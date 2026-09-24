"""System 1: fast, calibrated typed decisions from a local LLM through AnyJev.

The questions are fixed (so AnyJev L2 heads can be fit once and reused every night); what
changes is the state text (`observation.render`). "Which target next?" is asked per candidate
("how much would observing THIS candidate next cost, relative to the best choice?"), which
works for any target list with any names; the assistant picks the candidate with the lowest
expected regret and knows how sure it is.

Levels: every decision carries AnyJev's calibration level (L0 = zero labels, L2 = a closed-form
head fit on simulator-labelled states). The policy layer refuses to act autonomously on L0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from obsassist.assistant.anyjev_backend import load_anyjev, make_local_backend
from obsassist.assistant.observation import REGRET_CENTERS, REGRET_LEVELS, render

load_anyjev()
from anyjev import Decider, Question  # noqa: E402
from anyjev.readout import build_prompt, render_chat_parts, resolve_labels  # noqa: E402

SYSTEM_PROMPT = "You are an observing assistant's decision function. Reply with the answer label only."

QUESTIONS: Dict[str, Question] = {
    "regret": Question.score(
        "If the evaluated candidate is observed next, how does the night's total compare with the best choice now?",
        levels=REGRET_LEVELS,
        name="regret",
    ),
    "sky": Question.choice(
        "What is the sky transparency right now?", ["photometric", "thin cirrus", "thick cloud"], name="sky"
    ),
    "dome_risk": Question.noul("Will the dome have to close for weather within the next 30 minutes?", name="dome_risk"),
}
# Each question must have a distinct (kind, options) layout: with level="auto", AnyJev serves a
# question that has no head of its own with any head of the same kind and option texts
# (label-free re-centring), e.g. every yes/no question would borrow the dome_risk head.
PER_CANDIDATE = ("regret",)
PER_STATE = ("sky", "dome_risk")


@dataclass
class CandidateJudgement:
    name: str
    expected_regret: float
    p_best: Optional[float]
    level: str
    distribution: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return {
            "name": self.name,
            "expected_regret": round(self.expected_regret, 4),
            "p_best": None if self.p_best is None else round(self.p_best, 3),
            "level": self.level,
            "distribution": {k[:18]: round(v, 3) for k, v in self.distribution.items()},
        }


class System1:
    """AnyJev over a local Hugging Face causal LM (default Qwen3-1.7B, on MPS or CUDA when present).
    `loaded_backend` shares the weights of another System1 (e.g. to compare with and without heads)."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-1.7B",
        heads: Optional[str] = None,
        prior: str = "content_free",
        batch_size: int = 16,
        loaded_backend=None,
    ):
        self.model_name = model
        self.backend = (
            loaded_backend if loaded_backend is not None else make_local_backend(model, batch_size=batch_size)
        )
        self.decider = Decider(self.backend, level="auto", prior=prior, system=SYSTEM_PROMPT)
        self.heads_path = heads
        self.n_heads = 0
        if heads and Path(heads).exists():
            self.n_heads = self.decider.load_artifacts(heads)
        self.last_latency_s = 0.0
        self.trace = None  # obsassist.assistant.trace.Trace: record every call's prompts and answers

    # ------------------------------------------------------------------ inference
    def level_of(self, qname: str) -> str:
        return "L2" if self.decider.route(QUESTIONS[qname]) else "L0"

    def prompt(self, text: str, qname: str) -> str:
        """The exact prompt the model reads for a state text and a question: chat template applied,
        options in canonical order (L0 also reads other orders of the same options)."""
        q = QUESTIONS[qname]
        tok = self.backend.tokenizer
        labels, _ = resolve_labels(tok, q)
        pre, suf = render_chat_parts(tok, build_prompt(text, q, list(range(q.k)), SYSTEM_PROMPT, labels))
        return pre + suf

    def _record(self, qnames, prompts: Dict[str, str], decisions: Dict[str, Any], latency: float, decision_id):
        qs = {
            n: {"kind": QUESTIONS[n].kind, "text": QUESTIONS[n].text, "options": QUESTIONS[n].options} for n in qnames
        }
        out = {
            k: {
                "level": d.level,
                "probs": d.distribution,
                "answer": d.answer,
                "diagnostics": {
                    x: d.diagnostics.get(x)
                    for x in ("readout", "exit_layer", "n_blocks", "temperature", "permutations", "routed_from")
                    if x in d.diagnostics
                },
            }
            for k, d in decisions.items()
        }
        self.trace.add(
            "system1",
            {"model": self.model_name, "heads": self.heads_path, "questions": qs, "prompts": prompts},
            out,
            decision=decision_id,
            summary=f"{', '.join(qnames)} on {len(prompts)} prompt(s)",
            latency_s=latency,
        )

    def judge_candidates(
        self,
        obs: Dict[str, Any],
        names: Optional[Sequence[str]] = None,
        questions: Sequence[str] = ("regret",),
        decision_id: Optional[int] = None,
    ) -> List[CandidateJudgement]:
        feas = [c["name"] for c in obs["candidates"] if c.get("feasible")]
        names = list(names) if names is not None else feas
        if not names:
            return []
        texts = [render(obs, focus=n) for n in names]
        t0 = time.time()
        out: Dict[str, Dict[str, Any]] = {n: {} for n in names}
        for qn in questions:
            decs = self.decider.decide_batch(texts, QUESTIONS[qn], level="auto")
            for n, d in zip(names, decs):
                out[n][qn] = d
        self.last_latency_s = time.time() - t0
        if self.trace is not None:
            self._record(
                questions,
                {n: self.prompt(t, questions[0]) for n, t in zip(names, texts)},
                {f"{n} / {qn}": out[n][qn] for n in names for qn in questions},
                self.last_latency_s,
                decision_id,
            )
        res = []
        for n in names:
            d = out[n].get("regret")
            er = float(np.dot(d.probs, REGRET_CENTERS)) if d is not None else float("nan")
            pb = float(d.probs[0]) if d is not None else None  # P(best) = P(lowest regret level)
            lvl = min((x.level for x in out[n].values()), key=lambda s: {"raw": 0, "L0": 1, "L1": 2, "L2": 3}[s])
            res.append(CandidateJudgement(n, er, pb, lvl, d.distribution if d is not None else {}))
        return res

    def state_questions(self, obs: Dict[str, Any], names: Sequence[str] = PER_STATE) -> Dict[str, Any]:
        text = render(obs)
        t0 = time.time()
        r = self.decider.decide(text, [QUESTIONS[n] for n in names], level="auto")
        if self.trace is not None:
            self._record(
                names, {n: self.prompt(text, n) for n in names}, {n: r[n] for n in names}, time.time() - t0, None
            )
        out = {}
        for n in names:
            d = r[n]
            out[n] = {
                "level": d.level,
                "answer": d.answer if not isinstance(d.answer, (np.bool_,)) else bool(d.answer),
                "confidence": round(d.confidence, 3),
                **({"p_true": round(d.p_true, 3)} if d.question.kind == "noul" else {"distribution": d.distribution}),
            }
        return out

    def info(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "heads": self.heads_path,
            "n_heads": self.n_heads,
            "levels": {q: self.level_of(q) for q in QUESTIONS},
            "last_latency_s": round(self.last_latency_s, 2),
        }
