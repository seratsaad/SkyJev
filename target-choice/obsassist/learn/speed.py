"""Jev choice against text generation: the same held-out states, the same model, timed.

    python -m obsassist.learn.speed --test data/decisions/test_decisions.jsonl --heads heads/qwen3-1.7b.json \
        --device cuda --dtype float16 --out reports/speed_classical.json

Methods (all pick one candidate per state; regret is looked up in the hindsight labels):
  jev_heads   AnyJev with L2 heads, one regret question per candidate, highest P(best)
  jev_l0      AnyJev without heads (zero labels), lowest expected regret
  jev_choice  one AnyJev multiple-choice question listing the candidates (L0)
  gen_direct  the model writes the name (thinking off)
  gen_think   the model reasons, then writes the name (thinking on, capped at --max-think-tokens)
A generated answer that names no candidate is a failure and scores the state's mean regret.
The report is rewritten after every state, so an interrupted job keeps what it measured.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from obsassist.assistant.observation import render
from obsassist.learn.fit import load_rows
from obsassist.paths import REPORTS

QUESTION = "Which candidate should be observed next to maximise the night's total score?"
METHODS = ("jev_heads", "jev_l0", "jev_choice", "gen_direct", "gen_think")


def parse_pick(text: str, names: List[str]) -> Optional[str]:
    """The candidate named in a generated answer: the text after </think> when there is one, the
    longest name that occurs (names can contain each other), the last one mentioned if several."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    pattern = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    found = re.findall(pattern, text) if names else []
    return found[-1] if found else None


class Timer:
    def __init__(self, cuda: bool):
        self.cuda = cuda

    def __enter__(self):
        self._sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.s = time.perf_counter() - self.t0

    def _sync(self):
        if self.cuda:
            import torch

            torch.cuda.synchronize()


def generate(backend, text: str, think: bool, max_new: int):
    """Greedy generation with the chat template. Returns (answer text, new tokens)."""
    import torch

    tok, model = backend.tokenizer, backend.model
    msgs = [{"role": "user", "content": f"{text}\n\n{QUESTION} Answer with the candidate's exact name only."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    new = out[0, enc["input_ids"].shape[1] :]
    return tok.decode(new, skip_special_tokens=False), int(new.shape[0])


def summarise(res: Dict[str, List[dict]]) -> Dict[str, dict]:
    out = {}
    for m, rs in res.items():
        if not rs:
            continue
        r = np.array([x["regret"] for x in rs])
        s = np.array([x["seconds"] for x in rs if "seconds" in x])
        d = {"mean_regret_pct": round(100 * r.mean(), 3), "within_0.5pct": round(float((r < 0.005).mean()), 3)}
        d["n"] = len(rs)
        if len(s):
            d["s_per_decision_mean"] = round(float(s.mean()), 4)
            d["s_per_decision_median"] = round(float(np.median(s)), 4)
        if any("failed" in x for x in rs):
            d["failures"] = int(sum(x.get("failed", False) for x in rs))
        if any("new_tokens" in x for x in rs):
            d["mean_new_tokens"] = round(float(np.mean([x["new_tokens"] for x in rs])), 1)
        out[m] = d
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--n-states", type=int, default=60)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--heads", default="", help="L2 heads for jev_heads (empty: skip that method)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16", help="float16 on a V100")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--max-think-tokens", type=int, default=1024)
    ap.add_argument("--out", default=str(REPORTS / "speed.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    from anyjev import Decider, Question

    from obsassist.assistant.system1 import SYSTEM_PROMPT, System1

    methods = [m for m in a.methods.split(",") if m and (m != "jev_heads" or a.heads)]
    te = load_rows(a.test)
    keys = list(te)
    random.Random(1).shuffle(keys)
    keys = keys[: a.n_states]

    s1 = System1(model=a.model, heads=a.heads or None, device=a.device, dtype=a.dtype)
    s0 = System1(model=a.model, loaded_backend=s1.backend)  # same weights, no heads
    chooser = Decider(s1.backend, level="L0", prior="content_free", system=SYSTEM_PROMPT)
    cuda = str(s1.backend.device).startswith("cuda")
    k0 = keys[0]  # warm-up on one state, untimed
    s1.judge_candidates(te[k0][0]["obs"], [r["candidate"] for r in te[k0]])
    generate(s1.backend, render(te[k0][0]["obs"]), False, 8)

    info = {
        "test": Path(a.test).name,
        "model": a.model,
        "dtype": a.dtype,
        "heads": a.heads or None,
        "n_states": len(keys),
        "device": torch.cuda.get_device_name() if cuda else str(s1.backend.device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "failure_rule": "an answer naming no candidate scores the state's mean regret",
    }
    res: Dict[str, List[dict]] = {m: [] for m in methods + ["greedy", "queue_rule", "random"]}
    for n_done, k in enumerate(keys, 1):
        rows = te[k]
        obs = rows[0]["obs"]
        names = [r["candidate"] for r in rows]
        reg = {r["candidate"]: r["regret"] for r in rows}
        mean_reg = float(np.mean(list(reg.values())))
        text = render(obs, max_cands=len(names))  # every labelled candidate is listed
        for m in methods:
            rec: dict = {}
            with Timer(cuda) as t:
                if m == "jev_heads":
                    pick = max(s1.judge_candidates(obs, names), key=lambda j: j.p_best or 0).name
                elif m == "jev_l0":
                    pick = min(s0.judge_candidates(obs, names), key=lambda j: j.expected_regret).name
                elif m == "jev_choice":
                    q = Question.choice(QUESTION, names, name="next_target")
                    pick = chooser.decide(text, [q], level="L0")[0].answer
                else:
                    ans, rec["new_tokens"] = generate(
                        s1.backend, text, m == "gen_think", a.max_think_tokens if m == "gen_think" else 32
                    )
                    pick = parse_pick(ans, names)
            rec["seconds"] = t.s
            if pick is None:
                rec.update(regret=mean_reg, failed=True)
            else:
                rec.update(regret=reg[pick], failed=False, pick=pick)
            res[m].append(rec)
        g = [r["regret"] for r in rows if r["is_greedy"]]
        res["greedy"].append({"regret": g[0] if g else mean_reg})
        q = [r["regret"] for r in rows if r.get("is_queue_rule")]
        if q:
            res["queue_rule"].append({"regret": q[0]})
        res["random"].append({"regret": mean_reg})
        report = {"info": info, "states_done": n_done, "summary": summarise(res)}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1))
        print(f"{n_done}/{len(keys)} " + json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
