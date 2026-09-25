"""Whole-night benchmark: every policy plays the same held-out nights against the truth.

    python -m obsassist.learn.evaluate --nights 20 --policies list,greedy,gbdt,anyjev,gpt \
        --heads heads/qwen3-1.7b.json --model Qwen/Qwen3-1.7B --out reports/nights.json

Nights: random programs (different seeds, dates, telescopes from the training set) plus the two
demo programs under several weather seeds. For every night: the score of each policy, the
pilot-oracle score (hindsight-optimal search) and the upper bound; reported as the fraction of
the oracle and of the bound, with the number of targets finished. Decisions per night and
wall-clock per decision are recorded (the LLM policies are slow; the planner is not).

Policies:
  list      - go down the list in order
  greedy    - the planner's merit policy (what the console recommends without models)
  lookahead - top candidates re-ranked by projecting the rest of the night under the forecast
  gbdt    - greedy candidates ranked by a gradient-boosted regret model on planner numbers (no LLM)
  anyjev  - candidates ranked by System 1 (AnyJev L2 heads on a local LLM)
  gpt     - candidates chosen by System 2 (OpenAI) from the same state text, one call per decision
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import numpy as np

from obsassist.paths import DECISIONS, HEADS, REPORTS


class RankedPolicy:
    """Greedy's candidate list, re-ranked by a scoring function over (obs, names)."""

    def __init__(self, name: str, score_fn, max_cands: int = 8):
        from obsassist.planning.planner import GreedyPolicy

        self.name = name
        self.score_fn = score_fn
        self.max_cands = max_cands
        self.greedy = GreedyPolicy()
        self.calls = 0
        self.seconds = 0.0

    def act(self, model, state, now=None):
        from obsassist.assistant.observation import from_model
        from obsassist.planning.planner import Action, Nowcast

        a, cands = self.greedy.act(model, state, now)
        feas = [c for c in cands if c.feasible]
        if a.kind != "observe" or len(feas) < 2:
            return a, cands
        now = now or Nowcast.from_history(model, state.t)
        obs = from_model(model, state, now, cands)
        top = sorted(feas, key=lambda c: -c.merit)[: self.max_cands]
        t0 = time.time()
        scores = self.score_fn(obs, [c.name for c in top])
        self.seconds += time.time() - t0
        self.calls += 1
        c = top[int(np.argmin(scores))]
        return Action("observe", target=c.i, t_exp=c.t_exp, reason=self.name), cands


def gbdt_policy(train_path: str):
    from sklearn.ensemble import HistGradientBoostingClassifier

    from obsassist.assistant.observation import REGRET_CENTERS
    from obsassist.learn.baselines import features, matrix
    from obsassist.learn.fit import load_rows

    tr = load_rows(train_path)
    X, y, _ = matrix(tr, list(tr))
    mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, random_state=0).fit(X, y)
    cvec = np.array([REGRET_CENTERS[c] for c in mdl.classes_])
    return RankedPolicy(
        "gbdt", lambda obs, names: mdl.predict_proba(np.array([features(obs, n) for n in names])) @ cvec
    )


def anyjev_policy(model: str, heads: str):
    from obsassist.assistant.system1 import System1

    s1 = System1(model=model, heads=heads)
    return RankedPolicy("anyjev", lambda obs, names: [j.expected_regret for j in s1.judge_candidates(obs, names)])


def gpt_policy(model: str = "gpt-5.6-luna", effort: str = "low"):
    from openai import OpenAI

    from obsassist.assistant.observation import render
    from obsassist.assistant.system2 import PERSONA, api_key

    client = OpenAI(api_key=api_key())
    schema = {
        "type": "object",
        "properties": {"target": {"type": "string"}},
        "required": ["target"],
        "additionalProperties": False,
    }

    def score(obs, names):
        r = client.responses.create(
            model=model,
            instructions=PERSONA,
            reasoning={"effort": effort},
            input=[
                {
                    "role": "user",
                    "content": render(obs) + "\n\nWhich candidate should be observed next "
                    "to maximise the night's total science? Answer with its exact name.",
                }
            ],
            text={"format": {"type": "json_schema", "name": "pick", "schema": schema, "strict": True}},
        )
        try:
            pick = json.loads(r.output_text)["target"]
        except Exception:
            pick = ""
        return [0.0 if n == pick else 1.0 for n in names]

    return RankedPolicy(f"gpt:{model}", score)


def nights(n_random: int, seed0: int = 500000) -> List[tuple]:
    from obsassist.programs.generator import random_date

    rng = np.random.default_rng(seed0)
    out = []
    for k in range(n_random):
        tel = ("clay", "keck1")[k % 2]
        out.append(("random", tel, random_date(rng), int(rng.integers(1_000_000)), int(rng.integers(1_000_000))))
    for s in (11, 12, 13):
        out.append(("demo", "clay_mike_darktime", None, None, s))
        out.append(("demo", "keck1_lris_tonight", None, None, s))
    return out


def build_model(spec):
    from obsassist.programs.generator import generate_program
    from obsassist.sim.night import NightModel
    from obsassist.targets import builtin_programs, load_program

    kind, a, date, pseed, wseed = spec
    prog = generate_program(a, date, seed=pseed) if kind == "random" else load_program(builtin_programs()[a])
    return NightModel(prog, seed=wseed)


def _light_policy(name):
    from obsassist.planning.planner import GreedyPolicy, ListOrderPolicy

    if name == "lookahead":
        from obsassist.planning.lookahead import LookaheadPolicy

        return LookaheadPolicy()
    return {"list": ListOrderPolicy, "greedy": GreedyPolicy}[name]()


def _night_row(args, pols=None) -> dict:
    """Every policy on one night, with the hindsight oracle and the upper bound."""
    from obsassist.planning.oracle import pilot, upper_bound
    from obsassist.planning.planner import run_policy

    k, spec, names, width, with_bias, gpt_nights = args
    pols = pols if pols is not None else {n: _light_policy(n) for n in names}
    m = build_model(spec)
    orc = pilot(m, width=width)
    row = {
        "night": k,
        "spec": spec,
        "program": m.program.name,
        "regime": m.weather.regime,
        "max": m.max_score(),
        "oracle": round(orc.score, 3),
        "upper_bound": round(upper_bound(m), 3),
        "policies": {},
    }
    if with_bias:
        row["projection"] = projection_bias(m)
    for pn, pol in pols.items():
        if pn == "gpt" and k >= gpt_nights:
            continue
        t0 = time.time()
        s, _ = run_policy(m, pol)
        sc = m.score(s)
        row["policies"][pn] = {
            "score": round(sc, 3),
            "frac_oracle": round(sc / max(orc.score, 1e-9), 4),
            "done": int(m.done(s).sum()),
            "seconds": round(time.time() - t0, 1),
        }
    print(
        json.dumps(
            {
                "night": k,
                "program": row["program"][:40],
                "oracle": row["oracle"],
                **{p: v["frac_oracle"] for p, v in row["policies"].items()},
            }
        ),
        flush=True,
    )
    return row


def projection_bias(m) -> Dict[str, float]:
    """(projected - achieved) / max score for the greedy night, projected at 0/25/50/75 % of it."""
    from obsassist.planning.planner import GreedyPolicy, Nowcast, apply_action, project, run_policy

    g = GreedyPolicy()
    final, _ = run_policy(m, g)
    truth = m.score(final)
    marks = {q: m.t_start + int(q) / 100 * (m.t_end - m.t_start) for q in ("0", "25", "50", "75")}
    out: Dict[str, float] = {}
    s = m.initial_state()
    for _ in range(500):
        for q, tq in marks.items():
            if q not in out and s.t >= tq:
                out[q] = (project(m, s, Nowcast.from_history(m, s.t)).score - truth) / m.max_score()
        if m.finished(s) or len(out) == len(marks):
            break
        a, _ = g.act(m, s)
        s2, _ = apply_action(m, s, a)
        s = s2 if s2.t > s.t else m.wait(s2, 5.0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nights", type=int, default=20)
    ap.add_argument("--policies", default="list,greedy,gbdt")
    ap.add_argument("--train", default=str(DECISIONS / "train_decisions.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--heads", default=str(HEADS / "qwen3-1.7b.json"))
    ap.add_argument("--gpt-model", default="gpt-5.6-luna")
    ap.add_argument("--gpt-nights", type=int, default=6, help="GPT is slow and costs money: only the first N nights")
    ap.add_argument("--oracle-width", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8, help="parallel nights for the model-free policies")
    ap.add_argument(
        "--projection-bias",
        action="store_true",
        help="also measure the projection's bias: projected vs achieved greedy score at 0/25/50/75%% of the night",
    )
    ap.add_argument("--out", default=str(REPORTS / "nights.json"))
    a = ap.parse_args(argv)
    names = a.policies.split(",")
    pols = {n: _light_policy(n) for n in names if n in ("list", "greedy", "lookahead")}
    if "gbdt" in names:
        pols["gbdt"] = gbdt_policy(a.train)
    if "anyjev" in names:
        pols["anyjev"] = anyjev_policy(a.model, a.heads)
    if "gpt" in names:
        pols["gpt"] = gpt_policy(a.gpt_model)
    specs = nights(a.nights)
    light = set(names) <= {"list", "greedy", "lookahead"}
    if light and a.workers > 1:  # model-free policies: one night per process
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(
                ex.map(
                    _night_row, [(k, spec, names, a.oracle_width, a.projection_bias, 0) for k, spec in enumerate(specs)]
                )
            )
    else:
        rows = [
            _night_row((k, spec, names, a.oracle_width, a.projection_bias, a.gpt_nights), pols)
            for k, spec in enumerate(specs)
        ]
    summ = {}
    for pn in pols:
        v = [r["policies"][pn]["frac_oracle"] for r in rows if pn in r["policies"]]
        g = [r["policies"]["greedy"]["frac_oracle"] for r in rows if pn in r["policies"] and "greedy" in r["policies"]]
        summ[pn] = {
            "mean_frac_oracle": round(float(np.mean(v)), 4),
            "min": round(float(np.min(v)), 4),
            "n": len(v),
            "wins_vs_greedy": int(sum(x > y + 1e-9 for x, y in zip(v, g))) if g else None,
            "losses_vs_greedy": int(sum(x < y - 1e-9 for x, y in zip(v, g))) if g else None,
        }
        if hasattr(pols[pn], "calls") and getattr(pols[pn], "calls"):
            summ[pn]["s_per_decision"] = round(pols[pn].seconds / pols[pn].calls, 3)
    summ["oracle_over_bound"] = round(float(np.mean([r["oracle"] / max(r["upper_bound"], 1e-9) for r in rows])), 4)
    if a.projection_bias:
        for q in ("0", "25", "50", "75"):
            v = np.array([r["projection"][q] for r in rows if q in r["projection"]])
            summ[f"projection_bias_at_{q}pct"] = {
                "mean_pct_of_max": round(100 * float(v.mean()), 2),
                "rms_pct_of_max": round(100 * float(np.sqrt((v**2).mean())), 2),
                "n": len(v),
            }
    print(json.dumps(summ, indent=1))
    json.dump({"summary": summ, "nights": rows}, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
