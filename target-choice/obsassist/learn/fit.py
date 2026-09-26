"""Fit AnyJev L2 heads on simulator-labelled decisions and measure them against the oracle.

    python -m obsassist.learn.fit --l0-states 120     # -> heads/qwen3-1.7b.json, reports/fit_qwen3-1.7b.json
    python -m obsassist.learn.fit --train data/decisions/queue_train.jsonl --test data/decisions/queue_test.jsonl \
        --out heads/qwen3-1.7b-queue.json --report reports/fit_qwen3-1.7b-queue.json --device cuda --dtype float16

Reports, on held-out nights (different programs, dates and weather):
  * next-target choice: mean hindsight regret (fraction of the program's max score) of the
    candidate System 1 ranks first, next to the greedy planner's pick, the queue rule's pick (rows
    with `is_queue_rule`), a random pick and the oracle (0 by definition); the share of states
    where each picks a within-0.5% choice;
  * calibration of the regret distribution (ECE of the "best choice" probability);
  * accuracy of the per-state questions (sky transparency, dome closure within 30 min);
  * with --l0-states, the same choice from AnyJev L0 (no heads, zero labels).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from obsassist.assistant.observation import render
from obsassist.paths import DECISIONS, HEADS, REPORTS


def load_rows(path: str) -> Dict[tuple, List[dict]]:
    states: Dict[tuple, List[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            states[(r["night"], r["t"])].append(r)
    # nights finish in any order in the parallel builder: sort, so sampling depends on the data only
    return dict(sorted(states.items()))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def sample_training(states: Dict[tuple, List[dict]], n_rows: int, seed: int = 0):
    keys = list(states)
    random.Random(seed).shuffle(keys)
    rows, st = [], []
    for k in keys:
        if len(rows) >= n_rows:
            break
        rows.extend(states[k])
        st.append(states[k][0])
    return rows, st


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--train", default=str(DECISIONS / "train_decisions.jsonl"))
    ap.add_argument("--test", default=str(DECISIONS / "test_decisions.jsonl"))
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-test-states", type=int, default=120)
    ap.add_argument("--questions", default="regret,sky,dome_risk")
    ap.add_argument("--out", default=str(HEADS / "qwen3-1.7b.json"))
    ap.add_argument(
        "--layers", default="12,16,20", help="comma-separated block indices to try (empty: AnyJev's 5 depths)"
    )
    ap.add_argument("--l0-states", type=int, default=0, help="also score zero-label L0 on this many test states")
    ap.add_argument("--report", default=str(REPORTS / "fit_qwen3-1.7b.json"))
    ap.add_argument("--eval-only", action="store_true", help="evaluate the heads already in --out, fit nothing")
    ap.add_argument("--device", default=None, help="torch device (default: MPS, else CUDA, else CPU)")
    ap.add_argument("--dtype", default="bfloat16", help="weights dtype (float16 on a V100)")
    a = ap.parse_args(argv)

    from obsassist.assistant.system1 import PER_CANDIDATE, QUESTIONS, System1

    qs = [q for q in a.questions.split(",") if q]
    tr = load_rows(a.train)
    te = load_rows(a.test)
    rows, st_rows = sample_training(tr, a.n_train)
    print(f"train: {len(rows)} candidate rows from {len(st_rows)} states; test states available: {len(te)}", flush=True)
    s1 = System1(model=a.model, heads=a.out if a.eval_only else None, device=a.device, dtype=a.dtype)
    report = {"model": a.model, "n_train_rows": len(rows), "n_train_states": len(st_rows), "fits": {}}
    t0 = time.time()
    for qn in [] if a.eval_only else qs:
        q = QUESTIONS[qn]
        if qn in PER_CANDIDATE:
            texts = [render(r["obs"], focus=r["candidate"]) for r in rows]
            y = [int(r["level"]) for r in rows]
        else:
            texts = [render(r["obs"]) for r in st_rows]
            if qn == "sky":
                y = [q.options.index(r["sky"]) for r in st_rows]
            else:
                y = [0 if r["dome_closes_30"] else 1 for r in st_rows]
        if len(set(y)) < 2:
            print(f"skip {qn}: one class only", flush=True)
            continue
        t1 = time.time()
        layers = [int(x) for x in a.layers.split(",")] if a.layers else None
        art = s1.decider.fit_head(q, texts, y, layers=layers)
        report["fits"][qn] = {
            "n": len(y),
            "layer": art["layer_abs"],
            "kind": art.get("method"),
            "cv": art.get("cv"),
            "seconds": round(time.time() - t1, 1),
            "class_counts": np.bincount(y).tolist(),
        }
        print(
            f"fit {qn}: n={len(y)} layer={art['layer_abs']} {art.get('method')} cv={art.get('cv')} "
            f"({time.time() - t1:.0f}s)",
            flush=True,
        )
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        s1.decider.save_artifacts(a.out)  # after every head: an interrupted run keeps what it fit
    if not a.eval_only:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        s1.decider.save_artifacts(a.out)
    report["fit_seconds"] = round(time.time() - t0, 1)

    # ---------------------------------------------------------------- evaluation on held-out nights
    keys = list(te)
    random.Random(1).shuffle(keys)
    keys = keys[: a.n_test_states]
    res = defaultdict(list)
    pb_all, yb_all = [], []
    t2 = time.time()
    for k in keys:
        cand_rows = te[k]
        obs = cand_rows[0]["obs"]
        names = [r["candidate"] for r in cand_rows]
        reg = {r["candidate"]: r["regret"] for r in cand_rows}
        js = s1.judge_candidates(obs, names)
        if "regret" in qs:
            pick = min(js, key=lambda j: j.expected_regret).name
            res["s1_regret"].append(reg[pick])
            pick_b = max(js, key=lambda j: j.p_best or 0).name
            res["s1_pbest"].append(reg[pick_b])
            for j in js:
                pb_all.append(j.p_best)
                yb_all.append(1.0 if reg[j.name] < 0.005 else 0.0)
        g = [r for r in cand_rows if r["is_greedy"]]
        res["greedy"].append(g[0]["regret"] if g else float(np.mean(list(reg.values()))))
        q = [r["regret"] for r in cand_rows if r.get("is_queue_rule")]
        if q:  # the queue rule's pick, when it is among the labelled candidates
            res["queue_rule"].append(q[0])
        res["random"].append(float(np.mean(list(reg.values()))))
    summ = {}
    for k, v in res.items():
        v = np.array(v)
        summ[k] = {
            "mean_regret_pct": round(100 * v.mean(), 3),
            "within_0.5pct": round(float((v < 0.005).mean()), 3),
            "n": len(v),
        }
    if pb_all:
        summ["p_best_ece"] = round(ece(np.array(pb_all), np.array(yb_all)), 4)
    # per-state questions
    for qn in [q for q in qs if q not in PER_CANDIDATE]:
        texts, truth = [], []
        for k in keys:
            r = te[k][0]
            texts.append(render(r["obs"]))
            truth.append(r["sky"] if qn == "sky" else r["dome_closes_30"])
        decs = s1.decider.decide_batch(texts, QUESTIONS[qn], level="auto")
        if qn == "sky":
            acc = np.mean([d.answer == t for d, t in zip(decs, truth)])
            summ[qn] = {
                "accuracy": round(float(acc), 3),
                "level": decs[0].level,
                "base_rate": round(float(np.mean([t == "photometric" for t in truth])), 3),
            }
        else:
            p = np.array([d.p_true for d in decs])
            y = np.array(truth, dtype=float)
            summ[qn] = {
                "accuracy": round(float(np.mean((p >= 0.5) == (y > 0.5))), 3),
                "ece": round(ece(p, y), 4),
                "base_rate": round(float(y.mean()), 3),
                "level": decs[0].level,
            }
    summ["eval_seconds"] = round(time.time() - t2, 1)
    if a.l0_states and "regret" in qs:
        # the same questions with zero labels: AnyJev L0 (no heads), on the first l0_states test states
        from obsassist.assistant.system1 import System1 as _S1

        z = _S1(model=a.model, loaded_backend=s1.backend)  # same weights, no heads
        regs = []
        for k in keys[: a.l0_states]:
            cand_rows = te[k]
            reg = {r["candidate"]: r["regret"] for r in cand_rows}
            js = z.judge_candidates(cand_rows[0]["obs"], list(reg))
            regs.append(reg[min(js, key=lambda j: j.expected_regret).name])
        v = np.array(regs)
        g = np.array([next((r["regret"] for r in te[k] if r["is_greedy"]), np.nan) for k in keys[: a.l0_states]])
        summ["s1_L0_zero_labels"] = {
            "mean_regret_pct": round(100 * v.mean(), 3),
            "within_0.5pct": round(float((v < 0.005).mean()), 3),
            "n": len(v),
            "greedy_same_states_pct": round(100 * float(np.nanmean(g)), 3),
        }
    report["eval"] = summ
    print(json.dumps(summ, indent=1))
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    main()
