"""Next-target check: SkyJev acts on a state only when P(best) of its top candidate reaches a threshold,
and otherwise defers to the planner (or the observer).

Heads come from target-choice's fit (obsassist.learn.fit, per-candidate regret question). On held-out
simulated nights (the *_test.jsonl files, never fit on), the nights are split in two: states from the
validation nights choose the threshold, states from the test nights are reported. Per state we record
every labelled candidate's P(best) (probability of the lowest regret level, within 0.5% of the best
choice), expected regret and hindsight regret, plus the greedy planner's pick and, in queue mode, the
queue rule's pick.

Reported for each gate (threshold chosen on validation for 98% or 95% within-0.5% when acting, fixed
0.90 / 0.95, and no gate): coverage; mean regret of SkyJev's pick on the states it acts on, next to the
planner's (and the queue rule's) on the same states; the share of acted picks within 0.5% of best, with
a Wilson interval; agreement with the planner's pick; and the whole-night mix (act when confident,
otherwise take the planner's pick) against the planner alone.

    python -m routine.target --model Qwen/Qwen3-4B --heads routine/reports/target_heads_qwen3-4b_queue.json \
        --test target-choice/data/decisions/queue_test.jsonl --out routine/reports/target_gate_qwen3-4b_queue.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "target-choice"))
from routine.common import pick_threshold, wilson  # noqa: E402

BEST = 0.005  # within 0.5% of the best choice: regret level 0


def split_keys(states: Dict[tuple, List[dict]], n_val: int, n_test: int, seed: int = 0):
    """Nights split in half (validation / test); up to n states sampled from each half."""
    nights = sorted({k[0] for k in states})
    random.Random(seed).shuffle(nights)
    val_n = set(nights[: len(nights) // 2])
    kv = [k for k in states if k[0] in val_n]
    kt = [k for k in states if k[0] not in val_n]
    rv, rt = random.Random(seed + 1), random.Random(seed + 2)
    return sorted(rv.sample(kv, min(n_val, len(kv)))), sorted(rt.sample(kt, min(n_test, len(kt))))


def judge(s1, states, keys) -> List[Dict]:
    import torch

    recs = []
    for k in keys:
        rows = states[k]
        names = [r["candidate"] for r in rows]
        reg = {r["candidate"]: float(r["regret"]) for r in rows}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        js = s1.judge_candidates(rows[0]["obs"], names)
        torch.cuda.synchronize()
        g = next((r["candidate"] for r in rows if r["is_greedy"]), None)
        q = next((r["candidate"] for r in rows if r.get("is_queue_rule")), None)
        recs.append({"night": k[0], "t": k[1], "names": names, "regret": [reg[n] for n in names],
                     "p_best": [j.p_best for j in js], "exp_regret": [j.expected_regret for j in js],
                     "levels": sorted({j.level for j in js}), "greedy": g, "queue_rule": q,
                     "s": time.perf_counter() - t0})
    return recs


def arrays(recs: List[Dict]) -> Dict[str, np.ndarray]:
    top = [int(np.argmax(r["p_best"])) for r in recs]
    reg = [r["regret"] for r in recs]
    idx = lambda r, name: r["names"].index(name)  # noqa: E731
    return {
        "conf": np.array([r["p_best"][i] for r, i in zip(recs, top)]),
        "s1": np.array([g[i] for g, i in zip(reg, top)]),
        # the planner's pick; its row is missing only if the planner's choice was not labelled (as fit.py: mean)
        "greedy": np.array([r["regret"][idx(r, r["greedy"])] if r["greedy"] else float(np.mean(r["regret"])) for r in recs]),
        "queue": np.array([r["regret"][idx(r, r["queue_rule"])] if r["queue_rule"] else np.nan for r in recs]),
        "agree": np.array([r["greedy"] is not None and r["names"][i] == r["greedy"] for r, i in zip(recs, top)]),
    }


def gate_stats(a: Dict[str, np.ndarray], thr: float) -> Dict:
    act = a["conf"] >= thr
    n, na = len(act), int(act.sum())
    ok = a["s1"] < BEST
    pct = lambda v: round(100 * float(np.mean(v)), 3) if len(v) else None  # noqa: E731
    out = {"threshold": round(thr, 4), "n": n, "acted": na, "coverage": round(na / n, 4),
           "acting": {"skyjev_regret_pct": pct(a["s1"][act]), "planner_regret_pct": pct(a["greedy"][act]),
                      "skyjev_within_0.5pct": round(float(ok[act].mean()), 4) if na else None,
                      "skyjev_within_ci95": wilson(int(ok[act].sum()), na),
                      "planner_within_0.5pct": round(float((a["greedy"][act] < BEST).mean()), 4) if na else None,
                      "agree_with_planner": round(float(a["agree"][act].mean()), 4) if na else None},
           # SkyJev when it acts, the planner's pick when it defers, against the planner alone
           "mixed_regret_pct": pct(np.where(act, a["s1"], a["greedy"])), "planner_all_regret_pct": pct(a["greedy"])}
    q = a["queue"]
    if not np.isnan(q).all():
        out["acting"]["queue_rule_regret_pct"] = pct(q[act & ~np.isnan(q)])
        out["queue_rule_all_regret_pct"] = pct(q[~np.isnan(q)])
    return out


def report(val: List[Dict], test: List[Dict]) -> Dict:
    av, at = arrays(val), arrays(test)
    okv = av["s1"] < BEST
    thr = {"val_target_0.98": pick_threshold(av["conf"], okv, 0.98), "val_target_0.95": pick_threshold(av["conf"], okv, 0.95),
           "fixed_0.90": 0.90, "fixed_0.95": 0.95, "no_gate": 0.0}
    # when no threshold reaches the targets: the most confident 10% / 25% / 50% of validation states
    for c in (0.10, 0.25, 0.50):
        thr[f"val_top_{int(100 * c)}pct"] = float(np.quantile(av["conf"], 1 - c))
    return {"thresholds": thr,
            "top_p_best_quantiles_test": {q: round(float(np.quantile(at["conf"], q)), 4) for q in (0.5, 0.9, 0.99, 1.0)},
            "val": {g: gate_stats(av, t) for g, t in thr.items()},
            "test": {g: gate_stats(at, t) for g, t in thr.items()},
            "agree_with_planner_all": round(float(at["agree"].mean()), 4),
            "s_per_state_median": round(float(np.median([r["s"] for r in test])), 4),
            "levels": sorted({lv for r in val + test for lv in r["levels"]})}


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--test", required=True, help="held-out decisions jsonl (never fit on)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--rescore", action="store_true", help="recompute the report from the saved states (no GPU)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    raw = out.with_name(out.stem + "_states.json")
    if a.rescore:
        d = json.loads(raw.read_text())
    else:
        import torch
        import transformers
        from obsassist.assistant.system1 import System1
        from obsassist.learn.fit import load_rows

        states = load_rows(a.test)
        kv, kt = split_keys(states, a.n_val, a.n_test)
        s1 = System1(model=a.model, heads=a.heads, device="cuda", dtype=a.dtype)
        assert s1.level_of("regret") == "L2", "no regret head loaded"
        judge(s1, states, kv[:2])  # warm-up, untimed
        d = {"model": a.model, "heads": a.heads, "test_file": Path(a.test).name, "dtype": a.dtype,
             "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                          "transformers": transformers.__version__},
             "n_nights": len({k[0] for k in states}), "val": judge(s1, states, kv), "test": judge(s1, states, kt)}
        raw.write_text(json.dumps(d, default=float))
    rep = {k: d[k] for k in ("model", "heads", "test_file", "dtype", "hardware", "n_nights")}
    rep |= {"n_val_states": len(d["val"]), "n_test_states": len(d["test"])} | report(d["val"], d["test"])
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
