"""Non-LLM controls for the decision benchmark.

The same hindsight labels, learned from the planner's numbers alone (no text, no LLM): a
gradient-boosted classifier over per-candidate features plus a few state features. If this
matches the LLM heads, the LLM adds nothing on the next-target question and its value is
elsewhere (reading PI notes, explaining, talking); if the LLM wins, it is reading something
the numbers miss. Also reports greedy, random, and the oracle (0).

    python -m obsassist.learn.baselines       # data/decisions/{train,test}_decisions.jsonl -> reports/baselines.json
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np

from obsassist.assistant.observation import REGRET_CENTERS
from obsassist.learn.fit import load_rows
from obsassist.paths import DECISIONS, REPORTS


def features(obs: Dict, cand: str) -> List[float]:
    """One candidate's planner numbers and the state's conditions, as a feature vector (no text)."""
    feas = [c for c in obs["candidates"] if c.get("feasible")]
    ranks = {c["name"]: k for k, c in enumerate(sorted(feas, key=lambda c: -(c.get("merit") or 0)))}
    mmax = max([c.get("merit") or 0 for c in feas] + [1e-12])
    c = next(x for x in obs["candidates"] if x["name"] == cand)
    win = min(c["window_left_min"] or 0, 900)
    ms = c.get("max_seeing")
    return [
        c["priority"],
        (c["snr_now"] or 0) / max(c["goal"] or 1, 1e-9),
        min(c["n_needed"] or 0, 60),
        (c["t_exp"] or 0) / 60,
        min(c["time_needed_min"] or 0, 900),
        c["overhead_min"] or 0,
        win,
        win - min(c["time_needed_min"] or 0, 900),
        c["airmass"] or 0,
        1.0 if c.get("rising") else 0.0,
        c["efficiency"] or 0,
        c["urgency"] or 0,
        ranks.get(cand, 9),
        (c.get("merit") or 0) / mmax,
        (ms - c["fwhm_pred"]) if ms else 1.0,
        1.0 if c.get("has_window") else 0.0,
        1.0 if c.get("is_current") else 0.0,
        1.0 if c.get("kind") == "backup" else 0.0,
        1.0 if c.get("kind") == "standard" else 0.0,
        1.0 if c.get("kind") == "too" else 0.0,
        obs["dimm"] or 0,
        obs.get("dimm_trend") or 0,
        obs["guider_flux_ratio"] or 0,
        (obs["humidity_limit"] or 90) - (obs["humidity"] or 0),
        obs.get("humidity_trend") or 0,
        obs["night_left_min"],
        len(feas),
        obs["remaining_p1"],
        obs["done"] / max(obs["total"], 1),
    ]


def matrix(states, keys):
    X, y, grp = [], [], []
    for k in keys:
        for r in states[k]:
            X.append(features(r["obs"], r["candidate"]))
            y.append(r["level"])
            grp.append(k)
    return np.array(X, dtype=float), np.array(y), grp


def evaluate_choice(states, keys, score_fn) -> Dict:
    regs = []
    for k in keys:
        rows = states[k]
        s = score_fn(k, rows)
        pick = int(np.argmin(s))
        regs.append(rows[pick]["regret"])
    v = np.array(regs)
    return {
        "mean_regret_pct": round(100 * v.mean(), 3),
        "within_0.5pct": round(float((v < 0.005).mean()), 3),
        "n": len(v),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(DECISIONS / "train_decisions.jsonl"))
    ap.add_argument("--test", default=str(DECISIONS / "test_decisions.jsonl"))
    ap.add_argument("--n-train-states", type=int, default=0, help="0 = all")
    ap.add_argument("--n-test-states", type=int, default=0)
    ap.add_argument("--out", default=str(REPORTS / "baselines.json"))
    a = ap.parse_args(argv)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr, te = load_rows(a.train), load_rows(a.test)
    ktr, kte = list(tr), list(te)
    rng = np.random.default_rng(0)
    if a.n_train_states:
        ktr = list(rng.choice(ktr, size=min(a.n_train_states, len(ktr)), replace=False))
    if a.n_test_states:
        kte = list(np.random.default_rng(1).choice(len(kte), size=min(a.n_test_states, len(kte)), replace=False))
        kte = [list(te)[i] for i in kte]
    Xtr, ytr, _ = matrix(tr, ktr)
    out = {"n_train_rows": len(ytr), "n_test_states": len(kte)}
    centers = np.array(REGRET_CENTERS)
    models = {
        "gbdt": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, random_state=0),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
    }
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        classes = list(mdl.classes_)
        cvec = np.array([centers[c] for c in classes])

        def score(k, rows, mdl=mdl, cvec=cvec):
            X = np.array([features(r["obs"], r["candidate"]) for r in rows])
            return mdl.predict_proba(X) @ cvec

        out[name] = evaluate_choice(te, kte, score)
    out["greedy"] = evaluate_choice(te, kte, lambda k, rows: [0.0 if r["is_greedy"] else 1.0 for r in rows])
    out["merit_top"] = evaluate_choice(
        te,
        kte,
        lambda k, rows: [
            -(next(c for c in r["obs"]["candidates"] if c["name"] == r["candidate"]).get("merit") or 0) for r in rows
        ],
    )
    out["random_mean"] = {
        "mean_regret_pct": round(100 * float(np.mean([np.mean([r["regret"] for r in te[k]]) for k in kte])), 3)
    }
    out["oracle"] = {"mean_regret_pct": 0.0, "within_0.5pct": 1.0}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
