"""Tune the greedy planner's merit weights on training nights, check them on held-out nights.

    python -m obsassist.learn.tune            # 80 training nights, 34 held-out nights

The weights (planning/planner.py `MeritWeights`): alpha (exponent on the airmass/sky efficiency
now vs later tonight), beta (urgency boost for targets whose window is closing), continue_bonus
(staying on the current target, no overhead). Scored as the fraction of the maximum program
score on training nights; the held-out check reports the fraction of the hindsight oracle.
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from obsassist.paths import REPORTS

GRID = [
    dict(alpha=a, beta=b, continue_bonus=c)
    for a, b, c in itertools.product([0.0, 0.5, 1.0, 2.0], [0.0, 1.5, 3.0, 6.0], [1.0, 1.15, 1.4])
]


def _night(args):
    spec, with_oracle = args
    from obsassist.learn.evaluate import build_model
    from obsassist.planning.oracle import pilot
    from obsassist.planning.planner import GreedyPolicy, MeritWeights, run_policy

    m = build_model(spec)
    scores = [m.score(run_policy(m, GreedyPolicy(MeritWeights(**w)))[0]) for w in GRID]
    return {"max": m.max_score(), "scores": scores, "oracle": pilot(m).score if with_oracle else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-nights", type=int, default=80)
    ap.add_argument("--test-nights", type=int, default=34)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(REPORTS / "merit_tuning.json"))
    a = ap.parse_args(argv)
    from obsassist.learn.evaluate import nights
    from obsassist.programs.generator import random_date

    rng = np.random.default_rng(777)
    train = [
        (
            "random",
            ("clay", "keck1")[k % 2],
            random_date(rng),
            int(rng.integers(1_000_000)),
            int(rng.integers(1_000_000)),
        )
        for k in range(a.train_nights)
    ]
    with ProcessPoolExecutor(a.workers) as ex:
        tr = list(ex.map(_night, [(s, False) for s in train]))
        te = list(ex.map(_night, [(s, True) for s in nights(a.test_nights)]))
    T = np.array([np.array(r["scores"]) / r["max"] for r in tr])
    E = np.array([np.array(r["scores"]) / r["oracle"] for r in te])
    best, default = int(np.argmax(T.mean(0))), GRID.index(dict(alpha=1.0, beta=3.0, continue_bonus=1.15))
    out = {
        "default": GRID[default],
        "tuned": GRID[best],
        "train_frac_of_max": {
            "default": round(float(T[:, default].mean()), 4),
            "tuned": round(float(T[:, best].mean()), 4),
        },
        "test_frac_of_oracle": {
            "default": round(float(E[:, default].mean()), 4),
            "tuned": round(float(E[:, best].mean()), 4),
        },
        "test_wins_losses": [
            int((E[:, best] > E[:, default] + 1e-9).sum()),
            int((E[:, best] < E[:, default] - 1e-9).sum()),
        ],
        "grid": GRID,
        "train_mean": T.mean(0).round(5).tolist(),
        "test_mean": E.mean(0).round(5).tolist(),
    }
    print(json.dumps({k: v for k, v in out.items() if k not in ("grid", "train_mean", "test_mean")}, indent=1))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
