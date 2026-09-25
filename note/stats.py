"""Every number quoted in main.tex, recomputed from the saved results.

    python3 stats.py            # numpy only; no model calls, no network

Reads target-choice/reports/ and focus/results/ in this repository.
"""

import json
from math import sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(p):
    return json.loads((ROOT / p).read_text())


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def frac(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%}  [{lo:.0%}, {hi:.0%}]"


def boot_mean(x, rng, n=10000):
    x = np.asarray(x)
    return np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)], [2.5, 97.5])


def nights():
    rng = np.random.default_rng(0)
    d = load("target-choice/reports/nights_modelfree.json")
    s = d["summary"]
    print(f"== whole nights (n={len(d['nights'])}), fraction of the oracle score, 95% bootstrap")
    x = {p: [n["policies"][p]["frac_oracle"] for n in d["nights"]] for p in ("list", "greedy", "lookahead")}
    for p, v in x.items():
        lo, hi = boot_mean(v, rng)
        print(f"  {p:10s} {np.mean(v):.3f} [{lo:.3f}, {hi:.3f}]  worst {s[p]['min']:.2f}")
    diff = np.subtract(x["greedy"], x["lookahead"])
    lo, hi = boot_mean(diff, rng)
    print(f"  greedy - lookahead {diff.mean():.3f} [{lo:.3f}, {hi:.3f}]; lookahead wins/losses "
          f"{s['lookahead']['wins_vs_greedy']}/{s['lookahead']['losses_vs_greedy']}")
    print(f"  oracle / upper bound {s['oracle_over_bound']:.2f}")


def next_target():
    b, f = load("target-choice/reports/baselines.json"), load("target-choice/reports/fit_qwen3-1.7b.json")
    e = f["eval"]
    print(f"\n== next-target choice, {b['n_test_states']} held-out states, {b['n_train_rows']} training decisions")
    for k in ("greedy", "logreg", "gbdt"):
        print(f"  {k:8s} regret {b[k]['mean_regret_pct']:.2f}%  within 0.5% {b[k]['within_0.5pct']:.1%}")
    print(f"  random   regret {b['random_mean']['mean_regret_pct']:.2f}%")
    fr = f["fits"]["regret"]
    print(f"\n== AnyJev on {f['model']}: heads fit on {fr['n']} candidates from {f['n_train_states']} states "
          f"(layer {fr['layer']}, {fr['kind']}); scored on {e['greedy']['n']} held-out states")
    rows = [("random", e["random"]), ("L0 (no labels)", e["s1_L0_zero_labels"]),
            ("L2, lowest E[regret]", e["s1_regret"]), ("L2, highest P(best)", e["s1_pbest"]),
            ("greedy planner", e["greedy"])]
    for name, r in rows:
        print(f"  {name:22s} regret {r['mean_regret_pct']:.2f}%  within 0.5% {r['within_0.5pct']:.1%}")
    print(f"  P(best) ECE {e['p_best_ece']:.2f}")
    print(f"  sky: accuracy {e['sky']['accuracy']:.1%} vs always-photometric {e['sky']['base_rate']:.1%}")
    print(f"  dome within 30 min: accuracy {e['dome_risk']['accuracy']:.1%} vs always-no "
          f"{1 - e['dome_risk']['base_rate']:.1%}")


def focus():
    print("\n== focus: which of nine images is sharpest (within one step of the fitted best focus)")
    for m in ("gpt-4.1", "gpt-5.6-sol", "gpt-5.5", "gpt-4.1-mini"):
        d = load(f"focus/results/decider_sharpest_high_{m}.json")
        pick = lambda c, a: int(max(c[a], key=c[a].get))  # noqa: E731
        raw = sum(abs(pick(c, "raw") - c["truth"]) <= 1 for c in d)
        avg = sum(abs(pick(c, "debiased") - c["truth"]) <= 1 for c in d)
        print(f"  {m:12s} one order {raw}/{len(d)}   orders averaged {frac(avg, len(d))}")
    code = sum(abs(c["argmin"] - c["truth"]) <= 1 for c in d)
    print(f"  {'code':12s} narrowest measured image {frac(code, len(d))}")

    print("\n== focus: accept / move toward step 1 / move toward last step / retake")
    for m in ("gpt-4.1-mini", "gpt-4.1"):
        d = load(f"focus/results/decider_scored_{m}.json")
        runs = {c["run"] for c in d}
        top = lambda c: max(c["debiased"], key=c["debiased"].get)  # noqa: E731
        print(f"  {m}: {len(d)} cases from {len(runs)} runs; median {np.median([c['seconds'] for c in d]):.2f} s")
        for key, what in (("key2", "full nine-step fit (adopted after seeing answers)"),
                          ("key1", "rules written before any answer")):
            k = sum(top(c) == c[key] for c in d)
            maj = max(sum(c[key] == x for c in d) for x in "ABCD")
            print(f"    vs {what}: {frac(k, len(d))}; always-majority {maj / len(d):.0%}")
        sel = [top(c) == c["key2"] for c in d if max(c["debiased"].values()) >= 0.9]
        print(f"    probability >= 0.9: {frac(sum(sel), len(sel))}")
    rule = sum(c["key1"] == c["key2"] for c in d)
    print(f"  the pre-written rule on the same points vs the full nine-step fit: {frac(rule, len(d))}")


if __name__ == "__main__":
    nights()
    next_target()
    focus()
