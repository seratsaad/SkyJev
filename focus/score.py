"""Rescore the saved decider answers against two answer keys. No new model calls.

Key 1, the window rules (decide.truth_action): judge each five-step window from its own five points.
Written before any model output was seen. Reading the disagreements showed it is often wrong: a window
whose widths fall all the way to its last step was called "accept" because a parabola through five
noisy points put its vertex just inside.

Key 2, the full run: judge each window by where the fit to all nine steps puts best focus. It uses the
four steps the model never saw, so it is the better truth. It was adopted after seeing the model's
answers, which is why both keys are reported.

  uv run --with astropy --with scipy python focus/score.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decide import truth_action  # noqa: E402

RES = Path(__file__).resolve().parent / "results"


def robust_full_fit(fw):
    """Best step from all nine points, leaving out one bad image if that helps a lot (as in decide.py)."""
    s, f = np.arange(1, 10, dtype=float), np.asarray(fw, float)

    def rms(ss, ff):
        cc = np.polyfit(ss, ff ** 2, 2)
        return float(np.sqrt(np.mean((ff ** 2 - np.polyval(cc, ss)) ** 2))), cc

    base, c = rms(s, f)
    best, drop = min((rms(np.delete(s, i), np.delete(f, i))[0], i) for i in range(9))
    if best < base / 3:
        s, f = np.delete(s, drop), np.delete(f, drop)
        _, c = rms(s, f)
    if c[0] <= 0 or f.max() / f.min() < 1.3:
        return None
    return float(-c[1] / (2 * c[0]))


def truth_full(run, steps):
    if run["n_good"] < 3 or not run["median_fwhm_arcsec"]:
        return "D"
    s0 = robust_full_fit(run["median_fwhm_arcsec"])
    if s0 is None:
        return "D"
    lo, hi = steps[0], steps[-1]
    return "B" if s0 < lo + 0.5 else "C" if s0 > hi - 0.5 else "A"


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4.1-mini"
    d = json.loads((RES / f"decider_{model}.json").read_text())
    runs = {r.name.replace(".json", ".fits"): json.loads(r.read_text()) for r in RES.glob("WFI.*.json")}
    cases = d["action"]
    for c in cases:
        c["key1"] = truth_action(c["steps"], c["fwhm"], c["n_good"])
        c["key2"] = truth_full(runs[c["run"]], c["steps"])
    print(f"{len(cases)} cases; keys disagree on {sum(c['key1'] != c['key2'] for c in cases)}")
    for key in ("key1", "key2"):
        mix = Counter(c[key] for c in cases)
        base = mix.most_common(1)[0][1] / len(cases)
        line = f"{key}: mix {dict(sorted(mix.items()))}, always-{mix.most_common(1)[0][0]} scores {base:.0%}"
        for ans in ("raw", "debiased"):
            acc = np.mean([max(c[ans], key=c[ans].get) == c[key] for c in cases])
            line += f"  | model {ans} {acc:.0%}"
        print(line)
    print("\nstill wrong under key 2 (order-averaged):")
    for c in cases:
        p = max(c["debiased"], key=c["debiased"].get)
        if p != c["key2"]:
            s0 = robust_full_fit(runs[c["run"]]["median_fwhm_arcsec"]) if runs[c["run"]]["median_fwhm_arcsec"] else None
            print(f"  {c['run'][4:14]} {c['window']:9} key2 {c['key2']} model {p} ({c['debiased'][p]:.2f})  full-run best step "
                  f"{s0 if s0 is None else round(s0, 2)}  " + (" ".join(f"{v:.2f}" for v in c["fwhm"]) if c["fwhm"] else "-"))
    conf = [(max(c["debiased"].values()), max(c["debiased"], key=c["debiased"].get) == c["key2"]) for c in cases]
    for th in (0.9, 0.99):
        sel = [ok for p, ok in conf if p >= th]
        print(f"when the model's probability is at least {th}: {len(sel)} cases, {np.mean(sel):.0%} right")
    (RES / f"decider_scored_{model}.json").write_text(json.dumps(cases, indent=1, default=float))


if __name__ == "__main__":
    main()
