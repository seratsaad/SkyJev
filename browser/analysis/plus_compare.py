"""gpt-jev (baseline v0) against gpt-jev-plus, on the dev and held-out sets, with Wilson intervals.

Scoring rules, all written down in IMPROVING_JEV.md before the measured runs:
  - vizier_hipparcos_102: baseline v0 runs are rescored under verifier v2. All three ended on the VizieR
    front page, which v2 rejects, so they count as failures.
  - aladin_canvas_402 and eso_etc_403 are left out of the headline. Aladin's v1 verifier could never
    pass, so the baseline measured nothing there; ETC has no validated gold page. Both are reported
    separately.
  - Smoke runs (results/gpt_jev_plus__*) were used to build the fixes and are never counted.

    uv run python analysis/plus_compare.py
"""

from __future__ import annotations

import glob
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"aladin_canvas_402", "eso_etc_403"}
RESCORED_FAIL = {"vizier_hipparcos_102": "gpt-jev"}  # v1 false passes, see IMPROVING_JEV.md

ARMS = {
    ("dev", "gpt-jev"): "results/gpt_jev/*.json",
    ("dev", "gpt-jev-plus"): "results/gpt_jev_plus/*.json",
    ("heldout", "gpt-jev"): "results/gpt_jev_heldout/*.json",
    ("heldout", "gpt-jev-plus"): "results/gpt_jev_plus_heldout/*.json",
    ("dev", "browser-use"): "results/baseline/*__interface__*.json",
    ("heldout", "browser-use"): "results/baseline_heldout/*__interface__*.json",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def load(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(ROOT / pattern))):
        r = json.load(open(path))
        success = bool(r.get("success"))
        if RESCORED_FAIL.get(r["task_id"]) == r.get("agent") and "gpt_jev_plus" not in path:
            success = False
        # Same rule as analysis/summarize_results.py: the agent said it finished.
        claimed = bool(r.get("agent_claimed_success") if r.get("agent_claimed_success") is not None
                       else r.get("status") == "done" or r.get("agent_claimed_done"))
        rows.append({"task": r["task_id"], "category": r["category"], "success": success, "claimed": claimed,
                     "seconds": r.get("duration_seconds"), "failure": r.get("failure_type"),
                     "tokens": (r.get("tokens") or 0) if r.get("agent") == "browser-use"
                     else (r.get("chooser_tokens") or 0) + (r.get("text_tokens") or 0)})
    return rows


def median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    m = len(values) // 2
    return values[m] if len(values) % 2 else (values[m - 1] + values[m]) / 2


def main():
    report = {}
    for (task_set, arm), pattern in ARMS.items():
        rows = load(pattern)
        if not rows:
            continue
        head = [r for r in rows if r["task"] not in EXCLUDED]
        k, n = sum(r["success"] for r in head), len(head)
        lo, hi = wilson(k, n)
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task"]].append(r["success"])
        by_cat = defaultdict(list)
        for r in head:
            by_cat[r["category"]].append(r["success"])
        report[f"{task_set}/{arm}"] = {
            "runs": n, "successes": k, "rate": round(k / n, 3), "wilson95": [round(lo, 3), round(hi, 3)],
            "median_seconds": median([r["seconds"] for r in head]),
            "false_done_rate": round(sum(r["claimed"] and not r["success"] for r in head) / n, 3),
            "claim_agrees_with_page": round(sum(r["claimed"] == r["success"] for r in head) / n, 3),
            "median_tokens": median([r["tokens"] for r in head]),
            "by_category": {c: f"{sum(v)}/{len(v)}" for c, v in sorted(by_cat.items())},
            "by_task": {t: f"{sum(v)}/{len(v)}" for t, v in sorted(by_task.items())},
            "failures": dict(sorted(defaultdict(int, {f: sum(1 for r in head if r["failure"] == f and not r["success"])
                                                       for f in {r["failure"] for r in head}}).items(),
                                    key=lambda kv: -kv[1])) if head else {},
            "excluded_tasks": {t: f"{sum(v)}/{len(v)}" for t, v in by_task.items() if t in EXCLUDED},
        }
    # Paired view: only the held-out tasks every arm ran. Browser Use's batch hit its spend ceiling after
    # 11 of 13 tasks (Exoplanet Archive and Aladin were not run), so the full-set rates are not comparable.
    common = None
    for (task_set, arm), pattern in ARMS.items():
        if task_set == "heldout" and load(pattern):
            tasks = {r["task"] for r in load(pattern)}
            common = tasks if common is None else common & tasks
    for (task_set, arm), pattern in ARMS.items():
        if task_set != "heldout" or not common:
            continue
        rows = [r for r in load(pattern) if r["task"] in common]
        if not rows:
            continue
        k, n = sum(r["success"] for r in rows), len(rows)
        lo, hi = wilson(k, n)
        report[f"paired/{arm}"] = {
            "tasks": sorted(common), "runs": n, "successes": k, "rate": round(k / n, 3),
            "wilson95": [round(lo, 3), round(hi, 3)], "median_seconds": median([r["seconds"] for r in rows]),
            "false_done_rate": round(sum(r["claimed"] and not r["success"] for r in rows) / n, 3),
            "claim_agrees_with_page": round(sum(r["claimed"] == r["success"] for r in rows) / n, 3),
            "median_tokens": median([r["tokens"] for r in rows]),
        }
    out = ROOT / "results" / "summary" / "plus_compare.json"
    out.write_text(json.dumps(report, indent=1))
    for name, r in report.items():
        print(f"{name:24} {r['successes']:>3}/{r['runs']:<3} {r['rate']:.0%}  95% CI {r['wilson95'][0]:.0%}-{r['wilson95'][1]:.0%}"
              f"  median {r['median_seconds']}s  false-done {r['false_done_rate']:.0%}"
              f"  claim=page {r['claim_agrees_with_page']:.0%}  tok {r['median_tokens']}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
