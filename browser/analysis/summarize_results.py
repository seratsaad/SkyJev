"""Aggregate every saved run into tidy tables. Reads results/*/*.json, writes results/summary/.

  python analysis/summarize_results.py [--print]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(ROOT_SRC))

from astroweb.tasks import load_tasks
from astroweb.verification import goal_page_reached

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "summary"
CATEGORY_ORDER = ["lookup", "search", "multistep", "interactive", "adversarial"]
ARM_ORDER = ["jev-ultrafast", "gpt-jev", "browser-use", "browser-use (free)"]

KEEP = [
    "task_id", "category", "agent", "arm", "mode", "run_index", "started_at", "success", "status",
    "failure_type", "duration_seconds", "wall_seconds", "setup_seconds", "steps", "model_calls",
    "chooser_calls", "text_helper_calls", "chooser_tokens", "text_tokens", "tokens",
    "chooser_latency_ms_median", "protocol_calls", "stale_retries", "final_url", "navigated_by_url",
    "agent_claimed_done", "agent_claimed_success", "chooser_model", "baseline_model", "jev_commit",
]


def ambiguity_from_tables(tables) -> dict:
    """Recompute action-space ambiguity from a run's first saved element table."""
    if not tables or not tables[0]:
        return {}
    elements = tables[0]
    names = Counter((e.get("role"), e.get("label")) for e in elements)
    ambiguous = sum(count for name, count in names.items() if count > 1)
    unnamed = sum(1 for e in elements if e.get("role") and e.get("label") in (e["role"], "Open " + e["role"]))
    return {"elements": len(elements), "ambiguous": ambiguous, "unnamed": unnamed,
            "fraction": round(ambiguous / len(elements), 3)}


def reviewed_labels() -> dict:
    """Labels confirmed or corrected by hand, from analysis/review_failures.py."""
    path = RESULTS / "failure_review.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_runs() -> pd.DataFrame:
    reviewed = reviewed_labels()
    # A verifier can be tightened after a run was saved, and the stored label then reflects the old
    # definition. "Reached the goal and left" is recomputed here from the run's own visited URLs
    # against the current verifier, so the taxonomy always describes the checks in force today.
    goal_urls = {task.id: task.goal_url for task in load_tasks()}
    rows = []
    for path in sorted(RESULTS.glob("*/*.json")):
        # A leading underscore marks a quarantined directory: runs kept for the record but
        # excluded from every table, figure and slide. results/<dir>/README.md says why.
        # "shipped" holds compacted duplicates of runs already counted from their own directories.
        if path.parent.name in {"summary", "shipped"} or path.parent.name.startswith("_"):
            continue
        # The follow-up (IMPROVING_JEV.md) has its own table, analysis/plus_compare.py: the held-out
        # set and the gpt-jev-plus arm must not mix into the original fifteen-task tables.
        if "heldout" in path.parent.name or path.parent.name.startswith("gpt_jev_plus"):
            continue
        record = json.loads(path.read_text())
        if "task_id" not in record:
            continue
        row = {k: record.get(k) for k in KEEP}
        row["file"] = path.name
        row["failure_type_auto"] = record.get("failure_type")
        # A hand-checked label wins over the automatic one.
        if path.name in reviewed:
            row["failure_type"] = reviewed[path.name]
        row["verify_settle_seconds"] = (record.get("verification") or {}).get("settle_seconds")
        visited = record.get("urls") or (
            [record.get("start_url")] + [h.get("url") for h in (record.get("history") or [])]
        )
        transient = goal_page_reached(goal_urls.get(record["task_id"]), [u for u in visited if u])
        row["transient_reached"] = transient.get("reached")
        if not record.get("success") and path.name not in reviewed:
            if transient.get("reached"):
                row["failure_type"] = "abandoned_success"
            elif row["failure_type"] == "abandoned_success":
                # Labelled from a loose verifier URL clause at run time. The goal page was never
                # actually reached, so fall back to what the run ended as.
                row["failure_type"] = "false_done" if record.get("status") == "done" else "no_progress"
        env = record.get("env_probe") or {}
        row["iframes"] = env.get("iframes")
        row["canvas"] = env.get("canvas")
        row["omitted_actions"] = (record.get("last_page") or {}).get("omitted_actions")
        # Older runs predate the field; the saved element tables still carry the information.
        ambiguity = record.get("ambiguity") or ambiguity_from_tables(record.get("element_tables"))
        row["elements_first_page"] = ambiguity.get("elements")
        row["ambiguous_elements"] = ambiguity.get("ambiguous")
        row["ambiguous_fraction"] = ambiguity.get("fraction")
        # The agent claimed done but the page did not verify.
        # One column for "the agent said it was finished", whichever agent it was.
        # One token column for both arms: the chooser plus the text helper on one side, the
        # agent's own usage on the other. Reported as tokens, not dollars, because prices move.
        row["total_tokens"] = (
            (record.get("chooser_tokens") or 0) + (record.get("text_tokens") or 0)
            if record.get("agent") != "browser-use"
            else (record.get("tokens") or 0)
        ) or None
        # Free mode lets the baseline type a record URL straight into the address bar, which JEV
        # cannot do at all. Mixing those runs into the interface-mode totals would compare the two
        # architectures on different affordances, so they are a separate arm.
        if record.get("mode") == "free":
            row["agent"] = "browser-use (free)"
        row["claimed"] = bool(
            record.get("agent_claimed_success")
            if record.get("agent") == "browser-use"
            else record.get("status") == "done"
        )
        row["false_done"] = bool(
            (record.get("status") == "done" or record.get("agent_claimed_done")) and not record.get("success")
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["agent"] = pd.Categorical(df["agent"], [a for a in ARM_ORDER if a in set(df["agent"])], ordered=True)
    df["category"] = pd.Categorical(df["category"], CATEGORY_ORDER, ordered=True)
    return df.sort_values(["agent", "category", "task_id", "run_index"])


def per_agent(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("agent", observed=True).apply(
        lambda g: pd.Series(
            {
                "runs": len(g),
                "success_rate": g["success"].mean(),
                "median_seconds": g["duration_seconds"].median(),
                "median_seconds_success": g.loc[g["success"], "duration_seconds"].median(),
                "median_steps": g["steps"].median(),
                "median_model_calls": g["model_calls"].median(),
                "false_done_rate": g["false_done"].mean(),
                # What the agent reported, against what the page actually showed. JEV reports by
                # choosing DONE; Browser Use reports with its own success flag and its own judge.
                "claimed_success_rate": g["claimed"].mean(),
                "claim_agrees_with_page": (g["claimed"] == g["success"].astype(bool)).mean(),
                "abandoned_success": (g["failure_type"] == "abandoned_success").sum(),
                "median_tokens": g["total_tokens"].median(),
                "tokens_per_success": (
                    g["total_tokens"].sum() / g["success"].sum() if g["success"].sum() else float("nan")
                ),
                "seconds_per_success": (
                    g["duration_seconds"].sum() / g["success"].sum() if g["success"].sum() else float("nan")
                ),
                "efficiency": g["success"].sum() / max(g["steps"].sum(), 1),
            }
        ),
        include_groups=False,
    )
    return out.reset_index()


def per_category(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby(["agent", "category"], observed=True).apply(
        lambda g: pd.Series(
            {
                "runs": len(g),
                "success_rate": g["success"].mean(),
                "median_seconds": g["duration_seconds"].median(),
                "median_steps": g["steps"].median(),
                "median_model_calls": g["model_calls"].median(),
            }
        ),
        include_groups=False,
    )
    return out.reset_index()


def per_task(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby(["task_id", "category", "agent"], observed=True).apply(
        lambda g: pd.Series(
            {
                "runs": len(g),
                "successes": int(g["success"].sum()),
                "success_rate": g["success"].mean(),
                "median_seconds": g["duration_seconds"].median(),
                "median_steps": g["steps"].median(),
                "failures": ",".join(sorted({str(f) for f in g.loc[~g["success"], "failure_type"].dropna()})),
            }
        ),
        include_groups=False,
    )
    return out.reset_index().sort_values(["category", "task_id", "agent"])


def failure_table(df: pd.DataFrame) -> pd.DataFrame:
    failures = df.loc[~df["success"].astype(bool)]
    if failures.empty:
        return pd.DataFrame(columns=["failure_type", "agent", "count"])
    out = failures.groupby(["failure_type", "agent"], observed=True).size().reset_index(name="count")
    return out.sort_values("count", ascending=False)


def comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per task: success and median time for each agent, side by side."""
    wide = df.pivot_table(
        index=["category", "task_id"],
        columns="agent",
        values=["success", "duration_seconds"],
        aggfunc={"success": "mean", "duration_seconds": "median"},
        observed=True,
    )
    wide.columns = [f"{b}_{a}" for a, b in wide.columns]
    return wide.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", dest="show")
    args = parser.parse_args()
    df = load_runs()
    if df.empty:
        print("no runs found under results/")
        return
    SUMMARY.mkdir(parents=True, exist_ok=True)
    tables = {
        "runs": df,
        "per_agent": per_agent(df),
        "per_category": per_category(df),
        "per_task": per_task(df),
        "failures": failure_table(df),
        "comparison": comparison_table(df),
    }
    for name, table in tables.items():
        table.to_csv(SUMMARY / f"{name}.csv", index=False)
    print(f"{len(df)} runs -> {SUMMARY}")
    if args.show:
        for name in ("per_agent", "per_category", "failures"):
            print(f"\n== {name}")
            print(tables[name].to_string(index=False))


if __name__ == "__main__":
    main()
