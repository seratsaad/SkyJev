"""Read every failed run and confirm or correct its automatic label.

The automatic label is a first pass. Some of it is certain, because it comes from the environment
(an iframe count, a canvas count) or from the run's own URL history. The rest is a guess at intent
and deserves a human pass.

  python analysis/review_failures.py            # print every failure, grouped, with its trace
  python analysis/review_failures.py --summary  # counts only
  python analysis/review_failures.py --label task_id=arxiv_lookup_001 --as incorrect_target

Labels set by hand are stored in results/failure_review.json, keyed by result filename, and
summarize_results.py prefers them over the automatic label.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REVIEW = RESULTS / "failure_review.json"

# Labels that come from a measurement rather than a judgement, so a human pass adds nothing.
OBJECTIVE = {"iframe", "canvas", "dense_action_space", "abandoned_success", "browser_timeout",
             "provider_error", "timeout", "popup_tab", "agent_reported_failure"}

VOCABULARY = [
    "abandoned_success", "false_done", "unexecutable_target", "incorrect_target", "incorrect_text",
    "iframe", "canvas", "dense_action_space", "no_progress", "navigation_loop", "stale_page",
    "browser_timeout", "timeout", "action_budget", "provider_error", "element_not_exposed",
    "popup_tab", "agent_reported_failure", "error",
]


def load_failures() -> list[dict]:
    out = []
    for path in sorted(RESULTS.glob("*/*.json")):
        if path.parent.name in {"summary", "hybrid", "shipped"} or path.parent.name.startswith("_"):
            continue
        record = json.loads(path.read_text())
        if "task_id" not in record or record.get("success"):
            continue
        record["_file"] = path.name
        record["_dir"] = path.parent.name
        out.append(record)
    return out


def review_map() -> dict:
    return json.loads(REVIEW.read_text()) if REVIEW.exists() else {}


def _status_line(record: dict, reviewed: dict) -> str:
    seen = f"  reviewed={reviewed[record['_file']]}" if record["_file"] in reviewed else ""
    return f"  status   {record.get('status')}  auto={record.get('failure_type')}{seen}"


def describe(record: dict, reviewed: dict) -> str:
    lines = [
        f"{record['_file']}",
        f"  task     {record['task_id']}  ({record['category']}, {record['agent']})",
        _status_line(record, reviewed),
        (
            f"  time     {record.get('duration_seconds')} s, {record.get('steps')} steps, "
            f"{record.get('model_calls')} model calls"
        ),
        f"  ended at {record.get('final_url')}",
    ]
    if record.get("error"):
        lines.append(f"  error    {record['error'][:110]}")
    verification = record.get("verification") or {}
    detail = verification.get("detail")
    if isinstance(detail, list):
        for item in detail:
            lines.append(f"  check    {'pass' if item.get('passed') else 'FAIL'}  {str(item.get('detail'))[:90]}")
    elif detail:
        lines.append(f"  check    {str(detail)[:100]}")
    env = record.get("env_probe") or {}
    if env:
        lines.append(f"  page     iframes={env.get('iframes')} canvas={env.get('canvas')} "
                     f"shadow={env.get('shadow')} omitted={(record.get('last_page') or {}).get('omitted_actions')}")
    history = record.get("history") or []
    for h in history[:10]:
        lines.append(f"    {h['step']:2}. {h['operation']:10} {str(h.get('action'))[:32]:32} "
                     f"{'text=' + str(h.get('text'))[:22] + ' ' if h.get('text') else ''}{h['url'][:58]}")
    if len(history) > 10:
        lines.append(f"    ... {len(history) - 10} more actions")
    if record.get("actions"):  # baseline runs record action names instead
        lines.append(f"    actions  {', '.join(record['actions'][:12])}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--label", help="file=<result filename> or task_id=<id>")
    parser.add_argument("--as", dest="new_label", choices=VOCABULARY)
    parser.add_argument("--needs-review", action="store_true", help="only labels that are a judgement call")
    args = parser.parse_args()

    failures = load_failures()
    reviewed = review_map()

    if args.label and args.new_label:
        key, value = args.label.split("=", 1)
        touched = [f for f in failures if (f["_file"] == value if key == "file" else f["task_id"] == value)]
        for record in touched:
            reviewed[record["_file"]] = args.new_label
        REVIEW.write_text(json.dumps(reviewed, indent=1, sort_keys=True))
        print(f"labelled {len(touched)} run(s) as {args.new_label} -> {REVIEW}")
        return

    if args.needs_review:
        failures = [f for f in failures if f.get("failure_type") not in OBJECTIVE]

    if args.summary:
        auto = Counter(f.get("failure_type") for f in failures)
        confirmed = Counter(reviewed.get(f["_file"], f.get("failure_type")) for f in failures)
        print(f"{len(failures)} failed runs, {len(reviewed)} labelled by hand\n")
        print(f"{'label':24} {'auto':>6} {'final':>6}")
        for label in sorted(set(auto) | set(confirmed), key=lambda x: -confirmed[x]):
            print(f"{label!s:24} {auto[label]:>6} {confirmed[label]:>6}")
        return

    by_label: dict[str, list[dict]] = {}
    for record in failures:
        by_label.setdefault(str(record.get("failure_type")), []).append(record)
    for label in sorted(by_label, key=lambda k: -len(by_label[k])):
        marker = "" if label in OBJECTIVE else "   <- judgement call, check these"
        print(f"\n{'=' * 78}\n{label}  ({len(by_label[label])} runs){marker}\n{'=' * 78}")
        for record in by_label[label]:
            print(describe(record, reviewed))
            print()


if __name__ == "__main__":
    main()
