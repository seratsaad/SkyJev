"""Replay a saved run at its original speed. The demo fallback: no network, no keys, no browser.

If a live demo does not start within about fifteen seconds, run this instead and keep talking. It
prints what the agent was offered, what it chose, how long the decision took, and what the page did
about it, using the timestamps from the saved trace.

  python analysis/replay.py --task simbad_vega_101            # the fastest verified run of that task
  python analysis/replay.py --file <name>.json --speed 2      # a specific run, twice as fast
  python analysis/replay.py --list                            # what is available to replay
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BOLD, DIM, GREEN, RED, GOLD, CYAN, OFF = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m"
)


def runs() -> list[dict]:
    out = []
    for path in sorted(RESULTS.glob("*/*.json")):
        if path.parent.name in {"summary", "hybrid", "shipped"} or path.parent.name.startswith("_"):
            continue
        record = json.loads(path.read_text())
        if "task_id" in record and record.get("history"):
            record["_file"] = path.name
            out.append(record)
    return out


def pick(task: str | None, filename: str | None, want_success: bool) -> dict:
    available = runs()
    if filename:
        for record in available:
            if record["_file"] == filename:
                return record
        raise SystemExit(f"no saved run called {filename}")
    candidates = [r for r in available if not task or r["task_id"] == task]
    if want_success and any(r.get("success") for r in candidates):
        candidates = [r for r in candidates if r.get("success")]
    if not candidates:
        raise SystemExit(f"no saved run for {task!r}")
    return min(candidates, key=lambda r: r.get("duration_seconds") or 1e9)


def replay(record: dict, speed: float, table_rows: int):
    history = record["history"]
    tables = record.get("element_tables") or []
    print(f"\n{BOLD}{record['task_id']}{OFF}  {DIM}{record['agent']}, {record['started_at']}{OFF}")
    print(f"{DIM}goal{OFF}  {record['goal']}")
    print(f"{DIM}from{OFF}  {record['start_url']}\n")

    previous_ms = 0
    for index, step in enumerate(history):
        table = tables[index] if index < len(tables) else None
        wait = max(0.0, (step.get("executed_ms", step["elapsed_ms"]) - previous_ms) / 1000 / max(speed, 0.01))
        time.sleep(min(wait, 6.0))
        previous_ms = step.get("elapsed_ms", previous_ms)

        if table:
            offered = [e for e in table if step["operation"] in e.get("operations", [])]
            print(f"{DIM}  offered for {step['operation']}: {len(offered)} of {len(table)} elements{OFF}")
            # Always show the element that was chosen, even when it sits far down the list. Seeing
            # the choice without seeing what was chosen is the one thing this view must not do.
            target = str(step.get("target", "")).split(":")[0]
            shown = offered[:table_rows]
            chosen_row = next((e for e in offered if str(e.get("index")) == target), None)
            elided = len(offered) - len(shown)
            if chosen_row is not None and chosen_row not in shown:
                shown = shown[:-1] + [None, chosen_row]
                elided = len(offered) - len(shown) + 1
            for element in shown:
                if element is None:
                    print(f"{DIM}     ... {elided} more{OFF}")
                    continue
                chosen = str(element.get("index")) == target
                mark = f"{GOLD}>{OFF}" if chosen else " "
                style = BOLD if chosen else DIM
                print(f"   {mark} {style}[{element.get('index'):>3}] {element.get('label', '')[:56]}{OFF}")
            if elided > 0 and None not in shown:
                print(f"{DIM}     ... {elided} more{OFF}")

        text = f'  "{step["text"]}"' if step.get("text") else ""
        changed = f"{GREEN}page changed{OFF}" if step.get("page_changed") else f"{RED}no change{OFF}"
        print(f"  {CYAN}{step['elapsed_ms']:>6} ms{OFF}  {BOLD}{step['operation']:<10}{OFF} "
              f"[{step.get('target')!s:>3}] {step['action'][:34]:34}{text}")
        print(f"          {DIM}decision {step['latency_ms']} ms{OFF}  {changed}  {DIM}{step['url'][:70]}{OFF}\n")

    verdict = f"{GREEN}VERIFIED{OFF}" if record.get("success") else f"{RED}NOT VERIFIED{OFF}"
    print(f"{BOLD}  {record.get('duration_seconds')} s  ·  {len(history)} actions  ·  "
          f"{record.get('model_calls')} model calls  ·  {verdict}{OFF}")
    if not record.get("success"):
        print(f"  {DIM}failure: {record.get('failure_type')}  ·  status: {record.get('status')}{OFF}")
    print(f"  {DIM}ended at {record.get('final_url')}{OFF}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--file")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--rows", type=int, default=8, help="element-table rows to show per step")
    parser.add_argument("--failure", action="store_true", help="prefer a failed run over a verified one")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for record in sorted(runs(), key=lambda r: (r["task_id"], r["run_index"])):
            flag = f"{GREEN}ok  {OFF}" if record.get("success") else f"{RED}fail{OFF}"
            print(f"{flag} {record['task_id']:24} {record.get('duration_seconds')!s:>7} s  "
                  f"{len(record['history']):>2} actions  {record['_file']}")
        return
    if not args.task and not args.file:
        parser.error("give --task or --file, or use --list")
    replay(pick(args.task, args.file, want_success=not args.failure), args.speed, args.rows)


if __name__ == "__main__":
    sys.exit(main())
