"""CLI: validate verifiers, run batches, inspect a page through JEV's own snapshot.

  python -m astroweb.runner validate            # every verifier on its gold page and its start page
  python -m astroweb.runner run --agent jev --repeat 3 [--task ID] [--category C]
  python -m astroweb.runner snapshot --url URL  # what JEV sees, no model calls
"""

from __future__ import annotations

import argparse
import json
import os
import time

from .budget import Budget, BudgetExceeded
from .cdp_direct import Session
from .tasks import ROOT, load_tasks
from .verification import Outcome, verify_settled

RESULTS = ROOT / "results"


def cmd_validate(args):
    """A verifier is only trustworthy if it passes on the gold page and fails on the start page."""
    report = []
    for task in load_tasks(task_set=args.set):
        if args.task and task.id != args.task:
            continue
        row = {"task": task.id, "category": task.category}
        for label, url in (("gold", task.gold_url), ("start", task.start_url)):
            if not url:
                row[label] = "n/a"
                continue
            try:
                with Session(url=url, settle=args.settle) as s:
                    result = verify_settled(
                        task.verifier,
                        lambda s=s: Outcome(url=s.url(), text=s.text(), evaluate=s.evaluate),
                        timeout=task.verify_timeout_seconds,
                    )
                row[label] = "PASS" if result["passed"] else "fail"
                row[label + "_settle"] = result.get("settle_seconds")
            except Exception as e:  # noqa: BLE001
                row[label] = f"ERROR {type(e).__name__}"
        expected = row.get("gold") in {"PASS", "n/a"} and row.get("start") in {"fail", "n/a"}
        row["verdict"] = "ok" if expected else "CHECK"
        report.append(row)
        print(f"{row['task']:26} gold={row['gold']:>6}  start={row['start']:>6}  {row['verdict']}", flush=True)
    bad = [r for r in report if r["verdict"] == "CHECK"]
    if args.task:
        # A filtered run is a spot check. Writing it over the full report would leave the deck
        # quoting a validation of one task while claiming to have validated fifteen.
        print(f"\n{len(report) - len(bad)}/{len(report)} checked (spot check; the saved report is unchanged)")
        return
    path = RESULTS / ("verifier_validation.json" if args.set == "dev" else f"verifier_validation_{args.set}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
    print(f"\n{len(report) - len(bad)}/{len(report)} verifiers behave as expected -> {path}")


def cmd_run(args):
    tasks = [
        t
        for t in load_tasks(task_set=args.set)
        if (not args.task or t.id in args.task) and (not args.category or t.category in args.category)
    ]
    out_dir = RESULTS / {"jev": "jev", "gpt-jev": "gpt_jev", "gpt-jev-plus": "gpt_jev_plus", "baseline": "baseline"}[args.agent]
    if args.set == "heldout":
        out_dir = out_dir.with_name(out_dir.name + "_heldout")
    if args.label:
        out_dir = out_dir.with_name(out_dir.name + "__" + args.label)
    if args.agent in {"jev", "gpt-jev", "gpt-jev-plus"}:
        from .jev_runner import run_jev

        def run_one(task, index, out_dir):
            return run_jev(task, index, out_dir, arm=args.agent)
    else:
        from .baseline_runner import run_baseline

        def run_one(task, index, out_dir):
            return run_baseline(task, index, out_dir, mode=args.mode)
    model = os.environ.get("BASELINE_MODEL" if args.agent == "baseline" else "GPT_CHOOSER_MODEL", "gpt-4.1-mini")
    budget = Budget(max_usd=args.max_usd, model=model)
    planned = args.repeat * len(tasks)
    print(f"{planned} runs planned, ceiling ${budget.max_usd:.2f} on {model}\n", flush=True)
    for index in range(1, args.repeat + 1):
        for task in tasks:
            try:
                budget.check()
            except BudgetExceeded as e:
                print(f"\n  STOPPING: {e}\n  Runs already saved are unaffected. "
                      f"Raise --max-usd or run fewer tasks to continue.", flush=True)
                return
            print(f"[{args.agent}] {task.id} run {index}/{args.repeat} ...", flush=True)
            try:
                record = run_one(task, index, out_dir)
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "OutOfCredit":
                    # Every later run would record the provider's refusal as agent behaviour.
                    print(f"\n  STOPPING: the model account is out of credit.\n  {e}\n"
                          f"  Runs already saved are unaffected. Add credit and start this arm again.",
                          flush=True)
                    return
                print(f"    runner error: {type(e).__name__}: {e}", flush=True)
                continue
            budget.add(
                total_tokens=(record.get("tokens") or 0)
                or (record.get("chooser_tokens") or 0) + (record.get("text_tokens") or 0)
            )
            print(
                f"    success={record.get('success')} status={record.get('status')} "
                f"{record.get('duration_seconds')}s steps={record.get('steps')} "
                f"calls={record.get('model_calls')} failure={record.get('failure_type')}",
                flush=True,
            )
            print(f"    {budget.line()}", flush=True)
            time.sleep(args.pause)
    print(f"\nfinished: {budget.line()}", flush=True)


def cmd_snapshot(args):
    """What JEV's own observation layer sees on a page. No model calls, no API key needed."""
    from jev_ultrafast.browser import Browser

    from .jev_runner import ENV_PROBE

    if args.plus:
        from .jev_plus import install

        print("with gpt-jev-plus fixes:", install())
    browser = Browser(args.url)
    time.sleep(args.settle)
    page = browser.observe(screenshot=False)
    probe = browser.evaluate(ENV_PROBE)
    print(f"{page['url']}\n{page['title']}")
    print(f"actions={len(page['actions'])} omitted={page['omitted_actions']} env={probe}")
    for action in page["actions"][: args.limit]:
        print(f"  [{action['id']:>12}] {action['kind']:6} {action.get('role', ''):10} {action['label'][:70]!r}")
    browser.close()


def main():
    parser = argparse.ArgumentParser(prog="astroweb.runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check verifiers against gold and start pages")
    p.add_argument("--task")
    p.add_argument("--settle", type=float, default=1.5)
    p.add_argument("--set", choices=["dev", "heldout"], default="dev")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("run", help="run a batch")
    p.add_argument("--agent", choices=["jev", "gpt-jev", "gpt-jev-plus", "baseline"], required=True)
    p.add_argument("--label", help="suffix for the results directory, e.g. an ablation name")
    p.add_argument("--mode", choices=["interface", "free"], default="interface", help="baseline only")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--task", action="append")
    p.add_argument("--category", action="append")
    p.add_argument("--pause", type=float, default=2.0)
    p.add_argument("--set", choices=["dev", "heldout"], default="dev", help="task set; heldout is for the final test only")
    p.add_argument("--max-usd", type=float, default=None, dest="max_usd",
                   help="stop the batch once estimated spend reaches this (default: MAX_SPEND_USD, or 5)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("snapshot", help="show JEV's element table for a URL")
    p.add_argument("--url", required=True)
    p.add_argument("--settle", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--plus", action="store_true", help="show the element table with the gpt-jev-plus fixes applied")
    p.set_defaults(func=cmd_snapshot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
