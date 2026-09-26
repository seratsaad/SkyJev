"""Can a local, calibrated yes/no model decide "am I finished?" better than the agent does?

The constrained arm's largest single failure is reaching a page that satisfies the goal and then
leaving it. That is a failure of one binary decision, DONE, and a binary decision is exactly what
AnyJev (github.com/nokia-applied-research/AnyJev) is built for: a Yes/No read straight off the model's
next-token distribution, with position and label-prior bias removed, and no labels needed (level L0).

This replays that decision on trajectories the agent actually took. Nothing is re-run: every step of
every saved gpt-jev run has its URL, and each task's `goal_url` pattern says whether that URL is the
goal page. So each visited page is a labelled example, for free.

Only the seven tasks whose goal page is identifiable from its URL take part. An earlier version
labelled pages with the verifier's URL clauses instead, which for several tasks accept nearly any
page, and those labels were mostly noise.

The counterfactual is deliberately narrow. A DONE gate can only *stop* the agent earlier; it cannot
send it anywhere it did not go. So for each run we ask: had we stopped at the first step where
p(done) cleared a threshold, would the run have ended on a page that verifies?

  uv run python analysis/anyjev_done_gate.py                      # Qwen3-0.6B on CPU
  uv run python analysis/anyjev_done_gate.py --model Qwen/Qwen3-1.7B

No API calls, no browser. Runs on CPU; MPS weight loading crashed on the machine this was built on.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import re

from astroweb.tasks import load_tasks

OUT = ROOT / "results" / "anyjev"
QUESTION = "Does the current page already satisfy the user's goal, so that the task is finished?"


def is_goal_page(goal_url: str, url: str) -> bool:
    """The task's own goal-page pattern, applied to one visited URL. Deterministic, no model."""
    return bool(url and re.search(goal_url, url))


def build_dataset() -> list[dict]:
    tasks = {t.id: t for t in load_tasks()}
    rows = []
    for path in sorted(glob.glob(str(ROOT / "results" / "gpt_jev" / "*.json"))):
        run = json.loads(Path(path).read_text())
        task = tasks.get(run["task_id"])
        if task is None or not task.goal_url:
            continue  # the goal page is not identifiable from a URL, so a visited page cannot be labelled
        history = run.get("history") or []
        for i, h in enumerate(history):
            rows.append({
                "run": Path(path).name,
                "task_id": run["task_id"],
                "category": run["category"],
                "step": i + 1,
                "n_steps": len(history),
                "url": h["url"],
                "last_action": h.get("action"),
                "label": is_goal_page(task.goal_url, h["url"]),
                "goal": run["goal"],
                "run_success": bool(run.get("success")),
                "run_status": run.get("status"),
            })
    return rows


def state_for(row: dict) -> dict:
    """What the decision model sees. The goal and where the agent is, nothing from the verifier."""
    return {"goal": row["goal"], "current_url": row["url"], "last_action": row["last_action"]}


def score(rows: list[dict], model: str, level: str) -> np.ndarray:
    from anyjev import Decider, Question
    from anyjev.backends.hf import HFBackend

    backend = HFBackend(model, device="cpu", dtype="float32", batch_size=16)
    decider = Decider(backend)
    question = Question.noul(QUESTION, name="done")
    # decide_batch sees every state for the question at once, which is what lets the label-free
    # batch prior estimate and remove the model's lean toward Yes or No.
    result = decider.decide_batch([state_for(r) for r in rows], question, level=level)
    return np.array([d.p_true for d in result])


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability a random positive outranks a random negative. 0.5 is chance."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def counterfactual(rows: list[dict], p: np.ndarray, threshold: float) -> dict:
    """Stop at the first step where p(done) >= threshold. Did the run end on a verifying page?"""
    by_run: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_run.setdefault(r["run"], []).append(i)
    actual = gated = rescued = broken = 0
    for idx in by_run.values():
        idx = sorted(idx, key=lambda i: rows[i]["step"])
        was = rows[idx[0]]["run_success"]
        fire = next((i for i in idx if p[i] >= threshold), None)
        now = rows[fire]["label"] if fire is not None else was
        actual += was
        gated += now
        rescued += (now and not was)
        broken += (was and not now)
    return {"threshold": threshold, "runs": len(by_run), "actual": actual, "gated": gated,
            "rescued": rescued, "broken": broken}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    rows = build_dataset()
    labels = np.array([r["label"] for r in rows])
    print(f"{len(rows)} visited pages from {len({r['run'] for r in rows})} runs, "
          f"{labels.sum()} of them satisfy the goal\n")

    out = {"model": args.model, "question": QUESTION, "n": len(rows), "positives": int(labels.sum())}
    for level in ("raw", "L0"):
        t = time.perf_counter()
        p = score(rows, args.model, level)
        seconds = time.perf_counter() - t
        a = auroc(p, labels)
        sweep = [counterfactual(rows, p, th) for th in (0.5, 0.6, 0.7, 0.8, 0.9)]
        best = max(sweep, key=lambda s: (s["gated"], -s["broken"]))
        out[level] = {"auroc": a, "seconds": seconds, "ms_per_page": 1000 * seconds / len(rows),
                      "p_true": p.tolist(), "sweep": sweep, "best": best}
        print(f"[{level}]  AUROC {a:.3f}   {1000 * seconds / len(rows):.0f} ms per page")
        print(f"       mean p(done) on pages that satisfy the goal:   {p[labels].mean():.3f}")
        print(f"       mean p(done) on pages that do not:             {p[~labels].mean():.3f}")
        for s in sweep:
            print(f"       gate at {s['threshold']:.1f}: {s['gated']:>2}/{s['runs']} verify "
                  f"(actual {s['actual']}), rescued {s['rescued']}, broken {s['broken']}")
        print()

    # What the agent itself did at each labelled page, for comparison: it either stopped there or not.
    stopped = np.array([r["step"] == r["n_steps"] and r["run_status"] == "done" for r in rows])
    out["agent"] = {
        "stopped_on_satisfying_page": int((stopped & labels).sum()),
        "satisfying_pages_it_walked_past": int((~stopped & labels).sum()),
        "stopped_on_wrong_page": int((stopped & ~labels).sum()),
    }
    print(f"[agent's own DONE]  stopped on a satisfying page {out['agent']['stopped_on_satisfying_page']} times, "
          f"walked past one {out['agent']['satisfying_pages_it_walked_past']} times, "
          f"stopped on a wrong page {out['agent']['stopped_on_wrong_page']} times")

    OUT.mkdir(parents=True, exist_ok=True)
    name = args.model.split("/")[-1]
    (OUT / f"done_gate_{name}.json").write_text(json.dumps({**out, "rows": rows}, indent=1))
    print(f"\n-> {OUT / f'done_gate_{name}.json'}")


if __name__ == "__main__":
    main()
