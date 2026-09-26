"""One instrumented JEV run on one task.

Clock boundaries follow JEV's own report: `elapsed_ms` starts at the first prediction after the initial
observation and ends at the accepted DONE/BLOCKED (or our timeout). Browser setup and initial navigation
are reported separately as `setup_seconds`. Verification happens after the clock stops, from an
independent read of the page.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from jev_ultrafast import Agent
from jev_ultrafast.browser import StalePage

from .jev_compat import CdpCounter, patch_text_helper
from .tasks import Task
from .verification import Outcome, goal_page_reached, verify_settled

JEV_ROOT = Path(__file__).resolve().parents[3] / "jev-ultrafast"

ENV_PROBE = """(() => {
  let shadow=0; const walk=n=>{ if(n.shadowRoot) shadow++; for(const c of n.children) walk(c); };
  walk(document.documentElement);
  return {iframes: document.querySelectorAll('iframe').length,
          canvas: document.querySelectorAll('canvas').length, shadow};
})()"""


def jev_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(JEV_ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def auto_failure_label(
    status: str,
    verified: bool,
    page: dict,
    probe: dict,
    history: list,
    error: str | None,
    transient: dict | None = None,
    tabs: dict | None = None,
) -> str | None:
    if verified:
        return None
    if (transient or {}).get("reached"):
        return "abandoned_success"
    if error:
        if "Model provider" in error or "Model connection" in error or "Model unavailable" in error:
            return "provider_error"
        if "budget" in error:
            return "action_budget"
        if "Timeout" in error or "timed out" in error:
            return "browser_timeout"
        if "StalePage" in error:
            return "stale_page"
        return "error"
    if status == "timeout":
        return "timeout"
    if status == "stale_loop":
        return "unexecutable_target"
    if status == "done":
        return "false_done"
    # blocked
    if probe.get("iframes"):
        return "iframe"
    if probe.get("canvas"):
        return "canvas"
    if page.get("omitted_actions"):
        return "dense_action_space"
    tail = history[-3:]
    if len(tail) == 3 and all(h.get("page_changed") is False for h in tail):
        if (tabs or {}).get("opened"):
            return "popup_tab"
        return "no_progress"
    return "blocked_unclassified"


def page_targets() -> int:
    """How many page tabs the browser has open. A click on a target="_blank" link opens one more
    and leaves the agent's own tab unchanged, which is otherwise indistinguishable from a dead link."""
    try:
        import json as _json
        import urllib.request

        url = os.environ.get("BU_CDP_URL", "http://127.0.0.1:9222")
        with urllib.request.urlopen(f"{url}/json/list", timeout=3) as response:
            return sum(1 for item in _json.load(response) if item.get("type") == "page")
    except Exception:  # noqa: BLE001
        return -1


def _is_transport_timeout(error: Exception) -> bool:
    """A browser-harness IPC timeout, as opposed to anything the agent did."""
    return isinstance(error, TimeoutError) or "timed out" in str(error).lower()


def run_jev(
    task: Task,
    run_index: int,
    out_dir: Path,
    timeout: float | None = None,
    arm: str = "jev",
    stale_loop_limit: int = 8,
    setup_retries: int = 3,
) -> dict:
    """arm="jev" uses the TypeSafe chooser; arm="gpt-jev" swaps in an LLM over the same action space;
    arm="gpt-jev-plus" adds the engineering fixes in jev_plus, as switched on by JEV_PLUS."""
    patch_text_helper()
    plus_fixes = None
    if arm == "gpt-jev":
        from .gpt_chooser import install

        install()
    elif arm == "gpt-jev-plus":
        from .jev_plus import install as install_plus

        plus_fixes = install_plus()
    timeout = timeout or task.timeout_seconds
    counter = CdpCounter().install()
    started_wall = datetime.now(UTC).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    agent = None
    error = None
    status = "error"
    setup_seconds = None
    setup_attempts = 0
    tabs_before = page_targets()
    try:
        # Browser Harness allows a CDP call five seconds to answer, and some archive front pages take
        # longer than that just to begin navigating. A run lost before the agent has made a single
        # decision measures the harness, not the agent, so opening the page gets a few attempts.
        for attempt in range(1, setup_retries + 1):
            setup_attempts = attempt
            try:
                agent = Agent(task.start_url, task.goal)
                break
            except Exception as e:
                if attempt == setup_retries or not _is_transport_timeout(e):
                    raise
                time.sleep(3.0 * attempt)
        setup_seconds = round(time.perf_counter() - t0, 3)
        deadline = time.perf_counter() + timeout
        # JEV's own `run()` is `while status not in {done, blocked}: tick()`. We drive the same ticks
        # ourselves, unchanged, so we can also stop on two conditions JEV's loop does not count:
        # the wall-clock timeout, and a decision that the executor keeps refusing. The second one
        # matters: an element can be listed in the action space and still fail the hit test every
        # time, and the loop will then re-predict it forever without ever writing to history.
        consecutive_rejected = 0
        while agent.state["status"] not in {"done", "blocked"}:
            before = len(agent.state["history"])
            state = agent.command("tick")
            status = state["status"]
            consecutive_rejected = 0 if len(state["history"]) > before else consecutive_rejected + 1
            if consecutive_rejected >= stale_loop_limit:
                status = "stale_loop"
                break
            if time.perf_counter() > deadline:
                status = "timeout"
                break
    except StalePage as e:
        # A stale read at the very end is normal: the page navigated while we were observing.
        error = f"StalePage: {e}"
    except Exception as e:  # noqa: BLE001
        # Browser Harness raises its own timeout types. Losing the whole run to one would silently
        # bias the results toward whatever finished, so every run is recorded with its error.
        error = f"{type(e).__name__}: {e}"
    finally:
        counter.uninstall()

    record = {
        "task_id": task.id,
        "category": task.category,
        "agent": {"jev": "jev-ultrafast", "gpt-jev": "gpt-jev"}.get(arm, arm),
        "plus_fixes": plus_fixes,
        "arm": arm,
        "run_index": run_index,
        "started_at": started_wall,
        "goal": task.goal,
        "start_url": task.start_url,
        "jev_commit": jev_commit(),
        "chooser_model": os.environ.get("TYPESAFE_MODEL", "jev-latest")
        if arm == "jev"
        else os.environ.get("GPT_CHOOSER_MODEL", "gpt-4.1-mini"),
        "task_set": "heldout" if task.id.startswith("ho_") else "dev",
        "text_model": os.environ.get("TEXT_MODEL"),
        "status": status,
        "error": error,
        "setup_seconds": setup_seconds,
        "stale_loop_limit": stale_loop_limit,
        "setup_attempts": setup_attempts,
        "tabs_before": tabs_before,
    }
    if agent is None:
        record.update(success=False, failure_type="error", verification={"passed": False, "detail": error})
        return _save(record, out_dir)

    state = agent.state
    browser = agent.browser
    try:
        probe = browser.evaluate(ENV_PROBE) or {}
    except Exception:  # noqa: BLE001
        probe = {}
    try:
        def read():
            return Outcome(
                url=browser.evaluate("location.href"),
                text=browser.evaluate("document.body ? document.body.innerText : ''") or "",
                evaluate=browser.evaluate,
            )

        verification = verify_settled(task.verifier, read, timeout=task.verify_timeout_seconds)
        final_url = browser.evaluate("location.href")
    except Exception as e:  # noqa: BLE001
        final_url, verification = state["page"]["url"], {"passed": False, "detail": f"read failed: {e}"}
    finally:
        try:
            agent.close()
        except Exception:  # noqa: BLE001, S110  - the record is already complete; closing is best effort
            pass

    # Read the tab count before building the record: a click on a target="_blank" link leaves one
    # behind, and that is the difference between "the link is dead" and "the agent cannot follow it".
    tabs_after = page_targets()
    tabs_opened = tabs_after - tabs_before if tabs_before >= 0 and tabs_after >= 0 else 0

    history, decisions, text_calls = state["history"], state["decisions"], state["text_calls"]
    if plus_fixes is not None:
        from .jev_plus import run_report

        record["plus_report"] = run_report()
    chooser_tokens = sum((d.get("usage") or {}).get("total_tokens", 0) or 0 for d in decisions)
    text_tokens = sum((c.get("usage") or {}).get("total_tokens", 0) or 0 for c in text_calls)
    last_page = {k: state["page"].get(k) for k in ("url", "title", "omitted_actions")}
    last_page["n_actions"] = len(state["page"].get("actions", []))
    record.update(
        success=verification["passed"],
        verification=verification,
        final_url=final_url,
        duration_seconds=round(state["elapsed_ms"] / 1000, 3),
        wall_seconds=round(time.perf_counter() - t0, 3),
        steps=len(history),
        model_calls=len(decisions) + len(text_calls),
        chooser_calls=len(decisions),
        text_helper_calls=len(text_calls),
        chooser_tokens=chooser_tokens,
        text_tokens=text_tokens,
        chooser_latency_ms_median=_median([d["latency_ms"] for d in decisions]),
        protocol_calls=counter.total,
        protocol_calls_by_method=counter.calls,
        stale_retries=len(decisions) - len(history) - (1 if status in {"done", "blocked"} else 0),
        transient=goal_page_reached(task.goal_url, [task.start_url] + [h["url"] for h in history]),
        env_probe=probe,
        tabs_after=tabs_after,
        tabs_opened=tabs_opened,
        last_page=last_page,
        failure_type=auto_failure_label(
            status,
            verification["passed"],
            state["page"],
            probe,
            history,
            error,
            goal_page_reached(task.goal_url, [task.start_url] + [h["url"] for h in history]),
            {"opened": tabs_opened},
        ),
        history=history,
        decisions=[{k: v for k, v in d.items() if k not in {"request", "raw_answers"}} for d in decisions],
        text_calls=text_calls,
        element_tables=[d["request"]["state"]["elements"] for d in decisions],
        # Measured on the first observation, before the agent has changed anything.
        ambiguity=ambiguity(decisions[0]["request"]["state"]["elements"] if decisions else []),
    )
    return _save(record, out_dir)


def ambiguity(elements: list[dict]) -> dict:
    """How much of the action space is indistinguishable from the model's point of view.

    Two entries with the same role and the same label carry no information that could separate them,
    so a choice between them is a guess. Unnamed controls are the extreme case: their only label is
    their own role.
    """
    if not elements:
        return {"elements": 0, "ambiguous": 0, "unnamed": 0, "fraction": 0.0}
    names = Counter((e.get("role"), e.get("label")) for e in elements)
    ambiguous = sum(count for name, count in names.items() if count > 1)
    unnamed = sum(1 for e in elements if e.get("role") and e.get("label") in (e["role"], "Open " + e["role"]))
    return {
        "elements": len(elements),
        "ambiguous": ambiguous,
        "unnamed": unnamed,
        "fraction": round(ambiguous / len(elements), 3),
    }


def _median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _save(record: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = record["started_at"].replace(":", "").replace("+0000", "Z")
    path = out_dir / f"{record['task_id']}__{record.get('arm', 'jev')}__{record['run_index']:02d}__{stamp}.json"
    path.write_text(json.dumps(record, indent=1, default=str))
    record["_path"] = str(path)
    return record
