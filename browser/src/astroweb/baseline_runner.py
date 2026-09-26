"""One Browser Use run on one task. Same goal string, same verifier, same Chrome.

Two modes, because the two agents do not have the same affordances:

  free       Browser Use may navigate by typing a URL it already knows. JEV has no such action:
             it can only choose among the elements currently on the page.
  interface  The goal is prefixed with a rule to use the site's own interface rather than typing a
             URL. This is the fair comparison for browser interaction, and the default here.

Everything else is identical: same goal text, same start URL, same Chrome, same deterministic
verifier read afterwards through our own CDP client, never the agent's own `done` text.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from .cdp_direct import CDP_URL, Session, list_targets
from .tasks import Task
from .verification import Outcome, goal_page_reached, verify_settled

MAX_STEPS = 25
INTERFACE_RULE = (
    "Work inside the website's own interface: use its links, search boxes, forms and buttons. "
    "Do not type a full URL for a results or record page into the address bar."
)


async def _run(task: Task, model: str, timeout: float, mode: str) -> dict:
    """Run the agent, then verify the page it actually finished on, then tear the session down.

    Order matters. Stopping the session first closes the agent's tab, and a verifier that then
    attaches to "the newest tab" reads whatever is left over from an earlier run. That produced a
    round of results where a run ended correctly on an arXiv abstract page and the check was applied
    to an ADS page from the previous task. Verification now happens while the tab is still open,
    on the target whose URL matches the agent's own last URL.
    """
    from browser_use import Agent, Browser, ChatOpenAI

    goal = f"Start at {task.start_url} . {task.goal}"
    if mode == "interface":
        goal += " " + INTERFACE_RULE
    browser = Browser(cdp_url=CDP_URL, keep_alive=True)
    agent = Agent(task=goal, llm=ChatOpenAI(model=model), browser=browser, max_actions_per_step=3)
    t0 = time.perf_counter()
    error, history = None, None
    try:
        history = await asyncio.wait_for(agent.run(max_steps=MAX_STEPS), timeout=timeout)
        status = "done" if history.is_done() else "budget"
    except TimeoutError:
        status = "timeout"
    except Exception as e:  # noqa: BLE001
        status, error = "error", f"{type(e).__name__}: {e}"
    duration = round(time.perf_counter() - t0, 3)

    payload = {"status": status, "error": error, "duration_seconds": duration, "goal_sent": goal}
    if history is not None:
        usage = getattr(history, "usage", None)
        payload.update(
            steps=history.number_of_steps(),
            urls=[str(u) for u in history.urls()],
            actions=[str(a) for a in history.action_names()],
            agent_claimed_done=history.is_done(),
            agent_claimed_success=bool(history.is_successful()),
            final_result=history.final_result(),
            tokens=getattr(usage, "total_tokens", None) if usage else None,
            # The first navigate is the start URL, which both arms are given. Only a later one is
            # the agent choosing to jump straight to a page instead of using the site.
            navigated_by_url=sum(1 for a in history.action_names() if "navigate" in a.lower()) > 1,
            provider_errors=_provider_errors(history),
        )
    else:
        payload.update(steps=0, urls=[], actions=[], agent_claimed_done=False, agent_claimed_success=False)

    payload.update(_verify_live(task, payload.get("urls") or []))

    # Stop the agent's session inside this same loop. Never kill(): the Chrome is ours, not its own.
    try:
        await browser.stop()
    except Exception:  # noqa: BLE001, S110  - teardown is best effort; a failure here must not lose the run
        pass
    return payload


class OutOfCredit(RuntimeError):
    """The account cannot pay for a model call. Nothing after this measures the agent."""


def _provider_errors(history) -> list[str]:
    """Per-step LLM failures. Browser Use keeps stepping through them, so a dead API key looks
    exactly like an agent that cannot make progress. It is not, and the two must not be mixed."""
    out = []
    try:
        for error in history.errors():
            if error:
                out.append(str(error)[:200])
    except Exception:  # noqa: BLE001
        return out
    return out


def _verify_live(task: Task, agent_urls: list[str]) -> dict:
    """Attach to the tab the agent finished on and run the deterministic check there.

    The tab is chosen by matching the agent's own last URL, so a stale tab from an earlier run
    cannot be mistaken for this one. If no tab matches, the run is recorded as unverifiable rather
    than as a failure: a measurement that could not be taken is not evidence about the agent.
    """
    # Browser Use sometimes records an empty URL for the final step, so take the last real one.
    last_url = next((u for u in reversed(agent_urls) if u and u != "about:blank"), None)
    targets = list_targets()
    match = None
    if last_url:
        match = next((t for t in reversed(targets) if t.get("url") == last_url), None)
        if match is None:
            base = last_url.split("#")[0].split("?")[0]
            match = next((t for t in reversed(targets) if t.get("url", "").startswith(base)), None)
    if match is None:
        return {
            "verification": {"passed": False, "detail": f"no open tab matches the agent's last URL {last_url!r}"},
            "final_url": last_url,
            "verified_tab": None,
            "unverifiable": True,
        }
    try:
        session = Session(target_id=match["id"], timeout=15.0)
        session.owned = False
        verification = verify_settled(
            task.verifier,
            lambda: Outcome(url=session.url(), text=session.text(), evaluate=session.evaluate),
            timeout=task.verify_timeout_seconds,
        )
        final_url = session.url()
        session.close()
    except Exception as e:  # noqa: BLE001
        return {
            "verification": {"passed": False, "detail": f"read failed: {type(e).__name__}: {e}"},
            "final_url": last_url,
            "verified_tab": match["id"],
            "unverifiable": True,
        }
    return {"verification": verification, "final_url": final_url,
            "verified_tab": match["id"], "unverifiable": False}


def run_baseline(task: Task, run_index: int, out_dir: Path, timeout: float | None = None, mode: str = "interface") -> dict:
    model = os.environ.get("BASELINE_MODEL", "gpt-4.1")
    timeout = timeout or max(180, task.timeout_seconds * 2)
    known = {t["id"] for t in list_targets()}
    started_wall = datetime.now(UTC).isoformat(timespec="seconds")
    result = asyncio.run(_run(task, model, timeout, mode))

    # Verification already happened inside the run, before teardown. What is left is hygiene:
    # close every tab this run opened, so the next run cannot inherit one.
    verification = result.pop("verification")
    final_url = result.pop("final_url")
    leftovers = [t["id"] for t in list_targets() if t["id"] not in known]
    if leftovers:
        try:
            session = Session(target_id=leftovers[-1])
            session.owned = False
            for tid in leftovers:
                try:
                    session.send("Target.closeTarget", targetId=tid)
                except Exception:  # noqa: BLE001, S110  - best effort tab cleanup
                    pass
            session.close()
        except Exception:  # noqa: BLE001, S110  - hygiene only; never fails a run
            pass

    blocking = [e for e in (result.get("provider_errors") or []) if "credit" in e.lower() or "quota" in e.lower()]
    if blocking:
        raise OutOfCredit(blocking[0])

    record = {
        "task_id": task.id,
        "category": task.category,
        "agent": "browser-use",
        "mode": mode,
        "run_index": run_index,
        "started_at": started_wall,
        "goal": task.goal,
        "start_url": task.start_url,
        "baseline_model": model,
        "max_steps": MAX_STEPS,
        **{k: v for k, v in result.items() if k != "history"},
        "success": verification["passed"],
        "verification": verification,
        "final_url": final_url,
        "model_calls": result.get("steps", 0),
        "transient": goal_page_reached(task.goal_url, result.get("urls", [])),
        "failure_type": _label(
            result["status"],
            verification["passed"],
            result["error"],
            goal_page_reached(task.goal_url, result.get("urls", [])),
            claimed_success=bool(result.get("agent_claimed_success")),
            unverifiable=bool(result.get("unverifiable")),
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = started_wall.replace(":", "").replace("+0000", "Z")
    path = out_dir / f"{task.id}__{mode}__{run_index:02d}__{stamp}.json"
    path.write_text(json.dumps(record, indent=1, default=str))
    record["_path"] = str(path)
    return record


def _label(status, verified, error, transient=None, claimed_success=False, unverifiable=False):
    if verified:
        return None
    if unverifiable:
        return "unverifiable"
    if (transient or {}).get("reached"):
        return "abandoned_success"
    if error:
        return "provider_error" if any(w in error.lower() for w in ("rate", "api", "openai")) else "error"
    if status == "done":
        # Stopping is not a claim. An agent that finishes and reports that it could not do the task
        # has been honest, which is a different thing from asserting a success the page contradicts.
        return "false_done" if claimed_success else "agent_reported_failure"
    return {"budget": "action_budget", "timeout": "timeout"}.get(status, "error")
