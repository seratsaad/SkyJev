"""Deterministic verifiers. They read the final browser state independently of the agent.

An Outcome is what the runner reads after the agent stops:
  url       location.href
  text      document.body.innerText (full document, not the agent's own snapshot)
  evaluate  callable(js_expression) -> value, for app-state checks (canvas apps, frames)

Verifier spec (YAML):
  type: url_contains | url_regex | url_not_equal | page_contains | page_regex
        | frames_regex | js_expr | all_of | any_of
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

FRAMES_TEXT = """(() => {
  const out=[document.body ? document.body.innerText : ''];
  const walk=doc=>{ for (const f of doc.querySelectorAll('iframe')) {
    try { const d=f.contentDocument; if (d && d.body) { out.push(d.body.innerText); walk(d); } } catch (e) {}
  } };
  walk(document);
  return out.join('\\n');
})()"""


@dataclass
class Outcome:
    url: str
    text: str
    evaluate: Callable[[str], object] | None = None


def _ok(passed, detail):
    return {"passed": bool(passed), "detail": detail}


def verify(spec: dict, outcome: Outcome) -> dict:
    kind = spec["type"]
    value = spec.get("value")
    flags = 0 if spec.get("case_sensitive") else re.IGNORECASE
    if kind == "url_contains":
        return _ok(value.lower() in outcome.url.lower(), f"url={outcome.url}")
    if kind == "url_regex":
        return _ok(re.search(value, outcome.url, flags), f"url={outcome.url}")
    if kind == "url_not_equal":
        return _ok(outcome.url.rstrip("/") != value.rstrip("/"), f"url={outcome.url}")
    if kind == "page_contains":
        text = outcome.text if spec.get("case_sensitive") else outcome.text.lower()
        needle = value if spec.get("case_sensitive") else value.lower()
        return _ok(needle in text, f"{value!r} {'found' if needle in text else 'absent'} in {len(outcome.text)} chars")
    if kind == "page_regex":
        m = re.search(value, outcome.text, flags)
        return _ok(m, f"match={m.group(0)!r}" if m else f"no match for {value!r} in {len(outcome.text)} chars")
    if kind == "frames_regex":
        if outcome.evaluate is None:
            return _ok(False, "no evaluate() available")
        text = str(outcome.evaluate(FRAMES_TEXT) or "")
        m = re.search(value, text, flags)
        return _ok(m, f"match={m.group(0)!r}" if m else f"no match in {len(text)} chars incl. frames")
    if kind == "js_expr":
        if outcome.evaluate is None:
            return _ok(False, "no evaluate() available")
        try:
            result = outcome.evaluate(spec["expression"])
        except Exception as e:  # noqa: BLE001
            return _ok(False, f"js error: {e}")
        return _ok(result, f"js -> {result!r}")
    if kind in {"all_of", "any_of"}:
        results = [verify(item, outcome) for item in spec["items"]]
        passed = all(r["passed"] for r in results) if kind == "all_of" else any(r["passed"] for r in results)
        return {"passed": passed, "detail": [r for r in results]}
    raise ValueError(f"unknown verifier type {kind!r}")


def verify_settled(spec: dict, read: Callable[[], Outcome], timeout: float = 10.0, interval: float = 1.0) -> dict:
    """Poll the check while the page finishes loading.

    The agent's clock has already stopped. Both agents get the same window, so a slow site is not
    scored as an agent failure; the elapsed wait is reported so a slow verification stays visible.
    """
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    result = {"passed": False, "detail": "not evaluated"}
    while True:
        try:
            result = verify(spec, read())
        except Exception as e:  # noqa: BLE001
            result = {"passed": False, "detail": f"read failed: {type(e).__name__}: {e}"}
        if result["passed"] or time.monotonic() >= deadline:
            break
        time.sleep(interval)
    result["settle_seconds"] = round(time.monotonic() - started, 2)
    return result


def url_specs(spec: dict) -> list[dict]:
    """The URL-only parts of a verifier, usable on a list of visited URLs after the fact."""
    if spec["type"] in {"all_of", "any_of"}:
        return [s for item in spec["items"] for s in url_specs(item)]
    return [spec] if spec["type"].startswith("url_") else []


def transient_url_match(spec: dict, urls: list[str]) -> dict:
    """Did the agent ever stand on a URL that satisfies the verifier's URL checks?

    A run can reach the right page and then navigate away before stopping. That is a different
    failure from never getting there, and both agents expose the URLs they visited, so this is
    measured the same way on both sides.
    """
    checks = url_specs(spec)
    if not checks or not urls:
        return {"applicable": False, "reached": None, "at": None}
    for index, url in enumerate(urls):
        outcome = Outcome(url=url, text="")
        if all(verify(c, outcome)["passed"] for c in checks):
            return {"applicable": True, "reached": True, "at": index, "url": url}
    return {"applicable": True, "reached": False, "at": None}


def goal_page_reached(goal_url: str | None, urls: list[str]) -> dict:
    """Did the agent ever stand on the goal page itself, judged from the URLs it recorded?

    Uses the task's `goal_url`, a pattern naming the goal page, rather than the verifier's URL
    clauses. Some of those clauses only say "the agent left the home page", and read alone they
    accept nearly every page a run visits, which once inflated the count of runs that "reached the
    goal and left". When a task has no goal_url the question is not asked at all.
    """
    import re as _re

    if not goal_url or not urls:
        return {"applicable": False, "reached": None, "at": None}
    for index, url in enumerate(urls):
        if url and _re.search(goal_url, url):
            return {"applicable": True, "reached": True, "at": index, "url": url}
    return {"applicable": True, "reached": False, "at": None}
