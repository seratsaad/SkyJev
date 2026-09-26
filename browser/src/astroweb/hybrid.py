"""Reasoning and navigation, separated.

The argument on the slide is that a browsing task mixes two kinds of decision, and they do not
deserve the same model. Clicking Search is trivial. Knowing what a redshift implies is not.

This runs the arrangement rather than drawing it:

    reasoning model   what do I need to look at?   ->  a browse goal and a starting URL
    browser policy    get to that page             ->  JEV's loop, unchanged
    reasoning model   what does the page say?      ->  an answer, quoted from the page

The browser policy is never asked to interpret anything, and the reasoning model never picks an
element. The reasoning model also never sees the network: it only sees text the browser brought back.

  uv run --env-file .env python -m astroweb.hybrid "Is NGC 4993 near enough for its distance to matter?"
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import jev_ultrafast.model as jev_model

from .jev_compat import patch_text_helper
from .tasks import ROOT

PLAN_SYSTEM = """You decide what a browser should go and look at. You cannot browse yourself.
Return JSON with exactly these keys:
  start_url  a real starting page for a public astronomy website
  goal       one sentence telling a browser agent what to reach, in plain language
  wanted     the specific fact to read off the page once it is open
The browser agent can only click, type, select and scroll among the controls on the page. It cannot
type a URL, run code, or read anything you do not send it to. Keep the goal to one destination."""

READ_SYSTEM = """You are given a question and the visible text of a web page that a browser agent
reached. Answer the question from that text alone. Return JSON with exactly these keys:
  answer     a short answer, or null if the page does not contain it
  evidence   the phrase from the page that supports it, or null
  confident  true or false
Never use knowledge that is not in the page text. If the page does not answer the question, say so
by returning null rather than filling it in from memory."""


def ask(system: str, payload: dict, model: str | None = None) -> dict:
    model = model or os.environ.get("REASONING_MODEL", "gpt-4.1-mini")
    result = jev_model.post_json(
        os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions",
        os.environ["OPENAI_API_KEY"],
        {
            "model": model,
            "max_tokens": 600,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload)[:120000]},
            ],
        },
    )
    return json.loads(result["choices"][0]["message"]["content"])


def run(question: str, arm: str = "gpt-jev", timeout: float = 120) -> dict:
    from jev_ultrafast import Agent

    from .gpt_chooser import install

    patch_text_helper()
    if arm == "gpt-jev":
        install()

    started = time.perf_counter()
    plan = ask(PLAN_SYSTEM, {"question": question})
    plan_seconds = round(time.perf_counter() - started, 2)

    agent = Agent(plan["start_url"], plan["goal"])
    browse_started = time.perf_counter()
    status, steps = "error", []
    try:
        deadline = time.perf_counter() + timeout
        while agent.state["status"] not in {"done", "blocked"}:
            state = agent.command("tick")
            status = state["status"]
            if time.perf_counter() > deadline:
                status = "timeout"
                break
        steps = [{"op": h["operation"], "on": h["action"], "url": h["url"]} for h in agent.state["history"]]
    except Exception as e:  # noqa: BLE001
        status = f"error: {type(e).__name__}: {e}"
    browse_seconds = round(time.perf_counter() - browse_started, 2)

    # The browser policy stops when it has nothing left to choose, which on a slow archive can be
    # before the page has finished drawing. The reading step is not part of the agent's clock, so
    # let the page settle first: read until the visible text stops growing.
    page_text, final_url, settle_seconds = "", None, 0.0
    settle_started = time.perf_counter()
    try:
        final_url = agent.browser.evaluate("location.href")
        previous = -1
        while time.perf_counter() - settle_started < 15:
            page_text = agent.browser.evaluate("document.body ? document.body.innerText : ''") or ""
            if len(page_text) == previous and len(page_text) > 200:
                break
            previous = len(page_text)
            time.sleep(1.0)
        settle_seconds = round(time.perf_counter() - settle_started, 2)
    except Exception:  # noqa: BLE001, S110  - a read failure leaves the reading step with no text
        pass
    finally:
        agent.close()

    read_started = time.perf_counter()
    reading = ask(READ_SYSTEM, {"question": question, "wanted": plan.get("wanted"),
                                "page_url": final_url, "page_text": page_text[:40000]})
    read_seconds = round(time.perf_counter() - read_started, 2)

    return {
        "question": question,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "plan": plan,
        "browse": {"status": status, "steps": steps, "final_url": final_url,
                   "page_chars": len(page_text), "seconds": browse_seconds,
                   "settle_seconds": settle_seconds},
        "reading": reading,
        "seconds": {"plan": plan_seconds, "browse": browse_seconds, "settle": settle_seconds,
                    "read": read_seconds,
                    "total": round(plan_seconds + browse_seconds + settle_seconds + read_seconds, 2)},
        "models": {"reasoning": os.environ.get("REASONING_MODEL", "gpt-4.1-mini"),
                   "chooser": os.environ.get("GPT_CHOOSER_MODEL", "gpt-4.1-mini") if arm == "gpt-jev"
                   else os.environ.get("TYPESAFE_MODEL", "jev-latest")},
    }


def main():
    question = " ".join(sys.argv[1:]) or "What is the redshift of the galaxy NGC 4993?"
    record = run(question)
    print(f"\nquestion   {record['question']}")
    print(f"plan       {record['plan']['goal']}")
    print(f"           start at {record['plan']['start_url']}")
    for step in record["browse"]["steps"]:
        print(f"  browse   {step['op']:10} {step['on'][:38]}")
    print(f"           {record['browse']['status']} at {record['browse']['final_url']}")
    print(f"answer     {record['reading'].get('answer')}")
    print(f"evidence   {record['reading'].get('evidence')}")
    s = record["seconds"]
    print(f"\ntime       plan {s['plan']}s + browse {s['browse']}s + settle {s['settle']}s "
          f"+ read {s['read']}s = {s['total']}s")
    out = Path(ROOT) / "results" / "hybrid"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"hybrid__{record['started_at'].replace(':', '').replace('+0000', 'Z')}.json"
    path.write_text(json.dumps(record, indent=1, default=str))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
