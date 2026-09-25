"""System 2: an OpenAI model that deliberates, explains and talks to the observer.

Roles:
  * chat        - answer the observer's questions, calling tools (state, candidates, ETC,
                  projection, frame quick-look) instead of guessing numbers;
  * deliberate  - decide the next action when System 1 is not confident enough to act, and
                  return a structured recommendation with a confidence;
  * narrate     - write night-log summaries.

The model never computes photons or airmass itself: tools do. Tool calls that change the
telescope are not available to it; it can only recommend. Models are configurable
(defaults: gpt-5.6-luna for chat, gpt-5.6-sol for deliberation). The API key is read from
$OPENAI_API_KEY, or from `OPENAI=` in ~/.env.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

CHAT_MODEL = os.environ.get("OBSASSIST_CHAT_MODEL", "gpt-5.6-luna")
DELIBERATE_MODEL = os.environ.get("OBSASSIST_DELIBERATE_MODEL", "gpt-5.6-sol")

PERSONA = """You are the observing assistant for a night at a large optical telescope, working next to a
professional astronomer. Be concise and concrete, in the language observers use (seeing, airmass, S/N per
Angstrom, readout, twilight, slit losses). Always ground numbers in the tools (get_state, get_candidates,
etc_estimate, get_projection, get_frame_quicklook); never invent measurements. When you recommend an action,
say what to type or click (target, exposure time, number of exposures) and why, including the main risk.
You cannot move the telescope or start exposures yourself; you recommend, the observer decides.
Priorities: finish high-priority targets inside their constraints; do not waste good seeing on targets that
do not need it; respect time windows and ToO alerts; keep calibrations the program asks for."""


def api_key() -> Optional[str]:
    k = os.environ.get("OPENAI_API_KEY")
    if k:
        return k
    env = Path.home() / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            for name in ("OPENAI_API_KEY=", "OPENAI="):
                if line.startswith(name):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


TOOLS = [  # Responses API function-tool format
    {
        "type": "function",
        "name": "get_state",
        "description": "Current night state: time, conditions, dome, telescope, instrument arms, progress per target.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "get_candidates",
        "description": "Observable candidates now with the planner's numbers (time to finish, window left, "
        "efficiency, merit) and System 1's expected regret and P(best).",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "etc_estimate",
        "description": "Exposure time calculator for a program target under given (or current) seeing/cloud.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "seeing": {"type": "number", "description": "DIMM seeing at 500 nm, arcsec"},
                "cloud": {"type": "number", "description": "grey extinction, mag"},
                "t_exp": {"type": "number", "description": "single exposure seconds"},
            },
            "required": ["target"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_projection",
        "description": "Projected rest of the night under the planner's forecast: blocks, finish times, "
        "list completion.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "get_frame_quicklook",
        "description": "Quick-look of a FITS frame (latest if no name): trace, FWHM, S/N at the reference "
        "wavelength, saturation, flags.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": False},
    },
]

RECOMMEND_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["observe", "continue", "wait", "focus", "stop_exposure", "abort_exposure", "calibration"],
        },
        "target": {"type": "string", "description": "exact program target name (or empty for wait/focus)"},
        "t_exp": {"type": "number", "description": "length of ONE exposure in SECONDS (e.g. 1200), not minutes"},
        "n_exp": {"type": "integer", "description": "number of exposures of t_exp seconds"},
        "confidence": {"type": "number", "description": "probability (0-1) that this is the best action now"},
        "rationale": {"type": "string"},
        "risk": {"type": "string"},
    },
    "required": ["action", "target", "t_exp", "n_exp", "confidence", "rationale", "risk"],
    "additionalProperties": False,
}


class System2:
    def __init__(
        self,
        tools: Dict[str, Callable[..., Any]],
        chat_model: str = CHAT_MODEL,
        deliberate_model: str = DELIBERATE_MODEL,
    ):
        self.tools = tools
        self.chat_model = chat_model
        self.deliberate_model = deliberate_model
        key = api_key()
        self.enabled = key is not None
        self.client = None
        if self.enabled:
            from openai import OpenAI

            self.client = OpenAI(api_key=key)
        self.history: List[Dict[str, Any]] = []
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}
        self.trace = None  # obsassist.assistant.trace.Trace: record every request, response and tool call

    # ------------------------------------------------------------------ plumbing (Responses API)
    def _create(
        self,
        model: str,
        input_items,
        tools=None,
        effort: Optional[str] = None,
        text_format=None,
        instructions: str = PERSONA,
        tag: Optional[int] = None,
        role: str = "",
    ):
        kw: Dict[str, Any] = {"model": model, "input": input_items, "instructions": instructions}
        if tools:
            kw["tools"] = tools
        if effort:
            kw["reasoning"] = {"effort": effort}
        if text_format:
            kw["text"] = {"format": text_format}
        t0 = time.time()
        r = self.client.responses.create(**kw)
        dt = time.time() - t0
        self.usage["calls"] += 1
        self.usage["seconds"] += dt
        u = getattr(r, "usage", None)
        if u is not None:
            self.usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
            self.usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
        if self.trace is not None:
            calls = [it.name for it in r.output if getattr(it, "type", "") == "function_call"]
            self.trace.add(
                "system2",
                dict(kw, tools=[t["name"] for t in tools] if tools else None),
                r,
                decision=tag,
                summary=f"{role} {model}: " + (f"calls {', '.join(calls)}" if calls else "answers"),
                latency_s=dt,
            )
        return r

    def _loop(
        self,
        model: str,
        input_items: List[Any],
        max_rounds: int = 6,
        effort: Optional[str] = None,
        tag: Optional[int] = None,
        role: str = "",
    ) -> str:
        """Run tool calls until the model answers in text. `input_items` is extended in place."""
        for _ in range(max_rounds):
            r = self._create(model, list(input_items), tools=TOOLS, effort=effort, tag=tag, role=role)
            calls = [it for it in r.output if getattr(it, "type", "") == "function_call"]
            if not calls:
                return r.output_text or ""
            input_items.extend(r.output)  # reasoning items + the function calls
            for c in calls:
                try:
                    args = json.loads(c.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                t0 = time.time()
                try:
                    res = self.tools[c.name](**args)
                except Exception as e:
                    res = {"error": f"{type(e).__name__}: {e}"}
                if self.trace is not None:
                    self.trace.add(
                        "tool",
                        {"name": c.name, "arguments": args},
                        res,
                        decision=tag,
                        summary=c.name,
                        latency_s=time.time() - t0,
                    )
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": c.call_id,
                        "output": json.dumps(res, default=str)[:12000],
                    }
                )
        return "(stopped after too many tool rounds)"

    # ------------------------------------------------------------------ roles
    def chat(self, text: str) -> str:
        if not self.enabled:
            return "System 2 is off: no OpenAI key found (set OPENAI_API_KEY or OPENAI= in ~/.env)."
        self.history.append({"role": "user", "content": text})
        items: List[Any] = list(self.history[-16:])
        reply = self._loop(self.chat_model, items, effort="low", role="chat")
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def deliberate(self, context: str, effort: str = "medium", tag: Optional[int] = None) -> Dict[str, Any]:
        """A structured recommendation for the next action, grounded by tools first."""
        if not self.enabled:
            return {
                "action": "none",
                "confidence": 0.0,
                "rationale": "System 2 disabled (no key)",
                "target": "",
                "t_exp": 0,
                "n_exp": 0,
                "risk": "",
            }
        items: List[Any] = [
            {"role": "user", "content": context + "\n\nUse the tools you need, then give your recommendation."}
        ]
        notes = self._loop(self.deliberate_model, items, effort=effort, tag=tag, role="deliberate")
        items.append({"role": "assistant", "content": notes})
        items.append({"role": "user", "content": "Now return only the structured recommendation."})
        r = self._create(
            self.deliberate_model,
            items,
            effort="low",
            text_format={"type": "json_schema", "name": "recommendation", "schema": RECOMMEND_SCHEMA, "strict": True},
            tag=tag,
            role="deliberate (structured)",
        )
        try:
            rec = json.loads(r.output_text)
        except (json.JSONDecodeError, TypeError):
            rec = {
                "action": "none",
                "confidence": 0.0,
                "rationale": notes[:500],
                "target": "",
                "t_exp": 0,
                "n_exp": 0,
                "risk": "",
            }
        rec["notes"] = notes
        return rec

    def narrate(self, log_lines: List[str]) -> str:
        if not self.enabled:
            return ""
        r = self._create(
            self.chat_model,
            [{"role": "user", "content": "\n".join(log_lines[-80:])}],
            effort="low",
            instructions="Write a short observing-log entry (3-6 lines, plain text) summarising these "
            "console events for the night report. Keep numbers exact.",
            role="night log",
        )
        return r.output_text or ""
