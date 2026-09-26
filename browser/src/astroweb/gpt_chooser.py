"""Third arm: JEV's exact action space, an ordinary LLM as the chooser.

Why this exists. Comparing JEV against Browser Use changes two things at once: the page
representation (indexed action space vs. a general agent's own observation) and the decision model
(a specialised chooser vs. a large LLM). This arm holds the representation fixed and swaps only the
model, so the two effects can be separated:

    jev        JEV element table  +  TypeSafe chooser      (needs TYPESAFE_API_KEY)
    gpt-jev    JEV element table  +  GPT structured output  (needs OPENAI_API_KEY)
    baseline   Browser Use        +  GPT

It is a drop-in for `jev_ultrafast.model.choose`, so the loop, the executor, the freshness guards
and the instrumentation are all JEV's own, unchanged. The model still never emits selectors,
coordinates or JavaScript: it returns an operation name and an offered element index, exactly like
the TypeSafe head, and an unknown index is rejected before anything touches the browser.
"""

from __future__ import annotations

import json
import os
import time

import jev_ultrafast.model as jev_model
from jev_ultrafast.model import action_space
from jev_ultrafast.questions import NEXT_ACTION, TARGET

SYSTEM = (
    "You drive a browser by choosing one operation and one target from the offered lists.\n"
    + NEXT_ACTION
    + "\n"
    + TARGET
    + "\nAnswer with JSON: {\"operation\": <one offered operation>, \"target\": <one offered target index"
    " or null>, \"confidence\": <0..1>}. Choose a target only for CLICK, TYPE_TEXT or SELECT, and only"
    " an index that appears in the offered list for that operation."
)

# Two ways to ask the same model the same question.
#
#   free         (default, and what the benchmark measured) a JSON object the model writes itself.
#                Nothing stops it writing a target that is not on the list; 4 of 45 benchmark runs
#                ended that way, each refused before anything touched the browser.
#   constrained  OpenAI structured outputs with the offered operations and targets as enums, so the
#                decoder cannot produce an answer that is not on the menu. This is the difference
#                between generating an answer and choosing one, which is the point of the talk.
#
# Set GPT_CHOOSER_MODE=constrained. The live inspector uses it; the benchmark results do not.
MODE = os.environ.get("GPT_CHOOSER_MODE", "free")

# What the chooser is shown of each past action. The benchmark used these four; jev_plus adds "url".
RECENT_KEYS = ("action", "kind", "text", "page_changed")


def decision_schema(operations: dict, targets: dict) -> dict:
    """A JSON schema whose only valid instances are offered (operation, target) pairs."""
    branches = [
        {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": [operation]},
                "target": {"type": "string", "enum": list(candidates)},
            },
            "required": ["operation", "target"],
            "additionalProperties": False,
        }
        for operation, candidates in targets.items()
        if candidates
    ]
    targetless = [operation for operation in operations if operation not in targets]
    if targetless:
        branches.append({
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": targetless},
                "target": {"type": "null"},
            },
            "required": ["operation", "target"],
            "additionalProperties": False,
        })
    return {
        "type": "object",
        "properties": {"decision": {"anyOf": branches}, "confidence": {"type": "number"}},
        "required": ["decision", "confidence"],
        "additionalProperties": False,
    }


def choose_with_gpt(state, goal, history):
    """Same signature and return shape as jev_ultrafast.model.choose."""
    elements, targets, controls = action_space(state["actions"])
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")

    offered = {
        operation: [
            {"index": index, "element": f"[{index}] {a['label']}", "value": a.get("current_value", a.get("value", ""))}
            for index, a in candidates.items()
        ]
        for operation, candidates in targets.items()
    }
    payload = {
        "goal": goal,
        "page": {k: state[k] for k in ("url", "title", "text")},
        "elements": elements,
        "recent_actions": [{k: h.get(k) for k in RECENT_KEYS} for h in history[-10:]],
        "offered_operations": operations,
        "offered_targets": offered,
    }
    if state.get("agent_notes"):
        # Only jev_plus sets this: facts about the run the page itself does not show (a refused choice).
        payload["notes"] = state["agent_notes"]
    model = os.environ.get("GPT_CHOOSER_MODEL", "gpt-4.1-mini")
    constrained = os.environ.get("GPT_CHOOSER_MODE", MODE) == "constrained"
    # Structured outputs cap the number of enum values in one schema. A page with more offered targets
    # than that (VizieR with every dropdown option) falls back to the free answer, checked below.
    if constrained and sum(len(c) for c in targets.values()) > 900:
        constrained = False
    response_format = (
        {"type": "json_schema",
         "json_schema": {"name": "decision", "strict": True, "schema": decision_schema(operations, targets)}}
        if constrained else {"type": "json_object"}
    )
    system = SYSTEM + (
        '\nReturn {"decision": {"operation": ..., "target": ...}, "confidence": ...}.' if constrained else ""
    )
    started = time.perf_counter()
    result = jev_model.post_json(
        os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions",
        os.environ["OPENAI_API_KEY"],
        {
            "model": model,
            "max_tokens": 300,
            "response_format": response_format,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload)[:120000]},
            ],
        },
    )
    latency_ms = round((time.perf_counter() - started) * 1000)
    try:
        answer = json.loads(result["choices"][0]["message"]["content"])
        decision = answer["decision"] if constrained else answer
        operation = decision["operation"]
        target = decision.get("target")
        confidence = float(answer.get("confidence", 0.5))
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"Chooser returned no usable answer ({e}); no action executed.") from None

    if operation not in operations:
        raise ValueError(f"Chooser picked unoffered operation {operation!r}; no action executed.")
    probabilities = {}
    if operation in targets:
        target = str(target)
        if target not in targets[operation]:
            raise ValueError(f"Chooser picked unoffered target {target!r} for {operation}; no action executed.")
        choice = targets[operation][target]["id"]
        probabilities = {a["id"]: (1.0 if index == target else 0.0) for index, a in targets[operation].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        target = None
        probabilities = {choice: 1.0}
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": confidence,
        "probabilities": probabilities,
        "operation_probabilities": {operation: 1.0},
        "target_probabilities": {},
        "target_confidence": None,
        "raw_answers": {},
        "model": model,
        "usage": result.get("usage", {}),
        "latency_ms": latency_ms,
        "request": {"state": {"elements": elements}},
    }


def install():
    """Swap the chooser inside JEV's loop. Everything else stays JEV's own code."""
    import jev_ultrafast.agent as jev_agent

    jev_agent.choose = choose_with_gpt
    return choose_with_gpt
