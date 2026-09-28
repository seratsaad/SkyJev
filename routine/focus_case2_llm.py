"""Send the handed-off focus case (card 2 of the note figure) to the writing LLM once, as a writing question.

The case is WFI.2026-03-02T07-19-28.487, steps 1-5, from routine/reports/focus_qwen3-1.7b_cases.json: the
Qwen3-1.7B head gave accept 0.71 and move-beyond-last 0.27, below the gate, so SkyJev hands it on. We send the
same user prompt the Jev head saw (AnyJev build_prompt, options in the listed order) with its last line
"Answer with the letter only." replaced by a request for a short written answer, to obsassist's System 2
deliberate model (Responses API, reasoning effort low, the System 2 persona as instructions, no tools).

One call; a second only if the first raises (e.g. the model name is refused), then with the chat model.
Spend guard: max_output_tokens caps the call at the conservative prices in target-choice/reports/latency.json.

    python -m routine.focus_case2_llm      # on Pitzer, from the repo root
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "target-choice"))

from anyjev.readout import DEFAULT_SYSTEM, build_prompt  # noqa: E402
from obsassist.assistant import system2 as s2  # noqa: E402

from routine.focus import question, state_text  # noqa: E402

RUN, WINDOW = "WFI.2026-03-02T07-19-28.487", "steps 1-5"
ASK = "Explain briefly what the observer should do next and why."
BUDGET_USD = 0.10
MAX_OUT = 1200


def main() -> None:
    cases = json.loads((ROOT / "routine/reports/focus_qwen3-1.7b_cases.json").read_text())
    c = next(x for x in cases["cases"] if x["run"] == RUN and x["window"] == WINDOW)
    price = json.loads((ROOT / "target-choice/reports/latency.json").read_text())["s2"]["prices_usd_per_1e6"]
    q = question()
    jev_user = build_prompt(state_text(c["steps"], c["fwhm"], c["n_good"]), q, list(range(len(q.options)))).user
    last = "Answer with the letter only."
    assert jev_user.endswith(last)
    user = jev_user[: -len(last)] + ASK

    from openai import OpenAI

    client = OpenAI(api_key=s2.api_key())
    attempts, spent = [], 0.0
    for model in (s2.DELIBERATE_MODEL, s2.CHAT_MODEL):
        cap = (len(user + s2.PERSONA) / 2) * price["input"] / 1e6 + MAX_OUT * price["output"] / 1e6
        if spent + cap > BUDGET_USD:
            attempts.append({"model": model, "skipped": "budget"})
            break
        t0 = time.time()
        try:
            r = client.responses.create(model=model, instructions=s2.PERSONA, input=[{"role": "user", "content": user}],
                                        reasoning={"effort": "low"}, max_output_tokens=MAX_OUT)
        except Exception as e:  # noqa: BLE001
            attempts.append({"model": model, "error": f"{type(e).__name__}: {str(e)[:300]}",
                             "wall_s": round(time.time() - t0, 2)})
            continue
        dt = time.time() - t0
        u = r.usage
        tin, tout = u.input_tokens, u.output_tokens
        treason = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", None)
        usd = tin * price["input"] / 1e6 + tout * price["output"] / 1e6
        spent += usd
        attempts.append({"model": model, "wall_s": round(dt, 2), "status": r.status})
        out = {
            "case": {"run": RUN, "window": WINDOW, "steps": c["steps"], "fwhm": c["fwhm"], "n_good": c["n_good"],
                     "head_probs": c["probs"], "threshold": cases["threshold"], "key1_window_rule": "ABCD"[c["key1"]],
                     "key2_nine_step_fit": "ABCD"[c["key2"]]},
            "jev_system": DEFAULT_SYSTEM,
            "jev_user": jev_user,
            "llm_instructions": "obsassist.assistant.system2.PERSONA",
            "llm_user": user,
            "model": r.model,
            "model_requested": model,
            "reasoning_effort": "low",
            "api": "OpenAI Responses",
            "tools": None,
            "response": r.output_text,
            "wall_s": round(dt, 2),
            "input_tokens": tin,
            "output_tokens": tout,
            "reasoning_tokens": treason,
            "prices_usd_per_1e6": price,
            "est_usd": round(usd, 5),
            "attempts": attempts,
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (ROOT / "routine/reports/focus_case2_llm.json").write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps({k: out[k] for k in ("model", "wall_s", "input_tokens", "output_tokens", "est_usd")}))
        print(out["response"])
        return
    print(json.dumps(attempts))
    sys.exit(1)


if __name__ == "__main__":
    main()
