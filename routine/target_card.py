"""Next-target card of the note figure: one held-out simulated queue state that SkyJev hands on, and the
writing LLM's answer for it.

Two steps, both on Pitzer:

  judge   (GPU) pick the state from the saved gate run (routine/reports/target_gate_qwen3-4b_queue_states.json,
          test nights): among test states with three candidates whose hindsight regrets are not all equal, the
          one with the highest top P(best). Rerun System1.judge_candidates (Qwen3-4B, the routine queue heads)
          on it, check P(best) against the saved run, and write routine/reports/target_card.json with the
          rendered state, the planner numbers per candidate, P(best), hindsight regret, the planner's pick and
          the gate (validation threshold for 98 %, or the fixed 0.95 when no threshold reaches it).
  llm     (no GPU) send the same rendered state and the question below to the writing LLM once (Responses API,
          reasoning effort low, the System 2 persona, no tools) and write routine/reports/target_card_llm.json
          with the answer, wall time, tokens and cost.

    sbatch logs/fig_cards.sbatch                    # judge (and routine.slit_cases)
    python -m routine.target_card llm               # one API call
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "target-choice"))

REP = ROOT / "routine" / "reports"
STATES = REP / "target_gate_qwen3-4b_queue_states.json"
GATE = REP / "target_gate_qwen3-4b_queue.json"
HEADS = REP / "target_heads_qwen3-4b_queue.json"
TEST = ROOT / "target-choice" / "data" / "decisions" / "queue_test.jsonl"
CARD = REP / "target_card.json"
OUT = REP / "target_card_llm.json"
ASK = "Which candidate should be observed next and why? Answer briefly."
FIXED_GATE = 0.95
BUDGET_USD = 0.10
MAX_OUT = 1200


def pick(recs):
    ok = [r for r in recs if len(r["names"]) == 3 and len(set(r["regret"])) > 1]
    return max(ok, key=lambda r: max(r["p_best"]))


def judge() -> None:
    import torch
    import transformers
    from obsassist.assistant.observation import render
    from obsassist.assistant.system1 import System1
    from obsassist.learn.fit import load_rows

    d = json.loads(STATES.read_text())
    r = pick(d["test"])
    rows = load_rows(str(TEST))[(r["night"], r["t"])]
    obs = rows[0]["obs"]
    s1 = System1(model=d["model"], heads=str(HEADS), device="cuda", dtype="float16")
    assert s1.level_of("regret") == "L2"
    s1.judge_candidates(obs, r["names"])  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    js = s1.judge_candidates(obs, r["names"])
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    gate = json.loads(GATE.read_text())["thresholds"]
    thr = gate["val_target_0.98"]
    by = {c["name"]: c for c in obs["candidates"]}
    cands = []
    for n, j, pb_saved, reg in zip(r["names"], js, r["p_best"], r["regret"]):
        c = by[n]
        cands.append({"name": n, "priority": c["priority"], "min_to_finish": round(c["time_needed_min"]),
                      "window_left_min": round(c["window_left_min"]), "airmass": round(c["airmass"], 2),
                      "rising": c["rising"], "is_current": c["is_current"],
                      "p_best": round(j.p_best, 4), "p_best_saved": round(pb_saved, 4),
                      "expected_regret": round(j.expected_regret, 4), "level": j.level,
                      "hindsight_regret": reg})
    top = max(cands, key=lambda c: c["p_best"])
    from obsassist.assistant.system1 import QUESTIONS

    out = {"state": {"night": r["night"], "t": r["t"], "test_file": TEST.name, "rendered": render(obs)},
           "question": QUESTIONS["regret"].text, "question_options": list(QUESTIONS["regret"].options),
           "focus_line": render(obs, focus=r["names"][0]).split("\n")[-1],
           "selection": "test states with 3 candidates and unequal hindsight regrets; highest top P(best)",
           "model": d["model"], "heads": str(HEADS.relative_to(ROOT)), "dtype": "float16",
           "candidates": cands, "skyjev_top": top["name"], "top_p_best": top["p_best"],
           "planner_pick": r["greedy"], "queue_rule_pick": r["queue_rule"],
           "best_in_hindsight": [c["name"] for c in cands if c["hindsight_regret"] == min(r["regret"])],
           "gate_val_target_0.98": thr, "gate_used": thr if thr <= 1 else FIXED_GATE,
           "gate_note": "no validation threshold reaches 98 %; fixed 0.95 used" if thr > 1 else "validation threshold",
           "acts": top["p_best"] >= (thr if thr <= 1 else FIXED_GATE),
           "max_abs_p_best_diff_vs_saved": round(max(abs(c["p_best"] - c["p_best_saved"]) for c in cands), 4),
           "s_judge": round(secs, 4),
           "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                        "transformers": transformers.__version__}}
    CARD.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def llm() -> None:
    from obsassist.assistant import system2 as s2

    card = json.loads(CARD.read_text())
    price = json.loads((ROOT / "target-choice/reports/latency.json").read_text())["s2"]["prices_usd_per_1e6"]
    user = card["state"]["rendered"] + "\n\n" + ASK
    names = [c["name"] for c in card["candidates"]]
    model = s2.DELIBERATE_MODEL
    cap = (len(user + s2.PERSONA) / 2) * price["input"] / 1e6 + MAX_OUT * price["output"] / 1e6
    assert cap <= BUDGET_USD, cap
    from openai import OpenAI

    client = OpenAI(api_key=s2.api_key())
    t0 = time.time()
    r = client.responses.create(model=model, instructions=s2.PERSONA, input=[{"role": "user", "content": user}],
                                reasoning={"effort": "low"}, max_output_tokens=MAX_OUT)
    dt = time.time() - t0
    u = r.usage
    treason = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", None)
    usd = u.input_tokens * price["input"] / 1e6 + u.output_tokens * price["output"] / 1e6
    out = {"state": {"night": card["state"]["night"], "t": card["state"]["t"]},
           "llm_instructions": "obsassist.assistant.system2.PERSONA", "llm_user": user,
           "model": r.model, "model_requested": model, "reasoning_effort": "low", "api": "OpenAI Responses",
           "tools": None, "status": r.status, "response": r.output_text, "wall_s": round(dt, 2),
           "input_tokens": u.input_tokens, "output_tokens": u.output_tokens, "reasoning_tokens": treason,
           "prices_usd_per_1e6": price, "est_usd": round(usd, 5),
           "llm_pick_first_named": min((n for n in names if n in r.output_text), key=r.output_text.index, default=None),
           "planner_pick": card["planner_pick"], "skyjev_top": card["skyjev_top"],
           "best_in_hindsight": card["best_in_hindsight"],
           "hindsight_regret": {c["name"]: c["hindsight_regret"] for c in card["candidates"]},
           "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("model", "wall_s", "input_tokens", "output_tokens", "est_usd")}))
    print(out["response"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["judge", "llm"])
    {"judge": judge, "llm": llm}[ap.parse_args().step]()
