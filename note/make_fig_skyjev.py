"""fig_skyjev_data.tex: every number and quoted text in the TikZ figure fig_skyjev.tex, read from saved reports.

    ../target-choice/reports/latency.json       Jev (System 1) per call, and the writing-LLM roles ("s2")
    ../routine/reports/focus_qwen3-1.7b.json     Jev focus decision on the archival cases, options, gate
    ../routine/reports/focus_qwen3-1.7b_cases.json   per-case widths and head probabilities (card 1)
    ../routine/reports/focus_case2_llm.json      the focus prompt as the Jev head read it (card 1 wording)
    lbt_latency.json                             real LBT requests to a writing LLM with tools
    ../routine/reports/slit_easy_qwen3-4b_cases.json  anonymous slit-loss tables and head probabilities of
                                                 the 15 LBT replay cases (card 2); the gate is the saved one
    ../routine/reports/target_card.json          the simulated queue state, planner numbers, P(best), gate (card 3)
    ../routine/reports/target_card_llm.json      the writing LLM's answer for that state (card 3)

fig_skyjev.tex holds only the layout and reads the macros \\sj... defined here.

Run on the cluster:  python note/make_fig_skyjev.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

RUN, WINDOW = "WFI.2026-03-02T07-19-28.487", "all 9"
GATE = "val_target_0.98"
DX = 0.64  # bar spacing in the charts (cm)


def load(p):
    return json.loads((ROOT / p).read_text())


def sig1(x):
    return float(f"{x:.1g}")


def tex(s):
    """Plain text from a report to LaTeX."""
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = s.replace("″", "$''$").replace('"', "$''$").replace("–", "--").replace("—", "---").replace("’", "'")
    s = s.replace("<", "$<$").replace(">", "$>$")
    for a, b in (("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("_", r"\_")):
        s = s.replace(a, b)
    return s


def signed(n):
    return f"${n:+d}$"


def bars(labels, probs, best, colour, gate):
    """Foreach list x/label/p/value label/colour/inside for the chart: the top bar in `colour`, the rest grey;
    inside = 1 (white) or 2 (black) puts the value label inside the bar top, where above the bar it would cross the gate line."""
    items = []
    for i, (lab, p) in enumerate(zip(labels, probs)):
        col = colour if i == best else "black!30"
        inside = (1 if i == best else 2) if gate - 0.2 < p < gate else 0
        items.append(f"{DX * (i + 0.8):.2f}/{lab}/{p:.4f}/{p:.2f}/{col}/{inside}")
    return ", ".join(items), f"{DX * (len(probs) + 0.6):.2f}"


def gate_macros(M, tag, thr, label=None):
    M[f"{tag}GateY"] = f"{thr:.4f}"
    M[f"{tag}GateLab"] = label or f"gate {thr:.2f}"


lat = load("target-choice/reports/latency.json")
rep = load("routine/reports/focus_qwen3-1.7b.json")
cases = load("routine/reports/focus_qwen3-1.7b_cases.json")
lbt = load("note/lbt_latency.json")["overall"]
foc_llm = load("routine/reports/focus_case2_llm.json")
slit = load("routine/reports/slit_easy_qwen3-4b_cases.json")
slit_saved = load("routine/reports/slit_easy_qwen3-4b.json")
card = load("routine/reports/target_card.json")
tllm = load("routine/reports/target_card_llm.json")

jev_lo = sig1(rep["real"]["key2"]["L2"]["s_per_decision_median"])
jev_hi = sig1(max(lat[m]["s1_judge_candidates"]["median"] for m in ("classical", "queue")))
s2 = [lat["s2"][r]["median"] for r in ("deliberate", "chat", "narrate")]

M = {
    "JevLo": f"{jev_lo:g}", "JevHi": f"{jev_hi:g}",
    "LlmLo": f"{round(min(s2))}", "LlmHi": f"{round(max(s2))}",
    "LbtMin": f"{round(lbt['wall_s_median'] / 60)}",
}

# ---------------------------------------------------------------- card 1: focus, WFI archival run, SkyJev acts
thr = rep["thresholds"][GATE]
assert abs(cases["threshold"] - thr) < 1e-9
c = next(x for x in cases["cases"] if x["run"] == RUN and x["window"] == WINDOW)
p = c["probs"]
best = max(range(4), key=lambda i: p[i])
assert p[best] >= thr and best == c["key2"], "card 1 must be an acted, correct case"
user = foc_llm["jev_user"]
lead = re.search(r"(Median star FWHM in arcsec) at each step:", user)[1]
quest = re.search(r"^Question: (.+)$", user, re.M)[0]


def short(o):
    """'move: best focus lies before the first step, ...' -> 'move before first step'; 'accept: ...' -> 'accept'."""
    head, rest = o.split(": ", 1)
    if head != "move":
        return head
    return "move " + rest.split(",")[0].replace("best focus lies ", "").replace("the ", "")


opts = [f"{k}. {short(o)}" for k, o in zip("ABCD", rep["options"])]
steps = ", ".join(f"{w:.2f}" for w in c["fwhm"])
M["OneHead"] = f"WFI night {RUN[4:14]}, telescope focus"
M["OnePrompt"] = (f"\\ldots\\ {lead}, steps {c['steps'][0]}--{c['steps'][-1]}:\\\\ {steps}.\\\\ "
                  f"Stars measured: {c['n_good']}.\\\\ {quest}\\\\ "
                  f"{opts[0]}\\quad {opts[1]}\\\\ {opts[2]}\\quad {opts[3]}")
M["OneBars"], M["OneW"] = bars("ABCD", p, best, "oiGreen", thr)
gate_macros(M, "One", thr)
M["OneYlab"] = "model probability"
M["OneTag"] = "Acts: " + short(rep["options"][best])
M["OneTagCol"] = "oiGreen"

# ---------------------------------------------------------------- card 2: MODS slit angle, LBT replay, SkyJev acts
sthr = slit["threshold"]
assert slit["check_vs_saved"]["all_equal"] and sthr == slit_saved["threshold"]["0.98"]
ok = [x for x in slit["cases"] if x["correct"] and x["acts_at_gate_0.98"] and x["n_rows"] >= 3]
# the closest call among them: smallest gap between the lowest and the next-lowest blue loss
s = min(ok, key=lambda x: (sorted(x["blue_loss_pct"])[1] - sorted(x["blue_loss_pct"])[0], x["index"]))
sb = s["pick"]
rows = [f"row {i + 1} = PA {signed(pa)} deg: blue (0.35--0.45\\,$\\mu$m) loss {bl}\\,\\%"
        for i, (pa, bl) in enumerate(zip(s["pa_deg"], s["blue_loss_pct"]))]
M["TwoHead"] = "LBT replay, MODS slit angle"
M["TwoPrompt"] = ("\\ldots\\ Slit losses \\ldots\\ for each acquisition script:\\\\ "
                  + "\\\\ ".join(rows) + f"\\\\ Question: {tex(slit['question'])}")
M["TwoBars"], M["TwoW"] = bars([str(i + 1) for i in range(s["n_rows"])], s["probs"], sb, "oiGreen", sthr)
gate_macros(M, "Two", sthr)
M["TwoYlab"] = "model probability"
M["TwoTag"] = f"Acts: row {sb + 1} (PA {signed(s['pa_deg'][sb])})"
M["TwoTagCol"] = "oiGreen"

# ---------------------------------------------------------------- card 3: next target, simulated queue night, handed on
cs = card["candidates"]
pb = [x["p_best"] for x in cs]
tb = max(range(len(cs)), key=lambda i: pb[i])
assert not card["acts"] and card["max_abs_p_best_diff_vs_saved"] == 0
g3 = card["gate_used"]
first = card["state"]["rendered"].split("\n")[0]
seeing = re.search(r"Seeing [\d.]+\"", first)[0]
on = re.search(r"On (\S+)\.", card["state"]["rendered"])[1]
tab = ["\\begin{tabular}{@{}l@{\\ }l@{\\ \\ }r@{\\ \\ }l@{}}",
       " & priority & min to finish & airmass\\\\"]
for x in cs:
    tab.append(f"{x['name']} & P{x['priority']} & {x['min_to_finish']} & "
               f"{x['airmass']:.2f} {'rising' if x['rising'] else 'setting'}\\\\")
tab.append("\\end{tabular}")
M["ThrHead"] = "Simulated queue night, next target"
M["ThrPrompt"] = (f"{tex(first.split(', ')[0])} \\ldots\\ {tex(seeing)} \\ldots\\ On {on}.\\\\ " + "".join(tab)
                  + f"\\\\ {tex(card['focus_line'])} "
                  + f"Question: {tex(card['question'])}")
mid = [x["name"].split("-")[1] for x in cs]
assert len(set(mid)) == len(mid)
M["ThrBars"], M["ThrW"] = bars(mid, pb, tb, "oiOrange", g3)
gate_macros(M, "Thr", g3, f"fixed gate {g3:.2f}" if card["gate_val_target_0.98"] > 1 else None)
M["ThrYlab"] = "model probability"
M["ThrTag"] = "To writing LLM"
M["ThrTagCol"] = "oiOrange"

# the writing LLM's answer, trimmed: its bold recommendation, the first sentence of the explanation and the last
# sentence, with "..." where text is cut
paras = [x.strip() for x in tllm["response"].strip().split("\n\n")]
rec = re.match(r"\*\*(.+?)\*\*", paras[0])[1].rstrip(":. ")
sents = re.split(r"(?<=[.!?])\s+", paras[-1])
M["LlmText"] = (f"\\textbf{{{tex(rec)}}}\\,\\ldots\\ {tex(sents[0])} \\ldots\\ {tex(sents[-1])}")
M["LlmTime"] = f"{tllm['wall_s']:.1f}"
M["LlmModel"] = tex(tllm["model_requested"])

lines = ["%% made by note/make_fig_skyjev.py from saved reports; do not edit by hand"]
lines += [f"\\def\\sj{k}{{{v}}}" for k, v in M.items()]
(HERE / "fig_skyjev_data.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print("card 2 case index", s["index"], "probs", s["probs"], "threshold", sthr)
print("card 3 LLM pick", tllm["llm_pick_first_named"], "planner", card["planner_pick"],
      "hindsight best", card["best_in_hindsight"], "regrets", {x["name"]: x["hindsight_regret"] for x in cs})
