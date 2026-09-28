"""fig_skyjev_data.tex: every number and quoted text in the TikZ figure fig_skyjev.tex, read from saved reports.

    ../target-choice/reports/latency.json       Jev (System 1) per call, and the writing-LLM roles ("s2")
    ../routine/reports/focus_qwen3-1.7b.json     Jev focus decision on the archival cases, options, gate
    ../routine/reports/focus_qwen3-1.7b_cases.json   per-case widths and head probabilities
    lbt_latency.json                             real LBT requests to a writing LLM with tools
    ../routine/reports/focus_case2_llm.json      the writing LLM's answer for the handed-off case (card 2)

fig_skyjev.tex holds only the layout and reads the macros \\sj... defined here.

Run on the cluster:  python note/make_fig_skyjev.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

RUN = "WFI.2026-03-02T07-19-28.487"
SHOW = (("One", "all 9", "all nine steps"), ("Two", "steps 1-5", "steps 1--5"))
GATE = "val_target_0.98"
NAMES = ["accept", "move toward first step", "move beyond last step", "retake"]
PER_LINE = 3  # focus steps per prompt line


def load(p):
    return json.loads((ROOT / p).read_text())


def sig1(x):
    return float(f"{x:.1g}")


def tex(s):
    """Plain text from a report to LaTeX."""
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = s.replace("″", "$''$").replace("–", "--").replace("—", "---")
    for a, b in (("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("_", r"\_")):
        s = s.replace(a, b)
    return s


lat = load("target-choice/reports/latency.json")
rep = load("routine/reports/focus_qwen3-1.7b.json")
cases = load("routine/reports/focus_qwen3-1.7b_cases.json")
lbt = load("note/lbt_latency.json")["overall"]
llm = load("routine/reports/focus_case2_llm.json")

jev_lo = sig1(rep["real"]["key2"]["L2"]["s_per_decision_median"])
jev_hi = sig1(max(lat[m]["s1_judge_candidates"]["median"] for m in ("classical", "queue")))
s2 = [lat["s2"][r]["median"] for r in ("deliberate", "chat", "narrate")]
thr = rep["thresholds"][GATE]
assert abs(cases["threshold"] - thr) < 1e-9 and abs(llm["case"]["threshold"] - thr) < 1e-9

M = {
    "JevLo": f"{jev_lo:g}", "JevHi": f"{jev_hi:g}",
    "LlmLo": f"{round(min(s2))}", "LlmHi": f"{round(max(s2))}",
    "LbtMin": f"{round(lbt['wall_s_median'] / 60)}",
    "GateY": f"{thr:.4f}", "Gate": f"{thr:.2f}",
    "Date": RUN[4:14],
}
for k, o in zip("ABCD", rep["options"]):
    M[f"Opt{k}"] = tex(re.split(r", | \(", o)[0])

by_id = {(c["run"], c["window"]): c for c in cases["cases"]}
for tag, window, title in SHOW:
    c = by_id[(RUN, window)]
    p = c["probs"]
    best = max(range(4), key=lambda i: p[i])
    acts = p[best] >= thr
    if acts:
        assert best == c["key2"], "the acted example must be right against key 2"
    pairs = [f"step {s}: {w:.2f}" for s, w in zip(c["steps"], c["fwhm"])]
    rows = ["; ".join(pairs[i:i + PER_LINE]) for i in range(0, len(pairs), PER_LINE)]
    M[f"{tag}Title"] = title
    M[f"{tag}Steps"] = ";\\\\ ".join(rows) + "."
    M[f"{tag}Stars"] = f"{c['n_good']}"
    for i, k in enumerate("ABCD"):
        M[f"{tag}P{k}"] = f"{p[i]:.4f}"
        M[f"{tag}L{k}"] = f"{p[i]:.2f}"
        col = ("oiGreen" if acts else "oiOrange") if i == best else "black!30"
        M[f"{tag}C{k}"] = col
    M[f"{tag}Tag"] = f"Acts: {NAMES[best]}" if acts else "To writing LLM"
    M[f"{tag}TagCol"] = "oiGreen" if acts else "oiOrange"
    if tag == "Two":
        assert llm["case"]["run"] == RUN and llm["case"]["window"] == window and not acts

# the writing LLM's answer, trimmed: its first paragraph (the recommendation) and the second sentence of the
# explanation, with "..." where text is cut
paras = [p.strip() for p in llm["response"].strip().split("\n\n")]
expl = re.split(r"(?<=[.!?])\s+", paras[1])
M["LlmText"] = tex(paras[0]) + " \\ldots\\ " + tex(expl[1]) + "\\,\\ldots"
M["LlmTime"] = f"{llm['wall_s']:.1f}"
M["LlmModel"] = tex(llm["model_requested"])

lines = ["%% made by note/make_fig_skyjev.py from saved reports; do not edit by hand"]
lines += [f"\\def\\sj{k}{{{v}}}" for k, v in M.items()]
(HERE / "fig_skyjev_data.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
