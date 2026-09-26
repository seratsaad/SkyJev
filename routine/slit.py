"""MODS slit position angle: code computes the slit losses, the model picks a row, a gate decides.

Two versions, each one fixed four-row question ("row 1" .. "row 4"; a visit offers 2-4 PAs, listed in
ascending PA, and the rows it does not have are absent from the state):

  easy  the state is the loss table code computes with lbt_tools.optics.mods_slit_check, in the same
        format as lbt_replay/build_cases.build_slit; label = lowest mean blue loss. Code baseline:
        argmin of the computed losses (the label itself, 100 % by construction).
  hard  the state is a readme-style rule (UT or hour-angle boundaries), the start time in MST and UT,
        the hour angle at the start, and the scripts on offer, with no loss table; label = the row the
        rule gives. Code baseline: the rule applied by code (100 % by construction). The rules are
        written from the losses (routine/slit_data.py), so on the synthetic cases rule and argmin agree.

Synthetic visits come from routine/slit_data.py. Train on nights not in the evaluation nights (every
third night of the list); val and test on the evaluation nights, and the hard rules of val/test use a
different wording (family B) from train (family A). Real cases: the 15 LBT replay slit cases
(private_lbt/cases/slit.jsonl), re-rendered with row labels; for the hard version the state shows the
program's own readme text (read inside the job, never written out) and the reference stays the
lowest-loss PA, so there it measures whether the readme plus the model reach the physics answer.

    python -m routine.slit --model Qwen/Qwen3-4B --visits logs/routine_ls/slit_visits.jsonl \
        --private /fs/scratch/PAS2823/saadsm/private_lbt --out-prefix routine/reports/slit_
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
from pathlib import Path

import numpy as np

from routine.common import Split
from routine.slit_data import lbt, nights

ROWS = ["row 1", "row 2", "row 3", "row 4"]
Q_EASY_TEXT = "Which row of the table has the lowest blue slit loss for this visit?"
Q_HARD_TEXT = "Following the readme, which row's acquisition script should be used for this visit?"
REAL_ROW = re.compile(r"^PA ([+-]\d+) deg: blue \(0\.35-0\.45 um\) loss (-?\d+) %, red \(0\.65-0\.95 um\) loss (-?\d+) %, "
                      r"worst blue step keeps (-?\d+) %$")


def header(night, start, hours, width, seeing, parang) -> str:
    return (f"MODS long-slit visit on night {night}: start {start} MST, {hours} h, slit {width} arcsec, "
            f"seeing {seeing} arcsec, guiding at 0.65 um, no ADC. Parallactic angle at mid-visit {parang:+.0f} deg.")


def table_line(i: int, pa: int, blue: float, red: float, worst: float) -> str:
    """blue, red, worst in per cent."""
    return (f"row {i + 1} = PA {pa:+d} deg: blue (0.35-0.45 um) loss {blue:.0f} %, red (0.65-0.95 um) loss {red:.0f} %, "
            f"worst blue step keeps {worst:.0f} %")


def easy_state(v: dict) -> str:
    lines = [header(v["night"], v["start"], v["hours"], v["width"], v["seeing"], v["table"][0]["parang_mid"]),
             "Slit losses from atmospheric dispersion over the visit (mean relative to a centred star) for each acquisition script:"]
    lines += [table_line(i, r["pa"], 100 * r["blue_loss"], 100 * r["red_loss"], 100 * r["worst_blue"]) for i, r in enumerate(v["table"])]
    return "\n".join(lines)


def script(pa: int, style: int, tag: str) -> str:
    return [f"pa{pa}.acq", f"pa{pa:+d}.acq", f"{tag}_pa{pa}.acq"][style]


def rule_text(v: dict, family: str, ha: bool, names: list) -> str:
    segs = v["rule"]
    s = [names[g["pa_index"]] for g in segs]
    b = [g["from_ha"] if ha else g["from_ut"] for g in segs]
    fmt = (lambda x: f"{x:+.2f} h") if ha else (lambda x: f"{x} UT")
    n = len(segs)
    if family == "A":
        if n == 1:
            return f"Use {s[0]} for the whole night."
        cond = "the hour angle at the start is" if ha else "the visit starts"
        parts = [f"Use {s[0]} if {cond} before {fmt(b[1])}"]
        parts += [f"{s[i]} from {fmt(b[i])} to {fmt(b[i + 1])}" for i in range(1, n - 1)]
        parts.append(f"and {s[-1]} at or after {fmt(b[-1])}.")
        return ", ".join(parts)
    if n == 1:
        return f"Any hour angle: {s[0]}." if ha else f"{s[0]} at any time; no PA change is needed."
    if ha:
        lines = [f"HA < {fmt(b[1])} -> {s[0]}"] + [f"{fmt(b[i])} <= HA < {fmt(b[i + 1])} -> {s[i]}" for i in range(1, n - 1)]
        lines.append(f"HA >= {fmt(b[-1])} -> {s[-1]}")
    else:
        lines = [f"Starting before {fmt(b[1])}: acquire with {s[0]}."]
        lines += [f"Starting {b[i]}-{b[i + 1]} UT: {s[i]}." for i in range(1, n - 1)]
        lines.append(f"Starting {fmt(b[-1])} or later: {s[-1]}.")
    return "\n".join(lines)


def hard_head(night, start, start_ut, ha_start, hours, width, seeing) -> str:
    return (f"MODS long-slit visit on night {night}: start {start} MST ({start_ut} UT), {hours} h, slit {width} arcsec, "
            f"seeing {seeing} arcsec. Hour angle of the target at the start: {ha_start:+.2f} h.")


def hard_state(v: dict, family: str) -> str:
    rng = np.random.default_rng(v["seed"])
    style, ha = int(rng.integers(3)), bool(rng.random() < 0.4)
    names = [script(pa, style, "sci") for pa in v["pas"]]
    intro = "The program readme says:" if family == "A" else "From the program readme:"
    lines = [hard_head(v["night"], v["start"], v["start_ut"], v["ha_start"], v["hours"], v["width"], v["seeing"]),
             intro, rule_text(v, family, ha, names), "Acquisition scripts on offer (one per row):"]
    lines += [f"row {i + 1} = PA {pa:+d} deg ({n})" for i, (pa, n) in enumerate(zip(v["pas"], names))]
    return "\n".join(lines)


def build(visits: list):
    """Easy and hard splits from the synthetic visits, the code baselines, and data notes."""
    from collections import Counter
    ev = set(nights()[::3])
    tr = [v for v in visits if v["night"] not in ev]
    val = [v for v in visits if v["night"] in ev and v["seed"] % 2 == 0]
    test = [v for v in visits if v["night"] in ev and v["seed"] % 2 == 1]
    ok = lambda vs: [v for v in vs if not v["tie"]]  # noqa: E731  (rounded best loss equals the runner-up: no visible answer)
    easy = Split([(easy_state(v), v["best"]) for v in ok(tr)], [(easy_state(v), v["best"]) for v in ok(val)],
                 [(easy_state(v), v["best"]) for v in ok(test)])
    hard = Split([(hard_state(v, "A"), v["rule_pick"]) for v in tr], [(hard_state(v, "B"), v["rule_pick"]) for v in val],
                 [(hard_state(v, "B"), v["rule_pick"]) for v in test])
    common = {"n_visits": len(visits), "code_s_per_visit_median": round(float(np.median([v["code_seconds"] for v in visits])), 4),
              "fast_path_vs_mods_slit_check_max_diff": max(v["fast_check_diff"] for v in visits),
              "test_n_pas": dict(Counter(len(v["pas"]) for v in test))}
    info = {"easy": dict(common, ties_dropped={"train": len(tr) - len(ok(tr)), "val": len(val) - len(ok(val)),
                                               "test": len(test) - len(ok(test))}),
            "hard": dict(common, test_rule_segments=dict(Counter(len(v["rule"]) for v in test)),
                         test_rule_pick_equals_argmin=round(float(np.mean([v["rule_pick"] == v["best"] for v in test])), 4))}
    code = {"easy": {"test": [v["best"] for v in ok(test)]}, "hard": {"test": [v["rule_pick"] for v in test]}}
    return {"easy": easy, "hard": hard}, code, info


def real_cases(private: str):
    """The 15 replay slit cases: easy states re-rendered with row labels; hard states with the program
    readme, the start in MST and UT and the hour angle at the start. Kept in memory only."""
    from routine.slit_data import hhmm
    P = Path(private)
    cases = [json.loads(line) for line in open(P / "cases" / "slit.jsonl")]
    ann = {c["id"]: c for c in json.loads((P / "ann_slit.json").read_text())["cases"]}
    objs = {}
    for f in sorted((P / "src" / "data" / "cache").glob("queue_*.json")):
        for t in json.loads(f.read_text())["targets"]:
            objs.setdefault(t["object"], t)
    ephem, _ = lbt()
    easy, hard, code = [], [], []
    for c in cases:
        lines = c["state"].split("\n")
        rows = [REAL_ROW.match(x) for x in lines[2:]]
        assert all(rows) and len(rows) == len(c["options"]), [re.sub(r"\d", "9", x) for x in lines[2:]]
        pas = [int(m[1]) for m in rows]
        easy.append(("\n".join(lines[:2] + [table_line(i, pa, float(m[2]), float(m[3]), float(m[4]))
                                             for i, (pa, m) in enumerate(zip(pas, rows))]), c["ref"][0]))
        code.append(c["code_pick"])
        a = ann[c["id"]]
        t, prog = objs[a["object"]], a["object"].split("/")[0]
        rd = P / "src" / "inputs" / "00Readme"
        found = sorted(rd.glob(f"{glob.escape(prog)}*.readme")) + sorted(rd.glob(f"{glob.escape(prog)}*"))
        text = found[0].read_text(errors="replace") if found else ""
        fname = {int(n): m.group(0) for m in re.finditer(r"[\w.+-]*?pa([+-]?\d+)\.acq", text, re.I) for n in [m.group(1)]}
        hh, mm = (int(x) for x in a["start"].split(":"))
        t0 = hh + mm / 60 + (24 if hh < 12 else 0)
        r0 = ephem.target_track(dt.date.fromisoformat(a["night"]), t["ra"], t["dec"], step_min=10,
                                start_hour=t0, end_hour=t0)["rows"][0]
        st = [hard_head(a["night"], a["start"], hhmm(t0 + 7), r0["ha_h"], a["hours"], a["width"], a["seeing"]),
              "From the program readme:", text.strip(), "Acquisition scripts on offer (one per row):"]
        st += [f"row {i + 1} = PA {pa:+d} deg ({fname.get(pa, f'pa{pa}.acq')})" for i, pa in enumerate(pas)]
        hard.append(("\n".join(st), c["ref"][0]))
    return easy, hard, code


def main(argv=None):
    from anyjev import Question
    from routine.logging_eval import load_backend, private_guard, run_question, write_report

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--visits", default="logs/routine_ls/slit_visits.jsonl")
    ap.add_argument("--private", default=None, help="private_lbt directory (real cases); omit to skip them")
    ap.add_argument("--out-prefix", default="routine/reports/slit_")
    ap.add_argument("--versions", default="easy,hard")
    a = ap.parse_args(argv)
    visits = [json.loads(line) for line in open(a.visits)]
    splits, code, info = build(visits)
    guard = private_guard(a.private)
    if a.private:
        re_easy, re_hard, rcode = real_cases(a.private)
        splits["easy"].real, splits["hard"].real = re_easy, re_hard
        code["easy"]["real"] = rcode
        info["hard"]["real_reference"] = "lowest mean blue loss (the replay reference); the readme text is the program's own"
    qs = {"easy": Question.choice(Q_EASY_TEXT, ROWS, name="slit_row"),
          "hard": Question.choice(Q_HARD_TEXT, ROWS, name="slit_rule_row")}
    be = load_backend(a.model, batch_size=8)
    tag = a.model.split("/")[-1].lower()
    for name in a.versions.split(","):
        rep, _ = run_question(be, f"slit_{name}", qs[name], splits[name], a.model, code=code[name],
                              notes=__doc__.split("\n\n")[1].strip() if name == "easy" else "")
        rep["data"] = info[name]
        write_report(f"{a.out_prefix}{name}_{tag}.json", rep, guard)


if __name__ == "__main__":
    main()
