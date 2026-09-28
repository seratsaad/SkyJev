"""Per-case probabilities of the Qwen3-4B slit head (easy version) on the 15 real LBT replay cases, for the
note figure.

slit.py does not save its head or per-case probabilities, so this rebuilds the easy splits exactly as
slit.main does (same synthetic visits, same real cases, same L2 fit through logging_eval.run_question)
and checks the result against the saved report (routine/reports/slit_easy_qwen3-4b.json): gate
thresholds, and accuracy and acted counts on test and real.

The output keeps only anonymous numbers for each real case: the candidate position angles, their mean blue
slit losses in per cent, the reference row and the head probabilities. No night, start time, target,
program, PI or readme text is written, and the private-string guard runs on the output before it is saved.

    python -m routine.slit_cases --private /fs/scratch/PAS2823/saadsm/private_lbt \
        --out routine/reports/slit_easy_qwen3-4b_cases.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from routine.logging_eval import ROOT, load_backend, private_guard, run_question, write_report
from routine.slit import Q_EASY_TEXT, REAL_ROW, ROWS, build, real_cases

SAVED = ROOT / "routine" / "reports" / "slit_easy_qwen3-4b.json"


def main(argv=None):
    from anyjev import Question

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--visits", default="logs/routine_ls/slit_visits.jsonl")
    ap.add_argument("--private", required=True)
    ap.add_argument("--out", default="routine/reports/slit_easy_qwen3-4b_cases.json")
    a = ap.parse_args(argv)

    visits = [json.loads(line) for line in open(a.visits)]
    splits, code, _ = build(visits)
    guard = private_guard(a.private)
    re_easy, _, rcode = real_cases(a.private)
    split = splits["easy"]
    split.real = re_easy
    code["easy"]["real"] = rcode
    q = Question.choice(Q_EASY_TEXT, ROWS, name="slit_row")
    be = load_backend(a.model, batch_size=8)
    rep, probs = run_question(be, "slit_easy", q, split, a.model, code=code["easy"])

    saved = json.loads(SAVED.read_text())
    check = {"saved_threshold": saved["threshold"], "refit_threshold": rep["threshold"]}
    for part in ("test", "real"):
        for g in ("gate_0.98", "gate_0.95"):
            s, r = saved[part]["L2"][g], rep[part]["L2"][g]
            for k in ("accuracy", "acted", "accuracy_when_acting", "ece"):
                check[f"{part}_{g}_{k}"] = [s[k], r[k]]
    check["all_equal"] = all(v[0] == v[1] for k, v in check.items() if isinstance(v, list)) and \
        check["saved_threshold"] == check["refit_threshold"]

    thr = rep["threshold"]["0.98"]
    cases = []
    for i, ((text, ref), p) in enumerate(zip(re_easy, probs["real"])):
        rows = [REAL_ROW.match(x.split(" = ", 1)[1]) for x in text.split("\n")[2:]]
        k = len(rows)
        cases.append({"index": i, "n_rows": k, "pa_deg": [int(m[1]) for m in rows],
                      "blue_loss_pct": [int(m[2]) for m in rows], "ref_row": int(ref),
                      "probs": [round(float(x), 4) for x in p[:k]], "probs_absent_rows": [round(float(x), 4) for x in p[k:]],
                      "pick": int(np.argmax(p)), "correct": bool(int(np.argmax(p)) == ref),
                      "acts_at_gate_0.98": bool(p.max() >= thr)})
    out = {"model": a.model, "dtype": "float16", "task": "slit_easy", "question": Q_EASY_TEXT,
           "gate": "0.98 (validation target)", "threshold": thr, "thresholds": rep["threshold"],
           "head": rep["head"], "hardware": rep["hardware"], "check_vs_saved": check,
           "real_L2": rep["real"]["L2"], "note": "anonymous numbers only; row i is the i-th PA in ascending order",
           "cases": cases}
    write_report(a.out, out, guard)


if __name__ == "__main__":
    main()
