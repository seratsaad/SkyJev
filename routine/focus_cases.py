"""Per-case probabilities of the Qwen3-1.7B focus head on the 84 real WFI cases, for the note figure.

focus.py does not save its head, so this refits it exactly as focus.run does (same synthetic seeds
1 and 2 for train and validation, same L2 fit), re-picks the gate on validation, and reads the
L2 probabilities on the 84 archival cases. It then checks the result against the saved report
(routine/reports/focus_qwen3-1.7b.json): threshold, acted count and accuracy on key 2, and the
per-case probabilities. The output holds only numbers and the case index and run id.

    python -m routine.focus_cases --out routine/reports/focus_qwen3-1.7b_cases.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from routine.common import _probs, backend, score
from routine.focus import ROOT, as_case, code_rule, gates, question, real_cases, synthetic

SAVED = ROOT / "routine" / "reports" / "focus_qwen3-1.7b.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float16")
    a = ap.parse_args()

    import torch
    import transformers
    from anyjev import Decider

    q = question()
    tr, va = synthetic(600, 1), synthetic(250, 2)
    real = real_cases()
    be = backend(a.model, a.dtype)
    head = Decider(be, level="L2")
    t0 = time.time()
    art = head.fit_head(q, [as_case(c)[0] for c in tr], [as_case(c)[1] for c in tr])
    fit_s = time.time() - t0
    pv, _ = _probs(head, q, [as_case(c) for c in va], "L2")
    yv = np.array([c["label"] for c in va])
    thr = gates(pv, yv)
    pr, s_med = _probs(head, q, [as_case(c, "key2") for c in real], "L2")
    y2 = np.array([c["key2"] for c in real])
    gate = "val_target_0.98"
    sc = score(pr, y2, thr[gate])

    saved = json.loads(SAVED.read_text())
    sp = np.array([c["L2"] for c in saved["real_cases"]])
    ss = saved["real"]["key2"]["L2"][gate]
    check = {"saved_threshold": saved["thresholds"][gate], "refit_threshold": thr[gate],
             "saved_acted": ss["acted"], "refit_acted": sc["acted"],
             "saved_accuracy_when_acting": ss["accuracy_when_acting"],
             "refit_accuracy_when_acting": sc["accuracy_when_acting"],
             "saved_accuracy": ss["accuracy"], "refit_accuracy": sc["accuracy"],
             "max_abs_prob_diff_vs_saved": round(float(np.abs(np.round(pr, 4) - sp).max()), 4),
             "same_argmax_all": bool((pr.argmax(1) == sp.argmax(1)).all()),
             "same_acted_set": bool(((pr.max(1) >= thr[gate]) == (sp.max(1) >= saved["thresholds"][gate])).all())}

    rep = {"model": a.model, "dtype": a.dtype, "gate": gate, "threshold": thr[gate], "thresholds": thr,
           "head": {"layer": art.get("layer_abs"), "kind": art.get("method"), "cv": art.get("cv"), "fit_s": round(fit_s, 1)},
           "real_key2": sc, "s_per_decision_median": round(s_med, 4), "check_vs_saved": check,
           "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                        "transformers": transformers.__version__},
           "option_order": "accept, move before first step, move beyond last step, retake",
           "cases": [{"index": i, "run": c["run"].replace(".fits", ""), "window": c["window"], "steps": c["steps"],
                      "fwhm": [round(float(x), 4) for x in c["fwhm"]] if c["fwhm"] else None,
                      "n_good": c["n_good"], "key1": c["key1"], "key2": c["key2"],
                      "rule": code_rule(c["steps"], c["fwhm"], c["n_good"]),
                      "probs": [round(float(x), 4) for x in p]} for i, (c, p) in enumerate(zip(real, pr))]}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "cases"}, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
