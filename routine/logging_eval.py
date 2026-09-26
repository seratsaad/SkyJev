"""Evaluation runner shared by routine/logging.py and routine/slit.py.

It is routine.common.evaluate with four additions, built from the same pieces (common.backend,
_probs, pick_threshold, score), so the numbers mean the same thing:
  - one loaded model serves several questions (logging asks two);
  - both gates, 0.98 and 0.95, are picked on validation from one set of probabilities;
  - a code baseline is scored next to the model on test and real;
  - the per-case probabilities come back to the caller (for message-level logging metrics).
Reports are checked for private strings (lbt_replay/run_jev.private_strings) before they are written.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from routine.common import Split, _probs, backend, pick_threshold, score, wilson

GATES = (0.98, 0.95)
ROOT = Path(__file__).resolve().parents[1]


def code_score(pred: Sequence[int], y: np.ndarray) -> Dict:
    ok = np.asarray(pred) == np.asarray(y)
    k, n = int(ok.sum()), len(ok)
    return {"n": n, "accuracy": round(k / n, 4) if n else None, "accuracy_ci95": wilson(k, n)}


def hardware() -> Dict:
    import torch
    import transformers
    return {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "transformers": transformers.__version__, "python": platform.python_version()}


def run_question(be, name: str, question, split: Split, model: str, code: Optional[Dict] = None,
                 layers: Optional[Sequence[int]] = None, notes: str = ""):
    """Fit the L2 head on split.train, pick the gates on split.val, score test and real (L2, the L0
    zero-label reading, and the code baseline). Returns (report, probs by part)."""
    from anyjev import Decider

    head = Decider(be, level="L2")
    zero = Decider(be, level="L0", prior="content_free")
    _probs(zero, question, split.train[:1], "L0")  # warm-up, untimed
    t0 = time.time()
    art = head.fit_head(question, [c[0] for c in split.train], [c[1] for c in split.train], layers=layers)
    rep = {"task": name, "model": model, "dtype": "float16", "options": list(question.options),
           "question": question.text, "n_train": len(split.train),
           "train_label_counts": np.bincount([c[1] for c in split.train], minlength=question.k).tolist(),
           "head": {"layer": art.get("layer_abs"), "kind": art.get("method"), "cv": art.get("cv"),
                    "fit_s": round(time.time() - t0, 1)},
           "gates": list(GATES), "notes": notes, "hardware": hardware()}
    pv, _ = _probs(head, question, split.val, "L2")
    yv = np.array([c[1] for c in split.val])
    thr = {g: pick_threshold(pv.max(1), pv.argmax(1) == yv, g) for g in GATES}
    rep["threshold"] = {str(g): thr[g] for g in GATES}
    rep["val"] = {f"gate_{g}": score(pv, yv, thr[g]) for g in GATES}
    probs = {"val": pv}
    for part in ("test", "real"):
        cases = getattr(split, part)
        if not cases:
            continue
        y = np.array([c[1] for c in cases])
        p2, s2 = _probs(head, question, cases, "L2")
        p0, s0 = _probs(zero, question, cases, "L0")
        r = {"L2": {f"gate_{g}": score(p2, y, thr[g]) for g in GATES}, "L2_s_per_decision_median": round(s2, 4),
             "L0": score(p0, y, 1.01), "L0_s_per_decision_median": round(s0, 4)}
        if code and part in code:
            r["code"] = code_score(code[part], y)
        rep[part] = r
        probs[part], probs[part + "_L0"] = p2, p0
    return rep, probs


def private_guard(private: Optional[str]):
    """A function that raises if a text holds any private queue/annotation string (no-op without the
    private directory)."""
    if not private:
        return lambda text: None
    sys.path.insert(0, str(ROOT / "lbt_replay"))
    from run_jev import check_public, private_strings
    bad = private_strings(Path(private))

    def guard(text: str):
        hits = check_public(text, bad)
        if hits:
            raise SystemExit(f"refusing to write a report: {len(hits)} private strings in it")
    return guard


def write_report(path: str, rep: Dict, guard) -> None:
    text = json.dumps(rep, indent=1, default=float)
    guard(text)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
    print(text[:4000])


def load_backend(model: str, batch_size: int = 8):
    return backend(model, "float16", batch_size=batch_size)
