"""Shared harness for the routine-decision experiments: code prepares a small fixed menu, a Jev
choice picks from it, and a confidence gate decides whether SkyJev acts or hands the case on.

A task supplies labelled cases as (state_text, label_index) for one AnyJev Question with a fixed
option layout. `evaluate` fits an L2 head on the training cases, reads calibrated probabilities on
the validation and test cases (and on real cases, if any), picks the gate threshold on validation
so that accuracy on the decisions SkyJev acts on reaches the target, and reports on test and real:

  accuracy of every decision, and accuracy and coverage above the gate; the always-majority rate;
  the zero-label (L0) accuracy for comparison; ECE; seconds per decision (CUDA-synchronised).

Run on the cluster (V100, float16), never on a laptop:
    python -m routine.<task> --model Qwen/Qwen3-4B --out routine/reports/<task>_<model>.json
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Case = Tuple[str, int]  # (state text shown to the model, index of the correct option)


@dataclass
class Split:
    train: List[Case]
    val: List[Case]
    test: List[Case]
    real: List[Case] = field(default_factory=list)  # e.g. replayed LBT or archival cases; never fit on


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & (conf < b if b < 1 else conf <= b)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def backend(model: str, dtype: str = "float16", batch_size: int = 16):
    from obsassist.assistant.anyjev_backend import make_local_backend

    return make_local_backend(model, device="cuda", dtype=dtype, batch_size=batch_size)


def _probs(decider, question, cases: Sequence[Case], level: str) -> Tuple[np.ndarray, float]:
    """Probabilities for every case, and mean seconds per decision (one case at a time, timed)."""
    import torch

    out, secs = [], []
    for text, _ in cases:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        d = decider.decide(text, [question], level=level)[0]
        torch.cuda.synchronize()
        secs.append(time.perf_counter() - t0)
        out.append(np.asarray(d.probs, dtype=float))
    return np.array(out), float(np.median(secs)) if secs else float("nan")


def pick_threshold(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Lowest confidence threshold whose accepted validation decisions reach `target` accuracy
    (so coverage is as large as possible); 1.01 (act on nothing) if no threshold reaches it."""
    for t in sorted(set(np.round(conf, 4))):
        m = conf >= t
        if m.sum() >= 5 and correct[m].mean() >= target:
            return float(t)
    return 1.01


def score(probs: np.ndarray, labels: np.ndarray, thr: float) -> Dict:
    pred, conf = probs.argmax(1), probs.max(1)
    ok = pred == labels
    acted = conf >= thr
    n, k = len(labels), int(ok.sum())
    ka, na = int(ok[acted].sum()), int(acted.sum())
    return {
        "n": n,
        "accuracy": round(k / n, 4) if n else None,
        "accuracy_ci95": wilson(k, n),
        "acted": na,
        "coverage": round(na / n, 4) if n else None,
        "accuracy_when_acting": round(ka / na, 4) if na else None,
        "accuracy_when_acting_ci95": wilson(ka, na),
        "always_majority": round(float(np.bincount(labels).max() / n), 4) if n else None,
        "ece": round(ece(conf, ok.astype(float)), 4) if n else None,
    }


def evaluate(name: str, question, split: Split, model: str, out: str, target: float = 0.98,
             layers: Optional[Sequence[int]] = None, dtype: str = "float16", notes: str = "") -> Dict:
    import torch
    import transformers
    from anyjev import Decider

    be = backend(model, dtype)
    head = Decider(be, level="L2")
    zero = Decider(be, level="L0", prior="content_free")
    _probs(zero, question, split.train[:1], "L0")  # warm-up, untimed
    t0 = time.time()
    art = head.fit_head(question, [c[0] for c in split.train], [c[1] for c in split.train], layers=layers)
    fit_s = time.time() - t0
    rep = {
        "task": name,
        "model": model,
        "dtype": dtype,
        "options": list(question.options),
        "n_train": len(split.train),
        "head": {"layer": art.get("layer_abs"), "kind": art.get("method"), "cv": art.get("cv"), "fit_s": round(fit_s, 1)},
        "gate_target": target,
        "notes": notes,
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "python": platform.python_version()},
    }
    pv, _ = _probs(head, question, split.val, "L2")
    yv = np.array([c[1] for c in split.val])
    thr = pick_threshold(pv.max(1), pv.argmax(1) == yv, target)
    rep["threshold"] = thr
    rep["val"] = score(pv, yv, thr)
    for part in ("test", "real"):
        cases = getattr(split, part)
        if not cases:
            continue
        y = np.array([c[1] for c in cases])
        p2, s2 = _probs(head, question, cases, "L2")
        p0, s0 = _probs(zero, question, cases, "L0")
        rep[part] = {"L2": score(p2, y, thr) | {"s_per_decision_median": round(s2, 4)},
                     "L0": score(p0, y, 1.01) | {"s_per_decision_median": round(s0, 4)}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    return rep
