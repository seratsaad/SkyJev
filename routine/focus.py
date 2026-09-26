"""Focus action: given the measured FWHM at each focus step, accept, move toward the first step,
move toward the last step, or retake. Code measures; a Jev choice (AnyJev L2 head) picks the action;
a confidence gate decides whether SkyJev acts or hands the run back to the standard routine.

Training, validation and synthetic test cases are simulated focus runs: FWHM^2 is a parabola in step
around a true best focus s0, with seeing, slope, star counts, noise and occasional bad images drawn
around the 27 measured WFI runs (focus/results). Each run gives the nine-step window and the two
five-step windows (steps 1-5, 5-9), as in focus/decide.py, and every window is labelled from the TRUE
s0 with key 2's definitions (focus/score.py): retake if fewer than 3 stars or the true nine-step
curve is flatter than max/min 1.3, else move-first if s0 < first + 0.5, move-last if s0 > last - 0.5,
else accept.

The real test is the 84 archival cases in focus/results/decider_scored_gpt-4.1-mini.json, never fit
on, scored against key 1 (window rule written beforehand) and key 2 (full nine-step fit).

    python -m routine.focus --model Qwen/Qwen3-4B --out routine/reports/focus_qwen3-4b.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from routine.common import Split, _probs, backend, pick_threshold, score, wilson

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "focus" / "results" / "decider_scored_gpt-4.1-mini.json"
KEYS = "ABCD"
# the four actions, worded as in focus/decide.py (AnyJev adds the answer letters)
OPTIONS = ("accept: best focus is inside the range, set it there",
           "move: best focus lies before the first step, shift the sequence that way and repeat",
           "move: best focus lies beyond the last step, shift the sequence that way and repeat",
           "retake: too few stars or a curve too flat to trust (seeing too poor)")
WINDOWS = (("all 9", list(range(1, 10))), ("steps 1-5", list(range(1, 6))), ("steps 5-9", list(range(5, 10))))


def question():
    from anyjev import Question

    return Question.choice("What should happen next?", OPTIONS, name="focus_action")


def state_text(steps, fw, n_good) -> str:
    """The measured curve as numbers, worded as focus/decide.py's prompt (question and options apart)."""
    curve = "; ".join(f"step {int(s)}: {w:.2f}" for s, w in zip(steps, fw)) if fw else "no measurements"
    return (f"Telescope auto-focus. A focus sequence stepped the focus in equal increments. Median star FWHM in "
            f"arcsec at each step: {curve}. Stars measured: {n_good}.")


def code_rule(steps, fw, n_good) -> int:
    """The window rule written before any model output (a copy of focus/decide.truth_action, key 1)."""
    if n_good < 3 or fw is None:
        return 3
    s, f = np.asarray(steps, float), np.asarray(fw, float)

    def rms(ss, ff):
        cc = np.polyfit(ss, ff ** 2, 2)
        return float(np.sqrt(np.mean((ff ** 2 - np.polyval(cc, ss)) ** 2))), cc

    base, c = rms(s, f)
    if len(s) > 6:
        best_rms, drop = min((rms(np.delete(s, i), np.delete(f, i))[0], i) for i in range(len(s)))
        if best_rms < base / 3:
            s, f = np.delete(s, drop), np.delete(f, drop)
            _, c = rms(s, f)
    if f.max() / f.min() < 1.3:
        return 3
    if c[0] <= 0:
        return 2 if f[-1] < f[0] else 1
    s0 = -c[1] / (2 * c[0])
    return 1 if s0 < s[0] + 0.5 else 2 if s0 > s[-1] - 0.5 else 0


# ---- synthetic runs ------------------------------------------------------------------------------

def draw_run(rng: np.random.Generator) -> Dict:
    """One simulated nine-step run. Ranges from the 27 measured WFI runs: minimum FWHM 0.7-2.7 arcsec,
    defocus slope 0.4-0.75 arcsec per step, 3-50 stars (0 on three nights), 4-8% scatter per point,
    one bad image (too sharp or too wide) on about one run in nine."""
    s0 = rng.uniform(2.5, 7.5) if rng.random() < 0.5 else rng.uniform(-2.0, 12.0)
    seeing = float(np.clip(rng.lognormal(np.log(1.1), 0.35), 0.5, 3.2))
    k = rng.uniform(0.35, 0.8) if rng.random() < 0.85 else rng.uniform(0.05, 0.3)  # some flat curves
    u = rng.random()
    n_good = 0 if u < 0.06 else int(rng.integers(1, 3)) if u < 0.10 else int(np.clip(rng.lognormal(np.log(11), 0.6), 3, 60))
    s = np.arange(1, 10, dtype=float)
    true = np.sqrt(seeing ** 2 + (k * (s - s0)) ** 2)
    fw = None
    if n_good > 0:
        sig = float(np.clip(rng.uniform(0.025, 0.07) * np.sqrt(10 / n_good), 0.02, 0.25))
        fw = true * (1 + sig * rng.standard_normal(9))
        if rng.random() < 0.12:
            i = int(rng.integers(9))
            fw[i] *= rng.uniform(0.35, 0.75) if rng.random() < 0.4 else rng.uniform(1.3, 2.2)
        fw = np.maximum(fw, 0.3).tolist()
    return {"s0": float(s0), "seeing": seeing, "k": float(k), "n_good": n_good, "true": true.tolist(), "fwhm": fw}


def true_label(run: Dict, steps: List[int]) -> int:
    """Key 2's definitions applied to the true best focus and the true nine-step curve."""
    t = np.asarray(run["true"])
    if run["n_good"] < 3 or run["fwhm"] is None or t.max() / t.min() < 1.3:
        return 3
    lo, hi = steps[0], steps[-1]
    return 1 if run["s0"] < lo + 0.5 else 2 if run["s0"] > hi - 0.5 else 0


def synthetic(n_runs: int, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n_runs):
        run = draw_run(rng)
        for name, steps in WINDOWS if run["fwhm"] else WINDOWS[:1]:
            fw = [run["fwhm"][s - 1] for s in steps] if run["fwhm"] else None
            cases.append({"window": name, "steps": steps, "fwhm": fw, "n_good": run["n_good"],
                          "s0": run["s0"], "label": true_label(run, steps)})
    return cases


def real_cases() -> List[Dict]:
    out = []
    for c in json.loads(REAL.read_text()):
        fw = c["fwhm"]
        out.append({"run": c["run"], "window": c["window"], "steps": c["steps"], "fwhm": fw, "n_good": c["n_good"],
                    "key1": KEYS.index(c["key1"]), "key2": KEYS.index(c["key2"])})
    # the copied rule must reproduce key 1 exactly
    bad = [c["run"] + " " + c["window"] for c in out if code_rule(c["steps"], c["fwhm"], c["n_good"]) != c["key1"]]
    assert not bad, f"code_rule disagrees with key 1 on {bad}"
    return out


def as_case(c: Dict, label_key: str = "label"):
    return (state_text(c["steps"], c["fwhm"], c["n_good"]), c[label_key])


# ---- evaluation ----------------------------------------------------------------------------------

def gates(pv: np.ndarray, yv: np.ndarray) -> Dict[str, float]:
    """Thresholds: chosen on validation for 98% and 95% accuracy when acting, and fixed p >= 0.9, 0.95."""
    ok = pv.argmax(1) == yv
    return {"val_target_0.98": pick_threshold(pv.max(1), ok, 0.98), "val_target_0.95": pick_threshold(pv.max(1), ok, 0.95),
            "fixed_0.90": 0.90, "fixed_0.95": 0.95}


def scored(probs: np.ndarray, y: np.ndarray, thr: Dict[str, float], rule: np.ndarray, secs: float) -> Dict:
    out = {g: score(probs, y, t) for g, t in thr.items()}
    k = int((rule == y).sum())
    out["code_rule"] = {"accuracy": round(k / len(y), 4), "accuracy_ci95": wilson(k, len(y))}
    out["s_per_decision_median"] = round(secs, 4)
    return out


def run(model: str, out: str, n_train: int = 600, n_val: int = 250, n_test: int = 350, dtype: str = "float16") -> Dict:
    import torch
    import transformers
    from anyjev import Decider

    q = question()
    tr, va, te = synthetic(n_train, 1), synthetic(n_val, 2), synthetic(n_test, 3)
    real = real_cases()
    split = Split([as_case(c) for c in tr], [as_case(c) for c in va], [as_case(c) for c in te],
                  [as_case(c, "key2") for c in real])
    be = backend(model, dtype)
    head = Decider(be, level="L2")
    zero = Decider(be, level="L0", prior="content_free")
    _probs(zero, q, split.train[:1], "L0")  # warm-up, untimed
    t0 = time.time()
    art = head.fit_head(q, [c[0] for c in split.train], [c[1] for c in split.train])
    rep = {"task": "focus_action", "model": model, "dtype": dtype, "options": list(q.options),
           "n": {"train": len(tr), "val": len(va), "test": len(te), "real": len(real)},
           "label_mix": {p: np.bincount([c["label"] for c in cs], minlength=4).tolist() for p, cs in
                         (("train", tr), ("val", va), ("test", te))},
           "head": {"layer": art.get("layer_abs"), "kind": art.get("method"), "cv": art.get("cv"),
                    "fit_s": round(time.time() - t0, 1)},
           "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                        "transformers": transformers.__version__}}
    pv, _ = _probs(head, q, split.val, "L2")
    yv = np.array([c[1] for c in split.val])
    thr = gates(pv, yv)
    rep["thresholds"] = thr
    rep["val"] = {g: score(pv, yv, t) for g, t in thr.items()}
    yt = np.array([c["label"] for c in te])
    rule_t = np.array([code_rule(c["steps"], c["fwhm"], c["n_good"]) for c in te])
    p2, s2 = _probs(head, q, split.test, "L2")
    p0, s0 = _probs(zero, q, split.test, "L0")
    rep["test"] = {"L2": scored(p2, yt, thr, rule_t, s2), "L0": scored(p0, yt, {"none": 1.01}, rule_t, s0)}
    rule_r = np.array([code_rule(c["steps"], c["fwhm"], c["n_good"]) for c in real])
    p2, s2 = _probs(head, q, split.real, "L2")
    p0, s0 = _probs(zero, q, split.real, "L0")
    rep["real"] = {}
    for key in ("key1", "key2"):
        y = np.array([c[key] for c in real])
        rep["real"][key] = {"L2": scored(p2, y, thr, rule_r, s2), "L0": scored(p0, y, {"none": 1.01}, rule_r, s0)}
    rep["real_cases"] = [{"run": c["run"][4:14], "window": c["window"], "key1": KEYS[c["key1"]], "key2": KEYS[c["key2"]],
                          "rule": KEYS[r], "L2": [round(float(x), 4) for x in p], "L0": [round(float(x), 4) for x in z]}
                         for c, r, p, z in zip(real, rule_r, p2, p0)]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "real_cases"}, indent=1, default=float), flush=True)
    return rep


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=600, help="simulated runs (about 3 cases each)")
    ap.add_argument("--n-val", type=int, default=250)
    ap.add_argument("--n-test", type=int, default=350)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--dry", action="store_true", help="build the cases and check the code rule only (CPU)")
    a = ap.parse_args(argv)
    if a.dry:
        from collections import Counter
        te = synthetic(a.n_test, 3)
        rule = [code_rule(c["steps"], c["fwhm"], c["n_good"]) for c in te]
        print("test mix", Counter(c["label"] for c in te), "code rule", np.mean([r == c["label"] for r, c in zip(rule, te)]))
        real = real_cases()
        print("real", len(real), "code rule vs key2", np.mean([code_rule(c["steps"], c["fwhm"], c["n_good"]) == c["key2"] for c in real]))
        print(state_text(te[0]["steps"], te[0]["fwhm"], te[0]["n_good"]))
        return
    run(a.model, a.out, a.n_train, a.n_val, a.n_test, a.dtype)


if __name__ == "__main__":
    main()
