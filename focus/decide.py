"""Can a fast, cheap model make the auto-focus decisions? Scored against the standard fit, on archived runs.

Two tests, both with gpt-4.1-mini reading the probability of each answer from one token (no text written):

  sharpest  Show one star's nine images side by side, as an observer sees a focus frame, and ask which
            is sharpest. Truth: that star's own fitted best focus (measure.py).

  action    Show the measured focus curve and ask for the next move: accept, move toward step 1, move
            toward the last step, or retake. Real runs mostly have best focus mid-range, so each run is
            also cut to its first five and last five steps, which is what a badly centred sequence looks
            like. Truth: the fixed rules in `truth_action`, written before any model output was seen.

Position bias is measured the AnyJev way: each question is also asked with the options in reverse
(sharpest) or rotated (action) order, and the answers are averaged in log space.

  uv run --env-file .env --with astropy --with pillow python focus/decide.py
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure  # noqa: E402

import httpx  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MODEL = os.environ.get("FOCUS_MODEL", "gpt-4.1-mini")
MAX_USD = float(os.environ.get("FOCUS_MAX_USD", "0.50"))
DETAIL = os.environ.get("FOCUS_DETAIL", "low")      # image detail for the sharpest test: low or high
# US$ per input token, for the spend ceiling only; output is one token. The GPT-5 rates are a
# deliberately high guess, so the ceiling trips early rather than late. Check your dashboard.
PRICE_IN = {"gpt-4.1-mini": 0.40, "gpt-4.1": 2.00, "gpt-4o": 2.50}.get(MODEL, 5.00) / 1e6
REASONING_MODEL = MODEL.startswith(("gpt-5", "o3", "o4"))
spent = {"tokens": 0}


def ask(content, options: list[str]) -> tuple[dict, float]:
    """One call, one token. Returns the probability of each option label, renormalised, and seconds taken."""
    if spent["tokens"] * PRICE_IN > MAX_USD:
        raise SystemExit(f"stopping: about ${spent['tokens'] * PRICE_IN:.2f} spent, ceiling ${MAX_USD:.2f}")
    t = time.perf_counter()
    body = {"model": MODEL, "logprobs": True, "messages": [{"role": "user", "content": content}]}
    if REASONING_MODEL:
        # GPT-5 models: at most 5 alternatives per token, no temperature control, reasoning off, and a
        # little room beyond one token or the request is refused. Only the first token is read.
        body.update(top_logprobs=5, max_completion_tokens=4, reasoning_effort="none")
    else:
        body.update(top_logprobs=20, max_tokens=1, temperature=0)
    for attempt in range(5):
        r = httpx.post("https://api.openai.com/v1/chat/completions", timeout=90,
                       headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, json=body)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 4:
            time.sleep(2 * (attempt + 1))        # transient; one 502 once ended a whole run
            continue
        if r.status_code != 200:
            raise SystemExit(f"{MODEL}: HTTP {r.status_code}: {r.text[:300]}")
        break
    d = r.json()
    spent["tokens"] += d["usage"]["total_tokens"]
    top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    lp = {o: -30.0 for o in options}
    for item in top:
        tok = item["token"].strip()
        if tok in lp:
            lp[tok] = max(lp[tok], item["logprob"])
    z = max(lp.values())
    p = {o: math.exp(v - z) for o, v in lp.items()}
    s = sum(p.values())
    return {o: v / s for o, v in p.items()}, time.perf_counter() - t


def combine(dists: list[dict]) -> dict:
    """Average in log space, as AnyJev does across option orders."""
    keys = dists[0].keys()
    lp = {k: np.mean([math.log(max(d[k], 1e-12)) for d in dists]) for k in keys}
    z = max(lp.values())
    p = {k: math.exp(v - z) for k, v in lp.items()}
    s = sum(p.values())
    return {k: v / s for k, v in p.items()}


# ---- test 1: which image is sharpest -------------------------------------------------------------

def strip(img, xs, ys, order) -> str:
    """Nine cutouts side by side with one shared stretch, labelled 1..9 left to right."""
    cuts = []
    for x, y in zip(xs, ys, strict=True):
        x0, y0 = int(round(x)), int(round(y))
        cuts.append(img[y0 - 15:y0 + 16, x0 - 15:x0 + 16])
    hi = max(c.max() for c in cuts)
    tiles = []
    for i in order:
        c = np.arcsinh(np.clip(cuts[i], 0, None) / (0.02 * hi)) / np.arcsinh(1 / 0.02)
        tiles.append(Image.fromarray((255 * np.clip(c, 0, 1)).astype(np.uint8)).resize((96, 96), Image.NEAREST))
    canvas = Image.new("L", (9 * 104 + 8, 124), 30)
    draw = ImageDraw.Draw(canvas)
    for k, t in enumerate(tiles):
        canvas.paste(t, (8 + k * 104, 8))
        draw.text((8 + k * 104 + 44, 106), str(k + 1), fill=255)
    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def sharpest_test(max_stars=3):
    rows = []
    for rpath in sorted(RES.glob("WFI.*.json")):
        run = json.loads(rpath.read_text())
        good = [s for s in run["stars"] if not s["saturated"] and not s["faint"] and s["fit"]["best_step"]]
        good = [s for s in good if 1 <= s["fit"]["best_step"] <= 9]
        if not good:
            continue
        img, _ = measure.load(HERE / "data" / (Path(run["file"]).stem + ".fits"))
        src, _ = measure.detect(img)
        step = measure.spacing(src)
        good.sort(key=lambda s: -max(s["peak"]))
        for star in good[:max_stars]:
            ys = [star["y"] - k * step for k in range(9)]
            xs = [star["x"]] * 9
            prompt = ("These are nine images of the same star, taken at nine telescope focus settings, shown with the "
                      "same brightness scale. Which one is the sharpest, best-focused image? Answer with one digit, 1 to 9.")
            dists = []
            for order in (list(range(9)), list(range(8, -1, -1))):
                p, sec = ask([{"type": "text", "text": prompt},
                              {"type": "image_url", "image_url": {"url": strip(img, xs, ys, order), "detail": DETAIL}}],
                             [str(i) for i in range(1, 10)])
                # map shown position back to the true step
                dists.append({str(order[int(k) - 1] + 1): v for k, v in p.items()})
            raw, both = dists[0], combine(dists)
            truth = star["fit"]["best_step"]
            rows.append({"run": run["file"], "truth": truth, "argmin": int(np.argmin(star["fwhm_arcsec"])) + 1,
                         "raw": raw, "debiased": both, "seconds": sec})
    return rows


# ---- test 2: what to do next -------------------------------------------------------------------

ACTIONS = {"A": "accept: best focus is inside the range, set it there",
           "B": "move: best focus lies before the first step, shift the sequence that way and repeat",
           "C": "move: best focus lies beyond the last step, shift the sequence that way and repeat",
           "D": "retake: too few stars or a curve too flat to trust (seeing too poor)"}


def truth_action(steps, fw, n_good):
    """Fixed rules, written before any model output was seen."""
    if n_good < 3 or fw is None:
        return "D"
    s, f = np.asarray(steps, float), np.asarray(fw, float)

    def rms(ss, ff):
        cc = np.polyfit(ss, ff ** 2, 2)
        return float(np.sqrt(np.mean((ff ** 2 - np.polyval(cc, ss)) ** 2))), cc

    # One bad image can wreck the parabola. Try leaving out each point in turn and keep the removal
    # that helps most, but only if it helps a lot. (Removing the point with the largest residual was
    # tried first and picked the wrong point on a real run, 2026-05-14, where the first image is bad.)
    base, c = rms(s, f)
    if len(s) > 6:
        trials = [(rms(np.delete(s, i), np.delete(f, i))[0], i) for i in range(len(s))]
        best_rms, drop = min(trials)
        if best_rms < base / 3:
            s, f = np.delete(s, drop), np.delete(f, drop)
            _, c = rms(s, f)
    if f.max() / f.min() < 1.3:
        return "D"
    if c[0] <= 0:
        return "C" if f[-1] < f[0] else "B"
    s0 = -c[1] / (2 * c[0])
    if s0 < s[0] + 0.5:
        return "B"
    if s0 > s[-1] - 0.5:
        return "C"
    return "A"


def action_prompt(steps, fw, n_good, order):
    curve = "; ".join(f"step {int(s)}: {w:.2f}" for s, w in zip(steps, fw, strict=True)) if fw else "no measurements"
    opts = "\n".join(f"{lab}) {ACTIONS[k]}" for lab, k in zip("ABCD", order, strict=True))
    return (f"Telescope auto-focus. A focus sequence stepped the focus in equal increments. Median star FWHM in "
            f"arcsec at each step: {curve}. Stars measured: {n_good}.\nWhat should happen next?\n{opts}\n"
            f"Answer with one letter.")


def action_test():
    cases = []
    for rpath in sorted(RES.glob("WFI.*.json")):
        run = json.loads(rpath.read_text())
        fw, n = run["median_fwhm_arcsec"], run["n_good"]
        windows = [("all 9", list(range(1, 10)))]
        if fw:
            windows += [("steps 1-5", list(range(1, 6))), ("steps 5-9", list(range(5, 10)))]
        for name, steps in windows:
            f = [fw[s - 1] for s in steps] if fw else None
            cases.append({"run": run["file"], "window": name, "steps": steps, "fwhm": f, "n_good": n,
                          "truth": truth_action(steps, f, n)})
    rotations = ["ABCD", "BCDA", "CDAB", "DABC"]
    for c in cases:
        dists = []
        for rot in rotations:
            p, sec = ask(action_prompt(c["steps"], c["fwhm"], c["n_good"], rot), list("ABCD"))
            dists.append({rot["ABCD".index(lab)]: v for lab, v in p.items()})   # back to the true action
        c["raw"], c["debiased"], c["seconds"] = dists[0], combine(dists), sec
    return cases


def main():
    if "--sharpest-only" in sys.argv:
        # A fair rerun of the image test at a chosen detail, one star per run, saved separately.
        rows = sharpest_test(max_stars=1)
        (RES / f"decider_sharpest_{DETAIL}_{MODEL}.json").write_text(json.dumps(rows, indent=1, default=float))
        for key in ("raw", "debiased"):
            err = [abs(int(max(r[key], key=r[key].get)) - r["truth"]) for r in rows]
            print(f"sharpest detail={DETAIL} [{key:8}] {len(rows)} stars: within 1 step {np.mean([e <= 1 for e in err]):.0%}, mean error {np.mean(err):.2f}")
        err = [abs(r["argmin"] - r["truth"]) for r in rows]
        print(f"code's plain minimum on the same stars: within 1 step {np.mean([e <= 1 for e in err]):.0%}, mean error {np.mean(err):.2f}")
        print(f"tokens {spent['tokens']:,}, about ${spent['tokens'] * PRICE_IN:.3f}")
        return
    out = {"model": MODEL}
    out["sharpest"] = sharpest_test() if "--no-sharpest" not in sys.argv else []
    out["action"] = action_test()
    out["tokens"] = spent["tokens"]
    out["usd_estimate"] = spent["tokens"] * PRICE_IN
    (RES / f"decider_{MODEL}.json").write_text(json.dumps(out, indent=1, default=float))

    sh = out["sharpest"]
    for key in ("raw", "debiased") if sh else ():
        err = [abs(int(max(r[key], key=r[key].get)) - r["truth"]) for r in sh]
        print(f"sharpest [{key:8}] {len(sh)} stars: within 1 step of the fit {np.mean([e <= 1 for e in err]):.0%}, "
              f"mean error {np.mean(err):.2f} steps")
    err = [abs(r["argmin"] - r["truth"]) for r in sh] or [0]
    print(f"sharpest [argmin  ] the code's plain minimum: within 1 step {np.mean([e <= 1 for e in err]):.0%}, mean error {np.mean(err):.2f}")
    ac = out["action"]
    for key in ("raw", "debiased"):
        acc = np.mean([max(c[key], key=c[key].get) == c["truth"] for c in ac])
        print(f"action   [{key:8}] {len(ac)} cases: agrees with the rules {acc:.0%}")
    from collections import Counter
    print("truth mix:", dict(Counter(c["truth"] for c in ac)))
    print(f"seconds per call about {np.median([c['seconds'] for c in ac]):.2f}; tokens {spent['tokens']:,}; about ${out['usd_estimate']:.3f}")


if __name__ == "__main__":
    main()
