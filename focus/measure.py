"""Measure an archived through-focus frame the standard way, to get the answer a decider must match.

A WFI "Through Focus Sequence" (ESO template WFI_cal_FocusSeq) is one CCD frame holding nine exposures.
Between exposures the telescope focus steps by a fixed amount and the charge is shifted along the
columns, so every star appears as a vertical chain of nine images, 50 pixels apart. The sharpest image
in the chain marks best focus.

For each frame this finds the chains, measures every image's FWHM with a 2-D Gaussian fit, and fits the
usual focus curve, FWHM^2 = A + B s + C s^2 (a V in FWHM, a parabola in its square), per star and for
the median of all good stars. Outputs one JSON per frame in results/.

Step 1 is the first exposure, taken as the top image of each chain: the dark band at the bottom of the
chip is rows that saw fewer exposures, which means charge moved upward. The direction only matters for
mapping steps to focus values; "which step is best" does not depend on it.

  uv run --with astropy --with scipy python focus/measure.py            # every frame in focus/data
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage, optimize

HERE = Path(__file__).resolve().parent
# FOCUS_DATA / FOCUS_OUT move the frames and results (e.g. for a held-out set)
DATA = Path(os.environ.get("FOCUS_DATA", HERE / "data"))
OUT = Path(os.environ.get("FOCUS_OUT", HERE / "results"))
NSTEP, PIXSCALE, SATURATION = 9, 0.238, 50000.0      # WFI: 0.238 arcsec per pixel; stay clear of 65535


def load(path: Path):
    with fits.open(path) as h:
        hdr0, hdr1, raw = h[0].header, h[1].header, h[1].data.astype(float)
    pre = int(hdr1.get("HIERARCH ESO DET OUT1 PRSCX", 48))
    nx = int(hdr1.get("HIERARCH ESO DET OUT1 NX", 2046))
    ny = int(hdr1.get("HIERARCH ESO DET OUT1 NY", 4068))
    img = raw[:ny, pre:pre + nx]
    # The sky level changes along the columns (rows near the bottom saw fewer exposures), so remove a
    # smooth background row by row.
    rowbg = ndimage.median_filter(np.median(img, axis=1), 31)
    img = img - rowbg[:, None]
    info = {"date_obs": hdr0.get("DATE-OBS"), "filter": hdr0.get("HIERARCH ESO INS FILT1 NAME"),
            "exptime": hdr0.get("EXPTIME"), "airmass": hdr0.get("HIERARCH ESO TEL AIRM START"),
            "seeing_dimm": hdr0.get("HIERARCH ESO TEL AMBI FWHM START")}
    return img, info


def detect(img):
    noise = 1.4826 * np.median(np.abs(img - np.median(img)))
    mask = ndimage.gaussian_filter(img, 1.5) > 8 * noise
    lab, n = ndimage.label(mask)
    objs = ndimage.find_objects(lab)
    out = []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h < 3 or w < 3 or h > 45 or w > 45:
            continue
        sub = np.where(lab[sl] == i, img[sl], 0)
        cy, cx = ndimage.center_of_mass(sub)
        out.append((sl[1].start + cx, sl[0].start + cy, float(img[sl].max())))
    return np.array(out), noise


def spacing(src):
    """The charge shift in rows: the most common gap to the nearest source directly below.

    Only the nearest neighbour counts. Counting every pair lets two-step gaps compete with one-step
    gaps, and on a real frame the two-step gap won."""
    gaps = []
    for x, y, _ in src:
        below = src[(np.abs(src[:, 0] - x) < 4) & (src[:, 1] < y - 5) & (src[:, 1] > y - 150)]
        if len(below):
            gaps.append(y - below[:, 1].max())
    hist, edges = np.histogram(gaps, bins=np.arange(20, 150, 2))
    peak = edges[np.argmax(hist)] + 1.0
    near = [g for g in gaps if abs(g - peak) < 4]
    return float(np.median(near))


def chains(src, step):
    """Groups of nine sources in one column, separated by the charge shift."""
    used, found = set(), []
    order = np.argsort(-src[:, 1])
    for i in order:
        if i in used:
            continue
        chain = [i]
        for _ in range(1, NSTEP):
            # Step from the last image found, so small variations in the shift don't accumulate.
            x, y = src[chain[-1], 0], src[chain[-1], 1]
            ok = np.where((np.abs(src[:, 0] - x) < 4) & (np.abs(src[:, 1] - (y - step)) < 5))[0]
            if len(ok) == 0:
                break
            chain.append(int(ok[np.argmin(np.abs(src[ok, 1] - (y - step)))]))
        if len(chain) == NSTEP:
            used.update(chain)
            found.append(chain)          # top first: step 1 .. 9
    return found


def gauss2d(p, yy, xx):
    amp, x0, y0, sx, sy, bg = p
    return amp * np.exp(-0.5 * (((xx - x0) / sx) ** 2 + ((yy - y0) / sy) ** 2)) + bg


def fwhm(img, x, y, half=10):
    x0, y0 = int(round(x)), int(round(y))
    cut = img[y0 - half:y0 + half + 1, x0 - half:x0 + half + 1]
    if cut.shape != (2 * half + 1, 2 * half + 1):
        return None
    yy, xx = np.mgrid[:cut.shape[0], :cut.shape[1]]
    p0 = [cut.max(), half, half, 2.0, 2.0, 0.0]
    try:
        p, _ = optimize.leastsq(lambda p: (gauss2d(p, yy, xx) - cut).ravel(), p0, maxfev=400)
    except Exception:  # noqa: BLE001
        return None
    sx, sy = abs(p[3]), abs(p[4])
    if not (0.5 < sx < 12 and 0.5 < sy < 12):
        return None
    return {"fwhm_px": 2.3548 * np.sqrt(sx * sy), "ellipticity": 1 - min(sx, sy) / max(sx, sy),
            "peak": float(cut.max()), "flux": float(2 * np.pi * p[0] * sx * sy)}


def fit_curve(steps, widths):
    """FWHM^2 = A + B s + C s^2. Best focus at s0 = -B / 2C, if the curve opens upward."""
    s, w2 = np.asarray(steps, float), np.asarray(widths, float) ** 2
    C, B, A = np.polyfit(s, w2, 2)
    model = A + B * s + C * s ** 2
    rms = float(np.sqrt(np.mean((np.sqrt(np.clip(model, 1e-6, None)) - np.sqrt(w2)) ** 2)))
    if C <= 0:
        return {"best_step": None, "min_fwhm": None, "rms": rms, "opens_up": False}
    s0 = -B / (2 * C)
    return {"best_step": float(s0), "min_fwhm": float(np.sqrt(max(A - B * B / (4 * C), 1e-6))), "rms": rms, "opens_up": True}


def decide(best, n_good):
    """The standard routine's call, which the decider is scored against."""
    if n_good < 3 or best is None:
        return "retake"
    if best < 1.5:
        return "shift toward step 1"
    if best > NSTEP - 0.5:
        return "shift toward step 9"
    return "accept"


def measure(path: Path) -> dict:
    img, info = load(path)
    src, noise = detect(img)
    step = spacing(src)
    stars = []
    for chain in chains(src, step):
        pts = [fwhm(img, src[i, 0], src[i, 1]) for i in chain]
        if any(p is None for p in pts):
            continue
        peaks = [p["peak"] for p in pts]
        star = {"x": float(src[chain[0], 0]), "y": float(src[chain[0], 1]),
                "fwhm_arcsec": [p["fwhm_px"] * PIXSCALE for p in pts],
                "ellipticity": [p["ellipticity"] for p in pts], "peak": peaks,
                "saturated": max(peaks) > SATURATION, "faint": min(peaks) < 30 * noise}
        star["fit"] = fit_curve(range(1, NSTEP + 1), star["fwhm_arcsec"])
        stars.append(star)
    good = [s for s in stars if not s["saturated"] and not s["faint"]]
    med = np.median([s["fwhm_arcsec"] for s in good], axis=0).tolist() if good else None
    fit = fit_curve(range(1, NSTEP + 1), med) if med else {"best_step": None, "opens_up": False}
    per_star = [s["fit"]["best_step"] for s in good if s["fit"]["best_step"] is not None]
    result = {
        "file": path.name, **info, "shift_rows": step, "n_chains": len(stars), "n_good": len(good),
        "median_fwhm_arcsec": med, "fit": fit,
        "best_step_scatter": float(np.std(per_star)) if len(per_star) > 1 else None,
        "decision": decide(fit.get("best_step"), len(good)),
        "stars": stars,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / (path.stem + ".json")).write_text(json.dumps(result, indent=1, default=float))
    return result


def main():
    files = [Path(a) for a in sys.argv[1:]] or sorted(DATA.glob("WFI.*.fits"))
    for f in files:
        r = measure(f)
        fit = r["fit"]
        best = f"{fit['best_step']:.2f}" if fit.get("best_step") is not None else "none"
        curve = " ".join(f"{w:.2f}" for w in (r["median_fwhm_arcsec"] or []))
        print(f"{r['file'][:27]}  chains={r['n_chains']:3} good={r['n_good']:3}  best step {best:>5}  "
              f"-> {r['decision']:22} FWHM(\") {curve}")


if __name__ == "__main__":
    main()
