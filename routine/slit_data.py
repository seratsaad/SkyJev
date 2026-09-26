"""Synthetic MODS slit position-angle visits, computed with the lbt_tools physics.

Each visit is a random target (RA, Dec) on a random night (2026 Sep - 2027 Jun, no summer shutdown), a
start time on a 10-min grid inside the 18-deg-twilight night with the target above 30 deg for the
whole visit, a duration, slit width and seeing, and 2-4 candidate PAs (the acquisition scripts a
program offers), listed in ascending PA as in the real cases. For every PA, code computes with
`lbt_tools.optics.mods_slit_check` (what the `mods_slit_losses` tool calls) the mean blue
(0.35-0.45 um) and red (0.65-0.95 um) loss relative to a centred star and the worst blue step, exactly
as lbt_replay/build_cases.py does for the real cases. Easy label: the lowest mean blue loss.

Hard version: code also finds, on the 10-min grid of start times over the night, which PA gives the
lowest blue loss for a visit of this length starting then, merges runs shorter than 30 min into their
neighbour, and keeps the result as a readme-style rule (UT or hour-angle boundaries). The hard label is
the PA the rule gives at the visit start. The per-window losses of the rule come from one track of the
night and the same optics functions; they are checked against mods_slit_check at the actual start.

The lbt_tools code is vendored in routine/vendor_lbt_tools (code only, no data). CPU job on Pitzer:
    python -m routine.slit_data --n 1500 --procs 8 --out <scratch>/slit_visits.jsonl
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

VENDOR = Path(__file__).resolve().parent / "vendor_lbt_tools" / "mcp"
BLUE, RED = (0.35, 0.40, 0.45), (0.65, 0.75, 0.85, 0.95)
MIN_ALT = 30.0
DURATIONS = (0.5, 1.0, 1.5, 2.0, 2.4, 3.0)
WIDTHS = (0.6, 0.8, 1.0, 1.2)
SEEINGS = (0.6, 0.7, 0.8, 1.0, 1.2, 1.5)


def lbt():
    """lbt_tools.ephem and .optics from the vendored code, offline (no IERS download, no network)."""
    os.environ["LBT_OFFLINE"] = "1"
    os.environ.setdefault("LBT_STATE_DIR", str(VENDOR.parent / "state"))
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from lbt_tools import ephem, optics
    return ephem, optics


def nights() -> list:
    d0, out = dt.date(2026, 9, 1), []
    for i in range(0, 305, 5):
        d = d0 + dt.timedelta(days=i)
        if not (d.month in (7, 8)):
            out.append(d.isoformat())
    return out


def hhmm(h: float) -> str:
    m = int(round(h * 60)) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def sexa(x: float, sign: bool) -> str:
    s = "-" if x < 0 else ("+" if sign else "")
    x = abs(x)
    d, m = int(x), int((x * 60) % 60)
    return f"{s}{d:02d}:{m:02d}:{int(round((x * 3600) % 60)) % 60:02d}"


def table_losses(optics, date, ra, dec, pa, width, seeing, start, hours) -> dict:
    """One row of the loss table, from mods_slit_check, as build_cases.build_slit computes it."""
    d = optics.mods_slit_check(date, ra, dec, pa, width, seeing, start, hours)
    cen = d["throughput_if_centred"]
    blue = [r["throughput"][w] for r in d["rows"] for w in BLUE]
    red = [r["throughput"][w] for r in d["rows"] for w in RED]
    return {"pa": pa, "blue_loss": 1 - (sum(blue) / len(blue)) / cen, "red_loss": 1 - (sum(red) / len(red)) / cen,
            "worst_blue": min(blue) / cen, "parang_mid": d["parang_at_midvisit"]}


def window_blue_loss(optics, rows, pa, width, seeing) -> float:
    """Mean blue loss over track rows, with mods_slit_check's arithmetic and rounding."""
    cen = round(optics.slit_throughput(0.0, width, seeing), 3)
    thr = []
    for r in rows:
        if r["alt"] < 5:
            continue
        s = math.sin(math.radians(pa - r["parang"]))
        for w in BLUE:
            dR = optics.differential_refraction(90 - r["alt"], w, 0.65)
            thr.append(round(optics.slit_throughput(dR * s, width, seeing), 3))
    return 1 - (sum(thr) / len(thr)) / cen


def pick_pas(rng, k: int, parang_hint: float) -> list:
    """k distinct PAs, multiples of 5 in [-175, 180], at least 20 deg apart modulo 180 (PA and PA+180
    give the same loss); sometimes one of them near the parallactic angle."""
    out = []
    if rng.random() < 0.4:
        out.append(int(5 * round(((parang_hint + 180) % 360 - 180) / 5)) or 5)
    while len(out) < k:
        p = int(rng.choice(np.arange(-175, 181, 5)))
        if all(abs(((p - q + 90) % 180) - 90) >= 20 for q in out):
            out.append(p)
    return sorted(out)


def rule_runs(best: list, min_len: int = 3) -> list:
    """Runs [first, last, pa_index] of the best PA over consecutive start slots; runs shorter than
    min_len slots (30 min) are absorbed by the previous run (the next one if first)."""
    runs = []
    for i, b in enumerate(best):
        if runs and runs[-1][2] == b:
            runs[-1][1] = i
        else:
            runs.append([i, i, b])
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for j, r in enumerate(runs):
            if r[1] - r[0] + 1 < min_len:
                k = j - 1 if j > 0 else j + 1
                runs[k][0], runs[k][1] = min(runs[k][0], r[0]), max(runs[k][1], r[1])
                del runs[j]
                merged = []
                for x in runs:
                    if merged and merged[-1][2] == x[2]:
                        merged[-1][1] = x[1]
                    else:
                        merged.append(x)
                runs, changed = merged, True
                break
    return runs


def make_visit(seed: int):
    ephem, optics = lbt()
    rng = np.random.default_rng(seed)
    ns = nights()
    night = ns[int(rng.integers(len(ns)))]
    date = dt.date.fromisoformat(night)
    ni = ephem.night_info(night)
    ev = ni["events_local_hours"]
    g0, g1 = math.ceil(ev["evening_18deg"] * 6) / 6, math.floor(ev["morning_18deg"] * 6) / 6
    hours, width, seeing = float(rng.choice(DURATIONS)), float(rng.choice(WIDTHS)), float(rng.choice(SEEINGS))
    nrow = int(math.floor(hours * 6 + 1e-6)) + 1
    for _ in range(20):
        ra_h, dec = (ni["lst_at_local_midnight_h"] + rng.uniform(-5.5, 5.5)) % 24, float(rng.uniform(-15, 65))
        ra, dc = sexa(ra_h, False), sexa(dec, True)
        rows = ephem.target_track(date, ra, dc, step_min=10, start_hour=g0, end_hour=g1)["rows"]
        ok = [i for i in range(len(rows) - nrow + 1) if min(r["alt"] for r in rows[i:i + nrow]) >= MIN_ALT]
        if len(ok) >= 3 and ok[-1] - ok[0] + 1 == len(ok):
            break
    else:
        return None
    k = int(rng.choice([2, 3, 3, 4]))
    pas = pick_pas(rng, k, rows[(ok[0] + ok[-1] + nrow) // 2]["parang"])
    grid = [[window_blue_loss(optics, rows[i:i + nrow], pa, width, seeing) for pa in pas] for i in ok]
    runs = rule_runs([int(np.argmin(g)) for g in grid])
    i = int(rng.choice(ok))
    start = hhmm(g0 + i / 6)
    t0 = time.perf_counter()
    table = [table_losses(optics, date, ra, dc, pa, width, seeing, start, hours) for pa in pas]
    code_s = time.perf_counter() - t0
    best = int(np.argmin([r["blue_loss"] for r in table]))
    rounded = sorted(round(100 * r["blue_loss"]) for r in table)
    slot = i - ok[0]
    seg = [{"from_ut": rows[ok[0] + a]["utc"], "to_ut": rows[ok[0] + b + 1]["utc"] if ok[0] + b + 1 < len(rows) else None,
            "from_ha": rows[ok[0] + a]["ha_h"], "to_ha": rows[min(ok[0] + b + 1, len(rows) - 1)]["ha_h"], "pa_index": p}
           for a, b, p in runs]
    rule_pick = next(p for a, b, p in runs if a <= slot <= b)
    return {"seed": seed, "night": night, "ra": ra, "dec": dc, "start": start, "start_ut": rows[i]["utc"],
            "ha_start": rows[i]["ha_h"], "alt_start": rows[i]["alt"], "hours": hours, "width": width, "seeing": seeing,
            "pas": pas, "table": table, "best": best, "tie": rounded[0] == rounded[1], "rule": seg, "rule_pick": rule_pick,
            "grid_best_at_start": int(np.argmin(grid[slot])),
            "fast_check_diff": round(abs(grid[slot][best] - table[best]["blue_loss"]), 5), "code_seconds": code_s}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = []
    with Pool(a.procs) as pool:
        for v in pool.imap(make_visit, range(a.seed0, a.seed0 + 2 * a.n), chunksize=4):
            if v is not None:
                out.append(v)
            if len(out) >= a.n:
                break
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        for v in out:
            fh.write(json.dumps(v) + "\n")
    from collections import Counter
    print(json.dumps({"n": len(out), "k": Counter(len(v["pas"]) for v in out), "best_pos": Counter(v["best"] for v in out),
                      "ties": sum(v["tie"] for v in out), "rule_pick_eq_best": sum(v["rule_pick"] == v["best"] for v in out),
                      "grid_eq_best": sum(v["grid_best_at_start"] == v["best"] for v in out),
                      "rule_segments": Counter(len(v["rule"]) for v in out),
                      "fast_check_diff_max": max(v["fast_check_diff"] for v in out),
                      "code_s_median": float(np.median([v["code_seconds"] for v in out]))}, default=str))


if __name__ == "__main__":
    main()
