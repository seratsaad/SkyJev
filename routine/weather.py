"""Weather decisions for SkyJev: the dome call (observe / wait / close) and the sky class
(photometric / thin cirrus / thick cloud), as a Jev choice over a small fixed menu with a fitted L2
head and a confidence gate (routine/common.py).

Code does everything computable. At each moment it builds an observables-only summary: the site
limits, humidity / wind / temperature-minus-dew-point with their margins, 30-min trends and the
margin projected 60 min ahead at that trend, precipitation, the guider flux ratio (with its 30-min
range), DIMM seeing, time of night and the dome state (for a weather closure: how long it has been
closed and how long the reopen conditions have held). Nothing in the summary reads the future.

Labels come from the simulator's future truth (obsassist.weather, the weather NightModel(prog, seed)
holds): dome "observe" if the dome stays usable for the next HORIZON_MIN minutes, "close" if it must
close within that time and stays closed to the end of the observing night, "wait" otherwise; sky is
the true sky_state (asked only while the dome is open). Validation and test hold out whole nights.

Real cases (never fit on): the anonymized LBT replay cases in the private directory, converted to the
same summary with code (Mt Graham limits; forecast values where nothing was measured; missing
observables shown as n/a). They are read in memory only; reports hold aggregate numbers only.

On Pitzer (V100, float16):
    python -m routine.weather --build                       # class mix + rule baselines, no GPU
    python -m routine.weather --task dome --model Qwen/Qwen3-4B
Tasks: dome (3 options), dome2 (observe / do not observe), sky (3 options).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HORIZON_MIN = 60
TREND_MIN = 30
TELESCOPES = ("keck1", "keck2", "baade", "clay")  # both sites, both telescopes at each
PRIVATE = Path("/fs/scratch/PAS2823/saadsm/private_lbt")

DOME = ("observe", "wait", "close")
DOME_OPTIONS = ("observe: the dome is open or can open, and stays usable for the next 60 min",
                "wait: closed now or must close soon, but reopens later tonight",
                "close: must close (or stay closed) for the rest of the night")
DOME2_OPTIONS = ("observe: the dome stays usable for the next 60 min",
                 "do not observe: the dome is closed now or must close within 60 min")
SKY = ("photometric", "thin cirrus", "thick cloud")
REAL_SKY = {"clear": 0, "thin cirrus": 1, "cloudy": 2}


# ------------------------------------------------------------------ summary (observables only)

def _f(x, fmt: str, unit: str = "") -> str:
    return "n/a" if x is None else f"{x:{fmt}}{unit}"


def _signed(x, fmt: str, unit: str = "") -> str:
    return "n/a" if x is None else f"{x:+{fmt}}{unit}"


def margins(o: Dict) -> Dict[str, Optional[float]]:
    """Distance to each close limit now, and projected HORIZON_MIN ahead at the 30-min trend
    (positive = inside the limits). None where the observable or its trend is missing."""
    L = o["limits"]
    k = HORIZON_MIN / TREND_MIN
    now = {"humidity": None if o["humidity"] is None else L["humidity_close"] - o["humidity"],
           "wind": None if o["wind"] is None else L["wind_close_ms"] - o["wind"],
           "spread": None if o["spread"] is None else o["spread"] - L["dewpoint_margin_c"]}
    trend = {"humidity": o.get("humidity_trend"), "wind": o.get("wind_trend"), "spread": o.get("spread_trend")}
    sign = {"humidity": -1.0, "wind": -1.0, "spread": 1.0}
    proj = {q: None if now[q] is None or trend[q] is None else now[q] + sign[q] * k * trend[q] for q in now}
    return {"now": now, "proj": proj}


def summary(o: Dict) -> str:
    """The compact state text the model reads. `o` holds observables only (see sim_moment / real_obs)."""
    L, m = o["limits"], margins(o)
    src = {"humidity": o.get("humidity_src", ""), "wind": o.get("wind_src", "")}
    lines = [
        f"Site limits: close at humidity >= {L['humidity_close']:.0f} %, wind >= {L['wind_close_ms']:.1f} m/s, "
        f"temperature minus dew point < {L['dewpoint_margin_c']:.1f} C, or any precipitation; reopen only after "
        f"{L['reopen_wait_min']:.0f} min with humidity < {L['humidity_open']:.0f} %, wind < {L['wind_open_ms']:.1f} m/s "
        f"and temperature minus dew point >= {L['dewpoint_margin_c'] + 1:.1f} C.",
        f"Night: {o['h_since_start']:+.1f} h from evening twilight, {o['h_to_end']:.1f} h left until morning twilight.",
        "Dome: " + o["dome_text"] + ".",
        f"Humidity: {_f(o['humidity'], '.0f', ' %')}{src['humidity']} (close limit {L['humidity_close']:.0f} %, "
        f"margin {_f(m['now']['humidity'], '.0f', ' %')}); change over the last 30 min {_signed(o.get('humidity_trend'), '.0f', ' %')}.",
        f"Wind: {_f(o['wind'], '.1f', ' m/s')}{src['wind']}, gusts {_f(o.get('gust'), '.1f', ' m/s')} (close limit "
        f"{L['wind_close_ms']:.1f} m/s, margin {_f(m['now']['wind'], '.1f', ' m/s')}); change over the last 30 min "
        f"{_signed(o.get('wind_trend'), '.1f', ' m/s')}.",
        f"Temperature minus dew point: {_f(o['spread'], '.1f', ' C')} (limit {L['dewpoint_margin_c']:.1f} C, margin "
        f"{_f(m['now']['spread'], '.1f', ' C')}); change over the last 30 min {_signed(o.get('spread_trend'), '.1f', ' C')}.",
        f"Margins in 60 min if the 30-min trends continue: humidity {_f(m['proj']['humidity'], '.0f', ' %')}, "
        f"wind {_f(m['proj']['wind'], '.1f', ' m/s')}, temperature minus dew point {_f(m['proj']['spread'], '.1f', ' C')}.",
        "Precipitation: " + o["precip_text"] + ".",
        (f"Guider flux ratio (1.00 = photometric): {o['flux']:.2f} now, {o['flux_lo']:.2f} to {o['flux_hi']:.2f} over the last 30 min."
         if o.get("flux") is not None else "Guider flux ratio: n/a (" + o.get("flux_why", "not guiding") + ")."),
        f"DIMM seeing: {_f(o.get('dimm'), '.2f', ' arcsec')}.",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ code-only rules (baselines)

def rule_dome(o: Dict) -> int:
    """Limits and trends only. Closed by weather now, over a limit now, precipitation, or a margin
    projected to cross zero within 60 min -> not observe; then wait if at least 1.5 h of night is
    left, else close. Missing observables count as fine."""
    m = margins(o)
    bad = o["precip"] or any(v is not None and v <= 0 for v in list(m["now"].values()) + list(m["proj"].values()))
    if o["dome"] == "closed_weather":
        bad = True
    if not bad:
        return 0
    return 1 if o["h_to_end"] >= 1.5 else 2


def complete(o: Dict, task: str) -> bool:
    """Guard in code, before the gate: act only on a summary like the ones the head was fit on --
    humidity and wind measured (not forecast), 30-min trends and the dew-point spread present, the
    dome state and its reason known, no precipitation line (the simulator has none), and for the sky
    class a guider reading. Anything else is handed on whatever the head's confidence."""
    ok = (not o.get("humidity_src") and not o.get("wind_src") and o["spread"] is not None
          and o.get("humidity_trend") is not None and o["dome"] in ("open", "closed_weather")
          and o["precip_text"] == "none reported")
    return ok and (task != "sky" or o.get("flux") is not None)


def rule_sky(o: Dict) -> int:
    """Guider flux ratio >= 0.9 photometric, >= 0.45 thin cirrus, else thick cloud; n/a -> photometric."""
    f = o.get("flux")
    if f is None:
        return 0
    return 0 if f >= 0.9 else (1 if f >= 0.45 else 2)


# ------------------------------------------------------------------ simulator moments

def limits_dict(L) -> Dict[str, float]:
    """obsassist WeatherLimits -> the numbers the summary shows (reopen wind as in obsassist.weather: 0.9 x close)."""
    return {"humidity_close": L.humidity_close, "humidity_open": L.humidity_open, "wind_close_ms": L.wind_close_ms,
            "wind_open_ms": 0.9 * L.wind_close_ms, "dewpoint_margin_c": L.dewpoint_margin_c,
            "reopen_wait_min": L.reopen_wait_min}


def sim_night(i: int, seed0: int):
    """Night i: a random program (telescope, date) and weather seed. The weather is exactly what
    NightModel(prog, seed).weather holds (same NightEphem grid, same generate call), without the
    photon-budget precompute NightModel does for the targets."""
    from obsassist.astro.ephem import NightEphem
    from obsassist.programs.generator import generate_program
    from obsassist.weather import generate

    rng = np.random.default_rng(seed0 + i)
    tel = TELESCOPES[int(rng.integers(len(TELESCOPES)))]
    date = (dt.date(2026, 1, 1) + dt.timedelta(days=int(rng.integers(730)))).isoformat()
    seed = int(rng.integers(2**31 - 1))
    prog = generate_program(tel, date, seed=seed)
    eph = NightEphem(prog.site, date, 1.0)
    w = generate(prog.site, eph.t_min, seed)
    tw = eph.twilight
    start = {"sunset": tw.sunset, "6deg": tw.civil_dusk, "12deg": tw.nautical_dusk, "18deg": tw.astro_dusk}
    end = {"sunset": tw.sunrise, "6deg": tw.civil_dawn, "12deg": tw.nautical_dawn, "18deg": tw.astro_dawn}
    t0, t1 = int(round(eph.minute_of(start[prog.night_start]))), int(round(eph.minute_of(end[prog.night_end])))
    return prog, eph, w, seed, t0, t1


def sim_moment(site, w, j: int, t0: int, t1: int, rng) -> Tuple[Dict, int, int]:
    """Observables at grid minute j (past only) and the labels (dome call, sky class) from the future truth."""
    L = site.limits
    ok = w.dome_ok
    spread = w.temp_c - w.dewpoint_c
    k = j - TREND_MIN
    drop_trend, drop_spread = rng.random() < 0.08, rng.random() < 0.08  # station gaps, so n/a is familiar
    o = {"limits": limits_dict(L), "h_since_start": (j - t0) / 60.0, "h_to_end": (t1 - j) / 60.0,
         "humidity": float(w.humidity[j]), "wind": float(w.wind_ms[j]), "gust": float(w.wind_gust_ms[j]),
         "spread": None if drop_spread else float(spread[j]),
         "humidity_trend": None if drop_trend else float(w.humidity[j] - w.humidity[k]),
         "wind_trend": None if drop_trend else float(w.wind_ms[j] - w.wind_ms[k]),
         "spread_trend": None if (drop_trend or drop_spread) else float(spread[j] - spread[k]),
         "precip": False, "precip_text": "none reported",
         "dimm": float(w.seeing_dimm[j]) if np.isfinite(w.seeing_dimm[j]) else None}
    if ok[j]:
        o["dome"], o["dome_text"] = "open", "open"
    else:
        was_open = np.nonzero(ok[: j + 1])[0]
        closed = j - (int(was_open[-1]) if len(was_open) else 0)
        good = ((w.humidity < L.humidity_open) & (w.wind_ms < 0.9 * L.wind_close_ms)
                & (spread >= L.dewpoint_margin_c + 1))[: j + 1]
        bad_idx = np.nonzero(~good)[0]
        run = j - (int(bad_idx[-1]) if len(bad_idx) else -1)
        o["dome"] = "closed_weather"
        o["dome_text"] = (f"closed by weather for {closed} min; the reopen conditions have held for the last "
                          f"{run} min (reopening needs {L.reopen_wait_min:.0f} min)")
    # guider: only while open and guiding (15 % of open moments: acquisition, target change)
    if ok[j] and rng.random() > 0.15:
        idx = [i for i in range(k, j + 1) if ok[i]]
        fr = 10 ** (-0.4 * w.cloud_mag[idx]) * np.exp(rng.normal(0, 0.03, len(idx)))
        o["flux"], o["flux_lo"], o["flux_hi"] = float(fr[-1]), float(fr.min()), float(fr.max())
    else:
        o["flux"], o["flux_why"] = None, ("dome closed" if not ok[j] else "not guiding: acquisition or target change")
    je = min(j + HORIZON_MIN, t1)
    win = ok[j: je + 1]
    if win.all():
        dome = 0
    else:
        first = j + int(np.argmin(win))
        dome = 2 if not ok[first: t1 + 1].any() else 1
    return o, dome, int(w.sky_state[j])


def _night_rows(args) -> List[Dict]:
    """Moments of one night: per_night uniform ("natural") moments, and on a night with a weather
    closure 3 x per_night more drawn from 120 min before the first closed minute to 60 min after the
    last one ("enriched"), so wait and close are not a few percent of the cases."""
    i, per_night, seed0 = args
    prog, eph, w, seed, t0, t1 = sim_night(i, seed0)
    rng = np.random.default_rng(seed0 * 7 + i)
    lo, hi = max(t0, TREND_MIN + 1), t1 - 10
    js = [(int(j), True) for j in rng.integers(lo, hi, size=per_night)]
    shut = np.nonzero(~w.dome_ok[lo:hi])[0]
    if len(shut):
        a, b = max(lo, lo + int(shut[0]) - 120), min(hi, lo + int(shut[-1]) + 60)
        js += [(int(j), False) for j in rng.integers(a, b + 1, size=3 * per_night)]
    out = []
    for j, natural in sorted(js):
        o, dome, sky = sim_moment(prog.site, w, j, t0, t1, rng)
        out.append({"night": i, "site": prog.site.key, "natural": natural, "obs": o, "dome": dome, "sky": sky})
    return out


def build_sim(n_nights: int, per_night: int, seed0: int = 20260926, workers: int = 1) -> List[Dict]:
    """Moments of n_nights random nights (see _night_rows), built in `workers` processes."""
    jobs = [(i, per_night, seed0) for i in range(n_nights)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_night_rows, jobs, chunksize=8))
    else:
        rows = [_night_rows(a) for a in jobs]
    return [r for night in rows for r in night]


def night_part(i: int) -> str:
    """Whole nights go to one split: 70 % train, 15 % val, 15 % test."""
    r = i % 20
    return "train" if r < 14 else ("val" if r < 17 else "test")


# ------------------------------------------------------------------ real LBT cases (private, in memory only)

MPH = 0.44704
# LBTO operating guidelines as recorded in the lbt_tools config: humidity closes at 95 %, reopen below
# 90 %; wind closes at 20 m/s sustained; precipitation closes. Not given there, taken as elsewhere:
# dew-point margin 2 C, reopen wait 30 min, reopen wind 0.9 x close.
LBT_LIMITS = {"humidity_close": 95.0, "humidity_open": 90.0, "wind_close_ms": 20.0, "wind_open_ms": 18.0,
              "dewpoint_margin_c": 2.0, "reopen_wait_min": 30.0}
RE_HEAD = re.compile(r"Local time now (\d\d):(\d\d) MST, morning 18-degree twilight (\d\d):(\d\d)\. Dome now: (\w+)")
RE_ROW = re.compile(r"^(\d\d)-(\d\d) (\d\d):00\s+(.*)$")


def _mtgraham():
    from obsassist.astro.sites import SITES
    return dataclasses.replace(SITES["maunakea"], key="mtgraham", name="Mt Graham", lat_deg=32.7016,
                               lon_deg=-109.8719, elev_m=3221.0, tz_name="America/Phoenix", utc_offset_h=-7.0)


def _local(night: str, hh: int, mm: int) -> dt.datetime:
    d = dt.date.fromisoformat(night) + dt.timedelta(days=1 if hh < 12 else 0)
    return dt.datetime(d.year, d.month, d.day, hh, mm)


def real_obs(case: Dict, site) -> Dict:
    """One replay case -> the same observables dict the simulator gives. Uses only the header line,
    the measured-conditions line and the forecast rows; the observer's message is never read."""
    from obsassist.astro.ephem import NightEphem, jd_from_datetime

    lines = case["state"].splitlines()
    h = RE_HEAD.search(lines[0])
    night = case["night"]
    now = _local(night, int(h.group(1)), int(h.group(2)))
    morning = _local(night, int(h.group(3)), int(h.group(4)))
    dusk = NightEphem(site, night, 10.0).twilight.astro_dusk
    jd_now = jd_from_datetime((now + dt.timedelta(hours=7)).replace(tzinfo=dt.timezone.utc))
    fi = max([i for i, s in enumerate(lines) if s.startswith("Forecast (fetched")], default=len(lines))
    meas = [s for s in lines[:fi] if s.startswith("Latest measured conditions (")]
    num = lambda key: (lambda m: float(m.group(1)) if m else None)(re.search(key + r": ([0-9.]+)", meas[-1])) if meas else None
    rows = []
    for s in lines[fi + 1:]:
        m = RE_ROW.match(s)
        if m:
            left = m.group(4).split("|", 1)[0]
            t = dt.datetime(now.year if not (now.month == 1 and m.group(1) == "12") else now.year - 1,
                            int(m.group(1)), int(m.group(2)), int(m.group(3)))
            w, rh = re.search(r"(\d+) mph", left), re.search(r"(\d+)\s*$", left)
            wet = re.search(r"show|thunder|rain|storm|drizzle|snow", left, re.I)
            rows.append({"t": t, "wind": int(w.group(1)) * MPH if w else None, "rh": float(rh.group(1)) if rh else None,
                         "wet": bool(wet), "chance": bool(re.match(r"\s*(chance|slight)", left, re.I))})
    cur = min(rows, key=lambda r: abs((r["t"] - now).total_seconds()), default=None)
    hum, wind = num("humidity %"), num("wind mph")
    o = {"limits": dict(LBT_LIMITS), "h_since_start": (jd_now - dusk) * 24.0,
         "h_to_end": (morning - now).total_seconds() / 3600.0,
         "humidity": hum if hum is not None else (cur and cur["rh"]),
         "humidity_src": "" if hum is not None or not cur or cur["rh"] is None else " (forecast for this hour, not measured)",
         "wind": wind * MPH if wind is not None else (cur and cur["wind"]),
         "wind_src": "" if wind is not None or not cur or cur["wind"] is None else " (forecast for this hour, not measured)",
         "gust": None, "spread": None, "humidity_trend": None, "wind_trend": None, "spread_trend": None,
         "dimm": num("seeing arcsec"), "flux": None, "flux_why": "no guider data in the records"}
    wet = bool(cur and cur["wet"])
    o["precip"] = wet and not cur["chance"]
    o["precip_text"] = ("none reported" if not wet else "none measured; forecast for this hour: "
                        + ("a chance of showers or thunderstorms" if cur["chance"] else "showers or thunderstorms likely"))
    o["dome"] = "open" if h.group(5) == "open" else "closed_unknown"
    o["dome_text"] = "open" if o["dome"] == "open" else "closed (reason and duration not in the records)"
    return o


def build_real(question_name: str) -> List[Dict]:
    """Real replay cases of one question (dome or sky) as {obs, ref, tools_pick}; empty if the private
    directory is not readable. Kept in memory; nothing from here is written except aggregates."""
    f = PRIVATE / "cases" / "weather.jsonl"
    if not f.exists():
        return []
    site = _mtgraham()
    out = []
    for line in f.read_text().splitlines():
        c = json.loads(line)
        if c["question_name"] == question_name:
            out.append({"obs": real_obs(c, site), "ref": int(c["ref"][0]), "tools_pick": int(c["code_pick"])})
    return out


# ------------------------------------------------------------------ tasks, splits, run

QUESTIONS = {
    "dome": ("Dome call from the conditions above: what should the telescope do now?", DOME_OPTIONS),
    "dome2": ("Dome call from the conditions above: can the telescope observe for the next 60 min?", DOME2_OPTIONS),
    "sky": ("Which sky class describes the sky over the telescope now?", SKY),
}


def label(task: str, dome: int, sky: int) -> int:
    return {"dome": dome, "dome2": int(dome > 0), "sky": sky}[task]


def rule(task: str, o: Dict) -> int:
    return {"dome": rule_dome, "dome2": lambda x: int(rule_dome(x) > 0), "sky": rule_sky}[task](o)


def make_split(task: str, sim: List[Dict], n_train: int, seed: int = 0):
    """Split (text, label) per part, plus the rule picks and labels per part for the baseline.
    sky is asked only while the dome is open."""
    from routine.common import Split

    parts = {"train": [], "val": [], "test": []}
    for r in sim:
        if task == "sky" and r["obs"]["dome"] != "open":
            continue
        parts[night_part(r["night"])].append((summary(r["obs"]), label(task, r["dome"], r["sky"]), rule(task, r["obs"]), r["natural"], complete(r["obs"], task)))
    if len(parts["train"]) > n_train:
        keep = np.random.default_rng(seed).choice(len(parts["train"]), n_train, replace=False)
        parts["train"] = [parts["train"][i] for i in sorted(keep)]
    real = []
    for c in build_real("sky" if task == "sky" else "dome"):
        y = int(c["ref"] > 0) if task == "dome2" else c["ref"]
        tp = int(c["tools_pick"] > 0) if task == "dome2" else c["tools_pick"]
        real.append((summary(c["obs"]), y, rule(task, c["obs"]), tp, complete(c["obs"], task)))
    split = Split(*[[(t, y) for t, y, *_ in parts[p]] for p in ("train", "val", "test")],
                  real=[(t, y) for t, y, *_ in real])
    side = {p: {"y": [c[1] for c in parts[p]], "rule": [c[2] for c in parts[p]], "natural": [c[3] for c in parts[p]], "complete": [c[4] for c in parts[p]]} for p in parts}
    side["real"] = {"y": [c[1] for c in real], "rule": [c[2] for c in real], "tools": [c[3] for c in real], "complete": [c[4] for c in real]}
    return split, side


def agree(pred: List[int], y: List[int]) -> Dict:
    from routine.common import wilson

    k, n = sum(int(a == b) for a, b in zip(pred, y)), len(y)
    return {"n": n, "accuracy": round(k / n, 4) if n else None, "accuracy_ci95": wilson(k, n)}


def mix(y: List[int], k: int) -> List[int]:
    return np.bincount(np.asarray(y, dtype=int), minlength=k).tolist() if y else [0] * k


def nat(part: Dict, key: str) -> List[int]:
    """The uniformly sampled (natural class mix) moments of one part."""
    return [v for v, n in zip(part[key], part["natural"]) if n]


def baselines(task: str, side: Dict) -> Dict:
    k = len(QUESTIONS[task][1])
    out = {"class_mix": {p: mix(side[p]["y"], k) for p in side},
           "class_mix_test_natural": mix(nat(side["test"], "y"), k),
           "rule": {p: agree(side[p]["rule"], side[p]["y"]) for p in ("val", "test", "real")}}
    out["rule"]["test_natural"] = agree(nat(side["test"], "rule"), nat(side["test"], "y"))
    if side["real"]["y"]:
        out["rule"]["real_lbt_tools_forecast_rule"] = agree(side["real"]["tools"], side["real"]["y"])
    return out


def guarded(p: np.ndarray, y: np.ndarray, thr: float, ok: np.ndarray) -> Dict:
    """Acts only where the code guard passes (`complete`) and the head's confidence reaches thr."""
    from routine.common import wilson

    acted = ok & (p.max(1) >= thr)
    k, na = int((p.argmax(1) == y)[acted].sum()), int(acted.sum())
    return {"n": int(len(y)), "passes_guard": int(ok.sum()), "acted": na,
            "coverage": round(na / len(y), 4) if len(y) else None,
            "accuracy_when_acting": round(k / na, 4) if na else None, "accuracy_when_acting_ci95": wilson(k, na)}


def run(task: str, model: str, sim: List[Dict], n_train: int, out: str, targets=(0.98, 0.95)) -> Dict:
    """common.evaluate with the first gate target; the other gates are picked on the same validation
    probabilities (captured from common._probs, so nothing is recomputed) and scored on test and real."""
    from anyjev import Question
    from routine import common

    split, side = make_split(task, sim, n_train)
    q = Question.choice(QUESTIONS[task][0], list(QUESTIONS[task][1]), name=f"weather_{task}")
    seen, orig = [], common._probs

    def capture(decider, question, cases, level):
        p, s = orig(decider, question, cases, level)
        seen.append((level, p))
        return p, s

    common._probs = capture
    try:
        rep = common.evaluate(f"weather_{task}", q, split, model, out, target=targets[0],
                              notes=f"simulated nights (obsassist.weather), horizon {HORIZON_MIN} min; real = LBT replay")
    finally:
        common._probs = orig
    l2 = [p for lv, p in seen if lv == "L2"]  # val, test, real (in evaluate's order)
    yv, yt = np.array(side["val"]["y"]), np.array(side["test"]["y"])
    m = np.array(side["test"]["natural"], dtype=bool)
    rep["test_natural"] = {"L2": common.score(l2[1][m], yt[m], rep["threshold"])}
    for tgt in targets[1:]:
        thr = common.pick_threshold(l2[0].max(1), l2[0].argmax(1) == yv, tgt)
        g = {"threshold": thr, "val": common.score(l2[0], yv, thr), "test": common.score(l2[1], yt, thr),
             "test_natural": common.score(l2[1][m], yt[m], thr)}
        if len(l2) > 2:
            g["real"] = common.score(l2[2], np.array(side["real"]["y"]), thr)
        rep[f"gate_{tgt}"] = g
    parts = [("test", l2[1], yt, np.array(side["test"]["complete"], dtype=bool), np.ones(len(yt), bool)),
             ("test_natural", l2[1], yt, np.array(side["test"]["complete"], dtype=bool), m)]
    if len(l2) > 2:
        parts.append(("real", l2[2], np.array(side["real"]["y"]), np.array(side["real"]["complete"], dtype=bool),
                      np.ones(len(side["real"]["y"]), bool)))
    rep["guarded"] = {}
    for tgt in targets:
        thr = rep["threshold"] if tgt == targets[0] else rep[f"gate_{tgt}"]["threshold"]
        rep["guarded"][f"gate_{tgt}"] = {name: guarded(p[sel], y[sel], thr, ok[sel]) for name, p, y, ok, sel in parts}
    rep.update(baselines(task, side))
    rep["horizon_min"] = HORIZON_MIN
    Path(out).write_text(json.dumps(rep, indent=1, default=float))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", default="dome", help="dome, dome2, sky, or several comma-separated")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--nights", type=int, default=1400)
    ap.add_argument("--per-night", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8, help="processes for building the simulated nights")
    ap.add_argument("--build", action="store_true", help="only build cases: class mix and rule baselines (CPU)")
    a = ap.parse_args()
    sim = build_sim(a.nights, a.per_night, workers=a.workers)
    tasks = a.task.split(",")
    if a.build:
        from obsassist.sim.night import NightModel

        for i in (0, 1):  # the weather here is the weather NightModel(prog, seed) holds
            prog, _, w, seed, _, _ = sim_night(i, 20260926)
            nm = NightModel(prog, seed)
            print("night", i, "same as NightModel:", bool(np.array_equal(nm.weather.dome_ok, w.dome_ok)
                                                          and np.allclose(nm.weather.humidity, w.humidity)))
        for t in tasks:
            _, side = make_split(t, sim, a.n_train)
            print(t, "horizon", HORIZON_MIN, "moments", len(sim), json.dumps(baselines(t, side)))
        return
    short = a.model.split("/")[-1].lower()
    for t in tasks:
        run(t, a.model, sim, a.n_train, f"routine/reports/weather_{t}_{short}.json")


if __name__ == "__main__":
    main()
