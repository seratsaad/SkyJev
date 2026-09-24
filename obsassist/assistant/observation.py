"""What the observer can see, as a dict, and its rendering as text for the decision model.

Two producers build the same `Observation` dict:
  * `from_model(model, state, now)`   - the simulator (dataset building, evaluation), and
  * `from_snapshot(snapshot)`         - the live console API (or any adapter that fills a snapshot),
so the text the model is trained on and the text it reads at the telescope come from one
renderer (`render`). Only observables go in: DIMM, guider, weather station, what has been
taken, the planner's per-candidate numbers (computed from observables) and the PI's notes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

REGRET_LEVELS = ("best (within 0.5%)", "0.5-2% worse", "2-5% worse", "5-15% worse", "over 15% worse")
REGRET_EDGES = (0.005, 0.02, 0.05, 0.15)  # fractions of the program's maximum score
REGRET_CENTERS = (0.0, 0.0125, 0.035, 0.10, 0.25)


def regret_level(regret_frac: float) -> int:
    return int(np.searchsorted(REGRET_EDGES, regret_frac, side="right"))


def _trend(values: List[float], per_hours: float) -> Optional[float]:
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) < 5:
        return None
    x = np.linspace(0, per_hours, len(v))
    return float(np.polyfit(x, v, 1)[0])


# ---------------------------------------------------------------------------
# producers
# ---------------------------------------------------------------------------
def from_model(model, state, now, cands=None, t_min: Optional[float] = None) -> Dict[str, Any]:
    """Observation from the simulator (observables only)."""
    from obsassist.planning.planner import candidates as _cands

    t = state.t if t_min is None else t_min
    j = model.idx(t)
    w = model.weather
    j30 = max(0, j - int(30 / model.dt))
    cands = cands if cands is not None else _cands(model, state, now)
    dimm_hist = [float(x) for x in w.seeing_dimm[j30 : j + 1]]
    hum_hist = [float(x) for x in w.humidity[j30 : j + 1]]
    snr = np.sqrt(state.snr2)
    done = int((snr >= model.goals * 0.999).sum())
    remaining_p1 = sum(
        1
        for i, tg in enumerate(model.targets)
        if tg.priority == 1 and snr[i] < tg.snr_goal * 0.999 and t >= model.appears[i]
    )
    cur = model.targets[state.tel_target].name if state.tel_target >= 0 else None
    return {
        "utc": model.ephem.utc(t).strftime("%H:%M"),
        "night_left_min": float(max(0.0, model.t_end - t)),
        "twilight": model.ephem.twilight_phase(j),
        "moon_alt": float(model.ephem.moon_alt[j]),
        "moon_illum": float(model.ephem.moon_illum[j]),
        "site": model.site.name,
        "telescope": model.tel.name,
        "dimm": now.seeing,
        "dimm_trend": _trend(dimm_hist, 0.5),
        "site_median_seeing": model.site.seeing_median,
        "guider_flux_ratio": float(10 ** (-0.4 * now.cloud)),
        "cloud_mag_est": now.cloud,
        "humidity": float(w.humidity[j]),
        "humidity_trend": _trend(hum_hist, 0.5),
        "humidity_limit": model.site.limits.humidity_close,
        "wind_mph": float(w.wind_ms[j] * 2.237),
        "wind_limit_mph": model.site.limits.wind_close_ms * 2.237,
        "dome_open": bool(model.dome_ok[j]),
        "focus_dT": float(abs(model.temperature(t) - state.focus_temp)),
        "current": cur,
        "done": done,
        "total": model.N,
        "remaining_p1": remaining_p1,
        "score": model.score(state),
        "max_score": model.max_score(),
        "candidates": [_cand_dict(model, c, state) for c in cands],
    }


def _cand_dict(model, c, state) -> Dict[str, Any]:
    tg = model.targets[c.i]
    j = model.idx(state.t)
    alt_now = float(model.geo["alt"][c.i, j])
    alt_later = float(model.geo["alt"][c.i, min(model.ephem.n - 1, j + int(60 / model.dt))])
    return {
        "name": c.name,
        "priority": c.priority,
        "kind": tg.kind,
        "feasible": c.feasible,
        "note": c.note,
        "snr_now": c.snr_now,
        "goal": c.goal,
        "unit": tg.snr_unit,
        "snr1": c.snr1,
        "n_needed": c.n_needed,
        "t_exp": c.t_exp,
        "time_needed_min": c.time_needed_min,
        "overhead_min": c.overhead_min,
        "window_left_min": c.window_left_min,
        "airmass": c.airmass,
        "rising": alt_later > alt_now,
        "fwhm_pred": c.fwhm_pred,
        "max_seeing": tg.max_seeing,
        "efficiency": c.efficiency,
        "urgency": c.urgency,
        "merit": c.merit,
        "moon_sep": float(model.geo["moon_sep"][c.i, j]),
        "has_window": tg.window is not None,
        "notes": tg.notes,
        "is_current": state.tel_target == c.i,
    }


def from_snapshot(snap: Dict[str, Any]) -> Dict[str, Any]:
    """Observation from a console snapshot (GET /api/state), the live path."""
    hist = snap.get("history") or {}
    plan = snap.get("plan") or {}
    cmap = {c["name"]: c for c in plan.get("candidates", [])}
    now = plan.get("nowcast") or {}
    tmap = {t["name"]: t for t in snap.get("targets", [])}
    cands = []
    for name, c in cmap.items():
        t = tmap.get(name, {})
        cands.append(
            {
                "name": name,
                "priority": c.get("priority"),
                "kind": t.get("kind", "science"),
                "feasible": c.get("feasible"),
                "note": c.get("note", ""),
                "snr_now": c.get("snr_now"),
                "goal": c.get("goal"),
                "unit": t.get("unit", "per_A"),
                "snr1": c.get("snr1"),
                "n_needed": c.get("n_needed"),
                "t_exp": c.get("t_exp"),
                "time_needed_min": c.get("time_needed_min"),
                "overhead_min": c.get("overhead_min"),
                "window_left_min": c.get("window_left_min"),
                "airmass": c.get("airmass"),
                "rising": None,
                "fwhm_pred": c.get("fwhm_pred"),
                "max_seeing": t.get("max_seeing"),
                "efficiency": c.get("efficiency"),
                "urgency": c.get("urgency"),
                "merit": c.get("merit"),
                "moon_sep": t.get("moon_sep"),
                "has_window": bool(t.get("window")),
                "notes": t.get("notes", ""),
                "is_current": snap.get("tcs", {}).get("name") == name,
            }
        )
    w = snap.get("weather", {})
    dimm = [x for x in (hist.get("dimm") or [])[-30:]]
    hum = [x for x in (hist.get("humidity") or [])[-30:]]
    done = sum(1 for t in snap.get("targets", []) if t.get("done"))
    rem_p1 = sum(
        1 for t in snap.get("targets", []) if t.get("priority") == 1 and not t.get("done") and t.get("announced", True)
    )
    fr = snap.get("guider", {}).get("flux_ratio")
    return {
        "utc": snap.get("utc", "")[11:16],
        "night_left_min": snap.get("night", {}).get("minutes_left", 0.0),
        "twilight": snap.get("twilight", ""),
        "moon_alt": snap.get("moon", {}).get("alt"),
        "moon_illum": snap.get("moon", {}).get("illum"),
        "site": snap.get("site"),
        "telescope": snap.get("telescope"),
        "dimm": now.get("seeing", w.get("dimm")),
        "dimm_trend": _trend(dimm, 0.5),
        "site_median_seeing": None,
        "guider_flux_ratio": fr if fr is not None else 10 ** (-0.4 * now.get("cloud", 0.0)),
        "cloud_mag_est": now.get("cloud", 0.0),
        "humidity": w.get("humidity"),
        "humidity_trend": _trend(hum, 0.5),
        "humidity_limit": (w.get("limits") or {}).get("humidity"),
        "wind_mph": w.get("wind_mph"),
        "wind_limit_mph": (w.get("limits") or {}).get("wind_close_mph"),
        "dome_open": snap.get("dome", {}).get("open"),
        "focus_dT": (snap.get("focus") or {}).get("dT", 0.0),
        "current": snap.get("tcs", {}).get("name"),
        "done": done,
        "total": len(snap.get("targets", [])),
        "remaining_p1": rem_p1,
        "score": snap.get("score"),
        "max_score": snap.get("max_score"),
        "candidates": cands,
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _f(x, fmt="{:.2f}", none="?"):
    return none if x is None else fmt.format(x)


def _cand_line(c: Dict[str, Any]) -> str:
    """One candidate, compactly: name | P | S/N now/goal | minutes to finish now | minutes left |
    airmass and trend | efficiency now vs its best | flags."""
    done = c["snr_now"] is not None and c["goal"] and c["snr_now"] >= c["goal"] * 0.999
    need = "done" if done else f"{c['time_needed_min']:.0f}"
    left = ">600" if (c["window_left_min"] or 0) > 600 else f"{c['window_left_min']:.0f}"
    flags = []
    if c.get("max_seeing"):
        flags.append(f"FWHM<{c['max_seeing']:.1f} (pred {c['fwhm_pred']:.2f})")
    if c.get("has_window"):
        flags.append("window")
    if c.get("kind") in ("standard", "backup", "too"):
        flags.append(c["kind"])
    if c.get("is_current"):
        flags.append("on it")
    trend = "" if c.get("rising") is None else ("+" if c["rising"] else "-")
    return (
        f"{c['name']} | P{c['priority']} | {c['snr_now']:.0f}/{c['goal']:.0f} | {need} | {left} | "
        f"{c['airmass']:.2f}{trend} | {c['efficiency']:.2f}" + (f" | {', '.join(flags)}" if flags else "")
    )


def render(obs: Dict[str, Any], focus: Optional[str] = None, max_cands: int = 6, notes: bool = True) -> str:
    """Compact text for the decision model (~250 tokens). `focus` names the candidate under evaluation."""
    dt = obs.get("dimm_trend")
    ht = obs.get("humidity_trend")
    cloud = max(0.0, -2.5 * np.log10(max(obs["guider_flux_ratio"] or 1.0, 1e-3)))
    lines = [
        f"UT {obs['utc']}, {obs['night_left_min'] / 60:.1f} h left ({obs['twilight']}). "
        f'Seeing {_f(obs["dimm"])}"'
        + (f' ({dt:+.2f}"/h)' if dt is not None else "")
        + f", cloud {cloud:.2f} mag, RH {obs['humidity']:.0f}%"
        + (f" ({ht:+.0f}%/h)" if ht is not None else "")
        + f" limit {obs['humidity_limit']:.0f}, wind {obs['wind_mph']:.0f} mph limit {obs['wind_limit_mph']:.0f}, "
        + f"dome {'open' if obs['dome_open'] else 'CLOSED'}, focus dT {obs['focus_dT']:.1f} C.",
        f"Done {obs['done']}/{obs['total']}, P1 left {obs['remaining_p1']}, "
        f"score {obs['score']:.1f}/{obs['max_score']:.0f}." + (f" On {obs['current']}." if obs.get("current") else ""),
        "Candidates (name | priority | S/N now/goal | min to finish | min left | "
        "airmass +rising/-setting | efficiency):",
    ]
    feas = [c for c in obs["candidates"] if c.get("feasible")]
    feas.sort(key=lambda c: -(c.get("merit") or 0))
    shown = feas[:max_cands]
    if focus and focus not in [c["name"] for c in shown]:
        shown = shown[:-1] + [c for c in feas if c["name"] == focus]
    lines += [_cand_line(c) for c in shown] or ["none"]
    if focus:
        fc = next((c for c in obs["candidates"] if c["name"] == focus), None)
        if fc is not None:
            lines.append(f"Evaluate: {focus}." + (f" PI note: {fc['notes']}" if notes and fc.get("notes") else ""))
    return "\n".join(lines)
