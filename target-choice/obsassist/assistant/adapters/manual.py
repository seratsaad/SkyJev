"""Adapter for a real observing run at any telescope: the assistant advises, the observer acts.

Nothing here talks to observatory software. The assistant learns what happened from
  * the observer's reports in the assistant UI ("on target", "exposure started/done/aborted",
    current seeing / cloud / humidity / wind if no feed is available), and
  * optionally a FITS watcher on the night's data directory (new files -> OBJECT, EXPTIME,
    UT from the header -> an exposure on that target; an S/N measured by the observer's own
    quick-look can be entered, otherwise the ETC's prediction under the reported conditions
    is booked and marked as predicted).

The night model is the simulator's geometry and ETC with the *reported* conditions in place of
simulated weather (persistence between reports), rebuilt when conditions change. The clock is
real time (or a fixed "now" for rehearsals).
"""

from __future__ import annotations

import datetime as _dt
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from obsassist.assistant.adapters.base import ObservatoryAdapter
from obsassist.astro.ephem import NightEphem
from obsassist.planning.planner import Nowcast, candidates, project
from obsassist.sim.night import NightModel
from obsassist.targets import Program, load_program
from obsassist.weather import NightWeather


def _reported_weather(model_t: np.ndarray, site, reports: List[Dict[str, Any]]) -> NightWeather:
    """Weather arrays from observer reports: each report holds from its time until the next."""
    n = len(model_t)
    seeing = np.full(n, site.seeing_median)
    cloud = np.zeros(n)
    hum = np.full(n, 30.0)
    wind = np.full(n, 5.0)
    temp = np.full(n, site.temp_c)
    for r in sorted(reports, key=lambda r: r["t"]):
        m = model_t >= r["t"]
        if r.get("seeing") is not None:
            seeing[m] = r["seeing"]
        if r.get("cloud") is not None:
            cloud[m] = r["cloud"]
        if r.get("humidity") is not None:
            hum[m] = r["humidity"]
        if r.get("wind_ms") is not None:
            wind[m] = r["wind_ms"]
        if r.get("temp_c") is not None:
            temp[m] = r["temp_c"]
    a, b = 17.62, 243.12
    g = np.log(np.clip(hum, 1, 100) / 100.0) + a * temp / (b + temp)
    dew = b * g / (a - g)
    return NightWeather(
        t_min=model_t,
        seeing=seeing,
        seeing_dimm=seeing.copy(),
        cloud_mag=cloud,
        sky_state=np.where(cloud < 0.05, 0, np.where(cloud < 0.7, 1, 2)),
        humidity=hum,
        wind_ms=wind,
        wind_gust_ms=wind * 1.2,
        wind_dir=np.zeros(n),
        temp_c=temp,
        dewpoint_c=dew,
        pressure_hpa=np.full(n, site.pressure_hpa),
        dome_ok=np.ones(n, dtype=bool),
        seed=-1,
        regime="reported",
    )


class ManualAdapter(ObservatoryAdapter):
    name = "manual"
    can_actuate = False

    def __init__(
        self,
        program: str | Program,
        data_dir: Optional[str] = None,
        now: Optional[str] = None,
        keck_instrument: Optional[str] = None,
    ):
        self.program = program if isinstance(program, Program) else load_program(program)
        self.keck_instrument = keck_instrument  # e.g. "LRIS": print KTL lines, read DCS telemetry
        self.reports: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.fixed_now = now
        grid = NightEphem(self.program.site, self.program.date).t_min
        self.model = NightModel(self.program, weather=_reported_weather(grid, self.program.site, []))
        self.state = self.model.initial_state()
        self.on_target: Optional[int] = None
        self.exposing: Optional[Dict[str, Any]] = None
        self.data_dir = Path(data_dir) if data_dir else None
        self._seen_files: set = set()
        self._lock = threading.Lock()
        self.frames_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ time
    def now_min(self) -> float:
        if self.fixed_now:
            t = _dt.datetime.fromisoformat(self.fixed_now.replace("Z", "+00:00"))
        else:
            t = _dt.datetime.now(_dt.timezone.utc)
        return float(np.clip(self.model.minute_of_utc(t), 0, self.model.ephem.t_min[-1]))

    def log(self, who: str, text: str, level: str = "info"):
        t = self.now_min()
        self.events.append(
            {"t": t, "utc": self.model.ephem.utc(t).strftime("%H:%M:%S"), "who": who, "text": text, "level": level}
        )

    # ------------------------------------------------------------------ observer reports
    def report_conditions(self, seeing=None, cloud=None, humidity=None, wind_mph=None, temp_c=None):
        with self._lock:
            self.reports.append(
                {
                    "t": self.now_min(),
                    "seeing": seeing,
                    "cloud": cloud,
                    "humidity": humidity,
                    "wind_ms": None if wind_mph is None else wind_mph / 2.237,
                    "temp_c": temp_c,
                }
            )
            state = self.state
            self.model = NightModel(
                self.program, weather=_reported_weather(self.model.ephem.t_min, self.model.site, self.reports)
            )
            self.state = state
        self.log("OBS", f"conditions: seeing {seeing}, cloud {cloud}, RH {humidity}, wind {wind_mph} mph")

    def report_on_target(self, name: str):
        i = self.program.target_index(name)
        self.on_target = i
        self.state.tel_target, self.state.acquired = i, True
        j = self.model.idx(self.now_min())
        self.state.tel_alt, self.state.tel_az = float(self.model.geo["alt"][i, j]), float(self.model.geo["az"][i, j])
        self.state.setup = self.program.targets[i].cfg().setup_id
        self.log("OBS", f"on target {name}")

    def report_exposure_started(self, name: str, t_exp: float, n: int = 1):
        self.exposing = {"target": self.program.target_index(name), "t0": self.now_min(), "t_exp": float(t_exp), "n": n}
        self.log("OBS", f"started {n} x {t_exp:.0f} s on {name}")

    def report_exposure_done(
        self,
        name: Optional[str] = None,
        t_exp: Optional[float] = None,
        measured_snr: Optional[float] = None,
        aborted: bool = False,
    ):
        ex = self.exposing or {}
        i = self.program.target_index(name) if name else ex.get("target")
        if i is None:
            return
        t_exp = float(t_exp or ex.get("t_exp", 0.0))
        t0 = ex.get("t0", self.now_min() - t_exp / 60.0)
        self.exposing = None
        if aborted:
            self.log("OBS", f"aborted exposure on {self.program.targets[i].name}", level="warn")
            return
        if measured_snr is not None:
            snr, how = float(measured_snr), "measured"
        else:
            out = self.model.exposure(i, t0, t_exp, self.model.defocus(self.state, t0))
            snr, how = out.snr, "predicted"
        self.state.snr2[i] += snr**2
        self.state.n_exp[i] += 1
        self.state.open_s[i] += t_exp
        self.state.t = self.now_min()
        tg = self.program.targets[i]
        self.log(
            "DATA",
            f"{tg.name}: {t_exp:.0f} s, S/N {snr:.1f} ({how}) -> {math.sqrt(self.state.snr2[i]):.1f}/{tg.snr_goal:.0f}",
        )

    # ------------------------------------------------------------------ FITS watcher
    def poll_fits(self) -> List[str]:
        """Book new FITS files in the data directory (OBJECT / EXPTIME / IMAGETYP from the header)."""
        if not self.data_dir or not self.data_dir.exists():
            return []
        from astropy.io import fits

        new = []
        for p in sorted(self.data_dir.glob("*.fits")):
            if p.name in self._seen_files:
                continue
            self._seen_files.add(p.name)
            try:
                h = fits.getheader(p)
            except Exception:
                continue
            obj, texp = str(h.get("OBJECT", "")).strip(), float(h.get("EXPTIME", 0) or 0)
            typ = str(h.get("IMAGETYP", h.get("EXPTYPE", "object"))).lower()
            rec = {"file": p.name, "object": obj, "exptime": texp, "image_type": typ}
            self.frames_log.append(rec)
            new.append(p.name)
            if typ.startswith("obj") and obj:
                hits = [
                    k
                    for k, t in enumerate(self.program.targets)
                    if t.name.lower().replace(" ", "") in obj.lower().replace(" ", "")
                    or obj.lower().replace(" ", "") in t.name.lower().replace(" ", "")
                ]
                if len(hits) == 1:
                    self.report_exposure_done(self.program.targets[hits[0]].name, texp)
        return new

    # ------------------------------------------------------------------ snapshot
    def snapshot(self) -> Dict[str, Any]:
        self.poll_fits()
        m = self.model
        t = self.now_min()
        s = self.state.copy()
        s.t = t
        now = Nowcast.from_history(m, t)
        cands = candidates(m, s, now)
        proj = project(m, s, now)
        j = m.idx(t)
        snr = np.sqrt(s.snr2)
        tw = m.ephem.twilight
        rep = self.reports[-1] if self.reports else {}
        return {
            "t": t,
            "utc": m.ephem.utc(t).strftime("%Y-%m-%d %H:%M:%S"),
            "local": m.ephem.local(t).strftime("%H:%M:%S"),
            "site": m.site.name,
            "telescope": m.tel.name,
            "program": self.program.name,
            "twilight": m.ephem.twilight_phase(j),
            "sun_alt": float(m.ephem.sun_alt[j]),
            "moon": {"alt": float(m.ephem.moon_alt[j]), "illum": float(m.ephem.moon_illum[j])},
            "night": {
                "minutes_left": max(0.0, m.t_end - t),
                "twi12_dusk": m.ephem.utc(m.ephem.minute_of(tw.nautical_dusk)).strftime("%H:%M"),
                "twi12_dawn": m.ephem.utc(m.ephem.minute_of(tw.nautical_dawn)).strftime("%H:%M"),
            },
            "dome": {"open": True, "reason": ""},
            "weather": {
                "dimm": rep.get("seeing"),
                "humidity": rep.get("humidity") or 30.0,
                "wind_mph": (rep.get("wind_ms") or 5.0) * 2.237,
                "limits": {
                    "humidity": m.site.limits.humidity_close,
                    "wind_close_mph": m.site.limits.wind_close_ms * 2.237,
                },
            },
            "tcs": (self.keck_instrument and self._keck_telemetry())
            or {
                "name": self.program.targets[self.on_target].name if self.on_target is not None else "",
                "state": "guiding" if self.on_target is not None else "unknown",
            },
            "guider": {"flux_ratio": 10 ** (-0.4 * (rep.get("cloud") or 0.0))},
            "focus": {"dT": 0.0},
            "arms": {},
            "score": m.score(s),
            "max_score": m.max_score(),
            "targets": [
                {
                    "i": i,
                    "name": tg.name,
                    "priority": tg.priority,
                    "kind": tg.kind,
                    "snr": float(snr[i]),
                    "goal": tg.snr_goal,
                    "unit": tg.snr_unit,
                    "n_exp": int(s.n_exp[i]),
                    "airmass": float(m.geo["airmass"][i, j]),
                    "alt": float(m.geo["alt"][i, j]),
                    "moon_sep": float(m.geo["moon_sep"][i, j]),
                    "visible": m.is_visible(i, t),
                    "announced": t >= m.appears[i],
                    "done": bool(snr[i] >= tg.snr_goal * 0.999),
                    "notes": tg.notes,
                    "max_seeing": tg.max_seeing,
                    "window": tg.window,
                }
                for i, tg in enumerate(self.program.targets)
            ],
            "plan": {
                "candidates": [c.to_dict() for c in cands],
                "projection": proj.to_dict(m),
                "nowcast": now.__dict__,
            },
            "log": self.events[-60:],
            "frames": self.frames_log[-30:],
            "assistant_control": False,
        }

    def etc(self, target: str, seeing=None, cloud=None, t_exp=None) -> Dict[str, Any]:
        from obsassist.etc import compute_rates, plan_exposures

        m = self.model
        i = self.program.target_index(target)
        tg = self.program.targets[i]
        t = self.now_min()
        j = m.idx(t)
        now = Nowcast.from_history(m, t)
        r = compute_rates(
            tg.cfg(),
            m.tel,
            m.site,
            tg.source,
            seeing_500=seeing if seeing is not None else now.seeing,
            airmass=float(np.clip(m.geo["airmass"][i, j], 1, 6)),
            alt=float(m.geo["alt"][i, j]),
            cloud_mag=cloud if cloud is not None else now.cloud,
            sun_alt=m.ephem.sun_alt[j],
            moon_alt=m.ephem.moon_alt[j],
            moon_phase_angle=m.ephem.moon_phase_angle[j],
            moon_sep=m.geo["moon_sep"][i, j],
            lam=tg.lam_ref(),
            unit=tg.snr_unit,
        )
        p = plan_exposures(r, tg.cfg(), tg.snr_goal, float(np.sqrt(self.state.snr2[i])), t_exp or tg.t_exp)
        return {
            "target": tg.name,
            "plan": {"n_exp": p.n_exp, "t_exp": p.t_exp, "wall_s": p.wall_s, "snr_final": p.snr_final},
            "fwhm": float(r.fwhm),
            "sky_mag": float(r.sky_mag),
            "regime": r.regime(),
        }

    def instructions(self, target: str, t_exp: float, n_exp: int) -> List[str]:
        """Steps for the operator and the instrument GUI (or, at Keck, the KTL lines to type)."""
        from obsassist.astro.ephem import fmt_dec, fmt_ra

        t = self.program.targets[self.program.target_index(target)]
        rot = (
            "rotator OFF (MIKE)"
            if t.cfg().instrument == "MIKE"
            else (f"rotator PA {t.pa:.1f} (EQU)" if t.pa is not None else "rotator at the parallactic angle (HRZ)")
        )
        steps = [f"Operator: go to {t.name}  {fmt_ra(t.ra)} {fmt_dec(t.dec)} J2000, {rot}"]
        if self.keck_instrument:
            from obsassist.assistant.adapters.ktl import keck_instructions

            steps += keck_instructions(self.keck_instrument, t.name, t_exp, 1)
        else:
            steps.append(f"Instrument: Object '{t.name}', ImageType Object, Exp.Time {t_exp:.0f} s, Loops 1, Start")
        steps.append(f"(plan: {n_exp} exposure(s) of {t_exp:.0f} s at the current conditions; re-checked after each)")
        return steps

    def _keck_telemetry(self) -> Dict[str, Any]:
        """DCS telemetry through KTL `show`, when running on a Keck workstation (read-only)."""
        from obsassist.assistant.adapters.ktl import DCS_KEYWORDS, run_show

        try:
            dcs = run_show("dcs", list(DCS_KEYWORDS))
        except Exception:
            return {}
        guiding = dcs.get("GUIDING", "").lower() in ("1", "true", "yes")
        return {
            "name": dcs.get("TARGNAME", ""),
            "state": "guiding" if guiding else "tracking",
            "airmass": dcs.get("AIRMASS"),
            "alt": dcs.get("EL"),
            "az": dcs.get("AZ"),
        }

    def describe(self) -> Dict[str, Any]:
        return {
            "adapter": self.name,
            "program": self.program.name,
            "data_dir": str(self.data_dir) if self.data_dir else None,
            "can_actuate": False,
        }
