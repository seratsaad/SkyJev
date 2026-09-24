"""The real-time observatory behind the console.

Three actors, as in a real control room:

* the telescope operator (TO) runs the TCS and the dome: the observer *requests* a target,
  the TO slews, acquires the guide star and reports; the TO closes for weather;
* the instrument, one panel per arm (MIKE and LRIS are dual-arm): exposure time, loops,
  image type, binning, readout speed; Start / Pause / Resume / Stop (read out now) / Abort;
* the data system writes a FITS file per readout into the night's data directory and runs a
  quick-look on it.

Physics comes from `NightModel` (the same code the planner and the oracle use). The clock
runs at a chosen speed-up; `tick()` advances it and emits events.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as _dt
import json
import math
import shlex
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from obsassist.astro.ephem import fmt_dec, fmt_ra, parse_dec, parse_ra
from obsassist.astro.sky import delivered_fwhm, sky_ab
from obsassist.etc import compute_rates, saturation_factor
from obsassist.instruments.base import InstrumentConfig, get_config
from obsassist.planning.planner import Nowcast, candidates, project
from obsassist.sim.night import NightModel, ObsState
from obsassist.targets import Program

# arms per instrument: (arm name, config key, file prefix)
INSTRUMENT_ARMS = {
    "LRIS": [("blue", "LRIS-B600", "b"), ("red", "LRIS-R400", "r")],
    "MIKE": [("blue", "MIKE-BLUE", "b"), ("red", "MIKE-RED", "r")],
    "DEIMOS": [("main", "DEIMOS-600ZD", "d")],
    "MOSFIRE": [("main", "MOSFIRE-H", "m")],
    "HIRES": [("main", "HIRES-C2", "h")],
    "IMACS": [("main", "IMACS-f2-300", "i")],
    "LDSS3": [("main", "LDSS3-VPHALL", "c")],
    "FIRE": [("main", "FIRE-ECH", "f")],
}
IMAGE_TYPES = ("object", "flat", "arc", "bias", "dark", "sky")


@dataclass
class Arm:
    name: str
    cfg: InstrumentConfig
    prefix: str
    exp_time: float = 900.0
    loops: int = 1
    doing: int = 0
    image_type: str = "object"
    object: str = ""
    comment: str = ""
    binning: tuple = (1, 1)
    speed: str = "Slow"
    readout: str = "Full"
    state: str = "idle"  # idle | exposing | paused | reading | clearing
    image_no: int = 1
    t_open: Optional[float] = None  # sim minute the shutter last opened
    segments: List[tuple] = field(default_factory=list)  # (t0, t1) open-shutter intervals
    elapsed_s: float = 0.0
    read_end: Optional[float] = None
    read_total_s: float = 0.0
    ut_start: str = ""
    stop_requested: bool = False
    target_at_start: Optional[int] = None
    on_target_at_start: bool = True
    last_file: str = ""
    lamp: str = "off"

    def readout_s(self) -> float:
        base = (
            self.cfg.readout_speeds.get(self.speed.lower(), self.cfg.readout_s)
            if self.cfg.readout_speeds
            else self.cfg.readout_s
        )
        if not self.cfg.readout_speeds and self.speed.lower() == "fast":
            base *= 0.55
        # the configuration's readout times are quoted at its own binning: scale from that binning
        ref = max(1, self.cfg.bin_spatial * self.cfg.bin_spectral)
        return base / (max(1, self.binning[0] * self.binning[1]) / ref) ** 0.5

    def lines_total(self) -> int:
        return int(4224 / self.binning[1])

    def to_dict(self, now: float) -> dict:
        el = self.elapsed_s + (
            (now - self.t_open) * 60.0 if self.state == "exposing" and self.t_open is not None else 0.0
        )
        prog = None
        msg = ""
        if self.state == "reading" and self.read_end is not None:
            left = max(0.0, (self.read_end - now) * 60.0)
            frac = 1.0 - left / max(self.read_total_s, 1e-6)
            prog = frac
            msg = f"{left:.0f} seconds, {int(frac * self.lines_total())}/{self.lines_total()} lines"
        elif self.state == "exposing":
            msg = f"exposing {el:.0f}/{self.exp_time:.0f} s"
        elif self.state == "paused":
            msg = f"paused at {el:.0f}/{self.exp_time:.0f} s"
        return {
            "name": self.name,
            "config": self.cfg.key,
            "description": self.cfg.description,
            "prefix": self.prefix,
            "exp_time": self.exp_time,
            "loops": self.loops,
            "doing": self.doing,
            "image_type": self.image_type,
            "object": self.object,
            "comment": self.comment,
            "binning": list(self.binning),
            "speed": self.speed,
            "readout": self.readout,
            "state": self.state,
            "image_no": self.image_no,
            "elapsed_s": round(el, 1),
            "remaining_s": round(max(0.0, self.exp_time - el), 1),
            "read_progress": prog,
            "message": msg,
            "ut_start": self.ut_start,
            "shutter": "Open" if self.state == "exposing" else "Closed",
            "last_file": self.last_file,
            "lamp": self.lamp,
            "slit": f"{self.cfg.slit_width:.2f}x{self.cfg.slit_length:.2f}",
            "readout_s": round(self.readout_s(), 1),
        }


@dataclass
class TCS:
    state: str = "parked"  # parked | slewing | acquiring | tracking | guiding | limit
    target: Optional[int] = None  # program target index, or None for manual coordinates
    name: str = "zenith"
    ra: float = 0.0
    dec: float = 0.0
    pa: Optional[float] = None  # None = parallactic
    offset: tuple = (0.0, 0.0)  # arcsec
    busy_until: Optional[float] = None
    slew_from: tuple = (89.0, 180.0)
    alt: float = 89.0
    az: float = 180.0


class Observatory:
    def __init__(
        self,
        program: Program,
        seed: int = 0,
        data_root: Optional[str] = None,
        speed: float = 60.0,
        write_frames: bool = True,
        allow_assistant_control: bool = False,
    ):
        self.program = program
        self.seed = seed
        self.model = NightModel(program, seed=seed)
        self.state: ObsState = self.model.initial_state()
        self.t = float(max(0.0, self.model.ephem.minute_of(self.model.ephem.twilight.sunset)))
        self.speed = float(speed)
        self.paused = False
        self._n_alerts = 0  # alerts + TO warnings raised so far (fast_forward stops on them)
        self.allow_assistant_control = allow_assistant_control
        self.rng = np.random.default_rng(seed + 1000)
        inst = self._instrument_name()
        self.instrument = inst
        self.arms: Dict[str, Arm] = {}
        for name, key, prefix in INSTRUMENT_ARMS.get(inst, [("main", program.targets[0].config, "a")]):
            cfg = get_config(key)
            # start in the readout speed the configuration (and so the planner) assumes, e.g. LDSS3's
            # documented setup reads out fast (30 s); its slow mode takes 166 s
            speed = next(
                (
                    k.capitalize()
                    for k, v in cfg.readout_speeds.items()
                    if k.lower() in ("slow", "fast", "turbo") and abs(v - cfg.readout_s) < 1e-6
                ),
                "Slow",
            )
            self.arms[name] = Arm(
                name=name,
                cfg=cfg,
                prefix=prefix,
                exp_time=900.0,
                binning=(cfg.bin_spatial, cfg.bin_spectral),
                speed=speed,
            )
        self.tcs = TCS()
        self.dome_open = False
        self.dome_reason = "afternoon: dome closed"
        self.focus_busy_until: Optional[float] = None
        self.log: Deque[dict] = deque(maxlen=2000)
        self.events: Deque[dict] = deque(maxlen=500)
        self.frames: List[dict] = []
        self.guider = {"fwhm": None, "flux_ratio": None, "state": "off", "t": None}
        self.history: Dict[str, List] = {
            k: []
            for k in (
                "t",
                "utc",
                "dimm",
                "guider_fwhm",
                "humidity",
                "wind",
                "gust",
                "temp",
                "dewpoint",
                "pressure",
                "flux_ratio",
                "wind_dir",
            )
        }
        self._last_hist = -1e9
        self._last_plan = -1e9
        self.plan_cache: dict = {}
        self.pool = cf.ThreadPoolExecutor(max_workers=2)
        self.pending: List[cf.Future] = []
        self.write_frames = write_frames
        date_ut = self.model.ephem.utc(self.model.t_start).strftime("%Y%m%d")
        from obsassist.paths import FRAMES

        root = Path(data_root) if data_root else FRAMES
        self.data_dir = root / f"{self.model.tel.key}" / f"ut{date_ut}"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._last_real = time.time()
        self.say("SYS", f"Night of {program.date} at {self.model.site.name}. {program.name}.")
        self.say("SYS", f"Data directory {self.data_dir}")
        self.say(
            "TO", "Good evening. Dome is closed; we open at sunset if the weather allows. Afternoon cals are yours."
        )

    # ------------------------------------------------------------------ helpers
    def _instrument_name(self) -> str:
        names = {get_config(t.config).instrument for t in self.program.targets}
        return sorted(names)[0] if len(names) == 1 else get_config(self.program.targets[0].config).instrument

    def utc(self, t: Optional[float] = None) -> _dt.datetime:
        return self.model.ephem.utc(self.t if t is None else t)

    def say(self, who: str, text: str, level: str = "info", **extra) -> None:
        rec = {
            "t": round(float(self.t), 3),
            "utc": self.utc().strftime("%H:%M:%S"),
            "who": who,
            "text": text,
            "level": level,
        }
        rec.update(extra)
        if level == "alert" or (who == "TO" and level == "warn"):
            self._n_alerts = getattr(self, "_n_alerts", 0) + 1
        self.log.append(rec)
        self.events.append({"type": "log", **rec})

    # ------------------------------------------------------------------ clock
    def tick(self, real_dt: Optional[float] = None) -> None:
        now = time.time()
        if real_dt is None:
            real_dt = now - self._last_real
        self._last_real = now
        if self.paused:
            self._collect()
            return
        self.advance(min(real_dt, 1.0) * self.speed / 60.0)

    def advance(self, dmin: float, stop_on_alert: bool = False) -> None:
        """Advance simulated time by dmin minutes, in steps that do not skip events. With
        `stop_on_alert`, stop right after a step that raised an alert or a TO warning (ToO
        arrival, dome closing, elevation limit)."""
        end = min(self.t + dmin, self.model.ephem.t_min[-1])
        mark = self._n_alerts
        while self.t < end - 1e-9:
            step = min(end - self.t, 0.25)
            nxt = self._next_event_time()
            if nxt is not None and self.t < nxt < self.t + step:
                step = nxt - self.t + 1e-6
            self.t += step
            self._update()
            if stop_on_alert and self._n_alerts != mark:
                break
        self._collect()

    def fast_forward(self) -> None:
        """Jump to the next thing that needs the observer (end of exposure / readout / slew /
        focus, a ToO alert), stopping early if the TO warns (dome closing, elevation limit)."""
        nxt = self._next_event_time(include_too=False)
        if nxt is None:  # nothing running: a short hop, unless a ToO arrives sooner
            nxt = self.t + 5.0
        too = self._next_too_time()
        if too is not None:
            nxt = min(nxt, too)
        self.advance(max(0.0, nxt - self.t) + 0.01, stop_on_alert=True)

    def _next_too_time(self) -> Optional[float]:
        """Arrival time of the next ToO alert not raised yet."""
        ts = [
            float(self.model.appears[i])
            for i, tg in enumerate(self.program.targets)
            if tg.appears_at and not getattr(self, f"_too_{i}", False) and self.model.appears[i] > self.t
        ]
        return min(ts) if ts else None

    def _next_event_time(self, include_too: bool = True) -> Optional[float]:
        cands = []
        for a in self.arms.values():
            if a.state == "exposing" and a.t_open is not None:
                cands.append(a.t_open + (a.exp_time - a.elapsed_s) / 60.0)
            if a.state in ("reading", "clearing") and a.read_end is not None:
                cands.append(a.read_end)
        if self.tcs.busy_until is not None and self.tcs.state in ("slewing", "acquiring"):
            cands.append(self.tcs.busy_until)
        if self.focus_busy_until is not None:
            cands.append(self.focus_busy_until)
        too = self._next_too_time() if include_too else None
        if too is not None:  # step exactly onto a ToO alert's arrival time
            cands.append(too)
        return min(cands) if cands else None

    # ------------------------------------------------------------------ world update
    def _update(self) -> None:
        m, t = self.model, self.t
        j = m.idx(t)
        # dome (the TO follows the weather rules; opens at sunset)
        sun = m.ephem.sun_alt[j]
        want_open = bool(m.dome_ok[j]) and sun < 0.0 and t < m.ephem.minute_of(m.ephem.twilight.sunrise)
        if want_open and not self.dome_open:
            self.dome_open = True
            self.dome_reason = ""
            self.say("TO", "Opening the dome." if sun > -3 else "Conditions OK for 30 min: reopening the dome.")
        elif not want_open and self.dome_open:
            self.dome_open = False
            w = m.weather
            why = (
                "humidity %.0f%%" % w.humidity[j]
                if w.humidity[j] >= m.site.limits.humidity_close
                else "wind %.0f mph" % (w.wind_ms[j] * 2.237)
                if w.wind_ms[j] >= m.site.limits.wind_close_ms
                else "sunrise"
                if sun >= 0
                else "dew point"
            )
            self.dome_reason = why
            self.say("TO", f"Closing the dome: {why}.", level="warn")
            if self.tcs.state in ("guiding", "tracking", "acquiring"):
                self.tcs.state = "tracking"
                self.tcs.busy_until = None  # an acquisition in progress is abandoned
                self.state.acquired = False
            self.guider["state"] = "off"
        # telescope
        if self.tcs.busy_until is not None and t >= self.tcs.busy_until:
            if self.tcs.state == "slewing":
                if not self.dome_open:
                    self.tcs.state = "tracking"
                    self.tcs.busy_until = None
                    self.say("TO", f"On {self.tcs.name}, tracking. Dome is closed, no guiding.")
                else:
                    self.tcs.state = "acquiring"
                    acq = self._acq_seconds()
                    self.tcs.busy_until = t + acq / 60.0
                    self.say(
                        "TO",
                        f"Arrived at {self.tcs.name}. Acquiring guide star, putting the target on the slit "
                        f"(~{acq / 60:.0f} min).",
                    )
            elif self.tcs.state == "acquiring":
                self.tcs.state = "guiding"
                self.tcs.busy_until = None
                self.guider["state"] = "guiding"
                if self.tcs.target is not None:
                    self.state.tel_target, self.state.acquired = self.tcs.target, True
                self.say("TO", f"Guiding on {self.tcs.name}. Target is on the slit, you're good to go.")
        if self.focus_busy_until is not None and t >= self.focus_busy_until:
            self.focus_busy_until = None
            self.state.focus_temp = m.temperature(t)
            self.state.last_focus_t = t
            if self.tcs.state == "guiding" and self.dome_open:
                self.guider["state"] = "guiding"  # guiding resumes after the focus run
            self.say("TO", f"Focus done at T={m.temperature(t):.1f} C. Guider FWHM should be back down.")
        # pointing and limits
        if self.tcs.state != "parked":
            p = m.ephem.point(self.tcs.ra, self.tcs.dec, m.ephem.jd_of(t))
            if self.tcs.state == "slewing" and self.tcs.busy_until:
                frac = (
                    np.clip(1 - (self.tcs.busy_until - t) / self._slew_total_min, 0, 1)
                    if self._slew_total_min > 0
                    else 1
                )
                a0, z0 = self.tcs.slew_from
                self.tcs.alt = a0 + (p["alt"] - a0) * frac
                dz = ((p["az"] - z0) + 180) % 360 - 180
                self.tcs.az = (z0 + dz * frac) % 360
            else:
                self.tcs.alt, self.tcs.az = p["alt"], p["az"]
            from obsassist.astro.sites import extra_el_limit

            lim = extra_el_limit(m.tel, self.tcs.az)
            if self.tcs.state in ("guiding", "tracking", "acquiring") and self.tcs.alt < lim:
                self.tcs.state = "limit"
                self.tcs.busy_until = None
                self.state.acquired = False
                self.guider["state"] = "off"
                self.say(
                    "TO", f"{self.tcs.name} hit the elevation limit ({lim:.0f} deg). Stopping tracking.", level="warn"
                )
        # guider readings (every ~10 s sim)
        if self.guider["state"] == "guiding" and self.tcs.target is not None:
            i = self.tcs.target
            d = m.defocus(self.state, t)
            X = float(m.geo["airmass"][i, j])
            fw = float(delivered_fwhm(m.weather.seeing[j], X, 6500.0, m.tel.iq_floor_arcsec, d, m.tel.outer_scale_m))
            self.guider.update(
                fwhm=float(fw * np.exp(self.rng.normal(0, 0.05))),
                flux_ratio=float(10 ** (-0.4 * m.weather.cloud_mag[j]) * np.exp(self.rng.normal(0, 0.03))),
                t=t,
            )
        # instrument
        for a in self.arms.values():
            self._update_arm(a)
        # ToO alerts
        for i, tg in enumerate(self.program.targets):
            if tg.appears_at and m.appears[i] <= t and not getattr(self, f"_too_{i}", False):
                setattr(self, f"_too_{i}", True)
                self.say(
                    "ALERT",
                    f"Target of opportunity: {tg.name} (P{tg.priority}) at {fmt_ra(tg.ra)} {fmt_dec(tg.dec)}. "
                    f"{tg.notes}",
                    level="alert",
                )
        # history for the weather page (1 per sim minute)
        if t - self._last_hist >= 1.0:
            self._last_hist = t
            w = m.weather
            h = self.history
            h["t"].append(round(t, 2))
            h["utc"].append(self.utc().strftime("%H:%M"))
            dimm = w.seeing_dimm[j]
            h["dimm"].append(None if not np.isfinite(dimm) else round(float(dimm), 3))
            h["guider_fwhm"].append(
                None
                if self.guider["fwhm"] is None or self.guider["state"] != "guiding"
                else round(self.guider["fwhm"], 3)
            )
            h["flux_ratio"].append(
                None
                if self.guider["flux_ratio"] is None or self.guider["state"] != "guiding"
                else round(self.guider["flux_ratio"], 3)
            )
            for k, arr, nd in (
                ("humidity", w.humidity, 1),
                ("wind", w.wind_ms, 2),
                ("gust", w.wind_gust_ms, 2),
                ("temp", w.temp_c, 2),
                ("dewpoint", w.dewpoint_c, 2),
                ("pressure", w.pressure_hpa, 1),
                ("wind_dir", w.wind_dir, 0),
            ):
                h[k].append(round(float(arr[j]), nd))
        if t - self._last_plan >= 5.0:
            self._last_plan = t
            self.refresh_plan()

    def _acq_seconds(self) -> float:
        cfg = next(iter(self.arms.values())).cfg
        if self.tcs.target is None:
            return 120.0
        tg = self.program.targets[self.tcs.target]
        acq = cfg.acq_s if tg.source.mag > 16 else min(cfg.acq_s, 150.0)
        return float(acq * np.exp(self.rng.normal(0, 0.15)))

    def _update_arm(self, a: Arm) -> None:
        t = self.t
        if a.state == "exposing" and a.t_open is not None:
            el = a.elapsed_s + (t - a.t_open) * 60.0
            if el >= a.exp_time - 1e-6 or a.stop_requested:
                a.segments.append((a.t_open, t))
                a.elapsed_s = el
                a.t_open = None
                a.state = "reading"
                a.read_total_s = a.readout_s()
                a.read_end = t + a.read_total_s / 60.0
        elif a.state in ("reading", "clearing") and a.read_end is not None and t >= a.read_end:
            if a.state == "reading":
                self._finish_exposure(a)
            a.read_end = None
            a.segments = []
            a.elapsed_s = 0.0
            a.stop_requested = False
            if a.state == "reading" and a.doing < a.loops:
                a.state = "idle"
                self._start_arm(a, continuing=True)
            else:
                a.state = "idle"
                a.doing = 0 if a.doing >= a.loops else a.doing

    # ------------------------------------------------------------------ exposures
    def _start_arm(self, a: Arm, continuing: bool = False) -> str:
        if a.state != "idle":
            return f"{a.name}: busy ({a.state})"
        if a.image_type == "object" and not a.object:
            a.object = self.tcs.name
        a.doing += 1
        a.state = "exposing"
        a.t_open = self.t
        a.segments = []
        a.elapsed_s = 0.0
        a.ut_start = self.utc().strftime("%H:%M:%S")
        a.target_at_start = self.tcs.target
        a.on_target_at_start = self.tcs.state == "guiding" and self.dome_open
        if a.image_type in ("bias",):
            a.exp_time_eff = 0.0
        tag = f"{a.prefix}{a.image_no:04d}"
        self.say("OBS", f"{a.name}: start {tag} {a.image_type} '{a.object}' {a.exp_time:.0f}s ({a.doing}/{a.loops})")
        if a.image_type == "object" and not a.on_target_at_start:
            why = "the dome is closed" if not self.dome_open else f"the telescope is {self.tcs.state}"
            self.say("SYS", f"{a.name}: warning - shutter opened while {why}.", level="warn")
        return "ok"

    def _finish_exposure(self, a: Arm) -> None:
        """Physics of the exposure that just ended, the FITS file and its quick-look."""
        m = self.model
        tag = f"{a.prefix}{a.image_no:04d}"
        a.image_no += 1
        fname = f"{tag}.fits"
        a.last_file = fname
        t0 = a.segments[0][0] if a.segments else self.t
        t1 = a.segments[-1][1] if a.segments else self.t
        t_exp = sum((s1 - s0) * 60.0 for s0, s1 in a.segments)
        i = a.target_at_start
        info: Dict[str, Any] = {
            "file": fname,
            "arm": a.name,
            "config": a.cfg.key,
            "image_type": a.image_type,
            "object": a.object,
            "exptime": round(t_exp, 2),
            "ut_start": a.ut_start,
            "t_start": t0,
            "t_end": t1,
            "target": None,
            "counted": False,
        }
        seg = None
        on_target = False
        if a.image_type == "object" and i is not None:
            # was the target on the slit for the whole exposure?
            on_target = a.on_target_at_start and self.tcs.target == i and self.tcs.state == "guiding"
            d = m.defocus(self.state, t0)
            parts = [m.segment(i, s0, s1, d) for s0, s1 in a.segments]
            S = sum(p["S"] for p in parts)
            V = sum(p["V"] for p in parts)
            w = [p["dur_s"] for p in parts]
            wsum = max(sum(w), 1e-9)
            avg = lambda k: sum(p[k] * p["dur_s"] for p in parts) / wsum
            peak_e = sum(p["peak_e"] for p in parts)  # brightest pixel, accumulated over all segments
            seg = {
                "S": S,
                "V": V,
                "npix": avg("npix"),
                "fwhm": avg("fwhm"),
                "cloud": avg("cloud"),
                "airmass": avg("airmass"),
                "sky_mag": avg("sky_mag"),
                "peak_e": peak_e,
                "peak_rate": peak_e / wsum if parts else 0.0,
            }
            tg = self.program.targets[i]
            counts_arm = a.cfg.key == tg.config
            closed_during = not all(m.dome_ok[m.idx(s0) : m.idx(s1) + 1].all() for s0, s1 in a.segments)
            if counts_arm:
                var = V + seg["npix"] * m.rn2[i]
                snr = float(m.unit[i] * S / math.sqrt(max(var, 1e-9))) if on_target and not closed_during else 0.0
                reasons = []
                if not on_target:
                    reasons.append("target not on the slit (not guiding at start or telescope moved)")
                if closed_during:
                    reasons.append("dome closed during the exposure")
                if tg.max_seeing is not None and seg["fwhm"] > tg.max_seeing:
                    reasons.append(f'FWHM {seg["fwhm"]:.2f}" > {tg.max_seeing}"')
                if tg.max_cloud is not None and seg["cloud"] > tg.max_cloud:
                    reasons.append(f"cloud {seg['cloud']:.2f} mag")
                if seg["peak_e"] > 0.8 * a.cfg.full_well:
                    reasons.append("saturated")
                w0, w1 = m.window[i]
                if t0 < w0 or t1 > w1:
                    inside = max(0.0, min(t1, w1) - max(t0, w0)) * 60.0
                    if inside < 0.5 * t_exp:
                        reasons.append("outside time window")
                qc = not reasons
                if qc:
                    self.state.snr2[i] += snr**2
                self.state.n_exp[i] += 1
                self.state.open_s[i] += t_exp
                info.update(
                    target=tg.name,
                    counted=qc,
                    snr=round(snr, 2),
                    qc_reasons=reasons,
                    snr_total=round(float(math.sqrt(self.state.snr2[i])), 2),
                    goal=tg.snr_goal,
                    unit=tg.snr_unit,
                    lam=tg.lam_ref(),
                )
            else:
                info.update(target=tg.name, counted=False, note=f"S/N is booked on the {tg.config} arm")
            info.update(fwhm=round(seg["fwhm"], 3), airmass=round(seg["airmass"], 3), cloud=round(seg["cloud"], 3))
        rec = dict(info)
        self.frames.append(rec)
        self._last_plan = -1e9  # re-plan at once: the assistant must not decide on stale progress
        verdict = ""
        if info.get("target") and "snr" in info:
            verdict = f" S/N {info['snr']:.1f} -> {info['snr_total']:.1f}/{info['goal']:.0f}" + (
                "" if info["counted"] else f"  NOT COUNTED: {'; '.join(info['qc_reasons'])}"
            )
        self.say(
            "DATA",
            f"{a.name}: wrote {fname} ({a.image_type}, {t_exp:.0f}s){verdict}",
            level="warn" if ("snr" in info and not info["counted"]) else "info",
        )
        if self.write_frames:
            fut = self.pool.submit(self._make_frame_job, a.cfg, a, dict(info), seg, on_target, i, t_exp)
            fut.rec = rec
            self.pending.append(fut)

    def _make_frame_job(
        self, cfg, a: Arm, info: dict, seg: Optional[dict], on_target: bool, i: Optional[int], t_exp: float
    ) -> dict:
        try:
            from obsassist.sim import frames as F
        except Exception as e:  # frames module not available
            return {"ql_error": f"frames module unavailable: {e}"}
        m = self.model
        kw = dict(
            cfg=cfg,
            image_type=info["image_type"],
            t_exp_s=t_exp,
            site=m.site,
            tel=m.tel,
            seed=int(self.rng.integers(0, 2**31)),
            binning=tuple(a.binning),
        )
        if info["image_type"] == "object":
            j = m.idx(info["t_start"])
            if i is not None:
                tg = self.program.targets[i]
                X = seg["airmass"] if seg else float(m.geo["airmass"][i, j])
                sky = float(
                    sky_ab(
                        m.site,
                        cfg.lam_ref,
                        max(90 - math.degrees(math.acos(min(1, 1 / max(X, 1)))), 5),
                        X,
                        m.ephem.sun_alt[j],
                        m.ephem.moon_alt[j],
                        m.ephem.moon_phase_angle[j],
                        m.geo["moon_sep"][i, j],
                    )
                )
                fw = float(
                    delivered_fwhm(
                        m.weather.seeing[j],
                        X,
                        cfg.lam_ref,
                        m.tel.iq_floor_arcsec,
                        m.defocus(self.state, info["t_start"]),
                        m.tel.outer_scale_m,
                    )
                )
                kw.update(
                    source=tg.source,
                    template=tg.template,
                    z=tg.z,
                    fwhm_arcsec=fw,
                    airmass=X,
                    cloud_mag=seg["cloud"] if seg else 0.0,
                    sky_mag_ref=sky,
                    on_target=on_target,
                    slit_offset_arcsec=0.0 if on_target else 3.0,
                )
            else:
                kw.update(on_target=False, sky_mag_ref=20.5)
        req = F.FrameRequest(**kw)
        data, hdr_extra, truth = F.make_frame(req)
        hdr = self._header(a, info, i)
        hdr.update(hdr_extra)
        path = self.data_dir / info["file"]
        F.write_fits(str(path), data, hdr)
        ql = F.quicklook(data, hdr, cfg)
        return {"path": str(path), "ql": ql}

    def _header(self, a: Arm, info: dict, i: Optional[int]) -> dict:
        m = self.model
        t0 = info["t_start"]
        p = m.ephem.point(self.tcs.ra, self.tcs.dec, m.ephem.jd_of(t0)) if self.tcs.state != "parked" else {}
        ut = self.utc(t0)
        return {
            "TELESCOP": m.tel.name,
            "INSTRUME": a.cfg.instrument,
            "SITENAME": m.site.name[:60],
            "OBJECT": info["object"][:60],
            "IMAGETYP": info["image_type"],
            "EXPTIME": info["exptime"],
            "UT-DATE": ut.strftime("%Y-%m-%d"),
            "UT-TIME": ut.strftime("%H:%M:%S"),
            "MJD": round(m.ephem.jd_of(t0) - 2400000.5, 6),
            "RA": fmt_ra(self.tcs.ra),
            "DEC": fmt_dec(self.tcs.dec),
            "EQUINOX": 2000.0,
            "AIRMASS": round(p.get("airmass", 0.0), 3),
            "HA": round(p.get("ha", 0.0) / 15, 4),
            "ST": round(p.get("lst", 0.0) / 15, 4),
            "ROTANGLE": self.tcs.pa if self.tcs.pa is not None else round(p.get("parallactic", 0.0), 2),
            "SLITSIZE": f"{a.cfg.slit_width:.2f}x{a.cfg.slit_length:.2f}",
            "BINNING": f"{a.binning[0]}x{a.binning[1]}",
            "SPEED": a.speed,
            "FILENAME": info["file"],
            "OBSERVER": "obsassist",
            "ARM": a.name,
            "CONFIG": a.cfg.key,
            "DATAPATH": str(self.data_dir)[-60:],
        }

    def _collect(self) -> None:
        done = [f for f in self.pending if f.done()]
        for f in done:
            self.pending.remove(f)
            try:
                res = f.result()
            except Exception as e:  # never kill the night over a frame
                res = {"ql_error": f"{type(e).__name__}: {e}"}
            f.rec.update(res)
            ql = res.get("ql") or {}
            flags = ql.get("flags") or []
            if flags:
                self.say("QL", f"{f.rec['file']}: " + "; ".join(flags), level="warn")
            elif ql:
                snr = ql.get("snr_ref_per_A") or ql.get("snr_ref_per_pix")
                self.say(
                    "QL",
                    f"{f.rec['file']}: trace {'found' if ql.get('trace_found') else '-'}"
                    + (f', FWHM {ql["fwhm_arcsec"]:.2f}"' if ql.get("fwhm_arcsec") else "")
                    + (f", S/N {snr:.1f} at ref" if snr else ""),
                )
            self.events.append({"type": "frame", "file": f.rec["file"]})

    # ------------------------------------------------------------------ planning
    def nowcast(self) -> Nowcast:
        fr = self.guider["flux_ratio"] if self.guider["state"] == "guiding" else None
        return Nowcast.from_history(self.model, self.t, last_flux_ratio=fr)

    def refresh_plan(self) -> None:
        m = self.model
        s = self.state.copy()
        s.t = max(self.t, self._busy_until())
        now = self.nowcast()
        cands = candidates(m, s, now)
        proj = project(m, s, now)
        self.plan_cache = {
            "t": self.t,
            "candidates": [c.to_dict() for c in cands],
            "projection": proj.to_dict(m),
            "nowcast": now.__dict__,
        }

    def _busy_until(self) -> float:
        ends = [self.t]
        for a in self.arms.values():
            if a.state == "exposing" and a.t_open is not None:
                ends.append(a.t_open + (a.exp_time - a.elapsed_s) / 60.0 + a.readout_s() / 60.0)
            elif a.state == "reading" and a.read_end:
                ends.append(a.read_end)
        return max(ends)

    # ------------------------------------------------------------------ commands
    def command(self, text: str, source: str = "observer") -> str:
        """The console command line. Returns a one-line reply."""
        if source == "assistant" and not self.allow_assistant_control:
            self.say("SYS", f"assistant command refused (control not enabled): {text}", level="warn")
            return "refused: assistant control is not enabled on the console"
        try:
            words = shlex.split(text)
        except ValueError as e:
            return f"parse error: {e}"
        if not words:
            return ""
        cmd, args = words[0].lower(), words[1:]
        who = "OBS" if source == "observer" else "AST"
        try:
            reply = self._dispatch(cmd, args, who)
        except (ValueError, KeyError, IndexError) as e:
            reply = f"error: {e}"
        if reply and not reply.startswith("ok"):
            self.events.append({"type": "reply", "text": reply})
        return reply

    def _arms_from(self, token: Optional[str]) -> List[Arm]:
        if token is None or token in ("all", "both"):
            return list(self.arms.values())
        if token not in self.arms:
            raise KeyError(f"no arm '{token}' (arms: {', '.join(self.arms)})")
        return [self.arms[token]]

    def _dispatch(self, cmd: str, args: List[str], who: str) -> str:
        m = self.model
        if cmd == "help":
            return (
                "goto <target>|goto <ra> <dec> [name] · offset <dra> <ddec> · rot <pa>|parallactic · focus · "
                "<arm> exptime|loops|object|type|bin|speed <v> · start|pause|resume|stop|abort [arm|all] · "
                "log <text> · speed <x> · pause|resume sim · ff · etc <target>"
            )
        if cmd == "goto":
            if not args:
                raise ValueError("goto needs a target")
            coords = self._parse_coords(args[0], args[1]) if len(args) >= 2 else None
            if coords is not None:
                ra, dec = coords
                name = args[2] if len(args) > 2 else "manual"
                idx = None
            else:
                name = " ".join(args)
                idx = self._find_target(name)
                if self.t < m.appears[idx]:  # a ToO is unknown until its alert arrives
                    raise KeyError(f"no unique target matching '{name}'")
                tg = self.program.targets[idx]
                ra, dec, name = tg.ra, tg.dec, tg.name
            busy = [a.name for a in self.arms.values() if a.state == "exposing"]
            if busy:
                return f"refused: {', '.join(busy)} arm is exposing - stop or abort first"
            p = m.ephem.point(ra, dec, m.ephem.jd_of(self.t))
            from obsassist.astro.sites import extra_el_limit

            if p["alt"] < extra_el_limit(m.tel, p["az"]):
                return f"TO: {name} is below the limit (el {p['alt']:.1f}, az {p['az']:.0f}) - can't go there"
            sl = m.slew_seconds(self.tcs.alt, self.tcs.az, p["alt"], p["az"])
            self.tcs.slew_from = (self.tcs.alt, self.tcs.az)
            self.tcs.target, self.tcs.name, self.tcs.ra, self.tcs.dec = idx, name, ra, dec
            self.tcs.state = "slewing"
            self.tcs.busy_until = self.t + sl / 60.0
            self._slew_total_min = sl / 60.0
            self.tcs.offset = (0.0, 0.0)
            self.state.acquired = False
            self.guider["state"] = "off"
            self.say(who, f"Request: go to {name} ({fmt_ra(ra)} {fmt_dec(dec)})")
            self.say(
                "TO",
                f"Slewing to {name}: az {p['az']:.0f}, el {p['alt']:.0f}, airmass {p['airmass']:.2f}. "
                f"About {sl / 60:.1f} min.",
            )
            return "ok"
        if cmd == "offset":
            dra, ddec = float(args[0]), float(args[1])
            self.tcs.offset = (self.tcs.offset[0] + dra, self.tcs.offset[1] + ddec)
            self.say("TO", f'Offset {dra:+.1f}" E, {ddec:+.1f}" N applied.')
            return "ok"
        if cmd == "rot":
            self.tcs.pa = None if args[0].lower().startswith("par") else float(args[0])
            self.say("TO", "Rotator to parallactic." if self.tcs.pa is None else f"Rotator PA {self.tcs.pa:.1f}.")
            return "ok"
        if cmd == "focus":
            if not self.dome_open:
                return "TO: dome is closed, can't focus on sky"
            self.focus_busy_until = self.t + m.tel.focus_run_s / 60.0
            self.guider["state"] = "off"
            self.say("TO", f"Running a focus sequence (~{m.tel.focus_run_s / 60:.0f} min).")
            return "ok"
        if cmd in self.arms or cmd in ("all",):
            arms = self._arms_from(cmd)
            key, val = args[0].lower(), args[1:]
            for a in arms:
                if key in ("exptime", "exp", "time"):
                    a.exp_time = max(0.0, float(val[0]))
                elif key == "loops":
                    a.loops = max(1, int(val[0]))
                elif key == "object":
                    a.object = " ".join(val)
                elif key in ("type", "imagetype"):
                    if val[0].lower() not in IMAGE_TYPES:
                        raise ValueError(f"image type must be one of {IMAGE_TYPES}")
                    a.image_type = val[0].lower()
                    a.lamp = {"flat": "quartz", "arc": "ThAr" if a.cfg.resolution > 15000 else "HgNeAr"}.get(
                        a.image_type, "off"
                    )
                    if a.image_type == "bias":
                        a.exp_time = 0.0
                elif key == "bin":
                    a.binning = (int(val[0]), int(val[1]))
                elif key == "speed":
                    a.speed = val[0].capitalize()
                elif key == "comment":
                    a.comment = " ".join(val)
                else:
                    raise ValueError(f"unknown arm setting {key}")
            return "ok"
        if cmd == "start":
            return "; ".join(self._start_arm(a) for a in self._arms_from(args[0] if args else None))
        if cmd == "pause":
            if args and args[0] == "sim":
                self.paused = True
                return "ok"
            for a in self._arms_from(args[0] if args else None):
                if a.state == "exposing":
                    a.segments.append((a.t_open, self.t))
                    a.elapsed_s += (self.t - a.t_open) * 60.0
                    a.t_open = None
                    a.state = "paused"
                    self.say(who, f"{a.name}: paused")
            return "ok"
        if cmd == "resume":
            if args and args[0] == "sim":
                self.paused = False
                self._last_real = time.time()
                return "ok"
            for a in self._arms_from(args[0] if args else None):
                if a.state == "paused":
                    a.t_open = self.t
                    a.state = "exposing"
                    self.say(who, f"{a.name}: resumed")
            return "ok"
        if cmd == "stop":
            for a in self._arms_from(args[0] if args else None):
                if a.state in ("exposing", "paused"):
                    # close the shutter now (a paused arm's shutter is already closed) and read out
                    if a.state == "exposing" and a.t_open is not None:
                        a.segments.append((a.t_open, self.t))
                        a.elapsed_s += (self.t - a.t_open) * 60.0
                    a.t_open = None
                    a.state = "reading"
                    a.read_total_s = a.readout_s()
                    a.read_end = self.t + a.read_total_s / 60.0
                    a.loops = a.doing
                    self.say(who, f"{a.name}: stop - reading out now")
            return "ok"
        if cmd == "abort":
            for a in self._arms_from(args[0] if args else None):
                if a.state in ("exposing", "paused", "reading"):
                    a.state = "clearing"
                    a.t_open = None
                    a.read_end = self.t + 5.0 / 60.0
                    a.loops = a.doing
                    self.say(who, f"{a.name}: ABORT - exposure discarded", level="warn")
            return "ok"
        if cmd == "log":
            self.say(who, " ".join(args), level="note")
            return "ok"
        if cmd == "speed":
            self.speed = float(args[0])
            return "ok"
        if cmd == "ff":
            self.fast_forward()
            return "ok"
        if cmd == "etc":
            i = self._find_target(" ".join(args))
            return json.dumps(self.etc(i))
        raise ValueError(f"unknown command '{cmd}' (try help)")

    @staticmethod
    def _parse_coords(a: str, b: str):
        """(ra_deg, dec_deg) when both tokens are coordinates (sexagesimal or decimal degrees,
        e.g. '12:30:00 -20:15:00' or '187.5 -20.25'), else None (then it is a target name)."""
        ok = set("0123456789:.+-hmsd")
        if not (set(a) <= ok and set(b) <= ok) or not any(c.isdigit() for c in a) or not any(c.isdigit() for c in b):
            return None
        try:
            ra, dec = parse_ra(a), parse_dec(b)
        except ValueError:
            return None
        if not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
            raise ValueError(f"coordinates out of range: RA {a}, Dec {b}")
        return ra, dec

    def _find_target(self, name: str) -> int:
        n = name.lower().strip()
        for i, t in enumerate(self.program.targets):
            if t.name.lower() == n:
                return i
        hits = [i for i, t in enumerate(self.program.targets) if n in t.name.lower()]
        if len(hits) == 1:
            return hits[0]
        raise KeyError(f"no unique target matching '{name}'")

    # ------------------------------------------------------------------ ETC for the console
    def etc(
        self, i: int, seeing: Optional[float] = None, cloud: Optional[float] = None, t_exp: Optional[float] = None
    ) -> dict:
        m = self.model
        tg = self.program.targets[i]
        cfg = tg.cfg()
        j = m.idx(self.t)
        now = self.nowcast()
        s = seeing if seeing is not None else now.seeing
        c = cloud if cloud is not None else now.cloud
        X = float(np.clip(m.geo["airmass"][i, j], 1.0, 6.0))
        r = compute_rates(
            cfg,
            m.tel,
            m.site,
            tg.source,
            seeing_500=s,
            airmass=X,
            alt=float(m.geo["alt"][i, j]),
            cloud_mag=c,
            sun_alt=m.ephem.sun_alt[j],
            moon_alt=m.ephem.moon_alt[j],
            moon_phase_angle=m.ephem.moon_phase_angle[j],
            moon_sep=m.geo["moon_sep"][i, j],
            lam=tg.lam_ref(),
            unit=tg.snr_unit,
            defocus=m.defocus(self.state, self.t),
        )
        from obsassist.etc import plan_exposures

        have = float(math.sqrt(self.state.snr2[i]))
        sat = saturation_factor(cfg, m.tel, m.site, tg.source)
        plan = plan_exposures(r, cfg, tg.snr_goal, have, t_exp or tg.t_exp, sat)
        ts = np.array([60, 120, 300, 600, 900, 1200, 1800, 2700, 3600])
        return {
            "target": tg.name,
            "seeing": round(s, 2),
            "cloud": round(c, 2),
            "airmass": round(X, 3),
            "fwhm": round(float(r.fwhm), 3),
            "slit_frac": round(float(r.slit_frac), 3),
            "sky_mag": round(float(r.sky_mag), 2),
            "regime": r.regime(),
            "signal_e_s": float(r.signal),
            "sky_e_s_pix": float(r.sky_pix),
            "t_saturate": round(float(r.t_saturate(cfg.full_well)) / sat, 1),
            "snr_have": round(have, 2),
            "goal": tg.snr_goal,
            "unit": tg.snr_unit,
            "lam": tg.lam_ref(),
            "plan": {
                "n_exp": plan.n_exp,
                "t_exp": round(plan.t_exp, 1),
                "open_s": round(plan.open_shutter_s),
                "wall_s": round(plan.wall_s),
                "snr_final": round(plan.snr_final, 2),
                "limited_by": plan.limited_by,
            },
            "curve": {"t": ts.tolist(), "snr": [round(float(r.snr(x)), 2) for x in ts]},
        }

    # ------------------------------------------------------------------ snapshots
    def snapshot(self, full: bool = False) -> dict:
        m, t = self.model, self.t
        j = m.idx(t)
        tel = self.tcs
        p = (
            m.ephem.point(tel.ra, tel.dec, m.ephem.jd_of(t))
            if tel.state != "parked"
            else {
                "alt": tel.alt,
                "az": tel.az,
                "airmass": 1.0,
                "ha": 0.0,
                "parallactic": 0.0,
                "lst": float(m.ephem.lst[j]),
            }
        )
        w = m.weather
        loc = self.model.ephem.local(t)
        tw = m.ephem.twilight
        snr = np.sqrt(self.state.snr2)
        targets = []
        for i, tg in enumerate(self.program.targets):
            targets.append(
                {
                    "i": i,
                    "name": tg.name,
                    "priority": tg.priority,
                    "kind": tg.kind,
                    "config": tg.config,
                    "ra": fmt_ra(tg.ra),
                    "dec": fmt_dec(tg.dec),
                    "mag": tg.source.mag,
                    "band": tg.source.band,
                    "snr": round(float(snr[i]), 2),
                    "goal": tg.snr_goal,
                    "unit": tg.snr_unit,
                    "n_exp": int(self.state.n_exp[i]),
                    "open_s": round(float(self.state.open_s[i])),
                    "alt": round(float(m.geo["alt"][i, j]), 1),
                    "az": round(float(m.geo["az"][i, j]), 1),
                    "airmass": round(float(m.geo["airmass"][i, j]), 3),
                    "moon_sep": round(float(m.geo["moon_sep"][i, j]), 1),
                    "visible": bool(m.is_visible(i, t)),
                    "announced": bool(t >= m.appears[i]),
                    "done": bool(snr[i] >= tg.snr_goal * 0.999),
                    "notes": tg.notes,
                    "t_exp": tg.t_exp,
                    "max_seeing": tg.max_seeing,
                    "window": tg.window,
                    "pa": tg.pa,
                    "pi": tg.pi,
                    "program": tg.program,
                }
            )
        snap = {
            "t": round(t, 4),
            "utc": self.utc().strftime("%Y-%m-%d %H:%M:%S"),
            "local": loc.strftime("%H:%M:%S"),
            "lst": fmt_ra(float(m.ephem.lst[j]))[:8],
            "speed": self.speed,
            "paused": self.paused,
            "site": m.site.name,
            "telescope": m.tel.name,
            "instrument": self.instrument,
            "program": self.program.name,
            "sun_alt": round(float(m.ephem.sun_alt[j]), 2),
            "twilight": m.ephem.twilight_phase(j),
            "moon": {
                "alt": round(float(m.ephem.moon_alt[j]), 1),
                "az": round(float(m.ephem.moon_az[j]), 1),
                "illum": round(float(m.ephem.moon_illum[j]), 3),
            },
            "night": {
                "start": m.t_start,
                "end": m.t_end,
                "twi12_dusk": self.utc(m.ephem.minute_of(tw.nautical_dusk)).strftime("%H:%M"),
                "twi12_dawn": self.utc(m.ephem.minute_of(tw.nautical_dawn)).strftime("%H:%M"),
                "minutes_left": round(max(0.0, m.t_end - t), 1),
            },
            "dome": {"open": self.dome_open, "reason": self.dome_reason},
            "weather": {
                "dimm": None if not np.isfinite(w.seeing_dimm[j]) else round(float(w.seeing_dimm[j]), 2),
                "humidity": round(float(w.humidity[j]), 1),
                "wind_ms": round(float(w.wind_ms[j]), 1),
                "wind_mph": round(float(w.wind_ms[j]) * 2.237, 1),
                "gust_mph": round(float(w.wind_gust_ms[j]) * 2.237, 1),
                "wind_dir": round(float(w.wind_dir[j])),
                "temp": round(float(w.temp_c[j]), 1),
                "dewpoint": round(float(w.dewpoint_c[j]), 1),
                "pressure": round(float(w.pressure_hpa[j]), 1),
                "limits": {
                    "humidity": m.site.limits.humidity_close,
                    "wind_mph": round(m.site.limits.wind_high_ms * 2.237, 1),
                    "wind_close_mph": round(m.site.limits.wind_close_ms * 2.237, 1),
                },
            },
            "tcs": {
                "state": tel.state,
                "name": tel.name,
                "ra": fmt_ra(tel.ra),
                "dec": fmt_dec(tel.dec),
                "alt": round(tel.alt, 2),
                "az": round(tel.az, 2),
                "airmass": round(p.get("airmass", 1.0), 3),
                "ha": round(p.get("ha", 0.0) / 15.0, 3),
                "parallactic": round(p.get("parallactic", 0.0), 1),
                "rot_pa": tel.pa,
                "offset": list(tel.offset),
                "busy_s": round(max(0.0, (tel.busy_until - t) * 60.0), 0) if tel.busy_until else 0,
            },
            "guider": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.guider.items()},
            "focus": {
                "temp_at_focus": round(self.state.focus_temp, 2),
                "dT": round(abs(m.temperature(t) - self.state.focus_temp), 2),
                "running": self.focus_busy_until is not None,
            },
            "arms": {n: a.to_dict(t) for n, a in self.arms.items()},
            "targets": targets,
            "score": round(m.score(self.state), 3),
            "max_score": m.max_score(),
            "plan": self.plan_cache,
            "log": list(self.log)[-60:],
            "frames": [
                {k: v for k, v in f.items() if k != "ql"} | {"ql_flags": (f.get("ql") or {}).get("flags")}
                for f in self.frames[-30:]
            ],
            "temps": self.temps(),
            "assistant_control": self.allow_assistant_control,
            "data_dir": str(self.data_dir),
        }
        if full:
            snap["history"] = self.history
        return snap

    CCD_SETPOINT = {"blue": -120.0, "red": -115.0, "main": -110.0}

    def temps(self) -> dict:
        """Instrument and dome temperatures: the dome follows the outside air with a lag, the
        instrument enclosure is regulated, CCDs sit at their setpoints with controller noise."""
        m, j = self.model, self.model.idx(self.t)
        w = m.weather.temp_c
        lag = w[max(0, j - 45) : j + 1].mean()
        rng = np.random.default_rng(int(self.t * 10))
        return {
            "outside": round(float(w[j]), 2),
            "dome": round(float(lag + 0.8), 2),
            "instrument": round(17.0 + 0.1 * float(np.sin(self.t / 90.0)), 2),
            "ccd": {a: round(self.CCD_SETPOINT.get(a, -110.0) + float(rng.normal(0, 0.05)), 2) for a in self.arms},
        }

    def night_curves(self, step: int = 5) -> dict:
        """Airmass/altitude curves of every target on the night grid, for the planning chart."""
        m = self.model
        idx = np.arange(0, m.ephem.n, step)
        return {
            "t": m.ephem.t_min[idx].round(2).tolist(),
            "utc": [m.ephem.utc(x).strftime("%H:%M") for x in m.ephem.t_min[idx]],
            "sun_alt": m.ephem.sun_alt[idx].round(2).tolist(),
            "moon_alt": m.ephem.moon_alt[idx].round(2).tolist(),
            "targets": [
                {
                    "name": tg.name,
                    "alt": m.geo["alt"][i, idx].round(2).tolist(),
                    "az": m.geo["az"][i, idx].round(1).tolist(),
                    "airmass": np.clip(m.geo["airmass"][i, idx], 1, 9.9).round(3).tolist(),
                    "observable": m.observable[i, idx].tolist(),
                }
                for i, tg in enumerate(self.program.targets)
            ],
            "limits": {"el_min": m.tel.el_min_deg, "az_limits": [list(x) for x in m.tel.az_limits]},
            "night": {"start": m.t_start, "end": m.t_end},
        }
