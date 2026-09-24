"""The night model: ephemerides + weather + every target's photon budget, precomputed on the
1-minute grid, and the rules that turn an action into elapsed time and S/N.

Everything that evaluates a schedule - the real-time console, the greedy planner, the
brute-force oracle, the dataset builder - goes through `NightModel.observe`, so they all
agree on physics, overheads and data-quality rules.

Exposure S/N under time-varying conditions: the source and variance rates are integrated
over the exposure (cumulative sums on the grid, so any exposure costs O(1)); read noise is
added once per read; data count only when they pass the target's QC rules (mean delivered
FWHM and cloud during the exposure, time window).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from obsassist.astro.ephem import NightEphem, jd_from_datetime
from obsassist.astro.sites import Site, Telescope, extra_el_limit
from obsassist.etc import adr_offset_for, compute_rates, saturation_factor
from obsassist.targets import Program, Target
from obsassist.weather import NightWeather, generate

DEFOCUS_LEVELS = np.array([0.0, 0.2, 0.4, 0.6, 0.9])  # arcsec, precomputed grid
SEEING_LEVELS = np.array([0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0])  # planning tables (DIMM, arcsec)


def _parse_utc(s: str) -> _dt.datetime:
    t = _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=_dt.timezone.utc)


@dataclass
class ExposureOutcome:
    target: int
    t_start: float  # minutes since grid start
    t_req_s: float  # requested exposure time
    t_exp_s: float  # actual open-shutter time
    snr: float  # S/N of this exposure (target unit), before QC
    snr_counted: float  # S/N that counts toward the goal (0 when QC fails)
    qc_pass: bool
    qc_reasons: List[str]
    fwhm: float  # mean delivered FWHM (arcsec, at the target's wavelength)
    cloud: float  # mean cloud extinction (mag)
    airmass: float
    sky_mag: float
    signal_e: float  # e- in the extraction aperture (per spectral pixel for spec)
    sky_e_pix: float  # sky e- per pixel
    peak_e: float  # brightest pixel, e-
    saturated: bool
    truncated: Optional[str]  # why the exposure ended early, if it did
    npix: float
    t_end: float  # minutes, including readout


@dataclass
class ObsState:
    t: float  # minutes since grid start
    tel_target: int = -1  # target the telescope is on (-1: none / zenith)
    tel_alt: float = 89.0
    tel_az: float = 180.0
    setup: str = ""
    snr2: np.ndarray = None  # accumulated counted S/N^2 per target
    n_exp: np.ndarray = None
    open_s: np.ndarray = None
    focus_temp: float = 0.0
    last_focus_t: float = -1e9
    acquired: bool = False
    history: List[tuple] = field(default_factory=list)

    def copy(self) -> "ObsState":
        return replace(
            self, snr2=self.snr2.copy(), n_exp=self.n_exp.copy(), open_s=self.open_s.copy(), history=list(self.history)
        )


class NightModel:
    """One night of one program under one weather realisation (the ground truth)."""

    def __init__(
        self,
        program: Program,
        seed: int = 0,
        weather: Optional[NightWeather] = None,
        step_min: float = 1.0,
        **weather_kw,
    ):
        self.program = program
        self.tel: Telescope = program.telescope
        self.site: Site = program.site
        self.targets: List[Target] = program.targets
        self.N = len(self.targets)
        self.ephem = NightEphem(self.site, program.date, step_min)
        self.dt = self.ephem.step_min
        self.weather = weather or generate(self.site, self.ephem.t_min, seed, **weather_kw)
        self.seed = seed
        tw = self.ephem.twilight
        start = {"sunset": tw.sunset, "6deg": tw.civil_dusk, "12deg": tw.nautical_dusk, "18deg": tw.astro_dusk}
        end = {"sunset": tw.sunrise, "6deg": tw.civil_dawn, "12deg": tw.nautical_dawn, "18deg": tw.astro_dawn}
        self.t_start = self.ephem.minute_of(start[program.night_start])
        self.t_end = self.ephem.minute_of(end[program.night_end])
        self.t_start_twi = self.ephem.minute_of(tw.civil_dusk)  # twilight_ok targets may start here
        self.t_end_twi = self.ephem.minute_of(tw.civil_dawn)
        self._precompute()

    # ------------------------------------------------------------------ setup
    def _precompute(self) -> None:
        e, w, T = self.ephem, self.weather, self.targets
        n = e.n
        self.geo = e.targets([t.ra for t in T], [t.dec for t in T])
        alt, az = self.geo["alt"], self.geo["az"]
        el_lim = extra_el_limit(self.tel, az)
        self.el_limit = el_lim
        # geometric + program constraints
        obs = (alt >= el_lim) & (alt <= self.tel.el_max_deg)
        obs &= self.geo["airmass"] <= np.array([t.max_airmass for t in T])[:, None]
        moon_up = e.moon_alt[None, :] > -0.5
        obs &= ~moon_up | (self.geo["moon_sep"] >= np.array([t.min_moon_sep for t in T])[:, None])
        tgrid = e.t_min[None, :]
        t0 = np.array([self.t_start_twi if t.twilight_ok else self.t_start for t in T])[:, None]
        t1 = np.array([self.t_end_twi if t.twilight_ok else self.t_end for t in T])[:, None]
        obs &= (tgrid >= t0) & (tgrid <= t1)
        self.observable_geo = obs
        # windows and ToO arrival, in grid minutes
        self.window = np.full((self.N, 2), [-1e9, 1e9])
        self.appears = np.full(self.N, -1e9)
        for i, t in enumerate(T):
            if t.window:
                self.window[i] = [self.minute_of_utc(t.window[0]), self.minute_of_utc(t.window[1])]
            if t.appears_at:
                self.appears[i] = self.minute_of_utc(t.appears_at)
        # photon budgets: per target, per defocus level, per grid minute (truth conditions)
        L = len(DEFOCUS_LEVELS)
        self.sig = np.zeros((self.N, L, n))  # e-/s in aperture
        self.var = np.zeros((self.N, L, n))  # e-/s variance rate (source + sky + dark in aperture)
        self.npix = np.zeros((self.N, L, n))
        self.fwhm = np.zeros((self.N, L, n))
        self.peak = np.zeros((self.N, L, n))
        self.sky_pix = np.zeros((self.N, n))
        self.sky_mag = np.zeros((self.N, n))
        self.rn2 = np.zeros(self.N)
        self.unit = np.zeros(self.N)
        S = len(SEEING_LEVELS)
        # planning tables: cloud-free, in focus, at fixed DIMM seeing levels (what a planner can
        # compute without knowing the future): source rate, background variance rate, FWHM
        self.P_sig = np.zeros((self.N, S, n))
        self.P_bkg = np.zeros((self.N, S, n))
        self.P_npix = np.zeros((self.N, S, n))
        self.P_fwhm = np.zeros((self.N, S, n))
        self.P_peak = np.zeros((self.N, S, n))  # brightest-pixel e-/s (source + sky + dark)
        from obsassist.astro.sky import sky_ab

        for i, t in enumerate(T):
            cfg = t.cfg()
            lam = t.lam_ref()
            X = np.clip(self.geo["airmass"][i], 1.0, 6.0)
            adr = adr_offset_for(cfg, self.site, lam, X, t.pa, self.geo["parallactic"][i])
            sky_i = sky_ab(
                self.site,
                lam,
                np.clip(alt[i], 1, 90),
                X,
                e.sun_alt,
                e.moon_alt,
                e.moon_phase_angle,
                self.geo["moon_sep"][i],
            )
            common = dict(
                airmass=X, alt=np.clip(alt[i], 1, 90), lam=lam, unit=t.snr_unit, adr_offset=adr, sky_mag=sky_i
            )
            sat = saturation_factor(cfg, self.tel, self.site, t.source)  # brightest pixel of any arm
            for k, sv in enumerate(SEEING_LEVELS):
                r = compute_rates(
                    cfg, self.tel, self.site, t.source, seeing_500=sv, cloud_mag=0.0, defocus=0.0, **common
                )
                self.P_sig[i, k] = r.signal
                self.P_bkg[i, k] = r.npix * (r.sky_pix + r.dark_pix)
                self.P_npix[i, k] = r.npix
                self.P_fwhm[i, k] = r.fwhm
                self.P_peak[i, k] = r.peak_pix * sat
            for k, d in enumerate(DEFOCUS_LEVELS):
                r = compute_rates(
                    cfg, self.tel, self.site, t.source, seeing_500=w.seeing, cloud_mag=w.cloud_mag, defocus=d, **common
                )
                self.sig[i, k] = r.signal
                self.var[i, k] = r.var_rate()
                self.npix[i, k] = r.npix
                self.fwhm[i, k] = r.fwhm
                self.peak[i, k] = r.peak_pix * sat
                if k == 0:
                    self.sky_pix[i] = r.sky_pix
                    self.sky_mag[i] = r.sky_mag
                    self.rn2[i] = r.rn2
                    self.unit[i] = r.unit_scale
        # cumulative sums (per second rates x 60 s per minute) for O(1) exposure integrals
        sec = self.dt * 60.0
        pad = lambda a: np.concatenate([np.zeros(a.shape[:-1] + (1,)), np.cumsum(a * sec, axis=-1)], axis=-1)
        self.C_sig = pad(self.sig)
        self.C_var = pad(self.var)
        self.C_fwhm = pad(self.fwhm)
        self.C_npix = pad(self.npix)
        self.C_cloud = pad(self.weather.cloud_mag[None, :])[0]
        self.C_sky = pad(self.sky_pix)
        self.C_air = pad(np.clip(self.geo["airmass"], 1, 10))
        self.C_skymag = pad(self.sky_mag)
        self.C_peak = pad(self.peak)  # brightest-pixel e- accumulated (saturation check)
        # dome open minutes (truth) and high-wind pointing restriction
        self.dome_ok = self.weather.dome_ok.copy()
        lim = self.site.limits
        high = self.weather.wind_ms >= lim.wind_high_ms
        daz = np.abs(((az - self.weather.wind_dir[None, :]) + 180.0) % 360.0 - 180.0)
        self.wind_block = high[None, :] & (daz < lim.wind_avoid_deg)
        self.observable = self.observable_geo & ~self.wind_block
        self.weights = np.array([t.w for t in T])
        self.goals = np.array([t.snr_goal for t in T])

    # ------------------------------------------------------------------ time helpers
    def minute_of_utc(self, s) -> float:
        t = s if isinstance(s, _dt.datetime) else _parse_utc(s)
        return self.ephem.minute_of(jd_from_datetime(t))

    def idx(self, t_min: float) -> int:
        return int(np.clip(np.floor(t_min / self.dt), 0, self.ephem.n - 1))

    def _integral(self, C, t0: float, t1: float):
        """Integral of a per-second rate array over [t0, t1] minutes from its cumulative sum
        (linear interpolation inside a grid step). C has the time axis last."""
        n = C.shape[-1] - 1

        def at(t):
            x = np.clip(t / self.dt, 0, n)
            i = int(np.floor(x))
            f = x - i
            if i >= n:
                return C[..., n]
            return C[..., i] * (1 - f) + C[..., i + 1] * f

        return at(t1) - at(t0)

    def temperature(self, t_min: float) -> float:
        return float(self.weather.temp_c[self.idx(t_min)])

    def defocus(self, state: ObsState, t_min: float) -> float:
        dT = abs(self.temperature(t_min) - state.focus_temp)
        return float(np.clip(self.tel.focus_drift_arcsec_per_c * dT, 0.0, DEFOCUS_LEVELS[-1]))

    # ------------------------------------------------------------------ availability
    def is_visible(self, i: int, t_min: float) -> bool:
        """The ToO alert has arrived and the target is inside every pointing/program limit."""
        if t_min < self.appears[i]:
            return False
        return bool(self.observable[i, self.idx(t_min)])

    def usable_until(self, i: int, t_min: float) -> float:
        """Minutes from t_min until the target leaves its observable region (or the night ends)."""
        j = self.idx(t_min)
        row = self.observable[i, j:]
        if not row[0]:
            return 0.0
        bad = np.nonzero(~row)[0]
        end = (j + bad[0]) * self.dt if len(bad) else self.ephem.t_min[-1]
        return float(max(0.0, end - t_min))

    def dome_open_at(self, t_min: float) -> bool:
        return bool(self.dome_ok[self.idx(t_min)])

    def next_dome_change(self, t_min: float) -> float:
        j = self.idx(t_min)
        cur = self.dome_ok[j]
        ch = np.nonzero(self.dome_ok[j:] != cur)[0]
        return float((j + ch[0]) * self.dt) if len(ch) else float(self.ephem.t_min[-1])

    def slew_seconds(self, alt0, az0, alt1, az1) -> float:
        daz = abs(((az1 - az0) + 180.0) % 360.0 - 180.0)
        dalt = abs(alt1 - alt0)
        return max(daz / self.tel.slew_az_dps, dalt / self.tel.slew_el_dps) + self.tel.settle_s

    # ------------------------------------------------------------------ physics of one exposure
    def exposure(
        self, i: int, t0: float, t_exp_s: float, defocus: float = 0.0, stop_at: Optional[float] = None
    ) -> ExposureOutcome:
        """Integrate one exposure of target i starting at t0 (minutes). The exposure is cut
        short when the target hits a limit, the dome must close, or the night ends."""
        t = self.targets[i]
        cfg = t.cfg()
        t1 = t0 + t_exp_s / 60.0
        truncated = None
        # earliest hard stop inside the exposure
        j0, j1 = self.idx(t0), self.idx(min(t1, self.ephem.t_min[-1]))
        stop = t1
        seg_obs = self.observable[i, j0 : j1 + 1]
        seg_dome = self.dome_ok[j0 : j1 + 1]
        bad = np.nonzero(~(seg_obs & seg_dome))[0]
        if len(bad):
            cand = (j0 + bad[0]) * self.dt
            if cand < stop:
                stop = max(cand, t0)
                truncated = "dome closed" if not seg_dome[bad[0]] else "target limit"
        night_end = self.t_end_twi if t.twilight_ok else self.t_end
        if stop > night_end:
            stop, truncated = max(night_end, t0), "end of night"
        if stop_at is not None and stop_at < stop:
            stop, truncated = max(stop_at, t0), "stopped"
        dur_s = max(0.0, (stop - t0) * 60.0)
        # interpolate between the two precomputed defocus levels
        d = float(np.clip(defocus, 0.0, DEFOCUS_LEVELS[-1]))
        k = int(np.clip(np.searchsorted(DEFOCUS_LEVELS, d) - 1, 0, len(DEFOCUS_LEVELS) - 2))
        f = (d - DEFOCUS_LEVELS[k]) / (DEFOCUS_LEVELS[k + 1] - DEFOCUS_LEVELS[k])
        mix = lambda C: (1 - f) * self._integral(C[i, k], t0, stop) + f * self._integral(C[i, k + 1], t0, stop)
        if dur_s <= 0.5:
            return ExposureOutcome(
                i,
                t0,
                t_exp_s,
                0.0,
                0.0,
                0.0,
                False,
                ["no exposure"],
                0.0,
                0.0,
                0.0,
                99.0,
                0.0,
                0.0,
                0.0,
                False,
                truncated,
                1.0,
                t0 + cfg.readout_s / 60.0,
            )
        S = mix(self.C_sig)
        V = mix(self.C_var)
        span = dur_s
        npix = mix(self.C_npix) / span
        fwhm = mix(self.C_fwhm) / span
        cloud = self._integral(self.C_cloud, t0, stop) / span
        airmass = self._integral(self.C_air[i], t0, stop) / span
        sky_mag = self._integral(self.C_skymag[i], t0, stop) / span
        sky_e = self._integral(self.C_sky[i], t0, stop)
        var = V + npix * self.rn2[i]
        snr = float(self.unit[i] * S / np.sqrt(max(var, 1e-9)))
        # brightest pixel: integrated over the exposure and interpolated in defocus like S and V
        peak = float(mix(self.C_peak))
        saturated = peak > 0.8 * cfg.full_well
        reasons = []
        if t.max_seeing is not None and fwhm > t.max_seeing:
            reasons.append(f'FWHM {fwhm:.2f}" > {t.max_seeing:.2f}"')
        if t.max_cloud is not None and cloud > t.max_cloud:
            reasons.append(f"cloud {cloud:.2f} mag > {t.max_cloud:.2f}")
        if saturated:
            reasons.append("saturated")
        w0, w1 = self.window[i]
        if t0 < w0 or stop > w1:
            inside = max(0.0, min(stop, w1) - max(t0, w0)) * 60.0
            if inside < 0.5 * dur_s:
                reasons.append("outside time window")
        if truncated in ("dome closed",):
            reasons.append("interrupted by dome closure")
        qc = not reasons
        return ExposureOutcome(
            target=i,
            t_start=t0,
            t_req_s=t_exp_s,
            t_exp_s=dur_s,
            snr=snr,
            snr_counted=snr if qc else 0.0,
            qc_pass=qc,
            qc_reasons=reasons,
            fwhm=float(fwhm),
            cloud=float(cloud),
            airmass=float(airmass),
            sky_mag=float(sky_mag),
            signal_e=float(S),
            sky_e_pix=float(sky_e),
            peak_e=peak,
            saturated=saturated,
            truncated=truncated,
            npix=float(npix),
            t_end=stop + cfg.readout_s / 60.0,
        )

    def segment(self, i: int, t0: float, t1: float, defocus: float = 0.0) -> dict:
        """Raw integrals for target i over [t0, t1] minutes with no limit or QC logic (the
        real-time console enforces limits itself): source e-, variance e- (without read
        noise), and the mean delivered FWHM, cloud, airmass, sky and aperture pixels."""
        dur_s = max(0.0, (t1 - t0) * 60.0)
        if dur_s <= 0:
            return {
                "S": 0.0,
                "V": 0.0,
                "npix": 1.0,
                "fwhm": 0.0,
                "cloud": 0.0,
                "airmass": 0.0,
                "sky_mag": 99.0,
                "sky_e_pix": 0.0,
                "dur_s": 0.0,
                "peak_e": 0.0,
                "peak_rate": 0.0,
            }
        d = float(np.clip(defocus, 0.0, DEFOCUS_LEVELS[-1]))
        k = int(np.clip(np.searchsorted(DEFOCUS_LEVELS, d) - 1, 0, len(DEFOCUS_LEVELS) - 2))
        f = (d - DEFOCUS_LEVELS[k]) / (DEFOCUS_LEVELS[k + 1] - DEFOCUS_LEVELS[k])
        mix = lambda C: (1 - f) * self._integral(C[i, k], t0, t1) + f * self._integral(C[i, k + 1], t0, t1)
        peak_e = float(mix(self.C_peak))
        return {
            "S": float(mix(self.C_sig)),
            "V": float(mix(self.C_var)),
            "npix": float(mix(self.C_npix) / dur_s),
            "fwhm": float(mix(self.C_fwhm) / dur_s),
            "cloud": float(self._integral(self.C_cloud, t0, t1) / dur_s),
            "airmass": float(self._integral(self.C_air[i], t0, t1) / dur_s),
            "sky_mag": float(self._integral(self.C_skymag[i], t0, t1) / dur_s),
            "sky_e_pix": float(self._integral(self.C_sky[i], t0, t1)),
            "dur_s": dur_s,
            "peak_e": peak_e,
            "peak_rate": peak_e / dur_s,
        }

    # ------------------------------------------------------------------ state machine
    def initial_state(self) -> ObsState:
        return ObsState(
            t=self.t_start_twi if any(t.twilight_ok for t in self.targets) else self.t_start,
            snr2=np.zeros(self.N),
            n_exp=np.zeros(self.N, dtype=int),
            open_s=np.zeros(self.N),
            focus_temp=self.temperature(self.t_start),
            last_focus_t=self.t_start,
        )

    def overhead_to(self, state: ObsState, i: int) -> Tuple[float, Dict[str, float]]:
        """Seconds between 'go to target i' and the shutter opening (slew, acquisition, setup)."""
        t = self.targets[i]
        cfg = t.cfg()
        if state.tel_target == i and state.acquired:
            return 0.0, {}
        j = self.idx(state.t)
        alt1, az1 = float(self.geo["alt"][i, j]), float(self.geo["az"][i, j])
        slew = self.slew_seconds(state.tel_alt, state.tel_az, alt1, az1)
        setup = cfg.config_change_s if (state.setup and state.setup != cfg.setup_id) else 0.0
        acq = cfg.acq_s if t.source.mag > 16 else min(cfg.acq_s, 150.0)
        parts = {"slew": slew, "acquisition": acq, "setup": setup}
        return slew + acq + setup, parts

    def observe(
        self, state: ObsState, i: int, t_exp_s: Optional[float] = None, apply: bool = True
    ) -> Tuple[ObsState, ExposureOutcome]:
        """Go to target i if needed, take one exposure, read out. Returns the new state."""
        t = self.targets[i]
        cfg = t.cfg()
        t_exp_s = float(t_exp_s or t.t_exp or cfg.max_exp_s)
        s = state.copy() if apply else state
        oh, _ = self.overhead_to(s, i)
        t0 = s.t + oh / 60.0
        out = self.exposure(i, t0, t_exp_s, self.defocus(s, t0))
        # the telescope tracked the target to the end of the exposure: the next slew starts there
        j = self.idx(t0 + out.t_exp_s / 60.0)
        s.tel_target, s.acquired = i, True
        s.tel_alt, s.tel_az = float(self.geo["alt"][i, j]), float(self.geo["az"][i, j])
        s.setup = cfg.setup_id
        s.snr2[i] += out.snr_counted**2
        s.n_exp[i] += 1
        s.open_s[i] += out.t_exp_s
        s.t = out.t_end
        if out.truncated == "dome closed":
            s.acquired = False
            s.t = max(s.t, self.next_dome_change(out.t_end))
        s.history.append(("observe", i, round(t0, 3), round(out.t_exp_s, 1), round(out.snr_counted, 3)))
        return s, out

    def wait(self, state: ObsState, minutes: float) -> ObsState:
        s = state.copy()
        s.t = s.t + minutes
        s.history.append(("wait", round(state.t, 3), minutes))
        return s

    def focus(self, state: ObsState) -> ObsState:
        s = state.copy()
        s.t = s.t + self.tel.focus_run_s / 60.0
        s.focus_temp = self.temperature(s.t)
        s.last_focus_t = s.t
        s.history.append(("focus", round(state.t, 3)))
        return s

    def close_wait(self, state: ObsState) -> ObsState:
        """Dome closed: wait until weather allows reopening (or the night ends)."""
        s = state.copy()
        s.t = min(self.next_dome_change(s.t), self.t_end)
        s.acquired = False
        s.history.append(("closed", round(state.t, 3), round(s.t, 3)))
        return s

    # ------------------------------------------------------------------ scoring
    def snr(self, state: ObsState) -> np.ndarray:
        return np.sqrt(state.snr2)

    def done(self, state: ObsState) -> np.ndarray:
        return np.sqrt(state.snr2) >= self.goals * 0.999

    def credit(self, snr2: np.ndarray) -> np.ndarray:
        frac = np.sqrt(snr2) / self.goals
        pc = self.program.partial_credit
        return np.where(frac >= 0.999, 1.0, pc * np.clip(frac, 0, 1) ** 2)

    def score(self, state: ObsState) -> float:
        return float(np.sum(self.weights * self.credit(state.snr2)))

    def max_score(self) -> float:
        return float(self.weights.sum())

    def finished(self, state: ObsState) -> bool:
        return state.t >= self.t_end - 1.0 or bool(self.done(state).all())

    # ------------------------------------------------------------------ what the observer sees
    def telemetry(
        self, t_min: float, state: Optional[ObsState] = None, rng: Optional[np.random.Generator] = None
    ) -> dict:
        """Observed quantities only (DIMM, guider, weather station), with instrument noise."""
        rng = rng or np.random.default_rng(int(t_min * 1000) % (2**32))
        j = self.idx(t_min)
        w = self.weather
        dimm = w.seeing_dimm[j]
        out = {
            "t_min": t_min,
            "utc": self.ephem.utc(t_min).isoformat(timespec="seconds"),
            "sun_alt": float(self.ephem.sun_alt[j]),
            "twilight": self.ephem.twilight_phase(j),
            "moon_alt": float(self.ephem.moon_alt[j]),
            "moon_illum": float(self.ephem.moon_illum[j]),
            "lst_deg": float(self.ephem.lst[j]),
            "dimm_seeing": None if not np.isfinite(dimm) else float(dimm),
            "humidity": float(w.humidity[j]),
            "wind_ms": float(w.wind_ms[j]),
            "wind_gust_ms": float(w.wind_gust_ms[j]),
            "wind_dir": float(w.wind_dir[j]),
            "temp_c": float(w.temp_c[j]),
            "dewpoint_c": float(w.dewpoint_c[j]),
            "pressure_hpa": float(w.pressure_hpa[j]),
            "dome_ok": bool(self.dome_ok[j]),
            "high_wind": bool(w.wind_ms[j] >= self.site.limits.wind_high_ms),
        }
        if state is not None and state.tel_target >= 0 and self.dome_ok[j]:
            i = state.tel_target
            d = self.defocus(state, t_min)
            k = int(np.clip(np.searchsorted(DEFOCUS_LEVELS, d) - 1, 0, len(DEFOCUS_LEVELS) - 2))
            # guider FWHM (at ~ 6500 A) and flux relative to the photometric reference
            X = float(self.geo["airmass"][i, j])
            from obsassist.astro.sky import delivered_fwhm

            g_fwhm = float(delivered_fwhm(w.seeing[j], X, 6500.0, self.tel.iq_floor_arcsec, d, self.tel.outer_scale_m))
            out["guider_fwhm"] = float(g_fwhm * np.exp(rng.normal(0, 0.05)))
            out["guider_flux_ratio"] = float(10 ** (-0.4 * w.cloud_mag[j]) * np.exp(rng.normal(0, 0.03)))
            out["defocus_hint_c"] = abs(self.temperature(t_min) - state.focus_temp)
            out["fwhm_science"] = float(self.fwhm[i, k, j])
        return out
