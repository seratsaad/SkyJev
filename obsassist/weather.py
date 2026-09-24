"""Stochastic weather for one night, on the ephemeris grid (1 minute).

Truth channels (what the atmosphere does) and observed channels (what the DIMM, the guider
and the weather station report, with their noise and cadence) are kept separate: the
assistant only ever sees the observed ones.

Seeing: a night median drawn from the site's log-normal distribution; within the night an
Ornstein-Uhlenbeck process in log-seeing with the site's correlation time, plus occasional
jumps (a front, a change of wind), plus DIMM measurement noise.
Transparency: a three-state Markov chain (photometric / thin cirrus / thick cloud) whose
grey extinction wanders within each state.
Humidity, wind, temperature, dew point, pressure: mean-reverting processes, with the
occasional fog/humidity event that forces a dome closure (the site's limits).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from obsassist.astro.sites import Site

SKY_STATES = ("photometric", "thin cirrus", "thick cloud")


@dataclass
class NightWeather:
    t_min: np.ndarray  # minutes since grid start
    seeing: np.ndarray  # true DIMM seeing, 500 nm, zenith (arcsec)
    seeing_dimm: np.ndarray  # DIMM report (noisy, NaN when the DIMM is off)
    cloud_mag: np.ndarray  # grey extinction from cloud (mag)
    sky_state: np.ndarray  # index into SKY_STATES
    humidity: np.ndarray  # % RH
    wind_ms: np.ndarray
    wind_gust_ms: np.ndarray
    wind_dir: np.ndarray  # deg
    temp_c: np.ndarray
    dewpoint_c: np.ndarray
    pressure_hpa: np.ndarray
    dome_ok: np.ndarray  # weather permits the dome open (limits + reopen wait)
    seed: int
    regime: str  # "good" | "median" | "poor" (night seeing class)
    events: List[Dict] = field(default_factory=list)

    def at(self, i: int) -> dict:
        i = int(np.clip(i, 0, len(self.t_min) - 1))
        return {
            "seeing": float(self.seeing[i]),
            "seeing_dimm": float(self.seeing_dimm[i]),
            "cloud_mag": float(self.cloud_mag[i]),
            "sky_state": SKY_STATES[int(self.sky_state[i])],
            "humidity": float(self.humidity[i]),
            "wind_ms": float(self.wind_ms[i]),
            "wind_gust_ms": float(self.wind_gust_ms[i]),
            "wind_dir": float(self.wind_dir[i]),
            "temp_c": float(self.temp_c[i]),
            "dewpoint_c": float(self.dewpoint_c[i]),
            "pressure_hpa": float(self.pressure_hpa[i]),
            "dome_ok": bool(self.dome_ok[i]),
        }

    def closed_intervals(self) -> List[tuple]:
        """[(start_min, end_min)] where weather forces the dome closed."""
        out, start = [], None
        for i, ok in enumerate(self.dome_ok):
            if not ok and start is None:
                start = self.t_min[i]
            elif ok and start is not None:
                out.append((float(start), float(self.t_min[i])))
                start = None
        if start is not None:
            out.append((float(start), float(self.t_min[-1])))
        return out


def _ou(rng, n, dt, tau, sigma, x0=0.0):
    """Ornstein-Uhlenbeck path with stationary std `sigma` and correlation time `tau`."""
    a = np.exp(-dt / tau)
    b = sigma * np.sqrt(1 - a * a)
    x = np.empty(n)
    x[0] = x0
    eps = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + b * eps[i]
    return x


def generate(
    site: Site,
    t_min: np.ndarray,
    seed: int = 0,
    *,
    regime: Optional[str] = None,
    seeing_median: Optional[float] = None,
    clear: Optional[bool] = None,
    fog_event: Optional[bool] = None,
    cloud_scale: float = 1.0,
) -> NightWeather:
    """One night of weather on the grid t_min. Arguments override the random draws
    (regime "good"/"median"/"poor", a fixed night median seeing, a clear night, a fog event)."""
    rng = np.random.default_rng(seed)
    n = len(t_min)
    dt = float(t_min[1] - t_min[0]) if n > 1 else 1.0
    events: List[Dict] = []

    # ---- seeing ------------------------------------------------------------
    if seeing_median is None:
        z = rng.standard_normal()
        if regime == "good":
            z = -abs(z) - 0.4
        elif regime == "poor":
            z = abs(z) + 0.4
        elif regime == "median":
            z = 0.3 * z
        seeing_median = site.seeing_median * np.exp(site.seeing_sigma_ln * z)
    regime = regime or (
        "good"
        if seeing_median < 0.85 * site.seeing_median
        else "poor"
        if seeing_median > 1.2 * site.seeing_median
        else "median"
    )
    log_s = np.log(seeing_median) + _ou(
        rng, n, dt, site.seeing_tau_min, site.seeing_intranight_ln, x0=rng.normal(0, site.seeing_intranight_ln)
    )
    # fast component (a few minutes) on top of the slow wander
    log_s += _ou(rng, n, dt, 4.0, 0.08)
    # jumps: 0-2 per night, persistent shifts of +-30-60 %
    for _ in range(rng.poisson(0.9)):
        j = rng.integers(n // 8, n)
        amp = rng.choice([-1, 1]) * rng.uniform(0.25, 0.5)
        ramp = np.clip((np.arange(n) - j) / rng.uniform(5, 25), 0, 1)
        log_s += amp * ramp
        events.append({"t_min": float(t_min[j]), "kind": "seeing_jump", "factor": float(np.exp(amp))})
    seeing = np.clip(np.exp(log_s), 0.25, 3.5)
    seeing_dimm = seeing * np.exp(rng.normal(0, 0.07, n))
    # the DIMM samples ~ every minute and occasionally drops out (dome of the DIMM, target change)
    drop = rng.random(n) < 0.03
    seeing_dimm[drop] = np.nan

    # ---- transparency ----------------------------------------------------------
    if clear is None:
        clear = rng.random() < site.clear_fraction
    # transition rates per minute between states (mean dwell ~ hours)
    if clear:
        P = np.array(
            [[1 - 1 / 900, 1 / 900, 0.0], [1 / 60, 1 - 1 / 60 - 1 / 2000, 1 / 2000], [0.0, 1 / 30, 1 - 1 / 30]]
        )
        state0 = 0
    else:
        P = np.array(
            [
                [1 - 1 / 150, 1 / 180, 1 / 900],
                [1 / 120, 1 - 1 / 120 - 1 / 200, 1 / 200],
                [1 / 400, 1 / 90, 1 - 1 / 400 - 1 / 90],
            ]
        )
        state0 = int(rng.choice([0, 1, 2], p=[0.3, 0.45, 0.25]))
    P = P / P.sum(axis=1, keepdims=True)
    sky_state = np.empty(n, dtype=int)
    sky_state[0] = state0
    u = rng.random(n)
    cum = np.cumsum(P, axis=1)
    for i in range(1, n):
        sky_state[i] = int(np.searchsorted(cum[sky_state[i - 1]], u[i]))
    base = np.array([0.0, 0.25, 1.4])[sky_state]
    wander = np.abs(_ou(rng, n, dt, 8.0, 1.0))
    cloud = base + np.where(sky_state == 0, 0.01 * wander, np.where(sky_state == 1, 0.15 * wander, 0.7 * wander))
    cloud = np.clip(cloud * cloud_scale, 0.0, 5.0)
    if (sky_state > 0).any():
        first = int(np.argmax(sky_state > 0))
        events.append({"t_min": float(t_min[first]), "kind": "clouds", "state": SKY_STATES[int(sky_state[first])]})

    # ---- humidity / wind / temperature ------------------------------------------
    hum_mean = rng.uniform(10, 45) if clear else rng.uniform(35, 70)
    humidity = hum_mean + _ou(rng, n, dt, 60.0, 8.0)
    if fog_event is None:
        fog_event = rng.random() < (0.08 if clear else 0.3)
    if fog_event:
        j = int(rng.integers(n // 6, 5 * n // 6))
        dur = rng.uniform(25, 120)
        shape = np.exp(-0.5 * ((np.arange(n) - j) / (dur / 2.355)) ** 2)
        humidity = humidity + shape * (site.limits.humidity_close + 8 - hum_mean)
        events.append({"t_min": float(t_min[j]), "kind": "humidity_event", "duration_min": float(dur)})
    humidity = np.clip(humidity, 2, 100)
    wind_mean = rng.gamma(4.0, 1.6)
    wind = np.clip(wind_mean + _ou(rng, n, dt, 45.0, 2.2), 0, None)
    gust = wind * (1.15 + 0.1 * np.abs(rng.standard_normal(n)))
    wind_dir = np.mod(rng.uniform(0, 360) + np.cumsum(rng.normal(0, 1.2, n)), 360)
    temp = (
        site.temp_c
        + rng.normal(0, 2.0)
        - 2.5 * (t_min - t_min[0]) / max(t_min[-1] - t_min[0], 1)
        + _ou(rng, n, dt, 30.0, 0.35)
    )
    # dew point from RH (Magnus)
    a, b = 17.62, 243.12
    gamma = np.log(np.clip(humidity, 1, 100) / 100.0) + a * temp / (b + temp)
    dewpoint = b * gamma / (a - gamma)
    pressure = site.pressure_hpa + rng.normal(0, 1.5) + _ou(rng, n, dt, 180.0, 0.6)

    # ---- dome logic --------------------------------------------------------------
    L = site.limits
    bad = (humidity >= L.humidity_close) | (wind >= L.wind_close_ms) | ((temp - dewpoint) < L.dewpoint_margin_c)
    good_to_open = (
        (humidity < L.humidity_open) & (wind < 0.9 * L.wind_close_ms) & ((temp - dewpoint) >= L.dewpoint_margin_c + 1)
    )
    dome_ok = np.ones(n, dtype=bool)
    closed = False
    good_run = 0.0
    for i in range(n):
        if not closed and bad[i]:
            closed = True
            good_run = 0.0
            events.append(
                {
                    "t_min": float(t_min[i]),
                    "kind": "dome_close",
                    "reason": "humidity"
                    if humidity[i] >= L.humidity_close
                    else "wind"
                    if wind[i] >= L.wind_close_ms
                    else "dew point",
                }
            )
        elif closed:
            good_run = good_run + dt if good_to_open[i] else 0.0
            if good_run >= L.reopen_wait_min:
                closed = False
                events.append({"t_min": float(t_min[i]), "kind": "dome_reopen_allowed"})
        dome_ok[i] = not closed
    return NightWeather(
        t_min=t_min,
        seeing=seeing,
        seeing_dimm=seeing_dimm,
        cloud_mag=cloud,
        sky_state=sky_state,
        humidity=humidity,
        wind_ms=wind,
        wind_gust_ms=gust,
        wind_dir=wind_dir,
        temp_c=temp,
        dewpoint_c=dewpoint,
        pressure_hpa=pressure,
        dome_ok=dome_ok,
        seed=seed,
        regime=regime,
        events=sorted(events, key=lambda e: e["t_min"]),
    )
