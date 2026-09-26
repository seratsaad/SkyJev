"""Night ephemeris and target visibility for the LBT, using astropy only (offline ok).

Conventions
- A "night" is labelled by the LOCAL (MST, UTC-7) calendar date of the evening,
  the same convention as the OSURC queue page and the LBTO schedule "AZ date".
- Times are returned both as local MST strings and UTC ISO strings.
"""
from __future__ import annotations

import datetime as dt
import math
import warnings
from functools import lru_cache
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")
from astropy import units as u  # noqa: E402
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun  # noqa: E402
from astropy.time import Time  # noqa: E402
from astropy.utils import iers  # noqa: E402

iers.conf.auto_download = False
iers.conf.iers_degraded_accuracy = "ignore"

from .config import SITE, load_config, now_local  # noqa: E402

LOC = EarthLocation(lat=SITE["lat_deg"] * u.deg, lon=SITE["lon_deg"] * u.deg, height=SITE["height_m"] * u.m)
TZ = dt.timezone(dt.timedelta(hours=SITE["tz_offset_hours"]), SITE["tz_name"])


def parse_date(s: str | None) -> dt.date:
    """'today'/'tonight'/None -> tonight's local evening date (if before local noon, still 'last evening's' date? No: use today's date)."""
    if not s or s.lower() in ("today", "tonight", "now"):
        now = now_local()
        # Between local midnight and 09:00 we are still in "last night"; label by the evening date.
        if now.hour < 9:
            return (now - dt.timedelta(days=1)).date()
        return now.date()
    return dt.date.fromisoformat(s)


def local_to_utc(date: dt.date, hour: float) -> dt.datetime:
    """Local MST decimal hour (may exceed 24 for after midnight) on the evening `date` -> aware UTC datetime."""
    base = dt.datetime(date.year, date.month, date.day, tzinfo=TZ)
    return (base + dt.timedelta(hours=hour)).astimezone(dt.timezone.utc)


def utc_to_local_hour(date: dt.date, t: dt.datetime) -> float:
    base = dt.datetime(date.year, date.month, date.day, tzinfo=TZ)
    return (t.astimezone(TZ) - base).total_seconds() / 3600.0


def fmt_local(date: dt.date, hour: float) -> str:
    h = hour % 24
    return f"{int(h):02d}:{int(round((h - int(h)) * 60)) % 60:02d}"


def sexagesimal_to_deg(ra: str, dec: str) -> tuple[float, float]:
    c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    return float(c.ra.deg), float(c.dec.deg)


# ----------------------------------------------------------------------------- night events

def _crossing(times: np.ndarray, alts: np.ndarray, level: float, rising: bool) -> float | None:
    """Return the local hour at which alt crosses `level` (interpolated). times = local hours."""
    for i in range(len(alts) - 1):
        a0, a1 = alts[i] - level, alts[i + 1] - level
        if rising and a0 < 0 <= a1 or (not rising and a0 >= 0 > a1):
            f = a0 / (a0 - a1)
            return float(times[i] + f * (times[i + 1] - times[i]))
    return None


@lru_cache(maxsize=64)
def night_info(date_iso: str) -> dict[str, Any]:
    """Sun/moon events for the night starting on the local evening date `date_iso`."""
    cfg = load_config()
    date = dt.date.fromisoformat(date_iso)
    hours = np.arange(12.0, 36.0, 1 / 30)   # local 12:00 -> next-day 12:00, 2-min steps
    t = Time([local_to_utc(date, h) for h in hours])
    frame = AltAz(obstime=t, location=LOC)
    sun = get_sun(t).transform_to(frame)
    moon_icrs = get_body("moon", t, LOC)
    moon = moon_icrs.transform_to(frame)
    sun_alt = sun.alt.deg
    moon_alt = moon.alt.deg
    # fractional illumination from elongation
    elong = get_sun(t).separation(moon_icrs).rad
    fli = (1 - np.cos(elong)) / 2

    ev = {
        "sunset": _crossing(hours, sun_alt, -0.833, rising=False),
        "evening_civil": _crossing(hours, sun_alt, -6, rising=False),
        "evening_12deg": _crossing(hours, sun_alt, -12, rising=False),
        "evening_18deg": _crossing(hours, sun_alt, -18, rising=False),
        "morning_18deg": _crossing(hours, sun_alt, -18, rising=True),
        "morning_12deg": _crossing(hours, sun_alt, -12, rising=True),
        "morning_civil": _crossing(hours, sun_alt, -6, rising=True),
        "sunrise": _crossing(hours, sun_alt, -0.833, rising=True),
    }
    moon_events = []
    for i in range(len(hours) - 1):
        if moon_alt[i] < 0 <= moon_alt[i + 1]:
            moon_events.append(("moonrise", float(hours[i])))
        elif moon_alt[i] >= 0 > moon_alt[i + 1]:
            moon_events.append(("moonset", float(hours[i])))

    # LST at local midnight
    t_mid = Time(local_to_utc(date, 24.0))
    lst_mid = float(t_mid.sidereal_time("apparent", longitude=LOC.lon).hour)
    i_mid = int(np.argmin(np.abs(hours - 24.0)))
    fli_mid = float(fli[i_mid])
    dark_start, dark_end = ev["evening_18deg"], ev["morning_18deg"]

    # hourly sky-brightness classification across the science window
    sky = cfg["sky"]
    per_hour = []
    if dark_start is not None and dark_end is not None:
        h = math.floor(dark_start)
        while h < dark_end:
            hh = max(h, dark_start)
            j = int(np.argmin(np.abs(hours - (hh + 0.5))))
            ma, f = float(moon_alt[j]), float(fli[j])
            cls = classify_sky(ma, f, sky)
            frac = min(h + 1, dark_end) - max(h, dark_start)
            per_hour.append({"local": fmt_local(date, h), "utc": local_to_utc(date, h).strftime("%H:%M"),
                             "moon_alt": round(ma, 1), "fli": round(f, 3), "sky": cls, "hours_in_window": round(frac, 2)})
            h += 1
    dark_h = round(sum(x["hours_in_window"] for x in per_hour if x["sky"] == "dark"), 1)
    total_h = sum(x["hours_in_window"] for x in per_hour) or 1.0
    overall = "dark" if dark_h >= 0.6 * total_h else ("gray" if any(x["sky"] != "bright" for x in per_hour) else "bright")
    if per_hour and all(x["sky"] == "bright" for x in per_hour):
        overall = "bright"

    def L(h):
        return None if h is None else fmt_local(date, h)

    def U(h):
        return None if h is None else local_to_utc(date, h).strftime("%Y-%m-%d %H:%M UTC")

    return {
        "date_local": date_iso,
        "date_utc_of_midnight": local_to_utc(date, 24.0).date().isoformat(),
        "events_local_mst": {k: L(v) for k, v in ev.items()},
        "events_utc": {k: U(v) for k, v in ev.items()},
        "events_local_hours": ev,
        "moon_events_local_mst": [(k, fmt_local(date, v)) for k, v in moon_events],
        "science_night_hours_18deg": round((dark_end - dark_start), 2) if dark_start and dark_end else None,
        "lst_at_local_midnight_h": round(lst_mid, 2),
        "moon_illumination_at_midnight": round(fli_mid, 3),
        "moon_alt_at_midnight_deg": round(float(moon_alt[i_mid]), 1),
        "sky_by_hour": per_hour,
        "sky_overall": overall,
        "dark_hours_18deg": dark_h,
    }


def classify_sky(moon_alt: float, fli: float, sky_cfg: dict) -> str:
    if moon_alt < sky_cfg["moon_down_alt_deg"] and sky_cfg["dark_if_moon_down"]:
        return "dark"
    if fli <= sky_cfg["dark_fli_max"]:
        return "dark"
    if fli <= sky_cfg["gray_fli_max"]:
        return "gray"
    return "bright"


# ----------------------------------------------------------------------------- target visibility

def target_track(date: dt.date, ra: str, dec: str, step_min: int = 10,
                 start_hour: float | None = None, end_hour: float | None = None) -> dict[str, Any]:
    """Alt/airmass/HA/moon separation/parallactic angle track between the 12-deg twilights (or given hours)."""
    ni = night_info(date.isoformat())
    evh = ni["events_local_hours"]
    h0 = start_hour if start_hour is not None else (evh["evening_12deg"] or 19.0) - 0.5
    h1 = end_hour if end_hour is not None else (evh["morning_12deg"] or 30.0) + 0.5
    hours = np.arange(h0, h1 + 1e-6, step_min / 60)
    t = Time([local_to_utc(date, h) for h in hours])
    frame = AltAz(obstime=t, location=LOC)
    c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    aa = c.transform_to(frame)
    lst = t.sidereal_time("apparent", longitude=LOC.lon).hour
    ha = ((lst - c.ra.hour + 12) % 24) - 12
    moon_aa = get_body("moon", t, LOC).transform_to(frame)
    moon_sep = aa.separation(moon_aa).deg      # same (topocentric AltAz) frame: avoids the GCRS->ICRS parallax trap
    moon_alt = moon_aa.alt.deg
    sun_aa = get_sun(t).transform_to(frame)
    sun_alt = sun_aa.alt.deg
    elong = sun_aa.separation(moon_aa).deg
    alt = aa.alt.deg
    z = np.radians(90 - alt)
    with np.errstate(divide="ignore", invalid="ignore"):
        airmass = np.where(alt > 0, 1 / (np.cos(z) + 0.50572 * (6.07995 + alt) ** -1.6364), np.nan)
    lat = math.radians(SITE["lat_deg"])
    dec_r = c.dec.rad
    ha_r = np.radians(ha * 15)
    pa = np.degrees(np.arctan2(np.sin(ha_r), np.tan(lat) * np.cos(dec_r) - np.sin(dec_r) * np.cos(ha_r)))
    from .optics import moon_sky_brightness
    rows = []
    for i, h in enumerate(hours):
        msb = moon_sky_brightness(float(moon_alt[i]), float(alt[i]), float(moon_sep[i]), float(elong[i]))
        rows.append({"local": fmt_local(date, float(h)), "hour": round(float(h), 3), "utc": local_to_utc(date, float(h)).strftime("%H:%M"),
                     "alt": round(float(alt[i]), 1), "airmass": None if np.isnan(airmass[i]) else round(float(airmass[i]), 2),
                     "ha_h": round(float(ha[i]), 2), "parang": round(float(pa[i]), 1), "sun_alt": round(float(sun_alt[i]), 1),
                     "moon_alt": round(float(moon_alt[i]), 1), "moon_sep": round(float(moon_sep[i]), 1),
                     "sky_V": msb["sky_V"], "moon_V": msb["moon_V"]})
    tr_i = int(np.argmin(np.abs(ha)))
    return {"ra": ra, "dec": dec, "ra_deg": float(c.ra.deg), "dec_deg": float(c.dec.deg),
            "transit_local": fmt_local(date, float(hours[tr_i])), "transit_hour": round(float(hours[tr_i]), 2),
            "max_alt": round(float(alt.max()), 1), "rows": rows}


def observable_windows(track: dict[str, Any], min_alt: float, max_alt: float, ha_limits: list[list[float]] | None,
                       dark_start: float, dark_end: float, min_moon_sep: float | None = None) -> list[tuple[float, float]]:
    """Contiguous local-hour windows where the target satisfies altitude, HA, twilight and moon-separation limits."""
    ok = []
    for r in track["rows"]:
        good = min_alt <= r["alt"] <= max_alt and dark_start <= r["hour"] <= dark_end
        if ha_limits:
            good = good and any(lo <= r["ha_h"] <= hi for lo, hi in ha_limits)
        if min_moon_sep and r["moon_alt"] > 0:
            good = good and r["moon_sep"] >= min_moon_sep
        ok.append(good)
    wins, start = [], None
    for r, g in zip(track["rows"], ok):
        if g and start is None:
            start = r["hour"]
        elif not g and start is not None:
            wins.append((start, r["hour"]))
            start = None
    if start is not None:
        wins.append((start, track["rows"][-1]["hour"]))
    return wins
