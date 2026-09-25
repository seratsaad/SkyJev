"""Night ephemerides on a fixed time grid.

Sun and Moon come from astropy (built-in ephemeris, topocentric) on a coarse grid and are
interpolated to the working grid. Targets are precessed to the mean equinox of the night once
and then evaluated with closed-form spherical astronomy, vectorised over (target, time): this
is what makes the schedule search fast. `tests/test_ephem.py` checks the fast path against a
full astropy AltAz transform (agreement better than 0.02 deg above 15 deg elevation).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from obsassist.astro.sites import Site

DEG = np.pi / 180.0
JD_UNIX_EPOCH = 2440587.5


def jd_from_datetime(t: _dt.datetime) -> float:
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return JD_UNIX_EPOCH + t.timestamp() / 86400.0


def datetime_from_jd(jd: float) -> _dt.datetime:
    return _dt.datetime.fromtimestamp((jd - JD_UNIX_EPOCH) * 86400.0, tz=_dt.timezone.utc)


def gmst_deg(jd):
    """Greenwich mean sidereal time (IAU 1982), degrees. UT1 ~ UTC is fine at this level."""
    d = np.asarray(jd, dtype=float) - 2451545.0
    T = d / 36525.0
    g = 280.46061837 + 360.98564736629 * d + 0.000387933 * T**2 - T**3 / 38710000.0
    return np.mod(g, 360.0)


def lst_deg(jd, lon_east_deg: float):
    return np.mod(gmst_deg(jd) + lon_east_deg, 360.0)


def altaz(ha_deg, dec_deg, lat_deg):
    """True (geometric) altitude and azimuth (N=0, E=90) from hour angle and declination."""
    ha = np.asarray(ha_deg) * DEG
    dec = np.asarray(dec_deg) * DEG
    lat = lat_deg * DEG
    sin_alt = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    alt = np.arcsin(np.clip(sin_alt, -1, 1))
    az = np.arctan2(-np.cos(dec) * np.sin(ha), np.sin(dec) * np.cos(lat) - np.cos(dec) * np.cos(ha) * np.sin(lat))
    return alt / DEG, np.mod(az / DEG, 360.0)


def refraction_deg(alt_true_deg, pressure_hpa: float = 1010.0, temp_c: float = 10.0):
    """Saemundsson (1986) refraction for a true altitude, scaled by P and T. Degrees."""
    h = np.asarray(alt_true_deg, dtype=float)
    hh = np.maximum(h, -1.0)
    r_arcmin = 1.02 / np.tan((hh + 10.3 / (hh + 5.11)) * DEG)
    r_arcmin *= (pressure_hpa / 1010.0) * (283.0 / (273.0 + temp_c))
    return np.where(h > -1.0, r_arcmin / 60.0, 0.0)


def airmass_from_alt(alt_app_deg):
    """Kasten & Young (1989) relative air mass from apparent altitude; large below the horizon."""
    h = np.asarray(alt_app_deg, dtype=float)
    hh = np.maximum(h, 0.5)
    x = 1.0 / (np.sin(hh * DEG) + 0.50572 * (hh + 6.07995) ** -1.6364)
    return np.where(h > 0.5, x, 40.0)


def parallactic_angle(ha_deg, dec_deg, lat_deg):
    ha = np.asarray(ha_deg) * DEG
    dec = np.asarray(dec_deg) * DEG
    lat = lat_deg * DEG
    q = np.arctan2(np.sin(ha), np.tan(lat) * np.cos(dec) - np.sin(dec) * np.cos(ha))
    return q / DEG


def angular_sep_altaz(alt1, az1, alt2, az2):
    """Great-circle separation (deg) between two directions given in alt/az."""
    a1, z1, a2, z2 = (np.asarray(x) * DEG for x in (alt1, az1, alt2, az2))
    c = np.sin(a1) * np.sin(a2) + np.cos(a1) * np.cos(a2) * np.cos(z1 - z2)
    return np.arccos(np.clip(c, -1, 1)) / DEG


def parse_ra(ra) -> float:
    """RA in degrees from degrees or 'hh:mm:ss.s'."""
    if isinstance(ra, (int, float, np.floating)):
        return float(ra)
    s = str(ra).strip().replace("h", ":").replace("m", ":").replace("s", "").replace(" ", ":")
    parts = [p for p in s.split(":") if p]
    if len(parts) == 1:
        return float(parts[0])
    h, m, sec = (float(parts[0]), float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0)
    return 15.0 * (h + m / 60.0 + sec / 3600.0)


def parse_dec(dec) -> float:
    if isinstance(dec, (int, float, np.floating)):
        return float(dec)
    s = str(dec).strip().replace("d", ":").replace("m", ":").replace("s", "").replace(" ", ":")
    parts = [p for p in s.split(":") if p]
    if len(parts) == 1:
        return float(parts[0])
    sign = -1.0 if parts[0].strip().startswith("-") else 1.0
    d, m, sec = (abs(float(parts[0])), float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0)
    return sign * (d + m / 60.0 + sec / 3600.0)


def fmt_ra(ra_deg: float) -> str:
    """'hh:mm:ss.ss'. Rounded to 0.01 s *before* splitting, so the seconds never read 60.00."""
    cs = int(round(((float(ra_deg) / 15.0) % 24.0) * 360000.0)) % (24 * 360000)
    hh, rem = divmod(cs, 360000)
    mm, rem = divmod(rem, 6000)
    return f"{hh:02d}:{mm:02d}:{rem / 100.0:05.2f}"


def fmt_dec(dec_deg: float) -> str:
    """'+dd:mm:ss.s'. Rounded to 0.1 arcsec before splitting (no '60.0'), no '-00:00:00.0'."""
    ds = int(round(abs(float(dec_deg)) * 36000.0))
    sign = "-" if dec_deg < 0 and ds > 0 else "+"
    dd, rem = divmod(ds, 36000)
    mm, rem = divmod(rem, 600)
    return f"{sign}{dd:02d}:{mm:02d}:{rem / 10.0:04.1f}"


def precess_to_date(ra_deg: Sequence[float], dec_deg: Sequence[float], jd: float):
    """ICRS/J2000 -> mean equinox of date (precession only; nutation ~17'' is ignored)."""
    ra = np.atleast_1d(np.asarray(ra_deg, dtype=float))
    dec = np.atleast_1d(np.asarray(dec_deg, dtype=float))
    try:
        import astropy.units as u
        from astropy.coordinates import FK5, SkyCoord
        from astropy.time import Time

        _offline_astropy()
        c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        out = c.transform_to(FK5(equinox=Time(jd, format="jd")))
        return out.ra.deg, out.dec.deg
    except Exception:  # astropy missing: IAU 1976 precession, adequate to ~1''
        T = (jd - 2451545.0) / 36525.0
        zeta = (2306.2181 * T + 0.30188 * T**2) / 3600.0 * DEG
        z = (2306.2181 * T + 1.09468 * T**2) / 3600.0 * DEG
        theta = (2004.3109 * T - 0.42665 * T**2) / 3600.0 * DEG
        r, d = ra * DEG, dec * DEG
        A = np.cos(d) * np.sin(r + zeta)
        B = np.cos(theta) * np.cos(d) * np.cos(r + zeta) - np.sin(theta) * np.sin(d)
        C = np.sin(theta) * np.cos(d) * np.cos(r + zeta) + np.cos(theta) * np.sin(d)
        return np.mod((np.arctan2(A, B) + z) / DEG, 360.0), np.arcsin(np.clip(C, -1, 1)) / DEG


def _offline_astropy():
    """Never let astropy reach for the network (IERS tables): observatories are often offline,
    and UT1-UTC below a second is irrelevant here."""
    try:
        from astropy.utils import iers

        iers.conf.auto_download = False
        iers.conf.auto_max_age = None
        iers.conf.iers_degraded_accuracy = "ignore"
    except Exception:
        pass


def _body_altaz_grid(site: Site, jd_grid: np.ndarray):
    """Sun and Moon topocentric alt/az (no refraction) plus the Moon's illuminated fraction."""
    import warnings

    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_body, get_sun
    from astropy.time import Time

    _offline_astropy()
    warnings.filterwarnings("ignore", module="astropy")

    loc = EarthLocation(lat=site.lat_deg * u.deg, lon=site.lon_deg * u.deg, height=site.elev_m * u.m)
    t = Time(jd_grid, format="jd")
    frame = AltAz(obstime=t, location=loc)
    sun = get_sun(t)
    moon = get_body("moon", t, loc)
    sun_aa = sun.transform_to(frame)
    moon_aa = moon.transform_to(frame)
    elong = angular_sep_altaz(sun_aa.alt.deg, sun_aa.az.deg, moon_aa.alt.deg, moon_aa.az.deg)
    dist_moon = moon.distance.to(u.km).value
    dist_sun = 1.496e8
    phase_angle = np.arctan2(dist_sun * np.sin(elong * DEG), dist_moon - dist_sun * np.cos(elong * DEG)) / DEG
    illum = (1.0 + np.cos(phase_angle * DEG)) / 2.0
    return (
        sun_aa.alt.deg,
        sun_aa.az.deg,
        moon_aa.alt.deg,
        moon_aa.az.deg,
        np.abs(phase_angle),
        illum,
        moon.ra.deg,
        moon.dec.deg,
    )


def _interp_az(x_new, x, az):
    un = np.unwrap(az * DEG) / DEG
    return np.mod(np.interp(x_new, x, un), 360.0)


@dataclass
class Twilight:
    sunset: float
    civil_dusk: float  # -6
    nautical_dusk: float  # -12
    astro_dusk: float  # -18
    astro_dawn: float
    nautical_dawn: float
    civil_dawn: float
    sunrise: float  # all JD


class NightEphem:
    """Ephemerides for one night at one site, on a uniform grid of `step_min` minutes that
    spans sunset - 20 min to sunrise + 20 min. `date` is the local calendar date of the
    evening (YYYY-MM-DD)."""

    def __init__(self, site: Site, date: str, step_min: float = 1.0):
        self.site = site
        self.date = date
        self.step_min = float(step_min)
        y, m, d = (int(x) for x in date.split("-"))
        # local noon -> next local noon, as UTC
        noon_utc = _dt.datetime(y, m, d, 12, 0, tzinfo=_dt.timezone.utc) - _dt.timedelta(hours=site.utc_offset_h)
        jd0 = jd_from_datetime(noon_utc)
        coarse = jd0 + np.arange(0, 24 * 60 + 1, 10.0) / 1440.0
        s_alt, s_az, m_alt, m_az, phase, illum, m_ra, m_dec = _body_altaz_grid(site, coarse)
        # twilight crossings on the coarse grid, refined by linear interpolation
        self.twilight = self._twilight(coarse, s_alt)
        start = self.twilight.sunset - 20.0 / 1440.0
        end = self.twilight.sunrise + 20.0 / 1440.0
        n = int(np.ceil((end - start) * 1440.0 / self.step_min)) + 1
        self.jd = start + np.arange(n) * self.step_min / 1440.0
        self.n = n
        self.t_min = np.arange(n) * self.step_min  # minutes since grid start
        self.sun_alt = np.interp(self.jd, coarse, s_alt)
        self.sun_az = _interp_az(self.jd, coarse, s_az)
        self.moon_alt = np.interp(self.jd, coarse, m_alt)
        self.moon_az = _interp_az(self.jd, coarse, m_az)
        self.moon_phase_angle = np.interp(self.jd, coarse, phase)
        self.moon_illum = np.interp(self.jd, coarse, illum)
        self.moon_ra = _interp_az(self.jd, coarse, m_ra)
        self.moon_dec = np.interp(self.jd, coarse, m_dec)
        self.lst = lst_deg(self.jd, site.lon_deg)
        self._target_cache: Dict[tuple, dict] = {}

    # ---- time helpers ---------------------------------------------------
    @staticmethod
    def _twilight(jd, sun_alt) -> Twilight:
        def cross(level, rising):
            s = sun_alt - level
            idx = np.where((s[:-1] > 0) & (s[1:] <= 0))[0] if not rising else np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
            if len(idx) == 0:
                return float("nan")
            i = idx[0]
            f = s[i] / (s[i] - s[i + 1])
            return float(jd[i] + f * (jd[i + 1] - jd[i]))

        return Twilight(
            sunset=cross(-0.833, False),
            civil_dusk=cross(-6, False),
            nautical_dusk=cross(-12, False),
            astro_dusk=cross(-18, False),
            astro_dawn=cross(-18, True),
            nautical_dawn=cross(-12, True),
            civil_dawn=cross(-6, True),
            sunrise=cross(-0.833, True),
        )

    def index_of(self, jd: float) -> int:
        return int(np.clip(round((jd - self.jd[0]) * 1440.0 / self.step_min), 0, self.n - 1))

    def minute_of(self, jd: float) -> float:
        return (jd - self.jd[0]) * 1440.0

    def jd_of(self, minute: float) -> float:
        return self.jd[0] + minute / 1440.0

    def utc(self, minute: float) -> _dt.datetime:
        return datetime_from_jd(self.jd_of(minute))

    def local(self, minute: float) -> _dt.datetime:
        """Civil local time at the site (tz database, so Chile's daylight-saving switch is
        honoured); falls back to the fixed `utc_offset_h` when no tz data is available."""
        u = self.utc(minute)
        try:
            from zoneinfo import ZoneInfo

            return u.astimezone(ZoneInfo(self.site.tz_name))
        except Exception:
            return u.astimezone(_dt.timezone(_dt.timedelta(hours=self.site.utc_offset_h)))

    def twilight_phase(self, i: int) -> str:
        h = self.sun_alt[i]
        if h > -0.833:
            return "day"
        if h > -6:
            return "civil twilight"
        if h > -12:
            return "nautical twilight"
        if h > -18:
            return "astronomical twilight"
        return "night"

    # ---- targets --------------------------------------------------------
    def targets(self, ra_deg: Sequence[float], dec_deg: Sequence[float]) -> dict:
        """Alt/az/airmass/HA/parallactic angle for N targets on the grid: arrays [N, n]."""
        key = (tuple(np.round(ra_deg, 7)), tuple(np.round(dec_deg, 7)))
        if key in self._target_cache:
            return self._target_cache[key]
        mid = self.jd[len(self.jd) // 2]
        ra_d, dec_d = precess_to_date(ra_deg, dec_deg, mid)
        ha = np.mod(self.lst[None, :] - ra_d[:, None] + 180.0, 360.0) - 180.0
        dec2 = np.broadcast_to(dec_d[:, None], ha.shape)
        alt_true, az = altaz(ha, dec2, self.site.lat_deg)
        alt = alt_true + refraction_deg(alt_true, self.site.pressure_hpa, self.site.temp_c)
        out = {
            "ra_date": ra_d,
            "dec_date": dec_d,
            "ha": ha,
            "alt": alt,
            "az": az,
            "airmass": airmass_from_alt(alt),
            "parallactic": parallactic_angle(ha, dec2, self.site.lat_deg),
            "moon_sep": angular_sep_altaz(alt_true, az, self.moon_alt[None, :], self.moon_az[None, :]),
        }
        self._target_cache[key] = out
        return out

    def point(self, ra_deg: float, dec_deg: float, jd: float) -> dict:
        """Alt/az of one direction at an arbitrary time (for live telemetry)."""
        ra_d, dec_d = precess_to_date([ra_deg], [dec_deg], jd)
        lst = lst_deg(jd, self.site.lon_deg)
        ha = float(np.mod(lst - ra_d[0] + 180.0, 360.0) - 180.0)
        alt_t, az = altaz(ha, dec_d[0], self.site.lat_deg)
        alt = float(alt_t + refraction_deg(alt_t, self.site.pressure_hpa, self.site.temp_c))
        return {
            "alt": alt,
            "az": float(az),
            "ha": ha,
            "airmass": float(airmass_from_alt(alt)),
            "parallactic": float(parallactic_angle(ha, dec_d[0], self.site.lat_deg)),
            "lst": float(lst),
        }

    def summary(self) -> dict:
        tw = self.twilight
        f = lambda jd: datetime_from_jd(jd).strftime("%H:%M") if np.isfinite(jd) else "--"
        i_mid = self.index_of(0.5 * (tw.astro_dusk + tw.astro_dawn)) if np.isfinite(tw.astro_dusk) else self.n // 2
        return {
            "site": self.site.name,
            "date": self.date,
            "sunset_ut": f(tw.sunset),
            "twi12_dusk_ut": f(tw.nautical_dusk),
            "twi18_dusk_ut": f(tw.astro_dusk),
            "twi18_dawn_ut": f(tw.astro_dawn),
            "twi12_dawn_ut": f(tw.nautical_dawn),
            "sunrise_ut": f(tw.sunrise),
            "night_hours_12deg": (tw.nautical_dawn - tw.nautical_dusk) * 24.0,
            "moon_illum_mid": float(self.moon_illum[i_mid]),
        }
