"""Observatory sites and telescopes.

Every number that came from documentation carries a comment with its source class;
the researched values (with URLs) live in docs/research/*.json and override these
defaults through `obsassist.params` when present. Values marked APPROX are
engineering estimates chosen to be representative, not quoted specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# photometric bands: effective wavelength (Å) and width (Å)
# ---------------------------------------------------------------------------
BANDS: Dict[str, Tuple[float, float]] = {
    "U": (3600.0, 660.0),
    "B": (4400.0, 940.0),
    "V": (5500.0, 880.0),
    "R": (6500.0, 1380.0),
    "I": (8000.0, 1490.0),
    "u": (3560.0, 560.0),
    "g": (4830.0, 1300.0),
    "r": (6260.0, 1150.0),
    "i": (7670.0, 1230.0),
    "z": (9100.0, 1000.0),
    "Y": (10200.0, 1000.0),
    "J": (12500.0, 1600.0),
    "H": (16350.0, 2900.0),
    "K": (21900.0, 3900.0),
}

# AB - Vega offsets (Blanton & Roweis 2007, Willmer 2018)
AB_MINUS_VEGA: Dict[str, float] = {
    "U": 0.79,
    "B": -0.09,
    "V": 0.02,
    "R": 0.21,
    "I": 0.45,
    "u": 0.0,
    "g": 0.0,
    "r": 0.0,
    "i": 0.0,
    "z": 0.0,
    "Y": 0.63,
    "J": 0.91,
    "H": 1.39,
    "K": 1.85,
}

# Absolute magnitude of the Sun, AB (Willmer 2018) - used to colour moonlight
SUN_ABS_AB: Dict[str, float] = {
    "U": 6.33,
    "B": 5.31,
    "V": 4.80,
    "R": 4.60,
    "I": 4.51,
    "u": 6.39,
    "g": 5.11,
    "r": 4.65,
    "i": 4.53,
    "z": 4.50,
    "Y": 4.50,
    "J": 4.56,
    "H": 4.70,
    "K": 5.12,
}


@dataclass(frozen=True)
class WeatherLimits:
    humidity_close: float = 90.0  # % RH: close at or above
    humidity_open: float = 85.0  # % RH: may reopen below (hysteresis)
    wind_close_ms: float = 20.0  # sustained wind speed that closes the dome
    dewpoint_margin_c: float = 2.0  # close when T - Td below this
    reopen_wait_min: float = 30.0  # conditions must stay good this long before reopening
    wind_high_ms: float = 15.6  # "high wind": do not point into the wind above this
    wind_avoid_deg: float = 60.0  # half-width of the forbidden azimuth zone around the wind direction


@dataclass(frozen=True)
class Site:
    key: str
    name: str
    lat_deg: float
    lon_deg: float  # east positive
    elev_m: float
    tz_name: str
    utc_offset_h: float  # fixed offset: locates local noon for the night grid, and the
    # display fallback when tz data is missing (display uses tz_name)
    pressure_hpa: float
    temp_c: float  # typical night temperature
    dark_sky_ab: Dict[str, float]  # zenith, dark time, AB mag / arcsec^2
    extinction: Dict[str, float]  # mag / airmass
    seeing_median: float  # DIMM, 500 nm, zenith, arcsec
    seeing_sigma_ln: float  # night-to-night log-normal width
    seeing_intranight_ln: float  # within-night OU amplitude (log)
    seeing_tau_min: float  # within-night correlation time
    clear_fraction: float  # fraction of nights that start photometric
    limits: WeatherLimits = field(default_factory=WeatherLimits)
    notes: str = ""


# Maunakea (Keck). Sourced in docs/research/keck_parameters.json: extinction from the Buton et al.
# (2013) median curve at band centres (NIR from MKO/Leggett 2006); dark sky from CFHT MegaCam (AB,
# ugriz), UBVRI converted to AB, NIR from the MOSFIRE imaging sky; DIMM median 0.65" (MKAM), quartiles
# estimated (sigma_ln ~0.38); closure rules from Keck's weather table: sustained wind 45 mph, dew point
# within 2 C, reopen after 30 min below limits; there is no published humidity limit (95 % from a paper
# that modelled Keck's criteria).
MAUNAKEA = Site(
    key="maunakea",
    name="Maunakea (W. M. Keck Observatory)",
    lat_deg=19.8263,
    lon_deg=-155.4744,
    elev_m=4145.0,
    tz_name="Pacific/Honolulu",
    utc_offset_h=-10.0,
    pressure_hpa=615.0,
    temp_c=1.5,
    dark_sky_ab={
        "U": 22.7,
        "B": 22.6,
        "V": 21.7,
        "R": 21.1,
        "I": 20.2,
        "u": 22.7,
        "g": 22.0,
        "r": 21.3,
        "i": 20.3,
        "z": 19.4,
        "Y": 18.09,
        "J": 16.58,
        "H": 15.14,
        "K": 15.54,
    },
    extinction={
        "U": 0.40,
        "B": 0.185,
        "V": 0.106,
        "R": 0.070,
        "I": 0.027,
        "u": 0.424,
        "g": 0.143,
        "r": 0.081,
        "i": 0.031,
        "z": 0.019,
        "Y": 0.02,
        "J": 0.047,
        "H": 0.029,
        "K": 0.052,
    },
    seeing_median=0.65,
    seeing_sigma_ln=0.38,
    seeing_intranight_ln=0.22,
    seeing_tau_min=35.0,
    clear_fraction=0.72,
    limits=WeatherLimits(
        humidity_close=95.0,
        humidity_open=90.0,
        wind_close_ms=20.1,
        dewpoint_margin_c=2.0,
        reopen_wait_min=30.0,
        wind_high_ms=15.0,
        wind_avoid_deg=60.0,
    ),
    notes="See docs/research/keck_parameters.json (URLs per value).",
)

# Las Campanas (Magellan, Cerro Manqui). Sourced in docs/research/magellan_sky_parameters.json:
# extinction u/B/g/V/r/i/Y/J/H measured (CSP, Krisciunas+2017), U/R/I/z/K estimated; dark sky
# from the ESO SkyCalc model used by the official LCO ETC (new Moon), converted to AB; NIR sky
# from the FourStar page; DIMM percentiles from GMT site testing (median 0.62", sigma_ln 0.34);
# wind closure 15.6 m/s (35 mph) documented. The humidity limit is not published: 80 % is where
# weather.lco.cl draws its limit line (read from a photo), dew-point margin and reopen wait are
# ESO La Silla proxies. Outer scale 25 m (Magellan IQ studies).
LAS_CAMPANAS = Site(
    key="lco",
    name="Las Campanas Observatory (Magellan)",
    lat_deg=-29.015,
    lon_deg=-70.6917,
    elev_m=2380.0,
    tz_name="America/Santiago",
    utc_offset_h=-3.0,
    pressure_hpa=765.0,
    temp_c=9.0,
    dark_sky_ab={
        "U": 22.88,
        "B": 22.69,
        "V": 21.85,
        "R": 21.27,
        "I": 20.19,
        "u": 23.05,
        "g": 22.25,
        "r": 21.20,
        "i": 20.46,
        "z": 19.61,
        "Y": 18.24,
        "J": 16.51,
        "H": 15.19,
        "K": 14.95,
    },
    extinction={
        "U": 0.46,
        "B": 0.242,
        "V": 0.144,
        "R": 0.09,
        "I": 0.05,
        "u": 0.511,
        "g": 0.191,
        "r": 0.103,
        "i": 0.059,
        "z": 0.04,
        "Y": 0.044,
        "J": 0.076,
        "H": 0.041,
        "K": 0.07,
    },
    seeing_median=0.62,
    seeing_sigma_ln=0.34,
    seeing_intranight_ln=0.20,
    seeing_tau_min=40.0,
    clear_fraction=0.64,
    limits=WeatherLimits(
        humidity_close=80.0,
        humidity_open=75.0,
        wind_close_ms=15.6,
        dewpoint_margin_c=2.0,
        reopen_wait_min=30.0,
        wind_high_ms=12.5,
        wind_avoid_deg=60.0,
    ),
    notes="See docs/research/magellan_sky_parameters.json (URLs per value).",
)

SITES: Dict[str, Site] = {s.key: s for s in (MAUNAKEA, LAS_CAMPANAS)}


@dataclass(frozen=True)
class Telescope:
    key: str
    name: str
    site_key: str
    area_m2: float  # effective collecting area
    el_min_deg: float  # hard lower elevation limit
    el_max_deg: float  # upper limit (zenith keyhole for alt-az mounts)
    slew_az_dps: float
    slew_el_dps: float
    settle_s: float
    iq_floor_arcsec: float  # dome + optics + guiding, added in quadrature
    focus_drift_arcsec_per_c: float  # defocus blur growth per degC since last focus
    focus_run_s: float
    instruments: Tuple[str, ...]
    # (az_lo, az_hi, el_min): additional elevation limits over an azimuth range (e.g. Keck deck)
    az_limits: Tuple[Tuple[float, float, float], ...] = ()
    outer_scale_m: float = 25.0  # von Karman L0 for the FWHM correction


# Keck: effective area 72.3 m^2 (recommended in keck_parameters.json); slew 1.3 deg/s az, 0.5 deg/s el;
# Nasmyth-deck limits documented (K1: el >= 33.3 for az 5.3-146.2; K2: el >= 36.8 for az 185.3-332.8).
KECK1 = Telescope(
    key="keck1",
    name="Keck I",
    site_key="maunakea",
    area_m2=72.3,
    el_min_deg=18.0,
    el_max_deg=88.9,
    slew_az_dps=1.3,
    slew_el_dps=0.5,
    settle_s=15.0,
    iq_floor_arcsec=0.30,
    focus_drift_arcsec_per_c=0.12,
    focus_run_s=420.0,
    instruments=("LRIS", "MOSFIRE", "HIRES"),
    az_limits=((5.3, 146.2, 33.3),),
)
KECK2 = Telescope(
    key="keck2",
    name="Keck II",
    site_key="maunakea",
    area_m2=72.3,
    el_min_deg=18.0,
    el_max_deg=89.5,
    slew_az_dps=1.3,
    slew_el_dps=0.5,
    settle_s=15.0,
    iq_floor_arcsec=0.30,
    focus_drift_arcsec_per_c=0.12,
    focus_run_s=420.0,
    instruments=("DEIMOS", "KCWI"),
    az_limits=((185.3, 332.8, 36.8),),
)
# Magellan: 6.5 m, effective area 30.638 m^2 (LCO ETC PARAM files); slew/settle/el limit are estimates.
BAADE = Telescope(
    key="baade",
    name="Magellan Baade",
    site_key="lco",
    area_m2=30.638,
    el_min_deg=15.0,
    el_max_deg=89.0,
    slew_az_dps=2.0,
    slew_el_dps=1.0,
    settle_s=30.0,
    iq_floor_arcsec=0.30,
    focus_drift_arcsec_per_c=0.10,
    focus_run_s=300.0,
    instruments=("IMACS", "FIRE"),
)
CLAY = Telescope(
    key="clay",
    name="Magellan Clay",
    site_key="lco",
    area_m2=30.638,
    el_min_deg=15.0,
    el_max_deg=89.0,
    slew_az_dps=2.0,
    slew_el_dps=1.0,
    settle_s=30.0,
    iq_floor_arcsec=0.25,
    focus_drift_arcsec_per_c=0.10,
    focus_run_s=300.0,
    instruments=("LDSS3", "MIKE"),
)

TELESCOPES: Dict[str, Telescope] = {t.key: t for t in (KECK1, KECK2, BAADE, CLAY)}


def extra_el_limit(tel: Telescope, az_deg) -> "float | object":
    """Elevation limit (deg) at the given azimuth(s), including any azimuth-dependent
    limits. Works on scalars and numpy arrays."""
    import numpy as np

    az = np.asarray(az_deg, dtype=float) % 360.0
    lim = np.full_like(az, tel.el_min_deg, dtype=float)
    for lo, hi, el in tel.az_limits:
        inside = (az >= lo) & (az <= hi) if lo <= hi else ((az >= lo) | (az <= hi))
        lim = np.where(inside, np.maximum(lim, el), lim)
    return lim if lim.ndim else float(lim)
