"""Sky brightness and atmosphere: dark sky, moonlight (Krisciunas & Schaefer 1991),
twilight, extinction, and seeing (Kolmogorov scaling with an optional von Karman outer-scale
correction, Tokovinin 2002).

All surface brightnesses are AB mag / arcsec^2. Wavelengths are in Angstrom.
"""

from __future__ import annotations

import numpy as np

from obsassist.astro.sites import BANDS, SUN_ABS_AB, Site

_SPEC_BANDS = ("U", "B", "V", "R", "I", "Y", "J", "H", "K")
DEG = np.pi / 180.0


def _interp_band(values: dict, lam) -> np.ndarray:
    keys = [b for b in _SPEC_BANDS if b in values]
    lx = np.log([BANDS[b][0] for b in keys])
    ly = np.array([values[b] for b in keys], dtype=float)
    return np.interp(np.log(np.asarray(lam, dtype=float)), lx, ly)


def extinction_coeff(site: Site, lam) -> np.ndarray:
    """mag / airmass at wavelength(s) lam, interpolated in log-wavelength."""
    return _interp_band(site.extinction, lam)


def dark_sky_zenith(site: Site, lam) -> np.ndarray:
    return _interp_band(site.dark_sky_ab, lam)


def _mag_to_flux(m):
    return 10.0 ** (-0.4 * np.asarray(m, dtype=float))


def _flux_to_mag(f):
    return -2.5 * np.log10(np.maximum(np.asarray(f, dtype=float), 1e-30))


def _ks_nanolambert_to_vmag(b_nl):
    return (20.7233 - np.log(np.maximum(b_nl, 1e-12) / 34.08)) / 0.92104


def _xks(z_deg):
    """Krisciunas & Schaefer airmass approximation X(Z) = (1 - 0.96 sin^2 Z)^-1/2."""
    z = np.clip(np.asarray(z_deg, dtype=float), 0, 89.5) * DEG
    return (1.0 - 0.96 * np.sin(z) ** 2) ** -0.5


def ks91_moon_nl(phase_angle, sep_deg, z_moon_deg, z_target_deg, k_v: float = 0.172):
    """Krisciunas & Schaefer (1991) eq. 15/20/21: moonlight surface brightness in nanoLamberts (V)."""
    alpha = np.abs(np.asarray(phase_angle, dtype=float))
    I_star = 10.0 ** (-0.4 * (3.84 + 0.026 * alpha + 4e-9 * alpha**4))
    rho = np.maximum(np.asarray(sep_deg, dtype=float), 0.5)
    rayleigh = 10.0**5.36 * (1.06 + np.cos(rho * DEG) ** 2)
    mie = np.where(rho > 10.0, 10.0 ** (6.15 - rho / 40.0), 6.2e7 / rho**2)  # eq. 18 / eq. 19 (aureole)
    f_rho = rayleigh + mie
    return f_rho * I_star * 10.0 ** (-0.4 * k_v * _xks(z_moon_deg)) * (1.0 - 10.0 ** (-0.4 * k_v * _xks(z_target_deg)))


def moon_sky_ab(site: Site, lam, moon_alt, phase_angle, sep_deg, target_alt) -> np.ndarray:
    """Moonlight surface brightness (AB mag/arcsec^2) added to the sky, KS91.

    phase_angle: 0 = full moon, 180 = new. The V-band model is carried to other wavelengths
    by (a) using the extinction coefficient at that wavelength in the scattering term, which
    carries the Rayleigh/aerosol wavelength dependence, and (b) the solar colour. Returns
    +99 (no light) when the Moon is below the horizon."""
    lam = np.asarray(lam, dtype=float)
    k = extinction_coeff(site, lam)
    z_moon = 90.0 - np.asarray(moon_alt, dtype=float)
    z_tgt = 90.0 - np.clip(np.asarray(target_alt, dtype=float), 1.0, 90.0)
    b_nl = ks91_moon_nl(phase_angle, sep_deg, z_moon, z_tgt, k)
    v_mag = _ks_nanolambert_to_vmag(b_nl)
    # solar colour relative to V, interpolated to lam
    sun = {b: SUN_ABS_AB[b] for b in _SPEC_BANDS}
    colour = _interp_band(sun, lam) - SUN_ABS_AB["V"]
    out = v_mag + 0.02 + colour
    return np.where(np.asarray(moon_alt) > -0.5, out, 99.0)


def twilight_sky_ab(lam, sun_alt) -> np.ndarray:
    """Zenith twilight surface brightness (AB mag/arcsec^2) added to the dark sky.

    V-band slope of ~1 mag per degree of solar depression between -9 and -18 deg with
    V ~ 17.6 at -12 deg, representative of Patat et al. (2006, A&A 455, 385) at Paranal;
    the colour follows the Sun's (twilight is scattered sunlight). APPROX."""
    h = np.asarray(sun_alt, dtype=float)
    v = 17.6 + 1.0 * (-h - 12.0)
    lam = np.asarray(lam, dtype=float)
    sun = {b: SUN_ABS_AB[b] for b in _SPEC_BANDS}
    colour = _interp_band(sun, lam) - SUN_ABS_AB["V"]
    return np.where(h < -0.5, v + colour, 5.0)


def sky_ab(site: Site, lam, target_alt, target_airmass, sun_alt, moon_alt, moon_phase_angle, moon_sep) -> np.ndarray:
    """Total sky surface brightness (AB mag/arcsec^2) toward a target: dark sky brightened
    toward the horizon (KS91 van Rhijn-like term), moonlight and twilight added in flux."""
    lam = np.asarray(lam, dtype=float)
    X = np.clip(np.asarray(target_airmass, dtype=float), 1.0, 10.0)
    k = extinction_coeff(site, lam)
    dark = dark_sky_zenith(site, lam) - 2.5 * np.log10(X * 10.0 ** (-0.4 * k * (X - 1.0)))
    f = _mag_to_flux(dark)
    f = f + _mag_to_flux(moon_sky_ab(site, lam, moon_alt, moon_phase_angle, moon_sep, target_alt))
    f = f + _mag_to_flux(twilight_sky_ab(lam, sun_alt))
    return _flux_to_mag(f)


# ---------------------------------------------------------------------------
# seeing
# ---------------------------------------------------------------------------
def fwhm_atm(seeing_500, airmass, lam, outer_scale_m: float = 25.0, von_karman: bool = True):
    """Atmospheric FWHM (arcsec) at wavelength lam and airmass X from DIMM seeing at 500 nm,
    zenith: s * X^0.6 * (lam/5000)^-0.2, times the Tokovinin (2002) von Karman factor
    sqrt(1 - 2.183 (r0/L0)^0.356) when `von_karman` (DIMMs measure the Kolmogorov value)."""
    s = np.asarray(seeing_500, dtype=float)
    X = np.asarray(airmass, dtype=float)
    lam = np.asarray(lam, dtype=float)
    fwhm_k = s * X**0.6 * (lam / 5000.0) ** -0.2
    if not von_karman or outer_scale_m <= 0:
        return fwhm_k
    r0 = 0.98 * lam * 1e-10 / (np.maximum(fwhm_k, 1e-3) / 206265.0)  # metres, at lam and X
    corr = np.sqrt(np.clip(1.0 - 2.183 * (r0 / outer_scale_m) ** 0.356, 0.2, 1.0))
    return fwhm_k * corr


def delivered_fwhm(
    seeing_500,
    airmass,
    lam,
    iq_floor: float,
    defocus: float = 0.0,
    outer_scale_m: float = 25.0,
    von_karman: bool = True,
):
    atm = fwhm_atm(seeing_500, airmass, lam, outer_scale_m, von_karman)
    return np.sqrt(atm**2 + iq_floor**2 + np.asarray(defocus, dtype=float) ** 2)


def adr_arcsec(lam1, lam2, airmass, pressure_hpa=615.0, temp_c=2.0, rh=0.2):
    """Atmospheric differential refraction between lam1 and lam2 (Angstrom), arcsec,
    Filippenko (1982) with Edlen refractivity."""

    def n_minus_1(lam_a):
        s2 = (1e4 / np.asarray(lam_a, dtype=float)) ** 2
        n15 = 1e-6 * (64.328 + 29498.1 / (146.0 - s2) + 255.4 / (41.0 - s2))
        p_mmhg = pressure_hpa * 0.750062
        return n15 * p_mmhg * (1 + (1.049 - 0.0157 * temp_c) * 1e-6 * p_mmhg) / (720.883 * (1 + 0.003661 * temp_c))

    X = np.asarray(airmass, dtype=float)
    tanz = np.sqrt(np.maximum(X**2 - 1.0, 0.0))
    return 206265.0 * (n_minus_1(lam1) - n_minus_1(lam2)) * tanz
