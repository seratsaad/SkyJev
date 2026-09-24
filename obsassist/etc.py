"""Exposure time calculator.

Physics, per exposure of length t (all rates in electrons per second):

    S   = N_ph(lam) * A * eta(lam) * 10^(-0.4 (k(lam) X + cloud)) * dlam * f_slit * f_extract
    B   = I_sky(lam) * A * eta(lam) * dlam * (slit width x spatial pixel)        per pixel
    SNR = S t / sqrt(S t + n_pix (B t + D t + RN^2))

N_ph = f_nu / (h lam) photons s^-1 cm^-2 A^-1 for an AB magnitude. The PSF is a Moffat
(beta = 4.765, the Kolmogorov-like value of Trujillo et al. 2001) whose FWHM is the delivered
image quality: DIMM seeing scaled to airmass and wavelength (with the von Karman outer-scale
correction), plus telescope/dome image quality and any defocus, in quadrature. The slit
transmission is the Moffat's line-spread function integrated over the slit width (closed
form, incomplete beta function), including an optional centroid offset from atmospheric
differential refraction. Exposures combine with optimal weights: SNR_tot^2 = sum SNR_i^2.

Everything is vectorised over conditions so the simulator can precompute a whole night
per target in one call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import betainc, gammaln

from obsassist.astro.sites import AB_MINUS_VEGA, BANDS, Site, Telescope
from obsassist.astro.sky import adr_arcsec, delivered_fwhm, extinction_coeff, sky_ab
from obsassist.instruments.base import InstrumentConfig

H_CGS = 6.62607015e-27
K_CGS = 1.380649e-16
C_CGS = 2.99792458e10
BETA = 4.765
LINEARITY = 0.80  # fraction of full well treated as the saturation limit


# ---------------------------------------------------------------------------
# source
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    mag: float
    band: str = "r"
    system: str = "AB"  # "AB" | "Vega"
    kind: str = "point"  # "point" | "extended" (mag is then mag/arcsec^2)
    sed: str = "flat_fnu"  # "flat_fnu" | "flat_flam" | "bb:<T_K>"
    extent_arcsec: float = 1.0  # extended: length extracted along the slit / aperture radius

    def ab_at(self, lam) -> np.ndarray:
        m_ab = self.mag + (AB_MINUS_VEGA.get(self.band, 0.0) if self.system.lower() == "vega" else 0.0)
        lam = np.asarray(lam, dtype=float)
        lam_b = BANDS[self.band][0]
        if self.sed == "flat_fnu":
            return np.full_like(lam, m_ab, dtype=float)
        if self.sed == "flat_flam":
            return m_ab - 5.0 * np.log10(lam / lam_b)
        if self.sed.startswith("bb:"):
            T = float(self.sed[3:])

            def bnu(lam_a):
                nu = C_CGS / (np.asarray(lam_a) * 1e-8)
                return nu**3 / np.expm1(H_CGS * nu / (K_CGS * T))

            return m_ab - 2.5 * np.log10(bnu(lam) / bnu(lam_b))
        raise ValueError(f"unknown sed {self.sed!r}")


def photons_per_A(m_ab, lam) -> np.ndarray:
    """photons s^-1 cm^-2 A^-1 for AB magnitude m at wavelength lam (A)."""
    f_nu = 10.0 ** (-0.4 * (np.asarray(m_ab, dtype=float) + 48.6))
    return f_nu / (H_CGS * np.asarray(lam, dtype=float))


# ---------------------------------------------------------------------------
# Moffat PSF helpers
# ---------------------------------------------------------------------------
def moffat_alpha(fwhm, beta: float = BETA):
    return np.asarray(fwhm, dtype=float) / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))


def _lsf_cum(a, alpha, beta: float = BETA):
    """Fraction of a 2-D Moffat's flux with 0 <= x <= a (signed a): the 1-D marginal is
    (1 + x^2/alpha^2)^-(beta - 1/2); its CDF is an incomplete beta function."""
    a = np.asarray(a, dtype=float)
    s2 = np.sin(np.arctan(np.abs(a) / alpha)) ** 2
    return np.sign(a) * 0.5 * betainc(0.5, beta - 1.0, s2)


def strip_fraction(lo, hi, fwhm, beta: float = BETA):
    """Fraction of a centred Moffat's flux falling in the strip lo <= x <= hi (arcsec)."""
    alpha = moffat_alpha(fwhm, beta)
    return _lsf_cum(hi, alpha, beta) - _lsf_cum(lo, alpha, beta)


def slit_fraction(width, fwhm, offset=0.0, beta: float = BETA):
    """Slit transmission for a slit of `width` with the PSF centroid `offset` from its centre."""
    w = np.asarray(width, dtype=float)
    d = np.asarray(offset, dtype=float)
    return strip_fraction(-w / 2 - d, w / 2 - d, fwhm, beta)


def enclosed_energy(r, fwhm, beta: float = BETA):
    alpha = moffat_alpha(fwhm, beta)
    return 1.0 - (1.0 + (np.asarray(r, dtype=float) / alpha) ** 2) ** (1.0 - beta)


def lsf_peak_per_arcsec(fwhm, beta: float = BETA):
    """Peak of the normalised 1-D marginal (per arcsec)."""
    alpha = moffat_alpha(fwhm, beta)
    log_norm = 0.5 * np.log(np.pi) + gammaln(beta - 1.0) - gammaln(beta - 0.5)
    return 1.0 / (alpha * np.exp(log_norm))


def psf_peak_per_arcsec2(fwhm, beta: float = BETA):
    alpha = moffat_alpha(fwhm, beta)
    return (beta - 1.0) / (np.pi * alpha**2)


# ---------------------------------------------------------------------------
# rates
# ---------------------------------------------------------------------------
@dataclass
class Rates:
    """Per-second electron rates for one (source, configuration, conditions). Arrays allowed."""

    signal: np.ndarray  # e-/s in the extraction aperture (spec: per binned spectral pixel)
    sky_pix: np.ndarray  # e-/s per binned pixel
    dark_pix: float  # e-/s per binned pixel
    npix: np.ndarray  # binned pixels in the aperture (spec: spatial pixels per column)
    rn2: float  # read noise^2 per pixel per read
    peak_pix: np.ndarray  # e-/s in the brightest pixel (source + sky + dark)
    fwhm: np.ndarray  # delivered FWHM at lam (arcsec)
    slit_frac: np.ndarray
    ap_frac: np.ndarray
    sky_mag: np.ndarray  # AB mag / arcsec^2 at lam
    transmission: np.ndarray  # atmosphere + cloud
    lam: float
    unit_scale: float  # multiply per-pixel S/N by this to get the requested unit
    unit: str

    def var_rate(self):
        return self.signal + self.npix * (self.sky_pix + self.dark_pix)

    def snr(self, t, n_exp: int = 1):
        """S/N in the requested unit after n_exp exposures of t seconds each."""
        s = self.signal * t
        var = s + self.npix * ((self.sky_pix + self.dark_pix) * t + self.rn2)
        return self.unit_scale * np.sqrt(n_exp) * s / np.sqrt(np.maximum(var, 1e-12))

    def t_saturate(self, full_well: float) -> np.ndarray:
        return LINEARITY * full_well / np.maximum(self.peak_pix, 1e-9)

    def regime(self) -> str:
        """Which noise term dominates at a 600 s exposure (scalar rates only)."""
        s = float(np.mean(self.signal)) * 600
        sky = float(np.mean(self.npix * self.sky_pix)) * 600
        rn = float(np.mean(self.npix)) * self.rn2
        dk = float(np.mean(self.npix)) * self.dark_pix * 600
        terms = {"source": s, "sky": sky, "read noise": rn, "dark": dk}
        return max(terms, key=terms.get)


def unit_scale(cfg: InstrumentConfig, unit: str, lam: float) -> float:
    if cfg.mode == "img" or unit in ("per_pix", "aperture", "total"):
        return 1.0
    if unit == "per_A":
        return float(1.0 / np.sqrt(cfg.dlam_at(lam)))
    if unit == "per_resel":
        if cfg.resolution > 0:
            n = float((lam / cfg.resolution) / cfg.dlam_at(lam))
        else:
            n = cfg.slit_width / (cfg.pixscale * cfg.bin_spectral)
        return float(np.sqrt(max(n, 1.0)))
    raise ValueError(f"unknown S/N unit {unit!r}")


def compute_rates(
    cfg: InstrumentConfig,
    tel: Telescope,
    site: Site,
    src: Source,
    *,
    seeing_500,
    airmass,
    alt=None,
    cloud_mag=0.0,
    sun_alt=-30.0,
    moon_alt=-30.0,
    moon_phase_angle=180.0,
    moon_sep=90.0,
    defocus=0.0,
    lam: Optional[float] = None,
    unit: str = "per_A",
    adr_offset=0.0,
    sky_mag=None,
    von_karman: bool = True,
) -> Rates:
    """All condition arguments may be scalars or equal-length arrays."""
    lam = float(lam if lam is not None else cfg.lam_ref)
    X = np.asarray(airmass, dtype=float)
    if alt is None:
        alt = 90.0 - np.degrees(np.arccos(np.clip(1.0 / np.maximum(X, 1.0), -1, 1)))
    area = tel.area_m2 * 1e4  # cm^2
    eta = float(cfg.eff(lam))
    if eta <= 0:
        raise ValueError(f"{cfg.key}: no throughput at {lam:.0f} A (range {cfg.lam_min:.0f}-{cfg.lam_max:.0f})")
    k = float(extinction_coeff(site, lam))
    trans = 10.0 ** (-0.4 * (k * X + np.asarray(cloud_mag, dtype=float)))
    fwhm = delivered_fwhm(seeing_500, X, lam, tel.iq_floor_arcsec, defocus, tel.outer_scale_m, von_karman)
    if sky_mag is None:
        sky_mag = sky_ab(site, lam, alt, X, sun_alt, moon_alt, moon_phase_angle, moon_sep)
    sky_mag = np.asarray(sky_mag, dtype=float)
    n_sky = photons_per_A(sky_mag, lam)  # per arcsec^2
    m_src = float(src.ab_at(lam))
    n_src = photons_per_A(m_src, lam)
    dark = cfg.dark_e_per_hr / 3600.0 * cfg.bin_spatial * cfg.bin_spectral
    pix = cfg.spatial_scale

    if cfg.mode == "spec":
        dlam = float(cfg.dlam_at(lam))
        sky_pix = n_sky * area * eta * dlam * cfg.slit_width * pix * (cfg.sky_line_factor if cfg.nir else 1.0)
        if src.kind == "point":
            f_slit = slit_fraction(cfg.slit_width, fwhm, adr_offset)
            half = 1.0 * fwhm  # 2 x FWHM extraction aperture (the LCO ETC convention)
            f_ext = strip_fraction(-half, half, fwhm)
            npix = np.maximum(2.0 * half / pix, 1.0)
            through = n_src * area * eta * trans * dlam * f_slit
            signal = through * f_ext
            peak_src = through * lsf_peak_per_arcsec(fwhm) * pix
        else:
            L = src.extent_arcsec
            f_slit = np.ones_like(fwhm)
            f_ext = np.ones_like(fwhm)
            npix = np.full_like(fwhm, max(L / pix, 1.0))
            signal = n_src * area * eta * trans * dlam * cfg.slit_width * L
            peak_src = signal / npix
    else:
        width = cfg.band_width or BANDS[cfg.band][1]
        sky_pix = n_sky * area * eta * width * pix**2
        if src.kind == "point":
            r_ap = 0.8 * fwhm
            f_ext = enclosed_energy(r_ap, fwhm)
            f_slit = np.ones_like(fwhm)
            npix = np.maximum(np.pi * r_ap**2 / pix**2, 1.0)
            total = n_src * area * eta * trans * width
            signal = total * f_ext
            peak_src = total * psf_peak_per_arcsec2(fwhm) * pix**2
        else:
            r = src.extent_arcsec
            f_ext = np.ones_like(fwhm)
            f_slit = np.ones_like(fwhm)
            npix = np.full_like(fwhm, max(np.pi * r**2 / pix**2, 1.0))
            signal = n_src * area * eta * trans * width * np.pi * r**2
            peak_src = signal / npix
    return Rates(
        signal=np.asarray(signal, dtype=float),
        sky_pix=np.asarray(sky_pix, dtype=float),
        dark_pix=dark,
        npix=np.asarray(npix, dtype=float),
        rn2=cfg.read_noise**2,
        peak_pix=np.asarray(peak_src + sky_pix + dark, dtype=float),
        fwhm=np.asarray(fwhm, dtype=float),
        slit_frac=np.asarray(f_slit, dtype=float),
        ap_frac=np.asarray(f_ext, dtype=float),
        sky_mag=sky_mag,
        transmission=np.asarray(trans, dtype=float),
        lam=lam,
        unit_scale=unit_scale(cfg, unit, lam),
        unit=unit,
    )


def adr_offset_for(
    cfg: InstrumentConfig,
    site: Site,
    lam: float,
    airmass,
    pa_deg: Optional[float],
    parallactic_deg,
    lam_guide: float = 6500.0,
):
    """Centroid offset across the slit from differential refraction between lam and the
    guiding wavelength, when the slit is not at the parallactic angle and there is no ADC."""
    if cfg.mode != "spec" or cfg.adc or pa_deg is None:
        return 0.0
    d = adr_arcsec(lam, lam_guide, airmass, site.pressure_hpa, site.temp_c)
    return np.abs(d * np.sin(np.radians(np.asarray(parallactic_deg) - pa_deg)))


def saturation_factor(
    cfg: InstrumentConfig, tel: Telescope, site: Site, src: Source, seeing: float = 0.6, airmass: float = 1.2
) -> float:
    """How much sooner the brightest pixel anywhere in the exposure saturates than the pixel at
    cfg's reference wavelength: the maximum of peak / full well over wavelengths of every arm that
    shares the exposure (e.g. both MIKE arms), relative to cfg at lam_ref. At least 1. A hot star
    observed for a red-arm goal saturates first in the blue; an echelle order can saturate away
    from lam_ref."""
    from obsassist.instruments.base import siblings

    ref = float(compute_rates(cfg, tel, site, src, seeing_500=seeing, airmass=airmass).peak_pix) / cfg.full_well
    worst = ref
    for c in siblings(cfg):
        for lam in np.linspace(c.lam_min, c.lam_max, 13)[1:-1]:
            if float(c.eff(lam)) <= 0:
                continue
            r = compute_rates(c, tel, site, src, seeing_500=seeing, airmass=airmass, lam=float(lam))
            worst = max(worst, float(r.peak_pix) / c.full_well)
    return max(1.0, worst / max(ref, 1e-30))


# ---------------------------------------------------------------------------
# planning helpers
# ---------------------------------------------------------------------------
@dataclass
class ExposurePlan:
    t_exp: float
    n_exp: int
    snr_each: float
    snr_final: float
    open_shutter_s: float
    wall_s: float  # including readouts
    limited_by: str  # "max_exp" | "saturation" | "goal"
    regime: str


def plan_exposures(
    rates: Rates,
    cfg: InstrumentConfig,
    goal: float,
    snr_have: float = 0.0,
    t_pref: Optional[float] = None,
    sat_factor: float = 1.0,
) -> ExposurePlan:
    """Equal exposures of min(t_pref, max_exp, saturation) seconds until the goal is met
    (scalar rates). The last exposure is shortened when a shorter one reaches the goal.
    `sat_factor` (see `saturation_factor`) accounts for pixels that saturate before lam_ref's."""
    t_sat = float(rates.t_saturate(cfg.full_well)) / sat_factor
    t_max = min(t_pref or cfg.max_exp_s, cfg.max_exp_s, t_sat)
    limited = "saturation" if t_sat < min(t_pref or cfg.max_exp_s, cfg.max_exp_s) else "max_exp"
    t_max = max(t_max, cfg.min_exp_s)
    need_sq = max(goal**2 - snr_have**2, 0.0)
    s1 = float(rates.snr(t_max))
    if need_sq <= 0:
        return ExposurePlan(t_max, 0, s1, snr_have, 0.0, 0.0, "goal", rates.regime())
    n_full = int(np.floor(need_sq / max(s1**2, 1e-12)))
    rem = need_sq - n_full * s1**2
    t_last = 0.0
    if rem > 1e-9:
        # shortest single exposure delivering the remaining S/N^2 (bisection)
        lo, hi = cfg.min_exp_s, t_max
        if float(rates.snr(hi)) ** 2 < rem:
            t_last = t_max
        else:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if float(rates.snr(mid)) ** 2 >= rem:
                    hi = mid
                else:
                    lo = mid
            t_last = hi
    n = n_full + (1 if t_last > 0 else 0)
    if n == 0:
        return ExposurePlan(t_max, 0, s1, snr_have, 0.0, 0.0, "goal", rates.regime())
    # express as equal exposures when the saving from a short last exposure is small
    if t_last > 0 and t_last > 0.7 * t_max:
        t_last = t_max
    if n_full == 0:  # a single exposure shorter than the maximum does it
        s_last = float(rates.snr(t_last))
        return ExposurePlan(
            t_last,
            1,
            s_last,
            float(np.sqrt(snr_have**2 + s_last**2)),
            t_last,
            t_last + cfg.readout_s,
            "goal",
            rates.regime(),
        )
    open_s = n_full * t_max + t_last
    snr_final = float(np.sqrt(snr_have**2 + n_full * s1**2 + (float(rates.snr(t_last)) ** 2 if t_last > 0 else 0)))
    wall = open_s + n * cfg.readout_s
    return ExposurePlan(t_max, n, s1, snr_final, open_s, wall, limited if n_full else "goal", rates.regime())
