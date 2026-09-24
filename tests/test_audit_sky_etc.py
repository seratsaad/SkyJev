"""Audit: sky model, seeing, differential refraction and the ETC, checked against closed forms,
numerical integration and published values."""

import numpy as np
import pytest
from scipy import integrate

from obsassist.astro.sites import SITES, TELESCOPES
from obsassist.astro.sky import (
    _ks_nanolambert_to_vmag,
    adr_arcsec,
    delivered_fwhm,
    fwhm_atm,
    ks91_moon_nl,
    moon_sky_ab,
    sky_ab,
    twilight_sky_ab,
)
from obsassist.etc import (
    BETA,
    Source,
    compute_rates,
    enclosed_energy,
    lsf_peak_per_arcsec,
    moffat_alpha,
    photons_per_A,
    plan_exposures,
    psf_peak_per_arcsec2,
    slit_fraction,
    strip_fraction,
    unit_scale,
)
from obsassist.instruments.base import all_configs, get_config

LCO, MK = SITES["lco"], SITES["maunakea"]


# ----------------------------------------------------------------------------- sky
def test_ks91_nanolambert_conversion_inverts_eq1():
    # KS91 eq. 1: B[nL] = 34.08 exp(20.7233 - 0.92104 V)
    for v in (17.0, 19.5, 21.7, 23.0):
        assert float(_ks_nanolambert_to_vmag(34.08 * np.exp(20.7233 - 0.92104 * v))) == pytest.approx(v, abs=1e-9)


def test_moon_sky_ab_conversion_and_colour():
    # at V the AB value is the KS91 V (Vega) value + 0.02 (V_AB - V_Vega), no colour term
    b = float(
        ks91_moon_nl(40.0, 50.0, 40.0, 30.0, float(np.interp(np.log(5500), np.log([4400, 5500]), [0.242, 0.144])))
    )
    v = float(_ks_nanolambert_to_vmag(b))
    assert float(moon_sky_ab(LCO, 5500.0, 50.0, 40.0, 50.0, 60.0)) == pytest.approx(v + 0.02, abs=1e-6)
    # moonlight is brighter at full than at quarter, and brighter close to the Moon
    assert moon_sky_ab(LCO, 5500, 45, 0, 40, 60) < moon_sky_ab(LCO, 5500, 45, 90, 40, 60)
    assert moon_sky_ab(LCO, 5500, 45, 30, 15, 60) < moon_sky_ab(LCO, 5500, 45, 30, 90, 60)
    assert moon_sky_ab(LCO, 5500, -1.0, 0, 40, 60) == 99.0
    # redder than V in the red (Rayleigh k falls faster than the solar colour rises)
    assert moon_sky_ab(LCO, 8000, 45, 0, 45, 60) > moon_sky_ab(LCO, 5500, 45, 0, 45, 60)


def test_twilight_monotonic_and_dark_sky_brightens_with_airmass():
    h = np.linspace(-30, -0.6, 300)
    for lam in (4400.0, 5500.0, 8000.0):
        tw = twilight_sky_ab(lam, h)
        assert np.all(np.diff(tw) < 0)  # brighter (smaller mag) as the Sun rises
    X = np.array([1.0, 1.2, 1.5, 2.0, 3.0])
    alt = 90 - np.degrees(np.arccos(1 / X))
    s = sky_ab(LCO, 5500.0, alt, X, -40.0, -30.0, 180.0, 90.0)
    assert np.all(np.diff(s) < 0)  # KS91 eq. 2: X 10^(-0.4 k (X - 1)) > 1
    assert float(s[0]) == pytest.approx(LCO.dark_sky_ab["V"], abs=1e-6)


def test_sky_ab_is_always_finite():
    rng = np.random.default_rng(1)
    n = 20000
    for site in (LCO, MK):
        for lam in (3300.0, 5500.0, 9000.0, 21900.0):
            out = sky_ab(
                site,
                lam,
                rng.uniform(-10, 90, n),
                rng.uniform(0.5, 40, n),
                rng.uniform(-90, 10, n),
                rng.uniform(-90, 90, n),
                rng.uniform(0, 180, n),
                rng.uniform(0, 180, n),
            )
            assert np.isfinite(out).all() and out.min() > 4.0 and out.max() < 30.0


# ----------------------------------------------------------------------------- seeing and ADR
def test_von_karman_factor_is_tokovinin_2002():
    for s, X, lam, L0 in ((0.4, 1.0, 5000.0, 25.0), (0.8, 1.5, 7000.0, 25.0), (1.2, 2.0, 4000.0, 40.0)):
        fk = s * X**0.6 * (lam / 5000) ** -0.2
        r0 = 0.98 * lam * 1e-10 / (fk / 206264.8)
        want = fk * np.sqrt(1 - 2.183 * (r0 / L0) ** 0.356)
        assert float(fwhm_atm(s, X, lam, L0)) == pytest.approx(want, rel=1e-4)
    assert float(delivered_fwhm(0.6, 1.0, 5000, 0.3, 0.4, von_karman=False)) == pytest.approx(
        np.sqrt(0.36 + 0.09 + 0.16)
    )


def test_adr_matches_filippenko_1982():
    # Filippenko (1982) Table 1 (P = 600 mmHg, T = 7 C): airmass 1.5, 4000 A relative to 5000 A: 0.71"
    assert float(adr_arcsec(4000, 5000, 1.5, 600 / 0.750062, 7.0)) == pytest.approx(0.71, abs=0.02)
    # blue is lifted more than red: positive for lam1 < lam2; zero at the zenith
    assert float(adr_arcsec(4000, 6500, 1.5, LCO.pressure_hpa, LCO.temp_c)) == pytest.approx(1.15, abs=0.03)
    assert float(adr_arcsec(6500, 4000, 1.5)) < 0 and float(adr_arcsec(4000, 6500, 1.0)) == 0.0
    # scales with tan z and with pressure
    a1, a2 = adr_arcsec(4000, 6500, np.array([1.2, 2.0]), 765.0, 9.0)
    assert a2 / a1 == pytest.approx(np.sqrt(3.0) / np.sqrt(1.44 - 1), rel=1e-9)


# ----------------------------------------------------------------------------- ETC
def test_ab_zero_photon_rate_and_units():
    assert float(photons_per_A(0.0, 5500.0)) == pytest.approx(3631e-26 * 1e3 / (6.62607015e-27 * 5500.0), rel=1e-3)
    assert float(photons_per_A(0.0, 5500.0)) == pytest.approx(996, rel=0.005)
    assert float(photons_per_A(2.5, 5500.0) / photons_per_A(0.0, 5500.0)) == pytest.approx(0.1)


@pytest.mark.parametrize("fwhm", [0.5, 0.9, 1.6])
def test_moffat_helpers_integrate_correctly(fwhm):
    a = moffat_alpha(fwhm)

    def psf(x, y):
        return (BETA - 1) / (np.pi * a**2) * (1 + (x * x + y * y) / a**2) ** (-BETA)

    assert psf(fwhm / 2, 0) / psf(0, 0) == pytest.approx(0.5, abs=1e-12)  # it is the FWHM
    assert float(psf_peak_per_arcsec2(fwhm)) == pytest.approx(psf(0, 0))
    lsf = integrate.quad(lambda y: psf(0.0, y), -80, 80, epsabs=1e-12)[0]
    assert float(lsf_peak_per_arcsec(fwhm)) == pytest.approx(lsf, rel=1e-6)
    ee = integrate.quad(lambda r: 2 * np.pi * r * psf(r, 0), 0, 0.7 * fwhm)[0]
    assert float(enclosed_energy(0.7 * fwhm, fwhm)) == pytest.approx(ee, rel=1e-8)
    marg = lambda x: integrate.quad(lambda y: psf(x, y), -80, 80, epsabs=1e-12)[0]
    strip = integrate.quad(marg, -0.3 * fwhm, 0.55 * fwhm)[0]
    assert float(strip_fraction(-0.3 * fwhm, 0.55 * fwhm, fwhm)) == pytest.approx(strip, rel=1e-6)
    off = integrate.quad(marg, -0.5 - 0.2, 0.5 - 0.2)[0]
    assert float(slit_fraction(1.0, fwhm, 0.2)) == pytest.approx(off, rel=1e-6)
    assert float(slit_fraction(1.0, fwhm, 0.2)) == pytest.approx(float(slit_fraction(1.0, fwhm, -0.2)))


def test_snr_unit_scalings():
    cfg = get_config("LRIS-R400")
    assert unit_scale(cfg, "per_pix", 7000) == 1.0
    assert unit_scale(cfg, "per_A", 7000) == pytest.approx(1 / np.sqrt(cfg.dlam_pix))
    assert unit_scale(cfg, "per_resel", 7000) == pytest.approx(np.sqrt(7000 / cfg.resolution / cfg.dlam_pix))
    assert unit_scale(get_config("LRIS-IMG-R"), "per_A", 6500) == 1.0
    r = compute_rates(cfg, TELESCOPES["keck1"], MK, Source(21.0, "r"), seeing_500=0.7, airmass=1.2, unit="per_pix")
    rA = compute_rates(cfg, TELESCOPES["keck1"], MK, Source(21.0, "r"), seeing_500=0.7, airmass=1.2, unit="per_A")
    assert float(rA.snr(900)) == pytest.approx(float(r.snr(900)) / np.sqrt(cfg.dlam_pix))


def test_signal_and_sky_bookkeeping_from_first_principles():
    """Rebuild the point-source spectroscopic rates by hand from the documented formula."""
    cfg, tel, site = get_config("DEIMOS-600ZD"), TELESCOPES["keck2"], MK
    lam = 7000.0
    r = compute_rates(
        cfg,
        tel,
        site,
        Source(20.0, "r"),
        seeing_500=0.7,
        airmass=1.3,
        lam=lam,
        unit="per_pix",
        sky_mag=21.0,
        cloud_mag=0.2,
    )
    area, eta = tel.area_m2 * 1e4, float(cfg.eff(lam))
    k = float(np.interp(np.log(lam), np.log([6500.0, 8000.0]), [site.extinction["R"], site.extinction["I"]]))
    trans = 10 ** (-0.4 * (k * 1.3 + 0.2))
    fw = float(r.fwhm)
    want_sig = (
        float(photons_per_A(20.0, lam))
        * area
        * eta
        * trans
        * cfg.dlam_pix
        * float(slit_fraction(1.0, fw))
        * float(strip_fraction(-fw, fw, fw))
    )
    assert float(r.signal) == pytest.approx(want_sig, rel=1e-9)
    want_sky = float(photons_per_A(21.0, lam)) * area * eta * cfg.dlam_pix * cfg.slit_width * cfg.spatial_scale
    assert float(r.sky_pix) == pytest.approx(want_sky, rel=1e-9)
    assert float(r.npix) == pytest.approx(2 * fw / cfg.spatial_scale)


def test_imaging_and_extended_branches():
    tel = TELESCOPES["keck1"]
    img = get_config("LRIS-IMG-R")
    p = compute_rates(img, tel, MK, Source(22.0, "R"), seeing_500=0.6, airmass=1.1)
    assert float(p.ap_frac) == pytest.approx(float(enclosed_energy(0.8 * float(p.fwhm), float(p.fwhm))))
    # an extended source of surface brightness mu in an aperture of radius r carries mu x pi r^2
    e = compute_rates(img, tel, MK, Source(22.0, "R", kind="extended", extent_arcsec=2.0), seeing_500=0.6, airmass=1.1)
    tot = compute_rates(img, tel, MK, Source(22.0, "R"), seeing_500=0.6, airmass=1.1)
    assert float(e.signal) == pytest.approx(float(tot.signal) / float(tot.ap_frac) * np.pi * 4.0, rel=1e-9)
    spec = get_config("LRIS-R400")
    s = compute_rates(spec, tel, MK, Source(22.0, "r", kind="extended", extent_arcsec=3.0), seeing_500=0.6, airmass=1.1)
    assert float(s.npix) == pytest.approx(3.0 / spec.spatial_scale)
    assert float(s.slit_frac) == 1.0


def _rates(mag, cfg="LRIS-R400", band="r"):
    c = get_config(cfg)
    return c, compute_rates(c, TELESCOPES["keck1"], MK, Source(mag, band), seeing_500=0.6, airmass=1.2)


def test_plan_exposures_edge_cases():
    cfg, r = _rates(21.0)
    # goal already met: nothing to do
    p = plan_exposures(r, cfg, 10.0, snr_have=12.0)
    assert p.n_exp == 0 and p.wall_s == 0 and p.snr_final == 12.0
    # a single exposure shorter than the maximum, exactly reaching the goal
    p = plan_exposures(r, cfg, 10.0)
    assert p.n_exp == 1 and p.t_exp < cfg.max_exp_s and p.snr_final == pytest.approx(10.0, rel=1e-6)
    assert p.wall_s == pytest.approx(p.t_exp + cfg.readout_s)
    # several maximum-length exposures; S/N adds in quadrature
    p = plan_exposures(r, cfg, 50.0)
    assert p.limited_by == "max_exp" and p.n_exp >= 2 and p.snr_final >= 50.0 - 1e-6
    assert p.wall_s == pytest.approx(p.open_shutter_s + p.n_exp * cfg.readout_s)
    # saturation-limited: never longer than the linearity limit
    cfg, r = _rates(10.0)
    p = plan_exposures(r, cfg, 2000.0)
    assert p.limited_by == "saturation" and p.t_exp <= float(r.t_saturate(cfg.full_well)) + 1e-9
    assert p.snr_final >= 2000.0 - 1e-6
    # very faint: finite, honest, maximum-length exposures
    cfg, r = _rates(27.5)
    p = plan_exposures(r, cfg, 10.0)
    assert np.isfinite(p.wall_s) and p.n_exp > 1000 and p.t_exp == cfg.max_exp_s
    # a preferred length shorter than the maximum is respected
    cfg, r = _rates(21.0)
    p = plan_exposures(r, cfg, 40.0, t_pref=300.0)
    assert p.t_exp == pytest.approx(300.0) and p.snr_final >= 40.0 - 1e-6


def test_every_config_has_consistent_units():
    for key, cfg in all_configs().items():
        tel = TELESCOPES[cfg.telescopes[0]]
        r = compute_rates(cfg, tel, SITES[tel.site_key], Source(19.0, "V", "Vega"), seeing_500=0.7, airmass=1.3)
        assert np.isfinite(float(r.snr(600))) and float(r.peak_pix) > float(r.sky_pix) > 0, key
        assert 0 < float(r.slit_frac) <= 1 and 0 < float(r.ap_frac) <= 1, key
