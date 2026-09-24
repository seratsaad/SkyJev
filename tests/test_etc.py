"""Exposure time calculator: photon bookkeeping, PSF integrals, S/N units, exposure plans,
saturation, dispersion and the official LCO ETC anchors."""

import numpy as np
import pytest
from scipy import integrate

from obsassist.astro.sites import SITES, TELESCOPES
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

# official LCO ETC reruns (docs/research/magellan_sky_parameters.json, sensitivity_anchors): 1.0" slit,
# 0.6" Gaussian seeing, airmass 1.2, 3 x 1200 s, new Moon, S/N per unbinned pixel. Our ETC uses a Moffat
# PSF and is deliberately allowed to be up to ~35 % more conservative, never more optimistic.
LCO_ANCHORS = [
    ("LDSS3-VPHALL", "clay", 20, 6500, 42.5),
    ("IMACS-f2-300", "baade", 20, 6500, 43.2),
    ("MIKE-RED", "clay", 16, 6500, 43.2),
    ("MIKE-BLUE", "clay", 16, 4500, 37.7),
]


LCO, MK = SITES["lco"], SITES["maunakea"]


def _rates(mag, cfg="LRIS-R400", band="r"):
    c = get_config(cfg)
    return c, compute_rates(c, TELESCOPES["keck1"], MK, Source(mag, band), seeing_500=0.6, airmass=1.2)


def test_ab_zero_photon_rate():
    # an AB = 0 source gives ~1000 photons s^-1 cm^-2 A^-1 at 5500 A
    assert float(photons_per_A(0.0, 5500.0)) == pytest.approx(996, rel=0.01)


def test_moffat_integrals_are_normalised():
    assert float(strip_fraction(-50, 50, 0.7)) == pytest.approx(1.0, abs=1e-4)
    assert float(enclosed_energy(50, 0.7)) == pytest.approx(1.0, abs=1e-3)
    assert float(slit_fraction(1.0, 0.6)) < float(slit_fraction(1.0, 0.4)) < 1.0
    # a centroid offset loses light
    assert float(slit_fraction(1.0, 0.6, 0.3)) < float(slit_fraction(1.0, 0.6))


def test_snr_monotonic_and_sqrt_n():
    cfg, tel, site = get_config("LRIS-R400"), TELESCOPES["keck1"], SITES["maunakea"]
    r = compute_rates(cfg, tel, site, Source(21.0, "r"), seeing_500=0.6, airmass=1.2)
    s = [float(r.snr(t)) for t in (60, 300, 1200)]
    assert s[0] < s[1] < s[2]
    assert float(r.snr(600, 4)) == pytest.approx(2 * float(r.snr(600)), rel=1e-9)
    worse = compute_rates(cfg, tel, site, Source(21.0, "r"), seeing_500=1.4, airmass=1.8, cloud_mag=0.5)
    assert float(worse.snr(1200)) < s[2]


def test_every_config_computes():
    for key, cfg in all_configs().items():
        tel = TELESCOPES[cfg.telescopes[0]]
        site = SITES[tel.site_key]
        r = compute_rates(cfg, tel, site, Source(18.0, "V", "Vega"), seeing_500=0.7, airmass=1.3)
        assert np.isfinite(float(r.snr(600))) and float(r.snr(600)) > 0, key


def test_saturation_shortens_exposures():
    cfg = get_config("MIKE-RED")
    r = compute_rates(cfg, TELESCOPES["clay"], SITES["lco"], Source(5.5, "V", "Vega"), seeing_500=0.5, airmass=1.0)
    p = plan_exposures(r, cfg, goal=200)
    assert p.t_exp <= float(r.t_saturate(cfg.full_well)) + 1e-6


@pytest.mark.parametrize("key,telkey,mag,lam,ref", LCO_ANCHORS)
def test_lco_etc_anchors(key, telkey, mag, lam, ref):
    cfg = get_config(key).with_(bin_spatial=1, bin_spectral=1)
    t = TELESCOPES[telkey]
    t0 = t.__class__(**{**t.__dict__, "iq_floor_arcsec": 0.0})
    s500 = 0.6 / (1.2**0.6 * (lam / 5000) ** -0.2)
    r = compute_rates(
        cfg,
        t0,
        SITES["lco"],
        Source(mag, "r"),
        seeing_500=s500,
        airmass=1.2,
        lam=lam,
        unit="per_pix",
        von_karman=False,
        moon_alt=-30,
        sun_alt=-40,
    )
    ours = float(r.snr(1200, 3))
    assert 0.65 * ref < ours < 1.05 * ref


def test_echelle_dispersion_scales_with_wavelength():
    # an echelle order spans a fixed velocity per pixel, so A/pixel grows with lam; a grating's does not
    mike, lris = get_config("MIKE-RED"), get_config("LRIS-R400")
    assert mike.echelle and not lris.echelle
    assert float(mike.dlam_at(2 * mike.lam_ref)) == pytest.approx(2 * mike.dlam_pix)
    assert float(lris.dlam_at(lris.lam_ref + 1000)) == pytest.approx(lris.dlam_pix)
    # hence the same number of pixels per resolution element in every order
    assert unit_scale(mike, "per_resel", 5500) == pytest.approx(unit_scale(mike, "per_resel", 8500))


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
