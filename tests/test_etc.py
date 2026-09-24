"""Exposure time calculator: photon bookkeeping, PSF integrals, limits, and published anchors."""

import numpy as np
import pytest

from obsassist.astro.sites import SITES, TELESCOPES
from obsassist.etc import (
    Source,
    compute_rates,
    enclosed_energy,
    photons_per_A,
    plan_exposures,
    slit_fraction,
    strip_fraction,
    unit_scale,
)
from obsassist.instruments.base import all_configs, get_config


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


# official LCO ETC reruns (docs/research/magellan_sky_parameters.json, sensitivity_anchors): 1.0" slit,
# 0.6" Gaussian seeing, airmass 1.2, 3 x 1200 s, new Moon, S/N per unbinned pixel. Our ETC uses a Moffat
# PSF and is deliberately allowed to be up to ~35 % more conservative, never more optimistic.
LCO_ANCHORS = [
    ("LDSS3-VPHALL", "clay", 20, 6500, 42.5),
    ("IMACS-f2-300", "baade", 20, 6500, 43.2),
    ("MIKE-RED", "clay", 16, 6500, 43.2),
    ("MIKE-BLUE", "clay", 16, 4500, 37.7),
]


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
