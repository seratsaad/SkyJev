"""Sky brightness (dark sky, Moon, twilight), seeing and differential refraction against closed
forms and published values."""

import numpy as np
import pytest

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

# Krisciunas & Schaefer (1991) Table 2 (Moon zenith distance 60 deg, k=0.172; rho measured along the
# great circle through the Moon and the zenith, rho=60 is the zenith). Values in nanoLamberts.
KS91_TABLE2 = {30: {5: 7216, 30: 1160, 60: 530, 90: 437, 120: 818}}


LCO, MK = SITES["lco"], SITES["maunakea"]


@pytest.mark.parametrize("rho,expected", sorted(KS91_TABLE2[30].items()))
def test_ks91_reproduces_published_table(rho, expected):
    z_target = abs(60.0 - rho)
    b = float(ks91_moon_nl(30.0, rho, 60.0, z_target, 0.172))
    assert b == pytest.approx(expected, rel=0.05)


def test_sky_brighter_with_moon_and_twilight():
    s = SITES["lco"]
    dark = float(sky_ab(s, 5500, 70, 1.06, -40, -30, 180, 90))
    moon = float(sky_ab(s, 5500, 70, 1.06, -40, 45, 10, 40))
    twi = float(sky_ab(s, 5500, 70, 1.06, -13, -30, 180, 90))
    assert 21.0 < dark < 22.5
    assert moon < dark - 1.5
    assert twi < dark - 1.0


def test_seeing_scalings():
    assert float(fwhm_atm(0.6, 2.0, 5000, von_karman=False)) == pytest.approx(0.6 * 2**0.6)
    assert float(fwhm_atm(0.6, 1.0, 10000, von_karman=False)) == pytest.approx(0.6 * 2**-0.2)
    # von Karman makes a 6.5-10 m telescope's image sharper than the DIMM value (L0 = 25 m)
    assert float(fwhm_atm(0.62, 1.0, 5000)) < 0.62 * 0.85
    # Magellan: median DIMM 0.62" delivers ~0.57" (Magellan IQ studies)
    d = float(delivered_fwhm(0.62, 1.0, 5000, TELESCOPES["clay"].iq_floor_arcsec))
    assert 0.5 < d < 0.62


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
