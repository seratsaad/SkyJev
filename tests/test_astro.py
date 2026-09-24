"""Ephemerides, sky brightness and seeing against references."""

import numpy as np
import pytest

from obsassist.astro.ephem import NightEphem, airmass_from_alt, fmt_dec, fmt_ra, parse_dec, parse_ra
from obsassist.astro.sites import SITES, TELESCOPES, extra_el_limit
from obsassist.astro.sky import delivered_fwhm, fwhm_atm, ks91_moon_nl, sky_ab


@pytest.fixture(scope="module")
def eph():
    return NightEphem(SITES["maunakea"], "2026-09-24")


def test_fast_altaz_matches_astropy(eph):
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    ra, dec = [18.0, 330.0, 45.0, 120.0], [20.0, -5.0, 60.0, -30.0]
    tg = eph.targets(ra, dec)
    s = eph.site
    loc = EarthLocation(lat=s.lat_deg * u.deg, lon=s.lon_deg * u.deg, height=s.elev_m * u.m)
    idx = np.arange(0, eph.n, 45)
    frame = AltAz(
        obstime=Time(eph.jd[idx], format="jd"),
        location=loc,
        pressure=s.pressure_hpa * u.hPa,
        temperature=s.temp_c * u.deg_C,
        relative_humidity=0.2,
        obswl=0.55 * u.um,
    )
    for k in range(len(ra)):
        aa = SkyCoord(ra[k] * u.deg, dec[k] * u.deg).transform_to(frame)
        up = aa.alt.deg > 15
        if not up.any():
            continue
        assert np.abs(tg["alt"][k, idx][up] - aa.alt.deg[up]).max() < 0.02
        daz = ((tg["az"][k, idx][up] - aa.az.deg[up] + 180) % 360) - 180
        assert np.abs(daz * np.cos(np.radians(aa.alt.deg[up]))).max() < 0.02


def test_twilight_order_and_night_length(eph):
    tw = eph.twilight
    assert tw.sunset < tw.civil_dusk < tw.nautical_dusk < tw.astro_dusk < tw.astro_dawn < tw.nautical_dawn < tw.sunrise
    hours = (tw.nautical_dawn - tw.nautical_dusk) * 24
    assert 9.5 < hours < 11.5  # late-September night at 20 deg N


def test_airmass_limits():
    assert airmass_from_alt(90.0) == pytest.approx(1.0, abs=1e-3)
    assert airmass_from_alt(30.0) == pytest.approx(1.995, abs=0.01)
    assert airmass_from_alt(10.0) > 5.0


def test_keck_deck_limits_documented():
    k1, k2 = TELESCOPES["keck1"], TELESCOPES["keck2"]
    assert extra_el_limit(k1, 90.0) == pytest.approx(33.3)
    assert extra_el_limit(k1, 200.0) == pytest.approx(18.0)
    assert extra_el_limit(k2, 250.0) == pytest.approx(36.8)


def test_coordinate_round_trip():
    for ra, dec in [(0.0, 0.0), (123.456, -45.678), (359.9, 89.5)]:
        assert parse_ra(fmt_ra(ra)) == pytest.approx(ra, abs=1e-3)
        assert parse_dec(fmt_dec(dec)) == pytest.approx(dec, abs=1e-3)


# Krisciunas & Schaefer (1991) Table 2 (Moon zenith distance 60 deg, k=0.172; rho measured along the
# great circle through the Moon and the zenith, rho=60 is the zenith). Values in nanoLamberts.
KS91_TABLE2 = {30: {5: 7216, 30: 1160, 60: 530, 90: 437, 120: 818}}


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
