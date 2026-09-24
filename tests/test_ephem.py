"""Ephemerides against astropy (both sites, four dates): alt/az, LST, twilight, Moon, parallactic
angle, precession, refraction, airmass, local time and coordinate formatting."""

import datetime as dt

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import FK5, AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time

from obsassist.astro import ephem as E
from obsassist.astro.ephem import (
    NightEphem,
    airmass_from_alt,
    fmt_dec,
    fmt_ra,
    lst_deg,
    parallactic_angle,
    parse_dec,
    parse_ra,
    precess_to_date,
    refraction_deg,
)
from obsassist.astro.sites import SITES, TELESCOPES, extra_el_limit

E._offline_astropy()


DATES = ("2026-01-15", "2026-06-21", "2026-09-24", "2026-12-10")


CASES = [(s, d) for s in ("maunakea", "lco") for d in DATES]


# declinations -85..+85 incl. near-zenith (site latitude) and circumpolar ones for both hemispheres
RAS = np.linspace(0.0, 348.0, 30)


DECS = np.array([-85, -75, -60, -45, -29.0, -20, -10, 0, 10, 19.8, 30, 45, 60, 75, 85] * 2, dtype=float)


def _loc(site):
    return EarthLocation(lat=site.lat_deg * u.deg, lon=site.lon_deg * u.deg, height=site.elev_m * u.m)


@pytest.fixture(scope="module")
def ephs():
    return {(s, d): NightEphem(SITES[s], d) for s, d in CASES}


@pytest.fixture(scope="module")
def eph():
    return NightEphem(SITES["maunakea"], "2026-09-24")


@pytest.mark.parametrize("site_key,date", CASES)
def test_target_altaz_vs_astropy(ephs, site_key, date):
    e = ephs[(site_key, date)]
    s = e.site
    tg = e.targets(RAS, DECS)
    idx = np.arange(0, e.n, 40)
    frame = AltAz(
        obstime=Time(e.jd[idx], format="jd"),
        location=_loc(s),
        pressure=s.pressure_hpa * u.hPa,
        temperature=s.temp_c * u.deg_C,
        relative_humidity=0.0,
        obswl=0.55 * u.um,
    )
    aa = SkyCoord(RAS[:, None] * u.deg, DECS[:, None] * u.deg).transform_to(frame[None, :])
    up = aa.alt.deg > 10
    assert up.sum() > 50
    dalt = np.abs(tg["alt"][:, idx] - aa.alt.deg)[up]
    daz = (((tg["az"][:, idx] - aa.az.deg) + 180) % 360 - 180) * np.cos(np.radians(aa.alt.deg))
    assert dalt.max() < 0.02
    assert np.abs(daz[up]).max() < 0.02
    # azimuth convention N=0, E=90: a rising target (HA < 0) is in the east, a setting one in the west
    east = (tg["ha"] < -1) & (tg["ha"] > -179) & (tg["alt"] > 0) & (np.abs(DECS[:, None] - s.lat_deg) < 60)
    assert (np.sin(np.radians(tg["az"][east])) > 0).all()


@pytest.mark.parametrize("site_key,date", CASES)
def test_lst_and_hour_angle(ephs, site_key, date):
    e = ephs[(site_key, date)]
    idx = np.arange(0, e.n, 60)
    ref = Time(e.jd[idx], format="jd").sidereal_time("mean", longitude=e.site.lon_deg * u.deg).deg
    d = ((e.lst[idx] - ref + 180) % 360 - 180) * 240.0  # seconds of time
    assert np.abs(d).max() < 0.5
    ha = e.targets(RAS, DECS)["ha"]
    assert ha.min() >= -180 and ha.max() < 180  # hour angle wrapped to [-180, 180)


@pytest.mark.parametrize("site_key,date", CASES)
def test_twilight_times_vs_astropy_root_finding(ephs, site_key, date):
    e = ephs[(site_key, date)]
    tw = e.twilight
    levels = {
        "sunset": -0.833,
        "civil_dusk": -6,
        "nautical_dusk": -12,
        "astro_dusk": -18,
        "astro_dawn": -18,
        "nautical_dawn": -12,
        "civil_dawn": -6,
        "sunrise": -0.833,
    }
    off = np.arange(-15, 16) / 1440.0
    jds = np.concatenate([getattr(tw, k) + off for k in levels])
    t = Time(jds, format="jd")
    alt = get_sun(t).transform_to(AltAz(obstime=t, location=_loc(e.site))).alt.deg.reshape(len(levels), -1)
    for (k, lev), a in zip(levels.items(), alt):
        f = a - lev
        c = np.nonzero(np.sign(f[:-1]) != np.sign(f[1:]))[0]
        assert len(c) == 1, k
        i = c[0]
        t_ref = getattr(tw, k) + off[i] + f[i] / (f[i] - f[i + 1]) * (off[1] - off[0])
        assert abs(getattr(tw, k) - t_ref) * 1440 < 1.0, k


@pytest.mark.parametrize("site_key,date", CASES)
def test_night_grid_is_the_local_evening(ephs, site_key, date):
    e = ephs[(site_key, date)]
    tw = e.twilight
    assert e.jd[0] == pytest.approx(tw.sunset - 20 / 1440, abs=1e-9)
    assert e.jd[-1] >= tw.sunrise + 20 / 1440 - 1e-9 and e.jd[-1] < tw.sunrise + 22 / 1440
    y, m, d = (int(x) for x in date.split("-"))
    evening = dt.date(y, m, d)
    ss = e.local(e.minute_of(tw.sunset))
    sr = e.local(e.minute_of(tw.sunrise))
    assert ss.date() == evening and 16 <= ss.hour <= 21  # sunset on the evening of `date`
    assert sr.date() == evening + dt.timedelta(days=1) and 5 <= sr.hour <= 8


def test_local_time_follows_chile_daylight_saving():
    """Local time at Las Campanas follows Chile's daylight saving: UTC-4 in winter, UTC-3 in summer."""
    e = NightEphem(SITES["lco"], "2026-06-21")
    assert e.local(0.0).utcoffset() == dt.timedelta(hours=-4)
    e2 = NightEphem(SITES["lco"], "2026-01-15")
    assert e2.local(0.0).utcoffset() == dt.timedelta(hours=-3)
    assert NightEphem(SITES["maunakea"], "2026-06-21").local(0.0).utcoffset() == dt.timedelta(hours=-10)


@pytest.mark.parametrize("site_key,date", CASES)
def test_moon_position_illumination_and_phase_angle(ephs, site_key, date):
    e = ephs[(site_key, date)]
    idx = np.arange(0, e.n, 45)
    t = Time(e.jd[idx], format="jd")
    loc = _loc(e.site)
    m = get_body("moon", t, loc).transform_to(AltAz(obstime=t, location=loc))
    assert np.abs(e.moon_alt[idx] - m.alt.deg).max() < 0.15
    daz = ((e.moon_az[idx] - m.az.deg + 180) % 360 - 180) * np.cos(m.alt.radian)
    assert np.abs(daz).max() < 0.15
    # astroplan's definition (geocentric): i = atan2(R sin(elong), D - R cos(elong)), k = (1 + cos i) / 2
    sun, moon = get_sun(t), get_body("moon", t)
    el = sun.separation(moon)
    i = np.arctan2(sun.distance * np.sin(el), moon.distance - sun.distance * np.cos(el)).to(u.deg).value
    assert np.abs(e.moon_illum[idx] - (1 + np.cos(np.radians(i))) / 2).max() < 0.012
    # KS91 phase angle: 0 = full, 180 = new (ours is topocentric: differs by < the lunar parallax)
    assert np.abs(e.moon_phase_angle[idx] - i).max() < 1.2
    assert ((e.moon_phase_angle < 90) == (e.moon_illum > 0.5)).all()


@pytest.mark.parametrize("site_key,date", [("maunakea", "2026-09-24"), ("lco", "2026-06-21")])
def test_moon_target_separation(ephs, site_key, date):
    e = ephs[(site_key, date)]
    tg = e.targets(RAS[:10], DECS[:10])
    idx = np.arange(0, e.n, 90)
    t = Time(e.jd[idx], format="jd")
    loc = _loc(e.site)
    frame = AltAz(obstime=t, location=loc)
    moon = get_body("moon", t, loc).transform_to(frame)
    for k in range(10):
        c = SkyCoord(RAS[k] * u.deg, DECS[k] * u.deg).transform_to(frame)
        assert np.abs(tg["moon_sep"][k, idx] - c.separation(moon).deg).max() < 0.2


def test_parallactic_angle_vs_astropy_zenith_position_angle(ephs):
    """q = PA (E of N, north of date) of the zenith seen from the target; positive west of the
    meridian in the north. Reference: astropy, a point 1' toward the zenith, in FK5 of date."""
    for key in ("maunakea", "lco"):
        e = ephs[(key, "2026-09-24")]
        loc = _loc(e.site)
        tg = e.targets(RAS, DECS)
        for j in (60, 300, 540):
            tt = Time(e.jd[j], format="jd")
            frame = AltAz(obstime=tt, location=loc)
            ok = (tg["alt"][:, j] > 15) & (tg["alt"][:, j] < 80)
            c = SkyCoord(RAS[ok] * u.deg, DECS[ok] * u.deg).transform_to(frame)
            up = SkyCoord(alt=c.alt + 1 * u.arcmin, az=c.az, frame=frame)
            fk = FK5(equinox=tt)
            q = c.transform_to(fk).position_angle(up.transform_to(fk)).deg
            d = (tg["parallactic"][ok, j] - q + 180) % 360 - 180
            assert np.abs(d).max() < 0.1
    # sign convention: west of the meridian (HA > 0) at a northern site, south of zenith -> q > 0
    assert parallactic_angle(30.0, 0.0, 19.8) > 0 and parallactic_angle(-30.0, 0.0, 19.8) < 0
    assert parallactic_angle(0.0, 0.0, 19.8) == pytest.approx(0.0)
    assert abs(parallactic_angle(0.0, 40.0, 19.8)) == pytest.approx(180.0)  # north of zenith


def test_point_agrees_with_targets(ephs):
    e = ephs[("lco", "2026-12-10")]
    tg = e.targets(RAS, DECS)
    for j in (0, 200, e.n - 1):
        for k in (0, 7, 22):
            p = e.point(RAS[k], DECS[k], e.jd[j])
            assert p["alt"] == pytest.approx(tg["alt"][k, j], abs=1e-4)
            assert (p["az"] - tg["az"][k, j] + 180) % 360 - 180 == pytest.approx(0, abs=1e-3)
            assert p["ha"] == pytest.approx(tg["ha"][k, j], abs=1e-4)
            assert p["parallactic"] == pytest.approx(tg["parallactic"][k, j], abs=1e-3)
            assert p["lst"] == pytest.approx(e.lst[j], abs=1e-9)


def test_precession_to_date_both_paths(monkeypatch):
    ra, dec = np.array([0.0, 83.6, 201.3, 279.2]), np.array([0.0, 22.0, -43.0, 38.8])
    jd = Time("2026-09-25T08:00:00").jd
    ref = SkyCoord(ra * u.deg, dec * u.deg).transform_to(FK5(equinox=Time(jd, format="jd")))
    r1, d1 = precess_to_date(ra, dec, jd)
    # the fallback (IAU 1976) used when astropy is not importable
    monkeypatch.setattr(E, "_offline_astropy", lambda: (_ for _ in ()).throw(ImportError("no astropy")))
    r2, d2 = precess_to_date(ra, dec, jd)
    for r, d, tol in ((r1, d1, 1e-6), (r2, d2, 1.0 / 3600)):
        assert np.abs(((r - ref.ra.deg + 180) % 360 - 180) * np.cos(np.radians(dec))).max() < tol
        assert np.abs(d - ref.dec.deg).max() < tol
    # 26.7 yr of general precession moves RA by ~ 0.37 deg at the equator
    assert 0.3 < (r1[0] - 0.0) < 0.45


def test_refraction_and_pressure_temperature_scaling():
    """Saemundsson (1986) vs ERFA (astropy): within 6 % from 10 to 80 deg; the P/T scaling
    matches ERFA's to 1 %."""
    s = SITES["maunakea"]
    loc = _loc(s)
    t = Time("2026-09-25T08:00:00")
    alts = np.array([10.0, 20.0, 30.0, 45.0, 60.0, 80.0])
    true = SkyCoord(alt=alts * u.deg, az=np.full(6, 100.0) * u.deg, frame=AltAz(obstime=t, location=loc))
    ref = {}
    for P, T in ((615.0, 1.5), (1010.0, 10.0)):
        f = AltAz(
            obstime=t,
            location=loc,
            pressure=P * u.hPa,
            temperature=T * u.deg_C,
            relative_humidity=0.0,
            obswl=0.55 * u.um,
        )
        ref[P] = true.transform_to(f).alt.deg - alts
        ours = refraction_deg(alts, P, T)
        assert np.all(np.abs(ours / ref[P] - 1) < 0.06)
    ratio_ours = refraction_deg(alts, 615.0, 1.5) / refraction_deg(alts, 1010.0, 10.0)
    assert np.all(np.abs(ratio_ours / (ref[615.0] / ref[1010.0]) - 1) < 0.01)
    assert refraction_deg(-2.0) == 0.0 and refraction_deg(90.0) < 1e-5


def test_airmass_kasten_young():
    # Kasten & Young (1989) published values; secz at high altitude
    for h, x in ((90.0, 1.0), (60.0, 1.1547), (30.0, 1.9942), (10.0, 5.586), (5.0, 10.316), (1.0, 26.31)):
        assert float(airmass_from_alt(h)) == pytest.approx(x, rel=2e-3)
    h = np.array([40.0, 60.0, 80.0])
    assert np.allclose(airmass_from_alt(h), 1 / np.sin(np.radians(h)), rtol=3e-3)
    assert np.all(np.diff(airmass_from_alt(np.linspace(1, 90, 200))) < 0)
    assert float(airmass_from_alt(-5.0)) == 40.0


def test_sexagesimal_formatting_never_shows_60_seconds():
    """Coordinates round before splitting into h/m/s, so 60 seconds never appears."""
    assert fmt_ra(15 * (1 + 59 / 60 + 59.9999 / 3600)) == "02:00:00.00"
    assert fmt_ra(359.99999999) == "00:00:00.00"
    assert fmt_dec(-(10 + 59 / 60 + 59.99 / 3600)) == "-11:00:00.0"
    assert fmt_dec(89.99999999) == "+90:00:00.0"
    assert fmt_dec(-1e-7) == "+00:00:00.0"
    rng = np.random.default_rng(3)
    for ra, dec in zip(rng.uniform(0, 360, 300), rng.uniform(-90, 90, 300)):
        r, d = fmt_ra(ra), fmt_dec(dec)
        assert len(r) == 11 and len(d) == 11
        assert float(r.split(":")[2]) < 60 and float(d.split(":")[2]) < 60
        assert (parse_ra(r) - ra + 180) % 360 - 180 == pytest.approx(0, abs=0.006 * 15 / 3600 + 1e-9)
        assert parse_dec(d) == pytest.approx(dec, abs=0.051 / 3600)


def test_lst_function_matches_gmst_plus_longitude():
    jd = 2461308.5
    assert lst_deg(jd, -70.6917) == pytest.approx((E.gmst_deg(jd) - 70.6917) % 360)


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
