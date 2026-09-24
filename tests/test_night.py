"""Weather (ranges, medians, dome rules) and the night model (exposure integrals, limits, window
QC, overheads, saturation), and the program generator."""

import numpy as np
import pytest

from obsassist.astro.ephem import NightEphem
from obsassist.astro.sites import SITES, extra_el_limit
from obsassist.planning.planner import ListOrderPolicy, run_policy
from obsassist.programs.generator import TEMPLATE_SED, generate_program
from obsassist.sim.night import DEFOCUS_LEVELS, NightModel
from obsassist.targets import builtin_programs, load_program
from obsassist.weather import generate


def _brute(m, i, t0, t1, d):
    """Integrals by 1-s summation of the per-minute rates (independent of the cumulative sums)."""
    tt = t0 + (np.arange(int(round((t1 - t0) * 60))) + 0.5) / 60.0
    j = np.floor(tt / m.dt).astype(int)
    k = int(np.clip(np.searchsorted(DEFOCUS_LEVELS, d) - 1, 0, len(DEFOCUS_LEVELS) - 2))
    f = (d - DEFOCUS_LEVELS[k]) / (DEFOCUS_LEVELS[k + 1] - DEFOCUS_LEVELS[k])
    mix = lambda a: ((1 - f) * a[i, k, j] + f * a[i, k + 1, j])
    return mix(m.sig).sum(), mix(m.var).sum(), mix(m.npix).mean(), mix(m.peak).sum()


# ----------------------------------------------------------------------------- weather
@pytest.mark.parametrize("key", ["lco", "maunakea"])
def test_weather_ranges_reproducibility_and_median(key):
    site = SITES[key]
    t = np.arange(0.0, 720.0)
    a, b = generate(site, t, seed=7), generate(site, t, seed=7)
    for f in ("seeing", "seeing_dimm", "cloud_mag", "humidity", "wind_ms", "temp_c", "dome_ok"):
        assert np.array_equal(getattr(a, f), getattr(b, f), equal_nan=f != "dome_ok")
    assert not np.array_equal(a.seeing, generate(site, t, seed=8).seeing)
    for sd in range(20):
        w = generate(site, t, seed=sd)
        assert (w.seeing >= 0.25).all() and (w.seeing <= 3.5).all()
        assert (w.cloud_mag >= 0).all() and (w.cloud_mag <= 5).all()
        assert (w.humidity >= 2).all() and (w.humidity <= 100).all()
        assert (w.wind_ms >= 0).all() and (w.wind_gust_ms >= w.wind_ms).all()
        assert (w.dewpoint_c <= w.temp_c + 1e-9).all()
    tt = np.arange(0.0, 600.0, 10.0)
    med = np.median([np.median(generate(site, tt, seed=s).seeing) for s in range(150)])
    assert med == pytest.approx(site.seeing_median, rel=0.12)


def test_dome_closes_at_limits_and_reopens_after_the_wait():
    site = SITES["lco"]
    L = site.limits
    t = np.arange(0.0, 720.0)
    n_closures = 0
    for sd in range(30):
        w = generate(site, t, seed=sd, fog_event=True)
        bad = (
            (w.humidity >= L.humidity_close)
            | (w.wind_ms >= L.wind_close_ms)
            | ((w.temp_c - w.dewpoint_c) < L.dewpoint_margin_c)
        )
        good = (
            (w.humidity < L.humidity_open)
            & (w.wind_ms < 0.9 * L.wind_close_ms)
            & ((w.temp_c - w.dewpoint_c) >= L.dewpoint_margin_c + 1)
        )
        assert not (w.dome_ok & bad).any()  # never open while a limit is exceeded
        for a, b in w.closed_intervals():
            n_closures += 1
            ia, ib = int(a), int(b)
            assert bad[ia]  # closes because of a limit
            if ib < len(t) - 1:  # reopens only after 30 good minutes
                assert good[ib - int(L.reopen_wait_min) + 1 : ib + 1].all()
    assert n_closures >= 10


@pytest.mark.parametrize("d", [0.0, 0.1, 0.3, 0.9])
def test_exposure_integrals_match_brute_force(lris, d):
    m = lris
    i = m.program.target_index("SN 2026aer")
    ok = np.nonzero(m.observable[i] & m.dome_ok)[0]
    t0 = ok[len(ok) // 3] * m.dt + 0.37
    out = m.exposure(i, t0, 900.0, defocus=d)
    assert out.truncated is None and out.t_exp_s == pytest.approx(900.0)
    S, V, npix, peak = _brute(m, i, t0, t0 + 15.0, d)
    assert out.signal_e == pytest.approx(S, rel=2e-4)
    assert out.snr == pytest.approx(m.unit[i] * S / np.sqrt(V + npix * m.rn2[i]), rel=2e-4)
    assert out.peak_e == pytest.approx(peak, rel=2e-4)  # integrated and defocus-interpolated
    seg = m.segment(i, t0, t0 + 15.0, d)
    assert seg["S"] == pytest.approx(out.signal_e) and seg["peak_e"] == pytest.approx(out.peak_e)


def test_defocus_interpolation_is_continuous_and_monotonic(lris):
    m = lris
    i = m.program.target_index("SN 2026aer")
    t0 = np.nonzero(m.observable[i])[0][100] * m.dt
    ds = np.linspace(0, DEFOCUS_LEVELS[-1], 46)
    s = np.array([m.segment(i, t0, t0 + 10, d)["S"] for d in ds])
    fw = np.array([m.segment(i, t0, t0 + 10, d)["fwhm"] for d in ds])
    assert np.all(np.diff(s) <= 1e-9) and np.all(np.diff(fw) >= -1e-12)
    for lev_k, lev in enumerate(DEFOCUS_LEVELS):  # exact at the precomputed levels
        assert m.segment(i, t0, t0 + 10, lev)["S"] == pytest.approx(m._integral(m.C_sig[i, lev_k], t0, t0 + 10))


def test_segment_is_additive(lris):
    i = lris.program.target_index("QSO J0056+2513")
    t0 = lris.t_start + 97.3
    a, b, ab = lris.segment(i, t0, t0 + 13.1), lris.segment(i, t0 + 13.1, t0 + 40), lris.segment(i, t0, t0 + 40)
    for k in ("S", "V", "sky_e_pix", "peak_e"):
        assert a[k] + b[k] == pytest.approx(ab[k], rel=1e-9)
    assert (a["fwhm"] * a["dur_s"] + b["fwhm"] * b["dur_s"]) / ab["dur_s"] == pytest.approx(ab["fwhm"], rel=1e-9)


def test_truncation_at_target_limit_dome_and_night_end(lris):
    m = lris
    i = m.program.target_index("SN 2026aer")
    obs = m.observable[i]
    # the target leaves its limits: truncated at the first minute it is out
    last = np.nonzero(obs)[0]
    edge = next(j for j in last if not obs[j + 1])
    assert (edge + 1) * m.dt < m.t_end  # a real limit, not the end of the night
    out = m.exposure(i, edge * m.dt - 5.0, 1800.0)
    assert out.truncated == "target limit" and out.t_exp_s == pytest.approx(6 * 60.0, abs=1e-6)
    # end of night
    j = int(m.t_end / m.dt) - 3
    k = next(k for k in range(m.N) if m.observable[k, j] and not m.targets[k].twilight_ok)
    out = m.exposure(k, m.t_end - 2.0, 900.0)
    assert out.truncated == "end of night" and out.t_exp_s == pytest.approx(120.0, abs=1e-6)
    # dome closure: a copy of the night with the dome closing mid-exposure
    t0 = np.nonzero(obs)[0][50] * m.dt
    saved = m.dome_ok.copy()
    try:
        m.dome_ok[int(t0 / m.dt) + 4 :] = False
        out = m.exposure(i, t0 + 0.5, 900.0)
        assert out.truncated == "dome closed" and not out.qc_pass and out.snr_counted == 0
        assert out.t_exp_s == pytest.approx(3.5 * 60)
    finally:
        m.dome_ok[:] = saved


def test_window_qc(lris):
    i = lris.program.target_index("WD J1946+2957")
    w0, w1 = lris.window[i]
    assert lris.exposure(i, w0 + 1, 300).qc_pass
    half_out = lris.exposure(i, w0 - 2.4, 300)  # 48 % outside: still counts
    assert "outside time window" not in half_out.qc_reasons
    mostly_out = lris.exposure(i, w0 - 2.6, 300)  # 52 % outside: rejected
    assert "outside time window" in mostly_out.qc_reasons
    assert "outside time window" in lris.exposure(i, w1 + 1, 300).qc_reasons


def test_keck_deck_limits_and_wind_restriction_applied(lris):
    m = lris
    az, alt = m.geo["az"], m.geo["alt"]
    lim = extra_el_limit(m.tel, az)
    deck = (az >= 5.3) & (az <= 146.2)
    assert (lim[deck] == 33.3).all() and (lim[~deck] == 18.0).all()
    assert not (m.observable_geo & (alt < lim)).any()
    assert not (m.observable_geo & deck & (alt < 33.3)).any()
    # somewhere a target is between 18 and 33.3 deg in the deck zone: must be unobservable there
    zone = deck & (alt > 20) & (alt < 33) & (m.geo["airmass"] < 2.5)
    assert zone.any() and not m.observable_geo[zone].any()
    # high wind: never point within wind_avoid_deg of the wind direction
    L = m.site.limits
    high = m.weather.wind_ms >= L.wind_high_ms
    daz = np.abs((az - m.weather.wind_dir[None, :] + 180) % 360 - 180)
    assert not (m.observable & high[None, :] & (daz < L.wind_avoid_deg)).any()


def test_observe_overheads_slew_setup_and_position(lris):
    m = lris
    s = m.initial_state()
    s.t = m.t_start + 30
    # shortest azimuth path, both directions across north
    assert m.slew_seconds(45, 350, 45, 10) == pytest.approx(20 / m.tel.slew_az_dps + m.tel.settle_s)
    assert m.slew_seconds(45, 10, 45, 350) == pytest.approx(20 / m.tel.slew_az_dps + m.tel.settle_s)
    assert m.slew_seconds(30, 100, 80, 100) == pytest.approx(50 / m.tel.slew_el_dps + m.tel.settle_s)
    r = [k for k, t in enumerate(m.targets) if t.config == "LRIS-R400" and m.is_visible(k, s.t)]
    blue = next(k for k, t in enumerate(m.targets) if t.config == "LRIS-B600")
    s1, out = m.observe(s, r[0], 600)
    assert s1.setup == m.targets[r[0]].cfg().setup_id
    oh_same, parts = m.overhead_to(s1, blue)
    assert parts["setup"] == 0.0  # LRIS-B600 and LRIS-R400 share the dichroic setup
    s_other = s1.copy()
    s_other.setup = "something-else"
    assert m.overhead_to(s_other, blue)[1]["setup"] == m.targets[blue].cfg().config_change_s
    assert m.overhead_to(s1, r[0])[0] == 0.0  # continuing on the acquired target is free
    # the telescope ends the exposure where the target is when the shutter closes (it tracked it)
    j_end = m.idx(out.t_start + out.t_exp_s / 60.0)
    assert (s1.tel_alt, s1.tel_az) == (
        pytest.approx(m.geo["alt"][r[0], j_end]),
        pytest.approx(m.geo["az"][r[0], j_end]),
    )
    assert s1.t == pytest.approx(out.t_end) and out.t_end == pytest.approx(out.t_start + (600 + 54) / 60)


def test_saturation_uses_the_integrated_peak(lris):
    m = lris
    i = m.program.target_index("BD+28 4211")
    j = np.nonzero(m.observable[i])[0][5]
    t_sat = 0.8 * m.targets[i].cfg().full_well / m.peak[i, 0, j]
    assert not m.exposure(i, j * m.dt, 0.9 * t_sat).saturated
    assert m.exposure(i, j * m.dt, 1.2 * t_sat).saturated


def test_high_wind_blocks_pointing_into_the_wind():
    prog = load_program(builtin_programs()["clay_mike_darktime"])
    e = NightEphem(prog.site, prog.date)
    w = generate(prog.site, e.t_min, seed=2, clear=True, fog_event=False)
    w.wind_ms[:] = prog.site.limits.wind_high_ms + 1.0  # high, but below the closure limit
    w.wind_dir[:] = 180.0
    w.dome_ok[:] = True
    m = NightModel(prog, seed=2, weather=w)
    daz = np.abs((m.geo["az"] - 180.0 + 180) % 360 - 180)
    into = daz < prog.site.limits.wind_avoid_deg
    assert (m.observable_geo & into).any()  # some pointings are geometrically fine...
    assert not (m.observable & into).any()  # ...but blocked by the wind
    assert (m.observable == (m.observable_geo & ~into)).all()


def test_weather_reproducible_and_bounded():
    from obsassist.astro.sites import SITES

    t = np.arange(0, 700.0)
    a = generate(SITES["lco"], t, seed=11)
    b = generate(SITES["lco"], t, seed=11)
    assert np.array_equal(a.seeing, b.seeing) and np.array_equal(a.dome_ok, b.dome_ok)
    assert (a.seeing > 0.2).all() and (a.seeing < 4).all()
    assert (a.humidity >= 0).all() and (a.humidity <= 100).all()


def test_weather_median_matches_site():
    from obsassist.astro.sites import SITES

    t = np.arange(0, 600.0, 10.0)
    med = np.median([np.median(generate(SITES["lco"], t, seed=s).seeing) for s in range(200)])
    assert 0.5 < med < 0.8  # LCO DIMM median 0.62"


def test_exposure_is_additive_in_time(mike):
    i = 1
    t0 = mike.t_start + 120
    a = mike.segment(i, t0, t0 + 20)
    b = mike.segment(i, t0 + 20, t0 + 40)
    ab = mike.segment(i, t0, t0 + 40)
    assert a["S"] + b["S"] == pytest.approx(ab["S"], rel=1e-6)


def test_exposure_truncated_at_limits(lris):
    # AT 2026cvn sits next to the full Moon all night: never observable
    i = lris.program.target_index("AT 2026cvn")
    assert not lris.observable[i].any()
    out = lris.exposure(i, lris.t_start + 60, 900)
    assert out.snr_counted == 0 and out.t_exp_s == 0


def test_window_target_only_counts_inside_window(lris):
    i = lris.program.target_index("WD J1946+2957")
    w0, w1 = lris.window[i]
    inside = lris.exposure(i, w0 + 1, 300)
    before = lris.exposure(i, w0 - 60, 300)
    assert inside.qc_pass and not before.qc_pass


@pytest.mark.parametrize(
    "tel,date,seed", [("clay", "2026-03-12", 1), ("keck1", "2026-07-30", 2), ("keck2", "2026-11-05", 3)]
)
def test_generator_programs_are_valid(tel, date, seed):
    p = generate_program(tel, date, seed=seed)
    m = NightModel(p, seed=seed)
    assert len(p.targets) >= 8
    assert m.observable.any(axis=1).sum() >= len(p.targets) // 2
    s, log = run_policy(m, ListOrderPolicy())
    assert 0 <= m.score(s) <= m.max_score()
    # continuum colour follows the template (cool dwarfs are red, hot stars blue)
    for t in p.targets:
        assert t.source.sed == TEMPLATE_SED.get(t.template, "flat_fnu")
