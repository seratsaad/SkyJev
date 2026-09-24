"""Audit: planner candidates, projection, the oracle's guarantees (pilot >= greedy, the upper
bound is a bound), the batch CLI and the catalog exports."""

import json

import numpy as np
import pytest

from obsassist.astro.ephem import NightEphem, parse_dec, parse_ra
from obsassist.planning import oracle as O
from obsassist.planning.catalogs import keck_starlist, magellan_catalog
from obsassist.planning.planner import GreedyPolicy, Nowcast, _exp_len, candidates, project, run_policy
from obsassist.programs.generator import generate_program
from obsassist.sim.night import NightModel
from obsassist.targets import builtin_programs, load_program
from obsassist.weather import generate


@pytest.fixture(scope="module")
def lris():
    return NightModel(load_program(builtin_programs()["keck1_lris_tonight"]), seed=6)


def test_candidates_notes_and_feasibility(lris):
    m = lris
    s = m.initial_state()
    s.t = m.t_start + 30
    c = {x.name: x for x in candidates(m, s, Nowcast.from_history(m, s.t))}
    assert not c["AT 2026cvn"].feasible and c["AT 2026cvn"].note  # next to the Moon all night
    assert not c["TDE 2026sep"].feasible and "announced" in c["TDE 2026sep"].note
    wd = c["WD J1946+2957"]
    assert not wd.feasible and "window opens" in wd.note
    for x in c.values():
        if x.feasible:
            assert m.is_visible(x.i, s.t + x.overhead_min) and x.window_left_min > 0 and x.n_needed >= 1
            assert x.time_needed_min >= x.overhead_min + x.n_needed * x.t_exp / 60
    # a target that is done says so
    s2 = s.copy()
    k = m.program.target_index("SN 2026aer")
    s2.snr2[k] = m.goals[k] ** 2
    assert {x.name: x for x in candidates(m, s2, Nowcast.from_history(m, s.t))}["SN 2026aer"].note == "done"


def test_exposure_length_is_saturation_aware(lris):
    m = lris
    i = m.program.target_index("BD+28 4211")  # V = 10.5 standard, preferred 30 s
    j = np.nonzero(m.observable[i])[0][0]
    t = _exp_len(m, i, j * m.dt, 0.6)
    peak = m.P_peak[i, :, j].max()
    assert t <= m.targets[i].t_exp
    assert (
        t * np.interp(np.log(0.6), np.log([0.3, 0.45, 0.6]), m.P_peak[i, :3, j]) <= 0.8 * m.targets[i].cfg().full_well
    )
    q = m.program.target_index("QSO J0056+2513")  # faint, dark sky: the preferred length
    jq = np.nonzero(m.observable[q] & (m.ephem.sun_alt < -18))[0][10]
    assert _exp_len(m, q, jq * m.dt, 0.6) == m.targets[q].t_exp
    # ...but in bright twilight the sky alone saturates a long exposure, and the planner knows it
    assert _exp_len(m, q, j * m.dt, 0.6) < m.targets[q].t_exp
    assert peak > 0


def test_projection_time_is_monotonic(lris):
    m = lris
    s = m.initial_state()
    p = project(m, s, Nowcast.from_history(m, s.t))
    assert p.blocks
    for a, b in zip(p.blocks, p.blocks[1:]):
        assert a.start < a.end <= b.start + 1e-9
    assert all(v >= 0 for v in p.final_snr.values())
    assert json.dumps(p.to_dict(m))  # plain Python types (no numpy bools)


@pytest.mark.parametrize(
    "tel,date,seed", [("clay", "2026-06-21", 4), ("keck1", "2026-01-15", 7), ("keck2", "2026-09-24", 11)]
)
def test_pilot_between_greedy_and_upper_bound(tel, date, seed):
    m = NightModel(generate_program(tel, date, seed=seed), seed=seed)
    g, _ = run_policy(m, GreedyPolicy())
    r = O.pilot(m, time_budget_s=2.0)
    assert r.greedy_score == pytest.approx(m.score(g))
    assert m.score(g) - 1e-9 <= r.score <= r.upper_bound + 1e-9 <= m.max_score() + 1e-9
    assert m.score(r.state) == pytest.approx(r.score)


def test_upper_bound_counts_twilight_time():
    """Found in audit: the bound's capacity ignored the civil-twilight time in which twilight_ok
    targets are observed, so a night whose dome is open only in twilight had bound 0 while the
    standard was observed there for full credit."""
    p = load_program(builtin_programs()["keck1_lris_tonight"])
    e = NightEphem(p.site, p.date)
    w = generate(p.site, e.t_min, seed=1, clear=True, fog_event=False)
    m0 = NightModel(p, seed=1, weather=w)
    w.dome_ok[:] = True
    w.dome_ok[(e.t_min >= m0.t_start) & (e.t_min <= m0.t_end)] = False  # open only in twilight
    m = NightModel(p, seed=1, weather=w)
    g, _ = run_policy(m, GreedyPolicy())
    assert m.score(g) > 0
    assert O.upper_bound(m) >= m.score(g)


def test_batch_cli_runs(tmp_path, monkeypatch):
    from obsassist.planning import batch

    real = O.pilot
    monkeypatch.setattr(O, "pilot", lambda m, width=1: real(m, width=width, time_budget_s=1.0))
    out = tmp_path / "o.jsonl"
    batch.main(["--random", "baade", "--seeds", "3", "--out", str(out)])
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["seed"] == 3 and rec["greedy"] <= rec["oracle"] + 1e-9 <= rec["upper_bound"] + 1e-9
    assert batch.parse_seeds("0-2,5") == [0, 1, 2, 5]


def test_catalog_exports_parse_back():
    """Found in audit: both exports cut the declination to 10 characters, dropping the tenths of
    arcsec ('+28 51 50.' for +28:51:50.4)."""
    for key in ("keck1_lris_tonight", "clay_mike_darktime"):
        p = load_program(builtin_programs()[key])
        rows = [ln for ln in magellan_catalog(p).splitlines() if not ln.startswith("#")]
        assert len(rows) == len(p.targets)
        for t, row in zip(p.targets, rows):
            f = row.split()
            assert len(f) == 16 and f[4] == "2000.0" and f[8] in ("OFF", "EQU", "HRZ")
            assert abs((parse_ra(f[2]) - t.ra + 180) % 360 - 180) * 3600 < 0.08
            assert abs(parse_dec(f[3]) - t.dec) * 3600 < 0.051
        lines = [ln for ln in keck_starlist(p).splitlines() if not ln.startswith("#")]
        assert len(lines) == len(p.targets)
        for t, ln in zip(p.targets, lines):
            assert ln[15] == " " and ln[27] == " " and ln[39] == " "  # name 1-15, RA 17-27, Dec 29-39
            ra, dec = ln[16:27], ln[28:39]
            assert abs((parse_ra(ra.replace(" ", ":")) - t.ra + 180) % 360 - 180) * 3600 < 0.08
            assert abs(parse_dec(dec.replace(" ", ":")) - t.dec) * 3600 < 0.051
            assert ln[40:46] == "2000.0"
