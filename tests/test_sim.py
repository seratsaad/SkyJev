"""Weather, night model, planner and oracle invariants."""

import numpy as np
import pytest

from obsassist.planning.oracle import pilot, upper_bound
from obsassist.planning.planner import GreedyPolicy, ListOrderPolicy, Nowcast, candidates, project, run_policy
from obsassist.programs.generator import TEMPLATE_SED, generate_program
from obsassist.sim.night import NightModel
from obsassist.targets import builtin_programs, load_program
from obsassist.weather import generate


@pytest.fixture(scope="module")
def mike():
    return NightModel(load_program(builtin_programs()["clay_mike_darktime"]), seed=5)


@pytest.fixture(scope="module")
def lris():
    return NightModel(load_program(builtin_programs()["keck1_lris_tonight"]), seed=6)


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


def test_policies_are_ordered(mike, lris):
    for m in (mike, lris):
        g, _ = run_policy(m, GreedyPolicy())
        best = pilot(m)
        assert best.score >= m.score(g) - 1e-9  # the pilot method never loses to greedy
        assert best.score <= upper_bound(m) + 1e-6  # nothing beats the relaxation
        assert best.score <= m.max_score()


def test_projection_runs_from_any_state(mike):
    s = mike.initial_state()
    s.t = mike.t_start + 200
    p = project(mike, s, Nowcast.from_history(mike, s.t))
    assert p.score >= 0 and all(b.end > b.start for b in p.blocks)


def test_candidates_explain_infeasibility(lris):
    s = lris.initial_state()
    s.t = lris.t_start + 30
    c = {x.name: x for x in candidates(lris, s, Nowcast.from_history(lris, s.t))}
    assert not c["AT 2026cvn"].feasible and c["AT 2026cvn"].note
    assert not c["TDE 2026sep"].feasible and "announced" in c["TDE 2026sep"].note


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
