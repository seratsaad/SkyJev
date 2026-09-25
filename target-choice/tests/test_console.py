"""The console engine, headless: command grammar, instrument timing, the operator's behaviour at
limits and closures, ToO alerts, S/N bookkeeping and snapshots."""

import json

import numpy as np
import pytest

from obsassist.astro.ephem import fmt_dec, fmt_ra
from obsassist.console.engine import Observatory
from obsassist.targets import builtin_programs, load_program


def _obs(tmp_path, seed=5):
    p = load_program(builtin_programs()["clay_mike_darktime"])
    return Observatory(p, seed=seed, data_root=str(tmp_path), write_frames=False, allow_assistant_control=True)


def _on_target(o, name="CS 22892-052"):
    assert o.command(f'goto "{name}"') == "ok"
    for _ in range(4):
        if o.tcs.state == "guiding":
            break
        o.fast_forward()
    assert o.tcs.state == "guiding"
    return o.program.target_index(name)


@pytest.fixture()
def obs(tmp_path):
    p = load_program(builtin_programs()["clay_mike_darktime"])
    return Observatory(p, seed=5, data_root=str(tmp_path), write_frames=False, allow_assistant_control=True)


def test_goto_grammar_and_limits(tmp_path):
    o = _obs(tmp_path)
    o.advance(75)
    m = o.model
    j = m.idx(o.t)
    lst = float(m.ephem.lst[j])
    # by (partial, unique) name
    assert o.command("goto 22892") == "ok" and o.tcs.name == "CS 22892-052" and o.tcs.target is not None
    assert o.command("goto nonexistent").startswith("error")
    # by sexagesimal and by decimal-degree coordinates
    assert o.command(f"goto {fmt_ra(lst)} {fmt_dec(-35.0)} field1") == "ok"
    assert o.tcs.target is None and o.tcs.name == "field1" and o.tcs.dec == pytest.approx(-35.0)
    assert o.command(f"goto {lst:.3f} -35.5 field2") == "ok"
    assert o.tcs.ra == pytest.approx(lst, abs=1e-3) and o.tcs.dec == pytest.approx(-35.5)
    assert o.command(f"goto {int(lst)} -35 field3") == "ok" and o.tcs.dec == -35.0 and o.tcs.name == "field3"
    assert o.command("goto 400.0 -20.0").startswith("error")  # RA out of range
    # below the elevation limit: the TO refuses and nothing moves
    before = (o.tcs.name, o.tcs.ra)
    r = o.command(f"goto {fmt_ra(lst + 180)} {fmt_dec(20.0)} below")
    assert r.startswith("TO:") and "below the limit" in r and (o.tcs.name, o.tcs.ra) == before
    # refused while an arm is exposing
    _on_target(o)
    o.command("blue exptime 300")
    o.command("start blue")
    assert o.command("goto CS 31082-001").startswith("refused")


def test_offsets_rotator_focus_and_arm_settings(tmp_path):
    o = _obs(tmp_path)
    o.advance(75)
    _on_target(o)
    o.command("offset 1.5 -2")
    o.command("offset 0.5 1")
    assert o.tcs.offset == (2.0, -1.0)
    o.command("rot 45")
    assert o.tcs.pa == 45.0
    o.command("rot parallactic")
    assert o.tcs.pa is None
    assert o.command("focus") == "ok" and o.focus_busy_until is not None
    o.fast_forward()
    assert o.focus_busy_until is None and o.state.focus_temp == pytest.approx(o.model.temperature(o.t), abs=0.2)
    assert o.command("red exptime 450") == "ok" and o.arms["red"].exp_time == 450
    assert o.command("all loops 3") == "ok" and all(a.loops == 3 for a in o.arms.values())
    assert o.command("blue type flat") == "ok" and o.arms["blue"].lamp == "quartz"
    assert o.command("blue type bias") == "ok" and o.arms["blue"].exp_time == 0.0
    assert o.command("blue type banana").startswith("error")
    assert o.command("red bin 1 1") == "ok" and o.arms["red"].binning == (1, 1)
    assert o.command("red speed fast") == "ok" and o.arms["red"].speed == "Fast"
    assert o.command("green exptime 1").startswith("error")


def test_readout_time_matches_the_configuration(tmp_path):
    """Readout times are the configuration's (quoted at its own binning), e.g. 41 s for MIKE blue 2x2."""
    o = _obs(tmp_path)
    b, r = o.arms["blue"], o.arms["red"]
    assert b.binning == (2, 2) and b.readout_s() == pytest.approx(b.cfg.readout_s) == pytest.approx(41.0)
    assert r.readout_s() == pytest.approx(48.0)
    b.speed = "Fast"
    assert b.readout_s() == pytest.approx(28.0)
    b.speed, b.binning = "Slow", (1, 1)
    assert b.readout_s() == pytest.approx(82.0)  # 4x the pixels
    # the reading phase lasts exactly that long and the progress counts lines
    b.binning = (2, 2)
    o.advance(75)
    _on_target(o)
    o.command("blue exptime 60")
    o.command("start blue")
    o.fast_forward()
    assert b.state == "reading"
    t_read = o.t
    o.advance(20.5 / 60)
    d = b.to_dict(o.t)
    assert d["read_progress"] == pytest.approx(0.5, abs=0.02)
    assert f"/{b.lines_total()} lines" in d["message"] and int(d["message"].split()[2].split("/")[0]) > 0
    o.fast_forward()
    assert b.state == "idle" and (o.t - t_read) * 60 == pytest.approx(41.0, abs=1.0)


def test_loops_and_sn_booked_on_the_matching_arm_only(tmp_path):
    o = _obs(tmp_path)
    o.advance(75)
    i = _on_target(o)  # a MIKE-BLUE target
    o.command("all exptime 120")
    o.command("all loops 2")
    o.command("start all")
    for _ in range(30):
        if all(a.state == "idle" for a in o.arms.values()) and len(o.frames) >= 4:
            break
        o.fast_forward()
    files = {f["file"]: f for f in o.frames}
    assert set(files) == {"b0001.fits", "b0002.fits", "r0001.fits", "r0002.fits"}
    assert all(files[k]["counted"] and files[k]["snr"] > 0 for k in ("b0001.fits", "b0002.fits"))
    assert all(not files[k]["counted"] and "MIKE-BLUE" in files[k]["note"] for k in ("r0001.fits", "r0002.fits"))
    assert o.state.n_exp[i] == 2 and o.state.open_s[i] == pytest.approx(240.0, abs=0.1)
    s2 = files["b0001.fits"]["snr"] ** 2 + files["b0002.fits"]["snr"] ** 2
    assert o.state.snr2[i] == pytest.approx(s2, rel=1e-3)


def test_pause_resume_stop_abort(tmp_path):
    o = _obs(tmp_path)
    o.advance(75)
    i = _on_target(o)
    o.command("all exptime 600")
    o.command("start all")
    o.advance(2.0)
    o.command("pause blue")
    assert o.arms["blue"].state == "paused" and o.arms["red"].state == "exposing"
    o.advance(3.0)
    assert o.arms["blue"].to_dict(o.t)["elapsed_s"] == pytest.approx(120.0, abs=0.1)
    o.command("resume blue")
    o.advance(1.0)
    o.command("pause blue")
    o.command("stop all")  # stop: shutters close now, read out now
    assert o.arms["blue"].state == "reading" and o.arms["red"].state == "reading"
    o.fast_forward()
    o.fast_forward()
    f = {x["file"]: x for x in o.frames}
    assert f["b0001.fits"]["exptime"] == pytest.approx(180.0, abs=0.1)  # paused time is not exposure
    assert f["r0001.fits"]["exptime"] == pytest.approx(360.0, abs=0.1)
    assert o.state.open_s[i] == pytest.approx(180.0, abs=0.1)
    n = o.state.n_exp.sum()
    o.command("start blue")
    o.advance(1.0)
    o.command("abort all")
    assert o.arms["blue"].state == "clearing"
    o.advance(0.5)
    assert o.arms["blue"].state == "idle" and o.state.n_exp.sum() == n and len(o.frames) == 2


def test_elevation_limit_stops_tracking_and_the_frame_does_not_count(tmp_path):
    o = _obs(tmp_path)
    m = o.model
    i = o.program.target_index("CS 22892-052")  # MIKE-BLUE; sets through 15 deg late in the night
    a = m.geo["alt"][i]
    j_set = int(
        np.nonzero((a[:-1] >= m.tel.el_min_deg) & (a[1:] < m.tel.el_min_deg) & (m.ephem.t_min[:-1] > m.t_start))[0][0]
    )
    o.advance(j_set * m.dt - 22 - o.t)
    _on_target(o, "CS 22892-052")
    o.command("all exptime 1800")
    o.command("start all")
    o.fast_forward()  # stops at the limit, not at the end of the exposure
    assert o.tcs.state == "limit" and o.arms["blue"].state == "exposing"
    assert m.tel.el_min_deg - 0.3 < o.tcs.alt < m.tel.el_min_deg
    assert abs(o.t - j_set * m.dt) < 1.5
    assert any("elevation limit" in e["text"] for e in o.log)
    for _ in range(6):
        o.fast_forward()
    fr = {f["file"]: f for f in o.frames}["b0001.fits"]
    assert not fr["counted"] and any("not on the slit" in r for r in fr["qc_reasons"])
    assert o.state.snr2[i] == 0.0


def test_dome_closing_mid_exposure(tmp_path):
    o = _obs(tmp_path, seed=20)  # weather closes the dome at minute 177
    m = o.model
    ((c0, c1),) = m.weather.closed_intervals()
    o.advance(160 - o.t)
    i = _on_target(o, "Telluric B star T1")  # a MIKE-RED target
    o.command("all exptime 1800")
    o.command("start all")
    o.fast_forward()  # stops at the closure
    assert not o.dome_open and abs(o.t - c0) < 0.3 and o.arms["red"].state == "exposing"
    for _ in range(6):
        o.fast_forward()
    fr = {f["file"]: f for f in o.frames}["r0001.fits"]
    assert not fr["counted"] and fr["snr"] == 0.0
    assert "dome closed during the exposure" in fr["qc_reasons"]
    assert o.state.snr2[i] == 0.0


def test_too_alert_at_its_time_and_fast_forward_stops_there(tmp_path):
    o = _obs(tmp_path)
    m = o.model
    k = next(i for i, t in enumerate(o.program.targets) if t.appears_at)
    o.advance(m.appears[k] - 25 - o.t)
    _on_target(o, "CS 31082-001")
    o.command("all exptime 2400")
    o.command("start all")
    assert not any(e["who"] == "ALERT" for e in o.log)
    o.fast_forward()
    al = [e for e in o.log if e["who"] == "ALERT"]
    assert len(al) == 1 and al[0]["t"] == pytest.approx(m.appears[k], abs=0.01)
    assert o.t == pytest.approx(m.appears[k], abs=0.01) and o.arms["blue"].state == "exposing"
    snap = o.snapshot()
    assert next(t for t in snap["targets"] if t["i"] == k)["announced"]


def test_fast_forward_lands_on_each_event_in_turn(tmp_path):
    o = _obs(tmp_path)
    o.advance(75)
    _on_target(o)
    o.command("blue exptime 100")
    o.command("red exptime 170")
    o.command("all loops 2")
    o.command("start all")
    seen = []
    for _ in range(12):
        if all(a.state == "idle" for a in o.arms.values()):
            break
        nxt = o._next_event_time()
        before = {n: a.state for n, a in o.arms.items()}
        o.fast_forward()
        assert o.t <= nxt + 0.011  # never past the next event
        after = {n: a.state for n, a in o.arms.items()}
        seen.append(sum(before[n] != after[n] for n in before))
    assert all(x >= 1 for x in seen) and len([f for f in o.frames]) == 4


def test_snapshot_is_strict_json(tmp_path):
    """The snapshot, projection blocks included, is plain JSON (no numpy types)."""
    o = _obs(tmp_path)
    for dt in (0.0, 80.0, 30.0):
        o.advance(dt)
        o.refresh_plan()
        s = o.snapshot(full=True)
        json.dumps(s, allow_nan=False)


def test_dome_closing_during_acquisition_does_not_freeze_fast_forward(tmp_path):
    """A closure (or the elevation limit) during an acquisition clears the TCS busy time, so
    fast-forward keeps jumping to the next event."""
    o = _obs(tmp_path, seed=20)
    ((c0, _),) = o.model.weather.closed_intervals()
    o.advance(c0 - 1.5 - o.t)
    assert o.command('goto "Telluric B star T1"') == "ok"
    o.fast_forward()  # slew done, acquiring
    assert o.tcs.state == "acquiring"
    o.fast_forward()  # the dome closes mid-acquisition
    assert not o.dome_open and o.tcs.state == "tracking" and o.tcs.busy_until is None
    t = o.t
    o.fast_forward()
    assert o.t - t > 4.0  # a normal idle hop, not 0.01 min


def test_guider_resumes_after_focus(tmp_path):
    o = _obs(tmp_path)
    o.advance(75)
    _on_target(o)
    o.command("focus")
    assert o.guider["state"] == "off"
    o.fast_forward()
    assert o.focus_busy_until is None and o.guider["state"] == "guiding"
    t_before = o.guider["t"]
    o.advance(1.0)
    assert o.guider["t"] > t_before  # readings keep coming (nowcast cloud)


def test_too_cannot_be_observed_before_its_alert(tmp_path):
    o = _obs(tmp_path)
    o.advance(75)
    m = o.model
    k = next(i for i, t in enumerate(o.program.targets) if t.appears_at)
    name = o.program.targets[k].name
    assert o.command(f'goto "{name}"').startswith("error")
    o.advance(m.appears[k] + 0.5 - o.t)
    r = o.command(f'goto "{name}"')
    assert r == "ok" or r.startswith("TO:")  # known now (may still be below the limit)


def test_console_readout_matches_what_the_planner_assumes(tmp_path):
    """Every arm starts in the readout speed whose time the planner and the night model charge
    (LDSS3: fast, 30 s)."""
    from obsassist.console.engine import INSTRUMENT_ARMS
    from obsassist.instruments.base import all_configs

    for inst, arms in INSTRUMENT_ARMS.items():
        for _, key, _ in arms:
            cfg = all_configs()[key]
            tel = cfg.telescopes[0]
            prog = {
                "name": "t",
                "telescope": tel,
                "date": "2026-09-24",
                "targets": [{"name": "x", "ra": "00:00:00", "dec": "-10:00:00", "config": key, "mag": 18.0, "snr": 10}],
            }
            o = Observatory(load_program(prog), seed=1, data_root=str(tmp_path), write_frames=False)
            arm = next(a for a in o.arms.values() if a.cfg.key == key)
            assert arm.readout_s() == pytest.approx(cfg.readout_s), key


def test_manual_observing_sequence(obs):
    obs.advance(75)  # past 12-degree twilight
    assert obs.dome_open
    assert obs.command("goto CS 22892-052") == "ok"
    obs.fast_forward()
    obs.fast_forward()
    assert obs.tcs.state == "guiding"
    obs.command("all exptime 600")
    obs.command("start all")
    assert all(a.state == "exposing" for a in obs.arms.values())
    obs.fast_forward()
    obs.fast_forward()
    i = obs.program.target_index("CS 22892-052")
    assert obs.state.n_exp[i] == 1 and obs.state.snr2[i] > 0


def test_abort_discards_and_stop_reads_out(obs):
    obs.advance(75)
    obs.command("goto CS 22892-052")
    obs.fast_forward()
    obs.fast_forward()
    obs.command("blue exptime 1200")
    obs.command("start blue")
    obs.advance(5)
    obs.command("abort blue")
    obs.advance(1)
    assert obs.arms["blue"].state == "idle" and obs.state.n_exp.sum() == 0
    obs.command("start blue")
    obs.advance(5)
    obs.command("stop blue")
    obs.advance(2)
    assert obs.state.n_exp.sum() == 1 and 250 < obs.state.open_s.sum() < 330


def test_assistant_commands_refused_without_control(tmp_path):
    p = load_program(builtin_programs()["clay_mike_darktime"])
    o = Observatory(p, seed=5, data_root=str(tmp_path), write_frames=False, allow_assistant_control=False)
    assert o.command("goto CS 22892-052", source="assistant").startswith("refused")
