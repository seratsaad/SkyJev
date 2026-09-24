"""The console engine (headless) and the assistant loop against it, without models or network."""

import json

import pytest

from obsassist.assistant.adapters.sim import SimConsoleAdapter
from obsassist.assistant.observation import from_model, from_snapshot, render
from obsassist.assistant.policy import Assistant, Settings
from obsassist.console.engine import Observatory
from obsassist.planning.catalogs import keck_starlist, magellan_catalog
from obsassist.planning.planner import Nowcast
from obsassist.targets import builtin_programs, load_program


class LocalAdapter:
    """In-process adapter around an Observatory (what SimConsoleAdapter does over HTTP)."""

    name = "local"
    instructions = SimConsoleAdapter.instructions

    def __init__(self, obs):
        self.obs = obs
        self.can_actuate = True

    def snapshot(self):
        s = self.obs.snapshot(full=True)
        return s

    def frames(self):
        return self.obs.frames

    def frame_ql(self, name=None):
        return {}

    def etc(self, target, seeing=None, cloud=None, t_exp=None):
        return self.obs.etc(self.obs._find_target(target), seeing, cloud, t_exp)

    def send(self, command, confirm=False):
        assert confirm
        return self.obs.command(command, source="assistant")

    def describe(self):
        return {"adapter": self.name, "can_actuate": True}


@pytest.fixture()
def obs(tmp_path):
    p = load_program(builtin_programs()["clay_mike_darktime"])
    return Observatory(p, seed=5, data_root=str(tmp_path), write_frames=False, allow_assistant_control=True)


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


def test_autopilot_observes_a_night(obs):
    a = Assistant(LocalAdapter(obs), s1=None, s2=None, settings=Settings(mode="autopilot"))
    obs.advance(70)
    for _ in range(400):
        a.step()
        obs.advance(3.0)
        if obs.t >= obs.model.t_end:
            break
    assert a.stats["executed"] >= 5
    assert obs.state.n_exp.sum() >= 5
    assert obs.model.score(obs.state) > 0.3 * obs.model.max_score()
    # the trace holds every exchange behind the decisions, as plain JSON
    tr = a.trace.since(0, limit=100000)
    kinds = {e["kind"] for e in tr}
    assert {"observation", "planner", "recommendation", "console"} <= kinds
    cmd = next(e for e in tr if e["kind"] == "console")
    assert cmd["in"]["cmd"].startswith("goto ") and cmd["out"]["reply"] and cmd["decision"] is not None
    plan = next(e for e in tr if e["kind"] == "planner")
    assert plan["out"]["pick"] == plan["in"][0]["name"]  # the highest merit
    json.dumps(tr, allow_nan=False)


def test_observation_from_model_and_snapshot_render(obs):
    obs.advance(90)
    obs.refresh_plan()
    snap = obs.snapshot(full=True)
    o1 = from_snapshot(snap)
    txt = render(o1, focus=next(c["name"] for c in o1["candidates"] if c["feasible"]))
    assert "Evaluate:" in txt and "Seeing" in txt
    o2 = from_model(obs.model, obs.state, Nowcast.from_history(obs.model, obs.t))
    assert {c["name"] for c in o2["candidates"]} == {c["name"] for c in o1["candidates"]}


def test_catalog_exports():
    p = load_program(builtin_programs()["clay_mike_darktime"])
    cat = magellan_catalog(p)
    rows = [ln for ln in cat.splitlines() if not ln.startswith("#")]
    assert len(rows) == len(p.targets) and all(len(r.split()) == 16 for r in rows)
    sl = keck_starlist(load_program(builtin_programs()["keck1_lris_tonight"]))
    assert "2000.0" in sl and len(sl.splitlines()) == 15
