"""The assistant loop against the headless console (no models, no network), and the observation
it reads."""

import json
import time

import pytest

from obsassist.assistant.adapters.sim import SimConsoleAdapter
from obsassist.assistant.observation import from_model, from_snapshot, render
from obsassist.assistant.policy import Assistant, Settings
from obsassist.console.engine import Observatory
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


def test_system2_is_asked_only_when_there_is_a_choice(obs):
    asked = []

    class FakeSystem2:
        enabled, trace = True, None

        def deliberate(self, context, effort="low", tag=None):
            asked.append(tag)
            return {"action": "none", "target": "", "rationale": ""}

    a = Assistant(LocalAdapter(obs), s1=None, s2=FakeSystem2(), settings=Settings(mode="autopilot"))
    for _ in range(80):  # from dusk: in twilight only the telluric standard is observable
        a.step()
        obs.advance(3.0)
    time.sleep(0.2)  # escalations run in a background thread
    obs_entries = [e for e in a.trace.since(0, 100000) if e["kind"] == "observation"]
    feasible = {e["decision"]: int(e["summary"].split()[0]) for e in obs_entries}  # "N feasible candidates"
    assert 1 in feasible.values() and asked and all(feasible[d] > 1 for d in asked)
