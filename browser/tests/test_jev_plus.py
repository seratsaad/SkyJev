"""Offline checks on the gpt-jev-plus fixes. No browser, no network: the model calls are stubbed."""

from __future__ import annotations

import math

import pytest

from astroweb import gpt_chooser, jev_plus


@pytest.fixture
def plus(monkeypatch):
    monkeypatch.setenv("JEV_PLUS", "refusal,done,enter,popup,frames")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    jev_plus.install()
    yield jev_plus
    jev_plus.reset()


def _page(url="https://simbad.cds.unistra.fr/simbad/sim-basic?Ident=Vega", fp="f1"):
    return {
        "url": url, "title": "Vega", "text": "* alf Lyr -- Variable Star", "fingerprint": fp,
        "actions": [
            {"id": "e1", "kind": "click", "node": 11, "label": "Search", "role": "button"},
            {"id": "e2", "kind": "click", "node": 12, "label": "Continue", "role": "button"},
        ],
    }


def _logprobs(p_yes):
    return {"choices": [{"logprobs": {"content": [{"top_logprobs": [
        {"token": "yes", "logprob": math.log(p_yes)}, {"token": "no", "logprob": math.log(1 - p_yes)}]}]}}],
        "usage": {"total_tokens": 100}}


def test_enabled_parses_switches(monkeypatch):
    monkeypatch.setenv("JEV_PLUS", "none")
    assert jev_plus.enabled() == set()
    monkeypatch.setenv("JEV_PLUS", "refusal, enter")
    assert jev_plus.enabled() == {"refusal", "enter"}


def test_done_question_stops_on_a_confident_yes(plus, monkeypatch):
    monkeypatch.setattr(jev_plus.jev_model, "post_json", lambda *a: _logprobs(0.97))
    monkeypatch.setattr(gpt_chooser, "choose_with_gpt", lambda *a: pytest.fail("chooser should not be asked"))
    choose = jev_plus._make_choose(jev_plus.enabled())
    history = [{"kind": "click", "action": "SIMBAD search", "url": _page()["url"]}]
    decision = choose(_page(), "Open the SIMBAD page for Vega", history)
    assert decision["choice"] == "DONE"
    assert decision["usage"]["total_tokens"] == 100


def test_done_question_is_not_asked_before_any_action_or_right_after_typing(plus, monkeypatch):
    monkeypatch.setattr(jev_plus.jev_model, "post_json", lambda *a: pytest.fail("no done question expected"))
    monkeypatch.setattr(gpt_chooser, "choose_with_gpt", lambda state, goal, history: {"choice": "e1", "usage": {}})
    choose = jev_plus._make_choose(jev_plus.enabled())
    assert choose(_page(), "goal", [])["choice"] == "e1"
    assert choose(_page(fp="f2"), "goal", [{"kind": "fill", "action": "textbox"}])["choice"] == "e1"


def test_a_no_is_charged_to_the_next_decision_and_asked_once_per_page(plus, monkeypatch):
    calls = []
    monkeypatch.setattr(jev_plus.jev_model, "post_json", lambda *a: calls.append(1) or _logprobs(0.1))
    monkeypatch.setattr(gpt_chooser, "choose_with_gpt",
                        lambda state, goal, history: {"choice": "e1", "usage": {"total_tokens": 10}})
    choose = jev_plus._make_choose(jev_plus.enabled())
    history = [{"kind": "click", "action": "x"}]
    assert choose(_page(), "goal", history)["usage"]["total_tokens"] == 110
    choose(_page(), "goal", history)
    assert len(calls) == 1


def test_an_element_refused_twice_is_withdrawn_and_reported(plus, monkeypatch):
    seen = {}

    def base(state, goal, history):
        seen.update(nodes=[a["node"] for a in state["actions"]], notes=state.get("agent_notes"))
        return {"choice": "e2", "usage": {}}

    monkeypatch.setattr(gpt_chooser, "choose_with_gpt", base)
    choose = jev_plus._make_choose({"refusal"})
    page = _page()
    jev_plus.STATE["refusals"][(page["url"], 11)] = 1
    jev_plus.STATE["refused_labels"].append("Search")
    choose(page, "goal", [])
    assert seen["nodes"] == [11, 12] and "Search" in seen["notes"][0]
    jev_plus.STATE["refusals"][(page["url"], 11)] = 2
    choose(page, "goal", [])
    assert seen["nodes"] == [12]


def test_extra_options_become_controls():
    from jev_ultrafast.model import action_space

    class FakeBrowser:
        def evaluate(self, _):
            return {"enter": {"label": "Search arXiv", "value": "2306.06308"},
                    "frames": [{"src": "https://gea.esac.esa.int/archive-ui/q", "title": "query"}]}

    info = jev_plus._add_extras(FakeBrowser(), {"actions": [{"id": "wait", "kind": "wait", "label": "Wait"}]},
                                {"enter", "frames"})
    _, targets, controls = action_space(info["actions"])
    assert {"PRESS_ENTER", "OPEN_FRAME_1", "WAIT"} <= set(controls)
    assert not targets


def test_baseline_chooser_payload_is_unchanged_by_default():
    assert gpt_chooser.RECENT_KEYS[:4] == ("action", "kind", "text", "page_changed")
