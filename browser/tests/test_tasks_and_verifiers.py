"""Offline checks on the benchmark definition and the verifier logic. No browser, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astroweb.tasks import CATEGORIES, load_tasks
from astroweb.verification import Outcome, transient_url_match, url_specs, verify

ROOT = Path(__file__).resolve().parents[1]


def test_fifteen_tasks_across_five_categories():
    tasks = load_tasks()
    assert len(tasks) == 15
    counts = {c: sum(1 for t in tasks if t.category == c) for c in CATEGORIES}
    assert all(counts[c] == 3 for c in CATEGORIES), counts


def test_every_task_is_complete():
    for task in load_tasks():
        assert task.goal and task.goal[0].isupper(), task.id
        assert task.start_url.startswith("https://"), task.id
        assert task.timeout_seconds >= 60, task.id
        # A goal must not name the answer's URL, or the task is a navigation instruction.
        assert "http" not in task.goal, task.id


def test_adversarial_tasks_declare_what_they_probe():
    for task in load_tasks():
        if task.category == "adversarial":
            assert task.expected_limits, task.id


def test_verifier_types_are_supported():
    supported = {"url_contains", "url_regex", "url_not_equal", "page_contains", "page_regex",
                 "frames_regex", "js_expr", "all_of", "any_of"}

    def walk(spec):
        assert spec["type"] in supported, spec
        for item in spec.get("items", []):
            walk(item)

    for task in load_tasks():
        walk(task.verifier)


def test_verify_url_and_page():
    outcome = Outcome(url="https://arxiv.org/abs/2306.06308", text="arXiv:2306.06308 something")
    assert verify({"type": "url_contains", "value": "/abs/2306.06308"}, outcome)["passed"]
    assert verify({"type": "page_contains", "value": "arXiv:2306.06308"}, outcome)["passed"]
    assert not verify({"type": "page_contains", "value": "not here"}, outcome)["passed"]
    assert verify({"type": "url_not_equal", "value": "https://arxiv.org/"}, outcome)["passed"]


def test_all_of_requires_every_part():
    outcome = Outcome(url="https://example.org/x", text="hello")
    spec = {"type": "all_of", "items": [
        {"type": "url_contains", "value": "/x"},
        {"type": "page_contains", "value": "absent"},
    ]}
    assert not verify(spec, outcome)["passed"]
    spec["items"][1]["value"] = "hello"
    assert verify(spec, outcome)["passed"]


def test_any_of_needs_only_one():
    outcome = Outcome(url="https://example.org/x", text="hello")
    spec = {"type": "any_of", "items": [
        {"type": "url_contains", "value": "nope"},
        {"type": "page_contains", "value": "hello"},
    ]}
    assert verify(spec, outcome)["passed"]


def test_js_and_frames_need_an_evaluator():
    outcome = Outcome(url="u", text="t", evaluate=None)
    assert not verify({"type": "js_expr", "expression": "1"}, outcome)["passed"]
    assert not verify({"type": "frames_regex", "value": "x"}, outcome)["passed"]


def test_unknown_verifier_type_is_an_error():
    with pytest.raises(ValueError):
        verify({"type": "vibes", "value": "good"}, Outcome(url="u", text="t"))


def test_transient_match_finds_an_abandoned_success():
    spec = {"type": "all_of", "items": [
        {"type": "url_contains", "value": "Ident=Betelgeuse"},
        {"type": "page_contains", "value": "alf Ori"},
    ]}
    visited = ["https://simbad.example/", "https://simbad.example/sim-basic?Ident=Betelgeuse",
               "https://simbad.example/"]
    result = transient_url_match(spec, visited)
    assert result["applicable"] and result["reached"] and result["at"] == 1
    assert not transient_url_match(spec, ["https://simbad.example/"])["reached"]


def test_transient_match_is_inapplicable_without_url_checks():
    spec = {"type": "page_contains", "value": "x"}
    assert url_specs(spec) == []
    assert transient_url_match(spec, ["https://a/"])["applicable"] is False


def test_verifier_validation_report_is_clean():
    """The saved validation run must show every verifier passing on gold and failing on start."""
    path = ROOT / "results" / "verifier_validation.json"
    if not path.exists():
        pytest.skip("run `make validate` first")
    report = json.loads(path.read_text())
    assert len(report) == 15
    bad = [r["task"] for r in report if r["verdict"] != "ok"]
    assert not bad, f"verifiers that do not discriminate: {bad}"


# --- the goal page -----------------------------------------------------------------------------
#
# "Reached the goal and then left" is a headline finding, so the definition of "the goal page" is
# pinned here. It was once read off the verifier's URL clauses, some of which accept almost any
# page (for example "the agent left the home page"), and the count came out inflated.


def test_goal_url_names_the_gold_page_and_not_the_start():
    import re

    from astroweb.tasks import load_tasks

    for task in load_tasks():
        if not task.goal_url:
            continue
        assert task.gold_url and re.search(task.goal_url, task.gold_url), f"{task.id}: misses its gold page"
        assert not re.search(task.goal_url, task.start_url), f"{task.id}: already true at the start page"


def test_every_verified_success_ends_on_its_goal_page():
    """A goal pattern stricter than the verifier would label real successes as never arriving."""
    import glob
    import json
    import re

    from astroweb.tasks import load_tasks

    goal = {t.id: t.goal_url for t in load_tasks()}
    for path in glob.glob("results/*/*.json"):
        if any(part in path for part in ("/summary/", "/shipped/", "/hybrid/", "/anyjev/", "/_")):
            continue
        with open(path) as handle:
            record = json.load(handle)
        pattern = goal.get(record.get("task_id"))
        if pattern and record.get("success"):
            assert re.search(pattern, record.get("final_url") or ""), (
                f"{path}: verified success ended on {record.get('final_url')!r}, which goal_url rejects"
            )


def test_goal_page_is_not_asked_when_the_url_cannot_show_it():
    from astroweb.verification import goal_page_reached

    assert goal_page_reached(None, ["https://example.org/a"])["applicable"] is False
    hit = goal_page_reached(r"/abs/2306\.06308", ["https://arxiv.org/", "https://arxiv.org/abs/2306.06308"])
    assert hit["reached"] and hit["at"] == 1
