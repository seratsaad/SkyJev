"""The measurement code itself: failure labelling, the CDP counter, and the chooser's guard rails."""

from __future__ import annotations

import jev_ultrafast.browser as jev_browser
import jev_ultrafast.model as jev_model
import pytest

from astroweb.jev_compat import CdpCounter, patch_text_helper
from astroweb.jev_runner import auto_failure_label


def label(**kwargs):
    base = {"status": "blocked", "verified": False, "page": {}, "probe": {},
            "history": [], "error": None, "transient": None}
    base.update(kwargs)
    return auto_failure_label(**base)


def test_a_verified_run_has_no_failure_label():
    assert label(verified=True) is None
    # Even if it reached the goal transiently, a verified run is simply a success.
    assert label(verified=True, transient={"reached": True}) is None


def test_reaching_the_goal_and_leaving_is_its_own_label():
    assert label(transient={"reached": True}) == "abandoned_success"
    assert label(status="done", transient={"reached": True}) == "abandoned_success"


def test_claiming_done_without_verifying_is_a_false_done():
    assert label(status="done") == "false_done"


def test_structural_limits_are_named_before_generic_ones():
    assert label(probe={"iframes": 5}) == "iframe"
    assert label(probe={"canvas": 2}) == "canvas"
    assert label(page={"omitted_actions": 40}) == "dense_action_space"


def test_stale_loop_becomes_unexecutable_target():
    assert label(status="stale_loop") == "unexecutable_target"


def test_error_labels_distinguish_provider_from_browser():
    assert label(error="RuntimeError: Model provider returned HTTP 500") == "provider_error"
    assert label(error="_IPCResponseTimeout: Runtime.evaluate timed out") == "browser_timeout"
    assert label(error="StalePage: Document is navigating") == "stale_page"


def test_repeated_actions_that_change_nothing():
    history = [{"page_changed": False} for _ in range(3)]
    assert label(history=history) == "no_progress"


def test_cdp_counter_counts_and_restores():
    original = jev_browser.cdp
    counter = CdpCounter().install()
    assert jev_browser.cdp is not original
    jev_browser.cdp = lambda method, **kw: None  # a call made after install is not counted twice
    counter.uninstall()
    assert jev_browser.cdp is original
    assert counter.total == 0


def test_text_helper_shim_strips_reasoning_for_openai_only(monkeypatch):
    """OpenAI rejects JEV's `reasoning` field; OpenRouter needs it. Only one of them may be edited."""
    import astroweb.jev_compat as compat

    seen = {}

    def fake(url, key, body):
        seen[url] = body
        return {"ok": True}

    original = jev_model.post_json
    monkeypatch.setattr(compat, "_ORIGINAL_POST", fake)
    try:
        patch_text_helper()
        jev_model.post_json("https://api.openai.com/v1/chat/completions", "k", {"m": 1, "reasoning": {"x": 1}})
        jev_model.post_json("https://openrouter.ai/api/v1/chat/completions", "k", {"m": 1, "reasoning": {"x": 1}})
    finally:
        jev_model.post_json = original
    assert "reasoning" not in seen["https://api.openai.com/v1/chat/completions"]
    assert "reasoning" in seen["https://openrouter.ai/api/v1/chat/completions"]


def test_rate_limit_retry_gives_up_eventually(monkeypatch):
    """A 429 is retried patiently, but a run must not hang forever on a throttled account."""
    import astroweb.jev_compat as compat

    calls = {"n": 0}

    def always_throttled(url, key, body):
        calls["n"] += 1
        raise RuntimeError("Model provider returned HTTP 429; no action executed.")

    original = jev_model.post_json
    monkeypatch.setattr(compat, "_ORIGINAL_POST", always_throttled)
    monkeypatch.setattr(compat.time, "sleep", lambda _s: None)
    try:
        patch_text_helper(rate_limit_retries=3)
        with pytest.raises(RuntimeError, match="429"):
            jev_model.post_json("https://api.openai.com/v1/chat/completions", "k", {"m": 1})
    finally:
        jev_model.post_json = original
    assert calls["n"] == 3


def test_non_rate_limit_errors_are_not_retried(monkeypatch):
    import astroweb.jev_compat as compat

    calls = {"n": 0}

    def broken(url, key, body):
        calls["n"] += 1
        raise RuntimeError("Model provider returned HTTP 500; no action executed.")

    original = jev_model.post_json
    monkeypatch.setattr(compat, "_ORIGINAL_POST", broken)
    try:
        patch_text_helper(rate_limit_retries=3)
        with pytest.raises(RuntimeError, match="500"):
            jev_model.post_json("https://api.openai.com/v1/chat/completions", "k", {"m": 1})
    finally:
        jev_model.post_json = original
    assert calls["n"] == 1


class FakePost:
    def __init__(self, answer):
        self.answer = answer

    def __call__(self, url, key, body):
        self.body = body
        return {"choices": [{"message": {"content": self.answer}}], "usage": {"total_tokens": 10}}


PAGE = {
    "url": "https://example.org/",
    "title": "Example",
    "text": "hello",
    "actions": [
        {"id": "e1", "kind": "click", "node": 1, "role": "button", "label": "Search", "value": ""},
        {"id": "e2", "kind": "fill", "node": 2, "role": "textbox", "label": "Query", "value": ""},
        {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
    ],
}


def run_chooser(answer, monkeypatch):
    from astroweb import gpt_chooser

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(jev_model, "post_json", FakePost(answer))
    return gpt_chooser.choose_with_gpt(PAGE, "find something", [])


def test_chooser_returns_an_offered_element(monkeypatch):
    result = run_chooser('{"operation": "CLICK", "target": "1", "confidence": 0.8}', monkeypatch)
    assert result["choice"] == "e1"
    assert result["operation"] == "CLICK"
    assert result["confidence"] == 0.8


def test_chooser_rejects_an_index_that_was_not_offered(monkeypatch):
    with pytest.raises(ValueError, match="unoffered target"):
        run_chooser('{"operation": "CLICK", "target": "99", "confidence": 0.9}', monkeypatch)


def test_chooser_rejects_an_invented_operation(monkeypatch):
    with pytest.raises(ValueError, match="unoffered operation"):
        run_chooser('{"operation": "EXECUTE_JS", "target": "1", "confidence": 1}', monkeypatch)


def test_chooser_rejects_unparseable_output(monkeypatch):
    with pytest.raises(ValueError, match="no usable answer"):
        run_chooser("I'll click the search button.", monkeypatch)


def test_control_operations_need_no_target(monkeypatch):
    result = run_chooser('{"operation": "DONE", "target": null, "confidence": 0.7}', monkeypatch)
    assert result["choice"] == "DONE"
    assert result["target"] is None


# --- the baseline's verification target ------------------------------------------------------
#
# A round of baseline runs was thrown away because the verifier attached to "the newest tab" after
# the agent's session had already been stopped, and read a leftover tab from an earlier run. These
# pin the two rules that prevent it: match the agent's own last URL, and never guess.


def test_verify_target_matches_the_agents_last_url(monkeypatch):
    import astroweb.baseline_runner as runner

    monkeypatch.setattr(runner, "list_targets", lambda: [
        {"id": "stale", "url": "https://ui.adsabs.harvard.edu/search/q=old"},
        {"id": "ours", "url": "https://arxiv.org/abs/2306.06308"},
    ])
    seen = {}

    class FakeSession:
        def __init__(self, target_id=None, **kw):
            seen["target"] = target_id
            self.owned = True

        def url(self):
            return "https://arxiv.org/abs/2306.06308"

        def text(self):
            return "arXiv:2306.06308"

        def evaluate(self, expr):
            return None

        def close(self):
            pass

    monkeypatch.setattr(runner, "Session", FakeSession)
    task = type("T", (), {"verifier": {"type": "url_contains", "value": "/abs/2306.06308"},
                          "verify_timeout_seconds": 1})()
    result = runner._verify_live(task, ["https://arxiv.org/", "https://arxiv.org/abs/2306.06308"])
    assert seen["target"] == "ours", "verified the wrong tab"
    assert result["verification"]["passed"]
    assert result["unverifiable"] is False


def test_no_matching_tab_is_unverifiable_not_a_failure(monkeypatch):
    import astroweb.baseline_runner as runner

    monkeypatch.setattr(runner, "list_targets", lambda: [{"id": "stale", "url": "https://example.org/"}])
    task = type("T", (), {"verifier": {"type": "url_contains", "value": "x"}, "verify_timeout_seconds": 1})()
    result = runner._verify_live(task, ["https://arxiv.org/abs/2306.06308"])
    assert result["unverifiable"] is True
    assert not result["verification"]["passed"]
    assert runner._label("done", False, None, None, True, unverifiable=True) == "unverifiable"


def test_running_out_of_credit_is_not_agent_behaviour():
    """Browser Use absorbs per-step LLM failures, so a dead account looks like a stuck agent."""
    import astroweb.baseline_runner as runner

    class FakeHistory:
        def errors(self):
            return [None, "Error code: 429 - insufficient_quota, credit_balance_exhausted"]

    errors = runner._provider_errors(FakeHistory())
    assert any("credit" in e.lower() for e in errors)
    assert issubclass(runner.OutOfCredit, RuntimeError)


# --- the spend ceiling -----------------------------------------------------------------------


def test_budget_stops_a_batch_at_its_ceiling():
    from astroweb.budget import Budget, BudgetExceeded

    budget = Budget(max_usd=1.00, model="gpt-4.1-mini")   # $0.40 per million input tokens
    budget.add(total_tokens=1_000_000)
    budget.check()                                         # $0.40 spent, still under
    budget.add(total_tokens=2_000_000)
    with pytest.raises(BudgetExceeded, match="Stopping"):
        budget.check()


def test_a_bare_total_is_charged_at_the_input_rate():
    """Under-charging a ceiling is the failure that matters, so an unsplit total is not discounted."""
    from astroweb.budget import Budget

    budget = Budget(max_usd=10, model="gpt-4.1-mini")
    budget.add(total_tokens=1_000_000)
    assert budget.input_tokens == 1_000_000
    assert budget.usd == pytest.approx(0.40)


def test_prices_can_be_overridden_because_they_change(monkeypatch):
    from astroweb.budget import Budget, price_for

    monkeypatch.setenv("MODEL_PRICE_IN", "1.0")
    monkeypatch.setenv("MODEL_PRICE_OUT", "2.0")
    assert price_for("anything-at-all") == (1.0, 2.0)
    budget = Budget(max_usd=5, model="anything-at-all")
    budget.add(input_tokens=1_000_000, output_tokens=1_000_000)
    assert budget.usd == pytest.approx(3.0)


def test_projection_says_how_many_more_runs_fit():
    from astroweb.budget import Budget

    budget = Budget(max_usd=1.00, model="gpt-4.1-mini")
    budget.add(total_tokens=250_000)            # $0.10 for one run
    assert budget.projected_runs() == pytest.approx(9.0)


def test_an_empty_final_url_falls_back_to_the_last_real_one(monkeypatch):
    """Browser Use sometimes records an empty URL for the final step; that must not lose the run."""
    import astroweb.baseline_runner as runner

    monkeypatch.setattr(runner, "list_targets", lambda: [
        {"id": "ours", "url": "https://exoplanetarchive.ipac.caltech.edu/overview/TRAPPIST-1%20e"},
    ])

    class FakeSession:
        def __init__(self, target_id=None, **kw):
            self.owned = True

        def url(self):
            return "https://exoplanetarchive.ipac.caltech.edu/overview/TRAPPIST-1%20e"

        def text(self):
            return "TRAPPIST-1 e"

        def evaluate(self, expr):
            return None

        def close(self):
            pass

    monkeypatch.setattr(runner, "Session", FakeSession)
    task = type("T", (), {"verifier": {"type": "url_contains", "value": "TRAPPIST-1"},
                          "verify_timeout_seconds": 1})()
    result = runner._verify_live(task, [
        "https://exoplanetarchive.ipac.caltech.edu/",
        "https://exoplanetarchive.ipac.caltech.edu/overview/TRAPPIST-1%20e",
        "",
    ])
    assert result["unverifiable"] is False
    assert result["verification"]["passed"]


def test_cdp_reads_are_bounded(monkeypatch):
    """A blocking recv with no deadline once hung a whole batch for four hours."""
    import astroweb.cdp_direct as cdp

    class NeverAnswers:
        def send(self, _payload):
            pass

        def recv(self, timeout=None):
            raise TimeoutError

        def close(self):
            pass

    session = cdp.Session.__new__(cdp.Session)
    session.ws = NeverAnswers()
    session.ids = iter(range(1, 10))
    session.timeout = 0.2
    session.session_id = "s"
    with pytest.raises(cdp.CdpTimeout, match="did not answer"):
        session.send("Runtime.evaluate")
