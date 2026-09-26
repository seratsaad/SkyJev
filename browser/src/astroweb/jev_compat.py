"""Small shims around jev-ultrafast: OpenAI text helper compatibility and a CDP call counter."""

from __future__ import annotations

import time

import jev_ultrafast.browser as jev_browser
import jev_ultrafast.model as jev_model

_ORIGINAL_POST = jev_model.post_json
_ORIGINAL_CDP = jev_browser.cdp


def patch_text_helper(rate_limit_retries: int = 4):
    """Two compatibility fixes for running JEV against the OpenAI API.

    1. OpenAI's chat completions endpoint rejects JEV's `reasoning` field with HTTP 400. Drop it there.
    2. JEV retries a 429 twice at 0.5 s and 1 s. A per-minute token limit needs a longer wait, so
       retry the whole call with a widening pause. A run that dies on a rate limit is a measurement
       artefact, not a property of the agent.
    """

    def post_json(url, key, body):
        if "api.openai.com" in url and "reasoning" in body:
            body = {k: v for k, v in body.items() if k != "reasoning"}
        for attempt in range(rate_limit_retries):
            try:
                return _ORIGINAL_POST(url, key, body)
            except RuntimeError as e:
                if "HTTP 429" not in str(e) or attempt == rate_limit_retries - 1:
                    raise
                time.sleep(4 * (attempt + 1))
        raise RuntimeError("unreachable")

    jev_model.post_json = post_json


class CdpCounter:
    """Counts browser protocol calls made through JEV's browser layer, by method."""

    def __init__(self):
        self.calls: dict[str, int] = {}

    def install(self):
        counter = self

        def cdp(method, *args, **kwargs):
            counter.calls[method] = counter.calls.get(method, 0) + 1
            return _ORIGINAL_CDP(method, *args, **kwargs)

        jev_browser.cdp = cdp
        return self

    def uninstall(self):
        jev_browser.cdp = _ORIGINAL_CDP

    @property
    def total(self) -> int:
        return sum(self.calls.values())
