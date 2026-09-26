"""Minimal synchronous Chrome DevTools client used for independent verification.

Independent of both agents: it talks to the Chrome on BU_CDP_URL directly, so the verifier
reads the page the same way whether JEV or Browser Use drove it.
"""

from __future__ import annotations

import itertools
import json
import os
import time
import urllib.request

from websockets.sync.client import connect

CDP_URL = os.environ.get("BU_CDP_URL", "http://127.0.0.1:9222")


def browser_ws() -> str:
    with urllib.request.urlopen(f"{CDP_URL}/json/version", timeout=5) as r:
        return json.load(r)["webSocketDebuggerUrl"]


def list_targets() -> list[dict]:
    with urllib.request.urlopen(f"{CDP_URL}/json/list", timeout=5) as r:
        return [t for t in json.load(r) if t.get("type") == "page"]


class CdpTimeout(RuntimeError):
    """A browser call did not answer in time. The page is unresponsive, not the agent at fault."""


class Session:
    """One attached page target."""

    def __init__(self, target_id: str | None = None, url: str | None = None, settle: float = 1.5,
                 timeout: float = 20.0):
        # Every read is bounded. A blocking recv with no deadline turns one unresponsive page into a
        # hung batch: this client once sat on a single call for four hours while the runs it was
        # meant to verify never happened.
        self.timeout = timeout
        self.ws = connect(browser_ws(), max_size=64 * 1024 * 1024, open_timeout=timeout,
                          close_timeout=timeout)
        self.ids = itertools.count(1)
        self.owned = target_id is None
        if target_id is None:
            target_id = self.send("Target.createTarget", url="about:blank", background=True)["targetId"]
        self.target_id = target_id
        self.session_id = self.send("Target.attachToTarget", targetId=target_id, flatten=True)["sessionId"]
        if url:
            self.navigate(url, settle=settle)

    def send(self, method, session=False, timeout: float | None = None, **params):
        deadline = time.monotonic() + (timeout or self.timeout)
        msg = {"id": next(self.ids), "method": method, "params": params}
        if session:
            msg["sessionId"] = self.session_id
        self.ws.send(json.dumps(msg))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CdpTimeout(f"{method} did not answer within {timeout or self.timeout:g}s")
            try:
                reply = json.loads(self.ws.recv(timeout=remaining))
            except TimeoutError:
                raise CdpTimeout(f"{method} did not answer within {timeout or self.timeout:g}s") from None
            if reply.get("id") == msg["id"]:
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error']}")
                return reply.get("result", {})

    def evaluate(self, expression: str):
        r = self.send("Runtime.evaluate", session=True, expression=expression, returnByValue=True, awaitPromise=True)
        if r.get("exceptionDetails"):
            raise RuntimeError(r["exceptionDetails"].get("text", "js exception"))
        return r.get("result", {}).get("value")

    def navigate(self, url: str, settle: float = 1.5, timeout: float = 30):
        self.send("Page.navigate", session=True, url=url)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self.evaluate("document.readyState") == "complete":
                    break
            except RuntimeError:
                pass
            time.sleep(0.05)
        time.sleep(settle)

    def url(self) -> str:
        return self.evaluate("location.href")

    def text(self) -> str:
        return self.evaluate("document.body ? document.body.innerText : ''") or ""

    def close(self):
        try:
            if self.owned:
                self.send("Target.closeTarget", targetId=self.target_id)
        finally:
            self.ws.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def find_target(url: str) -> str | None:
    for t in list_targets():
        if t["url"] == url:
            return t["id"]
    return None
