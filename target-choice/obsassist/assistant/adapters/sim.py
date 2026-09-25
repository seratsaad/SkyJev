"""Adapter for the simulated console (obsassist console) over its HTTP API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from obsassist.assistant.adapters.base import ActuationRefused, ObservatoryAdapter


class SimConsoleAdapter(ObservatoryAdapter):
    name = "sim-console"

    def __init__(self, base_url: str = "http://127.0.0.1:8765", timeout: float = 10.0):
        self.base = base_url.rstrip("/")
        self.http = httpx.Client(timeout=timeout)
        self._hist: Dict[str, List] = {}

    @property
    def can_actuate(self) -> bool:
        try:
            return bool(self.http.get(f"{self.base}/api/state").json().get("assistant_control"))
        except Exception:
            return False

    def snapshot(self) -> Dict[str, Any]:
        snap = self.http.get(f"{self.base}/api/state").json()
        try:
            snap["history"] = self.history()
        except Exception:
            snap["history"] = self._hist
        return snap

    def history(self) -> Dict[str, List]:
        self._hist = self.http.get(f"{self.base}/api/history").json()
        return self._hist

    def program(self) -> Dict[str, Any]:
        return self.http.get(f"{self.base}/api/program").json()

    def frames(self) -> List[Dict[str, Any]]:
        return self.http.get(f"{self.base}/api/frames").json()

    def frame_ql(self, name: Optional[str] = None) -> Dict[str, Any]:
        if name is None:
            fr = self.frames()
            if not fr:
                return {"error": "no frames yet"}
            name = fr[-1]["file"]
        r = self.http.get(f"{self.base}/api/frame/{name}/ql")
        d = r.json()
        d["file"] = name
        return d

    def etc(
        self, target: str, seeing: Optional[float] = None, cloud: Optional[float] = None, t_exp: Optional[float] = None
    ) -> Dict[str, Any]:
        params = {"target": target}
        for k, v in (("seeing", seeing), ("cloud", cloud), ("t_exp", t_exp)):
            if v is not None:
                params[k] = v
        r = self.http.get(f"{self.base}/api/etc", params=params)
        return r.json() if r.status_code == 200 else {"error": r.text}

    def instructions(self, target: str, t_exp: float, n_exp: int) -> List[str]:
        """Console commands; one exposure at a time (the assistant re-decides after each readout)."""
        return [
            f"goto {target}",
            "all type object",
            f"all object {target}",
            f"all exptime {t_exp:.0f}",
            "all loops 1",
            "start all",
        ]

    def send(self, command: str, confirm: bool = False) -> str:
        if not confirm:
            raise ActuationRefused("commands need explicit confirmation")
        r = self.http.post(f"{self.base}/api/cmd", json={"cmd": command, "source": "assistant"}).json()
        reply = r.get("reply", "")
        if reply.startswith("refused"):
            raise ActuationRefused(reply)
        return reply

    def announce(self, url: str) -> None:
        """Tell the console where this assistant runs (its Assistant tab shows the trace)."""
        try:
            self.http.post(f"{self.base}/api/sim", json={"assistant_url": url})
        except httpx.HTTPError:
            pass  # the console is not up yet: its tab says how to start an assistant

    def describe(self) -> Dict[str, Any]:
        return {"adapter": self.name, "url": self.base, "can_actuate": self.can_actuate}
