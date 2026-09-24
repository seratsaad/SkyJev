"""The assistant talks to an observatory only through an adapter.

Every adapter produces a *snapshot* in the console's format (see console/engine.py:snapshot):
time, conditions, dome, TCS, instrument arms, targets with progress, the planner's candidates
and projection, the log and the latest frames. What differs is where the numbers come from and
whether the assistant may act:

    SimConsoleAdapter  - the simulated console over HTTP (can act if the console allows it)
    ManualAdapter      - any real telescope: the observer tells the assistant what happened
                         (or a FITS watcher sees new frames); the assistant only advises
                         (at Keck it can print KTL command lines and read DCS telemetry, adapters/ktl.py)

`send()` is the only path to actuation; it is refused unless `can_actuate` is true AND the
caller passes `confirm=True`. The assistant never sends commands on its own unless the
observer switched on autopilot in the assistant AND control is enabled on the console side.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ActuationRefused(RuntimeError):
    pass


class ObservatoryAdapter:
    name = "base"
    can_actuate = False

    def snapshot(self) -> Dict[str, Any]:
        raise NotImplementedError

    def history(self) -> Dict[str, List]:
        return {}

    def frames(self) -> List[Dict[str, Any]]:
        return []

    def frame_ql(self, name: Optional[str] = None) -> Dict[str, Any]:
        return {}

    def etc(
        self, target: str, seeing: Optional[float] = None, cloud: Optional[float] = None, t_exp: Optional[float] = None
    ) -> Dict[str, Any]:
        return {"error": "ETC not available through this adapter"}

    def instructions(self, target: str, t_exp: float, n_exp: int) -> List[str]:
        """How to carry out 'observe target now' at this observatory: console commands for an
        adapter that can act, plain steps for the observer and the operator otherwise."""
        return [
            f"Operator: go to {target}",
            f"Instrument: Object '{target}', Exp.Time {t_exp:.0f} s, Loops 1, Start",
            f"(plan: {n_exp} exposure(s) in total at the current conditions; the assistant re-checks after each)",
        ]

    def send(self, command: str, confirm: bool = False) -> str:
        raise ActuationRefused(f"{self.name}: this adapter cannot send commands")

    def describe(self) -> Dict[str, Any]:
        return {"adapter": self.name, "can_actuate": self.can_actuate}
