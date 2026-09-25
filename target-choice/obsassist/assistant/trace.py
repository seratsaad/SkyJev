"""The assistant's audit trail: every exchange, in and out, as JSON.

One entry per exchange, tagged with the decision it belongs to:

    observation     what the console reported, as the assistant reads it (the input to all tiers)
    planner         the feasible candidates with the planner's numbers -> its pick
    system1         AnyJev: the exact prompts the local model read -> probabilities per option,
                    calibration level, the head and layer that answered
    system2         one OpenAI Responses API call: the request -> the response
    tool            a tool System 2 called: arguments -> result
    console         a command sent to the console -> its reply
    recommendation  what the assistant recommends (again when System 2 changes it)

Entries are kept in memory for the UI (GET /api/trace) and appended to a JSON-lines file in
data/traces/, so a night can be audited afterwards.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import threading
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

MAX_STR = 20000  # longer strings are cut (a frame's spectrum does not belong in a trace)


def jsonable(x: Any) -> Any:
    """Plain JSON types: numpy, dataclasses and SDK objects converted, NaN/inf -> None."""
    if isinstance(x, float):
        return x if x == x and abs(x) != float("inf") else None
    if isinstance(x, (str, int, bool)) or x is None:
        return x[:MAX_STR] + "…" if isinstance(x, str) and len(x) > MAX_STR else x
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, deque)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return jsonable(dataclasses.asdict(x))
    if hasattr(x, "model_dump"):  # OpenAI SDK (pydantic) objects
        return jsonable(x.model_dump(exclude_none=True))
    return str(x)


class Trace:
    def __init__(self, path: Optional[Path] = None, keep: int = 3000):
        self.path = path
        self.clock: Callable[[], str] = lambda: ""  # the night's UT, set by the assistant
        self._items: deque = deque(maxlen=keep)
        self._n = 0
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        kind: str,
        inp: Any = None,
        out: Any = None,
        *,
        decision: Optional[int] = None,
        summary: str = "",
        latency_s: Optional[float] = None,
    ) -> int:
        with self._lock:
            self._n += 1
            e = {
                "id": self._n,
                "ut": self.clock(),
                "wall": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds"),
                "kind": kind,
                "decision": decision,
                "summary": summary,
                "latency_s": None if latency_s is None else round(latency_s, 3),
                "in": jsonable(inp),
                "out": jsonable(out),
            }
            self._items.append(e)
            if self.path is not None:
                with self.path.open("a") as f:
                    f.write(json.dumps(e) + "\n")
            return self._n

    def since(self, after: int = 0, limit: int = 400) -> List[Dict[str, Any]]:
        with self._lock:
            return [e for e in self._items if e["id"] > after][:limit]
