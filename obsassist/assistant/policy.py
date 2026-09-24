"""The assistant's decision loop: monitor, decide, escalate, recommend, (optionally) act.

Tiers, cheapest first:
  planner   - deterministic ETC + merit + projection. It chooses the next action: on held-out
              simulated nights it reaches 0.945 of the hindsight optimum and its single-step
              choices have lower regret than the learned choosers tested (README, Results);
  System 1  - AnyJev over a local LLM: calibrated per-candidate expected regret and P(best), sky
              transparency and dome-closure risk. When it disagrees with the planner the call goes
              to System 2; it may override the planner only above `s1_override_tau` (off by default);
  System 2  - GPT (OpenAI) with tools: a second opinion on disputed calls, questions, night log.

A recommendation carries the exact instructions for the adapter in use. In "advise" mode the
observer accepts or ignores it; in "autopilot" it is executed at once when (a) the adapter can act
(the console has assistant control enabled) and (b) it is an observe action. A System 2 answer
replaces an open recommendation; aborting exposures always needs a human. Every decision is logged
with its source, level, confidence and latency, and every exchange behind it (what the console
reported, the prompts and answers of both systems, the commands and replies) goes to the trace.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from obsassist.assistant.adapters.base import ActuationRefused, ObservatoryAdapter
from obsassist.assistant.observation import from_snapshot, render
from obsassist.assistant.trace import Trace

# the planner's numbers for each feasible candidate, as the trace shows them
PLANNER_FIELDS = (
    "name",
    "priority",
    "merit",
    "efficiency",
    "urgency",
    "t_exp",
    "n_needed",
    "time_needed_min",
    "window_left_min",
    "airmass",
    "fwhm_pred",
    "max_seeing",
    "snr_now",
    "goal",
)


@dataclass
class Settings:
    mode: str = "advise"  # advise | autopilot
    # System 1 may override the planner on its own only when P(best) >= s1_override_tau. Off by
    # default: on held-out decisions the planner's pick has lower regret than System 1's
    # (reports/fit_qwen3-1.7b.json); System 1 disagreeing sends the call to System 2 instead.
    s1_override_tau: float = 1.01
    escalate: bool = True  # use System 2 when System 1 is unsure
    s2_effort: str = "low"
    step_s: float = 1.0
    humidity_warn_margin: float = 8.0  # % RH below the limit
    wind_warn_margin_mph: float = 6.0


@dataclass
class Recommendation:
    id: int
    utc: str
    action: str  # observe | wait | focus | stop | abort | none
    target: str = ""
    t_exp: float = 0.0
    n_exp: int = 0
    source: str = "planner"  # planner | system1 | system2
    level: str = ""
    confidence: Optional[float] = None
    rationale: str = ""
    risk: str = ""
    commands: List[str] = field(default_factory=list)
    planner_pick: str = ""
    s1_pick: str = ""
    s2_pick: str = ""
    latency_s: Dict[str, float] = field(default_factory=dict)
    status: str = "open"  # open | accepted | executed | dismissed | superseded
    state_key: str = ""


class Assistant:
    def __init__(
        self,
        adapter: ObservatoryAdapter,
        s1=None,
        s2=None,
        settings: Optional[Settings] = None,
        trace: Optional[Trace] = None,
    ):
        self.adapter = adapter
        self.s1 = s1
        self.s2 = s2
        self.settings = settings or Settings()
        self.trace = trace or Trace()
        self.trace.clock = lambda: (self.snap.get("utc") or "")[11:19]
        for tier in (s1, s2):
            if tier is not None:
                tier.trace = self.trace
        self.rec: Optional[Recommendation] = None
        self.history: List[Recommendation] = []
        self.alerts: List[Dict[str, Any]] = []
        self.judgements: List[Dict[str, Any]] = []
        self.state_answers: Dict[str, Any] = {}
        self.snap: Dict[str, Any] = {}
        self.obs: Dict[str, Any] = {}
        self._n = 0
        self._seen_frames: set = set()
        self._alert_keys: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._s2_busy = False
        self.autopilot: Dict[str, Any] = {"stage": "idle"}
        self.errors: List[str] = []
        self.stats = {"steps": 0, "s1_calls": 0, "s2_calls": 0, "executed": 0, "agree_s1_planner": 0, "decisions": 0}

    # ------------------------------------------------------------------ helpers
    def _alert(self, key: str, text: str, level: str = "warn", every_min: float = 20.0):
        t = self.snap.get("t", 0.0)
        last = self._alert_keys.get(key)
        if last is not None and t - last < every_min:
            return
        self._alert_keys[key] = t
        self.alerts.append({"utc": self.snap.get("utc", "")[11:19], "key": key, "text": text, "level": level})
        self.alerts = self.alerts[-80:]

    def _arms_idle(self) -> bool:
        arms = self.snap.get("arms") or {}
        return all(a.get("state") == "idle" for a in arms.values()) if arms else True

    def _decision_point(self) -> bool:
        s = self.snap
        if not s.get("dome", {}).get("open", True):
            return False
        if s.get("night", {}).get("minutes_left", 0) <= 0:
            return False
        if s.get("tcs", {}).get("state") in ("slewing", "acquiring"):
            return False
        return self._arms_idle()

    def _state_key(self) -> str:
        """Re-decide when the telescope, the progress, the dome, the set of feasible targets
        or a 15-minute time bucket changes."""
        s = self.snap
        prog = tuple((t["name"], t.get("n_exp", 0)) for t in s.get("targets", []))
        feas = sorted(c["name"] for c in (s.get("plan") or {}).get("candidates", []) if c.get("feasible"))
        bucket = int((s.get("t") or 0) // 15)
        return json.dumps([s.get("tcs", {}).get("name"), prog, s.get("dome", {}).get("open"), feas, bucket])

    # ------------------------------------------------------------------ monitors
    def monitor(self):
        s, st = self.snap, self.settings
        w = s.get("weather", {})
        lim = w.get("limits") or {}
        if w.get("humidity") is not None and lim.get("humidity"):
            if w["humidity"] >= lim["humidity"] - st.humidity_warn_margin:
                self._alert(
                    "humidity",
                    f"Humidity {w['humidity']:.0f}% is within {st.humidity_warn_margin:.0f}% of the "
                    f"{lim['humidity']:.0f}% limit: avoid starting long exposures; be ready to close.",
                )
        if w.get("wind_mph") is not None and lim.get("wind_close_mph"):
            if w["wind_mph"] >= lim["wind_close_mph"] - st.wind_warn_margin_mph:
                self._alert(
                    "wind",
                    f"Wind {w['wind_mph']:.0f} mph, close limit {lim['wind_close_mph']:.0f} mph. "
                    f"Point away from the wind ({w.get('wind_dir', 0):.0f} deg) if you can.",
                )
        fr = (s.get("guider") or {}).get("flux_ratio")
        if fr is not None and (s.get("guider") or {}).get("state") == "guiding" and fr < 0.5:
            self._alert(
                "clouds",
                f"Guider flux at {fr:.2f} of clear sky: clouds. Consider a cloud-tolerant/bright target "
                f"or pausing a faint-target exposure.",
            )
        if (s.get("focus") or {}).get("dT", 0) > 1.5:
            self._alert(
                "focus",
                f"Temperature changed {s['focus']['dT']:.1f} C since the last focus: consider a focus run.",
                level="info",
                every_min=45,
            )
        left = s.get("night", {}).get("minutes_left", 999)
        if 0 < left < 40:
            self._alert(
                "dawn",
                f"{left:.0f} min of dark time left: short, bright targets and the end-of-night standard.",
                level="info",
                every_min=60,
            )
        # exposing on a seeing-limited target in bad seeing
        tcs_name = s.get("tcs", {}).get("name")
        tgt = next((t for t in s.get("targets", []) if t["name"] == tcs_name), None)
        g = (s.get("guider") or {}).get("fwhm")
        if tgt and tgt.get("max_seeing") and g and not self._arms_idle() and g > tgt["max_seeing"] * 1.1:
            self._alert(
                "seeing_" + tgt["name"],
                f'Guider FWHM {g:.2f}" exceeds {tgt["name"]}\'s {tgt["max_seeing"]}" limit: '
                f"these frames may not count. Consider stopping and switching target.",
            )
        for ev in (s.get("log") or [])[-10:]:
            if ev.get("who") == "ALERT":
                self._alert("too_" + ev["text"][:30], ev["text"], level="alert", every_min=600)
        # frame quick-look flags
        for fr_rec in s.get("frames", [])[-6:]:
            name = fr_rec.get("file")
            if name in self._seen_frames or not ({"ql", "ql_flags", "ql_error", "qc_reasons"} & set(fr_rec)):
                continue
            self._seen_frames.add(name)
            flags = list(fr_rec.get("ql_flags") or (fr_rec.get("ql") or {}).get("flags") or [])
            if fr_rec.get("qc_reasons"):
                flags += fr_rec["qc_reasons"]
            if flags:
                self._alert("frame_" + name, f"{name} ({fr_rec.get('object', '')}): " + "; ".join(flags), every_min=0)

    # ------------------------------------------------------------------ System 1 state questions
    def read_state(self, every_min: float = 10.0):
        """Sky transparency and dome-closure risk from System 1, every ~10 min."""
        if self.s1 is None or not self.obs:
            return
        t = self.snap.get("t", 0.0)
        if self.state_answers and t - self.state_answers.get("_t", -1e9) < every_min:
            return
        try:
            ans = self.s1.state_questions(self.obs)
        except Exception as e:
            self.errors.append(f"System 1 state: {type(e).__name__}: {e}")
            return
        ans["_t"] = t
        self.state_answers = ans
        dr = ans.get("dome_risk", {})
        if dr.get("level") == "L2" and (dr.get("p_true") or 0) >= 0.5:
            self._alert(
                "s1_dome",
                f"System 1: the dome is likely to close within 30 min (p={dr['p_true']:.2f}). "
                f"Prefer short exposures or targets close to done.",
            )

    # ------------------------------------------------------------------ decisions
    def decide(self):
        s = self.snap
        obs = self.obs
        feas = [c for c in obs["candidates"] if c.get("feasible")]
        key = self._state_key()
        if self.rec and self.rec.state_key == key and self.rec.status == "open":
            return
        self._n += 1
        rec = Recommendation(id=self._n, utc=s.get("utc", "")[11:19], action="none", state_key=key)
        self.trace.add("observation", None, obs, decision=rec.id, summary=f"{len(feas)} feasible candidates")
        if not feas:
            rec.action, rec.rationale = "wait", "Nothing observable meets its constraints right now."
            self._set(rec)
            return
        planner = max(feas, key=lambda c: c.get("merit") or 0)
        rec.planner_pick = planner["name"]
        self.trace.add(
            "planner",
            [{k: c.get(k) for k in PLANNER_FIELDS} for c in sorted(feas, key=lambda c: -(c.get("merit") or 0))],
            {"pick": planner["name"], "merit": planner.get("merit")},
            decision=rec.id,
            summary=f"highest merit: {planner['name']}",
        )
        pick, conf, level, src = planner, None, "", "planner"
        disagree = False
        if self.s1 is not None and len(feas) > 1:
            t0 = time.time()
            try:
                top = sorted(feas, key=lambda c: -(c.get("merit") or 0))[:8]
                js = self.s1.judge_candidates(obs, [c["name"] for c in top], decision_id=rec.id)
                self.stats["s1_calls"] += 1
                rec.latency_s["system1"] = round(time.time() - t0, 2)
                self.judgements = [
                    dict(j.to_dict(), merit=next(c.get("merit") for c in top if c["name"] == j.name)) for j in js
                ]
                best = min(js, key=lambda j: j.expected_regret)
                mine = next(j for j in js if j.name == planner["name"])
                rec.s1_pick, level = best.name, best.level
                disagree = best.name != planner["name"]
                conf = mine.p_best
                if disagree and best.level == "L2" and (best.p_best or 0) >= self.settings.s1_override_tau:
                    pick, conf, src = next(c for c in feas if c["name"] == best.name), best.p_best, "system1"
            except Exception as e:
                self.errors.append(f"System 1: {type(e).__name__}: {e}")
        self.stats["decisions"] += 1
        if rec.s1_pick and not disagree:
            self.stats["agree_s1_planner"] += 1
        rec.action, rec.target, rec.t_exp, rec.n_exp = (
            "observe",
            pick["name"],
            pick["t_exp"],
            max(1, int(pick["n_needed"] or 1)),
        )
        rec.source, rec.level, rec.confidence = src, level, conf
        rec.rationale = self._explain(pick, planner, rec.s1_pick, conf, level)
        rec.commands = self.commands_for(pick["name"], pick["t_exp"], rec.n_exp)
        self._set(rec)
        self.trace.add("recommendation", None, rec, decision=rec.id, summary=f"{rec.source}: observe {rec.target}")
        # a second opinion when System 1 disagrees with the planner (or cannot judge), if there is a choice
        if (
            len(feas) > 1
            and (disagree or level != "L2")
            and self.settings.escalate
            and self.s2 is not None
            and self.s2.enabled
            and not self._s2_busy
        ):
            threading.Thread(target=self._escalate, args=(rec,), daemon=True).start()

    def _explain(self, pick, planner, s1_pick, conf, level) -> str:
        bits = [
            f"{pick['name']} (P{pick['priority']}): {pick['n_needed']} x {pick['t_exp']:.0f} s "
            f"~{pick['time_needed_min']:.0f} min, {pick['window_left_min']:.0f} min left, X={pick['airmass']:.2f}"
        ]
        if s1_pick and s1_pick != pick["name"]:
            bits.append(f"System 1 would pick {s1_pick}; asking System 2")
        elif s1_pick:
            bits.append("System 1 agrees")
        if conf is not None:
            bits.append(f"P(best)={conf:.2f} [{level}]")
        return "; ".join(bits)

    def _escalate(self, rec: Recommendation):
        self._s2_busy = True
        t0 = time.time()
        try:
            ctx = (
                "Decide the next action for the observer.\n\nState (as System 1 sees it):\n"
                + render(self.obs)
                + "\n\nSystem 1 (fast model) judgements, lower expected regret is better:\n"
                + json.dumps(self.judgements, default=str)
                + f"\n\nThe deterministic planner's pick: {rec.planner_pick}. System 1's pick: {rec.s1_pick or 'n/a'} "
                f"(P(best)={rec.confidence}). Check the numbers with the tools before recommending."
            )
            out = self.s2.deliberate(ctx, effort=self.settings.s2_effort, tag=rec.id)
            self.stats["s2_calls"] += 1
            with self._lock:
                rec.latency_s["system2"] = round(time.time() - t0, 1)
                names = {c["name"] for c in self.obs["candidates"] if c.get("feasible")}
                tgt = out.get("target", "")
                rec.s2_pick = tgt
                if out.get("action") in ("observe", "continue") and tgt in names and rec.status == "open":
                    cand = next(c for c in self.obs["candidates"] if c["name"] == tgt)
                    # guardrail: exposure length from the ETC-backed planner, never from the LLM alone
                    t_exp = cand["t_exp"]
                    if out.get("t_exp") and 0.5 * cand["t_exp"] <= float(out["t_exp"]) <= 2.0 * cand["t_exp"]:
                        t_exp = float(out["t_exp"])
                    rec.target, rec.t_exp = tgt, t_exp
                    rec.n_exp = max(1, int(cand["n_needed"] or 1))
                    rec.commands = self.commands_for(tgt, t_exp, rec.n_exp)
                    rec.source, rec.confidence, rec.level = "system2", float(out.get("confidence", 0.0)), "LLM"
                    rec.rationale = out.get("rationale", "")
                    rec.risk = out.get("risk", "")
                elif rec.status == "open":
                    rec.risk = (
                        f"System 2 suggested '{out.get('action')} {tgt}' (not applied: "
                        f"{'not observable' if tgt and tgt not in names else 'not an observe action'}). "
                        + out.get("rationale", "")
                    )
                if rec.source == "system2":
                    verdict = "applied"
                elif tgt and tgt == rec.target:
                    verdict = "agrees"
                else:
                    verdict = f"not applied (recommendation already {rec.status})"
                self.trace.add(
                    "recommendation",
                    {k: v for k, v in out.items() if k != "notes"},
                    rec,
                    decision=rec.id,
                    summary=f"System 2 picks {tgt or out.get('action')}: {verdict}",
                )
        except Exception as e:
            self.errors.append(f"System 2: {type(e).__name__}: {e}")
        finally:
            self._s2_busy = False

    def _set(self, rec: Recommendation):
        with self._lock:
            if self.rec and self.rec.status == "open":
                self.rec.status = "superseded"
            self.rec = rec
            self.history.append(rec)
            self.history = self.history[-200:]

    # ------------------------------------------------------------------ commands and autopilot
    def commands_for(self, target: str, t_exp: float, n_exp: int) -> List[str]:
        return self.adapter.instructions(target, t_exp, n_exp)

    def _send(self, command: str) -> str:
        t0 = time.time()
        try:
            reply = self.adapter.send(command, confirm=True)
        except ActuationRefused as e:
            reply = f"refused: {e}"
            raise
        finally:
            self.trace.add(
                "console",
                {"cmd": command, "source": "assistant"},
                {"reply": reply},
                decision=self.autopilot.get("rec_id"),
                summary=command,
                latency_s=time.time() - t0,
            )
        return reply

    def accept(self, execute: bool = True) -> str:
        rec = self.rec
        if rec is None or rec.status != "open":
            return "no open recommendation"
        rec.status = "accepted"
        if execute and self.adapter.can_actuate:
            self.autopilot = {
                "stage": "goto",
                "rec_id": rec.id,
                "target": rec.target,
                "commands": rec.commands,
                "t": time.time(),
            }
            return "executing"
        return "accepted: please carry it out at the console"

    def dismiss(self) -> str:
        if self.rec and self.rec.status == "open":
            self.rec.status = "dismissed"
        return "dismissed"

    def _run_autopilot(self):
        """Advance the autopilot state machine as far as the current snapshot allows."""
        for _ in range(4):
            before = self.autopilot.get("stage")
            self._autopilot_once()
            if self.autopilot.get("stage") == before:
                break

    def _autopilot_once(self):
        ap = self.autopilot
        s = self.snap
        if ap.get("stage") == "idle":
            rec = self.rec
            if (
                self.settings.mode == "autopilot"
                and rec
                and rec.status == "open"
                and rec.action == "observe"
                and self.adapter.can_actuate
                and self._decision_point()
                and rec.source in ("planner", "system1", "system2")
            ):
                rec.status = "accepted"
                self.autopilot = {
                    "stage": "goto",
                    "rec_id": rec.id,
                    "target": rec.target,
                    "commands": rec.commands,
                    "t": time.time(),
                }
            return
        try:
            if ap["stage"] == "goto":
                if s.get("tcs", {}).get("name") == ap["target"] and s.get("tcs", {}).get("state") == "guiding":
                    ap["stage"] = "configure"
                else:
                    self._send(ap["commands"][0])
                    ap["stage"] = "wait_guiding"
            elif ap["stage"] == "wait_guiding":
                self.snap = s = self.adapter.snapshot()
                st = s.get("tcs", {}).get("state")
                if st == "guiding" and s.get("tcs", {}).get("name") == ap["target"]:
                    ap["stage"] = "configure"
                elif st in ("limit", "parked"):
                    ap["stage"] = "idle"
                    self._alert("autopilot", f"Autopilot: could not acquire {ap['target']} ({st}).")
            elif ap["stage"] == "configure":
                for c in ap["commands"][1:]:
                    self._send(c)
                ap["stage"] = "exposing"
                ap["started"] = s.get("t")
                self.stats["executed"] += 1
                if self.rec and self.rec.id == ap.get("rec_id"):
                    self.rec.status = "executed"
            elif ap["stage"] == "exposing" and self._arms_idle() and s.get("t", 0) > (ap.get("started") or 0) + 0.2:
                ap["stage"] = "idle"
        except ActuationRefused as e:
            self._alert("autopilot_refused", f"Autopilot stopped: {e}")
            self.autopilot = {"stage": "idle"}

    # ------------------------------------------------------------------ main step
    def step(self):
        try:
            self.snap = self.adapter.snapshot()
            if "error" in self.snap:
                self.errors.append(self.snap["error"])
                return
            self.obs = from_snapshot(self.snap)
            self.stats["steps"] += 1
            self.monitor()
            self.read_state()
            self._run_autopilot()  # finish an exposure sequence first
            if self.autopilot.get("stage", "idle") == "idle" and self._decision_point():
                self.decide()
                self._run_autopilot()  # and act on a fresh decision at once
        except Exception as e:
            self.errors.append(f"step: {type(e).__name__}: {e}")
            self.errors = self.errors[-50:]
            traceback.print_exc()

    def status(self) -> Dict[str, Any]:
        s = self.snap
        return {
            "utc": s.get("utc"),
            "local": s.get("local"),
            "site": s.get("site"),
            "telescope": s.get("telescope"),
            "program": s.get("program"),
            "twilight": s.get("twilight"),
            "minutes_left": (s.get("night") or {}).get("minutes_left"),
            "weather": s.get("weather"),
            "dome": s.get("dome"),
            "tcs": s.get("tcs"),
            "guider": s.get("guider"),
            "arms": {
                k: {kk: v.get(kk) for kk in ("state", "exp_time", "loops", "doing", "message", "object")}
                for k, v in (s.get("arms") or {}).items()
            },
            "score": s.get("score"),
            "max_score": s.get("max_score"),
            "targets": [
                {k: t.get(k) for k in ("name", "priority", "snr", "goal", "done", "visible", "airmass", "n_exp")}
                for t in s.get("targets", [])
            ],
            "projection": {k: v for k, v in ((s.get("plan") or {}).get("projection") or {}).items() if k != "blocks"},
            "recommendation": asdict(self.rec) if self.rec else None,
            "judgements": self.judgements,
            "alerts": self.alerts[-30:],
            "state_answers": {k: v for k, v in self.state_answers.items() if not k.startswith("_")},
            "history": [asdict(r) for r in self.history[-40:]],
            "settings": asdict(self.settings),
            "autopilot": self.autopilot,
            "stats": self.stats,
            "s1": self.s1.info() if self.s1 else None,
            "s2": (
                {
                    "enabled": self.s2.enabled,
                    "chat_model": self.s2.chat_model,
                    "deliberate_model": self.s2.deliberate_model,
                    "usage": self.s2.usage,
                    "busy": self._s2_busy,
                }
                if self.s2
                else None
            ),
            "adapter": self.adapter.describe(),
            "errors": self.errors[-8:],
            "state_text": render(self.obs) if self.obs else "",
        }
