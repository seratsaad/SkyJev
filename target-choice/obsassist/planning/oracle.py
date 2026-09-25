"""The theoretical best night, by search.

Given the ground truth of a night (the weather that actually happened), find the sequence
of actions that maximises the program score. The problem is a time-dependent orienteering
problem with sequence-dependent setup times, so it is solved heuristically, but with
guarantees that matter for evaluation:

* `pilot`  - the pilot method (Duin & Voss 1999): at every decision, try every feasible
  action, complete the night with the greedy policy from each, keep the action whose
  completed night scores best. Never worse than greedy (greedy's own choice is a candidate).
* `beam`   - a beam of pilot states ranked by their rollout value (width B); B=1 is `pilot`.
* `upper_bound` - a fractional-knapsack relaxation (each target at its best S/N rate of the
  night, overheads reduced to one acquisition, no sequencing, windows ignored). No schedule
  can beat it, so `score / upper_bound` is a lower bound on how close to optimal we are.

`hindsight=True` searches against the truth (what could have been done). `hindsight=False`
rolls out under the planner's forecast (what a perfect planner could do without knowing
the future); its first action at a state is the label the decision models learn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from obsassist.planning.planner import (
    Action,
    GreedyPolicy,
    Nowcast,
    apply_action,
    candidates,
    forecast_arrays,
    predict_exposure,
    project,
    run_policy,
)
from obsassist.sim.night import NightModel, ObsState


@dataclass
class OracleResult:
    score: float
    max_score: float
    upper_bound: float
    actions: List[dict]
    state: ObsState
    seconds: float
    method: str
    evaluations: int
    greedy_score: float = 0.0

    def summary(self, model: NightModel) -> dict:
        return {
            "method": self.method,
            "score": round(self.score, 3),
            "greedy": round(self.greedy_score, 3),
            "upper_bound": round(self.upper_bound, 3),
            "max_score": self.max_score,
            "done": int(model.done(self.state).sum()),
            "seconds": round(self.seconds, 2),
            "rollouts": self.evaluations,
        }


def action_space(model: NightModel, state: ObsState, include_wait: bool = True) -> List[Action]:
    """Every action worth considering now: one exposure on any target that can be observed,
    a focus run when the temperature has drifted, a short wait."""
    if not model.dome_open_at(state.t):
        return [Action("closed")]
    acts = []
    now = Nowcast.from_history(model, state.t)
    for c in candidates(model, state, now):
        if c.feasible:
            acts.append(Action("observe", target=c.i, t_exp=c.t_exp))
    if abs(model.temperature(state.t) - state.focus_temp) > 0.8 and state.t - state.last_focus_t > 30:
        acts.append(Action("focus"))
    if include_wait or not acts:
        acts.append(Action("wait", minutes=10.0))
    return acts


def _greedy_value(model: NightModel, state: ObsState, policy) -> float:
    s, _ = run_policy(model, policy, state)
    return model.score(s)


def pilot(
    model: NightModel,
    policy=None,
    state: Optional[ObsState] = None,
    width: int = 1,
    max_steps: int = 400,
    time_budget_s: Optional[float] = None,
) -> OracleResult:
    """Pilot method (width=1) or a beam of pilots (width>1) against the truth."""
    policy = policy or GreedyPolicy()
    t0 = time.time()
    start = state or model.initial_state()
    greedy_score = _greedy_value(model, start, policy)
    beam: List[Tuple[float, ObsState, List[dict]]] = [(greedy_score, start, [])]
    evals = 1
    finished: List[Tuple[float, ObsState, List[dict]]] = []
    for _ in range(max_steps):
        children: List[Tuple[float, ObsState, List[dict]]] = []
        for _, s, acts in beam:
            if model.finished(s):
                finished.append((model.score(s), s, acts))
                continue
            for a in action_space(model, s):
                s2, out = apply_action(model, s, a)
                if s2.t <= s.t:
                    s2 = model.wait(s2, 5.0)
                v = _greedy_value(model, s2, policy)
                evals += 1
                rec = dict(a.to_dict(), t=round(s.t, 2), snr=None if out is None else round(out.snr_counted, 3))
                children.append((v, s2, acts + [rec]))
        if not children:
            break
        # dedupe near-identical states (same time bucket, pointing and progress)
        seen = {}
        for v, s2, acts in sorted(children, key=lambda x: -x[0]):
            key = (round(s2.t / 5), s2.tel_target, tuple(np.round(np.sqrt(s2.snr2) / model.goals, 2)))
            if key not in seen:
                seen[key] = (v, s2, acts)
        beam = list(seen.values())[:width]
        if time_budget_s and time.time() - t0 > time_budget_s:
            break
    for v, s, acts in beam:
        # complete any unfinished beam member greedily
        s_end, log = run_policy(model, policy, s)
        finished.append(
            (model.score(s_end), s_end, acts + [dict(e["action"], t=round(e["t"], 2), snr=e["snr"]) for e in log])
        )
    best = max(finished, key=lambda x: x[0])
    return OracleResult(
        score=best[0],
        max_score=model.max_score(),
        upper_bound=upper_bound(model),
        actions=best[2],
        state=best[1],
        seconds=time.time() - t0,
        method=f"pilot(width={width})",
        evaluations=evals,
        greedy_score=greedy_score,
    )


def upper_bound(model: NightModel) -> float:
    """Fractional knapsack: value w_i, time = the least possible time to reach the goal (best
    S/N rate of the night under the true weather, one acquisition, readouts); capacity = the
    open-dome time of the night. Valid because partial credit is at most proportional to time.

    Capacity covers every minute in which a counted exposure can happen: from civil twilight when
    any target is `twilight_ok` (not only between the program's night limits), plus one readout
    per open-dome interval (a counted exposure may end right at a closure or at the end of the
    night and read out after it). A target counts as done at 0.999 of its goal, so its time is
    charged for (0.999 goal)^2."""
    items = []
    max_read_min = 0.0
    for i, t in enumerate(model.targets):
        vis = np.nonzero(model.observable[i] & model.dome_ok & (model.ephem.t_min >= model.appears[i]))[0]
        if not len(vis):
            continue
        cfg = t.cfg()
        max_read_min = max(max_read_min, cfg.readout_s / 60.0)
        texp = float(t.t_exp or cfg.max_exp_s)
        sig = model.sig[i, 0, vis]
        var = model.var[i, 0, vis]
        # best single-exposure S/N^2 per second achievable at any minute of the night
        s1 = model.unit[i] * sig * texp / np.sqrt(var * texp + model.npix[i, 0, vis] * model.rn2[i])
        rate = (s1**2) / (texp + cfg.readout_s)
        best = float(rate.max())
        if best <= 0:
            continue
        t_need = (0.999 * t.snr_goal) ** 2 / best + min(cfg.acq_s, 150.0)
        items.append((t.w, t_need / 60.0))
    twi = any(t.twilight_ok for t in model.targets)
    t_lo = min(model.t_start, model.t_start_twi) if twi else model.t_start
    t_hi = max(model.t_end, model.t_end_twi) if twi else model.t_end
    # grid minute k covers [k dt, (k+1) dt): count every open minute that overlaps [t_lo, t_hi]
    tm = model.ephem.t_min
    inside = (tm + model.dt > t_lo) & (tm <= t_hi)
    open_in = model.dome_ok & inside
    n_intervals = int(np.count_nonzero(open_in[1:] & ~open_in[:-1]) + (1 if open_in[0] else 0))
    cap = float(open_in.sum() * model.dt) + n_intervals * max_read_min
    ub = 0.0
    for w, tn in sorted(items, key=lambda x: -x[0] / x[1]):
        take = min(1.0, cap / tn) if tn > 0 else 1.0
        ub += w * take
        cap -= take * tn
        if cap <= 0:
            break
    return ub


# ---------------------------------------------------------------------------
# forecast-based best action (labels for learned decision models)
# ---------------------------------------------------------------------------
def best_action_forecast(
    model: NightModel, state: ObsState, now: Nowcast, actions: Optional[Sequence[Action]] = None
) -> Tuple[Action, Dict[int, float]]:
    """The action a perfect planner would take knowing only the telemetry: every candidate
    first action, then the projected rest of the night under the planner's forecast."""
    seeing_f, cloud_f = forecast_arrays(model, now, state.t)
    acts = list(actions) if actions is not None else action_space(model, state)
    values: Dict[int, float] = {}
    best, best_v = acts[0], -1e9
    for k, a in enumerate(acts):
        s = state.copy()
        if a.kind == "observe":
            i = a.target
            oh, _ = model.overhead_to(s, i)
            t0 = s.t + oh / 60.0
            pred = predict_exposure(model, i, t0, a.t_exp, seeing_f, cloud_f)
            if pred.qc_ok:
                s.snr2[i] += pred.snr**2
            cfg = model.targets[i].cfg()
            s.t = t0 + (a.t_exp + cfg.readout_s) / 60.0
            j = model.idx(t0 + a.t_exp / 60.0)  # telescope position when the shutter closes
            s.tel_target, s.acquired, s.setup = i, True, cfg.setup_id
            s.tel_alt, s.tel_az = float(model.geo["alt"][i, j]), float(model.geo["az"][i, j])
        elif a.kind == "focus":
            s.t += model.tel.focus_run_s / 60.0
            s.focus_temp = model.temperature(s.t)
        else:
            s.t += a.minutes or 10.0
        v = project(model, s, now).score
        values[k] = v
        if v > best_v:
            best, best_v = a, v
    return best, values
