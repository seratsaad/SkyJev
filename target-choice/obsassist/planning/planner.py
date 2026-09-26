"""Planning from what the observer can know: a nowcast from telemetry, a forecast that relaxes
toward the night's own conditions, predicted S/N from the planning tables, a greedy merit
policy, a queue observer's rule (priority first), and a projection of the rest of the night
(per-target finishing times, the list completion time).

The planner never reads the weather truth. It sees the DIMM, the guider flux ratio and the
S/N it measured on its own exposures, as an observer does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from obsassist.sim.night import SEEING_LEVELS, NightModel, ObsState


# ---------------------------------------------------------------------------
# nowcast and forecast
# ---------------------------------------------------------------------------
@dataclass
class Nowcast:
    seeing: float  # DIMM, 500 nm, zenith (arcsec), last ~15 min
    cloud: float  # grey extinction estimate (mag), latest
    seeing_trend: float = 0.0  # d ln(seeing) / hour over the last ~30 min
    humidity: float = 0.0
    wind_ms: float = 0.0
    dome_ok: bool = True
    night_seeing: Optional[float] = None  # median DIMM since sunset (the night's level so far)
    night_hours: float = 0.0  # hours of DIMM data behind night_seeing
    night_cloud: float = 0.0  # mean cloud over the last two hours
    source: str = "telemetry"

    @staticmethod
    def from_history(
        model: NightModel, t_min: float, lookback_min: float = 15.0, last_flux_ratio: Optional[float] = None
    ) -> "Nowcast":
        """Median of the recent DIMM reports; cloud from the most recent guider flux ratio; the
        night's level so far (DIMM median since sunset, cloud mean over two hours)."""
        j1 = model.idx(t_min)
        j0 = max(0, j1 - int(lookback_min / model.dt))
        d = model.weather.seeing_dimm[j0 : j1 + 1]
        d = d[np.isfinite(d)]
        s = float(np.median(d)) if len(d) else model.site.seeing_median
        jt = max(0, j1 - int(30 / model.dt))
        dd = model.weather.seeing_dimm[jt : j1 + 1]
        ok = np.isfinite(dd)
        trend = 0.0
        if ok.sum() > 5:
            x = np.arange(len(dd))[ok] * model.dt / 60.0
            trend = float(np.polyfit(x, np.log(dd[ok]), 1)[0])
        night = model.weather.seeing_dimm[: j1 + 1]
        night = night[np.isfinite(night)]
        rng = np.random.default_rng(int(t_min * 7919) % (2**32))
        if last_flux_ratio is None:
            last_flux_ratio = 10 ** (-0.4 * model.weather.cloud_mag[j1]) * np.exp(rng.normal(0, 0.03))
        cloud = float(max(0.0, -2.5 * np.log10(max(last_flux_ratio, 1e-3))))
        # two-hour cloud history as the guider / all-sky camera measures it: flux ratios with 3 % noise
        jc = max(0, j1 - int(120 / model.dt))
        hist = 10 ** (-0.4 * model.weather.cloud_mag[jc : j1 + 1]) * np.exp(rng.normal(0, 0.03, j1 + 1 - jc))
        night_cloud = float(max(0.0, -2.5 * np.log10(max(float(np.mean(hist)), 1e-3))))
        w = model.weather
        return Nowcast(
            seeing=s,
            cloud=cloud,
            seeing_trend=trend,
            humidity=float(w.humidity[j1]),
            wind_ms=float(w.wind_ms[j1]),
            dome_ok=bool(model.dome_ok[j1]),
            night_seeing=float(np.median(night)) if len(night) else None,
            night_hours=len(night) * model.dt / 60.0,
            night_cloud=night_cloud,
        )


FORECAST_SHRINK_H = 1.0  # hours of DIMM data at which the night's own level gets half the weight


def forecast_arrays(model: NightModel, now: Nowcast, t_min: float) -> Tuple[np.ndarray, np.ndarray]:
    """Seeing and cloud expected at every grid minute from t_min on.

    Seeing relaxes (log space, the site's correlation time) from its current value toward the
    night's own level: the DIMM median since sunset, shrunk toward the site median while little
    of the night has been seen (weight h / (h + 1 h)). Relaxing toward the site median instead
    makes every poor night look like it will improve, which biases projections high. Cloud
    relaxes from its current value toward its two-hour mean over ~1.5 h."""
    n = model.ephem.n
    h = np.maximum(model.ephem.t_min - t_min, 0.0)
    site = model.site
    level = np.log(site.seeing_median)
    if now.night_seeing:
        wt = now.night_hours / (now.night_hours + FORECAST_SHRINK_H)
        level = wt * np.log(now.night_seeing) + (1.0 - wt) * level
    decay = np.exp(-h / site.seeing_tau_min)
    s = np.exp(level + (np.log(max(now.seeing, 0.2)) - level) * decay)
    c = now.night_cloud + (now.cloud - now.night_cloud) * np.exp(-h / 90.0)
    return s[:n], np.maximum(c[:n], 0.0)


def _interp_levels(tab: np.ndarray, seeing: np.ndarray, j: np.ndarray) -> np.ndarray:
    """tab[S, n] sampled at seeing[j] (log interpolation across the seeing levels)."""
    x = np.log(np.clip(seeing, SEEING_LEVELS[0], SEEING_LEVELS[-1]))
    lx = np.log(SEEING_LEVELS)
    k = np.clip(np.searchsorted(lx, x) - 1, 0, len(lx) - 2)
    f = (x - lx[k]) / (lx[k + 1] - lx[k])
    return (1 - f) * tab[k, j] + f * tab[k + 1, j]


@dataclass
class Prediction:
    snr: float
    fwhm: float
    qc_ok: bool
    reason: str = ""


def predict_exposure(
    model: NightModel, i: int, t0: float, t_exp_s: float, seeing_f: np.ndarray, cloud_f: np.ndarray
) -> Prediction:
    """Predicted S/N of one exposure from the planning tables under the forecast."""
    t = model.targets[i]
    j0 = model.idx(t0)
    j1 = model.idx(t0 + t_exp_s / 60.0)
    j = np.arange(j0, max(j1, j0) + 1)
    s = seeing_f[j]
    c = cloud_f[j]
    sig = _interp_levels(model.P_sig[i], s, j) * 10 ** (-0.4 * c)
    bkg = _interp_levels(model.P_bkg[i], s, j)
    npix = _interp_levels(model.P_npix[i], s, j)
    fwhm = float(np.mean(_interp_levels(model.P_fwhm[i], s, j)))
    S = float(np.mean(sig)) * t_exp_s
    V = S + float(np.mean(bkg)) * t_exp_s + float(np.mean(npix)) * model.rn2[i]
    snr = float(model.unit[i] * S / np.sqrt(max(V, 1e-9)))
    ok, reason = True, ""
    if t.max_seeing is not None and fwhm > t.max_seeing:
        ok, reason = False, f'predicted FWHM {fwhm:.2f}" > {t.max_seeing}"'
    if t.max_cloud is not None and float(np.mean(c)) > t.max_cloud:
        ok, reason = False, "cloud"
    return Prediction(snr, fwhm, ok, reason)


# ---------------------------------------------------------------------------
# candidates and merit
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    i: int
    name: str
    priority: int
    weight: float
    t_exp: float
    snr_now: float  # S/N so far
    goal: float
    snr1: float  # predicted S/N of one exposure now
    n_needed: int
    time_needed_min: float  # overheads + exposures + readouts
    overhead_min: float
    window_left_min: float  # until the target sets / leaves its limits / its window closes
    airmass: float
    fwhm_pred: float
    efficiency: float  # (S/N rate now / best S/N rate later tonight)^2
    urgency: float
    merit: float
    feasible: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class MeritWeights:
    alpha: float = 1.0  # airmass/sky efficiency exponent
    beta: float = 3.0  # urgency boost
    continue_bonus: float = 1.15
    min_window_frac: float = 0.35


def _exp_len(model: NightModel, i: int, t0: Optional[float] = None, seeing: Optional[float] = None) -> float:
    """The planned single-exposure length: the target's preferred length, shortened to stay
    below 70 % of the linearity limit if the seeing turns out 20 % better than expected (a
    sharper image puts more light in the brightest pixel), rounded down to whole seconds."""
    t = model.targets[i]
    cfg = t.cfg()
    texp = float(t.t_exp or cfg.max_exp_s)
    if t0 is not None:
        j = model.idx(t0)
        s = 0.8 * min(seeing if seeing is not None else model.site.seeing_median, 0.6)
        peak = float(_interp_levels(model.P_peak[i], np.array([s]), np.array([j]))[0])
        t_sat = 0.8 * cfg.full_well / max(peak, 1e-9)
        texp = min(texp, max(cfg.min_exp_s, 0.7 * t_sat))
    return float(max(np.floor(texp), cfg.min_exp_s))


def candidates(
    model: NightModel,
    state: ObsState,
    now: Nowcast,
    weights: MeritWeights = MeritWeights(),
    seeing_f: Optional[np.ndarray] = None,
    cloud_f: Optional[np.ndarray] = None,
) -> List[Candidate]:
    """Every target, scored. Infeasible ones are kept (feasible=False) so the console can say why."""
    if seeing_f is None:
        seeing_f, cloud_f = forecast_arrays(model, now, state.t)
    out: List[Candidate] = []
    snr_now = np.sqrt(state.snr2)
    for i, t in enumerate(model.targets):
        goal = t.snr_goal
        texp = _exp_len(model, i)
        base = dict(i=i, name=t.name, priority=t.priority, weight=t.w, t_exp=texp, snr_now=float(snr_now[i]), goal=goal)
        if snr_now[i] >= goal * 0.999:
            out.append(
                Candidate(
                    **base,
                    snr1=0,
                    n_needed=0,
                    time_needed_min=0,
                    overhead_min=0,
                    window_left_min=0,
                    airmass=0,
                    fwhm_pred=0,
                    efficiency=0,
                    urgency=0,
                    merit=0,
                    feasible=False,
                    note="done",
                )
            )
            continue
        if state.t < model.appears[i]:
            out.append(
                Candidate(
                    **base,
                    snr1=0,
                    n_needed=0,
                    time_needed_min=0,
                    overhead_min=0,
                    window_left_min=0,
                    airmass=0,
                    fwhm_pred=0,
                    efficiency=0,
                    urgency=0,
                    merit=0,
                    feasible=False,
                    note="not yet announced",
                )
            )
            continue
        oh_s, _ = model.overhead_to(state, i)
        t0 = state.t + oh_s / 60.0
        j0 = model.idx(t0)
        texp = _exp_len(model, i, t0, float(seeing_f[j0]))
        base["t_exp"] = texp
        X = float(model.geo["airmass"][i, j0])
        left = model.usable_until(i, t0) if model.is_visible(i, t0) else 0.0
        w0, w1 = model.window[i]
        if w1 < 1e8:
            left = min(left, max(0.0, w1 - t0))
        pred = predict_exposure(model, i, t0, texp, seeing_f, cloud_f)
        common = dict(airmass=X, fwhm_pred=pred.fwhm, overhead_min=oh_s / 60.0)
        if left <= 0:
            nxt = np.nonzero(model.observable[i, j0:])[0]
            why = "below limits / Moon" if not len(nxt) else f"observable in {nxt[0] * model.dt:.0f} min"
            if t0 < w0:
                why = f"window opens in {w0 - t0:.0f} min"
            out.append(
                Candidate(
                    **base,
                    snr1=pred.snr,
                    n_needed=0,
                    time_needed_min=0,
                    window_left_min=0,
                    efficiency=0,
                    urgency=0,
                    merit=0,
                    feasible=False,
                    note=why,
                    **common,
                )
            )
            continue
        if t0 < w0:
            out.append(
                Candidate(
                    **base,
                    snr1=pred.snr,
                    n_needed=0,
                    time_needed_min=0,
                    window_left_min=left,
                    efficiency=0,
                    urgency=0,
                    merit=0,
                    feasible=False,
                    note=f"window opens in {w0 - t0:.0f} min",
                    **common,
                )
            )
            continue
        if not pred.qc_ok:
            out.append(
                Candidate(
                    **base,
                    snr1=pred.snr,
                    n_needed=0,
                    time_needed_min=0,
                    window_left_min=left,
                    efficiency=0,
                    urgency=0,
                    merit=0,
                    feasible=False,
                    note=pred.reason,
                    **common,
                )
            )
            continue
        need2 = max(goal**2 - state.snr2[i], 0.0)
        n_needed = int(np.ceil(need2 / max(pred.snr**2, 1e-9)))
        cfg = t.cfg()
        per = texp / 60.0 + cfg.readout_s / 60.0
        t_need = oh_s / 60.0 + n_needed * per
        # best S/N rate this target will get tonight (to value observing it near its best)
        vis = np.nonzero(model.observable[i, j0:])[0]
        jj = j0 + vis
        if len(jj):
            sig = _interp_levels(model.P_sig[i], seeing_f[jj], jj)
            bkg = _interp_levels(model.P_bkg[i], seeing_f[jj], jj)
            rate = sig**2 / np.maximum(sig + bkg, 1e-12)
            rate_now = rate[0]
            eff = float(np.clip(rate_now / max(rate.max(), 1e-12), 0, 1))
        else:
            eff = 1.0
        # value of the remaining work, given how much of it can fit before the window closes
        frac_time = min(1.0, left / max(t_need, 1e-3))
        pc = model.program.partial_credit
        frac_now = snr_now[i] / goal
        credit_now = 1.0 if frac_now >= 1 else pc * frac_now**2
        if frac_time >= 1.0:
            gain = t.w * (1.0 - credit_now)
        else:
            reach2 = state.snr2[i] + frac_time * need2
            gain = t.w * (pc * reach2 / goal**2 - credit_now)
        urg = float(np.clip(t_need / max(left, 1e-3), 0, 1))
        merit = gain / max(t_need, 1.0) * eff**weights.alpha * (1 + weights.beta * urg**2)
        if state.tel_target == i and state.acquired:
            merit *= weights.continue_bonus
        note = ""
        if n_needed > 40:
            note = "goal out of reach tonight at these conditions"
        out.append(
            Candidate(
                **base,
                snr1=pred.snr,
                n_needed=n_needed,
                time_needed_min=t_need,
                window_left_min=left,
                efficiency=eff,
                urgency=urg,
                merit=float(merit),
                feasible=True,
                note=note,
                **common,
            )
        )
    return out


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------
@dataclass
class Action:
    kind: str  # "observe" | "wait" | "focus" | "closed"
    target: int = -1
    t_exp: float = 0.0
    minutes: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class GreedyPolicy:
    """The classical observer heuristic: highest merit (value per minute x efficiency x urgency),
    refocus when the temperature has drifted, wait when nothing is feasible."""

    name = "greedy"

    def __init__(self, weights: MeritWeights = MeritWeights(), focus_dT: float = 1.5):
        self.w = weights
        self.focus_dT = focus_dT

    def act(self, model: NightModel, state: ObsState, now: Optional[Nowcast] = None) -> Tuple[Action, List[Candidate]]:
        if now is None:
            now = Nowcast.from_history(model, state.t)
        if not model.dome_open_at(state.t):
            return Action("closed", reason="dome closed by weather"), []
        if abs(model.temperature(state.t) - state.focus_temp) > self.focus_dT and state.t - state.last_focus_t > 45:
            return Action(
                "focus",
                reason=f"temperature drifted {abs(model.temperature(state.t) - state.focus_temp):.1f} C "
                "since last focus",
            ), []
        cands = candidates(model, state, now, self.w)
        feas = [c for c in cands if c.feasible]
        if not feas:
            return Action("wait", minutes=10.0, reason="nothing observable meets its constraints"), cands
        best, why = self.pick(feas)
        return Action("observe", target=best.i, t_exp=best.t_exp, reason=why), cands

    def pick(self, feas: List[Candidate]) -> Tuple[Candidate, str]:
        """The candidate to observe among the feasible ones, and why."""
        best = max(feas, key=lambda c: c.merit)
        return best, f"highest merit ({best.merit:.3g})"


class QueueRulePolicy(GreedyPolicy):
    """The rule a queue observer follows: among the feasible observing blocks, those that can still
    be finished (merit > 0) first, then the best queue priority, then the one that sets soonest.
    Focus and waiting as in the greedy policy."""

    name = "queue-rule"

    def pick(self, feas: List[Candidate]) -> Tuple[Candidate, str]:
        best = min([c for c in feas if c.merit > 0] or feas, key=lambda c: (c.priority, c.window_left_min))
        return best, f"queue rule: P{best.priority}, {best.window_left_min:.0f} min left"


class ListOrderPolicy:
    """Naive baseline: go down the list in order, finishing each target before the next."""

    name = "list-order"

    def act(self, model: NightModel, state: ObsState, now: Optional[Nowcast] = None):
        if not model.dome_open_at(state.t):
            return Action("closed"), []
        now = now or Nowcast.from_history(model, state.t)
        cands = candidates(model, state, now)
        for c in cands:
            if c.feasible:
                return Action("observe", target=c.i, t_exp=c.t_exp, reason="next on the list"), cands
        return Action("wait", minutes=10.0), cands


def apply_action(model: NightModel, state: ObsState, action: Action):
    """Advance the world by one action. Returns (new_state, exposure outcome or None)."""
    if action.kind == "closed":
        return model.close_wait(state), None
    if action.kind == "focus":
        return model.focus(state), None
    if action.kind == "wait":
        return model.wait(state, action.minutes), None
    return model.observe(state, action.target, action.t_exp)


def run_policy(model: NightModel, policy, state: Optional[ObsState] = None, max_steps: int = 500):
    """Play a whole night with `policy` against the truth. Returns (final state, log)."""
    s = state or model.initial_state()
    log = []
    for _ in range(max_steps):
        if model.finished(s):
            break
        a, _ = policy.act(model, s)
        s2, out = apply_action(model, s, a)
        log.append(
            {
                "t": s.t,
                "action": a.to_dict(),
                "snr": None if out is None else out.snr_counted,
                "qc": None if out is None else out.qc_reasons,
            }
        )
        if s2.t <= s.t:  # guarantee progress
            s2 = model.wait(s2, 5.0)
        s = s2
    return s, log


# ---------------------------------------------------------------------------
# projection: the rest of the night under the forecast
# ---------------------------------------------------------------------------
@dataclass
class Block:
    target: int
    name: str
    start: float
    end: float
    n_exp: int
    snr_after: float
    done: bool


@dataclass
class Projection:
    blocks: List[Block] = field(default_factory=list)
    finish_min: Dict[int, float] = field(default_factory=dict)  # target -> projected completion minute
    final_snr: Dict[int, float] = field(default_factory=dict)
    list_done_min: Optional[float] = None
    score: float = 0.0
    idle_min: float = 0.0

    def to_dict(self, model: NightModel) -> dict:
        fmt = lambda m: model.ephem.utc(m).strftime("%H:%M")
        return {
            "blocks": [
                {
                    "target": b.target,
                    "name": b.name,
                    "start": b.start,
                    "end": b.end,
                    "start_ut": fmt(b.start),
                    "end_ut": fmt(b.end),
                    "n_exp": b.n_exp,
                    "snr_after": round(b.snr_after, 2),
                    "done": b.done,
                }
                for b in self.blocks
            ],
            "finish_ut": {model.targets[i].name: fmt(m) for i, m in self.finish_min.items()},
            "final_snr": {model.targets[i].name: round(v, 2) for i, v in self.final_snr.items()},
            "list_done_ut": fmt(self.list_done_min) if self.list_done_min is not None else None,
            "score": round(self.score, 3),
            "max_score": model.max_score(),
            "idle_min": round(self.idle_min, 1),
        }


def project(model: NightModel, state: ObsState, now: Nowcast, policy=None, max_steps: int = 300) -> Projection:
    """Simulate the policy forward under the forecast (not the truth)."""
    policy = policy or GreedyPolicy()
    seeing_f, cloud_f = forecast_arrays(model, now, state.t)
    s = state.copy()
    proj = Projection()
    cur: Optional[Block] = None
    for _ in range(max_steps):
        if s.t >= model.t_end - 1 or bool(model.done(s).all()):
            break
        cands = candidates(model, s, now, policy.w if hasattr(policy, "w") else MeritWeights(), seeing_f, cloud_f)
        feas = [c for c in cands if c.feasible]
        if not feas:
            proj.idle_min += 10.0
            s = model.wait(s, 10.0)
            cur = None
            continue
        best = max(feas, key=lambda c: c.merit)
        i = best.i
        oh, _ = model.overhead_to(s, i)
        t0 = s.t + oh / 60.0
        pred = predict_exposure(model, i, t0, best.t_exp, seeing_f, cloud_f)
        cfg = model.targets[i].cfg()
        s = s.copy()
        s.snr2[i] += pred.snr**2
        s.n_exp[i] += 1
        s.t = t0 + (best.t_exp + cfg.readout_s) / 60.0
        j = model.idx(t0 + best.t_exp / 60.0)  # where the telescope is when the shutter closes
        s.tel_target, s.acquired, s.setup = i, True, cfg.setup_id
        s.tel_alt, s.tel_az = float(model.geo["alt"][i, j]), float(model.geo["az"][i, j])
        snr_i = float(np.sqrt(s.snr2[i]))
        done = bool(snr_i >= model.goals[i] * 0.999)
        if cur is not None and cur.target == i:
            cur.end, cur.n_exp, cur.snr_after, cur.done = s.t, cur.n_exp + 1, snr_i, done
        else:
            cur = Block(i, model.targets[i].name, t0 - oh / 60.0, s.t, 1, snr_i, done)
            proj.blocks.append(cur)
        if done and i not in proj.finish_min:
            proj.finish_min[i] = s.t
    for i in range(model.N):
        proj.final_snr[i] = float(np.sqrt(s.snr2[i]))
    reachable = [i for i in range(model.N) if model.observable[i].any()]
    done0 = model.done(state)  # same 0.999 threshold as everywhere else
    if all(i in proj.finish_min or done0[i] for i in reachable):
        proj.list_done_min = max(proj.finish_min.values()) if proj.finish_min else state.t
    proj.score = model.score(s)
    return proj
