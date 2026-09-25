"""Lookahead planning: try each good candidate, project the rest of the night, keep the best.

The greedy merit policy looks one decision ahead. `LookaheadPolicy` evaluates the top-K
candidates by merit (and "stay on the current target") with the projection machinery: observe
the candidate once under the planner's forecast, then let the greedy policy finish the night,
still under the forecast. It never sees the true future, so it is a legitimate policy (unlike the
hindsight oracle); it costs K projections (~0.03 s each) per decision.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from obsassist.planning.oracle import best_action_forecast
from obsassist.planning.planner import Action, Candidate, GreedyPolicy, Nowcast
from obsassist.sim.night import NightModel, ObsState


class LookaheadPolicy:
    name = "lookahead"

    def __init__(self, k: int = 6):
        self.k = k
        self.greedy = GreedyPolicy()

    def act(self, model: NightModel, state: ObsState, now: Optional[Nowcast] = None) -> Tuple[Action, List[Candidate]]:
        a, cands = self.greedy.act(model, state, now)
        feas = [c for c in cands if c.feasible]
        if a.kind != "observe" or len(feas) < 2:
            return a, cands
        now = now or Nowcast.from_history(model, state.t)
        top = sorted(feas, key=lambda c: -c.merit)[: self.k]
        acts = [Action("observe", target=c.i, t_exp=c.t_exp) for c in top]
        best, _ = best_action_forecast(model, state, now, acts)
        best.reason = "best projected night among the top candidates"
        return best, cands
