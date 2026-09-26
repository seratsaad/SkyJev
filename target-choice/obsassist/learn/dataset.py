"""Decision datasets from simulated nights, labelled by hindsight.

For each night (a random program under random weather), a behaviour policy walks through the
night; at every decision point with at least two observable candidates we record:

* the observation (observables only, `assistant.observation.from_model`),
* for every observable candidate c: the hindsight value V(c) = the night's final score when
  c is observed next and the greedy policy finishes the night, both under the weather that
  actually happens; regret(c) = (max_c V - V(c)) / max_score, and its level (0-4);
* per-state truth for auxiliary questions: the sky state (photometric / thin cirrus / thick
  cloud) and whether the dome closes within 30 minutes.

Rows are JSON lines with the observation stored as a dict, so the text rendering (the prompt)
can be changed and re-scored without re-simulating. Nights run in parallel processes.

    python -m obsassist.learn.dataset --nights 600 --out data/decisions/train_decisions.jsonl --workers 8
    python -m obsassist.learn.dataset --mode queue --nights 400 --out data/decisions/queue_train.jsonl

`--mode queue` draws queue nights (`programs.generator`); every row also records the queue
observer's rule pick (`is_queue_rule`, `planning.planner.QueueRulePolicy`), which is always among
the labelled candidates in queue mode.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np

from obsassist.paths import DECISIONS

TELESCOPES = ("clay", "keck1")


def night_rows(
    night_id: int,
    telescope: str,
    date: str,
    program_seed: int,
    weather_seed: int,
    mode: str = "classical",
    eps: float = 0.3,
    max_cands: int = 8,
) -> List[Dict]:
    from obsassist.assistant.observation import from_model, regret_level
    from obsassist.planning.planner import GreedyPolicy, Nowcast, QueueRulePolicy, apply_action, run_policy
    from obsassist.programs.generator import generate_program
    from obsassist.sim.night import NightModel
    from obsassist.weather import SKY_STATES

    prog = generate_program(telescope, date, seed=program_seed, mode=mode)
    m = NightModel(prog, seed=weather_seed)
    greedy = GreedyPolicy()
    queue_rule = QueueRulePolicy()
    rng = np.random.default_rng(weather_seed * 7 + 1)
    s = m.initial_state()
    rows: List[Dict] = []
    for step in range(400):
        if m.finished(s):
            break
        a, cands = greedy.act(m, s)
        feas = [c for c in cands if c.feasible]
        if a.kind == "observe" and len(feas) >= 2:
            now = Nowcast.from_history(m, s.t)
            obs = from_model(m, s, now, cands)
            vals = {}
            queue_pick, _ = queue_rule.pick(feas)
            evaluated = sorted(feas, key=lambda c: -c.merit)[:max_cands]
            if mode == "queue" and queue_pick not in evaluated:  # the queue rule's choice must have a label
                evaluated.append(queue_pick)
            for c in evaluated:
                s2, _ = m.observe(s, c.i, c.t_exp)
                s3, _ = run_policy(m, greedy, s2)
                vals[c.name] = m.score(s3)
            best = max(vals.values())
            j = m.idx(s.t)
            fut = m.idx(s.t + 30)
            sky = int(m.weather.sky_state[j])
            closes = bool(m.dome_ok[j] and not m.dome_ok[j : fut + 1].all())
            greedy_name = m.targets[a.target].name
            for name, v in vals.items():
                reg = (best - v) / m.max_score()
                rows.append(
                    {
                        "night": night_id,
                        "telescope": telescope,
                        "date": date,
                        "program_seed": program_seed,
                        "weather_seed": weather_seed,
                        "t": round(s.t, 3),
                        "candidate": name,
                        "value": round(v, 4),
                        "best": round(best, 4),
                        "regret": round(reg, 5),
                        "level": regret_level(reg),
                        "is_greedy": name == greedy_name,
                        "is_queue_rule": name == queue_pick.name,
                        "mode": mode,
                        "n_cands": len(vals),
                        "sky": SKY_STATES[sky],
                        "dome_closes_30": closes,
                        "regime": m.weather.regime,
                        "obs": obs,
                    }
                )
        # behaviour policy: mostly greedy, sometimes another feasible candidate (state diversity)
        if a.kind == "observe" and feas and rng.random() < eps:
            pick = feas[int(rng.integers(len(feas)))]
            from obsassist.planning.planner import Action

            a = Action("observe", target=pick.i, t_exp=pick.t_exp)
        s2, _ = apply_action(m, s, a)
        if s2.t <= s.t:
            s2 = m.wait(s2, 5.0)
        s = s2
    return rows


def _job(args):
    try:
        return night_rows(*args)
    except Exception as e:  # a bad random program must not kill the batch
        return [{"error": f"{type(e).__name__}: {e}", "night": args[0]}]


def build(
    n_nights: int, out: str, workers: int = 8, seed0: int = 0, telescopes=TELESCOPES, mode: str = "classical"
) -> Dict:
    from obsassist.programs.generator import random_date

    rng = np.random.default_rng(seed0)
    jobs = []
    for k in range(n_nights):
        tel = telescopes[k % len(telescopes)]
        jobs.append(
            (seed0 + k, tel, random_date(rng), int(rng.integers(1_000_000)), int(rng.integers(1_000_000)), mode)
        )
    t0 = time.time()
    n_rows = n_err = 0
    with open(out, "w") as f, ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs)):
            for r in fu.result():
                if "error" in r:
                    n_err += 1
                    continue
                f.write(json.dumps(r) + "\n")
                n_rows += 1
            if (k + 1) % 10 == 0:
                print(f"  {k + 1}/{n_nights} nights, {n_rows} rows, {time.time() - t0:.0f}s", flush=True)
    return {"nights": n_nights, "rows": n_rows, "errors": n_err, "seconds": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nights", type=int, default=100)
    ap.add_argument("--out", default=str(DECISIONS / "decisions.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--mode", choices=("classical", "queue"), default="classical", help="program generator mode")
    a = ap.parse_args(argv)
    import os

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    print(build(a.nights, a.out, a.workers, a.seed0, mode=a.mode))


if __name__ == "__main__":
    main()
