"""Per-call latency of the observing assistant's components on held-out decision states.

    source /fs/scratch/PAS2823/saadsm/SkyJev/pitzer_env.sh
    cd /fs/scratch/PAS2823/saadsm/SkyJev && python note/measure_latency.py --n 40   # add --s2 for System 2

States: the held-out decision files (classical and queue), grouped by (night, t) with
obsassist.learn.fit.load_rows and sampled as fit.py does (random.Random(1).shuffle(keys), first N).

Components, per mode:
  planner          one planner decision: Nowcast.from_history + planning.planner.candidates + the
                   greedy pick (highest merit among feasible). The row keeps only the rendered
                   observation, so the NightModel is rebuilt from (telescope, date, program_seed,
                   weather_seed, mode) and the night is REPLAYED to the row's t with the dataset's
                   own behaviour policy (greedy with eps=0.3 random picks, rng seeded as in
                   learn.dataset.night_rows), which reproduces the exact state of the row. If the
                   replay misses t, the state falls back to initial_state advanced with model.wait
                   to t (counted as `fallback_wait`). CPU only; each state is timed `--planner-reps`
                   times and its median kept.
  s1_judge         System1.judge_candidates(obs, names): the regret question over the row's
                   labelled candidates (the top-8 feasible by merit, as the assistant asks, plus the
                   queue-rule pick in queue mode), on the GPU, torch.cuda.synchronize around it.
  s1_state         System1.state_questions(obs): sky + dome_risk in one call.
  s2_deliberate    (--s2 only) System2.deliberate(context) on 5 states, context built as
  s2_chat          Assistant._escalate builds it; System2.chat on 3 fixed questions. Tools are
                   offline stand-ins built from the replayed NightModel (see s2_tools).
Seconds are summarised as n, mean, median, p10, p90.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "target-choice"
sys.path.insert(0, str(ROOT))

from obsassist.learn.fit import load_rows  # noqa: E402

MODES = {
    "classical": ("data/decisions/test_decisions.jsonl", "heads/qwen3-1.7b.json"),
    "queue": ("data/decisions/queue_test.jsonl", "heads/qwen3-1.7b-queue.json"),
}
CHAT_QUESTIONS = (
    "What should I observe next, and why?",
    "Is the seeing good enough for the highest-priority target right now?",
    "How much of the list will we finish tonight?",
)


def stats(xs: List[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, float)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 5),
        "median": round(float(np.median(a)), 5),
        "p10": round(float(np.percentile(a, 10)), 5),
        "p90": round(float(np.percentile(a, 90)), 5),
    }


class Timer:
    def __init__(self, cuda: bool = False):
        self.cuda = cuda

    def _sync(self):
        if self.cuda:
            import torch

            torch.cuda.synchronize()

    def __enter__(self):
        self._sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.s = time.perf_counter() - self.t0


def rebuild_state(row: Dict[str, Any], eps: float = 0.3):
    """(model, state, how) at the row's decision point. Replays learn.dataset.night_rows' behaviour
    policy; falls back to model.wait from the start of the night if the replay misses t."""
    from obsassist.planning.planner import Action, GreedyPolicy, apply_action
    from obsassist.programs.generator import generate_program
    from obsassist.sim.night import NightModel

    mode = row.get("mode", "classical")
    prog = generate_program(row["telescope"], row["date"], seed=row["program_seed"], mode=mode)
    m = NightModel(prog, seed=row["weather_seed"])
    greedy = GreedyPolicy()
    rng = np.random.default_rng(row["weather_seed"] * 7 + 1)
    t_goal = row["t"]
    s = m.initial_state()
    for _ in range(400):
        if m.finished(s) or s.t > t_goal + 1e-6:
            break
        a, cands = greedy.act(m, s)
        feas = [c for c in cands if c.feasible]
        if a.kind == "observe" and len(feas) >= 2 and round(s.t, 3) == t_goal:
            return m, s, "replay"
        if a.kind == "observe" and feas and rng.random() < eps:
            pick = feas[int(rng.integers(len(feas)))]
            a = Action("observe", target=pick.i, t_exp=pick.t_exp)
        s2, _ = apply_action(m, s, a)
        if s2.t <= s.t:
            s2 = m.wait(s2, 5.0)
        s = s2
    s0 = m.initial_state()
    return m, m.wait(s0, max(0.0, t_goal - s0.t)), "fallback_wait"


def planner_decision(m, s):
    """One planner decision as the assistant's planner tier makes it. Returns (n_targets, n_feasible)."""
    from obsassist.planning.planner import GreedyPolicy, Nowcast, candidates

    now = Nowcast.from_history(m, s.t)
    cands = candidates(m, s, now)
    feas = [c for c in cands if c.feasible]
    if feas:
        GreedyPolicy().pick(feas)
    return len(cands), len(feas)


def s2_tools(m, s, obs: Dict[str, Any], judgements: List[Dict[str, Any]]):
    """Offline stand-ins for the tools server.build_assistant gives System 2 (which read a live
    Assistant): same names and arguments, answered from the replayed NightModel and state."""
    from obsassist.etc import compute_rates, plan_exposures
    from obsassist.planning.planner import Nowcast, project

    now = Nowcast.from_history(m, s.t)

    def etc_estimate(target, seeing=None, cloud=None, t_exp=None):  # as adapters.manual.ManualAdapter.etc
        i = m.program.target_index(target)
        tg = m.program.targets[i]
        j = m.idx(s.t)
        r = compute_rates(
            tg.cfg(),
            m.tel,
            m.site,
            tg.source,
            seeing_500=seeing if seeing is not None else now.seeing,
            airmass=float(np.clip(m.geo["airmass"][i, j], 1, 6)),
            alt=float(m.geo["alt"][i, j]),
            cloud_mag=cloud if cloud is not None else now.cloud,
            sun_alt=m.ephem.sun_alt[j],
            moon_alt=m.ephem.moon_alt[j],
            moon_phase_angle=m.ephem.moon_phase_angle[j],
            moon_sep=m.geo["moon_sep"][i, j],
            lam=tg.lam_ref(),
            unit=tg.snr_unit,
        )
        p = plan_exposures(r, tg.cfg(), tg.snr_goal, float(np.sqrt(s.snr2[i])), t_exp or tg.t_exp)
        return {
            "target": tg.name,
            "plan": {"n_exp": p.n_exp, "t_exp": p.t_exp, "wall_s": p.wall_s, "snr_final": p.snr_final},
            "fwhm": float(r.fwhm),
            "sky_mag": float(r.sky_mag),
            "regime": r.regime(),
        }

    return {
        "get_state": lambda: {k: v for k, v in obs.items() if k != "candidates"},
        "get_candidates": lambda: {
            "candidates": [c for c in obs["candidates"] if c.get("feasible")],
            "system1": judgements,
        },
        "etc_estimate": etc_estimate,
        "get_projection": lambda: project(m, s, now).to_dict(m),
        "get_frame_quicklook": lambda name=None: {"error": "no frames in the offline latency run"},
    }


def s2_context(obs: Dict[str, Any], judgements: List[Dict[str, Any]], planner_pick: str, s1_pick: str, conf) -> str:
    """The deliberation prompt, as policy.Assistant._escalate writes it."""
    from obsassist.assistant.observation import render

    return (
        "Decide the next action for the observer.\n\nState (as System 1 sees it):\n"
        + render(obs)
        + "\n\nSystem 1 (fast model) judgements, lower expected regret is better:\n"
        + json.dumps(judgements, default=str)
        + f"\n\nThe deterministic planner's pick: {planner_pick}. System 1's pick: {s1_pick or 'n/a'} "
        f"(P(best)={conf}). Check the numbers with the tools before recommending."
    )


def hardware() -> Dict[str, Any]:
    import os
    import socket

    import torch
    import transformers

    cpu = platform.processor()
    try:
        cpu = next(ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "cpu": cpu,
        "cpus_allocated": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
    }


def measure_mode(mode: str, a, s1) -> Dict[str, Any]:
    data, _ = MODES[mode]
    te = load_rows(str(ROOT / data))
    keys = list(te)
    random.Random(1).shuffle(keys)
    keys = keys[: a.n]
    cuda = str(s1.backend.device).startswith("cuda") if s1 is not None else False
    out: Dict[str, Any] = {"test": data, "n_states": len(keys)}
    t_plan, t_judge, t_state = [], [], []
    n_tgt, n_feas, n_judged, how_n, consistent = [], [], [], {}, 0
    per_state = []  # (model, state, obs, judgement dicts) for System 2
    t_start = time.time()
    for k_i, k in enumerate(keys):
        rows = te[k]
        row, obs = rows[0], rows[0]["obs"]
        names = [r["candidate"] for r in rows]
        m, s, how = rebuild_state(row)
        how_n[how] = how_n.get(how, 0) + 1
        if k_i == 0:
            planner_decision(m, s)  # warm-up, untimed
        reps = []
        for _ in range(a.planner_reps):
            with Timer() as t:
                nt, nf = planner_decision(m, s)
            reps.append(t.s)
        t_plan.append(float(np.median(reps)))
        n_tgt.append(nt)
        n_feas.append(nf)
        n_judged.append(len(names))
        obs_feas = sorted(c["name"] for c in obs["candidates"] if c.get("feasible"))
        from obsassist.planning.planner import Nowcast, candidates

        re_feas = sorted(c.name for c in candidates(m, s, Nowcast.from_history(m, s.t)) if c.feasible)
        consistent += obs_feas == re_feas
        js = []
        if s1 is not None:
            if k_i == 0:  # warm-up, untimed
                s1.judge_candidates(obs, names)
                s1.state_questions(obs)
            with Timer(cuda) as t:
                js = s1.judge_candidates(obs, names)
            t_judge.append(t.s)
            with Timer(cuda) as t:
                s1.state_questions(obs)
            t_state.append(t.s)
        if len(per_state) < 5:
            per_state.append((m, s, obs, js))
        if (k_i + 1) % 10 == 0:
            print(f"  {mode} {k_i + 1}/{len(keys)} states, {time.time() - t_start:.0f}s", flush=True)
    out["planner"] = stats(t_plan)
    out["planner"]["reps_per_state"] = a.planner_reps
    out["planner"]["state_rebuild"] = how_n
    out["planner"]["replayed_feasible_set_matches_row"] = consistent
    out["candidates_per_state"] = {
        "judged_by_s1": stats(n_judged),
        "feasible": stats(n_feas),
        "program_targets": stats(n_tgt),
    }
    if s1 is not None:
        out["s1_judge_candidates"] = stats(t_judge)
        out["s1_state_questions"] = stats(t_state)
        out["s1_levels"] = s1.info()["levels"]
        out["s1_heads"] = s1.heads_path
    if a.s2:
        out.update(measure_s2(per_state, s1))
    return out


def measure_s2(per_state, s1) -> Dict[str, Any]:
    """System 2 wall time: deliberate on up to 5 states, chat on 3 fixed questions (state 0).
    Skipped, with the reason recorded, when no OpenAI key is found. The key is never printed."""
    from obsassist.assistant.system2 import System2, api_key

    if api_key() is None:
        return {"s2": {"skipped": "no OpenAI key (OPENAI_API_KEY or OPENAI= in ~/.env)"}}
    t_del, t_chat, usage = [], [], []
    for m, s, obs, js in per_state:
        jd = [j.to_dict() for j in js]
        feas = [c for c in obs["candidates"] if c.get("feasible")]
        planner = max(feas, key=lambda c: c.get("merit") or 0)["name"]
        s1_pick = min(js, key=lambda j: j.expected_regret).name if js else ""
        conf = next((j.p_best for j in js if j.name == planner), None)
        s2 = System2(tools=s2_tools(m, s, obs, jd))
        with Timer() as t:
            s2.deliberate(s2_context(obs, jd, planner, s1_pick, conf), effort="low")
        t_del.append(t.s)
        usage.append(dict(s2.usage, role="deliberate"))
    m, s, obs, js = per_state[0]
    jd = [j.to_dict() for j in js]
    for q in CHAT_QUESTIONS:
        s2 = System2(tools=s2_tools(m, s, obs, jd))  # fresh history: each question timed alone
        with Timer() as t:
            s2.chat(q)
        t_chat.append(t.s)
        usage.append(dict(s2.usage, role="chat"))
    return {
        "s2_deliberate": dict(stats(t_del), model=s2.deliberate_model, effort="low"),
        "s2_chat": dict(stats(t_chat), model=s2.chat_model, questions=list(CHAT_QUESTIONS)),
        "s2_usage": usage,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="states per mode")
    ap.add_argument("--modes", default="classical,queue")
    ap.add_argument("--planner-reps", type=int, default=5)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", help="float16 on a V100")
    ap.add_argument("--no-s1", action="store_true", help="planner only")
    ap.add_argument("--s2", action="store_true", help="also time System 2 (needs an OpenAI key)")
    ap.add_argument("--out", default=str(ROOT / "reports" / "latency.json"))
    a = ap.parse_args(argv)

    report: Dict[str, Any] = {
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "hardware": hardware(),
        "settings": {k: v for k, v in vars(a).items() if k != "out"},
        "units": "seconds per call (planner: per-state median of planner_reps repeats)",
    }
    base = None
    for mode in a.modes.split(","):
        s1 = None
        if not a.no_s1:
            from obsassist.assistant.system1 import System1

            heads = str(ROOT / MODES[mode][1])
            if base is None:
                base = s1 = System1(model=a.model, heads=heads, device=a.device, dtype=a.dtype)
            else:
                s1 = System1(model=a.model, heads=heads, loaded_backend=base.backend)
        report[mode] = measure_mode(mode, a, s1)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1, default=str))
        print(json.dumps({mode: report[mode]}, default=str), flush=True)
    if not a.s2:
        report["s2"] = {"skipped": "not requested (--s2 off)"}
        Path(a.out).write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
