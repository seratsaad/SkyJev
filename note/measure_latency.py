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
  s2               (--s2, or --s2-only to time System 2 alone and update only the "s2" key of --out)
                   deliberate: System2.deliberate(context, effort="low") on the first states of each
                     mode (--s2-deliberate, default classical:3,queue:2), context built as
                     Assistant._escalate builds it; tools are offline stand-ins answered from the
                     replayed NightModel (see s2_tools);
                   chat: System2.chat on CHAT_QUESTIONS, fresh history each, classical state 0;
                   narrate: System2.narrate once on a short console log written from a replayed
                     night's history.
                   A spend guard (SpendGuard) estimates the cost of every API call from the returned
                   token counts at conservative prices (PRICE_IN, PRICE_OUT per 1e6 tokens), keeps a
                   ledger across runs (--s2-ledger), refuses a call whose projected cost would take
                   the ledger to the hard cap (--s2-hard-cap), skips tasks projected past the
                   budget (--s2-budget) and aborts once the budget is reached. Completed
                   calls go to --s2-records (jsonl); a rerun skips calls already recorded.
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
    "What if the seeing goes to 1.2 arcsec?",
    "Is the moon a problem for the next target?",
    "How much of the list will we finish tonight?",
)
PRICE_IN, PRICE_OUT = 15.0, 60.0  # USD per 1e6 tokens, deliberately above list prices
FALLBACK_MODELS = ("gpt-5.5", "gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4o")


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
        if s1 is not None:
            if k_i == 0:  # warm-up, untimed
                s1.judge_candidates(obs, names)
                s1.state_questions(obs)
            with Timer(cuda) as t:
                s1.judge_candidates(obs, names)
            t_judge.append(t.s)
            with Timer(cuda) as t:
                s1.state_questions(obs)
            t_state.append(t.s)
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
    return out


class BudgetStop(RuntimeError):
    pass


class SpendGuard:
    """Estimated OpenAI spend at PRICE_IN / PRICE_OUT, kept in a JSON ledger shared by all runs.
    check() before a call refuses it once the budget is reached or if its projected cost would take
    the ledger to the hard cap; add() after a call records the returned token counts and aborts once
    the budget is reached. measure_s2 also skips a task whose projected cost would pass the budget."""

    def __init__(self, ledger: str, budget: float, hard: float):
        self.path, self.budget, self.hard = Path(ledger), budget, hard
        self.d = {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0, "max_output_tokens": 0}
        if self.path.exists():
            self.d.update(json.loads(self.path.read_text()))

    @staticmethod
    def usd(n_in: int, n_out: int) -> float:
        return (n_in * PRICE_IN + n_out * PRICE_OUT) / 1e6

    def check(self, est_in: int):
        proj = self.usd(est_in, max(1000, 2 * self.d["max_output_tokens"]))
        if self.d["usd"] >= self.budget or self.d["usd"] + proj >= self.hard:
            raise BudgetStop(f"spent ~${self.d['usd']:.3f}, next call projected ~${proj:.3f}, cap ${self.hard}")

    def add(self, n_in: int, n_out: int):
        d = self.d
        d["usd"] = round(d["usd"] + self.usd(n_in, n_out), 6)
        d["input_tokens"] += n_in
        d["output_tokens"] += n_out
        d["calls"] += 1
        d["max_output_tokens"] = max(d["max_output_tokens"], n_out)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d, indent=1))
        print(f"    call {d['calls']}: +{n_in} in / +{n_out} out tokens, est. total ${d['usd']:.4f}", flush=True)
        if d["usd"] >= self.budget:
            raise BudgetStop(f"estimated spend ${d['usd']:.3f} reached the budget ${self.budget}")


def guarded_system2(guard: SpendGuard, **kw):
    """A System2 whose every API call passes the spend guard (before and after)."""
    from obsassist.assistant.system2 import TOOLS, System2

    def size(x) -> int:
        return len(x.model_dump_json()) if hasattr(x, "model_dump_json") else len(json.dumps(x, default=str))

    class Guarded(System2):
        def _create(self, model, input_items, tools=None, **k):
            chars = sum(size(x) for x in input_items) + len(k.get("instructions", "")) + 2000
            chars += len(json.dumps(TOOLS)) if tools else 0
            guard.check(chars // 3)  # ~4 characters per token; /3 over-estimates on purpose
            n0, o0 = self.usage["input_tokens"], self.usage["output_tokens"]
            r = super()._create(model, input_items, tools=tools, **k)
            guard.add(self.usage["input_tokens"] - n0, self.usage["output_tokens"] - o0)
            return r

    return Guarded(**kw)


def pick_models() -> Dict[str, Any]:
    """Default chat / deliberate models if this key can list them, else the first listed fallback.
    models.list costs no tokens."""
    from openai import OpenAI

    from obsassist.assistant.system2 import CHAT_MODEL, DELIBERATE_MODEL, api_key

    ids = {m.id for m in OpenAI(api_key=api_key()).models.list()}
    out = {}
    for role, default in (("chat", CHAT_MODEL), ("deliberate", DELIBERATE_MODEL)):
        used = default if default in ids else next((f for f in FALLBACK_MODELS if f in ids), None)
        if used is None:
            raise SystemExit(f"no usable model for {role}: {default} not listed and no fallback listed")
        out[role] = {"default": default, "default_listed": default in ids, "used": used}
    print("models:", json.dumps(out), flush=True)
    return out


def s2_states(mode: str, n: int, s1) -> List[Any]:
    """(model, state, obs, judgements, how) for the first n held-out states of a mode, sampled as
    measure_mode samples them; judgements from System 1 when it is loaded, else empty."""
    te = load_rows(str(ROOT / MODES[mode][0]))
    keys = list(te)
    random.Random(1).shuffle(keys)
    out = []
    for k in keys[:n]:
        rows = te[k]
        obs = rows[0]["obs"]
        m, s, how = rebuild_state(rows[0])
        js = s1.judge_candidates(obs, [r["candidate"] for r in rows]) if s1 is not None else []
        out.append((m, s, obs, js, how))
    return out


def night_log(m, s, n: int = 12) -> List[str]:
    """A short console log ("HH:MM who: text", as server.nightlog feeds narrate) from a state's history."""
    utc = lambda t: m.ephem.utc(t).strftime("%H:%M")  # noqa: E731
    lines = []
    for h in s.history:
        if h[0] == "observe":
            _, i, t0, t_exp, snr = h
            lines.append(f"{utc(t0)} observer: {m.program.targets[i].name} 1 x {t_exp:.0f} s, S/N {snr:.1f} this exposure")
        elif h[0] == "wait":
            lines.append(f"{utc(h[1])} observer: waited {h[2]:.0f} min")
        elif h[0] == "focus":
            lines.append(f"{utc(h[1])} observer: focus run")
        elif h[0] == "closed":
            lines.append(f"{utc(h[1])} dome: closed until {utc(h[2])}")
    return lines[-n:]


def measure_s2(states: Dict[str, List[Any]], a) -> Dict[str, Any]:
    """System 2 wall time per role under the spend guard. Calls already in --s2-records are skipped,
    so a small test run and the main run add up. The key is never printed."""
    from obsassist.assistant.system2 import api_key

    if api_key() is None:
        return {"skipped": "no OpenAI key (OPENAI_API_KEY or OPENAI= in ~/.env)"}
    guard = SpendGuard(a.s2_ledger, a.s2_budget, a.s2_hard_cap)
    guard.check(0)
    models = pick_models()
    rec_path = Path(a.s2_records)
    recs = [json.loads(ln) for ln in rec_path.read_text().splitlines() if ln.strip()] if rec_path.exists() else []
    done = {(r["role"], r["mode"], r["i"]) for r in recs}
    want = dict(kv.split(":") for kv in a.s2_deliberate.split(",") if kv)
    delib = [("deliberate", md, i) for i in range(max(map(int, want.values()), default=0)) for md in want
             if i < int(want[md])]
    chats = [("chat", "classical", i) for i in range(a.s2_chat)]
    narr = [("narrate", "classical", i) for i in range(a.s2_narrate)]
    tasks = []  # interleaved, so that every role gets calls if the budget runs short
    for j in range(max(len(delib), len(chats), len(narr))):
        tasks += [x[j] for x in (delib, chats, narr) if j < len(x)]
    skipped = []
    for role, mode, i in tasks:
        if (role, mode, i) in done:
            continue
        seen = [r["est_usd"] for r in recs if r["role"] == role]
        if seen and guard.d["usd"] + 1.1 * max(seen) >= a.s2_budget:
            skipped.append(f"{role} {mode} {i}: budget")
            continue
        m, s, obs, js, how = states[mode][0 if role != "deliberate" else i]
        if role == "narrate":  # a state whose night already has exposures to log
            m, s, obs, js, how = next((x for x in states[mode] if any(h[0] == "observe" for h in x[1].history)),
                                      states[mode][0])
        jd = [j.to_dict() for j in js]
        s2 = guarded_system2(guard, tools=s2_tools(m, s, obs, jd), chat_model=models["chat"]["used"],
                             deliberate_model=models["deliberate"]["used"])
        print(f"  {role} {mode} {i} ...", flush=True)
        extra: Dict[str, Any] = {}
        try:
            with Timer() as t:
                if role == "deliberate":
                    feas = [c for c in obs["candidates"] if c.get("feasible")]
                    planner = max(feas, key=lambda c: c.get("merit") or 0)["name"]
                    s1_pick = min(js, key=lambda j: j.expected_regret).name if js else ""
                    conf = next((j.p_best for j in js if j.name == planner), None)
                    out = s2.deliberate(s2_context(obs, jd, planner, s1_pick, conf), effort="low")
                    extra = {k: out.get(k) for k in ("action", "target", "t_exp", "n_exp", "confidence")}
                    extra.update(planner_pick=planner, s1_pick=s1_pick, state=how)
                elif role == "chat":
                    extra = {"question": CHAT_QUESTIONS[i], "reply_chars": len(s2.chat(CHAT_QUESTIONS[i]))}
                else:
                    log = night_log(m, s)
                    extra = {"log_lines": len(log), "reply_chars": len(s2.narrate(log))}
        except BudgetStop as e:
            skipped.append(f"{role} {mode} {i}: {e}")
            print(f"  BUDGET STOP: {e}", flush=True)
            break
        u = s2.usage
        rec = dict(role=role, mode=mode, i=i, seconds=round(t.s, 3), model=s2.chat_model if role != "deliberate"
                   else s2.deliberate_model, calls=u["calls"], input_tokens=u["input_tokens"],
                   output_tokens=u["output_tokens"], est_usd=round(guard.usd(u["input_tokens"], u["output_tokens"]), 5),
                   **extra)
        recs.append(rec)
        with rec_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(f"  {role} {mode} {i}: {t.s:.1f} s, {u['calls']} calls, ~${rec['est_usd']:.4f}", flush=True)
    return s2_summary(recs, guard, models, skipped)


def s2_summary(recs, guard: SpendGuard, models, skipped) -> Dict[str, Any]:
    import os
    import socket

    out: Dict[str, Any] = {
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "units": "wall seconds per role call, including offline tool execution",
        "models": models,
        "effort": "low (policy.Settings.s2_effort default)",
        "prices_usd_per_1e6": {"input": PRICE_IN, "output": PRICE_OUT, "note": "conservative, for the spend guard"},
        "budget_usd": guard.budget,
        "hard_cap_usd": guard.hard,
        "estimated_spend_usd": round(guard.d["usd"], 4),
        "ledger": {k: guard.d[k] for k in ("calls", "input_tokens", "output_tokens")},
        "skipped": skipped,
    }
    for role in ("deliberate", "chat", "narrate"):
        rs = [r for r in recs if r["role"] == role]
        out[role] = dict(
            stats([r["seconds"] for r in rs]),
            model=sorted({r["model"] for r in rs}),
            api_calls=sum(r["calls"] for r in rs),
            input_tokens=sum(r["input_tokens"] for r in rs),
            output_tokens=sum(r["output_tokens"] for r in rs),
            est_usd=round(sum(r["est_usd"] for r in rs), 4),
        )
        if role == "deliberate":
            out[role]["states"] = {md: sum(r["mode"] == md for r in rs) for md in MODES}
        if role == "chat":
            out[role]["questions"] = sorted({r["question"] for r in rs})
    out["calls"] = recs
    return out


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
    ap.add_argument("--s2-only", action="store_true", help="time System 2 only; update just the 's2' key of --out")
    ap.add_argument("--s2-deliberate", default="classical:3,queue:2", help="deliberate calls per mode")
    ap.add_argument("--s2-chat", type=int, default=3, help="chat questions (<= len(CHAT_QUESTIONS))")
    ap.add_argument("--s2-narrate", type=int, default=1)
    ap.add_argument("--s2-budget", type=float, default=0.70, help="USD, estimated at PRICE_IN/PRICE_OUT")
    ap.add_argument("--s2-hard-cap", type=float, default=0.75, help="USD; no call is projected past it")
    ap.add_argument("--s2-ledger", default=str(ROOT / "reports" / "s2_spend.json"))
    ap.add_argument("--s2-records", default=str(ROOT / "reports" / "s2_calls.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "latency.json"))
    a = ap.parse_args(argv)
    a.s2 = a.s2 or a.s2_only
    want = {kv.split(":")[0]: int(kv.split(":")[1]) for kv in a.s2_deliberate.split(",") if kv}

    report: Dict[str, Any] = {
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "hardware": hardware(),
        "settings": {k: v for k, v in vars(a).items() if k != "out"},
        "units": "seconds per call (planner: per-state median of planner_reps repeats)",
    }
    base, states = None, {}
    for mode in a.modes.split(","):
        s1 = None
        if not a.no_s1:
            from obsassist.assistant.system1 import System1

            heads = str(ROOT / MODES[mode][1])
            if base is None:
                base = s1 = System1(model=a.model, heads=heads, device=a.device, dtype=a.dtype)
            else:
                s1 = System1(model=a.model, heads=heads, loaded_backend=base.backend)
        if not a.s2_only:
            report[mode] = measure_mode(mode, a, s1)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(report, indent=1, default=str))
            print(json.dumps({mode: report[mode]}, default=str), flush=True)
        if a.s2:
            states[mode] = s2_states(mode, max(want.get(mode, 0), 1), s1)
    if a.s2_only:  # keep every other key of the existing report
        report = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else report
    report["s2"] = measure_s2(states, a) if a.s2 else {"skipped": "not requested (--s2 off)"}
    if a.s2:
        report["s2"]["s1_judgements"] = "none (--no-s1)" if a.no_s1 else f"{a.model}, {a.device}"
    Path(a.out).write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report["s2"].items() if k != "calls"}, default=str), flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
