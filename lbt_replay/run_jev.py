#!/usr/bin/env python3
"""Run the Jev choices on the private LBT replay cases and write the aggregate result.

    source /fs/scratch/PAS2823/saadsm/SkyJev/pitzer_env.sh
    python lbt_replay/run_jev.py --private /fs/scratch/PAS2823/saadsm/private_lbt \
        --out lbt_replay/lbt_jev_replay.json            # on a V100: --device cuda --dtype float16

Each case is one zero-label Jev choice: anyjev.Question.choice over the case's fixed menu, decided by
Decider(backend, level="L0") (AnyJev defaults: all cyclic permutations of the menu read from the
option-letter probabilities of one prompt each, batch-mean prior correction), with the backend from
obsassist.assistant.anyjev_backend.make_local_backend. No tokens are generated.

Timing: one untimed warm-up decision per task on a separate Decider that shares the weights, then
every decision is timed with torch.cuda.synchronize() before and after. The code time to build the
menu (build_cases.py) is reported separately.

Per-case picks go to <private>/results/ only. The --out file holds aggregate numbers, task names and
generic labels; before writing it the script checks that no queue program or target name, and no
observer or reference text from the private files, occurs in it, and refuses to write otherwise.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "target-choice"))

TASKS = {
    "logging": {"label": "log an observation",
                "reference": "Status the assistant recorded for the target: its mark_observed call arguments (one case) or, "
                             "for the rest, its night-note / plan record of the night outcome after reading the same "
                             "observer message or night log. One case per (message, target named in it).",
                "code": "keyword rule on the line that names the target (skip words, then done words, then partial words; "
                        "nothing said = skipped)"},
    "weather": {"label": "weather call",
                "reference": "What happened next with the dome: observe = open or opened next with no closure in between; "
                             "wait = closed (or closed now) and reopened later that night; close = closed for the rest of "
                             "the night. Read from set_conditions calls, night notes and the OSURC night logs. Sky class: "
                             "the sky value of the assistant's set_conditions call in the same request.",
                "code": "rule on the lbt_tools weather limits: bad now if RH >= 90 %, precipitation in the NWS row or CSC "
                        "cloud >= 70 %; then wait if a later hour before twilight has RH < 85 %, no precipitation and "
                        "cloud <= 30 %, else close; sky class from the CSC cloud of the hour"},
    "next": {"label": "next queue target",
             "reference": "The target the telescope went to next (science, or a preset without science for two cases), "
                          "from the night notes, the observed log, the plans and the OSURC logs quoted in the transcripts.",
             "code": "lbt_tools planner: greedy_schedule from the decision time on the configured instrument, first target"},
    "slit": {"label": "slit position angle",
             "reference": "Acquisition-script PA with the lowest mean blue (0.35-0.45 um) slit loss over the visit, computed "
                          "with lbt_tools.optics.mods_slit_check (the mods_slit_losses tool); the Jev sees the same losses.",
             "code": "argmin of the computed loss (the reference itself, 100 % by construction)"},
}


def wilson(k: int, n: int, z: float = 1.959964) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def stats(xs) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, float)
    return {"n": int(a.size), "mean": round(float(a.mean()), 4), "median": round(float(np.median(a)), 4),
            "p10": round(float(np.percentile(a, 10)), 4), "p90": round(float(np.percentile(a, 90)), 4)}


def agreement(k: int, n: int) -> dict:
    return {"agree": k, "n": n, "pct": round(100 * k / n, 1) if n else None, "wilson95": wilson(k, n)}


def load_cases(private: Path, task: str) -> list[dict]:
    p = private / "cases" / f"{task}.jsonl"
    return [json.loads(line) for line in open(p)] if p.exists() else []


def private_strings(private: Path) -> set[str]:
    """Strings that must never appear in the public output: every queue program and target name, and
    every string value of the private annotation files longer than 5 characters."""
    bad: set[str] = set()
    for f in (private / "src" / "data" / "cache").glob("queue_*.json"):
        for t in json.loads(f.read_text())["targets"]:
            bad.update({t["program"], t["target"]})
            bad.update(p for p in t["program"].split("_") if len(p) > 3 and p not in {"MODS", "LUCI", "PEPSI"})

    generic = {"done", "partial", "skipped", "observe", "wait", "close", "clear", "thin cirrus", "cloudy", "open", "closed"}

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "refs":
                    bad.update(v.keys())
                elif k not in ("why", "_about", "ref_record", "from", "time", "start", "night", "prompt_time",
                               "window", "kind", "instrument", "default_ref", "dome_now", "id", "split", "queue", "what"):
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and len(x) > 5 and x not in generic:
            bad.add(x)
    for f in private.glob("ann_*.json"):
        walk(json.loads(f.read_text()))
    return {b for b in bad if len(b) >= 4}


def check_public(text: str, bad: set[str]) -> list[str]:
    hits = [b for b in bad if b in text]
    if re.search(r"\b\d{3}[- ]\d{3}[- ]\d{4}\b", text):
        hits.append("phone-number pattern")
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--private", required=True)
    ap.add_argument("--out", required=True, help="aggregate JSON (public: numbers and generic labels only)")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--tasks", default="logging,weather,next,slit")
    ap.add_argument("--batch-size", type=int, default=8, help="prompts per forward (backend default 8)")
    ap.add_argument("--shared-prefix", default="auto", choices=("auto", "off"),
                    help="off: score permutations as separate prompts (less memory; Qwen3-4B on a 16 GB V100)")
    a = ap.parse_args(argv)
    shared = "auto" if a.shared_prefix == "auto" else False
    priv = Path(a.private)
    os.umask(0o077)

    import torch
    import transformers
    import anyjev
    from anyjev import Decider, Question

    from obsassist.assistant.anyjev_backend import make_local_backend

    t_load = time.perf_counter()
    backend = make_local_backend(a.model, device=a.device, dtype=a.dtype, batch_size=a.batch_size)
    load_s = time.perf_counter() - t_load
    cuda = str(backend.device).startswith("cuda")

    def sync():
        if cuda:
            torch.cuda.synchronize()

    gpu = None
    if cuda:
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            gpu = torch.cuda.get_device_name()
    ct = json.loads((priv / "cases" / "claude_times.json").read_text())
    ann_drop = {t: json.loads((priv / f"ann_{t}.json").read_text()).get("dropped", []) for t in TASKS}
    resdir = priv / "results" / a.model.split("/")[-1]
    resdir.mkdir(parents=True, exist_ok=True)

    out = {"description": "Jev replay of routine LBT observing-assistant decisions (OSU partner queue, nights 2026 Sep 9-15). "
                          "Zero-label AnyJev L0 choices over code-built menus, compared with what was decided on the nights. "
                          "Aggregate numbers only.",
           "run_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
           "model": a.model, "dtype": a.dtype, "device": str(backend.device), "gpu": gpu,
           "torch": torch.__version__, "transformers": transformers.__version__,
           "anyjev": getattr(anyjev, "__version__", "unknown"), "python": platform.python_version(),
           "host": platform.node().split(".")[0], "decider": {"level": "L0", "prior": "batch (AnyJev default)",
                                                             "permutations": "all cyclic shifts (AnyJev default)",
                                                             "shared_prefix": a.shared_prefix, "batch_size": a.batch_size},
           "model_load_s": round(load_s, 1), "tasks": {}}

    for task in a.tasks.split(","):
        cases = load_cases(priv, task)
        if not cases:
            continue
        decider = Decider(backend, level="L0", shared_prefix=shared)
        warm = Decider(backend, level="L0", shared_prefix=shared)  # warm-up on its own decider: it must not feed the batch prior
        c0 = cases[0]
        warm.decide(c0["state"], [Question.choice(c0["question"], c0["options"], name=c0["question_name"])])
        sync()
        recs = []
        for c in cases:
            q = Question.choice(c["question"], c["options"], name=c["question_name"])
            sync()
            t0 = time.perf_counter()
            d = decider.decide(c["state"], [q], level="L0")[0]
            sync()
            sec = time.perf_counter() - t0
            if cuda:
                torch.cuda.empty_cache()  # outside the timed span
            assert d.level == "L0", d.level
            pick = c["options"].index(d.answer)
            recs.append({"id": c["id"], "question_name": c["question_name"], "pick": pick, "ref": c["ref"],
                         "code_pick": c["code_pick"], "probs": [round(float(p), 4) for p in d.probs], "seconds": sec,
                         "code_seconds": c["code_seconds"], "n_options": len(c["options"]),
                         "request_wall_s": c.get("request_wall_s"), "request_key": c.get("request_key")})
            print(f"{task} {c['id'][:6]} pick={pick} ref={c['ref']} code={c['code_pick']} {sec:.3f}s", flush=True)
        with open(resdir / f"{task}.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")

        qs = {}
        for qn in sorted({r["question_name"] for r in recs}):
            rr = [r for r in recs if r["question_name"] == qn]
            k = sum(r["pick"] in r["ref"] for r in rr)
            kc = sum(r["code_pick"] in r["ref"] for r in rr)
            opts0 = next(c["options"] for c in cases if c["question_name"] == qn)
            entry = {"jev": agreement(k, len(rr)), "code": agreement(kc, len(rr)),
                     "n_options": stats([r["n_options"] for r in rr])}
            if task in ("logging", "weather"):  # fixed menus: show the reference balance and the majority rate
                cnt = Counter(opts0[r["ref"][0]].split(":")[0] for r in rr)
                entry["reference_counts"] = dict(cnt)
                entry["majority_class_pct"] = round(100 * max(cnt.values()) / len(rr), 1)
                entry["jev_pick_counts"] = dict(Counter(opts0[r["pick"]].split(":")[0] for r in rr))
            if task == "next":
                entry["code_no_pick"] = sum(r["code_pick"] < 0 for r in rr)
                entry["jev_rank_of_reference_median"] = float(np.median(
                    [1 + int(np.sum(np.asarray(r["probs"]) > r["probs"][r["ref"][0]])) for r in rr if r["ref"]]))
            qs[qn] = entry
        walls = {}
        for r in recs:
            if r["request_wall_s"] is not None:
                walls[r["request_key"]] = r["request_wall_s"]
        wl = sorted(walls.values())
        out["tasks"][task] = {
            "label": TASKS[task]["label"], "where": "LBT records (replay)", "n_cases": len(recs), "questions": qs,
            "jev_seconds_per_decision": stats([r["seconds"] for r in recs]),
            "code_seconds_to_build_menu": stats([r["code_seconds"] for r in recs]),
            "claude_wall_s_matched_requests": {"n_requests": len(wl),
                                               "median": round(float(np.median(wl)), 1) if wl else None,
                                               "p10": round(float(np.percentile(wl, 10)), 1) if wl else None,
                                               "p90": round(float(np.percentile(wl, 90)), 1) if wl else None},
            "claude_wall_s_category": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in ct[task].items()},
            "reference": TASKS[task]["reference"], "code_baseline": TASKS[task]["code"],
            "dropped": [{"what": d.get("what", "one request"), "why": d["why"]} for d in ann_drop.get(task, [])],
        }
    text = json.dumps(out, indent=1)
    hits = check_public(text, private_strings(priv))
    if hits:
        sys.exit(f"refusing to write {a.out}: {len(hits)} private strings would leak")
    Path(a.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
