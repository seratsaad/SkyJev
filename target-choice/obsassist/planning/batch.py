"""Batch oracle runs: the hindsight-optimal night for many (program, weather) pairs.

    python -m obsassist oracle --program keck1_lris_tonight --seeds 0-99 --width 8 --out oracle.jsonl
    python -m obsassist oracle --random clay --seeds 0-999 --width 4 --shard 3/40 --out shard3.jsonl

`--shard i/n` runs every n-th seed starting at i, for Slurm job arrays (scripts/osc/).
Each line: program, seed, weather regime, oracle score (pilot/beam), greedy and list-order
scores, the upper bound, and the oracle's action sequence.
"""

from __future__ import annotations

import argparse
import json
import time


def parse_seeds(s: str):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--program", help="built-in program name or YAML path")
    g.add_argument("--random", help="telescope key: a new random program per seed")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--width", type=int, default=1, help="beam width (1 = pilot method)")
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--out", default="oracle.jsonl")
    a = ap.parse_args(argv)
    from obsassist.planning.oracle import pilot, upper_bound
    from obsassist.planning.planner import GreedyPolicy, ListOrderPolicy, run_policy
    from obsassist.programs.generator import generate_program, random_date
    from obsassist.sim.night import NightModel
    from obsassist.targets import builtin_programs, load_program

    seeds = parse_seeds(a.seeds)
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        seeds = seeds[i::n]
    with open(a.out, "a") as f:
        for sd in seeds:
            t0 = time.time()
            if a.program:
                progs = builtin_programs()
                prog = load_program(progs[a.program] if a.program in progs else a.program)
            else:
                import numpy as np

                prog = generate_program(a.random, random_date(np.random.default_rng(sd)), seed=sd)
            m = NightModel(prog, seed=sd)
            orc = pilot(m, width=a.width)
            g_s, _ = run_policy(m, GreedyPolicy())
            l_s, _ = run_policy(m, ListOrderPolicy())
            rec = {
                "program": prog.name,
                "seed": sd,
                "regime": m.weather.regime,
                "width": a.width,
                "oracle": orc.score,
                "greedy": m.score(g_s),
                "list": m.score(l_s),
                "upper_bound": upper_bound(m),
                "max": m.max_score(),
                "seconds": round(time.time() - t0, 1),
                "actions": orc.actions,
            }
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            print(
                f"seed {sd}: oracle {orc.score:.2f} greedy {rec['greedy']:.2f} list {rec['list']:.2f} "
                f"ub {rec['upper_bound']:.2f} ({rec['seconds']}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
