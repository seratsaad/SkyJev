# Working on astrojev

An observing assistant for optical telescopes: a simulated console, an exposure time calculator
(ETC) for Keck and Magellan, a hindsight oracle, and an assistant that combines a deterministic
planner with AnyJev (System 1, local LLM) and GPT (System 2). The package and command are
`obsassist`. What it does and what was measured: [README.md](README.md). Every observing decision
and who makes it: [docs/decisions.md](docs/decisions.md).

## Setup and checks

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[all]"
python -m pytest -q                                    # 244 tests, a minute or two, no network, no GPU
ruff check obsassist tests && ruff format --check obsassist tests
```

Run both before you finish a change. Live check: `obsassist console --assistant-control` and
`obsassist assistant --mode autopilot` (add `--no-s1` to skip the 3.8 GB model), then open the
console's Assistant tab: it shows every exchange behind every decision as JSON.

## How a decision flows

    console engine (sim/night.py: photons, weather truth; console/engine.py: TCS, dome, arms, FITS)
      -> snapshot JSON (GET /api/state) -> adapter (assistant/adapters/: sim, manual, ktl helpers)
      -> Assistant.step (assistant/policy.py) -> observation.from_snapshot -> render (the prompt text)
      -> planner pick (highest merit) -> System 1 judges it -> System 2 on disagreement
      -> Recommendation -> adapter.instructions -> console commands (autopilot) or text for the observer
    assistant/trace.py records each arrow (GET /api/trace, data/traces/*.jsonl).

## Rules

1. **The planner acts.** On held-out nights nothing learned beats it (README, Results). System 1
   annotates and escalates; its override setting stays off unless a benchmark in `reports/` shows
   a gain.
2. **Language models never compute photons, airmass or times.** The ETC and the ephemerides do. A
   System 2 exposure time is used only within 2x of the ETC plan.
3. **No real-hardware actuation.** At a real telescope the assistant only advises: the manual
   adapter prints instructions, the Keck helpers print KTL lines and never `modify`. Commands reach
   only the simulated console, and only when both the console and the assistant allow it.
4. **Observables only.** The planner, the prompts and System 2 never see the weather truth or the
   future; the truth is for the simulator, the oracle and the debrief.
5. **Numbers come from sources.** Instrument, telescope and site values carry a source URL and a
   confidence in `docs/research/`. Do not change one without a source.
6. **Every fix comes with a test that fails without it**, in the test file of its subject.
7. **README numbers come from `reports/`**, and `scripts/reproduce.sh` regenerates all of them. If
   you change the simulator, the ETC, the planner or the prompt, rerun it and update the README.
8. **Generated files go under `data/` (not versioned), `heads/` or `reports/`** (`obsassist/paths.py`).
9. **Keep it minimal**: no dead code, no leftover logs or scratch files; ruff clean.
10. **Secrets**: the OpenAI key comes from `$OPENAI_API_KEY` or `~/.env`; never write it anywhere.
    Do not log in to clusters or accounts for the user; write job scripts (`scripts/osc/`) instead.

## Common tasks

* **New instrument setup**: `register(InstrumentConfig(...))` in `obsassist/instruments/keck.py`
  or `magellan.py`, sources in `docs/research/`, console arms in `console/engine.py`
  (`INSTRUMENT_ARMS`), an anchor test in `tests/test_etc.py`.
* **New program**: a YAML file like `obsassist/programs/clay_mike_darktime.yaml` (give coloured
  targets a continuum, e.g. `sed: "bb:3200"`).
* **Changing the prompt** (`assistant/observation.py: render`) or a question
  (`assistant/system1.py: QUESTIONS`): the heads are fit on that text, so refit them
  (`python -m obsassist.learn.fit --l0-states 120`). Questions need distinct (kind, options)
  layouts, or AnyJev routes one question to another's head.
* **Another local model**: `obsassist assistant --model <hf id> --heads heads/<model>.json` after
  fitting heads for it. Qwen3.5 works but is slow on Apple GPUs (no Gated DeltaNet kernel); use CUDA.
* **Large sweeps**: `scripts/osc/` (Slurm job arrays, CPU only).
* **A real night**: `obsassist assistant --manual program.yaml --data /path/to/rawdata`; the trace
  in `data/traces/` is the record to compare against afterwards.

## Open directions

* Validate on real nights: the planner's projections and finishing times against what happened.
* Find where System 1 earns its place. It does not improve the next-target choice; its sky
  transparency head does work. Candidates: frame quality judgements, reading PI notes, dome risk
  with better weather features.
* Larger local models on CUDA; heads for them.
* A tighter upper bound for the oracle (the oracle reaches ~0.64 of the current bound, which is loose).
* Replace instrument numbers marked `approximate` or `estimate` with measured ones.
