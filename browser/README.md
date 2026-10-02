# browser: can a fast browser agent use astronomy websites?

AstroWeb-JEV is a small benchmark of 15 tasks on real astronomy websites: SIMBAD, VizieR, NED,
ADS, arXiv, the ESO archive and exposure time calculator, the NASA Exoplanet Archive, the Gaia
archive and Aladin. Each task gives an agent a goal and a start page, for example "Using the SIMBAD
basic search, search for the star Vega and open its result page", and the agent has to reach the
page the goal asks for.

The question is whether [Jev Ultrafast](https://github.com/browser-use/jev-ultrafast) (JEV), a
browser agent that is fast because it is constrained, still works on scientific sites. JEV turns
each page into a numbered list of controls, and its model answers two multiple-choice questions:
which operation, and on which element. It never writes a selector, a coordinate or a line of
JavaScript. The benchmark measures how often each agent finishes a task, how long it takes, how
many tokens it uses, and how it fails.

The tasks come in five groups of three: `lookup`, `search`, `multistep`, `interactive` (archive
query forms) and `adversarial` (a Gaia archive form, the Aladin sky viewer drawn on a canvas, and
the ESO exposure time calculator). They are in `tasks/*.yaml`. A second set of
13 held-out tasks, written later for a follow-up, is in `tasks_heldout/heldout.yaml`.

## How a run is scored

No model judges another model. Every task has a deterministic verifier (regular expressions on
the page text, frames and URL, and in one task a JavaScript expression) that reads the agent's own
tab through a separate browser connection after the agent stops. The agent saying "done" counts
for nothing on its own. Each verifier is itself checked: it must pass on a known-good page and
fail on the start page (`results/verifier_validation.json`). Both agents get the same goal text, the same start URL, the
same Chrome and the same settling time after the clock stops.

## Three arms

| arm | page seen as | chooser model | status |
|---|---|---|---|
| `jev` | JEV element table | TypeSafe's hosted `jev-latest` | **not run**: needs `TYPESAFE_API_KEY`, and access was waitlisted and full when checked on 2026-09-22 |
| `gpt-jev` | JEV element table | `gpt-4.1-mini` | 45 runs (15 tasks x 3) |
| `baseline` (Browser Use 0.13.10) | Browser Use's own page state | `gpt-4.1-mini` | 30 runs (15 tasks x 2), plus 3 in free mode |

`gpt-jev` swaps one function in JEV, the chooser, for `gpt-4.1-mini`. The loop, the executor and
the budgets are JEV's own code. It therefore measures JEV's page representation with an ordinary
model, not the published system. With the same model in `gpt-jev` and `baseline`, the difference
between them is the agent design.

The baseline runs in `interface` mode by default: it is asked to use the site's own search boxes
and links, because JEV has no address bar. `free` mode lets it type URLs; it was run once on each
of the three lookup tasks to see what that is worth.

## Headline numbers

From `results/summary/per_agent.csv`, runs taken 2026-09-22. Times run from the first model call
to the agent's stop and leave out opening the browser.

| | `gpt-jev` | `baseline` (interface) | `baseline` (free) |
|---|---|---|---|
| runs | 45 | 30 | 3 |
| verified successes | 14 (31%) | 19 (63%) | 3 (100%) |
| median time per run | 6.5 s | 71.6 s | 24.0 s |
| median time, successful runs | 7.6 s | 43.4 s | 24.0 s |
| total time / successes | 46.6 s | 144.5 s | 24.0 s |
| median tokens per run | 23,468 | 177,901 | 45,919 |
| total tokens / successes | 221,455 | 379,502 | 66,671 |
| said done, page failed the check | 13% | 33% | 0% |
| agent's own claim agrees with the page | 64% | 70% | 100% |
| reached the goal page, then left | 6 runs | 0 runs | 0 runs |

In short: with the same model, JEV's representation is about ten times faster per run but
succeeds half as often, so the time per verified success is about three times shorter rather
than ten. Both agents sometimes say they are finished when the page shows otherwise, which is why
the verifier, not the agent, decides. Per-task and per-category tables are in
`results/summary/per_task.csv`, `per_category.csv` and `comparison.csv`; `failures.csv` counts
failures by label and arm, and `runs.csv` has one row per run.

Two caveats. The `vizier_hipparcos_102` verifier was tightened after these runs (version 2); the
three `gpt-jev` passes on it ended on the VizieR front page and would fail today. `per_agent.csv`
keeps the scores as measured. Second, `results/summary/plus_compare.json` holds a later follow-up:
a patched `gpt-jev-plus` arm on the original tasks and on the 13 held-out tasks, with Wilson
intervals and the scoring rules given at the top of `analysis/plus_compare.py`. The write-up of
that follow-up (`IMPROVING_JEV.md`) was not copied here.

## Setup

You need macOS or Linux with Google Chrome, [uv](https://docs.astral.sh/uv/) and Python 3.12 or
3.13. JEV is used as a separate checkout, pinned to the commit the results were measured on. The
project file expects it at `../jev-ultrafast` relative to this folder, that is at
`SkyJev/jev-ultrafast`, which the top-level `.gitignore` keeps out of this repository:

```bash
cd browser
git clone https://github.com/browser-use/jev-ultrafast.git ../jev-ultrafast
git -C ../jev-ultrafast checkout 1231850
uv sync
cp .env.example .env     # add OPENAI_API_KEY (and TYPESAFE_API_KEY, if you have one)
make chrome              # a separate Chrome on port 9222 with a scratch profile in .chrome-profile/
make validate            # each verifier on its known-good page and on its start page
make test                # offline tests
```

`make chrome` uses the macOS path to Chrome; on Linux, edit `CHROME` in the `Makefile`. It never
touches your everyday browser profile.

## Rerunning

Run one arm at a time. Both arms share one Chrome, and two browser workloads at once cause
protocol timeouts that look like agent failures.

```bash
make gpt-jev     # 15 tasks x 3, about US$1.24 at the prices in src/astroweb/budget.py
make baseline    # 15 tasks x 2 in interface mode, then the 3 lookup tasks in free mode
make jev         # only with a TYPESAFE_API_KEY
make summary     # rebuild results/summary/ from the raw runs
```

Each run writes one JSON file to `results/<arm>/`; `results/README.md` explains the fields and
failure labels. A batch stops itself when its estimated spend reaches `MAX_SPEND_USD` in `.env`
(default US$5). Websites change, so new runs will not reproduce the table above exactly.

Raw run files are large and are not kept in git. `results/shipped/` holds a compact copy of each of
the 78 runs in the table (the first and last element table only) plus one run of the hybrid demo.
The analysis scripts skip `shipped/`, so to rebuild the tables from it, copy the files back into
per-arm folders first (not tested in this copy):

```bash
for f in results/shipped/*.json; do
  n=$(basename "$f"); arm=${n%%__*}
  mkdir -p "results/$arm" && cp "$f" "results/$arm/${n#*__}"
done
make summary
```

To see what JEV sees on any page, with no model calls and no API key:

```bash
uv run --env-file .env python -m astroweb.runner snapshot --url https://simbad.cds.unistra.fr/simbad/
```
