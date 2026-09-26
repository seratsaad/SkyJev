# LBT Jev replay

Can the routine requests that the writing assistant (a Claude session) handled on real LBT
partner-queue nights (OSU/ND/UM/UVa queue, 2026 Sep 9-15) be handled by code plus one Jev
choice instead? How fast is that, and how often does it agree with what was actually decided?

A Jev choice here is `anyjev.Question.choice` over a fixed menu, decided by
`Decider(backend, level="L0")` with Qwen3 on one V100 (float16): the answer is read from the
option-letter probabilities, no tokens are generated, and there are no heads (too few cases to fit
any). L0 means the AnyJev defaults: every cyclic permutation of the menu is scored (prompts share
the state prefix) and a label-free batch prior is divided out.

## Files

| file | what |
|---|---|
| `build_cases.py` | builds the private case files from the transcripts, the cached queue pages and forecasts, and the `lbt_tools` code |
| `run_jev.py` | runs the Jev choices, times them, writes `lbt_jev_replay.json` |
| `run_jev.sbatch` | the Pitzer job (gpudebug, 1 V100) |
| `lbt_jev_replay.json` | the aggregate result, Qwen3-1.7B (numbers and generic labels only); the figure uses this one |
| `lbt_jev_replay_qwen3-4b.json` | the same with Qwen3-4B |

Qwen3-4B runs out of memory on the 16 GB V100 on the 26-option menus when the permutations share
the prefix, so it was run with `MODEL=Qwen/Qwen3-4B OUT_SUFFIX=_qwen3-4b EXTRA="--batch-size 2
--shared-prefix off" sbatch --export=ALL lbt_replay/run_jev.sbatch`: the same L0 decisions, but
its times are not comparable with the 1.7B times.

To repeat the run on Pitzer:

```bash
source /fs/scratch/PAS2823/saadsm/SkyJev/pitzer_env.sh
cd /fs/scratch/PAS2823/saadsm/SkyJev
python lbt_replay/build_cases.py --private /fs/scratch/PAS2823/saadsm/private_lbt   # needs raw_transcripts/
sbatch lbt_replay/run_jev.sbatch                                                     # cases/ is enough
```

The raw transcript copies were deleted after the run, so `build_cases.py` needs them copied back
into `private_lbt/raw_transcripts/`; `run_jev.py` only needs the kept `private_lbt/cases/`.

## Tasks

Each task has a menu built by code, a reference read from the records of the nights, and a code
baseline where one exists. Cases are the real requests and events of the run; nothing is synthetic.

1. **Log an observation** (`logging`). Input: an observer message, either a night log the observer
   wrote (pasted in chat or fetched by the assistant) or a chat message. Code finds the queue targets
   the message names; the Jev picks the status of each (done / partial / skipped). Reference: the
   status the assistant recorded after reading the same message: its one `mark_observed` call, and
   otherwise its night-note or plan record of the night outcome. The assistant made only one
   `mark_observed` call in the whole run, so the strict reference has n = 1; the rest use the
   night-note records, and the question of *which* target a message is about is not scored (n = 1).
   Code baseline: a keyword rule on the line that names the target.
2. **Weather call** (`weather`). Input: the time, the dome state as the observer knew it, the
   observer's message, the latest measured conditions from the assistant's `set_conditions` calls
   (humidity, wind, seeing, the sky description with any sentence about the dome removed) and the
   cached forecast (the `weather` tool result of that request or the latest one before it for that
   night). Menu: observe / wait / close. Reference: what happened next with the dome, from the
   `set_conditions` calls, night notes and night logs (observe = open, or opened next without a
   closure; wait = closed and reopened later that night; close = closed for the rest of the night).
   A second menu, the sky class (clear / thin cirrus / cloudy), is scored where the assistant set a
   sky value in the same request. Code baseline: a rule on the `lbt_tools` weather limits.
   Pool: the 27 requests classified conditions/weather between 17:00 and 06:00 local, plus two
   weather-driven requests the classifier put elsewhere; the 15 daytime conditions requests are out
   (no dome decision at that time).
3. **Next queue target** (`next`). At each point in the nights where the telescope moved to a new
   target, the `lbt_tools` planner (`evaluate_targets`, offline replay) builds the menu: the queue
   targets on the configured instrument whose observable window (altitude, PI hour angle, twilight,
   moon, sky class) is open now, from the queue page cached before the decision, with the
   exclusions and instrument states the assistant had recorded by then and the targets finished
   earlier that night marked done. The state shows priority, time needed, when the window closes,
   altitude, and any standing instruction the observer had given in chat. Reference: the target the
   telescope went to next (for two cases a preset that never got science because of weather or a
   fault). Code baseline: the planner's own pick (`greedy_schedule` from that time, first target).
4. **Slit position angle** (`slit`). For MODS visits of the two programs whose readmes offer several
   acquisition scripts at different PAs, at each start time the assistant computed or planned, code
   computes the slit losses of each option with `lbt_tools.optics.mods_slit_check` (the
   `mods_slit_losses` tool) and the Jev picks from the table. Reference: the lowest mean blue loss,
   so the code baseline is 100 % by construction; this measures only whether the Jev reads the table.

Agreement is a count, a percentage and a Wilson 95 % interval. For the fixed menus the output also
gives the reference class counts and the majority-class rate, since most logged targets were
skipped (weather) and a constant answer would score well.

## Times

- Jev: seconds per decision, `torch.cuda.synchronize()` before and after, after one untimed warm-up
  per task on a separate `Decider` that shares the weights (so the warm-up does not feed the prior).
- Code: seconds to build the menu (planner windows and scores, slit-loss grid, keyword or weather
  rule), reported separately; most of the planner time is astropy ephemerides for every queue target.
- Claude: wall time of the request, from the human prompt to the last assistant message before the
  next prompt, as `note/lbt_latency.py` defines it. Two numbers per task: the median over the
  requests the cases came from (a request often did more than this one decision), and the median
  of the whole category in the `lbt_latency.py` classification (logging, conditions_or_weather,
  what_next_or_replan, position_or_angle_check).

## Privacy handling

The LBT records hold PI names, program names, target names and phone numbers.

- The sources (LBT_observation README, data/cache, state and logs, plans, `mcp/lbt_tools`, readmes)
  and the two transcripts were copied only into `/fs/scratch/PAS2823/saadsm/private_lbt`
  (`chmod 700`, files `600`). The references were transcribed by hand into `ann_*.json` there.
- `build_cases.py` and `run_jev.py` contain no names, targets or transcript text. The case files
  (`cases/`), per-case picks (`results/`) and job logs stay in the private directory.
- `lbt_jev_replay.json` holds aggregate numbers, task names and generic labels. Before writing it,
  `run_jev.py` searches it for every program and target name in the cached queue pages, every
  string of the private annotation files, and phone-number patterns, and refuses to write on a hit.
- After the run the raw transcript copies were deleted from Pitzer; the case files and annotations
  are kept (`chmod 700`) so the Jev run can be repeated.
