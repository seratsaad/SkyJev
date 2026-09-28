# Routine decisions

## Logging and slit position angle

Code does the computable part, then a Jev choice through AnyJev picks from a small fixed menu. A fitted
L2 head gives the probabilities, and a gate picked on validation (0.98, also 0.95) decides whether
SkyJev acts. The shared runner is `logging_eval.py`. It is `common.evaluate` extended to cover several
questions per model, both gates, a code baseline and per-case probabilities.

**Logging** (`logging.py`, `logging_data.py`). Code builds a fuzzy shortlist of queue targets from the
words and word pairs of the message. Two questions follow:
- `noul("Is this message about the queue target under review?")`, asked once for each candidate;
- done / partial / skipped, asked once for each target the message is about.

Code baselines are the fuzzy matcher and the replay's keyword rule. The messages are synthetic. Train
uses template family A; val and test use family B, which has other phrasings, frames, night-log
layout and catalogue prefixes, and shares no names with train. Real cases are the 48 LBT replay
status cases (7 messages), and the 61 shortlisted candidates of those messages.

**Slit PA** (`slit.py`, `slit_data.py`, code in `vendor_lbt_tools/`). There are 2000 synthetic MODS
visits, with 2-4 PAs each. Losses come from `lbt_tools.optics.mods_slit_check`. The question has four
fixed rows, `row 1` to `row 4`.
- *easy*: the state shows the loss table. The label is the lowest mean blue loss.
- *hard*: the state shows a readme-style rule in UT or hour angle, plus the start time and HA, and no
  table. The label is the row the rule gives.

Train and evaluation use disjoint nights. Real cases are the 15 replay slit cases; the hard version
shows the program's own readme, and its reference is still the lowest-loss PA.

To run on Pitzer (job scripts are in `logs/routine_ls/`):

```bash
N=2000 PROCS=16 SUFFIX=_x sbatch --export=ALL logs/routine_ls/slit_data.sbatch     # CPU, 1 min
# the visits must end up in logs/routine_ls/slit_visits.jsonl (the script's default suffix is _smoke)
MODEL=Qwen/Qwen3-4B sbatch --export=ALL -p gpu logs/routine_ls/ls_gpu.sbatch        # 8B: -p gpu-exp
```

The results are in `reports/{logging,slit}_*_<model>.json`.
- 1.7B and 4B ran on a V100 16 GB; 8B ran on a V100S 32 GB.
- All runs use float16. The hardware block was added from the job's `nvidia-smi` line.
- The reports hold aggregate numbers only, and are checked against the private strings before they
  are written.

Test is synthetic family B; "real" means the LBT replay cases. The table shows L2 accuracy / coverage
and accuracy when acting at the 0.98 gate:

| decision | model | test acc | cov @0.98 | acc acting | real acc | real cov | real acc acting |
|---|---|---|---|---|---|---|---|
| log status | 1.7B | 0.71 | 0.01 | 0.88 | 0.85 | 0.10 | 1.00 |
| log status | 4B | 0.84 | 0.54 | 0.96 | 0.88 | 0.63 | 0.97 |
| log status | 8B | 0.92 | 0.87 | 0.96 | 0.90 | 0.98 | 0.89 |
| log target | 1.7B | 0.84 | 0.48 | 0.97 | 0.39 | 0.38 | 0.61 |
| log target | 4B | 0.88 | 0.00 | - | 0.67 | 0.66 | 0.88 |
| log target | 8B | 0.96 | 0.97 | 0.97 | 0.84 | 0.98 | 0.85 |
| slit easy | 1.7B | 0.91 | 0.78 | 0.97 | 0.93 | 0.87 | 1.00 |
| slit easy | 4B | 0.98 | 0.99 | 0.98 | 1.00 | 1.00 | 1.00 |
| slit easy | 8B | 0.99 | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 |
| slit hard | 1.7B | 0.62 | 0.00 | - | 0.27 | 0.00 | - |
| slit hard | 4B | 0.64 | 0.01 | 1.00 | 0.40 | 0.00 | - |
| slit hard | 8B | 0.69 | 0.01 | 0.60 | 0.47 | 0.00 | - |

Code baselines:

| task | test | real |
|---|---|---|
| status keyword rule | 0.59 | 0.98 |
| target matcher | 0.99 | 0.97 |
| slit argmin | 1.00 | 1.00 |
| hard rule | 1.00 | n/a |

The majority rates are:

| task | test | real |
|---|---|---|
| status | 0.50 | 0.90 |
| target | 0.65 | 0.69 |
| slit easy | 0.36 | 0.47 |
| slit hard | 0.36 | 0.47 |

Limits:
- The real sets are tiny, and 41 of the 48 status cases come from 4 night logs.
- On the real cases the 0.98 gate is not calibrated: 8B acts on 98 % of status cases at 89 %.
- 2 of the 3 real chat messages refer to targets without naming them. No name shortlist can find these.

## Focus action and next-target check

Code measures the star widths (focus) or builds the candidate list (next target); a Jev choice with a
fitted L2 head picks; the gate threshold is chosen on validation for 98% accuracy when acting.
Scripts: `focus.py`, `target.py`. Reports: `reports/focus_*.json`, `reports/target_*.json`.

**Focus action** (accept / move toward first step / move toward last step / retake). Training on
synthetic focus curves; real test on the 84 archival ESO/MPG 2.2 m WFI cases (key 2 = full nine-step
fit, adopted after an earlier model's answers were seen; key 1 = rule written beforehand).

| model | real, key 2: accuracy | coverage | accuracy when acting (95% CI) | s/decision |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 89.3% | 76% | 100% (94.3-100), 64/64 | 0.03 |
| Qwen3-4B | 89.3% | 75% | 93.7% (84.8-97.5), 59/63 | 0.04 |
| Qwen3-8B | 91.7% | 68% | 96.5% (88.1-99.0), 55/57 | 0.05 |

Against key 1 the same models reach 77-81% when acting. The zero-label (L0) baseline is 13-52%.
Confident errors are boundary windows (best focus at a window edge).

**Next-target check** (simulated classical and queue nights, per-candidate regret heads, Qwen3-4B
and 8B). The gate does not reach high accuracy: on held-out states it acts on at most about 1% of
classical states and on no queue states at the validation-chosen threshold, and without a gate the
pick has a higher mean regret than the planner (0.83-0.85% against 0.71% classical, 1.59-1.64%
against 1.35% queue). The planner stays the default for the next target.

## Weather decisions

Code turns telemetry into margins to the site limits and trends over 30 min; a Jev choice picks.
Labels come from the simulator's future truth (horizon 60 min), with whole nights held out; the real
test is the 18 anonymized weather cases replayed from the LBT nights (4 for the sky class). Script:
`weather.py`. Reports: `reports/weather_*.json` (`dome` = observe / wait / close, `dome2` =
observe / not observe, `sky` = photometric / thin cirrus / thick cloud).

| task | model | simulated test: accuracy | coverage | accuracy when acting (95% CI) | real LBT |
|---|---|---:|---:|---:|---|
| sky class | Qwen3-4B | 95.4% | 93% | 98.3% (97.5-98.8) | 4 cases, not usable |
| dome, 2 options | Qwen3-8B | 89.8% | 37% | 98.5% (97.4-99.1) | acts on 3 of 18, all right |
| dome, 3 options | Qwen3-4B | 88.3% | 22% | 96.8% (94.8-98.0) | acts on 1 of 18, wrong |

Simulated always-majority rates are 71% (sky) and 68% (dome). The simulator's weather does not carry
over well to the real LBT cases (Mt Graham limits and the few real cases), so real-night use needs
labelled real weather data and a shadow run first.

## CLM test (contrastive heads on Qwen3-8B embeddings)

`clm/README.md` has the method, the V100 status and the numbers; reports are `reports/clm_*.json`.
CLM-v0.1-8B runs on a V100 through vLLM 0.10.1.1 (the last vLLM with sm_70 kernels) or an
equivalent transformers embedder. Zero-shot it is at about the majority or chance rate on next
target, logging and focus, and no validated gate acts. CLM's own fine-tuning on the Jev heads'
synthetic data helps on synthetic tests but stays below the Jev L2 heads, and does not transfer to
the real logging messages. Speed is similar to a Jev head on these short menus (0.07-0.17 s).

## Held-out focus test (plan fixed on 2026-09-28, before any held-out frame was measured)

- **Frames**: 300 WFI through-focus sequences (`WFI_cal_FocusSeq`, R band, one per night) drawn at
  random (seed 2026) from the 811 archive nights not used before, 2021-09 to 2026-02
  (`focus/fetch_heldout.py`, list in `focus/download_list_heldout.txt`).
- **Measurement**: `focus/measure.py` unchanged; cases windowed as in `focus/decide.py`.
- **Reference**: key 2, the full nine-step fit (`focus/score.truth_full`), unchanged. Key 1 (the window
  rule written before any model output) is reported as well.
- **Model and gate**: the Qwen3-1.7B head fitted on the same synthetic runs as before (`routine/focus.py`,
  same seeds); the gate is the threshold chosen on synthetic validation for 98% accuracy when acting.
  Qwen3-4B and 8B are reported alongside. No threshold or model is chosen on the held-out frames.
- **Reported**: accuracy on all cases, coverage and accuracy when acting (with 95% intervals), the code
  rule, and the always-majority rate, against both keys.
