# Can a fast, cheap model auto-focus a telescope? A test on archived runs

No telescope needed. ESO's archive keeps the frames from past auto-focus runs, and they are public.

## The data

30 **Through Focus Sequence** frames from the ESO/MPG 2.2 m Wide Field Imager at La Silla, February to
July 2026, one per night, R band (ESO template `WFI_cal_FocusSeq`). Each frame is one CCD holding nine
exposures: the telescope focus steps between exposures and the charge is shifted 50 rows, so every star
appears as a column of nine images, fuzzy at both ends and sharpest near best focus.


## The standard answer

[`measure.py`](measure.py) finds each star's column of nine images, fits every image with a 2-D Gaussian,
and fits the usual focus curve (FWHM² as a parabola in focus step) to the median of the good stars.
27 of 30 runs have enough unsaturated, isolated stars; best focus falls between steps 3 and 7.

## What the model was asked

[`decide.py`](decide.py): gpt-4.1-mini first, then gpt-4.1, gpt-5.5 and gpt-5.6-sol (`FOCUS_MODEL`). One token per answer, the probability of each option read from
the API's log-probabilities (the AnyJev approach). Each question is also asked with the options in other
orders and the answers averaged, to remove position bias.

1. **Which image is sharpest?** One star's nine images side by side, as an observer sees a focus frame.
2. **What next?** The measured curve as numbers, and four choices: accept, move toward step 1, move
   toward the last step, retake. Each run is also cut to its first and last five steps, so that some
   cases have best focus off the end, as a badly centred sequence would.

## Results

**Which star image is sharpest?** 26 stars, one per run, images at high detail. Right within one step:

| answered by | result |
|---|---:|
| gpt-4.1-mini, options in one order / both orders averaged | 15% / 46% |
| gpt-5.5, reasoning off | 77% |
| gpt-5.6-sol, reasoning off | 81% |
| **gpt-4.1** | **88%** |
| measuring the widths in code | 96% |

**What next: accept, move, or retake?** 84 cases, options in four orders averaged, scored against the
full nine-step fit:

| answered by | all cases | when its probability is at least 0.9 |
|---|---:|---:|
| always "accept" | 57% | |
| **gpt-4.1-mini** | **89%** | **96%** (70 of 84) |
| gpt-4.1 | 85% | 89% (76 of 84) |

About 0.6 to 0.7 s per call. Every model's answers are saved in `results/`, and all runs together cost
roughly $0.35 by the rates in `decide.py`.

## What it means

- **Whether the model can see focus depends on the model.** gpt-4.1-mini mostly picked a position, not an
  image: in one order it answered 7 or 8 for 22 of 26 stars whose best focus was at 4 to 6. gpt-4.1 lands
  within about half a step on average, 88% within one step, with no position bias. That is close to
  measuring the widths in code (96%), which is still simpler and exact. The GPT-5 models were run with
  reasoning off, to keep each answer to one fast pass, and with the five answer probabilities their API
  allows; with reasoning on they might do better, but they would no longer be fast decisions.
- **Bigger is not better at the calls.** gpt-4.1 sees better but reads the numbers slightly worse than
  gpt-4.1-mini, and is more often confidently wrong.
- **Do use it for the calls.** Given the measured curve, its accept / move / retake decisions agree with
  the full nine-step fit 89% of the time, and 96% when it is confident. That is the useful split: code
  measures, the fast model decides, and a low probability sends the case back to the standard routine or
  a person.
- **Averaging over option orders matters.** It moved the image test from 15% to 46% and the decision test
  from 81% to 89%.

## Honest caveats

- **My first answer key was wrong, and the model showed it.** The first rules judged each five-step
  window from its own five points. A window whose widths fall all the way to the last step was called
  "accept" because a parabola through five noisy points put its vertex just inside. The model said
  "best focus lies beyond the last step" with near certainty, and the full nine-step run agrees. The
  final key judges each window by the full nine-step fit, which uses four steps the model never saw.
  It was adopted after seeing the answers, so both keys are reported: 77% against the first, 89% against
  the second. [`score.py`](score.py) prints both.
- The outlier rule was also changed once before any model output: removing the point with the largest
  residual dropped the wrong point on 2026-05-14.
- Step 1 is taken as the top image of each column, inferred from the dark band of rows at the bottom of
  the chip that saw fewer exposures. The per-step focus values are not in the headers, so best focus is
  reported in steps, not micrometres.
- 30 runs from one instrument and one filter. Seeing-limited runs (2026-05-07, a curve between 2.5 and
  3.5 arcsec) are exactly where "accept" and "retake" are both defensible.

## Reproduce

```bash
focus/download.sh                                                    # 30 frames, about 240 MB, public
uv run --with astropy --with scipy python focus/measure.py
uv run --env-file .env --with astropy --with scipy --with pillow python focus/decide.py
FOCUS_DETAIL=high uv run --env-file .env --with astropy --with scipy --with pillow python focus/decide.py --sharpest-only
FOCUS_MODEL=gpt-4.1 uv run --env-file .env --with astropy --with scipy --with pillow python focus/decide.py --no-sharpest
uv run --with astropy --with scipy --with pillow python focus/score.py gpt-4.1
```

`decide.py` stops itself at `FOCUS_MAX_USD` (default $0.50).
