# The decisions of an observing night, and who makes them

Every decision below is listed with its inputs, the layer that handles it, and whether the
assistant may act on it by itself. Layers, cheapest first:

* **planner**: deterministic code: ETC, ephemerides, merit function, projection. Always on.
* **System 1**: AnyJev typed questions on a local LLM (Qwen3), calibrated heads (L2) fit on
  simulator hindsight labels. Returns probabilities, fast enough to call every decision point.
  It annotates the planner's choice and flags disagreements; it does not override the planner by
  default, because on held-out decisions the planner's choice has lower regret (README, Results).
* **System 2**: GPT-5.6 (sol/luna) with tools (state, candidates, ETC, projection, frame
  quick-look). A second opinion when System 1 disagrees with the planner, and whenever you ask.
* **you**: the observer; the telescope operator (TO) runs the telescope and the dome.

"Acts alone" means: in autopilot, with assistant control enabled on the console, the assistant
may carry it out without asking. Nothing irreversible ever acts alone.

## Before the night

| Decision | Inputs | Layer | Acts alone |
|---|---|---|---|
| Is each target observable tonight, and when? | coordinates, date, site, pointing limits (Keck Nasmyth deck), airmass and Moon limits, time windows | planner (ephemerides, `observable`) | n/a |
| Exposure plan per target (t_exp, number, readout speed, binning) | ETC under median seeing, saturation limit, instrument max exposure (cosmic rays, NIR sky) | planner (`plan_exposures`) | n/a |
| Is the list oversubscribed; what will not fit? | projection under the site median | planner (`project`) | n/a |
| Catalog for the TO | target list, rotator mode (MIKE: OFF; slit PA: EQU; parallactic: HRZ) | `planning/catalogs.py` (Magellan 16-field, Keck starlist) | n/a |
| Afternoon calibrations (bias, flats, arcs) | instrument setup | you (console: image type flat/arc/bias) | no |

## During the night

| Decision | Inputs | Layer | Acts alone |
|---|---|---|---|
| Open / close the dome | humidity, wind, dew point vs site limits, reopen wait | the TO (simulated: `weather.py` rules) | never (the TO) |
| **Which target next** | every observable candidate: time to finish now, time left before it sets or its window closes, airmass trend, efficiency vs its best, seeing requirement vs predicted FWHM, priority, overheads, PI notes | planner decides; System 1 gives expected regret and P(best) per candidate and sends disagreements to System 2, whose answer replaces the pick if it arrives before execution | yes (the planner's pick, or System 2's) |
| How long each exposure; stop when the goal is reached | measured S/N so far, ETC under current conditions | planner; assistant takes one exposure at a time and re-decides after each readout | yes |
| Continue the current target or switch | as "which target next" (continuing has no overhead) | same | yes |
| Should a seeing-critical target wait for better seeing? | predicted delivered FWHM vs `max_seeing`, seeing trend | planner (feasibility), System 1 (regret) | yes |
| Take the time-critical target now (eclipse window, ToO) | window, alert time, priority | planner urgency term; alert raised on ToO | yes (P1 ToO is still a normal decision) |
| Stop / pause an exposure in progress (clouds, seeing blew up, target lost) | guider flux, guider FWHM vs target limit, dome state | monitor raises an alert; you decide | **no** |
| Abort an exposure | same | you | **never** |
| Refocus | temperature change since last focus, guider FWHM vs DIMM | planner (refocus at ΔT > 1.5 °C), alert | yes (a focus run) |
| Is the sky photometric? (flux calibration, standards) | guider flux ratio | System 1 `sky` question | n/a |
| Will the dome close soon? (avoid starting a long exposure) | humidity/wind level and trend | System 1 `dome_risk` question, humidity/wind alerts | n/a |
| Does this frame look right? | quick-look of every FITS frame: trace found, spatial FWHM, S/N at the reference wavelength, saturation, cosmic rays, sky level | `sim/frames.py` quick-look + QC rules; alerts | n/a |
| Point away from the wind | wind speed and direction (high-wind sector) | planner (pointing restriction), alert | yes (as a constraint) |
| Standards and tellurics | program (twilight-OK standards), airmass matching | planner (targets with `kind: standard`) | yes |

## After the night

| Decision | Inputs | Layer |
|---|---|---|
| How good was the night; what would have been optimal? | the true weather, the program | oracle (`planning/oracle.py`): pilot/beam search in hindsight, fractional-knapsack upper bound; console "Debrief" |
| Night log | console events | System 2 (`narrate`) |

## Guardrails

* The LLMs never compute photons or airmass: tools do. A System 2 exposure time is only used if
  it is within a factor of 2 of the ETC plan; otherwise the ETC value is kept.
* Every System 1 answer carries its calibration level; autopilot ignores L0 (uncalibrated)
  answers.
* Actuation needs both the assistant in autopilot and assistant control enabled on the console.
  At a real telescope the assistant only advises: Magellan documents no scriptable TCS
  interface, and the Keck KTL adapter only reads telemetry and prints the commands to type.
