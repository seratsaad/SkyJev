# Correctness audit (2026-09-24)

Scope: `obsassist/astro/*`, `etc.py`, `weather.py`, `sim/night.py`, `planning/*`, `targets.py`,
`instruments/*`, `programs/generator.py` and `console/engine.py`. Independent references: astropy
(AltAz with refraction, `get_body`, `get_sun`, `sidereal_time`, FK5 precession), numerical
integration (scipy), published tables (Kasten & Young 1989, Krisciunas & Schaefer 1991,
Filippenko 1982, Tokovinin 2002). Every check is a test in `tests/test_audit_*.py` (about 30 s).

## Bugs found and fixed

| # | Area | What was wrong | Evidence | Fix | Test |
|---|---|---|---|---|---|
| 1 | Catalogs (`planning/catalogs.py`) | Both the Magellan catalog and the Keck starlist cut the Dec to 10 characters, dropping the tenths of an arcsecond (`+28 51 50.` for +28:51:50.4) and leaving a trailing dot. | Export of `keck1_lris_tonight`. | Print the full 11-character `+dd:mm:ss.s`. | `test_audit_planning.py::test_catalog_exports_parse_back` |
| 2 | Coordinate formatting (`astro/ephem.py`) | `fmt_ra`/`fmt_dec` split into h/m/s before rounding: `01:59:60.00`, `23:59:60.00`, `+89:59:60.0`; also `-00:00:00.0`. This reached the catalogs, FITS headers and the console. | `fmt_ra(15*(1+59/60+59.9999/3600))` returned `01:59:60.00`. | Round to 0.01 s / 0.1" first, then split; no signed zero. | `test_audit_ephem.py::test_sexagesimal_formatting_never_shows_60_seconds` |
| 3 | Oracle upper bound (`planning/oracle.py`) | The bound's capacity counted open-dome minutes only between the program's night limits. `twilight_ok` targets are observed in civil twilight, outside those limits, so the bound could be beaten. It also charged the full goal although a target counts as done at 0.999 of it, and it ignored the last readout of each open interval. | A night open only in twilight had bound 0.0 while greedy scored 4.0. | Capacity runs from civil dusk to civil dawn when any target is `twilight_ok`, plus one readout per open interval. Time is charged for (0.999 goal)^2. | `test_audit_planning.py::test_upper_bound_counts_twilight_time`, `test_pilot_between_greedy_and_upper_bound` |
| 4 | Console readout (`console/engine.py`) | `Arm.readout_s` divided by sqrt(binned pixels), but the configs quote readout times at their own binning. MIKE (2x2) read out in 20.5 s and 24 s instead of 41 s and 48 s, and HIRES (2x1) in 28 s instead of 40 s, while the planner charged the documented values. | `Observatory(...).arms['blue'].readout_s()` returned 20.5. | Scale relative to the configuration's binning. | `test_audit_console.py::test_readout_time_matches_the_configuration` |
| 5 | Console readout speed | Arms always started in "Slow". For configurations documented in fast readout (LDSS3-VPHALL: fast 30 s, slow 166 s) the console took 5.5x longer per frame than the planner and night model assume. | LDSS3 arm `readout_s()` returned 166. | Start in the named speed whose time equals the config's `readout_s`. | `test_audit_console.py::test_console_readout_matches_what_the_planner_assumes` |
| 6 | Console fast-forward | (a) If the dome closed, or the elevation limit was reached, during an acquisition, a stale TCS busy time was left behind, and every later `ff` advanced 0.6 s. (b) `ff` ran past ToO alerts, dome closures and elevation-limit stops. The events were logged, but the observer was not stopped at them. A ToO alert was also raised up to 15 s late. | Closure during acquisition (seed 20): `ff` went from 179.745 to 179.755 min, and so on. | Clear the busy time when an acquisition is abandoned, and ignore it unless slewing or acquiring. The ToO arrival is now an event time. `ff` stops after a step that raises an alert or TO warning. When idle, `ff` still hops 5 min. | `test_dome_closing_during_acquisition_does_not_freeze_fast_forward`, `test_too_alert_at_its_time_and_fast_forward_stops_there`, `test_dome_closing_mid_exposure`, `test_elevation_limit_stops_tracking_and_the_frame_does_not_count`, `test_fast_forward_lands_on_each_event_in_turn` |
| 7 | Console stop | `stop` did not close the shutter until the next simulation step (up to 15 s later). On a paused arm it reopened the shutter first, so time that was never exposed was added to the frame's EXPTIME and to the target's open-shutter time. | Pause, then stop: the frame gained up to 15 s. | `stop` closes the shutter and starts the readout immediately. | `test_audit_console.py::test_pause_resume_stop_abort` |
| 8 | Console `goto` grammar | Coordinates in integer degrees (`goto 150 -20`) were treated as a target name. Out-of-range coordinates (`goto 400 -20`) were accepted. | | Parse the coordinates when both tokens are numeric or sexagesimal, and reject out-of-range values. | `test_audit_console.py::test_goto_grammar_and_limits` |
| 9 | Console ToO | A ToO could be reached with `goto <name>` and observed, with S/N counted, before its alert arrived. | | `goto` by name refuses an unannounced target. | `test_audit_console.py::test_too_cannot_be_observed_before_its_alert` |
| 10 | Console guider | After a focus run, the guider stayed off while the TCS showed "guiding". Guider flux and FWHM readings, which feed the nowcast cloud, stopped until the next `goto`. | | Guiding resumes when the focus completes. | `test_audit_console.py::test_guider_resumes_after_focus` |
| 11 | Snapshot JSON (`planning/planner.py::project`) | Projection blocks carried `numpy.bool_` in `done`. `json.dumps(snapshot)` failed; the HTTP server hid this with its own cleaner. `list_done_min` also used a `>= goal^2` test while "done" everywhere else means 0.999 of the goal. | `json.dumps(obs.snapshot())` raised TypeError. | Use `bool(...)` and `model.done(state)`. | `test_audit_console.py::test_snapshot_is_strict_json`, `test_audit_planning.py::test_projection_time_is_monotonic` |
| 12 | Slew origin (`sim/night.py::observe`, and the same pattern in `planner.project` and `oracle.best_action_forecast`) | After an exposure, the telescope position was taken at the start of the exposure, not where it had tracked to when the shutter closed. A 30-min exposure moves a target by 7.5 deg of hour angle, so the next slew time was wrong. | | Use the position when the shutter closes. | `test_audit_weather_night.py::test_observe_overheads_slew_setup_and_position` |
| 13 | Saturation check (`sim/night.py::exposure`, `segment`, and the console) | The brightest-pixel count was the peak *rate at the start minute* times the duration. It used the lower of the two bracketing defocus levels and ignored changes in seeing, sky and airmass during the exposure. The console used only the first segment's rate. | | Accumulate the peak on the grid (cumulative sums, interpolated in defocus like S and V). `segment` returns `peak_e`, and the console sums it over segments. | `test_exposure_integrals_match_brute_force`, `test_saturation_uses_the_integrated_peak` |
| 14 | Local time (`astro/ephem.py::NightEphem.local`) | Used a fixed UTC-3 for Las Campanas, which is Chile's summer time. From April to early September Chile is on UTC-4, so the console's local clock was off by 1 h. | 2026-06-21 sunset showed 18:53 local; the correct time is 17:53. | Use the tz database (`zoneinfo`, `tz_name`), falling back to the fixed offset. | `test_audit_ephem.py::test_local_time_follows_chile_daylight_saving` |

The public signatures are unchanged. There are two new optional arguments:
`Observatory.advance(dmin, stop_on_alert=False)` and `_next_event_time(include_too=True)`. The
`NightModel.segment()` dict gains `peak_e`, and `peak_rate` is now its time average.

## Verified correct (tests kept as documentation)

* **Target alt/az** (both sites; 2026-01-15, 06-21, 09-24, 12-10; Dec -85 to +85, including the
  zenith and circumpolar targets): agrees with astropy AltAz (refraction on) to < 0.01 deg above
  10 deg. The azimuth convention is N=0, E=90.
* **LST** agrees with astropy mean sidereal time to < 0.13 s. The hour angle is wrapped to
  [-180, 180).
* **Precession to date** agrees with astropy FK5 to < 1e-6 deg. The IAU 1976 fallback agrees
  to < 1".
* **Twilight times** (sunset, -6, -12 and -18 deg, both ends) agree with astropy root finding
  to < 20 s. The grid spans sunset - 20 min to sunrise + 20 min of the local evening of `date`
  at both western-longitude sites.
* **Moon** alt/az agrees to < 0.08 deg; the 10-min interpolation error dominates. The
  illuminated fraction agrees with astroplan's formula to < 0.01. The KS91 phase angle is
  0 at full Moon; ours is topocentric, so it differs from the geocentric value by less than the
  lunar parallax (< 1 deg). Moon-target separation agrees to < 0.2 deg.
* **Parallactic angle** matches q = atan2(sin H, tan(lat) cos(dec) - sin(dec) cos H) and agrees
  with astropy's zenith position angle to < 0.1 deg (15-80 deg elevation). It is measured from
  the north of date: for J2000 PAs the difference is n sin(alpha) sec(dec) x 26.7 yr, about 0.4
  deg at Dec 70. This is negligible for the ADR slit loss.
* **`NightEphem.point()`** agrees with `targets()` to 1e-5 deg.
* **Refraction** (Saemundsson): 3-5 % above ERFA between 10 and 80 deg (3" at 45 deg; this is
  the known bias of the Bennett/Saemundsson form). Its pressure and temperature scaling matches
  ERFA's to 1 %. Kasten & Young airmass reproduces the published values (26.31 at 1 deg,
  10.32 at 5 deg).
* **Sky model**: KS91 reproduces Table 2 and its nL-to-V conversion inverts eq. 1. The AB
  conversion (+0.02 at V) and the solar colour are correct. Twilight brightness is monotonic.
  The dark sky brightens with airmass (eq. 2). `sky_ab` is finite over 160k random inputs.
* **Seeing**: Kolmogorov scaling, and the Tokovinin (2002) von Karman factor with r0 taken at
  the observing wavelength and airmass.
* **ADR** (Filippenko 1982): 0.71" at 4000 vs 5000 A, airmass 1.5, 600 mmHg, matching his
  Table 1. At LCO (765 hPa) 4000-6500 A at airmass 1.5 gives 1.15". The ~1.5-2" often quoted is
  a sea-level figure. Blue refracts more; ADR scales with tan z.
* **ETC**: an AB=0 source gives 996 photons s^-1 cm^-2 A^-1 at 5500 A. The Moffat FWHM, 2-D
  peak, LSF peak, strip/slit fractions (with offsets) and enclosed energy match numerical
  integrals to 1e-6. The per_A, per_pix and per_resel scalings are correct. The
  point-source spectroscopic rates were rebuilt by hand from the formula.
  `plan_exposures` handles a goal already met, a single short exposure, several
  maximum-length exposures, saturation limits, very faint targets and a preferred length.
* **Weather**: runs are reproducible and every channel stays in range. The seeing median is
  within 12 % of the site DIMM median at both sites. The dome never stays open above a limit
  and reopens only after 30 consecutive good minutes. It reopens at the 30th good sample, which
  is 1 grid minute early and was left as is.
* **Night model**: exposure integrals match 1-s brute-force summation at defocus 0, 0.1, 0.3
  and 0.9. Truncation at the target limit, the dome closure and the end of the night is
  correct. The window QC rejects above 50 % outside. `segment()` is additive. The defocus
  interpolation is monotonic and exact at the precomputed levels. Slews take the shortest
  azimuth path. A setup change is charged only when the setup differs; LRIS-B and LRIS-R share
  the D560 setup. Continuing on the same target is free. The Keck Nasmyth-deck limits are
  applied. In high wind, pointings within `wind_avoid_deg` of the wind are blocked.
* **Planner and oracle**: the feasibility notes are correct (Moon, not yet announced, window
  opens, done). The exposure length accounts for saturation, including sky saturation in
  twilight. Projection time is monotonic. Pilot >= greedy <= upper bound on random programs
  for Clay, Keck I and Keck II. `batch.py` runs.
* **Console**: goto by name and by coordinates, refusal below the limit, offsets, rotator,
  focus, arm settings, automatic loops, readout progress in lines, the elevation-limit stop,
  the dome closing mid-exposure, S/N counted only on the arm matching the target's
  configuration, and strict-JSON snapshots.

## Noted, not changed

* `NightModel.observe(apply=False)` changes the caller's state in place. The name suggests the
  opposite. Nothing calls it that way.
* `assistant/adapters/manual.py` also sets the telescope position at the start of the exposure
  (see bug 12). It is outside the audit scope.
* `planner.Nowcast.from_history` (being reworked by the lead) computes `night_cloud` from the
  true `weather.cloud_mag` with no noise. This is information the observer does not have
  unless guiding; the planner's docstring says it never reads the truth.
* The spectroscopic peak pixel uses the full Moffat marginal. For slits much narrower than the
  PSF, the transmitted profile is up to ~8 % more peaked. This is covered by the 0.8 linearity
  and the planner's 0.7 margin.
* For MIKE-RED and FIRE, the configured R differs from slit width / dispersion; the per_resel
  unit uses the configured R. This is instrument data, not code.
