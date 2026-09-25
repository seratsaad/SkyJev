# W. M. Keck Observatory parameters (simulator + ETC)

Compiled 2026-09-24 for `obsassist`. Machine-readable twin: [`keck_parameters.json`](keck_parameters.json). Every leaf is `{value, unit, source, confidence, note}`. The tables below are generated from the same Python dict as the JSON, so the two files agree. 530 leaves: 473 documented, 33 approximate, 24 estimate.

**Confidence key:** `doc` = stated in the cited source. `approx` = derived by arithmetic from documented numbers, read by me from a published data file, taken from a secondary source, or an older/proxy measurement. `**EST**` = not found publicly; the value is my reasoned estimate, and the note gives the reasoning.

## 0. Found vs. estimated (read this first)

- **Well documented (Keck pages):**
  - Telescope: pointing limits including the Nasmyth-deck az/el zones for K1 and K2, slew rates and accelerations, zenith blind spot, the 85 deg guiding limit, and the typical 5-min acquisition and 6-s offset/settle.
  - Night definition: the night runs between 12 deg twilights, and split nights divide at the midpoint between them.
  - Weather closure and reopening rules: wind, dew point, fog, precipitation, ice, clouds.
  - Detectors: read noise, gain, dark current for DEIMOS, MOSFIRE and HIRES, and linearity for every instrument.
  - Readout times for every instrument and mode: LRIS red and blue by binning and amps, DEIMOS, HIRES by binning, MOSFIRE MCDS, KCWI by amps/speed/binning.
  - Dispersers: dispersion, coverage and resolution for all LRIS grisms and gratings, DEIMOS 600ZD/830G/900ZD/1200G/1200B, MOSFIRE YJHK, HIRES deckers, and KCWI gratings.
  - Throughput curves for DEIMOS (2018, per grating), MOSFIRE (per band) and LRIS-B (per grism).
  - Recommended exposure times: MOSFIRE per band, the LRIS-R 20-min cosmic-ray limit, KCWI red, DEIMOS.
  - KTL `show`/`modify`/`waitfor` syntax, with DCS, LRIS, DEIMOS and MOSFIRE keyword names and real offset commands taken from Keck script sources.
- **Site data (published, but not Keck-specific):**
  - Maunakea extinction: the Buton+2013 curve, Krisciunas, MKO JHK, and MegaCam ugriz terms.
  - Sky brightness: MegaCam ugriz (AB), CFHT UBVRIJHK, Gemini percentiles, MOSFIRE NIR.
  - Seeing: MKAM DIMM median 0.65 arcsec, TMT 13N, and the Keck-2 visible median of 0.6 arcsec.
- **ETC anchors found:**
  - DEIMOS predicted S/N and the DEEP2 S/N-vs-R_AB table (1 h, 1200G).
  - MOSFIRE: Ks = 23.5 at S/N 10 in 1 h, the MOSDEF 3-sigma depths, and 5-sigma line fluxes.
  - HIRES: Kp = 13, 45 min, S/N 75 per pixel.
  - LRIS effective area of ~10 m^2 (blue) and ~15 m^2 (red), from Perley 2019.
  - Zero points for LRIS (1994), DEIMOS and MOSFIRE.
- **Estimated (not public):**
  - DIMM quartiles (lognormal fit).
  - Post-slew settle time and dome rotation rate.
  - Faint-target long-slit acquisition time (~10 min) and LRIS/MOSFIRE mask alignment times.
  - LRIS dark currents and the LRIS red Mark IV throughput.
  - Mechanism times: LRIS red filter/slitmask/tilt, DEIMOS filter/slitmask/zeroth-order tilt, MOSFIRE mode change.
  - DEIMOS full well, and HIRES and KCWI peak throughput.
  - A KCWI numeric sensitivity anchor was not found.
- **Relative-humidity limit:** the Keck table enforces humidity only through the 2 C dew-point margin. RH < 95% is a secondary-source proxy (van Kooten & Izett 2022).
- **Caveats and source conflicts:**
  - Two Keck coordinate sets: 4123 m (Keck guide) vs 4160 m (IRAF).
  - Four collecting areas are quoted (72.3 / 73.3 / 76.2 / 77.9 m^2). Use **72.3 m^2**, the net usable f/15 area.
  - The Keck 2002 guide gives U/B sky values 0.7-1.6 mag fainter than CFHT; prefer MegaCam/CFHT.
  - HIRES gains differ slightly between two Keck pages.
  - The DEIMOS primer readout times (40/80 s) are old.
  - The LRIS keyword list predates the 2021 red CCD.
  - Instrument status: DEIMOS CCD5 is dead, and MOSFIRE was warmed for service in Aug 2026.

## Quick-use defaults for the simulator

| Item | Default | Why |
|---|---|---|
| Collecting area | 72.3 m^2 | Nelson 1994 net f/15 area |
| Slew | az 1.3 deg/s (0.05 deg/s^2), el 0.5 deg/s (0.03 deg/s^2); + 15 s settle **EST** | Keck guide; settle estimated |
| Acquisition | 5 min bright/visible; +10 min **EST** blind-offset long-slit; +5 min DEIMOS mask align (DEEP2) | NIRC manual, LRIS/DEIMOS procedures |
| Elevation limit | K1: el >= 33.3 for az 5.3-146.2, else >= 18. K2: el >= 36.8 for az 185.3-332.8, else >= 18. Upper 88.9 (K1) / 89.5 (K2); guiding unreliable > 85 | TelLimits |
| Night | 12 deg twilight to 12 deg twilight | Keck metrics / split-night pages |
| Seeing | lognormal, median 0.65 arcsec (500 nm), sigma_ln 0.38; x airmass^0.6; x lambda^-0.2 | MKAM + TMT + Gemini |
| Extinction | Buton+2013 curve (optical) plus MKO J/H/K 0.05/0.03/0.05 | see site.extinction |
| Dark sky (AB) | u 22.7, g 22.0, r 21.3, i 20.3, z 19.4 | CFHT MegaCam |
| Close if | wind >= 45 mph sustained; T - Tdew <= 2 C for 5 min; fog within 200 m; precipitation | Keck weather table |
| Reopen after | wind <= 40 mph for 30 min (gusts < 45); dew margin > 2 C for 20-30 min | Keck weather table |


## 1. Maunakea site

### site

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| name | W. M. Keck Observatory, Maunakea, Hawaii |  | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) |  |
| latitude_deg | 19.8267 | deg (N positive) | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | Keck guide (2002): 19d49.6m N. IRAF/astropy 'keck' entry: 19.828333. |
| longitude_deg | -155.473 | deg (E positive) | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | Keck guide (2002): 155d28.4m W. IRAF/astropy 'keck' entry: 204.521667 E = -155.478333. |
| elevation_m | 4123 | m | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | Keck guide (2002) altitude. IRAF/astropy 'keck' entry uses 4160 m; commonly quoted 4145 m. Differences are below what matters for an ETC. |
| astropy_keck_site | {"lat": 19.828333, "lon": -155.478333, "height_m": 4160} | deg, deg, m | doc | [github.com/astropy/ast...coordinates/sites.json](https://github.com/astropy/astropy-data/blob/gh-pages/coordinates/sites.json) | Source field in the registry: 'IRAF Observatory Database'. Use EarthLocation.of_site('keck'). |
| telescope_separation_m | 85 | m | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | Keck II lies roughly NE of Keck I. |
| timezone | Pacific/Honolulu (HST = UTC-10, no DST) |  | doc | [keck/inst/common/facsum.html](https://www2.keck.hawaii.edu/inst/common/facsum.html) | FACSUM: UT is '10 hours ahead of HST'. |
| mean_pressure_mbar | 605 | mbar | doc | [cfht.hawaii.edu/Instru...oryManual_(Sec_2).html](https://www.cfht.hawaii.edu/Instruments/ObservatoryManual/CFHT_ObservatoryManual_(Sec_2).html) | CFHT manual: unit airmass at Maunakea equals 0.60 sea-level airmasses. Read via web.archive.org; the live page returns 403. |
| night_temperature_C | 2.5 +/- 4 (about 90% of the time) | degC | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | TMT 13N median night temperature at 2 m: 2.3 C. |
| median_wind_speed_m_s | 7 | m/s | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | Keck guide: ~7 m/s (~14 mph). CFHT: 50% &lt; 7 m/s, 84% &lt; 12 m/s, about 5% &gt; 30 m/s (telescope closed). |
| pwv_median_mm | 1.9 | mm | doc | [arxiv.org/abs/0904.1183](https://arxiv.org/abs/0904.1183) | TMT 13N site (4050 m): median 1.9 mm, and PWV &lt; 2 mm 54% of the time. The CFHT manual quotes a median of 0.9 mm (older, summit). |
| usable_night_fraction | {"usable": 0.8, "photometric": 0.55, "spectroscopic": 0.25} | fraction of nights | doc | [cfht.hawaii.edu/Instru...oryManual_(Sec_2).html](https://www.cfht.hawaii.edu/Instruments/ObservatoryManual/CFHT_ObservatoryManual_(Sec_2).html) | CFHT manual (2003). Gemini treats its 50th-percentile cloud-cover bin at MK as 'photometric' (no signal loss). |
| cloud_cover_percentiles_gemini | {"CC50": "photometric, no loss", "CC70": "patchy/thin cirrus, loss &lt;0.3 mag", "CC80": "cloudy, 0.3-1 mag loss", "any": "&gt;1 mag loss (ITC assumes 3 mag)"} |  | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) | Based on long-term Maunakea data. |

#### site / sky_brightness

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| note | Pick one table consistently. For optical ETC work in AB, use the MegaCam values. Use the Gemini percentiles to model the lunar phase, and the MOSFIRE tables for NIR spectroscopy between OH lines. |  | approx | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | Editorial guidance, not a measurement. |
| mosfire_band_avg_spec_sky_AB_derived | {"Y": 17.83, "J": 16.85, "H": 15.66, "K": 16.17} | mag/arcsec^2 (AB), OH lines included | approx | [keck/inst/mosfire/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/mosfire/exposure_recipes.html) | My flux-average of the MOSFIRE night-sky FITS spectra (data/MosfireSkyspec/*_abmag.fits), linked from sky_lines.html. |
| moon_brightening_cfht | {"U_quarter": 5, "U_full": 65, "V_quarter": 1.3, "V_full": 5} | multiplicative factor on dark sky | doc | [cfht.hawaii.edu/Instru...oryManual_(Sec_2).html](https://www.cfht.hawaii.edu/Instruments/ObservatoryManual/CFHT_ObservatoryManual_(Sec_2).html) | Rough, for cirrus-free nights. |
| airmass_scaling | sky brightness ~ proportional to airmass (first approximation) |  | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) |  |
| nir_oh_variability | {"timescale_min": "5-15", "amplitude_pct": "5-10"} |  | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) | OH declines steadily for the first 1-2 h after sunset. MOSFIRE team: OH lines vary by &gt;0.1% on ~30 s timescales. |

#### site / sky_brightness / megacam_dark_zenith_AB

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| u | 22.7 | mag/arcsec^2 (AB) | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | CFHT MegaCam, zenith, Moon 0%. Read via web.archive.org. |
| g | 22 | mag/arcsec^2 (AB) | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | CFHT MegaCam, zenith, Moon 0%. Read via web.archive.org. |
| r | 21.3 | mag/arcsec^2 (AB) | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | CFHT MegaCam, zenith, Moon 0%. Read via web.archive.org. |
| i | 20.3 | mag/arcsec^2 (AB) | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | CFHT MegaCam, zenith, Moon 0%. Read via web.archive.org. |
| z | 19.4 | mag/arcsec^2 (AB) | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | CFHT MegaCam, zenith, Moon 0%. Read via web.archive.org. |

#### site / sky_brightness / megacam_grey_zenith_AB_moon30pct

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| u | 21.2 | mag/arcsec^2 (AB) | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | Moon 30%. |
| g | 21.3 | mag/arcsec^2 (AB) | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | Moon 30%. |
| r | 20.8 | mag/arcsec^2 (AB) | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | Moon 30%. |
| i | 20.3 | mag/arcsec^2 (AB) | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | Moon 30%. |
| z | 19.4 | mag/arcsec^2 (AB) | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | Moon 30%. |

#### site / sky_brightness / megacam_bright_zenith_AB_moon60pct

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| u | 19.7 | mag/arcsec^2 (AB) | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | Moon 60%. |
| g | 20.5 | mag/arcsec^2 (AB) | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | Moon 60%. |
| r | 20.4 | mag/arcsec^2 (AB) | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | Moon 60%. |
| i | 20.3 | mag/arcsec^2 (AB) | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | Moon 60%. |
| z | 19.4 | mag/arcsec^2 (AB) | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | Moon 60%. |

#### site / sky_brightness / cfht_manual_dark_zenith_vega

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| U | 21.6 | mag/arcsec^2 (Vega, presumed) | doc | [cfht.hawaii.edu/Instru...oryManual_(Sec_2).html](https://www.cfht.hawaii.edu/Instruments/ObservatoryManual/CFHT_ObservatoryManual_(Sec_2).html) | CFHT Observatory Manual Sec. 2, 'average sky brightness at zenith during dark time'. |
| B | 22.3 | mag/arcsec^2 (Vega, presumed) | doc | [cfht.hawaii.edu/Instru...oryManual_(Sec_2).html](https://www.cfht.hawaii.edu/Instruments/ObservatoryManual/CFHT_ObservatoryManual_(Sec_2).html) | CFHT Observatory Manual Sec. 2, 'average sky brightness at zenith during dark time'. |
| V | 21.1 | mag/arcsec^2 (Vega, presumed) | doc | [cfht.hawaii.edu/Instru...oryManual_(Sec_2).html](https://www.cfht.hawaii.edu/Instruments/ObservatoryManual/CFHT_ObservatoryManual_(Sec_2).html) | CFHT Observatory Manual Sec. 2, 'average sky brightness at zenith during dark time'. |
| R | 20.3 | mag/arcsec^2 (Vega, presumed) | doc | [cfht.hawaii.edu/Instru...oryManual_(Sec_2).html](https://www.cfht.hawaii.edu/Instruments/ObservatoryManual/CFHT_ObservatoryManual_(Sec_2).html) | CFHT Observatory Manual Sec. 2, 'average sky brightness at zenith during dark time'. |
| I | 19.2 | mag/arcsec^2 (Vega, presumed) | doc | [cfht.hawaii.edu/Instru...oryManual_(Sec_2).html](https://www.cfht.hawaii.edu/Instruments/ObservatoryManual/CFHT_ObservatoryManual_(Sec_2).html) | CFHT Observatory Manual Sec. 2, 'average sky brightness at zenith during dark time'. |
| J | 14.8 | mag/arcsec^2 (Vega, presumed) | doc | [cfht.hawaii.edu/Instru...oryManual_(Sec_2).html](https://www.cfht.hawaii.edu/Instruments/ObservatoryManual/CFHT_ObservatoryManual_(Sec_2).html) | CFHT Observatory Manual Sec. 2, 'average sky brightness at zenith during dark time'. |
| H | 13.4 | mag/arcsec^2 (Vega, presumed) | doc | [cfht.hawaii.edu/Instru...oryManual_(Sec_2).html](https://www.cfht.hawaii.edu/Instruments/ObservatoryManual/CFHT_ObservatoryManual_(Sec_2).html) | CFHT Observatory Manual Sec. 2, 'average sky brightness at zenith during dark time'. |
| K | 12.6 | mag/arcsec^2 (Vega, presumed) | doc | [cfht.hawaii.edu/Instru...oryManual_(Sec_2).html](https://www.cfht.hawaii.edu/Instruments/ObservatoryManual/CFHT_ObservatoryManual_(Sec_2).html) | CFHT Observatory Manual Sec. 2, 'average sky brightness at zenith during dark time'. |

#### site / sky_brightness / keck_guide_2002_vega

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| U | 23.2 | mag/arcsec^2 | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | Keck guide 2002. U and B are 0.7-1.6 mag fainter than the CFHT manual values; treat with caution. |
| B | 23 | mag/arcsec^2 | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) |  |
| V_excl_OI5577 | 22.6 | mag/arcsec^2 | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | Excluding the [OI] 5577 line. |
| V_incl_OI | 21.25-22 | mag/arcsec^2 | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | 'various estimates that probably include the OI line'. |
| R | 22 | mag/arcsec^2 | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | ~22. |

#### site / sky_brightness / gemini_mk_V_percentiles_vega

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| SB20_darkest | 21.3 | mag/arcsec^2 V | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) | &lt;=3 nights from new moon. U=V+0.0, B=V+0.8, R=V-0.9. |
| SB50_dark | 20.7 | mag/arcsec^2 V | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) | &lt;=7 nights. U=V-1.5, B=V+0.2, R=V-0.8. |
| SB80_grey | 19.5 | mag/arcsec^2 V | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) | &lt;=11 nights. U=V-2.2, B=V-0.0, R=V-0.4. |
| any_bright | 18 | mag/arcsec^2 V | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) | &lt;=14 nights. U=V-3.0, B=V-0.5, R=V-0.1. |
| random_target_median | 20.78 | mag/arcsec^2 V | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) | Fainter than 20.78 for 50% of the time and than 21.37 for 20% of the time, for a random target. |
| ecliptic_penalty | 0.4 | mag | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) | V sky ~0.4 mag brighter at low ecliptic latitude. |

#### site / sky_brightness / gemini_mk_nir_vega

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| J | 16.1 | mag/arcsec^2 (Vega) | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) | Gemini Maunakea near-IR background, any lunar phase. |
| H | 13.8 | mag/arcsec^2 (Vega) | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) | Gemini Maunakea near-IR background, any lunar phase. |
| K | 14.8 | mag/arcsec^2 (Vega) | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) | Gemini Maunakea near-IR background, any lunar phase. |

#### site / sky_brightness / mosfire_imaging_sky

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| Y | {"vega": 17.44, "ab": 18.09} | mag/arcsec^2 | doc | [keck/inst/mosfire/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/mosfire/exposure_recipes.html) | MOSFIRE imaging. SPIE 2012 range 17.37-17.50 Vega (full moon). |
| J | {"vega": 15.78, "ab": 16.58} | mag/arcsec^2 | doc | [keck/inst/mosfire/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/mosfire/exposure_recipes.html) | SPIE range 15.49-15.87. |
| H | {"vega": 13.74, "ab": 15.14} | mag/arcsec^2 | doc | [keck/inst/mosfire/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/mosfire/exposure_recipes.html) | SPIE range 13.59-13.89. |
| Ks | {"vega": 13.68, "ab": 15.54} | mag/arcsec^2 | doc | [keck/inst/mosfire/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/mosfire/exposure_recipes.html) | SPIE range 13.68-14.25. Includes telescope thermal emission. |

#### site / sky_brightness / mosfire_spec_between_OH

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| Y | {"vega": 18.73, "ab": 19.38, "e_per_s_per_pix": 0.34} | mag/arcsec^2; e-/s/spatial pix | doc | [ucolick.org/documents/mosfire_spie.pdf](https://www.ucolick.org/documents/mosfire_spie.pdf) | 0.7 arcsec slit, per spatial pixel, between OH lines. Measured during full moon. |
| J | {"vega": 18.34, "ab": 19.24, "e_per_s_per_pix": 0.3} |  | doc | [ucolick.org/documents/mosfire_spie.pdf](https://www.ucolick.org/documents/mosfire_spie.pdf) |  |
| H | {"vega": 17.32, "ab": 18.72, "e_per_s_per_pix": 0.57} |  | doc | [ucolick.org/documents/mosfire_spie.pdf](https://www.ucolick.org/documents/mosfire_spie.pdf) |  |
| K_lt_2.2um | {"vega": 16.35, "ab": 18.21, "e_per_s_per_pix": 0.39} |  | doc | [ucolick.org/documents/mosfire_spie.pdf](https://www.ucolick.org/documents/mosfire_spie.pdf) |  |
| K_2.35_2.41um | {"vega": 14.05, "ab": 15.91, "e_per_s_per_pix": 8.0} |  | doc | [ucolick.org/documents/mosfire_spie.pdf](https://www.ucolick.org/documents/mosfire_spie.pdf) | Thermal part of K. |

#### site / extinction

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| buton2013_median_curve | *(69 rows; full table in JSON / appendix)* [[3200, 0.856], [3300, 0.588], [3400, 0.514], [3500, 0.448], [3600, 0.4], [3700, 0.359], [3800, 0.323], [3900, 0.292], [ ... | [wavelength_A, mag/airmass] | doc | [arxiv.org/abs/1210.2619](https://arxiv.org/abs/1210.2619) | SNfactory median, 3200-10000 A in 100 A bins (A&A 549, A8, Table 6). Continuum only: add O2/H2O telluric bands separately. Variability rms is ~0.057 mag/airmass at 3300 A and ~0.003 at 9000 A. |

#### site / extinction / megacam_airmass_terms_AB

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| u | 0.35 | mag/airmass | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | CFHT MegaCam photometric airmass term k. |
| g | 0.15 | mag/airmass | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | CFHT MegaCam photometric airmass term k. |
| r | 0.1 | mag/airmass | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | CFHT MegaCam photometric airmass term k. |
| i | 0.04 | mag/airmass | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | CFHT MegaCam photometric airmass term k. |
| z | 0.03 | mag/airmass | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | CFHT MegaCam photometric airmass term k. |

#### site / extinction / krisciunas_median_UKIRT_page

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| U | 0.358 | mag/airmass | doc | [keck/inst/common/exts.html](https://www2.keck.hawaii.edu/inst/common/exts.html) | n=1 only. Krisciunas et al. 1987 (PASP 99, 887), Dec 1980 to May 1996 compilation. |
| B | 0.194 | mag/airmass | doc | [keck/inst/common/exts.html](https://www2.keck.hawaii.edu/inst/common/exts.html) | median; mean 0.198+/-0.008 |
| V | 0.111 | mag/airmass | doc | [keck/inst/common/exts.html](https://www2.keck.hawaii.edu/inst/common/exts.html) | median; mean 0.119 |
| J | 0.102 | mag/airmass | doc | [keck/inst/common/exts.html](https://www2.keck.hawaii.edu/inst/common/exts.html) | Old UKIRT J filter. Much higher than MKO-filter values. |
| H | 0.059 | mag/airmass | doc | [keck/inst/common/exts.html](https://www2.keck.hawaii.edu/inst/common/exts.html) | old UKIRT filter |
| K | 0.088 | mag/airmass | doc | [keck/inst/common/exts.html](https://www2.keck.hawaii.edu/inst/common/exts.html) | old UKIRT filter |

#### site / extinction / mko_filters_leggett2006

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| J | 0.047 | mag/airmass | doc | [academic.oup.com/mnras/article/373/2/781/1277834](https://academic.oup.com/mnras/article/373/2/781/1277834) | +/-0.024 (21 photometric nights, UKIRT/WFCAM-era MKO filters) |
| H | 0.029 | mag/airmass | doc | [academic.oup.com/mnras/article/373/2/781/1277834](https://academic.oup.com/mnras/article/373/2/781/1277834) | +/-0.020 |
| K | 0.052 | mag/airmass | doc | [academic.oup.com/mnras/article/373/2/781/1277834](https://academic.oup.com/mnras/article/373/2/781/1277834) | +/-0.028 |

#### site / extinction / mko_filters_tokunaga2002_2mm_pwv

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| J | 0.015 | mag/airmass | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) | Tokunaga, Simons & Vacca 2002, as tabulated by Gemini, for 2 mm PWV. |
| H | 0.015 | mag/airmass | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) |  |
| K | 0.033 | mag/airmass | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) |  |

#### site / extinction / buton2013_at_band_centers_derived

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| u_3550 | 0.424 | mag/airmass | approx | [arxiv.org/abs/1210.2619](https://arxiv.org/abs/1210.2619) | Buton Table 6 interpolated at 3550 A. |
| B_4400 | 0.185 | mag/airmass | approx | [arxiv.org/abs/1210.2619](https://arxiv.org/abs/1210.2619) | Buton Table 6 interpolated at 4400 A. |
| g_4750 | 0.143 | mag/airmass | approx | [arxiv.org/abs/1210.2619](https://arxiv.org/abs/1210.2619) | Buton Table 6 interpolated at 4750 A. |
| V_5500 | 0.106 | mag/airmass | approx | [arxiv.org/abs/1210.2619](https://arxiv.org/abs/1210.2619) | Buton Table 6 interpolated at 5500 A. |
| r_6250 | 0.081 | mag/airmass | approx | [arxiv.org/abs/1210.2619](https://arxiv.org/abs/1210.2619) | Buton Table 6 interpolated at 6250 A. |
| R_6400 | 0.07 | mag/airmass | approx | [arxiv.org/abs/1210.2619](https://arxiv.org/abs/1210.2619) | Buton Table 6 interpolated at 6400 A. |
| i_7650 | 0.031 | mag/airmass | approx | [arxiv.org/abs/1210.2619](https://arxiv.org/abs/1210.2619) | Buton Table 6 interpolated at 7650 A. |
| I_8000 | 0.027 | mag/airmass | approx | [arxiv.org/abs/1210.2619](https://arxiv.org/abs/1210.2619) | Buton Table 6 interpolated at 8000 A. |
| z_9000 | 0.019 | mag/airmass | approx | [arxiv.org/abs/1210.2619](https://arxiv.org/abs/1210.2619) | Buton Table 6 interpolated at 9000 A. |

#### site / seeing

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| mkam_dimm_median | 0.65 | arcsec (500 nm, zenith) | approx | [academic.oup.com/mnras...cle/496/4/4734/5865137](https://academic.oup.com/mnras/article/496/4/4734/5865137) | MKAM DIMM (between CFHT and Gemini-N), 2009-2020, Lyman et al. 2020 mean/median table. Read from the MNRAS HTML through a text-extraction fetch because the PDF host was unreachable; two independent fetches agreed. The text also calls 0.65 arcsec the typical total seeing. |
| mkam_dimm_mean | 0.7 | arcsec | approx | [academic.oup.com/mnras...cle/496/4/4734/5865137](https://academic.oup.com/mnras/article/496/4/4734/5865137) | Same retrieval caveat. |
| mkam_mass_free_atm_median | 0.35 | arcsec | approx | [academic.oup.com/mnras...cle/496/4/4734/5865137](https://academic.oup.com/mnras/article/496/4/4734/5865137) | Mean 0.45. Same retrieval caveat. |
| mkam_ground_layer_median | 0.5 | arcsec | approx | [academic.oup.com/mnras...cle/496/4/4734/5865137](https://academic.oup.com/mnras/article/496/4/4734/5865137) | Mean 0.47. The ground layer is 2/3 to 3/4 of the total seeing. Same retrieval caveat. |
| tmt13n_dimm_median | 0.75 | arcsec | doc | [arxiv.org/abs/0904.1183](https://arxiv.org/abs/0904.1183) | TMT 13N (4050 m, ~150 m below summit, 7 m DIMM height). 10th percentile 0.46. |
| tmt13n_mass_median | 0.33 | arcsec | doc | [arxiv.org/abs/0904.1183](https://arxiv.org/abs/0904.1183) | 10th percentile 0.15 |
| tmt13n_isoplanatic_angle | 2.69 | arcsec | doc | [arxiv.org/abs/0904.1183](https://arxiv.org/abs/0904.1183) |  |
| tmt13n_tau0_ms | 5.1 | ms | doc | [arxiv.org/abs/0904.1183](https://arxiv.org/abs/0904.1183) |  |
| dimm_quartiles | {"p10": 0.4, "p25": 0.5, "p50": 0.65, "p75": 0.84, "p90": 1.06} | arcsec | **EST** | [academic.oup.com/mnras...cle/496/4/4734/5865137](https://academic.oup.com/mnras/article/496/4/4734/5865137) | No published quartile table found. Lognormal with median 0.65 (MKAM) and sigma_ln = 0.38. sigma_ln comes from the TMT 13N 10%/50% ratio: 0.46/0.75 gives sigma = -ln(0.613)/1.2816. The implied P(&gt;1.5 arcsec) is ~1.4%, consistent with Marcy+2014 (&lt;10%). |
| seeing_gt_1p5_fraction | &lt;0.10 | fraction of time | doc | [arxiv.org/abs/1401.4195](https://arxiv.org/abs/1401.4195) | Stated in passing in Marcy et al. 2014 (HIRES). |
| keck2_visible_median | 0.6 | arcsec (0.5 um) | doc | [keck/optics/lgsao/lgsbasics.html](https://www2.keck.hawaii.edu/optics/lgsao/lgsbasics.html) | 'approximately the median for the Keck 2 telescope' (LGS AO page). |
| keck_median_K_band | 0.5 | arcsec FWHM (K) | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | Keck guide 2002. |
| keck_telescope_optics_EE | 80% of energy within 0.4 arcsec diameter; FWHM ~0.2 arcsec |  | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | Segment + f/15 secondary + tertiary optical quality. Add in quadrature to the seeing. |
| cfht_free_atmosphere_median | 0.4 | arcsec | doc | [cfht.hawaii.edu/Instru...oryManual_(Sec_2).html](https://www.cfht.hawaii.edu/Instruments/ObservatoryManual/CFHT_ObservatoryManual_(Sec_2).html) | Winter 0.45, summer 0.35, 10th percentile ~0.25. |
| gemini_mk_iq_percentiles_zenith | {"g": [0.6, 0.85, 1.1, 1.9], "r": [0.5, 0.75, 1.05, 1.8], "i": [0.5, 0.75, 1.05, 1.7], "Z": [0.5, 0.7, 0.95, 1.7], "J": [0.4, 0.6, 0.85, 1.55], "H": [0.4, 0.6, 0.85, 1.5], "K": [0.35, 0.55, 0.8, 1.4]} | arcsec FWHM at [20%, 70%, 85%, 100%] percentile | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) | Delivered IQ at Gemini-N (8 m), a good proxy for Keck. |
| airmass_scaling | FWHM proportional to airmass^0.6 (visible and short-wave IR) |  | doc | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) |  |
| wavelength_scaling | FWHM proportional to lambda^-0.2 (Kolmogorov) |  | **EST** | [gemini.edu/observing/telescopes-and-sites/sites](https://www.gemini.edu/observing/telescopes-and-sites/sites) | Standard Kolmogorov scaling, not quoted by a Keck page. The Gemini IQ table is roughly consistent with it. |
| seasonal | best in summer (Jun-Sep) |  | doc | [academic.oup.com/mnras...cle/496/4/4734/5865137](https://academic.oup.com/mnras/article/496/4/4734/5865137) | June and September have the most nights with total seeing &lt; 0.6 arcsec. |

#### site / seeing / delivered_iq_examples

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| deimos_good | 0.5 | arcsec FWHM | doc | [keck/inst/deimos/primer.html](https://www2.keck.hawaii.edu/inst/deimos/primer.html) | 'FWHM=0.5 arcsec is achievable under good seeing conditions'. |
| mosdef_slit_star_range | 0.48-0.99 (typ. 0.55-0.80) | arcsec FWHM (NIR) | doc | [arxiv.org/abs/1412.1835](https://arxiv.org/abs/1412.1835) | MOSFIRE MOSDEF masks, Table 1. |
| mosfire_ks_stack | 0.56 | arcsec FWHM | doc | [ucolick.org/documents/mosfire_spie.pdf](https://www.ucolick.org/documents/mosfire_spie.pdf) | 1 h Ks stack, commissioning. |
| cfht_megacam_median | {"g": 0.77, "i": 0.66} | arcsec | doc | [cfht.hawaii.edu/Instru...eneralinformation.html](https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html) | 2003A sample, CFHT (not Keck). |

#### site / weather_closure

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| policy_note | Guidelines only. The Observing Assistant has ultimate authority. |  | doc | [keck/observing/Weather_Table.html](https://www2.keck.hawaii.edu/observing/Weather_Table.html) |  |
| wind_close | sustained 45 mph; OR sustained &gt;=25 mph with moderate dust accumulation | mph | doc | [keck/observing/Weather_Table.html](https://www2.keck.hawaii.edu/observing/Weather_Table.html) | 45 mph = 20.1 m/s |
| wind_reopen | sustained &lt;=40 mph for &gt;=30 min, gusts &lt;45 mph, minimal dust after a 30-min buildup test, no other adverse weather |  | doc | [keck/observing/Weather_Table.html](https://www2.keck.hawaii.edu/observing/Weather_Table.html) |  |
| wind_limit_2002_guide | 50 | mph | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | Older guide: 'allowable wind velocity with dome open: 50 mph', which may vary with dust and dome orientation. |
| dewpoint_close | dewpoint within 2 C of the outside, secondary or primary-mirror temperature, sustained 5 min; or condensation on outside surfaces | degC | doc | [keck/observing/Weather_Table.html](https://www2.keck.hawaii.edu/observing/Weather_Table.html) | Humidity is enforced through the dew-point margin; the table gives no RH%. |
| dewpoint_reopen | dewpoint margin &gt;2 C for &gt;=30 min (precip/condensation) or &gt;=20 min (fog) AND no condensation buildup after 30 min |  | doc | [keck/observing/Weather_Table.html](https://www2.keck.hawaii.edu/observing/Weather_Table.html) |  |
| fog_close | fog sighted within 200 m of telescope; or within 400 m and within 15 min of contact; sudden dewpoint jump |  | doc | [keck/observing/Weather_Table.html](https://www2.keck.hawaii.edu/observing/Weather_Table.html) |  |
| fog_reopen | no fog or clouds within 400 m of domes, no low clouds within 1/2 mile of summit, plus dewpoint rule |  | doc | [keck/observing/Weather_Table.html](https://www2.keck.hawaii.edu/observing/Weather_Table.html) |  |
| precipitation_close | any visible precipitation; stratus/cumulus clouds overhead of the domes |  | doc | [keck/observing/Weather_Table.html](https://www2.keck.hawaii.edu/observing/Weather_Table.html) |  |
| high_clouds_close | precipitation-producing clouds overhead; or cloud &gt;6 mag extinction AND conditions preclude the primary program |  | doc | [keck/observing/Weather_Table.html](https://www2.keck.hawaii.edu/observing/Weather_Table.html) |  |
| ice_snow_on_dome | close/do not open if ice or snow has accumulated on the dome, or snow falls in when opening; reopen when winds are sustained &lt;10 mph with no infalling material |  | doc | [keck/observing/Weather_Table.html](https://www2.keck.hawaii.edu/observing/Weather_Table.html) |  |
| wind_chill_laser_spotters | -15 | degC | doc | [keck/observing/Weather_Table.html](https://www2.keck.hawaii.edu/observing/Weather_Table.html) | Applies to laser spotters only. |
| relative_humidity_limit | 95 | % RH | approx | [arxiv.org/abs/2208.11794](https://arxiv.org/abs/2208.11794) | Not on the Keck table. van Kooten & Izett 2022 model the Keck criteria as U &lt; 20 m/s, RH &lt; 95% and T - Tdew &gt; 2 K. Use it as an operational proxy. |
| closure_statistics | {"dome_closed_any_reason_frac_2018_2021": 0.4, "weather_share_of_closures_hist": 0.15, "nights_exceeding_met_criteria_frac": "~0.15 -&gt; &gt;0.30 over 30 yr"} |  | doc | [arxiv.org/abs/2208.11794](https://arxiv.org/abs/2208.11794) | From Keck records cited by van Kooten & Izett 2022. Interpret with care. |

## 2. Telescope

### telescope

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| mirror_reflectivity | {"primary": 0.86, "secondary": 0.88} | fraction at ~6000 A | doc | [arxiv.org/abs/1903.07629](https://arxiv.org/abs/1903.07629) | Loss budget in Perley 2019 (from Oke et al. 1995). The LRIS-B page says multiply by ~0.8 for the telescope. |

#### telescope / primary

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| segments | 36 |  | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | hexagonal, 1.8 m across corners |
| max_diameter_m | 10.95 | m | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) |  |
| equivalent_circular_aperture_m | 9.96 | m | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) |  |
| primary_focal_length_m | 17.5 | m | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) |  |

#### telescope / collecting_area_m2

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| recommended_for_etc | 72.3 | m^2 | doc | [keck/inst/lris/specEffOldRed.html](https://www2.keck.hawaii.edu/inst/lris/specEffOldRed.html) | 72.295 m^2: 'net usable primary area in the f/15 configuration with the baffles retracted' (Nelson 1994, cited by J. Cohen on the LRIS efficiency page). |
| perley2019 | 73.3 | m^2 | doc | [arxiv.org/abs/1903.07629](https://arxiv.org/abs/1903.07629) | 'physical area of the primary mirror (73.3 m2)', LPipe paper. |
| lris_zp_1994 | 76.2 | m^2 | doc | [keck/inst/lris/photometric_zero_points.html](https://www2.keck.hawaii.edu/inst/lris/photometric_zero_points.html) | 0.97*pi*(10 m)^2/4, used for the 1994 LRIS zero points (0.97 for the central hole). |
| from_equivalent_aperture | 77.9 | m^2 | approx | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | pi*(9.96/2)^2, derived. |

#### telescope / f15_focus

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| focal_length_m | 149.6 | m | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) |  |
| plate_scale_mm_per_arcsec | 0.725 | mm/arcsec | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) |  |
| field_diameter_arcmin | 20 | arcmin | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) |  |

#### telescope / instrument_locations

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| LRIS | Keck I Cassegrain |  | doc | [keck/inst/lris/lrishome.html](https://www2.keck.hawaii.edu/inst/lris/lrishome.html) |  |
| MOSFIRE | Keck I Cassegrain |  | doc | [keck/inst/mosfire/home.html](https://www2.keck.hawaii.edu/inst/mosfire/home.html) |  |
| HIRES | Keck I right Nasmyth |  | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | Keck guide 2002. |
| DEIMOS | Keck II right Nasmyth |  | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) |  |
| KCWI | Keck II left Nasmyth |  | doc | [keck/inst/kcwi/specs.html](https://www2.keck.hawaii.edu/inst/kcwi/specs.html) |  |

#### telescope / pointing_limits

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| azimuth_convention | azimuth from North through East (N=0, E=90) |  | approx | [keck/inst/common/TelLimits.html](https://www2.keck.hawaii.edu/inst/common/TelLimits.html) | Inferred: the K1 deck range 5-146 deg is described as 'North and East' and the K2 range 185-333 deg as 'South and West'. |
| keck1_deck_az_range_deg | [5.3, 146.2] | deg az | doc | [keck/inst/common/TelLimits.html](https://www2.keck.hawaii.edu/inst/common/TelLimits.html) | Nasmyth-deck obstruction zone. |
| keck1_el_range_in_deck_zone_deg | [33.3, 88.9] | deg el | doc | [keck/inst/common/TelLimits.html](https://www2.keck.hawaii.edu/inst/common/TelLimits.html) |  |
| keck1_el_range_elsewhere_deg | {"unvignetted": [18.0, 88.9], "vignetted": [0.0, 18.0]} | deg el | doc | [keck/inst/common/TelLimits.html](https://www2.keck.hawaii.edu/inst/common/TelLimits.html) |  |
| keck2_deck_az_range_deg | [185.3, 332.8] | deg az | doc | [keck/inst/common/TelLimits.html](https://www2.keck.hawaii.edu/inst/common/TelLimits.html) |  |
| keck2_el_range_in_deck_zone_deg | [36.8, 89.5] | deg el | doc | [keck/inst/common/TelLimits.html](https://www2.keck.hawaii.edu/inst/common/TelLimits.html) |  |
| keck2_el_range_elsewhere_deg | {"unvignetted": [18.0, 89.5], "vignetted": [0.0, 18.0]} | deg el | doc | [keck/inst/common/TelLimits.html](https://www2.keck.hawaii.edu/inst/common/TelLimits.html) |  |
| practical_lower_el_deg | 18 | deg | doc | [keck/inst/common/TelLimits.html](https://www2.keck.hawaii.edu/inst/common/TelLimits.html) | Unvignetted limit outside the deck zone. From 0 to 18 deg the beam is vignetted. |
| guiding_upper_el_deg | 85 | deg | doc | [keck/inst/common/TelLimits.html](https://www2.keck.hawaii.edu/inst/common/TelLimits.html) | 'Stable guiding cannot be guaranteed for targets transiting at elevations higher than ~85 deg'. |
| mosfire_guiding_limits | {"el_deg": 86, "rotator_rate_deg_per_min": 2} |  | doc | [keck/inst/mosfire/rotator.html](https://www2.keck.hawaii.edu/inst/mosfire/rotator.html) | Guiding degrades above these. Between 87 and 88 deg the MOSFIRE team stopped guiding. |
| zenith_blind_spot_radius_deg | 0.5 | deg | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) |  |

#### telescope / drives

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| az_slew_rate_deg_s | 1.3 | deg/s | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | Also quoted in the NIRC manual. |
| el_slew_rate_deg_s | 0.5 | deg/s | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) |  |
| az_accel_deg_s2 | 0.05 | deg/s^2 | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) |  |
| el_accel_deg_s2 | 0.03 | deg/s^2 | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) |  |
| pointing_accuracy_arcsec_rms | 4 | arcsec rms | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | 2002 value; current pointing is likely better. |
| closed_loop_tracking_arcsec_rms | 0.08 | arcsec rms | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) |  |
| open_loop_tracking | ~0.1 arcsec/min |  | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) |  |
| open_loop_offset_accuracy | 0.1 arcsec over several arcsec |  | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) |  |
| slew_settle_s | 15 | s | **EST** | [keck/inst/nirc/manual/Manual.html](https://www2.keck.hawaii.edu/inst/nirc/manual/Manual.html) | No settle time is published. The NIRC manual says a small offset plus guider settle takes ~6 s, and a full acquisition ~5 min. Use 10-20 s for servo settle after a large slew, before guide-star acquisition. |
| dome_rotation | assume dome keeps pace with the azimuth slew (~1 deg/s) |  | **EST** | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) | No public dome rotation rate found. Assume the dome is not the bottleneck except on &gt;90 deg azimuth moves. |
| rotator_ranges | {"DEIMOS_physical_deg": [-330, 402], "MOSFIRE": "slightly &lt;530 deg"} | deg | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) | DEIMOS from its specs page; MOSFIRE from McLean+2012 SPIE. |

#### telescope / overheads

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| typical_acquisition_s | 300 | s | doc | [keck/inst/nirc/manual/Manual.html](https://www2.keck.hawaii.edu/inst/nirc/manual/Manual.html) | 'Including a typical slew, identifying the field, centering, and sending to the science detector, a round number to use is 5 minutes for an acquisition' (Keck NIRC manual). The rotator can move during the slew, and the instrument can be reconfigured. |
| small_offset_plus_guide_settle_s | 6 | s | doc | [keck/inst/nirc/manual/Manual.html](https://www2.keck.hawaii.edu/inst/nirc/manual/Manual.html) | 'Offsetting the telescope includes the actual move plus the time it takes for the guiding to settle. This is typically 6 sec.' |
| mosfire_dither_s | 3-4 | s | doc | [keck/inst/mosfire/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/mosfire/exposure_recipes.html) | per telescope nod |
| focus_mira_s | 600 | s | doc | [keck/inst/deimos/checklist-align-xbox.html](https://www2.keck.hawaii.edu/inst/deimos/checklist-align-xbox.html) | 'Wait about 10 minutes for the OA to acquire an appropriate star and run Mira.' The MOSFIRE SAT page recommends refocusing when elevation has changed by 30 deg or more since the last focus. |
| faint_longslit_acquisition_s | 600 | s | **EST** | [keck/inst/lris/longslit_setup.html](https://www2.keck.hawaii.edu/inst/lris/longslit_setup.html) | Estimated from the documented LRIS/DEIMOS blind-offset procedure after the ~5-min slew/acquire: a 30 s slit image and a 30 s field image, each plus ~50-90 s readout, a mov/movb offset, an optional confirmation image, a slit re-insert and a focus change. That sums to ~6-12 min. Use 10 min (plus 5 min slew). |
| slitmask_alignment_s | {"DEIMOS": 300, "LRIS": 600, "MOSFIRE": 600} | s | **EST** | [arxiv.org/abs/1203.3192](https://arxiv.org/abs/1203.3192) | DEIMOS ~5 min is documented (DEEP2: 'generally converges after two direct images and takes roughly 5 minutes'). LRIS and MOSFIRE are estimates. MOSFIRE mask execution (&lt;6 min) can run during the slew, then coarse + fine align. LRIS readouts are longer; aligning with blue-side images saves 35-45 min per night per Steidel. |

#### telescope / rotator_and_parallactic

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| rotator_modes | ["position angle mode (fixed sky PA)", "vertical angle mode (slit follows parallactic when set to 0)"] |  | doc | [keck/inst/common/parallactic.html](https://www2.keck.hawaii.edu/inst/common/parallactic.html) | Long bright on-slit guiding: vertical angle 0. Short offset-guided exposures: PA mode at the current parallactic angle. Long offset-guided: PA = parallactic angle at mid-exposure. |
| dcs_keywords | {"ROTMODE": "rotator tracking mode", "ROTDEST": "rotator user destination", "ROTPOSN": "rotator user position", "ROTPPOSN": "rotator physical position", "PARANG": "parallactic angle astrometric", "PARANTEL": "parallactic angle telescope"} |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) |  |
| example | dec +40, HA 2h east -&gt; parallactic angle 60 deg |  | doc | [keck/inst/common/parallactic.html](https://www2.keck.hawaii.edu/inst/common/parallactic.html) |  |

#### telescope / operations

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| night_definition | between 12-degree twilights |  | doc | [keck/inst/metrics/Intro.html](https://www2.keck.hawaii.edu/inst/metrics/Intro.html) | Keck Duty-Cycle Metrics: 'The time between 12 deg twilights is the formal definition of the night.' Split nights are divided at the midpoint between 12 deg twilights. |
| split_night_rule | first-half observers finish their last exposure at the midpoint |  | doc | [keck/observing/SplitNights.html](https://www2.keck.hawaii.edu/observing/SplitNights.html) | LRIS/DEIMOS cannot be reconfigured at night. Split-night PIs submit one shared configuration. |
| instrument_reconfig_at_night | LRIS gratings / DEIMOS gratings / HIRES cross-disperser fixed for the night |  | doc | [keck/observing/SplitNights.html](https://www2.keck.hawaii.edu/observing/SplitNights.html) | HIRES b&lt;-&gt;r XD swap is never done at night (blue_vs_red page). LRIS/DEIMOS gratings come from the configuration form. |

## 3. Instruments

### LRIS

#### LRIS / general

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| focus | Keck I Cassegrain |  | doc | [keck/inst/lris/lrishome.html](https://www2.keck.hawaii.edu/inst/lris/lrishome.html) |  |
| fov_arcmin | [6.0, 7.8] | arcmin | doc | [keck/inst/lris/lrishome.html](https://www2.keck.hawaii.edu/inst/lris/lrishome.html) |  |
| wavelength_range_A | [3200, 10000] | A | doc | [keck/inst/lris/lrishome.html](https://www2.keck.hawaii.edu/inst/lris/lrishome.html) | Blue arm ~3100-6000 (optimized), red arm via dichroic. |
| pixel_scale_arcsec | 0.135 | arcsec/pix | doc | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) | Both arms. |
| resolving_power_range | [300, 5000] |  | doc | [keck/inst/lris/lrishome.html](https://www2.keck.hawaii.edu/inst/lris/lrishome.html) |  |
| peak_system_efficiency | 0.5 | fraction | doc | [keck/inst/lris/lrishome.html](https://www2.keck.hawaii.edu/inst/lris/lrishome.html) | 'peak system efficiencies of ~50%' (home page). The home page does not say what is included; the efficiency pages quote instrument+CCD only (no telescope/atmosphere). |
| dichroics | {"D460": 4874, "D500": 5091, "D560": 5696, "D680": 6800, "mirror": "all light to blue"} | A (50% crossover) | doc | [keck/inst/lris/dichroic.html](https://www2.keck.hawaii.edu/inst/lris/dichroic.html) | D500 is 4800 in one column of the table; D680 is 6640/6800. Transmission &gt;95%, reflection &gt;98% outside the transition. |
| status_note | Red CCD Mark IV (4k x 4k LBNL) installed April/May 2021 |  | doc | [keck/inst/lris/news.html](https://www2.keck.hawaii.edu/inst/lris/news.html) |  |

#### LRIS / blue_detector

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| type | 2 x 2K x 4K Marconi (E2V), 15 um pixels |  | doc | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) |  |
| read_noise_e | [3.9, 4.2, 3.6, 3.6] | e- rms per amp 1-4 | doc | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) |  |
| gain_e_per_adu | [1.55, 1.56, 1.63, 1.7] | e-/ADU | doc | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) |  |
| linearity_adu | 62000 | ADU | doc | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) | saturates at 65535 |
| full_well_e | 99000 | e- | approx | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) | Derived: linear limit 62000 ADU x 1.55-1.70 e-/ADU = 96k-105k e-. ADC-limited. |
| dark_current_e_per_hr | 2 | e-/pix/hr | **EST** | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) | Not published. Typical of LN2-cooled E2V CCDs; negligible compared with sky. |
| erase_s | 8 | s | doc | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) |  |
| readout_s | 42 | s | doc | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) | Full frame, detector page. |
| bias_adu | 1000 | ADU | doc | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) | ~1000 |

#### LRIS / red_detector

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| type | LBNL 4k x 4k, 500 um deep-depletion, 15 um pixels (Mark IV) |  | doc | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) |  |
| read_noise_e | [3.64, 3.45, 3.65, 3.52] | e- rms per amp L1,L2,U1,U2 | doc | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) |  |
| gain_e_per_adu | [1.71, 1.64, 1.61, 1.67] | e-/ADU | doc | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) |  |
| linearity_adu | 59000 | ADU | doc | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) | saturates at 65535 |
| full_well_e | 98000 | e- | approx | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) | Derived: 59000 ADU x ~1.66 e-/ADU. |
| dark_current_e_per_hr | 5 | e-/pix/hr | **EST** | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) | Not published. A thick LBNL CCD at about -100 C: assume a few e-/pix/hr. |
| erase_s | 10 | s | doc | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) |  |
| readout_s | 90 | s | doc | [keck/inst/lris/detectors.html](https://www2.keck.hawaii.edu/inst/lris/detectors.html) | Detector-page value. The mode-specific table is below. |
| windowing | not available on red |  | doc | [keck/inst/lris/win_bin.html](https://www2.keck.hawaii.edu/inst/lris/win_bin.html) |  |
| binning_modes | {"imaging_4amp": ["1x1", "2x2"], "spectroscopy_2amp": ["1x1", "2x2", "2x1", "1x2"]} |  | doc | [keck/inst/lris/redccdmanual.html](https://www2.keck.hawaii.edu/inst/lris/redccdmanual.html) | Set with 'lredccd_setup imag/spec &lt;bx&gt; &lt;by&gt;'. Imaging uses 4 amps, spectroscopy 2 amps. |
| cosmic_ray_note | thick CCD -&gt; more CR hits |  | doc | [keck/inst/lris/news.html](https://www2.keck.hawaii.edu/inst/lris/news.html) |  |

#### LRIS / readout_wallclock

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| blue_1x1_full | 54 | s | doc | [keck/inst/lris/win_bin.html](https://www2.keck.hawaii.edu/inst/lris/win_bin.html) | Wall clock for a 1-s exposure, including erase + expose + readout. |
| blue_spatial_window_100pix_1x1 | 54 | s | doc | [keck/inst/lris/win_bin.html](https://www2.keck.hawaii.edu/inst/lris/win_bin.html) | Spatial windowing does not save time on blue. |
| blue_1x2_full | 36 | s | doc | [keck/inst/lris/win_bin.html](https://www2.keck.hawaii.edu/inst/lris/win_bin.html) |  |
| blue_1x2_half_frame | 23 | s | doc | [keck/inst/lris/win_bin.html](https://www2.keck.hawaii.edu/inst/lris/win_bin.html) |  |
| red_4amp_1x1 | 54 | s | doc | [keck/inst/lris/win_bin.html](https://www2.keck.hawaii.edu/inst/lris/win_bin.html) | Elapsed for a 0-s exposure, including erase, expose and readout. |
| red_4amp_2x2 | 24 | s | doc | [keck/inst/lris/win_bin.html](https://www2.keck.hawaii.edu/inst/lris/win_bin.html) |  |
| red_2amp_1x1 | 97 | s | doc | [keck/inst/lris/win_bin.html](https://www2.keck.hawaii.edu/inst/lris/win_bin.html) | Spectroscopy default (2 amps). |
| red_2amp_1x2 | 55 | s | doc | [keck/inst/lris/win_bin.html](https://www2.keck.hawaii.edu/inst/lris/win_bin.html) |  |
| red_2amp_2x1 | 65 | s | doc | [keck/inst/lris/win_bin.html](https://www2.keck.hawaii.edu/inst/lris/win_bin.html) |  |
| red_2amp_2x2 | 39 | s | doc | [keck/inst/lris/win_bin.html](https://www2.keck.hawaii.edu/inst/lris/win_bin.html) |  |

#### LRIS / blue_grisms

| Name | dispersion_A_per_pix | fwhm_1arcsec_A | R | coverage_A | instrument_efficiency | system_efficiency_peak_with_telescope | instrument_efficiency_peak |
|---|---|---|---|---|---|---|---|
| 600/4000 | 0.63 | [3.8, 4.1] | 1600 | {"MOS": [3330, 5910], "longslit": [3040, 5630]} | {"3500": 0.22, "4000": 0.43, "4221_peak": 0.49, "4500": 0.48, "5000": 0.445, "5500": 0.405} | 0.39 (approx) |  |
| 300/5000 | 1.43 | [8.4, 9.2] | 900 | {"MOS": [2240, 8090], "longslit": [1600, 7450]} |  |  | 0.56 |
| 400/3400 | 1.09 | [6.5, 7.1] | 1000 |  |  |  | 0.5 |
| 1200/3400 | 0.24 | 1.56 | 3600 | {"MOS": [3040, 4030], "longslit": [2940, 3920]} |  |  | 0.37 |

- `dispersion_A_per_pix` [A/pix]: source [keck/inst/lris/dispersive_elements.html](https://www2.keck.hawaii.edu/inst/lris/dispersive_elements.html)
- `fwhm_1arcsec_A` [A]: source [keck/inst/lris/dispersive_elements.html](https://www2.keck.hawaii.edu/inst/lris/dispersive_elements.html). blue-to-red
- `R` []: source [arxiv.org/abs/astro-ph/0401439](https://arxiv.org/abs/astro-ph/0401439). R ~= 1600 (Steidel+2004, 0.7-arcsec-class slits). ~1000 with a 0.7 arcsec slit
- `coverage_A` [A]: source [keck/inst/lris/dispersive_elements.html](https://www2.keck.hawaii.edu/inst/lris/dispersive_elements.html). Detector coverage. Usable &gt;~3100 A (atmosphere) and &lt;~7000 A (blue camera).
- `instrument_efficiency` [fraction vs A]: source [keck/inst/lris/data_files/600_mirr_eff.dat](https://www2.keck.hawaii.edu/inst/lris/data_files/600_mirr_eff.dat). Mirror in place of dichroic; instrument + CCD, excluding telescope and atmosphere. Data from 2002.
- `system_efficiency_peak_with_telescope` [fraction]: source [sites.astro.caltech.ed...s/lrisb/lrisb_new.html](https://sites.astro.caltech.edu/~ccs/lrisb/lrisb_new.html). 0.49 x 0.8 ('to get total system throughput including the telescope, multiply by about 0.8').
- `instrument_efficiency_peak` [fraction; fraction at ~3990 A; fraction at ~3774 A]: source [sites.astro.caltech.ed...s/lrisb/lrisb_new.html](https://sites.astro.caltech.edu/~ccs/lrisb/lrisb_new.html), [keck/inst/lris/data_files/400_mirr_eff.dat](https://www2.keck.hawaii.edu/inst/lris/data_files/400_mirr_eff.dat), [keck/inst/lris/data_files/1200_d460_eff.dat](https://www2.keck.hawaii.edu/inst/lris/data_files/1200_d460_eff.dat). '300/5000 grism provides a peak efficiency of about 56%'. The data file gives 0.55 at 5000 A. 0.14 at 3200 A, 0.31 at 3500 A with the D460 dichroic

#### LRIS / red_gratings

| Name | dispersion_A_per_pix | coverage_A | fwhm_1arcsec_A |
|---|---|---|---|
| 150/7500 | 3 | 12288 | 17.7 (**EST**) |
| 300/5000 | 1.59 | 6525 | 9.18 |
| 400/8500 | 1.16 | 4762 | 6.9 |
| 600/5000 | 0.8 | 3275 | 4.7 |
| 600/7500 | 0.8 | 3275 | 4.7 |
| 600/10000 | 0.8 | 3275 | 4.7 |
| 831/8200 | 0.58 | 2375 | 3.4 (**EST**) |
| 900/5500 | 0.53 | 2175 | 3.1 (**EST**) |
| 1200/7500 | 0.4 | 1638 | 2.4 (**EST**) |
| 1200/9000 | 0.4 | 1638 | 2.4 (**EST**) |

- `dispersion_A_per_pix` [A/pix]: source [keck/inst/lris/dispersive_elements.html](https://www2.keck.hawaii.edu/inst/lris/dispersive_elements.html). Calculated for the 4k detector.
- `coverage_A` [A]: source [keck/inst/lris/dispersive_elements.html](https://www2.keck.hawaii.edu/inst/lris/dispersive_elements.html). 4096 px x dispersion (calculated).
- `fwhm_1arcsec_A` [A]: source [keck/inst/lris/dispersive_elements.html](https://www2.keck.hawaii.edu/inst/lris/dispersive_elements.html). Not tabulated; assumed 5.9 pix x dispersion (the tabulated gratings give 5.8-5.9 pix FWHM for a 1 arcsec slit: 4.7/0.80, 6.9/1.16, 9.18/1.59).

#### LRIS / slits

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| longslits | {"long_0.7": [0.7, 175], "long_1.0": [1.0, 175], "long_1.5": [1.5, 175], "long_8.7": [8.7, 175], "pol_1.0": [1.0, 25], "pol_1.5": [1.5, 25]} | [width, length] arcsec | doc | [keck/inst/lris/slitmask.html](https://www2.keck.hawaii.edu/inst/lris/slitmask.html) |  |
| user_mask_slots | 8 |  | doc | [keck/inst/lris/slitmask.html](https://www2.keck.hawaii.edu/inst/lris/slitmask.html) | 10 positions, 2 reserved (direct + focus holes). |

#### LRIS / throughput

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| red_old_ccd_peaks_excl_tel_atm | {"300/5000": 0.34, "600/7500": 0.4, "1200/7500": 0.36} | fraction | doc | [keck/inst/lris/specEffOldRed.html](https://www2.keck.hawaii.edu/inst/lris/specEffOldRed.html) | 1995 measurement with the ORIGINAL Tek red CCD. Instrument + CCD only. |
| red_400/8500_total_peak | 0.3 | fraction incl. telescope, excl. atmosphere | **EST** | [keck/inst/lris/specEffOldRed.html](https://www2.keck.hawaii.edu/inst/lris/specEffOldRed.html) | No current curve is published for the Mark IV CCD. Estimate: ~0.38-0.40 instrument peak (similar to the 600/7500 old measurement; the Mark IV has 'similar QE' to the previous LBNL mosaic, which was much redder-sensitive than the Tek) x 0.76 telescope. Beyond 9000 A LRIS beats DEIMOS. |
| effective_area_observed | {"blue_m2": 10, "red_m2": 15, "total_throughput": [0.12, 0.2]} | m^2 | doc | [arxiv.org/abs/1903.07629](https://arxiv.org/abs/1903.07629) | Includes atmosphere, telescope, slit losses; clear weather, average seeing; over optimal-sensitivity region of each grism/grating (Perley 2019). |
| loss_budget_6000A | {"atmosphere": 0.9, "primary": 0.86, "secondary": 0.88, "slit_loss": 0.8, "dichroic": 0.95, "rest_of_spectrograph": 0.4, "detector_QE": 0.8} | fraction | doc | [arxiv.org/abs/1903.07629](https://arxiv.org/abs/1903.07629) | Typical conditions, after Oke et al. 1995. |

#### LRIS / exposure_limits

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| red_max_single_s | 1200 | s | doc | [keck/inst/lris/news.html](https://www2.keck.hawaii.edu/inst/lris/news.html) | 'we recommend maximum exposure times of 20 minutes to avoid excessive contamination by cosmic rays' (Mark IV). |
| blue_typical_mask_exposure_s | 1800 | s | doc | [arxiv.org/abs/astro-ph/0401439](https://arxiv.org/abs/astro-ph/0401439) | Steidel+2004 LBG masks: 3 x 1800 s, dithered 1-2 arcsec along the slit. |

#### LRIS / overheads

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| blue_element_change_s | 60 | s (max) | doc | [arxiv.org/abs/astro-ph/0401439](https://arxiv.org/abs/astro-ph/0401439) | Dichroic, grism or filter change '&lt;60 seconds'. |
| blue_grism_or_filter_deploy_s | 30 | s | doc | [arxiv.org/abs/astro-ph/0401439](https://arxiv.org/abs/astro-ph/0401439) | '~30 seconds' (LRIS-B page: 'less than 30 seconds'). |
| red_grating_change | several minutes |  | doc | [keck/inst/deimos/primer.html](https://www2.keck.hawaii.edu/inst/deimos/primer.html) | The DEIMOS primer says its grating change is 'comparable to the time required for an LRIS grating change'. Use 300 s. |
| red_grating_tilt_change_s | 60 | s | **EST** | [keck/inst/lris/dispersive_elements.html](https://www2.keck.hawaii.edu/inst/lris/dispersive_elements.html) | Not published. Grating-angle moves on this class of stage take tens of seconds. |
| red_filter_or_slitmask_change_s | 60 | s | **EST** | [keck/inst/lris/slitmask.html](https://www2.keck.hawaii.edu/inst/lris/slitmask.html) | Not published. The red side moves 2 mechanisms at once (grating + slitmask/filter/focus). |
| simultaneous_mechanisms | {"blue": 4, "red": 2} |  | doc | [sites.astro.caltech.ed...s/lrisb/lrisb_new.html](https://sites.astro.caltech.edu/~ccs/lrisb/lrisb_new.html) |  |
| faint_target_procedure | 30 s slit image; 30 s direct image; movb/movr offset; optional confirm; refocus MIRA/AUTOFOC |  | doc | [keck/inst/lris/longslit_setup.html](https://www2.keck.hawaii.edu/inst/lris/longslit_setup.html) | See telescope.overheads.faint_longslit_acquisition_s for the time estimate. |

#### LRIS / sensitivity_anchors

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| effective_area | ~10 m^2 blue, ~15 m^2 red (12-20% total) |  | doc | [arxiv.org/abs/1903.07629](https://arxiv.org/abs/1903.07629) | Observed from standard-star response functions. |
| old_red_AB20_count_rate | {"300/5000@6200A": 4.3, "600/7500@6600A_peak": 2.5} | photons/s/pixel above atmosphere, AB=20 | doc | [keck/inst/lris/specEffOldRed.html](https://www2.keck.hawaii.edu/inst/lris/specEffOldRed.html) | 1995, OLD Tek CCD with 24 um pixels. Use the efficiencies, not these per-pixel rates. |
| imaging_zeropoints_1994 | {"B": 27.34, "V": 27.51, "R": 27.52, "I": 27.4} | mag giving 1 DN/s at airmass 1 (Vega) | approx | [keck/inst/lris/photometric_zero_points.html](https://www2.keck.hawaii.edu/inst/lris/photometric_zero_points.html) | Mean of two nights, 1.72 e-/DN, ORIGINAL red CCD. Historical; recalibrate for Mark IV. |

### DEIMOS

#### DEIMOS / general

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| focus | Keck II right Nasmyth |  | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) |  |
| detector | 8192 x 8192 mosaic, 2 x 4 of 2K x 4K MIT/LL CCDs, 15 um |  | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) |  |
| pixel_scale_arcsec | 0.1185 | arcsec/pix | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) |  |
| imaging_fov_arcmin | [16.7, 5.0] | arcmin | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) | Mosaic gaps lose 0.4 x 5.0 arcmin. |
| slit_length_arcmin | {"total": 16.7, "usable": 16.3} | arcmin | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) |  |
| max_slitlets | 130 |  | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) | Up to 130 slitlets with 1.5 arcsec gaps; tilted slits up to 30 deg. |
| wavelength_range_A | {"spectra": [4100, 11000], "imaging": [4000, 10500]} | A | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) |  |
| filters | ["B", "V", "R", "I", "Z", "BAL12(clear)", "GG400", "GG455", "GG495", "OG550"] |  | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) | An order-blocking or broadband filter must always be in. |
| flexure_compensation | closed-loop FCS, &lt;0.25-0.5 px residual |  | doc | [keck/inst/deimos/primer.html](https://www2.keck.hawaii.edu/inst/deimos/primer.html) |  |
| mask_slots | 10 |  | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) | 11 in cassette, 1 for focus mask. |
| status_note | Science CCD5 inoperative; FCS CCD2 inoperative; grating in slider 3 preferred |  | doc | [keck/inst/deimos/news.html](https://www2.keck.hawaii.edu/inst/deimos/news.html) | Current status page: targets on CCD1 lose the red end of their spectrum. |

#### DEIMOS / detector

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| gain_e_per_dn_spectral | [1.179, 1.174, 1.312, 1.212, 1.213, 1.198, 1.204, 1.249] | e-/DN (amps 1B..8B) | doc | [keck/inst/obsdata/inst...mos_detector_data.html](https://www2.keck.hawaii.edu/inst/obsdata/inst/deimos/www/detector_data/deimos_detector_data.html) | Measured 2024-10-23. |
| read_noise_e_spectral | [2.472, 2.461, 2.752, 2.541, 2.542, 2.512, 2.524, 2.619] | e- rms | doc | [keck/inst/obsdata/inst...mos_detector_data.html](https://www2.keck.hawaii.edu/inst/obsdata/inst/deimos/www/detector_data/deimos_detector_data.html) | 2024-10-23 |
| dark_current_e_per_hr | [3.3, 3.6, 3.5, 3.7, 2.7, 3.8, 3.3, 3.7] | e-/pix/hr | doc | [keck/inst/obsdata/inst...mos_detector_data.html](https://www2.keck.hawaii.edu/inst/obsdata/inst/deimos/www/detector_data/deimos_detector_data.html) | 2024-10-23 |
| full_well_e | 79000 | e- | **EST** | [keck/inst/obsdata/inst...mos_detector_data.html](https://www2.keck.hawaii.edu/inst/obsdata/inst/deimos/www/detector_data/deimos_detector_data.html) | No linearity limit published. 65535 DN x ~1.2 e-/DN = ADC-limited ~79k e-. |
| readout_direct_s | 40 | s | doc | [keck/inst/deimos/primer.html](https://www2.keck.hawaii.edu/inst/deimos/primer.html) | 'about 40 seconds in imaging mode' (4 CCDs). |
| readout_spectral_s | 80 | s | doc | [keck/inst/deimos/primer.html](https://www2.keck.hawaii.edu/inst/deimos/primer.html) | '80 seconds in spectral readout mode' (8 CCDs). Primer dates from ~2002-2005; may differ slightly today. |

#### DEIMOS / gratings

| Name | blaze_A | dispersion_A_per_pix | coverage_A | fwhm_0p75arcsec_A | R_at_blaze_0p75arcsec |
|---|---|---|---|---|---|
| 600ZD | 7500 | 0.65 | 5300 | 3.5 | 2143 (approx) |
| 830G | 8640 | 0.47 | 3840 | 2.5 | 3456 (approx) |
| 900ZD | 5500 | 0.44 | 3530 | 2.1 | 2619 (approx) |
| 1200G | 7760 | 0.33 | 2630 | 1.1-1.6 | 5748 (approx) |
| 1200B | 4500 | 0.33 | 2630 | 1.1-1.6 | 3333 (approx) |

- `blaze_A` [A]: source [keck/inst/deimos/gratings.html](https://www2.keck.hawaii.edu/inst/deimos/gratings.html)
- `dispersion_A_per_pix` [A/pix]: source [keck/inst/deimos/gratings.html](https://www2.keck.hawaii.edu/inst/deimos/gratings.html)
- `coverage_A` [A (8192 px)]: source [keck/inst/deimos/gratings.html](https://www2.keck.hawaii.edu/inst/deimos/gratings.html). Spectral length. The instantaneous window depends on slit position.
- `fwhm_0p75arcsec_A` [A]: source [keck/inst/deimos/gratings.html](https://www2.keck.hawaii.edu/inst/deimos/gratings.html). 0.75 arcsec slit
- `R_at_blaze_0p75arcsec` []: source [keck/inst/deimos/gratings.html](https://www2.keck.hawaii.edu/inst/deimos/gratings.html). blaze / FWHM (1200-line uses 1.35 A mid-range). Derived.

#### DEIMOS / slits

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| longslits_arcsec | [0.7, 0.8, 1.0, 1.2, 1.5] | arcsec | doc | [keck/inst/deimos/standard_slitmasks.html](https://www2.keck.hawaii.edu/inst/deimos/standard_slitmasks.html) | Long0.7 ... Long1.5: 12 slitlets x 82 arcsec. LongMirr 0.7 x 205 arcsec. LVMslitC: 0.7-1.5 x 20 arcsec. |
| design_width_arcsec | {"nominal": 1.0, "good_seeing": 0.5} | arcsec | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) |  |
| alignment_accuracy_arcsec | 0.1 | arcsec | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) |  |

#### DEIMOS / throughput

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| measured_2018_incl_telescope | `600ZD_GG455_c6500`: {"5000": 0.127, "6000": 0.213, "7000": 0.244, "7400_peak": 0.246, "8000": 0.233, "8500": 0.206}<br>`600ZD_OG550_c7500`: {"6000": 0.206, "6840_peak": 0.243, "8000": 0.215, "9000": 0.158, "9500": 0.109, "10000": 0.039}<br>`830G_OG550_c8000`: {"6500": 0.175, "7000": 0.219, "8130_peak": 0.267, "9000": 0.219, "9500": 0.142}<br>`1200G_GG455_c7000`: {"6000": 0.182, "6500": 0.244, "7140_peak": 0.269, "8000": 0.239}<br>`1200G_OG550_c8000`: {"7000": 0.204, "7450_peak": 0.214, "8500": 0.182, "9000": 0.155}<br>`900ZD_GG400_c5000`: {"4500": 0.117, "5000": 0.17, "6000": 0.228, "6530_peak": 0.235}<br>`1200B_GG400_c5000`: {"4000": 0.051, "4500": 0.127, "5000": 0.167, "6000": 0.184} | fraction vs wavelength (A) | doc | [keck/inst/deimos/ripisc.html](https://www2.keck.hawaii.edu/inst/deimos/ripisc.html) | Nd/Nt: photons detected over photons hitting the primary (atmosphere removed). Sampled from the ECSV tables (ripisc/&lt;grating&gt;/*.txt), 2010-2018 standards (T. Lai). |
| spec_page_prediction_6000A | 0.29 | fraction incl. telescope | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) | Predicted (CCD QE 0.6, silver coatings). The primer says the total peaks near 30%. |

#### DEIMOS / exposure_limits

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| typical_science_exposure_s | [900, 1800] | s | doc | [keck/inst/deimos/checklist-align-xbox.html](https://www2.keck.hawaii.edu/inst/deimos/checklist-align-xbox.html) | 'typically 900-1800 sec' |
| deep2_exposures | 3 x 1200 s per mask, no dithering |  | doc | [arxiv.org/abs/1203.3192](https://arxiv.org/abs/1203.3192) |  |

#### DEIMOS / overheads

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| mirror_grating_mode_change_s | [120, 190] | s | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) | '120-190 sec changeover between mirror and grating, depending on slider'. |
| grating_change | several minutes |  | doc | [keck/inst/deimos/primer.html](https://www2.keck.hawaii.edu/inst/deimos/primer.html) | The primer recommends zeroth-order imaging (GO ZERO / GO BACK) for alignment instead of swapping to the mirror. |
| zeroth_order_tilt_s | 60 | s | **EST** | [keck/inst/deimos/faint-long.html](https://www2.keck.hawaii.edu/inst/deimos/faint-long.html) | Not published. A grating tilt move plus FCS re-lock. |
| fcs_lock_s | 120 | s | doc | [keck/inst/deimos/checklist-align-xbox.html](https://www2.keck.hawaii.edu/inst/deimos/checklist-align-xbox.html) | 'It may take a couple of minutes for the FCS to lock.' |
| mask_alignment_s | 300 | s | doc | [arxiv.org/abs/1203.3192](https://arxiv.org/abs/1203.3192) | DEEP2: converges after two direct images, ~5 min, and is re-checked once after the first exposure. |
| slitmask_change_s | 90 | s | **EST** | [keck/inst/deimos/checklist-align-xbox.html](https://www2.keck.hawaii.edu/inst/deimos/checklist-align-xbox.html) | Not published. Can be done during the slew. |
| filter_change_s | 60 | s | **EST** | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) | Not published. |

#### DEIMOS / sensitivity_anchors

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| spec_prediction_600l | {"V=22": 21, "V=23": 12, "V=24": 5} | S/N per pixel in 3600 s | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) | Point source, 0.75 arcsec slit, 0.7 arcsec seeing, 5500 A, 0.65 A/px, V sky 21.25 mag/arcsec^2. V is the magnitude through the slit (predicted). |
| achieved_1200G_faint_galaxies | {"R=21": "2.2-4.6", "R=22": "1.2-2.4", "R=23": "0.6-1.2", "R=24": "0.3-0.6"} | S/N per pixel in 3600 s | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) | 1.0 arcsec slit, ~7000 A, 0.32 A/px, average conditions, total mag (no slit-loss correction). |
| deep2_1200G_1hr | `18.5`: [5.53, 8.37, 10.89]<br>`19.0`: [4.81, 6.77, 9.74]<br>`19.5`: [3.34, 5.89, 8.28]<br>`20.0`: [2.92, 5.35, 7.43]<br>`20.5`: [2.67, 4.45, 6.18]<br>`21.0`: [2.19, 3.29, 4.6]<br>`21.5`: [1.67, 2.42, 3.25]<br>`22.0`: [1.23, 1.79, 2.4]<br>`22.5`: [0.88, 1.29, 1.73]<br>`23.0`: [0.61, 0.88, 1.24]<br>`23.5`: [0.42, 0.61, 0.83]<br>`24.0`: [0.32, 0.45, 0.61] | R_AB -&gt; [Q1, median, Q3] S/N per pixel in 1 h | doc | [keck/inst/deimos/deep2_s2n.html](https://www2.keck.hawaii.edu/inst/deimos/deep2_s2n.html) | ~50,000 DEEP2 galaxies, 1200G, 6600-6830 and 6960-7250 A, 1 arcsec slits. Galaxies are extended, so point sources do better. |
| count_rates_predicted | {"imaging_V21_e_per_s": 800, "spec_6000A_V21_e_per_s": 1.0} | e-/s | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) | Predicted; spectroscopic rate per pixel. |
| imaging_zeropoints_2015 | {"V": 27.5, "R": 27.9, "I": 28.0, "Z": 27.0} | Vega mag for 1 DN/s | doc | [keck/inst/deimos/deimos_imaging_zps.html](https://www2.keck.hawaii.edu/inst/deimos/deimos_imaging_zps.html) | 2015-12-17, airmass 1.04. |
| night_sky_between_OH | {"1200G_cont": 0.01, "1200G_OH_peak": 1.0, "830G_cont": 0.015, "830G_OH_peak": 1.5} | e-/s/px | doc | [keck/inst/deimos/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/deimos/exposure_recipes.html) |  |

### MOSFIRE

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| predicted_instrument_only_throughput | {"Y": 0.308, "J": 0.325, "H": 0.361, "K": 0.35} | band-average fraction | doc | [ucolick.org/documents/mosfire_spie.pdf](https://www.ucolick.org/documents/mosfire_spie.pdf) | Slit to detector; excludes telescope and atmosphere; agrees with on-sky measurements. |

#### MOSFIRE / general

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| focus | Keck I Cassegrain |  | doc | [keck/inst/mosfire/home.html](https://www2.keck.hawaii.edu/inst/mosfire/home.html) | since 2012 |
| fov_arcmin | {"imaging": [6.12, 6.12], "spectroscopy_typical": [6.12, 3.0]} | arcmin | doc | [keck/inst/mosfire/detector.html](https://www2.keck.hawaii.edu/inst/mosfire/detector.html) | Corners vignetted; the collimator field is 6.8 arcmin. |
| pixel_scale_arcsec | 0.1798 | arcsec/pix | doc | [keck/inst/mosfire/detector.html](https://www2.keck.hawaii.edu/inst/mosfire/detector.html) |  |
| slits | {"max": 46, "slit_length_per_bar_pair_arcsec": 7.0, "default_width_arcsec": 0.7} |  | doc | [ucolick.org/documents/mosfire_spie.pdf](https://www.ucolick.org/documents/mosfire_spie.pdf) | Configurable Slit Unit; any width. Recipes assume 0.7 arcsec. |
| csu_reconfig_s | 360 | s (max) | doc | [ucolick.org/documents/mosfire_spie.pdf](https://www.ucolick.org/documents/mosfire_spie.pdf) | 'Reconfiguration time is &lt; 6 minutes' (SPIE). The home page says 'under 5 minutes'. |
| status_note | Warmed for servicing after cold-head failure (2026-08-26) |  | doc | [keck/inst/mosfire/news.html](https://www2.keck.hawaii.edu/inst/mosfire/news.html) | Check availability for simulated dates after Aug 2026. |

#### MOSFIRE / detector

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| type | Teledyne H2RG HgCdTe 2048x2048 |  | doc | [keck/inst/mosfire/detector.html](https://www2.keck.hawaii.edu/inst/mosfire/detector.html) |  |
| gain_e_per_adu | 2.15 | e-/ADU | doc | [keck/inst/mosfire/detector.html](https://www2.keck.hawaii.edu/inst/mosfire/detector.html) |  |
| read_noise_e | {"CDS": 21, "MCDS4": 10.8, "MCDS8": 7.7, "MCDS16": 5.8, "MCDS32": 4.2, "MCDS64": 3.5, "MCDS128": 3.0} | e- rms | doc | [keck/inst/mosfire/detector.html](https://www2.keck.hawaii.edu/inst/mosfire/detector.html) | SPIE: CDS 23.2 e- at the telescope. |
| dark_current_e_per_s | 0.008 | e-/s/pix (upper limit) | doc | [keck/inst/mosfire/detector.html](https://www2.keck.hawaii.edu/inst/mosfire/detector.html) | &lt;0.008 |
| linearity | {"1pct_adu": 26000, "1pct_e": 56000, "5pct_adu": 37000, "5pct_e": 80000} |  | doc | [keck/inst/mosfire/detector.html](https://www2.keck.hawaii.edu/inst/mosfire/detector.html) |  |
| saturation | {"adu": 43000, "e": 97000} |  | doc | [keck/inst/mosfire/detector.html](https://www2.keck.hawaii.edu/inst/mosfire/detector.html) |  |
| persistence | {"fraction": 0.0004, "decay_time_s": 600} |  | doc | [keck/inst/mosfire/detector.html](https://www2.keck.hawaii.edu/inst/mosfire/detector.html) | The persistence page says e-folding ~10-12 min (&lt;1% after 51 min). 'When in doubt, go dark.' |
| min_integration_s | 1.455 | s | doc | [keck/inst/mosfire/detector.html](https://www2.keck.hawaii.edu/inst/mosfire/detector.html) | Exposure times are quantized by 1.455 s. |
| mcds_overhead_s | N_reads x 1.455 | s | doc | [keck/inst/mosfire/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/mosfire/exposure_recipes.html) | MCDS8 11.6 s, MCDS16 23.3 s, MCDS32 46.6 s. |

#### MOSFIRE / spectroscopy

| Name | order | dispersion_A_per_pix | resolution_pix | lambda_c_A | R | coverage_A | throughput_peak_mosfire_plus_tel | throughput_median_in_band | rec_exposure_s | rec_efficiency | noise_darkest_regions_e |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Y | 6 | 1.0855 | 2.82 | 10373 | 3388 | [9716, 11250] | 0.26 | 0.24 (approx) | 180 | 0.89 | [7.8, 9.7] |
| J | 5 | 1.3028 | 2.88 | 12450 | 3318 | [11530, 13520] | 0.32 | 0.28 (approx) | 120 | 0.84 | [6.0, 8.3] |
| H | 4 | 1.6269 | 2.74 | 16321 | 3660 | [14680, 18040] | 0.38 | 0.34 (approx) | 120 | 0.84 | [8.3, 10.1] |
| K | 3 | 2.1691 | 2.78 | 21760 | 3610 | [19540, 23970] | 0.36 | 0.31 (approx) | 180 | 0.89 | [8.4, 10.2] |

- `order` []: source [keck/inst/mosfire/grating.html](https://www2.keck.hawaii.edu/inst/mosfire/grating.html)
- `dispersion_A_per_pix` [A/pix]: source [keck/inst/mosfire/grating.html](https://www2.keck.hawaii.edu/inst/mosfire/grating.html)
- `resolution_pix` [pix (0.7 arcsec slit)]: source [keck/inst/mosfire/grating.html](https://www2.keck.hawaii.edu/inst/mosfire/grating.html)
- `lambda_c_A` [A]: source [keck/inst/mosfire/grating.html](https://www2.keck.hawaii.edu/inst/mosfire/grating.html). slit at field center
- `R` []: source [keck/inst/mosfire/grating.html](https://www2.keck.hawaii.edu/inst/mosfire/grating.html). 0.7 arcsec slit
- `coverage_A` [A]: source [keck/inst/mosfire/grating.html](https://www2.keck.hawaii.edu/inst/mosfire/grating.html). central slit; changes by ~6-12 A per mm of slit X position
- `throughput_peak_mosfire_plus_tel` [fraction]: source [keck/inst/mosfire/throughput.html](https://www2.keck.hawaii.edu/inst/mosfire/throughput.html). Peak of the measured 'Yeff_mosfire+tel.dat' (May 2012), at 10597 A. Peak of the measured 'Jeff_mosfire+tel.dat' (May 2012), at 13313 A. Peak of the measured 'Heff_mosfire+tel.dat' (May 2012), at 16154 A. Peak of the measured 'Keff_mosfire+tel.dat' (May 2012), at 21090 A.
- `throughput_median_in_band` [fraction]: source [keck/inst/mosfire/throughput.html](https://www2.keck.hawaii.edu/inst/mosfire/throughput.html). Median over the region above half of peak (my computation from the ASCII file).
- `rec_exposure_s` [s]: source [keck/inst/mosfire/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/mosfire/exposure_recipes.html). MCDS16. Tested to be near background-limited between OH lines.
- `rec_efficiency` [fraction]: source [keck/inst/mosfire/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/mosfire/exposure_recipes.html). texp / (texp + 23.3 s)
- `noise_darkest_regions_e` [[sky, total] e-]: source [keck/inst/mosfire/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/mosfire/exposure_recipes.html). 0.7 arcsec slit

#### MOSFIRE / imaging

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| filters_um | {"Y": [1.048, 0.152], "J": [1.253, 0.2], "H": [1.637, 0.341], "K": [2.162, 0.483], "Ks": [2.147, 0.314], "J2": [1.181, 0.129], "J3": [1.288, 0.122], "H1": [1.556, 0.165], "H2": [1.709, 0.167]} | [center, FWHM] um | doc | [keck/inst/mosfire/filters.html](https://www2.keck.hawaii.edu/inst/mosfire/filters.html) | K is not intended for imaging. |
| zeropoints_1e_per_s | {"Y": [28.13, 28.8], "J": [27.89, 28.79], "H": [27.54, 28.94], "Ks": [26.93, 28.78], "J2": [26.93, 27.83], "J3": [26.88, 27.78], "H1": [26.88, 28.28], "H2": [26.85, 28.25]} | [Vega, AB] mag for 1 e-/s, airmass 1 | doc | [keck/inst/mosfire/zeropoints.html](https://www2.keck.hawaii.edu/inst/mosfire/zeropoints.html) |  |
| optimal_exposure_s | {"Y": 30, "J": 10.2, "H": 2.9, "Ks": 5.9, "J2": 25, "J3": 30, "H1": 4.4, "H2": 4.3} | s | doc | [keck/inst/mosfire/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/mosfire/exposure_recipes.html) | For targets fainter than the sky. |
| sky_rate_e_per_pix_s | {"Y": [730, 820], "J": [2040, 2900], "H": [8940, 11980], "Ks": 16000} | e-/pix/s | doc | [keck/inst/mosfire/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/mosfire/exposure_recipes.html) |  |

#### MOSFIRE / observing_strategy

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| oh_variation_timescale_s | 30 | s | doc | [keck/inst/mosfire/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/mosfire/exposure_recipes.html) | &gt;0.1% OH variation on ~30 s timescales |
| dither_patterns | {"mask_nod": "A/B at +/- nod, typical 1-1.5 arcsec", "ABBA": "typical 1-1.5 arcsec", "ABA'B'": "outer 1.5, inner 1.2 arcsec", "slit_nod": "N-point along slit", "imaging": ["box4", "box5", "box9"]} |  | doc | [keck/inst/mosfire/dither_patterns.html](https://www2.keck.hawaii.edu/inst/mosfire/dither_patterns.html) | Default dither space 2.5 arcsec at design; 1.5 arcsec nod, 3 arcsec A-B separation. |
| example_clock_time | {"integration_s": 3600, "sequence": "20 x 180 s K MCDS16", "clock_s": 4150} |  | doc | [keck/inst/mosfire/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/mosfire/exposure_recipes.html) | 4066 s plus ~3-4 s per dither |
| csu_fatal_error | {"recovery_up_to_min": 90, "avoid_drive_angles": "within 10 deg of 0/180", "2025_test_rate": "1/210 moves"} |  | doc | [keck/inst/mosfire/observing_sequences.html](https://www2.keck.hawaii.edu/inst/mosfire/observing_sequences.html) | Rate from the 2025-04-29 news table. Suggested move elevation ~67 deg. |
| abort_modes | ["after frame (default)", "immediately (data corrupt)", "after read", "after group (UTR only)", "after coadd", "after repeat"] |  | doc | [keck/inst/mosfire/abort_options.html](https://www2.keck.hawaii.edu/inst/mosfire/abort_options.html) |  |
| mask_download_s | 60 | s | doc | [keck/inst/mosfire/dither_patterns.html](https://www2.keck.hawaii.edu/inst/mosfire/dither_patterns.html) | Masks need no advance submission. |
| mode_change_s | 90 | s | **EST** | [ucolick.org/documents/mosfire_spie.pdf](https://www.ucolick.org/documents/mosfire_spie.pdf) | Filter-wheel/pupil/grating move is not quantified (SPIE: filter-wheel cycle ~2x spec). Assume 60-120 s, overlapped with 'Wait & Go'. |

#### MOSFIRE / rotator

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| pa_definitions | {"SKYPA0_CSU_normal": "ROTDEST", "SKYPA1_guider_columns": "ROTDEST+90.00", "SKYPA2_detector_columns": "ROTDEST+0.22", "SKYPA3_longslit": "ROTDEST+4.00"} | deg | doc | [keck/inst/mosfire/skypa.html](https://www2.keck.hawaii.edu/inst/mosfire/skypa.html) | To put a longslit N-S, set ROTDEST = -4.00. |

#### MOSFIRE / sensitivity_anchors

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| ks_imaging_1hr | {"Ks_vega": 23.5, "snr": 10, "aperture_arcsec": 0.5, "fwhm_arcsec": 0.56, "t_s": 3600} |  | doc | [ucolick.org/documents/mosfire_spie.pdf](https://www.ucolick.org/documents/mosfire_spie.pdf) | 9-point dither, commissioning |
| mosdef_3sigma_continuum_per_pixel_2hr | {"J": [21.4, 22.1], "H": [21.1, 22.4], "K": [20.8, 21.6]} | AB mag range, per pixel (typ. ~2 h/filter, 60-245 min; seeing 0.5-0.8 arcsec) | doc | [arxiv.org/abs/1412.1835](https://arxiv.org/abs/1412.1835) | Kriek+2015 Table 1, total (slit-loss corrected), point source. The MOSFIRE ETC is optimistic by ~2x. |
| mosdef_5sigma_line_2hr | {"typical": 1.5e-17, "optimistic_H": 6.1e-18, "optimistic_K": 7.4e-18} | erg/s/cm^2 | doc | [arxiv.org/abs/1412.1835](https://arxiv.org/abs/1412.1835) | Slit-loss corrected |
| steidel2014_5sigma_line_1hr | {"H": 3.5e-18, "K": "4.5e-18 to 1.4e-17"} | erg/s/cm^2 | doc | [arxiv.org/abs/1412.1835](https://arxiv.org/abs/1412.1835) | As quoted by Kriek+2015; not slit-loss corrected |
| mosdef_exposures | {"Y": 180, "J": 120, "H": 120, "K": 180} | s per frame | doc | [arxiv.org/abs/1412.1835](https://arxiv.org/abs/1412.1835) |  |

### HIRES

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| deckers | {"B1-B4": [0.574, 72000, 67000], "B5/C1-C3": [0.861, 48000, 49000], "C4/C5/D1/D2": [1.148, 36000, 37000], "D3/D4": [1.722, 24000, 24000], "E1-E4": [0.4, 103000, 84000], "E5": [0.8, 51000, 52000]} | [slit width arcsec, R calculated, R measured UV-XD] | doc | [keck/inst/hires/slitres.html](https://www2.keck.hawaii.edu/inst/hires/slitres.html) | Lengths 3.5-28 arcsec by decker (B1 3.5, B2 7, B3 14, B4 28, C2 14, ...). The red-XD measured R is ~1% higher. |

#### HIRES / general

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| focus | Keck I right Nasmyth |  | doc | [keck/observing/kecktelgde/ktelinstupdate.pdf](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf) |  |
| wavelength_range_um | [0.3, 1.0] | um | doc | [keck/inst/hires/intro.html](https://www2.keck.hawaii.edu/inst/hires/intro.html) | Detector sensitivity 0.36-1.0 um (spec page). |
| configurations | {"HIRESb": "UV cross-disperser", "HIRESr": "red cross-disperser", "equal_efficiency_A": 4200} |  | doc | [keck/inst/hires/intro.html](https://www2.keck.hawaii.edu/inst/hires/intro.html) | No switching at night. |
| coverage_per_exposure_A | [3000, 4500] | A | doc | [keck/inst/hires/intro.html](https://www2.keck.hawaii.edu/inst/hires/intro.html) | Complete coverage shortward of 6200 A. Longer wavelengths need 2 settings. |
| image_rotator | optional; holds slit at a PA or at parallactic |  | doc | [keck/inst/hires/intro.html](https://www2.keck.hawaii.edu/inst/hires/intro.html) |  |
| iodine_cell_and_exposure_meter | True |  | doc | [keck/inst/hires/intro.html](https://www2.keck.hawaii.edu/inst/hires/intro.html) | Exposure meter PMT 300-650 nm; the pick-off costs 1% of the light. |

#### HIRES / detector

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| type | 3 x MIT/LL 2048x4096, 15 um (B, G, R) |  | doc | [keck/inst/hires/instrument_specifications.html](https://www2.keck.hawaii.edu/inst/hires/instrument_specifications.html) | since Aug 2004 |
| gain_low_e_per_adu | [1.95, 2.09, 2.09] | e-/ADU (B,G,R) | doc | [keck/inst/hires/ccdgain.html](https://www2.keck.hawaii.edu/inst/hires/ccdgain.html) | Spec page lists 1.9, 2.2, 2.2. Low gain is the default. |
| gain_high_e_per_adu | [0.78, 0.84, 0.89] | e-/ADU | doc | [keck/inst/hires/ccdgain.html](https://www2.keck.hawaii.edu/inst/hires/ccdgain.html) | Spec page lists 0.78, 0.86, 0.84. |
| read_noise_e | [2.8, 3.1, 3.1] | e- (B,G,R) | doc | [keck/inst/hires/instrument_specifications.html](https://www2.keck.hawaii.edu/inst/hires/instrument_specifications.html) |  |
| linear_limit_adu | [39800, 37900, 38400] | ADU | doc | [keck/inst/hires/instrument_specifications.html](https://www2.keck.hawaii.edu/inst/hires/instrument_specifications.html) |  |
| full_well_e_low_gain | 78000 | e- | approx | [keck/inst/hires/instrument_specifications.html](https://www2.keck.hawaii.edu/inst/hires/instrument_specifications.html) | Derived: 39800 ADU x 1.95. Nonlinearity becomes significant above ~half full well (ccdgain page). |
| dark_current_e_per_hr | 2 | e-/pix/hr | doc | [keck/inst/hires/hires_data.pdf](https://www2.keck.hawaii.edu/inst/hires/hires_data.pdf) |  |
| spatial_scale_arcsec | 0.12 | arcsec/pix | doc | [keck/inst/hires/binning.html](https://www2.keck.hawaii.edu/inst/hires/binning.html) | 'about 0.12 arcseconds on the sky, in the spatial direction' |
| velocity_per_pixel_km_s | 1.3 | km/s/pix | doc | [arxiv.org/abs/1401.4195](https://arxiv.org/abs/1401.4195) | 'each pixel spanned ~1.3 km/s' |
| dispersion_A_per_pix_5500 | 0.024 | A/pix | approx | [arxiv.org/abs/1401.4195](https://arxiv.org/abs/1401.4195) | Derived from 1.3 km/s x 5500 A / c. |

#### HIRES / readout

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| 1x1_s | 60 | s | doc | [keck/inst/hires/binning.html](https://www2.keck.hawaii.edu/inst/hires/binning.html) | 'about a minute' |
| 2x1_s | 40 | s | doc | [keck/inst/hires/binning.html](https://www2.keck.hawaii.edu/inst/hires/binning.html) | Binned 2 spatial (default mode). |
| 3x1_s | 30 | s | doc | [keck/inst/hires/binning.html](https://www2.keck.hawaii.edu/inst/hires/binning.html) |  |
| 2x2_s | 24 | s | doc | [keck/inst/hires/binning.html](https://www2.keck.hawaii.edu/inst/hires/binning.html) | 'roughly 24 seconds' |

#### HIRES / throughput

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| measured_page | plot only (Sept/Nov 2005) |  | doc | [keck/inst/hires/throughput.html](https://www2.keck.hawaii.edu/inst/hires/throughput.html) | Slitless 4x5 arcsec decker; no tabulated values. |
| total_peak_incl_telescope | 0.07 | fraction (excl. atmosphere & slit) | **EST** | [arxiv.org/abs/1401.4195](https://arxiv.org/abs/1401.4195) | My back-of-envelope from the Marcy+2014 anchor. The observed ~2% end-to-end includes the iodine cell, 0.86 arcsec slit losses and atmosphere; correcting those gives ~5-8%. Vogt+1994 (pre-2004 CCD) is ~8% near 6000 A per secondary summaries. |

#### HIRES / exposure_limits

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| typical_long_exposure_s | [600, 2700] | s | doc | [arxiv.org/abs/1401.4195](https://arxiv.org/abs/1401.4195) | 'typical long exposures of 10-45 minutes' (CPS / Kepler follow-up) |

#### HIRES / overheads

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| setup_changes | echelle/XD angle changes by software; XD swap impossible at night |  | doc | [keck/inst/hires/blue_vs_red.html](https://www2.keck.hawaii.edu/inst/hires/blue_vs_red.html) |  |
| exposure_meter_warmup_s | 600 | s | doc | [keck/inst/hires/exposure_meter.html](https://www2.keck.hawaii.edu/inst/hires/exposure_meter.html) | 'allow at least 10 minutes for the PMT to warm up'. |

#### HIRES / sensitivity_anchors

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| cps_kepler | {"Kp": 13.0, "t_s": 2700, "snr_per_pixel": 75, "decker": "C2 (0.861 x 14 arcsec)", "R": 50000, "iodine": true} |  | doc | [arxiv.org/abs/1401.4195](https://arxiv.org/abs/1401.4195) | 'at Kp = 13.0 mag, the typical exposure was 45 minutes, giving SNR=75 per pixel' |

### KCWI

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| gratings | `BL`: {"A_per_pix": 0.563, "R_LMS": [900, 1800, 3600], "total_A": [3500, 5600], "inst_A": 2000}<br>`BM`: {"A_per_pix": 0.24, "R_LMS": [2000, 4000, 8000], "total_A": [3500, 5500], "inst_A": [800, 900]}<br>`BH1`: {"A_per_pix": 0.09, "R_LMS": [4500, 9000, 18000], "total_A": [3500, 4100], "inst_A": 400}<br>`BH2`: {"A_per_pix": 0.111, "R_LMS": [4500, 9000, 18000], "total_A": [4000, 4800], "inst_A": [370, 440]}<br>`BH3`: {"A_per_pix": 0.129, "R_LMS": [4500, 9000, 18000], "total_A": [4700, 5600], "inst_A": [470, 530]}<br>`RL`: {"A_per_pix": 0.92, "R_LMS": ["&gt;500", "&gt;1000", "&gt;2000"], "total_A": [5400, 10800], "inst_A": [3300, 3700]}<br>`RM1`: {"A_per_pix": 0.355, "R_LMS": ["&gt;1400", "&gt;2800", "&gt;5600"], "total_A": [5500, 8600], "inst_A": [1260, 1430]}<br>`RM2`: {"A_per_pix": 0.5, "R_LMS": ["&gt;1400", "&gt;2800", "&gt;5600"], "total_A": [6700, 10800], "inst_A": [1750, 2000]}<br>`RH1`: {"A_per_pix": 0.162, "R_LMS": ["&gt;3250", "&gt;6500", "&gt;13000"], "total_A": [5500, 6800], "inst_A": [560, 630]}<br>`RH2`: {"A_per_pix": 0.19, "R_LMS": ["&gt;3250", "&gt;6500", "&gt;13000"], "total_A": [6300, 7800], "inst_A": [640, 730]}<br>`RH3`: {"A_per_pix": 0.223, "R_LMS": ["&gt;3250", "&gt;6500", "&gt;13000"], "total_A": [7700, 9500], "inst_A": [750, 850]}<br>`RH4`: {"A_per_pix": "0.28-0.33", "R_LMS": ["&gt;3250", "&gt;6500", "&gt;13000"], "total_A": [9200, 10800], "inst_A": [830, 1020]} |  | doc | [keck/inst/kcwi/configurations.html](https://www2.keck.hawaii.edu/inst/kcwi/configurations.html) | Red values are lab measurements (unbinned pixels). Nod-and-shuffle cuts the bandpass by ~4x. |
| readout_s | {"blue_1amp_slow": [337, 106], "blue_2amp_slow": [170, 53], "blue_4amp_slow": [85, 27], "blue_1amp_fast": [75, 25], "blue_2amp_fast": [38, 13], "blue_4amp_fast": [19, 7], "red_2amp_slow": [90, 30], "red_pre_exposure_erase": 13} | s [1x1, 2x2] | doc | [keck/inst/kcwi/overheads_estimated.html](https://www2.keck.hawaii.edu/inst/kcwi/overheads_estimated.html) | Blue 2-amp slow is the common mode. Do not use quad-amp with N&S. |
| overheads | {"target_acquisition_min": [0.5, 5], "slicer_change_s_per_station": 60, "blue_grating_change_min": 5, "blue_camera_rotation_s": 90, "red_grating_change_min": 2, "red_camera_rotation_s": 30, "cal_set_script_min": 3, "nas_settle_s_per_nod": [10, 20]} |  | doc | [keck/inst/kcwi/overheads_estimated.html](https://www2.keck.hawaii.edu/inst/kcwi/overheads_estimated.html) |  |
| exposure_limits | {"red_2x2_max_s": 300, "red_1x1_max_s": 600, "min_exposures_per_position": 3, "blue_example_s": [990, 1320]} |  | doc | [keck/inst/kcwi/detector-info.html](https://www2.keck.hawaii.edu/inst/kcwi/detector-info.html) | Blue exposures can only be shortened in progress; red can be lengthened or shortened. |

#### KCWI / general

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| focus | Keck II left Nasmyth |  | doc | [keck/inst/kcwi/specs.html](https://www2.keck.hawaii.edu/inst/kcwi/specs.html) |  |
| bandpass_nm | {"blue": [350, 560], "red": [560, 1080]} | nm | doc | [keck/inst/kcwi/specs.html](https://www2.keck.hawaii.edu/inst/kcwi/specs.html) | The red arm (KCRM) works simultaneously; dichroic at 560 nm. |
| slicers | {"Large": {"fov_arcsec": [33.1, 20.4], "slice_arcsec": 1.35}, "Medium": {"fov_arcsec": [16.5, 20.4], "slice_arcsec": 0.69}, "Small": {"fov_arcsec": [8.4, 20.4], "slice_arcsec": 0.35}} |  | doc | [keck/inst/kcwi/configurations.html](https://www2.keck.hawaii.edu/inst/kcwi/configurations.html) | R scales 1x : 2x : 4x for L : M : S. |
| pixel_scale_arcsec | 0.147 | arcsec/pix (along slice, 1x1) | doc | [keck/inst/kcwi/configurations.html](https://www2.keck.hawaii.edu/inst/kcwi/configurations.html) | The specs page says 0.15. |

#### KCWI / detector

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| blue | e2v CCD231-84 4k x 4k, 15 um |  | doc | [keck/inst/kcwi/specs.html](https://www2.keck.hawaii.edu/inst/kcwi/specs.html) |  |
| red | LBNL 4k x 4k, 500 um thick, Archon |  | doc | [keck/inst/kcwi/specs.html](https://www2.keck.hawaii.edu/inst/kcwi/specs.html) |  |
| read_noise_e | {"blue_slow": 2.7, "blue_fast": 5, "red_slow": 3} | e- | doc | [keck/inst/kcwi/overheads_estimated.html](https://www2.keck.hawaii.edu/inst/kcwi/overheads_estimated.html) |  |
| blue_dark_e_per_hr | 1 | e-/pix/hr | doc | [arxiv.org/abs/1807.10356](https://arxiv.org/abs/1807.10356) | at -110 C (Morrissey+2018) |
| saturation_counts | {"blue_high_gain": 65000, "red_high_gain": 63000} | DN | doc | [keck/inst/kcwi/detector-info.html](https://www2.keck.hawaii.edu/inst/kcwi/detector-info.html) | Nonlinearity starts ~60k (blue) / 58-60k (red). |
| red_cosmic_rays | 0.8-1.0% of pixels per 300 s at 2x2 |  | doc | [keck/inst/kcwi/detector-info.html](https://www2.keck.hawaii.edu/inst/kcwi/detector-info.html) |  |

#### KCWI / throughput

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| note | pre-ship and commissioning curves published only as plots |  | doc | [keck/inst/kcwi/throughput.html](https://www2.keck.hawaii.edu/inst/kcwi/throughput.html) | Commissioning curves include atmosphere at airmass 1 and 3 reflections; instrument alone is ~0.2 higher. |
| total_peak_BL_BM | 0.3 | fraction incl. telescope + atmosphere | **EST** | [keck/inst/kcwi/throughput.html](https://www2.keck.hawaii.edu/inst/kcwi/throughput.html) | Estimate: instrument ~0.45-0.5 at peak x telescope 0.8 x atmosphere 0.85. BH3 is ~25% below the others. |

#### KCWI / sensitivity_anchors

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| note | no numeric S/N anchor found on public pages |  | **EST** | [keck/inst/kcwi/throughput.html](https://www2.keck.hawaii.edu/inst/kcwi/throughput.html) | Use the KCWI ETC linked from the KCWI pages. |

## 4. Control interface (KTL)

### control

#### control / ktl_syntax

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| show | show [-binary] [-debug] [-terse] [-timestamp] -s &lt;service&gt; [keywords] |  | doc | [keck/inst/lris/cl.html](https://www2.keck.hawaii.edu/inst/lris/cl.html) | 'show -s &lt;service&gt; keywords' lists all keywords. -terse returns the bare value (for scripts). |
| modify | modify -s &lt;service&gt; keyword=value [keyword=value ...] [nowait] |  | doc | [keck/inst/lris/cl.html](https://www2.keck.hawaii.edu/inst/lris/cl.html) | Blocks until the move completes unless nowait. Scripts also use 'silent' (e.g. modify -s dcs silent raoff=..). |
| waitfor | waitfor -s &lt;service&gt; [-t &lt;timeout_s&gt;] keyword=value |  | doc | [keck/realpublic/observ..._info/deimos/procs/goi](https://www2.keck.hawaii.edu/realpublic/observing//public_instrument_info/deimos/procs/goi) | e.g. waitfor -s deiccd -t 10 wdisk=true |
| lris_shortcuts | {"s": "show -s lris", "sb": "show -s lrisblue", "m": "modify -s lris", "mb": "modify -s lrisblue"} |  | doc | [keck/inst/lris/cl.html](https://www2.keck.hawaii.edu/inst/lris/cl.html) |  |
| services | `Keck I telescope`: dcs<br>`primary ACS`: acs<br>`LRIS red+shared`: lris<br>`LRIS blue detector`: lrisblue<br>`DEIMOS CCD`: deiccd<br>`DEIMOS motors`: deimot<br>`DEIMOS rotator`: deirot<br>`DEIMOS FCS`: deifcs<br>`MOSFIRE global`: mosfire<br>`MOSFIRE detector`: mds<br>`MOSFIRE CSU`: mcsus<br>`HIRES CCD`: hiccd<br>`HIRES instrument`: hires |  | doc | [keck/inst/lris/cl.html](https://www2.keck.hawaii.edu/inst/lris/cl.html) | LRIS/ACS/DCS: cl.html; DEIMOS: deimos-keyword-summary.html; MOSFIRE: mosfire_keywords.html; HIRES: troublepage.html (show -s hiccd ..., show -s hires lfilname). On DEIMOS/K2 hosts the telescope service is also addressed as 'dcs' (DEIMOS scripts). |

#### control / dcs_keywords

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| AIRMASS | AIRMASS |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | air mass (double) |
| AZ | AZ |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | telescope azimuth (double) |
| EL | EL |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | telescope elevation (double) |
| RA | RA |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | telescope right ascension (double) |
| DEC | DEC |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | telescope declination (double) |
| HA | HA |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | telescope hour angle |
| LST | LST |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | local apparent sidereal time |
| UT | UT |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | coordinated universal time |
| PARANG | PARANG |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | parallactic angle astrometric |
| ROTMODE | ROTMODE |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | rotator tracking mode (enum) |
| ROTDEST | ROTDEST |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | rotator user destination |
| ROTPOSN | ROTPOSN |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | rotator user position |
| ROTPPOSN | ROTPPOSN |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | rotator physical position |
| TARGNAME | TARGNAME |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | target name |
| TARGRA | TARGRA |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | target RA |
| TARGDEC | TARGDEC |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | target Dec |
| RAOFF | RAOFF |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | right ascension offset |
| DECOFF | DECOFF |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | declination offset |
| AZOFF | AZOFF |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | azimuth offset |
| ELOFF | ELOFF |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | elevation offset |
| INSTXOFF | INSTXOFF |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | instrument x offset |
| INSTYOFF | INSTYOFF |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | instrument y offset |
| REL2BASE | REL2BASE |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | offset relative to base (enum) |
| REL2CURR | REL2CURR |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | offset relative to current (enum) |
| MARK | MARK |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | mark current as base (logical) |
| GOTOBASE | GOTOBASE |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | go to base |
| TELLOLM | TELLOLM |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | telescope lower el limit |
| TELUPLM | TELUPLM |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | telescope upper el limit |
| SLEWTIME | SLEWTIME |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | total slew time |
| DOMEPOSN | DOMEPOSN |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | dome azimuth |
| TRACKING | TRACKING |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | telescope servos tracking |
| GUIDING | GUIDING |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | guiding on |
| AUTACTIV | AUTACTIV |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | guider active |
| AUTGO | AUTGO |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | guider go flag |
| AUTRESUM | AUTRESUM |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | guider resume counter (used by wftel) |
| FLIMAGIN | FLIMAGIN |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | imaging dome-flat lamp (enum) |
| FLSPECTR | FLSPECTR |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | spectral dome-flat lamp (enum) |
| TELFOCUS | TELFOCUS |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | telescope focus compensated |
| EXPOSIP | EXPOSIP |  | doc | [keck/inst/lris/dcs_key_list.html](https://www2.keck.hawaii.edu/inst/lris/dcs_key_list.html) | instrument exposure in progress |

#### control / telescope_offset_commands

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| en_radec | modify -s dcs silent raoff=&lt;E_arcsec&gt; decoff=&lt;N_arcsec&gt; rel2curr=t |  | doc | [keck/realpublic/observ...t_info/deimos/procs/en](https://www2.keck.hawaii.edu/realpublic/observing//public_instrument_info/deimos/procs/en) | DEIMOS 'en' script |
| azel | modify -s dcs silent azoff=&lt;arcsec&gt; eloff=&lt;arcsec&gt; rel2curr=t |  | doc | [keck/realpublic/observ...info/deimos/procs/azel](https://www2.keck.hawaii.edu/realpublic/observing//public_instrument_info/deimos/procs/azel) |  |
| mxy_instrument | modify -s dcs silent instxoff=&lt;x&gt; instyoff=&lt;y&gt; rel2curr=t  (rel2base=t variant exists) |  | doc | [keck/realpublic/observ..._info/deimos/procs/mxy](https://www2.keck.hawaii.edu/realpublic/observing//public_instrument_info/deimos/procs/mxy) |  |
| markbase | modify -s dcs mark=true |  | doc | [keck/realpublic/observ.../deimos/procs/markbase](https://www2.keck.hawaii.edu/realpublic/observing//public_instrument_info/deimos/procs/markbase) |  |
| gotobase | modify -s dcs raoff=0 decoff=0 rel2base=true |  | doc | [keck/realpublic/observ.../deimos/procs/gotobase](https://www2.keck.hawaii.edu/realpublic/observing//public_instrument_info/deimos/procs/gotobase) |  |
| dome_flats_flag | modify -s dcs domecals=true / false |  | doc | [keck/inst/deimos/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/deimos/exposure_recipes.html) | Tells the summit crew not to enter the dome. |
| dome_lamps | modify -s dcs flimagin=0 flspectr=1 |  | doc | [keck/inst/deimos/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/deimos/exposure_recipes.html) |  |
| wftel | wftel: wait for telescope move / guider resume (polls dcs AUTACTIV, AUTRESUM, AUTGO) |  | doc | [keck/realpublic/observ...nfo/deimos/procs/wftel](https://www2.keck.hawaii.edu/realpublic/observing//public_instrument_info/deimos/procs/wftel) |  |

#### control / LRIS

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| keywords | `TTIME`: total (exposure) time, integer s (lris & lrisblue)<br>`EXPTIME`: exposure time (lris)<br>`OBJECT`: object name<br>`OBSERVER`: observer<br>`EXPOSE`: start exposure (logical)<br>`ABORTEX`: abort exposure<br>`PAUSE`: pause<br>`RESUME`: resume<br>`EXPOSIP`: exposure in progress<br>`ELAPTIME`: elapsed time<br>`FRAMENO`: frame number<br>`OUTFILE`: readout filename<br>`BINNING`: binning [2]<br>`WINDOW`: window [5]<br>`NUMAMPS`: number of amps<br>`SLITNAME`: slitmask name<br>`GRANAME`: grating name<br>`GRANGLE`: grating angle<br>`WAVELEN`: grating wavelength (longslit)<br>`MSWAVE`: multislit wavelength<br>`GRISNAME`: grism name<br>`BLUFILT`: blue filter<br>`REDFILT`: red filter<br>`DICHNAME`: dichroic name<br>`TRAPDOOR`: hatch<br>`LAMPS`: lamps [6] |  | doc | [keck/inst/lris/instrument_key_list.html](https://www2.keck.hawaii.edu/inst/lris/instrument_key_list.html) | The list predates the 2021 Mark IV red CCD. Red exposures are now driven by scripts (tintr/goir/abortr); the red-CCD keyword names may differ. |
| scripts | `tintr <s>`: set/show red exposure time<br>`tintb <s>`: set/show blue exposure time<br>`goir [N]`: take N red exposures<br>`goib [N]`: take N blue exposures<br>`goibr [N]`: red + blue<br>`abortr / abortb / abortbr`: abort (red: no readout/save)<br>`object <name>`: OBJECT<br>`slit <mask>`: slitmask<br>`grating [-mswave] <name> <wave>`: grating + central wavelength<br>`grism <name>`: blue grism<br>`filterb / filterr <name>`: filters<br>`dichroic <name>`: dichroic<br>`lamps <list/off>`: cal lamps<br>`trapopen / trapclose`: hatch<br>`lredccd_setup imag/spec <bx> <by>`: Mark IV red binning/amps<br>`windowb x y w h`: blue window<br>`binning1x2b / binning2x2b`: blue binning<br>`en <e> <n>`: offset E/N arcsec<br>`azel`: offset az/el<br>`mxy`: offset detector coords<br>`gxy`: offset guider coords<br>`sltmov <arcsec>`: move along slit<br>`movb/movr x1 y1 x2 y2`: put object at pixel<br>`markbase / gotobase`: base coords<br>`bxy5b/bxy9br ...`: imaging dithers |  | doc | [keck/inst/lris/procs/index.html](https://www2.keck.hawaii.edu/inst/lris/procs/index.html) | Also redccdmanual.html, dither.html, shell_scripts.html (csh example uses tintr, tintb, goibr, goir, goib, lamps, slit, grating -mswave). |

#### control / DEIMOS

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| keywords | `DEICCD.TTIME`: total time (s)<br>`DEICCD.EXPTIME`: exposure time<br>`DEICCD.OBJECT`: object<br>`DEICCD.EXPOSE`: start exposure (modify -s deiccd expose=t wait)<br>`DEICCD.ABORTEX`: abort<br>`DEICCD.STOPEX`: stop exposure<br>`DEICCD.PAUSE / RESUME`: pause/resume<br>`DEICCD.EXPOSIP`: exposure in progress<br>`DEICCD.ELAPTIME`: elapsed<br>`DEICCD.FRAMENO`: frame number<br>`DEICCD.OUTDIR / OUTFILE`: output<br>`DEICCD.WCRATE / WSERV / WDISK`: readout/transfer/write flags<br>`DEICCD.BINNING`: binning<br>`DEICCD.MOSMODE`: mosaic readout mode<br>`DEIMOT.GRATENAM`: grating<br>`DEIMOT.G3TLTWAV / G4TLTWAV`: slider 3/4 wavelength<br>`DEIMOT.SLMSKNAM`: slitmask<br>`DEIMOT.DWFILNAM`: filter<br>`DEIROT.ROTATVAL`: rotator value |  | doc | [keck/inst/deimos/deimos-keyword-summary.html](https://www2.keck.hawaii.edu/inst/deimos/deimos-keyword-summary.html) |  |
| scripts | `tint <s>`: modify -s deiccd ttime=&lt;s&gt;<br>`goi [-nowait] [-dark] [N]`: take N exposures (expose=t, then waitfor wcrate/wserv/wdisk)<br>`abortex`: modify -s deiccd abortex=1<br>`object <name>`: modify -s deiccd object=...<br>`wffcs`: wait for FCS lock<br>`wfi`: wait for new image<br>`wftel`: wait for telescope<br>`spectral / direct`: readout mode<br>`gozero`: zeroth-order imaging<br>`grating / slider / wavelen`: grating setup<br>`slitmask <name>`: mask<br>`filter <name>`: filter<br>`skypa / rotang`: rotator<br>`mov / gmov / mxy / en / azel / markbase / gotobase`: offsets<br>`abba / ab`: spectroscopic nod sequences<br>`bxy5/8/9`: imaging dithers |  | doc | [keck/inst/deimos/procs/](https://www2.keck.hawaii.edu/inst/deimos/procs/) | Typical science start from the checklist: 'wffcs ; goi'. |
| goi_sequence | ["modify -s deiccd silent expose=t wait", "waitfor -s deiccd wcrate=true", "waitfor -s deiccd wserv=true", "waitfor -s deiccd wcrate=false", "waitfor -s deiccd wserv=false", "waitfor -s deiccd -t 10 wdisk=true", "waitfor -s deiccd wdisk=false"] |  | doc | [keck/realpublic/observ..._info/deimos/procs/goi](https://www2.keck.hawaii.edu/realpublic/observing//public_instrument_info/deimos/procs/goi) | Documented end-of-exposure keyword transitions. |

#### control / MOSFIRE

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| keywords | `MOSFIRE.ITIME`: integration time per coadd in ms<br>`MOSFIRE.COADDS`: coadds<br>`MOSFIRE.SAMPMODE`: 1 Single, 2 CDS, 3 MCDS, 4 UTR<br>`MOSFIRE.NUMREADS`: reads per group<br>`MOSFIRE.GROUPS`: sampling groups<br>`MOSFIRE.OBJECT`: object<br>`MOSFIRE.OBSERVER`: observer<br>`MOSFIRE.GO`: start a saved exposure<br>`MOSFIRE.ABORT`: 1 now, 2 after read, 3 after group, 4 after coadd<br>`MOSFIRE.READY`: 0 busy/error, 1 ok to expose<br>`MOSFIRE.EXPOSING`: exposure in progress<br>`MOSFIRE.IMAGEDONE`: 1 when written<br>`MOSFIRE.PROGRESS`: percent complete<br>`MOSFIRE.LASTFILE`: last file<br>`MOSFIRE.FRAMENUM`: frame number<br>`MOSFIRE.OUTDIR`: outdir<br>`MOSFIRE.FILTER`: filter<br>`MOSFIRE.OBSMODE`: observing mode (e.g. 'Y-spectroscopy')<br>`MOSFIRE.MASKNAME`: current mask<br>`MOSFIRE.SETUPNAME`: set-up mask<br>`MOSFIRE.CSUREADY`: CSU state (2 ready, 3 moving, 4 configuring, -1 error)<br>`MOSFIRE.PATTERN`: dither pattern<br>`MOSFIRE.NODE`: RA nod offset (arcsec)<br>`MOSFIRE.NODN`: Dec nod offset (arcsec)<br>`MDS.* / MCSUS.*`: detector / CSU servers mirror the same names |  | doc | [keck/inst/mosfire/mosfire_keywords.html](https://www2.keck.hawaii.edu/inst/mosfire/mosfire_keywords.html) |  |
| commands | `modify -s mosfire node=<ra_arcsec>`: imaging dither size RA<br>`modify -s mosfire nodn=<dec_arcsec>`: imaging dither size Dec<br>`box4 / box5 / box9`: imaging dither scripts<br>`markbase`: mark base (on mosfireserver)<br>`acq_long2pos`: long2pos script<br>`Quick Dark / Wait & Go (GUI)`: dark filter / start sequence after mechanisms |  | doc | [keck/inst/mosfire/observing_sequences.html](https://www2.keck.hawaii.edu/inst/mosfire/observing_sequences.html) |  |

#### control / HIRES

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| services | {"hiccd": "CCD (e.g. show -s hiccd dwrxfrsw; modify -s hiccd autofill=1)", "hires": "instrument (e.g. show -s hires lfilname)"} |  | doc | [keck/inst/hires/troublepage.html](https://www2.keck.hawaii.edu/inst/hires/troublepage.html) | No public HIRES exposure keyword list found; exposure keywords not confirmed here. |

#### control / KCWI

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| scripts | `tintb <s>`: blue exposure time<br>`goib [N]`: blue exposures<br>`object`: object<br>`slicer`: slicer<br>`gratingb`: blue grating<br>`cwaveb / pwaveb`: central / peak wavelength<br>`filterb`: blue filter<br>`nsmaskb`: N&S mask<br>`binningb / ampmodeb / ccdspeedb`: detector<br>`goifpc / tintfc`: focal-plane camera<br>`moveFpc`: offset via FPC coords<br>`hatch / lamp / calmirror`: calibration |  | doc | [keck/inst/kcwi/scripts_docs/index.html](https://www2.keck.hawaii.edu/inst/kcwi/scripts_docs/index.html) | Red-side equivalents exist (framer, nextfiler, ...), but red exposure-script names were not seen in the public list. |

## 5. ETC calibration anchors (digest)

| Parameter | Value | Unit | Conf | Source | Note |
|---|---|---|---|---|---|
| LRIS: effective_area | ~10 m^2 blue, ~15 m^2 red (12-20% total) |  | doc | [arxiv.org/abs/1903.07629](https://arxiv.org/abs/1903.07629) | Observed from standard-star response functions. |
| LRIS: old_red_AB20_count_rate | {"300/5000@6200A": 4.3, "600/7500@6600A_peak": 2.5} | photons/s/pixel above atmosphere, AB=20 | doc | [keck/inst/lris/specEffOldRed.html](https://www2.keck.hawaii.edu/inst/lris/specEffOldRed.html) | 1995, OLD Tek CCD with 24 um pixels. Use the efficiencies, not these per-pixel rates. |
| LRIS: imaging_zeropoints_1994 | {"B": 27.34, "V": 27.51, "R": 27.52, "I": 27.4} | mag giving 1 DN/s at airmass 1 (Vega) | approx | [keck/inst/lris/photometric_zero_points.html](https://www2.keck.hawaii.edu/inst/lris/photometric_zero_points.html) | Mean of two nights, 1.72 e-/DN, ORIGINAL red CCD. Historical; recalibrate for Mark IV. |
| DEIMOS: spec_prediction_600l | {"V=22": 21, "V=23": 12, "V=24": 5} | S/N per pixel in 3600 s | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) | Point source, 0.75 arcsec slit, 0.7 arcsec seeing, 5500 A, 0.65 A/px, V sky 21.25 mag/arcsec^2. V is the magnitude through the slit (predicted). |
| DEIMOS: achieved_1200G_faint_galaxies | {"R=21": "2.2-4.6", "R=22": "1.2-2.4", "R=23": "0.6-1.2", "R=24": "0.3-0.6"} | S/N per pixel in 3600 s | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) | 1.0 arcsec slit, ~7000 A, 0.32 A/px, average conditions, total mag (no slit-loss correction). |
| DEIMOS: deep2_1200G_1hr | `18.5`: [5.53, 8.37, 10.89]<br>`19.0`: [4.81, 6.77, 9.74]<br>`19.5`: [3.34, 5.89, 8.28]<br>`20.0`: [2.92, 5.35, 7.43]<br>`20.5`: [2.67, 4.45, 6.18]<br>`21.0`: [2.19, 3.29, 4.6]<br>`21.5`: [1.67, 2.42, 3.25]<br>`22.0`: [1.23, 1.79, 2.4]<br>`22.5`: [0.88, 1.29, 1.73]<br>`23.0`: [0.61, 0.88, 1.24]<br>`23.5`: [0.42, 0.61, 0.83]<br>`24.0`: [0.32, 0.45, 0.61] | R_AB -&gt; [Q1, median, Q3] S/N per pixel in 1 h | doc | [keck/inst/deimos/deep2_s2n.html](https://www2.keck.hawaii.edu/inst/deimos/deep2_s2n.html) | ~50,000 DEEP2 galaxies, 1200G, 6600-6830 and 6960-7250 A, 1 arcsec slits. Galaxies are extended, so point sources do better. |
| DEIMOS: count_rates_predicted | {"imaging_V21_e_per_s": 800, "spec_6000A_V21_e_per_s": 1.0} | e-/s | doc | [keck/inst/deimos/specs.html](https://www2.keck.hawaii.edu/inst/deimos/specs.html) | Predicted; spectroscopic rate per pixel. |
| DEIMOS: imaging_zeropoints_2015 | {"V": 27.5, "R": 27.9, "I": 28.0, "Z": 27.0} | Vega mag for 1 DN/s | doc | [keck/inst/deimos/deimos_imaging_zps.html](https://www2.keck.hawaii.edu/inst/deimos/deimos_imaging_zps.html) | 2015-12-17, airmass 1.04. |
| DEIMOS: night_sky_between_OH | {"1200G_cont": 0.01, "1200G_OH_peak": 1.0, "830G_cont": 0.015, "830G_OH_peak": 1.5} | e-/s/px | doc | [keck/inst/deimos/exposure_recipes.html](https://www2.keck.hawaii.edu/inst/deimos/exposure_recipes.html) |  |
| MOSFIRE: ks_imaging_1hr | {"Ks_vega": 23.5, "snr": 10, "aperture_arcsec": 0.5, "fwhm_arcsec": 0.56, "t_s": 3600} |  | doc | [ucolick.org/documents/mosfire_spie.pdf](https://www.ucolick.org/documents/mosfire_spie.pdf) | 9-point dither, commissioning |
| MOSFIRE: mosdef_3sigma_continuum_per_pixel_2hr | {"J": [21.4, 22.1], "H": [21.1, 22.4], "K": [20.8, 21.6]} | AB mag range, per pixel (typ. ~2 h/filter, 60-245 min; seeing 0.5-0.8 arcsec) | doc | [arxiv.org/abs/1412.1835](https://arxiv.org/abs/1412.1835) | Kriek+2015 Table 1, total (slit-loss corrected), point source. The MOSFIRE ETC is optimistic by ~2x. |
| MOSFIRE: mosdef_5sigma_line_2hr | {"typical": 1.5e-17, "optimistic_H": 6.1e-18, "optimistic_K": 7.4e-18} | erg/s/cm^2 | doc | [arxiv.org/abs/1412.1835](https://arxiv.org/abs/1412.1835) | Slit-loss corrected |
| MOSFIRE: steidel2014_5sigma_line_1hr | {"H": 3.5e-18, "K": "4.5e-18 to 1.4e-17"} | erg/s/cm^2 | doc | [arxiv.org/abs/1412.1835](https://arxiv.org/abs/1412.1835) | As quoted by Kriek+2015; not slit-loss corrected |
| MOSFIRE: mosdef_exposures | {"Y": 180, "J": 120, "H": 120, "K": 180} | s per frame | doc | [arxiv.org/abs/1412.1835](https://arxiv.org/abs/1412.1835) |  |
| HIRES: cps_kepler | {"Kp": 13.0, "t_s": 2700, "snr_per_pixel": 75, "decker": "C2 (0.861 x 14 arcsec)", "R": 50000, "iodine": true} |  | doc | [arxiv.org/abs/1401.4195](https://arxiv.org/abs/1401.4195) | 'at Kp = 13.0 mag, the typical exposure was 45 minutes, giving SNR=75 per pixel' |
| KCWI: note | no numeric S/N anchor found on public pages |  | **EST** | [keck/inst/kcwi/throughput.html](https://www2.keck.hawaii.edu/inst/kcwi/throughput.html) | Use the KCWI ETC linked from the KCWI pages. |

## Appendix A. Buton+2013 Maunakea median extinction (mag/airmass, continuum only)

| lambda (A) | k | lambda (A) | k | lambda (A) | k |
|---|---|---|---|---|---|
| 3200 | 0.856 | 5500 | 0.106 | 7800 | 0.029 |
| 3300 | 0.588 | 5600 | 0.107 | 7900 | 0.028 |
| 3400 | 0.514 | 5700 | 0.108 | 8000 | 0.027 |
| 3500 | 0.448 | 5800 | 0.103 | 8100 | 0.026 |
| 3600 | 0.400 | 5900 | 0.098 | 8200 | 0.025 |
| 3700 | 0.359 | 6000 | 0.098 | 8300 | 0.024 |
| 3800 | 0.323 | 6100 | 0.092 | 8400 | 0.023 |
| 3900 | 0.292 | 6200 | 0.084 | 8500 | 0.023 |
| 4000 | 0.265 | 6300 | 0.078 | 8600 | 0.022 |
| 4100 | 0.241 | 6400 | 0.070 | 8700 | 0.021 |
| 4200 | 0.220 | 6500 | 0.065 | 8800 | 0.021 |
| 4300 | 0.202 | 6600 | 0.060 | 8900 | 0.020 |
| 4400 | 0.185 | 6700 | 0.056 | 9000 | 0.019 |
| 4500 | 0.171 | 6800 | 0.052 | 9100 | 0.019 |
| 4600 | 0.159 | 6900 | 0.048 | 9200 | 0.018 |
| 4700 | 0.147 | 7000 | 0.044 | 9300 | 0.018 |
| 4800 | 0.139 | 7100 | 0.042 | 9400 | 0.017 |
| 4900 | 0.130 | 7200 | 0.039 | 9500 | 0.017 |
| 5000 | 0.125 | 7300 | 0.037 | 9600 | 0.016 |
| 5100 | 0.119 | 7400 | 0.035 | 9700 | 0.016 |
| 5200 | 0.114 | 7500 | 0.033 | 9800 | 0.015 |
| 5300 | 0.113 | 7600 | 0.032 | 9900 | 0.015 |
| 5400 | 0.109 | 7700 | 0.030 | 10000 | 0.014 |

Source: [Buton et al. 2013, A&A 549, A8, Table 6](https://arxiv.org/abs/1210.2619). Telluric O2/H2O bands are not included. Use an ATRAN/SkyCalc transmission for those.

## 6. Source list

Keck pages (www2.keck.hawaii.edu):
- Telescope: TelLimits, the 2002 Telescope & Facility Instrument Guide (PDF), NIRC manual (overheads), Weather_Table, SplitNights, metrics/Intro, parallactic, lgsao/lgsbasics, common/exts.
- LRIS: lrishome, detectors, dispersive_elements (+ data_files), win_bin, slitmask, dichroic, news, redccdmanual, cl, instrument_key_list, dcs_key_list, procs, longslit_setup, photometric_zero_points, specEffOldRed.
- DEIMOS: specs, gratings, detector data, primer, ripisc (+ ECSV tables), deep2_s2n, imaging zps, standard_slitmasks, checklist-align-xbox, keyword summary, procs (+ script sources), news, exposure_recipes, faint-long.
- MOSFIRE: home, detector, grating, filters, exposure_recipes, throughput (+ ASCII), zeropoints, observing_sequences, dither_patterns, mosfire_keywords, skypa, rotator, persistence, news, abort_options.
- HIRES: instrument_specifications, binning, ccdgain, slitres, intro, hires_data.pdf, blue_vs_red, throughput, troublepage.
- KCWI: specs, configurations, overheads_estimated, detector-info, throughput, scripts_docs.

Papers and other observatories:
- Buton+2013 (arXiv:1210.2619), Leggett+2006 (MNRAS 373, 781), Schock+2009 TMT (arXiv:0904.1183), Lyman+2020 (MNRAS 496, 4734), van Kooten & Izett 2022 (arXiv:2208.11794).
- McLean+2012 MOSFIRE SPIE 8446, Kriek+2015 MOSDEF (arXiv:1412.1835), Newman+2013 DEEP2 (arXiv:1203.3192), Steidel+2004 (astro-ph/0401439), Perley 2019 LPipe (arXiv:1903.07629), Marcy+2014 (arXiv:1401.4195), Morrissey+2018 KCWI (arXiv:1807.10356).
- CFHT Observatory Manual Sec. 2 and the MegaCam spec page (via web.archive.org), Gemini "The Sites", C. Steidel's LRIS-B overview, astropy-data sites.json.
