# Exposure time calculator: physics and validation

`obsassist/etc.py` turns seeing, airmass, Moon, clouds and an instrument configuration into
electrons, and electrons into S/N, exposure plans and saturation limits. Every other part of
the project (the console, the planner, the oracle, the assistant) uses it.

## Physics

Per exposure of length *t*, for a point source through a slit:

    S   = N_ph(λ) · A · η(λ) · 10^(−0.4 (k(λ) X + cloud)) · Δλ · f_slit · f_extract     [e⁻/s]
    B   = I_sky(λ) · A · η(λ) · Δλ · (slit width × spatial pixel)                          [e⁻/s/pix]
    S/N = S t / sqrt(S t + n_pix (B t + D t + RN²))

* **Photons**: N_ph = f_ν / (h λ) for an AB magnitude (AB = 0 gives ~1000 photons s⁻¹ cm⁻² Å⁻¹ at 5500 Å).
  The continuum is flat in f_ν, flat in f_λ or a blackbody (`sed: "bb:3200"`), normalised in the
  target's band (AB or Vega).
* **Dispersion** Δλ, Å per binned pixel: constant for gratings and grisms; proportional to λ for
  the echelles (MIKE, HIRES), whose orders span the same velocity per pixel.
* **Throughput** η(λ): telescope + instrument + detector, from the observatory ETC zero points or
  instrument pages ([research files](research/)); the atmosphere is applied separately with the
  site's extinction curve.
* **Image quality**: DIMM seeing at 500 nm scaled as X^0.6 λ^−0.2, with the von Kármán outer-scale
  correction (Tokovinin 2002, L0 = 25 m), plus the telescope/dome floor and defocus in quadrature.
  At Magellan a median DIMM of 0.62″ gives 0.57″ delivered, matching the measured Magellan median
  of 0.575″.
* **PSF**: Moffat, β = 4.765 (Trujillo et al. 2001). Slit transmission is the Moffat line-spread
  function over the slit, in closed form (incomplete beta function), including a centroid
  offset from differential refraction when there is no ADC and the slit is not at the parallactic
  angle. Extraction over ±1 FWHM along the slit.
* **Sky**: dark-sky zenith brightness per band (site tables), brightened toward the horizon;
  moonlight from Krisciunas & Schaefer (1991), carried to other wavelengths through the extinction
  coefficient (scattering) and the solar colour; twilight after Patat et al. (2006); near-IR sky
  between OH lines through a per-configuration factor (MOSFIRE measured).
* **Combining exposures**: optimal weights, S/N_tot² = Σ S/N_i².
* **Saturation**: 80 % of the full well (or of the ADC limit, whichever is lower) in the brightest
  pixel of every arm exposed together (both MIKE arms) and at every wavelength of the continuum
  (a hot star observed for a red-arm goal saturates first in the blue); exposure plans are
  shortened to stay below it.

## Validation

| Check | Result |
|---|---|
| Fast ephemerides vs a full astropy AltAz transform | ≤ 0.005° in altitude and azimuth above 15° |
| Krisciunas & Schaefer Table 2 (Moon at 60°, phase 30°) | 5/5 values within 5 % (`tests/test_astro.py`) |
| Official LCO ETC, re-run with the same inputs (1.0″ slit, 0.6″ seeing, X = 1.2, 3 × 1200 s) | see below |
| Synthetic FITS frames through the quick-look, against the ETC at the reference wavelength | mean S/N within 3 % (single faint frames scatter by up to 10 %, as any measured S/N does); peak counts 0.86–1.07 of the ETC's (`tests/test_frames.py`) |

S/N per unbinned pixel, AB magnitude, new Moon:

| Configuration | Source | This ETC | LCO ETC | Ratio |
|---|---|---|---|---|
| LDSS3-C VPH-All, 6500 Å | AB 20 | 36.4 | 42.5 | 0.86 |
| LDSS3-C VPH-All, 6500 Å | AB 22 | 7.6 | 9.4 | 0.81 |
| IMACS f/2 300-line, 5000 Å | AB 20 | 42.9 | 49.0 | 0.88 |
| IMACS f/2 300-line, 6500 Å | AB 20 | 35.5 | 43.2 | 0.82 |
| IMACS f/2 300-line, 8000 Å | AB 20 | 20.4 | 28.0 | 0.73 |
| MIKE red, 6500 Å | AB 16 | 35.4 | 43.2 | 0.82 |
| MIKE blue, 4500 Å | AB 16 | 31.3 | 37.7 | 0.83 |

This ETC is 12–27 % more conservative than the official LCO calculator. The LCO ETC assumes a
Gaussian PSF; the Moffat wings used here lose more light at the slit, which explains about half
the difference. The rest comes from the throughput curves, which were derived from the LCO
zero-point files and are themselves approximations. No fudge factor is applied. During a night
the assistant compares measured and predicted counts on every frame, which is the
throughput check observers make anyway.

At Keck, for DEIMOS 600ZD and 1200G, the ETC sits between the two published numbers, as
expected for a point-source calculation:
* **Below the optimistic instrument-page prediction:** V = 22 in 1 h gives S/N 8.7 per pixel
  here, against a predicted 21.
* **Above what DEEP2 achieved on faint galaxies** (1200G, R = 22: 1.2–2.4 per pixel per hour).
  Galaxies are extended and lose more light at the slit, so a point-source calculation should
  come out 1.5–2× higher, and it does.

## Known limitations

* One S/N wavelength per target; no full spectral S/N curve in the planner (the synthetic
  frames do carry the full spectrum).
* Sky emission lines enter the optical sky only through band-averaged brightness; near-IR uses
  an inter-line factor.
* Dual-arm instruments book the S/N goal on the arm that matches the target's configuration.
* Instrument numbers marked "approximate" or "estimate" in `obsassist/instruments/*.py` should
  be replaced by your own measured values before relying on absolute exposure times.
