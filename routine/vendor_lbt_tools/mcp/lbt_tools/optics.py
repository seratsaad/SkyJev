"""Sky-brightness and slit-loss physics ported from Mark Whittle's LBTO_Planner.xlsx
(the spreadsheet behind Yifan Zhou's browser planner).

1. Scattered moonlight at a target: Krisciunas & Schaefer (1991, PASP 103, 1033),
   exactly as the spreadsheet's MODS!AH/AI columns and Calcs!B62/B63:
       I*      = 10^(-0.4 (3.84 + 0.026 |alpha| + 4e-9 alpha^4))      alpha = 180 - elongation (deg)
       f(rho)  = 10^5.36 (1.06 + cos^2 rho) + 10^(6.15 - rho/40)      rho = moon-target angle (deg)
       B_moon  = f(rho) I* 10^(-0.4 k X_moon) (1 - 10^(-0.4 k X_target))   [nanoLamberts], k = 0.143 mag/airmass
       V_moon  = 22.5 - 1.086 ln(B/34.08)  mag/arcsec^2
   The spreadsheet uses X = sec(ZD); we keep that for fidelity.  A dark-sky
   value (21.9 V mag/arcsec^2, Mt. Graham) is combined to give the total sky.

2. Differential atmospheric refraction (AtmDisp sheet, Filippenko 1982):
       dR(lambda) = F_atm tan(ZD) [6085 (1/(146 - lambda^-2) - 1/(146 - l0^-2)) + 52.6 (1/(41 - lambda^-2) - 1/(41 - l0^-2))]  arcsec
       F_atm = exp(-h/8400) * 288/(273 + T)      h = 3269 m, T = 2 C  ->  0.71
   relative to the guide wavelength l0 (0.65 um for the AGw guider).  NOTE: the
   spreadsheet applies F_atm tan(ZD) only to the 6085 term (a precedence slip);
   here both terms are scaled, as in Filippenko.

3. Slit throughput for a Gaussian PSF displaced across the slit by d:
       T = 0.5 [erf((w/2 - d)/(sigma sqrt2)) - erf((-w/2 - d)/(sigma sqrt2))],  sigma sqrt2 = FWHM / (2 sqrt(ln 2))
   The across-slit component of dR is dR sin(PA_slit - PA_parallactic).

MODS has no ADC, so this is the tool for deciding a long-slit PA and how long a
visit can run before the blue end walks out of the slit.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any

DARK_SKY_V = 21.9          # V mag/arcsec^2, moonless zenith, Mt. Graham (Clear Sky Chart quotes 21.89-21.81)
K_EXT = 0.143              # mag per airmass, spreadsheet default


def moon_sky_brightness(moon_alt: float, target_alt: float, moon_sep: float, elongation: float, k: float = K_EXT) -> dict[str, Any]:
    """Moonlight at the target in nL and V mag/arcsec^2, plus the total sky including the dark-sky term."""
    if moon_alt <= 0 or target_alt <= 0:
        return {"moon_nL": 0.0, "moon_V": None, "sky_V": DARK_SKY_V, "note": "moon down" if moon_alt <= 0 else "target down"}
    alpha = 180.0 - elongation
    istar = 10 ** (-0.4 * (3.84 + 0.026 * abs(alpha) + 4e-9 * alpha ** 4))
    rho = max(moon_sep, 0.1)
    f_rho = 10 ** 5.36 * (1.06 + math.cos(math.radians(rho)) ** 2) + 10 ** (6.15 - rho / 40.0)
    x_moon = 1 / math.sin(math.radians(moon_alt))
    x_targ = 1 / math.sin(math.radians(target_alt))
    b_moon = f_rho * istar * 10 ** (-0.4 * k * x_moon) * (1 - 10 ** (-0.4 * k * x_targ))
    v_moon = 22.5 - 1.086 * math.log(max(b_moon, 1e-6) / 34.08)
    b_dark = 34.08 * math.exp(20.7233 - 0.92104 * DARK_SKY_V)
    v_total = 22.5 - 1.086 * math.log((b_moon + b_dark) / 34.08)
    return {"moon_nL": round(b_moon, 1), "moon_V": round(v_moon, 2), "sky_V": round(v_total, 2),
            "brightening_mag": round(DARK_SKY_V - v_total, 2)}


def f_atm(elev_m: float = 3269.0, temp_c: float = 2.0) -> float:
    return math.exp(-elev_m / 8400.0) * 288.0 / (273.0 + temp_c)


def differential_refraction(zd_deg: float, wav_um: float, guide_um: float = 0.65, elev_m: float = 3269.0, temp_c: float = 2.0) -> float:
    """Image displacement (arcsec) of wavelength wav_um relative to guide_um, positive toward the zenith? No:
    positive means the image at wav_um is displaced *away from* the guide image along the parallactic
    direction toward the horizon for bluer light (blue is refracted more, appears higher). Sign convention
    follows the spreadsheet: dR > 0 for wav < guide."""
    if zd_deg >= 89.0:
        return float("nan")
    a = 1 / (146 - wav_um ** -2) - 1 / (146 - guide_um ** -2)
    b = 1 / (41 - wav_um ** -2) - 1 / (41 - guide_um ** -2)
    return f_atm(elev_m, temp_c) * math.tan(math.radians(zd_deg)) * (6085.0 * a + 52.6 * b)


def slit_throughput(offset_arcsec: float, slit_width: float, seeing_fwhm: float) -> float:
    s2 = seeing_fwhm / (2 * math.sqrt(math.log(2)))        # sigma*sqrt(2)
    lo = (-slit_width / 2 - offset_arcsec) / s2
    hi = (slit_width / 2 - offset_arcsec) / s2
    return 0.5 * (math.erf(hi) - math.erf(lo))


def mods_slit_check(date: dt.date, ra: str, dec: str, slit_pa: float | None, slit_width: float = 0.8,
                    seeing_fwhm: float = 1.0, start_local: str | None = None, duration_h: float = 1.0,
                    wavelengths_um: tuple[float, ...] = (0.35, 0.40, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95),
                    guide_um: float = 0.65, step_min: int = 10) -> dict[str, Any]:
    """Track a MODS long-slit visit: for each time step give alt, ZD, parallactic angle, the across-slit
    displacement and slit throughput per wavelength. slit_pa=None means 'parallactic at mid-visit'."""
    from .ephem import fmt_local, night_info, target_track

    ni = night_info(date.isoformat())
    evh = ni["events_local_hours"]
    if start_local:
        hh, mm = start_local.split(":")
        t0 = int(hh) + int(mm) / 60
        t0 = t0 + 24 if t0 < 12 else t0
    else:
        t0 = evh["evening_18deg"]
    t1 = t0 + duration_h
    tr = target_track(date, ra, dec, step_min=step_min, start_hour=t0, end_hour=t1)
    rows = tr["rows"]
    mid = rows[len(rows) // 2]
    pa_used = mid["parang"] if slit_pa is None else slit_pa
    out_rows = []
    worst = {w: 1.0 for w in wavelengths_um}
    for r in rows:
        zd = 90 - r["alt"]
        if r["alt"] < 5:
            continue
        delta = pa_used - r["parang"]
        thr, across = {}, {}
        for w in wavelengths_um:
            dR = differential_refraction(zd, w, guide_um)
            d_across = dR * math.sin(math.radians(delta))
            across[w] = round(d_across, 2)
            thr[w] = round(slit_throughput(d_across, slit_width, seeing_fwhm), 3)
            worst[w] = min(worst[w], thr[w])
        out_rows.append({"local": r["local"], "utc": r["utc"], "alt": r["alt"], "zd": round(zd, 1), "ha_h": r["ha_h"],
                         "parang": r["parang"], "slit_minus_parang": round(((delta + 180) % 360) - 180, 1),
                         "across_slit_arcsec": across, "throughput": thr})
    centred = slit_throughput(0.0, slit_width, seeing_fwhm)
    return {"ra": ra, "dec": dec, "date_local": date.isoformat(), "slit_pa_used": round(pa_used, 1),
            "slit_pa_is_parallactic_at_midvisit": slit_pa is None, "parang_at_midvisit": mid["parang"],
            "slit_width": slit_width, "seeing_fwhm": seeing_fwhm, "guide_um": guide_um,
            "throughput_if_centred": round(centred, 3), "worst_throughput_by_wavelength": {str(k): v for k, v in worst.items()},
            "window_local": [fmt_local(date, t0), fmt_local(date, t1)], "rows": out_rows,
            "advice": _slit_advice(worst, centred, wavelengths_um)}


def _slit_advice(worst: dict, centred: float, wavs) -> str:
    blue = [w for w in wavs if w <= 0.45]
    if not blue:
        return ""
    wb = min(worst[w] for w in blue)
    if wb >= 0.85 * centred:
        return "Blue end stays within 15% of the centred throughput: this PA is fine for the whole visit."
    if wb >= 0.6 * centred:
        return "Blue end loses 15-40% at the worst step: acceptable for red-optimised work, re-acquire at the parallactic angle for blue-sensitive targets."
    return "Blue end loses more than 40% at the worst step: shorten the visit, split it with a re-acquisition at the parallactic angle, or move the PA closer to parallactic."


def format_slit_check(d: dict[str, Any]) -> str:
    wavs = list(d["worst_throughput_by_wavelength"].keys())
    L = [f"MODS slit check {d['ra']} {d['dec']} on {d['date_local']}, window {d['window_local'][0]}-{d['window_local'][1]} local",
         f"  slit PA {d['slit_pa_used']} deg ({'parallactic at mid-visit' if d['slit_pa_is_parallactic_at_midvisit'] else 'as given'}; parang at mid-visit {d['parang_at_midvisit']}), "
         f"slit {d['slit_width']}\", seeing {d['seeing_fwhm']}\", guiding at {d['guide_um']} um",
         f"  throughput if perfectly centred: {d['throughput_if_centred']}",
         "  worst throughput by wavelength (um): " + ", ".join(f"{w}: {v}" for w, v in d["worst_throughput_by_wavelength"].items()),
         "  " + d["advice"], "",
         "  local  UT     alt   ZD   HA    parang  slit-par | across-slit offset (\") / throughput at " + ", ".join(wavs) + " um"]
    for r in d["rows"]:
        L.append(f"  {r['local']}  {r['utc']}  {r['alt']:4.0f} {r['zd']:4.0f} {r['ha_h']:+5.1f} {r['parang']:+7.1f} {r['slit_minus_parang']:+7.1f} | "
                 + "  ".join(f"{r['across_slit_arcsec'][float(w)]:+.2f}/{r['throughput'][float(w)]:.2f}" for w in wavs))
    return "\n".join(L)
