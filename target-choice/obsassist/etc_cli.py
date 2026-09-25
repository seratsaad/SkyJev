"""`python -m obsassist etc`: the exposure time calculator from the command line.

    python -m obsassist etc --config MIKE-RED --mag 16 --band V --system Vega --seeing 0.7 --airmass 1.2 \
        --goal 50 --unit per_pix --t 1800 [--moon-illum 0.5 --moon-sep 60 --moon-alt 40] [--cloud 0.2]
"""

from __future__ import annotations

import argparse

import numpy as np

from obsassist.astro.sites import SITES, TELESCOPES
from obsassist.etc import Source, compute_rates, plan_exposures, saturation_factor
from obsassist.instruments.base import all_configs, get_config


def main(argv=None):
    ap = argparse.ArgumentParser(description="exposure time calculator")
    ap.add_argument("--config", help="instrument configuration (--list to see all)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--telescope", default=None, help="default: the first telescope carrying the configuration")
    ap.add_argument("--mag", type=float, default=18.0)
    ap.add_argument("--band", default="r")
    ap.add_argument("--system", default="AB", choices=["AB", "Vega"])
    ap.add_argument("--extended", action="store_true", help="mag is a surface brightness (mag/arcsec^2)")
    ap.add_argument("--seeing", type=float, default=0.7, help="DIMM seeing at 500 nm, zenith (arcsec)")
    ap.add_argument("--airmass", type=float, default=1.2)
    ap.add_argument("--cloud", type=float, default=0.0, help="grey extinction (mag)")
    ap.add_argument("--moon-illum", type=float, default=0.0, help="illuminated fraction 0-1")
    ap.add_argument("--moon-sep", type=float, default=90.0)
    ap.add_argument("--moon-alt", type=float, default=-30.0)
    ap.add_argument("--lam", type=float, default=None, help="S/N wavelength (A)")
    ap.add_argument("--unit", default="per_A", choices=["per_A", "per_pix", "per_resel", "aperture"])
    ap.add_argument("--t", type=float, default=1200.0, help="single exposure (s)")
    ap.add_argument("--goal", type=float, default=None, help="S/N goal: prints the exposure plan")
    a = ap.parse_args(argv)
    if a.list or not a.config:
        for k, c in sorted(all_configs().items()):
            print(f"{k:16s} {'/'.join(c.telescopes):8s} {c.description}")
        return 0
    cfg = get_config(a.config)
    tel = TELESCOPES[a.telescope or cfg.telescopes[0]]
    site = SITES[tel.site_key]
    phase = float(np.degrees(np.arccos(np.clip(2 * a.moon_illum - 1, -1, 1))))  # illum -> phase angle
    src = Source(a.mag, a.band, a.system, "extended" if a.extended else "point")
    r = compute_rates(
        cfg,
        tel,
        site,
        src,
        seeing_500=a.seeing,
        airmass=a.airmass,
        cloud_mag=a.cloud,
        moon_alt=a.moon_alt if a.moon_illum > 0 else -30.0,
        moon_phase_angle=phase,
        moon_sep=a.moon_sep,
        lam=a.lam,
        unit=a.unit,
    )
    sat = saturation_factor(cfg, tel, site, src, seeing=a.seeing, airmass=a.airmass)  # any arm, any wavelength
    print(f"{cfg.key} on {tel.name}: {cfg.description}")
    print(
        f'  lambda {r.lam:.0f} A | delivered FWHM {float(r.fwhm):.2f}" | slit transmission {float(r.slit_frac):.2f} | '
        f"sky {float(r.sky_mag):.2f} AB/arcsec^2 | {r.regime()}-limited"
    )
    print(
        f"  source {float(r.signal):.3g} e-/s, sky {float(r.sky_pix):.3g} e-/s/pix, saturates in "
        f"{float(r.t_saturate(cfg.full_well)) / sat:.0f} s (brightest pixel of any arm)"
    )
    for t in (60, 300, 600, 1200, 1800, 3600):
        print(f"  t={t:5d} s  S/N {float(r.snr(t)):8.2f} ({a.unit})")
    if a.goal:
        p = plan_exposures(r, cfg, a.goal, 0.0, a.t, sat)
        print(
            f"  goal {a.goal}: {p.n_exp} x {p.t_exp:.0f} s -> S/N {p.snr_final:.1f}, "
            f"{p.wall_s / 60:.1f} min with readout"
            f" (limited by {p.limited_by})"
        )
    return 0


if __name__ == "__main__":
    main()
