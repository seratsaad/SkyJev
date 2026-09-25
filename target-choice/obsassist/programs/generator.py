"""Random, oversubscribed observing programs for training and evaluating decision policies.

A program is drawn for a telescope and a date: targets spread over the night's sidereal
range at declinations the site can reach, with magnitudes chosen so each needs between ~15
minutes and ~3 hours, priorities, and a mix of the complications observers actually face:
seeing-critical targets, time windows, fixed PAs, standards, backups and a ToO that arrives
mid-night. `oversubscription` sets requested time / available time.
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Optional

import numpy as np

from obsassist.astro.ephem import NightEphem, fmt_dec, fmt_ra
from obsassist.astro.sites import SITES, TELESCOPES
from obsassist.etc import Source, compute_rates
from obsassist.instruments.base import configs_for, get_config
from obsassist.targets import Program, load_program

TEMPLATES = {
    "spec_lowres": ["sn_ia", "sn_ii", "galaxy", "qso", "hot_star", "hii", "brown_dwarf", "m_dwarf"],
    "spec_echelle": ["galaxy", "hot_star", "m_dwarf", "white_dwarf"],
}
# continuum slope per template (the template itself only adds localised structure)
TEMPLATE_SED = {
    "hot_star": "bb:20000",
    "white_dwarf": "bb:12000",
    "galaxy": "bb:4700",
    "m_dwarf": "bb:3200",
    "brown_dwarf": "bb:1500",
}
DEFAULT_CONFIGS = {
    "keck1": ["LRIS-B600", "LRIS-R400"],
    "keck2": ["DEIMOS-600ZD"],
    "clay": ["MIKE-BLUE", "MIKE-RED"],
    "baade": ["IMACS-f2-300"],
}


def _need_seconds(cfg, tel, site, src, goal, unit, lam):
    r = compute_rates(cfg, tel, site, src, seeing_500=site.seeing_median, airmass=1.2, lam=lam, unit=unit)
    t = float(cfg.max_exp_s)
    s1 = float(r.snr(t))
    return (goal / max(s1, 1e-6)) ** 2 * (t + cfg.readout_s)


def generate_program(
    telescope: str,
    date: str,
    seed: int = 0,
    n_targets: Optional[int] = None,
    oversubscription: Optional[float] = None,
    configs: Optional[List[str]] = None,
) -> Program:
    rng = np.random.default_rng(seed)
    tel = TELESCOPES[telescope]
    site = SITES[tel.site_key]
    eph = NightEphem(site, date, step_min=10.0)
    tw = eph.twilight
    night_h = (tw.nautical_dawn - tw.nautical_dusk) * 24.0
    i0, i1 = eph.index_of(tw.nautical_dusk), eph.index_of(tw.nautical_dawn)
    lst0, lst1 = eph.lst[i0], eph.lst[i1]
    span = (lst1 - lst0) % 360.0
    configs = configs or DEFAULT_CONFIGS.get(telescope) or sorted(configs_for(telescope))
    n = n_targets or int(rng.integers(8, 15))
    over = oversubscription or float(rng.uniform(1.1, 1.9))
    budget_s = over * night_h * 3600.0
    # time shares (Dirichlet) -> per-target requested time
    shares = rng.dirichlet(np.full(n, 1.5))
    targets: List[Dict] = []
    dec_lo, dec_hi = (
        (site.lat_deg - 55, min(site.lat_deg + 50, 89))
        if site.lat_deg > 0
        else (max(site.lat_deg - 50, -89), site.lat_deg + 55)
    )
    t_start_utc = eph.utc(eph.minute_of(tw.nautical_dusk))
    for k in range(n):
        cfg = get_config(configs[int(rng.integers(len(configs)))])
        echelle = cfg.echelle
        # RA: transit somewhere in (or slightly outside) the night
        ra = float((lst0 + rng.uniform(-0.15, 1.15) * span) % 360.0)
        dec = float(rng.uniform(dec_lo, dec_hi))
        lam = float(np.clip(cfg.lam_ref + rng.normal(0, 300), cfg.lam_min + 200, cfg.lam_max - 200))
        unit = "per_pix" if echelle else ("per_A" if rng.random() < 0.8 else "per_pix")
        band = "V" if echelle else str(rng.choice(["g", "r", "i"]))
        system = "Vega" if echelle else "AB"
        want_s = float(np.clip(shares[k] * budget_s, 900.0, 4.5 * 3600.0))
        goal = float(rng.choice([10, 15, 20, 30, 50, 80] if echelle else [5, 8, 10, 15, 20, 30]))
        pri = int(rng.choice([1, 2, 3, 4], p=[0.25, 0.4, 0.25, 0.1]))
        t_exp = float(min(cfg.max_exp_s, rng.choice([600, 900, 1200, 1800])))
        template = str(rng.choice(TEMPLATES["spec_echelle" if echelle else "spec_lowres"]))
        sed = TEMPLATE_SED.get(template, "flat_fnu")
        # solve for the magnitude that makes the goal take ~want_s at median seeing
        lo, hi = 8.0, 25.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if _need_seconds(cfg, tel, site, Source(mid, band, system, sed=sed), goal, unit, lam) < want_s:
                lo = mid
            else:
                hi = mid
        mag = round(0.5 * (lo + hi), 2)
        d = {
            "name": f"T{k + 1:02d}-{fmt_ra(ra)[:5].replace(':', '')}{'+' if dec >= 0 else '-'}{abs(int(dec)):02d}",
            "ra": fmt_ra(ra),
            "dec": fmt_dec(dec),
            "config": cfg.key,
            "mag": mag,
            "band": band,
            "mag_system": system,
            "snr": goal,
            "snr_unit": unit,
            "lam": round(lam),
            "priority": pri,
            "t_exp": t_exp,
            "template": template,
            "sed": sed,
            "max_airmass": float(rng.choice([1.6, 1.8, 2.0, 2.2])),
            "min_moon_sep": float(rng.choice([15, 25, 35])),
        }
        if pri == 4:
            d["kind"] = "backup"
        if rng.random() < 0.25:
            d["max_seeing"] = float(rng.choice([0.7, 0.8, 0.9, 1.0]))
        if rng.random() < 0.12:
            w0 = t_start_utc + _dt.timedelta(hours=float(rng.uniform(0.5, night_h - 1.5)))
            d["window"] = [
                w0.isoformat().replace("+00:00", "Z"),
                (w0 + _dt.timedelta(minutes=float(rng.choice([40, 60, 90, 150])))).isoformat().replace("+00:00", "Z"),
            ]
        if rng.random() < 0.1:
            d["pa"] = float(rng.uniform(0, 180))
        targets.append(d)
    # one ToO in half the programs
    if rng.random() < 0.5:
        k = int(rng.integers(len(targets)))
        targets[k]["kind"] = "too"
        targets[k]["priority"] = 1
        targets[k]["appears_at"] = (
            (t_start_utc + _dt.timedelta(hours=float(rng.uniform(1.0, night_h - 2.5))))
            .isoformat()
            .replace("+00:00", "Z")
        )
        targets[k].pop("window", None)
    prog = {
        "name": f"random {telescope} {date} #{seed}",
        "telescope": telescope,
        "date": date,
        "description": f"generated: {n} targets, oversubscription {over:.2f}",
        "targets": targets,
        "extra": {"oversubscription": over, "generator_seed": seed},
    }
    return load_program(prog)


def random_date(rng: np.random.Generator, year: int = 2026) -> str:
    d0 = _dt.date(year, 1, 1) + _dt.timedelta(days=int(rng.integers(0, 365)))
    return d0.isoformat()
