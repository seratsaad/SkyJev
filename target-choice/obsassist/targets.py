"""Targets and observing programs (a night's list), loaded from YAML or JSON.

A target states what "done" means (an S/N goal in a unit at a wavelength with one
instrument configuration) and the constraints under which data count: airmass, Moon
distance, delivered image quality, transparency, a time window, a fixed slit PA.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from obsassist.astro.ephem import fmt_dec, fmt_ra, parse_dec, parse_ra
from obsassist.astro.sites import SITES, TELESCOPES, Site, Telescope
from obsassist.etc import Source
from obsassist.instruments.base import InstrumentConfig, get_config

PRIORITY_WEIGHT = {1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0}


@dataclass
class Target:
    name: str
    ra: float  # deg, ICRS/J2000
    dec: float
    config: str  # instrument configuration key
    source: Source
    snr_goal: float
    snr_unit: str = "per_A"
    lam: Optional[float] = None  # S/N wavelength (default: config lam_ref)
    priority: int = 2  # 1 highest ... 4 backup
    weight: Optional[float] = None  # overrides the priority weight
    t_exp: Optional[float] = None  # preferred single-exposure length (s)
    max_airmass: float = 2.0
    min_moon_sep: float = 25.0
    max_seeing: Optional[float] = None  # delivered FWHM at lam (arcsec) for data to count
    max_cloud: Optional[float] = None  # cloud extinction (mag) for data to count
    window: Optional[Tuple[str, str]] = None  # UTC ISO start, end: only data inside count
    pa: Optional[float] = None  # fixed slit/mask PA (deg); None = parallactic
    kind: str = "science"  # science | standard | backup | too
    appears_at: Optional[str] = None  # ToO: UTC ISO time the alert arrives
    twilight_ok: bool = False  # may be observed between 6 and 12 deg twilight
    template: str = "flat"  # spectral template for synthetic frames
    z: float = 0.0  # redshift for the template
    program: str = ""
    pi: str = ""
    notes: str = ""

    @property
    def w(self) -> float:
        return float(self.weight if self.weight is not None else PRIORITY_WEIGHT.get(self.priority, 1.0))

    def cfg(self) -> InstrumentConfig:
        return get_config(self.config)

    def lam_ref(self) -> float:
        return float(self.lam if self.lam is not None else self.cfg().lam_ref)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ra_hms"] = fmt_ra(self.ra)
        d["dec_dms"] = fmt_dec(self.dec)
        d["source"] = asdict(self.source)
        d["weight_eff"] = self.w
        return d


@dataclass
class Program:
    name: str
    telescope: Telescope
    date: str  # local date of the evening
    targets: List[Target]
    description: str = ""
    night_start: str = "12deg"  # "sunset" | "6deg" | "12deg" | "18deg"
    night_end: str = "12deg"
    partial_credit: float = 0.3
    observer_notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def site(self) -> Site:
        return SITES[self.telescope.site_key]

    def target_index(self, name: str) -> int:
        for i, t in enumerate(self.targets):
            if t.name == name:
                return i
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "telescope": self.telescope.key,
            "date": self.date,
            "description": self.description,
            "night_start": self.night_start,
            "night_end": self.night_end,
            "partial_credit": self.partial_credit,
            "observer_notes": self.observer_notes,
            "targets": [t.to_dict() for t in self.targets],
        }


def target_from_dict(d: Dict[str, Any]) -> Target:
    src = Source(
        mag=float(d["mag"]),
        band=d.get("band", "r"),
        system=d.get("mag_system", "AB"),
        kind=d.get("source_kind", "point"),
        sed=d.get("sed", "flat_fnu"),
        extent_arcsec=float(d.get("extent_arcsec", 1.0)),
    )
    win = d.get("window")
    return Target(
        name=str(d["name"]),
        ra=parse_ra(d["ra"]),
        dec=parse_dec(d["dec"]),
        config=str(d["config"]),
        source=src,
        snr_goal=float(d["snr"]),
        snr_unit=d.get("snr_unit", "per_A"),
        lam=d.get("lam"),
        priority=int(d.get("priority", 2)),
        weight=d.get("weight"),
        t_exp=d.get("t_exp"),
        max_airmass=float(d.get("max_airmass", 2.0)),
        min_moon_sep=float(d.get("min_moon_sep", 25.0)),
        max_seeing=d.get("max_seeing"),
        max_cloud=d.get("max_cloud"),
        window=(str(win[0]), str(win[1])) if win else None,
        pa=d.get("pa"),
        kind=d.get("kind", "science"),
        appears_at=d.get("appears_at"),
        twilight_ok=bool(d.get("twilight_ok", False)),
        template=d.get("template", "flat"),
        z=float(d.get("z", 0.0)),
        program=d.get("program", ""),
        pi=d.get("pi", ""),
        notes=d.get("notes", ""),
    )


def load_program(path_or_dict) -> Program:
    if isinstance(path_or_dict, dict):
        d = path_or_dict
    else:
        p = Path(path_or_dict)
        text = p.read_text()
        if p.suffix in (".yaml", ".yml"):
            import yaml

            d = yaml.safe_load(text)
        else:
            d = json.loads(text)
    tel = TELESCOPES[d["telescope"]]
    targets = [target_from_dict(t) for t in d["targets"]]
    for t in targets:
        cfg = get_config(t.config)
        if tel.key not in cfg.telescopes:
            raise ValueError(f"target {t.name}: configuration {t.config} is not on {tel.name}")
        lam = t.lam_ref()
        if not (cfg.lam_min <= lam <= cfg.lam_max):
            raise ValueError(f"target {t.name}: S/N wavelength {lam} outside {cfg.key} coverage")
    names = [t.name for t in targets]
    if len(set(names)) != len(names):
        raise ValueError("target names must be unique")
    return Program(
        name=d.get("name", "program"),
        telescope=tel,
        date=str(d["date"]),
        targets=targets,
        description=d.get("description", ""),
        night_start=d.get("night_start", "12deg"),
        night_end=d.get("night_end", "12deg"),
        partial_credit=float(d.get("partial_credit", 0.3)),
        observer_notes=d.get("observer_notes", ""),
        extra=d.get("extra", {}),
    )


def builtin_programs() -> Dict[str, Path]:
    root = Path(__file__).resolve().parent / "programs"
    return {p.stem: p for p in sorted(root.glob("*.yaml"))}
