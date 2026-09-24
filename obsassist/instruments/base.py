"""Instrument configurations: the numbers an exposure time calculator needs.

A configuration is one instrument in one setup (grating/grism/filter + slit + binning +
readout speed). Throughput is the total of telescope + instrument + detector (the atmosphere
is applied separately by the ETC), given as (wavelength [Å], efficiency) knots.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class InstrumentConfig:
    key: str  # registry key, e.g. "LRIS-R/400-8500/1.0"
    instrument: str  # e.g. "LRIS"
    telescopes: Tuple[str, ...]  # telescope keys that carry it
    mode: str  # "spec" | "img"
    description: str
    lam_min: float
    lam_max: float
    lam_ref: float  # default wavelength for S/N
    throughput: Tuple[Tuple[float, float], ...]
    pixscale: float  # arcsec / unbinned pixel (spatial)
    read_noise: float  # e- per (binned) read
    dark_e_per_hr: float  # e- / unbinned pixel / hour
    gain: float  # e- / ADU
    full_well: float  # e-
    readout_s: float  # at this binning and readout speed
    dispersion: float = 0.0  # Å / unbinned pixel (spec)
    resolution: float = 0.0  # R for this slit (informational)
    slit_width: float = 1.0  # arcsec (spec)
    slit_length: float = 20.0  # arcsec (spec)
    band: str = ""  # imaging filter band name
    band_width: float = 0.0  # imaging: effective width, Å
    bin_spatial: int = 1
    bin_spectral: int = 1
    max_exp_s: float = 1800.0  # longest sensible single exposure
    min_exp_s: float = 1.0
    acq_s: float = 420.0  # acquisition after a slew (blind offset / slit alignment)
    config_change_s: float = 120.0  # changing into this configuration
    adc: bool = False  # atmospheric dispersion corrector in the beam
    nir: bool = False
    sky_line_factor: float = 1.0  # NIR spec: sky between OH lines relative to band average
    readout_speeds: Dict[str, float] = field(default_factory=dict)  # name -> readout seconds
    setup: str = ""  # configs sharing a setup need no reconfiguration between them
    confidence: str = "approximate"  # "documented" | "approximate" | "estimate"
    sources: Tuple[str, ...] = ()

    def eff(self, lam) -> np.ndarray:
        x = np.array([p[0] for p in self.throughput])
        y = np.array([p[1] for p in self.throughput])
        lam = np.asarray(lam, dtype=float)
        out = np.interp(lam, x, y, left=0.0, right=0.0)
        return out

    @property
    def setup_id(self) -> str:
        return self.setup or self.key

    @property
    def spatial_scale(self) -> float:
        """arcsec per binned spatial pixel."""
        return self.pixscale * self.bin_spatial

    @property
    def dlam_pix(self) -> float:
        """Å per binned spectral pixel at lam_ref."""
        return self.dispersion * self.bin_spectral

    @property
    def echelle(self) -> bool:
        return self.mode == "spec" and self.resolution >= 15000

    def dlam_at(self, lam) -> np.ndarray:
        """Å per binned spectral pixel at lam: constant for gratings and grisms; proportional to
        lam for an echelle (every order spans the same velocity per pixel)."""
        lam = np.asarray(lam, dtype=float)
        return self.dlam_pix * (lam / self.lam_ref if self.echelle else np.ones_like(lam))

    def with_(self, **kw) -> "InstrumentConfig":
        return replace(self, **kw)


REGISTRY: Dict[str, InstrumentConfig] = {}


def register(cfg: InstrumentConfig) -> InstrumentConfig:
    REGISTRY[cfg.key] = cfg
    return cfg


def get_config(key: str) -> InstrumentConfig:
    if not REGISTRY:
        _load_builtin()
    if key not in REGISTRY:
        raise KeyError(f"unknown instrument configuration {key!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[key]


def configs_for(telescope_key: str) -> Dict[str, InstrumentConfig]:
    if not REGISTRY:
        _load_builtin()
    return {k: c for k, c in REGISTRY.items() if telescope_key in c.telescopes}


def _load_builtin() -> None:
    from obsassist.instruments import keck, magellan  # noqa: F401  (registers on import)


def siblings(cfg: InstrumentConfig) -> list:
    """Configurations exposed together with `cfg` (same setup, e.g. the blue and red arms of MIKE)."""
    if not REGISTRY:
        _load_builtin()
    return [c for c in REGISTRY.values() if c.setup_id == cfg.setup_id and set(c.telescopes) & set(cfg.telescopes)]


def all_configs() -> Dict[str, InstrumentConfig]:
    if not REGISTRY:
        _load_builtin()
    return dict(REGISTRY)
