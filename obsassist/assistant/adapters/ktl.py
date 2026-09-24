"""Keck KTL helpers: read-only telemetry and the command lines an observer types (never executed).

Keck exposes telescope and instrument state as KTL keywords through the `show` / `modify` /
`waitfor` command-line tools on the observing workstations (documented at
www2.keck.hawaii.edu; see docs/research/keck_parameters.json). This adapter:

  * reads telemetry with `show -s dcs -terse <kw>` (AIRMASS, AZ, EL, RA, DEC, HA, PARANG,
    TARGNAME, TRACKING, GUIDING) when the `show` binary exists on this machine;
  * turns the assistant's recommendations into the command lines an LRIS/DEIMOS observer
    types (tintb/tintr/goibr, tint/goi, object, dcs offsets) and returns them as TEXT.

It never executes `modify`. Actuation at Keck belongs to the observer and the OA; executing
commands would need Keck's approval and testing on their systems, which this project has not
had. `run_show` only runs read-only `show` commands. Used by the manual adapter
(`obsassist assistant --manual ... --keck-commands LRIS`).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Dict, List

DCS_KEYWORDS = ("AIRMASS", "AZ", "EL", "RA", "DEC", "HA", "PARANG", "TARGNAME", "TRACKING", "GUIDING", "UT", "LST")


def run_show(service: str, keywords: List[str], timeout: float = 5.0) -> Dict[str, str]:
    if not shutil.which("show"):
        raise FileNotFoundError("KTL 'show' is not available on this machine (not a Keck workstation)")
    out = subprocess.run(["show", "-s", service, "-terse", *keywords], capture_output=True, text=True, timeout=timeout)
    vals = out.stdout.split()
    return dict(zip(keywords, vals))


def lris_commands(target: str, t_blue: float, t_red: float, n: int = 1) -> List[str]:
    return [f'object "{target}"', f"tintb {int(round(t_blue))}", f"tintr {int(round(t_red))}", f"goibr {int(n)}"]


def deimos_commands(target: str, t_exp: float, n: int = 1) -> List[str]:
    return [f'object "{target}"', f"tint {int(round(t_exp))}", f"goi {int(n)}"]


def keck_instructions(instrument: str, target: str, t_exp: float, n: int = 1) -> List[str]:
    """What to type at a Keck instrument terminal for one exposure sequence (never executed here)."""
    inst = instrument.upper()
    if inst == "LRIS":
        return lris_commands(target, t_exp, t_exp, n)
    if inst == "DEIMOS":
        return deimos_commands(target, t_exp, n)
    return [f"# no command template for {instrument}: set {target} {n} x {t_exp:.0f} s by hand"]
