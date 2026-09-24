"""Where everything the project generates lives: inside the project directory.

    data/frames/<telescope>/utYYYYMMDD/   simulated FITS frames written by the console
    data/decisions/                       hindsight-labelled decision datasets (regenerable)
    data/traces/                          the assistant's audit trail, one JSON-lines file per run
    heads/                                fitted AnyJev heads (small JSON, one file per model)
    reports/                              benchmark results

`OBSASSIST_HOME` overrides the root (e.g. a fast scratch disk on a cluster).
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("OBSASSIST_HOME")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parent.parent  # the checkout that contains the package
    if (here / "pyproject.toml").exists():
        return here
    return Path.cwd()  # an installed package: the working directory


ROOT = project_root()
DATA = ROOT / "data"
FRAMES = DATA / "frames"
DECISIONS = DATA / "decisions"
TRACES = DATA / "traces"
HEADS = ROOT / "heads"
REPORTS = ROOT / "reports"
