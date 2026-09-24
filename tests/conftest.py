import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsassist.sim.night import NightModel  # noqa: E402
from obsassist.targets import builtin_programs, load_program  # noqa: E402


@pytest.fixture(scope="module")
def lris():
    """The Keck I/LRIS demo night (seed 6)."""
    return NightModel(load_program(builtin_programs()["keck1_lris_tonight"]), seed=6)


@pytest.fixture(scope="module")
def mike():
    """The Magellan Clay/MIKE demo night (seed 5)."""
    return NightModel(load_program(builtin_programs()["clay_mike_darktime"]), seed=5)
