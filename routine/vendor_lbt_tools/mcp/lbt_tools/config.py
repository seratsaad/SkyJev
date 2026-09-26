"""Site constants, paths and tunable defaults for the LBT lead-observer tools.

Everything an observer might want to tune lives in data/config.json (created on
first run from DEFAULTS below).  Values marked "estimate" are not published by
LBTO; treat them as starting points and correct them from experience.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../LBT_observation
DATA = ROOT / "data"
CACHE = DATA / "cache"
INPUTS = ROOT / "inputs"
LOCAL_READMES = INPUTS / "00Readme"
CONFIG_PATH = DATA / "config.json"
# Replay mode (used to re-run past nights offline, e.g. for a demo or a test):
#   LBT_STATE_DIR=<dir>   keep state.json / observed_log.json there instead of data/ (a clean state per run)
#   LBT_OFFLINE=1         never touch the network; queue/readmes/forecast come from data/cache only
#   LBT_REPLAY_NOW=<iso>  what "now" means (or write the same into <LBT_STATE_DIR>/clock.txt, which wins
#                         and can be changed while the server is running)
STATE_DIR = Path(os.environ["LBT_STATE_DIR"]) if os.environ.get("LBT_STATE_DIR") else DATA
STATE_PATH = STATE_DIR / "state.json"
OBSERVED_PATH = STATE_DIR / "observed_log.json"
OFFLINE = os.environ.get("LBT_OFFLINE", "") == "1"
SCHEDULE_TSV = DATA / "schedule_2026B.tsv"

# LBT / Mt. Graham
SITE = {
    "name": "LBT, Mt. Graham",
    "lat_deg": 32.7013,
    "lon_deg": -109.8891,
    "height_m": 3269,
    "tz_offset_hours": -7,          # MST all year (Arizona has no DST)
    "tz_name": "MST",
}

_TZ = _dt.timezone(_dt.timedelta(hours=-7), "MST")


def now_local() -> _dt.datetime:
    """The current local (MST) time, or the replay clock when one is set."""
    clock = STATE_DIR / "clock.txt"
    txt = clock.read_text().strip() if clock.exists() else os.environ.get("LBT_REPLAY_NOW", "")
    if txt:
        t = _dt.datetime.fromisoformat(txt)
        return t.astimezone(_TZ) if t.tzinfo else t.replace(tzinfo=_TZ)
    return _dt.datetime.now(_TZ)


def replay_forecast() -> str | None:
    """Scripted forecast text for a replayed night (<LBT_STATE_DIR>/forecast.txt), if any."""
    p = STATE_DIR / "forecast.txt"
    return p.read_text() if p.exists() else None


OSURC_BASE = "https://cgi.astronomy.osu.edu"
QUEUE_URL = OSURC_BASE + "/alt-cgi-bin/Queue/obsqueue.pl"
VISCALC_URL = OSURC_BASE + "/alt-cgi-bin/Tools/lbtVisCalc.pl"
OSURC_HOME = OSURC_BASE + "/lbtosurc/"

NWS_HOURLY_URL = "https://api.weather.gov/gridpoints/TWC/134,65/forecast/hourly"
NWS_POINTS_URL = "https://api.weather.gov/points/32.7018,-109.8708"
CSC_URL = "https://www.cleardarksky.com/txtc/MtGrahamAZcsp.txt"
USER_AGENT = "lbt-observation-planner (OSU LBT partner queue; contact saad.104@osu.edu)"

# Queue form vocabulary (values the CGI expects)
QUEUE_INSTRUMENTS = {
    "LBC": "LBC Cameras",
    "LUCI": "Lucifer All",
    "LUCI Imaging": "Lucifer Imaging",
    "LUCI Long Slit": "Lucifer Long Slit",
    "LUCI MOS": "Lucifer MOS",
    "MODS": "MODS All",
    "MODS Imaging": "MODS Imaging",
    "MODS Long Slit": "MODS Long Slit",
    "LMIRCAM": "LMIRCAM",
    "PEPSI": "PEPSI",
    "SHARK-Vis": "SHARK-Vis",
    "SHARK-NIR": "SHARK-NIR",
    "iLocater": "iLocater",
    "ALL": "All Instruments",
}

DEFAULTS = {
    "_comment": "Edit freely. Times in minutes unless stated. 'estimate' = not an LBTO-published number.",
    # Night definition used for planning
    "night": {
        "science_twilight_deg": -18.0,      # science window = between 18-deg twilights
        "bright_ok_twilight_deg": -12.0,    # bright-time PEPSI/LBC can start at 12-deg (estimate, common practice)
        "min_altitude_deg": 30.0,           # LBT adaptive secondary safe limit (hard)
        "max_altitude_deg": 85.0,           # zenith avoidance, advisory (tracking/rotator)
        "step_minutes": 10,
    },
    # Sky brightness classification (per hour). Fractional lunar illumination (FLI) thresholds.
    "sky": {
        "dark_if_moon_down": True,
        "dark_fli_max": 0.25,
        "gray_fli_max": 0.65,
        "moon_down_alt_deg": -3.0,
    },
    # Per-target overheads added to the queue "Time" (which is usually science + readout only)
    "overheads_min": {
        "PEPSI": 5, "MODS": 10, "LUCI": 12, "LUCI-AO": 25, "LBC": 8, "LBC-focus": 10,
        "SHARK-Vis": 20, "SHARK-NIR": 20, "iLocater": 15, "default": 10,
        "_note": "estimates from OSURC night logs: PEPSI preset+acq ~2-5 min, MODS acq ~10, LUCI ~10-15, AO close loop adds ~10-15",
    },
    # Cost of switching instruments mid-night (minutes). Symmetric matrix by family.
    "instrument_switch_min": {
        "_note": "estimates. LBC needs swing arms + mirror config; MODS/LUCI/PEPSI share the AGw/bent-Gregorian focal stations so switching is a reconfigure + pointing/collimation check. Tune from experience.",
        "same": 0,
        "LBC<->any": 40,
        "MODS<->LUCI": 20,
        "MODS<->PEPSI": 20,
        "LUCI<->PEPSI": 20,
        "any<->SHARK": 30,
        "any<->iLocater": 30,
        "default": 25,
    },
    # Scoring weights for plan_night
    "weights": {
        "time_critical_bonus": 5000.0,
        "dark_program_on_dark_night_bonus": 60.0,
        "gray_program_on_gray_or_dark_bonus": 20.0,
        "partner_behind_bonus_per_hour": 4.0,   # x hours behind (from queue time accounting)
        "transit_proximity_bonus": 15.0,        # max bonus at HA=0, linear to 0 at |HA|=4h
        "seeing_margin_bonus": 10.0,            # bonus when required seeing is comfortably above current
        "instrument_mismatch_penalty": 1e6,     # effectively excludes other instruments once chosen
        "short_target_filler_bonus": 5.0,
    },
    # Weather closure guidance (LBT Weather Guidelines; set from wiki if you have access)
    "weather_limits": {
        "_note": "LBTO operating guidelines (scienceops policies page, Sep 2026): close at sustained 20 m/s (45 mph) or gusts 22 m/s; humidity 95% closes, reopen below 90%; AdSec: 6 m/s kills diffraction-limited AO, 10 m/s gusts = move/close; mirror-ambient dT must be < 10 C to open; precipitation/ice closes.",
        "humidity_close_pct": 95,
        "humidity_caution_pct": 85,
        "wind_close_mph": 45,
        "wind_caution_mph": 25,
        "ao_wind_limit_mph": 13,
    },
    "seeing_default_arcsec": 1.0,
    # Programs that are calibrations, not science: scheduled as fillers, at most N per night
    "calibration_programs": ["MODSPhotCal", "LUCIPhotCal", "LBCPhotCal"],
    "calibration_score_penalty": 150.0,
    "max_calibration_targets_per_night": 1,
}


def load_config() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        try:
            user = json.loads(CONFIG_PATH.read_text())
        except Exception:
            user = {}
        cfg = json.loads(json.dumps(DEFAULTS))
        _deep_update(cfg, user)
        return cfg
    CONFIG_PATH.write_text(json.dumps(DEFAULTS, indent=2))
    return json.loads(json.dumps(DEFAULTS))


def _deep_update(base: dict, upd: dict) -> None:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def normalize_instrument(text: str) -> str:
    """Map a queue 'instrument/mode' string to a family: LBC, MODS, LUCI, PEPSI, SHARK-Vis, SHARK-NIR, iLocater, LBTI."""
    t = (text or "").upper()
    if "LBC" in t:
        return "LBC"
    if "MODS" in t:
        return "MODS"
    if "LUCI" in t or "LUCIFER" in t:
        return "LUCI"
    if "PEPSI" in t:
        return "PEPSI"
    if "SHARK" in t and "VIS" in t:
        return "SHARK-Vis"
    if "SHARK" in t:
        return "SHARK-NIR"
    if "ILOC" in t:
        return "iLocater"
    if "LMIR" in t or "LBTI" in t:
        return "LBTI"
    return text.strip() or "UNKNOWN"


def switch_cost_min(a: str, b: str, cfg: dict) -> float:
    if not a or not b or a == b:
        return 0.0
    m = cfg["instrument_switch_min"]
    fam = {a, b}
    if "LBC" in fam:
        return float(m.get("LBC<->any", m["default"]))
    if any("SHARK" in x for x in fam):
        return float(m.get("any<->SHARK", m["default"]))
    if "iLocater" in fam:
        return float(m.get("any<->iLocater", m["default"]))
    key = "<->".join(sorted(fam))
    for k, v in m.items():
        if k.startswith("_") or k in ("same", "default"):
            continue
        if set(k.split("<->")) == fam:
            return float(v)
    return float(m["default"])
