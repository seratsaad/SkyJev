#!/usr/bin/env python3
"""Build the private case files for the LBT Jev replay.

    source /fs/scratch/PAS2823/saadsm/SkyJev/pitzer_env.sh
    python lbt_replay/build_cases.py --private /fs/scratch/PAS2823/saadsm/private_lbt

Reads, all inside the private directory (chmod 700):
  raw_transcripts/*.jsonl   Claude Code session transcripts of the LBT observing nights
  src/                      copy of LBT_observation (data/cache queue pages and readmes, plans, mcp/lbt_tools)
  ann_logging.json, ann_weather.json, ann_next.json, ann_slit.json
                            hand-transcribed references; each names the transcript record it was read from
Writes, inside the private directory only:
  cases/<task>.jsonl        one JSON object per case: state text, question, options, reference,
                            code-baseline pick and time, the Claude wall time of the matching request
  cases/claude_times.json   per-category wall times of the Claude requests (aggregate numbers only)

This script holds no names, targets or transcript text; everything specific lives in the private
annotation files. Nothing here touches the network (lbt_tools runs with LBT_OFFLINE=1).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "note"))
import lbt_latency as LL  # noqa: E402  (request segmentation and classification, reused as is)

MST = dt.timedelta(hours=-7)
YEAR = 2026


# ------------------------------------------------------------------ transcripts

def local(ts: str) -> dt.datetime:
    return (LL.parse_ts(ts) + MST).replace(tzinfo=None)


def key(t: dt.datetime) -> str:
    return t.strftime("%m-%d %H:%M")


def parse_key(k: str) -> dt.datetime:
    return dt.datetime.strptime(f"{YEAR}-{k}", "%Y-%m-%d %H:%M")


def _result_text(b: dict) -> str:
    c = b.get("content")
    if isinstance(c, str):
        return c
    return " ".join(x.get("text", "") for x in (c or []) if isinstance(x, dict))


def load_requests(tdir: Path) -> list[dict]:
    """Requests as lbt_latency defines them (human prompt -> last assistant record before the next
    prompt), each with its tool calls (name, input, local time) and tool results (text, local time)."""
    reqs = []
    for f in sorted(glob.glob(str(tdir / "*.jsonl"))):
        cur = None
        pending: dict[str, dict] = {}
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = r.get("timestamp")
            text = LL.human_text(r)
            if text is not None and ts:
                cur = {"t0": local(ts), "t_end": None, "text": text, "tools": [], "calls": [], "results": []}
                reqs.append(cur)
                pending = {}
                continue
            if cur is None or not ts:
                continue
            msg = r.get("message") or {}
            if r.get("type") == "assistant":
                cur["t_end"] = local(ts)
                for b in msg.get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        call = {"t": local(ts), "name": LL.short_tool(b.get("name", "?")), "input": b.get("input") or {},
                                "result": None}
                        cur["tools"].append(b.get("name", "?"))
                        cur["calls"].append(call)
                        pending[b.get("id")] = call
            elif r.get("type") == "user" and isinstance(msg.get("content"), list):
                for b in msg["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        txt = _result_text(b)
                        cur["results"].append({"t": local(ts), "text": txt})
                        if b.get("tool_use_id") in pending:
                            pending.pop(b["tool_use_id"])["result"] = txt
    for r in reqs:
        r["wall"] = (r["t_end"] - r["t0"]).total_seconds() if r["t_end"] else None
        r["category"] = LL.classify(r["text"], r["tools"])
    return reqs


def find_request(reqs: list[dict], k: str, nth: int = 0) -> dict:
    hits = [r for r in reqs if key(r["t0"]) == k and r["t_end"] is not None]  # a prompt with no reply is not a request
    if len(hits) <= nth:
        raise KeyError(f"no request at {k}")
    return hits[nth]


def request_at(reqs: list[dict], t: dt.datetime) -> dict | None:
    """The request whose span contains local time t."""
    for r in reqs:
        if r["t_end"] and r["t0"] <= t <= r["t_end"] + dt.timedelta(seconds=59):
            return r
    return None


def all_calls(reqs: list[dict], name: str) -> list[dict]:
    return sorted((c for r in reqs for c in r["calls"] if c["name"] == name), key=lambda c: c["t"])


def night_of(t: dt.datetime) -> str:
    return (t - dt.timedelta(hours=12)).date().isoformat()


def hour_of(night: str, hhmm: str) -> float:
    """'01:15' on night D -> 25.25 (hours after midnight starting the evening date)."""
    hh, mm = hhmm.split(":")
    h = int(hh) + int(mm) / 60
    return h + 24 if h < 12 else h


def dt_of(night: str, hhmm: str) -> dt.datetime:
    d = dt.date.fromisoformat(night)
    return dt.datetime.combine(d, dt.time()) + dt.timedelta(hours=hour_of(night, hhmm))


# ------------------------------------------------------------------ lbt_tools in offline replay

class Tools:
    """lbt_tools imported in replay mode: LBT_OFFLINE=1 and a private scratch state directory, so the
    planner reads a per-case state.json / observed_log.json written by this script."""

    def __init__(self, private: Path):
        self.state_dir = private / "work" / "replay_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        os.environ["LBT_OFFLINE"] = "1"
        os.environ["LBT_STATE_DIR"] = str(self.state_dir)
        sys.modules.setdefault("requests", types.ModuleType("requests"))  # offline: never used
        from astropy.utils import iers
        iers.conf.auto_download = False
        sys.path.insert(0, str(private / "src" / "mcp"))
        from lbt_tools import config, ephem, optics, planner, readmes, state
        self.config, self.ephem, self.optics, self.planner, self.readmes, self.state = config, ephem, optics, planner, readmes, state
        self.cache = private / "src" / "data" / "cache"

    def queue(self, name: str) -> dict:
        return json.loads((self.cache / name).read_text())

    def all_objects(self) -> dict[str, dict]:
        objs = {}
        for f in sorted(self.cache.glob("queue_*.json")):
            for t in json.loads(f.read_text())["targets"]:
                objs.setdefault(t["object"], t)
        return objs

    def write_state(self, seeing: float, instruments: dict[str, tuple[str, str]], exclude_programs: list[str],
                    exclude_targets: list[str], done: list[str], night: str):
        s = self.state._empty_state()
        s["conditions"]["seeing_arcsec"] = seeing
        for name, (st, note) in instruments.items():
            if name in s["instruments"]:
                s["instruments"][name] = {"status": st, "note": note, "updated": None}
        s["plan_overrides"]["exclude_programs"] = list(exclude_programs)
        s["plan_overrides"]["exclude_targets"] = list(exclude_targets)
        (self.state_dir / "state.json").write_text(json.dumps(s))
        obs = [{"program": o.split("/")[0], "target": o.split("/", 1)[1], "date_local": night, "status": "done"} for o in done]
        (self.state_dir / "observed_log.json").write_text(json.dumps(obs))


def state_at(reqs: list[dict], t: dt.datetime) -> tuple[dict, list[str], list[str]]:
    """Instrument statuses and exclusions as the assistant had recorded them by local time t."""
    inst: dict[str, tuple[str, str]] = {}
    for c in all_calls(reqs, "report_instrument"):
        if c["t"] <= t:
            inst[c["input"].get("instrument", "")] = (c["input"].get("status", "unknown"), c["input"].get("note", ""))
    progs: list[str] = []
    targs: list[str] = []
    for c in all_calls(reqs, "exclude_targets"):
        if c["t"] <= t:
            if c["input"].get("clear"):
                progs, targs = [], []
            progs += c["input"].get("programs") or []
            targs += c["input"].get("targets") or []
    return inst, progs, targs


# ------------------------------------------------------------------ task 1: logging

STATUS = ["done", "partial", "skipped"]
STATUS_OPTIONS = ["done: observed and finished", "partial: started, not finished", "skipped: not observed"]
RE_SKIP = re.compile(r"not (be )?(observed|taken|done)|skip\w*|\blost\b|remov\w*|no observing|cancel\w*|abandon\w*", re.I)
RE_DONE = re.compile(r"\bdone\b|complet\w*|finish\w*", re.I)
RE_PART = re.compile(r"partial\w*|stopped|abort\w*|incomplete", re.I)


def mentions(text: str, objects) -> list[str]:
    """Queue objects named in text: the bare target name (4+ characters), or the program name when
    that program has a single target in the queue."""
    per_prog = defaultdict(list)
    for o in objects:
        per_prog[o.split("/")[0]].append(o)
    found = []
    for o in objects:
        prog, targ = o.split("/", 1)
        hit = len(targ) >= 4 and re.search(r"(?<![\w-])" + re.escape(targ) + r"(?![\w-])", text)
        if not hit and len(per_prog[prog]) == 1:
            hit = re.search(r"(?<![\w-])" + re.escape(prog) + r"(?![\w-])", text)
        if hit:
            found.append(o)
    return found


def code_status(text: str, obj: str) -> str:
    """Keyword rule on the rest of each line that names the target (the whole message if it names
    none): skip words first, then done words, then partial words; nothing said counts as skipped."""
    prog, targ = obj.split("/", 1)
    spots = [m.start() for m in re.finditer(re.escape(targ), text)] or [m.start() for m in re.finditer(re.escape(prog), text)]
    wins = [text[s: (text.find("\n", s) if text.find("\n", s) > 0 else len(text))] for s in spots] or [text]
    for w in wins:
        if RE_SKIP.search(w):
            return "skipped"
    for w in wins:
        if RE_DONE.search(w):
            return "done"
    for w in wins:
        if RE_PART.search(w):
            return "partial"
    return "skipped"


def build_logging(reqs, ann, tools, max_chars: int = 3500) -> list[dict]:
    objects = tools.all_objects()
    out = []
    for lg in ann["logs"]:
        src = lg["source"]
        t = parse_key(src["time"])
        text = None
        for r in reqs:
            if src["kind"] == "prompt" and key(r["t0"]) == src["time"] and src["contains"] in r["text"]:
                text, req = r["text"], r
            if src["kind"] == "tool_result":
                for res in r["results"]:
                    if key(res["t"]) == src["time"] and src["contains"] in res["text"]:
                        text, req = res["text"], r
        if text is None:
            raise KeyError(f"log source not found for {lg['id']}")
        text = text.replace("\\n", "\n")
        if src.get("split"):
            text = next(ch for ch in text.split(src["split"]) if src["contains"] in ch)
        text = text[text.find(src["contains"]):][:max_chars]
        for obj in mentions(text, objects):
            t0 = time.perf_counter()
            pick = code_status(text, obj)
            code_s = time.perf_counter() - t0
            ref = lg["refs"].get(obj, lg["default_ref"])
            out.append({"task": "logging", "question_name": "status", "id": f"{lg['id']}:{obj}", "night": lg["night"],
                        "state": f"Observing night of {lg['night']} (local evening date). Night log from the observer:\n{text}",
                        "question": f"According to this message, what happened to the queue target {obj} that night?",
                        "options": STATUS_OPTIONS, "ref": [STATUS.index(ref)], "code_pick": STATUS.index(pick),
                        "code_seconds": code_s, "source": "observer night log", "request_wall_s": req["wall"],
                        "request_key": key(req["t0"]), "ref_record": lg["ref_record"]})
    for ch in ann["chat"]:
        req = find_request(reqs, ch["prompt_time"])
        msg = req["text"]
        for obj, ref in ch["refs"].items():
            t0 = time.perf_counter()
            pick = code_status(msg, obj)
            code_s = time.perf_counter() - t0
            out.append({"task": "logging", "question_name": "status", "id": f"{ch['id']}:{obj}", "night": ch["night"],
                        "state": (f"Observing night of {ch['night']} (local evening date), local time {key(req['t0'])[6:]} MST, "
                                  f"instrument in use {ch['instrument']}. Message from the observer:\n{msg}"),
                        "question": f"According to this message, what is the status of the queue target {obj} tonight?",
                        "options": STATUS_OPTIONS, "ref": [STATUS.index(ref)], "code_pick": STATUS.index(pick),
                        "code_seconds": code_s, "source": "observer chat message", "request_wall_s": req["wall"],
                        "request_key": key(req["t0"]), "ref_record": ch["ref_record"]})
    return out


# ------------------------------------------------------------------ task 2: conditions and weather

DOME = ["observe", "wait", "close"]
DOME_OPTIONS = ["observe: open (or stay open) and observe now",
                "wait: keep the dome closed (or close it) and reopen later tonight",
                "close: closed for the rest of the night (weather loss)"]
SKY = ["clear", "thin cirrus", "cloudy"]
COND_FIELDS = ("humidity_pct", "wind_mph", "seeing_arcsec", "transparency_note")
RE_DECISION = re.compile(r"\b(re)?open\w*|\bclos\w*|shutter\w*", re.I)


def forecast_text(raw: str) -> str:
    try:
        return json.loads(raw).get("result", raw)
    except (json.JSONDecodeError, AttributeError):
        return raw


def forecast_rows(fc: str, now: dt.datetime, hours_after: int = 7) -> tuple[list[str], list[dict]]:
    """Header lines and the hourly rows from now-1 h to now+hours_after of a `weather` tool result."""
    head, rows = [], []
    for line in fc.splitlines():
        m = re.match(r"\s*(\d\d-\d\d) (\d\d):00\s+(.*)$", line)
        if m:
            t = parse_key(f"{m.group(1)} {m.group(2)}:00")
            if now - dt.timedelta(hours=1) <= t <= now + dt.timedelta(hours=hours_after):
                rows.append({"t": t, "line": line.strip()})
        elif line.strip().startswith(("csc_", "nws_", "closure", "Forecast", "hour")):
            head.append(line.strip())
        elif line.strip().startswith("!") or "FLAG" in line.upper():
            head.append(line.strip())
    return head, rows


def row_numbers(line: str) -> dict:
    """RH, precipitation words and cloud percent from one forecast row."""
    parts = line.split("|")
    left = parts[0]
    right = parts[1] if len(parts) > 1 else ""
    rh = re.findall(r"\s(\d{1,3})\s*$", left.strip() + " ")
    nums = re.findall(r"(\d{1,3})\s*$", left.strip())
    cloud = re.search(r"(\d{1,3})% cloud", right)
    cl = int(cloud.group(1)) if cloud else (0 if "clear" in right.lower() else (100 if "overcast" in right.lower() else None))
    return {"rh": int(nums[0]) if nums else (int(rh[0]) if rh else None),
            "precip": bool(re.search(r"show|thunder|rain|storm|t-storm", left, re.I)), "cloud": cl}


def code_dome(cond: dict, rows: list[dict], now: dt.datetime, morning: dt.datetime) -> tuple[str, str]:
    """Rule baseline from the lbt_tools weather limits: bad now if RH >= 90 %, precipitation in the
    NWS row, or CSC cloud >= 70 %; then wait if any later hour before morning twilight has RH < 85 %,
    no precipitation and cloud <= 30 %, else close. Sky class from the CSC cloud of the current hour."""
    cur = min(rows, key=lambda r: abs((r["t"] - now).total_seconds()), default=None)
    n = row_numbers(cur["line"]) if cur else {"rh": None, "precip": False, "cloud": None}
    rh = cond.get("humidity_pct") if cond.get("humidity_pct") is not None else n["rh"]
    bad = (rh is not None and rh >= 90) or n["precip"] or (n["cloud"] is not None and n["cloud"] >= 70)
    sky = "clear" if (n["cloud"] or 0) <= 10 else ("thin cirrus" if n["cloud"] <= 40 else "cloudy")
    if not bad:
        return "observe", sky
    for r in rows:
        if now < r["t"] <= morning:
            m = row_numbers(r["line"])
            if (m["rh"] is None or m["rh"] < 85) and not m["precip"] and (m["cloud"] is None or m["cloud"] <= 30):
                return "wait", sky
    return "close", sky


def build_weather(reqs, ann, tools) -> list[dict]:
    out = []
    setc = all_calls(reqs, "set_conditions")
    wx = [c for c in all_calls(reqs, "weather") if c["result"]]
    for c in ann["cases"]:
        req = find_request(reqs, c["prompt_time"], c.get("nth", 0))
        night, now = c["night"], req["t0"]
        t0c = time.perf_counter()
        noon = dt.datetime.combine(dt.date.fromisoformat(night), dt.time(12))
        mine = [s for s in req["calls"] if s["name"] == "set_conditions" and any(k in s["input"] for k in COND_FIELDS)]
        earlier = [s for s in setc if noon <= s["t"] < now and any(k in s["input"] for k in COND_FIELDS)]
        src = mine[0] if mine else (earlier[-1] if earlier else None)
        cond = {k: src["input"][k] for k in COND_FIELDS if src and k in src["input"]}
        if "transparency_note" in cond:  # keep the sky description, drop sentences about the dome
            cond["transparency_note"] = " ".join(s for s in re.split(r"(?<=[.;])\s+", cond["transparency_note"])
                                                 if not RE_DECISION.search(s))
        fcs = [w for w in wx if w["input"].get("date") == night and w["t"] <= (req["t_end"] or now)]
        fc = forecast_text(fcs[-1]["result"]) if fcs else ""
        head, rows = forecast_rows(fc, now)
        ni = tools.ephem.night_info(night)
        mt = ni["events_local_mst"].get("morning_18deg") or "04:40"
        morning = dt_of(night, mt[-5:])
        pick, sky_pick = code_dome(cond, rows, now, morning)
        code_s = time.perf_counter() - t0c
        lines = [f"Observing night of {night} (local evening date). Local time now {key(now)[6:]} MST, "
                 f"morning 18-degree twilight {mt[-5:]}. Dome now: {c['dome_now']}.",
                 f"Observer says: {req['text'][:600]}"]
        if cond:
            lines.append(f"Latest measured conditions ({key(src['t'])[6:]} MST): "
                         + "; ".join(f"{k.replace('_pct', ' %').replace('_mph', ' mph').replace('_arcsec', ' arcsec').replace('_', ' ')}: {v}"
                                     for k, v in cond.items()))
        else:
            lines.append("No measured conditions reported yet tonight.")
        if rows:
            lines.append(f"Forecast (fetched {key(fcs[-1]['t'])[6:]} MST):")
            lines += head[:8] + [r["line"] for r in rows]
        else:
            lines.append("No forecast for tonight fetched yet.")
        base = {"task": "weather", "night": night, "state": "\n".join(lines), "code_seconds": code_s,
                "request_wall_s": req["wall"], "request_key": key(req["t0"]), "ref_record": c["why"]}
        out.append({**base, "question_name": "dome", "id": c["id"] + ":dome",
                    "question": "What should the telescope do now?", "options": DOME_OPTIONS,
                    "ref": [DOME.index(c["ref"])], "code_pick": DOME.index(pick)})
        sky_ref = next((s["input"]["sky"] for s in req["calls"] if s["name"] == "set_conditions" and s["input"].get("sky") in SKY), None)
        if sky_ref:
            out.append({**base, "question_name": "sky", "id": c["id"] + ":sky",
                        "question": "Which sky class describes the sky over the telescope now?", "options": SKY,
                        "ref": [SKY.index(sky_ref)], "code_pick": SKY.index(sky_pick)})
    return out


# ------------------------------------------------------------------ task 3: what next

MAX_OPTIONS = 26  # one letter per option


def build_next(reqs, ann, tools) -> list[dict]:
    P, cfg = tools.planner, tools.config.load_config()
    out = []
    for c in ann["cases"]:
        night, now = c["night"], dt_of(c["night"], c["time"])
        inst, progs, targs = state_at(reqs, now)
        t0c = time.perf_counter()
        tools.write_state(c["seeing"], inst, progs, targs, c["done"], night)
        date = dt.date.fromisoformat(night)
        queue = tools.queue(c["queue"])
        ni = tools.ephem.night_info(night)
        state = tools.state.load_state()
        cands, excluded = P.evaluate_targets(date, queue, state, ni, cfg, c["seeing"], {})
        h = hour_of(night, c["time"])
        evh = ni["events_local_hours"]
        t_end = evh["morning_12deg"] or evh["morning_18deg"]
        pool = [x for x in cands if x["instrument"] == c["instrument"]]
        menu = []
        for x in pool:  # observable now: a window (alt/HA/twilight/moon) open within 15 min with 15+ min left
            for a, b in x["windows"]:
                st = max(h, a)
                if st - h <= 0.25 + 1e-9 and b - st >= 0.25:
                    menu.append((x, st, b))
                    break
        menu.sort(key=lambda m: -m[0]["score"])
        n_feasible = len(menu)
        menu = menu[:MAX_OPTIONS]
        plan = P.greedy_schedule(pool, c["instrument"], h, t_end, cfg)
        code_pick_obj = plan["sequence"][0]["object"] if plan["sequence"] else None
        code_s = time.perf_counter() - t0c
        names = [m[0]["object"] for m in menu]
        rows = []
        for x, st, b in menu:
            alt = P._alt_at(x, h)
            extra = [n for n in x["notes"] if n.startswith(("TIME-CRITICAL", "calibration", "readme:", "PI says", "partial"))]
            fits = st + x["need_hours"] <= b + 1e-9
            rows.append(f"{x['object']}: queue priority {x['priority']:.1f}, needs {x['need_hours']:.2f} h, "
                        f"window closes {tools.ephem.fmt_local(date, b)} ({'fits' if fits else 'does not fit in full'}), altitude now {alt:.0f} deg, "
                        f"transit {x['transit_local']}, sky {x['moon_req']}" + (f"; {'; '.join(extra)}" if extra else ""))
        inst_lines = [f"{k} {v[0]}: {v[1][:160]}" for k, v in inst.items() if v[0] in ("down", "degraded")]
        lines = [f"Observing night of {night} (local evening date). Local time now {c['time']} MST; "
                 f"morning 12-degree twilight {tools.ephem.fmt_local(date, t_end)}.",
                 f"Instrument configured: {c['instrument']}. Seeing {c['seeing']} arcsec."]
        if inst_lines:
            lines.append("Instrument status: " + " | ".join(inst_lines))
        lines.append("Done earlier tonight: " + (", ".join(c["done"]) if c["done"] else "nothing yet"))
        if c.get("observer"):
            lines.append(f"Observer's instruction: {c['observer']}")
        lines.append(f"Queue targets on {c['instrument']} observable now (window open for at least 15 more minutes):")
        lines += rows
        ref_ok = c["ref"] in names
        why_missing = None
        if not ref_ok:
            why_missing = next((e["reason"] for e in excluded if e["object"] == c["ref"]), None) or \
                ("outside window now" if any(x["object"] == c["ref"] for x in pool) else "not a candidate on this instrument")
        out.append({"task": "next", "question_name": "next_target", "id": c["id"], "night": night,
                    "state": "\n".join(lines), "question": "Which target should be observed next?",
                    "options": names, "ref": [names.index(c["ref"])] if ref_ok else [],
                    "code_pick": names.index(code_pick_obj) if code_pick_obj in names else -1,
                    "code_pick_outside_menu": code_pick_obj is not None and code_pick_obj not in names,
                    "code_seconds": code_s, "n_feasible": n_feasible, "ref_in_menu": ref_ok,
                    "ref_missing_reason": why_missing, "attempt": bool(c.get("attempt")), "request_wall_s": None,
                    "ref_record": c["ref_record"]})
    return out


# ------------------------------------------------------------------ task 4: slit position angle

BLUE, RED = (0.35, 0.40, 0.45), (0.65, 0.75, 0.85, 0.95)


def pa_options(tools, program: str) -> list[int]:
    try:
        text, _ = tools.readmes.get_readme_text(program)
    except Exception:
        return []
    return sorted({int(m) for m in re.findall(r"pa([+-]?\d+)\.acq", text, re.I)})


def build_slit(reqs, ann, tools) -> list[dict]:
    objs = tools.all_objects()
    calls = all_calls(reqs, "mods_slit_losses")
    out = []
    for c in ann["cases"]:
        t = objs[c["object"]]
        pas = pa_options(tools, c["object"].split("/")[0])
        if len(pas) < 2:
            continue
        date = dt.date.fromisoformat(c["night"])
        t0c = time.perf_counter()
        res = []
        for pa in pas:
            d = tools.optics.mods_slit_check(date, t["ra"], t["dec"], pa, c["width"], c["seeing"], c["start"], c["hours"])
            cen = d["throughput_if_centred"]
            blue = [r["throughput"][w] for r in d["rows"] for w in BLUE]
            red = [r["throughput"][w] for r in d["rows"] for w in RED]
            res.append({"pa": pa, "blue_loss": 1 - (sum(blue) / len(blue)) / cen, "red_loss": 1 - (sum(red) / len(red)) / cen,
                        "worst_blue": min(blue) / cen, "parang_mid": d["parang_at_midvisit"]})
        best = min(range(len(res)), key=lambda i: res[i]["blue_loss"])
        code_s = time.perf_counter() - t0c
        opts = [f"PA {r['pa']:+d} deg" for r in res]
        lines = [f"MODS long-slit visit on night {c['night']}: start {c['start']} MST, {c['hours']} h, slit {c['width']} arcsec, "
                 f"seeing {c['seeing']} arcsec, guiding at 0.65 um, no ADC. Parallactic angle at mid-visit {res[0]['parang_mid']:+.0f} deg.",
                 "Slit losses from atmospheric dispersion over the visit (mean relative to a centred star) for each acquisition script:"]
        lines += [f"{o}: blue (0.35-0.45 um) loss {100 * r['blue_loss']:.0f} %, red (0.65-0.95 um) loss {100 * r['red_loss']:.0f} %, "
                  f"worst blue step keeps {100 * r['worst_blue']:.0f} %" for o, r in zip(opts, res)]
        src_t = next((k["t"] for k in calls if key(k["t"]) == c["from"].split("mods_slit_losses ")[-1]), None) \
            if "mods_slit_losses" in c["from"] else None
        req = request_at(reqs, src_t) if src_t else None
        out.append({"task": "slit", "question_name": "slit_pa", "id": c["id"], "night": c["night"],
                    "state": "\n".join(lines), "question": "Which slit position angle gives the lowest slit loss for this visit?",
                    "options": opts, "ref": [best], "code_pick": best, "code_seconds": code_s,
                    "request_wall_s": req["wall"] if req else None, "request_key": key(req["t0"]) if req else None,
                    "ref_record": "argmin of the code-computed blue loss (" + c["from"] + ")"})
    return out


# ------------------------------------------------------------------ Claude wall times, main

CATEGORY_OF_TASK = {"logging": "logging", "weather": "conditions_or_weather", "next": "what_next_or_replan",
                    "slit": "position_or_angle_check"}


def claude_times(reqs, start="2026-09-09", end="2026-09-16") -> dict:
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    by = defaultdict(list)
    for r in reqs:
        if r["wall"] is None:
            continue
        nd = (r["t0"] - dt.timedelta(hours=12)).date()
        if d0 <= nd <= d1:
            by[r["category"]].append(r["wall"])
    out = {}
    for task, cat in CATEGORY_OF_TASK.items():
        w = by.get(cat, [])
        out[task] = {"category": cat, "n_requests": len(w), "wall_s_median": LL.pct(w, 50), "wall_s_p10": LL.pct(w, 10),
                     "wall_s_p90": LL.pct(w, 90)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--private", required=True, help="private directory (chmod 700)")
    ap.add_argument("--tasks", default="logging,weather,next,slit")
    a = ap.parse_args(argv)
    priv = Path(a.private)
    if (priv.stat().st_mode & 0o077) != 0:
        sys.exit(f"{priv} must not be readable by group or others (chmod 700)")
    os.umask(0o077)
    reqs = load_requests(priv / "raw_transcripts")
    tools = Tools(priv)
    outdir = priv / "cases"
    outdir.mkdir(exist_ok=True)
    builders = {"logging": build_logging, "weather": build_weather, "next": build_next, "slit": build_slit}
    ann_file = {"logging": "ann_logging.json", "weather": "ann_weather.json", "next": "ann_next.json", "slit": "ann_slit.json"}
    for task in a.tasks.split(","):
        ann = json.loads((priv / ann_file[task]).read_text())
        cases = builders[task](reqs, ann, tools)
        with open(outdir / f"{task}.jsonl", "w") as fh:
            for c in cases:
                fh.write(json.dumps(c, default=str) + "\n")
        n_by_q = defaultdict(int)
        for c in cases:
            n_by_q[c["question_name"]] += 1
        print(f"{task}: {dict(n_by_q)} cases, dropped {len(ann.get('dropped', []))}")
    ct = claude_times(reqs)
    # the pool the weather cases came from: conditions/weather requests on the run's nights, by time of day
    pool = [r for r in reqs if r["wall"] is not None and r["category"] == "conditions_or_weather"
            and dt.date(YEAR, 9, 9) <= (r["t0"] - dt.timedelta(hours=12)).date() <= dt.date(YEAR, 9, 16)]
    night_pool = [r for r in pool if r["t0"].hour >= 17 or r["t0"].hour < 6]
    ct["weather"]["pool"] = {"conditions_or_weather_requests": len(pool), "in_17h_06h_window": len(night_pool),
                             "daytime_dropped": len(pool) - len(night_pool)}
    (outdir / "claude_times.json").write_text(json.dumps(ct, indent=1))
    print(json.dumps(ct))


if __name__ == "__main__":
    main()
