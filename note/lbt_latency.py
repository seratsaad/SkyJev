#!/usr/bin/env python3
"""Per-request wall-time statistics for LBT observing-assistant sessions.

Usage:  python3 lbt_latency.py TRANSCRIPT_DIR [--out OUT.json]
        [--start 2026-09-09] [--end 2026-09-16]

Reads every Claude Code session transcript (*.jsonl, recursively) under
TRANSCRIPT_DIR. A "request" starts at a real human prompt (a `user` record
whose content is text, not a tool_result, not isMeta / isCompactSummary, and
not a harness tag such as <command-name> or "[Request interrupted ...") and
ends at the last `assistant` record before the next human prompt in the same
session file.

Only aggregate numbers and generic category labels are written. No prompt
text, names or other transcript content is ever emitted.

Night date: timestamps are UTC; the local (MST = UTC-7) evening date is
date(UTC - 7 h - 12 h), so e.g. 03:00 MST on Sep 10 belongs to night Sep 9.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

MST = timedelta(hours=-7)
CATEGORY_ORDER = ["full_night_plan", "what_next_or_replan", "position_or_angle_check",
                  "readme_reading", "conditions_or_weather", "logging", "other"]

RE_REPLAN = re.compile(r"\b(what'?s next|what next|next target|now what|replan|re-plan|"
                       r"switch\w*|swap\w*|instead|fail\w*|broke\w*|fault\w*|"
                       r"not working|stuck|problem\w*|down)\b", re.I)
RE_POSITION = re.compile(r"\b(gaia|simbad|vizier|parallactic|position angle|pa|"
                         r"angle|coord\w*|offset\w*|proper motion|slit loss\w*|"
                         r"visibility|visible|airmass)\b", re.I)
RE_README = re.compile(r"\breadme\w*\b", re.I)
RE_COND = re.compile(r"\b(seeing|cloud\w*|humid\w*|wind\w*|weather|dome|rain\w*|fog)\b", re.I)
RE_LOG = re.compile(r"\b(observed|log\w*|mark\w*|note\w*)\b", re.I)

POSITION_TOOLS = ("target_visibility", "mods_slit_losses")
LOG_TOOLS = ("mark_observed", "add_night_note", "observed_log")
COND_TOOLS = ("weather", "get_conditions", "set_conditions")


def parse_ts(s):
    return datetime.strptime(s.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S.%f%z")


def human_text(rec):
    """Return prompt text if rec is a real human prompt, else None."""
    if rec.get("type") != "user" or rec.get("isMeta") or rec.get("isCompactSummary") \
            or rec.get("isSidechain"):
        return None
    c = (rec.get("message") or {}).get("content")
    if isinstance(c, list):
        if any(b.get("type") == "tool_result" for b in c):
            return None
        c = " ".join(b.get("text", "") for b in c if b.get("type") == "text")
    if not isinstance(c, str):
        return None
    s = c.lstrip()
    if not s or s.startswith("<") or s.startswith("[Request interrupted") or s.startswith("Caveat:"):
        return None
    return s


def short_tool(name):
    return name.split("__")[-1] if name.startswith("mcp__") else name


def classify(text, tools):
    short = [short_tool(t) for t in tools]
    if "what_next" in short or RE_REPLAN.search(text):
        return "what_next_or_replan"
    if "plan_tonight" in short:
        return "full_night_plan"
    if any(t in POSITION_TOOLS for t in short) or any(t.startswith("mcp__astrodata__") for t in tools) \
            or RE_POSITION.search(text):
        return "position_or_angle_check"
    if "get_readme" in short or RE_README.search(text):
        return "readme_reading"
    if any(t in COND_TOOLS for t in short) or RE_COND.search(text):
        return "conditions_or_weather"
    if any(t in LOG_TOOLS for t in short) or RE_LOG.search(text):
        return "logging"
    return "other"


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def requests_from_file(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    out = []
    cur = None
    enq = []            # FIFO of enqueue timestamps
    pending_enq = None  # enqueue time of the prompt just dequeued
    open_tools = {}     # tool_use id -> start time
    for r in recs:
        ts = r.get("timestamp")
        t = r.get("type")
        if t == "queue-operation" and ts:
            op = r.get("operation")
            if op == "enqueue":
                enq.append(parse_ts(ts))
            elif op == "dequeue" and enq:
                pending_enq = enq.pop(0)
            elif op == "remove" and enq:
                enq.pop(0)
                if cur is not None:
                    cur["midturn_inputs"] += 1
            continue
        text = human_text(r)
        if text is not None and ts:
            if cur is not None:
                out.append(cur)
            t0 = parse_ts(ts)
            cur = {"t0": t0, "t_end": None, "text": text, "tools": [], "models": set(),
                   "midturn_inputs": 0, "interrupted": False, "tool_gaps": [],
                   "queued_wait": (t0 - pending_enq).total_seconds() if pending_enq else 0.0}
            pending_enq = None
            open_tools = {}
            continue
        if cur is None:
            continue
        msg = r.get("message") or {}
        if t == "assistant" and ts:
            cur["t_end"] = parse_ts(ts)
            m = msg.get("model")
            if m and m != "<synthetic>":
                cur["models"].add(m)
            for b in msg.get("content") or []:
                if b.get("type") == "tool_use":
                    cur["tools"].append(b.get("name", "?"))
                    open_tools[b.get("id")] = parse_ts(ts)
        elif t == "user" and ts:
            c = msg.get("content")
            if isinstance(c, list):
                for b in c:
                    if b.get("type") == "tool_result" and b.get("tool_use_id") in open_tools:
                        cur["tool_gaps"].append(
                            (parse_ts(ts) - open_tools.pop(b["tool_use_id"])).total_seconds())
                    if b.get("type") == "text" and b.get("text", "").startswith("[Request interrupted"):
                        cur["interrupted"] = True
            elif isinstance(c, str) and c.startswith("[Request interrupted"):
                cur["interrupted"] = True
    if cur is not None:
        out.append(cur)
    return out


def night_date(t):
    return (t + MST - timedelta(hours=12)).date()


def summarize(reqs):
    walls = [r["wall"] for r in reqs]
    ntool = [len(r["tools"]) for r in reqs]
    models = Counter(m for r in reqs for m in r["models"])
    tool_counts = Counter(short_tool(t) for r in reqs for t in r["tools"])
    long_gap = [sum(g for g in r["tool_gaps"] if g > 60) for r in reqs]
    return {
        "n_requests": len(reqs),
        "wall_s_median": round(pct(walls, 50), 1) if walls else None,
        "wall_s_p10": round(pct(walls, 10), 1) if walls else None,
        "wall_s_p90": round(pct(walls, 90), 1) if walls else None,
        "tool_calls_median": pct(ntool, 50) if ntool else None,
        "models_seen": dict(models),
        "n_no_tool_calls": sum(not r["tools"] for r in reqs),
        "n_interrupted": sum(r["interrupted"] for r in reqs),
        "n_with_midturn_user_input": sum(r["midturn_inputs"] > 0 for r in reqs),
        "n_with_single_tool_wait_over_60s": sum(g > 0 for g in long_gap),
        "top_tools": dict(tool_counts.most_common(8)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("transcript_dir")
    ap.add_argument("--out", default="lbt_latency.json")
    ap.add_argument("--start", default="2026-09-09")
    ap.add_argument("--end", default="2026-09-16")
    a = ap.parse_args()
    d0 = datetime.strptime(a.start, "%Y-%m-%d").date()
    d1 = datetime.strptime(a.end, "%Y-%m-%d").date()

    files = sorted(glob.glob(os.path.join(a.transcript_dir, "**", "*.jsonl"), recursive=True))
    allreq, n_total_prompts = [], 0
    for f in files:
        for r in requests_from_file(f):
            n_total_prompts += 1
            if r["t_end"] is None:
                continue
            nd = night_date(r["t0"])
            if not (d0 <= nd <= d1):
                continue
            r["wall"] = (r["t_end"] - r["t0"]).total_seconds()
            r["night"] = nd
            r["category"] = classify(r["text"], r["tools"])
            r.pop("text")  # never keep prompt text beyond classification
            allreq.append(r)

    bycat = defaultdict(list)
    for r in allreq:
        bycat[r["category"]].append(r)
    nights = Counter(str(r["night"]) for r in allreq)
    local_hour = [((r["t0"] + MST).hour) for r in allreq]
    queued = [r["queued_wait"] for r in allreq]
    result = {
        "description": "Wall time from human prompt to last assistant message before the next "
                       "human prompt, LBT observing-assistant sessions. Aggregates only.",
        "n_transcript_files": len(files),
        "n_human_prompts_all_dates": n_total_prompts,
        "n_requests": len(allreq),
        "night_date_range_requested": [a.start, a.end],
        "night_date_range_seen": [str(min(r["night"] for r in allreq)),
                                  str(max(r["night"] for r in allreq))] if allreq else None,
        "requests_per_night": dict(sorted(nights.items())),
        "n_requests_18h_to_07h_mst": sum(h >= 18 or h < 7 for h in local_hour),
        "queued_wait_s_median": round(pct(queued, 50), 1) if queued else None,
        "n_queued_over_5s": sum(q > 5 for q in queued),
        "overall": summarize(allreq),
        "categories": {c: summarize(bycat[c]) for c in CATEGORY_ORDER if bycat[c]},
        "classification_rule": "first match in order: what_next_or_replan (what_next tool or "
                               "failure/switch words) > full_night_plan (plan_tonight) > "
                               "position_or_angle_check (target_visibility, mods_slit_losses, "
                               "astrodata tools, or position/angle words) > readme_reading "
                               "(get_readme or 'readme') > conditions_or_weather (weather, get_conditions, "
                               "set_conditions, or seeing/cloud/humidity/wind/dome words) > logging (mark_observed, add_night_note, "
                               "observed_log, or log words) > other",
    }
    s = json.dumps(result, indent=1)
    print(s)
    with open(a.out, "w") as fh:
        fh.write(s + "\n")


if __name__ == "__main__":
    sys.exit(main())
