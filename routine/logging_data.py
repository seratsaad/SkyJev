"""Synthetic observer messages for the logging task, and the code side of logging.

Code side (used on real nights too):
  shortlist(text, queue)   fuzzy candidates: queue targets whose name (or program name, when the
                           program has one target) is close to a word or word pair of the message
  code_mentions(...)       the regex/fuzzy matcher baseline for "which target"
  code_status(...)         the replay's keyword rule (lbt_replay/build_cases.code_status), on the
                           lines that name the target

Synthetic side: invented queues (target names in the style of the LBT queue: mostly Title-case
catalogue prefix + digits, some upper-case prefix + digits, a few J-coordinates and lower-case names;
programs PARTNER_WORD) and messages written from two disjoint template families. Family A (train):
one chat phrasing set and one night-log layout; family B (val and test): other phrasings, frames and
layout. Evaluation names use catalogue prefixes half of which never occur in training, and no name is
shared between train and evaluation. Surface forms vary: exact, lower/upper case, separators, planet
suffix, one-letter typos. A target named only in a plan list and never reported is "skipped", the
replay's convention ("nothing said = skipped").
"""
from __future__ import annotations

import difflib
import re
from typing import Dict, List, Tuple

import numpy as np

STATUS = ["done", "partial", "skipped"]
STATUS_OPTIONS = ["done: observed and finished", "partial: started, not finished", "skipped: not observed"]

# ------------------------------------------------------------------ the replay's keyword rule, verbatim
RE_SKIP = re.compile(r"not (be )?(observed|taken|done)|skip\w*|\blost\b|remov\w*|no observing|cancel\w*|abandon\w*", re.I)
RE_DONE = re.compile(r"\bdone\b|complet\w*|finish\w*", re.I)
RE_PART = re.compile(r"partial\w*|stopped|abort\w*|incomplete", re.I)


def code_status(text: str, spots: List[int]) -> int:
    """Skip words first, then done words, then partial words, on the rest of each line that names the
    target (the whole message if none); nothing said counts as skipped."""
    wins = [text[s: (text.find("\n", s) if text.find("\n", s) > 0 else len(text))] for s in spots] or [text]
    for rx, lab in ((RE_SKIP, 2), (RE_DONE, 0), (RE_PART, 1)):
        if any(rx.search(w) for w in wins):
            return lab
    return 2


# ------------------------------------------------------------------ fuzzy shortlist and matcher
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+_.\-]*[A-Za-z0-9]|[A-Za-z0-9]")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def grams(text: str) -> List[Tuple[int, str]]:
    """(offset, normalised form) of every word and adjacent word pair; a trailing planet letter
    after digits is also tried without it."""
    toks = [(m.start(), norm(m.group())) for m in TOKEN.finditer(text)]
    out = []
    for i, (s, t) in enumerate(toks):
        out.append((s, t))
        if i + 1 < len(toks):
            out.append((s, t + toks[i + 1][1]))
    out += [(s, g[:-1]) for s, g in out if len(g) > 2 and g[-1] == "b" and g[-2].isdigit()]
    return [(s, g) for s, g in out if g]


def keys_of(queue: List[Dict]) -> Dict[str, List[str]]:
    """Normalised names each queue entry answers to: its target, and its program when that program
    has a single target in the queue."""
    per_prog: Dict[str, int] = {}
    for q in queue:
        per_prog[q["program"]] = per_prog.get(q["program"], 0) + 1
    return {q["object"]: [norm(q["target"])] + ([norm(q["program"])] if per_prog[q["program"]] == 1 else []) for q in queue}


def _ratio(m: difflib.SequenceMatcher, a: str) -> float:
    """Ratio of a against the matcher's fixed second sequence (a gram), cheap bounds first."""
    m.set_seq1(a)
    return m.ratio() if m.real_quick_ratio() >= 0.6 and m.quick_ratio() >= 0.6 else 0.0


def match_table(text: str, queue: List[Dict]) -> Dict[str, List[Tuple[float, int, bool]]]:
    """For every queue object: (ratio, offset, is_best_for_that_gram) over the grams it matches >= 0.6."""
    keys = keys_of(queue)
    out: Dict[str, List[Tuple[float, int, bool]]] = {o: [] for o in keys}
    for s, g in grams(text):
        if len(g) < 3:
            continue
        m = difflib.SequenceMatcher(None, "", g, autojunk=False)
        sc = {o: max(_ratio(m, k) for k in ks) for o, ks in keys.items()}
        top = max(sc.values()) if sc else 0.0
        for o, r in sc.items():
            if r >= 0.6:
                out[o].append((r, s, r == top))
    return out


def shortlist(table, max_n: int = 16) -> List[str]:
    """Queue objects with any gram at ratio >= 0.6, best first, at most max_n."""
    best = {o: max(r for r, _, _ in v) for o, v in table.items() if v}
    return sorted(best, key=lambda o: -best[o])[:max_n]


def code_mentions(table, thr: float = 0.85) -> Dict[str, List[int]]:
    """The matcher baseline: an object is named where a gram matches it at >= thr and no other queue
    object matches that gram better. Returns object -> offsets of those grams."""
    return {o: sorted({s for r, s, top in v if r >= thr and top}) for o, v in table.items()
            if any(r >= thr and top for r, _, top in v)}


# ------------------------------------------------------------------ invented queues
PREFIX_TRAIN = ["Wasp", "Kelt", "Hat", "Toi", "Kepler", "Tres", "Xo", "Hats", "Ngts", "Qatar", "Ross", "Wolf", "Ztf", "Gaia"]
PREFIX_EVAL = PREFIX_TRAIN[::2] + ["Mascara", "Ltt", "Tyc", "Epic", "Koi", "Corot", "Sdss"]
UPPER_TRAIN, UPPER_EVAL = ["HD", "TOI", "BD", "HIP"], ["HD", "BD", "KIC", "PG", "LP"]
PARTNERS = ["OSU", "ND", "UM", "UVA"]
SYLL = ["ka", "ro", "mi", "tel", "van", "dor", "si", "lan", "ber", "gu", "nes", "ol", "pra", "ven", "tu", "ash", "rim", "co"]
INSTR = ["PEPSI"] * 7 + ["MODS", "LUCI", "LBC"]


def make_name(rng, fam: str) -> str:
    r = rng.random()
    pre = PREFIX_TRAIN if fam == "A" else PREFIX_EVAL
    up = UPPER_TRAIN if fam == "A" else UPPER_EVAL
    if r < 0.62:
        return f"{rng.choice(pre)}{rng.integers(1, 10 ** int(rng.integers(1, 5)))}"
    if r < 0.84:
        return f"{rng.choice(up)}{rng.integers(100, 10 ** int(rng.integers(3, 7)))}"
    if r < 0.93:
        return f"J{rng.integers(0, 24):02d}{rng.integers(0, 60):02d}"
    return f"{rng.choice(['ngc', 'ugc', 'mrk'])}{rng.integers(10, 9999)}"


def make_program(rng) -> str:
    w = "".join(rng.choice(SYLL) for _ in range(int(rng.integers(2, 4)))).upper()
    return f"{rng.choice(PARTNERS)}_{w}" + (str(rng.integers(1, 4)) if rng.random() < 0.15 else "")


def near_name(rng, name: str) -> str:
    """A queue neighbour of a name: same prefix, digits shuffled or one changed."""
    m = re.match(r"([A-Za-z]+)(\d+)$", name)
    if not m or len(m.group(2)) < 2:
        return name + str(rng.integers(0, 10))
    d = list(m.group(2))
    if rng.random() < 0.5:
        i = int(rng.integers(len(d) - 1))
        d[i], d[i + 1] = d[i + 1], d[i]
    else:
        d[int(rng.integers(len(d)))] = str(rng.integers(0, 10))
    return m.group(1) + "".join(d)


def make_queue(rng, fam: str, used: set, forbidden: set) -> List[Dict]:
    """20-60 targets in 6-15 programs; a program often observes one catalogue family, so near-miss
    names sit next to each other. `used` collects names; `forbidden` holds names another split used
    (and, on Pitzer, the real queue names), all compared normalised."""
    queue, names = [], set()
    for _ in range(int(rng.integers(6, 16))):
        prog = make_program(rng)
        if norm(prog) in forbidden:
            continue
        base = make_name(rng, fam)
        for j in range(int(rng.choice([1, 1, 2, 3, 4, 6, 8]))):
            nm = base if j == 0 else (near_name(rng, base) if rng.random() < 0.6 else make_name(rng, fam))
            if norm(nm) in names or norm(nm) in forbidden or len(queue) >= 60:
                continue
            names.add(norm(nm))
            queue.append({"program": prog, "target": nm, "object": f"{prog}/{nm}", "instrument": str(rng.choice(INSTR))})
    used |= names | {norm(q["program"]) for q in queue}
    return queue


def surface(rng, name: str) -> str:
    """How an observer writes a name: mostly as is, sometimes cased, split, suffixed or misspelt."""
    r, m = rng.random(), re.match(r"([A-Za-z]+)(\d+)$", name)
    if r < 0.55 or not m:
        return name
    pre, dig = m.group(1), m.group(2)
    if r < 0.65:
        return name.lower()
    if r < 0.72:
        return name.upper()
    if r < 0.82:
        return f"{pre}{rng.choice(['-', ' '])}{dig}"
    if r < 0.88:
        return name + rng.choice(["b", " b"])
    if len(pre) >= 3:  # one-letter typo in the letters
        p = list(pre)
        i = int(rng.integers(len(p) - 1))
        if rng.random() < 0.5:
            p[i], p[i + 1] = p[i + 1], p[i]
        else:
            del p[i + 1]
        return "".join(p) + dig
    return name


# ------------------------------------------------------------------ messages, two template families
REASONS = ["clouds", "high humidity", "wind", "thick cirrus", "bad seeing", "an instrument fault", "twilight",
           "the dome closing", "ice on the dome"]
PH = {"A": {0: ["{T} done", "finished {T}", "{T} complete ({n}x{e}s)", "{T}: {n}/{n} exposures, finished",
                "completed {T}", "{T} done, S/N looks fine"],
            1: ["{T} partial, {k} of {n} exposures", "{T} stopped after {k} exposures ({r})", "partial on {T} - {r}",
                "got {k}/{n} on {T}, needs another visit", "{T}: {k}x{e}s of {n}, incomplete"],
            2: ["skipping {T}, {r}", "{T} not observed ({r})", "no time for {T}", "{T} skipped - {r}",
                "dropped {T} because of {r}"]},
      "B": {0: ["{T} wrapped up", "all {n} frames of {T} in hand", "{T} observed in full",
                "{T}: requested {n}x{e}s delivered", "got everything for {T}", "{T} is finished and looks good"],
            1: ["only {k} frames on {T} before {r} ended it", "{T} aborted mid-sequence ({k} of {n})",
                "{T} half way when {r} hit", "{T} cut short by {r}, {k} exposures usable", "{T} needs {m} more exposures"],
            2: ["never got to {T}", "{T} didn't start ({r})", "passed on {T}, {r}", "{T} left for another night",
                "{T} not attempted, {r}"]}}
PAIR = {"A": {0: "finished {T} and {U}", 1: "{T} and {U} both partial ({r})", 2: "skipping {T} and {U} ({r})"},
        "B": {0: "{T} + {U} wrapped up", 1: "{T} + {U} cut short by {r}", 2: "never got to {T} or {U}"}}
CHAT = {"A": ["{x}", "update: {x}", "{t} {x}", "log: {x}", "fyi {x}"],
        "B": ["quick note - {x}", "{x}. moving on.", "for the log: {x}", "status {t}: {x}", "{x} (from the control room)"]}
JOIN = {"A": [", ", "; "], "B": [". ", " / "]}
EVENTS = {"A": ["dome open", "closed, RH {rh}%", "reopened", "seeing {s}\" on the guider", "thin clouds passing",
                "closed, wind gusts {w} mph", "focus and collimation", "calibrations done"],
          "B": ["opened", "shut, RH at {rh}%", "open again", "DIMM {s}\"", "cirrus from the west", "shut for wind",
                "AGw acquisition trouble", "flats and arcs taken"]}
NOTES = {"A": ["All data transferred.", "Guider was flaky early on.", "Next night should favour the MODS targets.", ""],
         "B": ["Data copied to the archive.", "PEPSI throughput looked normal.", "Remember to restart the TCS GUI.", ""]}


def _fill(rng, tpl: str, **kw) -> str:
    n = int(rng.integers(2, 7))
    k = int(rng.integers(1, n))
    return tpl.format(n=n, k=k, m=n - k, e=int(rng.choice([300, 600, 900, 1200, 1800])), r=str(rng.choice(REASONS)),
                      rh=int(rng.integers(80, 99)), s=round(float(rng.uniform(0.6, 2.2)), 1), w=int(rng.integers(30, 60)), **kw)


def _clock(rng, n: int) -> List[str]:
    ts = np.sort(rng.uniform(19.0, 29.5, n))
    return [f"{int(t) % 24:02d}:{int((t % 1) * 60):02d}" for t in ts]


def make_message(rng, fam: str, queue: List[Dict], kind: str) -> Tuple[str, Dict[str, int]]:
    """One observer message about some of the queue's targets. Returns (text, {object: status index})."""
    k = int(rng.integers(1, 4)) if kind == "chat" else int(rng.integers(3, min(13, len(queue)) + 1))
    picked = [queue[i] for i in rng.choice(len(queue), size=min(k, len(queue)), replace=False)]
    closed = kind == "log" and rng.random() < 0.2
    truth = {q["object"]: (2 if closed else int(rng.choice(3, p=[0.35, 0.25, 0.40]))) for q in picked}
    say = {q["object"]: surface(rng, q["target"]) for q in picked}
    items, left = [], list(picked)
    while left:
        q = left.pop(0)
        mate = next((u for u in left if truth[u["object"]] == truth[q["object"]]), None)
        if mate is not None and rng.random() < 0.15:
            left.remove(mate)
            items.append(_fill(rng, PAIR[fam][truth[q["object"]]], T=say[q["object"]], U=say[mate["object"]]))
        else:
            items.append((q, _fill(rng, str(rng.choice(PH[fam][truth[q["object"]]])), T=say[q["object"]])))
    if kind == "chat":
        x = str(rng.choice(JOIN[fam])).join(i if isinstance(i, str) else i[1] for i in items)
        return _fill(rng, str(rng.choice(CHAT[fam])), x="{x}", t=_clock(rng, 1)[0]).format(x=x), truth
    return _night_log(rng, fam, queue, items, say, truth, closed), truth


def _night_log(rng, fam, queue, items, say, truth, closed) -> str:
    inst = queue[0]["instrument"]
    wx = _fill(rng, str(rng.choice(["clear most of the night", "variable cirrus, RH {rh}% at times",
                                    "windy, seeing {s}\"", "clouds after midnight"])))
    plan_line = rng.random() < 0.5 or closed
    head = ([f"{inst} queue night log", f"Observers: {'ABCDEFGH'[rng.integers(8)]}{'KLMNPRST'[rng.integers(8)]}", f"Weather: {wx}"]
            if fam == "A" else [f"== Night report ({inst}) ==", f"Conditions: {wx}"])
    plan_names = [say[o] for o in truth]
    rng.shuffle(plan_names)
    plan = ("Plan for tonight: " if fam == "A" else "Tonight's queue list: ") + ", ".join(plan_names)
    if closed:
        body = [(f"Closed all night ({rng.choice(REASONS)}). Nothing observed." if fam == "A"
                 else f"Dome never opened: {rng.choice(REASONS)}."), plan]
        return "\n".join(head + body)
    lines, unsaid, listed = [], [], []
    for it in items:
        if not isinstance(it, str) and truth[it[0]["object"]] == 2 and rng.random() < 0.5:
            (unsaid if (plan_line and rng.random() < 0.5) else listed).append(say[it[0]["object"]])
        else:
            lines.append(it if isinstance(it, str) else it[1])
    lines += [_fill(rng, str(rng.choice(EVENTS[fam]))) for _ in range(int(rng.integers(3, 11)))]
    order = rng.permutation(len(lines))
    times = _clock(rng, len(lines))
    body = [(f"{t} {lines[i]}" if fam == "A" else f"- {t}: {lines[i]}") for t, i in zip(times, order)]
    if fam == "B":
        body = ["Log:"] + body
    tail = []
    if listed:
        r = rng.choice(REASONS)
        tail.append(f"Not observed: {', '.join(listed)} ({r})." if fam == "A"
                    else f"Queue targets without data tonight ({r}): {', '.join(listed)}")
    note = str(rng.choice(NOTES[fam]))
    if note:
        tail.append(("Notes: " if fam == "A" else "Handover: ") + note)
    return "\n".join(head + ([plan] if plan_line else []) + body + tail)
