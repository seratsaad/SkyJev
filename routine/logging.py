"""Logging: turn an observer's message into log entries, which queue targets and done/partial/skipped.

Code shortlists the queue targets the message may name (routine/logging_data.shortlist: fuzzy match of
every word and word pair against the night's queue names), then two fixed Jev questions, each with a
fitted L2 head and a confidence gate:
  target  Question.noul("Is this message about the queue target under review?"), once per shortlisted
          candidate, the candidate named at the end of the state;
  status  Question.choice over done / partial / skipped, once per target the message is about.
Code baselines: the fuzzy matcher (logging_data.code_mentions) for target, the replay's keyword rule
(logging_data.code_status) for status. Message-level scores (every target right, and every target and
status right) come from combining the two.

Training messages are synthetic, template family A; val and test are family B (other phrasings,
frames and night-log layout; catalogue prefixes half unseen; no shared names). Real cases: the replay
logging cases (private_lbt/cases/logging.jsonl), read inside the job; status references are the
assistant's recorded statuses; target references are hand-annotated for the chat messages and the
replay's exact-name mentions for the night logs (so the matcher is close to the reference there).

    python -m routine.logging --model Qwen/Qwen3-4B --private /fs/scratch/PAS2823/saadsm/private_lbt
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from routine.common import Split
from routine import logging_data as D

FRAME_LOG = "Observing night of {night} (local evening date). Night log from the observer:\n{text}"
FRAME_CHAT = ("Observing night of {night} (local evening date), local time {t} MST, instrument in use {inst}. "
              "Message from the observer:\n{text}")
SUFFIX = "\n\nQueue target under review: {target} (program {program})"
Q_STATUS_TEXT = "According to this message, what happened to the queue target under review?"
Q_TARGET_TEXT = "Is this message about the queue target under review?"


def synth(n: int, fam: str, seed: int, forbidden: set) -> tuple:
    """n synthetic messages of family fam; returns (messages, normalised names used)."""
    rng, used, out = np.random.default_rng(seed), set(), []
    for _ in range(n):
        queue = D.make_queue(rng, fam, used, forbidden)
        if len(queue) < 4:
            continue
        kind = "chat" if rng.random() < 0.45 else "log"
        text, truth = D.make_message(rng, fam, queue, kind)
        night = (dt.date(2027, 1, 5) + dt.timedelta(days=int(rng.integers(0, 330)))).isoformat()
        head = (FRAME_LOG.format(night=night, text=text) if kind == "log" else
                FRAME_CHAT.format(night=night, t=D._clock(rng, 1)[0], inst=queue[0]["instrument"], text=text))
        out.append({"kind": kind, "text": text, "head": head, "queue": queue, "truth": truth})
    return out, used


def suffix(obj: str) -> str:
    prog, targ = obj.split("/", 1)
    return SUFFIX.format(target=targ, program=prog)


def cases_of(msgs: list, code_status_given=None) -> dict:
    """Status and target cases, code predictions, and per-message groups for combining them."""
    st, st_code, st_msg, st_obj, tg, tg_code, tg_msg, tg_obj, recall = [], [], [], [], [], [], [], [], []
    for i, m in enumerate(msgs):
        table = D.match_table(m["text"], m["queue"])
        sl, cm = D.shortlist(table), D.code_mentions(table)
        m["shortlist"], m["code_set"] = sl, set(cm)
        for o, s in m["truth"].items():
            st.append((m["head"] + suffix(o), s))
            st_code.append(code_status_given[(i, o)] if code_status_given else D.code_status(m["text"], cm.get(o, [])))
            st_msg.append(i)
            st_obj.append(o)
        for c in sl:
            tg.append((m["head"] + suffix(c), 0 if c in m["truth"] else 1))
            tg_code.append(0 if c in cm else 1)
            tg_msg.append(i)
            tg_obj.append(c)
        recall.append(len(set(sl) & set(m["truth"])) / len(m["truth"]))
    return {"status": st, "status_code": st_code, "status_msg": st_msg, "status_obj": st_obj, "target": tg, "target_code": tg_code,
            "target_msg": tg_msg, "target_obj": tg_obj, "shortlist_recall": round(float(np.mean(recall)), 4) if recall else None}


def message_scores(msgs: list, C: dict, ps, pt, thr_s: float, thr_t: float) -> dict:
    """Per message: is the set of targets right (candidates with P(yes) >= 0.5), and are the set and
    every status right; the gated version acts only when every decision of the message clears its gate.
    ps / pt: status and target probabilities (None: score the code baseline instead)."""
    by_s, by_t = defaultdict(list), defaultdict(list)
    for k, i in enumerate(C["status_msg"]):
        by_s[i].append(k)
    for j, i in enumerate(C["target_msg"]):
        by_t[i].append(j)
    set_ok, entry_ok, acted = [], [], []
    for i, m in enumerate(msgs):
        T = set(m["truth"])
        if pt is None:
            got, conf_t = m["code_set"], True
        else:
            got = {C["target_obj"][j] for j in by_t[i] if pt[j, 0] >= 0.5}
            conf_t = all(pt[j].max() >= thr_t for j in by_t[i])
        if ps is None:
            st_ok, conf_s = all(C["status_code"][k] == C["status"][k][1] for k in by_s[i]), True
        else:
            st_ok = all(int(ps[k].argmax()) == C["status"][k][1] for k in by_s[i])
            conf_s = all(ps[k].max() >= thr_s for k in by_s[i])
        set_ok.append(got == T)
        entry_ok.append(got == T and st_ok)
        acted.append(conf_t and conf_s)
    from routine.common import wilson
    n, a = len(msgs), np.array(acted)
    e = np.array(entry_ok)
    return {"n_messages": n, "target_set_exact": round(float(np.mean(set_ok)), 4), "target_set_exact_ci95": wilson(sum(set_ok), n),
            "entry_exact": round(float(e.mean()), 4), "entry_exact_ci95": wilson(int(e.sum()), n),
            "acted": int(a.sum()), "coverage": round(float(a.mean()), 4),
            "entry_exact_when_acting": round(float(e[a].mean()), 4) if a.any() else None,
            "entry_exact_when_acting_ci95": wilson(int(e[a].sum()), int(a.sum()))}


def real_messages(private: str) -> tuple:
    """The replay logging cases regrouped into messages: text, the night's queue (plus any referenced
    target), truth {object: recorded status}, the replay's code status pick. In memory only."""
    P = Path(private)
    cases = [json.loads(line) for line in open(P / "cases" / "logging.jsonl")]
    queues, allobj = defaultdict(dict), {}
    for f in sorted((P / "src" / "data" / "cache").glob("queue_*.json")):
        q = json.loads(f.read_text())
        for t in q["targets"]:
            e = {"program": t["program"], "target": t["target"], "object": t["object"], "instrument": t.get("instrument")}
            queues[str(q.get("date"))][t["object"]] = e
            allobj.setdefault(t["object"], e)
    groups, msgs, code = defaultdict(list), [], {}
    for c in cases:
        groups[c["state"]].append(c)
    kinds = Counter()
    for state, cs in groups.items():
        text = state.split("from the observer:\n", 1)[1]
        truth = {}
        for c in cs:
            o = c["id"].split(":", 1)[1] if c["id"].split(":", 1)[-1] in allobj else \
                re.search(r"queue target (.+?) (?:that night|tonight)\?", c["question"]).group(1)
            truth[o] = c["ref"][0]
            code[(len(msgs), o)] = c["code_pick"]
        night = cs[0]["night"]
        queue = dict(queues.get(night) or allobj)
        queue.update({o: allobj[o] for o in truth})
        kinds[cs[0]["source"]] += 1
        msgs.append({"kind": cs[0]["source"], "text": text, "head": state, "queue": list(queue.values()), "truth": truth})
    return msgs, code, {"messages_by_source": dict(kinds), "nights_with_queue_page": sum(bool(queues.get(m)) for m in {c["night"] for c in cases})}


def main(argv=None):
    from anyjev import Question
    from routine.logging_eval import GATES, load_backend, private_guard, run_question, write_report

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--private", default=None, help="private_lbt directory (real cases); omit to skip them")
    ap.add_argument("--out-prefix", default="routine/reports/logging_")
    ap.add_argument("--n-train", type=int, default=220)
    ap.add_argument("--n-val", type=int, default=80)
    ap.add_argument("--n-test", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dry", action="store_true", help="build the cases and print counts only (no model)")
    a = ap.parse_args(argv)

    real_names = set()
    if a.private:
        for f in (Path(a.private) / "src" / "data" / "cache").glob("queue_*.json"):
            for t in json.loads(f.read_text())["targets"]:
                real_names |= {D.norm(t["target"]), D.norm(t["program"])}
    train, used = synth(a.n_train, "A", 11, real_names)
    val, used_v = synth(a.n_val, "B", 22, real_names | used)
    test, _ = synth(a.n_test, "B", 33, real_names | used | used_v)
    parts = {"train": cases_of(train), "val": cases_of(val), "test": cases_of(test)}
    info = {p: {"messages": dict(Counter(m["kind"] for m in ms)), "status_cases": len(parts[p]["status"]),
                "status_labels": dict(Counter(D.STATUS[y] for _, y in parts[p]["status"])),
                "target_cases": len(parts[p]["target"]), "target_yes": sum(y == 0 for _, y in parts[p]["target"]),
                "shortlist_recall": parts[p]["shortlist_recall"]}
            for p, ms in (("train", train), ("val", val), ("test", test))}
    rmsgs = []
    if a.private:
        rmsgs, rcode, rinfo = real_messages(a.private)
        parts["real"] = cases_of(rmsgs, code_status_given=rcode)
        info["real"] = dict(rinfo, status_cases=len(parts["real"]["status"]),
                            status_labels=dict(Counter(D.STATUS[y] for _, y in parts["real"]["status"])),
                            target_cases=len(parts["real"]["target"]), target_yes=sum(y == 0 for _, y in parts["real"]["target"]),
                            shortlist_recall=parts["real"]["shortlist_recall"])
    guard = private_guard(a.private)
    guard(json.dumps(info))
    print(json.dumps(info, indent=1))
    if a.dry:
        return
    tag = a.model.split("/")[-1].lower()
    get = lambda q: Split(*(parts[p][q] if p in parts else [] for p in ("train", "val", "test", "real")))  # noqa: E731
    code = lambda q: {p: parts[p][q + "_code"] for p in ("test", "real") if p in parts}  # noqa: E731
    be = load_backend(a.model, batch_size=a.batch_size)
    reps, probs = {}, {}
    for q, question in (("status", Question.choice(Q_STATUS_TEXT, D.STATUS_OPTIONS, name="log_status")),
                        ("target", Question.noul(Q_TARGET_TEXT, name="log_target"))):
        reps[q], probs[q] = run_question(be, f"logging_{q}", question, get(q), a.model, code=code(q),
                                         notes="synthetic train (family A), val/test (family B); real = LBT replay")
        reps[q]["data"] = info
        write_report(f"{a.out_prefix}{q}_{tag}.json", reps[q], guard)
    msg = {"task": "logging_messages", "model": a.model,
           "definition": "per message: target_set_exact = the candidates with P(yes) >= 0.5 are exactly the targets it is "
                         "about (shortlist misses count as errors); entry_exact = that and every status right; the gate acts "
                         "on a message only if every target and status decision clears its validation-picked threshold"}
    for p, ms in (("test", test), ("real", rmsgs)):
        if not ms:
            continue
        C = parts[p]
        msg[p] = {"code": message_scores(ms, C, None, None, 0, 0)}
        for g in GATES:
            msg[p][f"L2_gate_{g}"] = message_scores(ms, C, probs["status"][p], probs["target"][p],
                                                    reps["status"]["threshold"][str(g)], reps["target"]["threshold"][str(g)])
    write_report(f"{a.out_prefix}messages_{tag}.json", msg, guard)


if __name__ == "__main__":
    main()
