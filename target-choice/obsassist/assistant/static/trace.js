// Assistant trace: every exchange behind every decision, newest first. Click a row for its JSON in and out.
"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const KIND = {
  observation: ["observation", ""], planner: ["planner", "good"], system1: ["System 1 · AnyJev", "s1"],
  system2: ["System 2 · GPT", "llm"], tool: ["tool", "llm"], console: ["console", "warn"], recommendation: ["recommendation", "good"],
};
let last = 0;
let topDecision;

// JSON as readable HTML: multi-line strings (prompts, answers) as text blocks, deep objects folded
function jhtml(v, depth = 0) {
  if (v === null || v === undefined) return '<span class="j-null">null</span>';
  if (typeof v === "number") return `<span class="j-num">${v}</span>`;
  if (typeof v === "boolean") return `<span class="j-num">${v}</span>`;
  if (typeof v === "string") {
    return v.includes("\n") || v.length > 140 ? `<pre class="j-block">${esc(v)}</pre>` : `<span class="j-str">"${esc(v)}"</span>`;
  }
  if (Array.isArray(v)) {
    if (!v.length) return "[]";
    if (v.every((x) => x === null || typeof x !== "object") && JSON.stringify(v).length < 120) {
      return "[" + v.map((x) => jhtml(x, depth + 1)).join(", ") + "]";
    }
    const body = v.map((x, i) => `<div><span class="j-key">${i}</span> ${jhtml(x, depth + 1)}</div>`).join("");
    return depth >= 2 ? `<details><summary>[${v.length} items]</summary><div class="j-obj">${body}</div></details>` : `<div class="j-obj">${body}</div>`;
  }
  const keys = Object.keys(v);
  if (!keys.length) return "{}";
  const body = keys.map((k) => `<div><span class="j-key">${esc(k)}</span>: ${jhtml(v[k], depth + 1)}</div>`).join("");
  return depth >= 3 ? `<details><summary>{${keys.length} keys}</summary><div class="j-obj">${body}</div></details>` : `<div class="j-obj">${body}</div>`;
}

function shown(kind) {
  const box = document.querySelector(`#filters input[value="${kind}"]`);
  return !box || box.checked;
}

function row(e) {
  const [label, cls] = KIND[e.kind] || [e.kind, ""];
  const d = document.createElement("div");
  d.className = "tr-row";
  d.dataset.kind = e.kind;
  d.hidden = !shown(e.kind);
  d.innerHTML = `<div class="tr-head"><span class="mono small muted">${esc(e.ut)}</span>
    <span class="pill ${cls}">${esc(label)}</span><span class="tr-sum">${esc(e.summary)}</span>
    <span class="mono small muted">${e.latency_s != null ? e.latency_s.toFixed(2) + " s" : ""}</span></div>`;
  d.querySelector(".tr-head").onclick = () => {
    let b = d.querySelector(".tr-body");
    if (b) { b.hidden = !b.hidden; return; }
    b = document.createElement("div");
    b.className = "tr-body";
    b.innerHTML = `<div><h4>in <button class="small" data-k="in">copy JSON</button></h4>${jhtml(e.in)}</div>
      <div><h4>out <button class="small" data-k="out">copy JSON</button></h4>${jhtml(e.out)}</div>`;
    for (const btn of b.querySelectorAll("button")) {
      btn.onclick = (ev) => { ev.stopPropagation(); navigator.clipboard.writeText(JSON.stringify(e[btn.dataset.k], null, 2)); };
    }
    d.appendChild(b);
  };
  return d;
}

function add(e) {
  const box = $("trace");
  if (e.decision !== topDecision) {
    const h = document.createElement("div");
    h.className = "tr-decision";
    h.textContent = e.decision != null ? `decision #${e.decision}` : "not tied to a decision (state questions, chat, night log)";
    box.prepend(h);
    topDecision = e.decision;
  }
  box.firstChild.after(row(e));
}

async function poll() {
  if ($("hold").checked) { setTimeout(poll, 1500); return; }
  try {
    const r = await (await fetch(`/api/trace?after=${last}`)).json();
    $("tracepath").textContent = r.path ? `· also written to ${r.path}` : "";
    for (const e of r.items) { add(e); last = e.id; }
  } catch (err) { /* the assistant is restarting */ }
  setTimeout(poll, 1500);
}

async function proposal() {
  try {
    const s = await (await fetch("/api/status")).json();
    const r = s.recommendation;
    $("proposal").textContent = r
      ? `#${r.id} ${r.utc} UT · ${r.action === "observe" ? `observe ${r.target}: ${r.n_exp} × ${Math.round(r.t_exp)} s` : r.action} · ${r.source}${r.level ? " " + r.level : ""} · ${r.status}` +
        ` · planner ${r.planner_pick || "–"} · System 1 ${r.s1_pick || (s.s1 ? "–" : "off")} · System 2 ${r.s2_pick || (s.s2 && s.s2.busy ? "thinking…" : "–")}`
      : "waiting for the first decision";
  } catch (err) { $("proposal").textContent = "assistant not reachable"; }
  setTimeout(proposal, 2000);
}

for (const box of document.querySelectorAll("#filters input[value]")) {
  box.onchange = () => { for (const d of document.querySelectorAll(`.tr-row[data-kind="${box.value}"]`)) d.hidden = !box.checked; };
}
poll();
proposal();
