// Observing Assistant UI: renders /api/status (pushed over /ws) and sends the observer's choices.
"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
let S = null;
let editing = new Set();

async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  return r.json().catch(() => ({}));
}

function renderHeader(s) {
  $("clock").textContent = `UT ${s.utc || "–"}  ·  local ${s.local || "–"}  ·  ${s.twilight || ""}  ·  ${fmt((s.minutes_left || 0) / 60, 1)} h left  ·  ${s.telescope || ""}`;
  $("adapter").textContent = s.adapter ? `${s.adapter.adapter}${s.adapter.can_actuate ? " · control ON" : ""}` : "";
  $("adapter").className = "pill" + (s.adapter && s.adapter.can_actuate ? " warn" : "");
  if (s.s1) {
    const lv = Object.values(s.s1.levels || {});
    const l2 = lv.filter((x) => x === "L2").length;
    $("s1info").textContent = `System 1: ${s.s1.model.replace("Qwen/", "")} · ${l2}/${lv.length} heads L2 · ${fmt(s.s1.last_latency_s, 1)} s`;
    $("s1info").className = "pill s1";
  } else { $("s1info").textContent = "System 1 off"; }
  if (s.s2) {
    $("s2info").textContent = s.s2.enabled ? `System 2: ${s.s2.chat_model} / ${s.s2.deliberate_model}${s.s2.busy ? " · thinking…" : ""}` : "System 2: no key";
    $("s2info").className = "pill llm";
    const u = s.s2.usage || {};
    $("s2usage").textContent = `${u.calls || 0} calls · ${((u.input_tokens || 0) / 1000).toFixed(1)}k in / ${((u.output_tokens || 0) / 1000).toFixed(1)}k out`;
  } else { $("s2info").textContent = "System 2 off"; }
  if (!editing.has("mode")) $("mode").value = s.settings.mode;
}

function renderNow(s) {
  const w = s.weather || {}, t = s.tcs || {}, g = s.guider || {};
  const lim = w.limits || {};
  const rows = [
    ["Dome", s.dome ? (s.dome.open ? "open" : `CLOSED (${s.dome.reason || ""})`) : "–"],
    ["Seeing (DIMM)", w.dimm != null ? `${fmt(w.dimm)}″` : "–"],
    ["Guider", g.fwhm != null ? `${fmt(g.fwhm)}″, flux ${fmt(g.flux_ratio)}` : (g.flux_ratio != null ? `flux ${fmt(g.flux_ratio)}` : "–")],
    ["Humidity", w.humidity != null ? `${fmt(w.humidity, 0)} % (limit ${fmt(lim.humidity, 0)})` : "–"],
    ["Wind", w.wind_mph != null ? `${fmt(w.wind_mph, 0)} mph (close ${fmt(lim.wind_close_mph, 0)})` : "–"],
    ["Telescope", `${t.name || "–"} · ${t.state || ""}`],
  ];
  for (const [k, a] of Object.entries(s.arms || {})) rows.push([`${k} arm`, `${a.state}${a.message ? " · " + a.message : ""}`]);
  $("nowgrid").innerHTML = rows.map(([k, v]) => `<div class="k">${esc(k)}</div><div>${esc(v)}</div>`).join("");
  const a = s.state_answers || {};
  const fmtq = (q, label) => {
    if (!q) return "";
    const val = q.p_true != null ? `p=${fmt(q.p_true)}` : `${esc(q.answer)} (${fmt(q.confidence)})`;
    return `<span class="pill ${q.level === "L2" ? "s1" : "warn"}" title="${esc(q.level)}">${label}: ${val}</span> `;
  };
  $("s1reads").innerHTML = a.sky ? `System 1 reads: ${fmtq(a.sky, "sky")}${fmtq(a.dome_risk, "dome closes <30 min")}` : "";
}

function renderProgress(s) {
  $("score").textContent = s.max_score ? `score ${fmt(s.score, 1)} / ${fmt(s.max_score, 0)}` : "";
  $("progress").innerHTML = (s.targets || []).map((t) => {
    const f = Math.min(1, (t.snr || 0) / (t.goal || 1));
    const cls = t.done ? "done" : (t.visible ? "" : "hidden");
    return `<div class="prow ${cls}"><div class="name" title="${esc(t.name)}">P${t.priority} ${esc(t.name)}</div>
      <div class="bar snr"><i style="width:${(f * 100).toFixed(0)}%"></i></div><div class="mono small">${fmt(t.snr, 0)}/${fmt(t.goal, 0)}</div></div>`;
  }).join("");
  const p = s.projection || {};
  $("proj").textContent = p.score != null ? `Projection (forecast): score ${fmt(p.score, 1)}/${fmt(p.max_score, 0)}${p.list_done_ut ? ` · list done ${p.list_done_ut} UT` : " · not everything finishes tonight"}` : "";
}

function renderRec(s) {
  const r = s.recommendation;
  if (!r) return;
  const srcCls = r.source === "system2" ? "llm" : r.source === "system1" ? "s1" : "";
  $("recsrc").textContent = { system1: "System 1 (AnyJev)", system2: "System 2 (GPT)", planner: "planner" }[r.source] || r.source;
  $("recsrc").className = "pill " + srcCls;
  $("reclevel").textContent = r.level || "";
  $("reclevel").className = "pill " + (r.level === "L2" ? "good" : r.level === "L0" ? "warn" : "");
  $("recconf").textContent = r.confidence != null ? `confidence ${fmt(r.confidence)}` : "";
  $("recstatus").textContent = r.status;
  $("recstatus").className = "pill " + (r.status === "open" ? "warn" : r.status === "executed" ? "good" : "");
  $("recmain").textContent = r.action === "observe" ? `Observe ${r.target}: ${r.n_exp} × ${fmt(r.t_exp, 0)} s` : (r.action === "wait" ? "Wait" : r.action);
  $("recwhy").textContent = r.rationale || "";
  $("recrisk").textContent = r.risk ? `Risk: ${r.risk}` : "";
  $("reccmds").textContent = (r.commands || []).join("\n");
  const lat = r.latency_s || {};
  $("picks").textContent = `planner: ${r.planner_pick || "–"} · System 1: ${r.s1_pick || "–"}${lat.system1 ? ` (${lat.system1}s)` : ""} · System 2: ${r.s2_pick || (s.s2 && s.s2.busy ? "thinking…" : "–")}${lat.system2 ? ` (${lat.system2}s)` : ""}`;
  $("accept").disabled = r.status !== "open";
}

function renderRank(s) {
  const js = (s.judgements || []).slice().sort((a, b) => a.expected_regret - b.expected_regret);
  const maxr = Math.max(0.05, ...js.map((j) => j.expected_regret || 0));
  const mmax = Math.max(1e-9, ...js.map((j) => j.merit || 0));
  $("rank").querySelector("tbody").innerHTML = js.map((j, k) => `<tr class="${k === 0 ? "best" : ""}">
    <td>${esc(j.name)}</td>
    <td><div class="bar" title="${fmt(j.expected_regret * 100, 2)}% of the night"><i style="width:${(100 * j.expected_regret / maxr).toFixed(0)}%"></i></div>
        <span class="mono small">${fmt(j.expected_regret * 100, 2)}%</span></td>
    <td class="mono">${fmt(j.p_best)}</td><td class="mono">${fmt(100 * (j.merit || 0) / mmax, 0)}</td><td>${esc(j.level)}</td></tr>`).join("");
}

function renderHist(s) {
  const st = s.stats || {};
  $("stats").textContent = `${st.decisions || 0} decisions · System 1 = planner ${st.agree_s1_planner || 0} · System 2 calls ${st.s2_calls || 0} · executed ${st.executed || 0}`;
  $("hist").querySelector("tbody").innerHTML = (s.history || []).slice().reverse().slice(0, 25).map((r) => `<tr>
    <td class="mono">${esc(r.utc)}</td><td>${esc(r.target || r.action)}</td><td>${esc(r.source)}${r.level ? " " + esc(r.level) : ""}</td>
    <td class="mono">${fmt(r.confidence)}</td><td>${esc(r.planner_pick)}</td><td>${esc(r.s1_pick)}</td><td>${esc(r.s2_pick)}</td><td>${esc(r.status)}</td></tr>`).join("");
}

function renderAlerts(s) {
  $("alerts").innerHTML = (s.alerts || []).slice().reverse().map((a) =>
    `<div class="alert ${esc(a.level)}"><span class="mono small muted">${esc(a.utc)}</span> ${esc(a.text)}</div>`).join("") || '<div class="muted small">No alerts.</div>';
  $("errors").textContent = (s.errors || []).slice(-3).join("\n");
}

function renderSettings(s) {
  if (!editing.has("tau")) $("tau").value = s.settings.s1_override_tau;
  if (!editing.has("escalate")) $("escalate").checked = s.settings.escalate;
  if (!editing.has("effort")) $("effort").value = s.settings.s2_effort;
  $("statetext").textContent = s.state_text || "";
  const manual = s.adapter && s.adapter.adapter === "manual";
  $("manualcard").hidden = !manual;
  if (manual && $("r_target").options.length !== (s.targets || []).length) {
    $("r_target").innerHTML = (s.targets || []).map((t) => `<option>${esc(t.name)}</option>`).join("");
  }
}

function render(s) {
  S = s;
  renderHeader(s); renderNow(s); renderProgress(s); renderRec(s); renderRank(s); renderHist(s); renderAlerts(s); renderSettings(s);
}

function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onmessage = (e) => { try { render(JSON.parse(e.data)); } catch (err) { console.error(err); } };
  ws.onclose = () => setTimeout(connect, 1500);
}

// ---- controls
for (const id of ["tau", "escalate", "effort", "mode"]) {
  $(id).addEventListener("focus", () => editing.add(id));
  $(id).addEventListener("blur", () => editing.delete(id));
}
$("mode").addEventListener("change", async (e) => {
  if (e.target.value === "autopilot" && !confirm("Autopilot sends commands to the console when it is confident and the console has assistant control enabled. Continue?")) {
    e.target.value = "advise"; return;
  }
  await post("/api/settings", { mode: e.target.value }); editing.delete("mode");
});
$("tau").addEventListener("change", (e) => post("/api/settings", { s1_override_tau: parseFloat(e.target.value) }));
$("escalate").addEventListener("change", (e) => post("/api/settings", { escalate: e.target.checked }));
$("effort").addEventListener("change", (e) => post("/api/settings", { s2_effort: e.target.value }));
$("accept").addEventListener("click", async () => { const r = await post("/api/accept", { execute: true }); addMsg("ai", r.reply); });
$("dismiss").addEventListener("click", () => post("/api/dismiss"));
$("deliberate").addEventListener("click", async () => { const r = await post("/api/deliberate"); addMsg("ai", r.reply || r.detail || ""); });
$("nightmode").addEventListener("click", () => document.body.classList.toggle("night"));
$("nightlog").addEventListener("click", async () => {
  addMsg("me", "(night log)");
  const r = await post("/api/nightlog");
  addMsg("ai", r.reply || r.detail || "(no reply)");
});

function addMsg(who, text) {
  const d = document.createElement("div");
  d.className = "msg " + who; d.textContent = text;
  $("chatlog").appendChild(d); $("chatlog").scrollTop = 1e9;
}
$("chatform").addEventListener("submit", async (e) => {
  e.preventDefault();
  const t = $("chatin").value.trim(); if (!t) return;
  $("chatin").value = ""; addMsg("me", t);
  const pending = document.createElement("div"); pending.className = "msg ai muted"; pending.textContent = "…";
  $("chatlog").appendChild(pending);
  const r = await post("/api/chat", { text: t });
  pending.remove(); addMsg("ai", r.reply || "(no reply)");
});

// manual reports
const num = (id) => { const v = $(id).value; return v === "" ? null : parseFloat(v); };
$("r_cond").addEventListener("click", () => post("/api/report/conditions", { seeing: num("r_seeing"), cloud: num("r_cloud"), humidity: num("r_hum"), wind_mph: num("r_wind") }));
$("r_on").addEventListener("click", () => post("/api/report/on_target", { name: $("r_target").value }));
$("r_start").addEventListener("click", () => post("/api/report/started", { name: $("r_target").value, t_exp: num("r_texp") || 1200 }));
$("r_done").addEventListener("click", () => post("/api/report/done", { name: $("r_target").value, t_exp: num("r_texp"), snr: num("r_snr") }));
$("r_abort").addEventListener("click", () => post("/api/report/done", { name: $("r_target").value, aborted: true }));

connect();
