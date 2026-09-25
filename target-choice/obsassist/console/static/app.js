/* obsassist console — core: state, WebSocket, REST helpers, status bar, tabs,
   night log / command line, debrief. Vanilla JS, no dependencies.
   Other files (charts.js, gui.js, weather.js, plan.js, data.js) register modules
   with OA.register({init, snap, reset, nightData, tab, redraw}). */
'use strict';

(function () {
  const OA = window.OA = {
    snap: null,          // latest snapshot
    night: null,         // /api/night
    program: null,       // /api/program
    programs: [],
    hist: null,          // weather/guider history (arrays)
    selTarget: null,     // selected target name
    tab: 'weather',
    nightMode: false,
    modules: [],
    expectReset: false,
    utBase: null,        // UT minute-of-day at sim t=0
  };

  // ------------------------------------------------------------------ DOM helpers
  OA.$ = (id) => document.getElementById(id);
  OA.h = function (tag, attrs, ...kids) {
    const el = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      const v = attrs[k];
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else if (k === 'html') el.innerHTML = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else if (k === 'style') el.style.cssText = v;
      else el.setAttribute(k, v === true ? '' : v);
    }
    for (const c of kids.flat()) if (c != null) el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    return el;
  };
  OA.setText = (el, s) => { s = s == null ? '' : String(s); if (el && el.textContent !== s) el.textContent = s; };
  OA.setCls = (el, c) => { if (el && el.className !== c) el.className = c; };
  OA.setAttr = (el, k, v) => { v = v == null ? '' : String(v); if (el && el.getAttribute(k) !== v) el.setAttribute(k, v); };
  OA.esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  OA.num = (v, d = 1) => (v == null || !isFinite(v)) ? '—' : Number(v).toFixed(d);
  OA.hm = (min) => {
    if (min == null || !isFinite(min)) return '—';
    const neg = min < 0; min = Math.abs(Math.round(min));
    return (neg ? '-' : '') + Math.floor(min / 60) + ':' + String(min % 60).padStart(2, '0');
  };
  OA.mmss = (s) => {
    if (s == null || !isFinite(s)) return '—';
    s = Math.max(0, Math.round(s));
    return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
  };
  OA.clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  OA.store = {
    get(k, d) { try { const v = localStorage.getItem('oa.' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('oa.' + k, JSON.stringify(v)); } catch (e) { /* ignore */ } },
  };

  // ------------------------------------------------------------------ colours / night mode
  const colCache = new Map();
  function parseColor(c) {
    if (c[0] === '#') {
      let h = c.slice(1);
      if (h.length === 3 || h.length === 4) h = h.split('').map((x) => x + x).join('');
      const a = h.length === 8 ? parseInt(h.slice(6, 8), 16) / 255 : 1;
      return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16), a];
    }
    const m = c.match(/rgba?\(([^)]+)\)/);
    if (m) { const p = m[1].split(',').map(parseFloat); return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1]; }
    return [128, 128, 128, 1];
  }
  /** Map any colour to a red-only equivalent (by luminance) in night mode. */
  OA.col = function (c) {
    if (!OA.nightMode || !c) return c;
    let v = colCache.get(c);
    if (v) return v;
    const [r, g, b, a] = parseColor(c);
    const L = 0.3 * r + 0.59 * g + 0.11 * b;
    const R = Math.round(70 + L * 0.72);
    v = a < 1 ? `rgba(${R},0,0,${a})` : `rgb(${R},0,0)`;
    colCache.set(c, v);
    return v;
  };
  /** Canvas theme tokens. */
  OA.th = function () {
    return OA.nightMode
      ? { bg: '#000', fg: '#c80000', dim: '#6a0000', grid: '#260000', axis: '#7a0000', panel: '#050000', sky: '#000', skyRing: '#3a0000', warn: '#ff0000' }
      : { bg: '#fff', fg: '#111', dim: '#666', grid: '#e4e4e4', axis: '#444', panel: '#f5f5f5', sky: '#0b1426', skyRing: '#2d3e5e', warn: '#e00000' };
  };
  OA.setNightMode = function (on) {
    OA.nightMode = !!on;
    document.documentElement.classList.toggle('night', OA.nightMode);
    OA.$('tg-night').checked = OA.nightMode;
    OA.store.set('night', OA.nightMode);
    OA.redrawAll(true);
  };

  // ------------------------------------------------------------------ REST
  OA.api = async function (path, opts = {}) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      let msg = r.statusText;
      try { const j = await r.json(); msg = j.detail || j.error || JSON.stringify(j); } catch (e) { /* not json */ }
      const err = new Error(`HTTP ${r.status}: ${msg}`);
      err.status = r.status;
      throw err;
    }
    return opts.raw ? r : r.json();
  };
  OA.post = (path, body) => OA.api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  /** Send a console command; echo it and its reply in the terminal. */
  // Commands are serialised so that each echo is followed by its own reply in the log.
  let cmdChain = Promise.resolve();
  OA.cmd = function (text, opts = {}) {
    text = String(text).trim();
    if (!text) return Promise.resolve('');
    const run = async () => {
      if (opts.echo !== false) OA.term.echo(text);
      try {
        const r = await OA.post('/api/cmd', { cmd: text, source: 'observer' });
        OA.term.reply(r.reply, text);
        return r.reply;
      } catch (e) {
        OA.term.reply('error: ' + e.message, text);
        return 'error: ' + e.message;
      }
    };
    const p = cmdChain.then(run, run);
    cmdChain = p.catch(() => {});
    return p;
  };

  // ------------------------------------------------------------------ module registry
  OA.register = (m) => { OA.modules.push(m); return m; };
  function each(fn, ...args) {
    for (const m of OA.modules) {
      if (typeof m[fn] !== 'function') continue;
      try { m[fn](...args); } catch (e) { console.error(`module ${m.name || '?'}.${fn}:`, e); }
    }
  }
  OA.redrawAll = (force) => each('redraw', !!force);

  // ------------------------------------------------------------------ time helpers
  /** UT minute-of-day of simulation minute t. */
  OA.utMin = function (t) {
    if (OA.utBase == null) return null;
    return ((OA.utBase + t) % 1440 + 1440) % 1440;
  };
  OA.fmtUT = function (t, sec) {
    const m = OA.utMin(t);
    if (m == null) return '—';
    const tot = Math.round(m * (sec ? 60 : 1));
    if (sec) return String(Math.floor(tot / 3600) % 24).padStart(2, '0') + ':' + String(Math.floor(tot / 60) % 60).padStart(2, '0') + ':' + String(tot % 60).padStart(2, '0');
    return String(Math.floor(tot / 60) % 24).padStart(2, '0') + ':' + String(tot % 60).padStart(2, '0');
  };
  /** Simulation minute of a "HH:MM" UT string, choosing the occurrence nearest to tRef. */
  OA.tOfUT = function (hhmm, tRef) {
    if (OA.utBase == null || !hhmm) return null;
    const p = hhmm.split(':').map(Number);
    const m = p[0] * 60 + p[1] + (p[2] || 0) / 60;
    let t = m - OA.utBase;
    const ref = tRef == null ? (OA.snap ? OA.snap.t : 0) : tRef;
    while (t < ref - 720) t += 1440;
    while (t > ref + 720) t -= 1440;
    return t;
  };
  function calibrateUT(s) {
    const hms = s.utc.slice(11).split(':').map(Number);
    OA.utBase = hms[0] * 60 + hms[1] + hms[2] / 60 - s.t;
  }

  /** Twilight times (sim minutes) from the night sun-altitude curve. */
  OA.twilights = function () {
    const n = OA.night;
    if (!n) return null;
    if (n._twi) return n._twi;
    const t = n.t, a = n.sun_alt, out = {};
    const cross = (lvl, down) => {
      for (let i = 1; i < a.length; i++) {
        if (down ? (a[i - 1] >= lvl && a[i] < lvl) : (a[i - 1] < lvl && a[i] >= lvl)) {
          return t[i - 1] + (t[i] - t[i - 1]) * (lvl - a[i - 1]) / (a[i] - a[i - 1]);
        }
      }
      return null;
    };
    out.sunset = cross(-0.833, true); out.dusk12 = cross(-12, true); out.dusk18 = cross(-18, true);
    out.dawn18 = cross(-18, false); out.dawn12 = cross(-12, false); out.sunrise = cross(-0.833, false);
    out.t0 = out.sunset != null ? out.sunset : t[0];
    out.t1 = out.sunrise != null ? out.sunrise : t[t.length - 1];
    n._twi = out;
    return out;
  };

  // ------------------------------------------------------------------ data flow
  let lastHistLen = 0;
  function extendHistory(s) {
    const h = OA.hist;
    if (!h || !h.t) return;
    const n = h.t.length;
    const lastT = n ? h.t[n - 1] : -1e9;
    if (s.t - lastT < 1) return;
    const w = s.weather, g = s.guider;
    const guiding = g && g.state === 'guiding';
    h.t.push(Math.round(s.t * 100) / 100);
    h.utc.push(s.utc.slice(11, 16));
    h.dimm.push(w.dimm);
    h.guider_fwhm.push(guiding ? g.fwhm : null);
    h.flux_ratio.push(guiding ? g.flux_ratio : null);
    h.humidity.push(w.humidity);
    h.wind.push(w.wind_ms);
    h.gust.push(w.gust_mph / 2.237);
    h.temp.push(w.temp);
    h.dewpoint.push(w.dewpoint);
    h.pressure.push(w.pressure);
    h.wind_dir.push(w.wind_dir);
    OA.histVersion++;
  }
  OA.histVersion = 0;

  function handleSnap(s) {
    const prev = OA.snap;
    const isReset = OA.expectReset || (prev && (s.program !== prev.program || s.t < prev.t - 0.5));
    calibrateUT(s);
    if (isReset) {
      OA.expectReset = false;
      OA.snap = s;
      doReset();
      return;
    }
    OA.snap = s;
    if (s.history) { OA.hist = s.history; OA.histVersion++; lastHistLen = s.history.t.length; }
    else extendHistory(s);
    each('snap', s, prev);
  }

  async function loadStatic() {
    const [night, program] = await Promise.all([
      OA.api('/api/night').catch((e) => { console.warn(e); return null; }),
      OA.api('/api/program').catch((e) => { console.warn(e); return null; }),
    ]);
    OA.night = night;
    OA.program = program;
    if (program) {
      if (program.key) OA.$('prog-sel').value = program.key;
      if (program.seed != null) OA.$('prog-seed').value = program.seed;
    }
    each('nightData', night, program);
  }

  async function doReset() {
    OA.selTarget = null;
    OA.hist = null;
    OA.term.clear();
    each('reset');
    OA.term.sys('— night reset — ' + (OA.snap ? OA.snap.program : ''));
    await loadStatic();
    try { OA.hist = await OA.api('/api/history'); OA.histVersion++; } catch (e) { /* comes with the ws */ }
    if (OA.snap) each('snap', OA.snap, null);
  }

  // WebSocket with auto-reconnect; polls /api/state while the socket is down.
  let ws = null, wsTries = 0, pollTimer = null, pollN = 0;
  function setConn(state, title) {
    const el = OA.$('conn');
    OA.setCls(el, 'conn ' + state);
    el.title = title;
  }
  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
      try {
        const s = await OA.api('/api/state' + (pollN++ % 10 === 0 ? '?full=true' : ''));
        handleSnap(s);
        setConn('poll', 'WebSocket down — polling /api/state');
      } catch (e) {
        setConn('off', 'server unreachable: ' + e.message);
      }
    }, 1500);
  }
  function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }
  function connect() {
    const url = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
    try { ws = new WebSocket(url); } catch (e) { scheduleReconnect(); return; }
    ws.onopen = () => { wsTries = 0; stopPolling(); setConn('on', 'WebSocket connected'); };
    ws.onmessage = (ev) => {
      let s;
      try { s = JSON.parse(ev.data); } catch (e) { console.warn('bad ws message', e); return; }
      try { handleSnap(s); } catch (e) { console.error('snapshot handling failed', e); }
    };
    ws.onclose = () => { setConn('off', 'WebSocket closed — reconnecting'); scheduleReconnect(); };
    ws.onerror = () => { try { ws.close(); } catch (e) { /* ignore */ } };
  }
  function scheduleReconnect() {
    ws = null;
    wsTries++;
    if (wsTries >= 2) startPolling();
    setTimeout(connect, Math.min(8000, 800 * wsTries));
  }

  // ------------------------------------------------------------------ status bar
  const SPEEDS = [1, 10, 60, 120, 300, 600];
  const statusBar = OA.register({
    name: 'statusbar',
    init() {
      const g = OA.$('sim-speeds');
      for (const v of SPEEDS) {
        g.appendChild(OA.h('button', { class: 'btn', 'data-speed': v, title: `simulated clock ${v}x real time`, onclick: () => OA.post('/api/sim', { speed: v }).catch(alertErr) }, v + 'x'));
      }
      OA.$('sim-pause').onclick = () => {
        const p = OA.snap ? !OA.snap.paused : true;
        OA.post('/api/sim', { paused: p }).catch(alertErr);
      };
      OA.$('sim-ff').onclick = () => OA.cmd('ff');
      OA.$('prog-reset').onclick = async () => {
        const program = OA.$('prog-sel').value, seed = parseInt(OA.$('prog-seed').value || '0', 10) || 0;
        if (!confirm(`Reset the night?\n\nProgram: ${program}\nSeed: ${seed}\n\nEverything observed so far is discarded.`)) return;
        try {
          OA.expectReset = true;
          await OA.post('/api/reset', { program, seed });
        } catch (e) { OA.expectReset = false; alertErr(e); }
      };
      OA.$('tg-assist').onchange = (e) => {
        OA.post('/api/sim', { assistant_control: e.target.checked }).catch(alertErr);
      };
      OA.$('tg-night').onchange = (e) => OA.setNightMode(e.target.checked);
      OA.$('btn-debrief').onclick = debrief;
    },
    nightData(n, p) {
      if (p) OA.$('prog-sel').title = p.description || '';
    },
    snap(s) {
      OA.setText(OA.$('sb-tel'), s.telescope);
      OA.setText(OA.$('sb-inst'), s.instrument);
      OA.setText(OA.$('sb-site'), s.site);
      const pe = OA.$('sb-program');
      OA.setText(pe, s.program);
      OA.setAttr(pe, 'title', s.program + (OA.program && OA.program.description ? '\n\n' + OA.program.description : ''));
      OA.setText(OA.$('sb-ut'), s.utc.slice(11));
      OA.setText(OA.$('sb-date'), s.utc.slice(0, 10));
      OA.setText(OA.$('sb-lt'), s.local);
      OA.setText(OA.$('sb-lst'), s.lst);
      OA.setText(OA.$('sb-sun'), (s.sun_alt >= 0 ? '+' : '') + s.sun_alt.toFixed(1) + '°');
      const tw = s.twilight || '';
      const twc = tw.startsWith('day') ? 'day' : tw.startsWith('civil') ? 'civil' : tw.startsWith('nautical') ? 'nautical' : tw.startsWith('astro') ? 'astro' : 'night';
      const twe = OA.$('sb-twi');
      OA.setText(twe, tw);
      OA.setCls(twe, 'twi ' + twc);
      OA.setText(OA.$('sb-moon'), (s.moon.alt >= 0 ? '+' : '') + s.moon.alt.toFixed(1) + '° az ' + Math.round(s.moon.az));
      OA.setText(OA.$('sb-moonill'), Math.round(s.moon.illum * 100) + '%' + (s.moon.alt < 0 ? ' (down)' : ''));
      const de = OA.$('sb-dome');
      OA.setText(de, s.dome.open ? 'DOME OPEN' : 'DOME CLOSED');
      OA.setCls(de, 'pill ' + (s.dome.open ? 'ok' : 'bad'));
      const why = s.dome.open ? '' : (s.dome.reason || '');
      OA.setText(OA.$('sb-domewhy'), why);
      OA.setAttr(OA.$('sb-domewhy'), 'title', why);
      const nt = s.night;
      const toStart = nt.start - s.t;
      OA.setText(OA.$('sb-left'), toStart > 0 ? OA.hm(nt.end - nt.start) + ' (in ' + OA.hm(toStart) + ')' : OA.hm(nt.minutes_left));
      OA.setText(OA.$('sb-twi12'), nt.twi12_dusk + '–' + nt.twi12_dawn);
      OA.setText(OA.$('sb-score'), OA.num(s.score, 1) + ' / ' + OA.num(s.max_score, 0));
      const pr = s.plan && s.plan.projection;
      OA.setText(OA.$('sb-proj'), pr ? OA.num(pr.score, 1) + (pr.list_done_ut ? ' · all by ' + pr.list_done_ut : '') : '—');
      const pb = OA.$('sim-pause');
      OA.setText(pb, s.paused ? '▶ Resume' : '❚❚ Pause');
      OA.setCls(pb, 'btn' + (s.paused ? ' on' : ''));
      for (const b of OA.$('sim-speeds').children) {
        OA.setCls(b, 'btn' + (Number(b.dataset.speed) === s.speed ? ' on' : ''));
      }
      const ac = OA.$('tg-assist');
      if (ac.checked !== !!s.assistant_control) ac.checked = !!s.assistant_control;
      document.title = `${s.utc.slice(11, 16)} UT · ${s.telescope} ${s.instrument} · obsassist`;
    },
  });
  function alertErr(e) { OA.term.reply('error: ' + (e && e.message ? e.message : e)); }

  // ------------------------------------------------------------------ tabs
  const tabs = OA.register({
    name: 'tabs',
    init() {
      for (const b of document.querySelectorAll('#tabbar .tab')) b.onclick = () => OA.showTab(b.dataset.tab);
      OA.$('tab-wide').onclick = () => {
        const m = OA.$('main');
        m.classList.toggle('wide');
        OA.store.set('wide', m.classList.contains('wide'));
        setTimeout(() => OA.redrawAll(true), 30);
      };
      if (OA.store.get('wide', false)) OA.$('main').classList.add('wide');
      OA.showTab(OA.store.get('tab', 'weather'));
    },
  });
  // the assistant watching this console announces its URL; this tab embeds its live trace
  OA.register({
    name: 'assistant',
    async tab(n) {
      if (n !== 'assistant' || OA.$('as-frame').src) return;
      const r = await OA.api('/api/assistant').catch(() => ({}));
      if (!r.url) return;
      const u = new URL('/trace', r.url);  // a loopback assistant is reached under this page's host name
      if (['127.0.0.1', 'localhost'].includes(u.hostname)) u.hostname = location.hostname;
      OA.$('as-frame').src = u.href;
      OA.$('as-frame').hidden = false;
      OA.$('as-none').hidden = true;
    },
  });
  OA.showTab = function (name) {
    if (!OA.$('tab-' + name)) name = 'weather';
    OA.tab = name;
    for (const b of document.querySelectorAll('#tabbar .tab')) b.classList.toggle('active', b.dataset.tab === name);
    for (const p of document.querySelectorAll('.tabpane')) p.classList.toggle('active', p.id === 'tab-' + name);
    OA.store.set('tab', name);
    each('tab', name);
  };

  // ------------------------------------------------------------------ night log / terminal
  const WHO = ['TO', 'OBS', 'AST', 'DATA', 'QL', 'ALERT', 'SYS'];
  const term = OA.term = OA.register({
    name: 'term',
    seen: new Set(),
    hidden: new Set(OA.store.get('logHidden', [])),
    hist: OA.store.get('cmdHist', []),
    hpos: -1,
    count: 0,
    init() {
      const f = OA.$('log-filters');
      for (const w of WHO) {
        const c = OA.h('span', { class: 'fchip' + (this.hidden.has(w) ? '' : ' on'), title: `show/hide ${w} messages` }, w);
        c.onclick = () => {
          if (this.hidden.has(w)) this.hidden.delete(w); else this.hidden.add(w);
          c.classList.toggle('on', !this.hidden.has(w));
          OA.store.set('logHidden', [...this.hidden]);
          for (const el of OA.$('term-out').querySelectorAll('.w-' + w)) el.classList.toggle('hide', this.hidden.has(w));
        };
        f.appendChild(c);
      }
      const inp = OA.$('term-in');
      OA.$('term-form').onsubmit = (e) => {
        e.preventDefault();
        const v = inp.value.trim();
        inp.value = '';
        this.hpos = -1;
        if (!v) return;
        if (this.hist[this.hist.length - 1] !== v) { this.hist.push(v); if (this.hist.length > 200) this.hist.shift(); OA.store.set('cmdHist', this.hist); }
        if (v === 'clear' || v === 'cls') { this.clear(); return; }
        if (v === 'help') {
          this.echo(v);
          this.local('console: clear · night (toggle night mode) · tab <weather|sky|targets|etc|data> · select <target>');
          OA.cmd('help', { echo: false });
          return;
        }
        if (v === 'night') { this.echo(v); OA.setNightMode(!OA.nightMode); return; }
        if (v.startsWith('tab ')) { this.echo(v); OA.showTab(v.slice(4).trim()); return; }
        if (v.startsWith('select ')) { this.echo(v); OA.selectTarget(v.slice(7).trim(), true); return; }
        OA.cmd(v);
      };
      inp.onkeydown = (e) => {
        if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
          if (!this.hist.length) return;
          e.preventDefault();
          if (e.key === 'ArrowUp') this.hpos = this.hpos < 0 ? this.hist.length - 1 : Math.max(0, this.hpos - 1);
          else this.hpos = this.hpos < 0 ? -1 : this.hpos + 1;
          if (this.hpos >= this.hist.length) this.hpos = -1;
          inp.value = this.hpos < 0 ? '' : this.hist[this.hpos];
          setTimeout(() => inp.setSelectionRange(inp.value.length, inp.value.length), 0);
        } else if (e.key === 'Escape') { inp.value = ''; this.hpos = -1; }
      };
      document.addEventListener('keydown', (e) => {
        if (e.key === '/' && !/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) { e.preventDefault(); inp.focus(); }
      });
    },
    atBottom() { const o = OA.$('term-out'); return o.scrollHeight - o.scrollTop - o.clientHeight < 30; },
    push(el) {
      const o = OA.$('term-out');
      const stick = this.atBottom();
      o.appendChild(el);
      if (++this.count > 4000) { o.removeChild(o.firstChild); this.count--; }
      if (stick) o.scrollTop = o.scrollHeight;
    },
    line(e) {
      const lvl = e.level || 'info';
      const el = OA.h('span', { class: `ll w-${e.who} l-${lvl}` + (this.hidden.has(e.who) ? ' hide' : '') },
        OA.h('span', { class: 'lt' }, e.utc + ' '), OA.h('span', { class: 'lw' }, e.who), ' ' + e.text);
      this.push(el);
    },
    echo(text) {
      this.push(OA.h('span', { class: 'll cmd' }, OA.h('span', { class: 'lt' }, (OA.snap ? OA.snap.utc.slice(11) : '') + ' '), OA.h('span', { class: 'lw' }, '>'), ' ' + text));
    },
    reply(text, cmd) {
      if (text == null || text === '') return;
      const err = /^(error|refused|parse error)/i.test(text) || /can't|below the limit|busy/.test(text);
      if (text === 'ok' || text.startsWith('ok;') || /^(ok; )*ok$/.test(text)) {
        this.push(OA.h('span', { class: 'll reply' }, '         ✓ ok'));
        return;
      }
      let t = text;
      if (cmd && cmd.startsWith('etc') && text[0] === '{') {
        try { t = JSON.stringify(JSON.parse(text), null, 1).replace(/\n\s*/g, ' '); } catch (e) { /* raw */ }
      }
      this.push(OA.h('span', { class: 'll reply' + (err ? ' err' : '') }, '         → ' + t));
    },
    local(text) { this.push(OA.h('span', { class: 'll reply' }, '         ' + text)); },
    sys(text) { this.push(OA.h('span', { class: 'll w-SYS' }, text)); },
    clear() { OA.$('term-out').textContent = ''; this.count = 0; this.seen.clear(); },
    reset() { this.seen.clear(); },
    snap(s) {
      for (const e of s.log) {
        const k = e.t + '|' + e.who + '|' + e.text;
        if (this.seen.has(k)) continue;
        this.seen.add(k);
        this.line(e);
      }
      if (this.seen.size > 6000) this.seen = new Set([...this.seen].slice(-3000));
    },
  });

  // ------------------------------------------------------------------ target selection (shared)
  OA.selectTarget = function (name, fromUser) {
    if (!name) return;
    if (OA.snap) {
      const n = name.toLowerCase();
      const hit = OA.snap.targets.find((t) => t.name.toLowerCase() === n) || OA.snap.targets.find((t) => t.name.toLowerCase().includes(n));
      if (hit) name = hit.name;
      else if (fromUser) { OA.term.local('no target matching ' + name); return; }
    }
    OA.selTarget = name;
    each('select', name, !!fromUser);
  };

  // ------------------------------------------------------------------ modal + debrief
  OA.modal = function (title, bodyEl) {
    OA.setText(OA.$('modal-title'), title);
    const b = OA.$('modal-body');
    b.textContent = '';
    if (bodyEl) b.appendChild(bodyEl);
    OA.$('modal').classList.remove('hidden');
  };
  OA.closeModal = () => OA.$('modal').classList.add('hidden');

  function debrief() {
    const box = OA.h('div', null,
      OA.h('p', null, 'The debrief runs the hindsight oracle: it uses the ', OA.h('b', null, 'true'), ' weather of the whole night (which you cannot know during the night) to find the best sequence, and compares it with the greedy planner and with what you have done so far.'),
      OA.h('p', null, 'Looking at it before the night is over spoils the exercise. Continue?'),
      OA.h('div', { class: 'modal-actions' },
        OA.h('button', { class: 'tk-btn', onclick: OA.closeModal }, 'Cancel'),
        OA.h('button', { class: 'tk-btn start', onclick: runDebrief }, 'Reveal the truth')));
    OA.modal('Debrief — reveals the truth', box);
  }
  async function runDebrief() {
    OA.modal('Debrief — hindsight oracle', OA.h('div', null, OA.h('span', { class: 'spin' }), 'Running the hindsight oracle and the greedy planner… (this can take a few seconds)'));
    let d;
    try { d = await OA.api('/api/debrief'); } catch (e) {
      OA.modal('Debrief — error', OA.h('div', null, 'Debrief failed: ' + e.message));
      return;
    }
    const o = d.oracle || {}, you = d.you || {};
    const max = o.max_score || (OA.snap && OA.snap.max_score) || 1;
    const bar = (label, v, color, note) => [
      OA.h('span', null, label),
      OA.h('div', { class: 'bar' }, OA.h('i', { style: `width:${OA.clamp(100 * (v || 0) / max, 0, 100)}%;background:${color}` })),
      OA.h('span', { class: 'mono' }, OA.num(v, 1) + ' / ' + OA.num(max, 0) + (note ? ' ' + note : '')),
    ];
    const seqRows = (d.sequence || []).map((a) => OA.h('tr', { class: a.kind === 'observe' ? '' : 'muted' },
      OA.h('td', { class: 'num' }, a.ut), OA.h('td', null, a.kind), OA.h('td', null, a.target || ''),
      OA.h('td', { class: 'num' }, a.t_exp ? OA.num(a.t_exp, 0) : ''), OA.h('td', { class: 'num' }, a.snr != null ? OA.num(a.snr, 1) : '')));
    const w = d.weather || {};
    const evs = (w.events || []).map((e) => {
      const ut = OA.fmtUT(e.t_min);
      const extra = Object.keys(e).filter((k) => k !== 't_min' && k !== 'kind').map((k) => `${k}=${typeof e[k] === 'number' ? OA.num(e[k], 2) : e[k]}`).join(' ');
      return OA.h('li', null, `${ut} UT  ${e.kind} ${extra}`);
    });
    const body = OA.h('div', null,
      OA.h('div', { class: 'cmp' },
        bar('You (so far, ' + (you.at_ut || '') + ' UT)', you.score, '#e07a10', `· ${you.done || 0} done`),
        bar('Greedy planner (full night)', d.greedy_score, '#2a6fdb'),
        bar('Hindsight oracle', o.score, '#1d9a3a', o.done != null ? `· ${o.done} done` : ''),
        o.upper_bound != null ? bar('Upper bound', o.upper_bound, '#999') : null),
      OA.h('div', { class: 'dim' }, `oracle method ${o.method || '?'} · ${o.rollouts || '?'} rollouts · ${OA.num(o.seconds, 2)} s`),
      OA.h('h3', null, 'Weather truth'),
      OA.h('div', null, 'Regime: ', OA.h('b', null, w.regime || '?')),
      evs.length ? OA.h('ul', { class: 'mono small' }, evs) : OA.h('div', { class: 'dim' }, 'no weather events'),
      OA.h('h3', null, 'Hindsight-optimal sequence'),
      OA.h('table', { class: 'grid small' },
        OA.h('thead', null, OA.h('tr', null, OA.h('th', null, 'UT'), OA.h('th', null, 'action'), OA.h('th', null, 'target'), OA.h('th', { class: 'num' }, 't_exp s'), OA.h('th', { class: 'num' }, 'S/N'))),
        OA.h('tbody', null, seqRows)));
    OA.modal('Debrief — hindsight oracle vs you', body);
  }

  // ------------------------------------------------------------------ start
  OA.start = async function () {
    OA.nightMode = !!OA.store.get('night', false);
    document.documentElement.classList.toggle('night', OA.nightMode);
    OA.$('tg-night').checked = OA.nightMode;
    OA.$('modal-x').onclick = OA.closeModal;
    OA.$('modal').addEventListener('click', (e) => { if (e.target.id === 'modal') OA.closeModal(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') OA.closeModal(); });
    each('init');
    let rt = null;
    window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => OA.redrawAll(true), 120); });
    try {
      const progs = await OA.api('/api/programs');
      OA.programs = progs;
      const sel = OA.$('prog-sel');
      for (const p of progs) sel.appendChild(OA.h('option', { value: p }, p));
    } catch (e) { console.warn('programs', e); }
    await loadStatic();
    try { OA.hist = await OA.api('/api/history'); OA.histVersion++; } catch (e) { /* ws brings it */ }
    try { handleSnap(await OA.api('/api/state')); } catch (e) { setConn('off', 'server unreachable'); }
    connect();
  };
})();
