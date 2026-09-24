/* obsassist console — Sky & plan tab (alt-az sky map, airmass chart with the projected plan as a
   Gantt strip, projected completion, airmass planner), Targets tab and ETC tab. */
'use strict';

(function () {
  const OA = window.OA, h = OA.h;
  const D2R = Math.PI / 180;

  // ------------------------------------------------------------------ target status
  const ST = {
    obs: { c: '#ff8c00', label: 'on telescope' },
    vis: { c: '#22b14c', label: 'observable now' },
    up: { c: '#d4b000', label: 'up, not observable' },
    down: { c: '#8a8a8a', label: 'not up' },
    done: { c: '#5b7fb3', label: 'done' },
    too: { c: '#d63bd6', label: 'ToO not yet announced' },
  };
  OA.tstatus = function (x, s) {
    if (!x.announced) return 'too';
    if (x.done) return 'done';
    if (s.tcs.name === x.name && s.tcs.state !== 'parked') return 'obs';
    if (x.visible) return 'vis';
    if (x.alt > 0) return 'up';
    return 'down';
  };
  const unitLabel = (u) => ({ per_pix: '/pix', per_A: '/Å', per_res: '/res', per_resel: '/resel' }[u] || (u ? ' ' + u.replace('per_', '/') : ''));
  const sig3 = (v) => (v == null || !isFinite(v) ? '—' : Math.abs(v) >= 100 ? v.toFixed(0) : Number(v).toPrecision(3));
  const tIndex = (name) => (OA.snap ? OA.snap.targets.findIndex((x) => x.name === name) : -1);

  // ------------------------------------------------------------------ sky & plan tab
  const sky = OA.register({
    name: 'sky',
    snap(s) {
      if (OA.tab !== 'sky') return;
      if (!OA.every('sky', 1000)) return;
      this.draw();
    },
    tab(n) { if (n === 'sky') requestAnimationFrame(() => this.draw(true)); },
    redraw() { if (OA.tab === 'sky') this.draw(true); },
    select() { if (OA.tab === 'sky') this.draw(true); },
    nightData() { this.tableSig = null; },
    draw(force) {
      const s = OA.snap;
      if (!s) return;
      this.map(s);
      this.airmass(s);
      this.legend(s);
      this.projection(s);
      if (force || OA.every('amtable', 5000)) this.table(s);
    },

    map(s) {
      const fit = OA.fit(OA.$('sky-map'));
      if (!fit) return;
      const { ctx, w, h: hh } = fit;
      const th = OA.th();
      const n = OA.night;
      ctx.fillStyle = th.bg; ctx.fillRect(0, 0, w, hh);
      const cx = w / 2, cy = hh / 2 + 4, R = Math.min(w, hh) / 2 - 18;
      const rOf = (alt) => (90 - alt) / 90 * R;
      const P = (alt, az) => { const r = rOf(alt), a = az * D2R; return [cx - r * Math.sin(a), cy - r * Math.cos(a)]; };
      const ang = (az) => -Math.PI / 2 - az * D2R;          // canvas angle of an azimuth (E left)
      // sky disc
      const skyCol = OA.nightMode ? '#000' : s.sun_alt > -0.8 ? '#6d86ad' : s.sun_alt > -6 ? '#3f5b86' : s.sun_alt > -12 ? '#1f3052' : s.sun_alt > -18 ? '#131f38' : '#0a1224';
      ctx.fillStyle = skyCol;
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.fill();
      // elevation limit (all azimuths) + azimuth-dependent limits
      const elMin = n && n.limits ? n.limits.el_min : 15;
      ctx.fillStyle = OA.nightMode ? 'rgba(90,0,0,.35)' : 'rgba(210,50,50,.28)';
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.arc(cx, cy, rOf(elMin), 0, 2 * Math.PI, true); ctx.fill();
      for (const [lo, hi, el] of (n && n.limits ? n.limits.az_limits : [])) {
        ctx.beginPath();
        ctx.arc(cx, cy, rOf(elMin), ang(lo), ang(hi), true);
        ctx.arc(cx, cy, rOf(el), ang(hi), ang(lo), false);
        ctx.closePath(); ctx.fill();
        const mid = (lo + hi) / 2 + (lo > hi ? 180 : 0);
        const [lx, ly] = P((el + elMin) / 2, mid);
        OA.text(ctx, `el>${el}°`, lx, ly + 3, { size: 9, align: 'center', color: '#ff9a9a' });
      }
      // high-wind forbidden sector
      const W = s.weather;
      if (W.wind_mph >= W.limits.wind_mph) {
        ctx.fillStyle = OA.nightMode ? 'rgba(160,0,0,.35)' : 'rgba(255,40,40,.30)';
        ctx.beginPath(); ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, R, ang(W.wind_dir - 60), ang(W.wind_dir + 60), true);
        ctx.closePath(); ctx.fill();
        ctx.strokeStyle = OA.col('#ff3030'); ctx.lineWidth = 1.5; ctx.stroke();
        const [lx, ly] = P(40, W.wind_dir);
        OA.text(ctx, `wind ${Math.round(W.wind_mph)} mph`, lx, ly, { size: 10, bold: true, align: 'center', color: '#ff5050', halo: 'rgba(0,0,0,.6)' });
      }
      // grid
      ctx.strokeStyle = OA.nightMode ? '#300000' : 'rgba(150,180,230,.35)'; ctx.lineWidth = 1;
      for (const a of [30, 60]) { ctx.beginPath(); ctx.arc(cx, cy, rOf(a), 0, 2 * Math.PI); ctx.stroke(); }
      for (let az = 0; az < 360; az += 45) { const [x, y] = P(0, az); ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke(); }
      ctx.setLineDash([4, 3]); ctx.strokeStyle = OA.col('#ff5050');
      ctx.beginPath(); ctx.arc(cx, cy, rOf(elMin), 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = th.axis; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
      for (const [lab, az] of [['N', 0], ['E', 90], ['S', 180], ['W', 270]]) {
        const [x, y] = P(-9, az);
        OA.text(ctx, lab, x, y + 4, { size: 11, bold: true, align: 'center' });
      }
      OA.text(ctx, '30°', ...P(30, 200), { size: 8, color: '#8aa0c8' });
      OA.text(ctx, '60°', ...P(60, 200), { size: 8, color: '#8aa0c8' });
      // tracks for the next 3 h
      if (n) {
        const i0 = n.t.findIndex((t) => t >= s.t);
        const i1 = n.t.findIndex((t) => t >= s.t + 180);
        s.targets.forEach((x, i) => {
          const tc = n.targets[i];
          if (!tc || i0 < 0 || !x.announced) return;
          ctx.strokeStyle = OA.col(ST[OA.tstatus(x, s)].c); ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
          ctx.beginPath();
          let pen = false;
          for (let k = i0; k <= (i1 < 0 ? n.t.length - 1 : i1); k++) {
            if (tc.alt[k] < 0) { pen = false; continue; }
            const [px, py] = P(tc.alt[k], tc.az[k]);
            if (!pen) { ctx.moveTo(px, py); pen = true; } else ctx.lineTo(px, py);
          }
          ctx.stroke(); ctx.globalAlpha = 1;
        });
      }
      // moon
      if (s.moon.alt > -1) {
        const [mx, my] = P(Math.max(0, s.moon.alt), s.moon.az);
        ctx.fillStyle = OA.col(`rgba(244,232,160,${(0.15 + 0.85 * s.moon.illum).toFixed(2)})`);
        ctx.beginPath(); ctx.arc(mx, my, 7, 0, 2 * Math.PI); ctx.fill();
        ctx.strokeStyle = OA.col('#f4e8a0'); ctx.lineWidth = 1; ctx.stroke();
        OA.text(ctx, `Moon ${Math.round(s.moon.illum * 100)}%`, mx + 9, my + 3, { size: 9, color: '#f4e8a0' });
      }
      // targets
      const sel = OA.selTarget;
      s.targets.forEach((x, i) => {
        if (x.alt < 0) return;
        const st = OA.tstatus(x, s);
        const [px, py] = P(x.alt, x.az);
        const col = OA.col(ST[st].c);
        ctx.lineWidth = 1.5;
        if (st === 'too') { ctx.strokeStyle = col; ctx.beginPath(); ctx.arc(px, py, 4, 0, 2 * Math.PI); ctx.stroke(); }
        else { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(px, py, st === 'done' ? 3.5 : 4.5, 0, 2 * Math.PI); ctx.fill(); }
        if (x.name === sel) { ctx.strokeStyle = OA.col('#ffffff'); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(px, py, 8, 0, 2 * Math.PI); ctx.stroke(); }
        const lab = (st === 'done' ? '✓ ' : '') + (st === 'too' ? '(ToO) ' : '') + x.name;
        ctx.font = '9.5px ' + OA.font('sans');
        const right = px + 6 + ctx.measureText(lab).width > w - 2;
        OA.text(ctx, lab, right ? px - 6 : px + 6, py - 4, { size: 9.5, align: right ? 'right' : 'left', color: st === 'down' ? '#999' : ST[st].c, halo: OA.nightMode ? '#000' : 'rgba(0,0,0,.65)', bold: x.name === sel });
      });
      // telescope
      const [tx, ty] = P(Math.max(0, s.tcs.alt), s.tcs.az);
      ctx.strokeStyle = OA.col('#ffffff'); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(tx - 11, ty); ctx.lineTo(tx - 4, ty); ctx.moveTo(tx + 4, ty); ctx.lineTo(tx + 11, ty);
      ctx.moveTo(tx, ty - 11); ctx.lineTo(tx, ty - 4); ctx.moveTo(tx, ty + 4); ctx.lineTo(tx, ty + 11); ctx.stroke();
      ctx.beginPath(); ctx.arc(tx, ty, 6, 0, 2 * Math.PI); ctx.stroke();
      OA.text(ctx, 'N up, E left (looking up)', 4, 11, { size: 9, color: th.dim });
      OA.text(ctx, s.utc.slice(11, 16) + ' UT', w - 4, 11, { size: 9, color: th.dim, align: 'right' });
    },

    airmass(s) {
      const n = OA.night;
      const cv = OA.$('am-chart');
      const fit = OA.fit(cv, 360);
      if (!fit || !n) return;
      const { ctx, w, h: hh } = fit;
      const th = OA.th();
      ctx.fillStyle = th.bg; ctx.fillRect(0, 0, w, hh);
      const tw = OA.twilights();
      const t0 = (tw ? tw.t0 : n.t[0]) - 10, t1 = (tw ? tw.t1 : n.t[n.t.length - 1]) + 10;
      const gH = 15, gGap = 3, xlab = 14;
      const ph = hh - 10 - xlab - 2 * (gH + gGap) - 8;
      const p = new OA.Plot(ctx, { x: 34, y: 8, w: w - 34 - 34, h: ph }, [t0, t1], [2.5, 1.0]);
      OA.shadeTwilight(p);
      p.ygrid([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.5], (v) => v.toFixed(1));
      p.y2([1, 1.2, 1.4, 1.6, 1.8, 2, 2.5], 2.5, 1.0, (v) => Math.round(Math.asin(1 / v) / D2R) + '°', '#888');
      OA.nightAxes(p, { twiLabels: true, labelBottom: true });
      p.clip();
      // moon
      const mx = n.moon_alt.map((a) => (a > 2 ? Math.min(3, 1 / Math.sin(a * D2R)) : null));
      p.line(n.t, mx, '#b8a040', 1.2, [2, 3]);
      const sel = OA.selTarget;
      const labels = [];
      n.targets.forEach((tc, i) => {
        const x = s.targets[i];
        if (!x) return;
        const col = OA.tcol(i);
        const st = OA.tstatus(x, s);
        const isSel = tc.name === sel;
        const faded = st === 'done' || st === 'too';
        ctx.globalAlpha = faded && !isSel ? 0.4 : 1;
        p.line(n.t, tc.airmass, col, isSel ? 1.6 : 0.9, [2, 3]);
        const obs = tc.airmass.map((v, k) => (tc.observable[k] ? v : null));
        p.line(n.t, obs, col, isSel ? 3.5 : 2);
        ctx.globalAlpha = 1;
        // label at minimum airmass inside the chart
        let kmin = -1;
        for (let k = 0; k < n.t.length; k++) if (n.t[k] >= t0 && n.t[k] <= t1 && (kmin < 0 || tc.airmass[k] < tc.airmass[kmin])) kmin = k;
        if (kmin >= 0 && tc.airmass[kmin] < 2.45) labels.push({ x: p.X(n.t[kmin]), y: Math.max(p.b.y + 20, p.Y(tc.airmass[kmin]) - 3), text: (st === 'done' ? '✓' : '') + tc.name, col, bold: isSel, faded });
      });
      // de-overlap labels a little
      labels.sort((a, b) => a.y - b.y);
      const placed = [];
      ctx.font = '9px sans-serif';
      for (const L of labels) {
        const wdt = ctx.measureText(L.text).width + 4;
        let y = L.y;
        for (let tries = 0; tries < 8; tries++) {
          const hit = placed.find((q) => Math.abs(q.x - L.x) < (q.w + wdt) / 2 && Math.abs(q.y - y) < 10);
          if (!hit) break;
          y = hit.y + 10;
        }
        placed.push({ x: L.x, y, w: wdt });
        ctx.globalAlpha = L.faded ? 0.55 : 1;
        OA.text(ctx, L.text, L.x, y, { size: 9, align: 'center', color: L.col, bold: L.bold, halo: OA.nightMode ? '#000' : 'rgba(255,255,255,.8)' });
        ctx.globalAlpha = 1;
      }
      p.unclip();
      p.vline(s.t, '#e00000', null, 'now ' + s.utc.slice(11, 16), false);
      p.frame();
      OA.text(ctx, 'X', 4, 12, { size: 10, bold: true });
      OA.text(ctx, 'alt', w - 30, 12, { size: 10, color: '#888' });
      // Gantt strips
      const gy1 = p.b.y + p.b.h + xlab + 4, gy2 = gy1 + gH + gGap;
      const strip = (y, label) => {
        ctx.fillStyle = OA.nightMode ? '#0a0000' : '#f2f2f2';
        ctx.fillRect(p.b.x, y, p.b.w, gH);
        ctx.strokeStyle = th.grid; ctx.strokeRect(p.b.x + 0.5, y + 0.5, p.b.w, gH);
        OA.text(ctx, label, p.b.x - 3, y + gH - 4, { size: 9, align: 'right', color: th.dim });
      };
      strip(gy1, 'done'); strip(gy2, 'plan');
      const bar = (y, ta, tb, col, text, alpha, hatch) => {
        let xa = p.X(ta), xb = p.X(tb);
        if (xb < p.b.x || xa > p.b.x + p.b.w) return;
        xa = Math.max(p.b.x, xa); xb = Math.min(p.b.x + p.b.w, xb);
        ctx.globalAlpha = alpha;
        ctx.fillStyle = OA.col(col);
        ctx.fillRect(xa, y + 1, Math.max(1.5, xb - xa), gH - 2);
        ctx.globalAlpha = 1;
        if (hatch) {
          ctx.strokeStyle = OA.col('#c00000'); ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(xa, y + gH - 1); ctx.lineTo(xb, y + 1); ctx.stroke();
        }
        if (text && xb - xa > 24) {
          ctx.save(); ctx.beginPath(); ctx.rect(xa, y, xb - xa, gH); ctx.clip();
          OA.text(ctx, text, xa + 3, y + gH - 4, { size: 9, color: '#ffffff', halo: OA.nightMode ? null : 'rgba(0,0,0,.35)' });
          ctx.restore();
        }
      };
      const frames = (OA.data && OA.data.frames && OA.data.frames.length ? OA.data.frames : s.frames) || [];
      for (const f of frames) {
        if (f.t_start == null) continue;
        const i = f.target ? tIndex(f.target) : -1;
        const col = f.image_type !== 'object' ? '#9a9a9a' : i >= 0 ? OA.tcol(i) : '#666';
        bar(gy1, f.t_start, Math.max(f.t_end, f.t_start + 0.5), col, f.image_type === 'object' ? (f.target || f.object) : f.image_type, f.counted || f.image_type !== 'object' ? 0.95 : 0.4, f.image_type === 'object' && f.target && !f.counted && f.snr != null);
      }
      const pr = s.plan && s.plan.projection;
      if (pr && pr.blocks) for (const b of pr.blocks) bar(gy2, b.start, b.end, OA.tcol(b.target), b.name + (b.n_exp > 1 ? ` ×${b.n_exp}` : ''), 0.85);
      const nx = p.X(s.t);
      ctx.strokeStyle = OA.col('#e00000'); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(nx, gy1); ctx.lineTo(nx, gy2 + gH); ctx.stroke();
    },

    legend(s) {
      const counts = {};
      for (const x of s.targets) { const st = OA.tstatus(x, s); counts[st] = (counts[st] || 0) + 1; }
      const el = OA.$('sky-legend');
      const below = s.targets.filter((x) => x.alt < 0 && x.announced).map((x) => x.name);
      const sig = JSON.stringify(counts) + OA.nightMode + below.join('|');
      if (el._sig === sig) return;
      el._sig = sig;
      el.textContent = '';
      for (const k of ['obs', 'vis', 'up', 'down', 'done', 'too']) {
        el.appendChild(h('span', { class: 'lg' }, h('span', { class: 'sw', style: `background:${OA.col(ST[k].c)}` }), `${ST[k].label} (${counts[k] || 0})`));
      }
      el.appendChild(h('span', { class: 'lg' }, '⊕ telescope'));
      el.appendChild(h('span', { class: 'lg dim' }, 'shaded: below elevation limit; red wedge: high-wind sector (wind dir ± 60°)'));
      if (below.length) el.appendChild(h('span', { class: 'lg dim', style: 'white-space:normal' }, 'below horizon (not on map): ' + below.join(', ')));
    },

    projection(s) {
      const box = OA.$('proj-box');
      const pr = s.plan && s.plan.projection;
      const sig = pr ? JSON.stringify([pr.finish_ut, pr.final_snr, pr.score, pr.list_done_ut, s.score]) : 'none';
      if (box._sig === sig) return;
      box._sig = sig;
      box.textContent = '';
      if (!pr) { box.appendChild(h('div', { class: 'dim' }, 'no projection yet')); return; }
      box.appendChild(h('div', { class: 'proj-sum' },
        'Projected (greedy plan, forecast): score ', h('b', null, `${OA.num(pr.score, 1)} / ${OA.num(pr.max_score, 0)}`),
        ' · now ', h('b', null, OA.num(s.score, 1)),
        ' · ', pr.list_done_ut ? h('span', null, 'whole list done by ', h('b', null, pr.list_done_ut + ' UT')) : h('span', null, 'list not completed tonight'),
        ` · idle ${OA.num(pr.idle_min, 0)} min`));
      const rows = s.targets.filter((x) => x.announced).map((x) => {
        const fin = pr.finish_ut ? pr.finish_ut[x.name] : null;
        const fs = pr.final_snr ? pr.final_snr[x.name] : null;
        return { x, fin, fs, tt: fin ? OA.tOfUT(fin) : (x.done ? -1e9 : 1e9) };
      }).sort((a, b) => a.tt - b.tt);
      const tb = h('tbody');
      for (const r of rows) {
        const i = tIndex(r.x.name);
        tb.appendChild(h('tr', { class: 'clickable' + (r.x.name === OA.selTarget ? ' sel' : ''), onclick: () => OA.selectTarget(r.x.name, true) },
          h('td', null, h('span', { class: 'dot', style: `background:${OA.col(OA.tcol(i))}` }), r.x.name),
          h('td', { class: 'num' }, r.x.done ? 'done' : r.fin || '—'),
          h('td', { class: 'num' }, `${OA.num(r.fs != null ? r.fs : r.x.snr, 0)} / ${OA.num(r.x.goal, 0)}`)));
      }
      box.appendChild(h('table', { class: 'grid small' }, h('thead', null, h('tr', null, h('th', null, 'target'), h('th', { class: 'num' }, 'finish UT'), h('th', { class: 'num' }, 'final S/N'))), tb));
    },

    table(s) {
      const n = OA.night;
      const tbl = OA.$('am-table');
      if (!n) { tbl.innerHTML = '<tr><td class="dim">no night data</td></tr>'; return; }
      const tw = OA.twilights();
      const a = tw && tw.dusk12 != null ? tw.dusk12 : n.night.start, b = tw && tw.dawn12 != null ? tw.dawn12 : n.night.end;
      const rows = n.targets.map((tc, i) => {
        let kmin = -1, first = null, last = null;
        for (let k = 0; k < n.t.length; k++) {
          if (n.t[k] < a || n.t[k] > b) continue;
          if (kmin < 0 || tc.airmass[k] < tc.airmass[kmin]) kmin = k;
          if (tc.observable[k]) { if (first == null) first = n.t[k]; last = n.t[k]; }
        }
        let left = 0;
        for (let k = 1; k < n.t.length; k++) if (tc.observable[k] && n.t[k] > s.t) left += n.t[k] - n.t[k - 1];
        return { i, tc, x: s.targets[i], tmin: kmin >= 0 ? n.t[kmin] : 1e9, xmin: kmin >= 0 ? tc.airmass[kmin] : null, first, last, left };
      }).sort((p, q) => p.tmin - q.tmin);
      tbl.textContent = '';
      tbl.appendChild(h('thead', null, h('tr', null, ['target', 'P', 'UT min X', 'min X', 'X now', 'alt now', 'observable (UT)', 'left'].map((c, k) => h('th', { class: k >= 2 && k !== 6 ? 'num' : '' }, c)))));
      const tb = h('tbody');
      for (const r of rows) {
        const x = r.x || {};
        tb.appendChild(h('tr', { class: 'clickable' + (x.name === OA.selTarget ? ' sel' : '') + (x.done || !x.announced ? ' muted' : ''), onclick: () => OA.selectTarget(r.tc.name, true), title: x.notes || '' },
          h('td', null, h('span', { class: 'dot', style: `background:${OA.col(OA.tcol(r.i))}` }), r.tc.name + (x.announced === false ? ' (ToO)' : '')),
          h('td', null, 'P' + (x.priority != null ? x.priority : '?')),
          h('td', { class: 'num' }, r.xmin != null ? OA.fmtUT(r.tmin) : '—'),
          h('td', { class: 'num' }, OA.num(r.xmin, 2)),
          h('td', { class: 'num' }, x.alt > 0 ? OA.num(x.airmass, 2) : '—'),
          h('td', { class: 'num' }, OA.num(x.alt, 0) + '°'),
          h('td', null, r.first != null ? `${OA.fmtUT(r.first)}–${OA.fmtUT(r.last)}` : 'not observable'),
          h('td', { class: 'num' }, OA.hm(r.left))));
      }
      tbl.appendChild(tb);
    },
  });

  // ------------------------------------------------------------------ targets tab
  const COLS = [
    { k: 'rank', t: '#', num: true, v: (r) => r.rank ?? 99, title: 'merit rank (planner)' },
    { k: 'name', t: 'Target', v: (r) => r.x.name },
    { k: 'pri', t: 'P', v: (r) => r.x.priority },
    { k: 'cfg', t: 'Arm', v: (r) => r.x.config, title: 'configuration whose S/N counts' },
    { k: 'mag', t: 'Mag', num: true, v: (r) => r.x.mag },
    { k: 'X', t: 'X', num: true, v: (r) => (r.x.alt > 0 ? r.x.airmass : 99), title: 'airmass now' },
    { k: 'alt', t: 'Alt', num: true, v: (r) => r.x.alt },
    { k: 'moon', t: 'Moon', num: true, v: (r) => r.x.moon_sep, title: 'Moon separation (deg)' },
    { k: 'snr', t: 'S/N', v: (r) => r.x.snr / Math.max(r.x.goal, 1e-9), title: 'accumulated S/N / goal' },
    { k: 'nexp', t: '#exp', num: true, v: (r) => r.x.n_exp },
    { k: 'status', t: 'Status', v: (r) => ['obs', 'vis', 'up', 'down', 'done', 'too'].indexOf(r.st) },
    { k: 'merit', t: 'Merit', num: true, v: (r) => (r.c ? r.c.merit : -1) },
    { k: 'plan', t: 'Needs', num: true, v: (r) => (r.c ? r.c.time_needed_min : 1e9), title: 'exposures still needed × t_exp; minutes incl. overheads' },
    { k: 'win', t: 'Win', num: true, v: (r) => (r.c ? r.c.window_left_min : -1), title: 'observable minutes left tonight' },
    { k: 'fin', t: 'Finish', num: true, v: (r) => (r.fin ? OA.tOfUT(r.fin) : 1e9), title: 'projected completion (UT)' },
  ];
  const targets = OA.register({
    name: 'targets',
    rows: new Map(),
    sortK: 'rank', asc: true,
    init() {
      const tbl = OA.$('tg-table');
      const tr = h('tr');
      for (const c of COLS) {
        const th = h('th', { class: 'sortable' + (c.num ? ' num' : ''), title: c.title || '' }, c.t);
        th.onclick = () => { if (this.sortK === c.k) this.asc = !this.asc; else { this.sortK = c.k; this.asc = c.k === 'rank' || c.k === 'name' || c.k === 'pri' || c.k === 'X' || c.k === 'fin' || c.k === 'status'; } this.render(true); };
        c.th = th;
        tr.appendChild(th);
      }
      tbl.appendChild(h('thead', null, tr));
      this.tbody = h('tbody');
      tbl.appendChild(this.tbody);
    },
    reset() { this.rows.clear(); this.tbody.textContent = ''; },
    snap() { if (OA.tab === 'targets' && OA.every('tg', 1000)) this.render(); },
    tab(n) { if (n === 'targets') this.render(true); },
    select() { if (OA.tab === 'targets') this.render(true); },
    redraw() { if (OA.tab === 'targets') this.render(true); },
    render() {
      const s = OA.snap;
      if (!s) return;
      const cands = (s.plan && s.plan.candidates) || [];
      const fin = (s.plan && s.plan.projection && s.plan.projection.finish_ut) || {};
      const ranked = cands.filter((c) => c.feasible).sort((a, b) => b.merit - a.merit);
      const rankOf = new Map(ranked.map((c, k) => [c.i, k + 1]));
      const data = s.targets.map((x) => ({ x, c: cands.find((c) => c.i === x.i), st: OA.tstatus(x, s), rank: rankOf.get(x.i), fin: fin[x.name] }));
      const col = COLS.find((c) => c.k === this.sortK) || COLS[0];
      data.sort((a, b) => {
        const va = col.v(a), vb = col.v(b);
        const r = typeof va === 'string' ? va.localeCompare(vb) : (va ?? 0) - (vb ?? 0);
        return this.asc ? r : -r;
      });
      for (const c of COLS) OA.setCls(c.th, 'sortable' + (c.num ? ' num' : '') + (c.k === this.sortK ? ' sorted' + (this.asc ? ' asc' : '') : ''));
      let order = [];
      for (const r of data) {
        let row = this.rows.get(r.x.i);
        if (!row) row = this.makeRow(r.x);
        this.fill(row, r, s);
        order.push(row.tr);
      }
      // reorder only when needed
      const cur = [...this.tbody.children];
      if (cur.length !== order.length || cur.some((el, k) => el !== order[k])) for (const tr of order) this.tbody.appendChild(tr);
    },
    makeRow(x) {
      const cells = {};
      const tr = h('tr', { class: 'clickable' });
      for (const c of COLS) { cells[c.k] = h('td', { class: c.num ? 'num' : '' }); tr.appendChild(cells[c.k]); }
      cells.snr.innerHTML = '<div class="snrbar"><i></i><span></span></div>';
      cells.bar = cells.snr.firstChild;
      tr.onclick = () => OA.selectTarget(x.name, true);
      const row = { tr, cells };
      this.rows.set(x.i, row);
      return row;
    },
    fill(row, r, s) {
      const { x, c, st } = r, C = row.cells;
      OA.setText(C.rank, r.rank ?? '');
      OA.setText(C.name, (x.name === OA.selTarget ? '▸ ' : '') + x.name);
      const kind = { standard: 'std', backup: 'bkp', too: 'ToO', science: '' }[x.kind] ?? (x.kind || '');
      OA.setText(C.pri, 'P' + x.priority + (kind ? ' ' + kind : ''));
      OA.setText(C.cfg, x.config.replace(/^[A-Z0-9]+-/, ''));
      OA.setAttr(C.cfg, 'title', x.config);
      OA.setText(C.mag, `${OA.num(x.mag, 1)} ${x.band}`);
      OA.setText(C.X, x.alt > 0 ? OA.num(x.airmass, 2) : '—');
      OA.setText(C.alt, OA.num(x.alt, 0) + '°');
      OA.setText(C.moon, OA.num(x.moon_sep, 0) + '°');
      const frac = OA.clamp(x.snr / Math.max(x.goal, 1e-9), 0, 1);
      C.bar.firstChild.style.width = (frac * 100).toFixed(1) + '%';
      OA.setCls(C.bar, 'snrbar' + (x.done ? ' done' : ''));
      OA.setText(C.bar.lastChild, `${OA.num(x.snr, 0)}/${OA.num(x.goal, 0)}${unitLabel(x.unit)}`);
      OA.setText(C.nexp, x.n_exp);
      const tagCls = { obs: 'obs', vis: 'vis', up: 'no', down: 'no', done: 'done', too: 'too' }[st];
      let stText = { obs: s.tcs.state, vis: 'observable', up: 'up', down: 'not up', done: 'done', too: 'ToO pending' }[st];
      if (c && !c.feasible && c.note && st !== 'done' && st !== 'too') stText = c.note.length > 26 ? c.note.slice(0, 25) + '…' : c.note;
      const tg = C.status.firstChild && C.status.firstChild.tagName === 'SPAN' ? C.status.firstChild : C.status.appendChild(h('span'));
      OA.setCls(tg, 'tag ' + (st === 'vis' && c && !c.feasible ? 'inf' : tagCls));
      OA.setText(tg, stText);
      OA.setAttr(tg, 'title', c && c.note ? 'planner: ' + c.note : '');
      OA.setText(C.merit, c && c.merit ? c.merit.toPrecision(3) : '—');
      OA.setText(C.plan, c && c.n_needed ? `${c.n_needed}×${OA.num(c.t_exp, 0)} ${OA.num(c.time_needed_min, 0)}m` : (x.done ? '—' : ''));
      OA.setText(C.win, c ? OA.hm(c.window_left_min) : '');
      OA.setText(C.fin, x.done ? 'done' : r.fin || '—');
      const tip = [x.name + ` (P${x.priority}, ${x.kind})`, x.notes, c && c.note ? 'planner: ' + c.note : '',
        `${x.ra} ${x.dec} · ${x.mag} ${x.band}` + (x.pi ? ` · PI ${x.pi}` : '') + (x.program ? ` · ${x.program}` : ''),
        x.max_seeing ? `max seeing ${x.max_seeing}″` : '', x.window ? 'window ' + JSON.stringify(x.window) : '', x.pa != null ? `PA ${x.pa}` : '',
        c ? `pred. FWHM ${OA.num(c.fwhm_pred, 2)}″ · S/N per exp ${OA.num(c.snr1, 1)} · overhead ${OA.num(c.overhead_min, 1)} min` : ''].filter(Boolean).join('\n');
      OA.setAttr(row.tr, 'title', tip);
      OA.setCls(row.tr, 'clickable' + (x.name === OA.selTarget ? ' sel' : '') + (st === 'done' || st === 'too' ? ' muted' : ''));
    },
  });

  // ------------------------------------------------------------------ ETC tab
  const etc = OA.register({
    name: 'etc',
    res: null, stale: true, optSig: null,
    init() {
      OA.$('etc-go').onclick = () => this.compute();
      OA.$('etc-now').onclick = () => { OA.$('etc-seeing').value = ''; OA.$('etc-cloud').value = ''; this.compute(); };
      OA.$('etc-target').onchange = (e) => { OA.$('etc-texp').value = ''; OA.selectTarget(e.target.value); };
      for (const id of ['etc-seeing', 'etc-cloud', 'etc-texp']) OA.$(id).addEventListener('keydown', (e) => { if (e.key === 'Enter') this.compute(); });
      OA.$('etc-apply').onclick = () => this.apply();
    },
    reset() { this.res = null; this.optSig = null; this.stale = true; },
    snap(s) {
      const sig = s.targets.map((x) => x.name + (x.announced ? 1 : 0)).join('|');
      if (sig !== this.optSig) {
        this.optSig = sig;
        const sel = OA.$('etc-target');
        const cur = sel.value || OA.selTarget || '';
        sel.textContent = '';
        sel.appendChild(h('option', { value: '' }, '— target —'));
        for (const x of s.targets) if (x.announced) sel.appendChild(h('option', { value: x.name }, `P${x.priority} ${x.name}`));
        if (cur) sel.value = cur;
      }
    },
    select(name) {
      const sel = OA.$('etc-target');
      if (sel.value !== name) { sel.value = name; OA.$('etc-texp').value = ''; }
      this.stale = true;
      if (OA.tab === 'etc') this.compute();
    },
    tab(n) { if (n === 'etc') { if (this.stale && OA.$('etc-target').value) this.compute(); else this.plot(); } },
    redraw() { if (OA.tab === 'etc') this.plot(); },
    async compute() {
      const name = OA.$('etc-target').value;
      const out = OA.$('etc-res');
      if (!name) { out.innerHTML = '<div class="dim pad">Select a target (here, in the Targets tab or on the sky map).</div>'; return; }
      const qs = new URLSearchParams({ target: name });
      for (const [k, id] of [['seeing', 'etc-seeing'], ['cloud', 'etc-cloud'], ['t_exp', 'etc-texp']]) {
        const v = OA.$(id).value.trim();
        if (v !== '') { if (!isFinite(parseFloat(v))) { out.innerHTML = `<div class="pad" style="color:#c00">${k} must be a number</div>`; return; } qs.set(k, v); }
      }
      this.stale = false;
      out.innerHTML = '<div class="dim pad"><span class="spin"></span>computing…</div>';
      try {
        this.res = await OA.api('/api/etc?' + qs.toString());
      } catch (e) {
        this.res = null;
        out.innerHTML = `<div class="pad" style="color:#c00">ETC failed: ${OA.esc(e.message)}</div>`;
        this.plot();
        return;
      }
      this.show();
      this.plot();
    },
    show() {
      const r = this.res, out = OA.$('etc-res');
      if (!r) return;
      const p = r.plan, u = unitLabel(r.unit);
      const kv = (k, v, title) => `<div class="kvr"${title ? ` title="${OA.esc(title)}"` : ''}><span>${k}</span><span>${v}</span></div>`;
      const wall = p.wall_s / 60;
      const need = Math.sqrt(Math.max(0, r.goal * r.goal - r.snr_have * r.snr_have));
      out.innerHTML =
        `<div class="etc-plan">${OA.esc(r.target)}: <b>${p.n_exp} × ${OA.num(p.t_exp, 0)} s</b> = ${OA.num(p.open_s / 60, 0)} min open, <b>${OA.hm(wall)}</b> wall · final S/N <b>${OA.num(p.snr_final, 1)}</b> / ${OA.num(r.goal, 0)}${u} at ${r.lam} Å · limited by <b>${OA.esc(p.limited_by)}</b></div>` +
        kv('Regime', OA.esc(r.regime), 'which noise term dominates') +
        kv('Seeing (500 nm, zenith)', OA.num(r.seeing, 2) + '″') +
        kv('Cloud extinction', OA.num(r.cloud, 2) + ' mag') +
        kv('Airmass now', OA.num(r.airmass, 3)) +
        kv('Delivered FWHM', OA.num(r.fwhm, 2) + '″') +
        kv('Slit fraction', OA.num(r.slit_frac * 100, 1) + ' %') +
        kv('Sky', OA.num(r.sky_mag, 2) + ' mag/″²') +
        kv('Signal', sig3(r.signal_e_s) + ' e⁻/s', 'object signal at the reference wavelength, per ' + (r.unit || 'unit')) +
        kv('Sky per pixel', sig3(r.sky_e_s_pix) + ' e⁻/s/pix') +
        kv('Saturation time', r.t_saturate > 1e6 ? '∞' : OA.num(r.t_saturate, 0) + ' s', 'time to reach the linearity limit at the peak') +
        kv('S/N so far', OA.num(r.snr_have, 1) + u) +
        kv('S/N still needed', OA.num(need, 1) + u, 'added in quadrature');
    },
    plot() {
      const fit = OA.fit(OA.$('etc-plot'));
      if (!fit) return;
      const { ctx, w, h: hh } = fit;
      const th = OA.th();
      ctx.fillStyle = th.bg; ctx.fillRect(0, 0, w, hh);
      const r = this.res;
      if (!r) { OA.text(ctx, 'S/N vs exposure time appears here', w / 2, hh / 2, { align: 'center', color: th.dim, size: 12 }); return; }
      const T = r.curve.t, S = r.curve.snr;
      const need = Math.sqrt(Math.max(0, r.goal * r.goal - r.snr_have * r.snr_have));
      const tmax = Math.max(T[T.length - 1], r.plan.t_exp * 1.1);
      const ymax = Math.max(...S, need, r.plan.snr_final ? Math.min(r.plan.snr_final, r.goal) : 0) * 1.12 || 1;
      const p = new OA.Plot(ctx, { x: 44, y: 10, w: w - 60, h: hh - 34 }, [0, tmax], [0, ymax]);
      p.ygrid(OA.niceTicks(0, ymax, 6), (v) => String(v));
      p.xgrid(OA.niceTicks(0, tmax, 8), (v) => v + ' s');
      p.clip();
      if (r.t_saturate && r.t_saturate < tmax) {
        ctx.fillStyle = OA.nightMode ? 'rgba(80,0,0,.4)' : 'rgba(230,60,60,.12)';
        ctx.fillRect(p.X(r.t_saturate), p.b.y, p.b.x + p.b.w - p.X(r.t_saturate), p.b.h);
        p.vline(r.t_saturate, '#d00000', [4, 3], 'saturates', false);
      }
      p.hline(r.goal, '#1d9a3a', [6, 4], `goal ${r.goal}${unitLabel(r.unit)}`);
      if (r.snr_have > 0) p.hline(need, '#e07a10', [3, 3], `still needed ${OA.num(need, 1)}`, 'left');
      p.vline(r.plan.t_exp, '#2a6fdb', [5, 3], `t_exp ${OA.num(r.plan.t_exp, 0)} s`, false);
      p.line([0, ...T], [0, ...S], '#2a6fdb', 2);
      p.dots(T, S, '#2a6fdb', 2.2);
      p.unclip();
      p.frame();
      OA.text(ctx, 'single-exposure S/N' + unitLabel(r.unit), p.b.x + 6, p.b.y + 14, { size: 11, bold: true });
      OA.text(ctx, `${r.target} · seeing ${r.seeing}″ · cloud ${r.cloud} mag · X ${r.airmass}`, p.b.x + 6, p.b.y + 28, { size: 10, color: th.dim });
    },
    async apply() {
      const r = this.res;
      if (!r || !OA.snap) { OA.term.local('compute an ETC first'); return; }
      const x = OA.snap.targets.find((t) => t.name === r.target);
      const arm = Object.values(OA.snap.arms).find((a) => x && a.config === x.config);
      await OA.cmd(`all exptime ${Math.round(r.plan.t_exp)}`);
      await OA.cmd(`all loops ${Math.max(1, r.plan.n_exp)}`);
      await OA.cmd(`all object ${r.target}`);
      OA.term.local(`applied ${r.plan.n_exp} × ${Math.round(r.plan.t_exp)} s to all arms` + (arm ? ` (S/N counts on the ${arm.name} arm, ${arm.config})` : ''));
    },
  });
})();
