/* obsassist console — tiny canvas plotting helpers (no libraries). */
'use strict';

(function () {
  const OA = window.OA;

  /** Size a canvas to its CSS box (x devicePixelRatio). Returns {ctx, w, h} in CSS px, or null if hidden. */
  let fonts = null;
  /** Font families from the CSS variables (cached). */
  OA.font = function (kind) {
    if (!fonts) {
      const cs = getComputedStyle(document.body);
      fonts = { sans: cs.getPropertyValue('--sans').trim() || 'sans-serif', mono: cs.getPropertyValue('--mono').trim() || 'monospace' };
    }
    return fonts[kind] || fonts.sans;
  };

  OA.fit = function (cv, cssH) {
    if (!cv) return null;
    const dpr = window.devicePixelRatio || 1;
    if (cssH != null) cv.style.height = cssH + 'px';
    else if (cv.dataset.h && !cv.style.height) cv.style.height = cv.dataset.h + 'px';
    const w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return null;
    const W = Math.round(w * dpr), H = Math.round(h * dpr);
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  };

  /** "Nice" tick values between lo and hi. */
  OA.niceTicks = function (lo, hi, n = 5) {
    if (!(hi > lo)) return [lo];
    const span = hi - lo;
    const raw = span / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const f = raw / mag;
    const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v / step) * step);
    return out;
  };

  /** Sim minutes at whole UT hours (or every `every` minutes) within [t0, t1]. */
  OA.utTicks = function (t0, t1, every = 60) {
    const out = [];
    const m0 = OA.utMin(t0);
    if (m0 == null) return out;
    let t = t0 + ((every - (m0 % every)) % every);
    for (; t <= t1 + 1e-6; t += every) out.push(t);
    return out;
  };

  OA.minMax = function (arrs, pad = 0) {
    let lo = Infinity, hi = -Infinity;
    for (const a of arrs) if (a) for (const v of a) if (v != null && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!isFinite(lo)) return null;
    const p = (hi - lo) * pad || pad;
    return [lo - p, hi + p];
  };

  class Plot {
    constructor(ctx, box, xr, yr) {
      this.ctx = ctx; this.b = box;
      this.x0 = xr[0]; this.x1 = xr[1]; this.y0 = yr[0]; this.y1 = yr[1];
      this.th = OA.th();
    }
    X(v) { return this.b.x + (v - this.x0) / (this.x1 - this.x0) * this.b.w; }
    Y(v) { return this.b.y + this.b.h - (v - this.y0) / (this.y1 - this.y0) * this.b.h; }
    Yr(v, r0, r1) { return this.b.y + this.b.h - (v - r0) / (r1 - r0) * this.b.h; }
    clip() { const c = this.ctx; c.save(); c.beginPath(); c.rect(this.b.x, this.b.y, this.b.w, this.b.h); c.clip(); }
    unclip() { this.ctx.restore(); }
    frame() {
      const c = this.ctx, b = this.b;
      c.strokeStyle = this.th.axis; c.lineWidth = 1; c.setLineDash([]);
      c.strokeRect(b.x + 0.5, b.y + 0.5, b.w, b.h);
    }
    xgrid(ticks, label, opts = {}) {
      const c = this.ctx, b = this.b;
      c.font = '10px ' + OA.font('sans');
      c.textAlign = 'center'; c.textBaseline = 'top';
      for (const t of ticks) {
        const x = Math.round(this.X(t)) + 0.5;
        if (x < b.x - 1 || x > b.x + b.w + 1) continue;
        c.strokeStyle = this.th.grid; c.lineWidth = 1;
        c.beginPath(); c.moveTo(x, b.y); c.lineTo(x, b.y + b.h); c.stroke();
        if (label && !opts.noLabels) { c.fillStyle = this.th.fg; c.fillText(label(t), x, b.y + b.h + 2); }
      }
    }
    ygrid(ticks, fmt, side = 'left', color, grid = true) {
      const c = this.ctx, b = this.b;
      c.font = '10px ' + OA.font('sans');
      c.textBaseline = 'middle';
      c.textAlign = side === 'left' ? 'right' : 'left';
      for (const v of ticks) {
        const y = Math.round(this.Y(v)) + 0.5;
        if (y < b.y - 1 || y > b.y + b.h + 1) continue;
        if (grid) { c.strokeStyle = this.th.grid; c.lineWidth = 1; c.beginPath(); c.moveTo(b.x, y); c.lineTo(b.x + b.w, y); c.stroke(); }
        c.fillStyle = color ? OA.col(color) : this.th.fg;
        c.fillText(fmt ? fmt(v) : String(v), side === 'left' ? b.x - 3 : b.x + b.w + 3, y);
      }
    }
    /** Axis ticks on the other side for a secondary range [r0, r1]. */
    y2(ticks, r0, r1, fmt, color) {
      const c = this.ctx, b = this.b;
      c.font = '10px ' + OA.font('sans');
      c.textBaseline = 'middle'; c.textAlign = 'left';
      c.fillStyle = color ? OA.col(color) : this.th.fg;
      for (const v of ticks) {
        const y = this.Yr(v, r0, r1);
        if (y < b.y - 1 || y > b.y + b.h + 1) continue;
        c.fillText(fmt ? fmt(v) : String(v), b.x + b.w + 3, y);
      }
    }
    /** Polyline; null/NaN values break the line. yMap optionally maps to pixel. */
    line(xs, ys, color, width = 1.5, dash = null, yMap = null) {
      const c = this.ctx;
      c.strokeStyle = OA.col(color); c.lineWidth = width; c.setLineDash(dash || []);
      c.lineJoin = 'round';
      c.beginPath();
      let pen = false;
      for (let i = 0; i < xs.length; i++) {
        const v = ys[i];
        if (v == null || !isFinite(v) || xs[i] == null) { pen = false; continue; }
        const x = this.X(xs[i]), y = yMap ? yMap(v) : this.Y(v);
        if (!pen) { c.moveTo(x, y); pen = true; } else c.lineTo(x, y);
      }
      c.stroke();
      c.setLineDash([]);
    }
    dots(xs, ys, color, r = 1.6, yMap = null) {
      const c = this.ctx;
      c.fillStyle = OA.col(color);
      for (let i = 0; i < xs.length; i++) {
        const v = ys[i];
        if (v == null || !isFinite(v)) continue;
        const x = this.X(xs[i]), y = yMap ? yMap(v) : this.Y(v);
        c.fillRect(x - r, y - r, 2 * r, 2 * r);
      }
    }
    vline(x, color, dash, label, labelTop = true) {
      const c = this.ctx, b = this.b;
      const px = Math.round(this.X(x)) + 0.5;
      if (px < b.x || px > b.x + b.w) return;
      c.strokeStyle = OA.col(color); c.lineWidth = 1; c.setLineDash(dash || []);
      c.beginPath(); c.moveTo(px, b.y); c.lineTo(px, b.y + b.h); c.stroke(); c.setLineDash([]);
      if (label) {
        c.font = '9px ' + OA.font('sans');
        c.fillStyle = OA.col(color); c.textAlign = 'left'; c.textBaseline = labelTop ? 'top' : 'bottom';
        c.fillText(label, px + 2, labelTop ? b.y + 2 : b.y + b.h - 2);
      }
    }
    hlinePx(py, color, dash, label, side = 'right') {
      const c = this.ctx, b = this.b;
      py = Math.round(py) + 0.5;
      c.strokeStyle = OA.col(color); c.lineWidth = 1; c.setLineDash(dash || []);
      c.beginPath(); c.moveTo(b.x, py); c.lineTo(b.x + b.w, py); c.stroke(); c.setLineDash([]);
      if (label) {
        c.font = '9px ' + OA.font('sans');
        c.fillStyle = OA.col(color); c.textBaseline = 'bottom';
        c.textAlign = side === 'right' ? 'right' : 'left';
        c.fillText(label, side === 'right' ? b.x + b.w - 3 : b.x + 3, py - 1);
      }
    }
    hline(y, color, dash, label, side) { this.hlinePx(this.Y(y), color, dash, label, side); }
  }
  OA.Plot = Plot;

  /** Draw text with the page's fonts. */
  OA.text = function (ctx, s, x, y, opts = {}) {
    ctx.font = (opts.bold ? 'bold ' : '') + (opts.size || 10) + 'px ' + (opts.mono ? OA.font('mono') : OA.font('sans'));
    ctx.fillStyle = opts.color ? OA.col(opts.color) : OA.th().fg;
    ctx.textAlign = opts.align || 'left';
    ctx.textBaseline = opts.base || 'alphabetic';
    if (opts.halo) {
      ctx.lineWidth = 3; ctx.strokeStyle = opts.halo; ctx.lineJoin = 'round';
      ctx.strokeText(s, x, y);
    }
    ctx.fillText(s, x, y);
    return ctx.measureText(s).width;
  };

  /** Draw the background + UT axis common to the night charts. */
  OA.nightAxes = function (p, opts = {}) {
    const tw = OA.twilights();
    const t0 = p.x0, t1 = p.x1;
    const span = t1 - t0;
    const every = span > 600 ? 60 : span > 240 ? 60 : 30;
    p.xgrid(OA.utTicks(t0, t1, every), opts.labels === false ? null : (t) => OA.fmtUT(t));
    if (tw && opts.twilight !== false) {
      const col = OA.nightMode ? '#660000' : '#8a6d3b';
      let lastX = -1e9;
      for (const [k, lab] of [['sunset', 'sunset'], ['dusk12', '12°'], ['dusk18', '18°'], ['dawn18', '18°'], ['dawn12', '12°'], ['sunrise', 'sunrise']]) {
        if (tw[k] == null) continue;
        const x = p.X(tw[k]);
        const show = opts.twiLabels && x - lastX > (lab.length > 3 ? 40 : 22);
        p.vline(tw[k], col, [3, 3], show ? lab : null, !opts.labelBottom);
        if (show) lastX = x;
      }
    }
  };

  /** Shade twilight on a chart from night.sun_alt. */
  OA.shadeTwilight = function (p) {
    const n = OA.night;
    if (!n) return;
    const c = p.ctx, b = p.b;
    const shade = (a) => {
      if (OA.nightMode) return a > -0.8 ? 'rgba(90,0,0,.35)' : a > -6 ? 'rgba(70,0,0,.28)' : a > -12 ? 'rgba(50,0,0,.22)' : a > -18 ? 'rgba(30,0,0,.15)' : null;
      return a > -0.8 ? 'rgba(250,215,120,.55)' : a > -6 ? 'rgba(250,190,120,.40)' : a > -12 ? 'rgba(160,150,210,.30)' : a > -18 ? 'rgba(140,160,210,.18)' : null;
    };
    for (let i = 0; i < n.t.length - 1; i++) {
      const f = shade(n.sun_alt[i]);
      if (!f) continue;
      const xa = p.X(n.t[i]), xb = p.X(n.t[i + 1]);
      if (xb < b.x || xa > b.x + b.w) continue;
      c.fillStyle = f;
      c.fillRect(Math.max(b.x, xa), b.y, Math.min(b.x + b.w, xb) - Math.max(b.x, xa) + 0.5, b.h);
    }
  };

  /** Distinct colours for targets. */
  OA.TCOL = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#8c564b', '#e377c2', '#6b6b6b', '#a5a21a', '#393b79', '#637939', '#843c39', '#7b4173', '#3182bd', '#e6550d'];
  OA.tcol = (i) => OA.TCOL[i % OA.TCOL.length];

  /** Throttle helper: returns true if at least `ms` elapsed since the last call with this key. */
  const lastRun = {};
  OA.every = function (key, ms) {
    const now = performance.now();
    if (lastRun[key] != null && now - lastRun[key] < ms) return false;
    lastRun[key] = now;
    return true;
  };

  OA.compass = (deg) => ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'][Math.round(((deg % 360) + 360) % 360 / 22.5) % 16];
})();
