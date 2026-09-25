/* obsassist console — Data / quick-look tab: frame list, "Overview" raw-frame display with
   zscale/asinh stretches, Magnifier (x4) following the mouse, extracted 1D spectrum, QL flags
   and a FITS header viewer. Frames may not exist (FITS writer missing / still working):
   every request degrades to a clear message. */
'use strict';

(function () {
  const OA = window.OA, h = OA.h;
  const LE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

  // ------------------------------------------------------------------ stretches
  function sample(a, nsamp) {
    const n = a.length, step = Math.max(1, Math.floor(n / nsamp)), s = [];
    for (let i = 0; i < n; i += step) s.push(a[i]);
    return s.sort((x, y) => x - y);
  }
  const pct = (s, p) => s[OA.clamp(Math.round(p / 100 * (s.length - 1)), 0, s.length - 1)];
  /** IRAF-style zscale (iterative line fit to the sorted sample). */
  function zscale(a, contrast = 0.25) {
    const s = sample(a, 4000), N = s.length;
    if (!N) return [0, 1];
    const med = s[Math.floor(N / 2)];
    const mask = new Uint8Array(N).fill(1);
    let slope = 0, icpt = med;
    for (let it = 0; it < 6; it++) {
      let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (let i = 0; i < N; i++) if (mask[i]) { sw++; sx += i; sy += s[i]; sxx += i * i; sxy += i * s[i]; }
      if (sw < N * 0.5) break;
      const d = sw * sxx - sx * sx;
      if (!d) break;
      slope = (sw * sxy - sx * sy) / d; icpt = (sy - slope * sx) / sw;
      let ss = 0, c = 0;
      for (let i = 0; i < N; i++) if (mask[i]) { const r = s[i] - (icpt + slope * i); ss += r * r; c++; }
      const sig = Math.sqrt(ss / Math.max(1, c));
      let changed = 0;
      for (let i = 0; i < N; i++) { const m = Math.abs(s[i] - (icpt + slope * i)) < 2.5 * sig ? 1 : 0; if (m !== mask[i]) { mask[i] = m; changed++; } }
      if (!changed) break;
    }
    const center = Math.floor(N / 2), sl = slope / contrast;
    let z1 = Math.max(s[0], med - center * sl), z2 = Math.min(s[N - 1], med + (N - center) * sl);
    if (!(z2 > z1)) { z1 = s[0]; z2 = s[N - 1] > s[0] ? s[N - 1] : s[0] + 1; }
    return [z1, z2];
  }

  function findSpectrum(ql) {
    if (!ql || typeof ql !== 'object') return null;
    const isArr = (v) => Array.isArray(v) && v.length > 1;
    const pick = (o, names) => { for (const n of names) if (isArr(o[n])) return o[n]; return null; };
    const F = ['flux', 'counts', 'spec', 'spectrum', 'flux_adu', 'signal', 'f'];
    const Wn = ['wave', 'wavelength', 'wave_A', 'lam', 'lambda', 'wl', 'wav'];
    const Sk = ['sky', 'sky_flux', 'sky_adu', 'bkg', 'background'];
    const Sn = ['snr', 'sn', 's_n', 'snr_per_pix', 'snr_pix', 'snr_curve'];
    const cands = [ql, ql.spectrum, ql.spec, ql.extracted, ql.extraction, ql.oned, ql['1d']].filter((o) => o && typeof o === 'object' && !Array.isArray(o));
    for (const o of cands) {
      const flux = pick(o, F);
      if (!flux) continue;
      return { wave: pick(o, Wn), flux, sky: pick(o, Sk), snr: pick(o, Sn) };
    }
    if (isArr(ql.orders)) {
      const out = { wave: [], flux: [], sky: [], snr: [] };
      for (const o of ql.orders) {
        const f = pick(o, F);
        if (!f) continue;
        const w = pick(o, Wn), sk = pick(o, Sk), sn = pick(o, Sn);
        out.flux.push(f); out.wave.push(w || null); out.sky.push(sk || null); out.snr.push(sn || null);
      }
      if (out.flux.length) return out;
    }
    return null;
  }
  // split into segments of plain numeric arrays (handles echelle orders = arrays of arrays)
  function segments(sp) {
    const segs = [];
    const is2d = Array.isArray(sp.flux[0]);
    const n = is2d ? sp.flux.length : 1;
    for (let k = 0; k < n; k++) {
      const f = is2d ? sp.flux[k] : sp.flux;
      const w0 = is2d ? (sp.wave && sp.wave[k]) : sp.wave;
      const w = w0 && w0.length === f.length ? w0 : f.map((_, i) => i);
      const sk = is2d ? (sp.sky && sp.sky[k]) : sp.sky;
      const sn = is2d ? (sp.snr && sp.snr[k]) : sp.snr;
      segs.push({ w, f, sky: sk && sk.length === f.length ? sk : null, snr: sn && sn.length === f.length ? sn : null });
    }
    return segs;
  }

  function fitsCard(k, v) {
    let val;
    if (typeof v === 'string') val = ("'" + v.replace(/'/g, "''").padEnd(8) + "'").padEnd(20);
    else if (typeof v === 'boolean') val = (v ? 'T' : 'F').padStart(20);
    else if (v == null) val = ''.padStart(20);
    else if (typeof v === 'object') val = JSON.stringify(v);
    else val = String(v).padStart(20);
    return (k.length <= 8 ? k.padEnd(8) : 'HIERARCH ' + k) + '= ' + val;
  }

  // ------------------------------------------------------------------ module
  const data = OA.data = OA.register({
    name: 'data',
    frames: [],
    sel: null,
    selReady: null,
    img: null,
    off: null,
    view: null,
    lock: null,
    lastSig: '',
    waiting: false,
    listGen: 0,
    init() {
      const cv = OA.$('dq-cv');
      cv.addEventListener('mousemove', (e) => { if (!this.lock) this.magAt(e); });
      cv.addEventListener('click', (e) => { this.lock = this.lock ? null : true; if (!this.lock) this.magAt(e); else this.magAt(e, true); });
      cv.addEventListener('mouseleave', () => { /* keep the last magnifier view */ });
      OA.$('dq-stretch').onchange = () => { this.stretch(); this.paint(); };
      const manual = () => { OA.$('dq-stretch').value = 'manual'; this.stretch(); this.paint(); };
      OA.$('dq-min').addEventListener('change', manual);
      OA.$('dq-max').addEventListener('change', manual);
      OA.$('dq-bin').onchange = () => { if (this.sel) this.loadPixels(this.sel); };
      OA.$('dq-hdrfilter').addEventListener('input', () => this.showHeader());
      this.magTable();
    },
    reset() {
      this.frames = []; this.sel = null; this.img = null; this.hdr = null; this.ql = null; this.lastSig = ''; this.listGen++;
      OA.$('dq-table').textContent = '';
      this.message('No frame selected.');
      OA.$('dq-hdr').textContent = '—';
      OA.$('dq-flags').textContent = '';
      OA.setText(OA.$('dq-qlsum'), '');
      this.plotSpec();
    },
    snap(s) {
      const fr = s.frames || [];
      const last = fr.length ? fr[fr.length - 1] : null;
      const sig = fr.length + '|' + (last ? last.file + '|' + ('path' in last) + ('ql_error' in last) + ('ql' in last) : '');
      if (sig !== this.lastSig) { this.lastSig = sig; this.refreshList(); }
      else if (this.waiting && OA.every('dqpoll', 2500)) this.refreshList();
    },
    tab(n) { if (n === 'data') requestAnimationFrame(() => { this.paint(); this.plotSpec(); }); },
    redraw() { if (OA.tab === 'data') { this.paint(); this.plotSpec(); } },
    async refreshList() {
      const gen = this.listGen;
      let list;
      try { list = await OA.api('/api/frames'); } catch (e) {
        OA.setText(OA.$('dq-count'), 'frame list unavailable: ' + e.message);
        return;
      }
      if (gen !== this.listGen) return;
      this.frames = list;
      this.waiting = list.some((f) => !('path' in f) && !('ql_error' in f));
      this.renderList();
      const newest = list[list.length - 1];
      if (!newest) return;
      if (OA.$('dq-follow').checked && newest.file !== this.sel) this.selectFrame(newest.file);
      else if (this.sel) {
        const f = list.find((x) => x.file === this.sel);
        const ready = f ? ('path' in f) + '|' + ('ql_error' in f) : null;
        if (f && ready !== this.selReady) this.selectFrame(this.sel);
      }
    },
    renderList() {
      const tbl = OA.$('dq-table');
      const list = this.frames;
      OA.setText(OA.$('dq-count'), `${list.length} frame${list.length === 1 ? '' : 's'}` + (this.waiting ? ' · data system writing…' : ''));
      const scroll = tbl.parentElement.scrollTop;
      tbl.textContent = '';
      tbl.appendChild(h('thead', null, h('tr', null, ['File', 'Arm', 'Type', 'Object', 'Exp s', 'UT start', 'X', 'FWHM', 'S/N (total/goal)', 'QC', 'Quick-look'].map((c, k) => h('th', { class: [4, 6, 7].includes(k) ? 'num' : '' }, c)))));
      const tb = h('tbody');
      if (!list.length) tb.appendChild(h('tr', null, h('td', { colspan: 11, class: 'dim' }, 'No frames yet tonight. Start an exposure in the instrument GUI; each readout writes a FITS file.')));
      for (const f of [...list].reverse()) {
        const q = f.ql_summary || {};
        let qc;
        if (f.image_type !== 'object') qc = h('span', { class: 'tag no' }, 'cal');
        else if (f.counted) qc = h('span', { class: 'tag good' }, '✓ counted');
        else if (f.note) qc = h('span', { class: 'tag no', title: f.note }, 'other arm');
        else if (f.target) qc = h('span', { class: 'tag bad', title: (f.qc_reasons || []).join('\n') }, '✗ ' + ((f.qc_reasons || [])[0] || 'not counted').slice(0, 28));
        else qc = h('span', { class: 'tag no', title: 'not a program target' }, '—');
        let ql;
        if (!('path' in f) && !('ql_error' in f)) ql = h('span', { class: 'dim' }, 'writing…');
        else if (f.ql_error) ql = h('span', { class: 'tag inf', title: f.ql_error }, 'QL error');
        else {
          const flags = q.flags || [];
          const snr = q.snr_ref_per_A != null ? q.snr_ref_per_A : q.snr_ref_per_pix;
          ql = h('span', { title: flags.join('\n') }, (q.trace_found ? 'trace' : q.trace_found === false ? 'no trace' : '') + (snr != null ? ` S/N ${OA.num(snr, 1)}` : '') + (flags.length ? ` ⚠${flags.length}` : ''));
        }
        const tr = h('tr', { class: 'clickable' + (f.file === this.sel ? ' sel' : ''), onclick: () => { OA.$('dq-follow').checked = f === list[list.length - 1]; this.selectFrame(f.file); } },
          h('td', { class: 'mono' }, f.file), h('td', null, f.arm), h('td', null, f.image_type), h('td', null, f.object || ''),
          h('td', { class: 'num' }, OA.num(f.exptime, 0)), h('td', { class: 'num' }, f.ut_start || ''),
          h('td', { class: 'num' }, f.airmass != null ? OA.num(f.airmass, 2) : ''), h('td', { class: 'num' }, f.fwhm != null ? OA.num(f.fwhm, 2) + '″' : ''),
          h('td', null, f.snr != null ? `${OA.num(f.snr, 1)} → ${OA.num(f.snr_total, 1)} / ${OA.num(f.goal, 0)}` : ''),
          h('td', null, qc), h('td', null, ql));
        tr.dataset.file = f.file;
        tb.appendChild(tr);
      }
      tbl.appendChild(tb);
      tbl.parentElement.scrollTop = scroll;
    },
    selectFrame(name) {
      this.sel = name;
      const f = this.frames.find((x) => x.file === name);
      this.selReady = f ? ('path' in f) + '|' + ('ql_error' in f) : null;
      for (const tr of OA.$('dq-table').querySelectorAll('tbody tr')) tr.classList.toggle('sel', tr.dataset.file === name);
      const inst = OA.snap ? OA.snap.instrument : '';
      OA.setText(OA.$('dq-title'), `Overview ${inst} ${f ? f.arm.charAt(0).toUpperCase() + f.arm.slice(1) : ''} — ${name}`);
      if (f && !('path' in f) && !('ql_error' in f)) {
        this.img = null;
        this.message(`${name}: the data system is still writing this frame…`);
        OA.$('dq-hdr').textContent = '(waiting for the file)';
        this.ql = null; this.showQL(f); this.plotSpec();
        return;
      }
      if (f && !('path' in f) && f.ql_error) {
        // the data system already told us the file was not written: don't ask for pixels
        this.img = null;
        const why = this.explain({ status: 409 }, name, 'pixels');
        this.message(why);
        this.hdr = null;
        OA.$('dq-hdr').textContent = this.explain({ status: 409 }, name, 'header');
        this.loadQL(name, f);
        return;
      }
      this.loadPixels(name);
      this.loadQL(name, f);
      this.loadHeader(name);
    },
    message(txt) {
      const m = OA.$('dq-msg');
      m.textContent = txt;
      m.classList.toggle('hidden', !txt);
      if (txt) {
        const fit = OA.fit(OA.$('dq-cv'));
        if (fit) { fit.ctx.fillStyle = '#000'; fit.ctx.fillRect(0, 0, fit.w, fit.h); }
      }
    },
    explain(e, name, what) {
      const f = this.frames.find((x) => x.file === name) || {};
      if (e.status === 409) return `${name}: ${what} not available — the FITS file has not been written.\n` + (f.ql_error ? `data system: ${f.ql_error}` : 'The frame writer (obsassist/sim/frames.py) may be disabled or still working.');
      if (e.status === 404) return `${name}: no such frame on the server (was the night reset?).`;
      return `${name}: ${what} failed — ${e.message}`;
    },
    async loadPixels(name) {
      const bin = OA.$('dq-bin').value;
      this.message(`loading ${name} (bin ${bin})…`);
      let r;
      try {
        r = await fetch(`/api/frame/${encodeURIComponent(name)}?bin=${bin}`);
        if (!r.ok) {
          let det = r.statusText;
          try { const j = await r.json(); det = j.detail || det; } catch (e) { /* not json */ }
          const err = new Error(det); err.status = r.status; throw err;
        }
        const shape = (r.headers.get('X-Shape') || '').split(',').map(Number);
        const buf = await r.arrayBuffer();
        if (name !== this.sel) return;
        const [ny, nx] = shape;
        if (!(ny > 0 && nx > 0) || buf.byteLength < 2 * nx * ny) throw new Error(`bad pixel payload (X-Shape ${shape}, ${buf.byteLength} bytes)`);
        let arr;
        if (LE) arr = new Uint16Array(buf, 0, nx * ny);
        else { const dv = new DataView(buf); arr = new Uint16Array(nx * ny); for (let i = 0; i < arr.length; i++) arr[i] = dv.getUint16(2 * i, true); }
        this.img = { name, nx, ny, arr, bin: parseInt(r.headers.get('X-Bin') || bin, 10) || 1 };
        this.message('');
        this.stretch();
        this.paint();
      } catch (e) {
        if (name !== this.sel) return;
        this.img = null;
        this.message(this.explain(e, name, 'pixels'));
      }
    },
    stretch() {
      const im = this.img;
      if (!im) return;
      const mode = OA.$('dq-stretch').value;
      let lo, hi;
      if (mode === 'manual') {
        lo = parseFloat(OA.$('dq-min').value); hi = parseFloat(OA.$('dq-max').value);
        if (!isFinite(lo) || !isFinite(hi) || hi <= lo) { [lo, hi] = zscale(im.arr); }
      } else if (mode === 'zscale') {
        [lo, hi] = zscale(im.arr);
      } else {
        const s = sample(im.arr, 20000);
        if (mode === 'minmax') { lo = s[0]; hi = s[s.length - 1]; let mx = 0, mn = 65535; for (const v of im.arr) { if (v > mx) mx = v; if (v < mn) mn = v; } lo = mn; hi = mx; }
        else if (mode === 'asinh') { lo = zscale(im.arr)[0]; hi = pct(s, 99.95); }
        else { lo = pct(s, 0.5); hi = pct(s, 99.5); }
        if (!(hi > lo)) hi = lo + 1;
      }
      im.lo = lo; im.hi = hi; im.mode = mode;
      if (mode !== 'manual') { OA.$('dq-min').value = Math.round(lo); OA.$('dq-max').value = Math.round(hi); }
      // LUT over the uint16 range
      const lut = new Uint8Array(65536);
      const beta = 0.04, an = Math.asinh(1 / beta);
      for (let v = 0; v < 65536; v++) {
        let x = (v - lo) / (hi - lo);
        x = x < 0 ? 0 : x > 1 ? 1 : x;
        if (mode === 'asinh') x = Math.asinh(x / beta) / an;
        lut[v] = Math.round(255 * x);
      }
      const { nx, ny, arr } = im;
      if (!this.off || this.off.width !== nx || this.off.height !== ny) {
        this.off = document.createElement('canvas');
        this.off.width = nx; this.off.height = ny;
      }
      const octx = this.off.getContext('2d');
      const id = octx.createImageData(nx, ny);
      const d = id.data, night = OA.nightMode;
      // display with row 0 at the bottom (FITS convention)
      for (let y = 0; y < ny; y++) {
        const src = (ny - 1 - y) * nx, dst = y * nx;
        for (let x = 0; x < nx; x++) {
          const c = lut[arr[src + x]], k = 4 * (dst + x);
          d[k] = c; d[k + 1] = night ? 0 : c; d[k + 2] = night ? 0 : c; d[k + 3] = 255;
        }
      }
      octx.putImageData(id, 0, 0);
    },
    paint() {
      const fit = OA.fit(OA.$('dq-cv'));
      if (!fit) return;
      const { ctx, w, h: hh } = fit;
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, hh);
      const im = this.img;
      if (!im || !this.off) return;
      if (im.night !== OA.nightMode) { im.night = OA.nightMode; this.stretch(); }
      const sc = Math.min(w / im.nx, hh / im.ny);
      const dw = im.nx * sc, dh = im.ny * sc, ox = (w - dw) / 2, oy = (hh - dh) / 2;
      ctx.imageSmoothingEnabled = sc < 1;
      ctx.drawImage(this.off, ox, oy, dw, dh);
      this.view = { ox, oy, sc };
      OA.text(ctx, `${im.nx}×${im.ny} (bin ${im.bin}) · ${im.mode} ${Math.round(im.lo)}…${Math.round(im.hi)}`, 6, hh - 6, { size: 10, mono: true, color: '#9ef59e', halo: 'rgba(0,0,0,.7)' });
      if (this.magPos) this.drawMag(this.magPos.x, this.magPos.y);
    },
    magAt(e) {
      const im = this.img, v = this.view;
      if (!im || !v) return;
      const r = e.target.getBoundingClientRect();
      const dx = (e.clientX - r.left - v.ox) / v.sc, dy = (e.clientY - r.top - v.oy) / v.sc;
      const x = Math.floor(dx), yDisp = Math.floor(dy);
      if (x < 0 || yDisp < 0 || x >= im.nx || yDisp >= im.ny) return;
      this.magPos = { x, y: im.ny - 1 - yDisp };
      this.drawMag(this.magPos.x, this.magPos.y);
    },
    drawMag(x, y) {
      const im = this.img;
      if (!im) return;
      const cv = OA.$('dq-magcv');
      const ctx = cv.getContext('2d');
      const N = 40, k = cv.width / N;
      ctx.imageSmoothingEnabled = false;
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, cv.width, cv.height);
      const yd = im.ny - 1 - y;
      ctx.drawImage(this.off, x - N / 2, yd - N / 2, N, N, 0, 0, cv.width, cv.height);
      ctx.strokeStyle = OA.nightMode ? '#ff0000' : '#00ff66'; ctx.lineWidth = 1;
      ctx.strokeRect(N / 2 * k + 0.5, N / 2 * k + 0.5, k, k);
      ctx.beginPath(); ctx.arc((N / 2 + 0.5) * k, (N / 2 + 0.5) * k, 10 * k, 0, 2 * Math.PI); ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);
      // stats within radius 10
      const R = 10, a = im.arr, nx = im.nx, ny = im.ny;
      const at = (i, j) => a[OA.clamp(j, 0, ny - 1) * nx + OA.clamp(i, 0, nx - 1)];
      const med3 = (p, q, r) => (p > q ? (q > r ? q : p > r ? r : p) : (p > r ? p : q > r ? r : q));
      // 3x3 median (robust to single-pixel cosmic rays) for the peak search
      const med9 = (i, j) => {
        const v = [];
        for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) v.push(at(i + di, j + dj));
        v.sort((p, q) => p - q);
        return v[4];
      };
      let n = 0, sum = 0, sum2 = 0, mn = Infinity, mx = -Infinity, px = x, py = y, pv = -Infinity;
      const vals = [];
      for (let j = Math.max(0, y - R); j <= Math.min(ny - 1, y + R); j++) {
        for (let i = Math.max(0, x - R); i <= Math.min(nx - 1, x + R); i++) {
          if ((i - x) ** 2 + (j - y) ** 2 > R * R) continue;
          const v = a[j * nx + i];
          n++; sum += v; sum2 += v * v; vals.push(v);
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
      }
      for (let j = Math.max(0, y - 5); j <= Math.min(ny - 1, y + 5); j++) {
        for (let i = Math.max(0, x - 5); i <= Math.min(nx - 1, x + 5); i++) {
          const v = med9(i, j);
          if (v > pv) { pv = v; px = i; py = j; }
        }
      }
      const mean = sum / Math.max(1, n), dev = Math.sqrt(Math.max(0, sum2 / Math.max(1, n) - mean * mean));
      vals.sort((p, q) => p - q);
      const bg = vals.length ? vals[Math.floor(vals.length * 0.2)] : 0;
      const half = bg + (pv - bg) / 2;
      // half-maximum crossings along x and y, each sample a 3-px median across the walk direction
      const width = (dxs, dys) => {
        let w = 0;
        for (const s of [-1, 1]) {
          let i = 1, prev = pv;
          while (i < 30) {
            const xx = px + s * i * dxs, yy = py + s * i * dys;
            if (xx < 0 || yy < 0 || xx >= nx || yy >= ny) { i = 30; break; }
            const v = med3(at(xx - dys, yy - dxs), at(xx, yy), at(xx + dys, yy + dxs));
            if (v < half) { w += i - 1 + (prev - half) / Math.max(prev - v, 1e-9); break; }
            prev = v; i++;
          }
          if (i >= 30) w += 30;
        }
        return w;
      };
      const contrast = pv - bg > 5 * Math.sqrt(Math.max(bg, 1));
      const fx = contrast ? width(1, 0) : null, fy = contrast ? width(0, 1) : null;
      const val = a[y * nx + x];
      const b = im.bin;
      this.magTable({
        pix: `(${x * b}, ${y * b})`, val, mn, mx, mean, dev,
        fwhm: fx != null ? `${OA.num(Math.min(fx, fy) * b, 1)} px` : '—',
        fxy: fx != null ? `${fx >= 59 ? '>60' : OA.num(fx * b, 1)} / ${fy >= 59 ? '>60' : OA.num(fy * b, 1)}` : '—', peak: `${pv}`, peakAt: `(${px * b}, ${py * b})`,
      });
    },
    magTable(m) {
      const rows = [
        ['Pixel (x,y)', m ? m.pix : '—'], ['value', m ? m.val : '—'], ['Radius', '10'],
        ['min / max', m ? `${m.mn} / ${m.mx}` : '—'], ['mean', m ? OA.num(m.mean, 1) : '—'], ['dev', m ? OA.num(m.dev, 1) : '—'],
        ['peak (med3)', m ? m.peak : '—'], ['at', m ? m.peakAt : '—'], ['flx.fwhm', m ? m.fwhm : '—'], ['fw x/y', m ? m.fxy : '—'],
      ];
      OA.$('dq-magtab').innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${OA.esc(v)}</td></tr>`).join('') +
        `<tr><td colspan="2" class="dim" style="text-align:left;font-family:inherit">unbinned detector px; click to ${this.lock ? 'unlock' : 'lock'}</td></tr>`;
    },
    async loadQL(name, f) {
      OA.setText(OA.$('dq-qlsum'), 'loading…');
      try {
        const ql = await OA.api(`/api/frame/${encodeURIComponent(name)}/ql`);
        if (name !== this.sel) return;
        this.ql = ql;
      } catch (e) {
        if (name !== this.sel) return;
        this.ql = { error: this.explain(e, name, 'quick-look') };
      }
      this.showQL(f);
      this.plotSpec();
    },
    showQL(f) {
      const ql = this.ql, fl = OA.$('dq-flags');
      fl.textContent = '';
      if (f) {
        if (f.image_type === 'object' && f.target) {
          if (f.counted) fl.appendChild(h('span', { class: 'tag good' }, `counted: S/N ${OA.num(f.snr, 1)} → ${OA.num(f.snr_total, 1)}/${OA.num(f.goal, 0)}`));
          else if (f.note) fl.appendChild(h('span', { class: 'tag no' }, f.note));
          else for (const r of f.qc_reasons || ['not counted']) fl.appendChild(h('span', { class: 'tag bad' }, r));
        }
      }
      if (!ql) { OA.setText(OA.$('dq-qlsum'), f ? 'waiting…' : ''); return; }
      if (ql.error) {
        OA.setText(OA.$('dq-qlsum'), '');
        fl.appendChild(h('span', { class: 'tag inf' }, 'no quick-look'));
        fl.appendChild(document.createTextNode(' ' + ql.error));
        return;
      }
      const snr = ql.snr_ref_per_A != null ? `S/N ${OA.num(ql.snr_ref_per_A, 1)}/Å` : ql.snr_ref_per_pix != null ? `S/N ${OA.num(ql.snr_ref_per_pix, 1)}/pix` : '';
      OA.setText(OA.$('dq-qlsum'), [ql.trace_found ? 'trace found' : ql.trace_found === false ? 'NO trace' : '',
        ql.fwhm_arcsec != null ? `FWHM ${OA.num(ql.fwhm_arcsec, 2)}″` : '', ql.peak_adu != null ? `peak ${OA.num(ql.peak_adu, 0)} ADU` : '',
        ql.saturated_frac ? `sat ${OA.num(ql.saturated_frac * 100, 2)}%` : '', ql.sky_adu_per_pix != null ? `sky ${OA.num(ql.sky_adu_per_pix, 1)} ADU/pix` : '', snr].filter(Boolean).join(' · '));
      for (const fg of ql.flags || []) fl.appendChild(h('span', { class: 'tag inf' }, '⚠ ' + fg));
      if (!(ql.flags || []).length) fl.appendChild(h('span', { class: 'tag good' }, 'QL: no flags'));
    },
    plotSpec() {
      const cv = OA.$('dq-speccv');
      const fit = OA.fit(cv);
      if (!fit) return;
      const { ctx, w, h: hh } = fit;
      const th = OA.th();
      ctx.fillStyle = th.bg; ctx.fillRect(0, 0, w, hh);
      const sp = findSpectrum(this.ql);
      if (!sp) {
        const msg = !this.sel ? 'Select a frame.' : this.ql && this.ql.error ? 'No quick-look spectrum for this frame.' : this.ql ? 'Quick-look has no extracted spectrum.' : 'Waiting for the quick-look…';
        OA.text(ctx, msg, w / 2, hh / 2, { align: 'center', color: th.dim, size: 11 });
        return;
      }
      const segs = segments(sp);
      const allW = segs.flatMap((s) => s.w), allF = segs.flatMap((s) => s.f).filter((v) => v != null && isFinite(v));
      const sortedF = [...allF].sort((a, b) => a - b);
      const hasW = !!(sp.wave && (Array.isArray(sp.wave[0]) ? sp.wave[0] : sp.wave));
      const wr = OA.minMax([allW]) || [0, 1];
      let fmax = sortedF.length ? sortedF[Math.floor(0.995 * (sortedF.length - 1))] * 1.15 : 1;
      let fmin = sortedF.length ? Math.min(0, sortedF[Math.floor(0.005 * (sortedF.length - 1))]) : 0;
      if (!(fmax > fmin)) fmax = fmin + 1;
      const snrs = segs.flatMap((s) => s.snr || []).filter((v) => v != null && isFinite(v));
      const smax = snrs.length ? Math.max(...snrs) * 1.15 : 0;
      const p = new OA.Plot(ctx, { x: 50, y: 8, w: w - 50 - (smax ? 40 : 10), h: hh - 26 }, wr, [fmin, fmax]);
      p.ygrid(OA.niceTicks(fmin, fmax, 5), (v) => (Math.abs(v) >= 1e4 ? (v / 1000).toFixed(0) + 'k' : String(+v.toPrecision(4))));
      p.xgrid(OA.niceTicks(wr[0], wr[1], 8), (v) => String(Math.round(v)));
      if (smax) p.y2(OA.niceTicks(0, smax, 4), 0, smax, (v) => String(v), '#d62020');
      p.clip();
      for (const s of segs) {
        if (s.sky) p.line(s.w, s.sky, '#2ca02c', 1);
        p.line(s.w, s.f, OA.nightMode ? '#ff4040' : '#111', 1);
        if (s.snr && smax) p.line(s.w, s.snr, '#d62020', 1, [3, 2], (v) => p.Yr(v, 0, smax));
      }
      const f = this.frames.find((x) => x.file === this.sel);
      if (f && f.lam && hasW) p.vline(f.lam, '#2a6fdb', [4, 3], `ref ${f.lam} Å`);
      p.unclip();
      p.frame();
      OA.text(ctx, hasW ? 'Å' : 'pix', p.b.x + p.b.w, hh - 3, { size: 9, align: 'right', color: th.dim });
      OA.text(ctx, 'flux (ADU)', p.b.x + 4, p.b.y + 11, { size: 9, color: th.fg });
      if (segs.some((s) => s.sky)) OA.text(ctx, 'sky', p.b.x + 64, p.b.y + 11, { size: 9, color: '#2ca02c' });
      if (smax) OA.text(ctx, 'S/N →', p.b.x + 90, p.b.y + 11, { size: 9, color: '#d62020' });
    },
    async loadHeader(name) {
      OA.$('dq-hdr').textContent = 'loading…';
      try {
        const hd = await OA.api(`/api/frame/${encodeURIComponent(name)}/header`);
        if (name !== this.sel) return;
        this.hdr = hd;
      } catch (e) {
        if (name !== this.sel) return;
        this.hdr = null;
        OA.$('dq-hdr').textContent = this.explain(e, name, 'header');
        return;
      }
      this.showHeader();
    },
    showHeader() {
      if (!this.hdr) return;
      const flt = OA.$('dq-hdrfilter').value.trim().toUpperCase();
      const lines = Object.entries(this.hdr).map(([k, v]) => fitsCard(k, v)).filter((l) => !flt || l.toUpperCase().includes(flt));
      OA.$('dq-hdr').textContent = lines.join('\n') + (flt ? '' : '\nEND');
    },
  });
})();
