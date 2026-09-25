/* obsassist console — instrument GUI (Tk style, one panel per arm), TCS / operator panel,
   guider / slit-viewer rendering, operator messages. */
'use strict';

(function () {
  const OA = window.OA, h = OA.h;

  // ------------------------------------------------------------------ editable-field helpers
  // A field is not overwritten while it has focus, nor for a short hold after the user changed it
  // (until the snapshot shows the new value).
  function hold(el, ms) { el._hold = Date.now() + ms; }
  function bindField(el, send) {
    el.addEventListener('focus', () => { el._focused = true; });
    el.addEventListener('blur', () => { el._focused = false; if (!el.classList.contains('pending')) hold(el, 600); });
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { el.value = el._server != null ? el._server : el.value; el.classList.remove('pending'); el.blur(); }
    });
    el.addEventListener('change', () => {
      const v = String(el.value).trim();
      if (v === '' && el.tagName !== 'SELECT' && !el.dataset.allowEmpty) { el.value = el._server || ''; return; }
      if (el._server != null && same(v, el._server)) return;
      el.classList.add('pending');
      el._want = v;
      hold(el, 4000);
      Promise.resolve(send(v)).then((r) => {
        if (r && r !== 'ok' && !/^(ok; )*ok$/.test(r)) { el.classList.remove('pending'); el._hold = 0; el.title = r; }
      });
    });
  }
  function same(a, b) {
    a = String(a).trim(); b = String(b).trim();
    if (a === b) return true;
    const x = parseFloat(a), y = parseFloat(b);
    return /^[-+.\d]+$/.test(a) && /^[-+.\d]+$/.test(b) && isFinite(x) && isFinite(y) && Math.abs(x - y) < 1e-6;
  }
  function setField(el, v) {
    v = v == null ? '' : String(v);
    el._server = v;
    if (el.classList.contains('pending')) {
      if (same(v, el._want)) { el.classList.remove('pending'); el.title = ''; }
      else if (Date.now() < el._hold) return;
      else el.classList.remove('pending');
    }
    if (el._focused || document.activeElement === el) return;
    if (el._hold && Date.now() < el._hold) return;
    if (el.value !== v) {
      if (el.tagName === 'SELECT' && ![...el.options].some((o) => o.value === v)) el.appendChild(h('option', { value: v }, v));
      el.value = v;
    }
  }
  const q = (s) => '"' + String(s).replace(/(["\\])/g, '\\$1') + '"';
  const fmtExp = (v) => (v == null ? '' : String(Math.round(v * 10) / 10));
  const IMAGE_TYPES = ['object', 'flat', 'arc', 'bias', 'dark', 'sky'];
  const DISPERSER = { 'MIKE-BLUE': 'echelle + prism (fixed)', 'MIKE-RED': 'echelle + prism (fixed)', 'LRIS-B600': '600/4000 grism', 'LRIS-R400': '400/8500 grating' };
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

  function ro(cls = '') { return h('span', { class: 'ro ' + cls }); }
  function nosim(txt = '—') { return h('span', { class: 'ro nosim', title: 'not simulated by the console backend' }, txt); }

  // ------------------------------------------------------------------ instrument GUI
  const inst = OA.register({
    name: 'instrument',
    sig: null,
    f: {},
    arms: {},
    snap(s) {
      const sig = s.instrument + '|' + Object.keys(s.arms).join(',');
      if (sig !== this.sig) this.build(s, sig);
      this.update(s);
    },
    reset() { this.sig = null; },
    build(s, sig) {
      this.sig = sig;
      const root = OA.$('inst');
      root.textContent = '';
      const f = this.f = {};
      const armNames = Object.keys(s.arms);
      OA.setText(OA.$('inst-title'), `${s.instrument} — ${s.telescope}`);
      // menubar row
      f.datapath = ro('mono datapath');
      f.disk = h('span', { class: 'disk', title: 'data disk usage' }, h('i'));
      f.diskTxt = h('span', { class: 'mono' });
      f.ut = h('span', { class: 'ro mono', style: 'width:6.5em;text-align:center;font-weight:bold' });
      const menu = (label, items) => h('span', { class: 'tk-menu', tabindex: '0' }, label,
        h('div', { class: 'tk-drop' }, items.map(([t, fn]) => h('div', { onclick: (e) => { e.stopPropagation(); document.activeElement.blur(); fn(); } }, t))));
      root.appendChild(h('div', { class: 'tk-menubar' },
        menu('File', [['Reset night…', () => OA.$('prog-reset').click()], ['Debrief…', () => OA.$('btn-debrief').click()]]),
        menu('Options', [['Toggle night mode', () => OA.setNightMode(!OA.nightMode)], ['Fast readout (all)', () => OA.cmd('all speed fast')], ['Slow readout (all)', () => OA.cmd('all speed slow')], ['Binning 1x1 (all)', () => OA.cmd('all bin 1 1')], ['Binning 2x2 (all)', () => OA.cmd('all bin 2 2')]]),
        h('label', null, 'DataPath'), f.datapath,
        h('label', null, 'Disk'), f.disk, f.diskTxt,
        h('label', null, 'UT'), f.ut));

      // shared block
      f.imgno = ro('mono');
      f.object = h('input', { title: 'object name for all arms (all object …)' });
      bindField(f.object, (v) => OA.cmd('all object ' + v));
      f.itype = h('select', { title: 'image type for all arms' }, h('option', { value: '' }, '(mixed)'), IMAGE_TYPES.map((t) => h('option', { value: t }, t)));
      bindField(f.itype, (v) => v && OA.cmd('all type ' + v));
      f.config = ro('mono');
      f.ra = ro('mono'); f.dec = ro('mono'); f.epoch = ro('mono'); f.airmass = ro('mono');
      f.az = ro('mono'); f.el = ro('mono'); f.rot = ro('mono'); f.slitang = ro('mono');
      f.slit = ro('mono');
      f.lampDot = h('span', { class: 'lampdot' }); f.lampTxt = h('span');
      f.lamp = h('span', { class: 'ro' }, f.lampDot, f.lampTxt);
      f.tOut = ro('mono'); f.focusdT = ro('mono'); f.tInst = ro('mono'); f.tDome = ro('mono'); f.tCcd = {};
      f.comment = h('input', { title: 'comment for all arms (all comment …)', 'data-allow-empty': '1' });
      bindField(f.comment, (v) => OA.cmd('all comment ' + v));
      const L = (t, title) => h('label', title ? { title } : null, t);
      const tSpecs = [];
      for (const a of armNames) { f.tCcd[a] = ro('mono'); tSpecs.push([L(`T.CCD-${cap(a)}`), f.tCcd[a]]); }
      while (tSpecs.length < 2) tSpecs.push([h('span'), h('span')]);
      root.appendChild(h('div', { class: 'shared' },
        L('Image#'), f.imgno, L('Object'), h('span', { class: 'span3', style: 'display:flex' }, f.object), L('ImageType'), f.itype,
        L('R.A.'), f.ra, L('Dec.'), f.dec, L('Epoch'), f.epoch, L('Airmass'), f.airmass,
        L('Azim.'), f.az, L('Elev.'), f.el, L('Rot.Angle'), f.rot, L('Slit Angle', 'slit position angle on the sky'), f.slitang,
        L('SlitSize'), f.slit, L('Lamp'), f.lamp, L(`Temp.${s.instrument}`), f.tInst, L('Temp.Dome'), f.tDome,
        ...tSpecs.flat(), L('Temp.Outs'), f.tOut, L('Focus ΔT', 'temperature change since the last telescope focus'), f.focusdT,
        L('Config'), h('span', { class: 'span3', style: 'display:flex' }, f.config), L('Comment'), h('span', { class: 'span3', style: 'display:flex' }, f.comment)));
      f.object.style.flex = '1'; f.comment.style.flex = '1'; f.config.style.flex = '1';
      for (const el of root.querySelectorAll('.shared .span3 > *')) el.style.width = '100%';
      const bAll = (label, cmd, cls) => h('button', { class: 'tk-btn ' + (cls || ''), onclick: () => OA.cmd(cmd) }, label);
      root.appendChild(h('div', { class: 'shared-btns' },
        h('b', null, 'All arms:'), bAll('Start all', 'start all', 'start'), bAll('Pause all', 'pause all'), bAll('Resume all', 'resume all'),
        bAll('Stop all', 'stop all'), bAll('Abort all', 'abort all', 'abort'),
        h('span', { class: 'spacer' }),
        h('span', { class: 'dim', title: 'Stop = end the exposure now and read it out (counts); Abort = discard it' }, 'Stop = read out now · Abort = discard')));

      // arm panels
      const armsBox = h('div', { class: 'arms' });
      this.arms = {};
      for (const name of armNames) {
        const a = s.arms[name];
        const g = this.arms[name] = {};
        g.imgno = ro('mono');
        g.exptime = h('input', { title: `${name} exptime <s>` });
        bindField(g.exptime, (v) => OA.cmd(`${name} exptime ${v}`));
        g.runtime = ro('mono'); g.utstart = ro('mono');
        g.loops = h('input', { title: `${name} loops <n>` });
        bindField(g.loops, (v) => OA.cmd(`${name} loops ${v}`));
        g.doing = ro('mono');
        g.shutterTxt = h('span'); g.shutterDot = h('span', { class: 'shutter-dot' });
        g.shutter = h('span', { class: 'ro' }, g.shutterTxt, g.shutterDot);
        g.itype = h('select', { title: `${name} type <t>` }, IMAGE_TYPES.map((t) => h('option', { value: t }, t)));
        bindField(g.itype, (v) => OA.cmd(`${name} type ${v}`));
        g.object = h('input', { title: `${name} object <name>` });
        bindField(g.object, (v) => OA.cmd(`${name} object ${v}`));
        g.comment = h('input', { title: `${name} comment <text>`, 'data-allow-empty': '1' });
        bindField(g.comment, (v) => OA.cmd(`${name} comment ${v}`));
        g.readout = ro('mono');
        g.speed = h('select', { title: `${name} speed fast|slow` }, h('option', { value: 'Fast' }, 'Fast'), h('option', { value: 'Slow' }, 'Slow'));
        bindField(g.speed, (v) => OA.cmd(`${name} speed ${v.toLowerCase()}`));
        const binOpts = () => [1, 2, 3, 4].map((b) => h('option', { value: String(b) }, String(b)));
        g.xbin = h('select', { title: 'spatial binning' }, binOpts());
        g.ybin = h('select', { title: 'spectral binning' }, binOpts());
        bindField(g.xbin, (v) => OA.cmd(`${name} bin ${v} ${g.ybin._server || g.ybin.value}`));
        bindField(g.ybin, (v) => OA.cmd(`${name} bin ${g.xbin._server || g.xbin.value} ${v}`));
        g.lampDot = h('span', { class: 'lampdot' }); g.lampTxt = h('span');
        g.lamp = h('span', { class: 'ro' }, g.lampDot, g.lampTxt);
        g.rdtime = ro('mono');
        g.pill = h('span', { class: 'pill st-idle' }, 'IDLE');
        g.barFill = h('i'); g.barTxt = h('span');
        g.bar = h('div', { class: 'progress' }, g.barFill, g.barTxt);
        g.msg = h('div', { class: 'arm-msg' });
        g.bStart = h('button', { class: 'tk-btn start', onclick: () => OA.cmd('start ' + name) }, 'Start');
        g.bPause = h('button', { class: 'tk-btn', onclick: () => OA.cmd((g.bPause.textContent === 'Resume' ? 'resume ' : 'pause ') + name) }, 'Pause');
        g.bStop = h('button', { class: 'tk-btn', title: 'end the exposure now and read out', onclick: () => OA.cmd('stop ' + name) }, 'Stop');
        g.bAbort = h('button', { class: 'tk-btn abort', title: 'discard the exposure', onclick: () => OA.cmd('abort ' + name) }, 'Abort');
        const grating = DISPERSER[a.config] || a.config;
        const panel = h('fieldset', { class: `tk-lf arm ${name}` },
          h('legend', { title: a.description }, h('span', { class: 'dot' }), `${s.instrument} ${cap(name)}`, h('span', { class: 'dim', style: 'font-weight:normal' }, `  ${a.config}`)),
          h('div', { class: 'arm-grid' },
            L('Image#'), g.imgno, L('Exp.Time'), g.exptime,
            L('RunTime'), g.runtime, L('UT-Start'), g.utstart,
            L('Loops'), g.loops, L('Doing'), g.doing,
            L('Shutter'), g.shutter, L('ImageType'), g.itype,
            L('Object'), h('span', { class: 'span3', style: 'display:flex' }, g.object),
            L('Comment'), h('span', { class: 'span3', style: 'display:flex' }, g.comment),
            L('Readout'), g.readout, L('Speed'), g.speed,
            L('X-Bin'), g.xbin, L('Y-Bin'), g.ybin,
            L('CamFocus'), nosim(), L('Grating', 'disperser'), h('span', { class: 'ro nosim', title: 'fixed by the configuration' }, grating),
            L('Filter'), nosim(), L('Lamp'), g.lamp),
          h('div', { class: 'arm-state' }, g.pill, g.bar),
          g.msg,
          h('div', { class: 'arm-btns' }, g.bStart, g.bPause, g.bStop, g.bAbort));
        g.object.style.flex = '1'; g.comment.style.flex = '1';
        armsBox.appendChild(panel);
      }
      root.appendChild(armsBox);
    },
    update(s) {
      const f = this.f, arms = Object.values(s.arms), t = s.tcs;
      OA.setText(OA.$('inst-sub'), arms.map((a) => a.description).join(' | '));
      OA.setText(f.datapath, s.data_dir);
      f.datapath.title = s.data_dir;
      const nFrames = (s.frames || []).length;
      const used = OA.clamp(37 + 0.08 * (arms.reduce((n, a) => n + a.image_no - 1, 0)), 0, 99);
      f.disk.firstChild.style.width = used.toFixed(1) + '%';
      OA.setText(f.diskTxt, used.toFixed(0) + '%');
      f.disk.title = `data disk ${used.toFixed(1)}% used (illustrative) · ${nFrames} recent frames`;
      OA.setText(f.ut, s.utc.slice(11));
      OA.setText(f.imgno, arms.map((a) => a.prefix + String(a.image_no).padStart(4, '0')).join('  '));
      const objs = new Set(arms.map((a) => a.object));
      setField(f.object, objs.size === 1 ? arms[0].object : '');
      f.object.placeholder = objs.size > 1 ? '(differs per arm)' : (t.name && t.state !== 'parked' ? t.name : '');
      const types = new Set(arms.map((a) => a.image_type));
      setField(f.itype, types.size === 1 ? arms[0].image_type : '');
      OA.setText(f.config, arms.map((a) => a.config).join(' + '));
      const parked = t.state === 'parked';
      OA.setText(f.ra, parked ? '—' : t.ra);
      OA.setText(f.dec, parked ? '—' : t.dec);
      OA.setText(f.epoch, '2000.0');
      OA.setText(f.airmass, parked ? '—' : OA.num(t.airmass, 3));
      OA.setText(f.az, OA.num(t.az, 2));
      OA.setText(f.el, OA.num(t.alt, 2));
      OA.setText(f.rot, t.rot_pa == null ? OA.num(t.parallactic, 1) : OA.num(t.rot_pa, 1));
      OA.setText(f.slitang, t.rot_pa == null ? 'parallactic' : 'PA ' + OA.num(t.rot_pa, 1));
      OA.setText(f.slit, arms[0].slit);
      const lamps = arms.map((a) => a.lamp).filter((l) => l && l !== 'off');
      const lamp = lamps.length ? [...new Set(lamps)].join('+') : 'off';
      OA.setCls(f.lampDot, 'lampdot ' + (lamps[0] || ''));
      OA.setText(f.lampTxt, lamp);
      OA.setText(f.tOut, OA.num(s.weather.temp, 1) + ' °C');
      if (s.temps) {
        OA.setText(f.tInst, OA.num(s.temps.instrument, 1) + ' °C');
        OA.setText(f.tDome, OA.num(s.temps.dome, 1) + ' °C');
        for (const a of Object.keys(f.tCcd)) OA.setText(f.tCcd[a], OA.num((s.temps.ccd || {})[a], 1) + ' °C');
      }
      OA.setText(f.focusdT, OA.num(s.focus.dT, 2) + ' °C' + (s.focus.running ? ' (focusing)' : ''));
      f.focusdT.style.color = s.focus.dT > 1.5 ? OA.col('#c62828') : '';
      setField(f.comment, new Set(arms.map((a) => a.comment)).size === 1 ? arms[0].comment : '');

      for (const [name, a] of Object.entries(s.arms)) {
        const g = this.arms[name];
        if (!g) continue;
        OA.setText(g.imgno, String(a.image_no));
        setField(g.exptime, fmtExp(a.exp_time));
        OA.setText(g.runtime, a.state === 'idle' ? '0' : OA.num(a.elapsed_s, 0));
        OA.setText(g.utstart, a.ut_start || '—');
        setField(g.loops, String(a.loops));
        OA.setText(g.doing, `${a.doing} / ${a.loops}`);
        const open = a.shutter === 'Open';
        OA.setText(g.shutterTxt, a.shutter);
        OA.setCls(g.shutterDot, 'shutter-dot' + (open ? ' open' : ''));
        setField(g.itype, a.image_type);
        setField(g.object, a.object);
        g.object.placeholder = a.image_type === 'object' && t.state !== 'parked' ? t.name : '';
        setField(g.comment, a.comment);
        OA.setText(g.readout, a.readout + ` (${OA.num(a.readout_s, 0)} s)`);
        setField(g.speed, cap(String(a.speed).toLowerCase()));
        setField(g.xbin, String(a.binning[0]));
        setField(g.ybin, String(a.binning[1]));
        OA.setCls(g.lampDot, 'lampdot ' + (a.lamp !== 'off' ? a.lamp : ''));
        OA.setText(g.lampTxt, a.lamp);
        // state
        const st = a.state;
        OA.setText(g.pill, st.toUpperCase());
        OA.setCls(g.pill, 'pill st-' + st);
        let frac = 0, txt = '';
        if (st === 'exposing' || st === 'paused') {
          frac = a.exp_time > 0 ? a.elapsed_s / a.exp_time : 1;
          txt = st === 'paused' ? `paused · ${OA.num(a.remaining_s, 0)} s left` : `${OA.num(a.remaining_s, 0)} s left  (${OA.num(a.elapsed_s, 0)} / ${OA.num(a.exp_time, 0)} s)`;
        } else if (st === 'reading') {
          frac = a.read_progress || 0;
          txt = 'READOUT ' + Math.round(frac * 100) + '%';
        } else if (st === 'clearing') { frac = 1; txt = 'clearing'; }
        else { txt = a.last_file ? 'last ' + a.last_file : 'ready'; }
        g.barFill.style.width = (OA.clamp(frac, 0, 1) * 100).toFixed(1) + '%';
        OA.setCls(g.bar, 'progress ' + st);
        OA.setText(g.barTxt, txt);
        const next = a.prefix + String(a.image_no).padStart(4, '0');
        OA.setText(g.msg, a.message || (st === 'idle' ? `idle · next ${next}.fits · readout ${a.speed} ${a.binning[0]}x${a.binning[1]} ~${OA.num(a.readout_s, 0)} s` : st));
        g.bStart.disabled = st !== 'idle';
        OA.setText(g.bPause, st === 'paused' ? 'Resume' : 'Pause');
        g.bPause.disabled = !(st === 'exposing' || st === 'paused');
        g.bStop.disabled = !(st === 'exposing' || st === 'paused');
        g.bAbort.disabled = !(st === 'exposing' || st === 'paused' || st === 'reading');
      }
    },
    redraw() { /* DOM only */ },
  });
  function L(t, title) { return h('label', title ? { title } : null, t); }

  // ------------------------------------------------------------------ TCS panel
  const tcs = OA.register({
    name: 'tcs',
    optSig: null,
    init() {
      this.msg = h('div', { class: 'dim small', style: 'min-height:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis' });
      OA.$('win-tcs').querySelector('fieldset').appendChild(this.msg);
      const say = (r) => { OA.setText(this.msg, r && r !== 'ok' ? r : ''); this.msg.style.color = r && r !== 'ok' ? OA.col('#b3261e') : ''; };
      OA.$('req-target').onchange = (e) => { if (e.target.value) OA.selectTarget(e.target.value); };
      OA.$('req-go').onclick = async () => {
        const n = OA.$('req-target').value;
        if (!n) { say('choose a target first'); return; }
        say(await OA.cmd('goto ' + q(n)));
      };
      OA.$('req-manual').onclick = async () => {
        const ra = OA.$('req-ra').value.trim(), dec = OA.$('req-dec').value.trim(), nm = OA.$('req-name').value.trim() || 'manual';
        if (!ra || !dec) { say('enter R.A. (hh:mm:ss) and Dec. (±dd:mm:ss)'); return; }
        say(await OA.cmd(`goto ${ra} ${dec} ${q(nm)}`));
      };
      OA.$('off-go').onclick = async () => {
        const a = parseFloat(OA.$('off-ra').value || '0'), d = parseFloat(OA.$('off-dec').value || '0');
        if (!isFinite(a) || !isFinite(d)) { say('offsets must be numbers (arcsec)'); return; }
        say(await OA.cmd(`offset ${a} ${d}`));
      };
      OA.$('focus-go').onclick = async () => say(await OA.cmd('focus'));
      OA.$('rot-set').onclick = async () => {
        const pa = parseFloat(OA.$('rot-pa').value);
        if (!isFinite(pa)) { say('enter a position angle (deg)'); return; }
        say(await OA.cmd('rot ' + pa));
      };
      OA.$('rot-par').onclick = async () => say(await OA.cmd('rot parallactic'));
      for (const id of ['req-ra', 'req-dec', 'req-name']) OA.$(id).addEventListener('keydown', (e) => { if (e.key === 'Enter') OA.$('req-manual').click(); });
      for (const id of ['off-ra', 'off-dec']) OA.$(id).addEventListener('keydown', (e) => { if (e.key === 'Enter') OA.$('off-go').click(); });
      OA.$('rot-pa').addEventListener('keydown', (e) => { if (e.key === 'Enter') OA.$('rot-set').click(); });
    },
    reset() { this.optSig = null; },
    select(name) {
      const sel = OA.$('req-target');
      if ([...sel.options].some((o) => o.value === name)) sel.value = name;
    },
    snap(s) {
      const t = s.tcs;
      const st = OA.$('tcs-state');
      OA.setText(st, t.state.toUpperCase() + (t.busy_s > 0 ? ' ' + OA.mmss(t.busy_s) : ''));
      OA.setCls(st, 'pill st-' + t.state);
      OA.setText(OA.$('tcs-name'), t.name);
      OA.setText(OA.$('tcs-busy'), t.busy_s > 0 ? `${t.state} · ${OA.mmss(t.busy_s)} left` : (s.dome.open ? '' : 'dome closed'));
      const parked = t.state === 'parked';
      OA.setText(OA.$('tcs-ra'), parked ? '—' : t.ra);
      OA.setText(OA.$('tcs-dec'), parked ? '—' : t.dec);
      OA.setText(OA.$('tcs-alt'), OA.num(t.alt, 2) + '°');
      OA.setText(OA.$('tcs-az'), OA.num(t.az, 2) + '°');
      OA.setText(OA.$('tcs-x'), parked ? '—' : OA.num(t.airmass, 3));
      const ha = t.ha;
      OA.setText(OA.$('tcs-ha'), parked ? '—' : (ha >= 0 ? '+' : '−') + OA.hm(Math.abs(ha) * 60) + ' h');
      OA.setText(OA.$('tcs-par'), parked ? '—' : OA.num(t.parallactic, 1) + '°');
      OA.setText(OA.$('tcs-rot'), t.rot_pa == null ? 'parallactic' : 'PA ' + OA.num(t.rot_pa, 1) + '°');
      OA.setText(OA.$('tcs-off'), `${OA.num(t.offset[0], 1)}″ E  ${OA.num(t.offset[1], 1)}″ N`);
      const fe = OA.$('tcs-focus');
      OA.setText(fe, s.focus.running ? 'running…' : `ΔT ${OA.num(s.focus.dT, 2)} °C`);
      fe.style.color = !s.focus.running && s.focus.dT > 1.5 ? OA.col('#c62828') : '';
      fe.title = `Temperature change since the last focus (focused at ${OA.num(s.focus.temp_at_focus, 1)} °C). Refocus when it drifts by more than ~1.5 °C.`;
      OA.$('focus-go').disabled = !!s.focus.running;
      // request selector: announced program targets
      const sig = s.targets.map((x) => x.name + (x.announced ? 1 : 0) + (x.done ? 1 : 0) + (x.visible ? 1 : 0)).join('|');
      if (sig !== this.optSig) {
        this.optSig = sig;
        const sel = OA.$('req-target');
        const cur = sel.value || OA.selTarget || '';
        sel.textContent = '';
        sel.appendChild(h('option', { value: '' }, '— program target —'));
        for (const x of s.targets) {
          if (!x.announced) continue;
          const mark = x.done ? '✓ ' : x.visible ? '' : '· ';
          sel.appendChild(h('option', { value: x.name, title: x.notes || '' }, `${mark}P${x.priority}  ${x.name}${x.visible ? '' : '  (not observable now)'}`));
        }
        if ([...sel.options].some((o) => o.value === cur)) sel.value = cur;
      }
      guider.render(s);
      chat.snap(s);
    },
  });

  // ------------------------------------------------------------------ guider / slit viewer
  let gaussSpare = null;
  function gauss() {
    if (gaussSpare != null) { const v = gaussSpare; gaussSpare = null; return v; }
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    v = Math.random();
    const r = Math.sqrt(-2 * Math.log(u));
    gaussSpare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }
  const guider = {
    W: 180, H: 100,           // image pixels (drawn 2x)
    scale: 0.14,              // arcsec per image pixel (0.07"/px on the 2x display)
    off: null, img: null,
    acqStart: null,
    render(s) {
      const cv = OA.$('guider-cv');
      const fit = OA.fit(cv);
      if (!fit) return;
      const ctx = fit.ctx;
      const W = this.W, H = this.H, sc = this.scale;
      if (!this.off) {
        this.off = document.createElement('canvas');
        this.off.width = W; this.off.height = H;
        this.img = this.off.getContext('2d').createImageData(W, H);
      }
      const t = s.tcs, g = s.guider;
      const arm = Object.values(s.arms)[0];
      const [sw, sl] = (arm.slit || '1.00x5.00').split('x').map(Number);
      const cx = W / 2, cy = H / 2;
      const halfW = sw / 2 / sc, halfL = Math.min(sl / 2 / sc, W);
      // sky level on the slit-viewer (ADU per pixel per frame)
      let sky;
      if (!s.dome.open) sky = 3;
      else if (s.sun_alt > -6) sky = 900;
      else if (s.sun_alt > -12) sky = 180;
      else if (s.sun_alt > -18) sky = 50;
      else sky = 22;
      if (s.dome.open && s.moon.alt > 0) sky += 60 * s.moon.illum * Math.sin(s.moon.alt * Math.PI / 180);
      // star
      const guiding = g.state === 'guiding' && s.dome.open;
      const acquiring = t.state === 'acquiring' && s.dome.open;
      let star = null;
      if (guiding || acquiring) {
        const tg = s.targets.find((x) => x.name === t.name);
        const mag = tg ? tg.mag : 14;
        const fr = g.flux_ratio != null ? g.flux_ratio : 1;
        const fwhm = g.fwhm != null ? g.fwhm : 1.0;
        const sig = fwhm / 2.3548 / sc;
        const total = 4e5 * Math.pow(10, -0.4 * (mag - 12)) * fr;
        let dx = t.offset[0] / sc, dy = t.offset[1] / sc;
        if (acquiring) {
          if (this.acqStart == null || this.acqName !== t.name) { this.acqStart = Math.max(t.busy_s, 1); this.acqName = t.name; }
          const f = OA.clamp(t.busy_s / this.acqStart, 0, 1);
          dx += 38 * f; dy += -18 * f;
        } else this.acqStart = null;
        const jit = guiding ? 0.05 / sc : 0.15 / sc;
        star = { x: cx + dx + jit * gauss(), y: cy + dy + jit * gauss(), sig, peak: total / (2 * Math.PI * sig * sig), fwhm, fr };
      }
      const d = this.img.data;
      const peak = star ? Math.min(star.peak, 60000) : 0;
      const hi = Math.max(sky * 6, peak * 0.9 + sky, 30);
      const soft = Math.max(sky * 1.5, 8);
      const norm = Math.asinh(hi / soft);
      const night = OA.nightMode;
      const slewing = t.state === 'slewing';
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const inSlit = Math.abs(y + 0.5 - cy) < halfW && Math.abs(x + 0.5 - cx) < halfL;
          const refl = inSlit ? 0.004 : 0.85;
          let v = sky;
          if (star) {
            const ddx = x + 0.5 - star.x, ddy = y + 0.5 - star.y;
            const r2 = (ddx * ddx + ddy * ddy) / (2 * star.sig * star.sig);
            if (r2 < 30) v += Math.min(star.peak, 60000) * Math.exp(-r2);
            // faint Moffat-like wings
            v += 0.02 * Math.min(star.peak, 60000) / (1 + (ddx * ddx + ddy * ddy) / (9 * star.sig * star.sig)) ** 2;
          }
          v *= refl;
          v += Math.sqrt(Math.max(v, 1)) * gauss() + 3 * gauss();
          if (slewing) v += (x * 7 + y * 13) % 17 === 0 ? 20 * Math.random() : 0;
          let c = Math.asinh(Math.max(v, 0) / soft) / norm;
          c = Math.round(255 * OA.clamp(c, 0, 1));
          const i = 4 * (y * W + x);
          d[i] = c; d[i + 1] = night ? 0 : c; d[i + 2] = night ? 0 : c; d[i + 3] = 255;
        }
      }
      this.off.getContext('2d').putImageData(this.img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(this.off, 0, 0, fit.w, fit.h);
      // overlays
      const k = fit.w / W;
      const fg = night ? '#d00000' : '#9ef59e';
      const fgw = night ? '#ff2020' : '#ffffff';
      ctx.strokeStyle = night ? '#600000' : 'rgba(120,200,255,.55)';
      ctx.lineWidth = 1;
      // slit end markers
      ctx.beginPath();
      for (const sx of [-1, 1]) { const x = (cx + sx * halfL) * k; ctx.moveTo(x, (cy - halfW) * k - 6); ctx.lineTo(x, (cy - halfW) * k - 1); ctx.moveTo(x, (cy + halfW) * k + 1); ctx.lineTo(x, (cy + halfW) * k + 6); }
      ctx.stroke();
      const T = (str, x, y, o) => OA.text(ctx, str, x, y, Object.assign({ size: 10, mono: true, color: fg, halo: 'rgba(0,0,0,.7)' }, o));
      const state = !s.dome.open ? 'DOME CLOSED' : guiding ? 'GUIDING' : acquiring ? 'ACQUIRING' : t.state === 'slewing' ? 'SLEWING' : 'NO GUIDE STAR';
      T(state, 5, 13, { bold: true, color: guiding ? fg : (night ? '#ff2020' : '#ffd24a') });
      T(t.name, 5, 25, { color: fgw });
      T(s.utc.slice(11) + ' UT', fit.w - 5, 13, { align: 'right' });
      if (guiding && g.fwhm != null) {
        T(`FWHM ${OA.num(g.fwhm, 2)}″   flux ${OA.num(g.flux_ratio, 2)}`, 5, fit.h - 6, { size: 11, bold: true, color: fgw });
      } else T(`slit ${arm.slit}″`, 5, fit.h - 6, { color: fg });
      // scale bar (2") + compass
      const bar = 2 / sc * k;
      const bx = fit.w - 10 - bar, by = fit.h - 10;
      ctx.strokeStyle = fg; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx + bar, by); ctx.stroke();
      T('2″', bx + bar / 2, by - 3, { align: 'center', size: 9 });
      const ax = fit.w - 22, ay = 44;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax, ay - 14); ctx.moveTo(ax, ay); ctx.lineTo(ax - 14, ay); ctx.stroke();
      T('N', ax, ay - 16, { align: 'center', size: 9 });
      T('E', ax - 17, ay + 3, { align: 'right', size: 9 });
      OA.setText(OA.$('guider-sub'), guiding ? `FWHM ${OA.num(g.fwhm, 2)}″ · flux ${OA.num(g.flux_ratio, 2)}` : state.toLowerCase());
      this.strip(s);
    },
    strip(s) {
      if (!OA.every('gstrip', 900)) return;
      const cv = OA.$('guider-strip');
      const fit = OA.fit(cv);
      if (!fit) return;
      const { ctx, w, h: hh } = fit;
      ctx.fillStyle = OA.nightMode ? '#000' : '#0a0f14';
      ctx.fillRect(0, 0, w, hh);
      const H = OA.hist;
      const t1 = s.t, t0 = t1 - 120;
      const p = new OA.Plot(ctx, { x: 26, y: 4, w: w - 30, h: hh - 16 }, [t0, t1], [0, 2]);
      p.th = Object.assign({}, p.th, { grid: OA.nightMode ? '#200000' : '#1f2a36', fg: OA.nightMode ? '#800000' : '#8fa3b8', axis: OA.nightMode ? '#300000' : '#33475c' });
      p.ygrid([0, 0.5, 1, 1.5, 2], (v) => v.toFixed(1));
      p.xgrid(OA.utTicks(t0, t1, 30), (t) => OA.fmtUT(t));
      if (H && H.t) {
        p.clip();
        p.line(H.t, H.dimm, '#4f8fff', 1);
        p.dots(H.t, H.guider_fwhm, '#ff5050', 1.6);
        p.unclip();
      }
      p.frame();
      OA.text(ctx, 'guider FWHM', 30, 13, { size: 9, color: '#ff6060' });
      OA.text(ctx, 'DIMM', 96, 13, { size: 9, color: '#6fa0ff' });
    },
  };
  OA.guider = guider;

  // ------------------------------------------------------------------ operator chat
  const chat = {
    seen: new Set(),
    n: 0,
    snap(s) {
      const box = OA.$('to-chat');
      let added = false;
      const stick = box.scrollHeight - box.scrollTop - box.clientHeight < 30;
      for (const e of s.log) {
        const mine = (e.who === 'OBS' || e.who === 'AST') && /^(Request:|.*: (ABORT|stop))/.test(e.text);
        if (e.who !== 'TO' && !mine) continue;
        const k = e.t + '|' + e.who + '|' + e.text;
        if (this.seen.has(k)) continue;
        this.seen.add(k);
        box.appendChild(h('div', { class: 'msg' + (mine ? ' me' : '') + (e.level === 'warn' ? ' warn' : '') },
          h('span', { class: 'mt' }, e.utc.slice(0, 5) + (mine ? ' you' : ' TO')), e.text));
        added = true;
        if (++this.n > 200) { box.removeChild(box.firstChild); this.n--; }
      }
      if (added && stick) box.scrollTop = box.scrollHeight;
    },
    reset() { this.seen.clear(); this.n = 0; OA.$('to-chat').textContent = ''; },
  };
  OA.register({ name: 'chat', reset() { chat.reset(); }, redraw() { if (OA.snap) guider.render(OA.snap); } });
})();
