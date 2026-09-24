/* obsassist console — weather page in the style of weather.lco.cl:
   seeing strip (DIMM + guider), temperature / dew point / humidity, wind + gusts,
   pressure, wind rose, big current values, limit lines and "High wind!" overlays. */
'use strict';

(function () {
  const OA = window.OA;
  const MPH = 2.237;
  const C = { dimm: '#1f4fd6', guider: '#d62020', temp: '#d01c1c', dew: '#8e3cc7', hum: '#1f5fe0', wind: '#169c16', gust: '#8fd08f', press: '#111111', limit: '#e00000', now: '#00a36c' };

  function frameBg(ctx, w, h) {
    ctx.fillStyle = OA.th().bg;
    ctx.fillRect(0, 0, w, h);
  }
  function bigValues(ctx, x, y, items) {
    // items: [text, color, size]
    let cx = x;
    for (const [txt, col, size] of items) {
      cx += OA.text(ctx, txt, cx, y, { size: size || 18, bold: true, color: col, halo: OA.nightMode ? '#000' : 'rgba(255,255,255,.85)' }) + 8;
    }
  }
  function overlay(ctx, p, msg) {
    const b = p.b;
    ctx.save();
    ctx.globalAlpha = 0.85;
    OA.text(ctx, msg, b.x + b.w / 2, b.y + b.h / 2 + 12, { size: Math.min(38, b.h * 0.4), bold: true, align: 'center', color: '#ff0000', halo: OA.nightMode ? '#000' : 'rgba(255,255,255,.7)' });
    ctx.restore();
  }
  const mph = (a) => a.map((v) => (v == null ? null : v * MPH));

  const wx = OA.register({
    name: 'weather',
    lastDraw: -1,
    snap(s) {
      if (OA.tab !== 'weather') return;
      if (!OA.every('wx', 1000)) return;
      this.draw();
    },
    tab(n) { if (n === 'weather') requestAnimationFrame(() => this.draw()); },
    redraw() { if (OA.tab === 'weather') this.draw(); },
    range(s) {
      const tw = OA.twilights();
      const H = OA.hist;
      let t0 = tw ? tw.t0 - 20 : s.t - 120, t1 = tw ? tw.t1 + 20 : s.t + 60;
      if (H && H.t && H.t.length) t0 = Math.min(t0, H.t[0]);
      t1 = Math.max(t1, s.t + 5);
      return [t0, t1];
    },
    draw() {
      const s = OA.snap;
      if (!s) return;
      const H = OA.hist && OA.hist.t ? OA.hist : { t: [], dimm: [], guider_fwhm: [], humidity: [], wind: [], gust: [], temp: [], dewpoint: [], pressure: [], wind_dir: [], flux_ratio: [] };
      const [t0, t1] = this.range(s);
      const w = s.weather, lim = w.limits;
      OA.setText(OA.$('wx-site'), s.site + ' — weather');
      OA.setText(OA.$('wx-upd'), 'updated ' + s.utc.slice(11, 16) + ' UT · wind in mph, pressure in mb');
      const M = { l: 42, r: 46, t: 16, b: 16 };
      const mk = (id) => {
        const fit = OA.fit(OA.$(id));
        if (!fit) return null;
        frameBg(fit.ctx, fit.w, fit.h);
        const p = new OA.Plot(fit.ctx, { x: M.l, y: M.t, w: fit.w - M.l - M.r, h: fit.h - M.t - M.b }, [t0, t1], [0, 1]);
        return { fit, p, ctx: fit.ctx };
      };
      const lastI = (arr) => { for (let i = arr.length - 1; i >= 0; i--) if (arr[i] != null) return arr[i]; return null; };

      // ---------------- seeing
      let c = mk('wx-seeing');
      if (c) {
        const { p, ctx } = c;
        p.y0 = 0; p.y1 = 2;
        OA.shadeTwilight(p);
        p.ygrid([0, 0.5, 1, 1.5, 2], (v) => v.toFixed(1) + '″');
        OA.nightAxes(p, { twiLabels: true });
        p.clip();
        p.dots(H.t, H.dimm, C.dimm, 1.5);
        p.dots(H.t, H.guider_fwhm, C.guider, 1.7);
        p.unclip();
        p.vline(s.t, C.now, null, null);
        p.frame();
        const g = s.guider.state === 'guiding' ? s.guider.fwhm : lastI(H.guider_fwhm.slice(-5));
        bigValues(ctx, p.b.x + 6, p.b.y + 22, [
          [w.dimm != null ? OA.num(w.dimm, 2) + '″' : '—', C.dimm, 20],
          [g != null ? '(' + OA.num(g, 2) + '″ guider)' : '(no guider)', C.guider, 13],
        ]);
        OA.text(ctx, 'Seeing (″)', 4, 10, { size: 10, bold: true });
        OA.text(ctx, '● DIMM', p.b.x + p.b.w - 110, p.b.y + p.b.h - 5, { size: 10, color: C.dimm });
        OA.text(ctx, '● guider', p.b.x + p.b.w - 56, p.b.y + p.b.h - 5, { size: 10, color: C.guider });
      }

      // ---------------- temperature / dew point / humidity
      c = mk('wx-temp');
      if (c) {
        const { p, ctx } = c;
        let r = OA.minMax([H.temp, H.dewpoint, [w.temp, w.dewpoint]], 0.1) || [-10, 15];
        if (r[1] - r[0] < 10) { const m = (r[0] + r[1]) / 2; r = [m - 5, m + 5]; }
        p.y0 = Math.floor(r[0]); p.y1 = Math.ceil(r[1]);
        OA.shadeTwilight(p);
        p.ygrid(OA.niceTicks(p.y0, p.y1, 5), (v) => v + '°', 'left', C.temp);
        p.y2([0, 20, 40, 60, 80, 100], 0, 100, (v) => v + '%', C.hum);
        OA.nightAxes(p);
        const hy = (v) => p.Yr(v, 0, 100);
        p.hlinePx(hy(lim.humidity), C.limit, [6, 4], `humidity limit ${lim.humidity}%`);
        p.clip();
        p.line(H.t, H.humidity, C.hum, 1.4, null, hy);
        p.line(H.t, H.dewpoint, C.dew, 1.6);
        p.line(H.t, H.temp, C.temp, 1.8);
        p.unclip();
        p.vline(s.t, C.now);
        p.frame();
        bigValues(ctx, p.b.x + 6, p.b.y + 22, [
          [OA.num(w.temp, 1) + '°C', C.temp, 20], ['(' + OA.num(w.temp * 9 / 5 + 32, 0) + '°F)', C.temp, 12],
          [OA.num(w.dewpoint, 1) + '°C', C.dew, 16], [OA.num(w.humidity, 0) + '%', C.hum, 20],
        ]);
        OA.text(ctx, '°C', 4, 10, { size: 10, bold: true, color: C.temp });
        OA.text(ctx, '% RH', p.b.x + p.b.w + 4, 10, { size: 10, bold: true, color: C.hum });
        OA.text(ctx, 'Temperature', p.b.x + p.b.w - 190, p.b.y + p.b.h - 5, { size: 10, color: C.temp });
        OA.text(ctx, 'Dew point', p.b.x + p.b.w - 120, p.b.y + p.b.h - 5, { size: 10, color: C.dew });
        OA.text(ctx, 'Humidity', p.b.x + p.b.w - 60, p.b.y + p.b.h - 5, { size: 10, color: C.hum });
        if (w.humidity >= lim.humidity) overlay(ctx, p, 'High humidity!');
      }

      // ---------------- wind
      c = mk('wx-wind');
      if (c) {
        const { p, ctx } = c;
        const gm = mph(H.gust), wm = mph(H.wind);
        const r = OA.minMax([gm, [w.gust_mph, lim.wind_close_mph]], 0) || [0, 40];
        p.y0 = 0; p.y1 = Math.max(40, Math.ceil((r[1] + 4) / 10) * 10);
        OA.shadeTwilight(p);
        p.ygrid(OA.niceTicks(0, p.y1, 5), (v) => String(v), 'left', C.wind);
        p.y2(OA.niceTicks(0, p.y1 / MPH, 4), 0, p.y1 / MPH, (v) => v + ' m/s', '#777');
        OA.nightAxes(p);
        p.hline(lim.wind_mph, '#e07000', [6, 4], `high wind ${lim.wind_mph} mph (pointing limits)`);
        p.hline(lim.wind_close_mph, C.limit, [6, 4], `close ${lim.wind_close_mph} mph`);
        p.clip();
        p.dots(H.t, gm, C.gust, 1.4);
        p.line(H.t, wm, C.wind, 1.8);
        p.unclip();
        p.vline(s.t, C.now);
        p.frame();
        bigValues(ctx, p.b.x + 6, p.b.y + 22, [
          [OA.num(w.wind_mph, 1) + ' mph', C.wind, 20], ['(' + OA.num(w.gust_mph, 1) + ')', C.wind, 14],
          [OA.compass(w.wind_dir) + ' ' + w.wind_dir + '°', C.wind, 12],
        ]);
        OA.text(ctx, 'mph', 4, 10, { size: 10, bold: true, color: C.wind });
        if (w.wind_mph >= lim.wind_mph) overlay(ctx, p, w.wind_mph >= lim.wind_close_mph ? 'High wind! (close)' : 'High wind!');
      }

      // ---------------- pressure
      c = mk('wx-press');
      if (c) {
        const { p, ctx } = c;
        let r = OA.minMax([H.pressure, [w.pressure]], 0) || [w.pressure - 2, w.pressure + 2];
        if (r[1] - r[0] < 3) { const m = (r[0] + r[1]) / 2; r = [m - 1.5, m + 1.5]; }
        p.y0 = r[0] - 0.3; p.y1 = r[1] + 0.3;
        OA.shadeTwilight(p);
        p.ygrid(OA.niceTicks(p.y0, p.y1, 3), (v) => v.toFixed(0), 'left');
        OA.nightAxes(p);
        p.clip();
        p.line(H.t, H.pressure, C.press, 1.6);
        p.unclip();
        p.vline(s.t, C.now);
        p.frame();
        bigValues(ctx, p.b.x + 6, p.b.y + 20, [[OA.num(w.pressure, 1) + ' mb', C.press, 16]]);
        OA.text(ctx, 'mb', 4, 10, { size: 10, bold: true });
      }

      this.rose(s, H);
      this.table(s, H);
    },
    rose(s, H) {
      const fit = OA.fit(OA.$('wx-rose'));
      if (!fit) return;
      const { ctx, w, h } = fit;
      const th = OA.th();
      frameBg(ctx, w, h);
      const cx = w / 2, cy = h / 2 + 6, R = Math.min(w, h) / 2 - 18;
      const lim = s.weather.limits;
      const recent = [];
      for (let i = H.t.length - 1; i >= 0 && H.t[i] > s.t - 120; i--) recent.push([H.t[i], H.wind[i] * MPH, H.wind_dir[i]]);
      let vmax = Math.max(40, lim.wind_close_mph + 5, ...recent.map((r) => r[1] + 3), s.weather.wind_mph + 3);
      vmax = Math.ceil(vmax / 10) * 10;
      const rr = (v) => (v / vmax) * R;
      ctx.lineWidth = 1;
      for (let v = 10; v <= vmax; v += 10) {
        ctx.strokeStyle = th.grid; ctx.beginPath(); ctx.arc(cx, cy, rr(v), 0, 2 * Math.PI); ctx.stroke();
        OA.text(ctx, String(v), cx + 2, cy - rr(v) + 9, { size: 8, color: '#888' });
      }
      for (let a = 0; a < 360; a += 45) {
        const r = a * Math.PI / 180;
        ctx.strokeStyle = th.grid; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + R * Math.sin(r), cy - R * Math.cos(r)); ctx.stroke();
      }
      for (const [lab, a] of [['N', 0], ['E', 90], ['S', 180], ['W', 270]]) {
        const r = a * Math.PI / 180;
        OA.text(ctx, lab, cx + (R + 9) * Math.sin(r), cy - (R + 9) * Math.cos(r) + 4, { size: 10, bold: true, align: 'center' });
      }
      ctx.setLineDash([5, 3]);
      ctx.strokeStyle = OA.col('#e07000'); ctx.beginPath(); ctx.arc(cx, cy, rr(lim.wind_mph), 0, 2 * Math.PI); ctx.stroke();
      ctx.strokeStyle = OA.col(C.limit); ctx.beginPath(); ctx.arc(cx, cy, rr(lim.wind_close_mph), 0, 2 * Math.PI); ctx.stroke();
      ctx.setLineDash([]);
      for (const [t, v, d] of recent) {
        const age = OA.clamp((s.t - t) / 120, 0, 1);
        const r = d * Math.PI / 180;
        ctx.fillStyle = OA.col(`rgba(22,156,22,${(1 - 0.8 * age).toFixed(2)})`);
        ctx.beginPath(); ctx.arc(cx + rr(v) * Math.sin(r), cy - rr(v) * Math.cos(r), 2.3, 0, 2 * Math.PI); ctx.fill();
      }
      const r = s.weather.wind_dir * Math.PI / 180, v = s.weather.wind_mph;
      ctx.strokeStyle = OA.col(C.wind); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + rr(v) * Math.sin(r), cy - rr(v) * Math.cos(r)); ctx.stroke();
      ctx.fillStyle = OA.col(C.wind);
      ctx.beginPath(); ctx.arc(cx + rr(v) * Math.sin(r), cy - rr(v) * Math.cos(r), 4.5, 0, 2 * Math.PI); ctx.fill();
      OA.text(ctx, 'Wind (from), last 2 h', 4, 11, { size: 10, bold: true });
    },
    table(s, H) {
      const w = s.weather, lim = w.limits, g = s.guider;
      const row = (k, v, over) => `<tr${over ? ' class="over"' : ''}><td>${k}</td><td class="v">${v}</td></tr>`;
      const guiding = g.state === 'guiding';
      OA.$('wx-now').innerHTML =
        row('DIMM seeing', w.dimm != null ? OA.num(w.dimm, 2) + '″' : '—') +
        row('Guider FWHM', guiding && g.fwhm != null ? OA.num(g.fwhm, 2) + '″' : '—') +
        row('Transparency', guiding && g.flux_ratio != null ? OA.num(g.flux_ratio, 2) + ' (flux)' : '—') +
        row('Temperature', OA.num(w.temp, 1) + ' °C') +
        row('Dew point', OA.num(w.dewpoint, 1) + ' °C', w.temp - w.dewpoint < 2) +
        row('T − T<sub>dew</sub>', OA.num(w.temp - w.dewpoint, 1) + ' °C', w.temp - w.dewpoint < 2) +
        row('Humidity', OA.num(w.humidity, 0) + ' %', w.humidity >= lim.humidity) +
        row('Wind', OA.num(w.wind_mph, 1) + ' mph ' + OA.compass(w.wind_dir), w.wind_mph >= lim.wind_mph) +
        row('Gust', OA.num(w.gust_mph, 1) + ' mph', w.gust_mph >= lim.wind_close_mph) +
        row('Direction', w.wind_dir + '°') +
        row('Pressure', OA.num(w.pressure, 1) + ' mb') +
        row('Limits', `RH ${lim.humidity}% · wind ${lim.wind_mph} / close ${lim.wind_close_mph} mph`) +
        row('Dome', s.dome.open ? 'OPEN' : 'CLOSED — ' + OA.esc(s.dome.reason), !s.dome.open);
    },
  });
})();
