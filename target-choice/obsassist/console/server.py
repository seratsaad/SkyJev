"""The console web server: the simulated observatory plus its operator UI.

    python -m obsassist console --program clay_mike_darktime --port 8765

REST + WebSocket API (also what the assistant app and any other client use):

    GET  /api/state              full snapshot (telemetry, TCS, arms, targets, plan, log)
    GET  /api/history            weather/guider history since sunset (weather page)
    GET  /api/night              airmass/altitude curves of all targets (planning chart, sky map)
    GET  /api/program            the target list as loaded
    GET  /api/programs           built-in programs
    GET  /api/frames             frames written tonight, with quick-look summaries
    GET  /api/frame/{name}       raw pixels (binary uint16, header X-Shape: ny,nx), ?bin=2
    GET  /api/frame/{name}/ql    quick-look (extracted spectrum, S/N, flags)
    GET  /api/frame/{name}/header
    GET  /api/etc?target=&seeing=&cloud=&t_exp=
    POST /api/cmd   {"cmd": "...", "source": "observer"|"assistant"}
    POST /api/sim   {"speed": 60, "paused": false, "assistant_control": false, "assistant_url": "..."}
    GET  /api/assistant          the URL of the assistant watching this console (its Assistant tab)
    POST /api/reset {"program": "...", "seed": 0, "speed": 60}
    GET  /api/debrief            hindsight oracle vs what was done (the truth; for after the night)
    WS   /ws                     snapshot every ~0.5 s real time
"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from obsassist.console.engine import Observatory
from obsassist.targets import builtin_programs, load_program

STATIC = Path(__file__).resolve().parent / "static"


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def dumps(x) -> str:
    return json.dumps(_clean(x), default=_json_default)


def _clean(x):
    """Replace NaN/inf (not valid JSON) with None, recursively."""
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating,)):
        return _clean(float(x))
    return x


def jresp(x) -> Response:
    return Response(content=dumps(x), media_type="application/json")


def create_app(
    program: str = "clay_mike_darktime",
    seed: int = 0,
    speed: float = 60.0,
    data_root: Optional[str] = None,
    assistant_control: bool = False,
) -> FastAPI:
    app = FastAPI(title="obsassist console")
    holder = {"obs": None}

    @app.middleware("http")
    async def revalidate(request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("Cache-Control", "no-cache")  # a browser must not keep an old UI after an upgrade
        return resp

    pool = cf.ThreadPoolExecutor(max_workers=2)

    def resolve_program(name: str):
        progs = builtin_programs()
        if name in progs:
            return load_program(progs[name])
        if os.path.exists(name):
            return load_program(name)
        raise HTTPException(404, f"unknown program {name}; built-in: {sorted(progs)}")

    def new_obs(name: str, sd: int, sp: float, ctl: bool):
        holder["obs"] = Observatory(
            resolve_program(name), seed=sd, speed=sp, data_root=data_root, allow_assistant_control=ctl
        )
        holder["program_key"] = name
        holder["seed"] = sd

    new_obs(program, seed, speed, assistant_control)

    async def ticker():
        while True:
            try:
                holder["obs"].tick()
            except Exception as e:  # keep the clock alive; report
                holder["obs"].say("SYS", f"engine error: {type(e).__name__}: {e}", level="error")
            await asyncio.sleep(0.2)

    @app.on_event("startup")
    async def _start():
        asyncio.create_task(ticker())

    # ---------------------------------------------------------------- pages
    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    # ---------------------------------------------------------------- API
    @app.get("/api/state")
    async def state(full: bool = False):
        return jresp(holder["obs"].snapshot(full=full))

    @app.get("/api/history")
    async def history():
        return jresp(holder["obs"].history)

    @app.get("/api/night")
    async def night(step: int = 5):
        return jresp(holder["obs"].night_curves(step))

    @app.get("/api/program")
    async def program_():
        o = holder["obs"]
        d = o.program.to_dict()
        d["key"] = holder.get("program_key")
        d["seed"] = holder.get("seed")
        return jresp(d)

    @app.get("/api/programs")
    async def programs():
        return jresp(sorted(builtin_programs()))

    @app.get("/api/frames")
    async def frames():
        out = []
        for r in holder["obs"].frames:
            d = {k: v for k, v in r.items() if k != "ql"}
            ql = r.get("ql") or {}
            d["ql_summary"] = {
                k: ql.get(k)
                for k in (
                    "trace_found",
                    "fwhm_arcsec",
                    "peak_adu",
                    "saturated_frac",
                    "snr_ref_per_A",
                    "snr_ref_per_pix",
                    "flags",
                    "sky_adu_per_pix",
                )
            }
            out.append(d)
        return jresp(out)

    def _frame_rec(name: str) -> dict:
        for r in holder["obs"].frames:
            if r["file"] == name:
                return r
        raise HTTPException(404, f"no frame {name}")

    @app.get("/api/frame/{name}")
    async def frame(name: str, bin: int = 2):
        r = _frame_rec(name)
        if "path" not in r:
            raise HTTPException(409, "frame not written yet (or frames disabled)")
        from obsassist.sim.frames import read_fits

        data, _ = read_fits(r["path"])
        b = max(1, int(bin))
        if b > 1:
            ny, nx = (data.shape[0] // b) * b, (data.shape[1] // b) * b
            data = data[:ny, :nx].reshape(ny // b, b, nx // b, b).mean(axis=(1, 3))
        arr = np.clip(data, 0, 65535).astype("<u2")
        return Response(
            content=arr.tobytes(),
            media_type="application/octet-stream",
            headers={
                "X-Shape": f"{arr.shape[0]},{arr.shape[1]}",
                "X-Bin": str(b),
                "Access-Control-Expose-Headers": "X-Shape, X-Bin",
            },
        )

    @app.get("/api/frame/{name}/ql")
    async def frame_ql(name: str):
        r = _frame_rec(name)
        return jresp(r.get("ql") or {"error": r.get("ql_error", "no quick-look yet")})

    @app.get("/api/frame/{name}/header")
    async def frame_header(name: str):
        r = _frame_rec(name)
        if "path" not in r:
            raise HTTPException(409, "frame not written yet")
        from obsassist.sim.frames import read_fits

        _, hdr = read_fits(r["path"])
        return jresp(dict(hdr))

    @app.get("/api/etc")
    async def etc(
        target: str, seeing: Optional[float] = None, cloud: Optional[float] = None, t_exp: Optional[float] = None
    ):
        o = holder["obs"]
        try:
            i = o._find_target(target)
        except KeyError as e:
            raise HTTPException(404, str(e))
        return jresp(o.etc(i, seeing, cloud, t_exp))

    @app.post("/api/cmd")
    async def cmd(req: Request):
        body = await req.json()
        reply = holder["obs"].command(str(body.get("cmd", "")), source=str(body.get("source", "observer")))
        return jresp({"reply": reply})

    @app.post("/api/sim")
    async def sim(req: Request):
        body = await req.json()
        o = holder["obs"]
        if "speed" in body:
            o.speed = float(body["speed"])
        if "paused" in body:
            o.paused = bool(body["paused"])
            import time as _t

            o._last_real = _t.time()
        if "assistant_control" in body:
            o.allow_assistant_control = bool(body["assistant_control"])
            o.say("SYS", "assistant control " + ("ENABLED" if o.allow_assistant_control else "disabled"), level="warn")
        if body.get("assistant_url"):
            holder["assistant_url"] = str(body["assistant_url"])
            o.say("SYS", f"assistant connected: {holder['assistant_url']}")
        return jresp({"speed": o.speed, "paused": o.paused, "assistant_control": o.allow_assistant_control})

    @app.get("/api/assistant")
    async def assistant():
        return jresp({"url": holder.get("assistant_url")})

    @app.post("/api/reset")
    async def reset(req: Request):
        body = await req.json()
        o = holder["obs"]
        new_obs(
            body.get("program", holder.get("program_key")),
            int(body.get("seed", holder.get("seed", 0))),
            float(body.get("speed", o.speed)),
            bool(body.get("assistant_control", o.allow_assistant_control)),
        )
        return jresp({"ok": True})

    @app.get("/api/debrief")
    async def debrief(width: int = 1):
        """The truth: the hindsight-optimal night vs the night so far. Not for use during the night."""
        o = holder["obs"]
        from obsassist.planning.oracle import pilot
        from obsassist.planning.planner import GreedyPolicy, run_policy

        def work():
            m = o.model
            best = pilot(m, width=width)
            g, _ = run_policy(m, GreedyPolicy())
            fmt = lambda x: m.ephem.utc(x).strftime("%H:%M")
            seq = [
                {
                    "ut": fmt(a["t"]),
                    "kind": a["kind"],
                    "target": m.targets[a["target"]].name if a.get("target", -1) >= 0 else "",
                    "t_exp": a.get("t_exp"),
                    "snr": a.get("snr"),
                }
                for a in best.actions
            ]
            return {
                "oracle": best.summary(m),
                "sequence": seq,
                "greedy_score": m.score(g),
                "you": {
                    "score": m.score(o.state),
                    "done": int(m.done(o.state).sum()),
                    "at_ut": o.utc().strftime("%H:%M"),
                },
                "weather": {"regime": m.weather.regime, "events": m.weather.events},
            }

        res = await asyncio.get_event_loop().run_in_executor(pool, work)
        return jresp(res)

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            n = 0
            while True:
                snap = holder["obs"].snapshot(full=False)
                if n % 20 == 0:
                    snap["history"] = holder["obs"].history
                await sock.send_text(dumps(snap))
                n += 1
                await asyncio.sleep(0.5)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


def main(argv=None):
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="obsassist console (simulated observatory)")
    ap.add_argument("--program", default="clay_mike_darktime")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=60.0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--data", default=None, help="root directory for FITS files (default <project>/data/frames)")
    ap.add_argument("--assistant-control", action="store_true", help="let the assistant send commands")
    a = ap.parse_args(argv)
    app = create_app(a.program, a.seed, a.speed, a.data, a.assistant_control)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
