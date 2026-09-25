"""The assistant app: a separate process and UI that watches a console and advises.

    python -m obsassist assistant --console http://127.0.0.1:8765 --port 8766 \
        --model Qwen/Qwen3-1.7B --heads heads/qwen3-1.7b.json
    python -m obsassist assistant --manual obsassist/programs/clay_mike_darktime.yaml --data /path/to/ut20261010

API:
    GET  /api/status               everything the UI shows
    GET  /api/trace?after=ID       every exchange in and out as JSON (also data/traces/*.jsonl)
    GET  /trace                    the trace as a page (the console embeds it in its Assistant tab)
    POST /api/settings  {mode, s1_override_tau, escalate, s2_effort}
    POST /api/accept    {execute: bool}   accept the open recommendation
    POST /api/dismiss
    POST /api/deliberate           ask System 2 now
    POST /api/chat      {text}     talk to System 2
    POST /api/nightlog             System 2 writes a night-log entry from the console events
    POST /api/report/*             manual adapter: conditions, on_target, started, done
    GET  /api/catalog?format=magellan|keck
    WS   /ws
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from obsassist.assistant.policy import Assistant
from obsassist.assistant.trace import Trace, jsonable
from obsassist.paths import TRACES

STATIC = Path(__file__).resolve().parent / "static"


def _dumps(x) -> str:
    return json.dumps(jsonable(x))


def build_assistant(
    console: Optional[str],
    manual: Optional[str],
    data: Optional[str],
    no_s1: bool,
    model: str,
    heads: Optional[str],
    chat_model: str,
    deliberate_model: str,
    no_s2: bool,
    keck_commands: Optional[str] = None,
    trace: Optional[Trace] = None,
) -> Assistant:
    if manual:
        from obsassist.assistant.adapters.manual import ManualAdapter

        adapter = ManualAdapter(manual, data_dir=data, keck_instrument=keck_commands)
    else:
        from obsassist.assistant.adapters.sim import SimConsoleAdapter

        adapter = SimConsoleAdapter(console or "http://127.0.0.1:8765")
    s1 = None
    if not no_s1:
        from obsassist.assistant.system1 import System1

        s1 = System1(model=model, heads=heads)
    s2 = None
    if not no_s2:
        from obsassist.assistant.system2 import System2

        a_ref = {}

        def tools():
            a = a_ref["a"]
            return {
                "get_state": lambda: {
                    k: v
                    for k, v in a.status().items()
                    if k
                    in (
                        "utc",
                        "twilight",
                        "minutes_left",
                        "weather",
                        "dome",
                        "tcs",
                        "guider",
                        "arms",
                        "targets",
                        "score",
                        "max_score",
                    )
                },
                "get_candidates": lambda: {
                    "candidates": [c for c in a.obs.get("candidates", []) if c.get("feasible")],
                    "system1": a.judgements,
                },
                "etc_estimate": lambda target, seeing=None, cloud=None, t_exp=None: a.adapter.etc(
                    target, seeing, cloud, t_exp
                ),
                "get_projection": lambda: (a.snap.get("plan") or {}).get("projection", {}),
                "get_frame_quicklook": lambda name=None: {
                    k: v for k, v in a.adapter.frame_ql(name).items() if k not in ("wave", "flux", "sky", "snr_curve")
                },
            }

        s2 = System2(tools={}, chat_model=chat_model, deliberate_model=deliberate_model)
        asst = Assistant(adapter, s1, s2, trace=trace)
        a_ref["a"] = asst
        s2.tools = tools()
        return asst
    return Assistant(adapter, s1, None, trace=trace)


def create_app(asst: Assistant, public_url: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="obsassist assistant")
    stop = threading.Event()

    @app.middleware("http")
    async def revalidate(request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("Cache-Control", "no-cache")  # a browser must not keep an old UI after an upgrade
        return resp

    def loop():
        while not stop.is_set():
            t0 = time.time()
            asst.step()
            stop.wait(max(0.5, asst.settings.step_s - (time.time() - t0)))

    @app.on_event("startup")
    async def _start():
        if public_url and hasattr(asst.adapter, "announce"):
            asst.adapter.announce(public_url)  # the console shows this assistant in its Assistant tab
        threading.Thread(target=loop, daemon=True).start()

    @app.on_event("shutdown")
    async def _stop():
        stop.set()

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/trace")
    async def trace_page():
        return FileResponse(STATIC / "trace.html")

    @app.get("/api/trace")
    async def trace(after: int = 0, limit: int = 400):
        return Response(
            _dumps({"path": str(asst.trace.path or ""), "items": asst.trace.since(after, limit)}),
            media_type="application/json",
        )

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/api/status")
    async def status():
        return Response(_dumps(asst.status()), media_type="application/json")

    @app.post("/api/settings")
    async def settings(req: Request):
        body = await req.json()
        for k, v in body.items():
            if hasattr(asst.settings, k):
                setattr(asst.settings, k, type(getattr(asst.settings, k))(v))
        return Response(_dumps(asst.settings.__dict__), media_type="application/json")

    @app.post("/api/accept")
    async def accept(req: Request):
        body = await req.json() if (await req.body()) else {}
        return {"reply": asst.accept(execute=bool(body.get("execute", True)))}

    @app.post("/api/dismiss")
    async def dismiss():
        return {"reply": asst.dismiss()}

    @app.post("/api/deliberate")
    async def deliberate():
        if asst.s2 is None or not asst.s2.enabled:
            raise HTTPException(409, "System 2 is not available")
        if asst.rec is None:
            raise HTTPException(409, "no decision yet")
        threading.Thread(target=asst._escalate, args=(asst.rec,), daemon=True).start()
        return {"reply": "System 2 is thinking"}

    @app.post("/api/chat")
    async def chat(req: Request):
        body = await req.json()
        if asst.s2 is None:
            return {"reply": "System 2 is off."}
        reply = await asyncio.get_event_loop().run_in_executor(None, asst.s2.chat, str(body.get("text", "")))
        return {"reply": reply}

    @app.post("/api/nightlog")
    async def nightlog():
        """A night-log entry written by System 2 from the console's recent events."""
        if asst.s2 is None or not asst.s2.enabled:
            raise HTTPException(409, "System 2 is not available")
        lines = [f"{e.get('utc', '')} {e.get('who', '')}: {e.get('text', '')}" for e in (asst.snap.get("log") or [])]
        text = await asyncio.get_event_loop().run_in_executor(None, asst.s2.narrate, lines)
        return {"reply": text}

    @app.post("/api/report/{kind}")
    async def report(kind: str, req: Request):
        body = await req.json()
        ad = asst.adapter
        if not hasattr(ad, "report_conditions"):
            raise HTTPException(409, "reports are for the manual adapter")
        if kind == "conditions":
            ad.report_conditions(**{k: body.get(k) for k in ("seeing", "cloud", "humidity", "wind_mph", "temp_c")})
        elif kind == "on_target":
            ad.report_on_target(body["name"])
        elif kind == "started":
            ad.report_exposure_started(body["name"], float(body["t_exp"]), int(body.get("n", 1)))
        elif kind == "done":
            ad.report_exposure_done(
                body.get("name"), body.get("t_exp"), body.get("snr"), bool(body.get("aborted", False))
            )
        else:
            raise HTTPException(404, kind)
        return {"ok": True}

    @app.get("/api/catalog")
    async def catalog(format: str = "magellan"):
        from obsassist.planning.catalogs import keck_starlist, magellan_catalog
        from obsassist.targets import load_program

        ad = asst.adapter
        prog = getattr(ad, "program", None)
        if callable(prog):  # sim adapter: fetch the program from the console
            try:
                prog = load_program(prog())
            except Exception as e:
                raise HTTPException(409, f"no program: {e}")
        text = magellan_catalog(prog) if format == "magellan" else keck_starlist(prog)
        return PlainTextResponse(text)

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                await sock.send_text(_dumps(asst.status()))
                await asyncio.sleep(1.0)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


def main(argv=None):
    import uvicorn

    ap = argparse.ArgumentParser(description="obsassist assistant")
    ap.add_argument("--console", default="http://127.0.0.1:8765", help="simulated console URL")
    ap.add_argument("--manual", default=None, help="program YAML: real-telescope mode (advise only)")
    ap.add_argument(
        "--keck-commands",
        default=None,
        choices=["LRIS", "DEIMOS"],
        help="manual mode at Keck: print KTL command lines and read DCS telemetry (never executes)",
    )
    ap.add_argument("--data", default=None, help="night data directory to watch for FITS files (manual mode)")
    ap.add_argument("--no-s1", action="store_true", help="no local model: planner and System 2 only")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--heads", default=None, help="AnyJev artifacts JSON (L2 heads)")
    ap.add_argument("--chat-model", default="gpt-5.6-luna")
    ap.add_argument("--deliberate-model", default="gpt-5.6-sol")
    ap.add_argument("--no-s2", action="store_true")
    ap.add_argument("--mode", default="advise", choices=["advise", "autopilot"])
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    import datetime as _dt

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace = Trace(TRACES / f"assistant_{stamp}.jsonl")
    asst = build_assistant(
        a.console,
        a.manual,
        a.data,
        a.no_s1,
        a.model,
        a.heads,
        a.chat_model,
        a.deliberate_model,
        a.no_s2,
        a.keck_commands,
        trace,
    )
    asst.settings.mode = a.mode
    uvicorn.run(create_app(asst, f"http://{a.host}:{a.port}"), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
