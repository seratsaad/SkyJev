#!/usr/bin/env python
"""Record the SkyJev demo video headlessly (no X server, no desktop).

It starts the assistant (System 1 on the local GPU) and the simulated Magellan Clay/MIKE console,
drives both UIs in headless Chromium through Playwright, takes JPEG screenshots while the pages
run, and encodes them with the ffmpeg from imageio-ffmpeg to demo/skyjev_demo.mp4 (H.264,
1920x1080). Waiting for the language models is cut from the video; the captions give the times.

GPT use is capped: escalation to System 2 is switched off before the console starts, so System 2
runs exactly once for "Ask System 2" and once for one chat question. `--no-s2` makes no OpenAI
calls at all. `--max-usd` kills the assistant if the System 2 spend, priced at gpt-5.6-sol rates,
passes it. `--browser-task ho_ned_m51_202` adds one live run of the browser agent in ../browser
(gpt-4.1-mini, about US$0.01-0.03, its own US$0.10 ceiling); without it the video shows a card
that points to browser/README.md.

On OSC Pitzer (one V100; float16, because the V100 has no bfloat16):

    cd /fs/scratch/PAS2823/saadsm/SkyJev
    sbatch -A PAS2823 -p gpudebug --gpus-per-node=1 -c 4 -t 0:50:00 -o demo/record_%j.log \
        --wrap "source ./pitzer_env.sh && python demo/record_demo.py --browser-task ho_ned_m51_202"

Needs `playwright` and `imageio-ffmpeg` in the venv and `playwright install chromium` (set
PLAYWRIGHT_BROWSERS_PATH if the browsers are not in the default cache).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
TC = ROOT / "target-choice"
BROWSER = ROOT / "browser"
DEMO = ROOT / "demo"
CONSOLE, ASSIST = "http://127.0.0.1:8765", "http://127.0.0.1:8766"
QUESTION = "What if the seeing goes to 1.2 arcsec?"
# US$ per million tokens (input, output) for the most expensive model used (gpt-5.6-sol, Sept 2026);
# every System 2 token is priced at this rate, so the estimate is an upper bound
PRICE_IN, PRICE_OUT = 5.00, 30.00


def s2_cost(usage: dict) -> float:
    return (usage.get("input_tokens", 0) * PRICE_IN + usage.get("output_tokens", 0) * PRICE_OUT) / 1e6
OUT_W, OUT_H = 1920, 1080

CARD = """<html><head><style>
body{margin:0;height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;
background:#0b1020;color:#e8ecf4;font-family:'DejaVu Sans','Liberation Sans',sans-serif;text-align:center}
.t{font-size:44px;font-weight:700;max-width:82%;line-height:1.3}
.s{margin-top:28px;font-size:23px;color:#9fb0cc;max-width:74%;line-height:1.45}
.k{margin-top:40px;font-size:15px;color:#5d6d8a;letter-spacing:.12em;text-transform:uppercase}
</style></head><body><div class="t">{title}</div><div class="s">{sub}</div><div class="k">{kicker}</div></body></html>"""

CAPTION_JS = """t => {
  let d = document.getElementById('demo-cap');
  if (!d) {
    d = document.createElement('div'); d.id = 'demo-cap';
    Object.assign(d.style, {position: 'fixed', left: '50%', bottom: '26px', transform: 'translateX(-50%)',
      zIndex: 2147483647, background: 'rgba(8,12,22,0.90)', color: '#fff', padding: '10px 22px',
      font: "600 21px/1.35 'DejaVu Sans', sans-serif", borderRadius: '10px', maxWidth: '82%',
      textAlign: 'center', pointerEvents: 'none', boxShadow: '0 4px 18px rgba(0,0,0,.45)'});
    document.body.appendChild(d);
  }
  d.textContent = t; d.style.display = t ? 'block' : 'none';
}"""


class Recorder:
    """Screenshots at ~fps while a page runs; each frame keeps its real duration."""

    def __init__(self, page, frames: Path, fps: float):
        self.page, self.dir, self.fps = page, frames, fps
        self.items: list[tuple[str, float]] = []
        self.n = 0
        self.shoot = None  # optional callable(path) used instead of page.screenshot
        frames.mkdir(parents=True, exist_ok=True)
        for f in frames.glob("f*.jpg"):
            f.unlink()

    def _shot(self) -> str:
        self.n += 1
        name = f"f{self.n:05d}.jpg"
        if self.shoot:
            self.shoot(self.dir / name)
        else:
            self.page.screenshot(path=str(self.dir / name), type="jpeg", quality=92)
        return name

    def hold(self, secs: float):
        end = time.monotonic() + secs
        while True:
            t0 = time.monotonic()
            name = self._shot()
            if t0 + 1 / self.fps >= end:
                self.items.append((name, max(end - t0, 1 / self.fps)))
                return
            time.sleep(max(0.0, t0 + 1 / self.fps - time.monotonic()))
            self.items.append((name, time.monotonic() - t0))

    def still(self, secs: float):
        self.items.append((self._shot(), secs))

    @property
    def length(self) -> float:
        return sum(d for _, d in self.items)

    def encode(self, out: Path, ffmpeg: str):
        lst = self.dir / "frames.txt"
        lines = [f"file '{n}'\nduration {d:.3f}" for n, d in self.items] + [f"file '{self.items[-1][0]}'"]
        lst.write_text("\n".join(lines) + "\n")
        vf = (
            f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2:color=0x0b1020,fps=25,format=yuv420p"
        )
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf", vf]
        cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)


class CdpShooter:
    """Screenshots of the newest http(s) page in a Chromium over raw CDP: a second, read-only client
    next to the agent's own, so that the agent's page is never touched."""

    def __init__(self, port: int):
        self.port, self.target, self.ws, self.i = port, None, None, 0

    def __call__(self, path: Path):
        import base64

        from websockets.sync.client import connect

        pages = [
            t for t in httpx.get(f"http://127.0.0.1:{self.port}/json/list", timeout=5).json()
            if t.get("type") == "page" and t.get("url", "").startswith("http")
        ]
        if not pages:
            raise LookupError("no page yet")
        t = pages[0]
        if t["id"] != self.target:
            self.close()
            self.ws, self.target = connect(t["webSocketDebuggerUrl"], max_size=None, open_timeout=5), t["id"]
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": "Page.captureScreenshot",
                                 "params": {"format": "jpeg", "quality": 90}}))
        while True:
            msg = json.loads(self.ws.recv(timeout=10))
            if msg.get("id") == self.i:
                break
        if "result" not in msg:
            raise LookupError(str(msg.get("error")))
        path.write_bytes(base64.b64decode(msg["result"]["data"]))

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
        self.ws = self.target = None


def wait_http(url: str, timeout: float, proc: subprocess.Popen) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"{url}: the server exited with code {proc.returncode}")
        try:
            if httpx.get(url, timeout=3).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError(url)


def browser_agent_segment(pw, rec: Recorder, card, task_id: str, work: Path, note, max_usd: float = 0.10) -> dict:
    """One held-out task for the browser agent in ../browser (arm gpt-jev-plus: JEV's menu of page
    controls with gpt-4.1-mini as the chooser; needs ../jev-ultrafast at commit 1231850 and
    `uv sync` in browser/). The agent drives its own headless Chromium over CDP; this function
    screenshots that tab without touching its DOM, and ends with the runner's verified result."""
    import yaml

    from obsassist.assistant.system2 import api_key

    tasks = yaml.safe_load((BROWSER / "tasks_heldout" / "heldout.yaml").read_text())
    tasks = tasks.get("tasks", tasks) if isinstance(tasks, dict) else tasks
    task = next(t for t in tasks if t["id"] == task_id)
    goal = " ".join(task["goal"].split())
    card(
        "A fast browser agent on astronomy websites",
        f"Held-out task {task_id}: “{goal}”",
        "JEV's menu of page controls · chooser gpt-4.1-mini · live run, real time",
        secs=6.0,
    )
    port = 9222
    shutil.rmtree(work / "cdp-profile", ignore_errors=True)  # a fresh profile: no cookies from earlier runs
    chrome = subprocess.Popen(
        [pw.chromium.executable_path, "--headless=new", f"--remote-debugging-port={port}",
         f"--user-data-dir={work / 'cdp-profile'}", "--window-size=1440,810", "--force-device-scale-factor=1.3333",
         "--no-first-run", "--no-default-browser-check", "about:blank"],
        stdout=open(work / "chrome.log", "w"), stderr=subprocess.STDOUT,
    )
    result = {}
    try:
        wait_http(f"http://127.0.0.1:{port}/json/version", 60, chrome)
        key = api_key()  # passed to the agent's environment only
        env = dict(os.environ, OPENAI_API_KEY=key, TEXT_MODEL_API_KEY=key, TEXT_MODEL="gpt-4.1-mini",
                   TEXT_MODEL_BASE_URL="https://api.openai.com/v1", TEXT_MODEL_REASONING="none",
                   GPT_CHOOSER_MODEL="gpt-4.1-mini", BU_CDP_URL=f"http://127.0.0.1:{port}", BU_NAME="astro")
        env.pop("VIRTUAL_ENV", None)
        cmd = [str(BROWSER / ".venv" / "bin" / "python"), "-m", "astroweb.runner", "run", "--agent", "gpt-jev-plus",
               "--set", "heldout", "--task", task_id, "--repeat", "1", "--pause", "0", "--label", "demo",
               "--max-usd", str(max_usd)]
        agent_log = work / "browser_agent.log"
        agent = subprocess.Popen(cmd, cwd=BROWSER, env=env, stdout=open(agent_log, "w"), stderr=subprocess.STDOUT)
        shooter = CdpShooter(port)
        rec.shoot = shooter
        t0, errors = time.time(), 0
        while agent.poll() is None and time.time() - t0 < 300:
            try:
                rec.hold(0.4)
            except Exception as e:  # no page yet, or the agent closed or replaced its tab mid-screenshot
                errors += 1
                if errors <= 3:
                    note("browser capture", error=f"{type(e).__name__}: {e}"[:200])
                shooter.close()
                time.sleep(0.2)
        rec.shoot = None
        shooter.close()
        text = agent_log.read_text()
        m = re.search(r"success=(\w+) status=(\w+) ([\d.]+)s steps=(\d+) calls=(\d+)", text)
        runs = sorted((BROWSER / "results" / "gpt_jev_plus_heldout__demo").glob(f"{task_id}__*.json"))
        if runs:
            r = json.loads(runs[-1].read_text())
            result = {k: r.get(k) for k in ("success", "status", "duration_seconds", "steps", "model_calls",
                                             "chooser_tokens", "text_tokens", "chooser_latency_ms_median", "final_url")}
        note("browser agent", parsed=m.groups() if m else None, result=result, tail=text[-400:])
    finally:
        rec.shoot = None
        chrome.terminate()
    if result:
        ok = "passed the verifier" if result.get("success") else "did not pass the verifier"
        tokens = (result.get("chooser_tokens") or 0) + (result.get("text_tokens") or 0)
        card(
            f"The run {ok}",
            f"{result.get('duration_seconds')} s from the first model call to the stop, {result.get('steps')} steps, "
            f"{result.get('model_calls')} model calls, {tokens:,} tokens. The page, not the agent, decides success.",
            "more tasks and all runs: browser/README.md",
            secs=6.0,
        )
    return result


def get(url: str) -> dict:
    return httpx.get(url, timeout=10).json()


def post(url: str, body: dict) -> dict:
    return httpx.post(url, json=body, timeout=10).json()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-s2", action="store_true", help="no OpenAI calls: planner and System 1 only")
    ap.add_argument("--seed", type=int, default=1, help="weather seed (1: a seeing jump ~165 min after sunset)")
    ap.add_argument("--speed", type=float, default=120)
    ap.add_argument("--css", default="1440x810", help="page size in CSS pixels; frames are scaled to 1920x1080")
    ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--jump-to-min", type=float, default=178, help="use ⏭ until this many minutes after sunset")
    ap.add_argument("--out", default=str(DEMO / "skyjev_demo.mp4"))
    ap.add_argument("--max-usd", type=float, default=0.60, help="stop the assistant if System 2 spend passes this")
    ap.add_argument("--browser-task", default=None, help="run the browser agent live on this held-out task")
    a = ap.parse_args(argv)

    import imageio_ffmpeg
    from playwright.sync_api import sync_playwright

    work = DEMO / "work"
    work.mkdir(parents=True, exist_ok=True)
    log: dict = {"args": vars(a), "events": []}

    def note(what: str, **kw):
        kw.update(what=what, t=round(time.time() - t_start, 1))
        log["events"].append(kw)
        print(json.dumps(kw, default=str), flush=True)

    t_start = time.time()
    try:
        r = httpx.get("https://api.openai.com/v1/models", timeout=10)
        note("network", openai_status=r.status_code)
    except httpx.HTTPError as e:
        note("network", error=type(e).__name__)

    env = dict(os.environ)
    py = sys.executable
    asst_cmd = [py, "-m", "obsassist", "assistant", "--heads", "heads/qwen3-1.7b.json", "--mode", "autopilot"]
    asst_cmd += ["--dtype", "float16"] + (["--no-s2"] if a.no_s2 else [])
    cons_cmd = [py, "-m", "obsassist", "console", "--program", "clay_mike_darktime", "--seed", str(a.seed)]
    cons_cmd += ["--speed", str(a.speed), "--assistant-control"]
    procs = []
    try:
        # the assistant first, so that System 2 escalation is off before any decision is made
        asst = subprocess.Popen(asst_cmd, cwd=TC, env=env, stdout=open(work / "assistant.log", "w"), stderr=subprocess.STDOUT)
        procs.append(asst)
        wait_http(f"{ASSIST}/api/status", 900, asst)
        settings = post(f"{ASSIST}/api/settings", {"escalate": False, "mode": "autopilot"})
        st = get(f"{ASSIST}/api/status")
        note("assistant up", settings=settings, s1=st.get("s1"), s2=st.get("s2"))
        if settings.get("escalate") is not False:
            raise RuntimeError("could not switch System 2 escalation off")

        def guard():  # a hard spending cap: kill the assistant if System 2 spend passes --max-usd
            while asst.poll() is None:
                try:
                    usage = (get(f"{ASSIST}/api/status").get("s2") or {}).get("usage") or {}
                except (httpx.HTTPError, ValueError):
                    usage = {}
                if s2_cost(usage) > a.max_usd:
                    note("spend cap reached: stopping the assistant", usage=usage)
                    asst.kill()
                    return
                time.sleep(1)

        threading.Thread(target=guard, daemon=True).start()

        cons = subprocess.Popen(cons_cmd, cwd=TC, env=env, stdout=open(work / "console.log", "w"), stderr=subprocess.STDOUT)
        procs.append(cons)
        wait_http(f"{CONSOLE}/api/state", 120, cons)
        post(f"{CONSOLE}/api/sim", {"paused": True, "assistant_url": ASSIST})
        s0 = get(f"{CONSOLE}/api/state")
        left0 = s0["night"]["minutes_left"]
        note("console up", utc=s0.get("utc"), minutes_left=left0)

        def night_min() -> tuple:
            s = get(f"{CONSOLE}/api/state")
            return left0 - s["night"]["minutes_left"], s

        cw, ch = (int(x) for x in a.css.split("x"))
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            ctx = browser.new_context(viewport={"width": cw, "height": ch}, device_scale_factor=OUT_W / cw)
            page = ctx.new_page()
            rec = Recorder(page, work / "frames", a.fps)

            def caption(text: str):
                page.evaluate(CAPTION_JS, text)

            def card(title: str, sub: str = "", kicker: str = "", secs: float = 5.0):
                page.set_content(CARD.replace("{title}", title).replace("{sub}", sub).replace("{kicker}", kicker))
                rec.still(secs)

            # 1. cards
            card(
                "SkyJev: fast menu choices and a language model for telescope observing",
                "A planner, a fast local model that answers menu questions (Jev on Qwen3-1.7B), and GPT for the hard calls.",
                "recorded headlessly on an OSC Pitzer V100 node",
                6.0,
            )
            card(
                "Keck and Magellan here are simulated consoles built from public documentation.",
                "Telescope, weather, instrument and frames are simulated. Nothing in this video moves real hardware.",
                secs=5.5,
            )

            # 2. the console, assistant in autopilot
            page.goto(CONSOLE)
            page.wait_for_timeout(2500)
            post(f"{CONSOLE}/api/sim", {"paused": False})
            caption("Simulated Magellan Clay / MIKE night (dark time), 120x real time. The assistant runs in autopilot.")
            rec.hold(7)
            caption("The planner picks a target; the assistant sends the slew and the MIKE exposure to the console.")
            rec.hold(7)
            caption("⏭ jumps to the next event: end of a slew, an exposure or a readout.")
            for i in range(24):
                m, s = night_min()
                if m >= a.jump_to_min:
                    break
                page.click("#sim-ff")
                rec.hold(2.3)
                note("ff", night_min=round(m), weather=s.get("weather"), tcs=(s.get("tcs") or {}).get("name"))
            m, s = night_min()
            note("after ff", night_min=round(m), weather=s.get("weather"))
            page.click('button[data-tab="weather"]')
            caption("Weather page: the seeing has jumped. The assistant re-decides after every readout.")
            rec.hold(8)
            page.click('button[data-tab="sky"]')
            caption("Sky map and airmass chart, with the planner's projection for the rest of the night.")
            rec.hold(6)
            page.click('button[data-tab="assistant"]')
            caption("The console's Assistant tab: every exchange behind every decision, as JSON in and out.")
            rec.hold(6)
            post(f"{CONSOLE}/api/sim", {"paused": True})

            # 3. the assistant app: System 1 and one System 2 deliberation, one chat question
            page.goto(ASSIST)
            # the only errors listed are "connection refused" from before the console started (this
            # script starts the assistant first on purpose); they are not part of the demo
            page.add_style_tag(content="#errors{display:none}")
            page.wait_for_timeout(3000)
            st = get(f"{ASSIST}/api/status")
            note("assistant page", rec=st.get("recommendation"), stats=st.get("stats"))
            caption("The assistant: planner pick, System 1 ranking (Jev on Qwen3-1.7B, local V100) and the decisions so far.")
            rec.hold(9)
            if not a.no_s2 and (st.get("s2") or {}).get("enabled"):
                rid = (st.get("recommendation") or {}).get("id")
                caption("Ask System 2: GPT deliberates on the current call, checking the numbers with tools.")
                page.click("#deliberate")
                rec.hold(3)
                t0 = time.time()
                lat = None
                while time.time() - t0 < 240:
                    st = get(f"{ASSIST}/api/status")
                    r = st.get("recommendation") or {}
                    if r.get("id") == rid and "system2" in (r.get("latency_s") or {}) and not st["s2"]["busy"]:
                        lat = r["latency_s"]["system2"]
                        break
                    time.sleep(1)
                note("deliberate", latency_s=lat, s2=st.get("s2"), s2_pick=(st.get("recommendation") or {}).get("s2_pick"))
                page.wait_for_timeout(2500)
                caption(f"System 2 answered in {lat:.0f} s (the wait is cut)." if lat else "System 2 is still thinking.")
                rec.hold(9)
                caption("One question to System 2 in the chat box.")
                page.click("#chatin")
                for chunk in [QUESTION[i : i + 3] for i in range(0, len(QUESTION), 3)]:
                    page.type("#chatin", chunk)
                    rec.still(0.12)
                page.press("#chatin", "Enter")
                rec.hold(1.5)
                t0 = time.time()
                try:
                    page.wait_for_function(
                        "() => !document.querySelector('#chatlog .msg.ai.muted') && document.querySelector('#chatlog .msg.ai')",
                        timeout=240000,
                    )
                except Exception as e:  # keep recording; the log says what happened
                    note("chat timeout", error=str(e)[:200])
                dt = time.time() - t0
                page.evaluate("() => document.getElementById('chatcard').scrollIntoView({block: 'center'})")
                caption(f"The answer, after {dt:.0f} s (the wait is cut).")
                rec.hold(12)
                st = get(f"{ASSIST}/api/status")
                note("chat", seconds=round(dt, 1), s2=st.get("s2"))
            else:
                caption("System 2 (GPT) is off in this run: planner and System 1 only.")
                rec.hold(5)

            # trace view: System 1 and System 2 rows with their latencies
            page.goto(f"{ASSIST}/trace")
            page.wait_for_timeout(3000)
            for kind in ("planner", "console", "recommendation", "tool"):
                box = page.locator(f'#filters input[value="{kind}"]')
                if box.is_checked():
                    box.click()
            items = get(f"{ASSIST}/api/trace?after=0&limit=5000").get("items", [])
            s1_lat = sorted(e["latency_s"] for e in items if e["kind"] == "system1" and e.get("latency_s") is not None)
            s2_lat = [e["latency_s"] for e in items if e["kind"] == "system2" and e.get("latency_s") is not None]
            note("trace", n_s1=len(s1_lat), s1_median=s1_lat[len(s1_lat) // 2] if s1_lat else None, s2=s2_lat)
            text = "The trace, newest first."
            if s1_lat:
                text += f" System 1 (Jev) calls: median {s1_lat[len(s1_lat) // 2]:.2f} s over {len(s1_lat)} calls."
            if s2_lat:
                text += f" System 2 (GPT) calls: {min(s2_lat):.0f}-{max(s2_lat):.0f} s each."
            caption(text)
            rec.hold(9)
            rows = page.locator('.tr-row[data-kind="system1"] .tr-head')
            if rows.count():
                rows.first.click()
                rows.first.scroll_into_view_if_needed()
                page.evaluate("() => window.scrollBy(0, -60)")
                caption("A System 1 row: the prompt, the menu of options and the calibrated answer.")
                rec.hold(7)

            # 4. the browser agent: one live held-out task, or a pointer to its README
            if a.browser_task:
                browser_agent_segment(pw, rec, card, a.browser_task, work, note)
            else:
                card(
                    "Also in the repository: a fast browser agent on astronomy websites",
                    "Menu choices instead of free text: which operation, on which element. "
                    "Tasks on SIMBAD, NED, VizieR, ADS and the ESO archive. Not recorded here; see browser/README.md.",
                    secs=6.0,
                )
            # 5. end
            card("Code and results: github.com/seratsaad/SkyJev", "", secs=5.0)
            ctx.close()
            browser.close()

        out = Path(a.out)
        rec.encode(out, imageio_ffmpeg.get_ffmpeg_exe())
        st = get(f"{ASSIST}/api/status")
        usd = s2_cost((st.get("s2") or {}).get("usage") or {})
        note("done", video=str(out), length_s=round(rec.length, 1), s2=st.get("s2"), s2_usd_max=round(usd, 3))
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                p.kill()
        (work / "record_log.json").write_text(json.dumps(log, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
