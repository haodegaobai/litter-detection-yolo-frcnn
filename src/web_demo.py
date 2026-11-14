"""Minimal web front end for the litter detector.

One page, one video stream, three numbers. No framework -- Python's own HTTP
server is enough, so there is nothing to install on the presentation machine.

The detection path is the same as `live_demo.py`: same checkpoint, same
filters, same person veto, same HUD. Only the transport changes, so a number
quoted on stage and a number shown in the browser come from identical code.

    python src/web_demo.py                     # webcam
    python src/web_demo.py --source clip.mp4   # file, loops (for testing)
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import models as M
from data_utils import CLASSES, ROOT
from live_demo import PersonVeto, Stabiliser, draw, filter_pred, hud

SAFE_CLASSES = [0, 1, 2]          # plastic, paper_cardboard, metal

STATE = {
    "conf": 0.60,
    "loop_fps": 0.0,
    "model_fps": 0.0,
    "ms": 0.0,
    "p50": 0.0,
    "p95": 0.0,
    "n_det": 0,
    "vetoed": 0,
    "frames": 0,
    "boxes_total": 0,
    "safe": True,
    "recording": False,
}
LATEST = {"jpeg": b""}
LOCK = threading.Lock()


class Capture(threading.Thread):
    """Grab, infer, annotate, encode -- one loop, one camera handle."""

    def __init__(self, source: str | None, weights: Path, width: int, height: int,
                 loop_file: bool):
        super().__init__(daemon=True)
        self.source = source
        self.loop_file = loop_file
        self.width, self.height = width, height
        self.model = M.load_yolo(weights)
        self.veto = PersonVeto(ROOT / "yolo11n.pt") if (ROOT / "yolo11n.pt").exists() else None
        self.stabiliser = Stabiliser(2)
        self.lat: list[float] = []
        self.stop = threading.Event()

    def _open(self):
        if self.source:
            cap = cv2.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def run(self) -> None:
        cap = self._open()
        if not cap.isOpened():
            print("[web_demo] cannot open source:", self.source or "camera 0", file=sys.stderr)
            return

        prev_t = time.perf_counter()
        while not self.stop.is_set():
            ok, frame = cap.read()
            if not ok:
                if self.source and self.loop_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            t0 = time.perf_counter()
            raw = M.predict_yolo(self.model, [frame], conf=0.05)[0]
            t1 = time.perf_counter()

            h, w = frame.shape[:2]
            area = float(h * w)
            conf = STATE["conf"]
            pred = filter_pred(raw, conf, SAFE_CLASSES, 4.0, 0.0005, area, 0)
            n_veto = 0
            if self.veto is not None:
                pred, n_veto = self.veto.apply(pred, self.veto.persons(frame), area)
            pred = self.stabiliser(pred)

            ms = (t1 - t0) * 1000
            self.lat.append(ms)
            self.lat = self.lat[-100:]

            now = time.perf_counter()
            loop_fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            arr = np.asarray(self.lat)
            with LOCK:
                STATE.update(
                    loop_fps=loop_fps,
                    model_fps=1000.0 / max(ms, 1e-6),
                    ms=ms,
                    p50=float(np.percentile(arr, 50)),
                    p95=float(np.percentile(arr, 95)),
                    n_det=int(len(pred["scores"])),
                    vetoed=n_veto,
                    frames=STATE["frames"] + 1,
                    boxes_total=STATE["boxes_total"] + int(len(pred["scores"])),
                )

            draw(frame, pred)
            hud(frame, STATE, "YOLO11n")
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                with LOCK:
                    LATEST["jpeg"] = buf.tobytes()

        cap.release()


PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Campus Litter Detection</title>
<style>
  :root { --navy:#1A2230; --blue:#1E4FA8; --line:#2C3648; --grey:#8B97A8; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--navy); color:#E8EEF7;
         font-family:"Microsoft YaHei",system-ui,sans-serif; }
  header { padding:18px 32px; border-bottom:1px solid var(--line); display:flex;
           align-items:baseline; gap:16px; }
  h1 { font-size:20px; margin:0; font-weight:600; }
  .sub { color:var(--grey); font-size:13px; }
  main { display:flex; gap:28px; padding:24px 32px; align-items:flex-start; }
  .video { flex:1 1 auto; background:#000; border:1px solid var(--line); border-radius:6px;
           overflow:hidden; line-height:0; }
  .video img { width:100%; height:auto; display:block; }
  aside { width:300px; flex:0 0 300px; display:flex; flex-direction:column; gap:14px; }
  .card { background:#212B3B; border:1px solid var(--line); border-radius:6px; padding:14px 16px; }
  .label { color:var(--grey); font-size:12px; letter-spacing:1px; }
  .value { font-size:34px; font-weight:700; line-height:1.15; }
  .row { display:flex; gap:12px; }
  .row .card { flex:1; }
  .unit { font-size:14px; color:var(--grey); font-weight:400; }
  input[type=range] { width:100%; accent-color:var(--blue); }
  .badge { display:inline-block; padding:2px 8px; border-radius:3px; font-size:12px;
           background:#12351F; color:#4FD37A; border:1px solid #2C6B42; }
  footer { padding:0 32px 24px; color:var(--grey); font-size:12px; }
</style></head><body>
<header>
  <h1>Campus Litter Detection</h1>
  <span class="sub">YOLO11n · 640 px · plastic / paper_cardboard / metal</span>
  <span class="badge" id="mode">SAFE MODE</span>
</header>
<main>
  <div class="video"><img src="/video" alt="live"></div>
  <aside>
    <div class="card">
      <div class="label">DETECTIONS</div>
      <div class="value" id="det">0</div>
    </div>
    <div class="row">
      <div class="card"><div class="label">LOOP FPS</div>
        <div class="value" id="loop">0<span class="unit"> fps</span></div></div>
      <div class="card"><div class="label">MODEL FPS</div>
        <div class="value" id="model">0<span class="unit"> fps</span></div></div>
    </div>
    <div class="card">
      <div class="label">INFERENCE</div>
      <div class="value" id="ms">0<span class="unit"> ms</span></div>
      <div class="label" style="margin-top:6px" id="p95">p95 –</div>
    </div>
    <div class="card">
      <div class="label">CONFIDENCE THRESHOLD <span id="confv">0.60</span></div>
      <input type="range" id="conf" min="0.20" max="0.90" step="0.05" value="0.60">
    </div>
    <div class="card">
      <div class="label">SESSION</div>
      <div class="label" id="session" style="margin-top:4px">0 frames · 0 boxes</div>
    </div>
  </aside>
</main>
<footer>同一份代码路径：checkpoint、过滤、人脸否决与桌面演示完全一致。</footer>
<script>
const $ = id => document.getElementById(id);
async function tick() {
  try {
    const s = await (await fetch('/stats')).json();
    $('det').textContent = s.n_det;
    $('loop').innerHTML = s.loop_fps.toFixed(1) + '<span class="unit"> fps</span>';
    $('model').innerHTML = s.model_fps.toFixed(1) + '<span class="unit"> fps</span>';
    $('ms').innerHTML = s.ms.toFixed(1) + '<span class="unit"> ms</span>';
    $('p95').textContent = 'p95 ' + s.p95.toFixed(1) + ' ms' + (s.vetoed ? ' · vetoed ' + s.vetoed : '');
    $('session').textContent = s.frames + ' frames · ' + s.boxes_total + ' boxes';
  } catch (e) {}
}
setInterval(tick, 500); tick();
const slider = $('conf');
slider.addEventListener('input', async () => {
  $('confv').textContent = Number(slider.value).toFixed(2);
  await fetch('/set?conf=' + slider.value);
});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 on purpose: an MJPEG stream has no Content-Length, so the body has
    # to be delimited by the connection closing. Under HTTP/1.1 the client waits
    # for a length that never arrives and renders an empty <img>.
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/stats":
            with LOCK:
                body = json.dumps({k: STATE[k] for k in
                                   ("conf", "loop_fps", "model_fps", "ms", "p50", "p95",
                                    "n_det", "vetoed", "frames", "boxes_total")}).encode()
            self._send(200, body, "application/json")
        elif self.path.startswith("/set"):
            try:
                q = self.path.split("?", 1)[1]
                conf = float(dict(p.split("=") for p in q.split("&"))["conf"])
                with LOCK:
                    STATE["conf"] = max(0.05, min(0.95, conf))
                self._send(200, b'{"ok":true}', "application/json")
            except Exception:
                self._send(400, b'{"ok":false}', "application/json")
        elif self.path == "/video":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with LOCK:
                        buf = LATEST["jpeg"]
                    if not buf:
                        time.sleep(0.02)
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
                    self.wfile.write(str(len(buf)).encode())
                    self.wfile.write(b"\r\n\r\n")
                    self.wfile.write(buf)
                    self.wfile.write(b"\r\n")
                    time.sleep(1 / 30)
            except OSError:
                # the browser closes the stream whenever the page reloads or the
                # tab is backgrounded; on Windows that surfaces as
                # ConnectionAbortedError, which is not a BrokenPipeError
                return
        else:
            self._send(404, b"not found", "text/plain")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="video file instead of the webcam")
    ap.add_argument("--weights", default="runs/yolo_scene/best.pt")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--conf", type=float, default=0.60)
    ap.add_argument("--open", action="store_true", help="open the browser once frames arrive")
    args = ap.parse_args()

    STATE["conf"] = args.conf
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights
    if not weights.exists():
        sys.exit(f"missing weights: {weights}")

    cam = Capture(args.source, weights, args.width, args.height, loop_file=bool(args.source))
    cam.start()
    # the first inference warms up CUDA; give it up to 20 s rather than guessing
    deadline = time.time() + 20
    while time.time() < deadline and not LATEST["jpeg"]:
        time.sleep(0.25)
    if not LATEST["jpeg"]:
        sys.exit("no frames captured -- is the camera in use by another program?")

    url = f"http://127.0.0.1:{args.port}/"
    print(f"serving on {url}")
    if args.open:
        webbrowser.open(url)
    print("press Ctrl+C to stop")
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        cam.stop.set()
        print("\nstopped")


if __name__ == "__main__":
    main()
