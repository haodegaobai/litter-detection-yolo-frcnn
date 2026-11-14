"""Upload a video, scan it, play back the annotated result.

Single page, no framework: Python's own HTTP server is enough, so the
presentation machine needs nothing installed. The detection path is the same
one the desktop demo uses -- same checkpoint, same filters, same person veto --
so the numbers on screen match the ones in the results.

    python src/web_upload.py --open
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import models as M
from data_utils import CLASSES, ROOT
from live_demo import PersonVeto, Stabiliser, draw, filter_pred

WORK = ROOT / "runs" / "web_jobs"
WORK.mkdir(parents=True, exist_ok=True)

SAFE_CLASSES = [0, 1, 2]
MAX_UPLOAD = 300 * 1024 * 1024          # 300 MB
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()

WEIGHTS = ROOT / "runs" / "yolo_scene" / "best.pt"
MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = M.load_yolo(WEIGHTS)
    return MODEL


def draw_hud(frame: np.ndarray, n_det: int, scan_fps: float, frames: int) -> None:
    """Minimal overlay: what was found, and how fast the scan is going.

    Deliberately does not print per-frame inference time -- that number depends
    on batching and would sit next to the 12.34 ms measured under the benchmark's
    own protocol, inviting a comparison that is not apples to apples.
    """
    lines = [
        f"YOLO11n   detections: {n_det}",
        f"scan {scan_fps:.1f} fps   frame {frames}",
    ]
    # a dark plate first: white text on a sandy background is unreadable, and
    # this clip is sand
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0][0] for t in lines)
    plate = frame.copy()
    cv2.rectangle(plate, (0, 0), (width + 28, 24 + 26 * len(lines)), (16, 22, 32), -1)
    cv2.addWeighted(plate, 0.72, frame, 0.28, 0, frame)
    for i, t in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(frame, t, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 246, 255), 1, cv2.LINE_AA)


def ffmpeg_to_h264(src: Path, dst: Path) -> bool:
    """Browsers will not play the mp4v stream VideoWriter produces."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    cmd = [exe, "-y", "-v", "error", "-i", str(src), "-c:v", "libx264",
           "-profile:v", "main", "-pix_fmt", "yuv420p", "-crf", "23",
           "-movflags", "+faststart", "-an", str(dst)]
    return subprocess.run(cmd).returncode == 0


def process(job_id: str, src: Path) -> None:
    job = JOBS[job_id]
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        job.update(status="error", message="无法读取这个视频文件")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # keep the working width at 1280 so a 4K phone clip does not crawl
    scale = min(1.0, 1280 / max(w, 1))
    if scale < 1.0:
        w, h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2

    tmp = WORK / f"{job_id}_raw.mp4"
    out = WORK / f"{job_id}.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    model = get_model()
    veto = PersonVeto(ROOT / "yolo11n.pt") if (ROOT / "yolo11n.pt").exists() else None
    stab = Stabiliser(2)

    per_class = {c: 0 for c in ("plastic", "paper_cardboard", "metal")}
    frames = boxes = frames_with = 0
    lat: list[float] = []
    t_start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

            # without the sync the timer measures kernel launch, not the work
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            raw = M.predict_yolo(model, [frame], conf=0.05)[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            area = float(w * h)
            pred = filter_pred(raw, 0.60, SAFE_CLASSES, 4.0, 0.0005, area, 0)
            if veto is not None:
                pred, _ = veto.apply(pred, veto.persons(frame), area)
            pred = stab(pred)
            lat.append((time.perf_counter() - t0) * 1000)

            n = len(pred["scores"])
            boxes += n
            frames_with += 1 if n else 0
            for lab in pred["labels"]:
                name = CLASSES[int(lab)]
                if name in per_class:
                    per_class[name] += 1

            elapsed_now = time.perf_counter() - t_start
            draw(frame, pred)
            draw_hud(frame, n, frames / max(elapsed_now, 1e-6), frames)
            writer.write(frame)

            frames += 1
            if frames % 3 == 0 or frames == total:
                job.update(frames=frames, total=total or frames,
                           pct=min(99, int(frames / max(total, frames) * 100)),
                           boxes=boxes, frames_with=frames_with,
                           per_class=dict(per_class))
    finally:
        cap.release()
        writer.release()

    job.update(status="encoding", pct=99)
    if not ffmpeg_to_h264(tmp, out):
        out = tmp
    for leftover in (tmp, src):
        try:
            leftover.unlink(missing_ok=True)
        except OSError:
            pass

    elapsed = time.perf_counter() - t_start
    job.update(
        status="done", pct=100, frames=frames, boxes=boxes, frames_with=frames_with,
        per_class=dict(per_class),
        mean_ms=round(float(np.mean(lat)), 1) if lat else 0.0,
        p95_ms=round(float(np.percentile(lat, 95)), 1) if lat else 0.0,
        seconds=round(elapsed, 1),
        video=f"/result/{job_id}.mp4",
        src_name=job.get("src_name", ""),
    )


PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Litter Detection — Video Scan</title>
<style>
  :root{--navy:#111823;--panel:#1B2432;--line:#2C3648;--blue:#3A7BD5;--grey:#8B97A8;--green:#4FD37A;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--navy);color:#E8EEF7;font-family:"Microsoft YaHei",system-ui,sans-serif}
  header{padding:20px 36px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:16px}
  h1{font-size:21px;margin:0;font-weight:600}
  .sub{color:var(--grey);font-size:13px}
  main{max-width:1180px;margin:0 auto;padding:28px 24px 60px}
  #drop{border:2px dashed #3A4658;border-radius:10px;padding:64px 24px;text-align:center;
        cursor:pointer;transition:.15s;background:#151D29}
  #drop.hot{border-color:var(--blue);background:#172234}
  #drop h2{margin:0 0 8px;font-size:20px;font-weight:600}
  #drop p{margin:0;color:var(--grey);font-size:14px}
  .bar{height:10px;background:#222C3C;border-radius:5px;overflow:hidden;margin:22px 0 10px}
  .bar > i{display:block;height:100%;width:0;background:var(--blue);transition:width .2s}
  .status{color:var(--grey);font-size:14px}
  .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
  .card .k{color:var(--grey);font-size:12px;letter-spacing:1px}
  .card .v{font-size:30px;font-weight:700;line-height:1.2}
  video{width:100%;border-radius:8px;background:#000;display:block}
  .cls{display:flex;gap:14px;margin-top:14px}
  .cls div{flex:1;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
  .cls .k{color:var(--grey);font-size:12px}
  .cls .v{font-size:24px;font-weight:700}
  .again{margin-top:22px;display:inline-block;padding:10px 20px;border:1px solid var(--line);
         border-radius:6px;color:#E8EEF7;text-decoration:none;font-size:14px;cursor:pointer;background:#1B2432}
  .hide{display:none}
</style></head><body>
<header><h1>Litter Detection</h1>
  <span class="sub">YOLO11n · 640 px · conf 0.60 · person veto on</span></header>
<main>
  <div id="drop">
    <h2>把视频拖进来，或者点击选择</h2>
    <p>mp4 / mov / avi，最大 300 MB。系统会逐帧扫描并输出带框的成片。</p>
  </div>
  <input type="file" id="file" accept="video/*" class="hide">

  <div id="progress" class="hide">
    <div class="bar"><i id="fill"></i></div>
    <div class="status" id="stat">上传中…</div>
  </div>

  <div id="result" class="hide">
    <video id="player" controls autoplay muted loop></video>
    <div class="grid">
      <div class="card"><div class="k">FRAMES</div><div class="v" id="m_frames">0</div></div>
      <div class="card"><div class="k">DETECTIONS</div><div class="v" id="m_boxes">0</div></div>
      <div class="card"><div class="k">BOXES / FRAME</div><div class="v" id="m_bpf">0</div></div>
      <div class="card"><div class="k">SCAN SPEED</div><div class="v" id="m_fps">0<span style="font-size:14px;color:#8B97A8"> fps</span></div><div class="k" id="m_sec" style="margin-top:4px">–</div></div>
    </div>
    <div class="cls">
      <div><div class="k">PLASTIC</div><div class="v" id="c_plastic">0</div></div>
      <div><div class="k">PAPER / CARDBOARD</div><div class="v" id="c_paper">0</div></div>
      <div><div class="k">METAL</div><div class="v" id="c_metal">0</div></div>
    </div>
    <div class="again" id="again">换一个视频</div>
  </div>
</main>
<script>
const $=id=>document.getElementById(id);
const drop=$('drop'), file=$('file'), prog=$('progress'), res=$('result');

drop.onclick=()=>file.click();
file.onchange=()=>file.files[0] && send(file.files[0]);
['dragenter','dragover'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hot')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('hot')}));
drop.addEventListener('drop',ev=>{const f=ev.dataTransfer.files[0]; if(f) send(f);});
$('again').onclick=()=>{res.classList.add('hide');prog.classList.add('hide');drop.classList.remove('hide');file.value='';};

async function send(f){
  drop.classList.add('hide'); prog.classList.remove('hide'); res.classList.add('hide');
  $('fill').style.width='0%'; $('stat').textContent='上传中… '+f.name;
  let r;
  try{
    r = await fetch('/upload?name='+encodeURIComponent(f.name), {method:'POST', body:f});
  }catch(e){ $('stat').textContent='上传失败：'+e; return; }
  const j = await r.json();
  if(!j.ok){ $('stat').textContent='失败：'+(j.message||''); return; }
  poll(j.id);
}

async function poll(id){
  const t = setInterval(async()=>{
    let s;
    try{ s = await (await fetch('/progress?id='+id)).json(); }catch(e){ return; }
    if(s.status==='error'){ clearInterval(t); $('stat').textContent='失败：'+s.message; return; }
    $('fill').style.width = s.pct+'%';
    if(s.status==='encoding'){ $('stat').textContent='正在编码输出视频…'; return; }
    if(s.status==='done'){
      clearInterval(t);
      $('fill').style.width='100%';
      prog.classList.add('hide'); res.classList.remove('hide');
      $('m_frames').textContent = s.frames;
      $('m_boxes').textContent = s.boxes;
      $('m_bpf').textContent = (s.boxes/Math.max(s.frames,1)).toFixed(2);
      $('m_fps').innerHTML = (s.frames/Math.max(s.seconds,0.01)).toFixed(1)+'<span style="font-size:14px;color:#8B97A8"> fps</span>';
      $('m_sec').textContent = s.seconds+' s total';
      $('c_plastic').textContent = s.per_class.plastic||0;
      $('c_paper').textContent = s.per_class.paper_cardboard||0;
      $('c_metal').textContent = s.per_class.metal||0;
      $('player').src = s.video;
      return;
    }
    $('stat').textContent = `扫描中 ${s.frames} / ${s.total} 帧 · 已检出 ${s.boxes} 个框`;
  }, 400);
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif u.path == "/progress":
            jid = parse_qs(u.query).get("id", [""])[0]
            job = JOBS.get(jid)
            self._json(job if job else {"status": "error", "message": "unknown job"})
        elif u.path.startswith("/result/"):
            self._serve_file(WORK / u.path.rsplit("/", 1)[-1])
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_file(self, path: Path):
        if not path.exists():
            self.send_response(404)
            self.end_headers()
            return
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
        length = end - start + 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except OSError:
                    return
                remaining -= len(chunk)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/upload":
            self.send_response(404)
            self.end_headers()
            return
        name = parse_qs(u.query).get("name", ["clip.mp4"])[0]
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > MAX_UPLOAD:
            self._json({"ok": False, "message": "文件为空或超过 300 MB"}, 400)
            return

        jid = uuid.uuid4().hex[:10]
        suffix = Path(name).suffix.lower() or ".mp4"
        src = WORK / f"{jid}_src{suffix}"
        got = 0
        with open(src, "wb") as fh:
            while got < length:
                chunk = self.rfile.read(min(1 << 20, length - got))
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)

        JOBS[jid] = {"status": "running", "pct": 0, "frames": 0, "total": 0,
                     "boxes": 0, "frames_with": 0, "per_class": {}, "src_name": name}
        threading.Thread(target=process, args=(jid, src), daemon=True).start()
        self._json({"ok": True, "id": jid})


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    if not WEIGHTS.exists():
        sys.exit(f"missing weights: {WEIGHTS}")

    print("loading checkpoint...")
    get_model()

    url = f"http://127.0.0.1:{args.port}/"
    print(f"serving on {url}\nCtrl+C to stop")
    if args.open:
        webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
