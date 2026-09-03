#!/usr/bin/env python3
"""Interactive labeler: mark sound event start/end intervals by watching the video.

Usage:
    python label_gui.py video.mp4
    python autocut.py video.mp4 --label

It starts a tiny local server (127.0.0.1 only), opens the video in your
browser and lets you mark the exact start/end of each sound event:

  - the audio waveform (traccia audio) under the video: *drag* over an event to
    mark it exactly, *click* to jump, blue rectangles show the labels;
  - keys: S = inizio, E = fine, [ / ] = un frame, Shift+[ / ] = 10 frame,
    Space = play/pausa, Z = annulla, Ctrl+S = salva;
  - 🔁 Ripeti ultimo replays the last label in a loop to check the borders.

On save it writes a label file like:

    <video>_etichette.txt
        1:54.55-1:55.20
        1:57.10-1:57.62

which you then pass to the editor with --teach @<file>.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import socket
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from wave_ui import WAVE_JS, serve_wave

PAGE = """<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Etichetta eventi audio</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 900px;
         margin: 20px auto; padding: 0 16px; background: #111; color: #eee; }
  video { width: 100%; max-height: 55vh; background: #000; border-radius: 8px; }
  .time { font-size: 28px; font-variant-numeric: tabular-nums; margin: 10px 0; }
  .btn { background: #2a6df4; color: #fff; border: 0; border-radius: 6px;
         padding: 10px 14px; font-size: 15px; cursor: pointer; margin: 3px; }
  .btn:hover { background: #1f56c4; }
  .btn.start { background: #1e9e4a; } .btn.end { background: #d64541; }
  .btn.small { padding: 6px 10px; font-size: 13px; background: #444; }
  .btn.small:hover { background: #555; }
  .row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
  ul { list-style: none; padding: 0; }
  li { background: #1c1c1c; border: 1px solid #333; border-radius: 6px;
       padding: 8px 12px; margin: 6px 0; display: flex; justify-content: space-between;
       align-items: center; cursor: pointer; }
  li:hover { border-color: #2a6df4; }
  .del { color: #d64541; cursor: pointer; border: 0; background: none; font-size: 18px; }
  .hint { color: #aaa; font-size: 14px; line-height: 1.6; }
  kbd { background: #333; border-radius: 4px; padding: 2px 6px; font-size: 13px; }
  .wave-wrap { position: relative; margin: 10px 0 4px; }
  #wave { width: 100%; height: 80px; background: #0d0e11; border: 1px solid #333;
          border-radius: 8px; cursor: crosshair; display: block; }
  #waveHint { color: #888; font-size: 12px; margin: 4px 0 10px; }
  select { background: #222; color: #eee; border: 1px solid #333; border-radius: 6px;
           padding: 6px 8px; font-size: 13px; }
</style>
</head>
<body>
  <h2>🎬 Etichetta eventi sonori</h2>
  <video id="v" controls preload="auto" src="/video"></video>
  <div class="time" id="t">0:00.000 / 0:00.000</div>
  <div class="wave-wrap">
    <canvas id="wave"></canvas>
    <div id="waveHint">caricamento traccia audio…</div>
  </div>
  <div class="row">
    <button class="btn start" id="bStart">▶ Inizio evento  <kbd>S</kbd></button>
    <button class="btn end" id="bEnd">⏹ Fine evento  <kbd>E</kbd></button>
    <button class="btn" id="bUndo">↩ Annulla  <kbd>Z</kbd></button>
    <button class="btn" id="bClear">🗑 Svuota</button>
    <button class="btn" id="bSave" style="background:#f4a72a">💾 Salva  <kbd>Ctrl+S</kbd></button>
  </div>
  <div class="row">
    <button class="btn small" id="bFPrev">◀◀ frame  <kbd>[</kbd></button>
    <button class="btn small" id="bFNext">frame ▶▶  <kbd>]</kbd></button>
    <select id="fpsSel" title="fotogrammi al secondo usati dallo step">
      <option value="auto" selected>fps auto</option>
      <option>24</option><option>25</option><option>30</option><option>60</option>
    </select>
    <button class="btn small" id="bLoop">🔁 Ripeti ultimo</button>
    <button class="btn small" id="bM5">−5s</button>
    <button class="btn small" id="bM05">−0.5s</button>
    <button class="btn small" id="bP05">+0.5s</button>
    <button class="btn small" id="bP5">+5s</button>
  </div>
  <p class="hint" id="stato">Premi <kbd>S</kbd> all'inizio dell'evento sonoro e <kbd>E</kbd> alla
    fine — oppure <b>trascina direttamente sulla traccia audio</b> qui sopra per marcare
    l'intervallo esatto. <kbd>←</kbd>/<kbd>→</kbd> = 0.1s, <kbd>[</kbd>/<kbd>]</kbd> = un
    frame, <kbd>Spazio</kbd> = play/pausa. Clicca un intervallo per andarci. Le etichette
    si salvano con precisione al centesimo di secondo.</p>
  <h3>Intervalli marcati: <span id="count">0</span></h3>
  <ul id="list"></ul>
<script>
const v = document.getElementById('v');
let labels = [];
let openStart = null;
window.waveUrl = '/wave';
window.wave = {
  video: () => v,
  labels: () => labels,
  refresh: () => refreshLabels(),
  add: (s, e) => { labels.push([s, e]); labels.sort((a, b) => a[0] - b[0]); refreshLabels(); },
  openStart: () => openStart,
};

function fmt(s) {
  s = Math.max(0, s);
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return m + ':' + sec.toFixed(3).padStart(6, '0');
}
function refreshLabels() {
  document.getElementById('t').textContent =
    fmt(v.currentTime || 0) + ' / ' + fmt(v.duration || 0);
  const ul = document.getElementById('list');
  ul.innerHTML = '';
  labels.forEach((it, i) => {
    const li = document.createElement('li');
    li.innerHTML = '<b>#' + (i + 1) + '</b> ' + fmt(it[0]) + ' → ' + fmt(it[1]) +
      ' <span style="color:#888">(' + (it[1] - it[0]).toFixed(2) + 's)</span>' +
      '<button class="del" title="elimina">×</button>';
    li.addEventListener('click', e => {
      if (e.target.className !== 'del') { v.currentTime = it[0]; v.play(); }
    });
    li.querySelector('.del').addEventListener('click', e => {
      e.stopPropagation(); labels.splice(i, 1);
      if (openStart !== null && openStart >= it[0] - 0.001 && openStart <= it[1] + 0.001) openStart = null;
      refreshLabels();
    });
    ul.appendChild(li);
  });
  document.getElementById('count').textContent = labels.length;
  document.getElementById('stato').style.color = openStart !== null ? '#1e9e4a' : '#aaa';
  if (window.redrawWave) redrawWave();
}
v.addEventListener('timeupdate', refreshLabels);
v.addEventListener('loadedmetadata', refreshLabels);

function markStart() { openStart = v.currentTime; refreshLabels(); }
function markEnd() {
  if (openStart === null) return;
  const e = v.currentTime;
  if (e - openStart >= 0.05) { labels.push([openStart, e]); labels.sort((a, b) => a[0] - b[0]); }
  openStart = null; refreshLabels();
}
function undo() {
  if (openStart !== null) { openStart = null; }
  else labels.pop();
  refreshLabels();
}
function clearAll() { labels = []; openStart = null; refreshLabels(); }

document.getElementById('bStart').onclick = markStart;
document.getElementById('bEnd').onclick = markEnd;
document.getElementById('bUndo').onclick = undo;
document.getElementById('bClear').onclick = clearAll;
document.getElementById('bM5').onclick = () => v.currentTime = Math.max(0, v.currentTime - 5);
document.getElementById('bM05').onclick = () => v.currentTime = Math.max(0, v.currentTime - 0.5);
document.getElementById('bP05').onclick = () => v.currentTime = Math.min(v.duration || 1e9, v.currentTime + 0.5);
document.getElementById('bP5').onclick = () => v.currentTime = Math.min(v.duration || 1e9, v.currentTime + 5);

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
  if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); save(); return; }
  const step = e.shiftKey ? 1 : 0.1;
  switch (e.key) {
    case ' ': e.preventDefault(); v.paused ? v.play() : v.pause(); break;
    case 's': case 'S': markStart(); break;
    case 'e': case 'E': markEnd(); break;
    case 'z': case 'Z': undo(); break;
    case 'ArrowLeft': e.preventDefault(); v.currentTime = Math.max(0, v.currentTime - step); break;
    case 'ArrowRight': e.preventDefault(); v.currentTime = Math.min(v.duration || 1e9, v.currentTime + step); break;
  }
});
document.getElementById('bSave').onclick = save;
function save() {
  fetch('/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ labels: labels })
  }).then(r => r.text()).then(msg => {
    document.getElementById('stato').textContent = '✅ ' + msg +
      ' — ora esegui:  ./autocut.command ' + 'VIDEO' + ' --teach @FILE';
  });
}
</script>
<script src="/wave.js"></script>
</body>
</html>
"""


def fmt_label(t: float) -> str:
    m = int(t // 60)
    s = t - m * 60
    return f"{m}:{s:05.2f}"


class LabelHandler(BaseHTTPRequestHandler):
    video_path: Path = Path()
    labels_path: Path = Path()
    saved_labels: list = []

    def log_message(self, *args):  # keep the console clean
        pass

    def do_GET(self):
        if self.path in ("/", "/etichetta", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/video":
            self._serve_file(self.video_path)
        elif self.path == "/wave":
            serve_wave(self, self.video_path)
        elif self.path == "/wave.js":
            body = WAVE_JS.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def _serve_file(self, path: Path):
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng[len("bytes="):].split(",", 1)[0]
            if "-" in spec:
                a, b = spec.split("-", 1)
                if a:
                    start = int(a)
                if b:
                    end = int(b)
                end = min(end, size - 1)
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_POST(self):
        if self.path != "/save":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length) or b"{}")
        labels = sorted(
            (float(s), float(e)) for s, e in data.get("labels", []) if e - s >= 0.05)
        lines = ["# Etichette create con la GUI (inizio-fine di ogni evento sonoro)\n"]
        lines += [f"{fmt_label(s)}-{fmt_label(e)}\n" for s, e in labels]
        self.labels_path.write_text("".join(lines), encoding="utf-8")
        msg = f"Salvate {len(labels)} etichette in {self.labels_path.name}"
        body = msg.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        print(f"\n  ✅ {msg}")
        print(f"     Ora esegui:  ./autocut.command {self.video_path.name} "
              f"--teach @{self.labels_path.name} --limit-minutes 10")


def run_gui(video: Path) -> None:
    if not video.is_file():
        print(f"ERROR: file non trovato: {video}", file=sys.stderr)
        sys.exit(1)
    labels_path = video.with_name(video.stem + "_etichette.txt")

    LabelHandler.video_path = video
    LabelHandler.labels_path = labels_path

    # Pick a free port on localhost.
    port = None
    for candidate in range(8765, 8790):
        try:
            server = HTTPServer(("127.0.0.1", candidate), LabelHandler)
            port = candidate
            break
        except OSError:
            continue
    if port is None:
        print("ERROR: nessuna porta libera per il server locale.", file=sys.stderr)
        sys.exit(1)

    url = f"http://127.0.0.1:{port}/"
    print(f"Video: {video}")
    print(f"Apri il browser: {url}")
    print("Nel browser: S = inizio evento, E = fine, oppure trascina sulla "
          "traccia audio. [ ] = un frame, Z = annulla, "
          "←/→ = 0.1s, Spazio = play/pausa, Ctrl+S = salva")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nChiuso. Le etichette (se salvate) sono in "
              f"{labels_path.name} — usale con --teach @{labels_path.name}")
        sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="video o audio da etichettare")
    args = parser.parse_args()
    run_gui(args.video)


if __name__ == "__main__":
    main()