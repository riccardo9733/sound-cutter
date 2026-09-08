#!/usr/bin/env python3
"""AutoCut — complete local web UI for audio event detection and video cutting.

Covers the whole workflow in one page:
  1. pick the source video and preview it in the browser
  2. mark target sound events (start/end) with keyboard shortcuts
  3. choose the detection mode (YAMNet / teach / train-from / saved model)
     and tune the parameters (threshold, pad, min-gap, min-duration,
     limit-minutes, safe, outdir)
  4. run the full pipeline with live log streaming and a Stop button
  5. browse the results: the detected events (click to jump the preview)
     and the two output cuts, playable inline and downloadable

Only the Python standard library is used (http.server + threads + subprocess);
the browser does the rest. The server binds to 127.0.0.1 only.

Usage:
    python app_ui.py [video]
    python autocut.py video.mp4 --ui
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

from wave_ui import WAVE_JS, serve_wave

ROOT = Path(__file__).resolve().parent
AUTOCUT_SCRIPT = ROOT / "autocut.py"

MEDIA_EXTS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".m4a",
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".wma", ".mpeg", ".mpg",
}

# ---------------------------------------------------------------- state --

class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.video: Path | None = None       # selected source video
        self.allowed: set[Path] = set()      # extra files permitted to serve
        self.proc: subprocess.Popen | None = None
        self.run_lines: list[str] = []
        self.running = False
        self.done = False
        self.exit_code: int | None = None
        self.command: str = ""

    def allow(self, path: Path):
        with self.lock:
            self.allowed.add(path)


ST = State()


def list_files() -> dict:
    """Discover media files and saved models under the project folder."""
    videos: list[str] = []
    models: list[str] = []
    for base in (ROOT, ROOT / "models", ROOT / "test"):
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir(), key=lambda q: q.name.lower()):
            if not p.is_file() or p.name.startswith("."):
                continue
            if p.suffix.lower() in MEDIA_EXTS:
                videos.append(str(p))
            elif p.suffix.lower() == ".npz":
                models.append(str(p))
    return {"videos": videos, "models": models}


def is_servable(path: Path) -> bool:
    """A file may be served if it lives under the project root or was
    explicitly chosen by the user (current selection / reference etc.)."""
    try:
        if path.resolve().is_relative_to(ROOT.resolve()):
            return True
    except ValueError:
        pass
    with ST.lock:
        return path in ST.allowed


# ---------------------------------------------------------------- run ----

def build_command(params: dict) -> tuple[list[str], Path]:
    """Translate the UI form into autocut.py CLI arguments.

    Raises ValueError with a friendly message on invalid input.
    """
    video = Path(str(params.get("video", "")))
    if not video.is_file():
        raise ValueError(f"File video non trovato: {video}")

    mode = params.get("mode", "yamnet")
    cmd = [str(AUTOCUT_SCRIPT), str(video)]
    labels_file: Path | None = None

    no_train = bool(params.get("no_train"))
    no_cut = bool(params.get("no_cut"))

    approved = params.get("approved_events")
    if approved is not None and not no_cut:
        if not isinstance(approved, list) or len(approved) == 0:
            raise ValueError("Tutti gli eventi sono stati eliminati o esclusi dalla soglia. Nessun evento da esportare.")
        tfile = tempfile.NamedTemporaryFile(
            mode="w", prefix="autocut_approved_", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(approved, tfile)
        tfile.close()
        cmd = [str(AUTOCUT_SCRIPT), str(video), "--events-file", tfile.name]
        export_mode = params.get("export_mode", "both")
        if export_mode and export_mode in ("both", "events", "clean"):
            cmd += ["--export", export_mode]
        if params.get("safe"):
            cmd += ["--safe"]
        outdir = str(params.get("outdir", "")).strip()
        if outdir:
            p = Path(outdir)
            if not p.is_absolute():
                p = ROOT / p
            cmd += ["--outdir", str(p)]
        return cmd, None

    if no_cut:
        cmd += ["--no-cut"]

    if no_train:
        # Pure scan: evaluate audio using existing model or YAMNet without retraining or saving
        model_path = None
        if params.get("model"):
            p = Path(str(params.get("model")).strip())
            if not p.is_file() and not p.is_absolute():
                if (ROOT / "models" / p).is_file():
                    p = ROOT / "models" / p
                elif (ROOT / p).is_file():
                    p = ROOT / p
            if p.is_file():
                model_path = p
        if not model_path and params.get("save_model"):
            p = Path(str(params.get("save_model")).strip())
            if not p.is_file() and not p.is_absolute():
                if (ROOT / "models" / p).is_file():
                    p = ROOT / "models" / p
                elif (ROOT / p).is_file():
                    p = ROOT / p
            if p.is_file():
                model_path = p
        if not model_path and mode in ("teach", "model"):
            candidate_npzs = list(ROOT.glob("*.npz"))
            if (ROOT / "models").is_dir():
                candidate_npzs += list((ROOT / "models").glob("*.npz"))
            npzs = sorted(candidate_npzs, key=lambda x: x.stat().st_mtime, reverse=True)
            if npzs:
                model_path = npzs[0]

        if model_path:
            ST.allow(model_path)
            cmd += ["--load-model", str(model_path)]
    else:
        labels_file: Path | None = None
        if mode == "teach":
            raw_labels = params.get("teach_labels") or []
            parsed = []
            for it in raw_labels:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    s, e = float(it[0]), float(it[1])
                    t = it[2] if len(it) > 2 else "pos"
                    if e - s >= 0.05:
                        parsed.append((s, e, t))
                elif isinstance(it, dict):
                    s, e = float(it["start"]), float(it["end"])
                    t = it.get("type", "pos")
                    if e - s >= 0.05:
                        parsed.append((s, e, t))
            pos_count = sum(1 for s, e, t in parsed if t == "pos")
            if not parsed:
                raise ValueError("Modalità 'impara dagli esempi': scrivi o seleziona almeno "
                                 "un intervallo positivo [+] (es. 1:54-1:56).")
            if pos_count == 0:
                raise ValueError(
                    "Modalità 'impara dagli esempi': hai selezionato solo esempi negativi [-]. "
                    "Devi includere almeno 1 esempio positivo [+] (il suono target da rilevare) "
                    "con '▶ Inizio [+] (S)' o '✏️ Come [+]', così il modello sa cosa deve trovare oltre a cosa scartare!"
                )
            labels_file = video.with_name(video.stem + "_etichette.txt")
            _write_labels_file(labels_file, parsed)
            cmd += ["--teach", "@" + str(labels_file)]
        elif mode == "train":
            ref = Path(str(params.get("reference", "")))
            if not ref.is_file():
                raise ValueError("Modalità 'impara da un riferimento': scegli un "
                                 "file di riferimento valido.")
            ST.allow(ref)
            cmd += ["--train-from", str(ref)]
        elif mode == "model":
            model_str = str(params.get("model", "")).strip()
            model = Path(model_str)
            if not model.is_file() and not model.is_absolute():
                if (ROOT / "models" / model).is_file():
                    model = ROOT / "models" / model
                elif (ROOT / model).is_file():
                    model = ROOT / model
            if not model.is_file():
                raise ValueError("Modalità 'usa modello salvato': scegli un file "
                                 ".npz valido.")
            ST.allow(model)
            cmd += ["--load-model", str(model)]
        # "yamnet" -> no extra flag, uses the pretrained model

        save_model = str(params.get("save_model", "")).strip()
        if not save_model and params.get("model") and mode == "teach":
            save_model = str(params.get("model", "")).strip()
        if save_model and mode in ("teach", "train"):
            p = Path(save_model)
            if not p.is_absolute():
                if (ROOT / "models").is_dir() and len(p.parts) == 1:
                    p = ROOT / "models" / p
                else:
                    p = ROOT / p
            cmd += ["--save-model", str(p)]

    if params.get("auto_threshold"):
        cmd += ["--auto-threshold"]
    elif params.get("threshold") not in (None, ""):
        try:
            th_val = float(params.get("threshold"))
            if th_val > 0:
                cmd += ["--threshold", str(th_val)]
        except (TypeError, ValueError):
            raise ValueError(f"Valore non valido per --threshold: {params.get('threshold')!r}")

    if no_cut:
        pad_val = params.get("pad")
        if pad_val not in (None, ""):
            try:
                p_val = float(pad_val)
                if p_val > 0:
                    cmd += ["--pad", str(p_val)]
                else:
                    cmd += ["--pad", "0"]
            except (TypeError, ValueError):
                cmd += ["--pad", "0"]
        else:
            cmd += ["--pad", "0"]
    else:
        pad_val = params.get("pad")
        if pad_val not in (None, ""):
            try:
                p_val = float(pad_val)
                if p_val >= 0:
                    cmd += ["--pad", str(p_val)]
            except (TypeError, ValueError):
                raise ValueError(f"Valore non valido per --pad: {pad_val!r}")

    for flag, key in [("--min-gap", "min_gap"),
                      ("--tail-hangover", "tail_hangover"),
                      ("--max-tail-extension", "max_tail_extension"),
                      ("--min-duration", "min_duration"),
                      ("--max-duration", "max_duration"),
                      ("--min-confidence", "min_confidence"),
                      ("--limit-minutes", "limit_minutes")]:
        val = params.get(key)
        try:
            val = float(val) if val not in (None, "") else None
        except (TypeError, ValueError):
            raise ValueError(f"Valore non valido per {flag}: {val!r}")
        if val is not None and val > 0:
            cmd += [flag, str(val)]

    st = params.get("silence_threshold")
    if st not in (None, ""):
        try:
            st_val = float(st)
            cmd += ["--silence-threshold", str(st_val)]
        except (TypeError, ValueError):
            raise ValueError(f"Valore non valido per --silence-threshold: {st!r}")

    sound_type = params.get("sound_type", "laughter")
    if sound_type:
        cmd += ["--sound-type", sound_type]

    export_mode = params.get("export_mode", "both")
    if export_mode and export_mode in ("both", "events", "clean"):
        cmd += ["--export", export_mode]

    if params.get("safe"):
        cmd += ["--safe"]

    outdir = str(params.get("outdir", "")).strip()
    if outdir:
        p = Path(outdir)
        if not p.is_absolute():
            p = ROOT / p
        cmd += ["--outdir", str(p)]

    return cmd, labels_file


def _write_labels_file(path: Path, labels: list) -> None:
    def fmt(t: float) -> str:
        m = int(t // 60)
        return f"{m}:{t - m * 60:05.2f}"

    lines = ["# Etichette (inizio-fine di ogni evento; prefisso '-' per falsi positivi/negativi)\n"]
    for item in sorted(labels, key=lambda x: (abs(float(x[0])) if isinstance(x, (list, tuple)) else 0)):
        if isinstance(item, (list, tuple)):
            s, e = float(item[0]), float(item[1])
            is_neg = len(item) > 2 and item[2] == "neg"
            prefix = "-" if is_neg else ""
            lines.append(f"{prefix}{fmt(s)}-{fmt(e)}\n")
        elif isinstance(item, dict):
            s, e = float(item["start"]), float(item["end"])
            is_neg = item.get("type") == "neg"
            prefix = "-" if is_neg else ""
            lines.append(f"{prefix}{fmt(s)}-{fmt(e)}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def _reader(p: subprocess.Popen, sink) -> None:
    """Stream the subprocess stdout/stderr live (splitting on \\n AND \\r so
    progress bars shown with \\r arrive in real time)."""
    buf = bytearray()
    while True:
        chunk = os.read(p.stdout.fileno(), 65536)
        if not chunk:
            break
        buf += chunk
        while True:
            nl = buf.find(b"\n")
            cr = buf.find(b"\r")
            if nl == -1 and cr == -1:
                break
            idx = nl if nl != -1 and (cr == -1 or nl < cr) else cr
            line = bytes(buf[:idx]).decode("utf-8", errors="replace").rstrip()
            del buf[:idx + 1]
            if line.strip():
                sink(line)
    if buf and bytes(buf).strip():
        sink(bytes(buf).decode("utf-8", errors="replace").rstrip())


def start_run(params: dict) -> tuple[str, list[str]]:
    with ST.lock:
        if ST.running:
            raise ValueError("C'è già un'elaborazione in corso. Aspetta che "
                             "finisca o premi 'Stop'.")
    cmd, _ = build_command(params)
    ST.run_lines = []
    ST.done = False
    ST.exit_code = None
    ST.command = " ".join(str(c) for c in cmd)

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        [sys.executable, *cmd],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    with ST.lock:
        ST.proc = proc
        ST.running = True
    threading.Thread(target=_monitor, args=(proc,), daemon=True).start()
    return ST.command, ST.run_lines


def _monitor(proc: subprocess.Popen) -> None:
    def sink(line: str) -> None:
        with ST.lock:
            ST.run_lines.append(line)
    _reader(proc, sink)
    proc.wait()
    with ST.lock:
        ST.running = False
        ST.done = True
        ST.exit_code = proc.returncode


def stop_run() -> None:
    with ST.lock:
        proc = ST.proc
    if proc and proc.poll() is None:
        proc.terminate()
        with ST.lock:
            ST.running = False
            ST.done = True
            ST.exit_code = -15


# ---------------------------------------------------------------- page ---

PAGE = r"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>✂️ AutoCut — Interfaccia</title>
<style>
  :root { --bg:#101114; --panel:#191b20; --border:#2a2d35; --txt:#e8e8ea;
          --muted:#9aa0a8; --accent:#2a6df4; --green:#1e9e4a; --red:#d64541;
          --amber:#f4a72a; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI",
         Roboto, sans-serif; background:var(--bg); color:var(--txt); }
  header { padding:12px 24px; border-bottom:1px solid var(--border);
           display:flex; align-items:center; justify-content:space-between; gap:14px; flex-wrap:wrap; }
  header h1 { font-size:19px; margin:0; }
  header p { margin:0; color:var(--muted); font-size:12px; }
  .grid { display:grid; grid-template-columns: minmax(480px, 1.65fr) 315px minmax(380px, 1.15fr); gap:16px;
          padding:14px 20px 40px; width:100%; max-width:100%; margin:0; }
  @media (max-width: 1350px) {
    .grid { grid-template-columns: 1.2fr 1fr; }
    .col-results { grid-column: 1 / -1; }
  }
  @media (max-width: 860px) {
    .grid { grid-template-columns: 1fr; }
    .col-results { grid-column: auto; }
  }
  section { background:var(--panel); border:1px solid var(--border);
            border-radius:10px; padding:16px; }
  .col-results { display:flex; flex-direction:column; }
  .col-results ul.events { flex:1; min-height:300px; max-height:calc(100vh - 340px); overflow-y:auto; }
  .col-results ul.events li { padding:5px 8px; font-size:12px; gap:4px; }
  section h2 { margin:0 0 12px; font-size:15px; color:var(--muted);
               font-weight:600; letter-spacing:.4px; }
  video { width:100%; max-height:46vh; background:#000; border-radius:8px; }
  .time { font-size:20px; font-variant-numeric:tabular-nums; margin:8px 0;
          color:var(--amber); }
  label { font-size:13px; color:var(--muted); display:block; margin:8px 0 3px; }
  input[type=text], input[type=number], select, textarea {
    background:#101114; color:var(--txt); border:1px solid var(--border);
    border-radius:6px; padding:8px 10px; font-size:13px; width:100%;
  }
  textarea { font-family:ui-monospace, Menlo, monospace; min-height:96px;
             resize:vertical; }
  input[type=range] { width:100%; accent-color:var(--accent); }
  button { background:#2a3a55; color:var(--txt); border:1px solid var(--border);
           border-radius:6px; padding:8px 12px; font-size:13px; cursor:pointer; }
  button:hover { filter:brightness(1.15); }
  button:disabled { opacity:.4; cursor:not-allowed; }
  .btn-run { background:var(--green); color:#fff; font-size:15px;
             padding:10px 18px; font-weight:600; }
  .btn-stop { background:var(--red); color:#fff; }
  .btn-start { background:var(--green); color:#fff; }
  .btn-end { background:var(--red); color:#fff; }
  .btn-warn { background:var(--amber); color:#111; }
  .row { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
  .param { display:grid; grid-template-columns: 1fr 1fr; gap:0 14px; }
  .param .full { grid-column: 1 / -1; }
  kbd { background:#2a2d35; border-radius:4px; padding:1px 6px; font-size:12px;
        font-family:ui-monospace, Menlo, monospace; }
  ul.events { list-style:none; padding:0; margin:8px 0; max-height:220px;
              overflow:auto; }
  ul.events li { background:#101114; border:1px solid var(--border);
                 border-radius:6px; padding:6px 10px; margin:4px 0;
                 display:flex; justify-content:space-between; cursor:pointer;
                 font-size:13px; font-variant-numeric:tabular-nums; }
  ul.events li:hover { border-color:var(--accent); }
  ul.events li.excluded { opacity:0.32; filter:grayscale(0.85); background:#090a0d; }
  ul.events li.excluded:hover { opacity:0.6; }
  ul.events li.deleted {
    opacity: 0.42;
    background: #180d0f;
    border-color: rgba(239, 68, 68, 0.4);
  }
  ul.events li.deleted .ev-info {
    text-decoration: line-through;
    color: #94a3b8;
  }
  ul.events li.deleted .ev-info b, ul.events li.deleted .ev-info span {
    color: #64748b !important;
    text-decoration: line-through;
  }
  .btn-delete-event {
    background: rgba(239, 68, 68, 0.14);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-delete-event:hover {
    background: #dc2626;
    color: #fff;
    border-color: #ef4444;
  }
  .btn-delete-event.restore {
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border-color: rgba(56, 189, 248, 0.4);
  }
  .btn-delete-event.restore:hover {
    background: #0284c7;
    color: #fff;
  }
  #log { background:#0a0b0d; border:1px solid var(--border); border-radius:8px;
         padding:10px; font-family:ui-monospace, Menlo, monospace; font-size:12px;
         white-space:pre-wrap; word-break:break-word; height:230px; overflow:auto;
         color:#c9d1d9; margin-top:10px; }
  .hint { color:var(--muted); font-size:12px; line-height:1.7; }
  a.out { color:var(--accent); text-decoration:none; font-size:13px; }
  .badge { display:inline-block; background:#23262e; border:1px solid var(--border);
           border-radius:20px; padding:2px 10px; font-size:12px; color:var(--muted);
           margin:2px 4px 2px 0; cursor:pointer; }
  .badge:hover { border-color:var(--accent); }
  .mode-card { border:1px solid var(--border); border-radius:8px; padding:8px 10px;
               margin:6px 0; }
  .mode-card.selected { border-color:var(--accent); background:#141823; }
  .hidden { display:none; }
  #status { font-size:13px; color:var(--muted); margin:6px 0; }

  .btn-detect {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    color: #fff;
    font-weight: 700;
    padding: 10px 16px;
    font-size: 13px;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    display: inline-flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(2, 132, 199, 0.35);
    transition: all 0.15s;
  }
  .btn-detect:hover {
    background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.5);
    transform: translateY(-1px);
  }
  .btn-detect:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  .btn-scan {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: #fff;
    font-weight: 700;
    padding: 10px 16px;
    font-size: 13px;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    display: inline-flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.35);
    transition: all 0.15s;
  }
  .btn-scan:hover {
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.5);
    transform: translateY(-1px);
  }
  .btn-scan:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  .btn-convert-pos {
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.4);
    padding: 4px 9px;
    font-size: 11px;
    font-weight: 600;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-convert-pos:hover, .btn-convert-pos.active {
    background: var(--green);
    color: #fff;
    border-color: var(--green);
  }
  .btn-convert-neg {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
    padding: 4px 9px;
    font-size: 11px;
    font-weight: 600;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-convert-neg:hover, .btn-convert-neg.active {
    background: var(--red);
    color: #fff;
    border-color: var(--red);
  }

  .train-workflow-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(59, 130, 246, 0.35);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 10px 0;
  }
  .badge-pos {
    background: rgba(34, 197, 94, 0.2);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.4);
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
  }
  .badge-neg {
    background: rgba(239, 68, 68, 0.2);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
  }
  .spin { display:inline-block; animation:spin 1s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .wave-wrap { position: relative; margin: 10px 0 4px; }
  #wave { width: 100%; height: 80px; background: #0d0e11; border: 1px solid var(--border);
          border-radius: 8px; cursor: crosshair; display: block; }
  #waveHint { font-size: 11px; color: var(--muted); margin: 4px 0 8px; }
</style>
</head>
<body>
<header>
  <h1>✂️ AutoCut</h1>
  <p>Rileva eventi sonori e genera automaticamente tagli del video: un cut con solo gli eventi e uno pulito senza eventi. Server locale
     (127.0.0.1) — chiudi con Ctrl+C nel terminale.</p>
</header>

<div class="grid">

  <!-- ============ COLONNA 1 (SINISTRA): video + etichette ============ -->
  <section class="col-video">
    <h2>1 · VIDEO SORGENTE</h2>
    <div class="row">
      <select id="videos" style="flex:2; min-width:200px">
        <option value="">— scegli un video dalla cartella —</option>
      </select>
      <button id="bRefresh" title="ricarica l'elenco">🔄</button>
    </div>
    <label>oppure scrivi un percorso (anche assoluto)</label>
    <div class="row">
      <input type="text" id="videoPath" placeholder="es. /Users/me/Video/clip.mp4">
      <button id="bLoadPath">Carica</button>
    </div>

    <video id="v" controls preload="auto"></video>
    <div class="time" id="pTime">—</div>

    <div class="wave-wrap">
      <canvas id="wave"></canvas>
      <div id="waveHint" class="hint">caricamento traccia audio…</div>
    </div>
    <div class="row">
      <button id="bFPrev" title="un frame indietro ([)">◀◀ frame</button>
      <button id="bFNext" title="un frame avanti (])">frame ▶▶</button>
      <select id="fpsSel" title="fotogrammi al secondo usati dallo step di un frame">
        <option value="auto" selected>fps auto</option>
        <option>24</option><option>25</option><option>30</option><option>60</option>
      </select>
      <button id="bPlayIso" title="Riascolta l'intervallo selezionato o ultimo in modo isolato (R)" style="font-weight:600; color:#60a5fa">▶ Riascolta isolato <kbd>R</kbd></button>
      <button id="bLoop" title="riproduci in loop l'ultimo intervallo o quello selezionato">🔁 Ripeti loop</button>
    </div>

    <h2 style="margin-top:14px">2 · ETICHETTE MANUALI <span style="font-weight:400; font-size:12px">
      (modificabili con le maniglie sulla waveform)</span></h2>
    <div class="row">
      <button class="btn-start" id="bStart" title="Marca inizio evento positivo (S)">▶ Inizio [+] <kbd>S</kbd></button>
      <button style="background:var(--red);color:#fff" id="bStartNeg" title="Marca inizio falso positivo da escludere (N)">▶ Inizio [-] <kbd>N</kbd></button>
      <button class="btn-end" id="bEnd" title="Marca fine intervallo (E)">⏹ Fine <kbd>E</kbd></button>
      <button id="bUndo">↩ Annulla <kbd>Z</kbd></button>
      <button id="bClear">🗑 Svuota</button>
      <button class="btn-warn" id="bSave">💾 Salva</button>
    </div>
    <p class="hint" style="font-size:11px; margin:6px 0"><kbd>S</kbd> = inizio [+] · <kbd>N</kbd> = inizio [-] · <kbd>E</kbd> = fine · <kbd>R</kbd> = riascolta isolato.<br>
      Clicca un'etichetta per riascoltarla isolata mantenendo lo zoom corrente. Trascina i bordi per rifinire.</p>
    <ul class="events" id="labelList" style="max-height:220px"></ul>
  </section>

  <!-- ============ COLONNA 2 (CENTRALE): RISULTATI TAGLIO / RILEVAMENTO AUTOMATICO ============ -->
  <section class="col-results">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:8px">
      <h2 style="margin:0">3 · RISULTATI TAGLIO AUTOMATICO</h2>
      <div id="lblCountBadge" style="font-size:11px; color:#94a3b8; font-family:ui-monospace, monospace"></div>
    </div>
    <div id="summary" class="hint">Esegui un rilevamento per vedere qui gli eventi trovati in tempo reale.</div>

    <!-- Filtro confidenza e soglia minima di taglio (SEMPRE VISIBILE) -->
    <div id="confFilterBox" style="background:#0f172a; border:1px solid var(--border); border-radius:8px; padding:10px 12px; margin:8px 0 10px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px">
        <label style="margin:0; font-size:12px; font-weight:600; color:var(--txt)">🎯 Soglia Precisione Minima: <b id="minConfLabel" style="color:var(--amber)">Tutti (0%)</b></label>
        <span id="filteredStats" style="font-size:11px; color:#38bdf8; font-weight:600">Tutti gli eventi</span>
      </div>
      <input type="range" id="minConfSlider" min="0" max="95" step="5" value="0" style="width:100%; margin:4px 0">
      <div style="display:flex; justify-content:space-between; font-size:10px; color:#64748b">
        <span>0% (Tutti)</span>
        <span>70% (Solo medi & certi)</span>
        <span>85% (Solo certissimi)</span>
      </div>
      <div id="col2Actions" style="display:none; justify-content:space-between; align-items:center; margin-top:8px; padding-top:6px; border-top:1px solid rgba(255,255,255,0.08)">
        <button id="bRestoreAll" style="display:none; font-size:11px; padding:4px 9px; background:#1e293b; color:#38bdf8; border:1px solid #334155; border-radius:4px; cursor:pointer" title="Ripristina tutti gli eventi precedentemente eliminati">↩️ Ripristina eliminati</button>
        <button id="bRunCol2" class="btn-run" style="font-size:12px; padding:5px 14px; font-weight:700; margin-left:auto" title="Esporta direttamente il video contenente solo gli eventi approvati sopra">✂️ Esporta Video</button>
      </div>
    </div>

    <div style="margin:8px 0 6px; font-size:12px; color:#94a3b8; line-height:1.4">
      Clicca un evento per ascoltare e zoomare. Usa <b>🗑️ Elimina</b> per toglierlo dal video finale <code>_events_only.mp4</code>, <b>✏️ [+]</b> per rifinirlo, o <b>🚫 [-]</b> per falsi positivi.
    </div>
    <ul class="events" id="events"></ul>
    <div id="outputs" style="margin-top:10px"></div>
    <video id="outV" controls preload="none" style="display:none; margin-top:10px; width:100%"></video>
  </section>

  <!-- ============ COLONNA 3 (DESTRA): RILEVAMENTO, PARAMETRI & LOG ============ -->
  <section class="col-controls">
    <h2>4 · PARAMETRI & ADDESTRAMENTO</h2>

    <div class="mode-card selected" data-mode="yamnet">
      <b>Modello di default (YAMNet)</b>
      <div class="hint">Pretrained generico. Usa le altre modalità se non trova nulla.</div>
    </div>
    <div class="mode-card" data-mode="teach">
      <b>Impara dagli esempi (etichette + e -)</b>
      <div class="hint">Usa le etichette marcate a sinistra (es. <code>1:54-1:56</code> per positivo, <code>-1:57-1:58</code> per escludere).</div>
      <textarea id="teachLabels" placeholder="1:54.55-1:55.20&#10;-1:57.10-1:57.62"></textarea>
      <p class="hint" style="font-size:11px">Prefisso <code>-</code> per i negativi. File: <code>&lt;video&gt;_etichette.txt</code>.</p>
    </div>
    <div class="mode-card" data-mode="train">
      <b>Impara da un video di riferimento</b>
      <div class="hint">Video contenente solo i suoni target (es. cut <code>*_events_only.mp4</code>).</div>
      <select id="refFiles" style="margin-top:4px">
        <option value="">— scegli il riferimento —</option>
      </select>
    </div>
    <div class="mode-card" data-mode="model">
      <b>Usa un modello già salvato</b>
      <div class="hint">Riusa un modello <code>.npz</code> imparato in precedenza.</div>
      <select id="modelFiles" style="margin-top:4px">
        <option value="">— scegli il modello —</option>
      </select>
    </div>

    <label>Salva il modello appreso (facoltativo)</label>
    <input type="text" id="saveModel" placeholder="es. mio_modello.npz">

    <div class="param">
      <div id="soundTypeContainer">
        <label>Tipo di suono (modello YAMNet)</label>
        <select id="soundType">
          <option value="laughter" selected>😄 Risata (Laughter)</option>
          <option value="applause">👏 Applauso (Applause)</option>
          <option value="speech">🗣️ Voce / Parlato (Speech)</option>
          <option value="cough">😷 Tosse / Gola (Cough)</option>
        </select>
      </div>
      <div>
        <div style="display:flex;justify-content:space-between;align-items:center">
          <label style="margin:0">Soglia: <b id="thVal">0.50</b></label>
          <label style="font-size:11px;cursor:pointer;display:flex;align-items:center;gap:4px;color:var(--amber)" title="Calcola automaticamente la soglia ideale in base ai positivi e negativi">
            <input type="checkbox" id="autoThresh" style="margin:0;width:auto"> 🎯 Auto
          </label>
        </div>
        <input type="range" id="threshold" min="0.05" max="0.95" step="0.05" value="0.50">
      </div>
      <div>
        <label>Contesto / Pad (s)</label>
        <input type="number" id="pad" value="0.5" min="0" step="0.1">
      </div>
      <div>
        <label>Coda / Hangover (s)</label>
        <input type="number" id="tailHangover" value="0.35" min="0" step="0.05" title="Buffer acustico naturale per non troncare la conclusione/decadimento del suono">
      </div>
      <div>
        <label>Unisci se vicini &lt; (s)</label>
        <input type="number" id="minGap" value="1.0" min="0" step="0.1">
      </div>
      <div>
        <label>Scarta se &lt; (s)</label>
        <input type="number" id="minDur" value="0.3" min="0" step="0.05">
      </div>
      <div>
        <label>Scarta se &gt; (s)</label>
        <input type="number" id="maxDur" placeholder="nessun limite" min="0.1" step="0.1">
      </div>
      <div>
        <label>Filtro silenzio (dB)</label>
        <input type="number" id="silenceThresh" placeholder="es. -40" step="1" max="0" min="-100">
      </div>
      <div>
        <label>Primi N minuti (test)</label>
        <input type="number" id="limitMin" placeholder="tutto il video" min="0.1" step="0.5">
      </div>
      <div>
        <label>Cartella output</label>
        <input type="text" id="outdir" placeholder="vuoto = accanto al video">
      </div>
      <div>
        <label>Cosa esportare</label>
        <select id="exportMode">
          <option value="both" selected>🎬 Entrambi (Eventi + Senza eventi)</option>
          <option value="events">🎯 Solo gli eventi (_events_only.mp4)</option>
          <option value="clean">✨ Solo senza eventi (_clean.mp4)</option>
        </select>
      </div>
    </div>

    <label style="margin-top:8px">
      <input type="checkbox" id="safe"> <b>Modalità safe</b> (CPU cooler)
    </label>

    <div class="train-workflow-card">
      <div style="font-weight:700; color:#93c5fd; margin-bottom:4px">🎯 Workflow di Addestramento (4 Fasi):</div>
      <ol style="margin:4px 0 0 16px; padding:0; font-size:11px; color:#cbd5e1; line-height:1.4">
        <li><b>Fase 1:</b> Marca ~10 eventi sulla waveform e stringi i bordi sul suono.</li>
        <li><b>Fase 2:</b> Clicca <b>🧠 Allena & Scansiona</b> e salva il modello <code>.npz</code>.</li>
        <li><b>Fase 3:</b> Nella colonna centrale usa <b>✏️ Come [+]</b> o <b>🚫 Come [-]</b>.</li>
        <li><b>Fase 4:</b> Riallena in 2s. Quando è perfetto, clicca <b>✂️ Esporta Video</b>!</li>
      </ol>
    </div>

    <div class="row" style="margin-top:10px; gap:6px; flex-wrap:wrap">
      <button class="btn-detect" id="bDetectOnly" title="Scansiona il video istantaneamente usando il modello attuale o YAMNet, senza riallenare e senza esportare tagli video (veloce, 2s)">🔍 Scansiona (Senza Allenare)</button>
      <button class="btn-scan" id="bScanOnly" title="Addestra il modello e scansiona la traccia audio in 2-3 secondi, senza attendere il rendering video">🧠 Allena Modello & Scansiona (2-3s)</button>
      <button class="btn-run" id="bRun" title="Esegue il rilevamento e ritaglia i file video scelti (eventi, senza eventi o entrambi)">✂️ Esporta Video Tagliato</button>
      <button class="btn-stop" id="bStop" disabled>⏹ Stop</button>
    </div>
    <div id="status">Nessuna elaborazione in corso.</div>
    <code id="cmd" style="font-size:11px; color:var(--muted)"></code>
    <div id="log" style="height:170px"></div>
  </section>
</div>

<script>
"use strict";
const $ = id => document.getElementById(id);
const v = $('v');
let labels = [];            // [[s, e, 'pos'|'neg'], ...]
let pollTimer = null;
let since = 0;
let openType = 'pos';

/* ---------- formattazione tempi ---------- */
function fmt(t) {
  t = Math.max(0, t || 0);
  const m = Math.floor(t / 60);
  return m + ':' + (t - m * 60).toFixed(2).padStart(5, '0');
}
function parseOne(tok) {
  tok = String(tok).trim();
  let m = tok.match(/^(\d+):(\d+(?:\.\d+)?)$/);
  if (m) return parseInt(m[1], 10) * 60 + parseFloat(m[2]);
  m = tok.match(/^(\d+(?:\.\d+)?)$/);
  if (m) return parseFloat(tok);
  return null;
}
function parseLabelLine(line) {
  line = line.trim();
  if (!line || line.startsWith('#')) return null;
  let isNeg = false;
  if (line.startsWith('!') || line.startsWith('-') || line.startsWith('neg:')) {
    isNeg = true;
    line = line.replace(/^[!\-]|^(neg:)/, '').trim();
  } else if (line.startsWith('+')) {
    line = line.substring(1).trim();
  }
  const parts = line.split('-').map(parseOne);
  if (parts.length === 2 && parts[0] !== null && parts[1] !== null) {
    const a = Math.min(parts[0], parts[1]), b = Math.max(parts[0], parts[1]);
    if (b - a >= 0.05) return [a, b, isNeg ? 'neg' : 'pos'];
  }
  return null;
}
function labelsText() {
  return labels.map(l => (l[2] === 'neg' ? '-' : '') + fmt(l[0]) + '-' + fmt(l[1])).join('\n');
}

/* ---------- elenco file ---------- */
function loadFiles() {
  fetch('/api/files').then(r => r.json()).then(d => {
    const sel = $('videos');
    const cur = sel.value;
    sel.innerHTML = '<option value="">— scegli un video dalla cartella —</option>' +
      d.videos.map(p => '<option value="' + p.replace(/&/g,'&amp;').replace(/"/g,'&quot;') + '">' +
        esc(p.replace(/^.*\//, '')) + '</option>').join('');
    if (cur) sel.value = cur;
    const ref = $('refFiles');
    ref.innerHTML = '<option value="">— scegli il riferimento —</option>' +
      d.videos.map(p => '<option value="' + p.replace(/&/g,'&amp;').replace(/"/g,'&quot;') + '">' +
        esc(p.replace(/^.*\//, '')) + '</option>').join('');
    const mdl = $('modelFiles');
    mdl.innerHTML = '<option value="">— scegli il modello —</option>' +
      d.models.map(p => '<option value="' + p.replace(/&/g,'&amp;').replace(/"/g,'&quot;') + '">' +
        esc(p.replace(/^.*\//, '')) + '</option>').join('');
    mdl.onchange = () => {
      const m = mdl.value;
      if (m && !$('saveModel').value.trim()) {
        $('saveModel').value = m.replace(/^.*\//, '');
      }
    };
  }).catch(err => console.error('files', err));
}

/* ---------- selezione video ---------- */
function selectVideo(path) {
  if (!path) return;
  fetch('/api/select', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({path: path})
  }).then(r => r.json()).then(d => {
    if (d.error) { alert(d.error); return; }
    v.src = '/video?' + Date.now();
    labels = d.labels || [];
    $('teachLabels').value = labelsText();
    refreshLabels();
    currentEvents = [];
    $('events').innerHTML = '';
    $('summary').textContent = 'Esegui un rilevamento per vedere qui gli eventi trovati in tempo reale.';
    if ($('col2Actions')) $('col2Actions').style.display = 'none';
    if ($('lblCountBadge')) $('lblCountBadge').textContent = '';
    if (window.wave && window.wave.setResults) window.wave.setResults([]);
    if ($('bRun')) $('bRun').textContent = '✂️ Esporta Video Tagliato';
    window.waveUrl = '/api/wave?path=' + encodeURIComponent(path);
    if (window.loadWave) window.loadWave();
    $('status').textContent = 'Video caricato: ' + d.name +
      ' (' + d.size_mb + ' MB). Ora scegli la modalità e premi Taglia.';
  }).catch(err => alert('Errore: ' + err));
}
$('bRefresh').onclick = loadFiles;
$('bLoadPath').onclick = () => selectVideo($('videoPath').value.trim());
$('videos').onchange = () => { const p = $('videos').value; if (p) selectVideo(p); };

/* ---------- etichette ---------- */
let selectedLabelIndex = -1;

function refreshLabelsHighlight() {
  const ul = $('labelList');
  if (!ul) return;
  const items = ul.querySelectorAll('li');
  items.forEach((li, idx) => {
    const isNeg = labels[idx] && labels[idx][2] === 'neg';
    const isSelected = (selectedLabelIndex === idx);
    li.style.borderColor = isSelected ? '#3b82f6' : (isNeg ? 'rgba(239, 68, 68, 0.4)' : 'rgba(34, 197, 94, 0.4)');
    li.style.background = isSelected ? 'rgba(59, 130, 246, 0.16)' : (isNeg ? 'rgba(239, 68, 68, 0.06)' : 'rgba(34, 197, 94, 0.06)');
    li.style.boxShadow = isSelected ? '0 0 0 1px #3b82f6' : 'none';
  });
}

function refreshLabels() {
  const ul = $('labelList');
  ul.innerHTML = '';
  const posCount = labels.filter(l => l[2] !== 'neg').length;
  const negCount = labels.filter(l => l[2] === 'neg').length;
  const badgeInfo = $('lblCountBadge');
  if (badgeInfo) {
    badgeInfo.innerHTML = 'Etichette attive: <b style="color:var(--green)">' + posCount + ' [+]</b> &nbsp;|&nbsp; <b style="color:var(--red)">' + negCount + ' [-]</b>';
  }

  labels.forEach((it, i) => {
    const isNeg = it[2] === 'neg';
    const isSelected = (selectedLabelIndex === i);
    const li = document.createElement('li');
    li.style.display = 'flex';
    li.style.alignItems = 'center';
    li.style.justifyContent = 'space-between';
    li.style.borderColor = isSelected ? '#3b82f6' : (isNeg ? 'rgba(239, 68, 68, 0.4)' : 'rgba(34, 197, 94, 0.4)');
    li.style.background = isSelected ? 'rgba(59, 130, 246, 0.16)' : (isNeg ? 'rgba(239, 68, 68, 0.06)' : 'rgba(34, 197, 94, 0.06)');
    li.style.boxShadow = isSelected ? '0 0 0 1px #3b82f6' : 'none';
    li.style.transition = 'all 0.15s';
    li.style.cursor = 'pointer';

    li.innerHTML = '<div style="display:flex;align-items:center;gap:8px;flex:1">' +
      '<span class="' + (isNeg ? 'badge-neg' : 'badge-pos') + '">' + (isNeg ? '[-] Negativo' : '[+] Positivo') + '</span>' +
      '<b>#' + (i + 1) + '</b> ' + fmt(it[0]) + ' → ' + fmt(it[1]) +
      ' <span style="color:#888">(' + (it[1] - it[0]).toFixed(2) + 's)</span>' +
      '</div>' +
      '<div style="display:flex;align-items:center;gap:4px">' +
      '<button class="btn-play-iso" title="Ascolta solo questo intervallo (' + (it[1] - it[0]).toFixed(2) + 's)" style="background:#1e293b;border:1px solid #3b82f6;color:#60a5fa;border-radius:4px;padding:2px 7px;cursor:pointer;font-size:11px;font-weight:600">▶ Ascolta</button>' +
      '<button class="btn-loop-iso" title="Ripeti in loop continuo questo intervallo" style="background:#1e293b;border:1px solid #475569;color:#e2e8f0;border-radius:4px;padding:2px 6px;cursor:pointer;font-size:11px">🔁</button>' +
      '<button class="btn-toggle" title="Inverti tipo (+/-)" style="background:#2a2d35;border:1px solid var(--border);border-radius:4px;padding:2px 6px;cursor:pointer;font-size:11px">⇄</button>' +
      '<button class="del" title="Elimina etichetta" style="color:var(--red);background:none;border:0;cursor:pointer;font-size:16px;padding:0 4px">×</button>' +
      '</div>';

    li.onclick = e => {
      if (e.target.closest('.del')) {
        labels.splice(i, 1);
        if (selectedLabelIndex === i) selectedLabelIndex = -1;
        else if (selectedLabelIndex > i) selectedLabelIndex--;
        $('teachLabels').value = labelsText();
        refreshLabels();
        return;
      }
      if (e.target.closest('.btn-toggle')) {
        labels[i][2] = (labels[i][2] === 'neg' ? 'pos' : 'neg');
        $('teachLabels').value = labelsText();
        refreshLabels();
        return;
      }
      if (e.target.closest('.btn-loop-iso')) {
        selectedLabelIndex = i;
        refreshLabelsHighlight();
        if (window.wave && window.wave.focusInterval) {
          window.wave.focusInterval(it[0], it[1]);
        }
        var isCurrentlyLooping = window.wave && window.wave.isLooping && window.wave.isLooping();
        var currentLoop = window.wave && window.wave.loopInterval && window.wave.loopInterval();
        if (isCurrentlyLooping && currentLoop && Math.abs(currentLoop[0] - it[0]) < 0.05) {
          if (window.wave && window.wave.stopLoop) window.wave.stopLoop();
        } else {
          if (window.wave && window.wave.playInterval) {
            window.wave.playInterval(it[0], it[1], true);
          }
        }
        return;
      }

      // Default click (sulla riga o sul pulsante ▶ Ascolta): focus mantenendo lo zoom impostato ed ascolto isolato!
      selectedLabelIndex = i;
      refreshLabelsHighlight();
      if (window.wave && window.wave.focusInterval) {
        window.wave.focusInterval(it[0], it[1]);
      }
      if (window.wave && window.wave.playInterval) {
        window.wave.playInterval(it[0], it[1], false);
      } else {
        v.currentTime = it[0];
        v.play();
      }
    };
    ul.appendChild(li);
  });
  $('teachLabels').value = labelsText();
  if (window.redrawWave) redrawWave();
}

let openStart = null;
window.wave = {
  video: () => v,
  labels: () => labels,
  selectedLabel: () => (selectedLabelIndex >= 0 && selectedLabelIndex < labels.length ? labels[selectedLabelIndex] : (labels.length ? labels[labels.length - 1] : null)),
  setSelectedLabelIndex: (idx) => {
    selectedLabelIndex = idx;
    refreshLabelsHighlight();
  },
  refresh: () => refreshLabels(),
  add: (s, e, type) => {
    labels.push([s, e, type || 'pos']);
    labels.sort((a, b) => a[0] - b[0]);
    selectedLabelIndex = labels.findIndex(l => Math.abs(l[0] - s) < 0.001 && Math.abs(l[1] - e) < 0.001);
    refreshLabels();
  },
  openStart: () => openStart,
};

function markStartPos() {
  openType = 'pos';
  openStart = v.currentTime;
  $('v').style.borderColor = 'var(--green)';
  if (window.redrawWave) redrawWave();
}
function markStartNeg() {
  openType = 'neg';
  openStart = v.currentTime;
  $('v').style.borderColor = 'var(--red)';
  if (window.redrawWave) redrawWave();
}
function markEnd() {
  if (openStart === null) return;
  const e = v.currentTime;
  if (e - openStart >= 0.05) { labels.push([openStart, e, openType]); }
  openStart = null; $('v').style.borderColor = '';
  labels.sort((a, b) => a[0] - b[0]);
  refreshLabels();
}
function undoLabel() {
  if (openStart !== null) openStart = null;
  else labels.pop();
  $('v').style.borderColor = '';
  refreshLabels();
}

$('bStart').onclick = markStartPos;
$('bStartNeg').onclick = markStartNeg;
$('bEnd').onclick = markEnd;
$('bUndo').onclick = undoLabel;
$('bClear').onclick = () => { labels = []; openStart = null; refreshLabels(); };
$('bSave').onclick = saveLabels;
function saveLabels() {
  const video = currentVideo();
  if (!video) { alert('Prima scegli il video.'); return; }
  fetch('/api/labels', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({path: video, labels: labels})
  }).then(r => r.text()).then(msg => {
    $('status').textContent = '✅ ' + msg;
  });
}
$('teachLabels').oninput = e => {
  const out = [];
  e.target.value.split(/[\n,]/).forEach(line => {
    const it = parseLabelLine(line);
    if (it) out.push(it);
  });
  labels = out;
  refreshLabelListOnly();
};
function refreshLabelListOnly() {
  refreshLabels();
}
function currentVideo() {
  const s = $('videos').value;
  if (s) return s;
  return $('videoPath').value.trim();
}
document.addEventListener('keydown', e => {
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); saveLabels(); return; }
  const step = e.shiftKey ? 1 : 0.1;
  switch (e.key) {
    case ' ': e.preventDefault(); v.paused ? v.play() : v.pause(); break;
    case 's': case 'S': markStartPos(); break;
    case 'n': case 'N': markStartNeg(); break;
    case 'e': case 'E': markEnd(); break;
    case 'z': case 'Z': undoLabel(); break;
    case 'r': case 'R': {
      e.preventDefault();
      const target = (window.wave && window.wave.selectedLabel) ? window.wave.selectedLabel() : (labels.length ? labels[labels.length - 1] : null);
      if (target && window.wave && window.wave.playInterval) {
        const s = typeof target[0] === 'number' ? Math.abs(target[0]) : (target.start || 0);
        const e = typeof target[1] === 'number' ? Math.abs(target[1]) : (target.end || 0);
        window.wave.playInterval(s, e, false);
      }
      break;
    }
    case 'ArrowLeft': e.preventDefault(); v.currentTime = Math.max(0, v.currentTime - step); break;
    case 'ArrowRight': e.preventDefault();
      v.currentTime = Math.min(v.duration || 1e9, v.currentTime + step); break;
    case 'Delete':
    case 'Backspace':
    case 'x': case 'X': {
      if (window.wave && window.wave.getCurrentEventIndex && currentEvents && currentEvents.length) {
        const cIdx = window.wave.getCurrentEventIndex();
        if (cIdx >= 0 && cIdx < currentEvents.length) {
          e.preventDefault();
          toggleDeleteEvent(currentEvents[cIdx].idx);
        }
      }
      break;
    }
  }
});
v.addEventListener('timeupdate', () => {
  $('pTime').textContent = fmt(v.currentTime || 0) + ' / ' + fmt(v.duration || 0);
});
v.addEventListener('loadedmetadata', () => {
  $('pTime').textContent = '0:00.00 / ' + fmt(v.duration || 0);
  if (pollTimer) loadLabelsForCurrent();
});

/* ---------- modalità ---------- */
let mode = 'yamnet';
function updateModeUI() {
  const isYamnet = (mode === 'yamnet');
  const stc = $('soundTypeContainer');
  if (stc) stc.style.display = isYamnet ? 'block' : 'none';
}

document.querySelectorAll('.mode-card').forEach(card => {
  card.onclick = () => {
    document.querySelectorAll('.mode-card').forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
    mode = card.dataset.mode;
    updateModeUI();
  };
});
updateModeUI();
$('autoThresh').onchange = e => {
  if (e.target.checked) {
    $('threshold').disabled = true;
    $('thVal').textContent = 'Auto (🎯)';
  } else {
    $('threshold').disabled = false;
    $('thVal').textContent = $('threshold').value;
  }
};
$('threshold').oninput = e => {
  $('thVal').textContent = e.target.value;
  $('autoThresh').checked = false;
  $('threshold').disabled = false;
};

/* ---------- esecuzione ---------- */
let currentEvents = []; // [{start, end, idx, dur, conf, deleted, excluded}]

function collectParams() {
  const p = {
    video: currentVideo(),
    mode: mode,
    teach_labels: labels,
    reference: $('refFiles').value,
    model: $('modelFiles').value,
    save_model: $('saveModel').value.trim(),
    auto_threshold: $('autoThresh').checked,
    threshold: parseFloat($('threshold').value),
    min_confidence: ($('minConfSlider') && parseFloat($('minConfSlider').value) > 0) ? (parseFloat($('minConfSlider').value) / 100.0) : 0.0,
    sound_type: $('soundType').value,
    pad: parseFloat($('pad').value),
    tail_hangover: $('tailHangover') ? parseFloat($('tailHangover').value) : 0.35,
    min_gap: parseFloat($('minGap').value),
    min_duration: parseFloat($('minDur').value),
    max_duration: $('maxDur').value ? parseFloat($('maxDur').value) : null,
    silence_threshold: $('silenceThresh').value ? parseFloat($('silenceThresh').value) : null,
    limit_minutes: $('limitMin').value ? parseFloat($('limitMin').value) : 0,
    export_mode: $('exportMode') ? $('exportMode').value : 'both',
    safe: $('safe').checked,
    outdir: $('outdir').value.trim(),
  };
  if (currentEvents && currentEvents.length > 0) {
    const active = currentEvents.filter(ev => !ev.deleted && !ev.excluded);
    p.approved_events = active.map(ev => [ev.start, ev.end]);
  }
  return p;
}
let renderedDuringRun = false;

function startRun(noCut, noTrain) {
  const params = collectParams();
  params.no_cut = !!noCut;
  params.no_train = !!noTrain;
  if (!params.video) { alert('Prima scegli il video sorgente.'); return; }
  if (!noCut && params.approved_events !== undefined && params.approved_events.length === 0) {
    alert('Tutti gli eventi sono stati eliminati o esclusi dalla soglia. Nessun evento da esportare nel video _only.');
    return;
  }
  $('log').textContent = '';
  if (noCut) {
    $('events').innerHTML = '';
    currentEvents = [];
    if ($('col2Actions')) $('col2Actions').style.display = 'none';
  }
  $('outputs').innerHTML = '';
  $('outV').style.display = 'none';
  renderedDuringRun = false;
  since = 0;
  fetch('/api/run', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(params)
  }).then(r => r.json()).then(d => {
    if (d.error) { $('log').textContent = '❌ ' + d.error; $('status').textContent = 'Errore: ' + d.error; return; }
    $('cmd').textContent = d.command;
    let stMsg = 'Elaborazione e taglio video in corso…';
    if (noTrain) stMsg = 'Scansione rapida in corso (senza allenare)…';
    else if (noCut) stMsg = 'Addestramento e scansione rapida in corso…';
    else {
      const exp = params.export_mode || 'both';
      const evCount = params.approved_events ? params.approved_events.length : null;
      const countMsg = evCount !== null ? ` (${evCount} eventi approvati)` : '';
      if (exp === 'events') stMsg = `Esportazione solo eventi in corso${countMsg}…`;
      else if (exp === 'clean') stMsg = `Esportazione solo video pulito in corso${countMsg}…`;
      else stMsg = `Esportazione tagli in corso${countMsg}…`;
    }
    $('status').innerHTML = '<span class="spin">⏳</span> ' + stMsg;
    $('bRun').disabled = true;
    if ($('bRunCol2')) $('bRunCol2').disabled = true;
    if ($('bScanOnly')) $('bScanOnly').disabled = true;
    if ($('bDetectOnly')) $('bDetectOnly').disabled = true;
    $('bStop').disabled = false;
    pollTimer = setInterval(poll, 600);
  }).catch(err => alert('Errore: ' + err));
}
$('bRun').onclick = () => startRun(false, false);
if ($('bRunCol2')) $('bRunCol2').onclick = () => startRun(false, false);
if ($('bScanOnly')) $('bScanOnly').onclick = () => startRun(true, false);
if ($('bDetectOnly')) $('bDetectOnly').onclick = () => startRun(true, true);

$('bStop').onclick = () => {
  fetch('/api/stop', {method: 'POST'}).then(() => {});
};
function poll() {
  fetch('/api/status?since=' + since).then(r => r.json()).then(d => {
    if (d.lines && d.lines.length) {
      since += d.lines.length;
      d.lines.forEach(l => appendLog(l));
      if (!renderedDuringRun && $('log').textContent.includes('Detected ') && ($('log').textContent.includes('duration:') || $('log').textContent.includes('Burp total:'))) {
        renderResults();
        renderedDuringRun = true;
      }
    }
    if (d.running) return;
    clearInterval(pollTimer); pollTimer = null;
    $('bRun').disabled = false;
    if ($('bRunCol2')) $('bRunCol2').disabled = false;
    if ($('bScanOnly')) $('bScanOnly').disabled = false;
    if ($('bDetectOnly')) $('bDetectOnly').disabled = false;
    $('bStop').disabled = true;
    if (d.done) {
      $('status').textContent = d.exit_code === 0
        ? '✅ Elaborazione completata.'
        : (d.exit_code === -15 ? "⏹ Interrotto dall'utente." : '❌ Terminato con errore (' + d.exit_code + ').');
      renderResults();
    }
  }).catch(() => {});
}
function appendLog(line) {
  const log = $('log');
  log.textContent += line + '\n';
  log.scrollTop = log.scrollHeight;
}

function updateResultsSummaryAndWave() {
  const activeEvents = currentEvents.filter(ev => !ev.deleted && !ev.excluded);
  const deletedEvents = currentEvents.filter(ev => ev.deleted);
  const totalCount = currentEvents.length;
  const activeCount = activeEvents.length;
  const activeDur = activeEvents.reduce((acc, ev) => acc + (ev.dur || (ev.end - ev.start)), 0);

  if (totalCount > 0) {
    let statText = `${activeCount}/${totalCount} inclusi (${activeDur.toFixed(1)}s)`;
    if (deletedEvents.length > 0) {
      statText += ` · ${deletedEvents.length} eliminat${deletedEvents.length === 1 ? 'o' : 'i'}`;
    }
    $('filteredStats').textContent = statText;
    if ($('lblCountBadge')) {
      $('lblCountBadge').textContent = statText;
    }
    if ($('col2Actions')) {
      $('col2Actions').style.display = 'flex';
    }
    if ($('bRunCol2')) {
      $('bRunCol2').textContent = `✂️ Esporta Video (${activeCount} event${activeCount === 1 ? 'o' : 'i'})`;
      $('bRunCol2').disabled = (activeCount === 0);
    }
    if ($('bRun')) {
      $('bRun').textContent = `✂️ Esporta Video Tagliato (${activeCount} event${activeCount === 1 ? 'o' : 'i'})`;
    }
    const restoreBtn = $('bRestoreAll');
    if (restoreBtn) {
      restoreBtn.style.display = deletedEvents.length > 0 ? 'inline-block' : 'none';
      restoreBtn.textContent = `↩️ Ripristina ${deletedEvents.length} eliminat${deletedEvents.length === 1 ? 'o' : 'i'}`;
    }
  }

  if (window.wave && window.wave.setResults) {
    window.wave.setResults(activeEvents);
  }
}

function toggleDeleteEvent(idx) {
  const ev = currentEvents.find(e => e.idx === idx);
  if (!ev) return;
  ev.deleted = !ev.deleted;
  const li = $('events').querySelector(`li[data-idx="${idx}"]`);
  if (li) {
    const bDel = li.querySelector('.btn-delete-event');
    const bPos = li.querySelector('.btn-convert-pos');
    const bNeg = li.querySelector('.btn-convert-neg');
    if (ev.deleted) {
      li.classList.add('deleted');
      if (bDel) {
        bDel.textContent = '↩️ Ripristina';
        bDel.classList.add('restore');
        bDel.title = 'Ripristina questo evento nei tagli';
      }
      if (bPos) bPos.style.display = 'none';
      if (bNeg) bNeg.style.display = 'none';
    } else {
      li.classList.remove('deleted');
      if (bDel) {
        bDel.textContent = '🗑️ Elimina';
        bDel.classList.remove('restore');
        bDel.title = 'Elimina questo evento (non apparirà nel video finale _events_only.mp4)';
      }
      if (bPos) bPos.style.display = '';
      if (bNeg) bNeg.style.display = '';
    }
  }
  updateResultsSummaryAndWave();
}

function restoreAllEvents() {
  currentEvents.forEach(ev => {
    if (ev.deleted) {
      ev.deleted = false;
      const li = $('events').querySelector(`li[data-idx="${ev.idx}"]`);
      if (li) {
        li.classList.remove('deleted');
        const bDel = li.querySelector('.btn-delete-event');
        if (bDel) {
          bDel.textContent = '🗑️ Elimina';
          bDel.classList.remove('restore');
          bDel.title = 'Elimina questo evento (non apparirà nel video finale _events_only.mp4)';
        }
        const bPos = li.querySelector('.btn-convert-pos');
        const bNeg = li.querySelector('.btn-convert-neg');
        if (bPos) bPos.style.display = '';
        if (bNeg) bNeg.style.display = '';
      }
    }
  });
  updateResultsSummaryAndWave();
}
if ($('bRestoreAll')) $('bRestoreAll').onclick = restoreAllEvents;

function applyMinConfFilter() {
  const slider = $('minConfSlider');
  if (!slider) return;
  const minVal = parseInt(slider.value, 10);
  $('minConfLabel').textContent = minVal === 0 ? 'Tutti (0%)' : '≥ ' + minVal + '%';

  currentEvents.forEach(ev => {
    const c = ev.conf !== null ? ev.conf : 100;
    ev.excluded = (c < minVal);
    const li = $('events').querySelector(`li[data-idx="${ev.idx}"]`);
    if (li) {
      if (ev.excluded) {
        li.classList.add('excluded');
      } else {
        li.classList.remove('excluded');
      }
    }
  });

  updateResultsSummaryAndWave();
}
if ($('minConfSlider')) $('minConfSlider').oninput = applyMinConfFilter;

function renderResults() {
  const text = $('log').textContent;
  const summary = $('summary');
  summary.textContent = '';
  $('events').innerHTML = '';
  let n = null;
  let autoThVal = null;
  currentEvents = [];

  text.split('\n').forEach(line => {
    let m_th = line.match(/\[Auto-Threshold\] (?:Optimal threshold determined|Applying calculated threshold): ([\d.]+)/);
    if (m_th) autoThVal = m_th[1];

    let m = line.match(/(?:Detected|Loaded)\s+(\d+)\s+(?:approved\s+)?(?:event|burp)\(s\):/i);
    if (m) n = parseInt(m[1], 10);
    m = line.match(/^\s*(\d+)\.\s+([\d.]+)\s*s\s*→\s*([\d.]+)\s*s\s*\(\s*([\d.]+)\s*s(?:[,\s]*conf:\s*(\d+)%)?\s*\)/);
    if (m) {
      const idx = parseInt(m[1], 10);
      const st = parseFloat(m[2]), en = parseFloat(m[3]);
      const dur = parseFloat(m[4]);
      const conf = m[5] ? parseInt(m[5], 10) : null;
      const evItem = { start: st, end: en, idx: idx, dur: dur, conf: conf, deleted: false, excluded: false };
      currentEvents.push(evItem);

      let badgeHtml = '';
      if (conf !== null) {
        let bBg = '#064e3b', bColor = '#34d399', bBorder = '#059669';
        if (conf < 70) {
          bBg = '#7f1d1d'; bColor = '#f87171'; bBorder = '#dc2626';
        } else if (conf < 85) {
          bBg = '#78350f'; bColor = '#fbbf24'; bBorder = '#d97706';
        }
        badgeHtml = `<span title="Confidenza del modello: ${conf}%" style="font-size:10px; font-weight:700; padding:1px 5px; border-radius:4px; background:${bBg}; color:${bColor}; border:1px solid ${bBorder}; font-family:ui-monospace, monospace">${conf}%</span>`;
      }

      const li = document.createElement('li');
      li.id = 'ev_li_' + idx;
      li.dataset.idx = idx;
      li.dataset.conf = conf !== null ? conf : 100;
      li.dataset.dur = dur;
      li.style.display = 'flex';
      li.style.alignItems = 'center';
      li.style.justifyContent = 'space-between';
      li.style.transition = 'all 0.15s';

      li.innerHTML = '<div class="ev-info" style="display:flex;align-items:center;gap:6px;cursor:pointer;flex:1">' +
        '<b style="color:var(--amber)">#' + idx + '</b> ' + fmt(st) + ' → ' + fmt(en) +
        ' <span style="color:#888">(' + dur.toFixed(2) + 's)</span> ' + badgeHtml +
        '</div>' +
        '<div class="ev-actions" style="display:flex;align-items:center;gap:5px">' +
        '<button class="btn-convert-pos" title="Aggiungi come positivo [+] e rifinisci i bordi sulla forma d\'onda">✏️ [+]</button>' +
        '<button class="btn-convert-neg" title="Aggiungi come falso positivo [-] per escluderlo dal modello e dal video">🚫 [-]</button>' +
        '<button class="btn-delete-event" title="Elimina questo evento dai risultati e dal video finale _events_only.mp4">🗑️ Elimina</button>' +
        '</div>';

      li.querySelector('.ev-info').onclick = () => {
        if (window.wave && window.wave.focusInterval) {
          window.wave.focusInterval(st, en);
        } else if (window.redrawWave) {
          redrawWave();
        }
        if (window.wave && window.wave.playInterval) {
          window.wave.playInterval(st, en, false);
        } else {
          v.currentTime = st;
          v.play();
        }
      };

      const bPos = li.querySelector('.btn-convert-pos');
      const bNeg = li.querySelector('.btn-convert-neg');
      const bDel = li.querySelector('.btn-delete-event');

      bPos.onclick = e => {
        e.stopPropagation();
        convertToLabel(st, en, 'pos', li);
        if (evItem.deleted) toggleDeleteEvent(idx);
      };
      bNeg.onclick = e => {
        e.stopPropagation();
        convertToLabel(st, en, 'neg', li);
        if (!evItem.deleted) toggleDeleteEvent(idx);
      };
      bDel.onclick = e => {
        e.stopPropagation();
        toggleDeleteEvent(idx);
      };

      $('events').appendChild(li);
    }
  });

  applyMinConfFilter();

  if (n !== null) {
    const p = document.createElement('div');
    const thText = autoThVal ? ' <span style="color:var(--amber);font-weight:normal">(🎯 soglia auto calcolata: <b>' + autoThVal + '</b>)</span>' : '';
    p.innerHTML = '<b>' + n + ' eventi rilevati.</b>' + thText + ' Clicca un evento per ascoltare. Usa <b>🗑️ Elimina</b> per toglierlo dal video finale <code>_events_only.mp4</code>.';
    summary.appendChild(p);
  } else if (!currentEvents.length) {
    summary.textContent = 'Nessun evento rilevato. Prova ad abbassare la soglia o a usare la modalità con etichette.';
  }

  const outs = [];
  text.split('\n').forEach(line => {
    let m = line.match(/(?:Events only|Burps only)\s*:\s*(.+)$/);
    if (m) outs.push(['events_only', m[1].trim()]);
    m = line.match(/(?:Clean video|No events|No burps)\s*:\s*(.+)$/);
    if (m) outs.push(['clean_video', m[1].trim()]);
  });
  const box = $('outputs');
  box.innerHTML = '';
  outs.forEach(([kind, path]) => {
    const span = document.createElement('span');
    span.className = 'badge';
    span.textContent = (kind === 'events_only' || kind === 'burps_only' ? '🎯 Solo eventi' : '✨ Solo senza eventi (pulito)') + ' — ' + path.replace(/^.*\//, '');
    span.onclick = () => previewOut(kind, path);
    box.appendChild(span);
    box.appendChild(document.createTextNode(' '));
  });
  if (outs.length) box.appendChild(document.createTextNode("(clicca qui per vedere/riprodurre l'output)"));
}

function convertToLabel(st, en, type, li) {
  // Remove any existing label overlapping this range
  labels = labels.filter(l => !(l[0] <= en && l[1] >= st));
  labels.push([st, en, type]);
  labels.sort((a, b) => a[0] - b[0]);
  refreshLabels();

  const bPos = li.querySelector('.btn-convert-pos');
  const bNeg = li.querySelector('.btn-convert-neg');
  if (type === 'pos') {
    if (bPos) { bPos.classList.add('active'); bPos.textContent = '✅ [+]'; }
    if (bNeg) { bNeg.classList.remove('active'); bNeg.style.opacity = '0.5'; }
    li.style.borderColor = 'rgba(34, 197, 94, 0.6)';
    li.style.background = 'rgba(34, 197, 94, 0.08)';
    if (window.wave && window.wave.focusInterval) {
      window.wave.focusInterval(st, en);
    }
    if (window.wave && window.wave.playInterval) {
      window.wave.playInterval(st, en, false);
    } else {
      v.currentTime = st;
      v.play();
    }
  } else {
    if (bNeg) { bNeg.classList.add('active'); bNeg.textContent = '🚫 [-]'; }
    if (bPos) { bPos.classList.remove('active'); bPos.style.opacity = '0.5'; }
    li.style.borderColor = 'rgba(239, 68, 68, 0.6)';
    li.style.background = 'rgba(239, 68, 68, 0.08)';
  }

  // Switch to teach mode automatically so labels are used
  if (mode !== 'teach') {
    document.querySelectorAll('.mode-card').forEach(c => c.classList.remove('selected'));
    const teachCard = document.querySelector('.mode-card[data-mode="teach"]');
    if (teachCard) {
      teachCard.classList.add('selected');
      mode = 'teach';
      updateModeUI();
    }
  }
}

function previewOut(kind, path) {
  const ov = $('outV');
  ov.src = '/file?path=' + encodeURIComponent(path) + '&t=' + Date.now();
  ov.style.display = 'block';
  ov.play();
}

/* ---------- etichette dal file (al caricamento video) ---------- */
function loadLabelsForCurrent() {
  const video = currentVideo();
  if (!video) return;
  fetch('/api/labels?path=' + encodeURIComponent(video)).then(r => r.json()).then(d => {
    if (d.labels) { labels = d.labels; refreshLabels(); }
  }).catch(() => {});
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

loadFiles();
</script>
<script src="/wave.js"></script>
</body>
</html>
"""


# ------------------------------------------------------------- server ---

def _send_json(handler, obj, code: int = 200) -> None:
    body = json.dumps(obj).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_text(handler, text: str, code: int = 200) -> None:
    body = text.encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class UIHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the console clean
        pass

    # ---------------- GET ----------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path in ("/", "/index.html", "/ui"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/files":
            _send_json(self, list_files())

        elif path == "/api/status":
            since = int(_query(self, "since") or 0)
            with ST.lock:
                lines = ST.run_lines[since:]
                status = {
                    "running": ST.running,
                    "done": ST.done,
                    "exit_code": ST.exit_code,
                    "lines": lines,
                }
            _send_json(self, status)

        elif path == "/api/labels":
            video = _query(self, "path") or ""
            labels = read_labels_file(Path(video)) if video else []
            _send_json(self, {"labels": labels})

        elif path == "/video":
            with ST.lock:
                video = ST.video
            if video is None or not is_servable(video):
                self.send_error(404, "nessun video selezionato")
                return
            self._serve_file(video)

        elif path == "/wave.js":
            body = WAVE_JS.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/wave":
            video = Path(_query(self, "path") or "")
            if not video.is_file() or not is_servable(video):
                _send_json(self, {"error": "video non disponibile"}, 400)
                return
            serve_wave(self, video)

        elif path == "/file":
            target = Path(_query(self, "path") or "")
            if not target.is_file() or not is_servable(target):
                self.send_error(403, "file non disponibile")
                return
            self._serve_file(target)

        else:
            self.send_error(404)

    # ---------------- POST ----------------
    def do_POST(self):
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            data = {}

        if path == "/api/select":
            p = Path(str(data.get("path", "")))
            if not p.is_file():
                _send_json(self, {"error": f"File non trovato: {p}"}, 400)
                return
            p = p.resolve()
            ST.allow(p)
            with ST.lock:
                ST.video = p
            labels = read_labels_file(p)
            _send_json(self, {
                "ok": True,
                "name": p.name,
                "size_mb": round(p.stat().st_size / 1e6, 1),
                "labels": labels,
            })

        elif path == "/api/labels":
            p = Path(str(data.get("path", "")))
            labels = data.get("labels") or []
            if not p.is_file():
                _send_json(self, {"error": "scegli prima il video"}, 400)
                return
            target = p.with_name(p.stem + "_etichette.txt")
            _write_labels_file(target, labels)
            _send_text(self, f"Salvate {len(labels)} etichette in {target.name} "
                            f"(usale ora con la modalità 'Impara dagli esempi').")

        elif path == "/api/run":
            try:
                command, _ = start_run(data)
            except ValueError as exc:
                _send_json(self, {"error": str(exc)}, 400)
                return
            _send_json(self, {"ok": True, "command": command})

        elif path == "/api/stop":
            stop_run()
            _send_json(self, {"ok": True})

        else:
            self.send_error(404)

    # ---------------- helpers ----------------
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
        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _query(handler, key: str) -> str | None:
    qs = handler.path.split("?", 1)[1] if "?" in handler.path else ""
    for part in qs.split("&"):
        if part.startswith(key + "="):
            return urllib.parse.unquote(part[len(key) + 1:])
    return None


def read_labels_file(path: Path) -> list:
    """Parse a labels file into [[s, e, 'pos'|'neg'], ...] numeric intervals."""
    labels_path = path.with_name(path.stem + "_etichette.txt")
    if not labels_path.is_file():
        return []
    out: list = []
    for raw in labels_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        is_neg = line.startswith("-") or line.startswith("!") or line.startswith("neg:")
        cleaned = line.lstrip("-!").removeprefix("neg:")
        m = re.match(r"^(\d+):(\d+(?:\.\d+)?)-(\d+):(\d+(?:\.\d+)?)$", cleaned)
        if m:
            a = int(m.group(1)) * 60 + float(m.group(2))
            b = int(m.group(3)) * 60 + float(m.group(4))
            if b - a >= 0.05:
                out.append([round(a, 2), round(b, 2), "neg" if is_neg else "pos"])
    return out


# ---------------------------------------------------------------- main --

def run_app(video: Path | None = None) -> None:
    if video is not None:
        video = video.resolve()
        if not video.is_file():
            print(f"ERROR: file non trovato: {video}", file=sys.stderr)
            sys.exit(1)
        ST.allow(video)
        with ST.lock:
            ST.video = video

    port = None
    server = None
    for candidate in range(8800, 8840):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), UIHandler)
            port = candidate
            break
        except OSError:
            continue
    if server is None:
        print("ERROR: nessuna porta libera per il server locale.", file=sys.stderr)
        sys.exit(1)

    url = f"http://127.0.0.1:{port}/"
    print("✂️ AutoCut — interfaccia completa")
    if ST.video is not None:
        print(f"  Video: {ST.video}")
    print(f"  Apri il browser: {url}")
    print("  Per chiudere: Ctrl+C in questo terminale.")
    if not os.environ.get("AB_NO_BROWSER"):
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nChiuso. I file di output restano nella cartella scelta.")
        sys.exit(0)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, nargs="?",
                        help="video da preselezionare (opzionale)")
    args = parser.parse_args()
    run_app(args.video)


if __name__ == "__main__":
    main()