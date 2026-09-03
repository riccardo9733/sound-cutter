#!/usr/bin/env python3
"""Shared audio-waveform support for the local web UIs (app_ui.py, label_gui.py).

Server side
    compute_wave_peaks(video) extracts the audio track with ffmpeg (mono 8 kHz
    s16le over a pipe), computes a high-resolution envelope (~50 values per
    second, up to 80000 buckets, normalized to 0..1) and caches the result in
    the system temp dir, so long videos are decoded only once and can be
    zoomed in with millisecond precision.

Client side
    WAVE_JS draws that envelope on a <canvas id="wave"> with zoom and pan
    controls, overview minimap navigator, dynamic time ruler, playhead
    tracking, labeled intervals and drag-to-mark capabilities.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

RATE = 8000          # Hz of the analysis stream
BLOCK = 160          # samples per envelope block (20 ms at 8 kHz -> 50 Hz peak resolution)
MAX_BUCKETS = 120000 # high fidelity envelope for zoom detail
MIN_BUCKETS = 200


def _find_ffmpeg() -> str | None:
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _cache_path(video: Path) -> Path:
    st = video.stat()
    key = hashlib.sha1(
        f"v2|{video.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()
    ).hexdigest()
    return Path(tempfile.gettempdir()) / "autocut_wave_v2" / f"{key}.json"


def compute_wave_peaks(video: Path) -> dict:
    """Return {"duration": s, "data": [amp, ...]} envelope of the audio track.

    The envelope is symmetric around 0: one value per bucket in 0..1.
    Cached in the system temp dir keyed by path + mtime + size.
    """
    try:
        cache = _cache_path(video)
        if cache.is_file():
            return json.loads(cache.read_text(encoding="utf-8"))
    except OSError:
        pass

    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy non disponibile (attiva il venv)"}

    exe = _find_ffmpeg()
    if exe is None:
        return {"error": "ffmpeg non trovato"}
    cmd = [exe, "-v", "error", "-i", str(video), "-vn", "-ac", "1",
           "-ar", str(RATE), "-f", "s16le", "-"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
    except OSError as exc:
        return {"error": f"ffmpeg non avviabile: {exc}"}

    block_peaks: list = []
    while True:
        raw = proc.stdout.read(1 << 20)          # ~1 MB per read
        if not raw:
            break
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        usable = len(arr) - (len(arr) % BLOCK)
        if usable <= 0:
            continue
        block_peaks.append(np.abs(arr[:usable].reshape(-1, BLOCK)).max(axis=1))
    proc.wait()

    if not block_peaks:
        return {"error": "nessuna traccia audio nel file"}
    bm = np.concatenate(block_peaks)
    duration = bm.size * BLOCK / RATE

    if bm.size > MAX_BUCKETS:
        starts = np.linspace(0, bm.size, MAX_BUCKETS + 1).astype(np.int64)[:-1]
        vals = np.maximum.reduceat(bm, starts)
    else:
        vals = bm

    gmax = float(bm.max()) or 1.0
    data = (vals / gmax).round(4).tolist()

    result = {"duration": round(duration, 3), "data": data}
    try:
        cache = _cache_path(video)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result), encoding="utf-8")
    except OSError:
        pass
    return result


def serve_wave(handler, video: Path) -> None:
    """Write the waveform JSON for `video` to a BaseHTTPRequestHandler."""
    body = json.dumps(compute_wave_peaks(video)).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


WAVE_JS = r"""
/* Traccia audio avanzata con zoom, pan, minimappa, griglia temporale e playhead tracking. */
(function () {
  'use strict';
  var cv = document.getElementById('wave');
  if (!cv) return;
  var w = window.wave;
  if (!w || !w.video) return;
  var v = w.video();
  var peaks = null;          // {duration, data: Float32Array}
  var waveErr = null;
  var dragT0 = null, dragT1 = null;
  var loopA = null, loopB = null, looping = false;
  var fps = 30;
  var dpr = window.devicePixelRatio || 1;

  // Zoom and viewport state
  var zoom = 1.0;            // 1.0 = Fit (whole track), up to 300.0
  var viewStart = 0.0;       // start time in seconds of visible window
  var followPlayhead = true; // auto-scroll during playback
  var hoverTime = null;
  var isPanning = false;
  var panStartX = 0;
  var panStartViewStart = 0;
  var isMinimapDragging = false;
  var minimapDragMode = 'pan'; // 'pan', 'left', 'right'

  var timelineMode = 'both'; // 'both', 'results', 'labels'
  var detectedResults = [];
  var currentEventIndex = -1;

  // Interactive label edge trimming state
  var hoverLabelEdge = null; // { index, edge: 'start'|'end', label }
  var isDraggingLabelEdge = false;
  var dragEdgeLabel = null;
  var dragEdgeType = null; // 'start' | 'end'

  function findLabelEdgeAt(xPx, W) {
    if (timelineMode === 'results') return null;
    var lbls = (w && w.labels ? w.labels() : []) || [];
    var tol = 7;
    for (var i = 0; i < lbls.length; i++) {
      var it = lbls[i];
      var s = typeof it[0] === 'number' ? Math.abs(it[0]) : (it.start || 0);
      var e = typeof it[1] === 'number' ? Math.abs(it[1]) : (it.end || 0);
      var sx = timeToX(s, W);
      var ex = timeToX(e, W);
      if (Math.abs(xPx - sx) <= tol) {
        return { index: i, edge: 'start', label: it, time: s };
      }
      if (Math.abs(xPx - ex) <= tol) {
        return { index: i, edge: 'end', label: it, time: e };
      }
    }
    return null;
  }

  // Inject UI Styles and Controls container if not already present
  var wrap = cv.parentElement || cv.parentNode;
  var miniCv = null;
  var zoomSlider = null, zoomValBadge = null, btnFollow = null;
  var btnModeBoth = null, btnModeResults = null, btnModeLabels = null;
  var navWrap = null, navCurrBadge = null;

  function updateModeButtons() {
    if (btnModeBoth) btnModeBoth.classList.toggle('active', timelineMode === 'both');
    if (btnModeResults) btnModeResults.classList.toggle('active', timelineMode === 'results');
    if (btnModeLabels) btnModeLabels.classList.toggle('active', timelineMode === 'labels');
    var lblCount = (w.labels() || []).length;
    var resCount = detectedResults.length;
    var elLbl = document.getElementById('wm-lbl-count');
    var elRes = document.getElementById('wm-res-count');
    if (elLbl) elLbl.textContent = lblCount;
    if (elRes) elRes.textContent = resCount;
  }

  function updateNavUI() {
    if (!navWrap) return;
    if (detectedResults.length > 0) {
      navWrap.style.display = 'inline-flex';
      if (navCurrBadge) {
        navCurrBadge.textContent = (currentEventIndex >= 0 ? (currentEventIndex + 1) : '-') + ' / ' + detectedResults.length;
      }
    } else {
      navWrap.style.display = 'none';
    }
  }

  function jumpEvent(delta) {
    if (!detectedResults.length) return;
    currentEventIndex = Math.max(0, Math.min(detectedResults.length - 1, currentEventIndex + delta));
    var ev = detectedResults[currentEventIndex];
    if (ev) {
      var st = ev.start !== undefined ? ev.start : (typeof ev[0] === 'number' ? ev[0] : 0);
      var en = ev.end !== undefined ? ev.end : (typeof ev[1] === 'number' ? ev[1] : st + 0.5);
      if (w.playInterval) {
        w.playInterval(st, en, false);
      } else {
        v.currentTime = st;
        v.play();
      }
      // Ensure the event is centered in the waveform view
      var dur = getDuration();
      var visDur = dur / zoom;
      viewStart = Math.max(0, Math.min(dur - visDur, st - visDur * 0.3));
      clampView();
      updateNavUI();
      draw();
    }
  }

  function injectUi() {
    if (document.getElementById('wave-styles')) return;
    var style = document.createElement('style');
    style.id = 'wave-styles';
    style.textContent = `
      .wave-controls {
        display: flex;
        align-items: center;
        gap: 6px;
        flex-wrap: wrap;
        margin: 6px 0;
        font-family: inherit;
        font-size: 12px;
        color: #a0aec0;
      }
      .wave-btn {
        background: #1e222d;
        color: #e2e8f0;
        border: 1px solid #333a4d;
        border-radius: 5px;
        padding: 3px 8px;
        font-size: 11px;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        transition: background 0.15s, border-color 0.15s;
        user-select: none;
      }
      .wave-btn:hover {
        background: #2a3040;
        border-color: #4b5563;
      }
      .wave-btn.active {
        background: #2563eb;
        border-color: #3b82f6;
        color: #fff;
      }
      .wave-btn.active-amber {
        background: #d97706;
        border-color: #f59e0b;
        color: #fff;
      }
      .wave-slider-wrap {
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .wave-slider {
        -webkit-appearance: none;
        appearance: none;
        width: 90px;
        height: 5px;
        border-radius: 3px;
        background: #2d3748;
        outline: none;
        cursor: pointer;
      }
      .wave-slider::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 13px;
        height: 13px;
        border-radius: 50%;
        background: #3b82f6;
        cursor: pointer;
      }
      .wave-slider::-moz-range-thumb {
        width: 13px;
        height: 13px;
        border-radius: 50%;
        background: #3b82f6;
        cursor: pointer;
        border: none;
      }
      .wave-badge {
        font-family: ui-monospace, Menlo, monospace;
        font-size: 11px;
        color: #93c5fd;
        min-width: 38px;
      }
      .wave-mini-wrap {
        position: relative;
        width: 100%;
        height: 22px;
        margin-top: 4px;
        background: #0b0c10;
        border: 1px solid #232733;
        border-radius: 4px;
        overflow: hidden;
      }
      #wave-mini {
        width: 100%;
        height: 100%;
        display: block;
        cursor: pointer;
      }
      .wave-tooltip {
        position: absolute;
        top: 4px;
        right: 8px;
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(4px);
        border: 1px solid #334155;
        border-radius: 4px;
        padding: 2px 6px;
        font-family: ui-monospace, Menlo, monospace;
        font-size: 11px;
        color: #cbd5e1;
        pointer-events: none;
        z-index: 10;
      }
    `;
    document.head.appendChild(style);

    // Create toolbar
    var bar = document.createElement('div');
    bar.className = 'wave-controls';
    bar.innerHTML = `
      <span style="font-weight:600; color:#94a3b8;">🔍 Zoom:</span>
      <button type="button" class="wave-btn" id="wz-out" title="Zoom indietro (−)">➖</button>
      <div class="wave-slider-wrap">
        <input type="range" class="wave-slider" id="wz-slider" min="1" max="300" step="0.5" value="1">
        <span class="wave-badge" id="wz-val">1.0×</span>
      </div>
      <button type="button" class="wave-btn" id="wz-in" title="Zoom avanti (+)">➕</button>
      <button type="button" class="wave-btn active" id="wz-fit" title="Vista completa (1x)">Adatta (1×)</button>
      <button type="button" class="wave-btn" id="wz-5x" title="Dettaglio 5x">5×</button>
      <button type="button" class="wave-btn" id="wz-20x" title="Dettaglio 20x">20×</button>
      <button type="button" class="wave-btn" id="wz-50x" title="Dettaglio 50x">50×</button>
      <button type="button" class="wave-btn" id="wz-75x" title="Dettaglio 75x">75×</button>
      <button type="button" class="wave-btn" id="wz-150x" title="Dettaglio 150x">150×</button>
      <button type="button" class="wave-btn" id="wz-300x" title="Dettaglio massimo 300x">300×</button>
      <button type="button" class="wave-btn active" id="wz-follow" title="Segui il cursore durante la riproduzione">🎯 Segui playhead</button>

      <div style="display:inline-flex; align-items:center; background:#181b24; border:1px solid #2a3142; border-radius:5px; padding:2px; margin-left:4px; gap:2px">
        <span style="font-size:11px; font-weight:600; color:#94a3b8; padding:0 4px">Traccia:</span>
        <button type="button" class="wave-btn active" id="wm-both" title="Mostra sia etichette che risultati" style="padding:2px 6px; font-size:11px">✨ Entrambi</button>
        <button type="button" class="wave-btn" id="wm-results" title="Mostra solo i risultati trovati dall'AI" style="padding:2px 6px; font-size:11px">🎯 Risultati (<span id="wm-res-count">0</span>)</button>
        <button type="button" class="wave-btn" id="wm-labels" title="Mostra solo le etichette (+/-)" style="padding:2px 6px; font-size:11px">🏷️ Etichette (<span id="wm-lbl-count">0</span>)</button>
      </div>

      <div id="wave-event-nav" style="display:none; align-items:center; gap:3px; margin-left:4px">
        <button type="button" class="wave-btn" id="wen-prev" title="Evento precedente (K)" style="padding:2px 6px; font-size:11px">⏮ Prec</button>
        <span style="font-size:11px; color:#f59e0b; font-family:ui-monospace, monospace; font-weight:bold" id="wen-curr">0/0</span>
        <button type="button" class="wave-btn" id="wen-next" title="Evento successivo (J)" style="padding:2px 6px; font-size:11px">Succ ⏭</button>
      </div>
    `;

    // Create minimap container
    var miniWrap = document.createElement('div');
    miniWrap.className = 'wave-mini-wrap';
    miniWrap.innerHTML = '<canvas id="wave-mini"></canvas>';

    // Tooltip for time under cursor
    var tip = document.createElement('div');
    tip.className = 'wave-tooltip';
    tip.id = 'wave-tip';
    tip.style.display = 'none';

    wrap.insertBefore(bar, cv);
    wrap.appendChild(miniWrap);
    wrap.appendChild(tip);

    miniCv = document.getElementById('wave-mini');
    zoomSlider = document.getElementById('wz-slider');
    zoomValBadge = document.getElementById('wz-val');
    btnFollow = document.getElementById('wz-follow');

    btnModeBoth = document.getElementById('wm-both');
    btnModeResults = document.getElementById('wm-results');
    btnModeLabels = document.getElementById('wm-labels');
    navWrap = document.getElementById('wave-event-nav');
    navCurrBadge = document.getElementById('wen-curr');

    // Controls event bindings
    document.getElementById('wz-in').onclick = function () { setZoom(zoom * 1.5); };
    document.getElementById('wz-out').onclick = function () { setZoom(zoom / 1.5); };
    document.getElementById('wz-fit').onclick = function () { setZoom(1.0); };
    document.getElementById('wz-5x').onclick = function () { setZoom(5.0); };
    document.getElementById('wz-20x').onclick = function () { setZoom(20.0); };
    document.getElementById('wz-50x').onclick = function () { setZoom(50.0); };
    document.getElementById('wz-75x').onclick = function () { setZoom(75.0); };
    document.getElementById('wz-150x').onclick = function () { setZoom(150.0); };
    document.getElementById('wz-300x').onclick = function () { setZoom(300.0); };
    btnFollow.onclick = function () {
      followPlayhead = !followPlayhead;
      btnFollow.classList.toggle('active', followPlayhead);
    };
    zoomSlider.oninput = function () {
      setZoom(parseFloat(zoomSlider.value));
    };

    btnModeBoth.onclick = function () { timelineMode = 'both'; updateModeButtons(); draw(); };
    btnModeResults.onclick = function () { timelineMode = 'results'; updateModeButtons(); draw(); };
    btnModeLabels.onclick = function () { timelineMode = 'labels'; updateModeButtons(); draw(); };

    document.getElementById('wen-prev').onclick = function () { jumpEvent(-1); };
    document.getElementById('wen-next').onclick = function () { jumpEvent(1); };
  }
  injectUi();

  function fmt(t) {
    t = Math.max(0, t || 0);
    var m = Math.floor(t / 60);
    var s = (t - m * 60).toFixed(2).padStart(5, '0');
    return m + ':' + s;
  }

  function fmtExact(t) {
    t = Math.max(0, t || 0);
    var m = Math.floor(t / 60);
    var s = (t - m * 60).toFixed(3).padStart(6, '0');
    return m + ':' + s;
  }

  function getDuration() {
    return (peaks && peaks.duration) || (v && v.duration) || 1;
  }

  function clampView() {
    var dur = getDuration();
    zoom = Math.max(1.0, Math.min(300.0, zoom));
    var visDur = dur / zoom;
    if (viewStart < 0) viewStart = 0;
    if (viewStart + visDur > dur) viewStart = Math.max(0, dur - visDur);
  }

  function timeToX(t, W) {
    var dur = getDuration();
    var visDur = dur / zoom;
    return ((t - viewStart) / visDur) * W;
  }

  function xToTime(x, W) {
    var dur = getDuration();
    var visDur = dur / zoom;
    var t = viewStart + (x / W) * visDur;
    return Math.max(0, Math.min(dur, t));
  }

  function setZoom(newZoom, centerTime) {
    var dur = getDuration();
    newZoom = Math.max(1.0, Math.min(300.0, newZoom));
    if (centerTime === undefined) {
      centerTime = viewStart + (dur / zoom) / 2;
    }
    var oldVisDur = dur / zoom;
    var ratio = (centerTime - viewStart) / oldVisDur;
    zoom = newZoom;
    var newVisDur = dur / zoom;
    viewStart = centerTime - ratio * newVisDur;
    clampView();
    updateControls();
    draw();
  }

  function updateControls() {
    if (zoomSlider) zoomSlider.value = zoom.toFixed(1);
    if (zoomValBadge) zoomValBadge.textContent = zoom.toFixed(1) + '×';
    var bFit = document.getElementById('wz-fit');
    if (bFit) bFit.classList.toggle('active', zoom <= 1.05);
    var b5 = document.getElementById('wz-5x');
    if (b5) b5.classList.toggle('active', Math.abs(zoom - 5.0) < 0.3);
    var b20 = document.getElementById('wz-20x');
    if (b20) b20.classList.toggle('active', Math.abs(zoom - 20.0) < 0.5);
    var b50 = document.getElementById('wz-50x');
    if (b50) b50.classList.toggle('active', Math.abs(zoom - 50.0) < 0.5);
    var b75 = document.getElementById('wz-75x');
    if (b75) b75.classList.toggle('active', Math.abs(zoom - 75.0) < 0.5);
    var b150 = document.getElementById('wz-150x');
    if (b150) b150.classList.toggle('active', Math.abs(zoom - 150.0) < 0.5);
    var b300 = document.getElementById('wz-300x');
    if (b300) b300.classList.toggle('active', Math.abs(zoom - 300.0) < 0.5);
  }

  /* --- Drawing Main Waveform & Time Ruler --- */
  function draw() {
    var W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return;
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    var ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // Dark background
    ctx.fillStyle = '#0d0e11';
    ctx.fillRect(0, 0, W, H);

    if (!peaks) {
      ctx.fillStyle = '#8b93a1';
      ctx.font = '12px ui-monospace, Menlo, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(waveErr || 'caricamento traccia audio…', W / 2, H / 2 + 4);
      drawMinimap();
      return;
    }

    var dur = getDuration();
    clampView();
    var visDur = dur / zoom;
    var viewEnd = viewStart + visDur;

    // 1. Time Ruler at Top (Height: 18px)
    var rulerH = 18;
    ctx.fillStyle = '#141720';
    ctx.fillRect(0, 0, W, rulerH);
    ctx.strokeStyle = '#222736';
    ctx.beginPath();
    ctx.moveTo(0, rulerH - 0.5);
    ctx.lineTo(W, rulerH - 0.5);
    ctx.stroke();

    // Pick dynamic tick step based on visible duration
    var tickSteps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
    var bestStep = tickSteps[tickSteps.length - 1];
    for (var sIdx = 0; sIdx < tickSteps.length; sIdx++) {
      var step = tickSteps[sIdx];
      var numTicks = visDur / step;
      if (numTicks >= 4 && numTicks <= 16) {
        bestStep = step;
        break;
      }
    }

    var firstTick = Math.ceil(viewStart / bestStep) * bestStep;
    ctx.font = '10px ui-monospace, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (var tCur = firstTick; tCur <= viewEnd + 0.0001; tCur += bestStep) {
      var tx = timeToX(tCur, W);
      if (tx < -10 || tx > W + 10) continue;

      // Tick notch
      ctx.strokeStyle = '#3b4254';
      ctx.beginPath();
      ctx.moveTo(tx, rulerH - 6);
      ctx.lineTo(tx, rulerH);
      ctx.stroke();

      // Tick label
      ctx.fillStyle = '#7e8b9f';
      var txt = bestStep < 0.1 ? fmtExact(tCur) : fmt(tCur);
      ctx.fillText(txt, tx, rulerH / 2);

      // Subtle vertical grid line on waveform area
      ctx.strokeStyle = 'rgba(35, 42, 58, 0.4)';
      ctx.beginPath();
      ctx.moveTo(tx, rulerH);
      ctx.lineTo(tx, H);
      ctx.stroke();
    }

    // 2. Audio Waveform Centerline
    var waveH = H - rulerH;
    var waveMid = rulerH + waveH / 2;
    ctx.strokeStyle = '#1e2330';
    ctx.beginPath();
    ctx.moveTo(0, waveMid);
    ctx.lineTo(W, waveMid);
    ctx.stroke();

    // 3. Render High-Detail Audio Waveform Envelope
    var data = peaks.data;
    var n = data.length;
    var samplesPerSec = n / dur;
    var half = (waveH / 2) - 4;

    // Gradient styling for waveform
    var grad = ctx.createLinearGradient(0, rulerH, 0, H);
    grad.addColorStop(0, '#60a5fa');
    grad.addColorStop(0.5, '#2563eb');
    grad.addColorStop(1, '#60a5fa');
    ctx.fillStyle = grad;

    var numColumns = Math.max(1, Math.floor(W));
    for (var px = 0; px < numColumns; px++) {
      var t0 = xToTime(px, W);
      var t1 = xToTime(px + 1, W);
      var idx0 = Math.max(0, Math.min(n - 1, Math.floor(t0 * samplesPerSec)));
      var idx1 = Math.max(0, Math.min(n - 1, Math.ceil(t1 * samplesPerSec)));

      var amp = 0;
      if (idx1 >= idx0) {
        for (var k = idx0; k <= idx1; k++) {
          if (data[k] > amp) amp = data[k];
        }
      } else {
        amp = data[idx0] || 0;
      }

      var barH = Math.max(1.5, amp * half);
      ctx.fillRect(px, waveMid - barH, 1, barH * 2);
    }

    // 4a. Highlight Regions: Labeled intervals (positive in blue, negative in red)
    if (timelineMode === 'labels' || timelineMode === 'both') {
      (w.labels() || []).forEach(function (it, idx) {
        var s = typeof it[0] === 'number' ? Math.abs(it[0]) : (it.start || 0);
        var e = typeof it[1] === 'number' ? Math.abs(it[1]) : (it.end || 0);
        var isNeg = it[2] === 'neg' || it.type === 'neg' || (typeof it[0] === 'number' && it[0] < 0);
        var x1 = timeToX(s, W), x2 = timeToX(e, W);
        var rx = Math.min(x1, x2), rw = Math.max(2, Math.abs(x2 - x1));
        if (rx + rw < 0 || rx > W) return;

        if (isNeg) {
          ctx.fillStyle = 'rgba(239, 68, 68, 0.35)';
          ctx.fillRect(rx, rulerH, rw, waveH);
          ctx.strokeStyle = '#ef4444';
          ctx.strokeRect(rx, rulerH, rw, waveH);

          if (rw > 28) {
            ctx.fillStyle = '#fca5a5';
            ctx.font = '10px ui-monospace, Menlo, monospace';
            ctx.textAlign = 'left';
            ctx.fillText('[-] #' + (idx + 1), rx + 6, rulerH + 12);
          }
        } else {
          ctx.fillStyle = 'rgba(37, 99, 235, 0.35)';
          ctx.fillRect(rx, rulerH, rw, waveH);
          ctx.strokeStyle = '#3b82f6';
          ctx.strokeRect(rx, rulerH, rw, waveH);

          if (rw > 28) {
            ctx.fillStyle = '#93c5fd';
            ctx.font = '10px ui-monospace, Menlo, monospace';
            ctx.textAlign = 'left';
            ctx.fillText('[+] #' + (idx + 1), rx + 6, rulerH + 12);
          }
        }

        // Visual resize handle indicators on left and right edges
        var isHovStart = (hoverLabelEdge && hoverLabelEdge.index === idx && hoverLabelEdge.edge === 'start');
        var isHovEnd = (hoverLabelEdge && hoverLabelEdge.index === idx && hoverLabelEdge.edge === 'end');

        // Left edge handle
        ctx.fillStyle = isHovStart ? '#ffffff' : (isNeg ? '#f87171' : '#60a5fa');
        ctx.fillRect(rx - (isHovStart ? 2 : 1), rulerH + 2, isHovStart ? 4 : 2, waveH - 4);

        // Right edge handle
        ctx.fillStyle = isHovEnd ? '#ffffff' : (isNeg ? '#f87171' : '#60a5fa');
        ctx.fillRect(rx + rw - (isHovEnd ? 2 : 1), rulerH + 2, isHovEnd ? 4 : 2, waveH - 4);

        // Edge dragging / hovering tooltip with exact millisecond readout
        if (isHovStart || isHovEnd) {
          var edgeT = isHovStart ? s : e;
          var edgeX = isHovStart ? rx : rx + rw;
          var edgeTxt = (isHovStart ? '◀ Inizio: ' : 'Fine: ▶ ') + fmtExact(edgeT) + ' (' + (e - s).toFixed(2) + 's)';
          ctx.font = '10px ui-monospace, Menlo, monospace';
          var bW = ctx.measureText(edgeTxt).width + 12;
          var bX = Math.max(4, Math.min(W - bW - 4, edgeX - bW / 2));
          ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
          ctx.fillRect(bX, rulerH + waveH - 24, bW, 20);
          ctx.strokeStyle = isHovStart ? '#38bdf8' : '#fb923c';
          ctx.strokeRect(bX, rulerH + waveH - 24, bW, 20);
          ctx.fillStyle = '#f8fafc';
          ctx.textAlign = 'center';
          ctx.fillText(edgeTxt, bX + bW / 2, rulerH + waveH - 10);
        }
      });
    }

    // 4b. Highlight Regions: AI Detected Results (in Glowing Amber / Gold)
    if (timelineMode === 'results' || timelineMode === 'both') {
      (detectedResults || []).forEach(function (it, idx) {
        var s = it.start !== undefined ? it.start : (typeof it[0] === 'number' ? it[0] : 0);
        var e = it.end !== undefined ? it.end : (typeof it[1] === 'number' ? it[1] : 0);
        var x1 = timeToX(s, W), x2 = timeToX(e, W);
        var rx = Math.min(x1, x2), rw = Math.max(3, Math.abs(x2 - x1));
        if (rx + rw < 0 || rx > W) return;

        var isCurr = (idx === currentEventIndex);
        ctx.fillStyle = isCurr ? 'rgba(245, 158, 11, 0.65)' : 'rgba(245, 158, 11, 0.35)';
        ctx.fillRect(rx, rulerH, rw, waveH);
        ctx.strokeStyle = isCurr ? '#fbbf24' : '#f59e0b';
        ctx.lineWidth = isCurr ? 2 : 1;
        ctx.strokeRect(rx, rulerH, rw, waveH);
        ctx.lineWidth = 1;

        if (rw > 24) {
          ctx.fillStyle = '#fde68a';
          ctx.font = 'bold 10px ui-monospace, Menlo, monospace';
          ctx.textAlign = 'left';
          var labelY = rulerH + (timelineMode === 'both' ? 24 : 12);
          ctx.fillText('🎯 #' + (idx + 1), rx + 3, labelY);
        }
      });
    }

    // 5. Open Interval (Mark Start in progress)
    var os = w.openStart();
    if (os !== null) {
      var ox1 = timeToX(os, W), ox2 = timeToX(v.currentTime, W);
      var orx = Math.min(ox1, ox2), orw = Math.max(2, Math.abs(ox2 - ox1));
      ctx.fillStyle = 'rgba(34, 197, 94, 0.3)';
      ctx.fillRect(orx, rulerH, orw, waveH);
      ctx.strokeStyle = '#22c55e';
      ctx.strokeRect(orx, rulerH, orw, waveH);
    }

    // 6. Active Drag Selection
    if (dragT0 !== null && dragT1 !== null) {
      var dx1 = timeToX(dragT0, W), dx2 = timeToX(dragT1, W);
      var drx = Math.min(dx1, dx2), drw = Math.max(2, Math.abs(dx2 - dx1));
      ctx.fillStyle = 'rgba(34, 197, 94, 0.45)';
      ctx.fillRect(drx, rulerH, drw, waveH);
      ctx.strokeStyle = '#4ade80';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(drx, rulerH, drw, waveH);
      ctx.lineWidth = 1;

      // Selection duration badge
      var selDur = Math.abs(dragT1 - dragT0);
      var badgeText = selDur.toFixed(2) + 's (' + fmt(Math.min(dragT0, dragT1)) + ' → ' + fmt(Math.max(dragT0, dragT1)) + ')';
      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      var tw = ctx.measureText(badgeText).width + 8;
      var bx = Math.max(4, Math.min(W - tw - 4, drx + drw / 2 - tw / 2));
      ctx.fillRect(bx, rulerH + 4, tw, 16);
      ctx.fillStyle = '#4ade80';
      ctx.font = '10px ui-monospace, Menlo, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(badgeText, bx + tw / 2, rulerH + 12);
    }

    // 7. Looping Region
    if (looping && loopA !== null && loopB !== null) {
      var lx1 = timeToX(loopA, W), lx2 = timeToX(loopB, W);
      var lrx = Math.min(lx1, lx2), lrw = Math.max(2, Math.abs(lx2 - lx1));
      ctx.fillStyle = 'rgba(244, 167, 42, 0.25)';
      ctx.fillRect(lrx, rulerH, lrw, waveH);
      ctx.strokeStyle = '#f4a72a';
      ctx.strokeRect(lrx, rulerH, lrw, waveH);
    }

    // 8. Playhead (yellow vertical cursor line)
    if (v && v.currentTime !== undefined) {
      var pxHead = timeToX(v.currentTime, W);
      ctx.strokeStyle = '#f4a72a';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(pxHead, 0);
      ctx.lineTo(pxHead, H);
      ctx.stroke();
      ctx.lineWidth = 1;

      // Playhead top triangle
      ctx.fillStyle = '#f4a72a';
      ctx.beginPath();
      ctx.moveTo(pxHead - 4, 0);
      ctx.lineTo(pxHead + 4, 0);
      ctx.lineTo(pxHead, 6);
      ctx.closePath();
      ctx.fill();
    }

    // 9. Hover cursor guide
    if (hoverTime !== null) {
      var hx = timeToX(hoverTime, W);
      if (hx >= 0 && hx <= W) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(hx, rulerH);
        ctx.lineTo(hx, H);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    drawMinimap();
  }
  window.redrawWave = draw;

  /* --- Drawing Minimap Overview --- */
  function drawMinimap() {
    if (!miniCv) return;
    var mW = miniCv.clientWidth, mH = miniCv.clientHeight;
    if (!mW || !mH) return;
    miniCv.width = Math.round(mW * dpr);
    miniCv.height = Math.round(mH * dpr);
    var ctx = miniCv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, mW, mH);

    ctx.fillStyle = '#0b0c10';
    ctx.fillRect(0, 0, mW, mH);

    if (!peaks) return;
    var dur = getDuration();
    var data = peaks.data;
    var n = data.length;

    // Full audio envelope in minimap
    var step = Math.max(1, Math.floor(n / mW));
    var half = mH / 2 - 2;
    ctx.fillStyle = '#334155';
    for (var x = 0; x < mW; x++) {
      var i = Math.floor(x * (n / mW));
      var amp = data[i] || 0;
      var h = Math.max(1, amp * half);
      ctx.fillRect(x, mH / 2 - h, 1, h * 2);
    }

    // Draw labeled markers on minimap
    if (timelineMode === 'labels' || timelineMode === 'both') {
      (w.labels() || []).forEach(function (it) {
        var s = typeof it[0] === 'number' ? Math.abs(it[0]) : (it.start || 0);
        var e = typeof it[1] === 'number' ? Math.abs(it[1]) : (it.end || 0);
        var isNeg = it[2] === 'neg' || it.type === 'neg' || (typeof it[0] === 'number' && it[0] < 0);
        var x1 = (s / dur) * mW, x2 = (e / dur) * mW;
        ctx.fillStyle = isNeg ? 'rgba(239, 68, 68, 0.7)' : 'rgba(59, 130, 246, 0.6)';
        ctx.fillRect(Math.min(x1, x2), 0, Math.max(2, Math.abs(x2 - x1)), mH);
      });
    }

    // Draw detected results markers on minimap
    if (timelineMode === 'results' || timelineMode === 'both') {
      (detectedResults || []).forEach(function (it) {
        var s = it.start !== undefined ? it.start : (typeof it[0] === 'number' ? it[0] : 0);
        var e = it.end !== undefined ? it.end : (typeof it[1] === 'number' ? it[1] : 0);
        var x1 = (s / dur) * mW, x2 = (e / dur) * mW;
        ctx.fillStyle = 'rgba(245, 158, 11, 0.85)';
        ctx.fillRect(Math.min(x1, x2), 0, Math.max(2, Math.abs(x2 - x1)), mH);
      });
    }

    // Playhead on minimap
    var pX = ((v.currentTime || 0) / dur) * mW;
    ctx.fillStyle = '#f4a72a';
    ctx.fillRect(pX - 0.5, 0, 1.5, mH);

    // Visible Window Viewport box on minimap
    var visDur = dur / zoom;
    var vx1 = (viewStart / dur) * mW;
    var vx2 = ((viewStart + visDur) / dur) * mW;
    var vw = Math.max(6, vx2 - vx1);

    ctx.fillStyle = 'rgba(59, 130, 246, 0.18)';
    ctx.fillRect(vx1, 0, vw, mH);
    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth = 1;
    ctx.strokeRect(vx1, 0, vw, mH);

    // Left and right handles
    ctx.fillStyle = '#93c5fd';
    ctx.fillRect(vx1, 0, 3, mH);
    ctx.fillRect(vx1 + vw - 3, 0, 3, mH);
  }

  function loadWave() {
    if (!window.waveUrl) { draw(); return; }
    fetch(window.waveUrl).then(function (r) { return r.json(); }).then(function (d) {
      if (d.error) { waveErr = d.error; draw(); return; }
      peaks = { duration: d.duration, data: new Float32Array(d.data) };
      viewStart = 0.0;
      zoom = 1.0;
      updateControls();
      var h = document.getElementById('waveHint');
      if (h) {
        h.textContent = 'Traccia audio (' + fmt(d.duration) + ') — Zoom con rotellina / tasti (+ / -), trascina per marcare intervallo, clicca per andare al punto';
      }
      draw();
    }).catch(function (e) { waveErr = 'Traccia non disponibile (' + e + ')'; draw(); });
  }
  window.loadWave = loadWave;

  /* --- Interazioni Canvas Principale --- */
  var tip = document.getElementById('wave-tip');

  cv.addEventListener('contextmenu', function (e) {
    e.preventDefault();
  });

  cv.addEventListener('mousedown', function (e) {
    e.preventDefault();
    var r = cv.getBoundingClientRect();
    var x = e.clientX - r.left;

    // Right click (button 2) or Alt+Left click = Pan view
    if (e.button === 2 || e.button === 1 || e.altKey) {
      isPanning = true;
      panStartX = e.clientX;
      panStartViewStart = viewStart;
      cv.style.cursor = 'grabbing';
      return;
    }

    // Check if clicking on an existing label edge to resize it
    var edge = findLabelEdgeAt(x, r.width);
    if (edge && e.button === 0) {
      isDraggingLabelEdge = true;
      dragEdgeLabel = edge.label;
      dragEdgeType = edge.edge;
      v.pause();
      cv.style.cursor = 'col-resize';
      draw();
      return;
    }

    // Left click = Select / Mark / Seek
    if (e.button === 0) {
      v.pause();
      dragT0 = xToTime(x, r.width);
      dragT1 = dragT0;
      draw();
    }
  });

  window.addEventListener('mousemove', function (e) {
    var r = cv.getBoundingClientRect();

    if (isPanning) {
      var dx = e.clientX - panStartX;
      var visDur = getDuration() / zoom;
      var dt = -(dx / r.width) * visDur;
      viewStart = panStartViewStart + dt;
      clampView();
      draw();
      return;
    }

    if (isDraggingLabelEdge && dragEdgeLabel) {
      var xDrag = Math.max(0, Math.min(r.width, e.clientX - r.left));
      var t = xToTime(xDrag, r.width);
      var it = dragEdgeLabel;
      if (dragEdgeType === 'start') {
        var endVal = typeof it[1] === 'number' ? Math.abs(it[1]) : (it.end || 0);
        t = Math.max(0, Math.min(endVal - 0.02, t));
        if (typeof it[0] === 'number') {
          it[0] = it[0] < 0 ? -t : t;
        } else {
          it.start = t;
        }
      } else {
        var startVal = typeof it[0] === 'number' ? Math.abs(it[0]) : (it.start || 0);
        t = Math.min(getDuration(), Math.max(startVal + 0.02, t));
        if (typeof it[1] === 'number') {
          it[1] = t;
        } else {
          it.end = t;
        }
      }
      if (w && w.refresh) w.refresh();
      draw();
      return;
    }

    if (dragT0 !== null) {
      var x = Math.max(0, Math.min(r.width, e.clientX - r.left));
      dragT1 = xToTime(x, r.width);
      draw();
      return;
    }

    // Check edge hover for resizing cursor
    if (!isPanning && dragT0 === null && !isDraggingLabelEdge) {
      var hEdge = findLabelEdgeAt(e.clientX - r.left, r.width);
      if (hEdge) {
        hoverLabelEdge = hEdge;
        cv.style.cursor = 'col-resize';
      } else {
        if (hoverLabelEdge) {
          hoverLabelEdge = null;
          cv.style.cursor = 'crosshair';
        }
      }
    }

    // Update hover tooltip when inside canvas
    if (e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom) {
      var hx = e.clientX - r.left;
      hoverTime = xToTime(hx, r.width);
      if (tip) {
        tip.style.display = 'block';
        tip.textContent = '⏱ ' + fmtExact(hoverTime);
      }
      draw();
    } else if (hoverTime !== null) {
      hoverTime = null;
      if (tip) tip.style.display = 'none';
      draw();
    }
  });

  window.addEventListener('mouseup', function (e) {
    if (isPanning) {
      isPanning = false;
      cv.style.cursor = 'crosshair';
      return;
    }

    if (isDraggingLabelEdge) {
      isDraggingLabelEdge = false;
      dragEdgeLabel = null;
      dragEdgeType = null;
      cv.style.cursor = 'crosshair';
      if (w && w.refresh) w.refresh();
      draw();
      return;
    }

    if (dragT0 === null) return;
    var a = dragT0, b = dragT1;
    dragT0 = dragT1 = null;

    var r = cv.getBoundingClientRect();
    var dx = Math.abs(timeToX(b, r.width) - timeToX(a, r.width));

    // If clicked without moving (or tiny movement < 4px), seek
    if (dx < 4 || Math.abs(b - a) < 0.04) {
      v.currentTime = b;
      draw();
      return;
    }

    var s = Math.min(a, b), end = Math.max(a, b);
    var os = w.openStart();
    if (os !== null && end - os >= 0.05) {
      w.add(os, end);
    } else if (end - s >= 0.05) {
      w.add(s, end);
      v.currentTime = s;
    }
    draw();
  });

  // Wheel zoom centered on cursor & horizontal pan
  cv.addEventListener('wheel', function (e) {
    e.preventDefault();
    var r = cv.getBoundingClientRect();
    var cursorTime = xToTime(e.clientX - r.left, r.width);

    if (e.ctrlKey || (!e.shiftKey && Math.abs(e.deltaY) >= Math.abs(e.deltaX))) {
      // Zoom
      var factor = e.deltaY < 0 ? 1.25 : (1 / 1.25);
      setZoom(zoom * factor, cursorTime);
    } else {
      // Horizontal pan
      var delta = (e.deltaX || e.deltaY);
      var visDur = getDuration() / zoom;
      viewStart += (delta / r.width) * visDur * 0.4;
      clampView();
      draw();
    }
  }, { passive: false });

  // Double click to toggle zoom
  cv.addEventListener('dblclick', function (e) {
    var r = cv.getBoundingClientRect();
    var cursorTime = xToTime(e.clientX - r.left, r.width);
    if (zoom > 1.2) {
      setZoom(1.0);
    } else {
      setZoom(8.0, cursorTime);
    }
  });

  /* --- Interazioni Minimappa --- */
  if (miniCv) {
    miniCv.addEventListener('mousedown', function (e) {
      e.preventDefault();
      var r = miniCv.getBoundingClientRect();
      var dur = getDuration();
      var visDur = dur / zoom;
      var mx = e.clientX - r.left;
      var clickT = (mx / r.width) * dur;

      var vx1 = (viewStart / dur) * r.width;
      var vx2 = ((viewStart + visDur) / dur) * r.width;

      if (Math.abs(mx - vx1) < 6) {
        minimapDragMode = 'left';
      } else if (Math.abs(mx - vx2) < 6) {
        minimapDragMode = 'right';
      } else if (mx >= vx1 && mx <= vx2) {
        minimapDragMode = 'pan';
      } else {
        // Jump viewport center to click
        viewStart = clickT - visDur / 2;
        clampView();
        minimapDragMode = 'pan';
      }

      isMinimapDragging = true;
      panStartX = e.clientX;
      panStartViewStart = viewStart;
      draw();
    });

    window.addEventListener('mousemove', function (e) {
      if (!isMinimapDragging) return;
      var r = miniCv.getBoundingClientRect();
      var dur = getDuration();
      var visDur = dur / zoom;
      var dx = e.clientX - panStartX;
      var dt = (dx / r.width) * dur;

      if (minimapDragMode === 'pan') {
        viewStart = panStartViewStart + dt;
        clampView();
      } else if (minimapDragMode === 'left') {
        var newStart = panStartViewStart + dt;
        newStart = Math.max(0, Math.min(viewStart + visDur - 0.2, newStart));
        var newVisDur = (viewStart + visDur) - newStart;
        zoom = dur / newVisDur;
        viewStart = newStart;
        clampView();
        updateControls();
      } else if (minimapDragMode === 'right') {
        var newEnd = (panStartViewStart + visDur) + dt;
        newEnd = Math.max(viewStart + 0.2, Math.min(dur, newEnd));
        var newVisDur = newEnd - viewStart;
        zoom = dur / newVisDur;
        clampView();
        updateControls();
      }
      draw();
    });

    window.addEventListener('mouseup', function () {
      if (isMinimapDragging) {
        isMinimapDragging = false;
      }
    });
  }

  /* --- Step di un frame --- */
  var fpsSel = document.getElementById('fpsSel');
  function frameStep(dir, mult) {
    var f = parseFloat(fpsSel ? fpsSel.value : NaN);
    if (!isFinite(f)) f = fps;
    var dt = (dir * mult) / f;
    v.currentTime = Math.max(0, Math.min(v.duration || 1e9, (v.currentTime || 0) + dt));
    if (followPlayhead && zoom > 1) {
      ensurePlayheadVisible();
    }
  }
  var bPrev = document.getElementById('bFPrev'), bNext = document.getElementById('bFNext');
  if (bPrev) bPrev.onclick = function () { frameStep(-1, 1); };
  if (bNext) bNext.onclick = function () { frameStep(1, 1); };

  document.addEventListener('keydown', function (e) {
    var tag = e.target && e.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

    if (e.key === '[' || e.key === ']') {
      e.preventDefault();
      frameStep(e.key === ']' ? 1 : -1, e.shiftKey ? 10 : 1);
    } else if (e.key === '+' || e.key === '=') {
      e.preventDefault();
      setZoom(zoom * 1.3);
    } else if (e.key === '-' || e.key === '_') {
      e.preventDefault();
      setZoom(zoom / 1.3);
    } else if (e.key === '0') {
      e.preventDefault();
      setZoom(1.0);
    } else if (e.key === 'j' || e.key === 'J') {
      e.preventDefault();
      jumpEvent(1);
    } else if (e.key === 'k' || e.key === 'K') {
      e.preventDefault();
      jumpEvent(-1);
    } else if (e.key === 'r' || e.key === 'R') {
      e.preventDefault();
      var L = w.labels ? w.labels() : [];
      var target = (w.selectedLabel && w.selectedLabel()) || (L.length ? L[L.length - 1] : null);
      if (target && w.playInterval) {
        var s = typeof target[0] === 'number' ? Math.abs(target[0]) : (target.start || 0);
        var e = typeof target[1] === 'number' ? Math.abs(target[1]) : (target.end || 0);
        w.playInterval(s, e, false);
      }
    }
  });

  /* --- Rileva i fotogrammi al secondo reali (fallback 30) --- */
  function detectFps() {
    if (!('requestVideoFrameCallback' in v)) return;
    var t0 = v.currentTime, times = [];
    v.pause();
    var handler = function (now) {
      times.push(now);
      if (times.length < 6) {
        try { v.requestVideoFrameCallback(handler); } catch (e) { return; }
        v.currentTime = Math.min((v.duration || 1e9) - 0.001, v.currentTime + 0.001);
      } else {
        var best = Infinity;
        for (var i = 1; i < times.length; i++) {
          var d = times[i] - times[i - 1];
          if (d > 5 && d < 1000) best = Math.min(best, d);
        }
        v.currentTime = t0;
        if (best !== Infinity) {
          fps = 1000 / best;
          var h = document.getElementById('waveHint');
          if (h && h.textContent.indexOf('frame') < 0) {
            h.textContent += ' · frame ≈ ' + Math.round(fps) + ' fps';
          }
        }
      }
    };
    try { v.requestVideoFrameCallback(handler); } catch (e) {}
  }
  setTimeout(detectFps, 300);

  /* --- Loop e Riproduzione Isolata Intervallo --- */
  var isolatedRaf = null;
  var isolatedEnd = null;

  function stopIsolated() {
    if (isolatedRaf) {
      cancelAnimationFrame(isolatedRaf);
      isolatedRaf = null;
    }
    isolatedEnd = null;
  }

  function startLoopRaf() {
    function loopStep() {
      if (looping && loopA !== null && loopB !== null && !v.paused) {
        if (v.currentTime >= loopB) {
          v.currentTime = loopA;
        }
        isolatedRaf = requestAnimationFrame(loopStep);
      } else {
        isolatedRaf = null;
      }
    }
    isolatedRaf = requestAnimationFrame(loopStep);
  }

  w.stopLoop = function () {
    looping = false;
    loopA = null;
    loopB = null;
    stopIsolated();
    var bLoop = document.getElementById('bLoop');
    if (bLoop) {
      bLoop.textContent = '🔁 Ripeti loop';
      bLoop.style.background = '';
    }
    draw();
  };

  w.isLooping = function () { return looping; };
  w.loopInterval = function () { return looping ? [loopA, loopB] : null; };

  w.playInterval = function (st, en, loop) {
    if (st === undefined || en === undefined) return;
    stopIsolated();

    var bLoop = document.getElementById('bLoop');
    if (loop) {
      loopA = st;
      loopB = en;
      looping = true;
      v.currentTime = st;
      v.play();
      if (bLoop) {
        bLoop.textContent = '⏹ Smetti loop';
        bLoop.style.background = 'var(--green, #1e9e4a)';
      }
      startLoopRaf();
      draw();
      return;
    }

    // Single isolated playback
    if (looping) {
      looping = false;
      loopA = null;
      loopB = null;
      if (bLoop) {
        bLoop.textContent = '🔁 Ripeti loop';
        bLoop.style.background = '';
      }
    }

    isolatedEnd = en;
    v.currentTime = st;
    v.play();

    function checkEnd() {
      if (v.paused || isolatedEnd === null) {
        isolatedEnd = null;
        isolatedRaf = null;
        return;
      }
      if (v.currentTime >= isolatedEnd) {
        v.pause();
        v.currentTime = isolatedEnd;
        isolatedEnd = null;
        isolatedRaf = null;
        draw();
        return;
      }
      isolatedRaf = requestAnimationFrame(checkEnd);
    }
    isolatedRaf = requestAnimationFrame(checkEnd);
    draw();
  };

  var bLoop = document.getElementById('bLoop');
  if (bLoop) bLoop.onclick = function () {
    var L = w.labels ? w.labels() : [];
    var target = (w.selectedLabel && w.selectedLabel()) || (L.length ? L[L.length - 1] : null);
    if (!target) return;
    var s = typeof target[0] === 'number' ? Math.abs(target[0]) : (target.start || 0);
    var e = typeof target[1] === 'number' ? Math.abs(target[1]) : (target.end || 0);
    if (looping) {
      w.stopLoop();
    } else {
      w.playInterval(s, e, true);
    }
  };

  var bPlayIso = document.getElementById('bPlayIso');
  if (bPlayIso) bPlayIso.onclick = function () {
    var L = w.labels ? w.labels() : [];
    var target = (w.selectedLabel && w.selectedLabel()) || (L.length ? L[L.length - 1] : null);
    if (!target) return;
    var s = typeof target[0] === 'number' ? Math.abs(target[0]) : (target.start || 0);
    var e = typeof target[1] === 'number' ? Math.abs(target[1]) : (target.end || 0);
    w.playInterval(s, e, false);
  };

  /* --- Playhead Tracking --- */
  function ensurePlayheadVisible() {
    var t = v.currentTime || 0;
    var dur = getDuration();
    var visDur = dur / zoom;
    if (t < viewStart || t > viewStart + visDur * 0.92) {
      viewStart = Math.max(0, t - visDur * 0.2);
      clampView();
    }
  }

  v.addEventListener('pause', function () {
    if (isolatedEnd !== null) {
      stopIsolated();
    }
  });

  v.addEventListener('timeupdate', function () {
    if (looping && loopA !== null && v.currentTime >= loopB) {
      v.currentTime = loopA;
    }
    if (isolatedEnd !== null && !looping && v.currentTime >= isolatedEnd) {
      v.pause();
      v.currentTime = isolatedEnd;
      stopIsolated();
    }
    if (!v.paused && followPlayhead && zoom > 1) {
      ensurePlayheadVisible();
    }
    draw();
  });

  v.addEventListener('loadedmetadata', function () {
    viewStart = 0.0;
    clampView();
    draw();
  });
  v.addEventListener('seeked', function () {
    if (followPlayhead && zoom > 1) {
      ensurePlayheadVisible();
    }
    draw();
  });
  window.addEventListener('resize', draw);

  w.setResults = function (res) {
    detectedResults = res || [];
    currentEventIndex = detectedResults.length > 0 ? 0 : -1;
    timelineMode = detectedResults.length > 0 ? 'both' : timelineMode;
    updateModeButtons();
    updateNavUI();
    draw();
  };
  w.getResults = function () { return detectedResults; };
  w.setTimelineMode = function (m) {
    timelineMode = m;
    updateModeButtons();
    draw();
  };
  w.nextEvent = function () { jumpEvent(1); };
  w.prevEvent = function () { jumpEvent(-1); };
  w.focusInterval = function (st, en, targetZoom) {
    if (st === undefined || en === undefined) return;
    if (typeof targetZoom !== 'number') {
      targetZoom = zoom;
    }
    var dur = getDuration();
    if (!dur) return;
    var mid = (st + en) / 2;
    setZoom(targetZoom, mid);
    var visDur = dur / zoom;
    viewStart = Math.max(0, Math.min(dur - visDur, mid - visDur / 2));
    clampView();
    updateControls();
    draw();
  };

  if (window.waveUrl) loadWave(); else draw();
})();
"""