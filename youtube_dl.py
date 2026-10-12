#!/usr/bin/env python3
"""YouTube Downloader integration for AutoCut.

Features:
  - Extract metadata & resolutions dynamically (up to 4K / 1080p / 720p / MP3 audio)
  - Integration with yt-dlp and FFmpeg
  - Time range trimming (start / end)
  - Background thread job manager with real-time progress, speed, ETA, and cancellation
  - Direct 1-click loading into AutoCut
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yt_dlp
from yt_dlp.utils import DownloadCancelled

from twitch_dl import find_ffmpeg

ROOT = Path(__file__).resolve().parent
BIN_DIR = ROOT / "bin"
DOWNLOADS_DIR = ROOT / "downloads"


def ensure_ffmpeg_in_path() -> str | None:
    """Ensure ffmpeg is in PATH and return its absolute path."""
    ff_exe = find_ffmpeg()
    if not ff_exe:
        return None

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    symlink_target = BIN_DIR / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")

    try:
        if not symlink_target.exists():
            try:
                symlink_target.symlink_to(ff_exe)
            except Exception:
                shutil.copy2(ff_exe, symlink_target)
                symlink_target.chmod(symlink_target.stat().st_mode | 0o755)
    except Exception:
        pass

    # Ensure BIN_DIR is at the beginning of PATH
    bin_str = str(BIN_DIR)
    current_path = os.environ.get("PATH", "")
    if bin_str not in current_path.split(os.pathsep):
        os.environ["PATH"] = bin_str + os.pathsep + current_path

    return ff_exe


def sanitize_filename(name: str) -> str:
    """Sanitize string for safe filenames across OS."""
    s = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:120] if s else "youtube_video"


def format_bytes(b: int | float | None) -> str:
    """Format bytes into human readable string."""
    if not b or b <= 0:
        return ""
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    if b < 1024 * 1024 * 1024:
        return f"{b / (1024 * 1024):.1f} MB"
    return f"{b / (1024 * 1024 * 1024):.2f} GB"


def format_seconds(sec: int | float | None) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    if sec is None or sec < 0:
        return "--:--"
    sec = int(round(sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_time_str(val: str | None) -> float | None:
    """Parse time string like '01:30', '00:05:20', '90s', or '90' into seconds."""
    if not val:
        return None
    val = str(val).strip().lower().rstrip("s")
    if not val:
        return None
    if ":" in val:
        parts = val.split(":")
        try:
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
        except ValueError:
            return None
    try:
        return float(val)
    except ValueError:
        return None


def fetch_video_info(url: str) -> dict[str, Any]:
    """Extract metadata and available resolutions for a YouTube video."""
    ff_path = ensure_ffmpeg_in_path()
    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
    }
    if ff_path:
        ydl_opts["ffmpeg_location"] = ff_path

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if not info:
            raise ValueError("Impossibile estrarre informazioni dal link fornito.")

        # If it's a playlist, take the first entry or basic playlist info
        if "entries" in info and info["entries"]:
            entries = list(info["entries"])
            if entries and entries[0]:
                info = entries[0]

        title = info.get("title") or "Video senza titolo"
        uploader = info.get("uploader") or info.get("channel") or "Sconosciuto"
        duration = info.get("duration")
        thumbnail = info.get("thumbnail") or ""
        view_count = info.get("view_count")

        # Aggregate unique resolutions
        res_map: dict[int, dict[str, Any]] = {}
        for f in info.get("formats", []):
            h = f.get("height")
            if h and f.get("vcodec") != "none":
                fps = f.get("fps")
                size = f.get("filesize") or f.get("filesize_approx")
                if h not in res_map or (size and not res_map[h].get("filesize")):
                    res_map[h] = {
                        "height": int(h),
                        "fps": int(fps) if fps else None,
                        "filesize": size,
                        "filesize_str": format_bytes(size) if size else "",
                    }

        sorted_res = sorted(res_map.values(), key=lambda x: x["height"], reverse=True)

        # Formatted resolution options
        resolutions = []
        for r in sorted_res:
            h = r["height"]
            label = f"{h}p"
            if h >= 2160:
                label += " (4K UHD)"
            elif h >= 1440:
                label += " (2K QHD)"
            elif h >= 1080:
                label += " (Full HD)"
            elif h >= 720:
                label += " (HD)"
            elif h >= 480:
                label += " (SD)"

            if r.get("fps") and r["fps"] > 30:
                label += f" {r['fps']}fps"
            if r.get("filesize_str"):
                label += f" • ~{r['filesize_str']}"

            resolutions.append({
                "value": f"{h}p",
                "height": h,
                "label": label,
                "filesize": r.get("filesize"),
                "filesize_str": r.get("filesize_str", ""),
            })

        return {
            "id": info.get("id"),
            "url": url,
            "title": title,
            "uploader": uploader,
            "duration": duration,
            "duration_str": format_seconds(duration),
            "thumbnail": thumbnail,
            "view_count": view_count,
            "resolutions": resolutions,
        }


class YouTubeDownloadJob:
    """Manages background download of YouTube videos and audio."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.running: bool = False
        self.done: bool = False
        self.cancelled: bool = False
        self.error: str | None = None
        self.progress: float = 0.0
        self.stage: str = "In attesa"
        self.speed_str: str = ""
        self.eta_str: str = ""
        self.downloaded_str: str = ""
        self.total_str: str = ""
        self.lines: list[str] = []
        self.output_file: str | None = None
        self.video_title: str = ""
        self.command_str: str = ""
        self.ydl_instance: yt_dlp.YoutubeDL | None = None

    def log(self, msg: str) -> None:
        with self.lock:
            self.lines.append(str(msg))

    def start(self, params: dict) -> tuple[bool, str]:
        with self.lock:
            if self.running:
                return False, "Un download è già in corso."

            self.running = True
            self.done = False
            self.cancelled = False
            self.error = None
            self.progress = 0.0
            self.stage = "Inizializzazione..."
            self.speed_str = ""
            self.eta_str = ""
            self.downloaded_str = ""
            self.total_str = ""
            self.lines = []
            self.output_file = None
            self.video_title = ""
            self.command_str = ""
            self.ydl_instance = None

        thread = threading.Thread(target=self._run_worker, args=(params,), daemon=True)
        thread.start()
        return True, "Download avviato."

    def stop(self) -> bool:
        with self.lock:
            if not self.running:
                return False
            self.cancelled = True
            self.stage = "Interruzione in corso..."
            self.lines.append("🛑 Richiesta di interruzione da parte dell'utente...")
        return True

    def get_status(self, since: int = 0) -> dict[str, Any]:
        with self.lock:
            lines = self.lines[since:]
            return {
                "running": self.running,
                "done": self.done,
                "cancelled": self.cancelled,
                "error": self.error,
                "progress": round(self.progress, 1),
                "stage": self.stage,
                "speed": self.speed_str,
                "eta": self.eta_str,
                "downloaded": self.downloaded_str,
                "total": self.total_str,
                "lines": lines,
                "output_file": self.output_file,
                "video_title": self.video_title,
                "command": self.command_str,
            }

    def _run_worker(self, params: dict) -> None:
        try:
            self.log("🔍 Verifica requisiti yt-dlp e FFmpeg...")
            ff_path = ensure_ffmpeg_in_path()
            if ff_path:
                self.log(f"🎬 FFmpeg rilevato: {ff_path}")
            else:
                self.log("⚠️ FFmpeg non trovato. Unione video/audio e ritagli potrebbero non riuscire.")

            raw_url = str(params.get("url", "")).strip()
            if not raw_url:
                raise ValueError("Inserisci un link YouTube valido.")

            quality = str(params.get("quality", "best")).strip().lower()
            trim_start_str = str(params.get("trim_start", "")).strip()
            trim_end_str = str(params.get("trim_end", "")).strip()
            custom_filename = str(params.get("filename", "")).strip()
            out_dir_param = str(params.get("outdir", "")).strip()

            out_dir = Path(out_dir_param) if out_dir_param else DOWNLOADS_DIR
            if not out_dir.is_absolute():
                out_dir = ROOT / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            t_start = parse_time_str(trim_start_str)
            t_end = parse_time_str(trim_end_str)

            # Build format selector and output template
            is_audio_only = quality in ("audio_mp3", "audio_m4a", "audio")
            
            ydl_opts: dict[str, Any] = {
                "quiet": True,
                "no_warnings": True,
                "no_color": True,
                "noprogress": False,
                "ffmpeg_location": ff_path,
            }

            if is_audio_only:
                ydl_opts["format"] = "bestaudio/best"
                if quality == "audio_mp3":
                    ydl_opts["postprocessors"] = [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "320",
                    }]
                    ext_target = "mp3"
                else:
                    ydl_opts["postprocessors"] = [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "m4a",
                    }]
                    ext_target = "m4a"
            else:
                ext_target = "mp4"
                ydl_opts["merge_output_format"] = "mp4"

                # Extract max height if specified e.g. "1080p" -> 1080
                m_height = re.search(r"(\d+)p?", quality)
                if m_height:
                    max_h = int(m_height.group(1))
                    ydl_opts["format"] = (
                        f"bestvideo[height<={max_h}][ext=mp4]+bestaudio[ext=m4a]/"
                        f"bestvideo[height<={max_h}]+bestaudio/"
                        f"best[height<={max_h}]/best"
                    )
                else:
                    # Best quality
                    ydl_opts["format"] = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best"

            # Download ranges for trimming
            if t_start is not None or t_end is not None:
                start_sec = t_start if t_start is not None else 0
                end_sec = t_end if t_end is not None else float("inf")
                ydl_opts["download_ranges"] = yt_dlp.utils.download_range_func(None, [(start_sec, end_sec)])
                ydl_opts["force_keyframes_at_cuts"] = True
                self.log(f"✂️ Ritaglio impostato: {format_seconds(start_sec)} ➔ {format_seconds(end_sec) if end_sec != float('inf') else 'Fine'}")

            # Subtitles if requested
            if params.get("subtitles"):
                ydl_opts["writesubtitles"] = True
                ydl_opts["subtitleslangs"] = ["it", "en"]
                ydl_opts["embedsubtitles"] = True

            # Outtmpl
            if custom_filename:
                base_fname = sanitize_filename(custom_filename)
                out_template = str(out_dir / f"{base_fname}.%(ext)s")
            else:
                out_template = str(out_dir / "%(title)s [%(id)s].%(ext)s")
            ydl_opts["outtmpl"] = out_template

            # Hooks
            downloaded_files: list[str] = []

            def clean_ansi(s: Any) -> str:
                if s is None:
                    return ""
                return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", str(s)).strip()

            def calc_stream_pct(d: dict[str, Any]) -> float | None:
                # 1. Exact numeric bytes
                d_bytes = d.get("downloaded_bytes")
                t_bytes = d.get("total_bytes") or d.get("total_bytes_estimate")
                if d_bytes is not None and t_bytes and t_bytes > 0:
                    return (d_bytes / t_bytes) * 100.0

                # 2. Fragment progress (HLS / DASH)
                f_idx = d.get("fragment_index")
                f_cnt = d.get("fragment_count")
                if f_idx is not None and f_cnt and f_cnt > 0:
                    return (f_idx / f_cnt) * 100.0

                # 3. Float _percent
                if d.get("_percent") is not None:
                    try:
                        v_pct = float(d["_percent"])
                        return v_pct if v_pct > 1.0 else v_pct * 100.0
                    except Exception:
                        pass

                # 4. _percent_str with ANSI stripping
                raw_str = clean_ansi(d.get("_percent_str"))
                if raw_str:
                    m = re.search(r"(\d+(?:\.\d+)?)%", raw_str)
                    if m:
                        try:
                            return float(m.group(1))
                        except ValueError:
                            pass
                return None

            def progress_hook(d: dict[str, Any]) -> None:
                with self.lock:
                    if self.cancelled:
                        raise DownloadCancelled("Download annullato dall'utente.")

                status = d.get("status")
                if status == "downloading":
                    raw_pct = calc_stream_pct(d)

                    speed = clean_ansi(d.get("_speed_str"))
                    eta = clean_ansi(d.get("_eta_str"))
                    downloaded = format_bytes(d.get("downloaded_bytes"))
                    total = format_bytes(d.get("total_bytes") or d.get("total_bytes_estimate"))

                    with self.lock:
                        if raw_pct is not None:
                            pct_val = max(0.0, min(99.9, raw_pct))
                            self.progress = pct_val
                            self.stage = f"Download in corso ({pct_val:.1f}%)..."
                        if speed:
                            self.speed_str = speed
                        if eta:
                            self.eta_str = eta
                        if downloaded:
                            self.downloaded_str = downloaded
                        if total:
                            self.total_str = total

                elif status == "finished":
                    filename = d.get("filename")
                    if filename and filename not in downloaded_files:
                        downloaded_files.append(filename)
                    with self.lock:
                        self.stage = "Muxing e post-elaborazione FFmpeg..."
                        self.log("⚙️ Conversione / unione flussi con FFmpeg...")

            def postprocessor_hook(d: dict[str, Any]) -> None:
                with self.lock:
                    if self.cancelled:
                        raise DownloadCancelled("Download annullato dall'utente.")
                status = d.get("status")
                if status == "processing":
                    with self.lock:
                        self.stage = f"Elaborazione: {d.get('postprocessor', 'FFmpeg')}..."
                elif status == "finished":
                    filepath = d.get("info_dict", {}).get("filepath")
                    if filepath and filepath not in downloaded_files:
                        downloaded_files.append(filepath)

            class CustomLogger:
                def __init__(self, job: YouTubeDownloadJob) -> None:
                    self.job = job

                def debug(self, msg: str) -> None:
                    if msg.startswith("[download]") and "%" in msg:
                        return
                    self.job.log(msg)

                def info(self, msg: str) -> None:
                    self.job.log(msg)

                def warning(self, msg: str) -> None:
                    self.job.log(f"⚠️ {msg}")

                def error(self, msg: str) -> None:
                    self.job.log(f"❌ {msg}")

            ydl_opts["progress_hooks"] = [progress_hook]
            ydl_opts["postprocessor_hooks"] = [postprocessor_hook]
            ydl_opts["logger"] = CustomLogger(self)

            with self.lock:
                self.stage = "Connessione a YouTube..."
                self.command_str = f"yt-dlp --format \"{ydl_opts['format']}\" {raw_url}"

            self.log(f"🚀 Avvio yt-dlp per: {raw_url}")
            self.log(f"🎯 Qualità richiesta: {quality} ({'Audio' if is_audio_only else 'Video MP4'})")
            self.log(f"📁 Cartella destinazione: {out_dir}")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                with self.lock:
                    self.ydl_instance = ydl
                info_result = ydl.extract_info(raw_url, download=True)
                video_title = info_result.get("title") if isinstance(info_result, dict) else ""
                with self.lock:
                    self.video_title = video_title or "YouTube Video"

            # Determine final destination file
            final_file: Path | None = None
            if downloaded_files:
                for f in reversed(downloaded_files):
                    cand = Path(f)
                    if cand.is_file() and cand.stat().st_size > 0:
                        final_file = cand.resolve()
                        break

            if not final_file:
                # Find most recently modified file in out_dir
                candidates = list(out_dir.glob("*.*"))
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    for c in candidates:
                        if c.suffix.lower() in (".mp4", ".m4a", ".mp3", ".webm", ".mkv"):
                            final_file = c.resolve()
                            break

            with self.lock:
                self.running = False
                if self.cancelled:
                    self.stage = "Download annullato"
                    self.lines.append("🛑 Download interrotto con successo.")
                elif final_file and final_file.is_file() and final_file.stat().st_size > 0:
                    self.done = True
                    self.progress = 100.0
                    self.stage = "Download completato!"
                    self.output_file = str(final_file)
                    size_mb = round(final_file.stat().st_size / (1024 * 1024), 2)
                    self.lines.append(f"🎉 File completato: {final_file.name} ({size_mb} MB)")
                else:
                    self.error = "File di output non trovato o download incompleto."
                    self.stage = "Errore durante il download"
                    self.lines.append(f"❌ {self.error}")

        except DownloadCancelled:
            with self.lock:
                self.running = False
                self.cancelled = True
                self.stage = "Download interrotto"
                self.lines.append("🛑 Download interrotto dall'utente.")
        except Exception as exc:
            with self.lock:
                self.running = False
                self.error = str(exc)
                self.stage = "Errore"
                self.lines.append(f"❌ Errore imprevisto: {exc}")


# Global YouTube job manager instance
YOUTUBE_JOB = YouTubeDownloadJob()


def get_python_exe() -> str:
    """Return best python interpreter path, preferring .venv if present."""
    venv_py = ROOT / ".venv" / "bin" / ("python.exe" if sys.platform == "win32" else "python3")
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def get_available_models() -> list[dict[str, str]]:
    """Return list of available models for pipeline selection."""
    models: list[dict[str, str]] = [
        {"id": "yamnet", "name": "⭐ YAMNet Predefinito (Modello universale)", "type": "builtin"}
    ]
    seen = set()
    for folder in (ROOT / "models", ROOT):
        if folder.is_dir():
            for p in sorted(folder.glob("*.npz"), key=lambda x: x.name.lower()):
                if p.name not in seen:
                    seen.add(p.name)
                    models.append({
                        "id": p.name,
                        "name": f"🧠 {p.stem} ({p.name})",
                        "type": "npz",
                        "path": str(p),
                    })
    return models


def normalize_playlist_url(url: str) -> str:
    """Normalize a YouTube URL containing a playlist parameter into a standard playlist URL."""
    url = str(url).strip()
    if not url:
        return url
    try:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(url)
        if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
            qs = parse_qs(parsed.query)
            list_id = qs.get("list", [None])[0]
            if list_id:
                return f"https://www.youtube.com/playlist?list={list_id}"
    except Exception:
        pass
    return url


def fetch_playlist_info(url: str) -> dict[str, Any]:
    """Extract playlist metadata and list of entries without downloading media."""
    url = normalize_playlist_url(url)
    ff_path = ensure_ffmpeg_in_path()
    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
    }
    if ff_path:
        ydl_opts["ffmpeg_location"] = ff_path

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if not info:
            raise ValueError("Impossibile estrarre informazioni dal link fornito.")

        entries_raw = info.get("entries")
        entries = []
        if entries_raw:
            for item in entries_raw:
                if not item:
                    continue
                v_id = item.get("id")
                if not v_id:
                    continue
                v_url = item.get("url") or f"https://www.youtube.com/watch?v={v_id}"
                if not v_url.startswith("http"):
                    v_url = f"https://www.youtube.com/watch?v={v_id}"
                dur = item.get("duration")

                thumbs = item.get("thumbnails") or []
                thumb_url = ""
                if isinstance(thumbs, list) and thumbs:
                    thumb_url = thumbs[-1].get("url") or ""
                elif isinstance(item.get("thumbnail"), str):
                    thumb_url = item.get("thumbnail")

                entries.append({
                    "id": v_id,
                    "title": item.get("title") or f"Video {v_id}",
                    "url": v_url,
                    "duration": dur,
                    "duration_str": format_seconds(dur),
                    "uploader": item.get("uploader") or item.get("channel") or "",
                    "thumbnail": thumb_url,
                })
        else:
            # Single video fallback if a non-playlist link was provided
            v_id = info.get("id")
            if v_id:
                dur = info.get("duration")
                entries.append({
                    "id": v_id,
                    "title": info.get("title") or f"Video {v_id}",
                    "url": url,
                    "duration": dur,
                    "duration_str": format_seconds(dur),
                    "uploader": info.get("uploader") or info.get("channel") or "",
                    "thumbnail": info.get("thumbnail") or "",
                })

        title = info.get("title") or "Playlist YouTube"
        uploader = info.get("uploader") or info.get("channel") or ""

        return {
            "title": title,
            "uploader": uploader,
            "video_count": len(entries),
            "entries": entries,
        }


class YouTubePlaylistPipelineJob:
    """Manages the end-to-end automated pipeline for YouTube playlists:
    1. Downloads each video with yt-dlp to a temporary folder
    2. Runs AutoCut audio detection & cuts events into destination folder
    3. Deletes the original downloaded video to save disk space
    4. Handles errors gracefully by continuing with the next video
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.running: bool = False
        self.done: bool = False
        self.cancelled: bool = False
        self.error: str | None = None
        self.progress: float = 0.0
        self.stage: str = "In attesa"
        self.current_video_idx: int = 0
        self.total_videos: int = 0
        self.current_video_title: str = ""
        self.current_stage: str = ""
        self.current_video_progress: float = 0.0
        self.speed_str: str = ""
        self.eta_str: str = ""
        self.lines: list[str] = []
        self.saved_files: list[dict[str, Any]] = []
        self.processed_count: int = 0
        self.active_proc: subprocess.Popen | None = None
        self.active_temp_file: Path | None = None

    def log(self, msg: str) -> None:
        with self.lock:
            self.lines.append(str(msg))

    def start(self, params: dict) -> tuple[bool, str]:
        with self.lock:
            if self.running:
                return False, "Una pipeline playlist è già in corso."

            self.running = True
            self.done = False
            self.cancelled = False
            self.error = None
            self.progress = 0.0
            self.stage = "Inizializzazione playlist..."
            self.current_video_idx = 0
            self.total_videos = 0
            self.current_video_title = ""
            self.current_stage = "Avvio"
            self.current_video_progress = 0.0
            self.speed_str = ""
            self.eta_str = ""
            self.lines = []
            self.saved_files = []
            self.processed_count = 0
            self.active_proc = None
            self.active_temp_file = None

        thread = threading.Thread(target=self._run_worker, args=(params,), daemon=True)
        thread.start()
        return True, "Pipeline playlist avviata con successo."

    def stop(self) -> bool:
        with self.lock:
            if not self.running:
                return False
            self.cancelled = True
            self.stage = "Interruzione in corso..."
            self.lines.append("🛑 Richiesta di interruzione della pipeline dall'utente...")
            proc = self.active_proc
            temp_file = self.active_temp_file

        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

        if temp_file and temp_file.is_file():
            try:
                temp_file.unlink()
            except Exception:
                pass

        return True

    def get_status(self, since: int = 0) -> dict[str, Any]:
        with self.lock:
            lines = self.lines[since:]
            return {
                "running": self.running,
                "done": self.done,
                "cancelled": self.cancelled,
                "error": self.error,
                "progress": round(self.progress, 1),
                "stage": self.stage,
                "current_video_idx": self.current_video_idx,
                "total_videos": self.total_videos,
                "current_video_title": self.current_video_title,
                "current_stage": self.current_stage,
                "current_video_progress": round(self.current_video_progress, 1),
                "speed": self.speed_str,
                "eta": self.eta_str,
                "lines": lines,
                "saved_files": list(self.saved_files),
                "processed_count": self.processed_count,
            }

    def _run_worker(self, params: dict) -> None:
        try:
            self.log("🚀 Avvio Pipeline Playlist AutoCut...")
            ff_path = ensure_ffmpeg_in_path()
            if ff_path:
                self.log(f"🎬 FFmpeg rilevato: {ff_path}")
            else:
                self.log("⚠️ FFmpeg non trovato. Verifica che ffmpeg sia installato.")

            py_exe = get_python_exe()
            self.log(f"🐍 Interprete Python: {py_exe}")

            raw_url = normalize_playlist_url(str(params.get("url", "")).strip())
            if not raw_url:
                raise ValueError("Inserisci un link YouTube valido.")

            quality = str(params.get("quality", "best")).strip().lower()
            model_choice = str(params.get("model", "yamnet")).strip()
            sound_type = str(params.get("sound_type", "laughter")).strip()
            threshold = params.get("threshold")
            pad = params.get("pad")
            min_duration = params.get("min_duration")
            safe = bool(params.get("safe", False))

            out_dir_param = str(params.get("outdir", "results")).strip()
            out_dir = Path(out_dir_param) if out_dir_param else (ROOT / "results")
            if not out_dir.is_absolute():
                out_dir = ROOT / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            self.log(f"📁 Cartella salvataggio segmenti: {out_dir}")

            temp_dir = ROOT / "downloads" / "pipeline_temp"
            temp_dir.mkdir(parents=True, exist_ok=True)

            self.log(f"📋 Estrazione lista video dalla playlist...")
            pl_info = fetch_playlist_info(raw_url)
            entries = pl_info.get("entries", [])
            total_videos = len(entries)

            if total_videos == 0:
                raise ValueError("Nessun video trovato nella playlist o link non valido.")

            with self.lock:
                self.total_videos = total_videos
                self.stage = f"Playlist: {pl_info.get('title', 'YouTube')} ({total_videos} video)"

            self.log(f"📑 Playlist caricata: '{pl_info.get('title')}' — {total_videos} video trovati.")
            self.log(f"🧠 Modello selezionato per l'analisi: {model_choice}")

            # Process each video in sequence
            for i, entry in enumerate(entries):
                with self.lock:
                    if self.cancelled:
                        break

                idx = i + 1
                v_title = entry.get("title", f"Video {idx}")
                v_url = entry.get("url")
                base_name = f"{idx:02d} - {sanitize_filename(v_title)}"

                with self.lock:
                    self.current_video_idx = idx
                    self.current_video_title = v_title
                    self.current_stage = "Download"
                    self.current_video_progress = 0.0
                    self.speed_str = ""
                    self.eta_str = ""
                    overall_base = (i / total_videos) * 100.0
                    self.progress = overall_base
                    self.stage = f"Video {idx}/{total_videos}: Download in corso..."

                self.log(f"\n------------------------------------------------------------")
                self.log(f"▶️ [{idx}/{total_videos}] Avvio elaborazione: '{v_title}'")
                self.log(f"🔗 Link: {v_url}")

                # 1. Download with yt-dlp
                downloaded_file: Path | None = None
                try:
                    out_template = str(temp_dir / f"{base_name}.%(ext)s")
                    ydl_opts: dict[str, Any] = {
                        "quiet": True,
                        "no_warnings": True,
                        "no_color": True,
                        "noprogress": False,
                        "ffmpeg_location": ff_path,
                        "merge_output_format": "mp4",
                        "outtmpl": out_template,
                    }

                    m_height = re.search(r"(\d+)p?", quality)
                    if m_height:
                        max_h = int(m_height.group(1))
                        ydl_opts["format"] = (
                            f"bestvideo[height<={max_h}][ext=mp4]+bestaudio[ext=m4a]/"
                            f"bestvideo[height<={max_h}]+bestaudio/"
                            f"best[height<={max_h}]/best"
                        )
                    else:
                        ydl_opts["format"] = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best"

                    def progress_hook(d: dict[str, Any]) -> None:
                        with self.lock:
                            if self.cancelled:
                                raise DownloadCancelled("Interruzione richiesta.")
                        if d.get("status") == "downloading":
                            raw_pct = d.get("downloaded_bytes", 0) / (d.get("total_bytes") or d.get("total_bytes_estimate") or 1) * 100.0
                            raw_pct = max(0.0, min(100.0, raw_pct))
                            spd = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", str(d.get("_speed_str", ""))).strip()
                            eta = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", str(d.get("_eta_str", ""))).strip()
                            with self.lock:
                                self.current_video_progress = raw_pct
                                self.speed_str = spd
                                self.eta_str = eta
                                self.progress = min(99.9, overall_base + (raw_pct * 0.45 / total_videos))

                    ydl_opts["progress_hooks"] = [progress_hook]

                    self.log(f"⬇️ [{idx}/{total_videos}] Download video da YouTube...")
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.extract_info(v_url, download=True)

                    # Locate the downloaded file
                    for cand in temp_dir.glob(f"{base_name}.*"):
                        if cand.is_file() and cand.stat().st_size > 0:
                            downloaded_file = cand
                            break

                    if not downloaded_file or not downloaded_file.is_file():
                        self.log(f"⚠️ [{idx}/{total_videos}] File non trovato dopo il download per '{v_title}'. Salto al successivo.")
                        continue

                except DownloadCancelled:
                    self.log(f"🛑 Interruzione richiesta durante il download.")
                    break
                except Exception as exc:
                    self.log(f"⚠️ [{idx}/{total_videos}] Errore download per '{v_title}': {exc}. Continuo con il prossimo video della playlist.")
                    # Clean up partial files
                    for cand in temp_dir.glob(f"{base_name}.*"):
                        try:
                            cand.unlink()
                        except Exception:
                            pass
                    continue

                with self.lock:
                    if self.cancelled:
                        if downloaded_file and downloaded_file.exists():
                            downloaded_file.unlink()
                        break
                    self.active_temp_file = downloaded_file
                    self.current_stage = "Analisi & Taglio"
                    self.stage = f"Video {idx}/{total_videos}: Analisi AutoCut e ritaglio..."

                # 2. Run AutoCut audio detection & cutting
                file_mb = round(downloaded_file.stat().st_size / (1024 * 1024), 1)
                self.log(f"🧠 [{idx}/{total_videos}] Avvio analisi AutoCut su: {downloaded_file.name} ({file_mb} MB)...")

                cmd = [
                    py_exe,
                    str(ROOT / "autocut.py"),
                    str(downloaded_file),
                    "--export",
                    "events",
                    "--outdir",
                    str(out_dir),
                ]

                # Model configuration
                if model_choice and model_choice != "yamnet":
                    m_path = ROOT / "models" / model_choice if (ROOT / "models" / model_choice).is_file() else ROOT / model_choice
                    if m_path.is_file():
                        cmd += ["--load-model", str(m_path)]
                    else:
                        self.log(f"⚠️ Modello '{model_choice}' non trovato come file .npz, uso YAMNet.")
                else:
                    if sound_type:
                        cmd += ["--sound-type", sound_type]

                if threshold not in (None, ""):
                    cmd += ["--threshold", str(threshold)]
                if pad not in (None, ""):
                    cmd += ["--pad", str(pad)]
                if min_duration not in (None, ""):
                    cmd += ["--min-duration", str(min_duration)]
                if safe:
                    cmd += ["--safe"]

                env = dict(os.environ)
                env.setdefault("PYTHONUNBUFFERED", "1")
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )

                with self.lock:
                    self.active_proc = proc

                # Stream subprocess output live
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if line_str:
                        # Log meaningful progress lines
                        if any(k in line_str.lower() for k in ("detected", "event", "evaluating", "extracting", "progress", "exporting", "done", "total")):
                            self.log(f"   {line_str}")

                proc.wait()
                with self.lock:
                    self.active_proc = None

                with self.lock:
                    if self.cancelled:
                        if downloaded_file and downloaded_file.exists():
                            downloaded_file.unlink()
                        break

                # 3. Check for generated cut file
                expected_cut = out_dir / f"{downloaded_file.stem}_events_only.mp4"
                if expected_cut.is_file() and expected_cut.stat().st_size > 0:
                    cut_size_mb = round(expected_cut.stat().st_size / (1024 * 1024), 2)
                    self.log(f"✅ [{idx}/{total_videos}] Segmenti salvati con successo: {expected_cut.name} ({cut_size_mb} MB)")
                    with self.lock:
                        self.saved_files.append({
                            "title": v_title,
                            "filename": expected_cut.name,
                            "path": str(expected_cut),
                            "size_mb": cut_size_mb,
                        })
                else:
                    self.log(f"ℹ️ [{idx}/{total_videos}] Nessun segmento rilevato per '{v_title}'.")

                # 4. Delete the original downloaded video to free disk space
                try:
                    if downloaded_file.is_file():
                        downloaded_file.unlink()
                        self.log(f"🗑️ [{idx}/{total_videos}] Video originale eliminato dal disco.")
                except Exception as del_err:
                    self.log(f"⚠️ Impossibile eliminare video temporaneo: {del_err}")

                with self.lock:
                    self.active_temp_file = None
                    self.processed_count += 1
                    self.progress = round((idx / total_videos) * 100.0, 1)

            # Pipeline loop finished
            with self.lock:
                self.running = False
                if self.cancelled:
                    self.stage = "Pipeline interrotta"
                    self.lines.append("🛑 Pipeline interrotta con successo.")
                else:
                    self.done = True
                    self.progress = 100.0
                    self.stage = "Pipeline completata!"
                    self.lines.append(f"\n🎉 Pipeline completata con successo! Salvati {len(self.saved_files)} file riassuntivi in '{out_dir.name}'.")

        except Exception as exc:
            with self.lock:
                self.running = False
                self.error = str(exc)
                self.stage = "Errore durante la pipeline"
                self.lines.append(f"❌ Errore pipeline: {exc}")


# Global YouTube playlist pipeline job manager instance
YOUTUBE_PIPELINE_JOB = YouTubePlaylistPipelineJob()

