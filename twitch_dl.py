#!/usr/bin/env python3
"""TwitchDownloader CLI integration for AutoCut.

Handles:
  - Auto-locating or auto-downloading TwitchDownloaderCLI (macOS arm64/x64, Linux, Windows)
  - Locating ffmpeg (system PATH or bundled in .venv imageio_ffmpeg)
  - Managing background download jobs for Twitch VODs and Clips
  - Real-time progress parsing, stage tracking, log streaming, and job cancellation
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN_DIR = ROOT / "bin"
DOWNLOADS_DIR = ROOT / "downloads"
OAUTH_FILE = ROOT / ".twitch_oauth"

TWITCH_DL_VERSION = "1.56.5"
GITHUB_RELEASE_BASE = (
    f"https://github.com/lay295/TwitchDownloader/releases/download/{TWITCH_DL_VERSION}"
)


def load_saved_oauth() -> str:
    """Load saved OAuth token from local file if present."""
    if OAUTH_FILE.is_file():
        try:
            return OAUTH_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return ""


def save_oauth_token(token: str) -> None:
    """Save or remove OAuth token in local file."""
    token = token.strip()
    if token:
        OAUTH_FILE.write_text(token, encoding="utf-8")
        try:
            OAUTH_FILE.chmod(0o600)
        except Exception:
            pass
    elif OAUTH_FILE.is_file():
        OAUTH_FILE.unlink(missing_ok=True)


def find_ffmpeg() -> str | None:
    """Find a usable ffmpeg executable, checking system PATH and virtualenv."""
    # 1. System PATH
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg

    # 2. imageio_ffmpeg in python environment
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe) and os.access(exe, os.X_OK):
            return exe
    except Exception:
        pass

    # 3. Look in .venv binaries folder
    venv_dir = ROOT / ".venv"
    if venv_dir.is_dir():
        for p in venv_dir.glob("**/ffmpeg*"):
            if p.is_file() and os.access(p, os.X_OK) and not p.name.endswith((".py", ".pyc", ".txt", ".json")):
                return str(p)

    return None


def find_twitch_cli() -> Path | None:
    """Check if TwitchDownloaderCLI exists in bin/ or in PATH."""
    exe_name = "TwitchDownloaderCLI.exe" if platform.system() == "Windows" else "TwitchDownloaderCLI"
    local_bin = BIN_DIR / exe_name
    if local_bin.is_file() and os.access(local_bin, os.X_OK):
        return local_bin

    in_path = shutil.which(exe_name)
    if in_path:
        return Path(in_path)

    return None


def get_platform_asset_name() -> str:
    """Get the appropriate TwitchDownloader release zip name for the current OS and arch."""
    sys_name = platform.system().lower()
    machine = platform.machine().lower()

    if sys_name == "darwin":
        if "arm" in machine or "aarch" in machine:
            return f"TwitchDownloaderCLI-{TWITCH_DL_VERSION}-MacOSArm64.zip"
        return f"TwitchDownloaderCLI-{TWITCH_DL_VERSION}-MacOS-x64.zip"
    elif sys_name == "linux":
        if "arm64" in machine or "aarch64" in machine:
            return f"TwitchDownloaderCLI-{TWITCH_DL_VERSION}-LinuxArm64.zip"
        elif "arm" in machine:
            return f"TwitchDownloaderCLI-{TWITCH_DL_VERSION}-LinuxArm.zip"
        return f"TwitchDownloaderCLI-{TWITCH_DL_VERSION}-Linux-x64.zip"
    elif sys_name == "windows":
        return f"TwitchDownloaderCLI-{TWITCH_DL_VERSION}-Windows-x64.zip"
    else:
        raise RuntimeError(f"Sistema operativo non supportato: {sys_name} ({machine})")


def ensure_twitch_cli(log_cb=None) -> Path:
    """Locate TwitchDownloaderCLI or download it automatically from GitHub Releases."""
    existing = find_twitch_cli()
    if existing:
        return existing

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    asset_name = get_platform_asset_name()
    url = f"{GITHUB_RELEASE_BASE}/{asset_name}"
    zip_path = BIN_DIR / asset_name

    if log_cb:
        log_cb(f"📥 Download TwitchDownloaderCLI ({asset_name}) in corso...")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AutoCut-TwitchDownloader"})
        with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out_file:
            shutil.copyfileobj(resp, out_file)

        if log_cb:
            log_cb("📦 Estrazione binario in bin/...")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(BIN_DIR)

        zip_path.unlink(missing_ok=True)

        exe_name = "TwitchDownloaderCLI.exe" if platform.system() == "Windows" else "TwitchDownloaderCLI"
        target_exe = BIN_DIR / exe_name
        if not target_exe.is_file():
            # Check nested files in case zip extracted with directory
            for f in BIN_DIR.glob(f"**/{exe_name}"):
                if f.is_file():
                    shutil.move(str(f), str(target_exe))
                    break

        if target_exe.is_file():
            target_exe.chmod(target_exe.stat().st_mode | 0o755)
            if log_cb:
                log_cb(f"✅ TwitchDownloaderCLI pronto: {target_exe}")
            return target_exe
        else:
            raise FileNotFoundError(f"Eseguibile {exe_name} non trovato nello zip scaricato.")

    except Exception as exc:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"Errore durante l'installazione di TwitchDownloaderCLI: {exc}") from exc


def is_twitch_clip(url_or_id: str) -> bool:
    """Determine whether the input is a Twitch Clip or a VOD."""
    s = str(url_or_id).strip().lower()
    if "clips.twitch.tv" in s:
        return True
    if "/clip/" in s:
        return True
    # If it is purely alphabetic (like 'BashfulTameFalconKreygasm'), it's often a clip slug
    if re.match(r"^[A-Za-z]+-[A-Za-z0-9_-]+$", s) or re.match(r"^[A-Z][a-z]+[A-Z][a-z]+[A-Za-z0-9]+$", s):
        return True
    return False


def sanitize_filename(name: str) -> str:
    """Sanitize string for safe filenames."""
    s = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:100] if s else "twitch_media"


class TwitchDownloadJob:
    """Manages background download of a Twitch VOD or Clip."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.running: bool = False
        self.done: bool = False
        self.cancelled: bool = False
        self.error: str | None = None
        self.progress: int = 0
        self.stage: str = "In attesa"
        self.lines: list[str] = []
        self.output_file: str | None = None
        self.target_video_path: Path | None = None
        self.command_str: str = ""

    def log(self, msg: str):
        with self.lock:
            self.lines.append(msg)

    def start(self, params: dict) -> tuple[bool, str]:
        with self.lock:
            if self.running:
                return False, "Un download è già in corso."

            self.running = True
            self.done = False
            self.cancelled = False
            self.error = None
            self.progress = 0
            self.stage = "Inizializzazione"
            self.lines = []
            self.output_file = None
            self.target_video_path = None

        self.thread = threading.Thread(target=self._run_worker, args=(params,), daemon=True)
        self.thread.start()
        return True, "Download avviato."

    def stop(self) -> bool:
        with self.lock:
            if not self.running or not self.proc:
                return False
            self.cancelled = True
            self.stage = "Interruzione in corso..."
            self.lines.append("🛑 Richiesta di interruzione da parte dell'utente...")
            proc = self.proc

        try:
            proc.terminate()
        except Exception:
            pass

        def force_kill():
            import time
            time.sleep(2.0)
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass

        threading.Thread(target=force_kill, daemon=True).start()
        return True

    def get_status(self, since: int = 0) -> dict:
        with self.lock:
            lines = self.lines[since:]
            return {
                "running": self.running,
                "done": self.done,
                "cancelled": self.cancelled,
                "error": self.error,
                "progress": self.progress,
                "stage": self.stage,
                "lines": lines,
                "output_file": self.output_file,
                "command": self.command_str,
            }

    def _run_worker(self, params: dict):
        try:
            self.log("🔍 Verifica requisiti TwitchDownloader e FFmpeg...")
            cli_path = ensure_twitch_cli(log_cb=self.log)
            ffmpeg_path = find_ffmpeg()

            if ffmpeg_path:
                self.log(f"🎬 FFmpeg rilevato: {ffmpeg_path}")
            else:
                self.log("⚠️ FFmpeg non trovato. Alcuni passaggi di unione video potrebbero fallire.")

            raw_url = str(params.get("url", "")).strip()
            if not raw_url:
                raise ValueError("Inserisci un link o un ID Twitch valido.")

            is_clip = bool(params.get("is_clip")) or is_twitch_clip(raw_url)
            quality = str(params.get("quality", "")).strip()
            trim_start = str(params.get("trim_start", "")).strip()
            trim_end = str(params.get("trim_end", "")).strip()
            oauth = str(params.get("oauth", "")).strip()
            if not oauth:
                oauth = load_saved_oauth()
            elif oauth:
                save_oauth_token(oauth)
            threads = int(params.get("threads") or 4)
            out_dir_param = str(params.get("outdir", "")).strip()
            custom_filename = str(params.get("filename", "")).strip()

            out_dir = Path(out_dir_param) if out_dir_param else DOWNLOADS_DIR
            if not out_dir.is_absolute():
                out_dir = ROOT / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            # Extract an identifier for default naming
            safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", raw_url.split("/")[-1].split("?")[0])
            if not safe_id:
                safe_id = "video"

            if custom_filename:
                fname = sanitize_filename(custom_filename)
                if not fname.lower().endswith(".mp4") and not fname.lower().endswith(".m4a"):
                    fname += ".mp4"
            else:
                prefix = "twitch_clip" if is_clip else "twitch_vod"
                fname = f"{prefix}_{safe_id}.mp4"

            dest_path = (out_dir / fname).resolve()
            self.target_video_path = dest_path

            # Build command
            cmd = [str(cli_path)]
            if is_clip:
                cmd += ["clipdownload", "-u", raw_url, "-o", str(dest_path), "--collision", "Overwrite"]
                if quality:
                    cmd += ["-q", quality]
                if ffmpeg_path:
                    cmd += ["--ffmpeg-path", ffmpeg_path]
            else:
                cmd += ["videodownload", "-u", raw_url, "-o", str(dest_path), "--collision", "Overwrite"]
                if quality:
                    cmd += ["-q", quality]
                if trim_start:
                    cmd += ["-b", trim_start]
                if trim_end:
                    cmd += ["-e", trim_end]
                if threads > 0:
                    cmd += ["-t", str(threads)]
                if oauth:
                    cmd += ["--oauth", oauth]
                if ffmpeg_path:
                    cmd += ["--ffmpeg-path", ffmpeg_path]

            with self.lock:
                # Mask oauth token in displayed command for privacy
                safe_cmd = []
                hide_next = False
                for part in cmd:
                    if hide_next:
                        safe_cmd.append("********")
                        hide_next = False
                    else:
                        if part == "--oauth":
                            hide_next = True
                        safe_cmd.append(part)
                self.command_str = " ".join(safe_cmd)
                self.stage = "Connessione a Twitch..."

            self.log(f"🚀 Comando: {self.command_str}")
            self.log(f"📁 Destinazione: {dest_path}")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            with self.lock:
                self.proc = proc

            percent_re = re.compile(r"(\d+(?:\.\d+)?)%")
            status_re = re.compile(r"\[STATUS\]\s*-\s*([^\r\n]+)", re.IGNORECASE)
            detected_error = None

            for line in proc.stdout:
                line_str = line.strip()
                if not line_str:
                    continue

                self.log(line_str)

                low = line_str.lower()
                if "insufficient access to vod" in low or "oauth may be required" in low:
                    detected_error = "Accesso insufficiente: questo VOD è riservato ai soli abbonati (Sub-Only). Per scaricarlo inserisci il token OAuth del tuo account Twitch abbonato."
                elif "invalid vod" in low or "deleted/expired vod" in low:
                    detected_error = "VOD non valido o scaduto/cancellato da Twitch."

                # Parse progress and stage
                m_pct = percent_re.search(line_str)
                pct_val = None
                if m_pct:
                    try:
                        pct_val = int(round(float(m_pct.group(1))))
                    except Exception:
                        pass

                stage_name = None
                m_st = status_re.search(line_str)
                if m_st:
                    stage_name = m_st.group(1).strip()
                elif "downloading" in line_str.lower():
                    stage_name = "Download frammenti..."
                elif "combining" in line_str.lower() or "ffmpeg" in line_str.lower():
                    stage_name = "Unione video con FFmpeg..."
                elif "finalizing" in line_str.lower() or "done" in line_str.lower():
                    stage_name = "Finalizzazione file..."

                with self.lock:
                    if pct_val is not None:
                        self.progress = max(self.progress, min(100, pct_val))
                    if stage_name:
                        self.stage = stage_name

            proc.wait()
            rc = proc.returncode

            with self.lock:
                self.running = False
                if self.cancelled:
                    self.stage = "Download annullato"
                    self.lines.append("🛑 Download interrotto con successo.")
                elif rc == 0 and dest_path.is_file() and dest_path.stat().st_size > 0:
                    self.done = True
                    self.progress = 100
                    self.stage = "Download completato!"
                    self.output_file = str(dest_path)
                    size_mb = round(dest_path.stat().st_size / 1e6, 2)
                    self.lines.append(f"🎉 File completato: {dest_path.name} ({size_mb} MB)")
                else:
                    self.error = detected_error or f"TwitchDownloader è terminato con codice di errore {rc}."
                    self.stage = "Errore durante il download"
                    self.lines.append(f"❌ {self.error}")

        except Exception as exc:
            with self.lock:
                self.running = False
                self.error = str(exc)
                self.stage = "Errore"
                self.lines.append(f"❌ Errore imprevisto: {exc}")


# Global job manager
TWITCH_JOB = TwitchDownloadJob()
