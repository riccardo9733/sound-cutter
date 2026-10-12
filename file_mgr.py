#!/usr/bin/env python3
"""File manager helper for AutoCut workspace.

Scans and organizes files across project folders:
downloads/, results/, models/, bites/, th/, eno/, test/, and root.
Provides safe reveal, delete, and rename operations.
"""

from __future__ import annotations

import datetime
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

MEDIA_EXTS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg",
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma",
}

PROTECTED_FILES = {
    "app_ui.py", "autocut.py", "youtube_dl.py", "twitch_dl.py", "wave_ui.py",
    "file_mgr.py", "requirements.txt", "autocut.command", "README.md",
}

KNOWN_FOLDERS = [
    ("downloads", "📥 downloads", "Video e audio scaricati da YouTube o Twitch"),
    ("results", "✂️ results", "Esportazioni e tagli automatici generati da AutoCut"),
    ("bites", "🎯 bites", "Video elaborati e frammenti"),
    ("th", "🎯 th", "Registrazioni e tagli Twitch"),
    ("eno", "🎯 eno", "Campioni ed estratti audio"),
    ("models", "🧠 models", "Modelli di classificazione sonori (.npz)"),
    ("test", "🧪 test", "Video e file di test"),
    (".", "🏠 Cartella Principale (Root)", "File multimediali e script nella directory base"),
]


def format_size(num_bytes: int) -> str:
    """Format bytes into human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_file_type(p: Path) -> str:
    """Determine file classification."""
    ext = p.suffix.lower()
    if ext in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg"}:
        return "video"
    elif ext in {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}:
        return "audio"
    elif ext == ".npz":
        return "model"
    elif ext in {".txt", ".json", ".csv", ".md"}:
        return "text"
    return "other"


def get_file_icon(ftype: str, ext: str) -> str:
    """Return appropriate emoji icon for file."""
    if ftype == "video":
        return "🎬"
    elif ftype == "audio":
        return "🎵"
    elif ftype == "model":
        return "🧠"
    elif ftype == "text":
        return "📄"
    return "📁" if ext == "" else "📦"


def get_project_tree() -> dict[str, Any]:
    """Scan project directories and return structured file data."""
    folders_data = []
    total_files_count = 0
    total_workspace_bytes = 0

    seen_folders = set()

    # Scan known and custom existing subdirectories
    folder_candidates: list[tuple[str, str, str]] = list(KNOWN_FOLDERS)

    for item in sorted(ROOT.iterdir(), key=lambda p: p.name.lower()):
        if item.is_dir() and not item.name.startswith((".", "__")) and item.name not in ("bin", ".venv"):
            rel = item.name
            if rel not in [f[0] for f in folder_candidates]:
                folder_candidates.append((rel, f"📁 {rel}", "Cartella di progetto"))

    for rel_key, label, desc in folder_candidates:
        if rel_key in seen_folders:
            continue
        seen_folders.add(rel_key)

        dir_path = (ROOT / rel_key).resolve()
        if not dir_path.is_dir():
            continue

        folder_files = []
        folder_bytes = 0

        try:
            # For root, only direct files (skip subfolders)
            # For subfolders, scan direct files
            entries = sorted(dir_path.iterdir(), key=lambda q: (not q.is_file(), q.name.lower()))
            for p in entries:
                if not p.is_file() or p.name.startswith("."):
                    continue

                try:
                    stat = p.stat()
                except Exception:
                    continue

                size = stat.st_size
                mtime = stat.st_mtime
                ext = p.suffix.lower()
                ftype = get_file_type(p)

                # Format mtime
                try:
                    mtime_dt = datetime.datetime.fromtimestamp(mtime)
                    mtime_str = mtime_dt.strftime("%d/%m/%Y %H:%M")
                except Exception:
                    mtime_str = "--"

                is_media = ext in MEDIA_EXTS
                is_protected = (p.name in PROTECTED_FILES) or ext in (".py", ".command")

                folder_files.append({
                    "name": p.name,
                    "path": str(p),
                    "rel_path": str(p.relative_to(ROOT)),
                    "folder": rel_key,
                    "size": size,
                    "size_str": format_size(size),
                    "mtime": mtime,
                    "mtime_str": mtime_str,
                    "ext": ext,
                    "type": ftype,
                    "icon": get_file_icon(ftype, ext),
                    "is_media": is_media,
                    "is_protected": is_protected,
                })
                folder_bytes += size

        except Exception:
            pass

        # Sort files: most recently modified first
        folder_files.sort(key=lambda x: x["mtime"], reverse=True)

        total_files_count += len(folder_files)
        total_workspace_bytes += folder_bytes

        folders_data.append({
            "key": rel_key,
            "label": label,
            "desc": desc,
            "path": str(dir_path),
            "file_count": len(folder_files),
            "size": folder_bytes,
            "size_str": format_size(folder_bytes),
            "files": folder_files,
        })

    return {
        "folders": folders_data,
        "total_files": total_files_count,
        "total_size": total_workspace_bytes,
        "total_size_str": format_size(total_workspace_bytes),
        "root_path": str(ROOT),
    }


def reveal_in_os(path_str: str) -> bool:
    """Reveal file or open folder in OS native file manager."""
    try:
        p = Path(path_str).resolve()
        if not p.is_relative_to(ROOT):
            return False

        if sys.platform == "darwin":
            if p.is_file():
                subprocess.Popen(["open", "-R", str(p)])
            else:
                subprocess.Popen(["open", str(p)])
            return True
        elif sys.platform == "win32":
            if p.is_file():
                subprocess.Popen(["explorer", "/select,", str(p)])
            else:
                subprocess.Popen(["explorer", str(p)])
            return True
        else:
            target = str(p.parent if p.is_file() else p)
            subprocess.Popen(["xdg-open", target])
            return True
    except Exception:
        return False


def delete_project_file(path_str: str) -> tuple[bool, str]:
    """Safely delete a project file."""
    try:
        p = Path(path_str).resolve()
        if not p.is_file():
            return False, "File non trovato."
        if not p.is_relative_to(ROOT):
            return False, "Operazione non consentita all'esterno del progetto."
        if p.name in PROTECTED_FILES or p.suffix.lower() in (".py", ".command"):
            return False, "Non è consentito eliminare file di sistema o codice sorgente."

        p.unlink()
        return True, "File eliminato con successo."
    except Exception as exc:
        return False, f"Errore durante l'eliminazione: {exc}"


def rename_project_file(path_str: str, new_name: str) -> tuple[bool, str, str | None]:
    """Safely rename a file within the same directory."""
    try:
        p = Path(path_str).resolve()
        if not p.is_file():
            return False, "File non trovato.", None
        if not p.is_relative_to(ROOT):
            return False, "Operazione non consentita all'esterno del progetto.", None
        if p.name in PROTECTED_FILES or p.suffix.lower() in (".py", ".command"):
            return False, "Non è consentito rinominare file di sistema o codice sorgente.", None

        clean_name = re.sub(r'[\\/*?:"<>|]', "", new_name).strip()
        if not clean_name:
            return False, "Nome file non valido.", None

        # Keep original extension if the user didn't specify one
        if "." not in clean_name and p.suffix:
            clean_name += p.suffix

        target = p.with_name(clean_name)
        if target.exists() and target != p:
            return False, "Un file con questo nome esiste già.", None

        p.rename(target)
        return True, "File rinominato con successo.", str(target)
    except Exception as exc:
        return False, f"Errore durante la ridenominazione: {exc}", None
