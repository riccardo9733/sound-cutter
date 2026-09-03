#!/bin/bash
# Double-clickable launcher for macOS: activates the project's Python
# environment and runs autocut.py with whatever arguments you pass.
# Usage from Terminal:
#   ./autocut.command video.mp4 --train-from reference.mp4
#   ./autocut.command --ui        (full web interface)
# Or double-click to open the full web UI; drag & drop a video onto it
# passes the video to the command line.
cd "$(dirname "$0")" || exit 1

if [ ! -d ".venv" ]; then
  echo "ERROR: .venv not found. Run first:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  read -r -p "Press Enter to close..."
  exit 1
fi

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found — is the virtualenv broken?"
  read -r -p "Press Enter to close..."
  exit 1
fi

SCRIPT="autocut.py"

if [ "$#" -eq 0 ]; then
  echo "Apro l'interfaccia AutoCut nel browser... (Ctrl+C per chiudere)"
  "$PY" "$SCRIPT" --ui
else
  "$PY" "$SCRIPT" "$@"
fi

echo
read -r -p "Premi Invio per chiudere... " _
