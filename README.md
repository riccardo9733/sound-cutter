# AutoCut 🎬

Automatically edit a video by detecting specific sound events in its audio track and
exporting **two cuts**:

- `<name>_events_only.mp4` — only the target sound moments (with configurable padding)
- `<name>_clean.mp4` — everything except the detected sound events (clean cut)

Detection options:
1. **Pretrained YAMNet** (Google AudioSet): built-in detection for common acoustic events such as **laughter**, **applause**, **speech**, **coughs**, etc.
2. **Few-shot Deep Detector (`--teach`)**: learn *any* custom target sound from a few labeled positive `[+]` and negative `[-]` intervals using 1024-D deep acoustic embeddings and prototype gating.
3. **Reference Video (`--train-from`)**: learn automatically from an audio/video clip containing only the target sound events.
4. **Saved Models (`--load-model`)**: reuse lightweight `.npz` trained acoustic models across different videos instantly without TensorFlow.

---

## Requirements

- Python 3.9+
- **ffmpeg** — used for audio extraction and frame-accurate video assembly. Prefers `ffmpeg` on your PATH, or falls back to the static binary bundled with `moviepy`/`imageio-ffmpeg`:
  - macOS: `brew install ffmpeg`
  - Windows: `winget install Gyan.FFmpeg`
  - Linux: `sudo apt install ffmpeg`
- Python packages: `tensorflow`, `tensorflow-hub`, `numpy`, `moviepy` (see `requirements.txt`).

> On Apple Silicon (M1/M2/M3/M4), standard `tensorflow` (2.16+) runs natively.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

```bash
# Detect target sound events (default: laughter)
python autocut.py video.mp4

# Custom detection parameters
python autocut.py video.mp4 --sound-type applause --pad 0.8 --threshold 0.6 --min-gap 1.5 --outdir cuts

# Learn any custom sound from labeled timestamps
python autocut.py video.mp4 --teach "1:54.5-1:56.0,3:44.2-3:45.0"
```

The script prints progress and the list of detected events with timestamps:

```
[1/4] Extracting audio from video.mp4 ...
      Audio duration: 120.0 s
[2/4] Running audio analysis & embedding extraction ...
[3/4] Post-processing detections ...
      Detected 3 event(s):
         1.    3.12 s →    4.44 s (1.32 s, conf: 88%)
         2.    7.90 s →    9.22 s (1.32 s, conf: 92%)
         3.   11.04 s →   12.36 s (1.32 s, conf: 76%)
      Total event duration: 4.0 s / 120.0 s
[4/4] Building 'events only' cut (3 segment(s)) via FFmpeg ...
[4/4] Building 'clean video' cut (4 segment(s)) via FFmpeg ...

Done!
  Events only : cuts/video_events_only.mp4
  Clean video : cuts/video_clean.mp4
```

---

## Full Web Interface (`--ui`)

The complete workflow in one local browser page:

```bash
./autocut.command                # macOS: double-click to open
python autocut.py --ui           # anywhere
python autocut.py video.mp4 --ui # with a video preselected
```

The interface opens at `http://127.0.0.1:8765` with four interactive panels:

1. **Source Video & Interactive Waveform**:
   - Preview video and zoom into the high-resolution audio waveform up to 150x.
   - Mark positive intervals `[+]` with **S** and negative intervals `[-]` with **N**, fine-adjusting boundaries directly with draggable waveform handles.
   - Listen to any interval in isolation with **R** or frame-step with `[` / `]`.
2. **Detection & Training**:
   - Choose detection mode: YAMNet presets, Learn from examples (`--teach`), Reference video (`--train-from`), or Saved `.npz` model.
   - Real-time parameter controls: threshold slider, Auto-Threshold (🎯), context padding, min gap, duration limits, and noise gate.
3. **Run Pipeline**:
   - Live streamed terminal log with real-time detection feedback, cancelable anytime with **Interrompi**.
   - Scan-only iteration mode to test parameters quickly without re-encoding video.
4. **Interactive Results & Outputs**:
   - Ranked event list with confidence badges (green/amber/red).
   - Click any detected event to zoom into the waveform and preview the sound.
   - One-click feedback: **✏️ Come [+]** to refine boundaries as a positive example, or **🚫 Come [-]** to mark false positives and update the model.
   - Direct inline playback and download of both generated cuts.

---

## CLI Options

| Option | Default | Description |
|---|---|---|
| `--sound-type` | `laughter` | Pretrained YAMNet sound event: `laughter`, `applause`, `speech`, `cough`, etc. |
| `--pad` | `0.5` | Seconds of context kept around each event in the *events only* cut. |
| `--threshold` | `0.5` | Detection confidence threshold (0.0–1.0). Lower = more sensitive; higher = stricter. |
| `--auto-threshold` | off | Automatically calculates the optimal threshold from score distributions. |
| `--min-gap` | `1.0` | Merge events separated by less than this many seconds into a single continuous event. |
| `--min-duration` | `0.3` | Drop detected events shorter than this duration (false positive filter). |
| `--max-duration` | None | Drop detected events longer than this duration (e.g. filter out long background noise). |
| `--silence-threshold` | None | Noise gate in dB (e.g. `-40`): zero out detections in silent audio frames. |
| `--teach` | — | Train from labeled intervals, e.g. `--teach "1:54-1:56,-2:10-2:12"` or `--teach @etichette.txt`. |
| `--train-from` | — | Learn from a reference video/audio containing only the target sounds. |
| `--save-model` | — | Save the trained detector to a `.npz` file for reuse across other videos. |
| `--load-model` | — | Run detection using a previously saved `.npz` detector (fast, runs without TensorFlow). |
| `--safe` | off | Lighter encoding preset (`veryfast` with thread caps) to keep CPUs cool on long encodes. |
| `--no-cut` | off | Scan and train only without exporting video files (fast iteration). |
| `--limit-minutes` | None | Process only the first N minutes of video for quick testing. |
| `--label` | off | Open the standalone browser labeling tool. |
| `--ui` | off | Open the full local web application. |

---

## Training Custom Detectors (`--teach`)

To detect a specific custom sound (e.g. a catchphrase, a sound effect, or an acoustic trigger not found in default models):

1. Mark a few positive examples of the sound in the UI (or with `--teach`).
2. Mark any confusing background sounds as negatives `[-]`.
3. AutoCut extracts 1024-dimensional neural embeddings, learns a linear acoustic classifier combined with cosine prototype gating, and saves the experience buffer so the model continually improves with feedback.
4. Reuse the model on any new footage:

```bash
python autocut.py target.mp4 --load-model models/my_sound_model.npz
```

---

## Project Structure

```text
sound-cutter/
├── autocut.py          # Core processing pipeline, CLI and acoustic models
├── app_ui.py           # Local full web interface & live runner
├── wave_ui.py          # High-resolution waveform renderer & audio decoder
├── label_gui.py        # Standalone manual event labeling tool
├── autocut.command     # Double-clickable macOS launcher
├── requirements.txt    # Python dependencies
├── models/             # Custom trained .npz sound models
├── results/            # Default directory for exported cuts & processed videos
└── test/               # Test suites and synthetic audio generation scripts
```

