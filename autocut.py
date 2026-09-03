#!/usr/bin/env python3
"""
autocut.py — automatic video editor that detects sound events
in a video's audio track and produces two cuts:

  <name>_events_only.mp4  only the target sound moments (with padding)
  <name>_clean.mp4        everything except the sound events (clean cut)

Detection uses Google's YAMNet (AudioSet) via tensorflow-hub or trained
few-shot embeddings from user-provided positive/negative examples.

Usage:
    python autocut.py video.mp4
    python autocut.py video.mp4 --pad 0.8 --threshold 0.6 --min-gap 1.5
    python autocut.py video.mp4 --teach "1:54,2:02,3:44"

Requirements (see requirements.txt):
    tensorflow, tensorflow-hub, numpy, moviepy
ffmpeg is used for audio extraction; the script uses one on PATH or falls
back to the static binary bundled with moviepy/imageio-ffmpeg.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print(
        "ERROR: numpy is not installed in this Python environment.\n"
        "  You are probably running the system Python instead of the project's.\n"
        "  Activate the environment first:\n"
        "    source .venv/bin/activate        (macOS/Linux)\n"
        "    .venv\\Scripts\\activate        (Windows)\n"
        "  or double-click autocut.command (macOS).",
        file=sys.stderr,
    )
    sys.exit(1)

# YAMNet class indices from AudioSet ontology:
LAUGHTER_CLASS_INDICES = [13, 14, 15, 16, 17, 18]   # Laughter, Giggle, Snicker, Belly laugh, Chuckle
APPLAUSE_CLASS_INDICES = [58, 61, 62]                # Clapping, Cheering, Applause
SPEECH_CLASS_INDICES = [0, 1, 2, 3]                  # Speech, child speech, conversation, narration
COUGH_CLASS_INDICES = [42, 43]                       # Cough, Throat clearing
GASP_CLASS_INDEX = 39                                # Gasp
SNORT_CLASS_INDEX = 41                               # Snort
HICCUP_CLASS_INDEX = 54                              # Hiccup

# Legacy indices kept for backward compatibility:
BURP_CLASS_INDEX = 53
FART_CLASS_INDEX = 55
THROAT_CLASS_INDEX = 43
COUGH_CLASS_INDEX = 42

N_EMBEDDING_FEATURES = 1024
YAMNET_HOP_SECONDS = 0.48

# YAMNet expects 16 kHz mono float audio in [-1, 1].
YAMNET_SAMPLE_RATE = 16000

# Minimum length of a kept segment in the clean cut, in seconds.
MIN_KEEP_SEGMENT = 0.2


@dataclass
class AudioEvent:
    """A detected sound event, in seconds, already padded, with confidence score."""
    start: float
    end: float
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


BurpEvent = AudioEvent


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def find_ffmpeg() -> str | None:
    """Return a usable ffmpeg binary: PATH first, then moviepy's bundled one.

    moviepy installs a static ffmpeg via imageio-ffmpeg, so the script works
    even on machines without a system ffmpeg.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def check_dependencies() -> str:
    """Fail fast with a clear message if no ffmpeg is available."""
    exe = find_ffmpeg()
    if exe is None:
        die(
            "No ffmpeg found. Install it (Windows: winget install Gyan.FFmpeg, "
            "macOS: brew install ffmpeg, Linux: sudo apt install ffmpeg), "
            "or run 'pip install -r requirements.txt' which bundles one "
            "via moviepy/imageio-ffmpeg."
        )
    return exe


def print_info() -> None:
    """Print the environment summary (python, versions, ffmpeg) and exit."""
    print(f"Python      : {sys.executable}")
    print(f"numpy       : {np.__version__}")
    for mod, name in [("moviepy", "moviepy"),
                      ("imageio_ffmpeg", "imageio-ffmpeg")]:
        try:
            m = __import__(mod)
            print(f"{name:<12}: {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"{name:<12}: MISSING")
    for mod, name in [("tensorflow", "tensorflow"),
                      ("tensorflow_hub", "tensorflow-hub")]:
        try:
            m = __import__(mod)
            print(f"{name:<12}: {m.__version__}")
        except ImportError:
            print(f"{name:<12}: missing (only needed for YAMNet mode)")
    print(f"ffmpeg      : {find_ffmpeg()}")
    sys.exit(0)


def extract_audio(video_path: Path, wav_path: Path, ffmpeg_exe: str) -> None:
    """Extract 16 kHz mono WAV from the video using ffmpeg."""
    print(f"[1/4] Extracting audio from {video_path.name} ...")
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(video_path),
        "-vn",                      # drop video
        "-ac", "1",                 # mono
        "-ar", str(YAMNET_SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        str(wav_path),
        "-loglevel", "error",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        die(f"ffmpeg failed to extract audio (exit code {exc.returncode}). "
            f"Is '{video_path}' a valid video with an audio track?")


def load_wav_as_float(wav_path: Path, normalize: bool = True) -> np.ndarray:
    """Load a 16-bit PCM WAV into a float32 numpy array in [-1, 1].

    Uses the stdlib `wave` module + numpy so we don't need scipy/scikit.
    If normalize=True and the audio peak is low (< 0.70, i.e. < -3.1 dBFS),
    applies peak normalization to ~0.89 (-1.0 dBFS) to ensure optimal log-mel
    dynamic range for neural network analysis without any clipping.
    """
    import wave

    with wave.open(str(wav_path), "rb") as wav:
        n_channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        raw = wav.readframes(n_frames)

    if sample_rate != YAMNET_SAMPLE_RATE:
        die(f"Unexpected sample rate {sample_rate} (expected {YAMNET_SAMPLE_RATE}).")

    if n_channels != 1:
        # Interleaved stereo -> average channels (shouldn't happen after -ac 1,
        # but be defensive).
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    else:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if normalize and len(samples) > 0:
        peak = float(np.max(np.abs(samples)))
        # Only boost if quiet (< 0.70 = -3.1 dBFS) and not complete silence (> 1e-4)
        if 1e-4 < peak < 0.70:
            target_peak = 0.891  # -1.0 dBFS
            gain = target_peak / peak
            samples = samples * gain
            gain_db = 20.0 * math.log10(gain)
            print(f"      [Audio Normalization] Applied +{gain_db:.1f} dB peak gain ({peak:.2f} -> {target_peak:.2f}).")

    return samples


def run_yamnet(audio: np.ndarray, chunk_seconds: float = 240.0) -> tuple[np.ndarray, np.ndarray]:
    """Run YAMNet over the audio with sample-accurate, continuous hop alignment.

    Returns (scores, embeddings):
      scores: shape (n_frames, 521)
      embeddings: shape (n_frames, 1024)
    YAMNet scores ~2 frames per second (0.96 s window, 0.48 s hop).
    Continuous chunking ensures ZERO timestamp drift across any video duration:
    global frame k corresponds strictly to audio interval [k * 0.48s, k * 0.48s + 0.96s].
    """
    print("[2/4] Running YAMNet audio analysis & embedding extraction ...")

    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

    try:
        import tensorflow_hub as hub
        import tensorflow as tf  # noqa: F401  (required by tensorflow_hub)
    except ImportError as exc:
        die(
            "TensorFlow / tensorflow-hub not installed. Run:\n"
            "  pip install -r requirements.txt\n"
            f"(import error: {exc})"
        )

    model = hub.load("https://tfhub.dev/google/yamnet/1")

    sr = YAMNET_SAMPLE_RATE
    hop_samples = int(YAMNET_HOP_SECONDS * sr)   # 7680 samples (0.48 s)
    win_samples = int(0.96 * sr)                 # 15360 samples (0.96 s)

    if len(audio) < win_samples:
        pad_len = win_samples - len(audio)
        audio = np.pad(audio, (0, pad_len))

    n_total_frames = 1 + max(0, (len(audio) - win_samples) // hop_samples)
    frames_per_chunk = max(50, int(chunk_seconds / YAMNET_HOP_SECONDS))
    n_chunks = max(1, math.ceil(n_total_frames / frames_per_chunk))

    all_scores: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []

    for chunk_idx, start_frame in enumerate(range(0, n_total_frames, frames_per_chunk), 1):
        end_frame = min(start_frame + frames_per_chunk, n_total_frames)
        chunk_frames = end_frame - start_frame

        start_sample = start_frame * hop_samples
        end_sample = start_sample + (chunk_frames - 1) * hop_samples + win_samples
        seg = audio[start_sample:end_sample]

        scores_t, emb_t, _ = model(seg.astype(np.float32))
        all_scores.append(scores_t.numpy())
        all_embeddings.append(emb_t.numpy())
        print(f"      chunk {chunk_idx}/{n_chunks} analyzed", end="\r")

    print(f"      done ({n_chunks} chunks).")
    if not all_scores:
        die("YAMNet produced no scores — audio too short?")
    return np.concatenate(all_scores, axis=0), np.concatenate(all_embeddings, axis=0)


def apply_silence_gate(
    scores: np.ndarray,
    audio: np.ndarray,
    sample_rate: int,
    hop_seconds: float,
    win_seconds: float,
    silence_thresh_db: float | None,
) -> np.ndarray:
    """Zero out detection scores for frames where audio RMS is below silence_thresh_db."""
    if silence_thresh_db is None:
        return scores

    n_frames = len(scores)
    win_samples = int(win_seconds * sample_rate)
    hop_samples = int(hop_seconds * sample_rate)

    muted = 0
    for i in range(n_frames):
        start = i * hop_samples
        end = min(len(audio), start + win_samples)
        if start >= len(audio):
            scores[i:] = 0.0
            muted += (n_frames - i)
            break
        frame = audio[start:end]
        if len(frame) == 0:
            scores[i] = 0.0
            muted += 1
            continue
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        db = 20.0 * math.log10(rms + 1e-9)
        if db < silence_thresh_db:
            scores[i] = 0.0
            muted += 1

    if muted > 0:
        pct = (muted / n_frames) * 100
        print(f"      Silence gate: filtered {muted}/{n_frames} silent frames ({pct:.1f}%) < {silence_thresh_db:.1f} dB.")
    return scores


# ---------------------------------------------------------------------------
# Deep Embeddings "Teach" Mode (1024-D YAMNet features)
# ---------------------------------------------------------------------------

def parse_one_time(token: str) -> float:
    """Parse '1:54', '1:54.5' or '114.5' into seconds."""
    if ":" in token:
        m, s = token.split(":", 1)
        try:
            return int(m) * 60 + float(s)
        except ValueError:
            die(f"Cannot parse time '{token}' (use m:ss or seconds).")
    try:
        return float(token)
    except ValueError:
        die(f"Cannot parse time '{token}' (use m:ss or seconds).")


def parse_teach_labels(spec: str) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Parse teach labels into (pos_labels, neg_labels) intervals in seconds.

    Supports positive labels ('1:54-1:56', '+1:54-1:56') and explicit negative
    labels ('-1:54-1:56', '!1:54-1:56', 'neg:1:54-1:56') to reject false positives.
    """
    if spec.startswith("@"):
        path = Path(spec[1:])
        if not path.is_file():
            die(f"Label file not found: {path}")
        lines = [ln.split("#", 1)[0].strip()
                 for ln in path.read_text(encoding="utf-8").splitlines()]
        spec = " ".join(ln for ln in lines if ln)
        if not spec.strip():
            die(f"Label file '{path}' is empty (only comments).")
    pos_out: list[tuple[float, float]] = []
    neg_out: list[tuple[float, float]] = []

    for raw_token in spec.replace(",", " ").split():
        raw_token = raw_token.strip()
        if not raw_token:
            continue
        is_neg = False
        token = raw_token
        if token.startswith("!"):
            is_neg = True
            token = token[1:]
        elif token.startswith("neg:"):
            is_neg = True
            token = token[4:]
        elif token.startswith("+"):
            token = token[1:]
        elif token.startswith("-"):
            is_neg = True
            token = token[1:]

        parts = token.split("-")
        if len(parts) > 2:
            die(f"Cannot parse label '{raw_token}' (use e.g. 1:54-2:00 or -1:54-2:00).")
        if len(parts) == 2:
            a = parse_one_time(parts[0])
            b = parse_one_time(parts[1])
            if b < a:
                a, b = b, a
            (neg_out if is_neg else pos_out).append((a, b))
        else:
            t = parse_one_time(token)
            (neg_out if is_neg else pos_out).append((t, t))
    return pos_out, neg_out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None,
                 iters: int = 1000, lr: float = 0.02, reg: float = 1e-3,
                 w_init: np.ndarray | None = None, b_init: float | None = None) -> tuple[np.ndarray, float]:
    """L2-regularized weighted logistic regression with Adam optimizer (pure numpy)."""
    n, d = X.shape
    if sample_weight is None:
        sample_weight = np.ones(n)

    total_w = float(np.sum(sample_weight))
    sw_norm = sample_weight / (total_w + 1e-9)

    if w_init is not None and len(w_init) == d:
        w = np.copy(w_init).astype(np.float64)
    else:
        w = np.zeros(d, dtype=np.float64)

    if b_init is not None:
        b = float(b_init)
    else:
        b = 0.0

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m_w = np.zeros(d, dtype=np.float64)
    v_w = np.zeros(d, dtype=np.float64)
    m_b = 0.0
    v_b = 0.0

    for t in range(1, iters + 1):
        p = _sigmoid(X @ w + b)
        err = (p - y) * sw_norm
        grad_w = X.T @ err + reg * w
        grad_b = float(np.sum(err))

        m_w = beta1 * m_w + (1.0 - beta1) * grad_w
        v_w = beta2 * v_w + (1.0 - beta2) * (grad_w ** 2)
        m_b = beta1 * m_b + (1.0 - beta1) * grad_b
        v_b = beta2 * v_b + (1.0 - beta2) * (grad_b ** 2)

        m_w_hat = m_w / (1.0 - beta1 ** t)
        v_w_hat = v_w / (1.0 - beta2 ** t)
        m_b_hat = m_b / (1.0 - beta1 ** t)
        v_b_hat = v_b / (1.0 - beta2 ** t)

        w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
        b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

    return w, b


def learn_embedding_model(
    embeddings: np.ndarray, centers: np.ndarray,
    pos_labels: list[tuple[float, float]],
    neg_labels: list[tuple[float, float]] | None = None,
    warm_start_path: Path | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Train regularized logistic classifier with positive & hard-negative feedback via Adam."""
    if neg_labels is None:
        neg_labels = []

    n_frames = len(centers)
    hard_neg_mask = np.zeros(n_frames, dtype=bool)
    for s, e in neg_labels:
        # Negatives cover surrounding false positive noise/words
        hard_neg_mask |= (centers >= s - 0.35) & (centers <= e + 0.35)

    pos_mask = np.zeros(n_frames, dtype=bool)
    for s, e in pos_labels:
        # Match frames whose window overlaps with the labeled interval
        m = (centers >= s - 0.25) & (centers <= e + 0.25)
        if not np.any(m):
            mid = (s + e) / 2.0
            idx = int(np.argmin(np.abs(centers - mid)))
            pos_mask[idx] = True
        else:
            pos_mask |= m

    # Explicit user negative [-] ALWAYS overrides any conflicting positive label:
    pos_mask &= ~hard_neg_mask
    pos = np.where(pos_mask)[0]
    hard_neg = np.where(hard_neg_mask)[0]

    # Background negatives from the rest of the audio
    bg_neg_mask = ~pos_mask & ~hard_neg_mask
    for s, e in pos_labels:
        # Guard band around positives so transitional sound edges don't pollute negative training
        bg_neg_mask &= ~((centers >= s - 0.60) & (centers <= e + 0.60))
    bg_neg = np.where(bg_neg_mask)[0]

    if len(pos) == 0:
        die("Devi specificare almeno un intervallo positivo [+] (es. 1:54-1:56). "
            "Gli esempi negativi [-] (es. -1:54-1:56) servono per escludere i falsi positivi, "
            "ma il modello ha bisogno di almeno un esempio del suono da trovare!")
    if len(bg_neg) == 0 and len(hard_neg) == 0:
        die("No non-target frames found for training.")

    # Unit-norm (L2) frame embeddings for volume and gain invariance
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)

    cur_pos = emb_norm[pos]
    cur_neg = emb_norm[hard_neg] if len(hard_neg) else np.zeros((0, N_EMBEDDING_FEATURES), dtype=np.float32)

    # Experience Replay: load historical positive and negative exemplars from previous videos
    hist_pos = np.zeros((0, N_EMBEDDING_FEATURES), dtype=np.float32)
    hist_neg = np.zeros((0, N_EMBEDDING_FEATURES), dtype=np.float32)
    if warm_start_path is not None and warm_start_path.is_file():
        try:
            old_data = np.load(str(warm_start_path))
            if "buffer_pos" in old_data and len(old_data["buffer_pos"]) > 0:
                hist_pos = old_data["buffer_pos"].astype(np.float32)
            elif "proto" in old_data and np.linalg.norm(old_data["proto"]) > 1e-6:
                hist_pos = old_data["proto"].astype(np.float32).reshape(1, -1)
            if "buffer_neg" in old_data and len(old_data["buffer_neg"]) > 0:
                hist_neg = old_data["buffer_neg"].astype(np.float32)
            elif "proto_neg" in old_data and old_data["proto_neg"] is not None and np.linalg.norm(old_data["proto_neg"]) > 1e-6:
                hist_neg = old_data["proto_neg"].astype(np.float32).reshape(1, -1)
            if len(hist_pos) > 0 or len(hist_neg) > 0:
                print(f"      [Experience Replay] Reusing {len(hist_pos)} historical positive and {len(hist_neg)} negative exemplar frames from '{warm_start_path.name}'.")
        except Exception as e:
            print(f"      [Experience Replay Warning] Could not load historical buffer: {e}")

    all_pos = np.vstack([hist_pos, cur_pos]) if len(hist_pos) else cur_pos
    all_neg = np.vstack([hist_neg, cur_neg]) if len(hist_neg) else cur_neg

    # Keep exemplar buffer bounded (max 1500 frames each ~6 MB max npz)
    if len(all_pos) > 1500:
        keep_idx = np.linspace(0, len(all_pos) - 1, 1500, dtype=int)
        all_pos = all_pos[keep_idx]
    if len(all_neg) > 1500:
        keep_idx = np.linspace(0, len(all_neg) - 1, 1500, dtype=int)
        all_neg = all_neg[keep_idx]

    # Compute normalized prototype centroids from all cumulative exemplars
    proto_pos = all_pos.mean(axis=0)
    proto_pos = proto_pos / (np.linalg.norm(proto_pos) + 1e-9)

    proto_neg = None
    if len(all_neg) > 0:
        proto_neg = all_neg.mean(axis=0)
        proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-9)

    # Sample background negatives from current video (up to 2000 frames)
    bg_frames = emb_norm[bg_neg]
    if len(bg_frames) > 2000:
        bg_idx = np.random.default_rng(0).choice(len(bg_frames), size=2000, replace=False)
        bg_frames = bg_frames[bg_idx]

    if len(all_neg) > 0:
        X_neg = np.vstack([all_neg, bg_frames]) if len(bg_frames) else all_neg
    else:
        X_neg = bg_frames

    X = np.vstack([all_pos, X_neg])
    y = np.concatenate([np.ones(len(all_pos)), np.zeros(len(X_neg))])

    # Balanced sample weighting
    weight_pos = len(X_neg) / len(all_pos)
    hard_weight = np.full(len(all_neg), 12.0) if len(all_neg) else np.array([])
    bg_weight = np.ones(len(bg_frames))
    sample_weight = np.concatenate([
        np.full(len(all_pos), weight_pos),
        hard_weight,
        bg_weight,
    ])

    mean = np.zeros(N_EMBEDDING_FEATURES, dtype=np.float64)
    std = np.ones(N_EMBEDDING_FEATURES, dtype=np.float64)

    w, b = fit_logistic(X, y, sample_weight, iters=1200, lr=0.04, reg=0.01)
    return w, b, mean, std, proto_pos, proto_neg, all_pos, all_neg


def calculate_deep_scores(
    embeddings: np.ndarray,
    w: np.ndarray,
    b: float,
    mean: np.ndarray,
    std: np.ndarray,
    proto_pos: np.ndarray | None = None,
    proto_neg: np.ndarray | None = None,
    is_unit_norm: bool = False,
) -> np.ndarray:
    """Calculate calibrated detection probabilities with prototype anchor gating."""
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)

    if np.linalg.norm(w) > 1e-6:
        if is_unit_norm:
            linear_p = _sigmoid(emb_norm @ w + b)
        else:
            linear_p = _sigmoid((embeddings - mean) / std @ w + b)
    else:
        linear_p = np.zeros(len(embeddings), dtype=np.float64)

    # Prototype Anchor Gating: prevents arbitrary out-of-distribution sounds from triggering 100% confidence
    if proto_pos is not None and np.linalg.norm(proto_pos) > 1e-6:
        sim_pos = np.maximum(0.0, emb_norm @ proto_pos)
        # Cosine ramp: sound must share acoustic spectral similarity with the target sound prototype
        pos_ramp = np.clip((sim_pos - 0.40) / (0.70 - 0.40), 0.0, 1.0)

        if proto_neg is not None and np.linalg.norm(proto_neg) > 1e-6:
            sim_neg = np.maximum(0.0, emb_norm @ proto_neg)
            margin = sim_pos - sim_neg
            # If closer to marked false positives than to target sound events, suppress smoothly
            neg_ramp = np.clip((margin + 0.06) / 0.14, 0.0, 1.0)
            gate = pos_ramp * neg_ramp
        else:
            gate = pos_ramp

        if np.linalg.norm(w) > 1e-6:
            probs = linear_p * gate
        else:
            probs = (sim_pos ** 2) * gate
    else:
        probs = linear_p

    return probs


def teach_event_scores(
    audio: np.ndarray, sr: int,
    pos_labels: list[tuple[float, float]],
    neg_labels: list[tuple[float, float]] | None = None,
    save_path: Path | None = None,
) -> tuple[np.ndarray, float]:
    """Learn a deep detector from labeled sound intervals (+ and -) using 1024-D YAMNet embeddings."""
    if neg_labels is None:
        neg_labels = []
    print(f"[2/4] Learning deep detector from {len(pos_labels)} positive + {len(neg_labels)} negative label(s) ...")
    scores, embeddings = run_yamnet(audio)
    # Frame window center is (i + 1.0) * hop (window starts at i * hop and ends at i * hop + 0.96)
    centers = np.arange(len(embeddings)) * YAMNET_HOP_SECONDS + (0.96 / 2.0)
    w, b, mean, std, proto_pos, proto_neg, buffer_pos, buffer_neg = learn_embedding_model(
        embeddings, centers, pos_labels, neg_labels, warm_start_path=save_path)

    if save_path is not None:
        save_teach_model(save_path, w, b, mean, std, proto_pos, proto_neg,
                         buffer_pos=buffer_pos, buffer_neg=buffer_neg)

    # Calibrated probability with prototype gating
    probs = calculate_deep_scores(embeddings, w, b, mean, std, proto_pos, proto_neg, is_unit_norm=True)

    pos_mask = np.zeros(len(centers), dtype=bool)
    for s, e in pos_labels:
        pos_mask |= (centers >= s - 0.25) & (centers <= e + 0.25)
    pos = np.where(pos_mask)[0]
    pos_mean = float(probs[pos].mean()) if len(pos) else 0.0
    frac = float((probs >= 0.5).mean())
    print(f"      Deep learning complete on {len(pos)} positive frames" +
          (f" and {len(neg_labels)} negative examples." if neg_labels else ".") +
          (f" [Cumulative Experience Buffer: {len(buffer_pos)} pos, {len(buffer_neg)} neg stored]" if len(buffer_pos) else ""))
    print(f"      Quality: positive frames mean p={pos_mean:.2f}; {frac:.1%} of video flagged at p>=0.5")

    return probs, YAMNET_HOP_SECONDS


teach_burp_scores = teach_event_scores


def train_from_reference(
    target_audio: np.ndarray, ref_audio: np.ndarray, sr: int,
    save_path: Path | None = None,
) -> tuple[np.ndarray, float]:
    """Learn a deep detector from reference audio clips containing only the target sounds."""
    print("[2/4] Extracting deep neural embeddings from reference and target audio ...")
    _, emb_t = run_yamnet(target_audio)
    _, emb_r = run_yamnet(ref_audio)

    if len(emb_r) < 2:
        die("Reference audio is too short.")

    emb_r_norm = emb_r / (np.linalg.norm(emb_r, axis=1, keepdims=True) + 1e-9)
    emb_t_norm = emb_t / (np.linalg.norm(emb_t, axis=1, keepdims=True) + 1e-9)

    proto = emb_r_norm.mean(axis=0)
    proto = proto / (np.linalg.norm(proto) + 1e-9)

    # Sample negatives from target video
    n_neg = min(2000, len(emb_t_norm))
    neg_idx = np.random.default_rng(0).choice(len(emb_t_norm), size=n_neg, replace=False)
    neg = emb_t_norm[neg_idx]

    X = np.vstack([emb_r_norm, neg])
    y = np.concatenate([np.ones(len(emb_r_norm)), np.zeros(len(neg))])
    sample_weight = np.concatenate([
        np.full(len(emb_r_norm), len(neg) / len(emb_r_norm)),
        np.ones(len(neg)),
    ])
    mean = np.zeros(N_EMBEDDING_FEATURES, dtype=np.float64)
    std = np.ones(N_EMBEDDING_FEATURES, dtype=np.float64)
    w, b = fit_logistic(X, y, sample_weight, iters=1500, lr=0.04, reg=0.01)

    if save_path is not None:
        save_teach_model(save_path, w, b, mean, std, proto, buffer_pos=emb_r_norm)

    probs = calculate_deep_scores(emb_t, w, b, mean, std, proto_pos=proto, is_unit_norm=True)

    frac = float((probs >= 0.5).mean())
    print(f"      Learned from {len(emb_r)} reference frames. {frac:.1%} of target flagged at p>=0.5")
    return probs, YAMNET_HOP_SECONDS


def save_teach_model(path: Path, w: np.ndarray, b: float,
                     mean: np.ndarray, std: np.ndarray,
                     proto_pos: np.ndarray | None = None,
                     proto_neg: np.ndarray | None = None,
                     buffer_pos: np.ndarray | None = None,
                     buffer_neg: np.ndarray | None = None) -> None:
    """Persist a trained 1024-D deep detector to an .npz file (with Experience Replay buffer)."""
    if proto_pos is None:
        proto_pos = np.zeros(N_EMBEDDING_FEATURES, dtype=np.float32)
    if proto_neg is None:
        proto_neg = np.zeros(N_EMBEDDING_FEATURES, dtype=np.float32)
    if buffer_pos is None:
        buffer_pos = np.zeros((0, N_EMBEDDING_FEATURES), dtype=np.float32)
    if buffer_neg is None:
        buffer_neg = np.zeros((0, N_EMBEDDING_FEATURES), dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        w=w.astype(np.float32),
        b=np.float32(b),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        proto=proto_pos.astype(np.float32),
        proto_neg=proto_neg.astype(np.float32),
        buffer_pos=buffer_pos.astype(np.float32),
        buffer_neg=buffer_neg.astype(np.float32),
        nfeat=np.int64(N_EMBEDDING_FEATURES),
    )
    print(f"      Deep model saved to {path} ({len(buffer_pos)} pos, {len(buffer_neg)} neg exemplars stored; reuse with --load-model).")

def compute_optimal_threshold(
    scores: np.ndarray,
    pos_mask: np.ndarray | None = None,
    hard_neg_mask: np.ndarray | None = None,
    min_thresh: float = 0.25,
    max_thresh: float = 0.75,
) -> float:
    """Calculate the mathematically optimal detection threshold.

    When labeled positive/negative frames are available:
      Finds the threshold maximizing F_1.5 score (prioritizing recall >= 90%)
      while strongly suppressing false positives.
    When unlabeled:
      Computes dynamic threshold based on high-confidence acoustic percentile.
    """
    if pos_mask is not None and np.any(pos_mask):
        pos_scores = scores[pos_mask]
        neg_mask = ~pos_mask
        if hard_neg_mask is not None and np.any(hard_neg_mask):
            neg_scores = np.concatenate([scores[hard_neg_mask], scores[neg_mask]])
        else:
            neg_scores = scores[neg_mask]

        candidates = np.linspace(min_thresh, max_thresh, 51)
        best_th = 0.50
        best_score = -1.0

        p10 = float(np.percentile(pos_scores, 10))
        p_med = float(np.median(pos_scores))

        for th in candidates:
            tp = float(np.sum(pos_scores >= th))
            fn = float(np.sum(pos_scores < th))
            fp = float(np.sum(neg_scores >= th))

            recall = tp / (tp + fn + 1e-9)
            precision = tp / (tp + fp + 1e-9)
            beta = 1.5
            f_score = (1.0 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-9)

            if recall >= 0.90:
                f_score += 0.5

            if f_score > best_score:
                best_score = f_score
                best_th = th

        if p10 > min_thresh:
            best_th = min(best_th, p10 * 0.95)
        best_th = float(np.clip(best_th, min_thresh, max_thresh))

        print(f"      [Auto-Threshold] Optimal threshold determined: {best_th:.2f} "
              f"(Positives: min {pos_scores.min():.2f}, med {p_med:.2f}, 10th perc {p10:.2f})")
        return best_th

    p99 = float(np.percentile(scores, 99.0))
    th = float(np.clip(p99 * 0.85, 0.35, 0.65))
    print(f"      [Auto-Threshold] Dynamic baseline threshold determined: {th:.2f}")
    return th


class TeachModelResult(tuple):
    """Backwards-compatible tuple (w, b, mean, std, proto_pos, proto_neg) with exemplar attributes."""
    def __new__(cls, w, b, mean, std, proto_pos, proto_neg, buffer_pos=None, buffer_neg=None, is_unit_norm=False):
        obj = super().__new__(cls, (w, b, mean, std, proto_pos, proto_neg))
        obj.buffer_pos = buffer_pos
        obj.buffer_neg = buffer_neg
        obj.is_unit_norm = is_unit_norm
        return obj


def load_teach_model(path: Path) -> TeachModelResult:
    """Load a previously trained 1024-D detector from an .npz file (supports legacy and exemplar models)."""
    if not path.is_file():
        script_dir = Path(__file__).resolve().parent
        cand1 = Path("models") / path
        cand2 = script_dir / "models" / path
        if cand1.is_file():
            path = cand1
        elif cand2.is_file():
            path = cand2
        else:
            die(f"Model file not found: {path}")
    data = np.load(str(path))
    try:
        w = data["w"].astype(np.float64)
    except KeyError:
        die(f"'{path}' is not a valid teach model.")
    b = float(data["b"])
    mean = data["mean"].astype(np.float64) if "mean" in data else np.zeros_like(w)
    std = data["std"].astype(np.float64) if "std" in data else np.ones_like(w)
    proto_pos = data["proto"].astype(np.float64) if "proto" in data else np.zeros_like(w)
    proto_neg = data["proto_neg"].astype(np.float64) if "proto_neg" in data else None
    if proto_neg is not None and np.linalg.norm(proto_neg) < 1e-6:
        proto_neg = None
    buffer_pos = data["buffer_pos"].astype(np.float32) if "buffer_pos" in data else None
    buffer_neg = data["buffer_neg"].astype(np.float32) if "buffer_neg" in data else None

    # Detect if model was trained with modern unit-norm
    is_unit_norm = (buffer_pos is not None) or (np.allclose(mean, 0.0) and np.allclose(std, 1.0))

    return TeachModelResult(
        w, b, mean, std, proto_pos, proto_neg,
        buffer_pos=buffer_pos, buffer_neg=buffer_neg, is_unit_norm=is_unit_norm
    )


def apply_teach_model(audio: np.ndarray, sr: int,
                      w: np.ndarray, b: float,
                      mean: np.ndarray, std: np.ndarray,
                      proto_pos: np.ndarray | None = None,
                      proto_neg: np.ndarray | None = None,
                      is_unit_norm: bool = False) -> tuple[np.ndarray, float]:
    """Score an audio track with a previously trained 1024-D deep detector."""
    _, embeddings = run_yamnet(audio)
    probs = calculate_deep_scores(
        embeddings, w, b, mean, std,
        proto_pos=proto_pos, proto_neg=proto_neg,
        is_unit_norm=is_unit_norm
    )
    return probs, YAMNET_HOP_SECONDS


def frames_to_seconds(frame_index: int, hop_seconds: float) -> float:
    """Center of the YAMNet frame -> time in seconds."""
    return (frame_index + 0.5) * hop_seconds


def refine_event_tail(
    audio: np.ndarray,
    sr: int,
    start_t: float,
    end_t: float,
    total_duration: float,
    max_extension: float = 0.85,
    tail_hangover: float = 0.25,
) -> float:
    """Trace the acoustic energy (RMS) decay of a detected sound event to prevent cutting off its tail.

    Follows the signal envelope until it merges back into the local ambient noise floor,
    and applies a natural hangover buffer (default 250ms).
    """
    if audio is None or len(audio) == 0:
        return min(total_duration, end_t + tail_hangover)

    win_len = int(0.025 * sr)  # 25 ms analysis window
    hop_len = int(0.010 * sr)  # 10 ms step

    # 1. Estimate local ambient noise floor (RMS) before event onset
    bg_start = max(0, int((start_t - 2.0) * sr))
    bg_end = max(0, int((start_t - 0.08) * sr))
    if bg_end - bg_start >= win_len:
        bg_chunk = audio[bg_start:bg_end]
        bg_rms = [float(np.sqrt(np.mean(bg_chunk[i:i + win_len] ** 2)))
                  for i in range(0, len(bg_chunk) - win_len, hop_len)]
        noise_floor = float(np.percentile(bg_rms, 30)) if bg_rms else 1e-4
    else:
        noise_floor = 1e-4

    # 2. Estimate event peak energy
    ev_start = max(0, int(start_t * sr))
    ev_end = min(len(audio), int(end_t * sr))
    if ev_end - ev_start >= win_len:
        ev_chunk = audio[ev_start:ev_end]
        ev_rms = [float(np.sqrt(np.mean(ev_chunk[i:i + win_len] ** 2)))
                  for i in range(0, len(ev_chunk) - win_len, hop_len)]
        peak_rms = float(np.max(ev_rms)) if ev_rms else noise_floor
    else:
        peak_rms = noise_floor

    # Dynamic decay threshold: noise floor + 10% of event dynamic range
    decay_threshold = noise_floor + 0.10 * max(0.0, peak_rms - noise_floor)

    # 3. Trace forward from (end_t - 0.20s) up to (end_t + max_extension)
    search_start = max(0, int((end_t - 0.20) * sr))
    search_end = min(len(audio), int((end_t + max_extension) * sr))
    if search_end - search_start < win_len:
        return min(total_duration, end_t + tail_hangover)

    search_chunk = audio[search_start:search_end]
    times: list[float] = []
    rms_vals: list[float] = []
    for i in range(0, len(search_chunk) - win_len, hop_len):
        t = (search_start + i + win_len / 2.0) / sr
        r = float(np.sqrt(np.mean(search_chunk[i:i + win_len] ** 2)))
        times.append(t)
        rms_vals.append(r)

    # Find where RMS drops below decay_threshold and stays low for >= 70ms (7 frames)
    refined_end = end_t
    consecutive_low = 0
    found_silence = False
    for t, r in zip(times, rms_vals):
        if t >= end_t:
            if r <= decay_threshold:
                consecutive_low += 1
                if consecutive_low >= 7:
                    refined_end = (t - (7 * 0.010)) + tail_hangover
                    found_silence = True
                    break
            else:
                consecutive_low = 0
                refined_end = t + tail_hangover

    if not found_silence and times:
        refined_end = max(end_t + tail_hangover, refined_end)

    # Never make shorter than original detected end_t
    refined_end = max(end_t, refined_end)
    return min(total_duration, refined_end)


def detect_events(
    scores: np.ndarray,
    hop_seconds: float,
    threshold: float,
    min_gap: float,
    min_duration: float,
    pad: float,
    total_duration: float,
    max_duration: float | None = None,
    sound_type: str = "laughter",
    min_confidence: float = 0.0,
    audio: np.ndarray | None = None,
    sr: int = YAMNET_SAMPLE_RATE,
    tail_hangover: float = 0.25,
) -> list[AudioEvent]:
    """Turn per-frame scores into full-length padded, merged events with confidence scores."""
    print("[3/4] Post-processing detections ...")

    if scores.ndim == 1:
        target_scores = scores
    elif scores.ndim == 2:
        if scores.shape[1] == 1:
            target_scores = scores[:, 0]
        elif sound_type == "laughter":
            target_scores = np.max(scores[:, LAUGHTER_CLASS_INDICES], axis=1)
        elif sound_type == "applause":
            target_scores = np.max(scores[:, APPLAUSE_CLASS_INDICES], axis=1)
        elif sound_type == "speech":
            target_scores = np.max(scores[:, SPEECH_CLASS_INDICES], axis=1)
        elif sound_type == "cough":
            target_scores = np.max(scores[:, COUGH_CLASS_INDICES], axis=1)
        elif sound_type == "fart":
            target_scores = scores[:, FART_CLASS_INDEX]
        elif sound_type == "both":
            target_scores = np.maximum(scores[:, BURP_CLASS_INDEX], scores[:, FART_CLASS_INDEX])
        elif sound_type == "burp_extended":
            target_scores = np.max(scores[:, [BURP_CLASS_INDEX, THROAT_CLASS_INDEX, SNORT_CLASS_INDEX, HICCUP_CLASS_INDEX]], axis=1)
        elif sound_type == "burp":
            target_scores = scores[:, BURP_CLASS_INDEX]
        else:
            target_scores = np.max(scores[:, LAUGHTER_CLASS_INDICES], axis=1)
    else:
        target_scores = scores.flatten()

    n_frames = len(target_scores)
    if n_frames == 0:
        return []

    # Adaptive Hysteresis (Schmitt trigger):
    # Trigger event on threshold (high), track the decaying tail until release_threshold (low).
    # Uses a gentle release floor (0.04 or 25% of threshold) to avoid cutting off decaying acoustic tails.
    release_threshold = max(0.04, threshold * 0.25)

    trigger_indices = np.where(target_scores >= threshold)[0]
    if len(trigger_indices) == 0:
        print("      No frames above threshold — no events detected.")
        return []

    # Actual sliding audio window duration for YAMNet is 0.96s (hop is 0.48s)
    window_duration = 0.96 if abs(hop_seconds - 0.48) < 0.05 else hop_seconds

    raw_intervals: list[tuple[float, float, float]] = []
    visited = np.zeros(n_frames, dtype=bool)

    for trig in trigger_indices:
        if visited[trig]:
            continue
        # Expand backward to find true onset where signal emerges from background
        left = trig
        while left > 0 and target_scores[left - 1] >= release_threshold and not visited[left - 1]:
            left -= 1
        # Expand forward to trace the acoustic decay
        right = trig
        while right < n_frames - 1 and target_scores[right + 1] >= release_threshold:
            right += 1

        visited[left:right + 1] = True

        start_t = left * hop_seconds
        end_t = right * hop_seconds + window_duration
        peak_conf = float(target_scores[left:right + 1].max())
        raw_intervals.append((start_t, end_t, peak_conf))

    raw_intervals.sort(key=lambda x: x[0])

    # Merge events closer than min_gap
    merged: list[list[float]] = []
    for s, e, c in raw_intervals:
        if merged and s - merged[-1][1] <= min_gap:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2] = max(merged[-1][2], c)
        else:
            merged.append([s, e, c])

    # Refine acoustic decay / conclusion using raw audio envelope tracking if audio is provided
    if audio is not None and len(audio) > 0:
        for i in range(len(merged)):
            s, e, c = merged[i]
            max_ext = 0.85
            if i + 1 < len(merged):
                gap_to_next = merged[i + 1][0] - e
                if gap_to_next > 0.1:
                    max_ext = min(0.85, gap_to_next - 0.05)
                else:
                    max_ext = 0.0
            if max_ext > 0.0:
                refined_e = refine_event_tail(
                    audio, sr, s, e, total_duration,
                    max_extension=max_ext, tail_hangover=tail_hangover
                )
                merged[i][1] = refined_e

        # Re-merge in case tail refinement closed a narrow gap
        re_merged: list[list[float]] = []
        for s, e, c in merged:
            if re_merged and s - re_merged[-1][1] <= min_gap:
                re_merged[-1][1] = max(re_merged[-1][1], e)
                re_merged[-1][2] = max(re_merged[-1][2], c)
            else:
                re_merged.append([s, e, c])
        merged = re_merged

    # Pad, clamp, and filter duration.
    # Merge any newly overlapping padded regions.
    padded: list[AudioEvent] = []
    for start, end, c in merged:
        start_padded = max(0.0, start - pad)
        end_padded = min(total_duration, end + pad)
        if end_padded - start_padded >= min_duration:
            if padded and start_padded <= padded[-1].end:
                padded[-1].end = max(padded[-1].end, end_padded)
                padded[-1].confidence = max(padded[-1].confidence, c)
            else:
                padded.append(AudioEvent(start_padded, end_padded, confidence=c))

    if max_duration is not None and max_duration > 0:
        padded = [ev for ev in padded if ev.duration <= max_duration]

    if min_confidence > 0.0:
        before_count = len(padded)
        padded = [ev for ev in padded if ev.confidence >= min_confidence]
        print(f"      [Confidence Filter] Kept {len(padded)}/{before_count} events with confidence >= {min_confidence*100:.0f}%.")

    return padded


detect_burp_events = detect_events


def complement_intervals(
    events: list[AudioEvent], total_duration: float
) -> list[tuple[float, float]]:
    """The inverse of the event intervals: everything without detected events (clean video)."""
    if not events:
        return [(0.0, total_duration)]

    segments: list[tuple[float, float]] = []
    cursor = 0.0
    for ev in events:
        if ev.start - cursor >= MIN_KEEP_SEGMENT:
            segments.append((cursor, ev.start))
        cursor = max(cursor, ev.end)
    if total_duration - cursor >= MIN_KEEP_SEGMENT:
        segments.append((cursor, total_duration))
    return segments


def export_cut_ffmpeg(
    video_path: Path,
    intervals: list[tuple[float, float]],
    out_path: Path,
    label: str,
    ffmpeg_exe: str,
    safe: bool = False,
) -> None:
    """Direct high-speed FFmpeg cut export (preserves timestamps, VFR/CFR, and sync)."""
    if not intervals:
        print(f"      No intervals for '{label}', skipping export.")
        return

    import tempfile

    print(f"[4/4] Building '{label}' cut ({len(intervals)} segment(s)) via FFmpeg ...")

    with tempfile.TemporaryDirectory() as tmpdir:
        seg_list_file = Path(tmpdir) / "list.txt"
        lines = []

        total = len(intervals)
        for i, (st, en) in enumerate(intervals, 1):
            seg_file = Path(tmpdir) / f"seg_{i:05d}.mp4"
            cmd_seg = [
                ffmpeg_exe, "-y",
                "-ss", f"{st:.3f}",
                "-to", f"{en:.3f}",
                "-i", str(video_path),
                "-c:v", "libx264",
                "-preset", "veryfast" if safe else "ultrafast",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "192k",
                str(seg_file),
            ]
            res = subprocess.run(cmd_seg, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if res.returncode != 0:
                die(f"FFmpeg segment extract failed for {st:.2f}s-{en:.2f}s: {res.stderr.decode('utf-8', errors='replace')[-300:]}")
            lines.append(f"file '{seg_file.resolve()}'")
            if i % 10 == 0 or i == total:
                print(f"      [{label}] processed {i}/{total} segment(s)...", flush=True)

        seg_list_file.write_text("\n".join(lines), encoding="utf-8")

        # Concat demuxer with fast copy
        cmd_concat = [
            ffmpeg_exe, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(seg_list_file),
            "-c", "copy",
            str(out_path),
        ]
        p = subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if p.returncode != 0:
            die(f"FFmpeg concat failed for '{label}': {p.stderr.decode('utf-8', errors='replace')[-300:]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AutoCut — Detect sound events in a video and export two cuts: "
            "<name>_events_only.mp4 and <name>_clean.mp4."
        )
    )
    parser.add_argument("video", type=Path, nargs="?", help="input video file")
    parser.add_argument(
        "--sound-type",
        choices=["laughter", "applause", "speech", "cough", "burp", "burp_extended", "fart", "both", "both_extended"],
        default="laughter",
        help="sound event to detect with YAMNet: 'laughter' (default), 'applause', 'speech', 'cough', etc."
    )
    parser.add_argument(
        "--pad", type=float, default=0.5,
        help="seconds of context kept around each detected event (default: 0.5)")
    parser.add_argument(
        "--tail-hangover", type=float, default=0.25,
        help="seconds of natural acoustic hangover added to the decaying conclusion of detected events (default: 0.25)")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="detection confidence threshold, 0.0–1.0 (default: 0.5). "
             "Lower = more sensitive, higher = stricter.")
    parser.add_argument(
        "--min-gap", type=float, default=1.0,
        help="merge events separated by less than this many seconds (default: 1.0)")
    parser.add_argument(
        "--min-duration", type=float, default=0.3,
        help="drop detected events shorter than this many seconds (default: 0.3)")
    parser.add_argument(
        "--max-duration", type=float, default=None,
        help="drop detected events longer than this many seconds (default: None, no max limit)")
    parser.add_argument(
        "--silence-threshold", type=float, default=None, metavar="DB",
        help="noise gate / silence trim: zero out detections in frames below this dB level, "
             "e.g. -40 (default: None, disabled). Helps eliminate false positives on silence.")
    parser.add_argument(
        "--outdir", type=Path, default=None,
        help="output directory (default: same folder as the input video)")
    parser.add_argument(
        "--teach", type=str, default=None, metavar="LABELS",
        help="learn from labeled sound intervals (+ and -) instead of YAMNet, e.g. "
             "--teach '1:54,2:02,3:44'. Use ranges to mark start-end of an "
             "event precisely: --teach '1:54-2:00,3:44'. Accepts m:ss, "
             "m:ss.s or plain seconds. Trains a small model and scans.")
    parser.add_argument(
        "--save-model", type=Path, default=None, metavar="PATH",
        help="with --teach: save the trained detector to PATH (.npz) for "
             "reuse on other videos with --load-model.")
    parser.add_argument(
        "--load-model", type=Path, default=None, metavar="PATH",
        help="use a detector previously saved with --save-model instead of "
             "YAMNet; no --teach needed. Fast and works without TensorFlow.")
    parser.add_argument(
        "--train-from", type=Path, default=None, metavar="REFERENCE",
        help="learn from a reference video/audio that contains ONLY the "
             "target sounds you want (e.g. cuts pasted together), instead "
             "of YAMNet or --teach. Combines with --save-model.")
    parser.add_argument(
        "--auto-threshold", action="store_true",
        help="automatically compute and apply the optimal detection threshold from audio score distributions")
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="disable automatic audio peak normalization (normalization to -1 dB is enabled by default)")
    parser.add_argument(
        "--min-confidence", type=float, default=0.0,
        help="only cut events with confidence score >= this value (0.0–1.0, e.g. 0.70)")
    parser.add_argument(
        "--safe", action="store_true",
        help="lighter encoding (veryfast preset, capped threads) to avoid "
             "CPU throttling/overheating on long videos. Slower, cooler.")
    parser.add_argument(
        "--no-cut", action="store_true",
        help="scan and train the model without exporting video cuts (fast iteration).")
    parser.add_argument(
        "--info", action="store_true",
        help="print the environment summary (Python, library versions, "
             "ffmpeg) and exit.")
    parser.add_argument(
        "--limit-minutes", type=float, default=None, metavar="N",
        help="only process the first N minutes of the video (for quick "
             "testing on long videos — the full encode of a 2-hour video "
             "takes a long time).")
    parser.add_argument(
        "--label", action="store_true",
        help="open the interactive labeling GUI in your browser: watch the "
             "video, press S at the start of an event and E at the end, then "
             "save. Writes <video>_etichette.txt for --teach @file.")
    parser.add_argument(
        "--ui", action="store_true",
        help="open the full AutoCut local web UI in your browser: choose the video, "
             "preview it, mark events, pick the detection mode and parameters, "
             "run with live progress and browse the results. Requires no "
             "extra dependencies (stdlib only).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.info:
        print_info()
    if args.label:
        if not args.video:
            die("--label requires a video/audio file.")
        try:
            from label_gui import run_gui
        except ImportError:
            die("label_gui.py missing — it must sit next to autocut.py.")
        run_gui(args.video)
        return
    if args.ui:
        try:
            from app_ui import run_app
        except ImportError:
            die("app_ui.py missing — it must sit next to autocut.py.")
        run_app(args.video)
        return
    if not args.video:
        die("No input video given. Run with --help for usage.")
    ffmpeg_exe = check_dependencies()

    video_path: Path = args.video
    if not video_path.is_file():
        die(f"Input video not found: {video_path}")

    outdir = args.outdir if args.outdir else video_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    events_only_path = outdir / f"{stem}_events_only.mp4"
    clean_path = outdir / f"{stem}_clean.mp4"

    with tempfile.TemporaryDirectory(prefix="autocut_") as tmp:
        wav_path = Path(tmp) / "audio.wav"

        # 1. Audio extraction & peak normalization
        extract_audio(video_path, wav_path, ffmpeg_exe)
        audio = load_wav_as_float(wav_path, normalize=not args.no_normalize)
        if args.limit_minutes:
            cut = int(args.limit_minutes * 60 * YAMNET_SAMPLE_RATE)
            audio = audio[:cut]
            print(f"      --limit-minutes: using only the first "
                  f"{args.limit_minutes:g} min of audio.")
        total_duration = len(audio) / YAMNET_SAMPLE_RATE
        print(f"      Audio duration: {total_duration:.1f} s")

        # 2. Detection
        detector_flags = sum([
            bool(args.teach), bool(args.load_model), bool(args.train_from)])
        if detector_flags > 1:
            die("Use only one of --teach, --load-model, --train-from.")
        if args.save_model and not (args.teach or args.train_from):
            die("--save-model requires --teach or --train-from.")

        pos_mask: np.ndarray | None = None
        hard_neg_mask: np.ndarray | None = None

        if args.load_model:
            mdl = load_teach_model(args.load_model)
            w, b, mean, std, proto_pos, proto_neg = mdl
            is_unit_norm = getattr(mdl, "is_unit_norm", False)
            print(f"[2/4] Applying saved detector '{args.load_model}' (unit_norm={is_unit_norm}) ...")
            scores, hop_seconds = apply_teach_model(
                audio, YAMNET_SAMPLE_RATE, w, b, mean, std, proto_pos, proto_neg, is_unit_norm=is_unit_norm)
        elif args.teach:
            pos_labels, neg_labels = parse_teach_labels(args.teach)
            for s, e in pos_labels + neg_labels:
                if s < 0 or e > total_duration:
                    die(f"Labeled interval {s:.1f}-{e:.1f}s is outside the "
                        f"video's 0-{total_duration:.1f}s range.")
            scores, hop_seconds = teach_event_scores(
                audio, YAMNET_SAMPLE_RATE, pos_labels, neg_labels, save_path=args.save_model)

            # Build masks for auto-thresholding
            pos_mask = np.zeros(len(scores), dtype=bool)
            centers = np.arange(len(scores)) * hop_seconds + (0.96 / 2.0)
            for s, e in pos_labels:
                pos_mask |= (centers >= s - 0.25) & (centers <= e + 0.25)
            hard_neg_mask = np.zeros(len(scores), dtype=bool)
            for s, e in neg_labels:
                hard_neg_mask |= (centers >= s - 0.35) & (centers <= e + 0.35)
        elif args.train_from:
            if not args.train_from.is_file():
                die(f"Reference file not found: {args.train_from}")
            ref_wav = Path(tmp) / "reference.wav"
            extract_audio(args.train_from, ref_wav, ffmpeg_exe)
            ref_audio = load_wav_as_float(ref_wav, normalize=not args.no_normalize)
            if len(ref_audio) == 0:
                die("Reference audio is empty.")
            scores, hop_seconds = train_from_reference(
                audio, ref_audio, YAMNET_SAMPLE_RATE,
                save_path=args.save_model)
        else:
            scores, _ = run_yamnet(audio)
            hop_seconds = YAMNET_HOP_SECONDS

        # 3. Silence gating (if enabled)
        if args.silence_threshold is not None:
            win_seconds = 0.96
            scores = apply_silence_gate(
                scores,
                audio,
                YAMNET_SAMPLE_RATE,
                hop_seconds=hop_seconds,
                win_seconds=win_seconds,
                silence_thresh_db=args.silence_threshold,
            )

        # 4. Post-processing & Auto-Thresholding
        if args.auto_threshold:
            if scores.ndim == 1:
                tgt = scores
            elif scores.ndim == 2:
                if scores.shape[1] == 1:
                    tgt = scores[:, 0]
                elif args.sound_type == "laughter":
                    tgt = np.max(scores[:, LAUGHTER_CLASS_INDICES], axis=1)
                elif args.sound_type == "applause":
                    tgt = np.max(scores[:, APPLAUSE_CLASS_INDICES], axis=1)
                elif args.sound_type == "speech":
                    tgt = np.max(scores[:, SPEECH_CLASS_INDICES], axis=1)
                elif args.sound_type == "cough":
                    tgt = np.max(scores[:, COUGH_CLASS_INDICES], axis=1)
                elif args.sound_type == "fart":
                    tgt = scores[:, FART_CLASS_INDEX]
                elif args.sound_type == "both":
                    tgt = np.maximum(scores[:, BURP_CLASS_INDEX], scores[:, FART_CLASS_INDEX])
                elif args.sound_type == "burp_extended":
                    tgt = np.max(scores[:, [BURP_CLASS_INDEX, THROAT_CLASS_INDEX, SNORT_CLASS_INDEX, HICCUP_CLASS_INDEX]], axis=1)
                elif args.sound_type == "burp":
                    tgt = scores[:, BURP_CLASS_INDEX]
                else:
                    tgt = np.max(scores[:, LAUGHTER_CLASS_INDICES], axis=1)
            else:
                tgt = scores.flatten()
            args.threshold = compute_optimal_threshold(
                tgt, pos_mask=pos_mask, hard_neg_mask=hard_neg_mask)
            print(f"      [Auto-Threshold] Applying calculated threshold: {args.threshold:.2f}")

        effective_pad = 0.0 if (args.no_cut and "--pad" not in sys.argv) else args.pad
        if args.no_cut and effective_pad == 0.0:
            print("      [Scan Only] Padding disabled (0.0 s) to show exact acoustic boundaries.")

        events = detect_events(
            scores,
            hop_seconds=hop_seconds,
            threshold=args.threshold,
            min_gap=args.min_gap,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            pad=effective_pad,
            total_duration=total_duration,
            sound_type=args.sound_type,
            min_confidence=args.min_confidence,
            audio=audio,
            sr=YAMNET_SAMPLE_RATE,
            tail_hangover=args.tail_hangover,
        )

        if not events:
            print("\nNo events detected — nothing to cut. "
                  "Try a lower --threshold (e.g. 0.3).")
            if args.no_cut:
                print("Detection completed (no cuts exported).")
            sys.exit(0)

        print(f"\n      Detected {len(events)} event(s):")
        for i, ev in enumerate(events, 1):
            print(f"        {i:2d}. {ev.start:7.2f} s → {ev.end:7.2f} s "
                  f"({ev.duration:.2f} s, conf: {ev.confidence*100:.0f}%)")

        event_total = sum(ev.duration for ev in events)
        print(f"      Total event duration: {event_total:.1f} s / {total_duration:.1f} s")

        if args.no_cut:
            print("\n      --no-cut: scanning and training complete. Video cuts skipped.")
            if args.save_model:
                print(f"      Model saved to: {args.save_model}")
            print("\nDone! (Scan only)")
            return

        # 4. Build and write both cuts
        if args.safe:
            print("      --safe: encoding with veryfast preset (cooler).")
        clean_intervals = complement_intervals(events, total_duration)

        event_intervals = [(ev.start, ev.end) for ev in events]
        export_cut_ffmpeg(
            video_path=video_path,
            intervals=event_intervals,
            out_path=events_only_path,
            label="events only",
            ffmpeg_exe=ffmpeg_exe,
            safe=args.safe,
        )

        export_cut_ffmpeg(
            video_path=video_path,
            intervals=clean_intervals,
            out_path=clean_path,
            label="clean video",
            ffmpeg_exe=ffmpeg_exe,
            safe=args.safe,
        )

    print(f"\nDone!")
    print(f"  Events only : {events_only_path}")
    print(f"  Clean video : {clean_path}")


if __name__ == "__main__":
    main()
