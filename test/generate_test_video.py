#!/usr/bin/env python3
"""Generate a test video with real burp sounds at known timestamps.

Smoke-test autocut.py without hunting for a real recording:

    python test/generate_test_video.py      # downloads a PD sound if needed
    python autocut.py test/test_video.mp4 --outdir test/out \
        --threshold 0.2

The burp sample comes from "Burping 5.ogg" (Wikimedia Commons, public
domain), a recording YAMNet classifies as *Burping, eructation* with high
confidence. The generator splices that burp window into a 15 s track of quiet
background noise at four known offsets and reports each placement's YAMNet
score, so you can pick a sensible --threshold.

Note: YAMNet's burp score for identical copies varies with what surrounds
them, so scores differ per placement even though the copies are the same.
"""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import wave

import numpy as np

SR = 16000  # YAMNet input rate
DURATION = 15.0
OUT_DIR = Path(__file__).resolve().parent
ASSET_DIR = OUT_DIR / "assets"
ASSET_URL = "https://commons.wikimedia.org/wiki/Special:FilePath/Burping%205.ogg"
ASSET_PATH = ASSET_DIR / "Burping 5.ogg"
WAV_PATH = OUT_DIR / "test_audio.wav"
MP4_PATH = OUT_DIR / "test_video.mp4"

# Burp window inside Burping 5.ogg (confirmed by YAMNet: score ~0.7 at 3.1 s).
SOURCE_BURP_START = 2.4
SOURCE_BURP_END = 4.4

# Where the burps go in the test video.
BURP_STARTS = [2.0, 6.0, 10.0, 14.0]

# Background levels (kept low so the burps stay detectable).
NOISE_AMP = 0.01
BURP_AMP = 0.9


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg  # bundled with moviepy
    return imageio_ffmpeg.get_ffmpeg_exe()


def fetch_asset() -> None:
    if ASSET_PATH.is_file() and ASSET_PATH.stat().st_size > 10000:
        return
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading burp sample (Wikimedia Commons, public domain) ...")
    urllib.request.urlretrieve(ASSET_URL, ASSET_PATH)


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        n = wav.getnframes()
        raw = wav.readframes(n)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def burp_scores(wav_path: Path, starts: list[float]) -> dict[float, float]:
    """Run YAMNet on a WAV and report the max burp score per placement."""
    import certifi
    import os
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    import tensorflow_hub as hub

    audio = load_wav(wav_path)
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    scores = model(audio.astype(np.float32))[0].numpy()
    burp = scores[:, 39]
    out = {}
    for st in starts:
        lo = max(0, int(st / 0.48))
        hi = min(len(burp), int((st + 2.0) / 0.48) + 1)
        out[st] = float(burp[lo:hi].max()) if hi > lo else 0.0
    return out


def main() -> None:
    fetch_asset()
    ffmpeg = find_ffmpeg()

    with tempfile.TemporaryDirectory() as tmp:
        # 1. Convert the source recording to 16 kHz mono WAV.
        src_wav = Path(tmp) / "src.wav"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(ASSET_PATH),
             "-ac", "1", "-ar", str(SR), str(src_wav)],
            check=True,
        )
        src = load_wav(src_wav)

    # 2. Cut the confirmed burp window out of the recording.
    a = int(SOURCE_BURP_START * SR)
    b = int(SOURCE_BURP_END * SR)
    burp = src[a:b]

    # 3. Build the 15 s track: quiet background + the burps at known offsets.
    rng = np.random.default_rng(7)
    n = int(SR * DURATION)
    t = np.arange(n, dtype=np.float32) / SR
    track = np.zeros(n, dtype=np.float32)
    track += NOISE_AMP * rng.normal(0, 1, n).astype(np.float32) \
        * (0.5 + 0.5 * np.sin(2 * np.pi * 3.2 * t)) ** 2

    for start in BURP_STARTS:
        piece = burp / (np.abs(burp).max() + 1e-9) * BURP_AMP
        i = int(start * SR)
        j = min(n, i + len(piece))
        track[i:j] += piece[: j - i]

    track = np.clip(track, -1.0, 1.0)

    # 4. Write WAV + mux into an MP4 (black frame, H.264 + AAC).
    with wave.open(str(WAV_PATH), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes((track * 32767).astype(np.int16).tobytes())

    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=black:s=320x240:d=15",
         "-i", str(WAV_PATH),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(MP4_PATH)],
        check=True,
    )

    print(f"Wrote {MP4_PATH} ({DURATION:.0f} s) — burps at {BURP_STARTS}.")
    print("YAMNet burp scores per placement:")
    for st, score in burp_scores(WAV_PATH, BURP_STARTS).items():
        print(f"  {st:>4.1f} s -> {score:.2f}")
    print("Hint: run autocut.py with --threshold 0.2 to catch the "
          "confident ones.")


if __name__ == "__main__":
    sys.exit(main())
