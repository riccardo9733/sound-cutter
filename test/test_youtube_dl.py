#!/usr/bin/env python3
"""Tests for youtube_dl module integration."""

import unittest
from pathlib import Path
from youtube_dl import (
    YOUTUBE_JOB,
    ensure_ffmpeg_in_path,
    fetch_video_info,
    format_bytes,
    format_seconds,
    parse_time_str,
    sanitize_filename,
)


class TestYouTubeDL(unittest.TestCase):

    def test_parse_time_str(self):
        self.assertEqual(parse_time_str("01:30"), 90.0)
        self.assertEqual(parse_time_str("00:02:15"), 135.0)
        self.assertEqual(parse_time_str("45s"), 45.0)
        self.assertEqual(parse_time_str("120"), 120.0)
        self.assertIsNone(parse_time_str(""))
        self.assertIsNone(parse_time_str(None))

    def test_sanitize_filename(self):
        cleaned = sanitize_filename("My Video: Test? / \\ * < > | Name")
        self.assertNotIn(":", cleaned)
        self.assertNotIn("?", cleaned)
        self.assertNotIn("/", cleaned)
        self.assertNotIn("<", cleaned)
        self.assertIn("My_Video", cleaned)

    def test_format_bytes(self):
        self.assertEqual(format_bytes(500), "500 B")
        self.assertEqual(format_bytes(1024 * 512), "512.0 KB")
        self.assertEqual(format_bytes(1024 * 1024 * 15), "15.0 MB")
        self.assertEqual(format_bytes(1024 * 1024 * 1024 * 2), "2.00 GB")

    def test_format_seconds(self):
        self.assertEqual(format_seconds(45), "00:45")
        self.assertEqual(format_seconds(125), "02:05")
        self.assertEqual(format_seconds(3665), "01:01:05")
        self.assertEqual(format_seconds(None), "--:--")

    def test_ffmpeg_found(self):
        ff = ensure_ffmpeg_in_path()
        self.assertIsNotNone(ff, "FFmpeg should be detected in PATH or virtualenv")
        self.assertTrue(Path(ff).is_file(), f"FFmpeg binary {ff} should exist")

    def test_job_initial_status(self):
        status = YOUTUBE_JOB.get_status()
        self.assertIn("running", status)
        self.assertIn("progress", status)
        self.assertIn("stage", status)
        self.assertIn("lines", status)

    def test_pipeline_job_initial_status(self):
        from youtube_dl import YOUTUBE_PIPELINE_JOB
        status = YOUTUBE_PIPELINE_JOB.get_status()
        self.assertIn("running", status)
        self.assertFalse(status["running"])
        self.assertIn("total_videos", status)
        self.assertIn("saved_files", status)
        self.assertIn("progress", status)

    def test_available_models(self):
        from youtube_dl import get_available_models
        models = get_available_models()
        self.assertIsInstance(models, list)
        self.assertTrue(any(m["id"] == "yamnet" for m in models))


if __name__ == "__main__":
    unittest.main()

