#!/usr/bin/env python3
"""Tests for file_mgr module."""

import unittest
from pathlib import Path
import file_mgr


class TestFileMgr(unittest.TestCase):

    def test_format_size(self):
        self.assertEqual(file_mgr.format_size(100), "100 B")
        self.assertEqual(file_mgr.format_size(1024 * 10), "10.0 KB")
        self.assertEqual(file_mgr.format_size(1024 * 1024 * 50), "50.0 MB")
        self.assertEqual(file_mgr.format_size(1024 * 1024 * 1024 * 3), "3.00 GB")

    def test_get_file_type(self):
        self.assertEqual(file_mgr.get_file_type(Path("video.mp4")), "video")
        self.assertEqual(file_mgr.get_file_type(Path("audio.mp3")), "audio")
        self.assertEqual(file_mgr.get_file_type(Path("model.npz")), "model")
        self.assertEqual(file_mgr.get_file_type(Path("notes.txt")), "text")
        self.assertEqual(file_mgr.get_file_type(Path("data.bin")), "other")

    def test_get_project_tree(self):
        tree = file_mgr.get_project_tree()
        self.assertIn("folders", tree)
        self.assertIn("total_files", tree)
        self.assertIn("total_size", tree)
        self.assertIn("total_size_str", tree)

        folder_keys = [f["key"] for f in tree["folders"]]
        self.assertIn("downloads", folder_keys)
        self.assertIn("results", folder_keys)
        self.assertIn("models", folder_keys)
        self.assertIn(".", folder_keys)


if __name__ == "__main__":
    unittest.main()
