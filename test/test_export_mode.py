import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import autocut
from app_ui import build_command


class TestExportMode(unittest.TestCase):

    def test_cli_argument_parsing(self):
        # Default is 'both'
        with patch.object(sys, "argv", ["autocut.py", "test.mp4"]):
            args = autocut.parse_args()
            self.assertEqual(args.export, "both")

        # Flag --export events
        with patch.object(sys, "argv", ["autocut.py", "test.mp4", "--export", "events"]):
            args = autocut.parse_args()
            self.assertEqual(args.export, "events")

        # Flag --export clean
        with patch.object(sys, "argv", ["autocut.py", "test.mp4", "--export", "clean"]):
            args = autocut.parse_args()
            self.assertEqual(args.export, "clean")

        # Alias --export-mode
        with patch.object(sys, "argv", ["autocut.py", "test.mp4", "--export-mode", "events"]):
            args = autocut.parse_args()
            self.assertEqual(args.export, "events")

    def test_app_ui_build_command(self):
        fake_video = Path("test/test_video.mp4").resolve()

        # Default / 'both'
        cmd, _ = build_command({"video": str(fake_video), "export_mode": "both"})
        self.assertIn("--export", cmd)
        self.assertEqual(cmd[cmd.index("--export") + 1], "both")

        # 'events'
        cmd, _ = build_command({"video": str(fake_video), "export_mode": "events"})
        self.assertIn("--export", cmd)
        self.assertEqual(cmd[cmd.index("--export") + 1], "events")

        # 'clean'
        cmd, _ = build_command({"video": str(fake_video), "export_mode": "clean"})
        self.assertIn("--export", cmd)
        self.assertEqual(cmd[cmd.index("--export") + 1], "clean")


if __name__ == "__main__":
    unittest.main()
