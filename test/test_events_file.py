import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import autocut
from app_ui import build_command


class TestEventsFile(unittest.TestCase):

    def setUp(self):
        self.fake_video = Path("test/test_video.mp4").resolve()

    def test_parse_events_file_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([[1.2, 3.4], {"start": 5.0, "end": 6.8}], f)
            f.flush()
            intervals = autocut.parse_events_file(Path(f.name))
            self.assertEqual(intervals, [(1.2, 3.4), (5.0, 6.8)])

    def test_parse_events_file_text(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# comment\n0:01.20-0:03.40\n5.0-6.8\n")
            f.flush()
            intervals = autocut.parse_events_file(Path(f.name))
            self.assertEqual(intervals, [(1.2, 3.4), (5.0, 6.8)])

    def test_app_ui_build_command_approved_events(self):
        approved = [[2.0, 4.0], [7.0, 9.0]]
        cmd, _ = build_command({
            "video": str(self.fake_video),
            "approved_events": approved,
            "export_mode": "events",
        })
        self.assertIn("--events-file", cmd)
        ev_file_idx = cmd.index("--events-file") + 1
        ev_file = Path(cmd[ev_file_idx])
        self.assertTrue(ev_file.is_file())
        parsed = json.loads(ev_file.read_text())
        self.assertEqual(parsed, approved)
        self.assertIn("--export", cmd)
        self.assertEqual(cmd[cmd.index("--export") + 1], "events")

    def test_app_ui_build_command_empty_approved_events_raises(self):
        with self.assertRaises(ValueError):
            build_command({
                "video": str(self.fake_video),
                "approved_events": [],
                "export_mode": "events",
            })

    def test_autocut_cli_events_file_arg(self):
        with patch("sys.argv", ["autocut.py", str(self.fake_video), "--events-file", "events.json"]):
            args = autocut.parse_args()
            self.assertEqual(args.events_file, Path("events.json"))


if __name__ == "__main__":
    unittest.main()
