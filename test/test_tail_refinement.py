import unittest
import numpy as np
import autocut


class TestTailRefinement(unittest.TestCase):
    def test_noise_floor(self):
        sr = 16000
        # 2 seconds of quiet noise (amplitude 0.005)
        np.random.seed(42)
        silence = np.random.normal(0, 0.005, int(2 * sr)).astype(np.float32)
        nf = autocut.calculate_track_noise_floor(silence, sr)
        self.assertGreater(nf, 0.001)
        self.assertLess(nf, 0.010)

    def test_tail_extension_on_trailing_signal(self):
        sr = 16000
        total_dur = 6.0
        np.random.seed(42)
        # Background: 0.002 amplitude
        audio = np.random.normal(0, 0.002, int(total_dur * sr)).astype(np.float32)
        # Event from 1.0s to 3.5s (peak at 1.5s amplitude 0.3, decaying to 0.03 at 3.5s)
        t = np.linspace(0, 2.5, int(2.5 * sr), endpoint=False)
        decay_envelope = 0.3 * np.exp(-t * 0.9)  # at t=2.0 (abs 3.0s), envelope ~ 0.05
        burst = (np.sin(2 * np.pi * 300 * t) * decay_envelope).astype(np.float32)
        ev_start_idx = int(1.0 * sr)
        audio[ev_start_idx:ev_start_idx + len(burst)] += burst

        # Simulated detector trigger: only caught 1.0s to 1.8s
        start_t = 1.0
        end_t = 1.8
        gnf = autocut.calculate_track_noise_floor(audio, sr)

        refined_end = autocut.refine_event_tail(
            audio, sr, start_t, end_t, total_dur,
            max_extension=3.0, tail_hangover=0.35,
            global_noise_floor=gnf, min_gap=1.0
        )
        self.assertGreater(refined_end, 2.8)
        self.assertLessEqual(refined_end, total_dur)

    def test_merge_safety_cap(self):
        # 10 discrete triggers spaced 0.8s apart (min_gap=1.0s)
        # Without a safety cap, all 10 would merge into a single 20s event.
        # With the safety cap, events must not exceed max_merge_duration (10.0s).
        scores = np.zeros(100, dtype=np.float32)
        # Place triggers every 4 frames (each frame is ~0.48s)
        for i in range(10, 60, 4):
            scores[i] = 0.9

        events = autocut.detect_events(
            scores=scores,
            hop_seconds=0.48,
            threshold=0.5,
            min_gap=2.5,
            min_duration=0.3,
            pad=0.0,
            total_duration=50.0,
        )
        self.assertGreater(len(events), 1)
        for ev in events:
            self.assertLessEqual(ev.duration, 10.0)


if __name__ == '__main__':
    unittest.main()
