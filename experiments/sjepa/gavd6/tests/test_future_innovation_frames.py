import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from gavd6_sjepa.research_directions.future_innovation.fi_contracts import (
    FRAME,
    FrameContract,
)
from gavd6_sjepa.research_directions.future_innovation.fi_video_pose import (
    decode_exact_window,
    normalize_skeleton,
)


class FutureInnovationFrameTests(unittest.TestCase):
    def test_exact_decode_absolute_zero_based_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.avi"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (48, 32)
            )
            self.assertTrue(writer.isOpened())
            for i in range(75):
                writer.write(np.full((32, 48, 3), i * 3, dtype=np.uint8))
            writer.release()
            frames, fps = decode_exact_window(path, 4)
            self.assertEqual(frames.shape, (64, 32, 48, 3))
            self.assertEqual(fps, 25)
            np.testing.assert_allclose(
                frames.mean(axis=(1, 2, 3)), np.arange(4, 68) * 3, atol=1
            )
            with self.assertRaises(ValueError):
                decode_exact_window(path, 20)

    def test_context_and_target_tubelets(self):
        self.assertEqual(FRAME.target_tubelet_start // FRAME.tubelet_size, 19)
        self.assertEqual(FRAME.context_stop_exclusive // FRAME.tubelet_size, 16)
        with self.assertRaises(ValueError):
            FrameContract(context_stop_exclusive=31)
        with self.assertRaises(ValueError):
            FrameContract(
                target_tubelet_start=30,
                target_tubelet_stop_exclusive=32,
                target_horizon_frames=0,
            )

    def test_past_normalization_preserves_missingness(self):
        raw = np.ones((32, 33, 4), dtype=np.float32) * 0.5
        raw[..., 3] = 0.9
        raw[:, 11:13, 1] = 0.25
        raw[0, 5, :2] = np.nan
        boxes = np.tile([0, 0, 1, 1], (32, 1))
        history, scale = normalize_skeleton(raw, boxes)
        self.assertAlmostEqual(scale, 0.25)
        np.testing.assert_array_equal(history[0, 5, :2], 0)
        self.assertEqual(history[0, 5, 3], 0)
        self.assertAlmostEqual(history[0, 5, 2], 0.9, places=6)
        with self.assertRaises(ValueError):
            normalize_skeleton(np.tile(raw, (2, 1, 1)), np.tile(boxes, (2, 1)))
