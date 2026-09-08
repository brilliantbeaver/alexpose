import unittest

import numpy as np

from gavd6_sjepa.research_directions.future_innovation.fi_validity_audits import (
    pixel_edits,
)
from gavd6_sjepa.research_directions.future_innovation.fi_video_pose import (
    CropGeometry,
    annotation_boxes,
)


class FutureInnovationBoxTests(unittest.TestCase):
    def test_boxes_and_pixels_share_resize_crop(self):
        geometry = CropGeometry(100, 200)
        self.assertEqual(geometry.resized_hw, (438, 876))
        np.testing.assert_array_equal(geometry.offset_xy, [246, 27])
        boxes, retention = geometry.boxes(np.array([[0.4, 0.2, 0.6, 0.8]]))
        expected = (
            np.array([[0.4 * 876, 0.2 * 438, 0.6 * 876, 0.8 * 438]])
            - [246, 27, 246, 27]
        ) / 384
        np.testing.assert_allclose(boxes, expected)
        np.testing.assert_allclose(retention, [1])
        video = np.zeros((1, 100, 200, 3), dtype=np.uint8)
        video[:, 20:80, 80:120] = 255
        resized = geometry.pixels(video)
        y, x = np.where(resized[0, :, :, 0] > 127)
        np.testing.assert_allclose(
            [x.min(), y.min(), x.max() + 1, y.max() + 1], boxes[0] * 384, atol=1
        )

    def test_severe_cropping_and_invalid_annotation(self):
        _, retention = CropGeometry(100, 200).boxes(np.array([[0, 0, 0.1, 1]]))
        self.assertLess(retention[0], 0.9)
        with self.assertRaises(ValueError):
            annotation_boxes([{"bbox": "{}", "vid_info": "{}"}])

    def test_pixel_edits_keep_past_and_opposite_regions_fixed(self):
        rng = np.random.default_rng(4)
        video = rng.integers(0, 256, (64, 32, 48, 3), dtype=np.uint8)
        donor = rng.integers(0, 256, video.shape, dtype=np.uint8)
        boxes = np.tile([0.25, 0.25, 0.75, 0.75], (64, 1))
        person, background = pixel_edits(video, boxes, donor, boxes, feather=3)
        np.testing.assert_array_equal(person[:32], video[:32])
        np.testing.assert_array_equal(background[:32], video[:32])
        np.testing.assert_array_equal(person[32:, :, :12], video[32:, :, :12])
        np.testing.assert_array_equal(
            background[32:, 8:24, 12:36], video[32:, 8:24, 12:36]
        )
        self.assertGreater(np.abs(person.astype(float) - video).sum(), 0)
        self.assertGreater(np.abs(background.astype(float) - video).sum(), 0)
