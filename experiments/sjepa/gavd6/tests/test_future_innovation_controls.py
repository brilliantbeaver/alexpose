import unittest

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.future_innovation.fi_controls import (
    block_shuffle,
    controlled_history,
    mismatch_donors,
)
from gavd6_sjepa.research_directions.future_innovation.fi_nested_training import (
    isolated_split,
    partition_control,
)
from gavd6_sjepa.research_directions.future_innovation.fi_residual_models import (
    SkeletonResidualHead,
)


class FutureInnovationControlTests(unittest.TestCase):
    def test_shuffle_preserves_joint_channels_and_four_frame_blocks(self):
        history = np.arange(32 * 33 * 4).reshape(32, 33, 4)
        result = block_shuffle(history, "w")
        self.assertEqual(result.shape, history.shape)
        self.assertFalse(np.array_equal(result, history))
        np.testing.assert_array_equal(result, block_shuffle(history, "w"))
        self.assertEqual(
            {tuple(row.flatten()) for row in result.reshape(8, 4, 33, 4)},
            {tuple(row.flatten()) for row in history.reshape(8, 4, 33, 4)},
        )

    def test_mismatch_is_partition_local_different_source_and_without_replacement(self):
        sources = np.array(["a", "a", "b", "b", "c", "c"])
        metadata = np.arange(12).reshape(6, 2)
        donors, replacement = mismatch_donors(sources, metadata)
        self.assertFalse(replacement)
        self.assertEqual(len(set(donors)), 6)
        self.assertFalse(np.any(sources == sources[donors]))
        with self.assertRaises(ValueError):
            mismatch_donors(["a", "a"], metadata[:2])
        cohort = pd.DataFrame(
            {"window_id": [f"w{i}" for i in range(6)], "video_id": sources}
        )
        arrays = {
            "skeleton": np.arange(6 * 32 * 33 * 4).reshape(6, 32, 33, 4),
            "matching": metadata,
        }
        audit = []
        partition_control(
            "clip-mismatch", np.array([0, 1, 2, 3]), cohort, arrays, "train", audit
        )
        self.assertLessEqual(
            set(audit[0]["donor_window_ids"]), {"w0", "w1", "w2", "w3"}
        )
        with self.assertRaises(ValueError):
            isolated_split(cohort, np.array([0, 2, 4]), np.array([1, 3, 5]))

    def test_validity_only_keeps_shape_and_parameter_count(self):
        history = np.ones((4, 32, 33, 4))
        result, _ = controlled_history(
            "no-skeleton", history, list("abcd"), list("abcd"), np.ones((4, 2))
        )
        np.testing.assert_array_equal(result[..., :3], 0)
        np.testing.assert_array_equal(result[..., 3], history[..., 3])
        counts = [
            sum(p.numel() for p in SkeletonResidualHead(132, 32).parameters())
            for _ in range(5)
        ]
        self.assertEqual(len(set(counts)), 1)
