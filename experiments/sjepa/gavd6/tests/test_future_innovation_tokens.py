import unittest

import numpy as np

from gavd6_sjepa.research_directions.future_innovation.fi_token_regions import (
    context_indices,
    pool_context,
    pool_target,
    region_masks,
    token_index,
)


class FutureInnovationTokenTests(unittest.TestCase):
    def test_temporal_major_pooling_known_indices(self):
        tokens = np.arange(32 * 24 * 24, dtype=float)[:, None]
        boxes = np.tile([0.4, 0.3, 0.6, 0.7], (64, 1))
        person, background = region_masks(boxes[0])
        pooled_person, pooled_background = pool_target(tokens, boxes)
        target = tokens.reshape(32, 576, 1)[19]
        np.testing.assert_array_equal(pooled_person, target[person].mean(axis=0))
        np.testing.assert_array_equal(
            pooled_background, target[background].mean(axis=0)
        )
        self.assertEqual(token_index(19, 4, 5), 19 * 576 + 4 * 24 + 5)
        pools = pool_context(tokens[context_indices()], boxes)
        self.assertAlmostEqual(pools[0], np.arange(16 * 576).mean())
        self.assertAlmostEqual(
            pools[1], tokens.reshape(32, 576, 1)[15, person].mean(), places=3
        )
        self.assertFalse((person & background).any())
        self.assertGreater((~person & ~background).sum(), 0)

    def test_empty_person_and_background_fail(self):
        with self.assertRaises(ValueError):
            region_masks([0.001, 0.001, 0.002, 0.002])
        with self.assertRaises(ValueError):
            region_masks([0, 0, 1, 1])
