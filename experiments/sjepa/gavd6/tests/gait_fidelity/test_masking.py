import numpy as np
import unittest

from gavd6_sjepa.research_directions.gait_fidelity.masking import POLICIES, sample_mask, coverage_audit, patch_support
from gavd6_sjepa.research_directions.gait_fidelity.training import normalize_batch


class MaskingTests(unittest.TestCase):

    def test_every_policy_has_same_realized_budget_and_retains_context(self):
        observed = np.ones((4, 64, 12), bool)
        observed[1, 4:22, 6:] = False
        observed[2] = False; observed[2, 2, 3] = True
        observed[3] = False
        available = patch_support(observed)
        counts = []
        for policy in POLICIES:
            mask, report = sample_mask(observed, policy, rng=np.random.default_rng(17), return_receipt=True)
            assert not (mask & ~available).any()
            assert ((available & ~mask).sum((1, 2))[:3] >= 1).all()
            assert not mask[2:].any()
            counts.append(mask.sum((1, 2)))
            assert report['sparse_no_added_mask'] == 2
        assert all(np.array_equal(counts[0], x) for x in counts)


    def test_every_joint_time_token_can_be_context_and_target(self):
        for policy in POLICIES:
            observed = np.ones((1, 32, 12), bool)
            audit = coverage_audit(observed, draws=128, seed=17, policy=policy)
            assert audit['never_hidden_eligible_slots'] == audit['always_hidden_eligible_slots'] == 0


    def test_masks_reproducible_but_advance(self):
        observed = np.ones((1, 64, 12), bool)
        a = np.random.default_rng(17)
        first = sample_mask(observed, 'graph_time', rng=a)
        second = sample_mask(observed, 'graph_time', rng=a)
        assert np.array_equal(first, sample_mask(observed, 'graph_time', rng=np.random.default_rng(17)))
        assert not np.array_equal(first, second)


    def test_context_only_normalization_does_not_reveal_hidden_coordinates(self):
        rng = np.random.default_rng(4)
        inputs = dict(xy=rng.normal(100, 40, (2, 32, 12, 2)).astype('float32'),
            observed=np.ones((2, 32, 12), bool),confidence=np.ones((2, 32, 12), 'float32'),
            timestamps=np.tile(np.arange(32)/25, (2,1)))
        hidden = sample_mask(inputs['observed'], 'graph_time', rng=rng)
        altered = {k: v.copy() for k, v in inputs.items()}
        frame_hidden = np.repeat(hidden, 4, axis=1)
        altered['xy'][frame_hidden] = 999999
        first, origin, scale, _ = normalize_batch(inputs, hidden)
        second, origin2, scale2, _ = normalize_batch(altered, hidden)
        np.testing.assert_array_equal(origin, origin2)
        np.testing.assert_array_equal(scale, scale2)
        np.testing.assert_array_equal(first['xy'][~frame_hidden], second['xy'][~frame_hidden])
