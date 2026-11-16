"""Scientific invariants of the isolated teaching mask laboratory."""
import numpy as np
import pytest

from worldmodels.mask_design import POLICIES, REFERENCE_JOINTS, design_prefix_mask
from worldmodels.data import JOINT_IDS


def observation():
    rng = np.random.default_rng(93)
    x = rng.normal(size=(16, 16, 2))
    valid = rng.random((16, 16)) > .15
    valid[-1] = True
    x[~valid] = np.nan
    return x, valid, np.arange(16) * .1


@pytest.mark.parametrize('policy', POLICIES)
def test_support_anchors_budget_and_replay(policy):
    x, valid, t = observation()
    hidden, protected, info = design_prefix_mask(x, valid, t, policy)
    assert not np.any(hidden & (~valid | protected))
    assert protected[-1].all()
    for joint in REFERENCE_JOINTS:
        j = JOINT_IDS.index(joint)
        np.testing.assert_array_equal(protected[:, j], valid[:, j])
    eligible = (valid & ~protected).sum()
    assert hidden.sum() == int(np.floor(.4 * eligible + .5)) == info['hidden_tokens']
    np.testing.assert_array_equal(hidden, design_prefix_mask(x, valid, t, policy)[0])
    changed = x.copy()
    changed[~valid] = 1e20
    np.testing.assert_array_equal(hidden, design_prefix_mask(changed, valid, t, policy)[0])


def test_future_excluded_and_budget_matched():
    prefix, valid, t = observation()
    whole = np.concatenate([prefix, np.ones((8, 16, 2))])
    counts = []
    for policy in POLICIES:
        before = design_prefix_mask(whole[:16], valid, t, policy)[0]
        whole[16:] = np.nan
        after = design_prefix_mask(whole[:16], valid, t, policy)[0]
        np.testing.assert_array_equal(before, after)
        counts.append(int(before.sum()))
    assert len(set(counts)) == 1


def test_invalid_contract_rejected():
    x, valid, t = observation()
    with pytest.raises(ValueError, match='timestamps'):
        design_prefix_mask(x, valid, t[::-1], 'uniform')
    with pytest.raises(ValueError, match='finite'):
        x[valid] = np.nan
        design_prefix_mask(x, valid, t, 'uniform')


def test_motion_policy_can_hide_observed_tokens_without_motion_pairs():
    x = np.zeros((16,16,2))
    valid = np.ones((16,16),bool)
    t = np.arange(16)*.1
    assert any(design_prefix_mask(x,valid,t,'motion_balanced',seed=s)[0][0,8]
               for s in range(100))
