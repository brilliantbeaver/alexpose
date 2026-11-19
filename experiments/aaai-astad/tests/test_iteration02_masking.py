import numpy as np
import pytest
from aaai_gait.masking import core_masks,draw_masks,make_mask_bank,rng_stream
from aaai_gait.context_prediction import ContextConfig,make_mask_bank as historical_bank

def test_uniform_bank_preserves_historical_bytes():
    actual=make_mask_bank("U")
    expected=historical_bank(ContextConfig(),17,16).numpy()
    np.testing.assert_array_equal(actual.hidden,expected)
    np.testing.assert_array_equal(actual.targets,expected)
    assert actual.sha256==make_mask_bank("U").sha256
    with pytest.raises(ValueError): actual.hidden[0,0]=False

@pytest.mark.parametrize("policy",["J","A"])
def test_exact_core_partition_and_probability(policy):
    cores=core_masks(policy)
    assert cores.shape==(16,128)
    np.testing.assert_array_equal(cores.sum(0),np.ones(128))
    np.testing.assert_array_equal(cores.sum(1),np.full(16,8))
    np.testing.assert_allclose(cores.mean(0)+(1-cores.mean(0))*56/120,.5,atol=0,rtol=0)
    signatures={row.tobytes() for row in cores};mirror=np.arange(16)^1
    for row in cores:
        assert row.reshape(8,16)[:,mirror].reshape(-1).tobytes() in signatures
        assert row.reshape(8,16)[::-1].reshape(-1).tobytes() in signatures

@pytest.mark.parametrize("policy",["U","J","A"])
def test_budget_and_separate_named_rng(policy):
    masks=draw_masks(policy,128,rng=rng_stream(42,"masks"))
    assert masks.dtype==bool and np.all(masks.sum(1)==64)
    source_before=rng_stream(42,"source_order").integers(100,size=20)
    draw_masks(policy,700,rng=rng_stream(42,"masks"))
    np.testing.assert_array_equal(source_before,rng_stream(42,"source_order").integers(100,size=20))

def test_leg_targets_are_core_only_and_repeat_twice():
    bank=make_mask_bank("E_leg")
    assert bank.seed==117
    assert np.all(bank.hidden.sum(1)==64) and np.all(bank.targets.sum(1)==8)
    assert not np.any(bank.targets&~bank.hidden)
    counts=bank.targets.reshape(16,8,16).sum(0)
    assert np.all(counts[:,:8]==0) and np.all(counts[:,8:]==2)
    assert bank.sha256!=make_mask_bank("A").sha256
    with pytest.raises(ValueError): make_mask_bank("E_leg",repeats=8)
