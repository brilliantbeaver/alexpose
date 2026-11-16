"""Method invariants, leakage guards and bounded software smoke training."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from worldmodels.demo import make_demo
from worldmodels.masking import JOINT_IDS, uniform_mask, forecast_mask
from worldmodels.model import ModelConfig, SkeletonJEPA, loss_terms, variance_covariance
from worldmodels.training import TrainConfig, fit, encode, anomaly_scores, save_checkpoint, load_checkpoint, seed_everything


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(latent_dim=0):
    seed_everything(13)
    data = make_demo(n=8,frames=4)
    model = SkeletonJEPA(ModelConfig(frames=4,dim=16,heads=2,encoder_layers=1,latent_dim=latent_dim))
    return model, data, torch.from_numpy(data['x']), torch.from_numpy(data['valid'])


def test_joint_contract_uniformity_and_missingness():
    valid = torch.ones(1,2,16,dtype=torch.bool)
    valid[0,0,0] = False
    counts = torch.zeros_like(valid,dtype=torch.float32)
    g = torch.Generator().manual_seed(8)
    for _ in range(800):
        mask = uniform_mask(valid, .5, generator=g)
        assert mask.sum() == 16
        assert not (mask & ~valid).any()
        counts += mask
    assert float(counts[valid].std()/counts[valid].mean()) < .08
    with pytest.raises(ValueError):
        uniform_mask(torch.ones(1,2,33,dtype=torch.bool))
    with pytest.raises(ValueError):
        ModelConfig(joint_ids=JOINT_IDS[:-1])


def test_no_future_coordinates_or_visibility_leak_into_predictions():
    model, _, x, valid = fixture()
    model.eval()
    mask = forecast_mask(valid,2)
    first = model(x,valid,mask)
    changed = x.clone()
    changed[:,2:] += 100
    second = model(changed,valid,mask)
    torch.testing.assert_close(first['coordinates'], second['coordinates'],rtol=0,atol=0)
    assert not torch.allclose(first['teacher'],second['teacher'])
    altered_valid=valid.clone()
    altered_valid[:,3,0] = False
    third=model(changed,altered_valid,forecast_mask(altered_valid,2))
    torch.testing.assert_close(first['coordinates'],third['coordinates'],rtol=0,atol=0)


def test_ema_teacher_stopgrad_and_vicreg_gradients():
    model,_,x,valid=fixture()
    mask=uniform_mask(valid)
    out=model(x,valid,mask)
    terms=loss_terms(model,out,x,valid,mask)
    terms['loss'].backward()
    assert all(p.grad is None for p in model.teacher.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.online.parameters())
    old=next(model.teacher.parameters()).detach().clone()
    with torch.no_grad():
        next(model.online.parameters()).add_(1)
    model.update_teacher()
    torch.testing.assert_close(next(model.teacher.parameters()),old+(1-model.config.ema))
    z=torch.randn(8,16,requires_grad=True)*.1
    var,cov=variance_covariance(z)
    grad=torch.autograd.grad(var+cov,z)[0]
    assert grad.abs().sum()>0


def test_normal_guard_parent_identity_and_safe_checkpoint(tmp_path):
    model,data,x,valid=fixture()
    with pytest.raises(ValueError,match='normal training'):
        fit(model,x,valid,TrainConfig(steps=1),labels=['stroke']*8,source_ids=data['source_ids'])
    with pytest.raises(ValueError,match='Validation'):
        fit(model,x,valid,TrainConfig(steps=1),labels=data['labels'],source_ids=data['source_ids'],partitions=['test']*8)
    history=fit(model,x,valid,TrainConfig(steps=3,batch_size=4),labels=data['labels'],source_ids=data['source_ids'])
    assert all(np.isfinite(h['loss']) for h in history)
    path=save_checkpoint(model,tmp_path/'model.pt')
    restored=load_checkpoint(path)
    np.testing.assert_array_equal(encode(model,x,valid),encode(restored,x,valid))
    with pytest.raises(FileExistsError):
        save_checkpoint(model,path)
    original=deepcopy(model.state_dict())
    adapted=deepcopy(model)
    labels=np.array(['normal']*4+['stroke']*4)
    fit(adapted,x,valid,TrainConfig(steps=2,regime='adaptation',run_id='add-stroke-v1'),labels=labels,source_ids=data['source_ids'])
    assert adapted.run_metadata['parent_run_id']=='normal-uniform-v1'
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())


def test_conditional_latent_inference_prior_and_sample_diversity():
    model,data,x,valid=fixture(latent_dim=3)
    history=fit(model,x,valid,TrainConfig(steps=2,batch_size=4,latent_mse_weight=1),labels=data['labels'],source_ids=data['source_ids'])
    assert history[-1]['kl'] >= 0
    mask=forecast_mask(valid,2)
    with pytest.raises(ValueError,match='forbidden'):
        model(x,valid,mask,posterior=True)
    scores=anomaly_scores(model,x,valid,context_frames=2,samples=3)
    assert scores['samples'].shape==(3,8,4,16,2)
    assert (scores['predictive_variance']>0).all()
    changed=x.clone()
    changed[:,2:]+=10
    scores_changed=anomaly_scores(model,changed,valid,context_frames=2,samples=3)
    np.testing.assert_array_equal(scores['samples'],scores_changed['samples'])
    assert not np.array_equal(scores['coordinate_mse'],scores_changed['coordinate_mse'])


def test_stochastic_positions_shared_and_disabled_in_eval():
    model,_,x,valid=fixture()
    model=SkeletonJEPA(ModelConfig(frames=4,dim=16,heads=2,position_jitter=.5))
    a,at=model.positions(8,x.device,x.dtype)
    b,bt=model.positions(8,x.device,x.dtype)
    torch.testing.assert_close(a,at)
    assert not torch.equal(a,b)
    model.eval()
    a,_=model.positions(8,x.device,x.dtype)
    b,_=model.positions(8,x.device,x.dtype)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_separate_forecast_stage_and_bounded_scoring():
    model,data,x,valid=fixture()
    fit(model,x,valid,TrainConfig(steps=1),labels=data['labels'],source_ids=data['source_ids'])
    with pytest.raises(ValueError,match='separately'):
        fit(model,x,valid,TrainConfig(steps=1,forecast_fraction=1),labels=data['labels'],source_ids=data['source_ids'])
    fit(model,x,valid,TrainConfig(steps=2,regime='forecast',run_id='normal-future-v1',
                                 forecast_fraction=1,context_frames=2),
        labels=data['labels'],source_ids=data['source_ids'])
    full=anomaly_scores(model,x,valid,context_frames=2)
    chunked=anomaly_scores(model,x,valid,context_frames=2,batch_size=3)
    np.testing.assert_allclose(full['coordinates'],chunked['coordinates'],rtol=1e-5,atol=1e-5)
    assert chunked['samples'].shape==(1,8,4,16,2)
    assert all(row['mask']=='forecast' for row in model.run_metadata['training_history'])
