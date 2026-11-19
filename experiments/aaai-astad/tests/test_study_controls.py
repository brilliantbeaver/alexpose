import numpy as np
import pandas as pd
import pytest
from aaai_gait.study_controls import paired_edits,common_intervention_rows,movement_annotation_template
from aaai_gait.data import JOINTS


def test_timing_probe_uses_interior_without_wraparound():
    x=np.arange(40,dtype=np.float32)[None,:,None,None]*np.ones((1,40,16,2),np.float32)
    variants,ledger=paired_edits(x,frames=32,shift=4)
    np.testing.assert_array_equal(variants['original'][0,:,0,0],np.arange(4,36))
    np.testing.assert_array_equal(variants['whole_body_shift'][0,:,0,0],np.arange(8,40))
    for joint in (23,25,27,29,31):
        np.testing.assert_array_equal(variants['relative_limb_shift'][0,:,JOINTS.index(joint),0],np.arange(8,40))
    np.testing.assert_array_equal(variants['relative_limb_shift'][0,:,JOINTS.index(28),0],np.arange(4,36))
    with pytest.raises(ValueError,match='longer'):
        paired_edits(x[:,:32])


def test_interventions_share_only_common_valid_targets():
    a=pd.DataFrame({'window_index':[0,0], 'joint':[25,27], 'patch_index':[0,0],
                    'mask_id':[0,0],'target_type':['coordinate','coordinate'],'valid':[True,True]})
    b=a.copy();b.loc[1,'valid']=False
    out=common_intervention_rows({'original':a,'edited':b})
    assert all(t.joint.tolist()==[25] for t in out.values())


def test_intervention_numerical_failures_cannot_become_qc_exclusions():
    a=pd.DataFrame({'window_index':[0,0], 'joint':[25,27], 'patch_index':[0,0],
                    'mask_id':[0,0],'target_type':['coordinate','coordinate'],'valid':[True,True],
                    'loss_own':[1.,2.],'loss_patch':[.8,1.8],'loss_full':[.5,1.]})
    b=a.copy();b.loc[1,'loss_patch']=np.nan
    with pytest.raises(FloatingPointError,match='Nonfinite'):
        common_intervention_rows({'original':a,'edited':b})
    b.loc[1,'valid']=False
    out=common_intervention_rows({'original':a,'edited':b})
    assert all(t.joint.tolist()==[25] for t in out.values())
    # If no explicit eligibility is supplied, all losses establish common support.
    out=common_intervention_rows({'original':a.drop(columns='valid'),
                                 'edited':b.drop(columns='valid')})
    assert all(t.joint.tolist()==[25] for t in out.values())
    for value in ('False',np.nan):
        invalid=a.copy();invalid['valid']=invalid.valid.astype(object);invalid.loc[0,'valid']=value
        with pytest.raises(ValueError,match='flag'):
            common_intervention_rows({'invalid':invalid})
    with pytest.raises(ValueError,match='Duplicate'):
        common_intervention_rows({'duplicate':pd.concat([a,a.iloc[[0]]])})


def test_edit_geometry_and_token_support_are_reported_separately():
    rng=np.random.default_rng(42)
    x=rng.normal(size=(2,40,16,2)).astype(np.float32)
    # A single missing joint-frame invalidates one entire four-frame token.
    x[0,5,JOINTS.index(25)]=np.nan
    variants,ledger=paired_edits(x,dropout_fraction=0,noise_sd=0)
    row=ledger.query("window_index == 0 and intervention == 'original'").iloc[0]
    assert row.finite_joint_fraction==511/512
    assert row.valid_token_fraction==127/128
    shifted=ledger[ledger.intervention.eq('relative_limb_shift')]
    assert (shifted.limb_donor_relative_length_change_median==0).all()
    assert (shifted.pelvis_relative_length_change_median>0).all()
    assert (shifted.limb_relative_length_change_median>0).all()
    # No input normalization or mutation occurs after an intervention.
    np.testing.assert_equal(variants['original'],x[:,4:36])


@pytest.mark.parametrize('kwargs',[{'noise_sd':np.nan},{'noise_sd':np.inf},
                                    {'dropout_fraction':np.nan},{'shift':4.5},{'frames':32.5}])
def test_invalid_edit_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        paired_edits(np.zeros((1,40,16,2)),**kwargs)


def test_annotation_table_is_source_unique_and_has_no_condition_lookup():
    metadata=pd.DataFrame({'source_id':['a','a','b','c'], 'sequence_id':['a1','a2','b1','c1'],
                           'label':['normal','normal','stroke','stroke']})
    blind,lookup=movement_annotation_template(metadata,per_label=1)
    assert len(blind)==len(lookup)==2
    assert lookup.source_id.is_unique
    assert set(lookup.label)=={'normal','stroke'}
    assert not {'label','source_id','sequence_id'} & set(blind)
    assert blind.video_reference.str.match(r'previews/clip-\d{3}\.mp4').all()
    with pytest.raises(ValueError,match='positive integer'):
        movement_annotation_template(metadata,per_label=0)
