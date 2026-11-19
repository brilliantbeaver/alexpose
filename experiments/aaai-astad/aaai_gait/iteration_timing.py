"""Fixed-answer distal timing diagnostics on a separately registered parent grid.

The donor is an observed image-plane limb, not a physiological intervention.
Every edit uses the original body reference. Hidden source-time values remain
inaccessible, including the hip needed to attach a shifted distal observation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from .masking import rng_stream
from .model import JOINT_INDICES, uniform_mask
from .retained_prediction import construct_features

TIMING_VERSION="paired_distal_parent48_common_support_v1"
OFFSETS=(-4,4)
NOISE_SD=.01
DROPOUT_FRACTION=.25
DEFAULT_PARENT_BUDGET=8
READOUTS=tuple(range(25,33))


def _write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n")


def parent_mask_bank(seed=217):
    """One exact 16-draw uniform bank on the 12-by-16 source-time grid."""
    masks=uniform_mask(16,192,.5,generator=torch.Generator().manual_seed(seed)).numpy().reshape(16,12,16)
    masks.setflags(write=False)
    return masks


def _token_valid(x):
    return np.isfinite(x.reshape(len(x)//4,4,16,2)).all(axis=(1,3))


@dataclass
class TimingPair:
    aligned: np.ndarray
    edited: np.ndarray
    hidden: np.ndarray
    edited_hidden: np.ndarray
    requested: np.ndarray
    eligible: np.ndarray
    own_count: np.ndarray
    other_count: np.ndarray
    own_times: np.ndarray
    other_times: np.ndarray
    diagnostics: dict
    fixed_target: bool=True


def prepare_timing_pair(raw48,central32,pelvis32,scale,masks,offset,side,*,
                        variant="timing",noise_seed=42,noise_sd=NOISE_SD,
                        dropout_fraction=DROPOUT_FRACTION):
    """Build paired permitted inputs without wrapping or recomputing Q.

    ``side`` is the target leg (0 left, 1 right). The edit affects the opposite
    distal leg. A patch is usable only when all four observations are valid.
    Returned arrays retain inaccessible values, but their masks prohibit reads.
    Noise and dropout share timing's paired support and do not define its cohort.
    """
    raw=np.asarray(raw48,np.float32);central=np.asarray(central32,np.float32)
    pelvis=np.asarray(pelvis32,np.float32);masks=np.asarray(masks)
    if raw.shape!=(48,16,2) or central.shape!=(32,16,2) or pelvis.shape!=(32,2):
        raise ValueError("Expected the registered raw48, central32 and fixed pelvis32 arrays")
    if masks.shape!=(16,12,16) or masks.dtype!=bool or not np.all(masks.sum((1,2))==96):
        raise ValueError("Expected sixteen exact96-of192 Boolean parent masks")
    if side not in (0,1) or offset not in (-4,0,4):
        raise ValueError("Only declared sides and no-wrap offsets -4,0,+4 are supported")
    if not np.isfinite(scale) or scale<=0: raise ValueError("Original scale must be finite and positive")
    if variant not in ("timing","noise","dropout","whole_body"):
        raise ValueError("Unknown timing diagnostic")
    if not 0<=dropout_fraction<1 or noise_sd<0: raise ValueError("Invalid measurement-control size")
    valid=_token_valid(raw);cvalid=_token_valid(central)
    hidden=masks[:,2:10,:].copy();original_hidden=hidden.copy()
    n=len(masks);aligned=np.broadcast_to(central,(n,32,16,2)).copy();edited=aligned.copy()
    opposite=1-side;distal=np.arange(8+opposite,16,2);hip=6+opposite
    original_patch=np.arange(2,10);donor_patch=original_patch+offset//4
    original_frame=np.arange(8,40);donor_frame=original_frame+offset
    if donor_frame.min()<0 or donor_frame.max()>=48: raise ValueError("Donor outside parent")
    if variant=="whole_body":
        permitted=(~hidden)&(~masks[:,donor_patch,:])&valid[donor_patch][None]&cvalid[None]
        hidden|=~permitted
        normalized=(raw[donor_frame]-pelvis[:,None])/np.float32(scale)
        edited=np.broadcast_to(normalized,aligned.shape).copy()
    else:
        # Every vector endpoint is permitted before its derived displacement is read.
        original_ok=(~masks[:,original_patch,:])&valid[original_patch][None]
        donor_ok=(~masks[:,donor_patch,:])&valid[donor_patch][None]
        permitted=(original_ok[:,:,distal]&donor_ok[:,:,distal]&
                   original_ok[:,:,hip,None]&donor_ok[:,:,hip,None]&cvalid[None,:,distal])
        for k,joint in enumerate(distal):
            hidden[:,:,joint]|=~permitted[:,:,k]
            for patch in range(8):
                use=permitted[:,patch,k]
                if not use.any(): continue
                old=np.arange(8+4*patch,12+4*patch);donor=old+offset
                displacement=((raw[donor,joint]-raw[donor,hip])-
                              (raw[old,joint]-raw[old,hip]))/np.float32(scale)
                # Adding a zero vector preserves exact identity at offset zero.
                edited[use,4*patch:4*patch+4,joint]+=displacement
    edited_hidden=hidden.copy();rng=rng_stream(noise_seed,f"timing_{variant}_{side}_{offset}")
    if variant=="noise":
        edited=aligned.copy()
        for m in range(n):
            for patch,jj in np.argwhere(~hidden[m,:,distal].T):
                joint=distal[jj]
                edited[m,4*patch:4*patch+4,joint]+=rng.normal(0,noise_sd,size=(4,2)).astype(np.float32)
    elif variant=="dropout":
        edited=aligned.copy()
        for m in range(n):
            cells=np.argwhere(~hidden[m,:,distal].T)
            count=int(np.ceil(len(cells)*dropout_fraction))
            if count:
                for patch,jj in cells[rng.choice(len(cells),count,replace=False)]: edited_hidden[m,patch,distal[jj]]=True
    visible=(~hidden)&cvalid[None]
    edited_valid=np.isfinite(edited.reshape(n,8,4,16,2)).all(axis=(2,4))
    visible_edited=(~edited_hidden)&edited_valid
    own=visible[:,:,6+side::2];other=visible[:,:,6+opposite::2]
    own_count=own.sum((1,2));other_count=other.sum((1,2))
    own_times=own.any(2).sum(1);other_times=other.any(2).sum(1)
    support=(own_count>=2)&(other_count>=2)&(own_times>=2)&(other_times>=2)
    edited_own=visible_edited[:,:,6+side::2];edited_other=visible_edited[:,:,6+opposite::2]
    edited_support=(edited_own.sum((1,2))>=2)&(edited_other.sum((1,2))>=2)&(edited_own.any(2).sum(1)>=2)&(edited_other.any(2).sum(1)>=2)
    target_slots=np.arange(8+side,16,2)
    requested=np.zeros((n,8,16),bool);requested[:,:,target_slots]=original_hidden[:,:,target_slots]
    eligible=requested&cvalid[None]&edited_valid&support[:,None,None]&edited_support[:,None,None]
    differences=edited-aligned
    allowed_frames=np.repeat((~hidden)&(~edited_hidden)&cvalid[None]&edited_valid,4,axis=1)
    affected=np.zeros((16,),bool);affected[:]=variant=="whole_body";affected[distal]=True
    allowed_frames&=affected[None,None,:]
    displacement=np.where(allowed_frames[...,None],differences,0.)
    denominator=np.maximum(allowed_frames.sum((1,2)),1)
    displacement_rms=np.sqrt(np.sum(displacement**2,axis=(1,2,3))/denominator)
    consecutive=allowed_frames[:,1:]&allowed_frames[:,:-1]
    velocities=np.where(consecutive[...,None],np.diff(differences,axis=1)*15,0.)
    velocity_rms=np.sqrt(np.sum(velocities**2,axis=(1,2,3))/np.maximum(consecutive.sum((1,2)),1))
    mean_shift=displacement.sum((1,2))/denominator[:,None]
    joint_denominator=np.maximum(allowed_frames.sum(1),1)[...,None]
    def excursion(values):
        observed=np.where(allowed_frames[...,None],values,0.)
        means=observed.sum(1)/joint_denominator
        squared=np.where(allowed_frames[...,None],(values-means[:,None])**2,0.)
        return np.sqrt(squared.sum((1,2,3))/denominator)
    diagnostics=dict(displacement_rms_torso=displacement_rms,velocity_change_rms_torso_per_s=velocity_rms,
        mean_context_shift_x_torso=mean_shift[:,0],mean_context_shift_y_torso=mean_shift[:,1],
        within_joint_excursion_change_torso=excursion(edited)-excursion(aligned),
        other_count_after=edited_other.sum((1,2)),removed_distal_tokens=(edited_hidden&~hidden).sum((1,2)),
        opposite_distal_permitted=((~hidden[:,:,distal])&cvalid[None,:,distal]).sum((1,2)),
        required_hip_unavailable=(~((~masks[:,original_patch,hip])&valid[original_patch,hip][None]&
                                   (~masks[:,donor_patch,hip])&valid[donor_patch,hip][None])).sum(1),
        same_body_reference=np.ones(n,bool),noise_sd_pixels=np.full(n,noise_sd*scale if variant=="noise" else 0.))
    return TimingPair(aligned,edited,hidden.reshape(n,128),edited_hidden.reshape(n,128),requested,eligible,
                      own_count,other_count,own_times,other_times,diagnostics,variant!="whole_body")


def select_parent_budget(metadata,max_parents_per_source=DEFAULT_PARENT_BUDGET):
    """Choose equally spaced sorted parent IDs before any model score is read."""
    if max_parents_per_source<1: raise ValueError("Parent budget must be positive")
    selected=[]
    for _,group in metadata.groupby("source_id",sort=True):
        group=group.sort_values(["sequence_id","parent_start_seconds","timing_window_index"])
        positions=np.unique(np.rint(np.linspace(0,len(group)-1,min(max_parents_per_source,len(group)))).astype(int))
        selected.extend(group.iloc[positions].timing_window_index.astype(int).tolist())
    return set(selected)


def _validate_data(data):
    n=len(data.parent_metadata)
    if data.parent_raw is None or data.parent_raw.shape!=(n,48,16,2):
        raise ValueError("Missing or misaligned parent48 data")
    if data.parent_x.shape!=(n,32,16,2) or data.parent_pelvis.shape!=(n,32,2) or data.parent_times.shape!=(n,48):
        raise ValueError("Parent reference arrays are misaligned")
    if not np.allclose(np.diff(data.parent_times,axis=1),1/15,rtol=0,atol=1e-8):
        raise ValueError("A parent has a recording-time gap")
    if not np.array_equal(data.parent_metadata.timing_window_index.to_numpy(int),np.arange(n)):
        raise ValueError("Parent indices must align exactly with saved arrays")


def _pair(data,index,masks,offset,side,**kwargs):
    row=data.parent_metadata.iloc[index]
    return prepare_timing_pair(data.parent_raw[index],data.parent_x[index],data.parent_pelvis[index],
                               float(row.normalization_scale),masks,offset,side,**kwargs)


def _coverage_table(targets,registry,offsets):
    records=[]
    counts=targets.groupby(["source_id","offset","joint"]).size() if len(targets) else pd.Series(dtype=int)
    for source in registry.source_id.astype(str):
        for offset in offsets:
            values=[int(counts.get((source,offset,j),0)) for j in READOUTS]
            records.append(dict(source_id=source,offset=offset,eligible=all(v>=4 for v in values),
                                minimum_readout_patches=min(values),distinct_target_patches=sum(values),
                                **{f"patches_{j}":v for j,v in zip(READOUTS,values)}))
    return pd.DataFrame(records).merge(registry,on="source_id",validate="many_to_one")


def build_timing_census(data,folder,*,offsets=OFFSETS,seed=217,max_parents_per_source=DEFAULT_PARENT_BUDGET):
    """Inspect every available parent without evaluating any predictor."""
    _validate_data(data);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    masks=parent_mask_bank(seed);selected=select_parent_budget(data.parent_metadata,max_parents_per_source)
    np.save(folder/"parent_uniform_masks.npy",masks)
    records=[];targets=[]
    for index,row in enumerate(data.parent_metadata.itertuples(index=False)):
        for offset in offsets:
            for side in (0,1):
                pair=_pair(data,index,masks,offset,side)
                records.append(dict(source_id=row.source_id,sequence_id=row.sequence_id,parent_index=index,
                    offset=offset,target_side=side,budget_selected=index in selected,
                    requested=int(pair.requested.sum()),eligible_mask_targets=int(pair.eligible.sum()),
                    masks_with_own_support=int(((pair.own_count>=2)&(pair.own_times>=2)).sum()),
                    masks_with_other_support=int(((pair.other_count>=2)&(pair.other_times>=2)).sum()),
                    missing_required_hip_patch_comparisons=int(pair.diagnostics["required_hip_unavailable"].sum()),
                    mask_targets_with_distal_context=int((pair.eligible* (pair.diagnostics["opposite_distal_permitted"]>0)[:,None,None]).sum()),
                    mean_permitted_distal_tokens=float(pair.diagnostics["opposite_distal_permitted"].mean()),
                    mean_own_tokens=float(pair.own_count.mean()),mean_other_tokens=float(pair.other_count.mean())))
                for patch,slot in np.argwhere(pair.eligible.any(0)):
                    targets.append(dict(source_id=row.source_id,sequence_id=row.sequence_id,offset=offset,
                        joint=JOINT_INDICES[slot],patch_start_ns=int(np.rint((row.start_seconds+patch*4/15)*1e9)),
                        budget_selected=index in selected))
    ledger=pd.DataFrame(records)
    targets=pd.DataFrame(targets,columns=["source_id","sequence_id","offset","joint","patch_start_ns","budget_selected"])
    keys=["source_id","sequence_id","offset","joint","patch_start_ns"]
    full=targets.drop_duplicates(keys);budget=targets.loc[targets.budget_selected].drop_duplicates(keys)
    full_coverage=_coverage_table(full,data.source_registry,offsets)
    budget_coverage=_coverage_table(budget,data.source_registry,offsets)
    ledger.to_csv(folder/"parent_eligibility.csv.gz",index=False)
    full_coverage.to_csv(folder/"source_eligibility_full.csv",index=False)
    budget_coverage.to_csv(folder/"source_eligibility_budget.csv",index=False)
    provenance=[]
    for index,row in enumerate(data.parent_metadata.itertuples(index=False)):
        positions=data.parent_times[index]*float(row.source_fps)+1
        nearest=np.rint(positions);positions=np.where(np.isclose(positions,nearest,atol=1e-6,rtol=0),nearest,positions)
        provenance.append(dict(parent_index=index,source_id=row.source_id,sequence_id=row.sequence_id,
            parent_window_id=row.parent_window_id,normalization_scale=float(row.normalization_scale),
            decoded_seconds=json.dumps(data.parent_times[index].tolist()),
            fractional_source_frame_positions=json.dumps(positions.tolist()),
            source_frame_lower=json.dumps(np.floor(positions).astype(int).tolist()),
            source_frame_upper=json.dumps(np.ceil(positions).astype(int).tolist()),
            budget_selected=index in selected))
    pd.DataFrame(provenance).to_csv(folder/"parent_frame_provenance.csv.gz",index=False)
    summary={}
    for name,table in (("full",full_coverage),("budget",budget_coverage)):
        summary[name]={}
        for offset,group in table.groupby("offset"):
            normal=group[group.binary_label.eq(0)&group.eligible]
            fold_counts={str(f):int(normal[f"role_fold_{f}"].eq("test").sum()) for f in range(4)}
            summary[name][str(offset)]=dict(eligible_sources=int(group.eligible.sum()),eligible_normal_sources=len(normal),
                normal_source_ids=normal.source_id.tolist(),outer_test_normal_counts=fold_counts,
                planning_floor_passed=len(normal)>=20 and min(fold_counts.values())>=4)
    audit=dict(version=TIMING_VERSION,model_outcomes_used=False,parent_seed=seed,parent_mask_sha256=hashlib.sha256(masks.tobytes()).hexdigest(),
        parent_count=len(data.parent_metadata),parent_sources=int(data.parent_metadata.source_id.nunique()),
        offsets=list(offsets),sample_hz=15,central=[8,40],max_parents_per_source=max_parents_per_source,
        budget_rule="Evenly spaced parents after sorting sequence ID, start time and parent index; fixed before scores",
        source_gate="At least four unique original temporal targets per each of eight readout joints, separately by offset",
        support_rule="Valid target and at least two permitted tokens at distinct times on each leg; no preferential redraw",
        donor_rule="Original distal, donor distal, donor hip and original attachment hip all valid and permitted",
        controls=dict(noise_sd_original_torso=NOISE_SD,dropout_fraction=DROPOUT_FRACTION),
        interpretation="Finite crops change empirical motion marginals and body configuration; no pure timing or neural-causality claim",
        source_frame_note="Fractional positions describe the 15 Hz interpolation; lower/upper are one-based original frame IDs",
        coverage=summary)
    _write_json(folder/"census.json",audit)
    return full_coverage,budget_coverage,audit


def _score_pair(pair,bundles,parent_row,offset,variant):
    mask_id,patch,slot=np.nonzero(pair.eligible)
    if not len(mask_id): return pd.DataFrame()
    joints=np.asarray(JOINT_INDICES)[slot]
    aligned=construct_features(pair.aligned[mask_id],pair.hidden[mask_id],joints,patch)
    edited=construct_features(pair.edited[mask_id],pair.edited_hidden[mask_id],joints,patch)
    if pair.fixed_target:
        np.testing.assert_array_equal(aligned.target,edited.target)
        np.testing.assert_array_equal(aligned.own,edited.own)
        np.testing.assert_array_equal(aligned.own_extra,edited.own_extra)
    common=dict(parent_index=parent_row.timing_window_index,sequence_id=parent_row.sequence_id,
        source_id=parent_row.source_id,mask_id=mask_id,patch_index=patch,joint=joints,offset=offset,
        variant=variant,fixed_target=pair.fixed_target,normalization_scale=float(parent_row.normalization_scale),
        patch_start_ns=np.rint((parent_row.start_seconds+patch*4/15)*1e9).astype(np.int64),
        own_context_tokens=pair.own_count[mask_id],other_context_tokens=pair.other_count[mask_id],
        own_context_times=pair.own_times[mask_id],other_context_times=pair.other_times[mask_id],
        **{k:np.asarray(v)[mask_id] for k,v in pair.diagnostics.items()})
    result=[]
    for policy,bundle in bundles.items():
        first=bundle.predict_features(aligned);second=bundle.predict_features(edited)
        if pair.fixed_target:
            np.testing.assert_array_equal(first["base"],second["base"])
            np.testing.assert_array_equal(first["own"],second["own"])
        record=pd.DataFrame(common);record["policy"]=policy
        record["both_multiplier"]=bundle.both_multiplier;record["own_multiplier"]=bundle.own_multiplier
        record["has_permitted_distal_context"]=record.opposite_distal_permitted.gt(0)
        record["has_nonzero_context_edit"]=record.displacement_rms_torso.gt(1e-12)|record.removed_distal_tokens.gt(0)
        for branch in ("base","own","both"):
            record[f"loss_{branch}_aligned"]=np.square(first[branch]-aligned.target).mean(1)
            record[f"loss_{branch}_edited"]=np.square(second[branch]-edited.target).mean(1)
        record["effect"]=record.loss_both_edited-record.loss_both_aligned
        record["adequacy_gain_base"]=record.loss_base_aligned-record.loss_both_aligned
        record["adequacy_gain_own"]=record.loss_own_aligned-record.loss_both_aligned
        record["own_prediction_max_change"]=np.max(np.abs(first["own"]-second["own"]),axis=1)
        record["target_max_change"]=np.max(np.abs(aligned.target-edited.target),axis=1)
        if variant=="zero":
            np.testing.assert_array_equal(first["both"],second["both"])
            assert np.equal(record.effect,0).all()
        result.append(record)
    return pd.concat(result,ignore_index=True)


def summarize_source_pairs(rows):
    """Mask -> target -> parent -> sequence -> source, with equal joint weight."""
    if rows.empty:return pd.DataFrame(),pd.DataFrame()
    identities=["source_id","sequence_id","parent_index","joint","patch_index","patch_start_ns","policy","variant","offset"]
    numeric=[c for c in rows.select_dtypes(include=[np.number,bool]).columns if c not in identities+["mask_id"]]
    targets=rows.groupby(identities,sort=True)[numeric].mean().reset_index()
    groups=["source_id","policy","variant","offset"]
    unique=targets.drop_duplicates(groups+["sequence_id","joint","patch_start_ns"])
    counts=unique.groupby(groups+["joint"]).size().rename("distinct_patches")
    parents=targets.groupby(groups+["sequence_id","parent_index","joint"])[numeric].mean()
    sequences=parents.groupby(groups+["sequence_id","joint"]).mean()
    joints=sequences.groupby(groups+["joint"]).mean().join(counts).reset_index()
    source_rows=[]
    for key,group in joints.groupby(groups,sort=True):
        present=set(group.joint);eligible=present==set(READOUTS) and group.distinct_patches.ge(4).all()
        values=group[numeric].mean().to_dict()
        values.update(dict(zip(groups,key)),eligible=bool(eligible),available_joints=len(present),
            minimum_readout_patches=int(group.distinct_patches.min()) if len(present)==8 else 0,
            effect_available_joints=values["effect"])
        if not eligible: values["effect"]=np.nan
        values["timing_evidence_status"]=("no_permitted_distal_context" if values.get("has_permitted_distal_context",0)==0
            else "frozen_zero_correction" if values.get("both_multiplier",0)==0
            else "descriptive_requires_paired_adequacy")
        source_rows.append(values)
    return pd.DataFrame(source_rows),joints


def run_timing_fold(data,fold,bundles,folder,*,source_ids=None,scope="outer_test_normal",
                    max_parents_per_source=DEFAULT_PARENT_BUDGET,offsets=OFFSETS,seed=217):
    """Score fixed normal-only development or outer-test diagnostic models.

    Pass selected inner-fold bundles and their disjoint validation IDs with
    ``scope='inner_validation'``. Explicit ``outer_test`` permits abnormal test
    sources for a later descriptive extension. Reference sources cannot be used.
    The complete score-free census is a separate ``build_timing_census`` call.
    """
    _validate_data(data);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    registry=data.source_registry.set_index("source_id",drop=False);role=f"role_fold_{fold}"
    if scope not in ("outer_test_normal","outer_test","inner_validation"): raise ValueError("Undeclared timing role")
    if source_ids is None:
        if scope=="inner_validation": raise ValueError("Explicit inner-validation identities are required")
        source_ids=registry.loc[registry[role].eq("test")&(registry.binary_label.eq(0) if scope=="outer_test_normal" else True),"source_id"].tolist()
    source_ids=sorted(set(source_ids))
    if not set(source_ids).issubset(registry.index): raise ValueError("Unknown requested source")
    selected_registry=registry.loc[source_ids]
    expected_role="train" if scope=="inner_validation" else "test"
    if not selected_registry[role].eq(expected_role).all(): raise ValueError("Timing source belongs to a forbidden role")
    if scope!="outer_test" and not selected_registry.binary_label.eq(0).all(): raise ValueError("Normal-only diagnostic received abnormal labels")
    for policy,bundle in bundles.items():
        if set(source_ids)&set(bundle.base.metadata["normal_sources"]): raise ValueError("Timing validation overlaps fitting sources")
        if bundle.policy!=policy: raise ValueError("Bundle policy key mismatch")
    masks=parent_mask_bank(seed);np.save(folder/"parent_uniform_masks.npy",masks)
    metadata=data.parent_metadata[data.parent_metadata.source_id.isin(source_ids)]
    selected=select_parent_budget(metadata,max_parents_per_source)
    protocol=dict(version=TIMING_VERSION,fold=fold,scope=scope,source_ids=source_ids,
        parent_mask_seed=seed,parent_mask_sha256=hashlib.sha256(masks.tobytes()).hexdigest(),
        max_parents_per_source=max_parents_per_source,selected_parent_indices=sorted(selected),
        offsets=list(offsets),noise_sd_original_torso=NOISE_SD,dropout_fraction=DROPOUT_FRACTION,
        fixed_before_scores=True,target_aggregation="Masks, targets, parents, sequences, then sources; equal joint weights",
        pure_timing_or_neurological_causality_claim=False,
        coordinate_access="No derived donor value without visible valid original distal, donor distal, donor hip and attachment hip",
        reference="Original per-frame pelvis and original first-half torso scale are unchanged",
        whole_body_note="Whole-body crop changes the answer and own context; it is never a fixed-target timing effect",
        dropout_note="Reduced-evidence control deliberately changes support; its eligibility is separate from timing",
        noise_note="Independent Gaussian image-plane perturbation; not displacement-matched or anatomy-preserving")
    _write_json(folder/"protocol.json",protocol)
    rowfolder=folder/"target_rows";rowfolder.mkdir(exist_ok=True)
    source_tables=[];joint_tables=[];ledger=[];zero_rows=0;fixed_target_rows=0
    for source in source_ids:
        parts=[]
        indexes=[int(i) for i in metadata.loc[metadata.source_id.eq(source),"timing_window_index"] if int(i) in selected]
        for index in indexes:
            row=data.parent_metadata.iloc[index]
            tasks=[(offset,side,variant) for offset in offsets for side in (0,1)
                   for variant in ("timing","noise","dropout","whole_body")]
            tasks.extend((0,side,"zero") for side in (0,1))
            for offset,side,variant in tasks:
                pair=_pair(data,index,masks,offset,side,variant="timing" if variant=="zero" else variant,
                           noise_seed=42+index)
                ledger.append(dict(source_id=source,parent_index=index,offset=offset,target_side=side,variant=variant,
                    requested=int(pair.requested.sum()),eligible=int(pair.eligible.sum()),
                    mean_own_tokens=float(pair.own_count.mean()),mean_other_tokens=float(pair.other_count.mean())))
                scored=_score_pair(pair,bundles,row,offset,variant)
                if len(scored):parts.append(scored)
        if parts:
            table=pd.concat(parts,ignore_index=True)
            zero_rows+=int(table.variant.eq("zero").sum())
            fixed_target_rows+=int(table.fixed_target.sum())
            table.to_csv(rowfolder/f"{source}.csv.gz",index=False)
            sources,joints=summarize_source_pairs(table);source_tables.append(sources);joint_tables.append(joints)
    sources=pd.concat(source_tables,ignore_index=True) if source_tables else pd.DataFrame()
    joints=pd.concat(joint_tables,ignore_index=True) if joint_tables else pd.DataFrame()
    if len(sources):
        sources["outer_fold"]=fold;sources["scope"]=scope
        sources=sources.merge(selected_registry[["source_id","binary_label"]].reset_index(drop=True),on="source_id",validate="many_to_one")
    sources.to_csv(folder/"source_pairs.csv",index=False);joints.to_csv(folder/"joint_pairs.csv",index=False)
    pd.DataFrame(ledger).to_csv(folder/"eligibility_ledger.csv.gz",index=False)
    eligible_summary=[]
    if len(sources):
        for key,group in sources.groupby(["policy","variant","offset"]):
            eligible_summary.append(dict(policy=key[0],variant=key[1],offset=int(key[2]),
                eligible_sources=int(group.eligible.sum()),available_source_rows=len(group),requested_sources=len(source_ids)))
    audit=dict(protocol,status="complete",model_calls_real=bool(source_tables),coverage=eligible_summary,
        sources_without_parents=sorted(set(source_ids)-set(metadata.source_id)),
        sources_without_eligible_predictions=sorted(set(source_ids)-(set(sources.source_id) if len(sources) else set())),
        zero_identity_assertion_rows=zero_rows,fixed_target_and_own_invariance_assertion_rows=fixed_target_rows,
        exact_zero_and_own_invariance_assertions_passed=True if zero_rows and fixed_target_rows else None,
        unchanged_normalizer_assertions="Original arrays and scale are never recomputed or mutated")
    _write_json(folder/"execution.json",audit)
    return sources,audit
