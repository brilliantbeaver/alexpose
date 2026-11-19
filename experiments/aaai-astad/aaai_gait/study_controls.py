"""Paired observation and timing probes, never synthetic clinical diagnoses.

Inputs contain a longer, already normalized sequence. Each edit uses common
interior support, with no wraparound or repeated boundary frames. The original
normalization is retained and never recomputed from an edited limb.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import JOINTS

LEFT_LEG = (23, 25, 27, 29, 31)
RIGHT_LEG = (24, 26, 28, 30, 32)
BONES = ((11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
         (23,25),(25,27),(27,29),(29,31),(24,26),(26,28),(28,30),(30,32))


def paired_edits(long_x, *, frames=32, shift=4, limb='left', noise_sd=.01,
                 dropout_fraction=.05, seed=42):
    """Return matched crops plus an explicit ledger of geometric changes.

    A limb shift preserves its internally articulated samples but can still
    distort its relationship to the torso. The base-referenced length change
    also includes ordinary gait-phase variation. Donor-referenced internal
    limb lengths isolate whether the edit preserved its articulated samples.
    No threshold here certifies physiological validity.
    """
    x = np.asarray(long_x, np.float32)
    if (x.ndim != 4 or x.shape[2:] != (16,2) or len(x) == 0
            or isinstance(frames, (bool, np.bool_)) or not isinstance(frames, (int, np.integer))
            or isinstance(shift, (bool, np.bool_)) or not isinstance(shift, (int, np.integer))
            or shift < 1 or frames < 2):
        raise ValueError('Expected [windows,time,16,2], positive shift and frames >= 2')
    if np.isinf(x).any():
        raise ValueError('Coordinates may be finite or NaN, never infinite')
    if x.shape[1] < frames + 2 * shift:
        raise ValueError('Timing probes need a longer sequence with common interior support')
    if (limb not in ('left','right') or not np.isfinite(noise_sd) or noise_sd < 0
            or not np.isfinite(dropout_fraction) or not 0 <= dropout_fraction < 1):
        raise ValueError('Invalid limb, noise, or dropout setting')
    base = x[:,shift:shift+frames].copy()
    shifted = x[:,2*shift:2*shift+frames].copy()
    relative = base.copy()
    indices = [JOINTS.index(j) for j in (LEFT_LEG if limb == 'left' else RIGHT_LEG)]
    relative[:,:,indices] = shifted[:,:,indices]
    rng = np.random.default_rng(seed)
    noise = base + rng.normal(0,noise_sd,base.shape).astype(np.float32)
    dropout = base.copy()
    # Observation deletion is a diagnostic, distinct from the learning mask.
    deleted = rng.random(base.shape[:-1]) < dropout_fraction
    dropout[deleted] = np.nan
    edits = {'original':base,'relative_limb_shift':relative,'whole_body_shift':shifted,
             'coordinate_noise':noise,'observation_dropout':dropout}
    records=[]
    for name, values in edits.items():
        for i in range(len(values)):
            record={'window_index':i,'intervention':name,
                    'finite_joint_fraction':float(np.isfinite(values[i]).all(-1).mean())}
            if frames % 4 == 0:
                record['valid_token_fraction'] = float(
                    np.isfinite(values[i]).reshape(frames//4,4,16,2).all(axis=(1,3)).mean())
            for prefix,pairs in [('limb',[(a,b) for a,b in BONES if a in (LEFT_LEG if limb=='left' else RIGHT_LEG) and b in (LEFT_LEG if limb=='left' else RIGHT_LEG)]),
                                 ('torso',[(11,23),(12,24)]), ('pelvis',[(23,24)])]:
                changes=[]
                for a,b in pairs:
                    ia,ib=JOINTS.index(a),JOINTS.index(b)
                    l0=np.linalg.norm(base[i,:,ia]-base[i,:,ib],axis=-1)
                    l1=np.linalg.norm(values[i,:,ia]-values[i,:,ib],axis=-1)
                    valid=np.isfinite(l0)&np.isfinite(l1)&(l0>1e-6)
                    changes.extend((np.abs(l1[valid]-l0[valid])/l0[valid]).tolist())
                record[prefix+'_relative_length_change_median']=float(np.median(changes)) if changes else np.nan
            donor = shifted if name in ('relative_limb_shift', 'whole_body_shift') else base
            donor_changes=[]
            for a,b in BONES:
                if JOINTS.index(a) not in indices or JOINTS.index(b) not in indices:
                    continue
                ia,ib=JOINTS.index(a),JOINTS.index(b)
                l0=np.linalg.norm(donor[i,:,ia]-donor[i,:,ib],axis=-1)
                l1=np.linalg.norm(values[i,:,ia]-values[i,:,ib],axis=-1)
                valid=np.isfinite(l0)&np.isfinite(l1)&(l0>1e-6)
                donor_changes.extend((np.abs(l1[valid]-l0[valid])/l0[valid]).tolist())
            record['limb_donor_relative_length_change_median'] = (
                float(np.median(donor_changes)) if donor_changes else np.nan)
            steps=np.linalg.norm(np.diff(values[i],axis=0),axis=-1)
            finite=steps[np.isfinite(steps)]
            record['largest_projected_step']=float(finite.max()) if len(finite) else np.nan
            records.append(record)
    return edits,pd.DataFrame(records)


def common_intervention_rows(tables):
    """Keep exactly the same eligible target/mask rows in every paired version.

    Explicitly valid rows must have three finite losses. Numerical failures
    cannot be hidden by a support intersection. Tables without a valid flag
    use all three finite losses to establish eligibility. Identity-only tables
    are also accepted, for inspecting observation support before prediction.
    """
    keys=['window_index','joint','patch_index','mask_id','target_type']
    losses=['loss_own','loss_patch','loss_full']
    common=None; filtered={}
    for name,table in tables.items():
        if not set(keys).issubset(table) or table[keys].isna().any().any():
            raise ValueError('Intervention tables need nonmissing target/mask keys')
        if table.duplicated(keys).any():
            raise ValueError('Duplicate target/mask rows in an intervention table')
        has_losses=bool(set(losses)&set(table.columns))
        if has_losses and not set(losses).issubset(table):
            raise ValueError('Intervention scores need all three predictor losses')
        finite=np.isfinite(table[losses].to_numpy(float)).all(axis=1) if has_losses else np.ones(len(table),bool)
        if 'valid' in table:
            if table.valid.isna().any() or not table.valid.isin([True, False]).all():
                raise ValueError('The valid flag must contain booleans or zero/one')
            eligible=table.valid.to_numpy(bool)
            if np.any(eligible & ~finite):
                raise FloatingPointError('Nonfinite loss on an eligible intervention target')
        else:
            eligible=finite
        t=table.loc[eligible].copy()
        index=pd.MultiIndex.from_frame(t[keys])
        common=index if common is None else common.intersection(index)
        filtered[name]=t
    for name,t in filtered.items():
        result=t.loc[pd.MultiIndex.from_frame(t[keys]).isin(common)].sort_values(keys).reset_index(drop=True)
        result.attrs['paired_support'] = dict(eligible_before_intersection=len(t),common_rows=len(result))
        filtered[name]=result
    return filtered


def movement_annotation_template(manifest, *, per_label=4, seed=42):
    """Create a reproducible source sample and a label-blinded annotation table.

    Return the restricted lookup separately. Do not expose it to annotators.
    Labels stratify sampling only; they never supply movement-phenotype truth.
    Opaque filenames blind the table, not any titles or condition clues in the
    videos themselves. Preview preparation needs a separate blinding review.
    """
    if isinstance(per_label,(bool,np.bool_)) or not isinstance(per_label,(int,np.integer)) or per_label < 1:
        raise ValueError('per_label must be a positive integer')
    fields=['label','source_id','sequence_id']
    if not set(fields).issubset(manifest) or manifest.empty or manifest[fields].isna().any().any():
        raise ValueError('Annotation sampling needs nonempty label/source/sequence metadata')
    rng=np.random.default_rng(seed); chosen=[]; used=set()
    for label, part in manifest.groupby('label',sort=True):
        rows=part.sort_values(['source_id','sequence_id']).drop_duplicates('source_id')
        for idx in rng.permutation(len(rows)):
            row=rows.iloc[idx]
            if row.source_id in used: continue
            chosen.append(row);used.add(row.source_id)
            if sum(r.label==label for r in chosen)>=per_label:break
    lookup=pd.DataFrame(chosen).sample(frac=1,random_state=seed).reset_index(drop=True)
    lookup.insert(0,'annotation_id',[f'clip-{i+1:03d}' for i in range(len(lookup))])
    blind=pd.DataFrame({'annotation_id':lookup.annotation_id,
        'video_reference':['previews/'+v+'.mp4' for v in lookup.annotation_id]})
    for field in ('reviewer_id','walking_visible','view','person_stable','left_right_confidence',
                  'motion_asymmetry','arm_excursion_reduction','sustained_knee_flexion',
                  'trunk_motion','walking_aid','uncertain_or_unscorable','notes'):
        blind[field]=''
    return blind,lookup
