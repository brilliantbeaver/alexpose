"""Matched target aggregation and frozen normal-reference predictive profiles.

Only twelve declared readout joints enter the primary three-region score. All
predictors retain the original sixteen training/masking joints. No score sign
is selected from abnormal outcomes. Unknown groups are never imputed.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
import numpy as np
import pandas as pd
from .study_data import STUDY_LABELS,source_table

GROUPS={'left_leg':(25,27,29,31),'right_leg':(26,28,30,32),'arms':(13,14,15,16)}
READOUT_JOINTS=tuple(sorted(j for group in GROUPS.values() for j in group))
LOSSES=('loss_own','loss_patch','loss_full')
TARGET_KEYS=('window_index','joint','patch_index','target_type')
PATCH_KEYS=('source_id','joint','sequence_id','patch_start_ns','patch_end_ns')


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def freeze_profile_eligibility(x,metadata,config=None,mask_seed=17,mask_repeats=16):
    """Freeze source eligibility using only missingness, time identity and masks.

    Return a boolean window mask and a source ledger. Neither coordinate values,
    binary labels nor prediction losses enter the decision. Labels are attached
    only to support the later attrition audit. Every retained source has four
    distinct eligible time patches for each of the twelve readout joints.
    """
    from .context_prediction import ContextConfig,make_mask_bank
    from .model import JOINT_INDICES
    cfg=config or ContextConfig()
    array=np.asarray(x)
    if array.shape!=(len(metadata),32,16,2) or not len(array):
        raise ValueError('Expected nonempty [windows,32,16,2] matching metadata')
    if np.isinf(array).any():raise ValueError('Infinite coordinates are corrupt input')
    if mask_repeats!=16:raise ValueError('The study fixes sixteen scoring masks')
    bank=make_mask_bank(cfg,mask_seed=mask_seed,mask_repeats=mask_repeats).cpu().numpy().reshape(mask_repeats,8,16)
    valid=np.isfinite(array).reshape(len(array),8,4,16,2).all(axis=(2,4))
    counts=np.zeros_like(valid,dtype=np.int64)
    for mask in bank:
        visible=(valid & ~mask).sum(axis=1)
        counts += mask & valid & (visible[:,None,:]>=2)
    slots=np.array([JOINT_INDICES.index(j) for j in READOUT_JOINTS])
    shape=(len(array),8,len(slots));window,patch,joint_slot=np.indices(shape)
    n=counts[:,:,slots].reshape(-1)
    requested=np.broadcast_to(bank[:,:,slots].sum(axis=0)[None],shape).reshape(-1)
    dummy=pd.DataFrame(dict(window_index=window.reshape(-1),joint=np.array(READOUT_JOINTS)[joint_slot.reshape(-1)],
        patch_index=patch.reshape(-1),target_type='coordinate',target_count=n,requested_count=requested,
        **{name:np.where(n>0,0.,np.nan) for name in LOSSES}))
    profiles,support,_=aggregate_profiles(dummy,metadata)
    ledger=profiles[['source_id','binary_label','categories','n_sequences','scorable','missing_joints',
                     'min_distinct_patches','requested_targets','common_target_exposures']].copy()
    ledger=ledger.rename(columns={'scorable':'profile_eligible'})
    for joint in READOUT_JOINTS:
        values=support[support.joint.eq(joint)].set_index('source_id').n_distinct_patches
        ledger[f'joint_{joint}_distinct_patches']=ledger.source_id.map(values)
    bank_hash=hashlib.sha256(bank.tobytes()).hexdigest()
    ledger['mask_seed']=mask_seed;ledger['mask_repeats']=mask_repeats;ledger['mask_bank_sha256']=bank_hash
    ledger['exclusion_reason']=np.where(ledger.profile_eligible,'','insufficient_distinct_readout_patches')
    keep=metadata.source_id.isin(ledger.loc[ledger.profile_eligible,'source_id']).to_numpy()
    ledger.attrs['eligibility_sha256']=_hash(dict(mask_bank_sha256=bank_hash,min_distinct_patches=4,
        min_own_context=2,readout_joints=READOUT_JOINTS,
        support=support[['source_id','joint','n_distinct_patches']].sort_values(['source_id','joint']).to_dict('records')))
    ledger.attrs['decision_inputs']='Finite-coordinate validity, fixed masks and sequence/time patch identities only'
    return keep,ledger


def compress_loss_rows(rows):
    """Average masks per window/joint/patch while retaining exact target weights.

    This reduces the large 16-mask table before joining source metadata. Raw
    duplicate identities do not add weight. Conflicting duplicates are errors.
    A valid target with nonfinite loss is a model failure, never a QC exclusion.
    Compressed rows retain requested_count and target_count for coverage audits.
    """
    table=rows.rename(columns={'joint_index':'joint'}).copy()
    required=set(TARGET_KEYS)|set(LOSSES)
    if not required.issubset(table):raise ValueError(f'Missing loss fields: {required-set(table)}')
    if 'target_count' in table and 'mask_id' not in table:
        if table.duplicated(list(TARGET_KEYS)).any():raise ValueError('Duplicate compressed target keys')
        count=table.target_count.to_numpy(float)
        if not np.isfinite(count).all() or np.any(count<0) or not np.equal(count,np.floor(count)).all():
            raise ValueError('Compressed target_count must be a nonnegative integer')
        if 'requested_count' not in table:table['requested_count']=count
        requested=table.requested_count.to_numpy(float)
        if not np.isfinite(requested).all() or np.any(requested<count) or not np.equal(requested,np.floor(requested)).all():
            raise ValueError('Requested count must be an integer at least as large as valid count')
        if not np.isfinite(table.loc[count>0,list(LOSSES)].to_numpy(float)).all():raise FloatingPointError('Nonfinite eligible prediction loss')
        return table[list(TARGET_KEYS)+list(LOSSES)+['target_count','requested_count']]
    if 'mask_id' not in table:raise ValueError('Raw losses need mask_id; compressed losses need target_count')
    mask_ids=table.mask_id.to_numpy(float)
    if not np.isfinite(mask_ids).all() or not np.equal(mask_ids,np.floor(mask_ids)).all() or np.any((mask_ids<0)|(mask_ids>=16)):
        raise ValueError('The fixed mask bank has integer IDs 0 through 15')
    keys=list(TARGET_KEYS)+['mask_id']
    duplicates=table.duplicated(keys,keep=False)
    if duplicates.any():
        checked=table.loc[duplicates].groupby(keys,dropna=False)[list(LOSSES)].nunique(dropna=False)
        if (checked>1).any().any():raise ValueError('Conflicting losses for the same target/mask identity')
        if 'valid' in table and table.loc[duplicates].groupby(keys).valid.nunique().gt(1).any():
            raise ValueError('Conflicting eligibility for the same target/mask identity')
        table=table.drop_duplicates(keys)
    finite=np.isfinite(table[list(LOSSES)].to_numpy(float)).all(axis=1)
    eligible=table.valid.to_numpy(bool) if 'valid' in table else finite
    if 'target_valid' in table and np.any(eligible & ~table.target_valid.to_numpy(bool)):
        raise ValueError('Eligible row has an unobserved target')
    if 'own_visible_count' in table and np.any(eligible & (table.own_visible_count.to_numpy()<2)):
        raise ValueError('Eligible row lacks two visible own-joint patches')
    if np.any(eligible & ~finite):raise FloatingPointError('Nonfinite eligible prediction loss')
    table.loc[~eligible,list(LOSSES)]=np.nan
    table['target_count']=eligible.astype(np.int64);table['requested_count']=1
    grouped=table.groupby(list(TARGET_KEYS),sort=True,observed=True,dropna=False)
    out=grouped[list(LOSSES)].mean()
    out[['target_count','requested_count']]=grouped[['target_count','requested_count']].sum()
    return out.reset_index()


def common_target_intersection(*tables):
    """Keep identical eligible raw target identities for paired methods/edits."""
    if not tables:return []
    keys=list(TARGET_KEYS)+['mask_id'];sets=[];normalized=[]
    for original in tables:
        table=original.rename(columns={'joint_index':'joint'}).copy()
        if not set(keys+list(LOSSES)).issubset(table):raise ValueError('Pairing requires raw target/mask rows')
        valid=table.valid.to_numpy(bool) if 'valid' in table else np.ones(len(table),bool)
        finite=np.isfinite(table[list(LOSSES)].to_numpy(float)).all(axis=1)
        if np.any(valid & ~finite):raise FloatingPointError('Nonfinite eligible prediction loss')
        normalized.append(table)
        sets.append(pd.MultiIndex.from_frame(table.loc[valid,keys]).unique())
    common=sets[0]
    for index in sets[1:]:common=common.intersection(index)
    return [t.loc[pd.MultiIndex.from_frame(t[keys]).isin(common)].copy() for t in normalized]


def _attach_metadata(compressed,metadata,patch_frames,window_index_is_global):
    meta=metadata.reset_index(drop=True).copy()
    required={'source_id','sequence_id','label','start_seconds','fps'}
    if not required.issubset(meta):raise ValueError(f'Missing window metadata: {required-set(meta)}')
    meta['_score_index']=meta.window_index if window_index_is_global else np.arange(len(meta))
    if meta._score_index.duplicated().any():raise ValueError('Window indices must be unique')
    if not set(compressed.window_index).issubset(set(meta._score_index)):raise ValueError('Loss window_index is outside supplied metadata')
    for column in ('split','outer_fold'):
        if column in meta and meta.groupby('source_id')[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f'A source has multiple {column} assignments')
    source_info=source_table(meta)
    for column in ('split','outer_fold','prior_use'):
        if column in meta:source_info[column]=source_info.source_id.map(meta.groupby('source_id')[column].first())
    fields=['_score_index','source_id','sequence_id','start_seconds','fps']
    if 'end_seconds' in meta:fields.append('end_seconds')
    merged=compressed.merge(meta[fields],left_on='window_index',right_on='_score_index',validate='many_to_one',sort=False)
    if not np.isfinite(merged[['start_seconds','fps','patch_index']].to_numpy(float)).all() or merged.fps.le(0).any() or merged.patch_index.lt(0).any():
        raise ValueError('Invalid time grid or patch index')
    if not np.equal(merged.patch_index,np.floor(merged.patch_index)).all():raise ValueError('Patch indices must be integers')
    start=merged.start_seconds+merged.patch_index*patch_frames/merged.fps
    end=start+patch_frames/merged.fps
    if 'end_seconds' in merged and (end>merged.end_seconds+1/merged.fps+1e-8).any():
        raise ValueError('Target patch extends beyond its window')
    merged['patch_start_ns']=np.rint(start*1e9).astype(np.int64)
    merged['patch_end_ns']=np.rint(end*1e9).astype(np.int64)
    return merged,source_info


def aggregate_profiles(long_rows,metadata,min_patches=4,patch_frames=4,fixed_budget=None,
                       seed=42,selection=None,window_index_is_global=False):
    """Return (source profiles, joint support table, fixed-budget selection).

    Loss indices are positional within supplied metadata unless explicitly
    declared global. Average common targets within a window, windows within a
    sequence, then sequences within a source separately for each joint. Regions
    give each of their four joints equal weight. Repeated masks and overlapping
    windows never increase distinct support. Physical support keys contain the
    sequence and nanosecond-rounded patch interval, not the window identifier.

    A supplied budget selection is never refilled after a paired edit removes
    support. Fixed-budget selections intersect support across supplied target
    types and depend only on source, joint, time identity and the fixed seed.
    """
    if min_patches<4 or patch_frames<1 or (fixed_budget is not None and fixed_budget!=4):
        raise ValueError('The study requires at least four patches; fixed-budget sensitivity uses four')
    compact=compress_loss_rows(long_rows)
    allowed={11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32}
    if not set(compact.joint).issubset(allowed):raise ValueError('joint must use original approved BlazePose indices')
    target_types=sorted(compact.target_type.unique())
    if not target_types or not set(target_types).issubset({'coordinate','latent'}):
        raise ValueError('Explicit coordinate/latent target_type is required')
    attached,source_info=_attach_metadata(compact,metadata,patch_frames,window_index_is_global)
    attached=attached[attached.joint.isin(READOUT_JOINTS)].copy()
    requested=attached.groupby(['source_id','target_type'],observed=True).requested_count.sum()
    usable=attached[attached.target_count.gt(0)].copy()
    selected=pd.DataFrame(columns=list(PATCH_KEYS))
    if selection is not None or fixed_budget is not None:
        if selection is None:
            # Only physical patches eligible for every supplied target type are
            # candidates. This keeps coordinate/latent and raw/profile methods
            # on a common fixed observation budget.
            types=usable.groupby(list(PATCH_KEYS),observed=True).target_type.nunique()
            candidates=types[types.eq(len(target_types))].reset_index()[list(PATCH_KEYS)]
            pieces=[]
            for (source,joint),group in candidates.groupby(['source_id','joint'],sort=True):
                group=group.copy()
                group['_key']=[_hash([int(seed),str(source),int(joint),str(r.sequence_id),int(r.patch_start_ns),int(r.patch_end_ns)]) for r in group.itertuples()]
                pieces.append(group.sort_values('_key').head(fixed_budget).drop(columns='_key'))
            selected=pd.concat(pieces,ignore_index=True) if pieces else selected
        else:
            if not set(PATCH_KEYS).issubset(selection):raise ValueError('Incomplete fixed support selection')
            selected=selection[list(PATCH_KEYS)].drop_duplicates().copy()
            if selected.groupby(['source_id','joint']).size().gt(4).any():raise ValueError('Budget selection exceeds four patches')
        usable=usable.merge(selected,on=list(PATCH_KEYS),how='inner',validate='many_to_one')
    key=['source_id','target_type','joint']
    distinct=usable.drop_duplicates(key+['sequence_id','patch_start_ns','patch_end_ns'])
    counts=distinct.groupby(key,observed=True).size()
    exposure=usable.groupby(key,observed=True).target_count.sum()
    time_support=distinct.groupby(key,observed=True).agg(n_sequences=('sequence_id','nunique'),
        first_patch_ns=('patch_start_ns','min'),last_patch_ns=('patch_end_ns','max'))
    support_index=pd.MultiIndex.from_product([source_info.source_id,target_types,READOUT_JOINTS],names=key)
    support=pd.DataFrame(index=support_index)
    support['n_distinct_patches']=counts.reindex(support_index,fill_value=0).astype(int)
    support['n_target_exposures']=exposure.reindex(support_index,fill_value=0).astype(int)
    support=support.join(time_support)
    support['n_sequences']=support.n_sequences.fillna(0).astype(int)
    support['time_span_seconds']=(support.last_patch_ns-support.first_patch_ns)/1e9
    support['supported']=support.n_distinct_patches.ge(min_patches)
    # Weighted patch means exactly reconstruct the original common-target mean
    # within a window after mask compression.
    weighted=usable.copy()
    for column in LOSSES:weighted[column]=weighted[column]*weighted.target_count
    win_keys=['source_id','target_type','joint','sequence_id','window_index']
    window=weighted.groupby(win_keys,observed=True)[list(LOSSES)+['target_count']].sum()
    window[list(LOSSES)]=window[list(LOSSES)].div(window.target_count,axis=0)
    sequence=window.reset_index().groupby(['source_id','target_type','joint','sequence_id'],observed=True)[list(LOSSES)].mean()
    joint=sequence.reset_index().groupby(key,observed=True)[list(LOSSES)].mean()
    support=support.join(joint)
    support['benefit']=support.loss_own-support.loss_full
    support['patch_benefit']=support.loss_own-support.loss_patch
    support=support.reset_index()
    records=[]
    info=source_info.set_index('source_id')
    for source in source_info.source_id:
        for target_type in target_types:
            group=support[support.source_id.eq(source)&support.target_type.eq(target_type)].set_index('joint')
            result={'source_id':source,'target_type':target_type,**info.loc[source].to_dict()}
            missing=[int(j) for j in READOUT_JOINTS if not bool(group.loc[j,'supported'])]
            result.update(scorable=not missing,missing_joints='|'.join(map(str,missing)),
                          min_distinct_patches=int(group.n_distinct_patches.min()),
                          requested_targets=int(requested.get((source,target_type),0)),
                          common_target_exposures=int(group.n_target_exposures.sum()),
                          fixed_budget=fixed_budget if fixed_budget is not None else (4 if selection is not None else None))
            for name,joints in GROUPS.items():
                valid=group.loc[list(joints),'supported'].all()
                for field,column in [('own','loss_own'),('patch','loss_patch'),('full','loss_full'),('benefit','benefit')]:
                    result[f'{field}_{name}']=float(group.loc[list(joints),column].mean()) if valid else np.nan
            for field in ('own','patch','full','benefit'):
                result[f'raw_{field}']=float(np.mean([result[f'{field}_{name}'] for name in GROUPS])) if result['scorable'] else np.nan
            result['anomaly_contrast']=result['raw_full']-result['raw_own']
            records.append(result)
    profiles=pd.DataFrame(records)
    protocol=dict(groups=GROUPS,min_patches=min_patches,patch_frames=patch_frames,
                  fixed_budget=fixed_budget,seed=seed,selection=selected.to_dict('records'))
    profiles.attrs['aggregation_sha256']=_hash(protocol)
    profiles.attrs['coverage']=dict(n_source_target_pairs=len(profiles),scorable_pairs=int(profiles.scorable.sum()),
                                    scorable_sources=int(profiles[profiles.scorable].source_id.nunique()))
    support.attrs.update(profiles.attrs);selected.attrs.update(protocol_sha256=profiles.attrs['aggregation_sha256'])
    return profiles,support,selected


@dataclass(frozen=True)
class ProfileReference:
    """Frozen diagonal reference fitted only to inner normal validation sources."""
    field:str
    target_type:str
    source_ids:tuple[str,...]
    mean:tuple[float,...]
    sample_variance:tuple[float,...]
    reference_variance:tuple[float,...]
    excluded_source_ids:tuple[str,...]=()
    sensitivity:bool=False
    variance_floor:float=1e-8
    shrinkage:float=0.5
    outer_fold:int|None=None

    @classmethod
    def fit(cls,profiles,field='benefit',target_type='latent',min_sources=4,
            training_source_ids=(),test_source_ids=()):
        if min_sources<4:raise ValueError('A main reference needs at least four sources')
        return cls._fit(profiles,field,target_type,min_sources,False,training_source_ids,test_source_ids)

    @classmethod
    def _fit(cls,profiles,field,target_type,min_sources,sensitivity,training_ids=(),test_ids=()):
        if field not in ('benefit','own','full','patch'):raise ValueError('Unknown profile field')
        if target_type not in ('coordinate','latent'):raise ValueError('Target units must be explicit')
        required={'source_id','target_type','binary_label','split','scorable'}|{f'{field}_{g}' for g in GROUPS}
        if not required.issubset(profiles):raise ValueError(f'Missing reference fields: {required-set(profiles)}')
        rows=profiles[profiles.target_type.eq(target_type)].copy()
        if rows.source_id.duplicated().any():raise ValueError('Reference sources must be unique')
        if 'outer_fold' in rows and rows.outer_fold.nunique(dropna=False)!=1:
            raise ValueError('Reference fitting requires exactly one outer fold')
        if rows.binary_label.ne(0).any() or rows.split.ne('val').any():
            raise ValueError('Reference fitting permits inner normal validation sources only')
        if set(rows.source_id)&(set(training_ids)|set(test_ids)):
            raise ValueError('Reference source overlaps predictor fitting or outer test')
        excluded=tuple(sorted(rows.loc[~rows.scorable,'source_id'].astype(str)))
        rows=rows[rows.scorable].sort_values('source_id')
        if len(rows)<min_sources:raise ValueError(f'Insufficient eligible inner normal references: {len(rows)} < {min_sources}')
        matrix=rows[[f'{field}_{g}' for g in GROUPS]].to_numpy(float)
        if not np.isfinite(matrix).all():raise ValueError('Reference profile has missing/nonfinite group values')
        variance=matrix.var(axis=0,ddof=1)
        regularized=.5*variance+.5*variance.mean()
        return cls(field,target_type,tuple(rows.source_id.astype(str)),tuple(matrix.mean(axis=0)),
                   tuple(variance),tuple(np.maximum(regularized,1e-8)),excluded,sensitivity,
                   outer_fold=int(rows.outer_fold.iloc[0]) if 'outer_fold' in rows else None)

    def to_dict(self):
        value=dict(field=self.field,target_type=self.target_type,source_ids=list(self.source_ids),
                   mean=list(self.mean),sample_variance=list(self.sample_variance),
                   reference_variance=list(self.reference_variance),excluded_source_ids=list(self.excluded_source_ids),
                   sensitivity=self.sensitivity,variance_floor=self.variance_floor,shrinkage=self.shrinkage,outer_fold=self.outer_fold,
                   group_order=list(GROUPS),groups={k:list(v) for k,v in GROUPS.items()})
        value['parameters_sha256']=_hash(value)
        return value

    @property
    def parameters_sha256(self):return self.to_dict()['parameters_sha256']

    def score(self,profiles):
        if not profiles.target_type.eq(self.target_type).all():raise ValueError('Do not mix coordinate and latent units')
        if self.outer_fold is not None and ('outer_fold' not in profiles or profiles.outer_fold.ne(self.outer_fold).any()):
            raise ValueError('Reference belongs to a different outer fold')
        values=profiles[[f'{self.field}_{g}' for g in GROUPS]].to_numpy(float)
        eligible=profiles.scorable.to_numpy(bool)
        if not np.isfinite(values[eligible]).all():raise ValueError('Scorable source has missing profile entries')
        result=np.full(len(profiles),np.nan,float)
        result[eligible]=np.mean(np.square(values[eligible]-np.asarray(self.mean))/np.asarray(self.reference_variance),axis=1)
        return result


def score_profiles(profiles,references):
    """Apply frozen references; retain raw/full/one-sided comparators unchanged."""
    out=profiles.copy()
    names={'benefit':'profile_score','own':'own_distance','full':'full_distance','patch':'patch_distance'}
    for field,reference in references.items():
        if reference.field!=field:raise ValueError('Reference field mismatch')
        out[names[field]]=reference.score(out)
        out[f'{field}_reference_sha256']=reference.parameters_sha256
    return out


def leave_one_reference_out(reference_profiles,evaluation_profiles,reference=None,
                           field='benefit',target_type='latent'):
    """Refit reference parameters only, with predictor losses and support fixed.

    A four-source main reference yields three-source diagnostic references.
    These are explicitly marked sensitivities and may not replace the main fit.
    The repeated scores are not additional independent observations.
    """
    reference=reference or ProfileReference.fit(reference_profiles,field,target_type)
    if reference.field!=field or reference.target_type!=target_type:raise ValueError('Reference specification mismatch')
    rows=[]
    base=reference_profiles[reference_profiles.source_id.isin(reference.source_ids)&reference_profiles.target_type.eq(target_type)]
    for omitted in reference.source_ids:
        fitted=ProfileReference._fit(base[base.source_id.ne(omitted)],field,target_type,3,True)
        scores=fitted.score(evaluation_profiles)
        for source,score in zip(evaluation_profiles.source_id,scores):
            rows.append(dict(source_id=source,left_out_source=omitted,field=field,target_type=target_type,
                score=float(score),n_reference_sources=len(fitted.source_ids),main_reference_minimum_met=len(fitted.source_ids)>=4,
                reference_sha256=fitted.parameters_sha256,sensitivity=True))
    return pd.DataFrame(rows)


def _score_cohort(scores,columns,expected_folds):
    if not columns:raise ValueError('At least one score column is required')
    required={'source_id','outer_fold','binary_label'}|set(columns)
    if not required.issubset(scores):raise ValueError(f'Missing evaluation fields: {required-set(scores)}')
    if scores.source_id.duplicated().any():raise ValueError('Each source needs one outer-test row; separate seeds/target types')
    if 'split' in scores and scores.split.ne('test').any():raise ValueError('Ranking evaluation permits outer-test sources only')
    if not set(scores.binary_label).issubset({0,1}):raise ValueError('Binary labels must be 0 or 1')
    if not set(scores.outer_fold).issubset(set(expected_folds)):raise ValueError('Unexpected outer fold')
    if 'target_type' in scores and scores.target_type.nunique()>1:raise ValueError('Evaluate coordinate and latent targets separately')
    finite=np.isfinite(scores[list(columns)].to_numpy(float)).all(axis=1)
    if 'scorable' in scores:finite &= scores.scorable.to_numpy(bool)
    table=scores.loc[finite].copy()
    coverage=dict(n_sources_supplied=len(scores),n_common_scorable_sources=len(table),n_excluded_sources=int((~finite).sum()),
                  excluded_source_ids=sorted(scores.loc[~finite,'source_id'].astype(str)))
    return table,coverage


def fold_ranking_metrics(source_scores,score_columns,expected_folds=(0,1,2,3),categories=None):
    """Within-fold AUROC/AP weighted by eligible normal-source count.

    Fitted-model score scales are never pooled. Category analyses reuse normals
    and may reuse mixed-category abnormal sources, so they are dependent.
    """
    from sklearn.metrics import roc_auc_score,average_precision_score
    columns=list(score_columns);table,coverage=_score_cohort(source_scores,columns,expected_folds)
    categories=list(categories or [c for c in STUDY_LABELS if c!='normal']);results={}
    for column in columns:
        folds=[]
        for fold in expected_folds:
            part=table[table.outer_fold.eq(fold)];y=part.binary_label.to_numpy(int)
            nn=int((y==0).sum());na=int((y==1).sum());supported=bool(nn and na)
            folds.append(dict(outer_fold=int(fold),n_normal=nn,n_abnormal=na,
                abnormal_prevalence=float(na/len(y)) if len(y) else None,
                auroc=float(roc_auc_score(y,part[column])) if supported else None,
                average_precision=float(average_precision_score(y,part[column])) if supported else None))
        complete=all(row['auroc'] is not None for row in folds);total=sum(row['n_normal'] for row in folds)
        summary=dict(primary_auroc=sum(row['n_normal']*row['auroc'] for row in folds)/total if complete else None,
            normal_weighted_ap=sum(row['n_normal']*row['average_precision'] for row in folds)/total if complete else None,
            folds=folds,all_folds_have_binary_support=complete,categories={})
        if 'categories' in table:
            membership=table.categories.fillna('').map(lambda text:set(str(text).split('|')))
            for category in categories:
                chosen=table.loc[table.binary_label.eq(0)|membership.map(lambda values:category in values)];parts=[]
                for fold in expected_folds:
                    part=chosen[chosen.outer_fold.eq(fold)];y=part.binary_label.to_numpy(int)
                    nn=int((y==0).sum());na=int((y==1).sum())
                    parts.append(dict(outer_fold=int(fold),n_normal=nn,n_category=na,
                        abnormal_prevalence=float(na/len(y)) if len(y) else None,
                        auroc=float(roc_auc_score(y,part[column])) if nn and na else None,
                        average_precision=float(average_precision_score(y,part[column])) if nn and na else None))
                supported=[p for p in parts if p['auroc'] is not None];weight=sum(p['n_normal'] for p in supported)
                n_category=int(chosen.binary_label.eq(1).sum())
                summary['categories'][category]=dict(n_category_sources=n_category,descriptive_only=n_category<=3,
                    supported_folds=len(supported),folds=parts,
                    normal_weighted_auroc=sum(p['n_normal']*p['auroc'] for p in supported)/weight if weight else None,
                    normal_weighted_ap=sum(p['n_normal']*p['average_precision'] for p in supported)/weight if weight else None)
        results[column]=summary
    return dict(methods=results,coverage=coverage,
                weighting='Within-fold metrics weighted by eligible normal test source counts',
                category_dependence='Normals and mixed-category sources may recur across category summaries',
                source_cohort_sha256=_hash(table[['source_id','outer_fold','binary_label']].sort_values('source_id').to_dict('records')))


def paired_fold_bootstrap(source_scores,score_a='profile_score',score_b='raw_full',
                          repeats=1000,seed=42,expected_folds=(0,1,2,3)):
    """Paired source resampling within fold and binary label, with fixed fits.

    This excludes retraining/reference uncertainty and unknown repeated-person
    dependence. Windows are never resampled as independent observations.
    """
    from sklearn.metrics import roc_auc_score
    if repeats<1:raise ValueError('Positive repeat count required')
    table,coverage=_score_cohort(source_scores,[score_a,score_b],expected_folds)
    table=table.reset_index(drop=True);strata={};weights={}
    for fold in expected_folds:
        for label in (0,1):
            indices=np.flatnonzero(table.outer_fold.eq(fold).to_numpy()&table.binary_label.eq(label).to_numpy())
            if not len(indices):raise ValueError('Every fold needs both binary labels for bootstrap')
            strata[(fold,label)]=indices
        weights[fold]=len(strata[(fold,0)])
    total=sum(weights.values());rng=np.random.default_rng(seed)
    y=table.binary_label.to_numpy();a=table[score_a].to_numpy(float);b=table[score_b].to_numpy(float)
    def difference(draws):
        return sum(weights[f]*(roc_auc_score(y[idx],a[idx])-roc_auc_score(y[idx],b[idx])) for f,idx in draws.items())/total
    original={f:np.concatenate([strata[(f,0)],strata[(f,1)]]) for f in expected_folds}
    point=float(difference(original));values=[]
    for _ in range(repeats):
        draws={f:np.concatenate([rng.choice(strata[(f,label)],size=len(strata[(f,label)]),replace=True) for label in (0,1)]) for f in expected_folds}
        values.append(difference(draws))
    return dict(score_a=score_a,score_b=score_b,point_difference=point,
                interval_95=np.quantile(values,[.025,.975]).tolist(),repeats=repeats,seed=seed,n_sources=len(table),
                normal_sources_by_fold={str(k):v for k,v in weights.items()},coverage=coverage,
                conditional_on='Fixed folds, predictor weights, reference parameters and common source cohort',
                paired=True,bootstrap_unit='source within outer fold and binary label')
