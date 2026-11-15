"""Paired timing and observation probes on development normal sources only."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from aaai_gait.settings import Settings
from aaai_gait.study_data import load_study_windows
from aaai_gait.context_prediction import load_context_bundle,score_context_models
from aaai_gait.predictive_profile import aggregate_profiles,compress_loss_rows,ProfileReference
from aaai_gait.study_controls import paired_edits,common_intervention_rows
from aaai_gait.study_runner import dump_json,prepare_study
from aaai_gait.data import JOINTS,file_sha256


def normalize_control_windows(x,meta,shift=4,frames=32):
    """Freeze pelvis centering and the original crop's first-16-frame scale.

    Forty-frame loading initially uses its first twenty frames. This division
    cancels that scale and replaces it with the median torso length over long
    frames 4:20, which are the original 32-frame crop's prefix. All edits retain
    this reference, including frames brought in by a temporal shift.
    """
    values=np.asarray(x,dtype=np.float32).copy()
    if values.shape!=(len(meta),frames+2*shift,16,2) or frames!=32 or shift!=4:
        raise ValueError('Controls require matching [windows,40,16,2] and shift=4')
    if np.isinf(values).any():raise ValueError('Infinite coordinates are corrupt input')
    pelvis=(values[:,:,JOINTS.index(23)]+values[:,:,JOINTS.index(24)])/2
    shoulder=(values[:,:,JOINTS.index(11)]+values[:,:,JOINTS.index(12)])/2
    lengths=np.linalg.norm(shoulder[:,shift:shift+16]-pelvis[:,shift:shift+16],axis=-1)
    scale=np.array([np.median(row[np.isfinite(row)]) if np.isfinite(row).any() else np.nan for row in lengths])
    keep=np.isfinite(scale)&(scale>1e-6)
    audit=meta[['source_id','sequence_id','start_seconds']].copy()
    audit['normalization_eligible']=keep;audit['prefix_torso_scale']=scale
    audit['exclusion_reason']=np.where(keep,'','original_crop_prefix_scale_unavailable')
    values=(values[keep]-pelvis[keep,:,None,:])/scale[keep,None,None,None]
    return values,meta.loc[keep].reset_index(drop=True),audit


def restore_reference(data,fold,validation_sources):
    """Restore every frozen parameter and verify its fold and fingerprint."""
    fields=('field','target_type','mean','sample_variance','reference_variance','source_ids',
            'excluded_source_ids','sensitivity','variance_floor','shrinkage','outer_fold')
    if not (set(fields)|{'parameters_sha256'})<=set(data):
        raise ValueError('Incomplete frozen-reference record; use current study output')
    kwargs={name:data[name] for name in fields}
    for name in ('mean','sample_variance','reference_variance','source_ids','excluded_source_ids'):
        kwargs[name]=tuple(kwargs[name])
    reference=ProfileReference(**kwargs)
    if reference.outer_fold!=fold or reference.parameters_sha256!=data['parameters_sha256']:
        raise ValueError('Reference fold or parameter fingerprint does not match')
    if (set(reference.source_ids)|set(reference.excluded_source_ids))!=set(validation_sources):
        raise ValueError('Reference sources disagree with frozen normal-validation roles')
    return reference


def _paired_profiles(table,requested,meta):
    # Zero common targets must produce unscorable profiles, never normal scores.
    keys=['window_index','joint','patch_index','target_type']
    scaffold=compress_loss_rows(requested)[keys+['requested_count']]
    if len(table):
        compact=scaffold.merge(compress_loss_rows(table).drop(columns='requested_count'),on=keys,how='left',validate='one_to_one')
    else:compact=scaffold.assign(target_count=0,loss_own=np.nan,loss_patch=np.nan,loss_full=np.nan)
    compact['target_count']=compact.target_count.fillna(0).astype(int)
    return aggregate_profiles(compact,meta)


def run_controls(run_dir,fold=0,synthetic_smoke=False,teacher_decomposition=False):
    torch.set_num_threads(2);run_dir=Path(run_dir).resolve()
    protocol=json.loads((run_dir/'protocol.json').read_text());synthetic=bool(protocol['synthetic'])
    if synthetic!=bool(synthetic_smoke):
        raise ValueError('Synthetic checks require --synthetic-smoke; omit it for a real run')
    if fold not in range(4) or protocol['mask_repeats']!=16:
        raise ValueError('Use a declared outer fold and the fixed sixteen-mask study')
    folder=run_dir/f'fold_{fold}';roles=json.loads((folder/'source_roles.json').read_text())
    if set(roles['train'])&(set(roles['val'])|set(roles['test'])):raise ValueError('Development roles overlap validation or test')
    provenance=dict(run_id=protocol['run_id'],fold=fold,method='trained',roles=roles)
    bundle=load_context_bundle(folder/'trained.pt',expected_provenance=provenance)
    if synthetic:
        from aaai_gait.demo import synthetic_corpus
        x,meta=synthetic_corpus(seed=protocol['seed'],frames=40,sources_per_class=12)
        use=meta.source_id.isin(roles['train'])&meta.label.eq('normal')
        x=x[use];meta=meta.loc[use].reset_index(drop=True)
        meta['start_seconds']=0.;meta['end_seconds']=39/15;meta['binary_label']=0
        ledger=meta.copy();corpus_hash=hashlib.sha256(x.tobytes()+meta.to_csv(index=False).encode()).hexdigest()
    else:
        settings=Settings(demo=False)
        _,current_meta,manifest,_=prepare_study(demo=False,seed=protocol['seed'])
        expected=protocol.get('pre_profile_corpus_sha256')
        if not expected or current_meta.attrs['corpus_sha256']!=expected:
            raise ValueError('Current annotation or pose corpus differs from the frozen study; preserve or restore its inputs')
        manifest=manifest[manifest.source_id.isin(roles['train'])&manifest.label.eq('normal')]
        x,meta,ledger=load_study_windows(manifest,settings.pose_dir,frames=40,stride=20,target_fps=15.,
                                       expected_frame_number_base=settings.frame_number_base)
        corpus_hash=meta.attrs['corpus_sha256']
    if not len(x):raise ValueError('No eligible long windows from development normal sources')
    x,meta,normalization_audit=normalize_control_windows(x,meta)
    if not len(x):raise ValueError('No development window has a valid original prefix scale')
    chosen=meta.sort_values(['source_id','sequence_id','start_seconds']).drop_duplicates('source_id').index.to_numpy()
    x=x[chosen];meta=meta.loc[chosen].reset_index(drop=True)
    meta['start_seconds']+=4/15;meta['end_seconds']=meta.start_seconds+31/15
    meta['split']='development';meta['outer_fold']=fold
    if not meta.label.eq('normal').all() or not set(meta.source_id)<=set(roles['train']):
        raise ValueError('Controls may use development normal sources only')
    edits,geometry=paired_edits(x,seed=42)
    raw={name:score_context_models(bundle,values,mask_seed=17,mask_repeats=16) for name,values in edits.items()}
    paired=common_intervention_rows(raw)
    keys=['window_index','joint','patch_index','mask_id','target_type']
    for table in paired.values():
        if not table[keys].equals(paired['original'][keys]):raise ValueError('Paired target identities differ')
    reference_data=json.loads((folder/'trained_references.json').read_text())
    records=[];coverage=[];support_tables=[]
    for name,table in paired.items():
        profiles,support,_=_paired_profiles(table,raw[name],meta)
        support['intervention']=name;support_tables.append(support)
        for kind in ('coordinate','latent'):
            reference=restore_reference(reference_data[f'trained_all_{kind}']['benefit'],fold,roles['val'])
            part=profiles[profiles.target_type.eq(kind)].copy()
            part['profile_score']=reference.score(part);part['reference_sha256']=reference.parameters_sha256
            part['intervention']=name;part['synthetic']=synthetic;records.append(part)
            coverage.append(dict(intervention=name,target_type=kind,
                requested_rows=int(raw[name].target_type.eq(kind).sum()),
                eligible_before_pairing=int((raw[name].target_type.eq(kind)&raw[name].valid).sum()),
                paired_rows=int(table.target_type.eq(kind).sum()),scorable_sources=int(part.scorable.sum()),
                unscorable_source_ids=part.loc[~part.scorable,'source_id'].tolist()))
    geometry=geometry.merge(meta[['source_id','sequence_id']].reset_index(names='window_index'),on='window_index',validate='many_to_one')
    out=folder/'development_controls';out.mkdir(exist_ok=True)
    pd.concat(records,ignore_index=True).to_csv(out/'paired_profiles.csv',index=False)
    pd.concat(support_tables,ignore_index=True).to_csv(out/'paired_support.csv',index=False)
    geometry.to_csv(out/'geometry.csv',index=False);meta.to_csv(out/'windows.csv',index=False)
    ledger.to_csv(out/'long_window_eligibility.csv',index=False)
    normalization_audit.to_csv(out/'normalization_eligibility.csv',index=False)
    for name,table in paired.items():table.to_csv(out/f'{name}_paired_losses.csv.gz',index=False,compression='gzip')
    teacher_report=None
    if teacher_decomposition:
        from aaai_gait.teacher_decomposition import score_teacher_decomposition,decompose_teacher_losses
        # The relative edit moves the left leg. These eight readout joints keep
        # every local measurement fixed, including their visible own histories.
        joints=(13,14,15,16,26,28,30,32);slots=[JOINTS.index(j) for j in joints]
        if not np.array_equal(edits['original'][:,:,slots],edits['relative_limb_shift'][:,:,slots],equal_nan=True):
            raise ValueError('Teacher isolation requires unchanged local readout measurements')
        factors=score_teacher_decomposition(bundle,edits['original'],edits['relative_limb_shift'],
            mask_seed=17,mask_repeats=16,joints=joints)
        decomposition=decompose_teacher_losses(factors)
        for component in ('predictor','teacher','interaction','joint'):
            decomposition[f'{component}_contrast']=decomposition[f'{component}_full']-decomposition[f'{component}_own']
        target=out/'teacher_decomposition';target.mkdir(exist_ok=True)
        for name,table in factors.items():table.to_csv(target/f'{name}.csv.gz',index=False,compression='gzip')
        decomposition.to_csv(target/'raw_kl_components.csv.gz',index=False,compression='gzip')
        components=[c for c in decomposition if any(c.startswith(name+'_') for name in ('predictor','teacher','interaction','joint'))]
        window=decomposition.loc[decomposition.valid].groupby(['window_index','joint'])[components].mean().reset_index()
        window=window.merge(meta[['source_id','sequence_id']].reset_index(names='window_index'),on='window_index',validate='many_to_one')
        sequence=window.groupby(['source_id','sequence_id','joint'])[components].mean().reset_index()
        sequence.groupby(['source_id','joint'])[components].mean().reset_index().to_csv(target/'source_joint_components.csv',index=False)
        teacher_report=dict(joints=list(joints),unchanged_local_measurements=True,
            requested_targets=len(decomposition),common_eligible_targets=int(decomposition.valid.sum()),
            code_sha256=file_sha256(Path(__file__).resolve().parents[1]/'aaai_gait/teacher_decomposition.py'),
            interpretation='Teacher component is a contextual model-target effect, not a neural or disease mechanism.',
            aggregation='Raw KL components per unchanged joint; no three-region profile is formed from eight joints.')
        dump_json(target/'protocol.json',{**teacher_report,**factors['p0y0'].attrs})
    dump_json(out/'protocol.json',dict(fold=fold,synthetic=synthetic,training_sources_only=True,
        selected_source_ids=meta.source_id.tolist(),unavailable_development_source_ids=sorted(set(roles['train'])-set(meta.source_id)),
        one_window_per_source=True,selection='First eligible sequence/time window per source, before edits and scoring',
        normalization='Original crop frames 0:16 correspond to long-window frames 4:20; fixed across all edits',
        shift_frames=4,noise_sd=.01,observation_dropout=.05,mask_seed=17,mask_repeats=16,
        control_code_sha256=file_sha256(Path(__file__)),checkpoint_sha256=file_sha256(folder/'trained.pt'),
        long_window_corpus_sha256=corpus_hash,references_sha256=file_sha256(folder/'trained_references.json'),
        common_target_sha256=hashlib.sha256(paired['original'][keys].to_csv(index=False).encode()).hexdigest(),
        coverage=coverage,teacher_decomposition=teacher_report,
        interpretation='Software sensitivity, not simulated disease or causal intervention',
        reference_caveat='Reference stays frozen; reduced paired support changes precision. Unscorable profiles remain missing.'))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--synthetic-smoke',action='store_true')
    parser.add_argument('--teacher-decomposition',action='store_true')
    args=parser.parse_args();print(run_controls(args.run_dir,args.fold,args.synthetic_smoke,args.teacher_decomposition))

if __name__=='__main__':main()
