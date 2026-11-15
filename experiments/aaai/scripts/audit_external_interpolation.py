"""Post-outcome interpolation check on frozen external normal test targets.

This derivative does not fit, select, gate, calibrate, or replace any model.
"""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
import numpy as np
import pandas as pd
from aaai_gait.external_iteration02 import angle_patches,permitted_features
BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'artifacts/iteration02/external_fb9d0d23b56ce29fd846'
PREP=BASE/'artifacts/iteration02/preparation/external_replication'
OUT=RUN/'derivatives/interpolation_audit_v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,x):Path(path).write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')

def independent_interpolation(p,hidden,targets):
    """Direct row loop verifies feature positions without using learned values."""
    out=np.zeros((len(p),4));mode=[]
    for i,target in enumerate(targets):
        joint=target%6;time=target//6;tokens=np.arange(8)*6+joint;allowed=~hidden[i,tokens]&np.isfinite(p[i,tokens]).all(1)
        before=np.flatnonzero(allowed&(np.arange(8)<time));after=np.flatnonzero(allowed&(np.arange(8)>time))
        if len(before) and len(after):
            left=int(before[-1]);right=int(after[0]);a=float(p[i,tokens[left],3]);b=float(p[i,tokens[right],0])
            q=4*time+np.arange(4);out[i]=a+(q-(4*left+3))/(4*right-(4*left+3))*(b-a);mode.append('two_sided_linear')
        elif len(before):out[i]=p[i,tokens[int(before[-1])],3];mode.append('previous_endpoint_hold')
        elif len(after):out[i]=p[i,tokens[int(after[0])],0];mode.append('next_endpoint_hold')
        else:mode.append('no_visible_target_angle_zero_fallback')
    return out,np.asarray(mode)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists():raise ValueError('Preserve the completed derivative audit')
    protocol=json.loads((RUN/'protocol.json').read_text())
    for name in ['windows.npy','window_index.csv','mask_bank.npy','participant_registry.csv']:
        if sha(PREP/name)!=protocol['identity']['data'][name]:raise ValueError('Frozen input hash changed: '+name)
    current=ast.parse((BASE/'aaai_gait/external_iteration02.py').read_text());original=ast.parse((RUN/'source_snapshot/external_iteration02.py').read_text())
    for name in ['angle_patches','permitted_features']:
        a=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
        if ast.dump(a)!=ast.dump(b):raise ValueError('Frozen feature producer changed')
    x=np.load(PREP/'windows.npy');allmeta=pd.read_csv(PREP/'window_index.csv');bank=np.load(PREP/'mask_bank.npy');p=angle_patches(x);lookup=allmeta.set_index('window_id').window_index
    folds=[];participants=[];ranks=[];modes=[];audits=[];sourcefiles={}
    methods=['interpolation','base','own','both']
    for fold in range(4):
        folder=OUT/f'fold_{fold}';folder.mkdir(exist_ok=True);frozen=RUN/f'fold_{fold}';meta=pd.read_csv(frozen/'scoring_window_index.csv');scores=pd.read_csv(frozen/'scores.csv');normals=set(scores.loc[scores.binary_label.eq(0),'source_id']);normal_window=meta.source_id.isin(normals).to_numpy();global_windows=meta.window_id.map(lookup).to_numpy(int);pieces=[];max_difference=0.;files=0
        for path in sorted((frozen/'predictions').glob('predictions_*.npz')):
            sourcefiles[str(path.relative_to(RUN))]=sha(path)
            with np.load(path) as data:
                keep=normal_window[data['window_index']]
                if not keep.any():continue
                saved={n:data[n][keep] for n in data.files}
            row=pd.DataFrame({k:saved[k] for k in ['window_index','angle','patch_index','mask_id']});patches=p[global_windows[row.window_index.to_numpy()]];hidden=bank[row.mask_id.to_numpy()];target=row.angle.to_numpy()+6*row.patch_index.to_numpy();f=permitted_features(patches,hidden,target)
            if not f['eligible'].all():raise ValueError('A frozen target lost support')
            np.testing.assert_array_equal(f['target'],saved['target'])
            prediction=f['own'][:,:4];independent,mode=independent_interpolation(patches,hidden,target);delta=float(np.max(np.abs(prediction-independent)))
            np.testing.assert_allclose(prediction,independent,rtol=1e-12,atol=1e-12);max_difference=max(max_difference,delta);row['interpolation_mode']=mode
            for method in methods:row['loss_'+method]=np.square((prediction if method=='interpolation' else saved[method])-saved['target']).mean(1)
            np.savez_compressed(folder/path.name,**saved,interpolation=prediction,interpolation_mode=mode);files+=1;pieces.append(row)
        raw=pd.concat(pieces,ignore_index=True);losses=['loss_'+m for m in methods];keys=['window_index','angle','patch_index'];compact=raw.groupby(keys)[losses].mean().reset_index();compact.to_csv(folder/'target_losses.csv.gz',index=False)
        old=pd.read_csv(frozen/'predictions/target_losses.csv.gz');old=old[old.window_index.isin(meta.loc[normal_window,'window_index'])].set_index(keys).sort_index();new=compact.set_index(keys).sort_index();np.testing.assert_allclose(new[['loss_base','loss_own','loss_both']],old[['loss_base','loss_own','loss_both']],rtol=1e-12,atol=1e-12)
        table=compact.merge(meta[['window_index','source_id']],on='window_index',validate='many_to_one');window=table.groupby(['source_id','window_index','angle'])[losses].mean();profile=window.groupby(['source_id','angle']).mean();profile.to_csv(folder/'participant_angle_profiles.csv');subject=profile.groupby('source_id').mean();subject['outer_fold']=fold;participants.append(subject.reset_index())
        for method in methods:folds.append(dict(outer_fold=fold,participants=len(subject),method=method,mean_participant_mse=float(subject['loss_'+method].mean())))
        for mode,count in raw.interpolation_mode.value_counts().items():modes.append(dict(outer_fold=fold,mode=mode,mask_target_exposures=int(count)))
        audit=json.loads((frozen/'fit_audit.json').read_text());recipe=json.loads((frozen/'selection.json').read_text())
        ranks.append(dict(outer_fold=fold,base_feature_width=36,base_rank=audit['base_rank'],residual_width=72,
            selected_own_rank=audit['residual_ranks']['own'][str(recipe['own_alpha'])],selected_both_rank=audit['residual_ranks']['both'][str(recipe['both_alpha'])],own_alpha=recipe['own_alpha'],both_alpha=recipe['both_alpha'],own_multiplier=recipe['own_multiplier'],both_multiplier=recipe['both_multiplier']))
        audits.append(dict(outer_fold=fold,normal_test_participants=len(subject),requested_frozen_target_rows=len(raw),unique_window_angle_patch_targets=len(compact),prediction_files=files,independent_interpolation_max_difference=max_difference,target_and_saved_loss_identity='passed'))
    folds=pd.DataFrame(folds);folds.to_csv(OUT/'fold_comparison.csv',index=False);participants=pd.concat(participants,ignore_index=True);participants.to_csv(OUT/'participant_comparison.csv',index=False);pd.DataFrame(ranks).to_csv(OUT/'basis_rank_audit.csv',index=False);pd.DataFrame(modes).to_csv(OUT/'interpolation_modes.csv',index=False)
    means={m:float(participants['loss_'+m].mean()) for m in methods};summary=dict(status='complete',audit_type='post_outcome_descriptive_missing_baseline_check_no_refit_or_selection',created_at_utc=datetime.now(timezone.utc).isoformat(),original_protocol_hash=protocol['protocol_hash'],normal_test_participants=len(participants),normal_test_participants_unique=int(participants.source_id.nunique()),mean_participant_mse=means,
        relative_reduction_vs_interpolation={m:(means['interpolation']-means[m])/means['interpolation'] for m in ['base','own','both']},folds=folds.to_dict('records'),rank_audit=ranks,verification=audits,
        interpolation_rule='Target-angle piecewise linear interpolation between nearest permitted previous-patch endpoint and next-patch start; one-sided endpoint hold; no-visible-target-angle zero fallback. No model fit.',aggregation='Mask draws within identical target; targets within window; windows within participant/angle; six angles equally within participant; participants equally within fold.',
        scope='Frozen outer-test normal targets only; no abnormal scoring, no anomaly calibration, no protocol replacement.',code_sha256=sha(__file__),input_prediction_file_hashes=sourcefiles)
    write(OUT/'summary.json',summary);write(OUT/'artifact_hashes.json',{str(path.relative_to(OUT)):sha(path) for path in sorted(OUT.rglob('*')) if path.is_file() and path.name!='artifact_hashes.json'})
    print(json.dumps({k:v for k,v in summary.items() if k not in ['input_prediction_file_hashes']},indent=2))
if __name__=='__main__':main()
