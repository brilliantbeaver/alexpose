"""Independent reconstruction of saved iteration02 predictions and scores."""
from pathlib import Path
import argparse,hashlib,json,time
import joblib
import numpy as np
import pandas as pd
from .data import file_sha256
from .iteration_scoring import (aggregate_target_profiles,bilateral_profiles,BilateralReference,
    ranking_metrics,endpoint_frame,collapse_recording_duplicates)
from .study_runner import dump_json
from .masking import make_mask_bank


def verify_run(out, *, all_predictions=True):
    out=Path(out);started=time.perf_counter();checks=[]
    protocol=json.loads((out/'protocol.json').read_text());registry=pd.read_csv(out/'source_registry.csv')
    manifest=json.loads((out/'run_manifest.json').read_text())
    if manifest['mode']!='full' or manifest['data_kind']!='real_gavd':
        raise ValueError('Synthetic or development runs cannot verify as efficacy evidence')
    if registry.source_id.duplicated().any():raise ValueError('Repeated source registry row')
    if file_sha256(out/'source_registry.csv')!=protocol['identity']['registry_sha256']:
        raise ValueError('Source registry differs from the frozen protocol')
    for fold in range(4):
        role=f'role_fold_{fold}';inner=f'inner_fold_{fold}'
        if registry.groupby('cluster_id')[role].nunique().max()!=1:raise ValueError('Duplicate footage leaks across roles')
        if registry.loc[registry[role].isin(['train','reference']),'binary_label'].ne(0).any():raise ValueError('Abnormal normal-only role')
        record=json.loads((out/f'fold_{fold}/internal_fit_audit.json').read_text())
        for item in record:
            fit=set(item['fit_sources']);val=set(item['validation_sources'])
            if fit&val or fit|val!=set(registry.loc[registry[role].eq('train'),'source_id']):raise ValueError('Internal role mismatch')
        checks.append(dict(name=f'fold{fold}_source_roles',passed=True))
    for name,seed in [('U',17),('E_leg',117)]:
        bank=make_mask_bank(name,16,seed);saved=np.load(out/f'mask_bank_{name}.npz')
        np.testing.assert_array_equal(saved['hidden'],bank.hidden);np.testing.assert_array_equal(saved['targets'],bank.targets)
        if bank.sha256!=protocol['identity']['banks'][name]['sha256']:raise ValueError('Evaluation bank hash changed')
    checks.append(dict(name='frozen_mask_banks',passed=True))
    raw_count=0;target_count=0;files_checked=0
    for fold in range(4):
        role=f'role_fold_{fold}';fit=set(registry.loc[registry[role].eq('train'),'source_id'])
        ref=set(registry.loc[registry[role].eq('reference'),'source_id'])
        base_parameters=[]
        for policy in ('U','J','A'):
            model=joblib.load(out/f'fold_{fold}/model_{policy}.joblib')
            if set(model.base.metadata['normal_sources'])!=fit:raise ValueError('Model fit sources changed')
            base_parameters.append((model.base.ridge.coef,model.base.ridge.intercept))
            for bank in ('U','E_leg'):
                folder=out/f'fold_{fold}/{policy}_{bank}';meta=pd.read_csv(folder/'window_metadata.csv')
                compact=pd.read_csv(folder/'target_losses.csv.gz');profiles=pd.read_csv(folder/'source_joint_profiles.csv')
                reconstructed,_=aggregate_target_profiles(compact,meta)
                pd.testing.assert_frame_equal(reconstructed,profiles,check_dtype=False,rtol=1e-10,atol=1e-11)
                references=json.loads((folder/'references.json').read_text()) if (folder/'references.json').exists() else None
                if references:
                    for name,reference in references.items():
                        if set(reference['sources'])!=ref:raise ValueError('Reference source substitution')
                        rebuilt=BilateralReference.fit(bilateral_profiles(profiles,name),sorted(ref))
                        np.testing.assert_allclose(rebuilt.mean,reference['mean'],rtol=1e-9,atol=1e-12)
                        np.testing.assert_allclose(rebuilt.variance,reference['variance'],rtol=1e-9,atol=1e-12)
                target_count+=len(compact)
                if not all_predictions:continue
                sums=np.zeros((len(meta)*64,3));counts=np.zeros(len(meta)*64,dtype=np.int64)
                for path in sorted(folder.glob('predictions_*.npz')):
                    z=np.load(path);keys=z['window_index']*64+(z['joint']-25)*8+z['patch_index']
                    target=np.stack([z[f'Y_{k}'] for k in range(8)],axis=1)
                    counts+=np.bincount(keys,minlength=len(counts));raw_count+=len(keys);files_checked+=1
                    for k,branch in enumerate(('base','own','both')):
                        predicted=np.stack([z[f'{branch}_{j}'] for j in range(8)],axis=1)
                        error=np.square(predicted-target).mean(1)
                        np.testing.assert_allclose(error,z[f'loss_{branch}'],rtol=1e-11,atol=1e-12)
                        sums[:,k]+=np.bincount(keys,weights=error,minlength=len(counts))
                keys=compact.window_index.to_numpy()*64+(compact.joint.to_numpy()-25)*8+compact.patch_index.to_numpy()
                np.testing.assert_array_equal(counts[keys],compact.mask_draws)
                if np.count_nonzero(counts)!=len(compact):raise ValueError('Compact target census differs from raw predictions')
                np.testing.assert_allclose(sums[keys]/counts[keys,None],compact[['loss_base','loss_own','loss_both']],rtol=1e-10,atol=1e-11)
        for coef,intercept in base_parameters[1:]:
            np.testing.assert_array_equal(coef,base_parameters[0][0]);np.testing.assert_array_equal(intercept,base_parameters[0][1])
    checks.append(dict(name='retained_shared_baselines_and_all_target_predictions',passed=True,raw_rows=raw_count,
                       distinct_window_targets=target_count,npz_files=files_checked,all_predictions=all_predictions))
    scores=pd.read_csv(out/'source_scores.csv')
    for (fold,policy,bank),part in scores.groupby(['outer_fold','policy','bank']):
        if part.source_id.duplicated().any():raise ValueError('Duplicate source score')
        expected=set(registry.loc[registry[f'role_fold_{fold}'].eq('test'),'source_id'])
        if set(part.source_id)!=expected:raise ValueError('Outer source population changed')
        np.testing.assert_allclose(part.S,.5*(part.M+part.R),rtol=1e-12,atol=1e-12,equal_nan=True)
        if part.fallback.all() and not np.equal(part.loc[part.R.notna(),'R'],0).all():raise ValueError('Fallback is not zero')
        folder=out/f'fold_{fold}/{policy}_{bank}';refs_path=folder/'references.json'
        if refs_path.exists():
            profiles=pd.read_csv(folder/'source_joint_profiles.csv');params=json.loads(refs_path.read_text())['benefit']
            reference=BilateralReference(np.asarray(params['mean']),np.asarray(params['variance']),tuple(params['sources']))
            expected_R=reference.score(bilateral_profiles(profiles)).reindex(part.source_id)
            np.testing.assert_allclose(expected_R.to_numpy(),part.R_raw,rtol=1e-8,atol=1e-10,equal_nan=True)
    checks.append(dict(name='all_source_scores_and_fallbacks',passed=True,rows=len(scores)))
    if (out/'rawmetrics.json').exists():
        metrics=json.loads((out/'rawmetrics.json').read_text())
        for key,methods in metrics.items():
            policy,bank,endpoint=key.split('/')
            table=scores.loc[scores.policy.eq(policy)&scores.bank.eq(bank)]
            table=collapse_recording_duplicates(endpoint_frame(table,endpoint),list(methods))
            for method,value in methods.items():
                rebuilt=ranking_metrics(table,method)
                if value['status']!=rebuilt['status']:raise ValueError('Metric support status differs')
                if 'weighted_auroc' in value:np.testing.assert_allclose(rebuilt['weighted_auroc'],value['weighted_auroc'],atol=1e-13)
        checks.append(dict(name='metrics_recomputed_from_saved_source_scores',passed=True))
    old=out.parents[2]/'docs/draft01/manuscript.pdf'
    if file_sha256(old)!='bf66987f4ed2b96ab8dda157afd24fcc8eddb9c58c42d446fa7c29e35a6b7cec':
        raise ValueError('Historical manuscript changed')
    checks.append(dict(name='draft01_manuscript_preserved',passed=True))
    result=dict(passed=True,status='verified_core' if (out/'rawmetrics.json').exists() else 'verified_predictions_pending_metrics',
                checks=checks,protocol_hash=out.name,data_kind='real_gavd',wall_seconds=time.perf_counter()-started,
                scope='core data, models, raw predictions, source aggregation and metrics; extensions separately audited')
    dump_json(out/'verification.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--quick',action='store_true')
    args=parser.parse_args();print(json.dumps(verify_run(args.run,all_predictions=not args.quick),indent=2))
