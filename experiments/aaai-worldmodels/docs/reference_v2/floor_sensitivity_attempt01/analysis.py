"""Post hoc sensitivity of saved normal calibration to the numerical log floor.

Forecasts, source partitions, score direction, and the primary endpoint are fixed.
An isolated in-memory copy of the frozen evaluator varies only its FLOOR constant.
No original module, checkpoint, prediction, or result is overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from worldmodels import v2_evaluation as ev

CORE=PROJECT/'artifacts/runs/reference-v2-core-v1'
PROTOCOL=PROJECT/'docs/reference_v2/floor_sensitivity_protocol.json'
REVIEW=PROJECT/'docs/reference_v2/floor_sensitivity_review.json'
OUTPUT=PROJECT/'artifacts/runs/reference-v2-floor-posthoc-v1'
FLOORS=(1e-10,1e-8,1e-6)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    def serial(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=serial)+'\n')


def evaluator_for_floor(floor):
    if float(floor) not in FLOORS:raise ValueError('Only the three declared floors are allowed')
    path=PROJECT/'worldmodels/v2_evaluation.py'
    # The worldmodels package name preserves the frozen module's relative import.
    specification=importlib.util.spec_from_file_location('worldmodels._floor_sensitivity_copy',path)
    module=importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.FLOOR=float(floor)
    assert ev.FLOOR==1e-8,'The frozen evaluator global must remain unchanged'
    return module


def self_test():
    frame=pd.DataFrame({'mse':[0.,2.],'prefix_motion':[0.,3.],'prefix_missingness':[.2,.1]})
    for floor in FLOORS:
        module=evaluator_for_floor(floor)
        np.testing.assert_array_equal(module._z(frame)[:,0],np.log(np.array([0.,3.])+floor))
        np.testing.assert_array_equal(module._log_error(frame),np.log(np.array([0.,2.])+floor))
        np.testing.assert_array_equal(module.empirical_mid_cdf([0.,0.,1.],[-1.,0.,2.]),[.125,.375,.875])
        assert module.FLOOR==floor and ev.FLOOR==1e-8
    return {'status':'passed_synthetic_software_checks_only','zero_and_nonzero_log_values':True,
            'midrank_and_endpoint_smoothing':True,'original_evaluator_global_unchanged':True}


def verify_declaration(protocol_path,review_path):
    declaration=json.loads(Path(protocol_path).read_text());review=json.loads(Path(review_path).read_text())
    if declaration['script_sha256']!=sha(__file__):raise ValueError('Script changed after declaration')
    if declaration['floors']!=list(FLOORS):raise ValueError('Numerical floors differ from the declaration')
    if review.get('status')!='accepted_before_posthoc_fitting':raise ValueError('Independent pre-execution review is required')
    if review.get('protocol_sha256')!=sha(protocol_path) or review.get('script_sha256')!=sha(__file__):
        raise ValueError('Review does not match this exact implementation and declaration')
    for path,expected in declaration['input_sha256'].items():
        if sha(PROJECT/path)!=expected:raise ValueError('A declared input changed: '+path)
    return declaration


def replay_component(module,entry):
    saved=json.loads((CORE/entry['calibrator_path']).read_text())
    reference=pd.DataFrame(saved['nuisance_reference_rows'])
    calibration=pd.DataFrame(saved['calibration_source_rows'])
    heldout=pd.read_csv(CORE/entry['scores_path'])
    calibrator=module.fit_source_calibrator(reference,calibration,schedule=entry['schedule'])
    calibrator['nuisance_model']['response']=f'log(source_mse + {module.FLOOR:g})'
    calibrator['identity']='reference-v2-posthoc-floor-calibrator-v1'
    scores=module.score_source_calibrator(calibrator,heldout)
    scores['rotation']=entry['rotation']
    return calibrator,scores,(reference,calibration,heldout)


def direction_agreement(a,b):
    """Count changes in all unordered score-pair directions, retaining ties."""
    upper=np.triu_indices(len(a),1)
    left=np.sign(a[:,None]-a[None,:])[upper]
    right=np.sign(b[:,None]-b[None,:])[upper]
    return {'all_source_pairs':len(left),'changed_pair_directions':int(np.count_nonzero(left!=right)),
            'agreement_fraction':float(np.mean(left==right))}


def run(output=OUTPUT,protocol_path=PROTOCOL,review_path=REVIEW):
    declaration=verify_declaration(protocol_path,review_path)
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve the completed analysis; use a fresh reviewed identity for changes')
    started=time.monotonic();software=self_test()
    manifest=json.loads((CORE/'artifact_hashes.json').read_text())
    for key in declaration['core_artifacts']:
        if sha(CORE/key)!=manifest[key]:raise ValueError('Core artifact differs from the frozen manifest: '+key)
    ledger=json.loads((CORE/'calibration_models.json').read_text())
    if len(ledger)!=15 or any(item['family']!='ridge_posture_velocity' for item in ledger):
        raise ValueError('Expected all15 frozen primary calibration components')
    baseline=pd.read_csv(CORE/'source_anomaly_scores.csv').sort_values('source_id').reset_index(drop=True)
    completed=json.loads((CORE/'results.json').read_text())
    all_scores={};all_components={};component_rows=[];covariate_rows=[]
    for floor in FLOORS:
        module=evaluator_for_floor(floor);rotations={};fitted=[]
        for entry in ledger:
            calibrator,scores,inputs=replay_component(module,entry)
            key=f"fold_{entry['outer_fold']}_rotation_{entry['rotation']}"
            fitted.append({'outer_fold':entry['outer_fold'],'rotation':entry['rotation'],'calibrator':calibrator})
            rotations.setdefault(entry['outer_fold'],[]).append(scores)
            component_rows.extend(scores.assign(numerical_floor=floor).to_dict('records'))
            if floor==1e-8:
                saved=json.loads((CORE/entry['calibrator_path']).read_text())
                np.testing.assert_allclose(calibrator['calibration_residuals'],saved['calibration_residuals'],atol=1e-12,rtol=0)
                original=pd.read_csv(CORE/entry['scores_path'])
                np.testing.assert_allclose(scores.two_sided,original.two_sided,atol=1e-12,rtol=0)
            for role,frame in zip(('nuisance_oof','calibration','outer'),inputs):
                for row in frame.itertuples():
                    covariate_rows.append({'floor':floor,'outer_fold':entry['outer_fold'],'rotation':entry['rotation'],
                        'role':role,'source_id':row.source_id,'mse':row.mse,'prefix_motion':row.prefix_motion})
        pooled=pd.concat([module.ensemble_rotation_scores(values) for values in rotations.values()],ignore_index=True)
        pooled=pooled.rename(columns={'two_sided':'candidate'})
        pooled=pooled.merge(baseline[['source_id','low_motion']],on='source_id',validate='one_to_one')
        pooled=pooled.sort_values('source_id').reset_index(drop=True)
        assert pooled.source_id.tolist()==baseline.source_id.tolist()
        if floor==1e-8:
            np.testing.assert_allclose(pooled.candidate,baseline.calibrated_two_sided,atol=1e-12,rtol=0)
        all_scores[floor]=pooled;all_components[floor]=fitted
    records=[];base=all_scores[1e-8].candidate.to_numpy()
    for floor in FLOORS:
        frame=all_scores[floor]
        primary=ev.paired_within_fold_bootstrap(frame,'candidate','low_motion',reps=declaration['bootstrap']['draws'],seed=declaration['bootstrap']['seed'])
        if floor==1e-8:
            np.testing.assert_allclose(primary['estimate'],completed['primary_anomaly']['estimate'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(primary['ci95'],completed['primary_anomaly']['ci95'],atol=1e-12,rtol=0)
        rows=pd.DataFrame(covariate_rows);rows=rows[rows.floor.eq(floor)]
        measures={}
        for name in ['mse','prefix_motion']:
            values=rows[name].to_numpy(float)
            measures[name]={'minimum':float(values.min()),'maximum':float(values.max()),
                'repeated_source_role_rows':len(values),'at_or_below_floor':int(np.count_nonzero(values<=floor)),
                'max_abs_log_change_vs_1e8':float(np.max(np.abs(np.log(values+floor)-np.log(values+1e-8))))}
        difference=frame.candidate.to_numpy()-base
        records.append({'floor':floor,'primary_anomaly':primary,'max_abs_score_change_vs_1e8':float(np.max(np.abs(difference))),
            'changed_source_scores_above_1e12':int(np.count_nonzero(np.abs(difference)>1e-12)),
            'ranking_agreement_vs_1e8':direction_agreement(frame.candidate.to_numpy(),base),'log_inputs':measures})
    output.mkdir(parents=True)
    for floor in FLOORS:
        tag=f'{floor:.0e}'
        all_scores[floor].to_csv(output/f'source_scores_floor_{tag}.csv',index=False)
        write_json(output/f'calibrators_floor_{tag}.json',all_components[floor])
    pd.DataFrame(component_rows).to_csv(output/'component_scores.csv',index=False)
    pd.DataFrame(covariate_rows).to_csv(output/'log_inputs.csv',index=False)
    result={'identity':declaration['identity'],'status':'complete','analysis_status':'Post hoc, outcome-exposed exploratory numerical sensitivity.',
        'scope':'Internal primary candidate only; all15 saved calibration components, same normal roles and outer predictions.',
        'forecasts_refit':False,'core_modules_written':False,'sources':len(baseline),'components':len(ledger),
        'floors':list(FLOORS),'results':records,'software_test':software,'baseline_replay':True,
        'protocol_sha256':sha(protocol_path),'review_sha256':sha(review_path),'script_sha256':sha(__file__),
        'inputs':declaration['input_sha256'],'seconds':time.monotonic()-started,
        'limitations':['Exploratory sensitivity chosen after outcomes were known; no primary result is replaced.',
            'Saved source-level errors and motion estimates are fixed; only nuisance regression and normal empirical tails are refitted.',
            'The floor is an additive logarithm stabilizer rather than a clipping threshold, so even large values change slightly.',
            'Repeated source/role rows are not independent observations. Reported affected-row counts describe computation only.',
            'Intervals condition on the saved forecasts and fixed fold memberships. External calibration is outside this sensitivity.']}
    write_json(output/'results.json',result)
    write_json(output/'artifact_hashes.json',{p.name:sha(p) for p in output.glob('*') if p.is_file()})
    assert ev.FLOOR==1e-8
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--protocol',type=Path,default=PROTOCOL);parser.add_argument('--review',type=Path,default=REVIEW)
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2))
    else:
        result=run(args.output,args.protocol,args.review)
        print(json.dumps({key:result[key] for key in ['identity','status','seconds']},indent=2))
        for row in result['results']:print(json.dumps({k:row[k] for k in ['floor','max_abs_score_change_vs_1e8','ranking_agreement_vs_1e8']}))
