"""Frozen GAVD-to-external video transfer with participant-first scoring.

This code fits nothing. It preserves the order: observed coordinate errors,
window-to-recording-to-participant means, normal-reference tail transformation,
then averaging the fifteen fixed GAVD calibration components.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits

from .data import sha256_file
from .settings import PROJECT
from .v2_models import causal_features, predict_forecaster, TARGET_JOINTS
from . import v2_evaluation as ev


def write_json(path, value):
    def serial(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=serial)+'\n')


def participant_errors(meta, truth, valid, prediction, horizon=8):
    """Use common targets, equal legs, then equal recordings per participant."""
    truth,valid,prediction=np.asarray(truth),np.asarray(valid),np.asarray(prediction)
    if truth.shape[1:]!=(24,16,2) or valid.shape!=truth.shape[:-1] or prediction.shape!=(len(truth),8,8,2):
        raise ValueError('Incorrect observation/prediction array shapes')
    if not {'source_id','sequence_id','component_id','participant_id','label'}.issubset(meta):
        raise ValueError('External metadata must retain participants and recordings')
    if not meta.source_id.eq(meta.participant_id).all() or not meta.component_id.eq(meta.participant_id).all():
        raise ValueError('External scoring groups must be dataset-coded participants')
    known=valid[:,:16,TARGET_JOINTS].any(1)
    target=truth[:,16:16+horizon,TARGET_JOINTS]
    mask=valid[:,16:16+horizon,TARGET_JOINTS]&known[:,None]
    per_leg=[]
    for start in (0,4):
        m=np.repeat(mask[:,:,start:start+4,None],2,axis=-1)
        count=m.sum((1,2,3))
        if np.any(count==0):raise ValueError('No observed outcome for one target leg')
        square=np.where(m,(target[:,:,start:start+4]-prediction[:,:horizon,start:start+4])**2,0.)
        per_leg.append(square.sum((1,2,3))/count)
    return ev.aggregate_windows(meta,np.mean(per_leg,axis=0),name='mse')


def average_components(frames, columns):
    """Average component scores after participant aggregation and calibration."""
    if len(frames)!=15:
        raise ValueError('All fifteen frozen GAVD calibration components are required')
    ids=set(frames[0].source_id)
    indexed=[]
    for frame in frames:
        if frame.source_id.duplicated().any() or set(frame.source_id)!=ids:
            raise ValueError('Component scores must have identical unique participants')
        indexed.append(frame.set_index('source_id').loc[sorted(ids),list(columns)].to_numpy())
    values=np.mean(indexed,axis=0)
    result=frames[0].drop_duplicates('source_id').set_index('source_id').loc[sorted(ids),['label','component_id']].reset_index()
    result[list(columns)]=values
    return result


def score_frozen_component(calibrator, participant_mse, covariates):
    frame=participant_mse.merge(covariates[['source_id','prefix_motion','prefix_missingness']],on='source_id',validate='one_to_one')
    scored=ev.score_source_calibrator(calibrator,frame)
    scored['conditional_one_sided']=scored.cdf
    marginal=ev.empirical_mid_cdf(
        np.log(np.array([r['mse'] for r in calibrator['calibration_source_rows']])+ev.FLOOR),
        np.log(scored.mse.to_numpy()+ev.FLOOR))
    scored['marginal_two_sided']=np.abs(2*marginal-1)
    return scored


def transfer(core_dir,external_dir,output_dir):
    """Evaluate one locked candidate without fitting target-domain parameters."""
    core,external,output=map(Path,(core_dir,external_dir,output_dir))
    if output.exists():raise FileExistsError('External evaluations require a new output identity')
    # Imports occur at execution so data preparation and a frozen fit can finish
    # independently. No import of this module invokes extraction or evaluation.
    from .v2_external import load_external
    x,valid,meta,times=load_external(external)
    for name,expected in json.loads((core/'artifact_hashes.json').read_text()).items():
        if sha256_file(core/name)!=expected:raise ValueError('Core fit hash mismatch: '+name)
    model_sets={}
    for family,name in [('primary','calibration_models.json'),('posture','posture_calibration_models.json')]:
        records=json.loads((core/name).read_text())
        if len(records)!=15:raise ValueError('Expected five folds times three rotations')
        # The runner uses explicit checkpoint names. Normalize that documented
        # schema here, without discovering or selecting alternative model files.
        records=[dict(item,fold=item['outer_fold'],forecaster=item['forecast_checkpoint'],
                      calibrator=item['calibrator_path']) for item in records]
        if len({(m['fold'],m['rotation']) for m in records})!=15:
            raise ValueError('Duplicated calibration component')
        model_sets[family]=records
    output.mkdir(parents=True)
    covariates=ev.source_covariates(x,valid,times,meta)
    _,anchor,_=causal_features(x[:,:16],valid[:,:16],times[:,:16])
    persistence=np.broadcast_to(anchor[:,None],(len(x),8,8,2))
    persistent_errors=participant_errors(meta,x,valid,persistence)
    predictions={};scored_by_family={'primary':[],'posture':[]};shifts=[];provenance=[]
    for family,item in [(family,item) for family,records in model_sets.items() for item in records]:
        checkpoint=joblib.load(core/item['forecaster'])
        calibrator=json.loads((core/item['calibrator']).read_text())
        expected_mode='ridge_posture_velocity' if family=='primary' else 'ridge_posture'
        if checkpoint['config']['mode']!=expected_mode:
            raise ValueError('External candidate family differs from the locked primary')
        if set(checkpoint['training_source_ids'])!=set(calibrator['schedule']['predictor_fit_sources']):
            raise ValueError('Forecast and calibrator source memberships differ')
        if set(meta.source_id)&(set(checkpoint['training_source_ids'])|set(calibrator['calibration_sources'])):
            raise ValueError('External participant identifier overlaps source training/reference IDs')
        pred=predict_forecaster(checkpoint,x[:,:16],valid[:,:16],times[:,:16])
        label=f"{family}_fold{item['fold']}_rotation{item['rotation']}"
        predictions[label]=pred
        error=participant_errors(meta,x,valid,pred)
        table=score_frozen_component(calibrator,error,covariates)
        table['component']=label
        scored_by_family[family].append(table)
        z=np.column_stack([np.log(table.prefix_motion.to_numpy()+ev.FLOOR),table.prefix_missingness])
        low=np.asarray(calibrator['nuisance_model']['covariate_min'])
        high=np.asarray(calibrator['nuisance_model']['covariate_max'])
        calres=np.array(calibrator['calibration_residuals'])
        shifts.append({'component':label,'normal_training_covariate_min':low.tolist(),
            'normal_training_covariate_max':high.tolist(),'external_participants':len(table),
            'outside_any_training_covariate_range':int(np.any((z<low)|(z>high),axis=1).sum()),
            'below_motion_range':int((z[:,0]<low[0]).sum()),'above_motion_range':int((z[:,0]>high[0]).sum()),
            'below_reference_residual':int((table.residual<calres.min()).sum()),
            'above_reference_residual':int((table.residual>calres.max()).sum()),
            'reference_sources':len(calres)})
        provenance.append({'component':label,'forecaster':item['forecaster'],
            'forecaster_sha256':sha256_file(core/item['forecaster']),
            'calibrator':item['calibrator'],'calibrator_sha256':sha256_file(core/item['calibrator']),
            'training_sources':checkpoint['training_source_ids'],
            'calibration_sources':calibrator['calibration_sources']})
    scores=average_components(scored_by_family['primary'],['two_sided','conditional_one_sided','marginal_two_sided','mse'])
    posture=average_components(scored_by_family['posture'],['two_sided','mse']).set_index('source_id')
    scores['posture_two_sided']=scores.source_id.map(posture.two_sided)
    scores['posture_mse']=scores.source_id.map(posture.mse)
    scores['low_motion']=-scores.source_id.map(persistent_errors.set_index('source_id').mse)
    scores['persistence_mse']=-scores.low_motion
    scores['excess_error']=scores.mse-scores.persistence_mse
    scores['prefix_motion']=scores.source_id.map(covariates.set_index('source_id').prefix_motion)
    scores['prefix_missingness']=scores.source_id.map(covariates.set_index('source_id').prefix_missingness)
    scores['low_prefix_motion']=-scores.prefix_motion
    # One single external evaluation population, not cross-validation folds.
    scores['outer_fold']=0
    names=['two_sided','posture_two_sided','low_motion','conditional_one_sided','marginal_two_sided',
           'mse','excess_error','low_prefix_motion','prefix_missingness']
    primary=ev.paired_within_fold_bootstrap(scores,'two_sided','low_motion',reps=5000,seed=5703)
    primary['unit']='dataset-coded external participant'
    primary['resampling']='paired participant resampling within normal/abnormal strata; single external cohort'
    posture_comparison=ev.paired_within_fold_bootstrap(scores,'two_sided','posture_two_sided',reps=5000,seed=5703)
    posture_comparison['unit']='dataset-coded external participant'
    posture_comparison['resampling']=primary['resampling']
    summaries={name:ev.pooled_rank_summary(scores,name,reps=2000,seed=5705) for name in names}
    conditions=[]
    for condition in sorted(set(scores.label)-{'normal'}):
        part=scores[scores.label.isin(['normal',condition])]
        contrast=ev.paired_within_fold_bootstrap(part,'two_sided','low_motion',reps=5000,seed=5703)
        conditions.append({'label':condition,'normal':int(part.label.eq('normal').sum()),
            'abnormal':int(part.label.ne('normal').sum()),'paired_comparison':contrast,
            'scores':{name:{'auroc':float(roc_auc_score(part.label.ne('normal'),part[name])),
                'average_precision':float(average_precision_score(part.label.ne('normal'),part[name]))} for name in names}})
    all_components=pd.concat([table.assign(family=family) for family,tables in scored_by_family.items() for table in tables],ignore_index=True)
    all_components.to_csv(output/'participant_component_scores.csv',index=False)
    scores.to_csv(output/'participant_scores.csv',index=False)
    meta.to_csv(output/'window_metadata.csv',index=False)
    covariates.to_csv(output/'participant_covariates.csv',index=False)
    np.savez_compressed(output/'component_predictions.npz',**predictions)
    write_json(output/'covariate_shift.json',shifts)
    write_json(output/'frozen_components.json',provenance)
    summary={'identity':'reference-v2-frozen-external-transfer-v1','status':'complete',
        'core_run':str(core.relative_to(PROJECT)),'external_preparation':str(external.relative_to(PROJECT)),
        'adaptation':'none; no external participant used for model fitting or calibration',
        'aggregation':'observed coordinates within leg, equal legs, windows per recording, recordings per participant, CDF tail per component, mean15 tails',
        'windows':len(meta),'recordings':int(meta.sequence_id.nunique()),'participants':len(scores),
        'participants_by_label':scores.label.value_counts().to_dict(),'primary_anomaly':primary,
        'posture_calibration_contrast':posture_comparison,
        'anomaly_scores':summaries,'condition_results':conditions,'covariate_shift':shifts,
        'inputs':{'core_results_sha256':sha256_file(core/'results.json'),
            'external_identity_sha256':sha256_file(external/'identity.json'),'code_sha256':sha256_file(__file__)},
        'limitations':['Dataset-coded identity and labels were not independently clinically reverified.',
            'External age, clothing, camera view, reflective markers, and acquisition protocol differ from GAVD.',
            'Byte-hash checks cannot exclude reencoded GAVD overlap or repeated people.',
            'Small GAVD normal calibration sets yield coarse ranks and can saturate under domain shift.',
            'Participant-bootstrap uncertainty conditions on fixed source-trained models.',
            'Pooled abnormality combines Parkinson and osteoarthritis and is not neurological specificity.']}
    write_json(output/'results.json',summary)
    write_json(output/'artifact_hashes.json',{str(p.relative_to(output)):sha256_file(p)
        for p in output.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',type=Path,default=PROJECT/'artifacts/runs/reference-v2-core-v1')
    parser.add_argument('--external',type=Path,default=PROJECT/'artifacts/runs/reference-v2-external-v1')
    parser.add_argument('--output',type=Path,default=PROJECT/'artifacts/runs/reference-v2-transfer-v1')
    args=parser.parse_args()
    with threadpool_limits(limits=1):summary=transfer(args.core,args.external,args.output)
    print(json.dumps({k:summary[k] for k in ['status','participants','windows']},indent=2))


if __name__=='__main__':
    main()
