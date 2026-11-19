"""Generate and independently execute the progressive reference-v2 notebooks.

Existing notebooks and historical execution records are preserved. New research
fits are performed by the frozen runner; notebook 23 fits a labeled teaching
example only and never substitutes its training error for held-out evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import nbformat as nbf

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT/'artifacts/executed-reference-v2'
RECORD = PROJECT/'docs/reference_v2/notebook_execution.json'

SETUP = '''from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
candidates = [Path.cwd(), *Path.cwd().parents]
PROJECT = next((p for p in candidates if (p/'worldmodels/v2_data.py').is_file()), None)
if PROJECT is None:
    PROJECT = next(p/'experiments/aaai-worldmodels' for p in candidates if (p/'experiments/aaai-worldmodels/worldmodels/v2_data.py').is_file())
sys.path.insert(0, str(PROJECT))
REPOSITORY = PROJECT.parents[1]
from worldmodels.v2_data import load_cohort
from worldmodels import v2_evaluation as ev
from worldmodels import v2_models as models
from worldmodels.v2_runner import window_errors, verify_freeze
COHORT = PROJECT/'artifacts/runs/reference-v2-cohort-v1'
RUN = PROJECT/'artifacts/runs/reference-v2-core-v1'
TRANSFER = PROJECT/'artifacts/runs/reference-v2-transfer-v1'
FIGURES = PROJECT/'artifacts/executed-reference-v2/figures'
FIGURES.mkdir(parents=True, exist_ok=True)
USED_INPUTS={}
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record_input(path):
    path=Path(path).resolve()
    key=str(path.relative_to(REPOSITORY)) if path.is_relative_to(REPOSITORY) else str(path)
    USED_INPUTS[key]=sha(path)
def read_json(path):
    path = Path(path)
    assert path.is_file(), f'Required completed artifact is absent: {path}. Read docs/reference_v2/README.md for prerequisites.'
    record_input(path)
    return json.loads(path.read_text())
def read_csv(path, **kwargs):
    record_input(path)
    return pd.read_csv(path, **kwargs)
def artifact(name, directory=RUN):
    path = directory/name
    manifest = read_json(directory/'artifact_hashes.json')
    assert name in manifest and path.is_file(), f'Unrecorded or missing artifact: {path}'
    assert sha(path) == manifest[name], f'Artifact hash differs from the completed run: {path}'
    record_input(path)
    return path
def completed_run():
    result = read_json(artifact('results.json'))
    assert result['status']=='complete' and not result.get('test_only', False)
    return result
def save_figure(figure, name):
    figure.savefig(FIGURES/(name+'.svg'), bbox_inches='tight')
    figure.savefig(FIGURES/(name+'.png'), dpi=150, bbox_inches='tight')
freeze = verify_freeze()
record_input(PROJECT/'docs/reference_v2/protocol_freeze.json')
x, valid, metadata, actual_times = load_cohort(COHORT, verify=True)
cohort_identity=read_json(COHORT/'identity.json');record_input(COHORT/'cohort.csv')
for name in cohort_identity['prepared_inputs']:
    record_input(REPOSITORY/cohort_identity['prepared_directory']/name)
assert x.shape == (len(metadata),24,16,2) and len(metadata)==2815
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
print('Verified real observations:',len(metadata),'windows;',metadata.source_id.nunique(),'source videos.')
print('Observation units: pelvis-relative 2D coordinates divided by a prefix torso scale. Times are decoder seconds.')
'''


def cells20():
    return [
        ('''## Rebuild the annotation inventory

One source video can contribute several sequences, and each sequence can contribute overlapping windows. We count these units separately because windows from one recording do not establish independent participants. The transformed sequence files are the annotation inventory; the upstream source CSVs are a provenance copy and are excluded from this count.''', '''from worldmodels.v2_data import all_annotations
catalog = all_annotations(REPOSITORY)
USED_INPUTS.update(dict(zip(catalog.csv_path,catalog.csv_sha256)))
display(catalog.groupby('label').agg(sequences=('sequence_id','nunique'), rows=('annotation_rows','sum'), sources=('source_id','nunique')))
assert len(catalog)==1874 and catalog.annotation_rows.sum()==458116 and catalog.source_id.nunique()==348
print('Annotated normal source ceiling:',catalog.loc[catalog.label.eq('normal'),'source_id'].nunique())
'''),
        ('''## Follow the population through the gates

The study reuses all eligible observed spans from the five selected annotations. The unchanged observation and target-support gates define a common population for every model. A gate can select a different fraction of each annotation, so the table includes both retained coverage and attrition.''', '''audit = read_json(COHORT/'audit.json')
display(pd.read_csv(COHORT/'attrition_by_label.csv'))
display(metadata.groupby('label').agg(windows=('window_id','size'),sequences=('sequence_id','nunique'),sources=('source_id','nunique')))
assert audit['prepared_windows']==2818 and audit['model_windows']==2815
assert audit['reconstruction']['maximum_absolute_error']==0
print('Full-span windows were reconstructed from sequence poses without changing the frozen arrays.')
'''),
        ('''## Keep related observations together

The relation graph uses source identity and verified byte-identical footage. A copy under another local dataset folder remains the same recording. Different file hashes do not prove that the people are different. Unknown repeated participants therefore remain a generalization limit.''', '''relations = pd.read_csv(COHORT/'media_relations.csv')
copies = relations.groupby('source_id').agg(local_files=('path','size'),different_hashes=('sha256','nunique'))
display(copies.query('local_files > 1').head(12))
graph = read_json(COHORT/'relation_graph.json')
print('Verified cross-source footage edges:',graph['verified_cross_source_edges'])
mixed=catalog.groupby('source_id').label.agg(lambda values:sorted(set(values)))
display(mixed[mixed.map(len)>1].rename('annotation_labels').to_frame())
print('Current model sources/components:',metadata.source_id.nunique(),metadata.component_id.nunique())
assert metadata.groupby('component_id').outer_fold.nunique().eq(1).all()
display(metadata[['source_id','component_id','label','outer_fold']].drop_duplicates().groupby(['outer_fold','label']).size().unstack(fill_value=0))
'''),
        ('''## Check the time axis and observation support

The nominal rate is ten samples per second, but every sample is an actual decoded frame. The final forecast extends about 0.8 seconds beyond the last observed frame. Future observation support is used after prediction to score available outcomes, while prefix measurements determine the predictor's inputs.''', '''gaps = np.diff(actual_times,axis=1)
display(metadata[['sample_fps','prefix_valid_fraction','future_valid_fraction','final_horizon_seconds','maximum_grid_error_seconds']].describe())
print('Actual decoder interval range:',float(gaps.min()),float(gaps.max()),'seconds')
assert np.isfinite(actual_times).all() and (gaps>0).all()
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
fig,ax=plt.subplots(figsize=(10,4.5))
ax.imshow(valid[0].T,aspect='auto',interpolation='nearest',cmap='Greens',vmin=0,vmax=1)
ax.axvline(15.5,color='#a45d21',lw=2)
ax.set_yticks(range(16),labels=[11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32])
ax.set(xlabel='Sample index; the future begins at 16',ylabel='BlazePose joint ID',title='Natural support in the first retained window in the pinned membership table')
fig.legend(handles=[Patch(facecolor='#00441b',label='Measured joint above the confidence threshold'),
                    Patch(facecolor='#f7fcf5',edgecolor='#888888',label='Naturally missing joint'),
                    Line2D([0],[0],color='#a45d21',label='Prefix/future boundary')],
           loc='lower center',bbox_to_anchor=(.5,-.21),ncol=1,frameon=False,fontsize=10)
save_figure(fig,'20_observed_support');plt.show()
'''),
        ('''## Identify the external transfer population

The external dataset is evaluated with frozen source-trained models after a separate compatibility audit. Recording names are mapped to participant codes using their annotation context. The external population is reported separately from GAVD, and its different acquisition conditions remain visible.''', '''external = read_json(PROJECT/'docs/reference_v2/data_external_summary.json')
display(external)
display(pd.read_csv(PROJECT/'docs/reference_v2/data_external_inventory.csv').head())
print('GAVD is a heterogeneous source-video benchmark; participant and site independence are not established by its source IDs.')
'''),
    ]


def cells21():
    return [
        ('''## Reproduce the simple reference comparison

Persistence repeats the most recent measured joint positions. Its future error describes how far the observed articulation moved away from those anchors. A negative persistence error assigns a higher anomaly score to lower observed movement. That direction was identified post hoc in the preceding analysis and is locked as a baseline in this study.''', '''result = completed_run()
source = pd.read_csv(artifact('source_anomaly_scores.csv'))
np.testing.assert_allclose(source.low_motion,-source.persistence)
display(source[['source_id','label','persistence','low_motion','ridge_posture_velocity','calibrated_two_sided']].head())
display(result['primary_anomaly'])
print('Larger anomaly scores indicate more abnormal annotation; lower forecast MSE indicates a better forecast.')
'''),
        ('''## Construct nuisance covariates from the past

We measure relative 2D movement from adjacent observed joints divided by actual decoder time gaps. We also measure the fraction of missing prefix joints. Each quantity is averaged over windows, sequences, and sources. Neither quantity requires a future observation or a diagnosis.''', '''covariates = ev.source_covariates(x,valid,actual_times,metadata)
display(covariates.describe())
changed=x[:3].copy();changed[:,16:]=np.nan
changed_support=valid[:3].copy();changed_support[:,16:]=~changed_support[:,16:]
changed_times=actual_times[:3].copy();changed_times[:,16:]=np.nan
a=ev.window_covariates(x[:3],valid[:3],actual_times[:3])
b=ev.window_covariates(changed,changed_support,changed_times)
pd.testing.assert_frame_equal(a,b)
print('Future-value, future-support, and future-time changes leave prefix covariates unchanged.')
'''),
        ('''## Separate predictor fitting, nuisance fitting, and calibration

Each outer evaluation fold is kept separate. Within its normal training pool, one set of sources fits the predictor and nuisance reference, while another set calibrates residuals. The nuisance regression uses out-of-fit normal errors. The same final predictor scores both calibration and outer sources, so calibration sources never enter their own reference construction.''', '''schedule = ev.calibration_schedule(metadata,outer_fold=0)[0]
display({key:schedule[key] for key in ['outer_fold','rotation','predictor_fit_sources','calibration_sources','heldout_sources']})
assert ev.validate_calibration_schedule(schedule)
ledger=read_json(artifact('calibration_models.json'))
entry=next(row for row in ledger if row['outer_fold']==0 and row['rotation']==0)
calibrator=read_json(artifact(entry['calibrator_path']))
display(pd.DataFrame(calibrator['nuisance_reference_rows'])[['source_id','mse','prefix_motion','prefix_missingness']].head())
display(calibrator['nuisance_model'])
'''),
        ('''## Interpret a two-sided normal-reference score

The scalar residual asks whether a source is easier or harder to predict than expected from its prefix movement and missingness. The empirical reference distribution is fitted only from the held-out normal calibration sources. A two-sided score is large at either tail. It does not assume that every impaired movement must be more erratic.''', '''scores=pd.read_csv(artifact(entry['scores_path']))
replayed=ev.score_source_calibrator(calibrator,scores)
np.testing.assert_allclose(replayed.two_sided,scores.two_sided,atol=1e-12,rtol=0)
reference=np.asarray(calibrator['calibration_residuals'])
grid=np.linspace(min(reference.min(),scores.residual.min())-.2,max(reference.max(),scores.residual.max())+.2,180)
fig,axes=plt.subplots(1,2,figsize=(11,3.8))
axes[0].step(grid,ev.empirical_mid_cdf(reference,grid),where='mid',color='#197c80')
axes[0].plot(reference,ev.empirical_mid_cdf(reference,reference),'o',color='#a45d21',label='Normal calibration source')
axes[0].legend(loc='lower right',fontsize=9)
axes[0].set(xlabel='Scalar log-error residual',ylabel='Smoothed empirical cumulative rank',title='Only normal calibration residuals define the reference.')
axes[1].step(grid,abs(2*ev.empirical_mid_cdf(reference,grid)-1),where='mid',color='#197c80')
axes[1].set(xlabel='Scalar log-error residual',ylabel='Two-sided anomaly score',title='Either tail can indicate an unusual observation.')
fig.tight_layout();save_figure(fig,'21_calibration_reference');plt.show()
display(scores[['source_id','label','mse','residual','cdf','two_sided']].head())
'''),
        ('''## Read the calibration limits before choosing a threshold

Seven or eight calibration sources produce coarse reference ranks. Averaging rotations improves score resolution but creates neither new independent people nor an exact coverage guarantee. The reported endpoint is ranking; this study does not derive a clinically validated operating threshold.''', '''calibration_sizes=pd.DataFrame([{'fold':r['outer_fold'],'rotation':r['rotation'],'normal_calibration_sources':len(r['calibration_sources'])} for r in ledger])
display(calibration_sizes)
print('Generic split-conformal resolution with n reference observations is 1/(n+1).')
display(pd.DataFrame({'n':[7,8,19],'rank_resolution':[1/8,1/9,1/20]}))
print('The current cross-fitted two-sided score is not claimed to have split-conformal coverage.')
producer_path=artifact('calibrated_source_scores.csv')
producer=read_csv(producer_path).sort_values('source_id').reset_index(drop=True)
alternative_parser=read_csv(producer_path,float_precision='round_trip').sort_values('source_id').reset_index(drop=True)
audit=source.sort_values('source_id').reset_index(drop=True).copy()
audit['alternative_parser']=alternative_parser.calibrated_two_sided.to_numpy()
np.testing.assert_array_equal(audit.calibrated_two_sided,producer.calibrated_two_sided)
print('Largest score change from a different CSV parser:',float(np.max(np.abs(audit.alternative_parser-audit.calibrated_two_sided))))
display({'frozen_producer_AUROC':ev.within_fold_summary(audit,'calibrated_two_sided')['pair_weighted_auroc'],
         'alternative_parser_AUROC':ev.within_fold_summary(audit,'alternative_parser')['pair_weighted_auroc']})
print('The frozen producer reads its ensemble CSV with the default pandas parser before evaluating. Tiny numerical differences can break exact empirical-rank ties, so reproductions must retain this transport path. No score rounding or primary result is changed.')
print('OOF nuisance forecasts use fewer training sources than the final rotation predictor; calibration and evaluation share that final predictor.')
'''),
    ]


def cells22():
    return [
        ('''## Decode one real source frame through Ambient

This is a fresh image inference call using the pinned local pose weights. The large study reuses verified full-span caches. GAVD frame numbers are one-based and the OpenCV decoder index is zero-based; the conversion is made once before calling the RGB image API.''', '''import cv2
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
from worldmodels.data import JOINT_IDS
row=metadata.sort_values(['source_id','sequence_id','first_frame']).iloc[0]
media=pd.read_csv(COHORT/'source_cache_manifest.csv')
video=REPOSITORY/media.set_index('source_id').loc[row.source_id,'video_path']
weights=REPOSITORY/'data/models/pose_landmarker_lite.task'
assert sha(weights) in set(media.pose_model_sha256)
assert sha(video)==media.set_index('source_id').loc[row.source_id,'video_sha256']
record_input(weights);record_input(video)
capture=cv2.VideoCapture(str(video));capture.set(cv2.CAP_PROP_POS_FRAMES,int(row.first_frame)-1)
ok,frame=capture.read();capture.release();assert ok,'The source frame could not be decoded.'
extractor=SequenceKeypointExtractor()
try:
    pose=extractor.extract_from_image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),str(weights))
finally:
    extractor.cleanup()
fig,ax=plt.subplots(figsize=(8,4.5));ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
if pose is not None:
    points=[point for point in pose.keypoints if point.id in JOINT_IDS and point.confidence>=.5]
    ax.scatter([p.x for p in points],[p.y for p in points],color='#00b0a0',s=25)
ax.set_title('One actual source frame and the newly estimated observed joints.');ax.axis('off')
plt.show()
print('Fresh pose calls in this teaching cell: 1. This is not a new full-cohort extraction.')
'''),
        ('''## Build equal-budget corruption policies

A token contains the two coordinates of one joint at one time. We hide only naturally observed tokens. Every observed shoulder and hip used for normalization remains visible, and every joint retains its latest anchor. The four principal corruption policies and their controls hide the same rounded fraction of eligible tokens.''', '''from worldmodels.v2_masks import POLICIES,MaskConfig,make_mask
index=int(metadata.index[metadata.window_id.eq(row.window_id)][0])
prefix,observed,t=x[index,:16],valid[index,:16],actual_times[index,:16]
designs={policy:make_mask(prefix,observed,t,MaskConfig(policy),seed=27) for policy in POLICIES}
joint_names=dict(zip(JOINT_IDS,['L shoulder','R shoulder','L elbow','R elbow','L wrist','R wrist','L hip','R hip','L knee','R knee','L ankle','R ankle','L heel','R heel','L foot index','R foot index']))
display(pd.DataFrame({'BlazePose ID':JOINT_IDS,'joint':[joint_names[j] for j in JOINT_IDS]}))
display(pd.DataFrame([entry[2] for entry in designs.values()]))
counts=[int(value[0].sum()) for policy,value in designs.items() if policy!='none']
assert len(set(counts))==1
print('M0 is the uncorrupted anchor and deliberately retains more training information.')
'''),
        ('''## Inspect what is visible and what is predicted

The colored grid distinguishes natural missingness, deliberate corruption, protected observations, and fixed future queries. Purple query locations request the same distal coordinates for every method. Their existence does not depend on whether those future joints will later be observed. A block or tube may be truncated to match the exact budget.''', '''from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
policies=['uniform','time_blocks','anatomical_tubes','motion_balanced']
colors=['#dce1e5','#c6e5df','#eea04d','#468dbc','#9d89c8','#ffffff']
fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True)
for ax,policy in zip(axes.flat,policies):
    hidden,protected,audit=designs[policy]
    panel=np.full((24,16),5,int);panel[:16]=observed.astype(int)
    panel[:16][hidden]=2;panel[:16][protected]=3;panel[16:,models.TARGET_JOINTS]=4
    ax.imshow(panel.T,cmap=ListedColormap(colors),vmin=0,vmax=5,aspect='auto',interpolation='nearest')
    ax.axvline(15.5,color='#17394c',lw=1)
    ax.set(title=policy.replace('_',' ').capitalize()+f": {audit['hidden_tokens']}/{audit['eligible_tokens']} eligible tokens hidden",xlabel='Sample index; future begins at 16')
    ax.set_yticks(range(16),[f'{joint_names[j]} ({j})' for j in JOINT_IDS],fontsize=8)
labels=['Naturally missing','Visible prefix','Artificially hidden','Protected context','Fixed future query','Future non-query']
fig.legend(handles=[Patch(facecolor=c,edgecolor='#777777',label=l) for c,l in zip(colors,labels)],loc='lower center',ncol=3,frameon=False)
fig.tight_layout(rect=[0,.1,1,1]);save_figure(fig,'22_matched_mask_policies');plt.show()
'''),
        ('''## Verify a hidden observation cannot survive in filled features

Coordinates are erased before causal filling. For a fixed mask, changing the hidden coordinates cannot alter the visible feature vector or anchors. This checks the input boundary. It does not eliminate information in an adaptive mask pattern, because the policy observed movement before deciding what to hide.''', '''hidden,protected,_=designs['motion_balanced']
a,anchor,known=models.causal_features(prefix[None],observed[None],t[None],hidden=hidden[None])
changed=prefix.copy();changed[hidden|~observed]=1e20
b,other_anchor,other_known=models.causal_features(changed[None],observed[None],t[None],hidden=hidden[None])
np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(anchor,other_anchor)
np.testing.assert_array_equal(known,other_known)
assert not (hidden&protected).any() and not (hidden&~observed).any()
print('Fixed-mask hidden-value invariance passed; natural missingness never became a training target.')
'''),
        ('''## Connect the training task to the fitted experiment

Every mask uses the same backbone, future objective, normal training groups, optimizer settings, and selected number of updates. M3 is compared with random joint groups of the same sizes and time extent. M4 is compared with a motion-focused policy and a pattern-only residual diagnostic. The complete results, including unsuccessful policies, remain visible.''', '''result=completed_run()
forecast=pd.DataFrame(result['normal_forecast'])
display(forecast[forecast.arm.str.startswith('mlp_') & forecast.horizon.eq(8)][['arm','mse','ci95','normal_sources']])
principal=forecast.query('horizon==8').set_index('arm')
print('The uniform, anatomical-tube, and motion-balanced models have normal-source MSE values of',
      ', '.join(f"{principal.loc[name,'mse']:.5f}" for name in ['mlp_uniform','mlp_anatomical_tubes','mlp_motion_balanced']), 'respectively.')
print('These point estimates must be read with the paired controls, rather than interpreted as an anatomical advantage by design.')
selection=read_json(artifact('selection.json'))
durations=[{'training_sources':len(v['training_sources']),'selected_steps':v['selected_steps']} for v in selection.values() if 'selected_steps' in v]
display(pd.DataFrame(durations))
print('An anatomical masking gain requires its matched controls. A low latent loss is not the endpoint in this coordinate-only study.')
'''),
    ]


def cells23():
    return [
        ('''## Fit a small real normal-only teaching model

This cell performs a real fit on a declared subset of normal sources outside fold zero. It uses a fixed penalty and a few windows per source for a quick demonstration. The displayed training error is not held-out research evidence and does not retune the completed study.''', '''from worldmodels.v2_models import ForecastConfig,fit_forecaster,predict_forecaster
training=metadata[metadata.label.eq('normal') & metadata.outer_fold.ne(0)].groupby('source_id',sort=True).head(4).index.to_numpy()
training_meta=metadata.iloc[training].copy();training_meta['group_id']=training_meta.component_id
configuration=ForecastConfig(mode='ridge_posture_velocity',ridge_alpha=10.,seed=27)
teaching_model=fit_forecaster(x[training,:16],valid[training,:16],actual_times[training,:16],x[training,16:],valid[training,16:],training_meta,configuration)
training_prediction=predict_forecaster(teaching_model,x[training,:16],valid[training,:16],actual_times[training,:16])
teaching_error=ev.aggregate_windows(training_meta,window_errors(x[training],valid[training],training_prediction))
print('Actual fit:',len(training),'windows;',training_meta.source_id.nunique(),'normal source groups;',round(teaching_model['seconds'],3),'seconds.')
print('Didactic training MSE, not evaluation MSE:',float(teaching_error.value.mean()))
'''),
        ('''## Inspect how the compact state represents movement

The posture model receives the latest distal-joint positions. The posture-plus-velocity model adds recent median coordinate slopes, their dispersion, and observation counts. A small residual neural model receives the full observed prefix. These feature sets ask different questions about whether history adds useful information.''', '''dimensions=[]
for mode in ['ridge_posture','ridge_posture_velocity','mlp']:
    features,anchors,known=models.causal_features(x[training[:3],:16],valid[training[:3],:16],actual_times[training[:3],:16],mode=mode)
    dimensions.append({'mode':mode,'input_features':features.shape[1],'future_coordinate_queries':8*8*2})
display(pd.DataFrame(dimensions))
weights=models.hierarchical_weights(training_meta)
mass=training_meta.assign(weight=weights).groupby('source_id').weight.sum()
np.testing.assert_allclose(mass.to_numpy(),mass.iloc[0])
print('Training mass is equal across the current independent singleton source groups.')
'''),
        ('''## Inspect actual inner selection rather than choosing from anomaly results

The completed runner selects regularization from normal held-out source errors. Neural duration is chosen using the uncorrupted model and then shared across masking policies. Every inner source contributes equal weight even when validation folds have different sizes.''', '''result=completed_run();selection=read_json(artifact('selection.json'))
summary=[]
for record in selection.values():
    summary.append({'mode':record['mode'],'normal_training_sources':len(record['training_sources']),
                    'selected':record.get('selected_alpha',record.get('selected_alphas',record.get('selected_steps')))})
display(pd.DataFrame(summary).groupby('mode').agg(selection_records=('mode','size'),smallest_training_pool=('normal_training_sources','min'),largest_training_pool=('normal_training_sources','max')))
example=next(v for v in selection.values() if v['mode']=='ridge_posture_velocity')
display(example['inner_source_mse'])
assert not set(example['membership'][0]['train_sources'])&set(example['membership'][0]['validation_sources'])
'''),
        ('''## Check what was learned and what remains a separate hypothesis

The neural model has a coordinate-prediction objective in this study. Its hidden-feature variance and effective rank are measured on normal training prefixes to detect a collapsed or numerically unstable representation. An untrained-encoder coordinate probe is an additional reference. These checks do not by themselves establish clinically meaningful features.''', '''import joblib
ledger=read_json(artifact('model_fit_ledger.json'))
rows=[]
for item in ledger.values():
    identity=item['identity']
    if identity.get('kind')!='compact' or identity['config']['mode']!='mlp': continue
    if not any('outer0:mask_' in role for role in item['roles']): continue
    checkpoint=joblib.load(artifact(item['checkpoint']))
    rows.append({'policy':checkpoint['config']['mask_policy'],'seed':checkpoint['config']['seed'],
                 'parameters':checkpoint['parameter_count'],'steps':checkpoint['config']['steps'],
                 'effective_rank':checkpoint['diagnostics']['effective_rank'],
                 'mean_feature_std':checkpoint['diagnostics']['mean_feature_std']})
display(pd.DataFrame(rows))
print('JEPA/teacher, phase, learned-mask, and stochastic extensions require their own gate decision and run identity.')
'''),
        ('''## Keep inference independent of the withheld future

The public inference API receives exactly sixteen prefix samples. It always emits the same eight-time/eight-joint coordinate queries. Future values and future support are used later to measure error; they cannot enter prediction through an accidentally supplied full window.''', '''prediction=predict_forecaster(teaching_model,x[training[:2],:16],valid[training[:2],:16],actual_times[training[:2],:16])
assert prediction.shape==(2,8,8,2)
try:
    predict_forecaster(teaching_model,x[training[:2]],valid[training[:2]],actual_times[training[:2]])
except ValueError as exc:
    print('Full-window input was correctly rejected:',str(exc))
else:
    raise AssertionError('The predictor accepted future measurements.')
'''),
    ]


def cells24():
    return [
        ('''## Compare normal-source forecasts on common targets

We first ask whether the models predict normal movement more accurately. The reported MSE averages observed coordinates within each leg, then legs, windows, sequences, and sources. It covers all eight future samples. A positive paired difference means the listed method reduces error relative to its comparator.''', '''result=completed_run()
means=pd.DataFrame(result['normal_forecast'])
display(means.query('horizon==8')[['arm','mse','ci95','normal_sources']])
contrasts=pd.DataFrame(result['forecast_contrasts'])
display(contrasts[contrasts.method.isin(['ridge_posture_velocity','mlp_anatomical_tubes','mlp_motion_balanced'])][['method','reference','estimate','ci95','relative_reduction']])
mean_priors=read_json(artifact('results.json',PROJECT/'artifacts/runs/reference-v2-mean-prior-posthoc-v1'))
display(pd.DataFrame(mean_priors['comparisons'])[['method','reference','method_mse','reference_mse','estimate','ci95']])
print('The two untuned normal-mean forecast controls were declared after the core outcomes were known. Their separate post hoc identity preserves that history.')
'''),
        ('''## Recompute the primary anomaly endpoint

An anomaly score is evaluated by whether it ranks annotated abnormal sources above normal sources. The primary comparison uses only pairs scored within the same outer fold. Its average weights each fold by the number of abnormal-normal source pairs. We compare the locked calibrated detector with low motion and separately with an identically calibrated posture-only model.''', '''source=pd.read_csv(artifact('source_anomaly_scores.csv'))
primary=ev.paired_within_fold_bootstrap(source,'calibrated_two_sided','low_motion',reps=5000,seed=5701)
np.testing.assert_allclose(primary['estimate'],result['primary_anomaly']['estimate'],rtol=0,atol=1e-12)
np.testing.assert_allclose(primary['ci95'],result['primary_anomaly']['ci95'],rtol=0,atol=1e-12)
display(pd.DataFrame(primary['folds']))
display({k:primary[k] for k in ['candidate_auroc','reference_auroc','estimate','ci95','uncertainty']})
print(f"The calibrated model has within-fold AUROC {primary['candidate_auroc']:.3f}, compared with {primary['reference_auroc']:.3f} for low motion.")
print(f"The paired difference is {primary['estimate']:.3f}, with conditional interval [{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}].")
print('This is a result about incremental ranking in the declared source population. It does not measure clinical diagnostic accuracy.')
display(result['posture_calibration_contrast'])
'''),
        ('''## Retain simple alternative explanations

The current-posture, movement, and observation-quality controls are part of the result. Calibrating an error does not prove that its signal comes from coordination. Pooled rankings also compare scores across separately fitted folds and therefore need a comparability assumption that within-fold pairs avoid.''', '''ranking=[]
for row in result['rankings']:
    if row['score'] not in ['calibrated_two_sided','posture_two_sided','low_motion','posture_distance','motion_quality_distance','negative_prefix_motion','high_prefix_missingness']: continue
    ranking.append({'score':row['score'],'within_fold_AUROC':row['within_fold']['pair_weighted_auroc'],
                    'pooled_AUROC':row['pooled']['auroc'],'average_precision':row['pooled']['average_precision'],
                    'prevalence':row['pooled']['prevalence']})
display(pd.DataFrame(ranking))
display(pd.read_csv(artifact('condition_scores.csv')).query("score in ['calibrated_two_sided','posture_two_sided','low_motion']"))
'''),
        ('''## Read the external population separately

The frozen source-trained model is applied to a different dataset without target-domain fitting or calibration. Errors and prefix covariates are aggregated within recordings and then within participant before each normal-reference transform. The fifteen frozen component scores are then averaged. This differs from averaging scores from individual short windows.''', '''transfer=read_json(artifact('results.json',TRANSFER))
assert transfer['status']=='complete'
participants=pd.read_csv(artifact('participant_scores.csv',TRANSFER))
display(transfer['participants_by_label'])
print('External windows, recordings, participants:',transfer['windows'],transfer['recordings'],transfer['participants'])
display(transfer['primary_anomaly'])
display(transfer['posture_calibration_contrast'])
external_contrast=transfer['primary_anomaly']
print(f"On the external participant-coded population, the calibrated and low-motion AUROCs are {external_contrast['candidate_auroc']:.3f} and {external_contrast['reference_auroc']:.3f}.")
print('Movement amount may itself contain clinically relevant information; conditioning on it does not automatically create a better detector.')
print('The external outer_fold=0 marker denotes one external population, not a new cross-validation fit.')
assert participants.source_id.nunique()==len(participants)
'''),
        ('''## Separate uncertainty from new independent evidence

The source and participant bootstrap intervals condition on saved fitted models. Two additional normal-source partitions measure sensitivity of compact forecast fits to training membership. Those correlated partitions and the three optimization seeds do not increase the number of participants.''', '''retraining=read_json(artifact('retraining_summary.json'))
display(retraining)
print('Intervals describe saved-fit source variation; repeated-partition changes describe an additional, separately reported fitting sensitivity.')
print('Neither internal source grouping nor a different dataset name proves that every evaluated person is independent.')
'''),
    ]


def cells25():
    return [
        ('''## Check the fixed observation perturbations

The robustness analysis changes declared prefix observations while holding future targets fixed. It measures sensitivity to observation loss, noise, or time-order changes. These are information controls and do not simulate a neurological lesion or treatment.''', '''result=completed_run()
ROBUSTNESS=PROJECT/'artifacts/runs/reference-v2-robustness-v1'
robustness=read_json(artifact('results.json',ROBUSTNESS))
display(pd.DataFrame(robustness['normal_forecast_contrasts']))
print('Read each perturbation together with its retained support and definition. A changed input asks a different information-access question.')
'''),
        ('''## Compare recording views and observation quality

Camera view changes projected articulation. Pose confidence and missingness can also alter which motion is visible. View or quality slices must therefore show source counts and their changed population. A sparse slice cannot establish the same conclusion as the complete cohort.''', '''source=pd.read_csv(artifact('source_anomaly_scores.csv'))
display(pd.read_csv(COHORT/'coverage_by_view.csv'))
display(robustness['nuisance_sensitivity'])
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
for label,color in [('normal','#197c80'),('other','#ad632d')]:
    keep=source.label.eq('normal') if label=='normal' else source.label.ne('normal')
    name=('Normal annotation' if label=='normal' else 'Abnormal annotations')+f' ({int(keep.sum())} source videos)'
    axes[0].scatter(source.loc[keep,'prefix_motion'],source.loc[keep,'calibrated_two_sided'],color=color,label=name,alpha=.75)
    axes[1].scatter(source.loc[keep,'prefix_missingness'],source.loc[keep,'calibrated_two_sided'],color=color,alpha=.75)
axes[0].set(xlabel='Observed prefix motion, normalized coordinates per second',ylabel='Calibrated two-sided score',title='Motion remains a competing explanation.')
axes[1].set(xlabel='Prefix missingness fraction',ylabel='Calibrated two-sided score',title='Observation quality remains a competing explanation.')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
fig.tight_layout(rect=[0,.12,1,1]);save_figure(fig,'25_motion_quality_controls');plt.show()
'''),
        ('''## Inspect external reference extrapolation

The nuisance reference was fitted on a small normal-source population. Different acquisition conditions can place external observations outside that population's motion or missingness range. Scores at an empirical-reference endpoint are coarse ranks, not evidence of a clinically calibrated probability.''', '''transfer=read_json(artifact('results.json',TRANSFER))
display(transfer['covariate_shift'])
participants=pd.read_csv(artifact('participant_scores.csv',TRANSFER))
display(participants.groupby('label').agg(participants=('source_id','size'),motion_median=('prefix_motion','median'),missingness_median=('prefix_missingness','median')))
print('The frozen reference is reported without fitting a new transform to improve the external labels.')
'''),
        ('''## Connect physiology to observations we actually measured

Bilateral timing, muscle coordination, and compensatory movement motivate useful questions. Our measurements are projected joint positions and their forecasts. We therefore test movement relationships and alternative explanations, while treating the physiological cause of an individual score as unmeasured.''', '''display(pd.DataFrame([
    {'motivation':'Bilateral timing may be altered.','observed_test':'Compare aligned and shifted histories.','limit':'A timing perturbation is not a neural intervention.'},
    {'motivation':'Compensation can be repeatable.','observed_test':'Compare raw error, low motion, and both normal-reference tails.','limit':'Easy prediction does not establish healthy movement.'},
    {'motivation':'Several joints move together.','observed_test':'Compare anatomical tubes with matched random joint groups.','limit':'An anatomical mask is not a measurement of muscle synergies.'},
    {'motivation':'Movement differs across clinical annotations.','observed_test':'Show condition profiles and shared normal references.','limit':'Folder labels do not independently confirm diagnosis.'},
]))
print('Reliable phase, learned-mask, latent-objective, and stochastic extensions require separately declared gate decisions.')
display(robustness['source_identity_probe'])
'''),
        ('''## Read every limitation at the correct unit

Many windows improve the description of a recording, but they do not create more independent people. Condition slices share normal references; altered observation support changes the population. The internal cohort and external transfer answer complementary questions, and neither alone supplies a treatment-response model.''', '''display(result['limitations'])
print('Internal sources:',result['sources'],'retained normal sources:',result['normal_sources'])
print('The output README states the remaining participant, site, label, and clinical validation requirements.')
'''),
    ]


def cells26():
    return [
        ('''## Lead with gait sequences and explain the clinical-video scale

GAVD was introduced as the largest publicly accessible collection of abnormal-gait videos annotated by expert clinicians. That is an attributed claim about a video collection, rather than a current ranking across all gait-measurement technologies. Our benchmark evaluates 456 unique annotated gait sequences, including 141 normal sequences, through 2,815 forecast windows. These sequences are movement observations. Their 91 original recordings define the groups that stay together in evaluation, and recording identifiers do not establish patient identity.

A manuscript claim needs a measured result and an exact input identity. We read the primary-source scale review, the annotation audit, and the new reporting ledger so the presentation can explain its observation count without treating repeated observations as independent people.''', '''result=completed_run()
manuscript=PROJECT/'docs/draft03'
scale_review=read_json(manuscript/'reviewer_support/literature_scale.json')
sequence_audit=read_json(manuscript/'reviewer_support/sequence_audit.json')
statistical_review=read_json(manuscript/'reviewer_support/statistical_review.json')
normal_metadata=metadata.loc[metadata.label.eq('normal')]
counts={'unique_gait_sequences':metadata.sequence_id.nunique(),'normal_gait_sequences':normal_metadata.sequence_id.nunique(),
        'forecast_windows':len(metadata),'recording_groups':metadata.source_id.nunique(),
        'normal_recording_groups':normal_metadata.source_id.nunique()}
assert counts=={'unique_gait_sequences':456,'normal_gait_sequences':141,'forecast_windows':2815,
                'recording_groups':91,'normal_recording_groups':29}
display(pd.DataFrame.from_dict(counts,orient='index',columns=['count']))
print('Primary-source scale review:',scale_review['identity'])
print('Annotation audit:',sequence_audit['analysis_identity'])
print('Statistical review:',statistical_review['identity'])
claims=read_json(manuscript/'claim_ledger.json')
assert isinstance(claims,dict) and isinstance(claims['claims'],list)
assert claims['identity']=='medwm-reference-v2-draft03-evidence-v3','The revised reporting evidence is required.'
print('Manuscript evidence identity:',claims['identity'])
display(pd.DataFrame(claims['claims'])[['id','claim','artifact','scope']])
print('The preserved previous manuscript is docs/draft02. This notebook reads only the new draft03.')
'''),
        ('''## Separate sequence coverage from the uncertainty unit

Several gait sequences can come from one recording, and some different sequence identifiers share video frames. The retained audit finds 17 overlapping sequence pairs. A distinct identifier or a single-direction walking annotation does not establish an independent participant. The annotation contract describes walking in one camera-relative direction; this study has not visually reverified turns in every sequence. The original primary forecast mean gives each normal recording group equal weight. A separately declared post hoc sensitivity gives each normal sequence equal weight while still resampling whole recording groups. It changes the weighting question, not the observations, fitted model, primary endpoint, or number of independent groups.

We show both forecast summaries with their identities and retain the original anomaly comparison against low motion. Forecast improvement and abnormality ranking remain separate questions.''', '''source=pd.read_csv(artifact('source_anomaly_scores.csv'))
assert result['windows']==len(metadata) and result['sources']==metadata.source_id.nunique()
assert result['normal_sources']==metadata.loc[metadata.label.eq('normal'),'source_id'].nunique()
np.testing.assert_allclose(source.low_motion,-source.persistence)
computed=ev.within_fold_summary(source,'calibrated_two_sided')['pair_weighted_auroc']
np.testing.assert_allclose(computed,result['primary_anomaly']['candidate_auroc'],atol=1e-12,rtol=0)
display(pd.DataFrame(result['forecast_contrasts']).query("method=='ridge_posture_velocity'"))
display({name:result[name] for name in ['primary_anomaly','posture_calibration_contrast']})
overlap=sequence_audit['populations']['retained_sequences']['raw_annotation_overlap']
assert overlap['unordered_same_source_sequence_pairs_sharing_exact_annotation_frames']==17
display({'different_sequence_pairs_sharing_frames':17,'participating_sequences':overlap['participating_sequences'],
         'fold_grouping':sequence_audit['fold_integrity']})
SEQUENCE_SCALE=PROJECT/'artifacts/runs/reference-v2-sequence-scale-posthoc-v2'
sequence_sensitivity=read_json(artifact('results.json',SEQUENCE_SCALE))
assert sequence_sensitivity['identity']=='reference-v2-sequence-scale-posthoc-v2' and sequence_sensitivity['status']=='complete'
assert sequence_sensitivity['forecast_refit'] is False
sequence_errors=pd.read_csv(artifact('per_sequence_errors.csv',SEQUENCE_SCALE))
assert sequence_errors.sequence_id.is_unique and len(sequence_errors)==456
membership_columns=['source_id','sequence_id','label','component_id','outer_fold']
expected_membership=metadata[membership_columns].drop_duplicates().sort_values('sequence_id').reset_index(drop=True)
actual_membership=sequence_errors[membership_columns].sort_values('sequence_id').reset_index(drop=True)
pd.testing.assert_frame_equal(actual_membership,expected_membership,check_dtype=False)
normal_sequence_errors=sequence_errors.loc[sequence_errors.label.eq('normal')]
assert len(normal_sequence_errors)==141 and normal_sequence_errors.component_id.nunique()==29
for arm in ['persistence','ridge_posture']:
    np.testing.assert_allclose(normal_sequence_errors[arm].mean(),sequence_sensitivity['sequence_weighted_mse'][arm],atol=1e-12,rtol=0)
primary_forecast=next(row for row in result['forecast_contrasts'] if row['method']=='ridge_posture' and row['reference']=='persistence')
np.testing.assert_allclose(sequence_sensitivity['recording_weighted_primary']['relative_reduction'],primary_forecast['relative_reduction'],atol=1e-12,rtol=0)
display(pd.DataFrame([
    {'weighting':'Equal original recording groups','status':'Original primary forecast endpoint','relative_reduction_percent':100*primary_forecast['relative_reduction'],
     'conditional_interval_percent':(100*np.asarray(primary_forecast['relative_ci95'])).tolist(),'normal_sequences':141,'resampled_recording_groups':29},
    {'weighting':'Equal annotated gait sequences','status':'Separate post hoc weighting sensitivity','relative_reduction_percent':100*sequence_sensitivity['relative_reduction'],
     'conditional_interval_percent':(100*np.asarray(sequence_sensitivity['relative_ci95'])).tolist(),'normal_sequences':141,'resampled_recording_groups':29},
]))
print('Sensitivity identity:',sequence_sensitivity['identity'])
print('Uncertainty resamples:',sequence_sensitivity['resampling'])
print('The equal-sequence sensitivity is not a new cohort or a replacement for the original recording-weighted primary result.')
'''),
        ('''## Rebuild every manuscript table from its source results

The publication tables are formatted views of the completed runs. We reconstruct all six tables from source-level summaries and cohort memberships, then compare every cell with the manuscript CSV. The renderer supplies names and decimal formatting, while the numerical values come from the declared evidence. This computation writes only into the notebook's own output area.''', '''import importlib.util
renderer_path=PROJECT/'scripts/build_next_draft03.py';record_input(renderer_path)
sys.path.insert(0,str(PROJECT/'scripts'))
specification=importlib.util.spec_from_file_location('medwm_publication_renderer',renderer_path)
renderer=importlib.util.module_from_spec(specification);specification.loader.exec_module(renderer)
transfer=read_json(artifact('results.json',TRANSFER))
external_windows=pd.read_csv(artifact('window_metadata.csv',TRANSFER))
mean_priors=read_json(artifact('results.json',PROJECT/'artifacts/runs/reference-v2-mean-prior-posthoc-v1'))
forecasts={row['arm']:row for row in result['normal_forecast'] if row['horizon']==8}
for row in mean_priors['comparisons']:
    if row['reference']=='persistence':
        forecasts[row['method']]={'mse':row['method_mse'],'ci95':row['method_ci95']}
ranks={row['score']:row for row in result['rankings']}
forecast_arms=['persistence','damped_velocity','own','bilateral','static','ridge_posture','ridge_posture_velocity','untrained_probe','mean_pose','mean_residual']
anomaly_arms=['calibrated_two_sided','posture_two_sided','low_motion','one_sided','marginal_two_sided','posture_distance','motion_quality_distance','negative_prefix_motion','high_prefix_missingness','excess_bilateral','excess_ridge_posture_velocity']
def forecast_row(arm):
    row=forecasts[arm]
    return [renderer.name(arm),renderer.fmt(row['mse']*1000,2),renderer.interval(np.array(row['ci95'])*1000,2)]
table_rows={
    'cohort':[[label,str(group.sequence_id.nunique()),str(len(group)),str(group.source_id.nunique())] for label,group in metadata.groupby('label')],
    'external_cohort':[[label,str(group.participant_id.nunique()),str(group.recording_id.nunique()),str(len(group))] for label,group in external_windows.groupby('label')],
    'forecasts':[forecast_row(arm) for arm in forecast_arms],
    'complete_forecasts':[forecast_row(arm) for arm in forecasts],
    'rankings':[[renderer.SCORES[arm],renderer.fmt(ranks[arm]['within_fold']['pair_weighted_auroc']),renderer.fmt(ranks[arm]['pooled']['auroc']),renderer.fmt(ranks[arm]['pooled']['average_precision'])] for arm in anomaly_arms],
    'external_rankings':[[renderer.name(arm),renderer.fmt(row['auroc']),renderer.interval(row['auroc_ci95']),renderer.fmt(row['average_precision'])] for arm,row in transfer['anomaly_scores'].items()],
}
rebuild=PROJECT/'artifacts/executed-reference-v2/manuscript_rebuild';rebuild.mkdir(parents=True,exist_ok=True)
table_checks=[]
for name,rows in table_rows.items():
    published=pd.read_csv(manuscript/(name+'.csv'),dtype=str)
    if name=='cohort':
        assert published.columns.tolist()==['GAVD annotation','Sequences','Windows','Recordings']
    rebuilt=pd.DataFrame(rows,columns=published.columns).astype(str)
    pd.testing.assert_frame_equal(rebuilt,published)
    rebuilt.to_csv(rebuild/(name+'.csv'),index=False)
    table_checks.append({'table':name,'rows':len(rebuilt),'all_cells_match':True})
display(pd.DataFrame(table_checks))
previous_rankings=manuscript/'revisions/before_score_figure_v3/rankings.csv'
record_input(previous_rankings)
assert len(table_rows['rankings'])==11 and sha(manuscript/'rankings.csv')==sha(previous_rankings)
print('The complete 11-score table retains all values and metrics unchanged in the appendix. The main comparison is now explained by a two-method chart.')
'''),
        ('''## Keep the evidence states visible

Successful software checks, fitted models, evaluated predictions, and proposed methods are different accomplishments. The execution ledger records what actually ran. We retain an exact copy of the coordinator's ledger at execution time because its administrative acceptance status can change afterward. A latent or stochastic model must not appear as a measured contribution merely because its illustration or mathematical idea was described.''', '''ledger_bytes=(PROJECT/'docs/reference_v2/run_record.json').read_bytes()
ledger_digest=hashlib.sha256(ledger_bytes).hexdigest()
ledger_snapshot=PROJECT/'artifacts/executed-reference-v2/inputs'/f'coordinator_ledger_{ledger_digest}.json'
ledger_snapshot.parent.mkdir(parents=True,exist_ok=True)
if ledger_snapshot.exists():
    assert ledger_snapshot.read_bytes()==ledger_bytes
else:
    ledger_snapshot.write_bytes(ledger_bytes)
record=read_json(ledger_snapshot)
print('Exact coordinator ledger snapshot:',ledger_digest)
request=read_json(artifact('request.json'));configuration=request['config']
ledger=read_json(artifact('model_fit_ledger.json'))
available=set(pd.read_csv(artifact('refs_source_scores.csv')).arm) | set(pd.read_csv(artifact('masks_source_scores.csv')).arm)
registry=[]
references=['persistence','damped_velocity','own','bilateral','shifted','static','ridge_posture','ridge_posture_velocity']
if configuration['untrained_probe']: references.append('untrained_probe')
for name in references:
    analytic=name in ['persistence','damped_velocity']
    registry.append({'candidate':name,'family':'reference','declared':True,'fitting':'analytic reference' if analytic else 'fitted normal-only model',
                     'status':'evaluated' if name in available else 'MISSING from completed evaluation; inspect run ledger'})
for policy in configuration['policies']:
    for seed in configuration['seeds']:
        name=f'mlp_{policy}_seed{seed}'
        roles=[f'outer{fold}:mask_{policy}:seed{seed}' for fold in range(5)]
        fitted=sum(any(role in item['roles'] for item in ledger.values()) for role in roles)
        registry.append({'candidate':name,'family':'mask policy and optimization seed','declared':True,'fitting':f'{fitted}/5 outer refits recorded',
                         'status':'evaluated' if name in available else 'MISSING from completed evaluation; inspect run ledger'})
    name=f'mlp_{policy}'
    registry.append({'candidate':name,'family':'fixed seed-mean prediction','declared':True,'fitting':'mean of the three declared coordinate predictions',
                     'status':'evaluated' if name in available else 'MISSING from completed evaluation; inspect run ledger'})
if 'motion_balanced' in configuration['policies']:
    registry.append({'candidate':'mlp_pattern_only','family':'mask-pattern residual diagnostic','declared':True,'fitting':'normal-only seed 27',
                     'status':'evaluated' if 'mlp_pattern_only' in available else 'MISSING from completed evaluation; inspect run ledger'})
for name in ['calibration_models.json','posture_calibration_models.json']:
    models_record=read_json(artifact(name))
    registry.append({'candidate':name,'family':'normal-reference calibration ensemble','declared':True,'fitting':f'{len(models_record)}/15 components recorded',
                     'status':'evaluated' if len(models_record)==15 else 'INCOMPLETE; inspect run ledger'})
mean_priors=read_json(artifact('results.json',PROJECT/'artifacts/runs/reference-v2-mean-prior-posthoc-v1'))
for name in ['mean_pose','mean_residual']:
    registry.append({'candidate':name,'family':'separately declared post hoc mean-prior control','declared':True,
                     'fitting':'five untuned normal-only outer fits','status':mean_priors['status']})
with pd.option_context('display.max_rows',None,'display.max_colwidth',None):
    display(pd.DataFrame(registry))
    display(pd.DataFrame([{'score':item['score'],'status':'evaluated with its declared direction',
                           'within_fold_AUROC':item['within_fold']['pair_weighted_auroc']}
                          for item in result['rankings']]))
floor_sensitivity=read_json(artifact('results.json',PROJECT/'artifacts/runs/reference-v2-floor-posthoc-v2'))
display({'separate_posthoc_analysis':floor_sensitivity['identity'],'status':floor_sensitivity['status'],
         'frozen_floors':floor_sensitivity['floors'],'forecast_refit':floor_sensitivity['forecasts_refit']})
display(pd.DataFrame([{'floor':row['floor'],'maximum_score_change':row['max_abs_score_change_vs_1e8'],
                       'changed_rank_pairs':row['ranking_agreement_vs_1e8']['changed_pair_directions'],
                       'within_fold_AUROC':row['primary_anomaly']['candidate_auroc']}
                      for row in floor_sensitivity['results']]))
print(floor_sensitivity['serialization'])
print('The failed first floor replay is preserved under its original identity. Its guard exposed an omitted CSV round trip; the revised sensitivity reproduces the exact producer path without altering the frozen scores.')
failed_floor=read_json(PROJECT/'artifacts/runs/reference-v2-floor-posthoc-v1/failure.json')
display({'preserved_attempt':failed_floor['identity'],'status':failed_floor['status'],'cause':failed_floor['cause']})
display({'separate_posthoc_analysis':sequence_sensitivity['identity'],'status':sequence_sensitivity['status'],
         'forecast_refit':sequence_sensitivity['forecast_refit'],'primary_recording_weighted_endpoint':'preserved'})
assert 'extension_gates' in record,'The final run ledger must state the extension-gate decisions.'
gates=record['extension_gates']
display(pd.DataFrame.from_dict(gates,orient='index') if isinstance(gates,dict) else pd.DataFrame(gates))
display(record)
print('Notebook 23 includes an explicitly labeled real training demonstration. Its training error is absent from the held-out result table.')
print('Read the resource/extension gate record before describing deferred models as implemented or evaluated.')
'''),
        ('''## Verify the manuscript evidence and inspect its figures

Draft03 has editable prose and a dedicated generator. Its evidence record links numerical tables and figures to completed runs. The main anomaly chart tests whether a learned forecast with a normal reference adds ranking value beyond measuring movement alone. It shows two clearly defined methods, internal and external panels, one AUROC metric, and a chance reference at 0.5. All eleven score definitions and their additional metrics remain available in the unchanged appendix table.

Both plotted methods use the observed future after it arrives. They are not prefix-only early warnings. The forecasting model still predicts from the past alone; future measurements enter when the resulting score is computed. Read the uncertainty intervals at the recording-group or coded-participant level, and keep the full controls beside the chart when interpreting its result.''', '''evidence=read_json(manuscript/'evidence.json')
print('Evidence identity:',evidence['identity'])
assert evidence['identity']==claims['identity']=='medwm-reference-v2-draft03-evidence-v3'
assert evidence['method_identity']=='medwm-reference-masking-v2'
assert evidence['run_identities']['sequence_weighting']=='artifacts/runs/reference-v2-sequence-scale-posthoc-v2'
assert evidence['counts']['gavd_sequences']==456 and evidence['counts']['gavd_normal_sequences']==141
for input_name,expected_hash in evidence['inputs'].items():
    declared_input=PROJECT/input_name
    assert sha(declared_input)==expected_hash,f'A manuscript evidence input changed: {input_name}'
    record_input(declared_input)
display(pd.DataFrame.from_dict(evidence['counts'],orient='index',columns=['count']))
display(evidence['displayed_values'])
display(evidence['inputs'])
print('Manuscript PDF exists:',(manuscript/'manuscript.pdf').is_file())
artifact('refs_predictions.npz');artifact('source_scores.csv')
renderer.OUT=rebuild;(rebuild/'figures').mkdir(exist_ok=True)
with plt.rc_context():
    plt.rcdefaults()
    renderer.figures(result,transfer,forecasts,ranks,metadata)
from IPython.display import SVG, Markdown
chart_methods={
    'Future movement only':{'internal_score':'low_motion','external_score':'low_motion'},
    'Forecast + normal reference':{'internal_score':'calibrated_two_sided','external_score':'two_sided'},
}
figure_checks=[]
for figure in sorted((manuscript/'figures').glob('*.svg')):
    record_input(figure)
    regenerated=rebuild/'figures'/figure.name
    assert regenerated.is_file() and sha(regenerated)==sha(figure),f'The rebuilt figure differs: {figure.name}'
    figure_checks.append({'figure':figure.name,'sha256':sha(figure),'exact_svg_match':True})
    if figure.stem=='ranking_comparison':
        display(Markdown('**Read the main anomaly comparison.** The chart tests whether forecast-based normal-reference scoring ranks abnormal walking better than a simple movement measurement.'))
        display(Markdown('**Future movement only** measures how far each future joint position moves from its last available prefix position. It uses negative persistence error, so less movement receives a higher abnormality score. It does not fit a forecast model.'))
        display(Markdown('**Forecast + normal reference** predicts the future from past observations, then asks whether its error is unusually small or large relative to normal examples with similar prefix movement and missing observations. It is the locked conditional two-sided score, with its original fitted models and calibration.'))
        display(Markdown('AUROC measures how often an abnormal group receives a higher score than a normal group, with half credit for a tie. Higher values mean better ranking; 0.5 is chance. The internal panel uses the original within-fold recording-group comparison, while the external panel uses coded participants. The 95% intervals condition on saved fits and do not imply clinical diagnostic accuracy.'))
    display(SVG(filename=str(regenerated)))
assert len(figure_checks)==5 and len(table_checks)==6
display(pd.DataFrame(figure_checks))
(PROJECT/'docs/reference_v2/notebook_26_publication_rebuild.json').write_text(json.dumps({'tables':table_checks,'figures':figure_checks,'renderer_sha256':sha(renderer_path),
    'reporting_identity':evidence['identity'],'evidence_sha256':sha(manuscript/'evidence.json'),
    'main_anomaly_chart':{'methods':chart_methods,'metric':'AUROC','chance':0.5,'full_appendix_score_rows':11,
                          'full_score_table_unchanged':True,'forecasts_refitted':False},
    'sequence_sensitivity_identity':sequence_sensitivity['identity'],'sequence_audit_identity':sequence_audit['analysis_identity']},indent=2)+'\\n')
print('Read the independent final review for exact accepted artifact identities after all material edits.')
'''),
        ('''## Explain the contribution in complete sentences

The last check is conceptual. Explain the observation model, the role of its simple reference, the strongest measured finding, and the remaining generalization requirement in four connected sentences. The conclusion must match the verified comparison, including the strongest simple alternative. It should help a reader understand what has been learned and why that finding matters for future medical world models.''', '''print('Evidence paths:')
for path in [RUN/'results.json',TRANSFER/'results.json',manuscript/'manuscript.pdf',PROJECT/'docs/reference_v2/protocol.md']:
    record_input(path)
    print(path.relative_to(PROJECT),sha(path))
print('The manuscript generator and independent review, rather than this notebook, determine the final publication claim.')
forecast=next(row for row in result['forecast_contrasts'] if row['method']=='ridge_posture' and row['reference']=='persistence')
print('A plain-language explanation grounded in the completed evidence follows.')
print('The study evaluates 456 unique annotated gait sequences from GAVD, including 141 normal sequences, using 2,815 windows of joint observations.')
print(f"A compact current-posture model reduced normal forecast error by {100*forecast['relative_reduction']:.1f}% relative to repeating the last pose when each original recording received equal weight.")
print(f"A separately declared post hoc analysis gave a similar {100*sequence_sensitivity['relative_reduction']:.1f}% reduction with equal sequence weights, while the masking and anomaly comparisons retained the need for stronger simple references.")
print('Uncertainty keeps related sequences together in their recording groups; these movement observations and frozen external transfer support a reproducible benchmark, while patient-level use still requires independent clinical validation.')
'''),
    ]


SPECS = [
    ('20_cohort_identity_and_coverage.ipynb','Count the observations and preserve their identities',
     'This notebook audits the complete annotation inventory, retained source groups, observation times, and external-data eligibility before interpreting any performance.',cells20),
    ('21_reference_scores_and_calibration.ipynb','Understand the reference score and its calibration',
     'This notebook explains how observed movement and prediction error enter the score, then replays a normal-only calibration reference without refitting the research model.',cells21),
    ('22_structured_masking_laboratory.ipynb','Hide observations while preserving a fair prediction task',
     'This notebook makes one fresh Ambient pose call and constructs the exact masking policies on real cached poses. It connects input construction with the fitted, controlled study.',cells22),
    ('23_compact_models_and_fitting.ipynb','Fit a compact model and inspect the complete study',
     'This notebook performs a small real normal-only teaching fit, then inspects the separate completed source-grouped evaluation. Its didactic training error is not a test result.',cells23),
    ('24_source_and_external_evaluation.ipynb','Read source-level and external-participant evidence',
     'This notebook reproduces the primary paired ranking calculation and presents forecasting, masking, nuisance controls, and external transfer at their correct evaluation units.',cells24),
    ('25_robustness_and_movement_interpretation.ipynb','Challenge the movement interpretation with controls',
     'This notebook connects observation perturbations, recording conditions, source support, and external reference shift to the limits of a physiological interpretation.',cells25),
    ('26_manuscript_evidence_and_claims.ipynb','Turn reproducible evidence into a clear manuscript',
     'This notebook maps the completed experiments to the new draft03, verifies central numbers, and distinguishes measured results from proposed extensions.',cells26),
]


def build(numbers=None):
    paths=[]
    for name,title,introduction,factory in SPECS:
        if numbers and int(name[:2]) not in numbers: continue
        book=nbf.v4.new_notebook()
        book.metadata={'kernelspec':{'display_name':'Med-WM reference-v2 Python3.12','language':'python','name':'python3'},
                       'language_info':{'name':'python','version':'3.12'}}
        setup_code=SETUP
        if name.startswith('26_'):
            setup_code=setup_code.replace("print('Verified real observations:',len(metadata),'windows;',metadata.source_id.nunique(),'source videos.')",
                "print('Verified real observations:',metadata.sequence_id.nunique(),'unique gait sequences;',len(metadata),'windows in',metadata.source_id.nunique(),'recording groups.')")
        book.cells=[nbf.v4.new_markdown_cell('# '+title+'\n\n'+introduction),nbf.v4.new_code_cell(setup_code)]
        for explanation,code in factory():
            code=code.replace('pd.read_csv(', 'read_csv(')
            book.cells.extend([nbf.v4.new_markdown_cell(explanation),nbf.v4.new_code_cell(code)])
        input_record = '''\n# Retain the exact inputs used by this independent-kernel execution.
input_path=PROJECT/'artifacts/executed-reference-v2/inputs/INPUT_NAME.json'
input_path.parent.mkdir(parents=True,exist_ok=True)
input_path.write_text(json.dumps({'path_base':'repository root','inputs':USED_INPUTS},indent=2)+'\\n')
'''.replace('INPUT_NAME',name.removesuffix('.ipynb'))
        book.cells[-1].source+=input_record
        for i,cell in enumerate(book.cells):
            cell['id']=hashlib.sha256((name+str(i)+cell.source).encode()).hexdigest()[:12]
        path=PROJECT/name;nbf.write(book,path);paths.append(path)
    return paths


def execute(paths):
    if sys.version_info[:2]!=(3,12):
        raise RuntimeError('Execute with the experiment Python3.12 environment.')
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import KernelSpecManager
    OUTPUT.mkdir(parents=True,exist_ok=True);RECORD.parent.mkdir(parents=True,exist_ok=True)
    kernel=OUTPUT/'kernel';kernel.mkdir(exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Med-WM reference-v2 verified interpreter','language':'python'},indent=2)+'\n')
    manager=KernelSpecManager();kernel_name='medwm-reference-v2-verified'
    manager.install_kernel_spec(str(kernel),kernel_name=kernel_name,user=True)
    report={'identity':'medwm-reference-v2-notebook-execution-v3','status':'running','interpreter':sys.executable,
            'python_version':sys.version,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_runs':['reference-v2-cohort-v1','reference-v2-core-v1','reference-v2-transfer-v1'],
            'requested_notebooks':[path.name for path in paths],'full_sequence':len(paths)==len(SPECS),'notebooks':[]}
    retained={};invalidated=[];requested={path.name for path in paths}
    if RECORD.is_file():
        previous=json.loads(RECORD.read_text())
        for entry in previous.get('notebooks',[]):
            if entry['name'] in requested or entry.get('status')!='complete': continue
            source=PROJECT/entry['name'];output=PROJECT/entry['output']
            matches=source.is_file() and output.is_file()
            matches=matches and hashlib.sha256(source.read_bytes()).hexdigest()==entry['source_sha256']
            matches=matches and hashlib.sha256(output.read_bytes()).hexdigest()==entry['output_sha256']
            input_trace=OUTPUT/'inputs'/f'{source.stem}.json'
            matches=matches and input_trace.is_file() and hashlib.sha256(input_trace.read_bytes()).hexdigest()==entry.get('input_record_sha256')
            for key,expected in entry.get('inputs',{}).get('inputs',{}).items():
                input_path=Path(key) if Path(key).is_absolute() else PROJECT.parents[1]/key
                matches=matches and input_path.is_file() and hashlib.sha256(input_path.read_bytes()).hexdigest()==expected
            if matches:
                entry.setdefault('generator_sha256',previous.get('generator_sha256'))
                retained[entry['name']]=entry
            else:
                invalidated.append(entry['name'])
    report['invalidated_previous_records']=invalidated
    report['retained_previous_records']=sorted(retained)
    try:
        for path in paths:
            book=nbf.read(path,as_version=4);started=time.monotonic()
            entry={'name':path.name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                   'generator_sha256':report['generator_sha256'],'status':'running'}
            error=None
            try:
                NotebookClient(book,timeout=600,kernel_name=kernel_name,resources={'metadata':{'path':str(PROJECT)}}).execute()
                entry.update(status='complete',code_cells=sum(c.cell_type=='code' for c in book.cells))
            except Exception as exc:
                entry.update(status='failed',error_type=type(exc).__name__,error=str(exc)[-4000:]);error=exc
            target=OUTPUT/path.name;nbf.write(book,target)
            entry.update(seconds=time.monotonic()-started,output=str(target.relative_to(PROJECT)),
                         output_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            inputs_path=OUTPUT/'inputs'/f'{path.stem}.json'
            if error is None:
                entry['inputs']=json.loads(inputs_path.read_text())
                entry['input_record_sha256']=hashlib.sha256(inputs_path.read_bytes()).hexdigest()
            retained[path.name]=entry
            report['notebooks']=[retained[name] for name in sorted(retained)]
            RECORD.write_text(json.dumps(report,indent=2)+'\n')
            console={key:entry[key] for key in ['name','status','code_cells','seconds','output_sha256'] if key in entry}
            console['input_hashes_recorded']=len(entry.get('inputs',{}).get('inputs',{}))
            print(json.dumps(console),flush=True)
            if error is not None:
                report['status']='failed';RECORD.write_text(json.dumps(report,indent=2)+'\n')
                raise RuntimeError('Notebook failed; inspect the saved execution artifact.') from error
        report.update(status='complete',total_code_cells=sum(n['code_cells'] for n in report['notebooks']),
                      full_sequence={n['name'] for n in report['notebooks']}=={s[0] for s in SPECS})
        direct_runs=set()
        for entry in report['notebooks']:
            for input_name in entry.get('inputs',{}).get('inputs',{}):
                parts=Path(input_name).parts
                if 'artifacts' in parts:
                    index=parts.index('artifacts')
                    if parts[index+1:index+2]==('runs',):direct_runs.add(parts[index+2])
        report['input_runs']=sorted(direct_runs)
        RECORD.write_text(json.dumps(report,indent=2)+'\n')
    finally:
        manager.remove_kernel_spec(kernel_name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--numbers',nargs='*',type=int,help='Restrict generation and execution to specified notebook numbers.')
    args=parser.parse_args();paths=build(args.numbers)
    if not paths:raise ValueError('No requested notebook number matched the generated sequence.')
    if args.execute:execute(paths)
    else:print('Built '+str(len(paths))+' source notebooks. Execution requires the completed declared study artifacts.')


if __name__=='__main__':
    main()
