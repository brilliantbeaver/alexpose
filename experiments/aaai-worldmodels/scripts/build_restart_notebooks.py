"""Build the independent, real-data restart tutorial sequence."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
SETUP='''from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
candidates = [Path.cwd(), *Path.cwd().parents]
PROJECT = next((p for p in candidates if (p/'worldmodels/restart_data.py').exists()), None)
if PROJECT is None:
    PROJECT = next(p/'experiments/aaai-worldmodels' for p in candidates if (p/'experiments/aaai-worldmodels/worldmodels/restart_data.py').exists())
sys.path.insert(0, str(PROJECT))
from worldmodels.settings import configure
PROJECT, REPOSITORY = configure()
DATA = PROJECT/'artifacts/runs/restart-full10hz-v1'
RUN = PROJECT/'artifacts/runs/restart-dynamics-v1'
assert (DATA/'preparation.json').exists(), 'Run the full extraction command in README first.'
meta = pd.read_csv(DATA/'windows.csv')
with np.load(DATA/'windows.npz', allow_pickle=False) as z:
    x, valid = z['x'], z['valid']
print('Real GAVD observations:', x.shape, 'sources:', meta.source_id.nunique())
'''
def build(name,title,intro,parts):
    nb=nbf.v4.new_notebook()
    nb.metadata={'kernelspec':{'display_name':'Med-WM gait (Python 3.12)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}
    nb.cells=[nbf.v4.new_markdown_cell('# '+title+'\n\n'+intro),nbf.v4.new_code_cell(SETUP)]
    for explanation,code in parts:
        nb.cells += [nbf.v4.new_markdown_cell(explanation),nbf.v4.new_code_cell(code)]
    import hashlib
    for i,cell in enumerate(nb.cells):
        cell['id']=hashlib.sha256((name+str(i)+cell.source).encode()).hexdigest()[:12]
    nbf.write(nb,ROOT/name)

build('10_restart_question_and_inventory.ipynb','From gait videos to a falsifiable dynamics question',
'''This restart asks whether past articulation supports useful future prediction and whether opposite-leg timing adds information. We use real videos and source groups. Earlier GAVD outcomes were known, so the new study is exploratory. This notebook audits the complete annotation inventory and the actual retained cohort.''',[
('The fresh inventory reads each transformed annotation once. The upstream source CSV is provenance, not a second set of observations.', '''from worldmodels.data import build_inventory
inventory, audit = build_inventory(REPOSITORY)
display(pd.DataFrame(audit['by_label']).T)
print('All sequences:',audit['all_sequences'],'all sources:',audit['all_sources'])'''),
('Trace attrition before counting model evidence. A source can contribute many sequences and windows but still counts once for held-out source evaluation.', '''display(meta.groupby('label').agg(windows=('source_id','size'),sequences=('sequence_id','nunique'),sources=('source_id','nunique')))
attrition=pd.read_csv(DATA/'attrition.csv')
display(attrition.head())
print(json.loads((DATA/'preparation.json').read_text())['actual_model_calls_this_invocation'])'''),
('GAVD contains all 32 annotated normal source videos in our selected scope. Extra labels add breadth but cannot create more normal sources or verified patients.', '''audit_path=PROJECT/'artifacts/restart_audit/full_gavd_availability.json'
display(json.loads(audit_path.read_text()))'''),
('Neurological evidence motivates coordination, but video coordinates do not identify neural circuits. Read the source registry and its explicit claim boundaries.', '''sources=json.loads((PROJECT/'docs/restart/primary_sources.json').read_text())['sources']
display(pd.DataFrame(sources)[['key','title','supports','does_not_support']])''')])

build('11_restart_video_pose_and_time.ipynb','Decode a real frame and call the pose model',
'''The full run decodes complete annotated spans through Ambient. Here we replay one actual frame and make a fresh MediaPipe model call. GAVD frames are one-based; OpenCV frame positions are zero-based. Source videos remain under data/videos-gavd.''',[
('Select the first retained source in deterministic order. This is a pipeline example, not a favorable-result selection.', '''from worldmodels.data import find_video
row=meta.sort_values(['source_id','sequence_id','first_frame']).iloc[0]
video=find_video(REPOSITORY,row.source_id,row.label)
assert video is not None
print('Source:',row.source_id,'annotation frame:',row.first_frame,'video:',video.relative_to(REPOSITORY))'''),
('Convert the frame index once and execute the shared image API. Missing joints remain missing.', '''import cv2
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
cap=cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES,int(row.first_frame)-1)
ok,frame=cap.read();cap.release();assert ok
extractor=SequenceKeypointExtractor()
try:
    pose=extractor.extract_from_image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),str(REPOSITORY/'data/models/pose_landmarker_lite.task'))
finally:
    extractor.cleanup()
fig,ax=plt.subplots(figsize=(9,5));ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
points=[] if pose is None else [(k.x,k.y,k.confidence) for k in pose.keypoints]
if points:
    points=np.asarray(points); keep=points[:,2]>=.5
    ax.scatter(points[keep,0],points[keep,1],s=18,c='#00ccaa')
ax.axis('off');plt.show()
print('Fresh pose call completed; detected joints:',len(points))'''),
('Nominal sampling is 10 Hz. Actual frame numbers and decoder timestamps remain recorded; no interpolated coordinate is called observed.', '''display(meta.filter(regex='time|fps|frame|horizon').head())
print((PROJECT/'docs/restart/data_contract.md').read_text().split('## Person selection')[0])'''),
('Inspect the source cache provenance and frame-level person-gate decisions. Pelvis containment does not prove identity in overlapping people.', '''with np.load(DATA/'source_poses'/f'{row.source_id}.npz',allow_pickle=False) as raw:
    print('Raw cache fields:',raw.files)
gates=pd.read_csv(DATA/'person_gate_frames.csv')
display(gates.head())''')])

build('12_restart_representation_and_controls.ipynb','Build a past-only articulation state',
'''The model predicts knee, ankle, heel and foot-index coordinates relative to the pelvis and prefix torso scale. Both hips already contribute to the shared reference. Own-leg means own distal articulation conditional on that reference.''',[
('Observe the actual valid-joint mask. Context and future each require adequate support.', '''fig,ax=plt.subplots(figsize=(10,3));ax.imshow(valid[0].T,aspect='auto',cmap='Greens',vmin=0,vmax=1)
ax.axvline(15.5,color='orange');ax.set(xlabel='Sample index',ylabel='Declared joint index',title='Actual observed support; orange line starts the forecast');plt.show()'''),
('Features must be invariant to every future coordinate and future validity bit.', '''from worldmodels.restart_experiment import features
f=features(x[:3],valid[:3],0,'bilateral')[0]
changed=x[:3].copy();changed[:,16:]=12345
changed_valid=valid[:3].copy();changed_valid[:,16:]=~changed_valid[:,16:]
np.testing.assert_array_equal(f,features(changed,changed_valid,0,'bilateral')[0])
print('Future-coordinate and future-mask invariance passed:',f.shape)'''),
('A temporal control must alter alignment while preserving the available values. A single global permutation can be learned by a linear model and is insufficient.', '''print((PROJECT/'docs/restart/protocol.md').read_text().split('## Control-design correction')[1].split('## Common')[0])
print('Available observed context:',16,'frames; forecast:',8,'frames')'''),
('A zero is only a storage convention when the validity mask is false. No future target is imputed into observed evidence.', '''from worldmodels.restart_experiment import targets
residual,mask=targets(x[:5],valid[:5],0)
print('Target residual shape:',residual.shape,'observed coordinates:',mask.sum())
assert residual.shape==mask.shape''')])

build('13_restart_fit_and_cross_validation.ipynb','Fit a real normal-source dynamics model',
'''This notebook performs a small real ridge fit and then inspects the complete saved nested source evaluation. It does not retune the published experiment. Inner folds select settings and outer folds provide held-out source predictions.''',[
('Use only normal source observations for this didactic model fit. Its training error is not held-out evidence.', '''from worldmodels.restart_experiment import features,targets,fit_grid,hierarchical_weights,predict
training=np.flatnonzero(meta.label.eq('normal'))[:120]
f=features(x[training],valid[training],0,'bilateral')[0]
y,m=targets(x[training],valid[training],0)
models=fit_grid(f,y,m,hierarchical_weights(meta.iloc[training]),alphas=[10.])
pred=predict(models[10.],f)
print('Real fitted coefficient predictions:',pred.shape,'observed training residual MSE:',float(((pred-y)[m]**2).mean()))'''),
('Every source remains in one outer fold. The complete run saves normal-only training source lists for every model.', '''cohort=pd.read_csv(RUN/'cohort.csv')
assert cohort.groupby('source_id').outer_fold.nunique().max()==1
print('Prepared windows:',len(meta),'model windows:',len(cohort),'support exclusions:',len(meta)-len(cohort))
display(pd.read_csv(RUN/'support_exclusions.csv'))
display(cohort.groupby(['outer_fold','label']).source_id.nunique().unstack(fill_value=0))'''),
('The prospective shrinkage rule may select persistence. That choice is reported instead of being hidden.', '''selection=json.loads((RUN/'selection.json').read_text())
display(pd.DataFrame(selection).drop(columns=['all_grid_mse'],errors='ignore'))'''),
('The fitted run pins input data, protocol and implementation. These identities allow an independent replay to detect stale artifacts.', '''display(json.loads((RUN/'identity.json').read_text()))''')])

build('14_restart_results_and_uncertainty.ipynb','Read the paired source-level evidence',
'''The complete experiment is exploratory. Source bootstrap intervals condition on fixed fitted folds; they are not patient-level clinical confidence intervals. All declared comparisons remain visible.''',[
('Compare errors in the declared coordinate units. Lower error is better. Horizon eight averages the eight nominal future samples from 0.1 to 0.8 seconds; it is not an endpoint-only error. Three prepared windows fail the common distal-target support gate.', '''results=json.loads((RUN/'results.json').read_text())
display(pd.DataFrame(results['normal_forecast']).query('horizon == 8'))'''),
('A favorable difference must be read alongside its interval and control. We separate forecast utility from evidence that correct bilateral timing matters.', '''display(pd.DataFrame(results['paired_contrasts']).query('horizon == 8'))'''),
('All anomaly directions were fixed before evaluation. Higher scores mean more abnormal even if a score ranks the annotations poorly. Pooling out-of-fold rankings assumes score comparability across separately fitted normal-source models.', '''display(pd.DataFrame(results['anomaly']).T)
print('Abnormality labels are annotations, not verified clinical diagnoses.')'''),
('Condition tables and source distributions reveal heterogeneity that a single pooled number conceals.', '''condition=pd.read_csv(RUN/'condition_results.csv');display(condition.query('horizon == 8'))
source=pd.read_csv(RUN/'source_scores.csv')
fig,ax=plt.subplots(figsize=(9,4))
for label,group in source.query("arm == 'bilateral' and horizon == 8").groupby('label'):
    ax.scatter([label]*len(group),group.mse,alpha=.65)
ax.set(ylabel='Source-average articulation MSE');ax.tick_params(axis='x',rotation=20);plt.show()''')])

build('15_restart_manuscript_and_claims.ipynb','Turn verified results into bounded claims',
'''A strong manuscript connects a clear contribution to evidence and states its limits. We keep the original narrative, the revised results, and an independent critique so the revisions are auditable.''',[
('The study can support observation-model claims. Treatment simulation and neurological mechanism require different evidence.', '''print((PROJECT/'docs/restart/protocol.md').read_text().split('## Interpretation and limits')[1].split('## Prospective')[0])'''),
('Read the source-backed research rationale rather than inferring neural mechanisms from forecast error.', '''print((PROJECT/'docs/restart/research_review.md').read_text()[:6000])'''),
('The completion record distinguishes extraction, fitting, notebooks and manuscript review.', '''print('Extraction status:',json.loads((DATA/'preparation.json').read_text()).get('mode'))
print('Model status:',json.loads((RUN/'results.json').read_text())['status'])
print('Manuscript path:',PROJECT/'docs/draft01/manuscript.md')'''),
('The next study should add independent patient identity and recording settings. More windows alone cannot meet this requirement.', '''display(pd.DataFrame(json.loads((RUN/'results.json').read_text())['cohort_by_label']))
print('No diagnosis, treatment recommendation, or clinical deployment decision is justified by this benchmark.')''')])
print('Built six real-data restart notebooks.')
