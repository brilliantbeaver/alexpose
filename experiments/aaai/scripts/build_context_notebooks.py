"""Build independently runnable tutorials for the revised research direction."""
from pathlib import Path
import textwrap
import nbformat as nbf
BASE=Path(__file__).resolve().parents[1]
def md(text):return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())
def code(text):return nbf.v4.new_code_cell(textwrap.dedent(text).strip())
SETUP='''
from pathlib import Path
import os, sys, json
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'ambient').is_dir() and (p/'data/gavd').is_dir())
BASE=ROOT/'experiments/aaai'
os.chdir(ROOT)
from dotenv import load_dotenv as load_env, find_dotenv as find_env
load_env(find_env())
os.chdir(BASE)
if str(BASE) not in sys.path:sys.path.insert(0,str(BASE))
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
from aaai_gait.settings import Settings
from aaai_gait.study_data import build_study_manifest,make_outer_folds,fold_metadata
from aaai_gait.study_runner import prepare_study
from aaai_gait.media import show_video_context,plot_motion
settings=Settings();torch.set_num_threads(2)
fresh_manifest=build_study_manifest(ROOT,settings.pose_dir)
print('Mode:', 'SYNTHETIC SOFTWARE CHECK' if settings.demo else 'REAL GAVD, subject to quality exclusions')
print('Fresh annotations:',len(fresh_manifest),'sequences;',fresh_manifest.source_id.nunique(),'source videos')
video_row=show_video_context(fresh_manifest,frame_number_base=settings.frame_number_base)
'''
LOAD='''
x,meta,manifest,ledger=prepare_study(demo=settings.demo,seed=settings.seed)
print('Available QC-passing windows:',x.shape)
display(meta.groupby('label').agg(sources=('source_id','nunique'),sequences=('sequence_id','nunique'),windows=('sequence_id','size')))
'''
def start(title,intro):
    return [md(f'''# {title}\n\n{intro}\n\n## Run this notebook on its own\n\nFrom `experiments/aaai`, run `uv sync`, then `uv run jupyter lab`. Select the Python environment created here. Python 3.12, PyTorch, pandas, scikit-learn, matplotlib, python-dotenv, the notebook libraries, and the local `ambient` package are declared in `pyproject.toml`. Each notebook imports and loads everything it needs.\n\nAll configuration and credentials remain in the repository root `.env`. Set `AAAI_DEMO=1` for artificial arrays used to check the software, or `AAAI_DEMO=0` for available real poses. Restart the kernel after changing settings. Artificial examples have no medical meaning. The setup uses aliases for python-dotenv's actual function names so that `load_env(find_env())` works.\n\nThe video below provides real walking context in either mode. It is separate from any artificial example. These research notebooks do not launch a large download or full experiment without an explicit command.'''),code(SETUP)]
def finish(cells,name):
    from build_iteration02_notebooks import navigation_cell
    cells.insert(1,navigation_cell())
    cells.append(md('''## Record what this experiment establishes\n\nKeep the run protocol, source roles, exclusions, training curves, frozen reference parameters, and source scores together. A diagram explains an intended comparison. Only a completed real-data run supplies evidence for it. Read the [implemented protocol](docs/context_study_protocol.md), [current results](docs/context_study_results.md), and [review of the research direction](docs/draft00/README.md) before making a manuscript claim.'''))
    book=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (AAAI gait)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    nbf.validate(book);nbf.write(book,BASE/name)

cells=start('08. Turn a biological question into a test','We will connect neurological and muscle limitations to observable movement, count all current GAVD folders, and freeze source roles without using model performance.')
cells += [md('''## 1. Start with movement that the video can support\n\nAfter stroke, a person may move one leg differently to maintain forward progress. That pattern can be highly repeatable. Parkinson's disease can affect movement size and timing differently. Cerebral palsy covers varied developmental motor impairments, and visible walking patterns change with growth and muscle properties. Myopathic gait describes a family of muscle-related problems. It is not one disease of the brain.\n\nThese observations motivate checking movement relationships. They do not let us infer a lesion, dopamine level, muscle force, or diagnosis from a web video. The [neurological review](docs/draft00/neurological_research_direction.md) links the primary clinical studies behind each hypothesis.\n\n![Biology, movement, and camera observations](docs/draft00/images/neurological_chain.svg)\n\n![Different causes can produce overlapping movements](docs/draft00/images/condition_routes.svg)'''),md('''## 2. Ask a question with a concrete comparison\n\nHide four consecutive positions of the left ankle. First predict them from the ankle's other visible positions. Then predict the same positions with the other visible joints available. The difference in prediction error measures how much the extra context helped this particular predictor. We repeat the comparison over random masks and twelve readout joints.\n\nBoth reduced and increased help can be unusual. A compensation may make two body parts move together more tightly. A scalar error can also be low for an abnormal but predictable movement. Our proposed three-region profile preserves this possibility without choosing a score direction after viewing the test results.\n\n![An example of predictive help](docs/draft00/images/predictive_benefit.svg)\n\nThe task is masked completion. Visible positions may come before and after a hidden section. Notebook 05 separately teaches forecasting with a fully hidden future.'''),code('''
illustration=pd.DataFrame({'situation':['Familiar relationship','Less help','More help'],
                          'own_error':[.20,.20,.20],'full_error':[.08,.18,.03]})
illustration['help']=illustration.own_error-illustration.full_error
print('Invented teaching values, not measured GAVD results.')
display(illustration)
'''),md('''## 3. Use all folders for the binary question\n\nOnly `normal` is normal. Every other current condition folder is abnormal for this experiment, including `exercise`, `style`, and `pregnant`. These labels describe the task and are not all medical diagnoses. The four condition families remain the focus of biological discussion and the separate five-class classifier. Stroke appears once.\n\nRead the current CSVs each time. The `source` directory is a distribution cache and must not add duplicate observations. A source video can contain several sequences, and a sequence can provide overlapping windows. These are not additional independent participants.'''),code('''
census=fresh_manifest.groupby('label').agg(sequences=('sequence_id','size'),rows=('n_frames','sum'),sources=('source_id','nunique'))
display(census)
assert fresh_manifest.loc[fresh_manifest.label.eq('normal'),'binary_label'].eq(0).all()
assert fresh_manifest.loc[fresh_manifest.label.ne('normal'),'binary_label'].eq(1).all()
print('Manifest fingerprint:',fresh_manifest.attrs['manifest_sha256'])
'''),md('''## 4. Retrieve media, extract poses, and record failures\n\nRun the following command in a terminal when ready. The fresh inventory covers all twelve folders. Downloads go to `data/videos-gavd/<source_id>/`. Source IDs avoid duplicate downloads when a video has several annotated sequences. Existing decodable local media are reused.\n\n```bash\nuv run python scripts/ingest_study.py --download-missing --download-budget-seconds 900 --source-timeout 45 --download-workers 4 --workers 4\n```\n\nA fixed acquisition budget prevents a few unavailable videos from blocking the study. Repeat the command to continue acquisition. Logs record videos not attempted, timeouts, and failures. The pose pipeline uses `ambient` through the annotated-person crop adapter, retains raw confidence and frame indices, and hashes the annotation, video, model, and extraction settings. Fixed observation-quality rules apply before folds. A score-dependent numerical failure must raise an error rather than become a quiet exclusion.\n\n![Pipeline and controls](docs/draft00/images/neuromechanics_experiment.svg)'''),code(LOAD),md('''## 5. Separate fitting, reference estimation, and testing\n\nFour outer folds distribute normal sources and abnormal category memberships without seeing scores. In each fold, five remaining normal sources estimate the normal profile reference. Other normal sources fit the encoder and predictors. Non-test abnormal sources are auxiliary for the separate supervised classifier only. They never fit the normal-only anomaly model.\n\nBefore assigning folds, the full runner also checks that every source has four distinct eligible physical patches for each readout joint. This check reads masks and observation validity, not losses. Repeated masks and overlapping windows do not create additional physical support.'''),code('''
from aaai_gait.predictive_profile import freeze_profile_eligibility
from aaai_gait.context_prediction import ContextConfig
keep,eligibility=freeze_profile_eligibility(x,meta,config=ContextConfig())
eligible_meta=meta.loc[keep].reset_index(drop=True)
registry=make_outer_folds(eligible_meta,seed=settings.seed)
roles=fold_metadata(eligible_meta,registry,outer_fold=0)
display(registry.groupby(['outer_fold','binary_label']).size().unstack(fill_value=0))
display(roles.groupby('split').agg(sources=('source_id','nunique'),windows=('source_id','size')))
assert roles.loc[roles.split.isin(['train','val']),'label'].eq('normal').all()
'''),md('''## 6. Review visible movement without making a diagnosis\n\nA small blinded review should describe whether the person is walking, the camera view, tracking identity, visible asymmetry, arm excursion, knee flexion, trunk motion, and aids. Keep the label lookup separate. Uncertain observations remain uncertain. Two independent reviewers and agreement reporting would strengthen the biological interpretation. A blank template is not evidence that review occurred.'''),code('''
from aaai_gait.study_controls import movement_annotation_template
blind,lookup=movement_annotation_template(fresh_manifest,per_label=2,seed=settings.seed)
display(blind.head())
print('Blank review rows:',len(blind),'; label lookup is a separate table.')
''')]
cells.insert(4,md('''## Watch an annotated abnormal example

The dataset's condition label provides context, not a confirmed diagnosis. This example previews a stroke-labeled sequence. Compare the visible movement with the normal preview above without assuming which side is affected. Set `AAAI_PREVIEW_LABEL` in the root `.env` to `parkinsons`, `cerebral palsy`, or `myopathic` to inspect another family.'''))
cells.insert(5,code('''
condition_to_view=os.getenv('AAAI_PREVIEW_LABEL','stroke')
condition_video=show_video_context(fresh_manifest,label=condition_to_view,frame_number_base=settings.frame_number_base)
'''))
finish(cells,'08_biology_cohort_and_source_folds.ipynb')

cells=start('09. Train matched predictors without information leakage','We will follow the six predictor heads, prove what each head may see, run a small normal-only training exercise, and inspect the exact same masked targets.')
cells += [md('''## 1. Keep the three kinds of randomness separate\n\nUniform masking samples hidden positions without reading coordinates, movement speed, condition, or confidence. All sixteen approved joints participate. Observation validity is checked after the mask is drawn, and an invalid target is not replaced by another draw.\n\nStochastic positional embeddings add noise to position queries. Stochastic latent variables represent alternative completions through a learned distribution. These are different ideas. The primary context experiment uses deterministic positions and no stochastic latent variable so that its information-access comparison is interpretable. The earlier optional positional-noise ablation remains available in Notebook 07. Add latent uncertainty only after checking calibration and diversity against a deterministic comparator.\n\n![Separate sources of stochasticity](images/stochasticity.svg)'''),md((BASE/'docs/joint_table.md').read_text()),md('''## 2. Freeze a normal-only encoder, then compare access\n\nPretrain the skeleton JEPA with uniform masks, an EMA teacher, centered target distributions, and VICReg. Freeze the resulting encoder, teacher, and target center. Each prediction head learns on normal sources only. Its inputs are restricted before any contextual attention.\n\nThe own-joint head sees only visible patches of the target joint. The patch-context head also sees other joints in the target four-frame interval. The full-context head sees all visible approved joints. Each target needs at least two visible patches of its own joint.\n\nThere are six independent heads: three access conditions for coordinate targets and three for latent targets. Identical initialization within a target type, matching masks, separate optimizers, and fixed update budgets make the comparison easier to interpret. Coordinate MSE and latent KL have different units and must never be subtracted from one another.\n\n![Matched experiment](docs/draft00/images/neuromechanics_experiment.svg)'''),code('''
from aaai_gait.context_prediction import ContextConfig,context_access,fit_context_models,score_context_models,make_mask_bank
cfg=ContextConfig(pretrain_updates=8,predictor_updates=8,batch_size=8,validation_interval=8,validation_batches=2)
mask=make_mask_bank(cfg,mask_repeats=1)
valid=torch.ones((1,128),dtype=torch.bool)
hidden_ankle_patches=torch.nonzero(mask[0,10::16],as_tuple=True)[0]
target=torch.tensor([int(hidden_ankle_patches[0])*16+10])  # actual hidden left-ankle patch
assert bool(mask[0,target.item()])
access={branch:context_access(mask,valid,target,branch) for branch in ('own','patch','full')}
fig,axes=plt.subplots(1,3,figsize=(12,3))
for ax,(name,allowed) in zip(axes,access.items()):
    ax.imshow(allowed.reshape(8,16),vmin=0,vmax=1,aspect='auto');ax.set_title(name);ax.set_xlabel('Joint slot')
axes[0].set_ylabel('Time patch');plt.tight_layout();plt.show()
assert not (access['own'] & ~access['full']).any()
'''),code(LOAD),code('''
from aaai_gait.predictive_profile import freeze_profile_eligibility
keep,_=freeze_profile_eligibility(x,meta,config=cfg)
a=x[keep];m=meta.loc[keep].reset_index(drop=True)
registry=make_outer_folds(m,seed=settings.seed);m=fold_metadata(m,registry,0)
train=m.split.eq('train').to_numpy();val=m.split.eq('val').to_numpy()
exercise_bundle,history=fit_context_models(a[train],a[val],config=cfg,seed=settings.seed,
    train_source_ids=m.loc[train,'source_id'].to_numpy(),validation_source_ids=m.loc[val,'source_id'].to_numpy(),
    train_labels=m.loc[train,'label'].to_numpy(),validation_labels=m.loc[val,'label'].to_numpy())
print('Eight-update teaching exercise. This is not a converged research model.')
display(pd.DataFrame(history['pretrain']))
display(pd.DataFrame(history['predictors']))
'''),md('''## 3. Inspect matched losses and support\n\nThe table records the original BlazePose index, temporal patch, shared mask ID, eligibility, and three losses. A missing target or insufficient own-joint context makes the row ineligible for all three branches. Valid rows with a nonfinite model loss are implementation or optimization failures.\n\nThe main profile reads twelve joints in three groups: four left-leg joints, four right-leg joints, and four arm joints. The hips and shoulders still participate in training and masking. They are excluded from the primary score because pelvis centering creates an exact relationship between the hips, which could make the test unusually easy.'''),code('''
example_rows=score_context_models(exercise_bundle,a[val][:4],mask_repeats=2)
display(example_rows.head(12))
assert np.isfinite(example_rows.loc[example_rows.valid,['loss_own','loss_patch','loss_full']]).all().all()
print(example_rows.groupby('target_type',observed=True).valid.agg(['size','sum']))
'''),md('''## 4. Run the full experiment from a terminal\n\n```bash\nuv run python scripts/run_context_study.py --pretrain-updates 512 --predictor-updates 512 --masks 16 --ridge-draws 256\n```\n\nThis command reads current annotations and poses, freezes eligibility and folds, trains each fold independently, saves compatible checkpoints, and evaluates held-out sources. It also fits a simple source-balanced ridge predictor and an untrained-encoder control. A different corpus, implementation, mask count, or update budget produces a different run directory.\n\nThe update count is a budget, not proof of convergence. Inspect validation curves, teacher drift, effective rank, and variance before arguing that representation learning helped. Fix a larger-budget sensitivity using normal validation behavior rather than choosing the run with the best abnormal test score.\n\nA frozen encoder trained with broad context may behave differently when given only one joint. Coordinate targets, the ridge control, and the untrained control help investigate this alternative explanation. Predictive help alone is not a measurement of neural coupling.''')]
finish(cells,'09_matched_context_prediction.ipynb')

READ_RUN='''
kind='synthetic' if settings.demo else 'gavd'
latest=BASE/'artifacts/context_study'/kind/'latest.json'
run_dir=None
if latest.exists():
    pointer=json.loads(latest.read_text());run_dir=Path(pointer['run_dir'])
    if not run_dir.is_absolute():run_dir=ROOT/run_dir
    summary=json.loads((run_dir/'summary.json').read_text())
    assert bool(summary['synthetic'])==settings.demo
    print('Recorded artifact:',run_dir,'status:',summary['status'])
    display(summary)
else:
    print('No completed artifact in this mode. Run scripts/run_context_study.py first.')
    print('The calculations below teach the protocol without inventing results.')
'''
cells=start('10. Evaluate the profile and compare classifiers','We will combine matched errors into source-level profiles, freeze the normal reference, compare methods on the same sources, and keep supervised condition recognition separate from anomaly detection.')
cells += [md('''## 1. Build a profile that a reader can explain\n\nFor each joint, subtract full-context error from own-joint error. Average common targets within a window, then windows within a sequence, then sequences within a source video. Give each of the four joints in a body region equal weight. The result has three numbers: left-leg help, right-leg help, and arm help.\n\nThis hierarchy prevents a long sequence from dominating a source. It also prevents a highly visible joint from dominating its region. At least four distinct physical patches must support every readout joint. Missing regions make a source unscorable; they never become zero-valued evidence of normality.\n\n![Three regions and a normal reference](docs/draft00/images/support_profile.svg)'''),code('''
# This matrix is invented solely to explain the reference calculation.
normal_profiles=np.array([[.10,.11,.06],[.12,.10,.05],[.09,.12,.07],[.11,.09,.06],[.10,.10,.05]])
example_profile=np.array([.03,.16,.06])
mean=normal_profiles.mean(0);variance=normal_profiles.var(0,ddof=1)
reference_variance=np.maximum(.5*variance+.5*variance.mean(),1e-8)
contribution=(example_profile-mean)**2/reference_variance
print('Illustrative normal-reference distance:',contribution.mean())
display(pd.DataFrame({'region':['left leg','right leg','arms'],'normal_mean':mean,
                      'example_help':example_profile,'squared_standardized_deviation':contribution}))
'''),md('''## 2. Estimate normal variation without touching the test sources\n\nThe five inner normal sources estimate the mean and variation of each region. Half of each region's sample variance is combined with half of the mean variance across regions. This shrinkage limits unstable estimates from a small reference group, but it does not solve the small-sample problem. A floor prevents division by zero. The anomaly score is the mean squared standardized distance across the three regions.\n\nEvery fold has its own fitted models and reference. Raw distances from different folds need not have the same scale. We calculate AUROC within each fold and average those values using the number of normal test sources as weights. We also report average precision with each fold's abnormal prevalence. A high average precision can be less informative when most videos are labeled abnormal.\n\n![Separate the normal reference and test](docs/draft00/images/neuromechanics_experiment.svg)'''),code(READ_RUN),code('''
if run_dir is not None:
    metrics=pd.read_csv(run_dir/'ranking_metrics.csv')
    key_methods=['trained_latent_all_profile_score','trained_latent_all_raw_full',
                 'trained_coordinate_all_profile_score','ridge_coordinate_all_profile_score',
                 'untrained_latent_all_profile_score']
    display(metrics[metrics.method.isin(key_methods)])
    display(json.loads((run_dir/'primary_paired_bootstrap.json').read_text()))
    print('Synthetic checks have no scientific interpretation.' if settings.demo else 'These are recorded real-run estimates, not a guarantee of generalization.')
'''),md('''## 3. Compare the proposed contribution with simpler explanations\n\nThe primary comparison is the trained latent profile against full-context latent error on the same twelve joints, masks, targets, and sources. The coordinate profile checks whether the effect remains with a local measurement target. Ridge regression checks whether a linear completion model already captures the effect. An untrained encoder checks whether representation learning is necessary.\n\nThe patch-context head asks whether simultaneous geometry provides most of the help. Separate normal-reference distances for own-joint and full-context errors test whether subtraction adds value. The fixed-budget sensitivity retains four physical patches per joint and source. Reference sensitivity removes one inner normal source at a time while keeping predictor weights fixed. These comparisons must not be selected because they produce the largest test score.'''),code('''
if run_dir is not None:
    paired=pd.read_csv(run_dir/'paired_source_scores.csv')
    print('Unique outer-test sources:',paired.source_id.nunique())
    display(paired.groupby('outer_fold').binary_label.agg(['size','sum']))
    assert not paired.source_id.duplicated().any()
    files=sorted(run_dir.glob('fold_*/trained_reference_sensitivity.csv'))
    if files:
        sensitivity=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
        ranges=sensitivity.groupby(['outer_fold','target_type','source_id']).score.agg(lambda values: values.max()-values.min()).rename('reference_score_range')
        display(ranges.reset_index().groupby(['outer_fold','target_type']).reference_score_range.agg(['min','median','max']))
'''),md('''## 4. Keep uncertainty at the source level\n\nThe paired bootstrap resamples source IDs within each outer fold and binary label. Each sampled source carries both methods' scores. Windows do not become independent observations. The interval conditions on the fitted models, reference samples, and fold assignments. It omits training uncertainty, reference-selection uncertainty, and repeated-person dependence across source IDs.\n\nA positive estimate with an interval spanning zero does not establish improvement. A negative result also matters: it would show that this context comparison did not add value under this protocol. Do not reverse the score or search for a favorable split after reading it.'''),md('''## 5. Compare the five-condition classifiers fairly\n\nThe Random Forest uses exp5-style angle and motion summaries from the same normalized pose windows. The JEPA classifier uses the frozen normal-trained encoder followed by a linear classifier. Both receive normal and abnormal labels from non-test fitting sources, excluding the five normal reference sources. The encoder itself remains normal-trained.\n\nThese classifiers answer a supervised five-label question. They receive more label information than the normal-only anomaly detectors, so their macro-F1 scores belong in a separate table. Hyperparameters are fixed. Probabilities are averaged within sequences and then sources. A source with several focal labels needs explicit source-label handling and cannot be counted as independent people. Labels are dataset annotations, not verified diagnoses.\n\nThe full runner executes `condition_capstone` from `aaai_gait.study_baselines` in each fold. Notebook 06 retains the detailed earlier exp5 comparison and video-level prediction tutorial.'''),code('''
if run_dir is not None:
    rows=[]
    for path in sorted(run_dir.glob('fold_*/condition_metrics.json')):
        data=json.loads(path.read_text())
        for name,value in data['metrics'].items():
            rows.append({'fold':path.parent.name,'method':name,'macro_f1':value['macro_f1'],
                         'balanced_accuracy':value['balanced_accuracy'],'test_sources':value['n_test_sources']})
    display(pd.DataFrame(rows))
'''),md('''## 6. Interpret condition-level results cautiously\n\nReport each category's source count, supported folds, AUROC, average precision, and prevalence. Condition summaries reuse the same normal sources and are dependent. Categories with two or three sources are descriptive examples. Even a strong result could reflect age, camera, assistive devices, or uploader style. GAVD cannot establish performance on new clinical participants, sites, or disease stages without an independent cohort.''')]
finish(cells,'10_source_profiles_and_classifier_comparison.ipynb')

cells=start('11. Challenge the explanation before writing the paper','We will edit a longer motion sequence, keep observation support paired, inspect geometric damage, and identify the evidence needed for a neurological interpretation.')
cells += [md('''## 1. Make an edit whose meaning is narrow and testable\n\nShift the entire left leg forward in time while keeping the rest of the body at its original time. Compare that edit with shifting the whole body by the same amount. The relative shift changes relationships between body regions; the whole-body shift changes the crop's phase. Both use interior samples from a longer sequence, with no circular wraparound or repeated boundary frames.\n\nThe leg retains its donor sequence's internal articulation, but its connection to the torso may become distorted. Measure that distortion explicitly. These edits test model sensitivity. They do not simulate stroke, Parkinson's disease, cerebral palsy, or muscle weakness.\n\n![Paired timing and observation controls](docs/draft00/images/intervention_controls.svg)'''),code('''
from aaai_gait.demo import synthetic_corpus
from aaai_gait.study_controls import paired_edits,common_intervention_rows
long_x,_=synthetic_corpus(seed=42,frames=40,sources_per_class=2)
edits,geometry=paired_edits(long_x[:2],shift=4,seed=42)
print('Artificial software demonstration only.')
display(geometry)
for name in ('original','relative_limb_shift','whole_body_shift'):
    plot_motion(edits[name][0],title=name);plt.show()
assert np.array_equal(edits['whole_body_shift'],long_x[:2,8:40])
'''),md('''## 2. Separate changed motion from degraded observation\n\nCoordinate noise and observation dropout are separate controls. The learning mask still comes from the same uniform bank. It must not adapt to the edit. A prediction difference can otherwise come from a changed set of evaluated joints rather than changed movement.\n\nFor every version of a source, intersect valid target identities before comparing scores. An identity contains the window, original joint index, time patch, mask ID, and target type. A dropout experiment may leave too little support for a complete profile. Report that loss of coverage instead of replacing missing regions with zero.\n\nThe real control command uses development normal sources only, keeps the original crop's normalization fixed across edits, and applies the fitted reference without refitting it on edited data.'''),code('''
from aaai_gait.context_prediction import ContextConfig,fit_context_models,score_context_models
small,small_meta=synthetic_corpus(seed=9,sources_per_class=6)
# First12 rows are artificial normal examples, six for each role.
mini,mini_history=fit_context_models(small[:6],small[6:12],
    config=ContextConfig(pretrain_updates=2,predictor_updates=2,batch_size=4,validation_batches=1),seed=42,
    train_source_ids=small_meta.source_id[:6].to_numpy(),validation_source_ids=small_meta.source_id[6:12].to_numpy(),
    train_labels=small_meta.label[:6].to_numpy(),validation_labels=small_meta.label[6:12].to_numpy())
raw={name:score_context_models(mini,values,mask_repeats=2) for name,values in edits.items()}
paired=common_intervention_rows(raw)
display(pd.DataFrame({'requested_rows':{k:len(v) for k,v in raw.items()},
                      'paired_rows':{k:len(v) for k,v in paired.items()}}))
keys=['window_index','joint','patch_index','mask_id','target_type']
for table in paired.values():
    assert table[keys].reset_index(drop=True).equals(paired['original'][keys].reset_index(drop=True))
'''),md('''## 3. Run the real development controls\n\nAfter a completed real study, replace the path below with its printed run directory:\n\n```bash\nuv run python scripts/run_study_controls.py --run-dir artifacts/context_study/gavd/<run_id> --fold 0\n```\n\nThe command chooses one eligible long window per development normal source using sequence and time order, not scores. Outputs contain paired profiles, source IDs, observation coverage, and geometry checks. Inspect the video and skeleton alongside these numbers. A relative shift producing an impossible pelvis or large torso changes cannot support a specific coordination explanation.\n\nThe teacher target is contextual. Editing other joints can change that target as well as predictor context. The coordinate branch is essential. The optional `--teacher-decomposition` flag now separates changes in predictor inputs from changes in the frozen teacher target. It evaluates four combinations on identical target support. For a relative left-leg edit, the diagnostic reads only unchanged arm and right-leg joints. A teacher-side change then demonstrates contextualization of the model target, not a change in those local joint measurements.'''),code(READ_RUN),code('''
if run_dir is not None:
    control_dir=run_dir/'fold_0/development_controls'
    if (control_dir/'protocol.json').exists():
        display(json.loads((control_dir/'protocol.json').read_text()))
        display(pd.read_csv(control_dir/'geometry.csv').groupby('intervention').median(numeric_only=True))
    else:
        print('No real development-control result recorded for this run.')
'''),md('''## 4. Decide which manuscript claim survives\n\nA strong contribution needs a reproducible improvement over simpler scores, evidence that learning matters, stability to reference and observation choices, and a plausible movement-level explanation. If only the latent target improves, discuss teacher contextualization as an alternative. If ridge performs equally well, emphasize the useful comparison rather than claiming a necessary JEPA innovation. If the untrained encoder matches the trained one, the experiment has not demonstrated an advantage from self-supervised learning.\n\nA published skeleton anomaly baseline still needs a validated adaptation to these sixteen joints and missing observations. Read the [feasibility review](docs/published_baseline_feasibility.md). An inspired implementation or repaired forward pass is not a reproduced baseline.\n\n![Mechanistic ambiguity](docs/draft00/images/mechanism_ambiguity.svg)'''),md('''## 5. Keep the longer-term world model grounded\n\nA physical gait model would separate a controller, muscle and body properties, ground contact, and the camera. Several combinations can produce similar two-dimensional motion. A useful research direction is to return a set of compatible explanations and identify which extra measurement would distinguish them. Force, EMG, measured strength, medication state, or clinical examination could provide additional evidence.\n\nGAVD can help study visible motion and observation bias. It does not supply the ground truth needed to identify hidden biological causes. The current completion study is a controlled first step toward that program.\n\n![A physical world-model research path](docs/draft00/images/physical_world_model.svg)''')]
cells.insert(-2,md('''## Inspect the contextual teacher directly

Let p0 and p1 denote predictor inputs before and after the edit. Let y0 and y1 denote teacher targets before and after it. Evaluate p0y0, p1y0, p0y1, and p1y1. The total change equals the predictor-only change, teacher-only change, and their interaction. This identity concerns raw losses. A nonlinear profile distance would require recomputing all four complete profiles against the same frozen reference.

Here we inspect only the unchanged right ankle while the left leg is shifted. The values below come from an artificial teaching sequence and a tiny model, so they establish software behavior only.'''))
cells.insert(-2,code('''
from aaai_gait.teacher_decomposition import score_teacher_decomposition,decompose_teacher_losses
factors=score_teacher_decomposition(mini,edits['original'],edits['relative_limb_shift'],mask_repeats=2,joints=(28,))
deltas=decompose_teacher_losses(factors)
display(deltas.head())
print('Each factor retains the same target identities and common eligibility.')
'''))
finish(cells,'11_interventions_and_manuscript_evidence.ipynb')

for path in sorted(BASE.glob('0[0-7]_*.ipynb')):
    book=nbf.read(path,as_version=4)
    from build_iteration02_notebooks import navigation_cell,NAVIGATION_MARKER
    markers=('## How this tutorial fits the revised study',NAVIGATION_MARKER)
    book.cells=[cell for cell in book.cells if not (cell.cell_type=='markdown' and cell.source.startswith(markers))]
    book.cells.insert(1,navigation_cell())
    nbf.validate(book);nbf.write(book,path)
print('Built notebooks08 through11 and updated the eight foundation notebooks.')
