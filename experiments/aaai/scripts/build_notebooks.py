"""Author the tutorials from readable Markdown and executable Python cells."""
from pathlib import Path
import textwrap
import nbformat as nbf
BASE=Path(__file__).resolve().parents[1]
def md(s):return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())
def code(s):return nbf.v4.new_code_cell(textwrap.dedent(s).strip())
SETUP='''
from pathlib import Path
import os, sys
# Find the repository from either its root or this notebook directory.
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'ambient').is_dir() and (p / 'data/gavd').is_dir())
BASE = ROOT / 'experiments/aaai'
os.chdir(ROOT)
from dotenv import load_dotenv as load_env, find_dotenv as find_env
load_env(find_env())  # requested spelling, using aliases for python-dotenv's real API
os.chdir(BASE)
if str(BASE) not in sys.path: sys.path.insert(0, str(BASE))
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from aaai_gait.settings import Settings
from aaai_gait.data import build_manifest, grouped_split, LABELS
from aaai_gait.media import show_video_context, plot_motion
settings = Settings()
torch.set_num_threads(2)
manifest = grouped_split(build_manifest(ROOT), seed=settings.seed)
print('Mode:', 'SYNTHETIC SOFTWARE DEMONSTRATION' if settings.demo else 'GAVD RESEARCH RUN')
print('Current annotation sequences:', len(manifest), 'unique sources:', manifest.source_id.nunique())
'''
LOAD='''
from aaai_gait.workflow import prepare_corpus
x, meta, provenance = prepare_corpus(settings)
print('Window tensor:', x.shape, '[windows, frames, selected joints, XY]')
display(meta.groupby(['split','label']).agg(windows=('sequence_id','size'), sequences=('sequence_id','nunique'), sources=('source_id','nunique')))
'''
VIDEO='''
video_row = show_video_context(manifest, frame_number_base=settings.frame_number_base)
'''
def start(title,goals):
    return [md(f'''# {title}\n\n{goals}\n\n## Run this notebook independently\n\nFrom `experiments/aaai`, run:\n\n```bash\nuv sync\nuv run jupyter lab\n```\n\nSelect the kernel from this environment. Python 3.12 and all notebook libraries are declared in `pyproject.toml`. Every notebook repeats its setup and loads its own inputs. No variable from another notebook is required.\n\nA fresh installation defaults to `AAAI_DEMO=1` for a small synthetic software check. This prepared workspace uses `AAAI_DEMO=0` after real-data ingestion and validation. Its condition parameters are invented and its scores are not research evidence. Set `AAAI_DEMO=0` in the repository root `.env` after the ingestion workflow has produced a complete pose corpus. All notebooks use the same root `.env`. No API key is needed for JEPA or Random Forest training. Restart the kernel after changing configuration.\n\nOnly the root environment file contains credentials. `load_env` and `find_env` below are aliases for the real `python-dotenv` functions, `load_dotenv` and `find_dotenv`.'''),code(SETUP),md('## Connect the experiment to walking\n\nWatch a short annotated normal training sequence when local media is available. The preview is an observed video. Synthetic arrays used below are a separate software demonstration.'),code(VIDEO)]
def finish(cells,filename):
    from build_iteration02_notebooks import navigation_cell
    cells.insert(1,navigation_cell())
    cells.append(md('''## What to record before a paper run\n\nSave the current annotation hash, source assignments, retained windows, exclusions, software versions, model configuration, seed, and exact command. Keep synthetic artifacts separate. Revisit the video when an apparent anomaly could instead be a crop error, occlusion, or viewpoint change.\n\nThe [research review](docs/literature_review.md) explains source methods and adaptations. The [paper plan](docs/paper_plan.md) defines claims, ablations, and limits.'''))
    book=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (aaai-gait)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    nbf.validate(book);nbf.write(book,BASE/filename)

cells=start('00. Research question, sources, and current GAVD inventory','We will define normal-only anomaly detection, distinguish it from supervised condition recognition, and count the current annotations before training anything.')
cells += [md('''## 1. Define the unit of evidence\n\nA source video may contain several walking sequences. A sequence may provide many overlapping windows. These windows share a person, camera, and recording conditions. The experiment therefore splits source video IDs before building windows. A source ID still does not guarantee a unique participant.\n\nThe five labels are `normal`, `parkinsons`, `stroke`, `cerebral palsy`, and `myopathic`. Repeated mentions of stroke create no extra class. Myopathic gait is principally associated with muscle disease, so the paper should describe a neurological and neuromuscular cohort rather than imply that every label has the same neurological cause.\n\n![Research tasks](images/research_map.svg)'''),code('''
summary=manifest.groupby('label').agg(sequences=('sequence_id','size'), annotation_rows=('n_frames','sum'), source_videos=('source_id','nunique')).reindex(LABELS)
display(summary)
print('Selected rows:', manifest.n_frames.sum())
from aaai_gait.data import manifest_digest
print('Fresh annotation SHA256:', manifest_digest(manifest))
'''),md('''## 2. Understand the starting architecture\n\nSkeleton S-JEPA predicts teacher features at hidden joint-time tokens. An online encoder sees visible tokens. A predictor receives those context features and learned positional queries. A slowly updated target encoder sees the complete clean skeleton and supplies stopped-gradient targets.\n\nOur adaptation retains centered and sharpened target distributions with cross-entropy. It changes masking to uniform sampling, adds VICReg, uses a small laptop model, and pools the online encoder for downstream classification. The original paper uses a different benchmark setup and target-encoder readout. We cannot claim an exact reproduction.\n\nMAMP predicts motion targets rather than teacher latent distributions. Its published default favors motion, while its official code also contains a uniform random masking option. StoP adds positional noise. The 2026 stochastic-latent JEPA learns a prior and posterior over latent variables. These are three distinct ideas.\n\n![Randomness](images/stochasticity.svg)'''),md('''## 3. Inspect source separation\n\nThe splitter uses label and source counts only. It never searches for a split that produces a favorable model score. Freeze these assignments before extraction and record every exclusion afterwards. Current class proportions differ from a clinic's prevalence, so a probability learned here should not be interpreted as a clinical risk.'''),code('''
display(manifest.groupby(['split','label']).agg(sequences=('sequence_id','size'),sources=('source_id','nunique')))
assert manifest.groupby('source_id').split.nunique().max() == 1
'''),md('''## 4. See why more data can still leave uncertainty\n\nThe current subset contains 666 sequences, 140,641 annotation rows, and 103 source IDs. Only 11 source videos contribute Parkinson's examples and 11 contribute cerebral-palsy examples. Frames increase temporal detail but do not create independent people. The README explains missing participant metadata, recording bias, and the need for an independent cohort.\n\n![Generalizability](images/generalization.svg)\n\nThe [ASTAD4 site](https://ismart.ece.mcgill.ca/astad4/) lists November 20, 2026 as the deadline and permits seven content pages plus two reference pages. A successful submission needs measured evidence and careful limits, not just a working model.''')]
finish(cells,'00_research_and_gavd_inventory.ipynb')

cells=start('01. Download, extract, and inspect walking sequences','We will trace each skeleton back to a current CSV, decode the annotated person, check frame timing, and produce reproducible pose caches. This notebook can preview ingestion independently of model training.')
cells += [md('''## 1. Keep one source video in one place\n\nEvery new download goes to `data/videos-gavd/<source_id>/<source_id>.mp4` or the source container's extension. Existing valid flat files in `data/videos-gavd` are reused. A clip with two conditions is still one source. We do not duplicate it into class folders.\n\nDownload failures are records, not permission to replace the dataset with whatever happens to be cached. The CSV manifest is rebuilt from the five selected folders each time.\n\n![Ingestion](images/data_pipeline.svg)'''),code('''
from aaai_gait.data import download_sources, find_video
# A preview is deliberately small. Use scripts/run_pipeline.py ingest for the full cohort.
if not settings.demo:
    limit = int(os.getenv('AAAI_DOWNLOAD_LIMIT', '3'))
    preview_manifest=manifest[manifest.label.eq('normal') & manifest.split.eq('train')]
    preview_sources=preview_manifest.drop_duplicates('source_id').sort_values('source_id').head(limit)
    if all(find_video(ROOT,source) is not None for source in preview_sources.source_id):
        print('The preview training sources are already locally decodable.')
    else:
        download_report = download_sources(preview_manifest, ROOT, limit=limit)
        display(download_report[['source_id','status','error']])
    video_row=show_video_context(manifest,frame_number_base=settings.frame_number_base)
else:
    print('Demo mode: no downloads. Existing local videos may still be previewed.')
'''),md('''## 2. Verify the annotated frame and person\n\nGAVD stores source frame numbers, an image size, and a person bounding box. The repository adapter treats frame numbers as one-based. OpenCV uses zero-based decoded positions, so frame 1 maps to decoded index 0. The configurable frame base is recorded in the pose cache. Verify several actual overlays before a large run.\n\nBounding boxes must scale when a downloaded video has a different resolution. We crop the annotated box with a small margin and restore detected points to image coordinates. A crop helps associate the pose with the annotated person, but it cannot guarantee identity when people overlap.'''),code('''
from aaai_gait.media import annotated_frame
if video_row is not None:
    annotated_frame(video_row, frame_number_base=settings.frame_number_base)
    plt.show()
else:
    print('An overlay becomes available after the first successful download.')
'''),md('''## 3. Extract with ambient\n\n`SequenceKeypointExtractor` supplies the MediaPipe model and extracts each annotated crop. Raw coordinates, confidence, source FPS, decoded-frame flags, and source frame indices are retained. Cache provenance includes the CSV, video, pose-model hash, visibility threshold, and extraction settings. A stale cache raises an error instead of silently reusing it.\n\nAll 33 raw points can be stored for inspection. The learning pipeline selects only the declared 16-joint whitelist. Confidence defines observation validity and is not supplied as a numerical encoder feature. Missingness can still be a shortcut, so the capstone includes a pose-availability control.'''),code('''
from aaai_gait.data import extract_sequence, load_pose
if not settings.demo and video_row is not None:
    pose_path=extract_sequence(video_row, find_video(ROOT,video_row['source_id']), settings.pose_dir,
                               frame_number_base=settings.frame_number_base)
    pose=load_pose(pose_path)
    print('Frames:',len(pose['frame_numbers']),'FPS:',float(pose['fps']))
    print('Decoded fraction:',pose['decoded'].mean())
    print('Selected-joint visibility fraction:',(pose['confidence'][:,[11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]]>=.5).mean())
else:
    print('Extraction is enabled by AAAI_DEMO=0. No cached MS dataset is read.')
'''),md('''## 4. Preserve time and audit exclusions\n\nThe common corpus uses 15 Hz, 32-frame windows, and a 16-frame stride. Interpolation only joins adjacent observed source frames. Sources below the requested sample rate are excluded instead of upsampled. It never bridges a missing annotation gap. Each four-frame joint token is valid only when its coordinates are observed throughout the patch. A retained window needs at least 75% valid tokens and at least 50% valid tokens in each half. Missing values remain NaN and cannot become teacher targets. This gate can still reject substantial data. Report attrition by class and source, including partially rejected sequences.\n\nEach frame is centered on its pelvis. Scale comes from median torso length in the first half of the window, so future torso measurements cannot change the observed prefix. This is relative image-plane motion, not physical walking speed or depth.\n\nTo ingest all selected videos locally:\n\n```bash\nuv run python scripts/run_pipeline.py ingest\nuv run python scripts/run_pipeline.py audit\n```\n\nThe first command can take hours and encounters unavailable media. The second reports retained data and refuses to start a five-class evaluation if any split has lost a class. Do not change the split after inspecting performance.'''),code('''
from aaai_gait.data import load_windows
preview_x,preview_meta=load_windows(manifest,settings.pose_dir,frames=settings.frames,stride=settings.stride,target_fps=settings.target_fps,require_coverage=False,
                                   expected_frame_number_base=settings.frame_number_base)
print('Locally available valid windows:',len(preview_x))
display(pd.DataFrame(preview_meta.attrs['exclusions']).groupby(['label','reason']).size().rename('sequences').reset_index() if preview_meta.attrs['exclusions'] else pd.DataFrame())
''')]
finish(cells,'01_video_pose_and_quality.ipynb')

cells=start('02. Joint whitelist, tokens, and uniform masks','We will build the exact model input, verify the deduplicated joint list, and check that masking never depends on movement intensity.')
cells += [md((BASE/'docs/joint_table.md').read_text()),md('![Selected joints](images/joint_whitelist.svg)'),code(LOAD),code('''
from aaai_gait.model import JOINT_INDICES, ModelConfig, SJEPA, uniform_mask
from aaai_gait.data import JOINTS
assert tuple(JOINTS)==JOINT_INDICES
assert list(JOINT_INDICES)==sorted(set(JOINT_INDICES))
display(pd.DataFrame({'compact_position':range(16),'BLAZEPOSE_33 index':JOINT_INDICES}))
plot_motion(x[0],title='Synthetic software check' if settings.demo else 'Retained GAVD motion');plt.show()
'''),md('''## 1. Build a temporal patch for one joint\n\nThe input tensor has shape `[B, 32, 16, 2]`. Four consecutive XY pairs for the same joint become eight numbers. A learned linear map converts those eight numbers into a 64-dimensional token. Eight time patches times 16 joints gives 128 tokens. Learned temporal and joint embeddings retain identity.\n\n![Tokens](images/tokenization.svg)'''),code('''
cfg=ModelConfig()
model=SJEPA(cfg)
batch=torch.as_tensor(x[:4])
tokens=model.view_encoder.tokenize(batch)
print('Input:',tuple(batch.shape),'tokens:',tuple(tokens.shape))
assert tokens.shape==(4,128,64)
'''),md('''## 2. Sample without reading the skeleton\n\n`uniform_mask` accepts batch size, token count, ratio, and a random generator. It accepts no coordinates, label, velocity, or motion-intensity argument. At a 0.5 ratio, 64 positions per example are hidden without replacement. Every permitted joint-time token has the same marginal selection probability.\n\nMasking a token hides four frames of one joint from the context encoder. The complete target encoder sees all observed tokens in the clean sequence. Missing patches remain invalid and cannot become target labels. A joint can be entirely hidden in one draw, but no joint is permanently reserved as context.\n\n![Uniform masks](images/uniform_mask.svg)'''),code('''
generator=torch.Generator().manual_seed(settings.seed)
masks=uniform_mask(2000,cfg.tokens,.5,generator=generator)
print('Hidden count per example:',masks.sum(1).unique().tolist())
frequency=masks.float().mean(0).reshape(cfg.time_tokens,cfg.joints).numpy()
plt.figure(figsize=(10,3));plt.imshow(frequency,aspect='auto',vmin=.4,vmax=.6,cmap='viridis')
plt.xticks(range(16),JOINT_INDICES);plt.xlabel('BlazePose joint index');plt.ylabel('Time patch');plt.colorbar(label='Empirical hidden fraction');plt.title('Uniform-mask software check');plt.show()
'''),md('''## 3. Understand positional stochasticity\n\nOur optional StoP-inspired mode adds Gaussian noise to hidden predictor queries through the same learned projection that maps context features. It does not shuffle joint identities or perturb teacher coordinates. The default model uses deterministic learned positions. Inference switches positional noise off.\n\nThis is an ablation to evaluate, not an assumed improvement. Compare identical splits and seeds with `spe=False` and `spe=True`. Read the [literature review](docs/literature_review.md) before describing it as the original StoP implementation.'''),code('''
spe_model=SJEPA(ModelConfig(spe=True))
mask=uniform_mask(len(batch),cfg.tokens,generator=torch.Generator().manual_seed(9))
spe_model.eval()
a=spe_model.embed(batch);b=spe_model.embed(batch)
assert torch.equal(a,b)
print('Evaluation embeddings are deterministic.')
''')]
finish(cells,'02_joint_tokens_and_uniform_masks.ipynb')

cells=start('03. Train the normal-only skeleton JEPA','We will follow one prediction through the context encoder, teacher, and loss, then train a protected normal-only reference and inspect collapse diagnostics.')
cells += [code(LOAD),md('''## 1. Separate context, target, and predictor\n\nThe online encoder receives visible tokens only. Attention therefore cannot read hidden coordinates. Invalid patches are also excluded as attention keys, and the loss uses valid hidden targets only. The predictor restores the time-joint grid using learned queries and predicts a distribution over feature channels. The full-input teacher supplies the target distribution without receiving gradients.\n\n![Branches](images/jepa_branches.svg)\n\nFor token `i`, let `h_i` be the teacher feature and `p_i` the predictor output. The target is `softmax((h_i - center) / target_temperature)`. The prediction is `softmax(p_i / student_temperature)`. Cross-entropy is averaged only over hidden positions. The moving center discourages a persistent dominant teacher channel.'''),code('''
from aaai_gait.model import ModelConfig, SJEPA
from aaai_gait.training import seed_everything
seed_everything(settings.seed)
example=SJEPA(ModelConfig())
normal_train=(meta.split.eq('train')&meta.label.eq('normal')).to_numpy()
example_batch=torch.as_tensor(x[normal_train][:4])
losses=example.loss(example_batch,generator=torch.Generator().manual_seed(3))
print({k:round(float(losses[k].detach()),4) for k in ['loss','prediction','vicreg','projection_std','embedding_std']})
losses['loss'].backward()
assert all(p.grad is None for p in example.target_encoder.parameters())
print('Teacher gradients are absent.')
'''),md('''## 2. Add anti-collapse regularization\n\nTwo randomly masked views of the same walking window produce pooled online features. An unnormalized projector feeds VICReg. Its invariance term encourages view agreement. Its variance term penalizes dimensions with low batch variation. Its covariance term reduces redundant dimensions.\n\nThe loss is `prediction + 0.1 * VICReg`, with internal invariance and variance weights of 25 and covariance weight of 1. These are starting settings. VICReg is not a supervised repulsion between clinical classes. A small prediction loss alone does not prove useful learning.\n\n![VICReg](images/vicreg.svg)'''),md('''## 3. Run normal-only pretraining\n\nThe training function selects only rows with `split == train` and `label == normal`. Source-balanced sampling prevents a long video from dominating. Each optimizer step updates the online network first, then performs `teacher = 0.996 * teacher + 0.004 * online`. Target-center statistics come from training batches only.\n\nThe helper stores configuration, source split identity, and data identity. It refuses to overwrite an existing anchor. Reopening the notebook reuses a compatible checkpoint. A new corpus or implementation produces a new artifact directory.'''),code('''
from aaai_gait.workflow import normal_anchor
anchor,history=normal_anchor(x,meta,provenance,settings)
display(pd.DataFrame(history))
print('Training sources used:',meta.loc[normal_train,'source_id'].nunique())
print('Artifacts:',provenance['run_dir'])
'''),md('''## 4. Check the actual encoder\n\nInspect both the projector's standard deviation and the encoder's variation. A projector can hide problems in the representation it receives. Effective rank describes how many directions carry variation, but it is bounded by the number of examples minus one. Neither rank nor a visually separated plot establishes clinical validity.'''),code('''
from aaai_gait.training import embed_windows, representation_diagnostics
train=meta.split.eq('train').to_numpy()
z=embed_windows(anchor,x[train],batch_size=settings.batch_size)
display(representation_diagnostics(z))
plt.figure(figsize=(7,3));plt.plot([row['prediction'] for row in history],label='Latent cross-entropy');plt.plot([row['vicreg'] for row in history],label='VICReg');plt.xlabel('Epoch');plt.legend();plt.title('Synthetic check' if settings.demo else 'Normal training losses');plt.show()
'''),md('''## 5. Know what this model predicts\n\nThe predictor outputs latent features. It does not yet generate joint coordinates or a video of a future gait. Normal-only masked completion also differs from a dedicated past-to-future forecasting model. Notebook 05 makes that distinction executable. The frozen online encoder supplies features for the condition probes in later notebooks.''')]
finish(cells,'03_normal_only_jepa_and_vicreg.ipynb')

cells=start('04. Add conditions while retaining a normal anchor','We will continue representation learning with additional conditions, replay normal data, and use validation labels to compare frozen feature readouts.')
cells += [code(LOAD),md('''## 1. Define the progressive curriculum\n\nThe initial model learns only normal training sources. A copy then learns from normal plus Parkinson's data, followed by stroke, cerebral palsy, and myopathic data. Each stage keeps all earlier conditions. Every training batch includes normal replay and samples sources within labels. Labels regulate sampling, but the JEPA and VICReg losses themselves use no class targets.\n\nThe order is a research hypothesis. It is not justified by a presumed severity ranking. Compare it with pooled training and reversed order before claiming a curriculum advantage.\n\n![Curriculum](images/curriculum.svg)'''),code('''
from aaai_gait.workflow import normal_anchor, progressive_model
anchor,normal_history=normal_anchor(x,meta,provenance,settings)
before={k:v.clone() for k,v in anchor.state_dict().items()}
adapted,stage_history=progressive_model(anchor,x,meta,provenance,settings)
assert all(torch.equal(v,before[k]) for k,v in anchor.state_dict().items())
display(pd.DataFrame(stage_history)[['stage','epoch','prediction','vicreg','embedding_std']])
print('The normal anchor is unchanged.')
'''),md('''## 2. Train a separate condition readout\n\nA linear probe turns a complete observed window into five condition probabilities. The encoder stays frozen. Standardization and logistic-regression weights use training data only. We compare three regularization strengths on validation sources, aggregate overlapping windows into sequences, and give each source equal total metric weight.\n\nHere we compare normal and final progressive encoders. The capstone selects a JEPA stage using validation macro-F1 before reading test performance. The original skeleton S-JEPA uses a teacher readout; our online readout is a declared adaptation.'''),code('''
from aaai_gait.training import embed_windows
from aaai_gait.workflow import select_probe
comparisons=[]
for name,encoder in [('normal',anchor),('progressive',adapted)]:
    z=embed_windows(encoder,x,batch_size=settings.batch_size)
    probe,selection=select_probe(z,meta,settings.seed)
    comparisons.append({'stage':name,**selection})
display(pd.DataFrame(comparisons))
'''),md('''## 3. Keep anomaly detection anchored\n\nAfter learning an abnormal pattern, a predictor may find it less surprising. That is useful for condition recognition but can weaken normal-only anomaly detection. The normal anchor remains a separate checkpoint and supplies the main anomaly score. The adapted encoder is evaluated as a supervised feature extractor. Do not describe its error as an untouched normality reference.'''),md('''## 4. Decide when stochastic latent variables are warranted\n\nNormal gait has several plausible continuations from a short prefix. A deterministic predictor can average incompatible futures. The 2026 stochastic-latent JEPA paper introduces a training posterior `q(z | context, target)`, a test-time prior `p(z | context)`, a predictor conditioned on `z`, and a KL penalty aligning posterior and prior. Sampling the prior can then produce diverse latent completions.\n\nThis tutorial implements uniform masking and the optional StoP-inspired positional-noise ablation. It does not implement that separate stochastic-latent architecture. Adding independent noise without a learned prior, posterior, and calibration experiment would not reproduce the paper. A defensible extension would compare expected predictive error, uncertainty-error correlation, diversity, and representation quality across matched compute budgets.\n\nKeep this extension secondary for a near-deadline paper. The primary claims can be tested with the deterministic normal anchor.\n\n![Stochasticity](images/stochasticity.svg)''')]
finish(cells,'04_progressive_condition_adaptation.ipynb')

cells=start('05. Forecasting, anomaly scores, and local explanations','We will train a separate past-to-future model, calibrate a normal-only score, and inspect which time-joint targets contribute to a prediction error.')
cells += [code(LOAD),md('''## 1. Separate completion from forecasting\n\nRandom masked completion can observe both earlier and later frames. A forecasting experiment hides every joint in the future half. The online encoder receives only the past half. Learned future time and joint positions are allowed because they do not contain future coordinates. The teacher sees all observed tokens in the window to construct training targets. Missing patches are excluded from its attention keys and the prediction loss.\n\nThe shared preprocessing estimates scale from the observed first half. Cohort admission also checks future observation coverage, so this is retrospective evaluation on evaluable futures rather than a streaming admission policy. This prevents future torso dimensions from entering past inputs. Prefix normalization assumes the configured forecast boundary remains at half the window. Changing that boundary requires updating preprocessing as well.\n\n![Forecasting](images/forecast.svg)'''),code('''
from aaai_gait.workflow import normal_anchor
from aaai_gait.model import forecast_mask
forecast_model,history=normal_anchor(x,meta,provenance,settings,mode='forecast')
mask=forecast_mask(1,forecast_model.cfg).reshape(8,16)
assert not mask[:4].any() and mask[4:].all()
print('Observed patches:',(~mask).any(1).nonzero().ravel().tolist())
display(pd.DataFrame(history))
'''),md('''## 2. Define predictive surprise\n\nFor each hidden token, we compare the target and predicted feature distributions using `KL(target || prediction)`. This subtracts the target entropy from cross-entropy, so a broad teacher distribution does not create a large score by itself. Masked completion averages a fixed bank of eight uniform masks. Every window uses the same bank, making scores independent of batching order.\n\nForecast scoring hides the same complete suffix on each call. We average window scores within sequence and then within each source-label pair. If a source has several labels, its rows remain dependent and should stay together in uncertainty calculations. Residuals are an experimental signal, not established anomaly probabilities.'''),code('''
from aaai_gait.workflow import anomaly_evaluation
anchor,_=normal_anchor(x,meta,provenance,settings)
source_scores,metrics=anomaly_evaluation(anchor,x,meta,batch_size=settings.batch_size,alpha=.05)
display(pd.Series(metrics))
forecast_scores,forecast_metrics=anomaly_evaluation(forecast_model,x,meta,batch_size=settings.batch_size,alpha=.05,mode='forecast')
from aaai_gait.robustness import anomaly_bootstrap
display(pd.DataFrame([{'method':'masked',**metrics,**anomaly_bootstrap(source_scores,repeats=200)},
                      {'method':'forecast',**forecast_metrics,**anomaly_bootstrap(forecast_scores,repeats=200)}]))
from aaai_gait.workflow import write_json
from aaai_gait.normal_distance import normal_distance
distance_scores,distance_metrics=normal_distance(anchor,x,meta,batch_size=settings.batch_size)
display(pd.Series(distance_metrics))
source_scores.to_csv(provenance['run_dir']/'masked_anomaly_sources.csv',index=False)
forecast_scores.to_csv(provenance['run_dir']/'forecast_anomaly_sources.csv',index=False)
write_json(provenance['run_dir']/'forecast_notebook_metrics.json',forecast_metrics)
'''),md('''## 3. Compare a simple normal-space baseline\n\nThe notebook also fits a shrinkage covariance model to one mean embedding per normal training source. Squared Mahalanobis distance measures departure from that reference. No abnormal source trains the reference, and normal validation sources set the same finite-sample threshold. This baseline uses a completed observed sequence; it is not a future-coordinate generator. Source aggregation gives long recordings the same fitting weight as short ones.\n\n## 4. Understand a small calibration set\n\nThe threshold uses only normal validation sources. The finite-sample upper rank is `ceil((n + 1) * (1 - alpha))`. If the rank exceeds `n`, the threshold is infinity. At a nominal 5% false-alarm target, at least 19 exchangeable normal calibration observations are needed for a finite threshold. The default real split has only six normal validation sources.\n\nAn infinite threshold means that this calibration guarantee is unsupported at the chosen resolution. It is not evidence that the detector recognizes every normal gait. Report ranking metrics, calibration size, and this limitation. An empirical 95th percentile can be reported only as an exploratory operating point with uncertain false-alarm behavior. Do not tune a threshold on test labels.\n\n![Calibration](images/anomaly.svg)'''),code('''
# Keep the test table hidden during development. Inspect training/validation distributions.
visible=source_scores[source_scores.split.isin(['train','val'])]
fig,ax=plt.subplots(figsize=(8,3))
for label,rows in visible.groupby('label'):ax.scatter([label]*len(rows),rows.score,alpha=.65,label=label)
ax.set_ylabel('Latent KL surprise');ax.tick_params(axis='x',rotation=20);ax.set_title('Synthetic check' if settings.demo else 'Development-source scores');plt.show()
'''),md('''## 5. Localize mismatch without claiming causation\n\nA heatmap describes where the model made larger prediction errors. A bright knee region can arise from unusual motion, an occluded knee, a pose-estimator error, or a camera change. Inspect the walking video and pose confidence alongside the heatmap. It is not a diagnosis or evidence that a joint caused the condition.'''),code('''
from aaai_gait.model import JOINT_INDICES
chosen=np.flatnonzero(meta.split.eq('val').to_numpy())[0]
residual=forecast_model.latent_residual(torch.as_tensor(x[chosen:chosen+1]),mode='forecast')
from aaai_gait.media import show_window_context
show_window_context(meta.iloc[chosen],manifest)
heat=residual['per_token'][0].numpy()
fig,ax=plt.subplots(figsize=(10,3));im=ax.imshow(np.ma.masked_invalid(heat),aspect='auto',cmap='magma')
ax.set_xticks(range(16),JOINT_INDICES);ax.set_xlabel('BlazePose joint index');ax.set_ylabel('Time patch');ax.set_title('Forecast error on a validation example');fig.colorbar(im,label='Latent KL');plt.show()
plot_motion(x[chosen],title='Synthetic validation example' if settings.demo else 'Validation skeleton');plt.show()
''')]
finish(cells,'05_forecast_anomalies_and_explanations.ipynb')

cells=start('06. Capstone: Random Forest versus skeleton JEPA','We will fit both approaches on the same retained gait windows, select settings on validation sources, and evaluate prespecified methods on held-out sequences.')
cells += [code(LOAD),md('''## 1. Rebuild the classical baseline on this cohort\n\nThe exp5 notebooks motivate a Random Forest on gait-angle summaries. Here we recompute all features from the exact same windows used by JEPA. We do not reuse exp5's saved feature table or historical scores because those involve different data and protocols.\n\nThe baseline contains 36 measured values: 15 core summaries of left/right hip, knee, and ankle angles; six angle standard deviations; 12 wrist/ankle motion summaries; and three ankle-separation/torso-orientation summaries. The optional `build_ambient_features` adapter reproduces the 15-field ambient core-angle vector. We do not advertise an 82-feature vector populated with uncomputed zeros.\n\n![Comparison](images/evaluation.svg)'''),code('''
from aaai_gait.features import build_features
features,feature_names=build_features(x,settings.target_fps)
print('Feature matrix:',features.shape)
display(pd.DataFrame({'feature':feature_names}))
'''),md('''## 2. Keep fitting and selection separate\n\nThe Random Forest pipeline fits its median imputer on training data only. It compares maximum tree depths of 4, 8, and unrestricted on validation macro-F1, using 300 trees and a fixed seed. The JEPA probes compare regularization strengths 0.1, 1, and 10. Every method sees the same train/validation/test source membership.\n\nThe condition task includes all five labels during supervised readout fitting, including the normal-only encoder's probe. Normal-only describes encoder pretraining, not an unsupervised five-way classifier. The separate anomaly task in notebook 05 uses no abnormal labels to learn its reference or threshold.\n\nWe also compare an untrained encoder, mean pose, pose availability with camera view, and a training-prior baseline. These controls test whether learned dynamics add value beyond architecture, static posture, and class frequency.'''),code('''
from aaai_gait.workflow import capstone
result=capstone(x,meta,provenance,settings,bootstrap_repeats=200 if settings.demo else 1000)
rows=[]
for method,record in result['methods'].items():
    rows.append({'method':method,**{k:record['source_weighted'][k] for k in ['macro_f1','balanced_accuracy','abnormal_auroc','abnormal_average_precision']}})
display(pd.DataFrame(rows))
print('JEPA selected using validation:',result['selected_jepa'])
print('Interpretation:',result['interpretation'])
'''),md('''## 3. Read paired uncertainty correctly\n\nAverage overlapping-window probabilities inside each annotated sequence. Report both sequence-weighted and source-weighted metrics. The latter gives each source equal total weight across its sequences.\n\nThe bootstrap resamples complete source groups, using exactly the same sampled groups for both methods in a paired comparison. Its reported differences are sequence-weighted. Replicates missing a prespecified class are counted and omitted. With few test sources, many replicates may be invalid and intervals may be unavailable. These intervals do not include training-seed uncertainty, and source groups do not prove participant independence.'''),code('''
display(result['paired_selected_minus_rf_sequence_weighted'])
chosen=result['selected_jepa']
record=result['methods'][chosen]['source_weighted']
confusion=np.asarray(record['confusion_matrix'])
fig,ax=plt.subplots(figsize=(7,5));image=ax.imshow(confusion,cmap='Blues')
ax.set_xticks(range(5),LABELS,rotation=35,ha='right');ax.set_yticks(range(5),LABELS)
ax.set_xlabel('Predicted label');ax.set_ylabel('Dataset label');ax.set_title('Synthetic software check' if settings.demo else 'Source-weighted test confusion')
for i in range(5):
    for j in range(5):ax.text(j,i,f'{confusion[i,j]:.1f}',ha='center',va='center')
fig.colorbar(image);fig.tight_layout();plt.show()
'''),md('''## 4. Predict one held-out gait sequence\n\nA video must pass the same annotation-aware pose extraction, temporal resampling, and QC before either classifier receives it. The example below fits a frozen probe using training rows, then predicts all windows of one test sequence and averages their probabilities. The Random Forest follows the identical interface using `build_features`. For a new unannotated video, a human-reviewed person crop and frame interval must first be supplied. Do not classify a random bystander or silently assume a new video belongs to a known condition.'''),code('''
from aaai_gait.workflow import normal_anchor, progressive_model, select_probe
from aaai_gait.training import embed_windows
anchor,_=normal_anchor(x,meta,provenance,settings)
encoder=anchor if chosen=='normal_jepa' else progressive_model(anchor,x,meta,provenance,settings)[0]
z=embed_windows(encoder,x,batch_size=settings.batch_size)
probe,_=select_probe(z,meta,settings.seed)
sequence_id=meta.loc[meta.split.eq('test'),'sequence_id'].iloc[0]
sequence_rows=meta.sequence_id.eq(sequence_id).to_numpy()
from aaai_gait.media import show_window_context
show_window_context(meta.loc[sequence_rows].iloc[0],manifest)
p=probe.predict_proba(z[sequence_rows]).mean(0)
display(pd.Series(p,index=probe.classes_,name=f'Observed sequence {sequence_id}'))
'''),md('''## 5. Run multiple prespecified splits\n\nUse a fixed list of seeds chosen before examining results. Each run retrains normal and progressive encoders from scratch within its training partition. Reusing an encoder trained on all sources would leak test motion even without labels. Repeated split scores are dependent because sources recur. Report the distribution across seeds and keep source-cluster intervals within each run.\n\n```bash\nuv run python scripts/run_pipeline.py train --seeds 42 43 44\n```\n\nThe results are an internal GAVD comparison, not a direct numerical reproduction of exp5 and not evidence of diagnostic accuracy.''')]
finish(cells,'06_capstone_rf_vs_jepa.ipynb')

cells=start('07. Ablations, representation checks, and the paper evidence ledger','We will inspect representations without fitting on held-out data, define a manageable ablation plan, and distinguish completed checks from evidence still needed for a paper.')
cells += [code(LOAD),md('''## 1. Fit visualization on training data\n\nA two-dimensional projection can help find outliers and source clusters, but it cannot prove meaningful class separation. PCA is fitted on training embeddings only. Validation embeddings are transformed with that fitted map. Test plots should be produced only after the analysis has been locked.'''),code('''
from aaai_gait.workflow import normal_anchor
from aaai_gait.training import embed_windows, representation_diagnostics
from sklearn.decomposition import PCA
anchor,_=normal_anchor(x,meta,provenance,settings)
z=embed_windows(anchor,x,batch_size=settings.batch_size)
train=meta.split.eq('train').to_numpy();val=meta.split.eq('val').to_numpy()
pca=PCA(n_components=2).fit(z[train]);points=pca.transform(z[val])
fig,ax=plt.subplots(figsize=(7,4))
for label in LABELS:
    keep=meta.loc[val,'label'].to_numpy()==label
    ax.scatter(points[keep,0],points[keep,1],label=label)
ax.legend(fontsize=8);ax.set_title('Synthetic validation projection' if settings.demo else 'Validation features in training-fitted PCA');plt.show()
display(representation_diagnostics(z[train]))
'''),md('''## 2. Predefine ablations before reading their test scores\n\n|Comparison|Fixed elements|Question|\n|---|---|---|\n|Uniform JEPA with and without VICReg|Sources, seed, steps, joint whitelist|Does the encoder retain useful variance?|\n|Deterministic positions versus StoP-inspired noise|Sources, seeds, compute budget|Does positional noise improve held-out features?|\n|Normal anchor versus progressive adaptation|Readout labels and validation procedure|Does adaptation help condition recognition?|\n|Progressive versus pooled training|Total optimizer steps and label exposure|Does stage order matter beyond compute?|\n|Masked completion versus suffix forecasting|Temporal scale and normal sources|Does future prediction add anomaly information?|\n|Untrained encoder and mean pose controls|Same classifiers and splits|Does learned temporal structure add value?|\n\nThe main pipeline implements the normal/progressive comparison and simple controls. The optional SPE training call below implements a position-noise candidate. Pooled, reversed-order, and compute-matched schedules remain additional experiments to run and report, not completed results.'''),code('''
# Optional locally configured ablation. Leave off for the short walkthrough.
if os.getenv('AAAI_RUN_SPE_ABLATION','0')=='1':
    spe_anchor,spe_history=normal_anchor(x,meta,provenance,settings,spe=True)
    print(pd.DataFrame(spe_history).tail(1))
else:
    print('SPE ablation not run. Enable AAAI_RUN_SPE_ABLATION=1 in the root .env.')
'''),md('''## 3. Test robustness on development data first\n\nPose jitter is a technical perturbation, not a disease simulation. The example measures how a small coordinate perturbation changes validation embeddings. Camera view, lower frame rate, missing joints, and person identity require separate controlled audits. The validity gate rejects windows with too little observed motion. In retained windows, missing tokens are excluded from attention and target loss. The pose-availability control combines per-joint and per-time-patch validity with training-fitted camera-view indicators. It checks whether recording quality and view can predict labels.'''),code('''
rng=np.random.default_rng(23)
perturbed=x[val]+rng.normal(0,.01,x[val].shape).astype(np.float32)
original=embed_windows(anchor,x[val],batch_size=settings.batch_size)
changed=embed_windows(anchor,perturbed,batch_size=settings.batch_size)
print('Mean validation embedding displacement after 0.01 torso-unit noise:',np.linalg.norm(changed-original,axis=1).mean())
'''),md('''## 4. Assemble an honest evidence ledger\n\nA notebook that executes demonstrates an implementation. It does not demonstrate anomaly-detection performance. Save separately: the full annotation inventory, downloaded media inventory, pose exclusions, final retained cohort, split registry, checkpoints, validation selection, per-sequence predictions, and statistical summaries.\n\n|Evidence|Meaning|\n|---|---|\n|Synthetic notebook execution|Software paths run with explicit toy data|\n|Model invariant tests|Selected leakage, gradient, and masking properties hold|\n|Real-media extraction pilot|The ambient integration can process some actual annotations|\n|Full GAVD evaluation|Required before claiming within-dataset performance|\n|Independent cohort evaluation|Required for stronger generalization claims|\n\nThe paper should lead with normal-only anomaly detection, then present condition classification as supporting analysis. Myopathic gait broadens the cohort beyond exclusively neurological disease. Source labels, disease severity, participant identities, and filming bias limit interpretation. The [paper plan](docs/paper_plan.md) contains a claim-to-evidence map and the verified submission rules.'''),code('''
from aaai_gait.workflow import write_json
import importlib.metadata as im
versions={name:im.version(name) for name in ['torch','numpy','pandas','scikit-learn','mediapipe','opencv-python','python-dotenv']}
write_json(provenance['run_dir']/'software_versions.json',versions)
display(pd.Series(versions))
print('Use the exact run directory in the experiment ledger:',provenance['run_dir'])
''')]
finish(cells,'07_ablation_and_paper_evidence.ipynb')
print('Built eight self-contained tutorial notebooks')
