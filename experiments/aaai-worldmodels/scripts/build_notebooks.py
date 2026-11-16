"""Authoritative source for eight independently runnable teaching notebooks."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = '''from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display, Markdown, SVG
from worldmodels.settings import PROJECT, REPOSITORY, load_env, find_env
load_env(find_env())
torch.set_num_threads(2)
from worldmodels.training import seed_everything
from worldmodels.notebook_support import (data_mode, lesson_data, real_inputs, real_example,
    bounded_preview, lesson_directory, persist_lesson, prepared_path, comparison_path)
REAL = data_mode() == "real"
RELEASE_TEST_RESULTS = os.getenv("WM_RELEASE_TEST_RESULTS", "0") == "1"
EVIDENCE_LABEL = "Real GAVD" if REAL else "Synthetic teaching"
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats("svg")
seed_everything(27)
print("Data mode:", data_mode(), "| Prepared source:", prepared_path() if REAL else "synthetic fixture")
print("Project:", PROJECT.name, "| Python kernel ready")
'''
INTRO = '''Run this notebook independently from the experiment directory. No previous notebook execution is required. The environment installs this local `worldmodels` package and the repository's editable Ambient package together.

```bash
cd experiments/aaai-worldmodels
uv sync
uv run python -m ipykernel install --user --name medwm-gait --display-name "Med-WM gait (Python 3.12)"
uv run jupyter lab
```

Select **Med-WM gait (Python 3.12)**. The project requires Python 3.12. Place local settings in this project's ignored `.env`. The requested `load_env(find_env())` spelling uses aliases for `python-dotenv`; it resolves this experiment's `.env` explicitly and never displays credentials. These lessons need no paid model API.

Set `WM_DATA_MODE=real` to use only actual prepared GAVD skeletons and real MediaPipe/PyTorch/classifier calls. Real mode requires the immutable person-gated clean cohort and fails if those inputs are absent; it never substitutes synthetic data. It embeds a short local walking excerpt and persists numerical outputs and trained lesson checkpoints under a separately named real execution. Run all real lessons with `uv run python scripts/execute_notebooks.py --data-mode real`.

The default `synthetic` mode is an optional fast teaching fixture. Its motions are not medical simulations and cannot establish clinical accuracy. Real lesson training is also a bounded tutorial exercise, distinct from the primary controlled study. Keep all three evidence types separate.
'''


def md(text): return nbf.v4.new_markdown_cell(text.strip())
def code(text): return nbf.v4.new_code_cell(text.strip())
def diagram(name): return code(f'display(SVG(filename=str(PROJECT / "images/{name}.svg")))')
def build(name,title,description,body):
    context = [md("### Walking-video context\n\nReal mode embeds only a short original walking excerpt from an accepted normal training sequence. The preview is saved under `data/videos-gavd/previews/`, with its source hash and frame range. It provides context for this lesson without duplicating the full original. Synthetic mode enables the same real viewing aid only when `WM_EMBED_VIDEO=1`."), code("""if REAL or os.getenv("WM_EMBED_VIDEO", "0") == "1":
    from IPython.display import Video
    preview_path = bounded_preview()
    display(Video(str(preview_path), embed=True, html_attributes='controls preload="metadata"'))
    print("Bounded source excerpt:", preview_path.relative_to(REPOSITORY))
else:
    print("Synthetic mode: set WM_EMBED_VIDEO=1 for a real-video context excerpt.")""")]
    cells=[md('# '+title+'\n\n'+description),md('## Run locally\n\n'+INTRO),code(SETUP)]+context+body
    cells.append(code(f'lesson_evidence = persist_lesson("{Path(name).stem}", globals())\nprint("Evidence mode:", lesson_evidence.get("mode"))'))
    cells.append(md('## What to retain\n\nRecord the data identity, settings, executed commands, and limitations with any result. A successful cell execution is implementation evidence, not proof that the method generalizes. See the [data protocol](docs/data_protocol.md), [method review](docs/literature.md), and [README](README.md) before changing the study.'))
    book=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Med-WM gait (Python 3.12)','language':'python','name':'medwm-gait'},'language_info':{'name':'python','version':'3.12'}})
    nbf.validate(book)
    nbf.write(book,ROOT/name)


build('00_inventory_and_research_question.ipynb','00. Start with the research question and the real cohort',
'''We want to learn a reference for normal movement, predict what follows a short observed past, and investigate whether prediction errors and learned representations help identify gait deviations. This lesson separates that research question from a clinical diagnosis claim. By the end, you will have rebuilt the annotation inventory and checked a source-disjoint partition.''',[
md('''## Step 1. Decide what the experiment can answer

A skeleton is a time series of estimated landmarks. A predictive representation can learn recurring relationships between limbs without being given diagnostic labels. A classifier can then use the learned representation to predict the supplied dataset label. These are different tasks: useful classification does not prove forecasting, and accurate short-horizon forecasting does not prove clinical understanding.

The primary supervised comparison is normal versus the four selected non-normal labels, evaluated with source-level balanced accuracy. Five-class macro-F1 is secondary. The anomaly branch keeps a normal-only reference and uses future prediction errors. Folder labels remain annotations, not verified diagnoses.'''),diagram('cohort'),
md('''## Step 2. Rebuild from current CSVs

We read current per-sequence CSVs and exclude `data/gavd/source/`, which contains the same upstream rows in their original parts. Counting both would double-count observations. Each row below represents one annotated sequence, not one patient. The inventory checks schema, sequence identity, source identity, label consistency, and duplicate frame indexes.'''),
code('''from worldmodels.data import build_inventory, assign_splits, LABELS
inventory, audit = build_inventory(REPOSITORY)
display(pd.DataFrame(audit["by_label"]).T)
print({key: audit[key] for key in ("all_sequences", "all_rows", "all_sources", "selected_sequences", "selected_sources")})
assert tuple(LABELS) == ("normal", "parkinsons", "stroke", "cerebral palsy", "myopathic")'''),
md('''## Step 3. Count the independent units

The selected cohort contains 666 sequences from 103 YouTube sources before media attrition. There are 140,641 annotation rows. These are different denominators. Cutting a source into many clips increases training examples without creating independent patients. True person identity and edited repost relationships are not established by a YouTube identifier.

Stroke was repeated in the brief and is included once. Myopathic gait motivates a neuromuscular question; it should not automatically be described as a central neurological condition. Restricting the first study to these five labels makes its scope clear, but excludes many other forms of abnormal gait.'''),
code('''selected_counts = inventory.groupby("label").agg(sequences=("sequence_id", "size"), sources=("source_id", "nunique"), rows=("n_frames", "sum"))
display(selected_counts)
display(inventory.loc[~inventory.eligible, ["label", "source_id", "sequence_id", "exclusion_reason"]])
print("Annotation identity:", audit["annotation_identity"])'''),
md('''## Step 4. Split sources before training

All sequences from one source inherit one split. This prevents a model from learning from one part of a video and being tested on another part of the same recording. The split is established before pose quality filtering. We do not replace an excluded test source with a more convenient one after seeing performance.

Training fits the representation and classifiers. Validation chooses declared development settings and thresholds. Test sources are reserved for final evaluation. Unlabeled test motion is still test data and must not enter self-supervised training.'''),diagram('source_split'),
code('''partition = assign_splits(inventory, seed=27)
assert partition.groupby("source_id").split.nunique().max() == 1
display(partition.loc[partition.eligible].groupby(["split", "label"]).agg(sources=("source_id", "nunique"), sequences=("sequence_id", "size")))
if REAL:
    _, _, clean_metadata = real_inputs()
    print("Active person-gated clean cohort, after pose and source exclusions:")
    display(clean_metadata.groupby(["split", "label"]).agg(sources=("source_id", "nunique"), windows=("sequence_id", "size")))'''),
md('''## Step 5. Interpret the limitations before a score exists

The fixed split has only one media-eligible cerebral-palsy test source. Its many sequences cannot estimate between-person variation. The broader GAVD inventory has 348 sources, but this five-label study has far fewer, and missing media reduces it further. Source grouping is necessary but cannot by itself establish generalization to new patients, clinics, diagnoses, or camera systems.

A credible workshop claim is an auditable feasibility study of predictive gait representations under source shifts. A stronger clinical claim needs independently verified labels and external patient/site evaluation. The supplied neurological framing can motivate hypotheses without turning uncertain annotations into clinical ground truth.'''),
code('''display(Markdown((PROJECT / "docs/workshop.md").read_text()))'''),
md('''## Step 6. Freeze evidence without silently overwriting it

The command below records the exact current CSV identity and source partition. Changed inputs require a new output directory. This behavior protects earlier evidence from being relabeled after a method or cohort changes.'''),
code('''from worldmodels.data import save_inventory
# Reusing the same unchanged inventory is safe; changed evidence refuses overwrite.
report = save_inventory(partition, audit, PROJECT / "artifacts/runs/notebook-inventory")
print("Frozen partition:", report["partition_sha256"])''')])

build('01_video_pose_and_quality.ipynb','01. Connect walking video to trustworthy skeletons',
'''The model never sees a diagnosis. It sees coordinates estimated from frames. This lesson makes that conversion visible, preserves frame numbering, and explains why a pose cache is only useful when its provenance and missingness are explicit.''',[
md('''## Step 1. Find a real source without duplicating media

Existing originals live under `data/videos-gavd/<label>/`. The resolver also supports `sources/` for new downloads, so each YouTube ID can be stored once. New previews belong under `data/videos-gavd/previews/`. We preserve existing media and never copy a full video into multiple condition folders for this experiment.'''),
code('''from worldmodels.data import build_inventory, find_video
inventory, audit = build_inventory(REPOSITORY)
example = real_example() if REAL else inventory.loc[inventory.eligible & (inventory.label == "normal")].sort_values("sequence_id").iloc[0]
video_path = find_video(REPOSITORY, example.source_id, example.label)
annotation = pd.read_csv(REPOSITORY / example.csv_path, keep_default_na=False)
display(annotation[["seq", "frame_num", "cam_view", "gait_pat"]].head())
print("Local source:", video_path.relative_to(REPOSITORY))'''),
md('''## Step 2. Watch the annotated interval

Set `WM_EMBED_VIDEO=1` in `.env` to embed this local original. Embedded video makes a notebook larger; clear its output before sharing the notebook. Playback starts near the annotated interval, which is more useful than watching an unrelated introduction. Real mode always embeds the bounded excerpt. The original can also be opened in your local video player.

Check that the visible person matches the intended sequence, that walking is present, and that there are no cuts across the requested interval. A skeleton detector can choose the wrong person in a multi-person scene. The current adapter selects the first full-image pose, so pilot outputs remain unverified for person identity until reviewed.'''),
code('''import cv2
from worldmodels.pose import video_preview
cap = cv2.VideoCapture(str(video_path))
source_fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
start_seconds = (int(example.first_frame) - 1) / source_fps
end_seconds = int(example.last_frame) / source_fps
if REAL or os.getenv("WM_EMBED_VIDEO", "0") == "1":
    from IPython.display import Video
    display(Video(str(bounded_preview(example)), embed=True))
else:
    print(f"Open {video_path.name} near {start_seconds:.2f} to {end_seconds:.2f} seconds; set WM_EMBED_VIDEO=1 for inline playback.")'''),
md('''## Step 3. Preserve the frame boundary exactly once

GAVD frame 1 means the first source frame. OpenCV's decoded index 0 means the same frame. The extraction adapter converts `frame_num - 1` once and then passes the decoded RGB image to Ambient. It does not pass a second shifted index to another frame-extraction API.

Bounding boxes were annotated against an earlier image representation. They should not be applied blindly when the downloaded image differs in scale, borders, or framing. The current pilot uses full images and records that limitation.'''),
code('''mapping = annotation[["frame_num"]].head(8).copy()
mapping["decoded_index"] = mapping.frame_num - 1
mapping["nominal_seconds"] = mapping.decoded_index / source_fps
display(mapping)
assert (mapping.decoded_index >= 0).all()'''),
md('''## Step 4. Run a bounded real extraction

Real mode always executes MediaPipe on 16 sampled frames in a fresh lesson-specific cache. In synthetic mode, set `WM_RUN_POSE=1` to request the same actual extraction. It uses the local MediaPipe model and Ambient's public `SequenceKeypointExtractor.extract_from_image`. The cache records model, video, annotation, extractor, and adapter fingerprints. Failed detections retain their frame positions with zero confidence and missing coordinates. They are not removed and mistaken for adjacent time steps.

This small extraction checks the pipeline. It is not enough to characterize a full gait cycle or establish performance. The full preparation command in the capstone can process the selected cohort.'''),
code('''from worldmodels.pose import extract_sequence, make_windows
pose = None
if REAL or os.getenv("WM_RUN_POSE", "0") == "1":
    model_path = Path(os.getenv("WM_POSE_MODEL") or REPOSITORY / "data/models/pose_landmarker_lite.task")
    pose = extract_sequence(REPOSITORY / example.csv_path, REPOSITORY, model_path,
                            (lesson_directory("01_video_pose_and_quality") / "poses") if REAL else PROJECT / "artifacts/poses", frame_step=2, max_frames=16)
    print("Pixels:", pose["xy"].shape, "| confidence:", pose["confidence"].shape)
    print("Observed frame numbers:", pose["frame_numbers"])
    x, valid, metadata = make_windows(pose)
    print("Accepted windows:", len(x))
else:
    print("Real extraction is available with WM_RUN_POSE=1. No pose result is fabricated.")'''),
md('''## Step 5. Understand normalization and missingness

The 16 selected landmarks are centered on the pelvis. The median observed torso length in the first eight frames supplies a single scale. Future frames cannot change this scale. Missing pelvis observations invalidate the affected frame. This representation describes relative articulation rather than global travel through the scene.

Random completion has a separate caveat: normalization is performed before the random mask is drawn, so hidden hips or shoulders may have contributed to the transform. The task is completion of a pre-normalized articulation sequence. See the joint contract before making stronger claims about raw-coordinate conditional prediction.'''),
code('''from worldmodels.notebook_support import lesson_data
fixture = lesson_data(n=2)
x_demo = fixture["x"][0]
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(x_demo[0,:,0], x_demo[0,:,1], c=np.arange(16), cmap="viridis")
ax.invert_yaxis(); ax.set_aspect("equal")
ax.set(xlabel="Relative x in torso units", ylabel="Relative y in torso units", title=EVIDENCE_LABEL + " skeleton, first frame")
plt.show()'''),
md('''## Step 6. Audit what did not survive

The fresh inventory excludes known missing videos, an insufficient source range, and four sequences with a documented video-reference mismatch. Pose processing creates additional attrition. Keep its reasons and count retained sources by label and partition. Confidence thresholds are study settings, not universal Ambient defaults.

The tutorials use a confidence cutoff of 0.5 and minimum observed fraction of 0.7. A sensitivity analysis should vary these using development data and report how many sources disappear. Do not select a quality cutoff because it makes the final test score look better.'''),
code('''display(inventory.loc[~inventory.eligible].groupby(["label", "exclusion_reason"]).size().rename("sequences").reset_index())
print("Pose filtering has not been inferred from media availability.")''')])

build('02_joint_tokens_and_uniform_masks.ipynb','02. Turn the approved joints into uniformly masked tokens',
'''A token is one joint at one time. This lesson fixes the anatomical indexing, draws uniform masks, and distinguishes random completion from prediction of the future. The mask never consults motion intensity or velocity.''',[
md('''## Step 1. Fix the deduplicated anatomy

The user supplied feature requirements rather than a literal landmark list. We derive an explicit 16-joint union from exp5's lower-limb geometry and the declared arm/foot extension. These are the only joints that enter the model or its masking universe.'''),
md((ROOT/'docs/joints.md').read_text().split('## What the features measure')[0]),
code('''from worldmodels.data import JOINT_IDS
from worldmodels.masking import JOINT_IDS as MASK_JOINT_IDS
assert JOINT_IDS == MASK_JOINT_IDS
assert list(JOINT_IDS) == sorted(set(JOINT_IDS))
print("Tensor slot -> BlazePose ID:", dict(enumerate(JOINT_IDS)))'''),
md('''## Step 2. Give shape a physical meaning

`[batch, time, joint, coordinate]` is the tensor order. A 16-frame window and 16 joints give 256 possible joint-time tokens. The last dimension has x and y. Visibility is a separate Boolean array and does not create another coordinate dimension. Index 0 in the reduced tensor means BlazePose index 11, not the nose.'''),
code('''from worldmodels.notebook_support import lesson_data
from worldmodels.masking import uniform_mask, forecast_mask
fixture = lesson_data(n=4)
x = torch.as_tensor(fixture["x"])
valid = torch.as_tensor(fixture["valid"])
print("coordinates:", tuple(x.shape), "validity:", tuple(valid.shape))
if not REAL:
    valid[0, 3, 5] = False  # synthetic missingness exercise; real mode preserves measured validity'''),diagram('uniform_mask'),
md('''## Step 3. Draw a uniform mask without replacement

A `True` target flag means the model must predict that token. The routine samples a fixed-size subset of valid positions uniformly. Missing positions are excluded from both targets and visible context. Every example retains at least one context and one target.

MAMP's main published strategy favors stronger motion. This study intentionally follows the requested uniform variant. Random numbers inside a motion-weighted sampler would not make its final sampling distribution uniform.'''),
code('''mask = uniform_mask(valid, ratio=0.5, generator=torch.Generator().manual_seed(27))
assert not (mask & ~valid).any()
print("Targets per window:", mask.flatten(1).sum(1).tolist())
fig, ax = plt.subplots(figsize=(9,4))
ax.imshow(mask[0].T, aspect="auto", interpolation="nearest", cmap="Blues")
ax.set(yticks=np.arange(16), yticklabels=JOINT_IDS, xlabel="Frame slot", ylabel="BlazePose joint ID", title="Blue cells are prediction targets")
plt.show()'''),
md('''## Step 4. Verify independence from movement

The masking function accepts validity and a random generator, but no coordinates. Multiplying every displacement by a thousand cannot change the chosen positions when validity and the seed are fixed. This test checks the intended causal dependency of the software rather than simply inspecting a random-looking plot.'''),
code('''changed_motion = x * 1000
again = uniform_mask(valid, ratio=0.5, generator=torch.Generator().manual_seed(27))
assert torch.equal(mask, again)
print("Changing motion values has no input path into mask selection.")'''),
md('''## Step 5. Distinguish completion from forecasting

Random masks can leave visible joints before and after a target time. A completion model may therefore use both temporal directions. Forecasting requires a different mask: all later observations are hidden from the online encoder, including joints that would otherwise be easy to reveal. The teacher can see the later outcomes only when building training or retrospective scoring targets.'''),
code('''future = forecast_mask(valid, context_frames=8)
assert not future[:, :8].any()
assert torch.equal(future[:, 8:], valid[:, 8:])
fig, ax = plt.subplots(figsize=(9,3))
ax.imshow(future[0].T, aspect="auto", interpolation="nearest", cmap="Oranges")
ax.set(xlabel="Frame slot", ylabel="Joint slot", title="Future-only targets: frames 8 through 15")
plt.show()'''),
md('''## Step 6. Keep masks consistent across controlled variants

A positional-noise or latent-variable experiment should reuse the same eligible inputs, seed, source split, and masking rule. Otherwise an apparent stochasticity benefit may reflect easier masks or a different cohort. Never add face or finger landmarks just because a new code path expects 33 coordinates.'''),
code('''print("Allowed targets only:", JOINT_IDS)
print("Example token targeted?", bool(mask[0,3,5]))
if REAL:
    from worldmodels.training import load_checkpoint
    saved_encoder = load_checkpoint(comparison_path() / "normal_completion.pt")
    with torch.no_grad():
        actual_completion = saved_encoder(x, valid, mask)
    print("Actual trained-model masked latent prediction:", tuple(actual_completion["latent"].shape))''')])

build('03_normal_only_jepa_and_vicreg.ipynb','03. Learn a normal-motion reference without collapse',
'''This lesson follows one tiny normal-only training run from visible tokens through teacher targets and the loss. The implementation is a compact Skeleton-JEPA adaptation with a coordinate readout and VICReg-style regularization, not a numerical reproduction of the original action-recognition paper.''',[
md('''## Step 1. Follow the three networks

The online view encoder sees visible joint-time tokens. The predictor combines their embeddings with positioned mask tokens and predicts hidden representations. The target encoder sees the complete observed sequence and produces stop-gradient targets. Its weights follow the online encoder through an exponential moving average.

This teaching model uses one joint-time token and final-layer teacher features. Published Skeleton-JEPA uses a different tokenization and target construction. We preserve the conceptual learning path while declaring the adaptation.'''),diagram('jepa'),
code('''from worldmodels.notebook_support import lesson_data
from worldmodels.model import ModelConfig, SkeletonJEPA, loss_terms, variance_covariance
from worldmodels.masking import uniform_mask
from worldmodels.training import TrainConfig, fit, encode
fixture = lesson_data(n=16, seed=27)
model = SkeletonJEPA(ModelConfig(dim=24, encoder_layers=1))
print("Normal-only labels:", np.unique(fixture["labels"]))
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))'''),
md('''## Step 2. Inspect one prediction before training

The teacher is not an independently supervised clinician. It is a slowly updated copy of the online encoder. Its output becomes a centered and sharpened categorical distribution across latent dimensions. The predictor distribution is trained against that target using cross-entropy at masked positions. A separate coordinate head makes approximate skeleton trajectories visible; latent prediction alone does not automatically reconstruct coordinates.'''),
code('''bx = torch.as_tensor(fixture["x"][:4]); bv = torch.as_tensor(fixture["valid"][:4])
target = uniform_mask(bv, generator=torch.Generator().manual_seed(27))
model.train()
output = model(bx, bv, target)
terms = loss_terms(model, output, bx, bv, target)
display(pd.Series({k: float(v.detach()) for k,v in terms.items()}, name="Before training"))
assert all(not p.requires_grad for p in model.teacher.parameters())'''),
md('''## Step 3. Explain collapse and the actual role of VICReg

If every input becomes the same embedding, prediction can become uninformative even when a loss appears stable. The variance term penalizes dimensions with too little batch variation. The covariance term reduces redundant dimensions. Together with centered teacher targets and EMA, these are explicit anti-collapse mechanisms.

VICReg does not push diagnosis clusters as far apart as possible. It uses no class labels here. Whether useful clinical structure appears is an empirical question. Batches should contain distinct sources, because many nearly identical windows from one recording do not provide independent variation.'''),
code('''spread = output["representation"].detach() if REAL else torch.randn(8,24)
collapsed = torch.zeros_like(spread)
comparison = {name: [float(v) for v in variance_covariance(z)] for name,z in [("collapsed", collapsed),("observed representation" if REAL else "random spread", spread)]}
display(pd.DataFrame(comparison, index=["variance penalty", "covariance penalty"]))'''),
md('''## Step 4. Train only on normal training sources

In real mode the lesson loads normal training windows with actual source and sequence IDs. In synthetic mode the fixture labels are normal and its source IDs are explicitly artificial. The API refuses non-normal labels in the normal-reference regime and refuses any partition value other than `train`. Four updates are enough to exercise the mechanics; they are not convergence evidence. The full experiment uses actual source IDs and separately recorded update budgets.'''),
code('''history = fit(model, fixture["x"], fixture["valid"],
              TrainConfig(steps=4, batch_size=8, seed=27, regime="normal", run_id="lesson03-normal-v1"),
              labels=fixture["labels"], source_ids=fixture["source_ids"], partitions=["train"]*len(fixture["x"]))
display(pd.DataFrame(history))'''),
md('''## Step 5. Check representations rather than only loss

Look at variation across examples, dimensions, and sources. An embedding with zero variation is unusable even if training finishes. A nonzero standard deviation is only a diagnostic, not proof of useful structure. Clinical separation and robustness must be tested on held-out sources with suitable baselines.'''),
code('''embeddings = encode(model, fixture["x"], fixture["valid"])
print("Embedding matrix:", embeddings.shape)
print("Median dimension standard deviation:", float(np.median(embeddings.std(0))))
fig, ax = plt.subplots(figsize=(7,3))
ax.plot(pd.DataFrame(history)["step"], pd.DataFrame(history)["loss"], marker="o")
ax.set(xlabel="Update", ylabel="Training objective", title=EVIDENCE_LABEL + " normal-only training exercise")
plt.show()'''),
md('''## Step 6. Preserve the reference identity

Adaptation should start from a copy of this normal checkpoint. Replacing it in place would destroy the reference needed to ask whether a new gait deviates from learned normal motion. Checkpoint metadata includes sources, data identity, code hashes, training settings, and parent identity. New training stages require new run names.'''),
code('''print({k: model.run_metadata[k] for k in ("run_id", "regime", "n_windows", "data_sha256", "parent_run_id")})
print("Teacher remains gradient-free:", all(p.grad is None for p in model.teacher.parameters()))''')])

build('04_progressive_condition_adaptation.ipynb','04. Add conditions while preserving the normal reference',
'''A normal-only encoder can serve as a reference, but a five-class classifier may benefit from exposure to more motion patterns. This lesson compares a cumulative condition curriculum with a mixed adaptation control. Every training observation remains inside the training partition.''',[
md('''## Step 1. Keep the question and labels separate

The curriculum adds Parkinsonian, stroke, cerebral-palsy, and myopathic examples after normal pretraining. Their order is an experimental choice, not a claim about disease progression. We retain normal replay at every stage. An adapted representation may become better for classification while becoming a weaker normality reference, so the original normal model is preserved.

Real mode uses actual normal and condition-labeled training windows from the frozen GAVD split. Synthetic mode uses artificial motions whose condition names merely exercise the API. Those synthetic labels must never be interpreted medically.'''),diagram('adaptation'),
code('''import copy
from worldmodels.data import LABELS
from worldmodels.notebook_support import lesson_data
from worldmodels.model import ModelConfig, SkeletonJEPA
from worldmodels.training import TrainConfig, fit, encode
fixture = lesson_data(n=40, seed=27, conditions=True)
normal_rows = fixture["labels"] == "normal"
reference = SkeletonJEPA(ModelConfig(dim=24, encoder_layers=1))
fit(reference, fixture["x"][normal_rows], fixture["valid"][normal_rows],
    TrainConfig(steps=4, seed=27, regime="normal", run_id="lesson04-normal-v1"),
    labels=fixture["labels"][normal_rows], source_ids=fixture["source_ids"][normal_rows], partitions=["train"]*int(normal_rows.sum()))
reference_weights = {k: v.clone() for k,v in reference.state_dict().items()}
adapted = copy.deepcopy(reference)'''),
md('''## Step 2. Add one label at a time

At each stage, eligible labels are the conditions already introduced plus normal. The sampler reserves normal examples for replay and draws distinct sources. This reduces immediate forgetting of the reference population, but it does not guarantee that earlier representations remain unchanged. Keep each stage's identity and evaluate forgetting using held-out development sources.'''),
code('''stages = []
for stage, condition in enumerate(LABELS[1:], start=1):
    rows = np.isin(fixture["labels"], LABELS[:stage+1])
    fit(adapted, fixture["x"][rows], fixture["valid"][rows],
        TrainConfig(steps=4, seed=27, regime="adaptation", run_id=f"lesson04-stage-{stage}-v1"),
        labels=fixture["labels"][rows], source_ids=fixture["source_ids"][rows], partitions=["train"]*int(rows.sum()))
    stages.append({"stage": stage, "added": condition, "sources": int(rows.sum()), "parent": adapted.run_metadata["parent_run_id"]})
display(pd.DataFrame(stages))'''),
md('''## Step 3. Verify the reference was preserved

Copying a model before adaptation is a small implementation detail with an important scientific consequence. It allows normal-reference anomaly scores and adapted classification scores to be compared honestly without silently changing what “normal” means.'''),
code('''assert all(torch.equal(reference_weights[k], reference.state_dict()[k]) for k in reference_weights)
print("Original normal weights are unchanged.")
print("Adapted identity:", adapted.run_metadata["run_id"])'''),
md('''## Step 4. Build a mixed-data control

The mixed model starts from the same normal checkpoint and receives all five labels immediately, with the same total number of adaptation updates. This controls update budget. It does not exactly control exposure to each condition, because curriculum stages have different membership. Report that distinction instead of calling the comparison perfectly matched.'''),
code('''mixed = copy.deepcopy(reference)
fit(mixed, fixture["x"], fixture["valid"],
    TrainConfig(steps=16, seed=27, regime="adaptation", run_id="lesson04-mixed-v1"),
    labels=fixture["labels"], source_ids=fixture["source_ids"], partitions=["train"]*len(fixture["x"]))
print("Curriculum adaptation updates:", 4*4)
print("Mixed adaptation updates:", mixed.run_metadata["config"]["steps"])'''),
md('''## Step 5. Inspect representation change without claiming accuracy

A change in embeddings shows that training changed the model. It does not establish improvement. Use source-level validation classification, normal-reference error, and representation diagnostics to choose a declared checkpoint. The final test set cannot choose the curriculum order, epoch count, or best-looking visualization.'''),
code('''representations = {"normal": encode(reference, fixture["x"], fixture["valid"]),
                   "curriculum": encode(adapted, fixture["x"], fixture["valid"]),
                   "mixed": encode(mixed, fixture["x"], fixture["valid"])}
display(pd.DataFrame({name: {"median_dimension_std": float(np.median(z.std(0))), "mean_norm": float(np.linalg.norm(z,axis=1).mean())} for name,z in representations.items()}))'''),
md('''## Step 6. Transfer the protocol to real data

Use the frozen source table from `worldmodels.data.assign_splits`, select rows whose split is `train`, and then select normal labels for Stage A. Use only the training subset again during Stage B. For each source-grouped cross-validation fold, repeat pretraining and adaptation from a fresh model. A checkpoint trained on another fold's test subjects cannot be reused merely because the architecture is identical.'''),
code('''print("Real cumulative label sets:")
for i in range(1, len(LABELS)):
    print(i, LABELS[:i+1])''')])

build('05_forecasting_and_stochasticity.ipynb','05. Predict the future and distinguish two kinds of stochasticity',
'''Random completion can use both temporal directions. This lesson trains a separate future-only branch, visualizes an approximate coordinate forecast, and explains why positional noise and latent samples answer different questions.''',[
md('''## Step 1. Make the future inaccessible to the predictor

The first eight frames are context; the last eight are targets. All future joints are hidden from the online encoder. The teacher may observe the outcome when computing a training target or retrospective residual. It must not feed those outcomes into deployment predictions. Prefix-only normalization avoids using future scale statistics, while future pelvis centering defines an articulation target rather than global translation.'''),diagram('forecast'),
code('''import copy
from worldmodels.notebook_support import lesson_data
from worldmodels.model import ModelConfig, SkeletonJEPA
from worldmodels.training import TrainConfig, fit, anomaly_scores
fixture = lesson_data(n=16, seed=27)
def train_stage(net, name, regime, forecast_fraction=0.0):
    return fit(net, fixture["x"], fixture["valid"],
        TrainConfig(steps=4, seed=27, regime=regime, run_id=name, forecast_fraction=forecast_fraction, latent_mse_weight=1.0),
        labels=fixture["labels"], source_ids=fixture["source_ids"], partitions=["train"]*len(fixture["x"]))
normal = SkeletonJEPA(ModelConfig(dim=24, encoder_layers=1))
normal_initial_state = copy.deepcopy(normal.state_dict())
deterministic_history = {"normal": train_stage(normal, "lesson05-normal-v1", "normal")}
forecast = copy.deepcopy(normal)
deterministic_history["forecast"] = train_stage(forecast, "lesson05-forecast-v1", "forecast", 1.0)'''),
md('''## Step 2. Inspect one coordinate forecast

The coordinate readout is trained alongside the latent objective. A tiny run is expected to produce rough trajectories. Real mode plots actual GAVD articulation; synthetic mode plots a software fixture. In either mode this is a diagnostic for units, horizon, and target visibility, rather than clinical validation. A useful future model must beat simple controls such as repeating the last observed articulation.'''),
code('''evaluation_fixture = lesson_data(n=8, seed=27, split="validation") if REAL else fixture
scores = anomaly_scores(forecast, evaluation_fixture["x"][:4], evaluation_fixture["valid"][:4], context_frames=8)
joint_slot = 10  # left ankle, BlazePose 27
fig, ax = plt.subplots(figsize=(8,3))
ax.plot(np.arange(16), evaluation_fixture["x"][0,:,joint_slot,0], label=EVIDENCE_LABEL + " observed trajectory")
ax.plot(np.arange(8,16), scores["coordinates"][0,8:,joint_slot,0], label="Predicted future")
ax.axvline(7.5,color="gray",linestyle=":")
ax.set(xlabel="Frame slot",ylabel="Relative x",title=EVIDENCE_LABEL + " left-ankle forecast")
ax.legend(); plt.show()'''),
md('''## Step 3. Keep a persistence baseline

Persistence predicts that the relative articulation stays at its last observed value. It can be surprisingly strong at short horizons. Compute its error on exactly the same target mask and coordinate units. If a learned model cannot improve this reference, claims of useful dynamics learning need revision.'''),
code('''target = scores["target_mask"]
persistence = np.repeat(evaluation_fixture["x"][:4,7:8], 16, axis=1)
squared = ((persistence-evaluation_fixture["x"][:4])**2).mean(-1)
persistence_error = np.array([squared[i][target[i]].mean() for i in range(4)])
display(pd.DataFrame({"learned_coordinate_mse": scores["coordinate_mse"], "persistence_coordinate_mse": persistence_error}))'''),
md('''## Step 4. Distinguish positional noise from latent possibilities

Positional noise perturbs where a token is represented. The implemented temporal jitter is a proposed ablation and is not a faithful implementation of the published StoP weight-tying mechanism. Anatomical joint identities remain fixed.

A stochastic latent variable represents alternative completions conditioned on the same visible past. Its posterior can use targets during training, while its deployment prior uses context only. The KL term limits the gap between these distributions. It does not replace anti-collapse regularization. The recent Stochastic-JEPA image results do not establish a benefit for gait. This lesson gives both deterministic and stochastic branches the same latent-MSE weight of 1.0 alongside categorical teacher loss, coordinate readout, and VICReg terms. The stochastic branch additionally uses KL regularization. This is a declared hybrid adaptation, not a reproduction of the published latent-MSE-plus-KL objective alone.'''),diagram('stochasticity'),
md('''### Compare temporal-position jitter on the same real inputs

The next branch uses temporal jitter with standard deviation 0.2 in frame-slot units. It starts from exactly the deterministic branch's initial weights, uses the same source batches, uniform-mask seed, hybrid objective, and four updates in each normal and forecast stage. The implementation perturbs shared temporal encodings during training; it preserves joint identities and uses exact positions at evaluation. This is a simple positional-stochasticity experiment, not the published StoP weight-tying mechanism.

Real mode trains both branches on actual normal GAVD training sources and evaluates the same normal validation windows. The tiny, one-seed table demonstrates a controlled software comparison. It does not establish that positional noise improves performance and is separate from the main three-seed revision study.'''),
code('''positional = SkeletonJEPA(ModelConfig(dim=24, encoder_layers=1, position_jitter=0.2, latent_dim=0))
positional.load_state_dict(normal_initial_state)
assert all(torch.equal(normal_initial_state[key], positional.state_dict()[key]) for key in normal_initial_state)
positional_history = {"normal": train_stage(positional, "lesson05-position-normal-v1", "normal")}
positional_history["forecast"] = train_stage(positional, "lesson05-position-forecast-v1", "forecast", 1.0)
assert positional.run_metadata["data_sha256"] == forecast.run_metadata["data_sha256"]
assert all(value == forecast.run_metadata["config"][key] for key, value in positional.run_metadata["config"].items() if key != "run_id")
positional_scores = anomaly_scores(positional, evaluation_fixture["x"][:4], evaluation_fixture["valid"][:4], context_frames=8)
positional_comparison = pd.DataFrame({
    "source_id": evaluation_fixture["source_ids"][:4],
    "deterministic_mse": scores["coordinate_mse"],
    "position_jitter_mse": positional_scores["coordinate_mse"],
    "persistence_mse": persistence_error,
})
display(positional_comparison)
print("Matched initial weights and batches; four updates per stage; one teaching seed.")'''),
code('''seed_everything(27)
stochastic = SkeletonJEPA(ModelConfig(dim=24, encoder_layers=1, latent_dim=4))
stochastic.load_state_dict(normal_initial_state, strict=False)
stochastic_history = {"normal": train_stage(stochastic, "lesson05-latent-normal-v2", "normal")}
stochastic_history["forecast"] = train_stage(stochastic, "lesson05-latent-forecast-v2", "forecast", 1.0)
sampled = anomaly_scores(stochastic, evaluation_fixture["x"][:4], evaluation_fixture["valid"][:4], context_frames=8, samples=8, seed=27)
print("Prior sample tensor:", sampled["samples"].shape)
display(pd.DataFrame({"prediction_error": sampled["coordinate_mse"], "uncalibrated_dispersion": sampled["predictive_variance"]}))'''),
md('''## Step 5. Do not call dispersion calibrated uncertainty

Multiple prior samples can differ without their spread tracking predictive error. Test error versus dispersion on held-out sources, assess risk when uncertain cases are deferred, and compare uncertainty with pose visibility and camera view. High spread may reflect acquisition noise rather than disease. A few teaching examples cannot establish calibration, even when they come from real GAVD recordings.

For a normal-reference detector, choose a threshold on normal validation sources only. With six such sources, the conservative finite-sample threshold for a 5% false-positive target is infinite. This means there is not enough calibration support for a useful finite guarantee.'''),
code('''from worldmodels.evaluation import validation_threshold_report
if REAL:
    from worldmodels.evaluation import aggregate_sources
    validation_fixture = lesson_data(n=32, conditions=False, split="validation")
    validation_scores = anomaly_scores(forecast, validation_fixture["x"], validation_fixture["valid"], context_frames=8)["coordinate_mse"]
    _, _, source_calibration = aggregate_sources(validation_scores[:,None], validation_fixture["labels"], validation_fixture["source_ids"], validation_fixture["metadata"].sequence_id)
    calibration_example = validation_threshold_report(source_calibration[:,0])
else:
    calibration_example = validation_threshold_report([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
display(pd.Series(calibration_example))'''),
md('''## Step 6. Bound explanations

A large left-ankle residual identifies where the prediction disagreed with an observed estimate. It does not identify the neurological cause. Check occlusion, tracking failures, camera direction, and unusual action before interpreting a residual as abnormal gait. The normal-only reference and adapted classifier should keep separate names and evidence records.'''),
code('''joint_residual = ((scores["coordinates"][0]-evaluation_fixture["x"][0])**2).mean(-1)
fig, ax = plt.subplots(figsize=(9,3))
ax.imshow(joint_residual[8:].T, aspect="auto", cmap="magma")
ax.set(xlabel="Future frame slot",ylabel="Joint slot",title=EVIDENCE_LABEL + " residual localization, not causal explanation")
plt.show()''')])

build('06_capstone_rf_vs_jepa.ipynb','06. Compare Random Forest and JEPA on the same held-out sources',
'''This capstone connects original video, shared pose extraction, features, learned representations, and source-level evaluation. It reads actual experiment artifacts when available and otherwise gives executable commands. It never replaces missing real results with synthetic accuracy.''',[
md('''## Step 1. Fix the comparison before looking at test scores

Both classifiers must receive identical accepted windows from identical sources. The RF uses measured joint-angle and motion summaries. JEPA uses an encoder representation with a training-only standardized logistic probe. Data splits, pose failures, quality rules, and target labels are shared. Hyperparameter selection uses validation sources with a declared candidate budget.

Exp5 demonstrates the classical feature route. Historical scores from a different cohort are not a controlled baseline. This project retrains the classical model on the new five-label cohort.'''),diagram('comparison'),
code('''from worldmodels.data import build_inventory, assign_splits
inventory, audit = build_inventory(REPOSITORY)
partition = assign_splits(inventory, seed=27)
if REAL:
    _, _, clean_metadata = real_inputs()
    display(clean_metadata.groupby(["split","label"]).source_id.nunique().unstack())
else:
    display(partition.loc[partition.eligible].groupby(["split","label"]).source_id.nunique().unstack())'''),
md('''## Step 2. Understand the 37 observed features

The first 15 follow the core exp5 construction: six mean hip/knee/ankle angles, three bilateral differences, and six angle ranges. The extension adds angle standard deviations, elbow summaries, foot-axis inclination, wrist motion, and ankle separation. Shared Ambient geometry computes lower-limb angles. Missing observations remain missing, and training-only imputation handles incomplete feature vectors.

The feature input is the same normalized 16-joint window used by JEPA. No source identifier, class label, or filename becomes a predictor. Units are projected angles and torso-normalized movement, not measured 3D anatomy or physical gait speed.'''),
code('''from worldmodels.notebook_support import lesson_data
from worldmodels.features import build_features
fixture = lesson_data(n=4)
feature_matrix, feature_names = build_features(fixture["x"], fixture["valid"], fps=fixture["metadata"].sample_fps.to_numpy() if REAL else 15)
print(EVIDENCE_LABEL + " feature matrix:", feature_matrix.shape)
display(pd.DataFrame({"column": np.arange(len(feature_names)), "name": feature_names}))'''),
md('''## Step 3. Prepare and run a real bounded pilot

Run these commands from the experiment directory. The clean builder requires the previously extracted real 16-frame pose caches and the original `extended-input-v1` cohort. It verifies provenance, reapplies the person gate using fresh GAVD bounding-box annotations, and excludes the complete sources `CMls0LwK5OU` from validation and `eBvzFkcvScg` from training. It creates `pilot-input-clean-v1` and `extended-input-clean-v1` without overwriting the earlier cohorts. Source assignments remain fixed. This is an intentionally short classifier pilot, distinct from the longer forecasting study. If the clean inputs already exist, reuse their verified artifacts instead of rerunning the immutable builder.

```bash
uv run python -m worldmodels.clean_data
uv run python -m worldmodels.workflow compare --input artifacts/runs/pilot-input-clean-v1 --output artifacts/runs/pilot-comparison-clean-v1 --epochs 5
```

The person-gate ledger is `docs/source_exclusions.json`. The longer clean forecasting cohort has its own preparation identity and reuse certificate. A new temporal horizon requires a new declared cohort and renewed person/quality checks; a raw `workflow prepare` call alone does not establish person eligibility. Output directories refuse accidental overwrite.'''),
code('''input_dir = prepared_path()
result_dir = comparison_path()
if (input_dir / "preparation.json").exists():
    import json
    display(pd.Series(json.loads((input_dir / "preparation.json").read_text())))
else:
    print("No prepared real pilot exists yet. Run the preparation command above.")'''),
md('''## Step 4. Inspect attrition before accuracy

A model comparison becomes misleading if one method quietly excludes difficult sources. The runner loads one prepared matrix for every approach. It requires all five labels in every split after extraction and refuses to proceed if a class disappears. Inspect the remaining number of sources, not only the number of windows.'''),
code('''if (input_dir / "windows.csv").exists():
    window_metadata = pd.read_csv(input_dir / "windows.csv")
    display(window_metadata.groupby(["split","label"]).agg(windows=("label","size"),sources=("source_id","nunique")))
    if (input_dir / "attrition.csv").exists():
        attrition = pd.read_csv(input_dir / "attrition.csv", keep_default_na=False)
        display(attrition.loc[attrition.exclusion_reason != ""].head(20))
else:
    print("Pose attrition is unavailable until preparation completes.")'''),
md('''## Step 5. Read actual source-level results

During development, `WM_RELEASE_TEST_RESULTS=0` displays validation selections and source support only. Final test tables require an explicit release after the revision protocol is frozen. Training-source input-video inference remains available before release.

Predictions are averaged within sequence and then within source. The primary comparison is binary balanced accuracy at the fixed supervised probability threshold of 0.5. Five-class macro-F1, AUROC, average precision, sensitivity, specificity, and per-class recall provide complementary views.

A paired source bootstrap resamples identical held-out source indexes for both methods. With very few sources in a condition, the interval can still underrepresent population uncertainty. It does not measure training-seed variability or external clinical generalization.'''),
code('''import json
results_path = result_dir / "results.json"
raw_result_record = json.loads(results_path.read_text()) if results_path.exists() else None
results = raw_result_record
if results is not None and not RELEASE_TEST_RESULTS:
    permitted = ("identity", "status", "support", "validation_selection", "representation_diagnostics", "limitations", "seed", "n_windows")
    results = {key: raw_result_record[key] for key in permitted if key in raw_result_record}
    display(pd.DataFrame(results.get("validation_selection", [])))
    display(pd.DataFrame(results.get("support", [])))
    print("Final test metrics are sealed. Set WM_RELEASE_TEST_RESULTS=1 only after the revision protocol is frozen.")
if results is not None and RELEASE_TEST_RESULTS:
    score_columns = ["binary_balanced_accuracy", "macro_f1", "abnormal_auroc", "sensitivity", "specificity"]
    display(pd.DataFrame(results["classifiers"]).T.reindex(columns=score_columns))
    display(pd.Series(results["paired_jepa_minus_rf"]))
else:
    print("No test metric is displayed while results are absent or the test-release gate is closed.")'''),
md('''## Step 6. Read the error structure and reference detector separately

The supervised classifier predicts labels after training on the selected conditions. The anomaly branch is trained only on normal sources and scores future prediction errors. Their thresholds and intended uses differ. Preserve the normal checkpoint when adapting a separate classification encoder.

A confusion matrix can reveal a model that predicts the majority condition, but it cannot explain disease mechanisms. Inspect original video and pose visibility for misclassified sources. Report the one-source cerebral-palsy test limitation and incomplete person identity review with every pilot result.'''),
code('''if results is not None and RELEASE_TEST_RESULTS:
    chosen = results["classifiers"]["jepa_adapted"]
    display(pd.DataFrame(chosen["confusion_matrix"], index=chosen["classes"], columns=chosen["classes"]))
    display(pd.Series(results["anomaly"]))
    display(Markdown("**Recorded limitations:**"))
    display(pd.Series(results["limitations"]))
else:
    print("Test predictions and anomaly test metrics remain withheld until WM_RELEASE_TEST_RESULTS=1.")''')])

# Add an independently callable input-video demonstration to the capstone.
capstone_path = ROOT / '06_capstone_rf_vs_jepa.ipynb'
capstone = nbf.read(capstone_path, as_version=4)
capstone.cells[-2:-2] = [
    md("""## Step 7. Classify an input walking interval

The inference adapter takes an existing video and a declared walking interval without requiring an input condition label. It applies the same pose/window contract, loads the fitted RF and JEPA probes, and reports sequence probabilities plus normal-reference forecast error. The interval should contain the intended walking person. A predicted condition remains a dataset-label prediction rather than a diagnosis.

After the real comparison has completed, the command-line interface is:

```bash
uv run python -m worldmodels.inference --video /absolute/path/to/walking.mp4 --run artifacts/runs/pilot-comparison-clean-v1 --start-frame 1 --end-frame 96 --pose-model ../../data/models/pose_landmarker_lite.task
```

Real mode always runs this actual input-video inference against the fitted clean classifiers. Synthetic mode runs it only when `WM_RUN_INFERENCE=1`. No condition label is passed into the classifier."""),
    code("""if REAL or os.getenv("WM_RUN_INFERENCE", "0") == "1":
    from worldmodels.inference import classify_video
    from worldmodels.data import find_video
    sample = real_example() if REAL else partition.loc[partition.eligible & (partition.label == "normal")].iloc[0]
    source_video = find_video(REPOSITORY, sample.source_id, sample.label)
    report = classify_video(source_video, result_dir,
                            REPOSITORY / "data/models/pose_landmarker_lite.task",
                            start_frame=int(sample.first_frame), end_frame=min(int(sample.last_frame), int(sample.first_frame)+95))
    if REAL and not report.get("classifiers"):
        raise RuntimeError("Real input-video inference produced no classifier predictions")
    display(report)
else:
    print("Input-video inference is available after a fitted comparison: set WM_RUN_INFERENCE=1.")""")]
nbf.write(capstone, capstone_path)

build('07_ablation_and_paper_evidence.ipynb','07. Turn a runnable tutorial into a defensible research claim',
'''A working pipeline is the beginning of the evidence, not the conclusion. This lesson designs controlled ablations, checks what has actually run, and connects the available evidence to the Med-WM submission scope.''',[
md('''## Step 1. Separate the proposed mechanisms

Uniform masking tests learning from missing joint-time observations. Positional noise changes the location representation. Stochastic latent variables allow multiple context-conditioned completions. VICReg terms constrain representation statistics. Changing all of these together would make an improvement hard to interpret.

The default model is a compact Skeleton-JEPA adaptation. The positional-noise switch is temporal jitter, not the full published StoP method. The stochastic latent extension borrows the prior/posterior idea but does not inherit image-paper performance guarantees.'''),diagram('stochasticity'),
code('''ablations = pd.DataFrame([
    {"identity":"deterministic", "position_jitter":0.0,"latent_dim":0,"question":"Reference predictive representation"},
    {"identity":"position-only", "position_jitter":0.2,"latent_dim":0,"question":"Does temporal position perturbation help?"},
    {"identity":"latent-only", "position_jitter":0.0,"latent_dim":4,"question":"Does prior diversity track future error?"},
    {"identity":"both", "position_jitter":0.2,"latent_dim":4,"question":"Do the two mechanisms interact?"},
])
display(ablations)'''),
md('''## Step 2. Declare the controls before testing

Use identical source folds, accepted windows, normal-pretraining data, masks, update budgets, and seed lists. Fit every fold's representation from scratch. Keep an untrained random-encoder probe to test whether useful structure already follows from architecture and pose input. Keep persistence for short-horizon prediction. Evaluate normal-only, cumulative adaptation, and mixed adaptation with separate identities.

Removing variance or covariance penalties is an anti-collapse ablation, not an instruction to seek maximally distant diagnosis clusters. Monitor dimension variance and effective representation rank alongside task performance.'''),
code('''controls = pd.DataFrame([
    ["Random encoder", "Same architecture and training-only probe", "Representation architecture versus learned dynamics"],
    ["Persistence", "Same prefix, horizon, target mask, units", "Whether future learning beats a short-horizon rule"],
    ["Mixed adaptation", "Same parent and total updates", "Curriculum ordering, with exposure caveat"],
    ["Missingness/view audit", "Source-level nuisance metadata", "Acquisition patterns versus motion evidence"],
], columns=["Control","What stays matched","Why it matters"])
display(controls)'''),
md('''## Step 3. Make unfinished work explicit

A model configuration that instantiates successfully has not completed an ablation. A synthetic training run has not measured medical accuracy. A single real pilot has not established seed stability. Track these stages with distinct status fields so planned work is never presented as a result.'''),
code('''from worldmodels.model import ModelConfig, SkeletonJEPA
rows = []
for record in ablations.to_dict("records"):
    config = ModelConfig(dim=24, encoder_layers=1, position_jitter=record["position_jitter"], latent_dim=record["latent_dim"])
    model = SkeletonJEPA(config)
    rows.append({"identity":record["identity"], "parameters":sum(p.numel() for p in model.parameters() if p.requires_grad), "evidence":"instantiation only; no real ablation run"})
display(pd.DataFrame(rows))'''),
md('''## Step 4. Inventory evidence for the paper

The workshop deadline and format must be checked against the live official call before submission. The locally verified call is documented below, including a date discrepancy between the website's update label and the workspace date. An honest position paper may be more appropriate than a strong empirical claim if the source review and controlled experiments are incomplete.'''),
code('''display(Markdown((PROJECT / "docs/workshop.md").read_text()))'''),
md('''## Step 5. Read available results without assuming success

This cell lists completed local result records. An absent file is not a failed scientific result, but it is also not completed evidence. Keep the pilot's exact input identities, code hashes, versions, source support, and unresolved limitations in any manuscript supplement.'''),
code('''import json
records = []
for path in sorted((PROJECT / "artifacts/runs").glob("*/results.json")):
    item = json.loads(path.read_text())
    records.append({"run":path.parent.name,"status":item.get("status"),"windows":item.get("n_windows"),"seed":item.get("seed"),"seconds":item.get("seconds")})
if records:
    display(pd.DataFrame(records))
else:
    print("No completed real result records were found.")'''),
md('''## Step 6. Write claims that the evidence can support

A reasonable initial claim concerns an auditable normal-reference prediction protocol and its behavior under source shifts. A claim that JEPA improves classification needs matched held-out predictions and uncertainty against RF and appropriate controls. A claim about uncertainty needs error-dispersion and calibration evidence. A claim of clinical generalization requires external patient or site validation.

Do not describe a residual heatmap as a causal neurological explanation. Do not describe projected coordinate motion as measured physical speed. Do not count overlapping windows as independent patients. Negative findings can be informative when the comparison is controlled and the limits are clear.'''),
code('''claim_gate = pd.DataFrame([
    ["Runnable implementation", "Executed notebooks and focused tests", "Verify execution record"],
    ["Source-independent pilot", "Frozen sources, pose attrition, matched predictions", "Inspect pilot artifacts"],
    ["Stochasticity advantage", "Matched repeated-seed real-data ablations", "Not established by instantiation"],
    ["Clinical generalization", "External verified patient/site cohort", "Not established by this GAVD study"],
], columns=["Claim","Required evidence","Current interpretation"])
display(claim_gate)''')])

print('Generated', len(list(ROOT.glob('*.ipynb'))), 'notebooks')

# Real notebook07 consumes completed development evidence. It cannot substitute a
# configuration-only demonstration when actual revision results are required.
revision_path = ROOT / '07_ablation_and_paper_evidence.ipynb'
revision_book = nbf.read(revision_path, as_version=4)
for cell in revision_book.cells:
    if cell.cell_type=='code' and cell.source.startswith('from worldmodels.model import ModelConfig, SkeletonJEPA'):
        cell.source = '''if REAL:
    import json
    revision_development = Path(os.getenv("WM_REVISION_DEVELOPMENT") or PROJECT / "artifacts/runs/revision-development-v1")
    development_file = revision_development / "development.json"
    if not development_file.is_file():
        raise FileNotFoundError("Real notebook07 requires completed revision-development-v1/development.json; no synthetic ablation replacement is permitted.")
    revision_development_record = json.loads(development_file.read_text())
    report = revision_development_record
    normal_development_columns = ["seed", "arm", "normal_mse", "normal_energy", "persistence_gain", "normal_control_mse", "representation_diagnostics"]
    display(pd.DataFrame(revision_development_record["records"]).reindex(columns=normal_development_columns))
    print("Actual development evidence:", development_file)
    print("Validation gate:", revision_development_record.get("validation_gate"))
else:
    from worldmodels.model import ModelConfig, SkeletonJEPA
    rows = []
    for record in ablations.to_dict("records"):
        config = ModelConfig(dim=24, encoder_layers=1, position_jitter=record["position_jitter"], latent_dim=record["latent_dim"])
        model = SkeletonJEPA(config)
        rows.append({"identity":record["identity"], "parameters":sum(p.numel() for p in model.parameters() if p.requires_grad), "evidence":"instantiation only; no real ablation run"})
    display(pd.DataFrame(rows))'''
revision_book.cells[-2:-2] = [
    md('''## Step 7. Audit the executed revision and call its actual model

The real revision separates a direct coordinate predictor, a deterministic residual predictor, and a stochastic residual predictor. A residual predictor starts from a simple last-observation reference and learns corrections. Its contribution must be evaluated against persistence and the declared controls on matched sources. The development record identifies the actual settings, seeds, horizon, and validation evidence. The earlier four-row stochasticity table is a design map, not a claim that every listed option was executed.

This cell loads the saved deterministic residual model and calls it on an actual normal validation sequence from the extended cohort. The printed tensor dimensions verify the real forward path. Per-source CSVs provide the development errors used for the scientific comparison. Neither the input window nor these CSVs are drawn from the final test partition. The clean cohort reuses the original forecast checkpoints only because `docs/forecast_reuse_certificate.json` verifies exact equality of normal training, normal validation, and all test arrays, validity masks, and metadata. The source-score table is the filtered original validation prediction file, preserving its original stochastic draws. The forward call here is a pedagogical check and does not replace the inherited paper statistics. The 16-frame classification cohort and 32-frame forecasting cohort have different quality requirements and can retain different sources. Compare methods within their declared common cohort, and do not treat scores across those two populations as a paired comparison.'''),
    code('''if REAL:
    from worldmodels.revision import RevisionProtocol, load_revision_checkpoint, predict
    from worldmodels.workflow import load_prepared
    revision_input = Path(os.getenv("WM_REVISION_INPUT") or PROJECT / "artifacts/runs/extended-input-clean-v1")
    revision_x, revision_valid, revision_meta = load_prepared(revision_input)
    certificate_path = PROJECT / "docs/forecast_reuse_certificate.json"
    forecast_reuse_certificate = json.loads(certificate_path.read_text())
    required_checks = ("normal_train", "normal_validation", "all_test")
    assert all(forecast_reuse_certificate["checks"][name]["equal"] for name in required_checks)
    display(pd.DataFrame(forecast_reuse_certificate["checks"]).T)
    protocol = RevisionProtocol(**json.loads((revision_development / "protocol.json").read_text()))
    revision_model = load_revision_checkpoint(revision_development / "seed_27/residual_det.pt")
    validation_rows = revision_meta.index[(revision_meta.split == "validation") & (revision_meta.label == "normal")].to_numpy()
    if not len(validation_rows):
        raise ValueError("No normal validation windows in the real revision input")
    selected = validation_rows[:1]
    means, variances, samples = predict(revision_model, revision_x[selected], revision_valid[selected], protocol, seed=7028, control="actual")
    scores = {"revision_prediction": np.asarray(means), "revision_variance": np.asarray(variances), "revision_samples": np.asarray(samples)}
    print("Actual validation source:", revision_meta.loc[selected, "source_id"].tolist())
    print("Prediction shapes:", {name: value.shape for name, value in scores.items()})
    validation_source_scores = pd.read_csv(PROJECT / "artifacts/runs/revision-validation-clean-v1/revision-development-v1/seed_27/residual_det_actual_sources.csv")
    display(validation_source_scores.head())
    validation_source_scores.to_csv(lesson_directory("07_ablation_and_paper_evidence") / "revision_validation_sources.csv", index=False)
else:
    print("The real revision forward pass requires WM_DATA_MODE=real and completed development artifacts.")'''),
    md('''## Step 8. Release final test evidence only after the protocol is frozen

The development evidence can motivate a declared method change. Once the method, primary endpoint, and analysis are fixed, the separate test command produces a new record. This notebook keeps the final test result behind `WM_RELEASE_TEST_RESULTS=1`. An unsuccessful validation gate is a result to report, not a reason to keep changing the method against the same final test outcomes.'''),
    code('''if REAL and RELEASE_TEST_RESULTS:
    revision_test = Path(os.getenv("WM_REVISION_TEST") or PROJECT / "artifacts/runs/revision-test-v1/results.json")
    if not revision_test.is_file():
        raise FileNotFoundError("Released real notebook07 requires a completed revision-test-v1/results.json")
    released_revision_test = json.loads(revision_test.read_text())
    display(released_revision_test)
    destination = lesson_directory("07_ablation_and_paper_evidence") / "released_revision_test.json"
    destination.write_text(json.dumps(released_revision_test, indent=2) + "\\n")
else:
    print("Final revised test evidence remains sealed until WM_RELEASE_TEST_RESULTS=1.")''')]
nbf.validate(revision_book)
nbf.write(revision_book, revision_path)

# The revised binary endpoint and consolidated research report are released
# only after their corresponding protocols have been frozen.
capstone = nbf.read(ROOT / '06_capstone_rf_vs_jepa.ipynb', as_version=4)
capstone.cells[-2:-2] = [
    md("""## Step 8. Compare the direct binary endpoint

A five-class classifier and a direct normal-versus-abnormal classifier optimize different targets. Deriving abnormal probability by adding four condition probabilities can behave differently from fitting the binary endpoint directly, especially when class weighting changes the effective training distribution. The revision fixes the binary target and chooses its probe settings using validation sources before releasing the final test table.

The table below reads the actual `binary-test-clean-v1` result. It includes the matched Random Forest and JEPA probes, identifies the validation-selected primary JEPA representation, and reports its paired comparison with RF. It does not infer an improvement merely because the endpoint changed. This remains a small source-grouped study without external clinical validation."""),
    code("""if REAL and RELEASE_TEST_RESULTS:
    binary_path = Path(os.getenv("WM_BINARY_TEST") or PROJECT / "artifacts/runs/binary-test-clean-v1/results.json")
    if not binary_path.is_file():
        raise FileNotFoundError("Released capstone requires the completed direct-binary test record.")
    binary_report = json.loads(binary_path.read_text())
    binary_columns = ["binary_balanced_accuracy", "sensitivity", "specificity", "abnormal_auroc", "abnormal_average_precision"]
    display(pd.DataFrame(binary_report["test_reports"]).T.reindex(columns=binary_columns))
    primary_binary_jepa = binary_report["primary_jepa"]
    print("Validation-selected primary JEPA:", primary_binary_jepa)
    display(binary_report["paired_against_rf"][primary_binary_jepa])
    display(pd.DataFrame(binary_report["old_five_class_derived_reports"]).T.reindex(columns=binary_columns))
else:
    print("Direct-binary final test results remain sealed until WM_RELEASE_TEST_RESULTS=1.")""")]
nbf.write(capstone, ROOT / '06_capstone_rf_vs_jepa.ipynb')
revision_book = nbf.read(ROOT / '07_ablation_and_paper_evidence.ipynb', as_version=4)
revision_book.cells[-2:-2] = [
    md("""## Step 9. Inspect the consolidated evidence and figures

When the released analysis exists, this section reads its recorded interpretation and displays the evidence-linked figures. The binary comparison, forecast gain across seeds, and anomaly-score analysis answer different questions. A representation may support classification without beating persistence, and a prediction error may reflect pose quality rather than a medical condition. Read the uncertainty intervals and nuisance analyses before deciding which claims belong in a paper. The optional quality-only control is loaded from the separately released clean-cohort result, `quality-test-clean-v1/results.json`.

The teaching positional-jitter run in notebook 05 is a separate small demonstration. It should not be pooled with the three-seed revision or described as evidence of a stochasticity advantage."""),
    code("""quality_test_path = Path(os.getenv("WM_QUALITY_TEST") or PROJECT / "artifacts/runs/quality-test-clean-v1/results.json")
if REAL and RELEASE_TEST_RESULTS and quality_test_path.is_file():
    quality_control_report = json.loads(quality_test_path.read_text())
    display(quality_control_report)
    quality_output = lesson_directory("07_ablation_and_paper_evidence") / "quality_control_test.json"
    quality_output.write_text(json.dumps(quality_control_report, indent=2))
research_path = PROJECT / "docs/research_results.json"
if REAL and RELEASE_TEST_RESULTS and research_path.is_file():
    research_report = json.loads(research_path.read_text())
    display(research_report.get("interpretation"))
    display({key: research_report[key] for key in ("forecast_validation_gate", "temporal_controls", "stochastic_energy_gain") if key in research_report})
    for figure_name in ("binary_revision.svg", "forecast_revision.svg", "anomaly_revision.svg"):
        figure_path = PROJECT / "images" / figure_name
        if figure_path.is_file():
            display(SVG(filename=str(figure_path)))
else:
    print("Consolidated research results are displayed after test release and analysis generation.")""")]
nbf.write(revision_book, ROOT / '07_ablation_and_paper_evidence.ipynb')

# This audit was declared from training-gradient diagnostics. It remains
# secondary regardless of its validation result and never replaces the primary.
revision_book = nbf.read(ROOT / '07_ablation_and_paper_evidence.ipynb', as_version=4)
audit_insert = next(i for i, cell in enumerate(revision_book.cells)
                    if cell.cell_type == 'markdown' and cell.source.startswith('## Step 8.'))
revision_book.cells[audit_insert:audit_insert] = [
    md("""### Secondary validation audit: coordinate-loss weight

Training-gradient diagnostics motivated a separately declared audit of the coordinate-loss weight, increasing it from 1 to 10 while retaining the same normal pretraining, source schedule, and three seeds. This is a secondary sensitivity analysis. The original weight-1 method remains the primary method regardless of whether the audit improves validation error. Its normal-source statistics are reused under the same clean-cohort equality certificate; filtered validation predictions retain their original lineage.

When its development artifact exists, the table reports actual normal-source error, the original method's error, persistence, and the predeclared context controls. It contains validation evidence only. An absent artifact is shown as pending and is never replaced with a synthetic result."""),
    code("""objective_audit_path = Path(os.getenv("WM_OBJECTIVE_AUDIT") or PROJECT / "artifacts/runs/objective-audit-development-v1/development.json")
if REAL and objective_audit_path.is_file():
    objective_audit_report = json.loads(objective_audit_path.read_text())
    if objective_audit_report.get("declaration", {}).get("identity") != "coordinate-weight-ten-secondary-audit-v1":
        raise ValueError("Unexpected objective-audit identity")
    display(pd.DataFrame(objective_audit_report["records"]))
    print("Secondary validation audit only. The original weight-1 method remains primary.")
    objective_audit_output = lesson_directory("07_ablation_and_paper_evidence") / "objective_audit_validation.json"
    objective_audit_output.write_text(json.dumps(objective_audit_report, indent=2))
else:
    print("Secondary coordinate-weight audit is pending; no result has been substituted.")""")]
nbf.write(revision_book, ROOT / '07_ablation_and_paper_evidence.ipynb')

# The capstone exposes the actual frozen direct-binary endpoint for a video,
# alongside the separate five-class interpretation of the same interval.
capstone = nbf.read(ROOT / '06_capstone_rf_vs_jepa.ipynb', as_version=4)
binary_video_insert = next(i for i, cell in enumerate(capstone.cells)
                          if cell.cell_type == 'markdown' and cell.source.startswith('## Step 8.'))
capstone.cells[binary_video_insert:binary_video_insert] = [
    md("""### Predict normal versus abnormal for that same walking interval

The five-class probabilities above describe the condition-label task. The direct binary classifiers below are separately fitted frozen readouts for normal versus abnormal. They use the same input video interval and the same frozen feature/encoder contracts. This call applies the actual Random Forest and primary JEPA binary classifiers; it does not derive its prediction by summing the five-class probabilities.

The input example belongs to a training source for this tutorial demonstration, so its prediction is not a held-out performance estimate. The reported probabilities are research outputs for dataset annotations, not a clinical diagnosis. No truth label is passed to either video-inference function."""),
    code("""if REAL or os.getenv("WM_RUN_INFERENCE", "0") == "1":
    from worldmodels.binary_inference import classify_video as classify_binary_video
    binary_development = Path(os.getenv("WM_BINARY_DEVELOPMENT") or PROJECT / "artifacts/runs/binary-development-clean-v1")
    binary_video_report = classify_binary_video(
        source_video, binary_development, REPOSITORY / "data/models/pose_landmarker_lite.task",
        start_frame=int(sample.first_frame),
        end_frame=min(int(sample.last_frame), int(sample.first_frame)+95))
    if not binary_video_report.get("classifiers"):
        raise RuntimeError("Actual direct-binary video inference returned no predictions")
    binary_video_rows = []
    for name, result in binary_video_report["classifiers"].items():
        binary_video_rows.append({"model": name, **result["probabilities"],
                                  "predicted_annotation": result["predicted_annotation"],
                                  "threshold": result["threshold"]})
    display(pd.DataFrame(binary_video_rows))
    report["direct_binary"] = binary_video_report
    binary_video_output = lesson_directory("06_capstone_rf_vs_jepa") / "binary_video_report.json"
    binary_video_output.write_text(json.dumps(binary_video_report, indent=2))
else:
    print("Direct-binary input-video inference runs in real mode or with WM_RUN_INFERENCE=1.")""")]
nbf.write(capstone, ROOT / '06_capstone_rf_vs_jepa.ipynb')

# Stable cell IDs make regeneration reproducible and preserve evidence identities
# for lessons whose content did not change.
import hashlib
for notebook_path in sorted(ROOT.glob('*.ipynb')):
    notebook = nbf.read(notebook_path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        identity = f'{notebook_path.name}:{index}:{cell.cell_type}:{cell.source}'
        cell.id = hashlib.sha256(identity.encode()).hexdigest()[:16]
    nbf.write(notebook, notebook_path)
