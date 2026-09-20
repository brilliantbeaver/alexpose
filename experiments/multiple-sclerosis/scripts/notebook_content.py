"""Content for the seven S-JEPA tutorial notebooks.

Each ``nb_XX`` function returns the ordered list of cells for one notebook. The
``build`` function at the bottom assembles them with the shared header. Kept in a
separate module so the builder script stays readable. All code cells use the
verified sjepa package APIs, so what runs in the notebook matches what we tested.
"""

from __future__ import annotations


def nb_00(md, code, badge, boot):
    c = [badge("00_overview_and_video_gallery.ipynb")]
    c += [md(
        "# 00 - Overview and video gallery\n",
        "Welcome. This series trains **S-JEPA**, a model that learns structure from walking motion "
        "without condition labels, then tests whether that representation helps distinguish three "
        "labels:\n",
        "- **Normal** gait\n- **MS**: multiple sclerosis\n- **PD**: Parkinson's disease\n",
        "These are dataset labels, not diagnoses made by this project. We compare a supervised "
        "probe with a classical Random Forest on matched source-grouped splits.\n",
        "This first notebook sets the scene. We look at the data, count it honestly, and actually "
        "watch a few clips so the later math stays grounded in real movement.\n",
        "### How to run\n",
        "You can run every notebook two ways:\n",
        "1. **Locally** with `uv`. From the repo root: `cd experiments/multiple-sclerosis && uv sync`, "
        "then open the notebooks in Jupyter or VS Code.\n",
        "2. **In Google Colab** by clicking the badge at the top. The setup cells install what is "
        "missing and clone the repo so `import sjepa` works.\n",
    )]
    c += boot(need_torch=False)
    c += [md(
        "## The pipeline at a glance\n",
        "Both the learned approach and the classical baseline start from the same pose front-end, "
        "then split into two branches, and finally meet again for a fair comparison.\n",
    )]
    c += [code(
        "from IPython.display import SVG, display",
        "display(SVG(filename=str(IMAGES_DIR / 'pipeline_flowchart.svg')))",
    )]
    c += [md(
        "## The dataset, counted honestly\n",
        "The current `video-data-full/` collection contains **91 MP4 clips from 41 source videos**: "
        "26 clips from 16 Normal sources, 30 from 13 MS sources, and 35 from 12 PD sources. The first "
        "11 filename characters identify the source video; `_P...` suffixes distinguish clips. A "
        "source ID is a conservative grouping key, not a verified participant ID.\n",
    )]
    c += [code(
        "import pandas as pd",
        "from sjepa.data import source_id_from_name",
        "",
        "CLASS_DIRS = {'normal': 'Normal', 'ms': 'MS', 'pd': 'PD'}",
        "rows = []",
        "for label, folder in CLASS_DIRS.items():",
        "    for vid in sorted((VIDEO_DIR / folder).glob('*.mp4')):",
        "        rows.append(dict(label=label, clip=vid.name, source_id=source_id_from_name(vid.name)))",
        "manifest = pd.DataFrame(rows)",
        "summary = manifest.groupby('label').agg(clips=('clip', 'count'),",
        "                                        sources=('source_id', 'nunique'))",
        "print(summary)",
        "print('\\ntotal clips:', len(manifest), '| total sources:', manifest.source_id.nunique())",
        "manifest.to_csv(ARTIFACT_DIR / 'manifest_grouped.csv', index=False)",
    )]
    c += [md(
        "One MS source contributes 14 clips and one PD source contributes seven. We therefore split "
        "by source, not by clip. Recording conditions also differ by label, so later notebooks compare "
        "learned features with acquisition-related nuisance controls.\n",
        "## Watch a few walks\n",
        "The cell below embeds one clip per label. Inspect the walk and the camera angle, framing, "
        "resolution, background, and duration. One web clip cannot establish a diagnosis or represent "
        "its whole label.\n",
    )]
    c += [code(
        "from sjepa.viz import show_video",
        "from IPython.display import display",
        "",
        "for label, folder in CLASS_DIRS.items():",
        "    clip = sorted((VIDEO_DIR / folder).glob('*.mp4'))[0]",
        "    print(f'{label}: {clip.name}')",
        "    display(show_video(clip, width=360))",
    )]
    c += [md(
        "## Roadmap\n",
        "| Notebook | What you build |\n|---|---|\n"
        "| 00 overview | this tour of the data and the plan |\n"
        "| 01 pose extraction | turn clips into skeleton sequences with MediaPipe |\n"
        "| 02 mask and tokens | build stochastic masks and turn skeletons into tokens |\n"
        "| 03 label-free pretraining | train S-JEPA only on each fold's training sources |\n"
        "| 04 adaptation | compare extra label-free training with a supervised probe |\n"
        "| 05 representations | compare learned and nuisance-feature projections |\n"
        "| 06 capstone | compare RF, S-JEPA, and controls on source-grouped folds |\n",
        "On to notebook 01, where we turn these clips into skeletons.\n",
    )]
    return c


def nb_01(md, code, badge, boot):
    c = [badge("01_pose_extraction_from_raw_video.ipynb")]
    c += [md(
        "# 01 - Pose extraction from raw video\n",
        "The model reads **skeleton sequences**, not image pixels directly. We run MediaPipe BlazePose "
        "over each of the 91 clips and estimate 33 landmarks per sampled frame. Each cache record keeps "
        "both its clip name and its source-video ID.\n",
        "Each usable clip becomes an array of shape `(T, 33, 3)`: sampled frames, landmarks, and pixel "
        "x, pixel y, and visibility. We clean and normalize it, then cache it for later stages.\n",
    )]
    c += boot(need_torch=False)
    c += [md(
        "## One frame at a time\n",
        "The raw collection ranges from about 24 to 60 fps and uses several resolutions. The loader "
        "retains frames at an approximately 15 fps cadence and asks MediaPipe for a pose in each one. "
        "Unlike the GAVD pipeline, it reads the whole frame without bounding boxes or annotation CSVs.\n",
    )]
    c += [code(
        "from ambient.pose.model_management import MediaPipeModelManager",
        "from ambient.pose.keypoint_extractor import SequenceKeypointExtractor",
        "from sjepa.data import load_video_sequence, clean_sequence, normalize_sequence",
        "",
        "MediaPipeModelManager().ensure_model_available()  # downloads the model once",
        "extractor = SequenceKeypointExtractor()",
        "",
        "CLASS_DIRS = {'normal': 'Normal', 'ms': 'MS', 'pd': 'PD'}",
        "sample = sorted((VIDEO_DIR / CLASS_DIRS['normal']).glob('*.mp4'))[0]",
        "seq = load_video_sequence(sample, target_fps=15, extractor=extractor, verbose=True)",
        "print('raw sequence shape:', seq.shape, '  (frames, joints, [x, y, visibility])')",
    )]
    c += [md(
        "## Clean and normalize\n",
        "Real videos have frames where the detector loses the person. We interpolate short gaps and "
        "drop videos that are mostly empty. Then we **normalize**: we move the pelvis to the origin "
        "and scale by the torso length. This reduces translation and apparent-size differences, but "
        "does not remove viewpoint, frame-rate, resolution, compression, or detector-confidence cues.\n",
    )]
    c += [code(
        "cleaned = clean_sequence(seq)",
        "normalized = normalize_sequence(cleaned)",
        "print('cleaned:', cleaned.shape, '| normalized:', normalized.shape)",
        "print('normalized x range:', round(float(normalized[:,:,0].min()),2),",
        "      'to', round(float(normalized[:,:,0].max()),2))",
    )]
    c += [md(
        "## See the skeleton move\n",
        "Here is the skeleton the model will actually train on. The animation draws the BlazePose "
        "stick figure over time. This is the same view we will reuse in notebook 02 to show the mask.\n",
    )]
    c += [code(
        "from sjepa.viz import skeleton_animation",
        "from IPython.display import Image",
        "",
        "gif = skeleton_animation(normalized, ARTIFACT_DIR / 'demo_skeleton.gif',",
        "                         fps=15, title='normal gait (normalized)')",
        "Image(filename=str(gif))",
    )]
    c += [md(
        "## Extract and cache every clip\n",
        "We now process all 91 raw clips and write one `.npz` per usable clip. The final count may be "
        "smaller because clips with too little valid pose signal are skipped. The committed `g1` cache "
        "comes from an earlier 47-clip collection and must not be mixed with this versioned full cache.\n",
    )]
    c += [code(
        "from sjepa.data import save_sequence_npz, source_id_from_name",
        "import numpy as np",
        "",
        "KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)",
        "index = []",
        "for label, folder in CLASS_DIRS.items():",
        "    for vid in sorted((VIDEO_DIR / folder).glob('*.mp4')):",
        "        sid = source_id_from_name(vid.name)",
        "        out = KEYPOINTS_DIR / f'{label}__{sid}__{vid.stem}.npz'",
        "        if out.exists():",
        "            with np.load(out, allow_pickle=True) as z:",
        "                n = int(z['keypoints_norm'].shape[0])",
        "        else:",
        "            raw = load_video_sequence(vid, target_fps=15, extractor=extractor)",
        "            cl = clean_sequence(raw)",
        "            if cl is None or cl.shape[0] < 8:",
        "                print('skip (too few valid frames):', vid.name); continue",
        "            nm = normalize_sequence(cl)",
        "            save_sequence_npz(out, cl, nm, 15, sid, label, vid.stem)",
        "            n = nm.shape[0]",
        "        index.append(dict(label=label, source_id=sid, clip_name=vid.stem, n_frames=n))",
        "",
        "import pandas as pd",
        "idx = pd.DataFrame(index)",
        "idx.to_parquet(ARTIFACT_DIR / 'keypoints_index_full.parquet', index=False)",
        "print(idx.groupby('label').agg(clips=('clip_name','count'),",
        "                               sources=('source_id','nunique'),",
        "                               frames=('n_frames','sum')))",
    )]
    c += [md(
        "### Quick checks\n",
        "Shape checks verify structure, not validity. Also report accepted and rejected clips by label "
        "and source before training.\n",
    )]
    c += [code(
        "from sjepa.data import load_index",
        "recs = load_index(KEYPOINTS_DIR)",
        "assert len(recs) > 0",
        "for r in recs[:5]:",
        "    a = r.load_norm()",
        "    assert a.ndim == 3 and a.shape[1] == 33 and a.shape[2] == 3",
        "print(f'cached {len(recs)} sequences, all shaped (T, 33, 3). Ready for notebook 02.')",
    )]
    return c


def nb_02(md, code, badge, boot):
    c = [badge("02_anatomical_mask_and_tokenization.ipynb")]
    c += [md(
        "# 02 - Masking and tokenization\n",
        "S-JEPA learns by hiding part of a skeleton sequence and predicting it in feature space. The "
        "token and mask operations apply to either dataset version; the dataset changes the number of "
        "windows and source groups, not the token definition. Run notebook 01 first to build the "
        "versioned `keypoints-full/` cache.\n",
        "> **What changed, and why.** An earlier version of this project hid the *same* twelve clinical "
        "joints on every single step. That turned out to be a real bug: the encoder never saw those "
        "joints as context, so their internal position settings received no learning signal, yet the "
        "classifier then pooled exactly those joints. We now use **stochastic graph-time masks**: a "
        "different connected group of joints is hidden each step, so every joint is sometimes context "
        "and sometimes a target. Clinical knowledge still guides us, but gently, by choosing the "
        "leg and shoulder joints as targets a bit more often. We also do **not** bias toward the "
        "busiest joints (the paper's motion-aware masking), because in MS and PD the telling sign is "
        "often *reduced* motion, which a high-motion mask would hide.\n",
    )]
    c += boot(need_torch=False)
    c += [md(
        "## Tokenizing a window\n",
        "A training window is a short movie of stick figures. We group `l = 4` adjacent frames of one "
        "joint into a single token, so each token summarizes how that joint moved over a moment. With "
        "32 frames and 33 joints that gives `(32 / 4) x 33 = 264` tokens. Token index is `t * V + v` "
        "(time block `t`, joint `v`).\n",
    )]
    c += [code(
        "from IPython.display import SVG, display",
        "display(SVG(filename=str(IMAGES_DIR / 'tokenization.svg')))",
    )]
    c += [code(
        "from sjepa.config import get_config, describe",
        "cfg = get_config()  # honours SJEPA_PROFILE",
        "print(describe(cfg))",
        "print('tokens per window N =', cfg.num_tokens,",
        "      f'= {cfg.num_time_tokens} time blocks x {cfg.num_joints} joints')",
    )]
    c += [md(
        "## The clinical joints (domain context, not a permanent mask)\n",
        "The file `mapping-data/ms-pd-mapping.md` lists the joints clinicians care about for ms and "
        "pd. After removing duplicates and sorting, we get exactly twelve BlazePose landmarks: both "
        "shoulders and both complete legs. We keep this table as **domain knowledge** that biases how "
        "often a joint is chosen as a target, but every joint can still be both context and target.\n",
    )]
    c += [code(
        "from sjepa.masking_v2 import CLINICAL_JOINTS",
        "from ambient.pose.keypoint_data import MEDIAPIPE_33_NAMES",
        "import pandas as pd",
        "",
        "features_for = {",
        "    11: 'shoulder_symmetry_index, trunk_lean_angle',",
        "    12: 'shoulder_symmetry_index, trunk_lean_angle',",
        "    23: 'walking_speed_ms, hip_asymmetry, knee_range, trunk_lean_angle',",
        "    24: 'walking_speed_ms, hip_asymmetry, knee_range, trunk_lean_angle',",
        "    25: 'knee_range, ankle_range', 26: 'knee_range, ankle_range',",
        "    27: 'knee_range, ankle_range, step_width_m', 28: 'knee_range, ankle_range, step_width_m',",
        "    29: 'stride_length_m, double_support_pct, stride_time_cv, ankle_range',",
        "    30: 'stride_length_m, double_support_pct, stride_time_cv, ankle_range',",
        "    31: 'stride_length_m, double_support_pct, stride_time_cv, ankle_range',",
        "    32: 'stride_length_m, double_support_pct, stride_time_cv, ankle_range',",
        "}",
        "table = pd.DataFrame([",
        "    {'BLAZEPOSE_33 index': j, 'Keypoint name': MEDIAPIPE_33_NAMES[j],",
        "     'Features involved': features_for[j]}",
        "    for j in sorted(CLINICAL_JOINTS)",
        "])",
        "table",
    )]
    c += [md(
        "## Stochastic graph-time masks\n",
        "Each step we sample a per-example mask: connected groups of joints (a limb or the trunk) over "
        "a contiguous span of time. The cell below samples a few masks and shows they differ, that "
        "every one keeps visible context, and that over a bank of masks every joint is both visible "
        "and targeted often enough (the coverage gates).\n",
    )]
    c += [code(
        "import numpy as np",
        "from sjepa.masking_v2 import sample_mask_batch, mask_bank_stats",
        "",
        "rng = np.random.default_rng(0)",
        "batch = sample_mask_batch(6, cfg.num_joints, cfg.num_time_tokens, rng)",
        "print('mask batch shape (B, N):', batch.shape)",
        "print('unique masks in the batch:', len({row.tobytes() for row in batch}), 'of 6')",
        "print('every row has context and target:',",
        "      bool((~batch).any(1).all() and batch.any(1).all()))",
        "",
        "stats = mask_bank_stats(cfg.num_joints, cfg.num_time_tokens, n_masks=512, seed=0)",
        "print(f'over 512 masks: min joint-visible {stats.joint_visible_frac.min():.2f} '",
        "      f'(gate >=0.20), min joint-target {stats.joint_target_frac.min():.2f} (gate >=0.10)')",
        "print(f'mean target fraction {stats.mean_target_frac:.2f}')",
    )]
    c += [md(
        "Here is the difference drawn out: a fixed mask hides the same joints forever (left), while "
        "stochastic masks rotate which joints are hidden (right).\n",
    )]
    c += [code(
        "display(SVG(filename=str(IMAGES_DIR / 'defect_mask_starvation.svg')))",
    )]
    c += [md(
        "## See one mask on a real skeleton\n",
        "The animation highlights one sampled set of masked joints in red on a real walking sequence. "
        "Next time you sample, a different group will be hidden.\n",
    )]
    c += [code(
        "from sjepa.data import load_index",
        "from sjepa.masking_v2 import sample_target_mask",
        "from sjepa.viz import skeleton_animation",
        "from IPython.display import Image",
        "",
        "recs = load_index(KEYPOINTS_DIR)",
        "seq = recs[0].load_norm()",
        "tgt = sample_target_mask(cfg.num_joints, cfg.num_time_tokens, np.random.default_rng(1))",
        "masked_joints = sorted({int(i % cfg.num_joints) for i in np.nonzero(tgt)[0]})",
        "gif = skeleton_animation(seq, ARTIFACT_DIR / 'mask_demo.gif',",
        "                         masked_joints=masked_joints, fps=15,",
        "                         title='red = one sampled set of masked joints')",
        "Image(filename=str(gif))",
    )]
    return c


def nb_03(md, code, badge, boot):
    c = [badge("03_sjepa_model_and_pretrain_normal.ipynb")]
    c += [md(
        "# 03 - Build S-JEPA and pretrain (label-free)\n",
        "Now we build the model and train it, with **no condition labels in the objective**, on walking motion "
        "itself. S-JEPA has three parts, all small transformers:\n",
        "- a **view encoder** that reads the visible joints of a slightly transformed view,\n"
        "- a **predictor** that guesses the hidden joints in feature space. Crucially, it is told "
        "*which* joint and *which* time each hidden slot is (a factorized position tag), so it can "
        "make a different guess per position. Without that tag every hidden guess is identical, which "
        "was a real bug in an earlier version.\n"
        "- a **target encoder** that reads the full skeleton and provides the answer. It is a slow "
        "moving average of the view encoder, which is what stops the model from collapsing every "
        "skeleton to the same features.\n",
        "> **A note on what trains here.** This notebook learns from every clip assigned to the fold's "
        "training sources. Labels stay attached for evaluation but are not read by the S-JEPA loss. "
        "Training only on Normal-labeled gait would answer a different anomaly-detection question.\n",
        "> **Dataset-version note.** The committed `g1` folds and checkpoints describe the earlier "
        "47-clip, 35-source-group benchmark. The current 91-clip, 41-source raw collection needs a "
        "new cache, fold registry, and checkpoints before it has a full-data result.\n",
    )]
    c += boot(need_torch=True, legacy_artifacts=True)
    c += [md(
        "## The two-lane design\n",
        "The picture below is the whole idea. The top lane makes a prediction from a masked view. The "
        "bottom lane makes the target from the complete skeleton with a slow teacher. They meet only "
        "at the loss.\n",
    )]
    c += [code(
        "from IPython.display import SVG, display",
        "display(SVG(filename=str(IMAGES_DIR / 'sjepa_two_lane.svg')))",
        "display(SVG(filename=str(IMAGES_DIR / 'defect_predictor_positions.svg')))",
    )]
    c += [code(
        "from sjepa.config import get_config, describe",
        "from sjepa.models import build_model, pick_device",
        "",
        "cfg = get_config()",
        "device = pick_device()",
        "print('device:', device)",
        "print(describe(cfg))",
        "model = build_model(cfg, device=device, repaired=True)  # PredictorV2 + per-example masks",
        "n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)",
        "print(f'trainable parameters: {n_params/1e6:.2f}M')",
    )]
    c += [md(
        "## Load the training windows (fold 0's training sources, no labels)\n",
        "For the retained benchmark, fold 0 has 37 training clips from 29 provisional source groups "
        "and ten held-out clips from six groups. Held-out motion remains unseen even though S-JEPA "
        "does not read labels. A full-data run needs a newly frozen registry using the updated `_P...` "
        "source grouping, especially for the 14 MS clips from one source video.\n",
    )]
    c += [code(
        "import json",
        "from sjepa.data import load_index, SequenceWindowDataset",
        "",
        "records = load_index(KEYPOINTS_DIR)",
        "by_clip = {r.clip_name: r for r in records}",
        "registry = json.loads((ARTIFACT_DIR / 'eval' / 'g1' / 'fold_registry.json').read_text())",
        "fold0 = registry['folds'][0]",
        "train_recs = [by_clip[c] for c in fold0['train_clips']]",
        "test_srcs = {by_clip[c].source_id for c in fold0['test_clips']}",
        "assert not ({r.source_id for r in train_recs} & test_srcs), 'held-out source leaked into SSL'",
        "ds_train = SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride)",
        "print(f'fold 0 training partition: {len(train_recs)} clips -> {len(ds_train)} windows',",
        "      '(labels unused in SSL; held-out clips excluded)')",
    )]
    c += [md(
        "## Train\n",
        "We run the two-lane objective for a fixed number of **optimizer updates** (not "
        "epochs), sampling sources uniformly so a long clip cannot dominate, and nudge the teacher "
        "with a responsive EMA. We log the loss and collapse diagnostics: the per-dimension spread, "
        "the effective rank (how many directions the features really use), and how far the teacher has "
        "drifted from the student.\n",
    )]
    c += [code(
        "from sjepa.train_v2 import train_sjepa_v2, save_checkpoint_v2",
        "",
        "# A small update budget keeps the notebook fast; raise it for stronger features.",
        "# `cfg.profile` ends in '+smoke' only when SJEPA_SMOKE is truthy (the config parses",
        "# '0'/'1'/'true' correctly, unlike a raw os.environ.get truthiness test).",
        "SMOKE = cfg.profile.endswith('smoke')",
        "UPDATES = 60 if SMOKE else 800",
        "state = train_sjepa_v2(model, ds_train, cfg, total_updates=UPDATES, device=device,",
        "                       mask_ratio=0.6, log_every=max(1, UPDATES // 4))",
        "save_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl.pt', model, cfg, train_state=state,",
        "                   extra={'stage': 'ssl_fold0_train'})",
        "print('EMA half-life (steps):', round(state.ema_half_life_steps, 1))",
    )]
    c += [code(
        "import matplotlib.pyplot as plt",
        "fig, ax = plt.subplots(1, 3, figsize=(12, 3))",
        "ax[0].plot(state.losses, color='#dd6b20'); ax[0].set_title('latent cross-entropy')",
        "ax[1].plot(state.eff_rank, color='#2563eb'); ax[1].set_title('effective rank (higher = richer)')",
        "ax[2].plot(state.teacher_drift, color='#16a34a'); ax[2].set_title('teacher drift from student')",
        "for a in ax: a.set_xlabel('update')",
        "plt.tight_layout(); plt.show()",
    )]
    c += [md(
        "### Sanity checks\n",
        "We avoid weak checks. A falling loss is necessary but not sufficient: a collapsed model can "
        "also lower the loss. So we also require the effective rank to stay well above 1 (the features "
        "use many directions, not one) and the teacher to have moved.\n",
    )]
    c += [code(
        "import numpy as np",
        "early = np.mean(state.losses[:5]); late = np.mean(state.losses[-5:])",
        "print(f'loss {early:.3f} -> {late:.3f} | final effective rank {state.eff_rank[-1]:.1f}')",
        "assert np.isfinite(state.losses).all(), 'loss went non-finite'",
        "assert state.eff_rank[-1] > 1.5, 'representation looks collapsed (effective rank near 1)'",
        "assert state.teacher_drift[-1] >= 0, 'teacher drift should be finite and non-negative'",
        "print('SSL looks healthy (no collapse). On to comparing training regimes.')",
    )]
    return c


def nb_04(md, code, badge, boot):
    c = [badge("04_progressive_finetune_ms_pd_vicreg.ipynb")]
    c += [md(
        "# 04 - Two ways to adapt the encoder: SSL continuation vs supervised adaptation\n",
        "Notebook 03 trained the encoder without condition labels in its objective, using only each "
        "fold's training sources. Here we ask where labels may enter and whether extra label-free "
        "training changes a supervised probe's performance.\n",
        "We compare two clearly named regimes, both starting from the same label-free checkpoint:\n",
        "1. **SSL continuation** - keep training with the *same* label-free objective for more updates. "
        "No labels touch the model.\n"
        "2. **Balanced supervised adaptation** - freeze the encoder and fit a small, class-balanced "
        "linear head on top. Labels are used *only* in this head, never inside the self-supervised "
        "objective.\n",
        "> **What changed, and why.** An earlier version of this notebook mixed the diagnosis label "
        "*into* the self-supervised loss (a 'class-aware VICReg' that rewarded within-class spread) and "
        "claimed it 'compacts the classes'. That was a leak: the SSL objective must not see labels, and "
        "the claim was not supported. We removed it. If labels help, they help in an explicitly "
        "supervised stage that we name and measure, not smuggled into the pretext task.\n",
        "> **Dataset-version note.** The executable example uses the earlier `g1` benchmark of 47 "
        "clips from 35 source groups. It does not measure the expanded 91-clip raw collection.\n",
    )]
    c += boot(need_torch=True, legacy_artifacts=True)
    c += [code(
        "from IPython.display import SVG, display",
        "display(SVG(filename=str(IMAGES_DIR / 'progressive_timeline.svg')))",
    )]
    c += [md(
        "## Use the locked, leakage-safe fold registry\n",
        "The comparison is meaningful only when clips from one source never straddle train and test. "
        "Phase 0 froze the earlier source-grouped benchmark in `artifacts/eval/g1/fold_registry.json`. "
        "A current full-data analysis needs a new versioned registry rather than modifying `g1`.\n",
        "> Source grouping is **provisional**: a `source_id` is a YouTube id, not a verified person, so "
        "everything here is a development estimate, never a clinical claim.\n",
    )]
    c += [code(
        "import json",
        "from sjepa.data import load_index",
        "",
        "records = load_index(KEYPOINTS_DIR)",
        "by_clip = {r.clip_name: r for r in records}",
        "registry = json.loads((ARTIFACT_DIR / 'eval' / 'g1' / 'fold_registry.json').read_text())",
        "fold0 = registry['folds'][0]",
        "train_recs = [by_clip[c] for c in fold0['train_clips']]",
        "test_recs  = [by_clip[c] for c in fold0['test_clips']]",
        "tr_src = {r.source_id for r in train_recs}; te_src = {r.source_id for r in test_recs}",
        "assert not (tr_src & te_src), 'source leakage across the fold'",
        "print('fold 0:', len(train_recs), 'train clips /', len(test_recs), 'test clips')",
        "print('no source in both sides:', not (tr_src & te_src))",
    )]
    c += [md(
        "## Regime 1 - SSL continuation (no labels)\n",
        "We rebuild the model, load the label-free checkpoint from notebook 03, and keep "
        "training with the exact same objective for a few hundred more updates. The label is never "
        "read. This asks: does more unlabeled training alone sharpen the representation?\n",
    )]
    c += [code(
        "from sjepa.config import get_config",
        "from sjepa.models import build_model, pick_device",
        "from sjepa.train_v2 import train_sjepa_v2, load_checkpoint_v2, save_checkpoint_v2",
        "from sjepa.data import SequenceWindowDataset",
        "",
        "cfg = get_config()",
        "device = pick_device()",
        "model = build_model(cfg, device=device, repaired=True)",
        "load_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl.pt', model, map_location=device)",
        "",
        "ds_train = SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride)",
        "SMOKE = cfg.profile.endswith('smoke')  # correct parse of SJEPA_SMOKE (not a raw truthiness test)",
        "MORE = 40 if SMOKE else 400",
        "state = train_sjepa_v2(model, ds_train, cfg, total_updates=MORE, device=device,",
        "                       mask_ratio=0.6, log_every=max(1, MORE // 4))",
        "save_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl_continued.pt', model, cfg, train_state=state,",
        "                   extra={'stage': 'ssl_continuation_fold0'})",
        "print('continued SSL: final effective rank', round(state.eff_rank[-1], 1))",
    )]
    c += [md(
        "## A fixed, label-free read-out\n",
        "To turn a clip into one vector we mean-pool the frozen target encoder over a **fixed** pool "
        "of target tokens. The pool is chosen once from a seeded RNG and never from the test labels, so "
        "no information leaks from the evaluation into the representation. Both regimes below use this "
        "same read-out.\n",
    )]
    c += [code(
        "import numpy as np, torch",
        "from sjepa.masking_v2 import sample_target_mask",
        "from sjepa.data import sliding_windows",
        "",
        "readout = sample_target_mask(cfg.num_joints, cfg.num_time_tokens,",
        "                             np.random.default_rng(0), target_ratio=0.6)",
        "tm = torch.from_numpy(readout).to(device)",
        "",
        "def embed_records(m, recs):",
        "    V, Y = [], []",
        "    for r in recs:",
        "        w = sliding_windows(r.load_norm(), cfg.window_frames, cfg.window_stride)",
        "        x = torch.from_numpy(w).float().to(device)",
        "        with torch.no_grad():",
        "            V.append(m.embed(x, tm).mean(0).cpu().numpy())",
        "        Y.append(r.label)",
        "    return np.stack(V), Y",
    )]
    c += [md(
        "## Regime 2 - balanced supervised adaptation (labels only in the head)\n",
        "Now we use the labels honestly: freeze the encoder and fit a class-balanced logistic head on "
        "the training embeddings, then score the held-out clips. The scaler and the head are fit on "
        "**training data only**. We do this on top of both the notebook-03 checkpoint and the "
        "SSL-continued one, so we can see whether extra unlabeled training moved the probe at all.\n",
    )]
    c += [code(
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.preprocessing import StandardScaler",
        "from sjepa.eval import evaluate",
        "LABELS = ['normal', 'ms', 'pd']",
        "",
        "def probe_and_score(ckpt):",
        "    m = build_model(cfg, device=device, repaired=True)",
        "    load_checkpoint_v2(ckpt, m, map_location=device)",
        "    Etr, ytr = embed_records(m, train_recs)",
        "    Ete, yte = embed_records(m, test_recs)",
        "    sc = StandardScaler().fit(Etr)               # fit on TRAIN only",
        "    clf = LogisticRegression(max_iter=2000, class_weight='balanced')",
        "    clf.fit(sc.transform(Etr), ytr)",
        "    return evaluate(yte, clf.predict(sc.transform(Ete)), LABELS)",
        "",
        "m_base = probe_and_score(ARTIFACT_DIR / 'sjepa_ssl.pt')",
        "m_cont = probe_and_score(ARTIFACT_DIR / 'sjepa_ssl_continued.pt')",
        "print(f'supervised probe on fold 0 held-out videos (macro-F1):')",
        "print(f'  notebook-03 checkpoint : {m_base.macro_f1:.3f}')",
        "print(f'  after SSL continuation : {m_cont.macro_f1:.3f}')",
    )]
    c += [md(
        "## Read this honestly\n",
        "This legacy fold has only ten held-out clips from six source groups. Its scores demonstrate "
        "the method boundary but are not a result for the expanded collection. Notebook 06 aggregates "
        "the five legacy folds; the full dataset still requires a complete rerun.\n",
    )]
    return c


def nb_05(md, code, badge, boot):
    c = [badge("05_representation_visualization.ipynb")]
    c += [md(
        "# 05 - Looking at the learned representation (diagnostics only)\n",
        "We summarize each cached clip with a frozen encoder and project those vectors to two "
        "dimensions. The plots show point arrangement; they do not identify which physical or "
        "acquisition features created it.\n",
        "> **These pictures are diagnostics, not evidence.** Two honest cautions run through this "
        "notebook. First, t-SNE and UMAP distort distances; a clean-looking blob can be an artifact of "
        "the projection. Second, and more important, apparent separation can come from a **shortcut** "
        "(camera frame rate, body size, how visible the joints are) rather than from gait. So we plot "
        "the S-JEPA embedding *and* a cheap nuisance feature side by side: if the nuisance separates "
        "just as well, the pretty S-JEPA plot is not telling us about gait. The verdict lives in "
        "notebook 06's leakage-safe scores, never in a scatter plot.\n",
        "We compare the label-free checkpoint from notebook 03 against the SSL-continued one from "
        "notebook 04. No test labels are ever used to fit, select, or color anything beyond the plain "
        "class of each point.\n",
        "> **Dataset-version note.** The committed embeddings describe the legacy 47-clip cache. The "
        "current 91-clip raw collection needs pose extraction and retraining before it has new plots.\n",
    )]
    c += boot(need_torch=True, legacy_artifacts=True)
    c += [code(
        "from IPython.display import SVG, display",
        "display(SVG(filename=str(IMAGES_DIR / 'vicreg_clusters.svg')))",
    )]
    c += [md(
        "## Embed every clip with the frozen encoder\n",
        "For each cached clip we mean-pool the encoder's features over its windows and over a **fixed**, "
        "seeded read-out pool of target tokens (the same pool used in notebooks 04 and 06). This pool "
        "is never chosen from labels. We do it for both checkpoints.\n",
    )]
    c += [code(
        "import numpy as np, torch",
        "from sjepa.config import get_config",
        "from sjepa.models import build_model, pick_device",
        "from sjepa.train_v2 import load_checkpoint_v2",
        "from sjepa.masking_v2 import sample_target_mask",
        "from sjepa.data import load_index, sliding_windows",
        "",
        "cfg = get_config(); device = pick_device()",
        "records = load_index(KEYPOINTS_DIR)",
        "readout = sample_target_mask(cfg.num_joints, cfg.num_time_tokens,",
        "                             np.random.default_rng(0), target_ratio=0.6)",
        "tm = torch.from_numpy(readout).to(device)",
        "",
        "def embed_all(ckpt):",
        "    m = build_model(cfg, device=device, repaired=True)",
        "    load_checkpoint_v2(ckpt, m, map_location=device)",
        "    vecs, labels = [], []",
        "    for r in records:",
        "        w = sliding_windows(r.load_norm(), cfg.window_frames, cfg.window_stride)",
        "        x = torch.from_numpy(w).float().to(device)",
        "        with torch.no_grad():",
        "            vecs.append(m.embed(x, tm).mean(0).cpu().numpy())",
        "        labels.append(r.label)",
        "    return np.stack(vecs), labels",
        "",
        "E_base, y = embed_all(ARTIFACT_DIR / 'sjepa_ssl.pt')",
        "E_cont, _ = embed_all(ARTIFACT_DIR / 'sjepa_ssl_continued.pt')",
        "np.savez(ARTIFACT_DIR / 'embeddings_3class.npz', E_base=E_base, E_continued=E_cont,",
        "         labels=np.array(y))",
        "print('embedded', len(y), 'clips into', E_cont.shape[1], 'dimensions')",
    )]
    c += [md(
        "## A nuisance baseline to keep us honest\n",
        "This nuisance representation contains only the per-joint mean and spread of MediaPipe "
        "visibility. In the current raw collection, all 30 MS clips are 30 fps landscape recordings "
        "at two resolutions, while 30 of 35 PD clips are about 30 fps at 1280 x 720 and Normal is more "
        "heterogeneous. The legacy benchmark had a different acquisition shortcut. A nuisance plot "
        "that separates labels supplies a competing explanation; it does not identify S-JEPA's cause.\n",
    )]
    c += [code(
        "def nuisance_vec(r):",
        "    vis = r.load_raw()[:, :, 2]",
        "    return np.nan_to_num(np.concatenate([np.nanmean(vis, 0), np.nanstd(vis, 0)]))",
        "N_nuis = np.stack([nuisance_vec(r) for r in records])",
        "print('nuisance feature shape:', N_nuis.shape)",
    )]
    c += [md(
        "## Project and plot\n",
        "t-SNE squeezes the high-dimensional vectors into a plane while trying to keep neighbors "
        "together. We color one hue per condition, and we place the nuisance baseline in the same row "
        "for the comparison the caution above demands.\n",
    )]
    c += [code(
        "from sklearn.manifold import TSNE",
        "from sjepa.viz import scatter_2d",
        "import matplotlib.pyplot as plt",
        "",
        "def tsne2d(E):",
        "    perp = min(15, max(2, len(E)//3))",
        "    return TSNE(n_components=2, perplexity=perp, random_state=42, init='pca').fit_transform(E)",
        "",
        "fig, ax = plt.subplots(1, 3, figsize=(15,4.4))",
        "scatter_2d(tsne2d(E_base), y, ax[0], 't-SNE: S-JEPA (label-free, nb 03)')",
        "scatter_2d(tsne2d(E_cont), y, ax[1], 't-SNE: S-JEPA (SSL continued, nb 04)')",
        "scatter_2d(tsne2d(N_nuis), y, ax[2], 't-SNE: nuisance (visibility only)')",
        "plt.tight_layout(); plt.savefig(IMAGES_DIR / 'tsne_sjepa_vs_nuisance.png', dpi=130)",
        "plt.show()",
    )]
    c += [code(
        "# UMAP view (falls back gracefully if umap-learn is missing).",
        "import matplotlib.pyplot as plt",
        "from sjepa.viz import scatter_2d",
        "try:",
        "    import umap",
        "    def umap2d(E):",
        "        nn = min(15, max(2, len(E)//3))",
        "        return umap.UMAP(n_neighbors=nn, min_dist=0.3, random_state=42).fit_transform(E)",
        "    fig, ax = plt.subplots(1, 3, figsize=(15,4.4))",
        "    scatter_2d(umap2d(E_base), y, ax[0], 'UMAP: S-JEPA (nb 03)')",
        "    scatter_2d(umap2d(E_cont), y, ax[1], 'UMAP: S-JEPA (nb 04)')",
        "    scatter_2d(umap2d(N_nuis), y, ax[2], 'UMAP: nuisance (visibility)')",
        "    plt.tight_layout(); plt.show()",
        "except Exception as e:",
        "    print('UMAP not available, skipping:', e)",
    )]
    c += [md(
        "## Put a (descriptive) number on the separation\n",
        "The silhouette score summarizes labeled point-cloud separation from -1 to +1. The retained "
        "value uses 47 correlated clips and is descriptive, not out-of-sample. A full-data value would "
        "use the usable subset of 91 raw clips after pose extraction.\n",
    )]
    c += [code(
        "from sjepa.eval import silhouette",
        "s_base = silhouette(E_base, y)",
        "s_cont = silhouette(E_cont, y)",
        "s_nuis = silhouette(N_nuis, y)",
        "print(f'silhouette (descriptive, whole set):')",
        "print(f'  S-JEPA label-free (nb 03): {s_base:.3f}')",
        "print(f'  S-JEPA SSL continued (nb 04): {s_cont:.3f}')",
        "print(f'  nuisance (visibility only): {s_nuis:.3f}')",
        "if s_nuis >= max(s_base, s_cont):",
        "    print('Note: the nuisance baseline separates at least as well -- a clean S-JEPA plot',",
        "          'here would NOT be evidence of learned gait. See notebook 06.')",
    )]
    return c


def nb_06(md, code, badge, boot):
    c = [badge("06_capstone_rf_vs_sjepa.ipynb")]
    c += [md(
        "# 06 - Capstone: Random Forest vs S-JEPA, on identical folds, with controls\n",
        "This notebook compares several systems on the **same clips**, the **same locked "
        "source-grouped folds**, and the **same pooled out-of-fold scoring**:\n",
        "1. a classical **Random Forest** on hand-made gait features (the exp5 recipe, three classes),\n"
        "2. the **label-free S-JEPA** with a frozen linear probe,\n"
        "3. cheap **shortcut controls** (visibility, body size, static pose) that any real "
        "representation must beat before we trust it.\n",
        "Pooled macro-F1 assigns one prediction to each held-out clip, pools all folds, and gives the "
        "three labels equal weight. Clips still receive equal weight when one source contributes many.\n",
        "> **Evidence status.** The frozen results belong to the earlier `g1` benchmark: 47 usable "
        "clips from 35 provisional source groups. The current raw collection has 91 clips from 41 "
        "source videos. It has no full-data model result until caches, folds, controls, and checkpoints "
        "are rebuilt.\n",
    )]
    c += boot(need_torch=True, legacy_artifacts=True)
    c += [code(
        "from IPython.display import SVG, display",
        "display(SVG(filename=str(IMAGES_DIR / 'rf_vs_sjepa.svg')))",
        "display(SVG(filename=str(IMAGES_DIR / 'grouped_split.svg')))",
        "display(SVG(filename=str(IMAGES_DIR / 'eval_firewall.svg')))",
    )]
    c += [md(
        "## The frozen result is produced by a script, not the notebook\n",
        "To preserve the earlier benchmark, its authoritative R1 run lives in "
        "`scripts/scripts_r1_repaired.py` and its output is committed under "
        "`artifacts/runs/r1_g1_1k_s42/`. That script and this notebook share the identical fold "
        "registry, so that comparison is paired by construction. A future full-data run should use new "
        "artifact names and must not overwrite `g1`.\n",
    )]
    c += [code(
        "import json",
        "frozen_path = ARTIFACT_DIR / 'runs' / 'r1_g1_1k_s42' / 'results.json'",
        "if frozen_path.exists():",
        "    frozen = json.loads(frozen_path.read_text())",
        "    sj = frozen['sjepa_pooled']; rf = frozen['rf_pooled']",
        "    print('Frozen R1 (1000 updates, seed 42, all 5 folds, pooled OOF):')",
        "    print(f\"  S-JEPA          : macro-F1 {sj['macro_f1']:.3f} | acc {sj['accuracy']:.3f}\"",
        "          f\" | PD-recall {sj['pd_recall']:.3f}\")",
        "    print(f\"  Random Forest   : macro-F1 {rf['macro_f1']:.3f} | acc {rf['accuracy']:.3f}\"",
        "          f\" | PD-recall {rf['pd_recall']:.3f}\")",
        "    print('  effective rank per fold:',",
        "          [round(d['eff_rank_final'], 1) for d in frozen['diagnostics']],",
        "          '(all >> 1 -> no collapse)')",
        "else:",
        "    print('Frozen run not found. Reproduce it with:')",
        "    print('  python scripts/scripts_r1_repaired.py --total-updates 1000 --seed 42 \\\\')",
        "    print('      --output-dir artifacts/runs/r1_g1_1k_s42')",
    )]
    c += [md(
        "## Shortcut controls: the bar S-JEPA has to clear\n",
        "Phase 0 scored nuisance features on the same legacy folds. If a control matches or beats "
        "S-JEPA, acquisition cues are a sufficient competing explanation for the predictive score. "
        "The expanded collection needs its own recomputed controls.\n",
    )]
    c += [code(
        "e0_path = ARTIFACT_DIR / 'eval' / 'g1' / 'E0_results.json'",
        "if e0_path.exists():",
        "    e0 = json.loads(e0_path.read_text())",
        "    # Use POOLED macro-F1 for the controls so they are the SAME metric as the",
        "    # pooled S-JEPA/RF above (pooling one prediction per clip across folds).",
        "    # Averaging per-fold macro-F1 is a different, non-comparable number.",
        "    print('Shortcut controls on g1 (best of logreg/rf, pooled OOF macro-F1):')",
        "    for name, res in e0['shortcut_controls'].items():",
        "        best = max(res['logreg']['pooled_macro_f1'], res['rf']['pooled_macro_f1'])",
        "        print(f'  {name:16s}: {best:.3f}')",
        "    print(f\"E0 Random Forest pooled macro-F1: {e0['E0_RF']['pooled_macro_f1']:.3f}\")",
        "else:",
        "    print('Run scripts/scripts_phase0_provenance.py to generate the control table.')",
    )]
    c += [md(
        "## Reproduce the mechanism live (one fold, small budget)\n",
        "For one locked legacy fold, we run the comparison mechanism: paired RF, label-free S-JEPA, "
        "a fixed mean-pooled read-out, and a class-balanced probe fitted only on training clips. The "
        "small update budget is a demonstration, not a result for all 91 current clips.\n",
    )]
    c += [code(
        "import numpy as np, torch, os",
        "from sjepa.config import get_config",
        "from sjepa.models import build_model, pick_device",
        "from sjepa.train_v2 import train_sjepa_v2",
        "from sjepa.masking_v2 import sample_target_mask",
        "from sjepa.data import load_index, SequenceWindowDataset, sliding_windows",
        "from sjepa.classical import build_feature_matrix, train_rf_and_predict",
        "from sjepa.eval import evaluate",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.preprocessing import StandardScaler",
        "",
        "cfg = get_config(); device = pick_device()",
        "LABELS = ['normal', 'ms', 'pd']",
        "records = load_index(KEYPOINTS_DIR)",
        "by_clip = {r.clip_name: r for r in records}",
        "registry = json.loads((ARTIFACT_DIR / 'eval' / 'g1' / 'fold_registry.json').read_text())",
        "readout = sample_target_mask(cfg.num_joints, cfg.num_time_tokens,",
        "                             np.random.default_rng(0), target_ratio=0.6)",
        "tm = torch.from_numpy(readout).to(device)",
        "",
        "def embed_records(model, recs):",
        "    V, Y = [], []",
        "    for r in recs:",
        "        w = sliding_windows(r.load_norm(), cfg.window_frames, cfg.window_stride)",
        "        x = torch.from_numpy(w).float().to(device)",
        "        with torch.no_grad():",
        "            V.append(model.embed(x, tm).mean(0).cpu().numpy())",
        "        Y.append(r.label)",
        "    return np.stack(V), Y",
    )]
    c += [code(
        "fold0 = registry['folds'][0]",
        "train_recs = [by_clip[c] for c in fold0['train_clips']]",
        "test_recs  = [by_clip[c] for c in fold0['test_clips']]",
        "",
        "# paired Random Forest (exp5 recipe) on this fold",
        "Xtr, ytr, _, _ = build_feature_matrix(train_recs, fps=cfg.target_fps)",
        "Xte, yte, _, _ = build_feature_matrix(test_recs, fps=cfg.target_fps)",
        "rf_pred = train_rf_and_predict(Xtr, ytr, Xte, seed=cfg.seed)",
        "rf_m = evaluate(yte, rf_pred, LABELS)",
        "",
        "# label-free S-JEPA on this fold's training sources",
        "SMOKE = cfg.profile.endswith('smoke')  # correct parse of SJEPA_SMOKE (not a raw truthiness test)",
        "UPDATES = 60 if SMOKE else 500",
        "model = build_model(cfg, device=device, repaired=True)",
        "ds = SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride)",
        "state = train_sjepa_v2(model, ds, cfg, total_updates=UPDATES, device=device, mask_ratio=0.6)",
        "Etr, ytr2 = embed_records(model, train_recs)",
        "Ete, yte2 = embed_records(model, test_recs)",
        "sc = StandardScaler().fit(Etr)                    # TRAIN only",
        "probe = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(Etr), ytr2)",
        "sj_m = evaluate(yte2, probe.predict(sc.transform(Ete)), LABELS)",
        "print(f'live fold-0 demo (budget={UPDATES} updates): RF f1={rf_m.macro_f1:.3f}',",
        "      f'| S-JEPA f1={sj_m.macro_f1:.3f} | eff_rank={state.eff_rank[-1]:.1f}')",
        "print('(The frozen 5-fold pooled numbers above are the ones to cite, not this one fold.)')",
    )]
    c += [md(
        "## Confusion of the frozen S-JEPA run\n",
        "These matrices contain one prediction for each of the 47 legacy clips. In `g1`, S-JEPA's "
        "largest off-diagonal error was PD labeled as MS. This should not be projected onto the "
        "expanded collection.\n",
    )]
    c += [code(
        "import numpy as np, matplotlib.pyplot as plt, seaborn as sns",
        "if frozen_path.exists():",
        "    fig, ax = plt.subplots(1, 2, figsize=(10,4))",
        "    for a, key, title in [(ax[0], 'rf_pooled', 'Random Forest (pooled OOF)'),",
        "                          (ax[1], 'sjepa_pooled', 'S-JEPA (pooled OOF)')]:",
        "        cm = np.array(frozen[key]['confusion'])",
        "        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,",
        "                    xticklabels=LABELS, yticklabels=LABELS, ax=a)",
        "        a.set_title(title); a.set_xlabel('predicted'); a.set_ylabel('true')",
        "    plt.tight_layout(); plt.show()",
        "else:",
        "    print('Frozen run not found; run scripts/scripts_r1_repaired.py first.')",
    )]
    c += [md(
        "## A combined scoreboard\n",
        "Every retained system uses the identical legacy `g1` folds and pooled macro-F1. This is not "
        "a scoreboard for `video-data-full`.\n",
    )]
    c += [code(
        "import pandas as pd",
        "rows = []",
        "if frozen_path.exists():",
        "    rows.append(('Random Forest (paired)', frozen['rf_pooled']['macro_f1']))",
        "    rows.append(('S-JEPA (R1, 1k updates)', frozen['sjepa_pooled']['macro_f1']))",
        "if e0_path.exists():",
        "    for name, res in e0['shortcut_controls'].items():",
        "        best = max(res['logreg']['pooled_macro_f1'], res['rf']['pooled_macro_f1'])",
        "        rows.append((f'control: {name}', best))",
        "rows.append(('chance (3 classes)', 1/3))",
        "board = pd.DataFrame(rows, columns=['system', 'macro_F1']).sort_values('macro_F1', ascending=False)",
        "display(board.reset_index(drop=True))",
        "results = {'frozen_run': str(frozen_path.relative_to(ARTIFACT_DIR)) if frozen_path.exists() else None,",
        "           'scoreboard': {n: float(v) for n, v in rows}}",
        "(ARTIFACT_DIR / 'capstone_results.json').write_text(json.dumps(results, indent=2))",
        "print('saved capstone_results.json')",
    )]
    c += [md(
        "## What this does and does not show\n",
        "**What the frozen benchmark supports.** Within `g1`, all systems used identical grouped folds "
        "and training-only fitted transforms. RF had the highest pooled macro-F1; S-JEPA scored below "
        "it and the nuisance controls. Effective rank above 1 rules out the simplest collapse account, "
        "but does not prove which shortcut caused each error.\n",
        "**What neither version establishes.** The legacy benchmark has 47 clips from 35 groups; the "
        "expanded inventory has 91 clips from 41 sources. Neither source count is a participant count, "
        "and neither dataset supports clinical or deployment claims.\n",
        "**The next analysis.** Extract the 91 clips into a clean cache; report exclusions; freeze a "
        "new `_P...`-aware source registry; rerun RF, S-JEPA, and controls; and report clip-weighted and "
        "source-aware summaries while preserving `g1` as historical evidence.\n",
    )]
    return c


def build(md, code, badge, boot, write_nb):
    write_nb("00_overview_and_video_gallery.ipynb", nb_00(md, code, badge, boot))
    write_nb("01_pose_extraction_from_raw_video.ipynb", nb_01(md, code, badge, boot))
    write_nb("02_anatomical_mask_and_tokenization.ipynb", nb_02(md, code, badge, boot))
    write_nb("03_sjepa_model_and_pretrain_normal.ipynb", nb_03(md, code, badge, boot))
    write_nb("04_progressive_finetune_ms_pd_vicreg.ipynb", nb_04(md, code, badge, boot))
    write_nb("05_representation_visualization.ipynb", nb_05(md, code, badge, boot))
    write_nb("06_capstone_rf_vs_sjepa.ipynb", nb_06(md, code, badge, boot))
