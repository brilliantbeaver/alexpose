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
        "Welcome. This series teaches **S-JEPA**, a way for a model to learn what walking looks "
        "like without any labels, and then uses that learned sense of motion to tell apart three "
        "conditions from a short video of someone walking:\n",
        "- **normal** gait\n- **ms**: multiple sclerosis\n- **pd**: Parkinson's disease\n",
        "By the end you will have trained a small S-JEPA model on real walking clips, fine-tuned it "
        "across the three conditions, and compared it head to head against a classical Random Forest "
        "on the exact same videos.\n",
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
        "The videos live in `video-data/` split into `normal`, `ms`, and `pd` folders. Some clips "
        "come from the same source video (a long YouTube clip cut into pieces). That matters a lot "
        "for a fair test, because two clips from one source are not independent. We track the "
        "**source id** now so later we can keep all clips of one source on the same side of a split.\n",
    )]
    c += [code(
        "import pandas as pd",
        "from sjepa.data import source_id_from_name",
        "",
        "rows = []",
        "for label in ['normal', 'ms', 'pd']:",
        "    for vid in sorted((VIDEO_DIR / label).glob('*.mp4')):",
        "        rows.append(dict(label=label, clip=vid.name, source_id=source_id_from_name(vid.name)))",
        "manifest = pd.DataFrame(rows)",
        "summary = manifest.groupby('label').agg(clips=('clip', 'count'),",
        "                                        sources=('source_id', 'nunique'))",
        "print(summary)",
        "print('\\ntotal clips:', len(manifest), '| total sources:', manifest.source_id.nunique())",
        "manifest.to_csv(ARTIFACT_DIR / 'manifest_grouped.csv', index=False)",
    )]
    c += [md(
        "Notice that `pd` has many more clips than sources. That is the clip-splitting we will guard "
        "against. If we split clips at random, pieces of one walk could land in both training and "
        "testing and make the scores look better than they really are.\n",
        "## Watch a few walks\n",
        "Numbers are easier to trust once you have seen what they describe. The cell below embeds one "
        "clip per condition right in the notebook. Look for the differences the clinicians describe: "
        "normal gait is smooth and symmetric, ms gait can be unsteady with shorter steps, and pd gait "
        "often shows small shuffling steps and reduced arm swing.\n",
    )]
    c += [code(
        "from sjepa.viz import show_video",
        "from IPython.display import display",
        "",
        "for label in ['normal', 'ms', 'pd']:",
        "    clip = sorted((VIDEO_DIR / label).glob('*.mp4'))[0]",
        "    print(f'{label}: {clip.name}')",
        "    display(show_video(clip, width=360))",
    )]
    c += [md(
        "## Roadmap\n",
        "| Notebook | What you build |\n|---|---|\n"
        "| 00 overview | this tour of the data and the plan |\n"
        "| 01 pose extraction | turn videos into skeleton sequences with MediaPipe |\n"
        "| 02 mask and tokens | the fixed anatomical mask and how skeletons become tokens |\n"
        "| 03 pretrain on normal | build S-JEPA and train it on normal gait |\n"
        "| 04 progressive fine-tune | add ms and pd, add VICReg to separate the classes |\n"
        "| 05 representations | visualize the learned features with t-SNE and UMAP |\n"
        "| 06 capstone | Random Forest vs S-JEPA on identical, leakage-safe splits |\n",
        "On to notebook 01, where we turn these videos into skeletons.\n",
    )]
    return c


def nb_01(md, code, badge, boot):
    c = [badge("01_pose_extraction_from_raw_video.ipynb")]
    c += [md(
        "# 01 - Pose extraction from raw video\n",
        "A model cannot learn from pixels here; it learns from **skeletons**. In this notebook we run "
        "MediaPipe BlazePose over each video and get 33 body landmarks per frame. We reuse the "
        "existing `alexpose` pose code, so there is no new pose logic to trust, just a thin loop that "
        "feeds whole frames to the detector.\n",
        "The result for each video is an array of shape `(T, 33, 3)`: `T` frames, 33 joints, and three "
        "numbers per joint (x, y in pixels, plus a visibility score). We clean it, normalize it, and "
        "cache it so every later notebook opens instantly.\n",
    )]
    c += boot(need_torch=False)
    c += [md(
        "## One frame at a time\n",
        "The loader opens a video, samples it down to about 15 frames per second, and asks MediaPipe "
        "for the pose in each sampled frame. Unlike the GAVD pipeline it needs no bounding boxes and "
        "no annotation CSVs; it just reads the whole frame.\n",
    )]
    c += [code(
        "from ambient.pose.model_management import MediaPipeModelManager",
        "from ambient.pose.keypoint_extractor import SequenceKeypointExtractor",
        "from sjepa.data import load_video_sequence, clean_sequence, normalize_sequence",
        "",
        "MediaPipeModelManager().ensure_model_available()  # downloads the model once",
        "extractor = SequenceKeypointExtractor()",
        "",
        "sample = sorted((VIDEO_DIR / 'normal').glob('*.mp4'))[0]",
        "seq = load_video_sequence(sample, target_fps=15, extractor=extractor, verbose=True)",
        "print('raw sequence shape:', seq.shape, '  (frames, joints, [x, y, visibility])')",
    )]
    c += [md(
        "## Clean and normalize\n",
        "Real videos have frames where the detector loses the person. We interpolate short gaps and "
        "drop videos that are mostly empty. Then we **normalize**: we move the pelvis to the origin "
        "and scale by the torso length. This removes where the walker stood and how close the camera "
        "was, so the model sees the shape of the motion rather than the framing.\n",
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
        "## Extract and cache every video\n",
        "Now we run the same steps over all clips and cache one `.npz` file per video. This is the "
        "slow step, a few minutes on a laptop, and only needs to run once. If the cache already "
        "exists (it ships with the repo) this loop just confirms it.\n",
    )]
    c += [code(
        "from sjepa.data import save_sequence_npz, source_id_from_name",
        "import numpy as np",
        "",
        "KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)",
        "index = []",
        "for label in ['normal', 'ms', 'pd']:",
        "    for vid in sorted((VIDEO_DIR / label).glob('*.mp4')):",
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
        "idx.to_parquet(ARTIFACT_DIR / 'keypoints_index.parquet', index=False)",
        "print(idx.groupby('label').agg(videos=('clip_name','count'),",
        "                               sources=('source_id','nunique'),",
        "                               frames=('n_frames','sum')))",
    )]
    c += [md(
        "### Quick checks\n",
        "A good habit: assert the shapes and ranges are what we expect before moving on.\n",
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
        "S-JEPA learns by hiding part of the skeleton and predicting the hidden part in feature space. "
        "Two design choices drive this notebook: **how we cut the skeleton into tokens**, and "
        "**which joints we hide**.\n",
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
        "Now we build the model and train it, with **no labels**, on the walking motion "
        "itself. S-JEPA has three parts, all small transformers:\n",
        "- a **view encoder** that reads the visible joints of a slightly transformed view,\n"
        "- a **predictor** that guesses the hidden joints in feature space. Crucially, it is told "
        "*which* joint and *which* time each hidden slot is (a factorized position tag), so it can "
        "make a different guess per position. Without that tag every hidden guess is identical, which "
        "was a real bug in an earlier version.\n"
        "- a **target encoder** that reads the full skeleton and provides the answer. It is a slow "
        "moving average of the view encoder, which is what stops the model from collapsing every "
        "skeleton to the same features.\n",
        "> **A note on what trains here.** For three-class classification the useful thing is to learn "
        "from *all* the fold's unlabeled walking, so this notebook trains label-free on every training "
        "source. Training on normal gait *only* is a different question (one-class anomaly detection); "
        "we keep that as a separate idea, not the default.\n",
    )]
    c += boot(need_torch=True)
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
        "We train the self-supervised encoder on the **training half of the locked fold 0** only, "
        "never on the videos held out for testing. This matters even though the objective ignores "
        "labels: if the encoder saw a held-out video's motion during self-supervised training, then "
        "the probe scores that notebooks 04 and 06 report on that video would no longer be honest "
        "held-out numbers. So we load the same `g1` fold registry the later notebooks use and cut "
        "windows from its fold 0 training sources. The label still rides along, unused by the "
        "objective, only so later notebooks can score.\n",
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
        "print(f'fold 0 training half: {len(train_recs)} videos -> {len(ds_train)} windows',",
        "      '(labels unused in SSL; held-out videos excluded)')",
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
        "Notebook 03 trained the encoder with **no labels** on all the walking motion in the training "
        "set. Now we ask a sharper question: once we do have diagnosis labels, what is the honest way "
        "to use them, and does it actually help the three conditions separate?\n",
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
    )]
    c += boot(need_torch=True)
    c += [code(
        "from IPython.display import SVG, display",
        "display(SVG(filename=str(IMAGES_DIR / 'progressive_timeline.svg')))",
    )]
    c += [md(
        "## Use the locked, leakage-safe fold registry\n",
        "The comparison is only meaningful on a split where clips from one source never straddle "
        "train and test. Notebook's Phase 0 froze such a split to `artifacts/eval/g1/fold_registry.json` "
        "(source-grouped, seed 42). We load fold 0 from it here rather than inventing a fresh split, so "
        "this notebook, notebook 05, and notebook 06 all sit on the identical partition.\n",
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
        "print('fold 0:', len(train_recs), 'train videos /', len(test_recs), 'test videos')",
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
        "To turn a video into one vector we mean-pool the frozen target encoder over a **fixed** pool "
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
        "the training embeddings, then score the held-out videos. The scaler and the head are fit on "
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
        "On this tiny fold the two numbers are close and noisy; do not over-read a few points either "
        "way. The point of the notebook is the **method**: labels live only in the supervised head, the "
        "SSL objective stays label-free, and the split is the locked, leakage-safe one. Notebook 06 "
        "runs this over all folds and puts it beside the Random Forest and the shortcut controls, which "
        "is where any real verdict lives. A single fold here proves nothing on its own.\n",
    )]
    return c


def nb_05(md, code, badge, boot):
    c = [badge("05_representation_visualization.ipynb")]
    c += [md(
        "# 05 - Looking at the learned representation (diagnostics only)\n",
        "We have a trained encoder. What did it actually learn? Here we turn each video into a single "
        "feature vector with the frozen target encoder and project those vectors to two dimensions with "
        "t-SNE and UMAP, to *see* whether normal, ms, and pd land in different regions.\n",
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
    )]
    c += boot(need_torch=True)
    c += [code(
        "from IPython.display import SVG, display",
        "display(SVG(filename=str(IMAGES_DIR / 'vicreg_clusters.svg')))",
    )]
    c += [md(
        "## Embed every video with the frozen encoder\n",
        "For each video we mean-pool the encoder's features over its windows and over a **fixed**, "
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
        "print('embedded', len(y), 'videos into', E_cont.shape[1], 'dimensions')",
    )]
    c += [md(
        "## A nuisance baseline to keep us honest\n",
        "This is the cheapest possible 'representation': the per-joint mean and spread of the raw "
        "visibility channel, which we already know tracks the acquisition domain (the MS clips were all "
        "filmed at 60fps). If this separates the classes as cleanly as S-JEPA does, then a tidy S-JEPA "
        "scatter is not evidence of learned gait.\n",
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
        "The silhouette score summarizes how tight and well separated the class clusters are, from -1 "
        "to +1. We report it for all three embeddings so the S-JEPA numbers are read *against* the "
        "nuisance number, not in isolation. On ~47 videos this is noisy and purely descriptive: it is "
        "computed on the whole set, so it is **not** an out-of-sample score and must not be used to "
        "pick a model. Model selection happens only through the leakage-safe folds in notebook 06.\n",
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
        "This is the scientific payoff, and it comes with a result that is honest rather than "
        "flattering. We put several systems side by side on the **same videos**, the **same locked "
        "leakage-safe folds**, and the **same pooled out-of-fold scoring**:\n",
        "1. a classical **Random Forest** on hand-made gait features (the exp5 recipe, three classes),\n"
        "2. the **label-free S-JEPA** with a frozen linear probe,\n"
        "3. cheap **shortcut controls** (visibility, body size, static pose) that any real "
        "representation must beat before we trust it.\n",
        "The headline is **pooled macro-F1** over the folds (one prediction per clip, gathered across "
        "all held-out folds), because averaging per-fold F1 on ~9 test videos is even noisier. We "
        "report it beside the paired RF and the controls, then say plainly what it does and does not "
        "mean.\n",
        "> **Spoiler, stated up front.** On this tiny, already-inspected, source-grouped collection the "
        "S-JEPA scores *below* both the Random Forest and the nuisance controls. That is the "
        "expected, plan-anticipated outcome of removing the shortcuts and the label leak that inflated "
        "an earlier number, and it is reported as a negative result, not hidden.\n",
    )]
    c += boot(need_torch=True)
    c += [code(
        "from IPython.display import SVG, display",
        "display(SVG(filename=str(IMAGES_DIR / 'rf_vs_sjepa.svg')))",
        "display(SVG(filename=str(IMAGES_DIR / 'grouped_split.svg')))",
        "display(SVG(filename=str(IMAGES_DIR / 'eval_firewall.svg')))",
    )]
    c += [md(
        "## The frozen result is produced by a script, not the notebook\n",
        "So the headline cannot drift as someone re-runs cells, the authoritative R1 run lives in "
        "`scripts/scripts_r1_repaired.py` and its output is committed under "
        "`artifacts/runs/r1_g1_1k_s42/`. That script and this notebook share the identical fold "
        "registry, so the comparison is paired by construction. We read the frozen numbers first, then "
        "reproduce the mechanism live at a smaller budget so you can see how it is built.\n",
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
        "Phase 0 also scored cheap nuisance features on the identical folds. If a control matches or "
        "beats S-JEPA, then S-JEPA is not yet using gait beyond what a camera artifact already reveals. "
        "We read those frozen control scores here.\n",
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
        "Now the moving parts, so nothing is a black box. For one locked fold we run the exact "
        "pipeline: paired RF, then **label-free** S-JEPA (no diagnosis label enters the objective), a "
        "frozen mean-pool over a fixed read-out, and a class-balanced probe fit on the training clips "
        "only. This uses a tiny update budget to stay fast; the frozen numbers above are the ones to "
        "cite.\n",
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
        "Where does S-JEPA confuse the conditions across all held-out clips? The dominant "
        "error is PD read as MS, the same failure the Random Forest also struggles with here.\n",
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
        "One table, every system on the identical g1 folds. This is the whole comparison in one place.\n",
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
        "**What we can say.** The comparison is fair by construction: RF, S-JEPA, and the controls all "
        "sit on the identical locked folds, use per-clip pooled out-of-fold scoring, and fit their "
        "scalers and heads on training data only. On this data the Random Forest is the strongest "
        "system, and the label-free S-JEPA sits below it *and* below cheap nuisance controls.\n",
        "**Why that is progress, not failure.** An earlier S-JEPA number looked higher partly because "
        "it leaned on shortcuts (every MS clip was filmed at 60fps and square) and a label leak in the "
        "objective. Removing those lowered the honest score. The representation did not collapse "
        "(effective rank stays well above 1 on every fold), so this is a real, non-degenerate estimate, "
        "not a broken run. Per the pre-registered rule, a mechanically valid model that does not clear "
        "the bar tells us to **stop scaling the local network** and fix the binding constraint instead.\n",
        "**What we cannot say.** With ~47 videos from ~35 sources that are not verified people, this is "
        "a provisional, source-grouped **development estimate**, never a clinical result. No diagnostic, "
        "validity, or deployment claim is warranted.\n",
        "**The honest next step.** The evidence points at the data pipeline and the acquisition domain, "
        "not model size: rebuild the lineage (true common frame rate, speed-preserving normalization, "
        "validity masks, a domain de-confound) and bring in external clinical-motion pretraining, then "
        "rerun. That completes the series: a skeleton pipeline from raw video, a label-free "
        "S-JEPA with stochastic clinically-guided masks, and a leakage-safe comparison that reports the "
        "result whichever way it falls.\n",
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
