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
    from notebook_full_data import split_cells
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
    c += split_cells(md, code)
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
        "every window keeps visible context somewhere, and that over a bank of masks every joint "
        "is both visible and targeted often enough (the coverage gates). A joint can be masked in "
        "every block of one window: overlapping regions can cover its full duration. The maximum "
        "span applies to each sampled region, not their union. Hips belong to both trunk and leg "
        "regions, so their target frequency can be higher. Coverage is measured across fresh "
        "mask draws; individual frames and windows do not have a per-joint visibility guarantee. "
        "There is also a temporal bias: intervals placed entirely inside a window cover middle "
        "blocks more often than the ends. The exact-time table below exposes this; passing "
        "the window-coverage gates does not establish balanced masking or optimal training.\n",
    )]
    c += [code(
        "import numpy as np",
        "from importlib import reload",
        "import sjepa.masking_v2 as masking",
        "masking = reload(masking)  # pick up local edits in an existing notebook kernel",
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
        "print(f'over 512 masks: min visible-at-least-once/window {stats.joint_visible_frac.min():.2f} '",
        "      f'(gate >=0.20), min targeted-at-least-once/window {stats.joint_target_frac.min():.2f} (gate >=0.10)')",
        "print(f'mean target fraction {stats.mean_target_frac:.2f}')",
        "display(pd.DataFrame({",
        "    'joint': MEDIAPIPE_33_NAMES,",
        "    'visible token %': 100 * stats.joint_visible_token_frac,",
        "    'windows with any context %': 100 * stats.joint_visible_frac,",
        "    'windows fully masked for this joint %': 100 * stats.joint_always_target_frac,",
        "}).round(1))",
        "# 'Ever visible' can conceal low coverage in the middle of a window.",
        "print('Hip context frequency at each exact time block (%):')",
        "display(pd.DataFrame(100 * stats.token_visible_frac[:, [23, 24]],",
        "                     columns=['left hip', 'right hip'],",
        "                     index=np.arange(1, cfg.num_time_tokens + 1)).round(1))",
    )]
    c += [md(
        "Here is the difference drawn out: a fixed mask hides the same joints forever (left), while "
        "stochastic masks rotate which joints are hidden (right).\n",
    )]
    c += [code(
        "display(SVG(filename=str(IMAGES_DIR / 'defect_mask_starvation.svg')))",
    )]
    c += [md(
        "## See successive mask draws on the same skeleton\n",
        "Two clocks are shown separately: **mask sample** changes the independently drawn mask, "
        "while **time block** advances within a single model window. We replay the same motion "
        "for eight fresh masks from one seeded RNG, without filtering or recoloring any samples. "
        "Blue circles are visible context; red X markers are hidden prediction targets.\n",
        "With the laptop profile, the first seed-0 sample hides both complete legs for the whole window. In later "
        "samples the hips become visible. An entirely red time block is also valid if context "
        "exists elsewhere in that window. The GIF repeats these eight saved samples; rerunning "
        "with the same seed reproduces them. Change `DEMO_SEED` to explore another batch.\n",
        "The static timeline shows **every joint and time block at once**, even if your notebook "
        "viewer freezes the GIF. Its rows 23 and 24 show the exact hip mask bits. The helper also "
        "refreshes both older GIF filenames and writes a manifest so stale output can be detected. "
        "From a terminal, `python scripts/scripts_mask_demo.py` produces the same demo and "
        "`python scripts/scripts_mask_demo.py --check` detects overwritten artifacts.\n",
    )]
    c += [code(
        "from importlib import reload",
        "import sjepa.masking_v2 as masking",
        "import sjepa.viz as viz",
        "import sjepa.mask_demo as mask_demo",
        "masking = reload(masking); viz = reload(viz)",
        "mask_demo = reload(mask_demo)",
        "from sjepa.data import sliding_windows",
        "from IPython.display import Image, display",
        "",
        "# Inspect only a training clip; no held-out motion guides the mask demo.",
        "seq = sliding_windows(train_recs[0].load_norm(),",
        "                      cfg.window_frames, cfg.window_stride)[0]",
        "DEMO_SEED = 0",
        "N_MASK_SAMPLES = 8",
        "demo = mask_demo.write_mask_demo(seq, ARTIFACT_DIR, frame_group=cfg.frame_group,",
        "                                fps=cfg.target_fps, seed=DEMO_SEED, n_samples=N_MASK_SAMPLES)",
        "demo_masks = demo.masks",
        "print('Every joint appears in both roles across these samples:',",
        "      bool(demo_masks.any((0, 1)).all() and (~demo_masks).any((0, 1)).all()))",
        "print(f'Each sample: {cfg.window_frames} frames; each block: {cfg.frame_group} frames.')",
        "display(pd.DataFrame({",
        "    'mask sample': np.arange(1, N_MASK_SAMPLES + 1),",
        "    'left hip visible blocks': (~demo_masks[:, :, 23]).sum(1),",
        "    'right hip visible blocks': (~demo_masks[:, :, 24]).sum(1),",
        "    'out of blocks': cfg.num_time_tokens,",
        "}))",
        "mask_demo.verify_mask_demo(ARTIFACT_DIR)",
        "display(Image(filename=str(demo.timeline)))",
        "display(Image(filename=str(demo.animation)))",
    )]
    return c


from notebook_full_data import nb_03, nb_04, nb_05, nb_06


def build(md, code, badge, boot, write_nb):
    write_nb("00_overview_and_video_gallery.ipynb", nb_00(md, code, badge, boot))
    write_nb("01_pose_extraction_from_raw_video.ipynb", nb_01(md, code, badge, boot))
    write_nb("02_anatomical_mask_and_tokenization.ipynb", nb_02(md, code, badge, boot))
    write_nb("03_sjepa_model_and_pretrain_normal.ipynb", nb_03(md, code, badge, boot))
    write_nb("04_progressive_finetune_ms_pd_vicreg.ipynb", nb_04(md, code, badge, boot))
    write_nb("05_representation_visualization.ipynb", nb_05(md, code, badge, boot))
    write_nb("06_capstone_rf_vs_sjepa.ipynb", nb_06(md, code, badge, boot))
