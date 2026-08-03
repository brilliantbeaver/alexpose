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
        "# 02 - The anatomical mask and tokenization\n",
        "S-JEPA learns by hiding part of the skeleton and predicting the hidden part in feature space. "
        "Two design choices drive this notebook: **how we cut the skeleton into tokens**, and "
        "**which joints we hide**.\n",
        "The original paper hides whichever joints move the most, a rule called motion-aware masking. "
        "For a gait study that is the wrong instinct, because in many conditions the telling sign is "
        "*less* motion: short steps, stiff knees, reduced arm swing. So we do something simpler and "
        "clinically grounded. We hide a **fixed set of neurologically relevant joints** and nothing "
        "else.\n",
    )]
    c += boot(need_torch=False)
    c += [md(
        "## Tokenizing a window\n",
        "A training window is a short movie of stick figures. We group `l = 4` adjacent frames of one "
        "joint into a single token, so each token summarizes how that joint moved over a moment. With "
        "32 frames and 33 joints that gives `(32 / 4) x 33 = 264` tokens.\n",
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
        "## The fixed anatomical mask\n",
        "The file `mapping-data/ms-pd-mapping.md` lists the joints clinicians care about for ms and "
        "pd. After removing duplicates and sorting, we get exactly twelve BlazePose landmarks: both "
        "shoulders and both complete legs. These are the joints we hide and ask the model to predict. "
        "No other joints are ever masked.\n",
    )]
    c += [code(
        "from sjepa.masking import MASKED_JOINTS, AnatomicalMaskSampler, masked_joint_names",
        "from ambient.pose.keypoint_data import MEDIAPIPE_33_NAMES",
        "import pandas as pd",
        "",
        "# The features from the mapping file that each masked joint supports.",
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
        "    for j in MASKED_JOINTS",
        "])",
        "table",
    )]
    c += [md(
        "That table is the whole masking rule. Twelve joints, chosen once, used every step. Here is "
        "the same set drawn on the skeleton.\n",
    )]
    c += [code(
        "display(SVG(filename=str(IMAGES_DIR / 'anatomical_mask.svg')))",
    )]
    c += [md(
        "## The mask on a real skeleton\n",
        "The animation below highlights the masked joints in red on one real walking sequence. Those "
        "red joints are what the model must reconstruct from the rest.\n",
    )]
    c += [code(
        "from sjepa.data import load_index",
        "from sjepa.viz import skeleton_animation",
        "from IPython.display import Image",
        "",
        "recs = load_index(KEYPOINTS_DIR)",
        "seq = recs[0].load_norm()",
        "gif = skeleton_animation(seq, ARTIFACT_DIR / 'mask_demo.gif',",
        "                         masked_joints=list(MASKED_JOINTS), fps=15,",
        "                         title='red = masked target joints')",
        "Image(filename=str(gif))",
    )]
    c += [md(
        "## How the mask becomes tokens\n",
        "The mask sampler turns those twelve joints into a boolean over the 264 tokens: every time "
        "block of a masked joint is a target token, everything else is visible context. It takes no "
        "motion input, so the split is identical for every clip.\n",
    )]
    c += [code(
        "sampler = AnatomicalMaskSampler(cfg.num_joints, cfg.num_time_tokens)",
        "print(sampler.summary())",
        "tm, cm = sampler.target_mask, sampler.context_mask",
        "assert not (tm & cm).any() and (tm | cm).all()  # no overlap, full cover",
        "print('target tokens:', int(tm.sum()), '| context tokens:', int(cm.sum()))",
        "print('This is what replaces motion-aware masking. Fixed, clinical, and simple.')",
    )]
    return c


def nb_03(md, code, badge, boot):
    c = [badge("03_sjepa_model_and_pretrain_normal.ipynb")]
    c += [md(
        "# 03 - Build S-JEPA and pretrain on normal gait\n",
        "Now we build the model and give it its first lesson: learn what ordinary walking looks like. "
        "We train only on **normal** clips here. This is phase one of progressive training; ms and pd "
        "come in notebook 04.\n",
        "S-JEPA has three parts, all small transformers:\n",
        "- a **view encoder** that reads the visible joints of a slightly rotated view,\n"
        "- a **predictor** that guesses the hidden joints in feature space,\n"
        "- a **target encoder** that reads the full skeleton and provides the answer. It is a slow "
        "moving average of the view encoder, which is what stops the model from cheating by "
        "collapsing every skeleton to the same features.\n",
    )]
    c += boot(need_torch=True)
    c += [md(
        "## The two-lane design\n",
        "The picture below is the whole idea. The top lane makes a prediction from a masked, rotated "
        "view. The bottom lane makes the target from the complete skeleton with a slow teacher. The "
        "only place they meet is the loss.\n",
    )]
    c += [code(
        "from IPython.display import SVG, display",
        "display(SVG(filename=str(IMAGES_DIR / 'sjepa_two_lane.svg')))",
    )]
    c += [code(
        "from sjepa.config import get_config, describe",
        "from sjepa.models import build_model, pick_device",
        "",
        "cfg = get_config()",
        "device = pick_device()",
        "print('device:', device)",
        "print(describe(cfg))",
        "model = build_model(cfg, device=device)",
        "n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)",
        "print(f'trainable parameters: {n_params/1e6:.2f}M')",
    )]
    c += [md(
        "## Load the normal-gait windows\n",
        "We cut each normal sequence into overlapping windows. Each window is one training example.\n",
    )]
    c += [code(
        "from sjepa.data import load_index, SequenceWindowDataset",
        "",
        "records = load_index(KEYPOINTS_DIR)",
        "normal_records = [r for r in records if r.label == 'normal']",
        "ds_normal = SequenceWindowDataset(normal_records, cfg.window_frames, cfg.window_stride)",
        "print(f'{len(normal_records)} normal videos -> {len(ds_normal)} training windows')",
    )]
    c += [md(
        "## Pretrain\n",
        "We run the two-lane objective: predict the hidden joints' features, match them to the slow "
        "teacher's features with a centered, sharpened cross-entropy, and nudge the teacher along with "
        "an exponential moving average. Watch the loss fall.\n",
    )]
    c += [code(
        "from sjepa.train import train_sjepa, save_checkpoint",
        "",
        "state = train_sjepa(model, ds_normal, cfg, epochs=cfg.pretrain_epochs,",
        "                    device=device, log_every=max(1, cfg.pretrain_epochs))",
        "save_checkpoint(ARTIFACT_DIR / 'sjepa_pretrain_normal.pt', model, cfg,",
        "                extra={'stage': 'pretrain_normal'})",
        "print('final loss (mean of last 5 steps):',",
        "      round(sum(state.losses[-5:]) / min(5, len(state.losses)), 4))",
    )]
    c += [code(
        "import matplotlib.pyplot as plt",
        "plt.figure(figsize=(7,3))",
        "plt.plot(state.losses, color='#dd6b20')",
        "plt.xlabel('training step'); plt.ylabel('loss'); plt.title('S-JEPA pretraining on normal gait')",
        "plt.tight_layout(); plt.show()",
    )]
    c += [md(
        "### Sanity checks\n",
        "Two things to confirm. First, the loss went down. Second, the teacher and the view encoder "
        "are not identical, which is the sign that the anti-collapse machinery is working.\n",
    )]
    c += [code(
        "import torch",
        "early = sum(state.losses[:5]) / 5",
        "late = sum(state.losses[-5:]) / 5",
        "print(f'loss {early:.3f} -> {late:.3f}')",
        "assert late <= early + 1e-3, 'loss did not decrease'",
        "diff = sum(torch.norm(t - v).item() for t, v in",
        "           zip(model.target_encoder.parameters(), model.view_encoder.parameters()))",
        "print('teacher vs student weight distance:', round(diff, 3))",
        "assert diff > 0, 'teacher collapsed onto student'",
        "print('Pretraining looks healthy. On to progressive fine-tuning.')",
    )]
    return c


def nb_04(md, code, badge, boot):
    c = [badge("04_progressive_finetune_ms_pd_vicreg.ipynb")]
    c += [md(
        "# 04 - Progressive fine-tuning with ms, pd, and VICReg\n",
        "The model now knows normal walking. In this notebook we grow its world: we keep training but "
        "add the ms and pd clips, and we switch on **VICReg**, an extra regularizer that pushes the "
        "three conditions toward separate regions of feature space.\n",
        "A note on honesty: VICReg is **not** part of the original S-JEPA. The paper prevents collapse "
        "with the slow teacher plus centering and sharpening. We add VICReg on top because our goal is "
        "classification, and we want the normal, ms, and pd clusters to pull apart. We label it "
        "clearly as an extension so nobody mistakes it for the original recipe.\n",
    )]
    c += boot(need_torch=True)
    c += [code(
        "from IPython.display import SVG, display",
        "display(SVG(filename=str(IMAGES_DIR / 'progressive_timeline.svg')))",
    )]
    c += [md(
        "## Pick the training videos, without leakage\n",
        "We must decide now which videos train the model and which are held out, and we must do it by "
        "**source id** so clips from one walk never straddle the split. We save that decision to "
        "`split_spec.json` so notebooks 05 and 06 reuse the exact same split. This is what makes the "
        "later comparison fair.\n",
    )]
    c += [code(
        "import json",
        "from sjepa.data import load_index, grouped_train_test_split",
        "",
        "records = load_index(KEYPOINTS_DIR)",
        "train_recs, test_recs = grouped_train_test_split(records, test_size=0.3, seed=42)",
        "",
        "split = {",
        "    'train_sources': sorted({r.source_id for r in train_recs}),",
        "    'test_sources': sorted({r.source_id for r in test_recs}),",
        "}",
        "assert not (set(split['train_sources']) & set(split['test_sources']))",
        "(ARTIFACT_DIR / 'split_spec.json').write_text(json.dumps(split, indent=2))",
        "print('train videos:', len(train_recs), '| test videos:', len(test_recs))",
        "print('no source appears in both:', not (set(split['train_sources']) & set(split['test_sources'])))",
    )]
    c += [md(
        "## Continue training on all three conditions\n",
        "We load the normal-pretrained weights, then keep training on the full training set with "
        "VICReg turned on. Because we have labels during fine-tuning, we use the class-aware VICReg "
        "variant, which keeps each condition compact while the variance floor keeps the condition "
        "centers from piling up.\n",
    )]
    c += [code(
        "from sjepa.config import get_config",
        "from sjepa.models import build_model, pick_device",
        "from sjepa.train import train_sjepa, load_checkpoint, save_checkpoint",
        "from sjepa.data import SequenceWindowDataset",
        "",
        "cfg = get_config()",
        "device = pick_device()",
        "model = build_model(cfg, device=device)",
        "load_checkpoint(ARTIFACT_DIR / 'sjepa_pretrain_normal.pt', model, map_location=device)",
        "",
        "ds_all = SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride)",
        "print(f'{len(train_recs)} training videos -> {len(ds_all)} windows across normal/ms/pd')",
        "state = train_sjepa(model, ds_all, cfg, epochs=cfg.finetune_epochs,",
        "                    use_vicreg=True, class_aware_vicreg=True,",
        "                    device=device, log_every=max(1, cfg.finetune_epochs))",
        "save_checkpoint(ARTIFACT_DIR / 'sjepa_finetuned_3class.pt', model, cfg,",
        "                extra={'stage': 'finetune_3class'})",
    )]
    c += [code(
        "import matplotlib.pyplot as plt",
        "fig, ax = plt.subplots(1, 2, figsize=(10,3))",
        "ax[0].plot(state.ce_losses, color='#2b6cb0'); ax[0].set_title('latent cross-entropy')",
        "ax[1].plot(state.vic_losses, color='#38a169'); ax[1].set_title('VICReg term')",
        "for a in ax: a.set_xlabel('step')",
        "plt.tight_layout(); plt.show()",
    )]
    c += [md(
        "## Does VICReg actually spread the features out?\n",
        "A quick check: pool the learned features per window and measure their spread. VICReg should "
        "keep the per-dimension spread comfortably above zero, meaning the representation did not "
        "collapse. We look at it below and visualize the clusters properly in notebook 05.\n",
    )]
    c += [code(
        "import torch, numpy as np",
        "from sjepa.masking import AnatomicalMaskSampler",
        "sampler = AnatomicalMaskSampler(cfg.num_joints, cfg.num_time_tokens)",
        "tm = torch.from_numpy(sampler.target_mask).to(device)",
        "xs = torch.stack([torch.from_numpy(ds_all.windows[i]) for i in range(len(ds_all))]).float().to(device)",
        "with torch.no_grad():",
        "    emb = model.embed(xs, tm).cpu().numpy()",
        "spread = float(emb.std(0).mean())",
        "print('per-dimension std (mean):', round(spread, 4))",
        "assert np.isfinite(emb).all(), 'embeddings went non-finite (training diverged)'",
        "if spread > 1e-3:",
        "    print('Features have healthy spread. Fine-tuned model saved.')",
        "else:",
        "    print('Spread is small. With the full profile (not SJEPA_SMOKE) it opens up;',",
        "          'in a 2-epoch smoke run this is expected.')",
    )]
    return c


def nb_05(md, code, badge, boot):
    c = [badge("05_representation_visualization.ipynb")]
    c += [md(
        "# 05 - Visualizing the learned representations\n",
        "We have a trained encoder. What did it actually learn? In this notebook we turn each video "
        "into a single feature vector using the frozen target encoder, then project those vectors to "
        "two dimensions with t-SNE and UMAP so we can see whether normal, ms, and pd land in different "
        "regions.\n",
        "We compare two moments: right after pretraining on normal only, and after progressive "
        "fine-tuning with VICReg. If the story holds, the clusters should be better separated after "
        "fine-tuning. On this small dataset the effect is a demonstration of mechanism, not proof, "
        "and we say so plainly.\n",
    )]
    c += boot(need_torch=True)
    c += [code(
        "from IPython.display import SVG, display",
        "display(SVG(filename=str(IMAGES_DIR / 'vicreg_clusters.svg')))",
    )]
    c += [md(
        "## Embed every video with the frozen encoder\n",
        "For each video we average the encoder's features over its windows and over the masked joints, "
        "giving one vector per video. We do this for both checkpoints.\n",
    )]
    c += [code(
        "import numpy as np, torch",
        "from sjepa.config import get_config",
        "from sjepa.models import build_model, pick_device",
        "from sjepa.train import load_checkpoint",
        "from sjepa.masking import AnatomicalMaskSampler",
        "from sjepa.data import load_index, sliding_windows",
        "",
        "cfg = get_config(); device = pick_device()",
        "records = load_index(KEYPOINTS_DIR)",
        "sampler = AnatomicalMaskSampler(cfg.num_joints, cfg.num_time_tokens)",
        "tm = torch.from_numpy(sampler.target_mask).to(device)",
        "",
        "def embed_all(ckpt):",
        "    m = build_model(cfg, device=device)",
        "    load_checkpoint(ckpt, m, map_location=device)",
        "    vecs, labels = [], []",
        "    for r in records:",
        "        w = sliding_windows(r.load_norm(), cfg.window_frames, cfg.window_stride)",
        "        x = torch.from_numpy(w).float().to(device)",
        "        with torch.no_grad():",
        "            vecs.append(m.embed(x, tm).mean(0).cpu().numpy())",
        "        labels.append(r.label)",
        "    return np.stack(vecs), labels",
        "",
        "E_pre, y = embed_all(ARTIFACT_DIR / 'sjepa_pretrain_normal.pt')",
        "E_ft, _ = embed_all(ARTIFACT_DIR / 'sjepa_finetuned_3class.pt')",
        "np.savez(ARTIFACT_DIR / 'embeddings_3class.npz', E_pretrain=E_pre, E_finetune=E_ft,",
        "         labels=np.array(y))",
        "print('embedded', len(y), 'videos into', E_ft.shape[1], 'dimensions')",
    )]
    c += [md(
        "## Project and plot\n",
        "t-SNE and UMAP both squeeze the high-dimensional vectors into a plane while trying to keep "
        "neighbors together. We color one hue per condition.\n",
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
        "fig, ax = plt.subplots(1, 2, figsize=(11,4.4))",
        "scatter_2d(tsne2d(E_pre), y, ax[0], 't-SNE: pretrain on normal only')",
        "scatter_2d(tsne2d(E_ft), y, ax[1], 't-SNE: after fine-tune + VICReg')",
        "plt.tight_layout(); plt.savefig(IMAGES_DIR / 'tsne_pretrain_vs_finetune.png', dpi=130)",
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
        "    fig, ax = plt.subplots(1, 2, figsize=(11,4.4))",
        "    scatter_2d(umap2d(E_pre), y, ax[0], 'UMAP: pretrain only')",
        "    scatter_2d(umap2d(E_ft), y, ax[1], 'UMAP: fine-tune + VICReg')",
        "    plt.tight_layout(); plt.show()",
        "except Exception as e:",
        "    print('UMAP not available, skipping:', e)",
    )]
    c += [md(
        "## Put a number on the separation\n",
        "The silhouette score summarizes how tight and well separated the clusters are, from -1 (bad) "
        "to +1 (clean). We compare the two checkpoints. On 47 videos this number is noisy, so treat it "
        "as a hint, not a verdict.\n",
    )]
    c += [code(
        "from sjepa.eval import silhouette",
        "s_pre = silhouette(E_pre, y)",
        "s_ft = silhouette(E_ft, y)",
        "print(f'silhouette  pretrain-only: {s_pre:.3f}   fine-tuned+VICReg: {s_ft:.3f}')",
        "if s_ft >= s_pre:",
        "    print('Fine-tuning helped separate the clusters (as hoped).')",
        "else:",
        "    print('No clear gain here. On this tiny dataset that can happen; see the caveats in 06.')",
    )]
    return c


def nb_06(md, code, badge, boot):
    c = [badge("06_capstone_rf_vs_sjepa.ipynb")]
    c += [md(
        "# 06 - Capstone: Random Forest vs S-JEPA\n",
        "This is the scientific payoff. We put two classifiers side by side on the **same videos** and "
        "the **same leakage-safe splits**:\n",
        "1. a classical **Random Forest** on hand-made gait features, exactly the exp5 recipe but for "
        "our three classes,\n"
        "2. an **S-JEPA linear probe** on top of the frozen encoder from notebook 04.\n",
        "We report grouped k-fold results with mean and standard deviation, because a single split of "
        "47 videos is too noisy to trust. Then we discuss honestly what this can and cannot show.\n",
    )]
    c += boot(need_torch=True)
    c += [code(
        "from IPython.display import SVG, display",
        "display(SVG(filename=str(IMAGES_DIR / 'rf_vs_sjepa.svg')))",
        "display(SVG(filename=str(IMAGES_DIR / 'grouped_split.svg')))",
    )]
    c += [md(
        "## The two branches as functions\n",
        "Both branches are thin wrappers around the verified `sjepa` package. The Random Forest branch "
        "reuses the `ambient` joint-angle and feature code (the exp5 pipeline). The S-JEPA branch "
        "trains a fresh model per fold and fits a logistic-regression probe on its frozen features.\n",
    )]
    c += [code(
        "import numpy as np, torch",
        "from sjepa.config import get_config",
        "from sjepa.models import build_model, pick_device",
        "from sjepa.train import train_sjepa",
        "from sjepa.masking import AnatomicalMaskSampler",
        "from sjepa.data import load_index, grouped_kfold, SequenceWindowDataset, sliding_windows",
        "from sjepa.classical import build_feature_matrix, train_rf_and_predict",
        "from sjepa.eval import evaluate, aggregate_folds, silhouette",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.preprocessing import StandardScaler",
        "",
        "cfg = get_config(); device = pick_device()",
        "LABELS = ['normal', 'ms', 'pd']",
        "records = load_index(KEYPOINTS_DIR)",
        "sampler = AnatomicalMaskSampler(cfg.num_joints, cfg.num_time_tokens)",
        "tm = torch.from_numpy(sampler.target_mask).to(device)",
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
    c += [md(
        "## Run grouped k-fold for both models\n",
        "For each fold we train from scratch (pretrain on the fold's normal videos, fine-tune on the "
        "fold's full training set with VICReg), then score both classifiers on the fold's held-out "
        "videos. Using a smaller epoch count keeps this runnable in the notebook; raise it in your "
        ".env profile for stronger results.\n",
    )]
    c += [code(
        "rf_metrics, sj_metrics, sils = [], [], []",
        "EPOCHS_PRE = max(6, cfg.pretrain_epochs // 4)",
        "EPOCHS_FT  = max(4, cfg.finetune_epochs // 4)",
        "",
        "for fold, (train_recs, test_recs) in enumerate(grouped_kfold(records, n_splits=5, seed=42)):",
        "    # --- Random Forest branch (exp5 recipe) ---",
        "    Xtr, ytr, _, _ = build_feature_matrix(train_recs, fps=cfg.target_fps)",
        "    Xte, yte, _, _ = build_feature_matrix(test_recs, fps=cfg.target_fps)",
        "    rf_pred = train_rf_and_predict(Xtr, ytr, Xte, seed=cfg.seed)",
        "    rf_metrics.append(evaluate(yte, rf_pred, LABELS))",
        "",
        "    # --- S-JEPA branch ---",
        "    model = build_model(cfg, device=device)",
        "    normal_tr = [r for r in train_recs if r.label == 'normal']",
        "    if normal_tr:",
        "        train_sjepa(model, SequenceWindowDataset(normal_tr, cfg.window_frames, cfg.window_stride),",
        "                    cfg, epochs=EPOCHS_PRE, device=device)",
        "    train_sjepa(model, SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride),",
        "                cfg, epochs=EPOCHS_FT, use_vicreg=True, class_aware_vicreg=True, device=device)",
        "    Etr, ytr2 = embed_records(model, train_recs)",
        "    Ete, yte2 = embed_records(model, test_recs)",
        "    sc = StandardScaler().fit(Etr)",
        "    probe = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(Etr), ytr2)",
        "    sj_pred = probe.predict(sc.transform(Ete))",
        "    sj_metrics.append(evaluate(yte2, sj_pred, LABELS))",
        "    sils.append(silhouette(np.vstack([Etr, Ete]), ytr2 + yte2))",
        "    print(f'fold {fold}: RF f1={rf_metrics[-1].macro_f1:.3f} | '",
        "          f'S-JEPA f1={sj_metrics[-1].macro_f1:.3f} | test n={len(test_recs)}')",
    )]
    c += [md(
        "## Headline: mean and standard deviation across folds\n",
    )]
    c += [code(
        "import json, pandas as pd",
        "rf_agg = aggregate_folds(rf_metrics)",
        "sj_agg = aggregate_folds(sj_metrics)",
        "",
        "def fmt(agg):",
        "    return {k: f\"{v['mean']:.3f} +/- {v['std']:.3f}\" for k, v in agg.items()}",
        "",
        "summary = pd.DataFrame({'Random Forest': fmt(rf_agg), 'S-JEPA probe': fmt(sj_agg)})",
        "display(summary)",
        "results = {'random_forest': rf_agg, 'sjepa': sj_agg,",
        "           'silhouette_mean': float(np.nanmean(sils)),",
        "           'n_folds': len(rf_metrics)}",
        "(ARTIFACT_DIR / 'capstone_results.json').write_text(json.dumps(results, indent=2))",
        "print('saved capstone_results.json')",
    )]
    c += [md(
        "## Confusion matrices side by side\n",
        "Averaged over folds, where does each model confuse the conditions?\n",
    )]
    c += [code(
        "import numpy as np, matplotlib.pyplot as plt, seaborn as sns",
        "def avg_cm(ms):",
        "    return np.mean([np.array(m.confusion) for m in ms], axis=0)",
        "fig, ax = plt.subplots(1, 2, figsize=(10,4))",
        "for a, ms, title in [(ax[0], rf_metrics, 'Random Forest'), (ax[1], sj_metrics, 'S-JEPA probe')]:",
        "    sns.heatmap(avg_cm(ms), annot=True, fmt='.1f', cmap='Blues', cbar=False,",
        "                xticklabels=LABELS, yticklabels=LABELS, ax=a)",
        "    a.set_title(title); a.set_xlabel('predicted'); a.set_ylabel('true')",
        "plt.tight_layout(); plt.show()",
    )]
    c += [md(
        "## Bonus: label efficiency\n",
        "S-JEPA's real promise is not beating a Random Forest when every video is labeled. It is "
        "learning useful features from unlabeled data, so it needs fewer labels to do well. We test "
        "that idea by training the probe on 25%, 50%, and 100% of the training labels and watching the "
        "score. The Random Forest, which has no pretraining to lean on, tends to fall off faster as "
        "labels shrink.\n",
    )]
    c += [code(
        "# Single grouped split for a quick, illustrative sweep.",
        "from sjepa.data import grouped_train_test_split",
        "train_recs, test_recs = grouped_train_test_split(records, test_size=0.3, seed=42)",
        "model = build_model(cfg, device=device)",
        "normal_tr = [r for r in train_recs if r.label == 'normal']",
        "if normal_tr:",
        "    train_sjepa(model, SequenceWindowDataset(normal_tr, cfg.window_frames, cfg.window_stride),",
        "                cfg, epochs=EPOCHS_PRE, device=device)",
        "train_sjepa(model, SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride),",
        "            cfg, epochs=EPOCHS_FT, use_vicreg=True, class_aware_vicreg=True, device=device)",
        "Etr, ytr = embed_records(model, train_recs); Ete, yte = embed_records(model, test_recs)",
        "Xtr, yrf, _, _ = build_feature_matrix(train_recs, fps=cfg.target_fps)",
        "Xte, yrf_te, _, _ = build_feature_matrix(test_recs, fps=cfg.target_fps)",
        "",
        "import numpy as np",
        "from sklearn.utils import resample",
        "rng = np.random.default_rng(0)",
        "fracs = [0.25, 0.5, 1.0]; sj_scores=[]; rf_scores=[]",
        "for f in fracs:",
        "    k = max(3, int(len(Etr)*f)); idx = rng.choice(len(Etr), k, replace=False)",
        "    sc = StandardScaler().fit(Etr[idx])",
        "    p = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(Etr[idx]), [ytr[i] for i in idx])",
        "    sj_scores.append(evaluate(yte, p.predict(sc.transform(Ete)), LABELS).macro_f1)",
        "    rf_pred = train_rf_and_predict(Xtr[idx], [yrf[i] for i in idx], Xte, seed=cfg.seed)",
        "    rf_scores.append(evaluate(yrf_te, rf_pred, LABELS).macro_f1)",
        "import matplotlib.pyplot as plt",
        "plt.figure(figsize=(6,3.5))",
        "plt.plot([int(f*100) for f in fracs], sj_scores, 'o-', color='#dd6b20', label='S-JEPA probe')",
        "plt.plot([int(f*100) for f in fracs], rf_scores, 's-', color='#38a169', label='Random Forest')",
        "plt.xlabel('percent of labels used'); plt.ylabel('macro F1'); plt.legend()",
        "plt.title('Label efficiency (single grouped split, illustrative)')",
        "plt.tight_layout(); plt.show()",
    )]
    c += [md(
        "## What this does and does not show\n",
        "**What we can say.** The comparison is fair: both models saw identical, leakage-safe splits "
        "and identical metrics. On this data the Random Forest is a strong baseline, which is exactly "
        "what we expect when every one of a few dozen videos is labeled and the hand-made features "
        "already encode clinical knowledge.\n",
        "**What we cannot say.** With 47 videos from about 35 independent sources, a few videos moving "
        "between folds swings the score by several points. These numbers are a methodology "
        "demonstration, not a clinical result. We do not claim S-JEPA beats the Random Forest here, "
        "and a single lucky split proves nothing.\n",
        "**Where S-JEPA earns its keep.** Its features come from unlabeled motion, so its edge shows up "
        "when labels are scarce or when the encoder is pretrained on far more walking than we have "
        "here. The label-efficiency sweep hints at that. The honest next step is more data and the "
        "`gpu` profile, not a bigger claim.\n",
        "That completes the series. You built a skeleton pipeline from raw video, an S-JEPA model with "
        "a clinically grounded mask, a VICReg-regularized fine-tune, and a fair comparison against a "
        "classical baseline.\n",
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
