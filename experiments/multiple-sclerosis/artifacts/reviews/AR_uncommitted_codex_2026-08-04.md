OpenAI Codex v0.146.0
--------
workdir: /Users/pmui/dev/alexpose/experiments/multiple-sclerosis
model: gpt-5.6-sol
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: medium
reasoning summaries: none
session id: 019fce63-4987-7481-be9e-9679c13eef25
--------
user
current changes
exec
/bin/zsh -lc "pwd && printf '\\n-- status --\\n' && git status --short && printf '\\n-- instruction files --\\n' && find .. -name AGENTS.md -o -name AGENTS.override.md" in /Users/pmui/dev/alexpose/experiments/multiple-sclerosis
 succeeded in 666ms:
exec
/bin/zsh -lc "git rev-parse --show-toplevel; printf '\\n-- diff stat --\\n'; git diff --stat; printf '\\n-- cached stat --\\n'; git diff --cached --stat; printf '\\n-- script diff --\\n'; git diff -- scripts/notebook_content.py; printf '\\n-- report diff headers --\\n'; git diff -- docs/06-0803-FINAL_REPORT.md | sed -n '1,240p'; printf '\\n-- untracked sizes --\\n'; find artifacts/reviews docs slides ../../notes/ms -maxdepth 2 -type f -print0 | xargs -0 ls -lh | tail -30" in /Users/pmui/dev/alexpose/experiments/multiple-sclerosis
 succeeded in 49ms:
diff --git a/experiments/multiple-sclerosis/scripts/notebook_content.py b/experiments/multiple-sclerosis/scripts/notebook_content.py
index 7a820d0..15045d1 100644
--- a/experiments/multiple-sclerosis/scripts/notebook_content.py
+++ b/experiments/multiple-sclerosis/scripts/notebook_content.py
@@ -208,22 +208,27 @@ def nb_01(md, code, badge, boot):
 def nb_02(md, code, badge, boot):
     c = [badge("02_anatomical_mask_and_tokenization.ipynb")]
     c += [md(
-        "# 02 - The anatomical mask and tokenization\n",
+        "# 02 - Masking and tokenization\n",
         "S-JEPA learns by hiding part of the skeleton and predicting the hidden part in feature space. "
         "Two design choices drive this notebook: **how we cut the skeleton into tokens**, and "
         "**which joints we hide**.\n",
-        "The original paper hides whichever joints move the most, a rule called motion-aware masking. "
-        "For a gait study that is the wrong instinct, because in many conditions the telling sign is "
-        "*less* motion: short steps, stiff knees, reduced arm swing. So we do something simpler and "
-        "clinically grounded. We hide a **fixed set of neurologically relevant joints** and nothing "
-        "else.\n",
+        "> **What changed, and why.** An earlier version of this project hid the *same* twelve clinical "
+        "joints on every single step. That turned out to be a real bug: the encoder never saw those "
+        "joints as context, so their internal position settings received no learning signal, yet the "
+        "classifier then pooled exactly those joints. We now use **stochastic graph-time masks**: a "
+        "different connected group of joints is hidden each step, so every joint is sometimes context "
+        "and sometimes a target. Clinical knowledge still guides us, but gently, by choosing the "
+        "leg and shoulder joints as targets a bit more often. We also do **not** bias toward the "
+        "busiest joints (the paper's motion-aware masking), because in MS and PD the telling sign is "
+        "often *reduced* motion, which a high-motion mask would hide.\n",
     )]
     c += boot(need_torch=False)
     c += [md(
         "## Tokenizing a window\n",
         "A training window is a short movie of stick figures. We group `l = 4` adjacent frames of one "
         "joint into a single token, so each token summarizes how that joint moved over a moment. With "
-        "32 frames and 33 joints that gives `(32 / 4) x 33 = 264` tokens.\n",
+        "32 frames and 33 joints that gives `(32 / 4) x 33 = 264` tokens. Token index is `t * V + v` "
+        "(time block `t`, joint `v`).\n",
     )]
     c += [code(
         "from IPython.display import SVG, display",
@@ -237,18 +242,17 @@ def nb_02(md, code, badge, boot):
         "      f'= {cfg.num_time_tokens} time blocks x {cfg.num_joints} joints')",
     )]
     c += [md(
-        "## The fixed anatomical mask\n",
+        "## The clinical joints (domain context, not a permanent mask)\n",
         "The file `mapping-data/ms-pd-mapping.md` lists the joints clinicians care about for ms and "
         "pd. After removing duplicates and sorting, we get exactly twelve BlazePose landmarks: both "
-        "shoulders and both complete legs. These are the joints we hide and ask the model to predict. "
-        "No other joints are ever masked.\n",
+        "shoulders and both complete legs. We keep this table as **domain knowledge** that biases how "
+        "often a joint is chosen as a target, but every joint can still be both context and target.\n",
     )]
     c += [code(
-        "from sjepa.masking import MASKED_JOINTS, AnatomicalMaskSampler, masked_joint_names",
+        "from sjepa.masking_v2 import CLINICAL_JOINTS",
         "from ambient.pose.keypoint_data import MEDIAPIPE_33_NAMES",
         "import pandas as pd",
         "",
-        "# The features from the mapping file that each masked joint supports.",
         "features_for = {",
         "    11: 'shoulder_symmetry_index, trunk_lean_angle',",
         "    12: 'shoulder_symmetry_index, trunk_lean_angle',",
@@ -264,75 +268,93 @@ def nb_02(md, code, badge, boot):
         "table = pd.DataFrame([",
         "    {'BLAZEPOSE_33 index': j, 'Keypoint name': MEDIAPIPE_33_NAMES[j],",
         "     'Features involved': features_for[j]}",
-        "    for j in MASKED_JOINTS",
+        "    for j in sorted(CLINICAL_JOINTS)",
         "])",
         "table",
     )]
     c += [md(
-        "That table is the whole masking rule. Twelve joints, chosen once, used every step. Here is "
-        "the same set drawn on the skeleton.\n",
+        "## Stochastic graph-time masks\n",
+        "Each step we sample a per-example mask: connected groups of joints (a limb or the trunk) over "
+        "a contiguous span of time. The cell below samples a few masks and shows they differ, that "
+        "every one keeps visible context, and that over a bank of masks every joint is both visible "
+        "and targeted often enough (the coverage gates).\n",
     )]
     c += [code(
-        "display(SVG(filename=str(IMAGES_DIR / 'anatomical_mask.svg')))",
+        "import numpy as np",
+        "from sjepa.masking_v2 import sample_mask_batch, mask_bank_stats",
+        "",
+        "rng = np.random.default_rng(0)",
+        "batch = sample_mask_batch(6, cfg.num_joints, cfg.num_time_tokens, rng)",
+        "print('mask batch shape (B, N):', batch.shape)",
+        "print('unique masks in the batch:', len({row.tobytes() for row in batch}), 'of 6')",
+        "print('every row has context and target:',",
+        "      bool((~batch).any(1).all() and batch.any(1).all()))",
+        "",
+        "stats = mask_bank_stats(cfg.num_joints, cfg.num_time_tokens, n_masks=512, seed=0)",
+        "print(f'over 512 masks: min joint-visible {stats.joint_visible_frac.min():.2f} '",
+        "      f'(gate >=0.20), min joint-target {stats.joint_target_frac.min():.2f} (gate >=0.10)')",
+        "print(f'mean target fraction {stats.mean_target_frac:.2f}')",
+    )]
+    c += [md(
+        "Here is the difference drawn out: a fixed mask hides the same joints forever (left), while "
+        "stochastic masks rotate which joints are hidden (right).\n",
+    )]
+    c += [code(
+        "display(SVG(filename=str(IMAGES_DIR / 'defect_mask_starvation.svg')))",
     )]
     c += [md(
-        "## The mask on a real skeleton\n",
-        "The animation below highlights the masked joints in red on one real walking sequence. Those "
-        "red joints are what the model must reconstruct from the rest.\n",
+        "## See one mask on a real skeleton\n",
+        "The animation highlights one sampled set of masked joints in red on a real walking sequence. "
+        "Next time you sample, a different group will be hidden.\n",
     )]
     c += [code(
         "from sjepa.data import load_index",
+        "from sjepa.masking_v2 import sample_target_mask",
         "from sjepa.viz import skeleton_animation",
         "from IPython.display import Image",
         "",
         "recs = load_index(KEYPOINTS_DIR)",
         "seq = recs[0].load_norm()",
+        "tgt = sample_target_mask(cfg.num_joints, cfg.num_time_tokens, np.random.default_rng(1))",
+        "masked_joints = sorted({int(i % cfg.num_joints) for i in np.nonzero(tgt)[0]})",
         "gif = skeleton_animation(seq, ARTIFACT_DIR / 'mask_demo.gif',",
-        "                         masked_joints=list(MASKED_JOINTS), fps=15,",
-        "                         title='red = masked target joints')",
+        "                         masked_joints=masked_joints, fps=15,",
+        "                         title='red = one sampled set of masked joints')",
         "Image(filename=str(gif))",
     )]
-    c += [md(
-        "## How the mask becomes tokens\n",
-        "The mask sampler turns those twelve joints into a boolean over the 264 tokens: every time "
-        "block of a masked joint is a target token, everything else is visible context. It takes no "
-        "motion input, so the split is identical for every clip.\n",
-    )]
-    c += [code(
-        "sampler = AnatomicalMaskSampler(cfg.num_joints, cfg.num_time_tokens)",
-        "print(sampler.summary())",
-        "tm, cm = sampler.target_mask, sampler.context_mask",
-        "assert not (tm & cm).any() and (tm | cm).all()  # no overlap, full cover",
-        "print('target tokens:', int(tm.sum()), '| context tokens:', int(cm.sum()))",
-        "print('This is what replaces motion-aware masking. Fixed, clinical, and simple.')",
-    )]
     return c
 
 
 def nb_03(md, code, badge, boot):
     c = [badge("03_sjepa_model_and_pretrain_normal.ipynb")]
     c += [md(
-        "# 03 - Build S-JEPA and pretrain on normal gait\n",
-        "Now we build the model and give it its first lesson: learn what ordinary walking looks like. "
-        "We train only on **normal** clips here. This is phase one of progressive training; ms and pd "
-        "come in notebook 04.\n",
-        "S-JEPA has three parts, all small transformers:\n",
-        "- a **view encoder** that reads the visible joints of a slightly rotated view,\n"
-        "- a **predictor** that guesses the hidden joints in feature space,\n"
+        "# 03 - Build S-JEPA and pretrain (label-free)\n",
+        "Now we build the model and train it, with **no labels**, on the walking motion "
+        "itself. S-JEPA has three parts, all small transformers:\n",
+        "- a **view encoder** that reads the visible joints of a slightly transformed view,\n"
+        "- a **predictor** that guesses the hidden joints in feature space. Crucially, it is told "
+        "*which* joint and *which* time each hidden slot is (a factorized position tag), so it can "
+        "make a different guess per position. Without that tag every hidden guess is identical, which "
+        "was a real bug in an earlier version.\n"
         "- a **target encoder** that reads the full skeleton and provides the answer. It is a slow "
-        "moving average of the view encoder, which is what stops the model from cheating by "
-        "collapsing every skeleton to the same features.\n",
+        "moving average of the view encoder, which is what stops the model from collapsing every "
+        "skeleton to the same features.\n",
+        "> **A note on what trains here.** For three-class classification the useful thing is to learn "
+        "from *all* the fold's unlabeled walking, so this notebook trains label-free on every training "
+        "source. Training on normal gait *only* is a different question (one-class anomaly detection); "
+        "we keep that as a separate idea, not the default.\n",
     )]
     c += boot(need_torch=True)
     c += [md(
         "## The two-lane design\n",
-        "The picture below is the whole idea. The top lane makes a prediction from a masked, rotated "
-        "view. The bottom lane makes the target from the complete skeleton with a slow teacher. The "
-        "only place they meet is the loss.\n",
+        "The picture below is the whole idea. The top lane makes a prediction from a masked view. The "
+        "bottom lane makes the target from the complete skeleton with a slow teacher. They meet only "
+        "at the loss.\n",
     )]
     c += [code(
         "from IPython.display import SVG, display",
         "display(SVG(filename=str(IMAGES_DIR / 'sjepa_two_lane.svg')))",
+        "display(SVG(filename=str(IMAGES_DIR / 'defect_predictor_positions.svg')))",
     )]
     c += [code(
         "from sjepa.config import get_config, describe",
@@ -342,61 +364,64 @@ def nb_03(md, code, badge, boot):
         "device = pick_device()",
         "print('device:', device)",
         "print(describe(cfg))",
-        "model = build_model(cfg, device=device)",
+        "model = build_model(cfg, device=device, repaired=True)  # PredictorV2 + per-example masks",
         "n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)",
         "print(f'trainable parameters: {n_params/1e6:.2f}M')",
     )]
     c += [md(
-        "## Load the normal-gait windows\n",
-        "We cut each normal sequence into overlapping windows. Each window is one training example.\n",
+        "## Load the training windows (all sources, no labels)\n",
+        "We cut every training sequence into overlapping windows. The label is ignored by the "
+        "self-supervised objective; it comes along only so later notebooks can score.\n",
     )]
     c += [code(
         "from sjepa.data import load_index, SequenceWindowDataset",
         "",
         "records = load_index(KEYPOINTS_DIR)",
-        "normal_records = [r for r in records if r.label == 'normal']",
-        "ds_normal = SequenceWindowDataset(normal_records, cfg.window_frames, cfg.window_stride)",
-        "print(f'{len(normal_records)} normal videos -> {len(ds_normal)} training windows')",
+        "ds_all = SequenceWindowDataset(records, cfg.window_frames, cfg.window_stride)",
+        "print(f'{len(records)} videos -> {len(ds_all)} training windows (labels unused in SSL)')",
     )]
     c += [md(
-        "## Pretrain\n",
-        "We run the two-lane objective: predict the hidden joints' features, match them to the slow "
-        "teacher's features with a centered, sharpened cross-entropy, and nudge the teacher along with "
-        "an exponential moving average. Watch the loss fall.\n",
+        "## Train\n",
+        "We run the two-lane objective for a fixed number of **optimizer updates** (not "
+        "epochs), sampling sources uniformly so a long clip cannot dominate, and nudge the teacher "
+        "with a responsive EMA. We log the loss and collapse diagnostics: the per-dimension spread, "
+        "the effective rank (how many directions the features really use), and how far the teacher has "
+        "drifted from the student.\n",
     )]
     c += [code(
-        "from sjepa.train import train_sjepa, save_checkpoint",
+        "from sjepa.train_v2 import train_sjepa_v2, save_checkpoint_v2",
         "",
-        "state = train_sjepa(model, ds_normal, cfg, epochs=cfg.pretrain_epochs,",
-        "                    device=device, log_every=max(1, cfg.pretrain_epochs))",
-        "save_checkpoint(ARTIFACT_DIR / 'sjepa_pretrain_normal.pt', model, cfg,",
-        "                extra={'stage': 'pretrain_normal'})",
-        "print('final loss (mean of last 5 steps):',",
-        "      round(sum(state.losses[-5:]) / min(5, len(state.losses)), 4))",
+        "# A small update budget keeps the notebook fast; raise it for stronger features.",
+        "UPDATES = 60 if __import__('os').environ.get('SJEPA_SMOKE') else 800",
+        "state = train_sjepa_v2(model, ds_all, cfg, total_updates=UPDATES, device=device,",
+        "                       mask_ratio=0.6, log_every=max(1, UPDATES // 4))",
+        "save_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl.pt', model, cfg, train_state=state,",
+        "                   extra={'stage': 'ssl_all_sources'})",
+        "print('EMA half-life (steps):', round(state.ema_half_life_steps, 1))",
     )]
     c += [code(
         "import matplotlib.pyplot as plt",
-        "plt.figure(figsize=(7,3))",
-        "plt.plot(state.losses, color='#dd6b20')",
-        "plt.xlabel('training step'); plt.ylabel('loss'); plt.title('S-JEPA pretraining on normal gait')",
+        "fig, ax = plt.subplots(1, 3, figsize=(12, 3))",
+        "ax[0].plot(state.losses, color='#dd6b20'); ax[0].set_title('latent cross-entropy')",
+        "ax[1].plot(state.eff_rank, color='#2563eb'); ax[1].set_title('effective rank (higher = richer)')",
+        "ax[2].plot(state.teacher_drift, color='#16a34a'); ax[2].set_title('teacher drift from student')",
+        "for a in ax: a.set_xlabel('update')",
         "plt.tight_layout(); plt.show()",
     )]
     c += [md(
         "### Sanity checks\n",
-        "Two things to confirm. First, the loss went down. Second, the teacher and the view encoder "
-        "are not identical, which is the sign that the anti-collapse machinery is working.\n",
+        "We avoid weak checks. A falling loss is necessary but not sufficient: a collapsed model can "
+        "also lower the loss. So we also require the effective rank to stay well above 1 (the features "
+        "use many directions, not one) and the teacher to have moved.\n",
     )]
     c += [code(
-        "import torch",
-        "early = sum(state.losses[:5]) / 5",
-        "late = sum(state.losses[-5:]) / 5",
-        "print(f'loss {early:.3f} -> {late:.3f}')",
-        "assert late <= early + 1e-3, 'loss did not decrease'",
-        "diff = sum(torch.norm(t - v).item() for t, v in",
-        "           zip(model.target_encoder.parameters(), model.view_encoder.parameters()))",
-        "print('teacher vs student weight distance:', round(diff, 3))",
-        "assert diff > 0, 'teacher collapsed onto student'",
-        "print('Pretraining looks healthy. On to progressive fine-tuning.')",
+        "import numpy as np",
+        "early = np.mean(state.losses[:5]); late = np.mean(state.losses[-5:])",
+        "print(f'loss {early:.3f} -> {late:.3f} | final effective rank {state.eff_rank[-1]:.1f}')",
+        "assert np.isfinite(state.losses).all(), 'loss went non-finite'",
+        "assert state.eff_rank[-1] > 1.5, 'representation looks collapsed (effective rank near 1)'",
+        "assert state.teacher_drift[-1] >= 0, 'teacher drift should be finite and non-negative'",
+        "print('SSL looks healthy (no collapse). On to comparing training regimes.')",
     )]
     return c
 
@@ -404,14 +429,21 @@ def nb_03(md, code, badge, boot):
 def nb_04(md, code, badge, boot):
     c = [badge("04_progressive_finetune_ms_pd_vicreg.ipynb")]
     c += [md(
-        "# 04 - Progressive fine-tuning with ms, pd, and VICReg\n",
-        "The model now knows normal walking. In this notebook we grow its world: we keep training but "
-        "add the ms and pd clips, and we switch on **VICReg**, an extra regularizer that pushes the "
-        "three conditions toward separate regions of feature space.\n",
-        "A note on honesty: VICReg is **not** part of the original S-JEPA. The paper prevents collapse "
-        "with the slow teacher plus centering and sharpening. We add VICReg on top because our goal is "
-        "classification, and we want the normal, ms, and pd clusters to pull apart. We label it "
-        "clearly as an extension so nobody mistakes it for the original recipe.\n",
+        "# 04 - Two ways to adapt the encoder: SSL continuation vs supervised adaptation\n",
+        "Notebook 03 trained the encoder with **no labels** on all the walking motion in the training "
+        "set. Now we ask a sharper question: once we do have diagnosis labels, what is the honest way "
+        "to use them, and does it actually help the three conditions separate?\n",
+        "We compare two clearly named regimes, both starting from the same label-free checkpoint:\n",
+        "1. **SSL continuation** - keep training with the *same* label-free objective for more updates. "
+        "No labels touch the model.\n"
+        "2. **Balanced supervised adaptation** - freeze the encoder and fit a small, class-balanced "
+        "linear head on top. Labels are used *only* in this head, never inside the self-supervised "
+        "objective.\n",
+        "> **What changed, and why.** An earlier version of this notebook mixed the diagnosis label "
+        "*into* the self-supervised loss (a 'class-aware VICReg' that rewarded within-class spread) and "
+        "claimed it 'compacts the classes'. That was a leak: the SSL objective must not see labels, and "
+        "the claim was not supported. We removed it. If labels help, they help in an explicitly "
+        "supervised stage that we name and measure, not smuggled into the pretext task.\n",
     )]
     c += boot(need_torch=True)
     c += [code(
@@ -419,84 +451,116 @@ def nb_04(md, code, badge, boot):
         "display(SVG(filename=str(IMAGES_DIR / 'progressive_timeline.svg')))",
     )]
     c += [md(
-        "## Pick the training videos, without leakage\n",
-        "We must decide now which videos train the model and which are held out, and we must do it by "
-        "**source id** so clips from one walk never straddle the split. We save that decision to "
-        "`split_spec.json` so notebooks 05 and 06 reuse the exact same split. This is what makes the "
-        "later comparison fair.\n",
+        "## Use the locked, leakage-safe fold registry\n",
+        "The comparison is only meaningful on a split where clips from one source never straddle "
+        "train and test. Notebook's Phase 0 froze such a split to `artifacts/eval/g1/fold_registry.json` "
+        "(source-grouped, seed 42). We load fold 0 from it here rather than inventing a fresh split, so "
+        "this notebook, notebook 05, and notebook 06 all sit on the identical partition.\n",
+        "> Source grouping is **provisional**: a `source_id` is a YouTube id, not a verified person, so "
+        "everything here is a development estimate, never a clinical claim.\n",
     )]
     c += [code(
         "import json",
-        "from sjepa.data import load_index, grouped_train_test_split",
+        "from sjepa.data import load_index",
         "",
         "records = load_index(KEYPOINTS_DIR)",
-        "train_recs, test_recs = grouped_train_test_split(records, test_size=0.3, seed=42)",
-        "",
-        "split = {",
-        "    'train_sources': sorted({r.source_id for r in train_recs}),",
-        "    'test_sources': sorted({r.source_id for r in test_recs}),",
-        "}",
-        "assert not (set(split['train_sources']) & set(split['test_sources']))",
-        "(ARTIFACT_DIR / 'split_spec.json').write_text(json.dumps(split, indent=2))",
-        "print('train videos:', len(train_recs), '| test videos:', len(test_recs))",
-        "print('no source appears in both:', not (set(split['train_sources']) & set(split['test_sources'])))",
+        "by_clip = {r.clip_name: r for r in records}",
+        "registry = json.loads((ARTIFACT_DIR / 'eval' / 'g1' / 'fold_registry.json').read_text())",
+        "fold0 = registry['folds'][0]",
+        "train_recs = [by_clip[c] for c in fold0['train_clips']]",
+        "test_recs  = [by_clip[c] for c in fold0['test_clips']]",
+        "tr_src = {r.source_id for r in train_recs}; te_src = {r.source_id for r in test_recs}",
+        "assert not (tr_src & te_src), 'source leakage across the fold'",
+        "print('fold 0:', len(train_recs), 'train videos /', len(test_recs), 'test videos')",
+        "print('no source in both sides:', not (tr_src & te_src))",
     )]
     c += [md(
-        "## Continue training on all three conditions\n",
-        "We load the normal-pretrained weights, then keep training on the full training set with "
-        "VICReg turned on. Because we have labels during fine-tuning, we use the class-aware VICReg "
-        "variant, which keeps each condition compact while the variance floor keeps the condition "
-        "centers from piling up.\n",
+        "## Regime 1 - SSL continuation (no labels)\n",
+        "We rebuild the model, load the label-free checkpoint from notebook 03, and keep "
+        "training with the exact same objective for a few hundred more updates. The label is never "
+        "read. This asks: does more unlabeled training alone sharpen the representation?\n",
     )]
     c += [code(
         "from sjepa.config import get_config",
         "from sjepa.models import build_model, pick_device",
-        "from sjepa.train import train_sjepa, load_checkpoint, save_checkpoint",
+        "from sjepa.train_v2 import train_sjepa_v2, load_checkpoint_v2, save_checkpoint_v2",
         "from sjepa.data import SequenceWindowDataset",
         "",
         "cfg = get_config()",
         "device = pick_device()",
-        "model = build_model(cfg, device=device)",
-        "load_checkpoint(ARTIFACT_DIR / 'sjepa_pretrain_normal.pt', model, map_location=device)",
+        "model = build_model(cfg, device=device, repaired=True)",
+        "load_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl.pt', model, map_location=device)",
         "",
-        "ds_all = SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride)",
-        "print(f'{len(train_recs)} training videos -> {len(ds_all)} windows across normal/ms/pd')",
-        "state = train_sjepa(model, ds_all, cfg, epochs=cfg.finetune_epochs,",
-        "                    use_vicreg=True, class_aware_vicreg=True,",
-        "                    device=device, log_every=max(1, cfg.finetune_epochs))",
-        "save_checkpoint(ARTIFACT_DIR / 'sjepa_finetuned_3class.pt', model, cfg,",
-        "                extra={'stage': 'finetune_3class'})",
+        "ds_train = SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride)",
+        "MORE = 40 if __import__('os').environ.get('SJEPA_SMOKE') else 400",
+        "state = train_sjepa_v2(model, ds_train, cfg, total_updates=MORE, device=device,",
+        "                       mask_ratio=0.6, log_every=max(1, MORE // 4))",
+        "save_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl_continued.pt', model, cfg, train_state=state,",
+        "                   extra={'stage': 'ssl_continuation_fold0'})",
+        "print('continued SSL: final effective rank', round(state.eff_rank[-1], 1))",
+    )]
+    c += [md(
+        "## A fixed, label-free read-out\n",
+        "To turn a video into one vector we mean-pool the frozen target encoder over a **fixed** pool "
+        "of target tokens. The pool is chosen once from a seeded RNG and never from the test labels, so "
+        "no information leaks from the evaluation into the representation. Both regimes below use this "
+        "same read-out.\n",
     )]
     c += [code(
-        "import matplotlib.pyplot as plt",
-        "fig, ax = plt.subplots(1, 2, figsize=(10,3))",
-        "ax[0].plot(state.ce_losses, color='#2b6cb0'); ax[0].set_title('latent cross-entropy')",
-        "ax[1].plot(state.vic_losses, color='#38a169'); ax[1].set_title('VICReg term')",
-        "for a in ax: a.set_xlabel('step')",
-        "plt.tight_layout(); plt.show()",
+        "import numpy as np, torch",
+        "from sjepa.masking_v2 import sample_target_mask",
+        "from sjepa.data import sliding_windows",
+        "",
+        "readout = sample_target_mask(cfg.num_joints, cfg.num_time_tokens,",
+        "                             np.random.default_rng(0), target_ratio=0.6)",
+        "tm = torch.from_numpy(readout).to(device)",
+        "",
+        "def embed_records(m, recs):",
+        "    V, Y = [], []",
+        "    for r in recs:",
+        "        w = sliding_windows(r.load_norm(), cfg.window_frames, cfg.window_stride)",
+        "        x = torch.from_numpy(w).float().to(device)",
+        "        with torch.no_grad():",
+        "            V.append(m.embed(x, tm).mean(0).cpu().numpy())",
+        "        Y.append(r.label)",
+        "    return np.stack(V), Y",
     )]
     c += [md(
-        "## Does VICReg actually spread the features out?\n",
-        "A quick check: pool the learned features per window and measure their spread. VICReg should "
-        "keep the per-dimension spread comfortably above zero, meaning the representation did not "
-        "collapse. We look at it below and visualize the clusters properly in notebook 05.\n",
-    )]
-    c += [code(
-        "import torch, numpy as np",
-        "from sjepa.masking import AnatomicalMaskSampler",
-        "sampler = AnatomicalMaskSampler(cfg.num_joints, cfg.num_time_tokens)",
-        "tm = torch.from_numpy(sampler.target_mask).to(device)",
-        "xs = torch.stack([torch.from_numpy(ds_all.windows[i]) for i in range(len(ds_all))]).float().to(device)",
-        "with torch.no_grad():",
-        "    emb = model.embed(xs, tm).cpu().numpy()",
-        "spread = float(emb.std(0).mean())",
-        "print('per-dimension std (mean):', round(spread, 4))",
-        "assert np.isfinite(emb).all(), 'embeddings went non-finite (training diverged)'",
-        "if spread > 1e-3:",
-        "    print('Features have healthy spread. Fine-tuned model saved.')",
-        "else:",
-        "    print('Spread is small. With the full profile (not SJEPA_SMOKE) it opens up;',",
-        "          'in a 2-epoch smoke run this is expected.')",
+        "## Regime 2 - balanced supervised adaptation (labels only in the head)\n",
+        "Now we use the labels honestly: freeze the encoder and fit a class-balanced logistic head on "
+        "the training embeddings, then score the held-out videos. The scaler and the head are fit on "
+        "**training data only**. We do this on top of both the notebook-03 checkpoint and the "
+        "SSL-continued one, so we can see whether extra unlabeled training moved the probe at all.\n",
+    )]
+    c += [code(
+        "from sklearn.linear_model import LogisticRegression",
+        "from sklearn.preprocessing import StandardScaler",
+        "from sjepa.eval import evaluate",
+        "LABELS = ['normal', 'ms', 'pd']",
+        "",
+        "def probe_and_score(ckpt):",
+        "    m = build_model(cfg, device=device, repaired=True)",
+        "    load_checkpoint_v2(ckpt, m, map_location=device)",
+        "    Etr, ytr = embed_records(m, train_recs)",
+        "    Ete, yte = embed_records(m, test_recs)",
+        "    sc = StandardScaler().fit(Etr)               # fit on TRAIN only",
+        "    clf = LogisticRegression(max_iter=2000, class_weight='balanced')",
+        "    clf.fit(sc.transform(Etr), ytr)",
+        "    return evaluate(yte, clf.predict(sc.transform(Ete)), LABELS)",
+        "",
+        "m_base = probe_and_score(ARTIFACT_DIR / 'sjepa_ssl.pt')",
+        "m_cont = probe_and_score(ARTIFACT_DIR / 'sjepa_ssl_continued.pt')",
+        "print(f'supervised probe on fold 0 held-out videos (macro-F1):')",
+        "print(f'  notebook-03 checkpoint : {m_base.macro_f1:.3f}')",
+        "print(f'  after SSL continuation : {m_cont.macro_f1:.3f}')",
+    )]
+    c += [md(
+        "## Read this honestly\n",
+        "On this tiny fold the two numbers are close and noisy; do not over-read a few points either "
+        "way. The point of the notebook is the **method**: labels live only in the supervised head, the "
+        "SSL objective stays label-free, and the split is the locked, leakage-safe one. Notebook 06 "
+        "runs this over all folds and puts it beside the Random Forest and the shortcut controls, which "
+        "is where any real verdict lives. A single fold here proves nothing on its own.\n",
     )]
     return c
 
@@ -504,15 +568,20 @@ def nb_04(md, code, badge, boot):
 def nb_05(md, code, badge, boot):
     c = [badge("05_representation_visualization.ipynb")]
     c += [md(
-        "# 05 - Visualizing the learned representations\n",
-        "We have a trained encoder. What did it actually learn? In this notebook we turn each video "
-        "into a single feature vector using the frozen target encoder, then project those vectors to "
-        "two dimensions with t-SNE and UMAP so we can see whether normal, ms, and pd land in different "
-        "regions.\n",
-        "We compare two moments: right after pretraining on normal only, and after progressive "
-        "fine-tuning with VICReg. If the story holds, the clusters should be better separated after "
-        "fine-tuning. On this small dataset the effect is a demonstration of mechanism, not proof, "
-        "and we say so plainly.\n",
+        "# 05 - Looking at the learned representation (diagnostics only)\n",
+        "We have a trained encoder. What did it actually learn? Here we turn each video into a single "
+        "feature vector with the frozen target encoder and project those vectors to two dimensions with "
+        "t-SNE and UMAP, to *see* whether normal, ms, and pd land in different regions.\n",
+        "> **These pictures are diagnostics, not evidence.** Two honest cautions run through this "
+        "notebook. First, t-SNE and UMAP distort distances; a clean-looking blob can be an artifact of "
+        "the projection. Second, and more important, apparent separation can come from a **shortcut** "
+        "(camera frame rate, body size, how visible the joints are) rather than from gait. So we plot "
+        "the S-JEPA embedding *and* a cheap nuisance feature side by side: if the nuisance separates "
+        "just as well, the pretty S-JEPA plot is not telling us about gait. The verdict lives in "
+        "notebook 06's leakage-safe scores, never in a scatter plot.\n",
+        "We compare the label-free checkpoint from notebook 03 against the SSL-continued one from "
+        "notebook 04. No test labels are ever used to fit, select, or color anything beyond the plain "
+        "class of each point.\n",
     )]
     c += boot(need_torch=True)
     c += [code(
@@ -521,25 +590,27 @@ def nb_05(md, code, badge, boot):
     )]
     c += [md(
         "## Embed every video with the frozen encoder\n",
-        "For each video we average the encoder's features over its windows and over the masked joints, "
-        "giving one vector per video. We do this for both checkpoints.\n",
+        "For each video we mean-pool the encoder's features over its windows and over a **fixed**, "
+        "seeded read-out pool of target tokens (the same pool used in notebooks 04 and 06). This pool "
+        "is never chosen from labels. We do it for both checkpoints.\n",
     )]
     c += [code(
         "import numpy as np, torch",
         "from sjepa.config import get_config",
         "from sjepa.models import build_model, pick_device",
-        "from sjepa.train import load_checkpoint",
-        "from sjepa.masking import AnatomicalMaskSampler",
+        "from sjepa.train_v2 import load_checkpoint_v2",
+        "from sjepa.masking_v2 import sample_target_mask",
         "from sjepa.data import load_index, sliding_windows",
         "",
         "cfg = get_config(); device = pick_device()",
         "records = load_index(KEYPOINTS_DIR)",
-        "sampler = AnatomicalMaskSampler(cfg.num_joints, cfg.num_time_tokens)",
-        "tm = torch.from_numpy(sampler.target_mask).to(device)",
+        "readout = sample_target_mask(cfg.num_joints, cfg.num_time_tokens,",
+        "                             np.random.default_rng(0), target_ratio=0.6)",
+        "tm = torch.from_numpy(readout).to(device)",
         "",
         "def embed_all(ckpt):",
-        "    m = build_model(cfg, device=device)",
-        "    load_checkpoint(ckpt, m, map_location=device)",
+        "    m = build_model(cfg, device=device, repaired=True)",
+        "    load_checkpoint_v2(ckpt, m, map_location=device)",
         "    vecs, labels = [], []",
         "    for r in records:",
         "        w = sliding_windows(r.load_norm(), cfg.window_frames, cfg.window_stride)",
@@ -549,16 +620,31 @@ def nb_05(md, code, badge, boot):
         "        labels.append(r.label)",
         "    return np.stack(vecs), labels",
         "",
-        "E_pre, y = embed_all(ARTIFACT_DIR / 'sjepa_pretrain_normal.pt')",
-        "E_ft, _ = embed_all(ARTIFACT_DIR / 'sjepa_finetuned_3class.pt')",
-        "np.savez(ARTIFACT_DIR / 'embeddings_3class.npz', E_pretrain=E_pre, E_finetune=E_ft,",
+        "E_base, y = embed_all(ARTIFACT_DIR / 'sjepa_ssl.pt')",
+        "E_cont, _ = embed_all(ARTIFACT_DIR / 'sjepa_ssl_continued.pt')",
+        "np.savez(ARTIFACT_DIR / 'embeddings_3class.npz', E_base=E_base, E_continued=E_cont,",
         "         labels=np.array(y))",
-        "print('embedded', len(y), 'videos into', E_ft.shape[1], 'dimensions')",
+        "print('embedded', len(y), 'videos into', E_cont.shape[1], 'dimensions')",
+    )]
+    c += [md(
+        "## A nuisance baseline to keep us honest\n",
+        "This is the cheapest possible 'representation': the per-joint mean and spread of the raw "
+        "visibility channel, which we already know tracks the acquisition domain (the MS clips were all "
+        "filmed at 60fps). If this separates the classes as cleanly as S-JEPA does, then a tidy S-JEPA "
+        "scatter is not evidence of learned gait.\n",
+    )]
+    c += [code(
+        "def nuisance_vec(r):",
+        "    vis = r.load_raw()[:, :, 2]",
+        "    return np.nan_to_num(np.concatenate([np.nanmean(vis, 0), np.nanstd(vis, 0)]))",
+        "N_nuis = np.stack([nuisance_vec(r) for r in records])",
+        "print('nuisance feature shape:', N_nuis.shape)",
     )]
     c += [md(
         "## Project and plot\n",
-        "t-SNE and UMAP both squeeze the high-dimensional vectors into a plane while trying to keep "
-        "neighbors together. We color one hue per condition.\n",
+        "t-SNE squeezes the high-dimensional vectors into a plane while trying to keep neighbors "
+        "together. We color one hue per condition, and we place the nuisance baseline in the same row "
+        "for the comparison the caution above demands.\n",
     )]
     c += [code(
         "from sklearn.manifold import TSNE",
@@ -569,10 +655,11 @@ def nb_05(md, code, badge, boot):
         "    perp = min(15, max(2, len(E)//3))",
         "    return TSNE(n_components=2, perplexity=perp, random_state=42, init='pca').fit_transform(E)",
         "",
-        "fig, ax = plt.subplots(1, 2, figsize=(11,4.4))",
-        "scatter_2d(tsne2d(E_pre), y, ax[0], 't-SNE: pretrain on normal only')",
-        "scatter_2d(tsne2d(E_ft), y, ax[1], 't-SNE: after fine-tune + VICReg')",
-        "plt.tight_layout(); plt.savefig(IMAGES_DIR / 'tsne_pretrain_vs_finetune.png', dpi=130)",
+        "fig, ax = plt.subplots(1, 3, figsize=(15,4.4))",
+        "scatter_2d(tsne2d(E_base), y, ax[0], 't-SNE: S-JEPA (label-free, nb 03)')",
+        "scatter_2d(tsne2d(E_cont), y, ax[1], 't-SNE: S-JEPA (SSL continued, nb 04)')",
+        "scatter_2d(tsne2d(N_nuis), y, ax[2], 't-SNE: nuisance (visibility only)')",
+        "plt.tight_layout(); plt.savefig(IMAGES_DIR / 'tsne_sjepa_vs_nuisance.png', dpi=130)",
         "plt.show()",
     )]
     c += [code(
@@ -584,28 +671,34 @@ def nb_05(md, code, badge, boot):
         "    def umap2d(E):",
         "        nn = min(15, max(2, len(E)//3))",
         "        return umap.UMAP(n_neighbors=nn, min_dist=0.3, random_state=42).fit_transform(E)",
-        "    fig, ax = plt.subplots(1, 2, figsize=(11,4.4))",
-        "    scatter_2d(umap2d(E_pre), y, ax[0], 'UMAP: pretrain only')",
-        "    scatter_2d(umap2d(E_ft), y, ax[1], 'UMAP: fine-tune + VICReg')",
+        "    fig, ax = plt.subplots(1, 3, figsize=(15,4.4))",
+        "    scatter_2d(umap2d(E_base), y, ax[0], 'UMAP: S-JEPA (nb 03)')",
+        "    scatter_2d(umap2d(E_cont), y, ax[1], 'UMAP: S-JEPA (nb 04)')",
+        "    scatter_2d(umap2d(N_nuis), y, ax[2], 'UMAP: nuisance (visibility)')",
         "    plt.tight_layout(); plt.show()",
         "except Exception as e:",
         "    print('UMAP not available, skipping:', e)",
     )]
     c += [md(
-        "## Put a number on the separation\n",
-        "The silhouette score summarizes how tight and well separated the clusters are, from -1 (bad) "
-        "to +1 (clean). We compare the two checkpoints. On 47 videos this number is noisy, so treat it "
-        "as a hint, not a verdict.\n",
+        "## Put a (descriptive) number on the separation\n",
+        "The silhouette score summarizes how tight and well separated the class clusters are, from -1 "
+        "to +1. We report it for all three embeddings so the S-JEPA numbers are read *against* the "
+        "nuisance number, not in isolation. On ~47 videos this is noisy and purely descriptive: it is "
+        "computed on the whole set, so it is **not** an out-of-sample score and must not be used to "
+        "pick a model. Model selection happens only through the leakage-safe folds in notebook 06.\n",
     )]
     c += [code(
         "from sjepa.eval import silhouette",
-        "s_pre = silhouette(E_pre, y)",
-        "s_ft = silhouette(E_ft, y)",
-        "print(f'silhouette  pretrain-only: {s_pre:.3f}   fine-tuned+VICReg: {s_ft:.3f}')",
-        "if s_ft >= s_pre:",
-        "    print('Fine-tuning helped separate the clusters (as hoped).')",
-        "else:",
-        "    print('No clear gain here. On this tiny dataset that can happen; see the caveats in 06.')",
+        "s_base = silhouette(E_base, y)",
+        "s_cont = silhouette(E_cont, y)",
+        "s_nuis = silhouette(N_nuis, y)",
+        "print(f'silhouette (descriptive, whole set):')",
+        "print(f'  S-JEPA label-free (nb 03): {s_base:.3f}')",
+        "print(f'  S-JEPA SSL continued (nb 04): {s_cont:.3f}')",
+        "print(f'  nuisance (visibility only): {s_nuis:.3f}')",
+        "if s_nuis >= max(s_base, s_cont):",
+        "    print('Note: the nuisance baseline separates at least as well -- a clean S-JEPA plot',",
+        "          'here would NOT be evidence of learned gait. See notebook 06.')",
     )]
     return c
 
@@ -613,44 +706,103 @@ def nb_05(md, code, badge, boot):
 def nb_06(md, code, badge, boot):
     c = [badge("06_capstone_rf_vs_sjepa.ipynb")]
     c += [md(
-        "# 06 - Capstone: Random Forest vs S-JEPA\n",
-        "This is the scientific payoff. We put two classifiers side by side on the **same videos** and "
-        "the **same leakage-safe splits**:\n",
-        "1. a classical **Random Forest** on hand-made gait features, exactly the exp5 recipe but for "
-        "our three classes,\n"
-        "2. an **S-JEPA linear probe** on top of the frozen encoder from notebook 04.\n",
-        "We report grouped k-fold results with mean and standard deviation, because a single split of "
-        "47 videos is too noisy to trust. Then we discuss honestly what this can and cannot show.\n",
+        "# 06 - Capstone: Random Forest vs S-JEPA, on identical folds, with controls\n",
+        "This is the scientific payoff, and it comes with a result that is honest rather than "
+        "flattering. We put several systems side by side on the **same videos**, the **same locked "
+        "leakage-safe folds**, and the **same pooled out-of-fold scoring**:\n",
+        "1. a classical **Random Forest** on hand-made gait features (the exp5 recipe, three classes),\n"
+        "2. the **label-free S-JEPA** with a frozen linear probe,\n"
+        "3. cheap **shortcut controls** (visibility, body size, static pose) that any real "
+        "representation must beat before we trust it.\n",
+        "The headline is **pooled macro-F1** over the folds (one prediction per clip, gathered across "
+        "all held-out folds), because averaging per-fold F1 on ~9 test videos is even noisier. We "
+        "report it beside the paired RF and the controls, then say plainly what it does and does not "
+        "mean.\n",
+        "> **Spoiler, stated up front.** On this tiny, already-inspected, source-grouped collection the "
+        "S-JEPA scores *below* both the Random Forest and the nuisance controls. That is the "
+        "expected, plan-anticipated outcome of removing the shortcuts and the label leak that inflated "
+        "an earlier number, and it is reported as a negative result, not hidden.\n",
     )]
     c += boot(need_torch=True)
     c += [code(
         "from IPython.display import SVG, display",
         "display(SVG(filename=str(IMAGES_DIR / 'rf_vs_sjepa.svg')))",
         "display(SVG(filename=str(IMAGES_DIR / 'grouped_split.svg')))",
+        "display(SVG(filename=str(IMAGES_DIR / 'eval_firewall.svg')))",
     )]
     c += [md(
-        "## The two branches as functions\n",
-        "Both branches are thin wrappers around the verified `sjepa` package. The Random Forest branch "
-        "reuses the `ambient` joint-angle and feature code (the exp5 pipeline). The S-JEPA branch "
-        "trains a fresh model per fold and fits a logistic-regression probe on its frozen features.\n",
+        "## The frozen result is produced by a script, not the notebook\n",
+        "So the headline cannot drift as someone re-runs cells, the authoritative R1 run lives in "
+        "`scripts/scripts_r1_repaired.py` and its output is committed under "
+        "`artifacts/runs/r1_g1_1k_s42/`. That script and this notebook share the identical fold "
+        "registry, so the comparison is paired by construction. We read the frozen numbers first, then "
+        "reproduce the mechanism live at a smaller budget so you can see how it is built.\n",
     )]
     c += [code(
-        "import numpy as np, torch",
+        "import json",
+        "frozen_path = ARTIFACT_DIR / 'runs' / 'r1_g1_1k_s42' / 'results.json'",
+        "if frozen_path.exists():",
+        "    frozen = json.loads(frozen_path.read_text())",
+        "    sj = frozen['sjepa_pooled']; rf = frozen['rf_pooled']",
+        "    print('Frozen R1 (1000 updates, seed 42, all 5 folds, pooled OOF):')",
+        "    print(f\"  S-JEPA          : macro-F1 {sj['macro_f1']:.3f} | acc {sj['accuracy']:.3f}\"",
+        "          f\" | PD-recall {sj['pd_recall']:.3f}\")",
+        "    print(f\"  Random Forest   : macro-F1 {rf['macro_f1']:.3f} | acc {rf['accuracy']:.3f}\"",
+        "          f\" | PD-recall {rf['pd_recall']:.3f}\")",
+        "    print('  effective rank per fold:',",
+        "          [round(d['eff_rank_final'], 1) for d in frozen['diagnostics']],",
+        "          '(all >> 1 -> no collapse)')",
+        "else:",
+        "    print('Frozen run not found. Reproduce it with:')",
+        "    print('  python scripts/scripts_r1_repaired.py --total-updates 1000 --seed 42 \\\\')",
+        "    print('      --output-dir artifacts/runs/r1_g1_1k_s42')",
+    )]
+    c += [md(
+        "## Shortcut controls: the bar S-JEPA has to clear\n",
+        "Phase 0 also scored cheap nuisance features on the identical folds. If a control matches or "
+        "beats S-JEPA, then S-JEPA is not yet using gait beyond what a camera artifact already reveals. "
+        "We read those frozen control scores here.\n",
+    )]
+    c += [code(
+        "e0_path = ARTIFACT_DIR / 'eval' / 'g1' / 'E0_results.json'",
+        "if e0_path.exists():",
+        "    e0 = json.loads(e0_path.read_text())",
+        "    print('Shortcut controls on g1 (best of logreg/rf, fold-mean macro-F1):')",
+        "    for name, res in e0['shortcut_controls'].items():",
+        "        best = max(res['logreg']['macro_f1']['mean'], res['rf']['macro_f1']['mean'])",
+        "        print(f'  {name:16s}: {best:.3f}')",
+        "    print(f\"E0 Random Forest pooled macro-F1: {e0['E0_RF']['pooled_macro_f1']:.3f}\")",
+        "else:",
+        "    print('Run scripts/scripts_phase0_provenance.py to generate the control table.')",
+    )]
+    c += [md(
+        "## Reproduce the mechanism live (one fold, small budget)\n",
+        "Now the moving parts, so nothing is a black box. For one locked fold we run the exact "
+        "pipeline: paired RF, then **label-free** S-JEPA (no diagnosis label enters the objective), a "
+        "frozen mean-pool over a fixed read-out, and a class-balanced probe fit on the training clips "
+        "only. This uses a tiny update budget to stay fast; the frozen numbers above are the ones to "
+        "cite.\n",
+    )]
+    c += [code(
+        "import numpy as np, torch, os",
         "from sjepa.config import get_config",
         "from sjepa.models import build_model, pick_device",
-        "from sjepa.train import train_sjepa",
-        "from sjepa.masking import AnatomicalMaskSampler",
-        "from sjepa.data import load_index, grouped_kfold, SequenceWindowDataset, sliding_windows",
+        "from sjepa.train_v2 import train_sjepa_v2",
+        "from sjepa.masking_v2 import sample_target_mask",
+        "from sjepa.data import load_index, SequenceWindowDataset, sliding_windows",
         "from sjepa.classical import build_feature_matrix, train_rf_and_predict",
-        "from sjepa.eval import evaluate, aggregate_folds, silhouette",
+        "from sjepa.eval import evaluate",
         "from sklearn.linear_model import LogisticRegression",
         "from sklearn.preprocessing import StandardScaler",
         "",
         "cfg = get_config(); device = pick_device()",
         "LABELS = ['normal', 'ms', 'pd']",
         "records = load_index(KEYPOINTS_DIR)",
-        "sampler = AnatomicalMaskSampler(cfg.num_joints, cfg.num_time_tokens)",
-        "tm = torch.from_numpy(sampler.target_mask).to(device)",
+        "by_clip = {r.clip_name: r for r in records}",
+        "registry = json.loads((ARTIFACT_DIR / 'eval' / 'g1' / 'fold_registry.json').read_text())",
+        "readout = sample_target_mask(cfg.num_joints, cfg.num_time_tokens,",
+        "                             np.random.default_rng(0), target_ratio=0.6)",
+        "tm = torch.from_numpy(readout).to(device)",
         "",
         "def embed_records(model, recs):",
         "    V, Y = [], []",
@@ -662,136 +814,93 @@ def nb_06(md, code, badge, boot):
         "        Y.append(r.label)",
         "    return np.stack(V), Y",
     )]
-    c += [md(
-        "## Run grouped k-fold for both models\n",
-        "For each fold we train from scratch (pretrain on the fold's normal videos, fine-tune on the "
-        "fold's full training set with VICReg), then score both classifiers on the fold's held-out "
-        "videos. Using a smaller epoch count keeps this runnable in the notebook; raise it in your "
-        ".env profile for stronger results.\n",
-    )]
     c += [code(
-        "rf_metrics, sj_metrics, sils = [], [], []",
-        "EPOCHS_PRE = max(6, cfg.pretrain_epochs // 4)",
-        "EPOCHS_FT  = max(4, cfg.finetune_epochs // 4)",
-        "",
-        "for fold, (train_recs, test_recs) in enumerate(grouped_kfold(records, n_splits=5, seed=42)):",
-        "    # --- Random Forest branch (exp5 recipe) ---",
-        "    Xtr, ytr, _, _ = build_feature_matrix(train_recs, fps=cfg.target_fps)",
-        "    Xte, yte, _, _ = build_feature_matrix(test_recs, fps=cfg.target_fps)",
-        "    rf_pred = train_rf_and_predict(Xtr, ytr, Xte, seed=cfg.seed)",
-        "    rf_metrics.append(evaluate(yte, rf_pred, LABELS))",
+        "fold0 = registry['folds'][0]",
+        "train_recs = [by_clip[c] for c in fold0['train_clips']]",
+        "test_recs  = [by_clip[c] for c in fold0['test_clips']]",
         "",
-        "    # --- S-JEPA branch ---",
-        "    model = build_model(cfg, device=device)",
-        "    normal_tr = [r for r in train_recs if r.label == 'normal']",
-        "    if normal_tr:",
-        "        train_sjepa(model, SequenceWindowDataset(normal_tr, cfg.window_frames, cfg.window_stride),",
-        "                    cfg, epochs=EPOCHS_PRE, device=device)",
-        "    train_sjepa(model, SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride),",
-        "                cfg, epochs=EPOCHS_FT, use_vicreg=True, class_aware_vicreg=True, device=device)",
-        "    Etr, ytr2 = embed_records(model, train_recs)",
-        "    Ete, yte2 = embed_records(model, test_recs)",
-        "    sc = StandardScaler().fit(Etr)",
-        "    probe = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(Etr), ytr2)",
-        "    sj_pred = probe.predict(sc.transform(Ete))",
-        "    sj_metrics.append(evaluate(yte2, sj_pred, LABELS))",
-        "    sils.append(silhouette(np.vstack([Etr, Ete]), ytr2 + yte2))",
-        "    print(f'fold {fold}: RF f1={rf_metrics[-1].macro_f1:.3f} | '",
-        "          f'S-JEPA f1={sj_metrics[-1].macro_f1:.3f} | test n={len(test_recs)}')",
-    )]
-    c += [md(
-        "## Headline: mean and standard deviation across folds\n",
-    )]
-    c += [code(
-        "import json, pandas as pd",
-        "rf_agg = aggregate_folds(rf_metrics)",
-        "sj_agg = aggregate_folds(sj_metrics)",
+        "# paired Random Forest (exp5 recipe) on this fold",
+        "Xtr, ytr, _, _ = build_feature_matrix(train_recs, fps=cfg.target_fps)",
+        "Xte, yte, _, _ = build_feature_matrix(test_recs, fps=cfg.target_fps)",
+        "rf_pred = train_rf_and_predict(Xtr, ytr, Xte, seed=cfg.seed)",
+        "rf_m = evaluate(yte, rf_pred, LABELS)",
         "",
-        "def fmt(agg):",
-        "    return {k: f\"{v['mean']:.3f} +/- {v['std']:.3f}\" for k, v in agg.items()}",
-        "",
-        "summary = pd.DataFrame({'Random Forest': fmt(rf_agg), 'S-JEPA probe': fmt(sj_agg)})",
-        "display(summary)",
-        "results = {'random_forest': rf_agg, 'sjepa': sj_agg,",
-        "           'silhouette_mean': float(np.nanmean(sils)),",
-        "           'n_folds': len(rf_metrics)}",
-        "(ARTIFACT_DIR / 'capstone_results.json').write_text(json.dumps(results, indent=2))",
-        "print('saved capstone_results.json')",
-    )]
-    c += [md(
-        "## Confusion matrices side by side\n",
-        "Averaged over folds, where does each model confuse the conditions?\n",
+        "# label-free S-JEPA on this fold's training sources",
+        "UPDATES = 60 if os.environ.get('SJEPA_SMOKE') else 500",
+        "model = build_model(cfg, device=device, repaired=True)",
+        "ds = SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride)",
+        "state = train_sjepa_v2(model, ds, cfg, total_updates=UPDATES, device=device, mask_ratio=0.6)",
+        "Etr, ytr2 = embed_records(model, train_recs)",
+        "Ete, yte2 = embed_records(model, test_recs)",
+        "sc = StandardScaler().fit(Etr)                    # TRAIN only",
+        "probe = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(Etr), ytr2)",
+        "sj_m = evaluate(yte2, probe.predict(sc.transform(Ete)), LABELS)",
+        "print(f'live fold-0 demo (budget={UPDATES} updates): RF f1={rf_m.macro_f1:.3f}',",
+        "      f'| S-JEPA f1={sj_m.macro_f1:.3f} | eff_rank={state.eff_rank[-1]:.1f}')",
+        "print('(The frozen 5-fold pooled numbers above are the ones to cite, not this one fold.)')",
+    )]
+    c += [md(
+        "## Confusion of the frozen S-JEPA run\n",
+        "Where does S-JEPA confuse the conditions across all held-out clips? The dominant "
+        "error is PD read as MS, the same failure the Random Forest also struggles with here.\n",
     )]
     c += [code(
         "import numpy as np, matplotlib.pyplot as plt, seaborn as sns",
-        "def avg_cm(ms):",
-        "    return np.mean([np.array(m.confusion) for m in ms], axis=0)",
-        "fig, ax = plt.subplots(1, 2, figsize=(10,4))",
-        "for a, ms, title in [(ax[0], rf_metrics, 'Random Forest'), (ax[1], sj_metrics, 'S-JEPA probe')]:",
-        "    sns.heatmap(avg_cm(ms), annot=True, fmt='.1f', cmap='Blues', cbar=False,",
-        "                xticklabels=LABELS, yticklabels=LABELS, ax=a)",
-        "    a.set_title(title); a.set_xlabel('predicted'); a.set_ylabel('true')",
-        "plt.tight_layout(); plt.show()",
+        "if frozen_path.exists():",
+        "    fig, ax = plt.subplots(1, 2, figsize=(10,4))",
+        "    for a, key, title in [(ax[0], 'rf_pooled', 'Random Forest (pooled OOF)'),",
+        "                          (ax[1], 'sjepa_pooled', 'S-JEPA (pooled OOF)')]:",
+        "        cm = np.array(frozen[key]['confusion'])",
+        "        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,",
+        "                    xticklabels=LABELS, yticklabels=LABELS, ax=a)",
+        "        a.set_title(title); a.set_xlabel('predicted'); a.set_ylabel('true')",
+        "    plt.tight_layout(); plt.show()",
+        "else:",
+        "    print('Frozen run not found; run scripts/scripts_r1_repaired.py first.')",
     )]
     c += [md(
-        "## Bonus: label efficiency\n",
-        "S-JEPA's real promise is not beating a Random Forest when every video is labeled. It is "
-        "learning useful features from unlabeled data, so it needs fewer labels to do well. We test "
-        "that idea by training the probe on 25%, 50%, and 100% of the training labels and watching the "
-        "score. The Random Forest, which has no pretraining to lean on, tends to fall off faster as "
-        "labels shrink.\n",
-    )]
-    c += [code(
-        "# Single grouped split for a quick, illustrative sweep.",
-        "from sjepa.data import grouped_train_test_split",
-        "train_recs, test_recs = grouped_train_test_split(records, test_size=0.3, seed=42)",
-        "model = build_model(cfg, device=device)",
-        "normal_tr = [r for r in train_recs if r.label == 'normal']",
-        "if normal_tr:",
-        "    train_sjepa(model, SequenceWindowDataset(normal_tr, cfg.window_frames, cfg.window_stride),",
-        "                cfg, epochs=EPOCHS_PRE, device=device)",
-        "train_sjepa(model, SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride),",
-        "            cfg, epochs=EPOCHS_FT, use_vicreg=True, class_aware_vicreg=True, device=device)",
-        "Etr, ytr = embed_records(model, train_recs); Ete, yte = embed_records(model, test_recs)",
-        "Xtr, yrf, _, _ = build_feature_matrix(train_recs, fps=cfg.target_fps)",
-        "Xte, yrf_te, _, _ = build_feature_matrix(test_recs, fps=cfg.target_fps)",
-        "",
-        "import numpy as np",
-        "from sklearn.utils import resample",
-        "rng = np.random.default_rng(0)",
-        "fracs = [0.25, 0.5, 1.0]; sj_scores=[]; rf_scores=[]",
-        "for f in fracs:",
-        "    k = max(3, int(len(Etr)*f)); idx = rng.choice(len(Etr), k, replace=False)",
-        "    sc = StandardScaler().fit(Etr[idx])",
-        "    p = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(Etr[idx]), [ytr[i] for i in idx])",
-        "    sj_scores.append(evaluate(yte, p.predict(sc.transform(Ete)), LABELS).macro_f1)",
-        "    rf_pred = train_rf_and_predict(Xtr[idx], [yrf[i] for i in idx], Xte, seed=cfg.seed)",
-        "    rf_scores.append(evaluate(yrf_te, rf_pred, LABELS).macro_f1)",
-        "import matplotlib.pyplot as plt",
-        "plt.figure(figsize=(6,3.5))",
-        "plt.plot([int(f*100) for f in fracs], sj_scores, 'o-', color='#dd6b20', label='S-JEPA probe')",
-        "plt.plot([int(f*100) for f in fracs], rf_scores, 's-', color='#38a169', label='Random Forest')",
-        "plt.xlabel('percent of labels used'); plt.ylabel('macro F1'); plt.legend()",
-        "plt.title('Label efficiency (single grouped split, illustrative)')",
-        "plt.tight_layout(); plt.show()",
+        "## A combined scoreboard\n",
+        "One table, every system on the identical g1 folds. This is the whole comparison in one place.\n",
+    )]
+    c += [code(
+        "import pandas as pd",
+        "rows = []",
+        "if frozen_path.exists():",
+        "    rows.append(('Random Forest (paired)', frozen['rf_pooled']['macro_f1']))",
+        "    rows.append(('S-JEPA (R1, 1k updates)', frozen['sjepa_pooled']['macro_f1']))",
+        "if e0_path.exists():",
+        "    for name, res in e0['shortcut_controls'].items():",
+        "        best = max(res['logreg']['macro_f1']['mean'], res['rf']['macro_f1']['mean'])",
+        "        rows.append((f'control: {name}', best))",
+        "rows.append(('chance (3 classes)', 1/3))",
+        "board = pd.DataFrame(rows, columns=['system', 'macro_F1']).sort_values('macro_F1', ascending=False)",
+        "display(board.reset_index(drop=True))",
+        "results = {'frozen_run': str(frozen_path.relative_to(ARTIFACT_DIR)) if frozen_path.exists() else None,",
+        "           'scoreboard': {n: float(v) for n, v in rows}}",
+        "(ARTIFACT_DIR / 'capstone_results.json').write_text(json.dumps(results, indent=2))",
+        "print('saved capstone_results.json')",
     )]
     c += [md(
         "## What this does and does not show\n",
-        "**What we can say.** The comparison is fair: both models saw identical, leakage-safe splits "
-        "and identical metrics. On this data the Random Forest is a strong baseline, which is exactly "
-        "what we expect when every one of a few dozen videos is labeled and the hand-made features "
-        "already encode clinical knowledge.\n",
-        "**What we cannot say.** With 47 videos from about 35 independent sources, a few videos moving "
-        "between folds swings the score by several points. These numbers are a methodology "
-        "demonstration, not a clinical result. We do not claim S-JEPA beats the Random Forest here, "
-        "and a single lucky split proves nothing.\n",
-        "**Where S-JEPA earns its keep.** Its features come from unlabeled motion, so its edge shows up "
-        "when labels are scarce or when the encoder is pretrained on far more walking than we have "
-        "here. The label-efficiency sweep hints at that. The honest next step is more data and the "
-        "`gpu` profile, not a bigger claim.\n",
-        "That completes the series. You built a skeleton pipeline from raw video, an S-JEPA model with "
-        "a clinically grounded mask, a VICReg-regularized fine-tune, and a fair comparison against a "
-        "classical baseline.\n",
+        "**What we can say.** The comparison is fair by construction: RF, S-JEPA, and the controls all "
+        "sit on the identical locked folds, use per-clip pooled out-of-fold scoring, and fit their "
+        "scalers and heads on training data only. On this data the Random Forest is the strongest "
+        "system, and the label-free S-JEPA sits below it *and* below cheap nuisance controls.\n",
+        "**Why that is progress, not failure.** An earlier S-JEPA number looked higher partly because "
+        "it leaned on shortcuts (every MS clip was filmed at 60fps and square) and a label leak in the "
+        "objective. Removing those lowered the honest score. The representation did not collapse "
+        "(effective rank stays well above 1 on every fold), so this is a real, non-degenerate estimate, "
+        "not a broken run. Per the pre-registered rule, a mechanically valid model that does not clear "
+        "the bar tells us to **stop scaling the local network** and fix the binding constraint instead.\n",
+        "**What we cannot say.** With ~47 videos from ~35 sources that are not verified people, this is "
+        "a provisional, source-grouped **development estimate**, never a clinical result. No diagnostic, "
+        "validity, or deployment claim is warranted.\n",
+        "**The honest next step.** The evidence points at the data pipeline and the acquisition domain, "
+        "not model size: rebuild the lineage (true common frame rate, speed-preserving normalization, "
+        "validity masks, a domain de-confound) and bring in external clinical-motion pretraining, then "
+        "rerun. That completes the series: a skeleton pipeline from raw video, a label-free "
+        "S-JEPA with stochastic clinically-guided masks, and a leakage-safe comparison that reports the "
+        "result whichever way it falls.\n",
     )]
     return c
 

-- report diff headers --
diff --git a/experiments/multiple-sclerosis/docs/06-0803-FINAL_REPORT.md b/experiments/multiple-sclerosis/docs/06-0803-FINAL_REPORT.md
index 2148f5d..6a43536 100644
--- a/experiments/multiple-sclerosis/docs/06-0803-FINAL_REPORT.md
+++ b/experiments/multiple-sclerosis/docs/06-0803-FINAL_REPORT.md
@@ -15,7 +15,7 @@ was run on the real cached skeletons on the locked source-grouped fold registry,
 Random Forest and shortcut controls, and pooled out-of-fold predictions saved.
 
 **The headline result is a negative one, and that is the point.** On this tiny, already-inspected,
-source-grouped collection, the *repaired* S-JEPA scores **lower** than both the old (partly
+source-grouped collection, the corrected S-JEPA scores **lower** than both the old (partly
 shortcut-driven) S-JEPA and the Random Forest:
 
 ![R1 result](../images/r1_results.svg)
@@ -26,14 +26,17 @@ shortcut-driven) S-JEPA and the Random Forest:
 | nuisance control: pose mean+std | 0.694 | – | – |
 | nuisance control: visibility only | 0.602 | – | – |
 | old broken S-JEPA (historical) | 0.570 | 0.573 | 0.353 |
-| **repaired S-JEPA (R1, 1000 updates, seed 42)** | **0.438** | 0.447 | 0.235 |
+| **S-JEPA (R1, 1000 updates, seed 42)** | **0.438** | 0.447 | 0.235 |
 | chance (3 classes) | 0.333 | – | – |
 
 This is scientific progress: the previous 0.570 leaned partly on shortcuts (all MS clips are 60fps
 and square; the label leak and fixed mask further inflated it). With a mechanically correct,
-label-free, source-uniform pipeline, the honest number is lower. The representation did **not**
-collapse (effective rank 7.7–9.5 across folds), so this is a real, non-degenerate estimate, not a
-broken run.
+label-free, source-uniform pipeline, the honest number is lower. (**Label-free** means the
+self-supervised objective never reads the `normal`/`ms`/`pd` diagnosis label — it learns to predict
+masked joint motion from visible motion; the label enters only later, in the supervised probe. The
+old code violated this via a class-aware term that fed the label into the loss; removing it is
+defect D3.) The representation did **not** collapse (effective rank 7.7–9.5 across folds), so this
+is a real, non-degenerate estimate, not a broken run.
 
 Per the pre-registered promotion rule, a mechanically valid R1 that does not clear the inner gate
 means: **stop scaling the local model; the next bottleneck is the data pipeline and external
@@ -120,12 +123,12 @@ All 15 tests pass on both the pinned experiment `.venv` (Python 3.12) and the re
 
 ## 7. Results table (separate systems, same registry g1)
 
-See the table in §1. RF, old S-JEPA, repaired S-JEPA, and nuisance controls are separate rows.
+See the table in §1. RF, old (broken) S-JEPA, the corrected S-JEPA, and nuisance controls are separate rows.
 No fusion, supervised-adaptation, or RGB branch was run (out of scope this session).
 
 ## 8. Shortcut / temporal controls and limitations
 
-- Nuisance controls (pose mean+std 0.694; visibility 0.602) **exceed** the repaired S-JEPA (0.438),
+- Nuisance controls (pose mean+std 0.694; visibility 0.602) **exceed** the S-JEPA score (0.438),
   which means the learned representation does not yet beat cheap nuisance signals on this cache. No
   clinical-representation claim is warranted.
 - Limitations: single seed, 1000 updates (no learning-curve sweep, no inner-fold model selection,
@@ -163,8 +166,11 @@ de-confound or explicit domain control), then rerun R1 on the corrected cache. T
 to the data pipeline and acquisition domain, not model size, as the binding constraint.
 
 **Pending (runnable now, handed off):** full R1 learning curve (300/1k/3k updates × 3–5 seeds) with
-inner-fold selection; participant registry; external clinical-motion pretraining; notebook
-regeneration to match the repaired method.
+inner-fold selection; participant registry; external clinical-motion pretraining.
+
+(The seven tutorial notebooks were regenerated to the corrected method — label-free SSL, stochastic
+clinically-guided masks, `PredictorV2` target positions, the locked `g1` firewall, and the honest
+R1 scoreboard — and smoke-execute end to end on the cached skeletons.)
 
 ## Reproducibility command sequence
 

-- untracked sizes --
-rw-r--r--@ 1 pmui  staff   6.6K Aug  2 16:15 ../../notes/ms/sjepa-ms-01-tutorials.md
-rw-r--r--@ 1 pmui  staff   563B Aug  2 22:55 ../../notes/ms/sjepa-ms-02-nextsteps.md
-rw-r--r--@ 1 pmui  staff    76K Aug  2 23:32 ../../notes/ms/sjepa-ms-03-improvements.md
-rw-r--r--@ 1 pmui  staff   915B Aug  3 22:05 ../../notes/ms/sjepa-ms-04-analysis.md
-rw-r--r--@ 1 pmui  staff   3.5K Aug  3 00:13 artifacts/reviews/AR1_codex_prompt.md
-rw-r--r--@ 1 pmui  staff   6.4K Aug  3 00:13 artifacts/reviews/AR1_codex_review_2026-08-03.md
-rw-r--r--@ 1 pmui  staff   1.1K Aug  4 13:07 artifacts/reviews/AR_uncommitted_codex_2026-08-04.md
-rw-r--r--@ 1 pmui  staff    14K Aug  2 17:14 docs/01-0802-PROGRESS.md
-rw-r--r--@ 1 pmui  staff   453K Aug  2 17:14 docs/01-0802-PROGRESS.pdf
-rw-r--r--@ 1 pmui  staff    48K Aug  2 17:30 docs/02-0802-NEXT_STEPS.md
-rw-r--r--@ 1 pmui  staff    18K Aug  3 00:16 docs/03-0802-PHASE_LEDGER.md
-rw-r--r--@ 1 pmui  staff    14K Aug  3 00:12 docs/04-0803-FIXES.md
-rw-r--r--@ 1 pmui  staff   6.0K Aug  2 23:35 docs/05-0803-LITERATURE_UPDATE.md
-rw-r--r--@ 1 pmui  staff    13K Aug  3 21:38 docs/06-0803-FINAL_REPORT.md
-rw-r--r--@ 1 pmui  staff    34K Aug  4 13:05 docs/07-0803-METHODOLOGY_ROADMAP.md
-rw-r--r--@ 1 pmui  staff   680B Aug  2 16:15 slides/README.md
-rw-r--r--@ 1 pmui  staff   119K Aug  3 22:08 slides/slides.html
-rw-r--r--@ 1 pmui  staff   5.6K Aug  2 16:15 slides/slides.md
-rw-r--r--@ 1 pmui  staff   166K Aug  3 22:08 slides/slides.pdf

exec
/bin/zsh -lc "python -m py_compile scripts/notebook_content.py && python - <<'PY'
import json,glob
for f in glob.glob('*.ipynb'):
 d=json.load(open(f)); print(f,len(d['cells']), d.get('nbformat'))
PY
printf '\\n-- repo files --\\n'; find . -maxdepth 3 -type f | sort | sed -n '1,240p'
printf '\\n-- scripts changed current selected --\\n'; sed -n '1,980p' scripts/notebook_content.py" in /Users/pmui/dev/alexpose/experiments/multiple-sclerosis
 succeeded in 7ms:
./.DS_Store
./.coverage
./.env
./.env.example
./.gitignore
./.python-version
./.venv/.gitignore
./.venv/.lock
./.venv/CACHEDIR.TAG
./.venv/bin/activate
./.venv/bin/activate.bat
./.venv/bin/activate.csh
./.venv/bin/activate.fish
./.venv/bin/activate.nu
./.venv/bin/activate.ps1
./.venv/bin/activate_this.py
./.venv/bin/alembic
./.venv/bin/alexpose
./.venv/bin/alexpose-server
./.venv/bin/celery
./.venv/bin/cffi-gen-src
./.venv/bin/coverage
./.venv/bin/coverage-3.12
./.venv/bin/coverage3
./.venv/bin/deactivate.bat
./.venv/bin/debugpy
./.venv/bin/debugpy-adapter
./.venv/bin/distro
./.venv/bin/dotenv
./.venv/bin/f2py
./.venv/bin/fastapi
./.venv/bin/fonttools
./.venv/bin/httpx
./.venv/bin/hypothesis
./.venv/bin/idna
./.venv/bin/ipython
./.venv/bin/ipython3
./.venv/bin/isympy
./.venv/bin/jupyter
./.venv/bin/jupyter-kernel
./.venv/bin/jupyter-kernelspec
./.venv/bin/jupyter-migrate
./.venv/bin/jupyter-run
./.venv/bin/jupyter-troubleshoot
./.venv/bin/mako-render
./.venv/bin/normalizer
./.venv/bin/numba
./.venv/bin/numpy-config
./.venv/bin/py.test
./.venv/bin/pydoc.bat
./.venv/bin/pyftmerge
./.venv/bin/pyftsubset
./.venv/bin/pygmentize
./.venv/bin/pyrsa-decrypt
./.venv/bin/pyrsa-encrypt
./.venv/bin/pyrsa-keygen
./.venv/bin/pyrsa-priv2pub
./.venv/bin/pyrsa-sign
./.venv/bin/pyrsa-verify
./.venv/bin/pysemver
./.venv/bin/pytest
./.venv/bin/torchfrtrace
./.venv/bin/torchrun
./.venv/bin/tqdm
./.venv/bin/ttx
./.venv/bin/uvicorn
./.venv/bin/watchfiles
./.venv/bin/websockets
./.venv/bin/yt-dlp
./.venv/pyvenv.cfg
./00_overview_and_video_gallery.ipynb
./01_pose_extraction_from_raw_video.ipynb
./02_anatomical_mask_and_tokenization.ipynb
./03_sjepa_model_and_pretrain_normal.ipynb
./04_progressive_finetune_ms_pd_vicreg.ipynb
./05_representation_visualization.ipynb
./06_capstone_rf_vs_sjepa.ipynb
./README.md
./__pycache__/notebook_content.cpython-312.pyc
./__pycache__/notebook_content.cpython-314.pyc
./__pycache__/scripts_build_notebooks.cpython-314.pyc
./__pycache__/scripts_make_diagrams.cpython-312.pyc
./__pycache__/scripts_make_diagrams.cpython-314.pyc
./artifacts/capstone_results.json
./artifacts/keypoints/ms__0NfZp2hmni4__0NfZp2hmni4.npz
./artifacts/keypoints/ms__0ecDhEQFrx4__0ecDhEQFrx4.npz
./artifacts/keypoints/ms__3T0BfK9HOzU__3T0BfK9HOzU.npz
./artifacts/keypoints/ms__6iCUKf9xgJg__6iCUKf9xgJg.npz
./artifacts/keypoints/ms__Ivxdl6r2z_o__Ivxdl6r2z_o.npz
./artifacts/keypoints/ms__L-41u-0tsFo__L-41u-0tsFo.npz
./artifacts/keypoints/ms__W35NeWDslAE__W35NeWDslAE.npz
./artifacts/keypoints/ms__WvoNYV6nZtM__WvoNYV6nZtM.npz
./artifacts/keypoints/ms__XIPXYpWuIX4__XIPXYpWuIX4.npz
./artifacts/keypoints/ms__tqEmrPDPIsU__tqEmrPDPIsU.npz
./artifacts/keypoints/ms__zOxtPrKySB8__zOxtPrKySB8.npz
./artifacts/keypoints/normal__-0wbleNgAwg__-0wbleNgAwg.npz
./artifacts/keypoints/normal__3FXUw98rrUY__3FXUw98rrUY.npz
./artifacts/keypoints/normal__DfRhvdCiUJk__DfRhvdCiUJk.npz
./artifacts/keypoints/normal__EHymg4AGMJs__EHymg4AGMJs_clip-01.npz
./artifacts/keypoints/normal__EHymg4AGMJs__EHymg4AGMJs_clip-02.npz
./artifacts/keypoints/normal__FTHc-TJOQ34__FTHc-TJOQ34.npz
./artifacts/keypoints/normal__JD1AGVpftps__JD1AGVpftps.npz
./artifacts/keypoints/normal__MN4vnaNwIsA__MN4vnaNwIsA.npz
./artifacts/keypoints/normal__PAeh4qBwsUk__PAeh4qBwsUk.npz
./artifacts/keypoints/normal__VL0AOiZt_lg__VL0AOiZt_lg.npz
./artifacts/keypoints/normal__WpARylM4UYU__WpARylM4UYU.npz
./artifacts/keypoints/normal___-Ubl8iD2B0___-Ubl8iD2B0.npz
./artifacts/keypoints/normal__eCCYhDSDlDc__eCCYhDSDlDc.npz
./artifacts/keypoints/normal__gp4H7Z2Vvn0__gp4H7Z2Vvn0_clip-01.npz
./artifacts/keypoints/normal__gp4H7Z2Vvn0__gp4H7Z2Vvn0_clip-02.npz
./artifacts/keypoints/normal__n-O8dHyYIF0__n-O8dHyYIF0.npz
./artifacts/keypoints/normal__nXuJIs25z1U__nXuJIs25z1U_clip-01.npz
./artifacts/keypoints/normal__nXuJIs25z1U__nXuJIs25z1U_clip-02.npz
./artifacts/keypoints/normal__tUT8Fh1zGKA__tUT8Fh1zGKA.npz
./artifacts/keypoints/pd__-oJM2wUUjws__-oJM2wUUjws.npz
./artifacts/keypoints/pd__B5hrxKe2nP8__B5hrxKe2nP8.npz
./artifacts/keypoints/pd__M-_cogKwXK4__M-_cogKwXK4_clip-01.npz
./artifacts/keypoints/pd__M-_cogKwXK4__M-_cogKwXK4_clip-02.npz
./artifacts/keypoints/pd___Wn9oYGpRdM___Wn9oYGpRdM_clip-01.npz
./artifacts/keypoints/pd___Wn9oYGpRdM___Wn9oYGpRdM_clip-02.npz
./artifacts/keypoints/pd___Wn9oYGpRdM___Wn9oYGpRdM_clip-03.npz
./artifacts/keypoints/pd__b0KrA_96Ks0__b0KrA_96Ks0.npz
./artifacts/keypoints/pd__bmi1hYOnTHs__bmi1hYOnTHs_clip-01.npz
./artifacts/keypoints/pd__bmi1hYOnTHs__bmi1hYOnTHs_clip-02.npz
./artifacts/keypoints/pd__bmi1hYOnTHs__bmi1hYOnTHs_clip-03.npz
./artifacts/keypoints/pd__pFLC9C-xH8E__pFLC9C-xH8E_clip-01.npz
./artifacts/keypoints/pd__pFLC9C-xH8E__pFLC9C-xH8E_clip-02.npz
./artifacts/keypoints/pd__pFLC9C-xH8E__pFLC9C-xH8E_clip-03.npz
./artifacts/keypoints/pd__pFLC9C-xH8E__pFLC9C-xH8E_clip-04.npz
./artifacts/keypoints/pd__pFLC9C-xH8E__pFLC9C-xH8E_clip-05.npz
./artifacts/keypoints/pd__v1SoZ_S31pk__v1SoZ_S31pk.npz
./artifacts/keypoints_index.parquet
./artifacts/manifest_grouped.csv
./artifacts/research/evidence_ledger.yaml
./artifacts/reviews/AR1_codex_prompt.md
./artifacts/reviews/AR1_codex_review_2026-08-03.md
./artifacts/reviews/AR_uncommitted_codex_2026-08-04.md
./docs/01-0802-PROGRESS.md
./docs/01-0802-PROGRESS.pdf
./docs/02-0802-NEXT_STEPS.md
./docs/03-0802-PHASE_LEDGER.md
./docs/04-0803-FIXES.md
./docs/05-0803-LITERATURE_UPDATE.md
./docs/06-0803-FINAL_REPORT.md
./docs/07-0803-METHODOLOGY_ROADMAP.md
./images/anatomical_mask.svg
./images/defect_domain_leakage.svg
./images/defect_mask_starvation.svg
./images/defect_predictor_positions.svg
./images/eval_firewall.svg
./images/experiments_tried.svg
./images/fixes_overview.svg
./images/grouped_split.svg
./images/pipeline_flowchart.svg
./images/progressive_timeline.svg
./images/project_status.svg
./images/r1_config.svg
./images/r1_results.svg
./images/repair_dependency_dag.svg
./images/results_bars.svg
./images/results_readout.svg
./images/rf_vs_sjepa.svg
./images/roadmap.svg
./images/sjepa_two_lane.svg
./images/steps_we_followed.svg
./images/tokenization.svg
./images/vicreg_clusters.svg
./images/why_it_matters.svg
./mapping-data/ms-pd-mapping.md
./pyproject.toml
./scripts/__pycache__/notebook_content.cpython-312.pyc
./scripts/__pycache__/notebook_content.cpython-314.pyc
./scripts/__pycache__/scripts_make_diagrams.cpython-312.pyc
./scripts/notebook_content.py
./scripts/scripts_build_notebooks.py
./scripts/scripts_build_progress_pdf.sh
./scripts/scripts_capstone_check.py
./scripts/scripts_extract_all.py
./scripts/scripts_make_diagrams.py
./scripts/scripts_make_fixes_diagrams.py
./scripts/scripts_phase0_provenance.py
./scripts/scripts_r1_repaired.py
./sjepa/__init__.py
./sjepa/__pycache__/__init__.cpython-312.pyc
./sjepa/__pycache__/__init__.cpython-314.pyc
./sjepa/__pycache__/augment.cpython-312.pyc
./sjepa/__pycache__/augment.cpython-314.pyc
./sjepa/__pycache__/classical.cpython-312.pyc
./sjepa/__pycache__/classical.cpython-314.pyc
./sjepa/__pycache__/config.cpython-312.pyc
./sjepa/__pycache__/config.cpython-314.pyc
./sjepa/__pycache__/data.cpython-312.pyc
./sjepa/__pycache__/data.cpython-314.pyc
./sjepa/__pycache__/eval.cpython-312.pyc
./sjepa/__pycache__/eval.cpython-314.pyc
./sjepa/__pycache__/losses.cpython-312.pyc
./sjepa/__pycache__/losses.cpython-314.pyc
./sjepa/__pycache__/masking.cpython-312.pyc
./sjepa/__pycache__/masking.cpython-314.pyc
./sjepa/__pycache__/masking_v2.cpython-312.pyc
./sjepa/__pycache__/masking_v2.cpython-314.pyc
./sjepa/__pycache__/models.cpython-312.pyc
./sjepa/__pycache__/models.cpython-314.pyc
./sjepa/__pycache__/tokenizer.cpython-312.pyc
./sjepa/__pycache__/tokenizer.cpython-314.pyc
./sjepa/__pycache__/train.cpython-312.pyc
./sjepa/__pycache__/train.cpython-314.pyc
./sjepa/__pycache__/train_v2.cpython-312.pyc
./sjepa/__pycache__/train_v2.cpython-314.pyc
./sjepa/__pycache__/viz.cpython-312.pyc
./sjepa/__pycache__/viz.cpython-314.pyc
./sjepa/augment.py
./sjepa/classical.py
./sjepa/config.py
./sjepa/data.py
./sjepa/eval.py
./sjepa/losses.py
./sjepa/masking.py
./sjepa/masking_v2.py
./sjepa/models.py
./sjepa/tests/__init__.py
./sjepa/tests/test_correctness.py
./sjepa/tests/test_smoke.py
./sjepa/tests/test_train_v2.py
./sjepa/tokenizer.py
./sjepa/train.py
./sjepa/train_v2.py
./sjepa/viz.py
./slides/README.md
./slides/slides.html
./slides/slides.md
./slides/slides.pdf
./uv.lock
./video-data/manifest.csv
./video-data/ms/0NfZp2hmni4.mp4
./video-data/ms/0ecDhEQFrx4.mp4
./video-data/ms/3T0BfK9HOzU.mp4
./video-data/ms/6iCUKf9xgJg.mp4
./video-data/ms/Ivxdl6r2z_o.mp4
./video-data/ms/JUsbspdRRJA.mp4

-- scripts changed current selected --
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
        "## Load the training windows (all sources, no labels)\n",
        "We cut every training sequence into overlapping windows. The label is ignored by the "
        "self-supervised objective; it comes along only so later notebooks can score.\n",
    )]
    c += [code(
        "from sjepa.data import load_index, SequenceWindowDataset",
        "",
        "records = load_index(KEYPOINTS_DIR)",
        "ds_all = SequenceWindowDataset(records, cfg.window_frames, cfg.window_stride)",
        "print(f'{len(records)} videos -> {len(ds_all)} training windows (labels unused in SSL)')",
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
        "UPDATES = 60 if __import__('os').environ.get('SJEPA_SMOKE') else 800",
        "state = train_sjepa_v2(model, ds_all, cfg, total_updates=UPDATES, device=device,",
        "                       mask_ratio=0.6, log_every=max(1, UPDATES // 4))",
        "save_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl.pt', model, cfg, train_state=state,",
        "                   extra={'stage': 'ssl_all_sources'})",
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
        "MORE = 40 if __import__('os').environ.get('SJEPA_SMOKE') else 400",
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
        "    print('Shortcut controls on g1 (best of logreg/rf, fold-mean macro-F1):')",
        "    for name, res in e0['shortcut_controls'].items():",
        "        best = max(res['logreg']['macro_f1']['mean'], res['rf']['macro_f1']['mean'])",
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
        "UPDATES = 60 if os.environ.get('SJEPA_SMOKE') else 500",
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
        "        best = max(res['logreg']['macro_f1']['mean'], res['rf']['macro_f1']['mean'])",
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

exec
/bin/zsh -lc "nl -ba scripts/notebook_content.py | sed -n '200,900p'" in /Users/pmui/dev/alexpose/experiments/multiple-sclerosis
 succeeded in 0ms:
   200	        "for r in recs[:5]:",
   201	        "    a = r.load_norm()",
   202	        "    assert a.ndim == 3 and a.shape[1] == 33 and a.shape[2] == 3",
   203	        "print(f'cached {len(recs)} sequences, all shaped (T, 33, 3). Ready for notebook 02.')",
   204	    )]
   205	    return c
   206	
   207	
   208	def nb_02(md, code, badge, boot):
   209	    c = [badge("02_anatomical_mask_and_tokenization.ipynb")]
   210	    c += [md(
   211	        "# 02 - Masking and tokenization\n",
   212	        "S-JEPA learns by hiding part of the skeleton and predicting the hidden part in feature space. "
   213	        "Two design choices drive this notebook: **how we cut the skeleton into tokens**, and "
   214	        "**which joints we hide**.\n",
   215	        "> **What changed, and why.** An earlier version of this project hid the *same* twelve clinical "
   216	        "joints on every single step. That turned out to be a real bug: the encoder never saw those "
   217	        "joints as context, so their internal position settings received no learning signal, yet the "
   218	        "classifier then pooled exactly those joints. We now use **stochastic graph-time masks**: a "
   219	        "different connected group of joints is hidden each step, so every joint is sometimes context "
   220	        "and sometimes a target. Clinical knowledge still guides us, but gently, by choosing the "
   221	        "leg and shoulder joints as targets a bit more often. We also do **not** bias toward the "
   222	        "busiest joints (the paper's motion-aware masking), because in MS and PD the telling sign is "
   223	        "often *reduced* motion, which a high-motion mask would hide.\n",
   224	    )]
   225	    c += boot(need_torch=False)
   226	    c += [md(
   227	        "## Tokenizing a window\n",
   228	        "A training window is a short movie of stick figures. We group `l = 4` adjacent frames of one "
   229	        "joint into a single token, so each token summarizes how that joint moved over a moment. With "
   230	        "32 frames and 33 joints that gives `(32 / 4) x 33 = 264` tokens. Token index is `t * V + v` "
   231	        "(time block `t`, joint `v`).\n",
   232	    )]
   233	    c += [code(
   234	        "from IPython.display import SVG, display",
   235	        "display(SVG(filename=str(IMAGES_DIR / 'tokenization.svg')))",
   236	    )]
   237	    c += [code(
   238	        "from sjepa.config import get_config, describe",
   239	        "cfg = get_config()  # honours SJEPA_PROFILE",
   240	        "print(describe(cfg))",
   241	        "print('tokens per window N =', cfg.num_tokens,",
   242	        "      f'= {cfg.num_time_tokens} time blocks x {cfg.num_joints} joints')",
   243	    )]
   244	    c += [md(
   245	        "## The clinical joints (domain context, not a permanent mask)\n",
   246	        "The file `mapping-data/ms-pd-mapping.md` lists the joints clinicians care about for ms and "
   247	        "pd. After removing duplicates and sorting, we get exactly twelve BlazePose landmarks: both "
   248	        "shoulders and both complete legs. We keep this table as **domain knowledge** that biases how "
   249	        "often a joint is chosen as a target, but every joint can still be both context and target.\n",
   250	    )]
   251	    c += [code(
   252	        "from sjepa.masking_v2 import CLINICAL_JOINTS",
   253	        "from ambient.pose.keypoint_data import MEDIAPIPE_33_NAMES",
   254	        "import pandas as pd",
   255	        "",
   256	        "features_for = {",
   257	        "    11: 'shoulder_symmetry_index, trunk_lean_angle',",
   258	        "    12: 'shoulder_symmetry_index, trunk_lean_angle',",
   259	        "    23: 'walking_speed_ms, hip_asymmetry, knee_range, trunk_lean_angle',",
   260	        "    24: 'walking_speed_ms, hip_asymmetry, knee_range, trunk_lean_angle',",
   261	        "    25: 'knee_range, ankle_range', 26: 'knee_range, ankle_range',",
   262	        "    27: 'knee_range, ankle_range, step_width_m', 28: 'knee_range, ankle_range, step_width_m',",
   263	        "    29: 'stride_length_m, double_support_pct, stride_time_cv, ankle_range',",
   264	        "    30: 'stride_length_m, double_support_pct, stride_time_cv, ankle_range',",
   265	        "    31: 'stride_length_m, double_support_pct, stride_time_cv, ankle_range',",
   266	        "    32: 'stride_length_m, double_support_pct, stride_time_cv, ankle_range',",
   267	        "}",
   268	        "table = pd.DataFrame([",
   269	        "    {'BLAZEPOSE_33 index': j, 'Keypoint name': MEDIAPIPE_33_NAMES[j],",
   270	        "     'Features involved': features_for[j]}",
   271	        "    for j in sorted(CLINICAL_JOINTS)",
   272	        "])",
   273	        "table",
   274	    )]
   275	    c += [md(
   276	        "## Stochastic graph-time masks\n",
   277	        "Each step we sample a per-example mask: connected groups of joints (a limb or the trunk) over "
   278	        "a contiguous span of time. The cell below samples a few masks and shows they differ, that "
   279	        "every one keeps visible context, and that over a bank of masks every joint is both visible "
   280	        "and targeted often enough (the coverage gates).\n",
   281	    )]
   282	    c += [code(
   283	        "import numpy as np",
   284	        "from sjepa.masking_v2 import sample_mask_batch, mask_bank_stats",
   285	        "",
   286	        "rng = np.random.default_rng(0)",
   287	        "batch = sample_mask_batch(6, cfg.num_joints, cfg.num_time_tokens, rng)",
   288	        "print('mask batch shape (B, N):', batch.shape)",
   289	        "print('unique masks in the batch:', len({row.tobytes() for row in batch}), 'of 6')",
   290	        "print('every row has context and target:',",
   291	        "      bool((~batch).any(1).all() and batch.any(1).all()))",
   292	        "",
   293	        "stats = mask_bank_stats(cfg.num_joints, cfg.num_time_tokens, n_masks=512, seed=0)",
   294	        "print(f'over 512 masks: min joint-visible {stats.joint_visible_frac.min():.2f} '",
   295	        "      f'(gate >=0.20), min joint-target {stats.joint_target_frac.min():.2f} (gate >=0.10)')",
   296	        "print(f'mean target fraction {stats.mean_target_frac:.2f}')",
   297	    )]
   298	    c += [md(
   299	        "Here is the difference drawn out: a fixed mask hides the same joints forever (left), while "
   300	        "stochastic masks rotate which joints are hidden (right).\n",
   301	    )]
   302	    c += [code(
   303	        "display(SVG(filename=str(IMAGES_DIR / 'defect_mask_starvation.svg')))",
   304	    )]
   305	    c += [md(
   306	        "## See one mask on a real skeleton\n",
   307	        "The animation highlights one sampled set of masked joints in red on a real walking sequence. "
   308	        "Next time you sample, a different group will be hidden.\n",
   309	    )]
   310	    c += [code(
   311	        "from sjepa.data import load_index",
   312	        "from sjepa.masking_v2 import sample_target_mask",
   313	        "from sjepa.viz import skeleton_animation",
   314	        "from IPython.display import Image",
   315	        "",
   316	        "recs = load_index(KEYPOINTS_DIR)",
   317	        "seq = recs[0].load_norm()",
   318	        "tgt = sample_target_mask(cfg.num_joints, cfg.num_time_tokens, np.random.default_rng(1))",
   319	        "masked_joints = sorted({int(i % cfg.num_joints) for i in np.nonzero(tgt)[0]})",
   320	        "gif = skeleton_animation(seq, ARTIFACT_DIR / 'mask_demo.gif',",
   321	        "                         masked_joints=masked_joints, fps=15,",
   322	        "                         title='red = one sampled set of masked joints')",
   323	        "Image(filename=str(gif))",
   324	    )]
   325	    return c
   326	
   327	
   328	def nb_03(md, code, badge, boot):
   329	    c = [badge("03_sjepa_model_and_pretrain_normal.ipynb")]
   330	    c += [md(
   331	        "# 03 - Build S-JEPA and pretrain (label-free)\n",
   332	        "Now we build the model and train it, with **no labels**, on the walking motion "
   333	        "itself. S-JEPA has three parts, all small transformers:\n",
   334	        "- a **view encoder** that reads the visible joints of a slightly transformed view,\n"
   335	        "- a **predictor** that guesses the hidden joints in feature space. Crucially, it is told "
   336	        "*which* joint and *which* time each hidden slot is (a factorized position tag), so it can "
   337	        "make a different guess per position. Without that tag every hidden guess is identical, which "
   338	        "was a real bug in an earlier version.\n"
   339	        "- a **target encoder** that reads the full skeleton and provides the answer. It is a slow "
   340	        "moving average of the view encoder, which is what stops the model from collapsing every "
   341	        "skeleton to the same features.\n",
   342	        "> **A note on what trains here.** For three-class classification the useful thing is to learn "
   343	        "from *all* the fold's unlabeled walking, so this notebook trains label-free on every training "
   344	        "source. Training on normal gait *only* is a different question (one-class anomaly detection); "
   345	        "we keep that as a separate idea, not the default.\n",
   346	    )]
   347	    c += boot(need_torch=True)
   348	    c += [md(
   349	        "## The two-lane design\n",
   350	        "The picture below is the whole idea. The top lane makes a prediction from a masked view. The "
   351	        "bottom lane makes the target from the complete skeleton with a slow teacher. They meet only "
   352	        "at the loss.\n",
   353	    )]
   354	    c += [code(
   355	        "from IPython.display import SVG, display",
   356	        "display(SVG(filename=str(IMAGES_DIR / 'sjepa_two_lane.svg')))",
   357	        "display(SVG(filename=str(IMAGES_DIR / 'defect_predictor_positions.svg')))",
   358	    )]
   359	    c += [code(
   360	        "from sjepa.config import get_config, describe",
   361	        "from sjepa.models import build_model, pick_device",
   362	        "",
   363	        "cfg = get_config()",
   364	        "device = pick_device()",
   365	        "print('device:', device)",
   366	        "print(describe(cfg))",
   367	        "model = build_model(cfg, device=device, repaired=True)  # PredictorV2 + per-example masks",
   368	        "n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)",
   369	        "print(f'trainable parameters: {n_params/1e6:.2f}M')",
   370	    )]
   371	    c += [md(
   372	        "## Load the training windows (all sources, no labels)\n",
   373	        "We cut every training sequence into overlapping windows. The label is ignored by the "
   374	        "self-supervised objective; it comes along only so later notebooks can score.\n",
   375	    )]
   376	    c += [code(
   377	        "from sjepa.data import load_index, SequenceWindowDataset",
   378	        "",
   379	        "records = load_index(KEYPOINTS_DIR)",
   380	        "ds_all = SequenceWindowDataset(records, cfg.window_frames, cfg.window_stride)",
   381	        "print(f'{len(records)} videos -> {len(ds_all)} training windows (labels unused in SSL)')",
   382	    )]
   383	    c += [md(
   384	        "## Train\n",
   385	        "We run the two-lane objective for a fixed number of **optimizer updates** (not "
   386	        "epochs), sampling sources uniformly so a long clip cannot dominate, and nudge the teacher "
   387	        "with a responsive EMA. We log the loss and collapse diagnostics: the per-dimension spread, "
   388	        "the effective rank (how many directions the features really use), and how far the teacher has "
   389	        "drifted from the student.\n",
   390	    )]
   391	    c += [code(
   392	        "from sjepa.train_v2 import train_sjepa_v2, save_checkpoint_v2",
   393	        "",
   394	        "# A small update budget keeps the notebook fast; raise it for stronger features.",
   395	        "UPDATES = 60 if __import__('os').environ.get('SJEPA_SMOKE') else 800",
   396	        "state = train_sjepa_v2(model, ds_all, cfg, total_updates=UPDATES, device=device,",
   397	        "                       mask_ratio=0.6, log_every=max(1, UPDATES // 4))",
   398	        "save_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl.pt', model, cfg, train_state=state,",
   399	        "                   extra={'stage': 'ssl_all_sources'})",
   400	        "print('EMA half-life (steps):', round(state.ema_half_life_steps, 1))",
   401	    )]
   402	    c += [code(
   403	        "import matplotlib.pyplot as plt",
   404	        "fig, ax = plt.subplots(1, 3, figsize=(12, 3))",
   405	        "ax[0].plot(state.losses, color='#dd6b20'); ax[0].set_title('latent cross-entropy')",
   406	        "ax[1].plot(state.eff_rank, color='#2563eb'); ax[1].set_title('effective rank (higher = richer)')",
   407	        "ax[2].plot(state.teacher_drift, color='#16a34a'); ax[2].set_title('teacher drift from student')",
   408	        "for a in ax: a.set_xlabel('update')",
   409	        "plt.tight_layout(); plt.show()",
   410	    )]
   411	    c += [md(
   412	        "### Sanity checks\n",
   413	        "We avoid weak checks. A falling loss is necessary but not sufficient: a collapsed model can "
   414	        "also lower the loss. So we also require the effective rank to stay well above 1 (the features "
   415	        "use many directions, not one) and the teacher to have moved.\n",
   416	    )]
   417	    c += [code(
   418	        "import numpy as np",
   419	        "early = np.mean(state.losses[:5]); late = np.mean(state.losses[-5:])",
   420	        "print(f'loss {early:.3f} -> {late:.3f} | final effective rank {state.eff_rank[-1]:.1f}')",
   421	        "assert np.isfinite(state.losses).all(), 'loss went non-finite'",
   422	        "assert state.eff_rank[-1] > 1.5, 'representation looks collapsed (effective rank near 1)'",
   423	        "assert state.teacher_drift[-1] >= 0, 'teacher drift should be finite and non-negative'",
   424	        "print('SSL looks healthy (no collapse). On to comparing training regimes.')",
   425	    )]
   426	    return c
   427	
   428	
   429	def nb_04(md, code, badge, boot):
   430	    c = [badge("04_progressive_finetune_ms_pd_vicreg.ipynb")]
   431	    c += [md(
   432	        "# 04 - Two ways to adapt the encoder: SSL continuation vs supervised adaptation\n",
   433	        "Notebook 03 trained the encoder with **no labels** on all the walking motion in the training "
   434	        "set. Now we ask a sharper question: once we do have diagnosis labels, what is the honest way "
   435	        "to use them, and does it actually help the three conditions separate?\n",
   436	        "We compare two clearly named regimes, both starting from the same label-free checkpoint:\n",
   437	        "1. **SSL continuation** - keep training with the *same* label-free objective for more updates. "
   438	        "No labels touch the model.\n"
   439	        "2. **Balanced supervised adaptation** - freeze the encoder and fit a small, class-balanced "
   440	        "linear head on top. Labels are used *only* in this head, never inside the self-supervised "
   441	        "objective.\n",
   442	        "> **What changed, and why.** An earlier version of this notebook mixed the diagnosis label "
   443	        "*into* the self-supervised loss (a 'class-aware VICReg' that rewarded within-class spread) and "
   444	        "claimed it 'compacts the classes'. That was a leak: the SSL objective must not see labels, and "
   445	        "the claim was not supported. We removed it. If labels help, they help in an explicitly "
   446	        "supervised stage that we name and measure, not smuggled into the pretext task.\n",
   447	    )]
   448	    c += boot(need_torch=True)
   449	    c += [code(
   450	        "from IPython.display import SVG, display",
   451	        "display(SVG(filename=str(IMAGES_DIR / 'progressive_timeline.svg')))",
   452	    )]
   453	    c += [md(
   454	        "## Use the locked, leakage-safe fold registry\n",
   455	        "The comparison is only meaningful on a split where clips from one source never straddle "
   456	        "train and test. Notebook's Phase 0 froze such a split to `artifacts/eval/g1/fold_registry.json` "
   457	        "(source-grouped, seed 42). We load fold 0 from it here rather than inventing a fresh split, so "
   458	        "this notebook, notebook 05, and notebook 06 all sit on the identical partition.\n",
   459	        "> Source grouping is **provisional**: a `source_id` is a YouTube id, not a verified person, so "
   460	        "everything here is a development estimate, never a clinical claim.\n",
   461	    )]
   462	    c += [code(
   463	        "import json",
   464	        "from sjepa.data import load_index",
   465	        "",
   466	        "records = load_index(KEYPOINTS_DIR)",
   467	        "by_clip = {r.clip_name: r for r in records}",
   468	        "registry = json.loads((ARTIFACT_DIR / 'eval' / 'g1' / 'fold_registry.json').read_text())",
   469	        "fold0 = registry['folds'][0]",
   470	        "train_recs = [by_clip[c] for c in fold0['train_clips']]",
   471	        "test_recs  = [by_clip[c] for c in fold0['test_clips']]",
   472	        "tr_src = {r.source_id for r in train_recs}; te_src = {r.source_id for r in test_recs}",
   473	        "assert not (tr_src & te_src), 'source leakage across the fold'",
   474	        "print('fold 0:', len(train_recs), 'train videos /', len(test_recs), 'test videos')",
   475	        "print('no source in both sides:', not (tr_src & te_src))",
   476	    )]
   477	    c += [md(
   478	        "## Regime 1 - SSL continuation (no labels)\n",
   479	        "We rebuild the model, load the label-free checkpoint from notebook 03, and keep "
   480	        "training with the exact same objective for a few hundred more updates. The label is never "
   481	        "read. This asks: does more unlabeled training alone sharpen the representation?\n",
   482	    )]
   483	    c += [code(
   484	        "from sjepa.config import get_config",
   485	        "from sjepa.models import build_model, pick_device",
   486	        "from sjepa.train_v2 import train_sjepa_v2, load_checkpoint_v2, save_checkpoint_v2",
   487	        "from sjepa.data import SequenceWindowDataset",
   488	        "",
   489	        "cfg = get_config()",
   490	        "device = pick_device()",
   491	        "model = build_model(cfg, device=device, repaired=True)",
   492	        "load_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl.pt', model, map_location=device)",
   493	        "",
   494	        "ds_train = SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride)",
   495	        "MORE = 40 if __import__('os').environ.get('SJEPA_SMOKE') else 400",
   496	        "state = train_sjepa_v2(model, ds_train, cfg, total_updates=MORE, device=device,",
   497	        "                       mask_ratio=0.6, log_every=max(1, MORE // 4))",
   498	        "save_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl_continued.pt', model, cfg, train_state=state,",
   499	        "                   extra={'stage': 'ssl_continuation_fold0'})",
   500	        "print('continued SSL: final effective rank', round(state.eff_rank[-1], 1))",
   501	    )]
   502	    c += [md(
   503	        "## A fixed, label-free read-out\n",
   504	        "To turn a video into one vector we mean-pool the frozen target encoder over a **fixed** pool "
   505	        "of target tokens. The pool is chosen once from a seeded RNG and never from the test labels, so "
   506	        "no information leaks from the evaluation into the representation. Both regimes below use this "
   507	        "same read-out.\n",
   508	    )]
   509	    c += [code(
   510	        "import numpy as np, torch",
   511	        "from sjepa.masking_v2 import sample_target_mask",
   512	        "from sjepa.data import sliding_windows",
   513	        "",
   514	        "readout = sample_target_mask(cfg.num_joints, cfg.num_time_tokens,",
   515	        "                             np.random.default_rng(0), target_ratio=0.6)",
   516	        "tm = torch.from_numpy(readout).to(device)",
   517	        "",
   518	        "def embed_records(m, recs):",
   519	        "    V, Y = [], []",
   520	        "    for r in recs:",
   521	        "        w = sliding_windows(r.load_norm(), cfg.window_frames, cfg.window_stride)",
   522	        "        x = torch.from_numpy(w).float().to(device)",
   523	        "        with torch.no_grad():",
   524	        "            V.append(m.embed(x, tm).mean(0).cpu().numpy())",
   525	        "        Y.append(r.label)",
   526	        "    return np.stack(V), Y",
   527	    )]
   528	    c += [md(
   529	        "## Regime 2 - balanced supervised adaptation (labels only in the head)\n",
   530	        "Now we use the labels honestly: freeze the encoder and fit a class-balanced logistic head on "
   531	        "the training embeddings, then score the held-out videos. The scaler and the head are fit on "
   532	        "**training data only**. We do this on top of both the notebook-03 checkpoint and the "
   533	        "SSL-continued one, so we can see whether extra unlabeled training moved the probe at all.\n",
   534	    )]
   535	    c += [code(
   536	        "from sklearn.linear_model import LogisticRegression",
   537	        "from sklearn.preprocessing import StandardScaler",
   538	        "from sjepa.eval import evaluate",
   539	        "LABELS = ['normal', 'ms', 'pd']",
   540	        "",
   541	        "def probe_and_score(ckpt):",
   542	        "    m = build_model(cfg, device=device, repaired=True)",
   543	        "    load_checkpoint_v2(ckpt, m, map_location=device)",
   544	        "    Etr, ytr = embed_records(m, train_recs)",
   545	        "    Ete, yte = embed_records(m, test_recs)",
   546	        "    sc = StandardScaler().fit(Etr)               # fit on TRAIN only",
   547	        "    clf = LogisticRegression(max_iter=2000, class_weight='balanced')",
   548	        "    clf.fit(sc.transform(Etr), ytr)",
   549	        "    return evaluate(yte, clf.predict(sc.transform(Ete)), LABELS)",
   550	        "",
   551	        "m_base = probe_and_score(ARTIFACT_DIR / 'sjepa_ssl.pt')",
   552	        "m_cont = probe_and_score(ARTIFACT_DIR / 'sjepa_ssl_continued.pt')",
   553	        "print(f'supervised probe on fold 0 held-out videos (macro-F1):')",
   554	        "print(f'  notebook-03 checkpoint : {m_base.macro_f1:.3f}')",
   555	        "print(f'  after SSL continuation : {m_cont.macro_f1:.3f}')",
   556	    )]
   557	    c += [md(
   558	        "## Read this honestly\n",
   559	        "On this tiny fold the two numbers are close and noisy; do not over-read a few points either "
   560	        "way. The point of the notebook is the **method**: labels live only in the supervised head, the "
   561	        "SSL objective stays label-free, and the split is the locked, leakage-safe one. Notebook 06 "
   562	        "runs this over all folds and puts it beside the Random Forest and the shortcut controls, which "
   563	        "is where any real verdict lives. A single fold here proves nothing on its own.\n",
   564	    )]
   565	    return c
   566	
   567	
   568	def nb_05(md, code, badge, boot):
   569	    c = [badge("05_representation_visualization.ipynb")]
   570	    c += [md(
   571	        "# 05 - Looking at the learned representation (diagnostics only)\n",
   572	        "We have a trained encoder. What did it actually learn? Here we turn each video into a single "
   573	        "feature vector with the frozen target encoder and project those vectors to two dimensions with "
   574	        "t-SNE and UMAP, to *see* whether normal, ms, and pd land in different regions.\n",
   575	        "> **These pictures are diagnostics, not evidence.** Two honest cautions run through this "
   576	        "notebook. First, t-SNE and UMAP distort distances; a clean-looking blob can be an artifact of "
   577	        "the projection. Second, and more important, apparent separation can come from a **shortcut** "
   578	        "(camera frame rate, body size, how visible the joints are) rather than from gait. So we plot "
   579	        "the S-JEPA embedding *and* a cheap nuisance feature side by side: if the nuisance separates "
   580	        "just as well, the pretty S-JEPA plot is not telling us about gait. The verdict lives in "
   581	        "notebook 06's leakage-safe scores, never in a scatter plot.\n",
   582	        "We compare the label-free checkpoint from notebook 03 against the SSL-continued one from "
   583	        "notebook 04. No test labels are ever used to fit, select, or color anything beyond the plain "
   584	        "class of each point.\n",
   585	    )]
   586	    c += boot(need_torch=True)
   587	    c += [code(
   588	        "from IPython.display import SVG, display",
   589	        "display(SVG(filename=str(IMAGES_DIR / 'vicreg_clusters.svg')))",
   590	    )]
   591	    c += [md(
   592	        "## Embed every video with the frozen encoder\n",
   593	        "For each video we mean-pool the encoder's features over its windows and over a **fixed**, "
   594	        "seeded read-out pool of target tokens (the same pool used in notebooks 04 and 06). This pool "
   595	        "is never chosen from labels. We do it for both checkpoints.\n",
   596	    )]
   597	    c += [code(
   598	        "import numpy as np, torch",
   599	        "from sjepa.config import get_config",
   600	        "from sjepa.models import build_model, pick_device",
   601	        "from sjepa.train_v2 import load_checkpoint_v2",
   602	        "from sjepa.masking_v2 import sample_target_mask",
   603	        "from sjepa.data import load_index, sliding_windows",
   604	        "",
   605	        "cfg = get_config(); device = pick_device()",
   606	        "records = load_index(KEYPOINTS_DIR)",
   607	        "readout = sample_target_mask(cfg.num_joints, cfg.num_time_tokens,",
   608	        "                             np.random.default_rng(0), target_ratio=0.6)",
   609	        "tm = torch.from_numpy(readout).to(device)",
   610	        "",
   611	        "def embed_all(ckpt):",
   612	        "    m = build_model(cfg, device=device, repaired=True)",
   613	        "    load_checkpoint_v2(ckpt, m, map_location=device)",
   614	        "    vecs, labels = [], []",
   615	        "    for r in records:",
   616	        "        w = sliding_windows(r.load_norm(), cfg.window_frames, cfg.window_stride)",
   617	        "        x = torch.from_numpy(w).float().to(device)",
   618	        "        with torch.no_grad():",
   619	        "            vecs.append(m.embed(x, tm).mean(0).cpu().numpy())",
   620	        "        labels.append(r.label)",
   621	        "    return np.stack(vecs), labels",
   622	        "",
   623	        "E_base, y = embed_all(ARTIFACT_DIR / 'sjepa_ssl.pt')",
   624	        "E_cont, _ = embed_all(ARTIFACT_DIR / 'sjepa_ssl_continued.pt')",
   625	        "np.savez(ARTIFACT_DIR / 'embeddings_3class.npz', E_base=E_base, E_continued=E_cont,",
   626	        "         labels=np.array(y))",
   627	        "print('embedded', len(y), 'videos into', E_cont.shape[1], 'dimensions')",
   628	    )]
   629	    c += [md(
   630	        "## A nuisance baseline to keep us honest\n",
   631	        "This is the cheapest possible 'representation': the per-joint mean and spread of the raw "
   632	        "visibility channel, which we already know tracks the acquisition domain (the MS clips were all "
   633	        "filmed at 60fps). If this separates the classes as cleanly as S-JEPA does, then a tidy S-JEPA "
   634	        "scatter is not evidence of learned gait.\n",
   635	    )]
   636	    c += [code(
   637	        "def nuisance_vec(r):",
   638	        "    vis = r.load_raw()[:, :, 2]",
   639	        "    return np.nan_to_num(np.concatenate([np.nanmean(vis, 0), np.nanstd(vis, 0)]))",
   640	        "N_nuis = np.stack([nuisance_vec(r) for r in records])",
   641	        "print('nuisance feature shape:', N_nuis.shape)",
   642	    )]
   643	    c += [md(
   644	        "## Project and plot\n",
   645	        "t-SNE squeezes the high-dimensional vectors into a plane while trying to keep neighbors "
   646	        "together. We color one hue per condition, and we place the nuisance baseline in the same row "
   647	        "for the comparison the caution above demands.\n",
   648	    )]
   649	    c += [code(
   650	        "from sklearn.manifold import TSNE",
   651	        "from sjepa.viz import scatter_2d",
   652	        "import matplotlib.pyplot as plt",
   653	        "",
   654	        "def tsne2d(E):",
   655	        "    perp = min(15, max(2, len(E)//3))",
   656	        "    return TSNE(n_components=2, perplexity=perp, random_state=42, init='pca').fit_transform(E)",
   657	        "",
   658	        "fig, ax = plt.subplots(1, 3, figsize=(15,4.4))",
   659	        "scatter_2d(tsne2d(E_base), y, ax[0], 't-SNE: S-JEPA (label-free, nb 03)')",
   660	        "scatter_2d(tsne2d(E_cont), y, ax[1], 't-SNE: S-JEPA (SSL continued, nb 04)')",
   661	        "scatter_2d(tsne2d(N_nuis), y, ax[2], 't-SNE: nuisance (visibility only)')",
   662	        "plt.tight_layout(); plt.savefig(IMAGES_DIR / 'tsne_sjepa_vs_nuisance.png', dpi=130)",
   663	        "plt.show()",
   664	    )]
   665	    c += [code(
   666	        "# UMAP view (falls back gracefully if umap-learn is missing).",
   667	        "import matplotlib.pyplot as plt",
   668	        "from sjepa.viz import scatter_2d",
   669	        "try:",
   670	        "    import umap",
   671	        "    def umap2d(E):",
   672	        "        nn = min(15, max(2, len(E)//3))",
   673	        "        return umap.UMAP(n_neighbors=nn, min_dist=0.3, random_state=42).fit_transform(E)",
   674	        "    fig, ax = plt.subplots(1, 3, figsize=(15,4.4))",
   675	        "    scatter_2d(umap2d(E_base), y, ax[0], 'UMAP: S-JEPA (nb 03)')",
   676	        "    scatter_2d(umap2d(E_cont), y, ax[1], 'UMAP: S-JEPA (nb 04)')",
   677	        "    scatter_2d(umap2d(N_nuis), y, ax[2], 'UMAP: nuisance (visibility)')",
   678	        "    plt.tight_layout(); plt.show()",
   679	        "except Exception as e:",
   680	        "    print('UMAP not available, skipping:', e)",
   681	    )]
   682	    c += [md(
   683	        "## Put a (descriptive) number on the separation\n",
   684	        "The silhouette score summarizes how tight and well separated the class clusters are, from -1 "
   685	        "to +1. We report it for all three embeddings so the S-JEPA numbers are read *against* the "
   686	        "nuisance number, not in isolation. On ~47 videos this is noisy and purely descriptive: it is "
   687	        "computed on the whole set, so it is **not** an out-of-sample score and must not be used to "
   688	        "pick a model. Model selection happens only through the leakage-safe folds in notebook 06.\n",
   689	    )]
   690	    c += [code(
   691	        "from sjepa.eval import silhouette",
   692	        "s_base = silhouette(E_base, y)",
   693	        "s_cont = silhouette(E_cont, y)",
   694	        "s_nuis = silhouette(N_nuis, y)",
   695	        "print(f'silhouette (descriptive, whole set):')",
   696	        "print(f'  S-JEPA label-free (nb 03): {s_base:.3f}')",
   697	        "print(f'  S-JEPA SSL continued (nb 04): {s_cont:.3f}')",
   698	        "print(f'  nuisance (visibility only): {s_nuis:.3f}')",
   699	        "if s_nuis >= max(s_base, s_cont):",
   700	        "    print('Note: the nuisance baseline separates at least as well -- a clean S-JEPA plot',",
   701	        "          'here would NOT be evidence of learned gait. See notebook 06.')",
   702	    )]
   703	    return c
   704	
   705	
   706	def nb_06(md, code, badge, boot):
   707	    c = [badge("06_capstone_rf_vs_sjepa.ipynb")]
   708	    c += [md(
   709	        "# 06 - Capstone: Random Forest vs S-JEPA, on identical folds, with controls\n",
   710	        "This is the scientific payoff, and it comes with a result that is honest rather than "
   711	        "flattering. We put several systems side by side on the **same videos**, the **same locked "
   712	        "leakage-safe folds**, and the **same pooled out-of-fold scoring**:\n",
   713	        "1. a classical **Random Forest** on hand-made gait features (the exp5 recipe, three classes),\n"
   714	        "2. the **label-free S-JEPA** with a frozen linear probe,\n"
   715	        "3. cheap **shortcut controls** (visibility, body size, static pose) that any real "
   716	        "representation must beat before we trust it.\n",
   717	        "The headline is **pooled macro-F1** over the folds (one prediction per clip, gathered across "
   718	        "all held-out folds), because averaging per-fold F1 on ~9 test videos is even noisier. We "
   719	        "report it beside the paired RF and the controls, then say plainly what it does and does not "
   720	        "mean.\n",
   721	        "> **Spoiler, stated up front.** On this tiny, already-inspected, source-grouped collection the "
   722	        "S-JEPA scores *below* both the Random Forest and the nuisance controls. That is the "
   723	        "expected, plan-anticipated outcome of removing the shortcuts and the label leak that inflated "
   724	        "an earlier number, and it is reported as a negative result, not hidden.\n",
   725	    )]
   726	    c += boot(need_torch=True)
   727	    c += [code(
   728	        "from IPython.display import SVG, display",
   729	        "display(SVG(filename=str(IMAGES_DIR / 'rf_vs_sjepa.svg')))",
   730	        "display(SVG(filename=str(IMAGES_DIR / 'grouped_split.svg')))",
   731	        "display(SVG(filename=str(IMAGES_DIR / 'eval_firewall.svg')))",
   732	    )]
   733	    c += [md(
   734	        "## The frozen result is produced by a script, not the notebook\n",
   735	        "So the headline cannot drift as someone re-runs cells, the authoritative R1 run lives in "
   736	        "`scripts/scripts_r1_repaired.py` and its output is committed under "
   737	        "`artifacts/runs/r1_g1_1k_s42/`. That script and this notebook share the identical fold "
   738	        "registry, so the comparison is paired by construction. We read the frozen numbers first, then "
   739	        "reproduce the mechanism live at a smaller budget so you can see how it is built.\n",
   740	    )]
   741	    c += [code(
   742	        "import json",
   743	        "frozen_path = ARTIFACT_DIR / 'runs' / 'r1_g1_1k_s42' / 'results.json'",
   744	        "if frozen_path.exists():",
   745	        "    frozen = json.loads(frozen_path.read_text())",
   746	        "    sj = frozen['sjepa_pooled']; rf = frozen['rf_pooled']",
   747	        "    print('Frozen R1 (1000 updates, seed 42, all 5 folds, pooled OOF):')",
   748	        "    print(f\"  S-JEPA          : macro-F1 {sj['macro_f1']:.3f} | acc {sj['accuracy']:.3f}\"",
   749	        "          f\" | PD-recall {sj['pd_recall']:.3f}\")",
   750	        "    print(f\"  Random Forest   : macro-F1 {rf['macro_f1']:.3f} | acc {rf['accuracy']:.3f}\"",
   751	        "          f\" | PD-recall {rf['pd_recall']:.3f}\")",
   752	        "    print('  effective rank per fold:',",
   753	        "          [round(d['eff_rank_final'], 1) for d in frozen['diagnostics']],",
   754	        "          '(all >> 1 -> no collapse)')",
   755	        "else:",
   756	        "    print('Frozen run not found. Reproduce it with:')",
   757	        "    print('  python scripts/scripts_r1_repaired.py --total-updates 1000 --seed 42 \\\\')",
   758	        "    print('      --output-dir artifacts/runs/r1_g1_1k_s42')",
   759	    )]
   760	    c += [md(
   761	        "## Shortcut controls: the bar S-JEPA has to clear\n",
   762	        "Phase 0 also scored cheap nuisance features on the identical folds. If a control matches or "
   763	        "beats S-JEPA, then S-JEPA is not yet using gait beyond what a camera artifact already reveals. "
   764	        "We read those frozen control scores here.\n",
   765	    )]
   766	    c += [code(
   767	        "e0_path = ARTIFACT_DIR / 'eval' / 'g1' / 'E0_results.json'",
   768	        "if e0_path.exists():",
   769	        "    e0 = json.loads(e0_path.read_text())",
   770	        "    print('Shortcut controls on g1 (best of logreg/rf, fold-mean macro-F1):')",
   771	        "    for name, res in e0['shortcut_controls'].items():",
   772	        "        best = max(res['logreg']['macro_f1']['mean'], res['rf']['macro_f1']['mean'])",
   773	        "        print(f'  {name:16s}: {best:.3f}')",
   774	        "    print(f\"E0 Random Forest pooled macro-F1: {e0['E0_RF']['pooled_macro_f1']:.3f}\")",
   775	        "else:",
   776	        "    print('Run scripts/scripts_phase0_provenance.py to generate the control table.')",
   777	    )]
   778	    c += [md(
   779	        "## Reproduce the mechanism live (one fold, small budget)\n",
   780	        "Now the moving parts, so nothing is a black box. For one locked fold we run the exact "
   781	        "pipeline: paired RF, then **label-free** S-JEPA (no diagnosis label enters the objective), a "
   782	        "frozen mean-pool over a fixed read-out, and a class-balanced probe fit on the training clips "
   783	        "only. This uses a tiny update budget to stay fast; the frozen numbers above are the ones to "
   784	        "cite.\n",
   785	    )]
   786	    c += [code(
   787	        "import numpy as np, torch, os",
   788	        "from sjepa.config import get_config",
   789	        "from sjepa.models import build_model, pick_device",
   790	        "from sjepa.train_v2 import train_sjepa_v2",
   791	        "from sjepa.masking_v2 import sample_target_mask",
   792	        "from sjepa.data import load_index, SequenceWindowDataset, sliding_windows",
   793	        "from sjepa.classical import build_feature_matrix, train_rf_and_predict",
   794	        "from sjepa.eval import evaluate",
   795	        "from sklearn.linear_model import LogisticRegression",
   796	        "from sklearn.preprocessing import StandardScaler",
   797	        "",
   798	        "cfg = get_config(); device = pick_device()",
   799	        "LABELS = ['normal', 'ms', 'pd']",
   800	        "records = load_index(KEYPOINTS_DIR)",
   801	        "by_clip = {r.clip_name: r for r in records}",
   802	        "registry = json.loads((ARTIFACT_DIR / 'eval' / 'g1' / 'fold_registry.json').read_text())",
   803	        "readout = sample_target_mask(cfg.num_joints, cfg.num_time_tokens,",
   804	        "                             np.random.default_rng(0), target_ratio=0.6)",
   805	        "tm = torch.from_numpy(readout).to(device)",
   806	        "",
   807	        "def embed_records(model, recs):",
   808	        "    V, Y = [], []",
   809	        "    for r in recs:",
   810	        "        w = sliding_windows(r.load_norm(), cfg.window_frames, cfg.window_stride)",
   811	        "        x = torch.from_numpy(w).float().to(device)",
   812	        "        with torch.no_grad():",
   813	        "            V.append(model.embed(x, tm).mean(0).cpu().numpy())",
   814	        "        Y.append(r.label)",
   815	        "    return np.stack(V), Y",
   816	    )]
   817	    c += [code(
   818	        "fold0 = registry['folds'][0]",
   819	        "train_recs = [by_clip[c] for c in fold0['train_clips']]",
   820	        "test_recs  = [by_clip[c] for c in fold0['test_clips']]",
   821	        "",
   822	        "# paired Random Forest (exp5 recipe) on this fold",
   823	        "Xtr, ytr, _, _ = build_feature_matrix(train_recs, fps=cfg.target_fps)",
   824	        "Xte, yte, _, _ = build_feature_matrix(test_recs, fps=cfg.target_fps)",
   825	        "rf_pred = train_rf_and_predict(Xtr, ytr, Xte, seed=cfg.seed)",
   826	        "rf_m = evaluate(yte, rf_pred, LABELS)",
   827	        "",
   828	        "# label-free S-JEPA on this fold's training sources",
   829	        "UPDATES = 60 if os.environ.get('SJEPA_SMOKE') else 500",
   830	        "model = build_model(cfg, device=device, repaired=True)",
   831	        "ds = SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride)",
   832	        "state = train_sjepa_v2(model, ds, cfg, total_updates=UPDATES, device=device, mask_ratio=0.6)",
   833	        "Etr, ytr2 = embed_records(model, train_recs)",
   834	        "Ete, yte2 = embed_records(model, test_recs)",
   835	        "sc = StandardScaler().fit(Etr)                    # TRAIN only",
   836	        "probe = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(Etr), ytr2)",
   837	        "sj_m = evaluate(yte2, probe.predict(sc.transform(Ete)), LABELS)",
   838	        "print(f'live fold-0 demo (budget={UPDATES} updates): RF f1={rf_m.macro_f1:.3f}',",
   839	        "      f'| S-JEPA f1={sj_m.macro_f1:.3f} | eff_rank={state.eff_rank[-1]:.1f}')",
   840	        "print('(The frozen 5-fold pooled numbers above are the ones to cite, not this one fold.)')",
   841	    )]
   842	    c += [md(
   843	        "## Confusion of the frozen S-JEPA run\n",
   844	        "Where does S-JEPA confuse the conditions across all held-out clips? The dominant "
   845	        "error is PD read as MS, the same failure the Random Forest also struggles with here.\n",
   846	    )]
   847	    c += [code(
   848	        "import numpy as np, matplotlib.pyplot as plt, seaborn as sns",
   849	        "if frozen_path.exists():",
   850	        "    fig, ax = plt.subplots(1, 2, figsize=(10,4))",
   851	        "    for a, key, title in [(ax[0], 'rf_pooled', 'Random Forest (pooled OOF)'),",
   852	        "                          (ax[1], 'sjepa_pooled', 'S-JEPA (pooled OOF)')]:",
   853	        "        cm = np.array(frozen[key]['confusion'])",
   854	        "        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,",
   855	        "                    xticklabels=LABELS, yticklabels=LABELS, ax=a)",
   856	        "        a.set_title(title); a.set_xlabel('predicted'); a.set_ylabel('true')",
   857	        "    plt.tight_layout(); plt.show()",
   858	        "else:",
   859	        "    print('Frozen run not found; run scripts/scripts_r1_repaired.py first.')",
   860	    )]
   861	    c += [md(
   862	        "## A combined scoreboard\n",
   863	        "One table, every system on the identical g1 folds. This is the whole comparison in one place.\n",
   864	    )]
   865	    c += [code(
   866	        "import pandas as pd",
   867	        "rows = []",
   868	        "if frozen_path.exists():",
   869	        "    rows.append(('Random Forest (paired)', frozen['rf_pooled']['macro_f1']))",
   870	        "    rows.append(('S-JEPA (R1, 1k updates)', frozen['sjepa_pooled']['macro_f1']))",
   871	        "if e0_path.exists():",
   872	        "    for name, res in e0['shortcut_controls'].items():",
   873	        "        best = max(res['logreg']['macro_f1']['mean'], res['rf']['macro_f1']['mean'])",
   874	        "        rows.append((f'control: {name}', best))",
   875	        "rows.append(('chance (3 classes)', 1/3))",
   876	        "board = pd.DataFrame(rows, columns=['system', 'macro_F1']).sort_values('macro_F1', ascending=False)",
   877	        "display(board.reset_index(drop=True))",
   878	        "results = {'frozen_run': str(frozen_path.relative_to(ARTIFACT_DIR)) if frozen_path.exists() else None,",
   879	        "           'scoreboard': {n: float(v) for n, v in rows}}",
   880	        "(ARTIFACT_DIR / 'capstone_results.json').write_text(json.dumps(results, indent=2))",
   881	        "print('saved capstone_results.json')",
   882	    )]
   883	    c += [md(
   884	        "## What this does and does not show\n",
   885	        "**What we can say.** The comparison is fair by construction: RF, S-JEPA, and the controls all "
   886	        "sit on the identical locked folds, use per-clip pooled out-of-fold scoring, and fit their "
   887	        "scalers and heads on training data only. On this data the Random Forest is the strongest "
   888	        "system, and the label-free S-JEPA sits below it *and* below cheap nuisance controls.\n",
   889	        "**Why that is progress, not failure.** An earlier S-JEPA number looked higher partly because "
   890	        "it leaned on shortcuts (every MS clip was filmed at 60fps and square) and a label leak in the "
   891	        "objective. Removing those lowered the honest score. The representation did not collapse "
   892	        "(effective rank stays well above 1 on every fold), so this is a real, non-degenerate estimate, "
   893	        "not a broken run. Per the pre-registered rule, a mechanically valid model that does not clear "
   894	        "the bar tells us to **stop scaling the local network** and fix the binding constraint instead.\n",
   895	        "**What we cannot say.** With ~47 videos from ~35 sources that are not verified people, this is "
   896	        "a provisional, source-grouped **development estimate**, never a clinical result. No diagnostic, "
   897	        "validity, or deployment claim is warranted.\n",
   898	        "**The honest next step.** The evidence points at the data pipeline and the acquisition domain, "
   899	        "not model size: rebuild the lineage (true common frame rate, speed-preserving normalization, "
   900	        "validity masks, a domain de-confound) and bring in external clinical-motion pretraining, then "

exec
/bin/zsh -lc "nl -ba scripts/notebook_content.py | sed -n '429,590p'; printf '\\n-- artifacts frozen presence --\\n'; find artifacts/eval artifacts/runs -maxdepth 4 -type f -print 2>&1 | sort | head -100; printf '\\n-- JSON keys --\\n'; .venv/bin/python - <<'PY'
import json, pathlib
for p in ['artifacts/eval/g1/E0_results.json','artifacts/runs/r1_g1_1k_s42/results.json','artifacts/eval/g1/fold_registry.json']:
 q=pathlib.Path(p); print(p,q.exists());
 if q.exists():
  d=json.load(open(q)); print(d.keys())
  if 'folds' in d: print(d['folds'][0].keys(),d['folds'][0])
PY
printf '\\n-- API source --\\n'; grep -R \"def load_checkpoint_v2\\|def save_checkpoint_v2\\|def train_sjepa_v2\\|def sample_target_mask\\|def build_feature_matrix\" -n sjepa" in /Users/pmui/dev/alexpose/experiments/multiple-sclerosis
 succeeded in 0ms:
   429	def nb_04(md, code, badge, boot):
   430	    c = [badge("04_progressive_finetune_ms_pd_vicreg.ipynb")]
   431	    c += [md(
   432	        "# 04 - Two ways to adapt the encoder: SSL continuation vs supervised adaptation\n",
   433	        "Notebook 03 trained the encoder with **no labels** on all the walking motion in the training "
   434	        "set. Now we ask a sharper question: once we do have diagnosis labels, what is the honest way "
   435	        "to use them, and does it actually help the three conditions separate?\n",
   436	        "We compare two clearly named regimes, both starting from the same label-free checkpoint:\n",
   437	        "1. **SSL continuation** - keep training with the *same* label-free objective for more updates. "
   438	        "No labels touch the model.\n"
   439	        "2. **Balanced supervised adaptation** - freeze the encoder and fit a small, class-balanced "
   440	        "linear head on top. Labels are used *only* in this head, never inside the self-supervised "
   441	        "objective.\n",
   442	        "> **What changed, and why.** An earlier version of this notebook mixed the diagnosis label "
   443	        "*into* the self-supervised loss (a 'class-aware VICReg' that rewarded within-class spread) and "
   444	        "claimed it 'compacts the classes'. That was a leak: the SSL objective must not see labels, and "
   445	        "the claim was not supported. We removed it. If labels help, they help in an explicitly "
   446	        "supervised stage that we name and measure, not smuggled into the pretext task.\n",
   447	    )]
   448	    c += boot(need_torch=True)
   449	    c += [code(
   450	        "from IPython.display import SVG, display",
   451	        "display(SVG(filename=str(IMAGES_DIR / 'progressive_timeline.svg')))",
   452	    )]
   453	    c += [md(
   454	        "## Use the locked, leakage-safe fold registry\n",
   455	        "The comparison is only meaningful on a split where clips from one source never straddle "
   456	        "train and test. Notebook's Phase 0 froze such a split to `artifacts/eval/g1/fold_registry.json` "
   457	        "(source-grouped, seed 42). We load fold 0 from it here rather than inventing a fresh split, so "
   458	        "this notebook, notebook 05, and notebook 06 all sit on the identical partition.\n",
   459	        "> Source grouping is **provisional**: a `source_id` is a YouTube id, not a verified person, so "
   460	        "everything here is a development estimate, never a clinical claim.\n",
   461	    )]
   462	    c += [code(
   463	        "import json",
   464	        "from sjepa.data import load_index",
   465	        "",
   466	        "records = load_index(KEYPOINTS_DIR)",
   467	        "by_clip = {r.clip_name: r for r in records}",
   468	        "registry = json.loads((ARTIFACT_DIR / 'eval' / 'g1' / 'fold_registry.json').read_text())",
   469	        "fold0 = registry['folds'][0]",
   470	        "train_recs = [by_clip[c] for c in fold0['train_clips']]",
   471	        "test_recs  = [by_clip[c] for c in fold0['test_clips']]",
   472	        "tr_src = {r.source_id for r in train_recs}; te_src = {r.source_id for r in test_recs}",
   473	        "assert not (tr_src & te_src), 'source leakage across the fold'",
   474	        "print('fold 0:', len(train_recs), 'train videos /', len(test_recs), 'test videos')",
   475	        "print('no source in both sides:', not (tr_src & te_src))",
   476	    )]
   477	    c += [md(
   478	        "## Regime 1 - SSL continuation (no labels)\n",
   479	        "We rebuild the model, load the label-free checkpoint from notebook 03, and keep "
   480	        "training with the exact same objective for a few hundred more updates. The label is never "
   481	        "read. This asks: does more unlabeled training alone sharpen the representation?\n",
   482	    )]
   483	    c += [code(
   484	        "from sjepa.config import get_config",
   485	        "from sjepa.models import build_model, pick_device",
   486	        "from sjepa.train_v2 import train_sjepa_v2, load_checkpoint_v2, save_checkpoint_v2",
   487	        "from sjepa.data import SequenceWindowDataset",
   488	        "",
   489	        "cfg = get_config()",
   490	        "device = pick_device()",
   491	        "model = build_model(cfg, device=device, repaired=True)",
   492	        "load_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl.pt', model, map_location=device)",
   493	        "",
   494	        "ds_train = SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride)",
   495	        "MORE = 40 if __import__('os').environ.get('SJEPA_SMOKE') else 400",
   496	        "state = train_sjepa_v2(model, ds_train, cfg, total_updates=MORE, device=device,",
   497	        "                       mask_ratio=0.6, log_every=max(1, MORE // 4))",
   498	        "save_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl_continued.pt', model, cfg, train_state=state,",
   499	        "                   extra={'stage': 'ssl_continuation_fold0'})",
   500	        "print('continued SSL: final effective rank', round(state.eff_rank[-1], 1))",
   501	    )]
   502	    c += [md(
   503	        "## A fixed, label-free read-out\n",
   504	        "To turn a video into one vector we mean-pool the frozen target encoder over a **fixed** pool "
   505	        "of target tokens. The pool is chosen once from a seeded RNG and never from the test labels, so "
   506	        "no information leaks from the evaluation into the representation. Both regimes below use this "
   507	        "same read-out.\n",
   508	    )]
   509	    c += [code(
   510	        "import numpy as np, torch",
   511	        "from sjepa.masking_v2 import sample_target_mask",
   512	        "from sjepa.data import sliding_windows",
   513	        "",
   514	        "readout = sample_target_mask(cfg.num_joints, cfg.num_time_tokens,",
   515	        "                             np.random.default_rng(0), target_ratio=0.6)",
   516	        "tm = torch.from_numpy(readout).to(device)",
   517	        "",
   518	        "def embed_records(m, recs):",
   519	        "    V, Y = [], []",
   520	        "    for r in recs:",
   521	        "        w = sliding_windows(r.load_norm(), cfg.window_frames, cfg.window_stride)",
   522	        "        x = torch.from_numpy(w).float().to(device)",
   523	        "        with torch.no_grad():",
   524	        "            V.append(m.embed(x, tm).mean(0).cpu().numpy())",
   525	        "        Y.append(r.label)",
   526	        "    return np.stack(V), Y",
   527	    )]
   528	    c += [md(
   529	        "## Regime 2 - balanced supervised adaptation (labels only in the head)\n",
   530	        "Now we use the labels honestly: freeze the encoder and fit a class-balanced logistic head on "
   531	        "the training embeddings, then score the held-out videos. The scaler and the head are fit on "
   532	        "**training data only**. We do this on top of both the notebook-03 checkpoint and the "
   533	        "SSL-continued one, so we can see whether extra unlabeled training moved the probe at all.\n",
   534	    )]
   535	    c += [code(
   536	        "from sklearn.linear_model import LogisticRegression",
   537	        "from sklearn.preprocessing import StandardScaler",
   538	        "from sjepa.eval import evaluate",
   539	        "LABELS = ['normal', 'ms', 'pd']",
   540	        "",
   541	        "def probe_and_score(ckpt):",
   542	        "    m = build_model(cfg, device=device, repaired=True)",
   543	        "    load_checkpoint_v2(ckpt, m, map_location=device)",
   544	        "    Etr, ytr = embed_records(m, train_recs)",
   545	        "    Ete, yte = embed_records(m, test_recs)",
   546	        "    sc = StandardScaler().fit(Etr)               # fit on TRAIN only",
   547	        "    clf = LogisticRegression(max_iter=2000, class_weight='balanced')",
   548	        "    clf.fit(sc.transform(Etr), ytr)",
   549	        "    return evaluate(yte, clf.predict(sc.transform(Ete)), LABELS)",
   550	        "",
   551	        "m_base = probe_and_score(ARTIFACT_DIR / 'sjepa_ssl.pt')",
   552	        "m_cont = probe_and_score(ARTIFACT_DIR / 'sjepa_ssl_continued.pt')",
   553	        "print(f'supervised probe on fold 0 held-out videos (macro-F1):')",
   554	        "print(f'  notebook-03 checkpoint : {m_base.macro_f1:.3f}')",
   555	        "print(f'  after SSL continuation : {m_cont.macro_f1:.3f}')",
   556	    )]
   557	    c += [md(
   558	        "## Read this honestly\n",
   559	        "On this tiny fold the two numbers are close and noisy; do not over-read a few points either "
   560	        "way. The point of the notebook is the **method**: labels live only in the supervised head, the "
   561	        "SSL objective stays label-free, and the split is the locked, leakage-safe one. Notebook 06 "
   562	        "runs this over all folds and puts it beside the Random Forest and the shortcut controls, which "
   563	        "is where any real verdict lives. A single fold here proves nothing on its own.\n",
   564	    )]
   565	    return c
   566	
   567	
   568	def nb_05(md, code, badge, boot):
   569	    c = [badge("05_representation_visualization.ipynb")]
   570	    c += [md(
   571	        "# 05 - Looking at the learned representation (diagnostics only)\n",
   572	        "We have a trained encoder. What did it actually learn? Here we turn each video into a single "
   573	        "feature vector with the frozen target encoder and project those vectors to two dimensions with "
   574	        "t-SNE and UMAP, to *see* whether normal, ms, and pd land in different regions.\n",
   575	        "> **These pictures are diagnostics, not evidence.** Two honest cautions run through this "
   576	        "notebook. First, t-SNE and UMAP distort distances; a clean-looking blob can be an artifact of "
   577	        "the projection. Second, and more important, apparent separation can come from a **shortcut** "
   578	        "(camera frame rate, body size, how visible the joints are) rather than from gait. So we plot "
   579	        "the S-JEPA embedding *and* a cheap nuisance feature side by side: if the nuisance separates "
   580	        "just as well, the pretty S-JEPA plot is not telling us about gait. The verdict lives in "
   581	        "notebook 06's leakage-safe scores, never in a scatter plot.\n",
   582	        "We compare the label-free checkpoint from notebook 03 against the SSL-continued one from "
   583	        "notebook 04. No test labels are ever used to fit, select, or color anything beyond the plain "
   584	        "class of each point.\n",
   585	    )]
   586	    c += boot(need_torch=True)
   587	    c += [code(
   588	        "from IPython.display import SVG, display",
   589	        "display(SVG(filename=str(IMAGES_DIR / 'vicreg_clusters.svg')))",
   590	    )]

-- artifacts frozen presence --
artifacts/eval/g1/COMPLETED.json
artifacts/eval/g1/E0_RF_oof.json
artifacts/eval/g1/E0_results.json
artifacts/eval/g1/fold_registry.json
artifacts/eval/g1/run_manifest.json
artifacts/runs/r1_g1_1k_s42/COMPLETED.json
artifacts/runs/r1_g1_1k_s42/oof.json
artifacts/runs/r1_g1_1k_s42/results.json
artifacts/runs/r1_g1_1k_s42/run_manifest.json
artifacts/runs/r1_g1_1k_s42/sjepa_fold0.pt
artifacts/runs/r1_g1_1k_s42/sjepa_fold1.pt
artifacts/runs/r1_g1_1k_s42/sjepa_fold2.pt
artifacts/runs/r1_g1_1k_s42/sjepa_fold3.pt
artifacts/runs/r1_g1_1k_s42/sjepa_fold4.pt

-- JSON keys --
artifacts/eval/g1/E0_results.json True
dict_keys(['generation', 'labels_order', 'E0_RF', 'shortcut_controls'])
artifacts/runs/r1_g1_1k_s42/results.json True
dict_keys(['run', 'generation', 'note', 'folds_run', 'n_folds_total', 'total_updates', 'mask_ratio', 'seed', 'sjepa_pooled', 'rf_pooled', 'diagnostics', 'wall_seconds'])
artifacts/eval/g1/fold_registry.json True
dict_keys(['generation', 'grouping', 'splitter', 'n_splits', 'seed', 'labels_order', 'folds'])
dict_keys(['fold', 'train_clips', 'test_clips', 'train_sources', 'test_sources', 'test_labels']) {'fold': 0, 'train_clips': ['-0wbleNgAwg', '-oJM2wUUjws', '0ecDhEQFrx4', '3FXUw98rrUY', '3T0BfK9HOzU', '6iCUKf9xgJg', 'B5hrxKe2nP8', 'DfRhvdCiUJk', 'EHymg4AGMJs_clip-01', 'EHymg4AGMJs_clip-02', 'FTHc-TJOQ34', 'Ivxdl6r2z_o', 'L-41u-0tsFo', 'M-_cogKwXK4_clip-01', 'M-_cogKwXK4_clip-02', 'PAeh4qBwsUk', 'W35NeWDslAE', 'WpARylM4UYU', 'XIPXYpWuIX4', '_-Ubl8iD2B0', '_Wn9oYGpRdM_clip-01', '_Wn9oYGpRdM_clip-02', '_Wn9oYGpRdM_clip-03', 'b0KrA_96Ks0', 'bmi1hYOnTHs_clip-01', 'bmi1hYOnTHs_clip-02', 'bmi1hYOnTHs_clip-03', 'eCCYhDSDlDc', 'gp4H7Z2Vvn0_clip-01', 'gp4H7Z2Vvn0_clip-02', 'n-O8dHyYIF0', 'nXuJIs25z1U_clip-01', 'nXuJIs25z1U_clip-02', 'tUT8Fh1zGKA', 'tqEmrPDPIsU', 'v1SoZ_S31pk', 'zOxtPrKySB8'], 'test_clips': ['0NfZp2hmni4', 'JD1AGVpftps', 'MN4vnaNwIsA', 'VL0AOiZt_lg', 'WvoNYV6nZtM', 'pFLC9C-xH8E_clip-01', 'pFLC9C-xH8E_clip-02', 'pFLC9C-xH8E_clip-03', 'pFLC9C-xH8E_clip-04', 'pFLC9C-xH8E_clip-05'], 'train_sources': ['-0wbleNgAwg', '-oJM2wUUjws', '0ecDhEQFrx4', '3FXUw98rrUY', '3T0BfK9HOzU', '6iCUKf9xgJg', 'B5hrxKe2nP8', 'DfRhvdCiUJk', 'EHymg4AGMJs', 'FTHc-TJOQ34', 'Ivxdl6r2z_o', 'L-41u-0tsFo', 'M-_cogKwXK4', 'PAeh4qBwsUk', 'W35NeWDslAE', 'WpARylM4UYU', 'XIPXYpWuIX4', '_-Ubl8iD2B0', '_Wn9oYGpRdM', 'b0KrA_96Ks0', 'bmi1hYOnTHs', 'eCCYhDSDlDc', 'gp4H7Z2Vvn0', 'n-O8dHyYIF0', 'nXuJIs25z1U', 'tUT8Fh1zGKA', 'tqEmrPDPIsU', 'v1SoZ_S31pk', 'zOxtPrKySB8'], 'test_sources': ['0NfZp2hmni4', 'JD1AGVpftps', 'MN4vnaNwIsA', 'VL0AOiZt_lg', 'WvoNYV6nZtM', 'pFLC9C-xH8E'], 'test_labels': {'0NfZp2hmni4': 'ms', 'WvoNYV6nZtM': 'ms', 'JD1AGVpftps': 'normal', 'MN4vnaNwIsA': 'normal', 'VL0AOiZt_lg': 'normal', 'pFLC9C-xH8E_clip-01': 'pd', 'pFLC9C-xH8E_clip-02': 'pd', 'pFLC9C-xH8E_clip-03': 'pd', 'pFLC9C-xH8E_clip-04': 'pd', 'pFLC9C-xH8E_clip-05': 'pd'}}

-- API source --
sjepa/masking_v2.py:95:def sample_target_mask(
sjepa/classical.py:54:def build_feature_matrix(records: Sequence[SequenceRecord], fps: int = 15
sjepa/train_v2.py:112:def train_sjepa_v2(
sjepa/train_v2.py:312:def save_checkpoint_v2(path, model: SJEPA, cfg: SJEPAConfig,
sjepa/train_v2.py:330:def load_checkpoint_v2(path, model: SJEPA, map_location: Optional[str] = None) -> Dict:

exec
/bin/zsh -lc "sed -n '1,220p' scripts/scripts_build_notebooks.py; sed -n '100,380p' sjepa/train_v2.py; sed -n '1,180p' sjepa/models.py; printf '\\n-- Extract and compile changed notebook code cells --\\n'; .venv/bin/python - <<'PY'
import ast,json,glob
for f in glob.glob('*.ipynb'):
 d=json.load(open(f)); errs=[]
 for i,c in enumerate(d['cells']):
  if c['cell_type']=='code':
   try: compile(''.join(c['source']),f'{f}:{i}','exec')
   except Exception as e: errs.append((i,e))
 print(f,errs)
PY
printf '\\n-- tests --\\n'; .venv/bin/pytest -q" in /Users/pmui/dev/alexpose/experiments/multiple-sclerosis
 exited 4 in 1778ms:
ERROR: Unknown config option: hypothesis_profiles




exec
/bin/zsh -lc "sed -n '1,180p' sjepa/masking_v2.py; grep -n \"def embed\\|def forward_repaired\" -A60 sjepa/models.py; sed -n '1,110p' sjepa/data.py; cat pyproject.toml; printf '\\n-- git diff name/status including top-level --\\n'; git status --short --untracked-files=all" in /Users/pmui/dev/alexpose/experiments/multiple-sclerosis
 succeeded in 0ms:
"""Repaired masking for S-JEPA: per-example stochastic graph-time masks.

This replaces the fixed 12-joint mask (see ``masking.py``, kept for the E0
reference) with the design the audit and literature call for:

* **Per-example** masks shaped ``(B, N)`` rather than one ``(N,)`` pattern reused
  across the whole batch.
* **Stochastic graph-time regions**: each masked region is a connected group of
  joints (a limb or the trunk) over a contiguous span of time blocks, so the
  mask has spatial and temporal structure instead of hiding whole joints for all
  time.
* **Every joint rotates** between context and target across steps, so no joint is
  starved of context gradient (the D2 defect).
* **Clinical target bias, not motion bias.** The lower-body and shoulder joints
  are *sampled as targets a little more often* (default 1.5x). We deliberately do
  NOT bias toward high-motion regions: reduced motion (hypokinesia, short steps)
  is exactly the clinical signal in MS/PD, so a high-motion mask would hide the
  evidence (MAMP's motion-aware masking is contraindicated here).
* **Full context coverage guaranteed**: at least one lower-body / contralateral
  cue is kept visible, and the target fraction is bounded so context is never
  empty.

Tokens are laid out as ``token index = t * V + v`` (time block ``t``, joint
``v``), matching the tokenizer. Functions are pure numpy and take an explicit
``numpy.random.Generator`` so masks are deterministic under a seed and diverse
across examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import ANATOMICAL_MASK_IDX

# BlazePose-33 connected joint groups (graph regions). Each region is a connected
# set of joints; a mask grows from one region over a contiguous time span. Every
# joint belongs to at least one region, and the face is split so its joints can
# also be targets (the coverage gate requires *every* joint to be targetable).
JOINT_GROUPS: Dict[str, Tuple[int, ...]] = {
    "head": (0, 1, 2, 3, 4, 5, 6, 7, 8),
    "mouth": (9, 10),
    "left_arm": (11, 13, 15, 17, 19, 21),
    "right_arm": (12, 14, 16, 18, 20, 22),
    "trunk": (11, 12, 23, 24),
    "left_leg": (23, 25, 27, 29, 31),
    "right_leg": (24, 26, 28, 30, 32),
}

# Clinically relevant joints (both shoulders + both legs) get a target-sampling
# boost. This is a *bias*, not a permanent hide.
CLINICAL_JOINTS = frozenset(ANATOMICAL_MASK_IDX)


@dataclass
class MaskBankStats:
    """Coverage accounting over a bank of sampled masks (for the gates/tests)."""

    joint_visible_frac: np.ndarray   # (V,) fraction of masks where joint is context
    joint_target_frac: np.ndarray    # (V,) fraction of masks where joint is target
    mean_target_frac: float          # mean fraction of tokens masked
    n_masks: int


def _region_joint_pool() -> List[Tuple[str, Tuple[int, ...]]]:
    """All connected regions we grow masks from (every joint is targetable)."""
    return list(JOINT_GROUPS.items())


def _region_weights(regions, clinical_bias: float) -> np.ndarray:
    """Region-selection weights.

    Joints that appear in several regions (e.g. shoulders 11/12 and hips 23/24 in
    the trunk) would otherwise be over-targeted, so we down-weight each region by
    the average multiplicity of its joints. Clinical regions get a mild boost. The
    net effect keeps per-joint target frequency roughly balanced while still
    favouring the clinically relevant lower body a little.
    """
    # How many regions each joint belongs to.
    mult = {}
    for _, joints in regions:
        for j in joints:
            mult[j] = mult.get(j, 0) + 1
    w = []
    for _, joints in regions:
        avg_mult = np.mean([mult[j] for j in joints])
        boost = clinical_bias if any(j in CLINICAL_JOINTS for j in joints) else 1.0
        w.append(boost / avg_mult)
    w = np.asarray(w, dtype=float)
    return w / w.sum()


def sample_target_mask(
    num_joints: int,
    num_time_tokens: int,
    rng: np.random.Generator,
    target_ratio: float = 0.6,
    clinical_bias: float = 1.5,
    max_time_span_frac: float = 0.75,
) -> np.ndarray:
    """Sample ONE per-example target mask, shape (num_tokens,), True = target.

    We grow connected graph-time regions until roughly ``target_ratio`` of the
    tokens are targeted, then guarantee a non-empty context by clearing targets
    if we overshot. Clinically relevant joints are chosen more often via
    ``clinical_bias``. Time spans are contiguous and bounded by
    ``max_time_span_frac`` of the window so masks keep temporal structure.
    """
    V, T = num_joints, num_time_tokens
    N = V * T
    target = np.zeros((T, V), dtype=bool)

    regions = _region_joint_pool()
    weights = _region_weights(regions, clinical_bias)

    max_span = max(1, int(round(max_time_span_frac * T)))
    budget = int(round(target_ratio * N))
    guard = 0
    while target.sum() < budget and guard < 4 * len(regions) + 8:
        guard += 1
        ri = rng.choice(len(regions), p=weights)
        _, joints = regions[ri]
        # contiguous temporal span
        span = int(rng.integers(1, max_span + 1))
        start = int(rng.integers(0, max(1, T - span + 1)))
        for t in range(start, min(T, start + span)):
            for j in joints:
                target[t, j] = True

    flat = target.reshape(-1)
    # Guarantee non-empty context: if we masked everything, free a random block.
    if flat.all():
        flat[rng.integers(0, N)] = False
    # Guarantee at least one target (degenerate tiny configs).
    if not flat.any():
        flat[rng.integers(0, N)] = True
    return flat


def sample_mask_batch(
    batch_size: int,
    num_joints: int,
    num_time_tokens: int,
    rng: np.random.Generator,
    target_ratio: float = 0.6,
    masks_per_sequence: int = 1,
    clinical_bias: float = 1.5,
) -> np.ndarray:
    """Return a per-example target-mask batch of shape (B * masks, num_tokens).

    When ``masks_per_sequence > 1`` the batch is expanded so each sequence
    contributes several independent masks (the (B, M, N) idea, flattened to
    (B*M, N) for vectorised training). ``context = ~target``.
    """
    rows = []
    for _ in range(batch_size):
        for _ in range(masks_per_sequence):
            rows.append(sample_target_mask(
                num_joints, num_time_tokens, rng,
                target_ratio=target_ratio, clinical_bias=clinical_bias))
    return np.stack(rows, axis=0)


def mask_bank_stats(
    num_joints: int,
    num_time_tokens: int,
    n_masks: int = 512,
    seed: int = 0,
    target_ratio: float = 0.6,
    clinical_bias: float = 1.5,
) -> MaskBankStats:
    """Sample a bank of masks and measure per-joint context/target coverage.

    Used by the promotion-gate tests: every joint must be visible in some masks
    and targeted in others.
    """
    rng = np.random.default_rng(seed)
    V, T = num_joints, num_time_tokens
282:    def forward_repaired(self, x_view: torch.Tensor, x_full: torch.Tensor,
283-                         context_mask: torch.Tensor):
284-        """Repaired two-lane forward with PER-EXAMPLE (B, N) context masks.
285-
286-        Uses :meth:`SJEPAEncoder.forward_context_per_example` (key-padding
287-        attention so every joint can be context) and :class:`PredictorV2` (target
288-        joint/time position identity). The target lane still masks the target
289-        encoder's OUTPUT by the caller, which the paper shows is essential.
290-        Returns ``(predicted (B, N, dim), target (B, N, dim))``.
291-        """
292-        view_features = self.view_encoder.forward_context_per_example(
293-            x_view, context_mask, self.mask_token)
294-        predicted = self.predictor(view_features)
295-        with torch.no_grad():
296-            target = self.target_encoder(x_full)
297-        return predicted, target
298-
299-    @torch.no_grad()
300:    def embed(self, x_full: torch.Tensor, target_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
301-        """Downstream representation: masked mean-pool of target-encoder tokens.
302-
303-        By default we pool over the masked (neurologically relevant) tokens, since
304-        those are the joints the model was trained to reason about. Pass a
305-        different mask to pool over other tokens.
306-        """
307-        self.target_encoder.eval()
308-        feats = self.target_encoder(x_full)             # (B, N, dim)
309-        if target_mask is None:
310-            return feats.mean(dim=1)
311-        idx = torch.nonzero(target_mask, as_tuple=False).squeeze(1)
312-        return feats[:, idx, :].mean(dim=1)
313-
314-
315-def build_model(cfg: SJEPAConfig, device: Optional[str] = None,
316-                repaired: bool = False) -> SJEPA:
317-    """Construct the model, seed it, and move it to the chosen device.
318-
319-    ``repaired=True`` builds the corrected model (PredictorV2 with target-position
320-    identity + per-example mask support). ``repaired=False`` keeps the E0 model.
321-    """
322-    torch.manual_seed(cfg.seed)
323-    np.random.seed(cfg.seed)
324-    model = SJEPA(cfg, repaired=repaired)
325-    if device is None:
326-        device = pick_device()
327-    return model.to(device)
328-
329-
330-def pick_device() -> str:
331-    """Prefer CUDA, then Apple MPS, then CPU."""
332-    if torch.cuda.is_available():
333-        return "cuda"
334-    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
335-        return "mps"
336-    return "cpu"
"""Data handling for the S-JEPA gait tutorials.

This module turns raw walking videos into skeleton sequences and then into the
fixed length windows the model trains on. It reuses the existing ``ambient`` pose
stack for the heavy lifting, so we never re-implement MediaPipe.

The flow is:

    mp4 file
      -> cv2 frame loop, sampled to target_fps
      -> ambient SequenceKeypointExtractor.extract_from_image  (33 landmarks)
      -> array of shape (T, 33, 3) with columns [x_pixels, y_pixels, visibility]
      -> normalise (root centre on the pelvis, scale by torso length)
      -> cache to a .npz file, one per video

There are no GAVD CSVs here. The GAVD pipeline needs per-frame bounding boxes and
YouTube URLs; our videos are already downloaded clips, so we just walk the folders
and run pose on whole frames.

Splitting is done by *source video id* (the YouTube id in manifest.csv), never by
clip, so clips cut from the same source never land on both sides of a split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Pelvis and shoulder landmark indices used for normalisation.
_LEFT_HIP, _RIGHT_HIP = 23, 24
_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12

_CLIP_SUFFIX = re.compile(r"_clip-\d+$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Pose extraction from raw video
# ---------------------------------------------------------------------------

def load_video_sequence(
    video_path: str | Path,
    target_fps: int = 15,
    max_frames: Optional[int] = None,
    extractor=None,
    verbose: bool = False,
) -> np.ndarray:
    """Run MediaPipe over a video and return a (T, 33, 3) array.

    Columns are ``[x_pixels, y_pixels, visibility]``. Frames with no detection
    become rows of NaN so :func:`clean_sequence` can interpolate them later.

    ``extractor`` may be a pre-built ``SequenceKeypointExtractor`` (reuse one
    across many videos to keep the MediaPipe singleton warm). If ``None`` we make
    one on first use.
    """
    import cv2

    if extractor is None:
        from ambient.pose.keypoint_extractor import SequenceKeypointExtractor

        extractor = SequenceKeypointExtractor()

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / float(target_fps))))

    rows: List[np.ndarray] = []
    frame_idx = 0
    kept = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                kps = extractor.extract_from_image(frame_rgb)
                rows.append(_keypointset_to_row(kps))
                kept += 1
                if max_frames is not None and kept >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()

    if not rows:
        return np.empty((0, 33, 3), dtype=np.float32)

    seq = np.stack(rows, axis=0).astype(np.float32)
    if verbose:
        valid = np.isfinite(seq[:, :, 0]).all(axis=1).mean() if len(seq) else 0.0
        print(f"  {video_path.name}: {len(seq)} frames @ ~{target_fps}fps, "
              f"{valid:.0%} fully detected")
    return seq


def _keypointset_to_row(kps) -> np.ndarray:
    """Convert an ambient KeypointSet (33 keypoints) into one (33, 3) row.

    An empty detection yields a row of NaNs. Keypoints carry pixel x, pixel y and
    a confidence value that is MediaPipe's visibility.
[project]
name = "ms-sjepa"
version = "0.1.0"
description = "Progressive S-JEPA tutorials for gait classification (normal / ms / pd)"
readme = "README.md"
# The repository targets Python 3.12 (see the root project's classifiers and
# tool configuration).  In particular, UMAP's numba/llvmlite stack does not
# currently support Python 3.14; leaving this open-ended lets uv select 3.14
# and backtrack to an ancient llvmlite release that cannot build there.
requires-python = ">=3.12,<3.13"
license = { text = "MIT" }

# These are the concrete libraries the notebooks and the sjepa package need.
# The Colab install cells pip-install this same set, so local and cloud match.
# torch is guarded in the notebooks so Colab's preinstalled GPU torch is kept.
dependencies = [
    # Pose estimation and video (already used by the alexpose ambient package)
    "mediapipe>=0.10.31",
    "opencv-python>=4.12.0.88",
    # Deep learning (new for this experiment; not in the repo root deps)
    "torch>=2.2.0",
    # Classical baseline and metrics
    "scikit-learn>=1.8.0",
    # Representation visualisation
    "umap-learn>=0.5.6",
    # Data, plotting, notebooks
    "numpy>=1.24.0",
    "pandas>=2.3.3",
    "pyarrow>=15.0.0",
    "matplotlib>=3.10.8",
    "seaborn>=0.12.0",
    "pillow>=10.0.0",
    "tqdm>=4.66.0",
    "ipython>=8.0.0",
    "ipykernel>=7.1.0",
    "python-dotenv>=1.2.1",
    # Local pose + classification pipeline used throughout the notebooks.
    "alexpose",
    "imageio-ffmpeg>=0.6.0",
]

# Reuse the ambient pose + classification code from the repo root. When you run
# `uv sync` from inside this folder, uv installs the root package in editable
# mode so `import ambient` works.
[tool.uv.sources]
alexpose = { path = "../..", editable = true }

[tool.uv]
# pynndescent's broad `numba>=0.51.2`/`llvmlite>=0.30` requirements allow uv
# to pair current NumPy with releases from 2021 whose Python compatibility was
# only enforced at build time. Keep resolution on the modern metadata branch.
constraint-dependencies = [
    "numba>=0.61.2",
    "llvmlite>=0.44.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["sjepa"]

-- git diff name/status including top-level --
 M 00_overview_and_video_gallery.ipynb
 M 01_pose_extraction_from_raw_video.ipynb
 M 02_anatomical_mask_and_tokenization.ipynb
 M 03_sjepa_model_and_pretrain_normal.ipynb
 M 04_progressive_finetune_ms_pd_vicreg.ipynb
 M 05_representation_visualization.ipynb
 M 06_capstone_rf_vs_sjepa.ipynb
 M docs/06-0803-FINAL_REPORT.md
 M scripts/notebook_content.py
?? artifacts/reviews/AR_uncommitted_codex_2026-08-04.md
?? docs/07-0803-METHODOLOGY_ROADMAP.md
?? slides/slides.html
?? slides/slides.pdf
?? ../../notes/ms/sjepa-ms-04-analysis.md

exec
/bin/zsh -lc "cat artifacts/reviews/AR_uncommitted_codex_2026-08-04.md; cat ../../notes/ms/sjepa-ms-04-analysis.md; head -80 docs/07-0803-METHODOLOGY_ROADMAP.md; grep -nE 'src=|href=' slides/slides.html | head -80; git diff --check; python scripts/scripts_build_notebooks.py --check" in /Users/pmui/dev/alexpose/experiments/multiple-sclerosis
 exited 1 in 5ms:
check 00_overview_and_video_gallery.ipynb: MATCH
check 01_pose_extraction_from_raw_video.ipynb: MATCH
check 02_anatomical_mask_and_tokenization.ipynb: MATCH
check 03_sjepa_model_and_pretrain_normal.ipynb: MATCH
check 04_progressive_finetune_ms_pd_vicreg.ipynb: MATCH
check 05_representation_visualization.ipynb: MATCH
check 06_capstone_rf_vs_sjepa.ipynb: DIFFERS

IDEMPOTENCE: SOME DIFFER

2026-08-04T20:08:25.403143Z ERROR codex_core::tools::router: error=exec_command failed for `/bin/zsh -lc "rm -rf /tmp/ms-nbs && python scripts/scripts_build_notebooks.py --output-dir /tmp/ms-nbs >/dev/null && diff -u /tmp/ms-nbs/06_capstone_rf_vs_sjepa.ipynb 06_capstone_rf_vs_sjepa.ipynb | sed -n '1,240p'"`: CreateProcess { message: "Rejected(\"`/bin/zsh -lc \\\"rm -rf /tmp/ms-nbs && python scripts/scripts_build_notebooks.py --output-dir /tmp/ms-nbs >/dev/null && diff -u /tmp/ms-nbs/06_capstone_rf_vs_sjepa.ipynb 06_capstone_rf_vs_sjepa.ipynb | sed -n '1,240p'\\\"` rejected: rm -f style commands are not permitted. Use a safer approach\")" }
exec
/bin/zsh -lc 'd=$(mktemp -d /tmp/ms-nbs.XXXXXX) && python scripts/scripts_build_notebooks.py --output-dir "$d" >/dev/null && diff -u "$d/06_capstone_rf_vs_sjepa.ipynb" 06_capstone_rf_vs_sjepa.ipynb | sed -n '"'1,260p'" in /Users/pmui/dev/alexpose/experiments/multiple-sclerosis
 succeeded in 0ms:
--- /tmp/ms-nbs.oZ1W29/06_capstone_rf_vs_sjepa.ipynb	2026-08-04 13:08:28
+++ 06_capstone_rf_vs_sjepa.ipynb	2026-08-03 21:43:26
@@ -2,14 +2,15 @@
  "cells": [
   {
    "cell_type": "markdown",
+   "id": "06_capstone_rf_vs_sjepa-00",
    "metadata": {},
    "source": [
     "<a href=\"https://colab.research.google.com/github/your-org/alexpose/blob/main/experiments/multiple-sclerosis/06_capstone_rf_vs_sjepa.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
-   ],
-   "id": "06_capstone_rf_vs_sjepa-00"
+   ]
   },
   {
    "cell_type": "markdown",
+   "id": "06_capstone_rf_vs_sjepa-01",
    "metadata": {},
    "source": [
     "# 06 - Capstone: Random Forest vs S-JEPA, on identical folds, with controls\n",
@@ -22,16 +23,23 @@
     "\n",
     "The headline is **pooled macro-F1** over the folds (one prediction per clip, gathered across all held-out folds), because averaging per-fold F1 on ~9 test videos is even noisier. We report it beside the paired RF and the controls, then say plainly what it does and does not mean.\n",
     "\n",
-    "> **Spoiler, stated up front.** On this tiny, already-inspected, source-grouped collection the S-JEPA scores *below* both the Random Forest and the nuisance controls. That is the expected, plan-anticipated outcome of removing the shortcuts and the label leak that inflated an earlier number, and it is reported as a negative result, not hidden.\n",
-    ""
-   ],
-   "id": "06_capstone_rf_vs_sjepa-01"
+    "> **Spoiler, stated up front.** On this tiny, already-inspected, source-grouped collection the S-JEPA scores *below* both the Random Forest and the nuisance controls. That is the expected, plan-anticipated outcome of removing the shortcuts and the label leak that inflated an earlier number, and it is reported as a negative result, not hidden.\n"
+   ]
   },
   {
    "cell_type": "code",
+   "execution_count": 11,
+   "id": "06_capstone_rf_vs_sjepa-02",
    "metadata": {},
-   "execution_count": null,
-   "outputs": [],
+   "outputs": [
+    {
+     "name": "stdout",
+     "output_type": "stream",
+     "text": [
+      "all light dependencies already present\n"
+     ]
+    }
+   ],
    "source": [
     "# --- Setup: install dependencies (Colab installs; local usually already has them) ---\n",
     "import importlib, importlib.util, subprocess, sys, os\n",
@@ -56,14 +64,23 @@
     "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *_pkgs])\n",
     "else:\n",
     "    print('all light dependencies already present')"
-   ],
-   "id": "06_capstone_rf_vs_sjepa-02"
+   ]
   },
   {
    "cell_type": "code",
+   "execution_count": 12,
+   "id": "06_capstone_rf_vs_sjepa-03",
    "metadata": {},
-   "execution_count": null,
-   "outputs": [],
+   "outputs": [
+    {
+     "name": "stdout",
+     "output_type": "stream",
+     "text": [
+      "experiment dir: /Users/pmui/dev/alexpose/experiments/multiple-sclerosis\n",
+      "repo root     : /Users/pmui/dev/alexpose\n"
+     ]
+    }
+   ],
    "source": [
     "# --- Make `sjepa` and `ambient` importable, locally and in Colab ---\n",
     "from pathlib import Path\n",
@@ -91,14 +108,22 @@
     "        sys.path.insert(0, p)\n",
     "print('experiment dir:', EXP_DIR)\n",
     "print('repo root     :', REPO_ROOT)"
-   ],
-   "id": "06_capstone_rf_vs_sjepa-03"
+   ]
   },
   {
    "cell_type": "code",
+   "execution_count": 13,
+   "id": "06_capstone_rf_vs_sjepa-04",
    "metadata": {},
-   "execution_count": null,
-   "outputs": [],
+   "outputs": [
+    {
+     "name": "stdout",
+     "output_type": "stream",
+     "text": [
+      "SJEPA_PROFILE = laptop | SJEPA_SMOKE = 0\n"
+     ]
+    }
+   ],
    "source": [
     "# --- Paths and profile (reads the root .env if python-dotenv is present) ---\n",
     "import os\n",
@@ -119,38 +144,208 @@
     "os.environ.setdefault('SJEPA_PROFILE', 'laptop')\n",
     "print('SJEPA_PROFILE =', os.environ['SJEPA_PROFILE'],\n",
     "      '| SJEPA_SMOKE =', os.environ.get('SJEPA_SMOKE', '0'))"
-   ],
-   "id": "06_capstone_rf_vs_sjepa-04"
+   ]
   },
   {
    "cell_type": "code",
+   "execution_count": 14,
+   "id": "06_capstone_rf_vs_sjepa-05",
    "metadata": {},
-   "execution_count": null,
-   "outputs": [],
+   "outputs": [
+    {
+     "data": {
+      "image/svg+xml": [
+       "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"400\" viewBox=\"0 0 900 400\">\n",
+       "<rect width=\"900\" height=\"400\" fill=\"#ffffff\"/>\n",
+       "<defs>\n",
+       "<marker id=\"arrow\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#94a3b8\"/></marker>\n",
+       "<marker id=\"arrowBlue\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#2563eb\"/></marker>\n",
+       "<marker id=\"arrowOrange\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#ea7317\"/></marker>\n",
+       "<marker id=\"arrowGreen\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#16a34a\"/></marker>\n",
+       "<marker id=\"arrowSlate\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#475569\"/></marker>\n",
+       "<marker id=\"arrowMute\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#64748b\"/></marker>\n",
+       "<marker id=\"arrowPurple\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#7c3aed\"/></marker>\n",
+       "</defs>\n",
+       "<text x=\"450.0\" y=\"34\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"21\" fill=\"#1a2733\" font-weight=\"700\" text-anchor=\"middle\">A fair head-to-head comparison</text>\n",
+       "<rect x=\"51.5\" y=\"112.5\" width=\"250\" height=\"74\" rx=\"14\" fill=\"#0f172a\" opacity=\"0.06\"/>\n",
+       "<rect x=\"50\" y=\"110\" width=\"250\" height=\"74\" rx=\"14\" fill=\"#fff7ed\" stroke=\"#fed7aa\" stroke-width=\"1.5\"/>\n",
+       "<text x=\"175.0\" y=\"143.0\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"14\" fill=\"#ea7317\" font-weight=\"700\" text-anchor=\"middle\">S-JEPA linear probe</text>\n",
+       "<text x=\"175.0\" y=\"162.0\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"12\" fill=\"#64748b\" font-weight=\"400\" text-anchor=\"middle\">frozen learned features</text>\n",
+       "<rect x=\"51.5\" y=\"238.5\" width=\"250\" height=\"74\" rx=\"14\" fill=\"#0f172a\" opacity=\"0.06\"/>\n",
+       "<rect x=\"50\" y=\"236\" width=\"250\" height=\"74\" rx=\"14\" fill=\"#f0fdf4\" stroke=\"#bbf7d0\" stroke-width=\"1.5\"/>\n",
+       "<text x=\"175.0\" y=\"269.0\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"14\" fill=\"#16a34a\" font-weight=\"700\" text-anchor=\"middle\">Random Forest</text>\n",
+       "<text x=\"175.0\" y=\"288.0\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"12\" fill=\"#64748b\" font-weight=\"400\" text-anchor=\"middle\">82 hand-made features</text>\n",
+       "<rect x=\"401.5\" y=\"175.5\" width=\"170\" height=\"74\" rx=\"14\" fill=\"#0f172a\" opacity=\"0.06\"/>\n",
+       "<rect x=\"400\" y=\"173\" width=\"170\" height=\"74\" rx=\"14\" fill=\"#f8fafc\" stroke=\"#e2e8f0\" stroke-width=\"1.5\"/>\n",
+       "<text x=\"485.0\" y=\"215.0\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"14\" fill=\"#475569\" font-weight=\"700\" text-anchor=\"middle\">Same test videos</text>\n",
+       "<rect x=\"641.5\" y=\"175.5\" width=\"200\" height=\"74\" rx=\"14\" fill=\"#0f172a\" opacity=\"0.06\"/>\n",
+       "<rect x=\"640\" y=\"173\" width=\"200\" height=\"74\" rx=\"14\" fill=\"#f8fafc\" stroke=\"#e2e8f0\" stroke-width=\"1.5\"/>\n",
+       "<text x=\"740.0\" y=\"206.0\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"13.5\" fill=\"#475569\" font-weight=\"700\" text-anchor=\"middle\">Accuracy, macro F1</text>\n",
+       "<text x=\"740.0\" y=\"225.0\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"12\" fill=\"#64748b\" font-weight=\"400\" text-anchor=\"middle\">mean ± std</text>\n",
+       "<path d=\"M 300 147 H 360 V 200 H 398\" fill=\"none\" stroke=\"#ea7317\" stroke-width=\"2.2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" marker-end=\"url(#arrowOrange)\"/>\n",
+       "<path d=\"M 300 273 H 360 V 220 H 398\" fill=\"none\" stroke=\"#16a34a\" stroke-width=\"2.2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" marker-end=\"url(#arrowGreen)\"/>\n",
+       "<path d=\"M 570 210 H 638\" fill=\"none\" stroke=\"#475569\" stroke-width=\"2.2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" marker-end=\"url(#arrowSlate)\"/>\n",
+       "<text x=\"450.0\" y=\"360\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"12\" fill=\"#64748b\" font-weight=\"400\" text-anchor=\"middle\">grouped k-fold over source videos, identical folds for both models</text>\n",
+       "</svg>"
+      ],
+      "text/plain": [
+       "<IPython.core.display.SVG object>"
+      ]
+     },
+     "metadata": {},
+     "output_type": "display_data"
+    },
+    {
+     "data": {
+      "image/svg+xml": [
+       "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"400\" viewBox=\"0 0 900 400\">\n",
+       "<rect width=\"900\" height=\"400\" fill=\"#ffffff\"/>\n",
+       "<defs>\n",
+       "<marker id=\"arrow\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#94a3b8\"/></marker>\n",
+       "<marker id=\"arrowBlue\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#2563eb\"/></marker>\n",
+       "<marker id=\"arrowOrange\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#ea7317\"/></marker>\n",
+       "<marker id=\"arrowGreen\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#16a34a\"/></marker>\n",
+       "<marker id=\"arrowSlate\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#475569\"/></marker>\n",
+       "<marker id=\"arrowMute\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#64748b\"/></marker>\n",
+       "<marker id=\"arrowPurple\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#7c3aed\"/></marker>\n",
+       "</defs>\n",
+       "<text x=\"450.0\" y=\"34\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"21\" fill=\"#1a2733\" font-weight=\"700\" text-anchor=\"middle\">Leakage-safe splitting by source video</text>\n",
+       "<text x=\"450.0\" y=\"57\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"13.5\" fill=\"#64748b\" font-weight=\"400\" text-anchor=\"middle\">All clips from one source stay on the same side of the split</text>\n",
+       "<text x=\"70\" y=\"150\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"14\" fill=\"#2563eb\" font-weight=\"700\" text-anchor=\"start\">Source A</text>\n",
+       "<rect x=\"70\" y=\"160\" width=\"66\" height=\"46\" rx=\"8\" fill=\"#eff6ff\" stroke=\"#bfdbfe\" stroke-width=\"1.4\"/>\n",
+       "<text x=\"103\" y=\"188\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"11\" fill=\"#2563eb\" font-weight=\"400\" text-anchor=\"middle\">clip 1</text>\n",
+       "<rect x=\"144\" y=\"160\" width=\"66\" height=\"46\" rx=\"8\" fill=\"#eff6ff\" stroke=\"#bfdbfe\" stroke-width=\"1.4\"/>\n",
+       "<text x=\"177\" y=\"188\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"11\" fill=\"#2563eb\" font-weight=\"400\" text-anchor=\"middle\">clip 2</text>\n",
+       "<rect x=\"218\" y=\"160\" width=\"66\" height=\"46\" rx=\"8\" fill=\"#eff6ff\" stroke=\"#bfdbfe\" stroke-width=\"1.4\"/>\n",
+       "<text x=\"251\" y=\"188\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"11\" fill=\"#2563eb\" font-weight=\"400\" text-anchor=\"middle\">clip 3</text>\n",
+       "<text x=\"70\" y=\"270\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"14\" fill=\"#ea7317\" font-weight=\"700\" text-anchor=\"start\">Source B</text>\n",
+       "<rect x=\"70\" y=\"280\" width=\"66\" height=\"46\" rx=\"8\" fill=\"#fff7ed\" stroke=\"#fed7aa\" stroke-width=\"1.4\"/>\n",
+       "<text x=\"103\" y=\"308\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"11\" fill=\"#ea7317\" font-weight=\"400\" text-anchor=\"middle\">clip 1</text>\n",
+       "<rect x=\"144\" y=\"280\" width=\"66\" height=\"46\" rx=\"8\" fill=\"#fff7ed\" stroke=\"#fed7aa\" stroke-width=\"1.4\"/>\n",
+       "<text x=\"177\" y=\"308\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"11\" fill=\"#ea7317\" font-weight=\"400\" text-anchor=\"middle\">clip 2</text>\n",
+       "<rect x=\"641.5\" y=\"152.5\" width=\"150\" height=\"66\" rx=\"14\" fill=\"#0f172a\" opacity=\"0.06\"/>\n",
+       "<rect x=\"640\" y=\"150\" width=\"150\" height=\"66\" rx=\"14\" fill=\"#ecfeff\" stroke=\"#a5f3fc\" stroke-width=\"1.5\"/>\n",
+       "<text x=\"715.0\" y=\"188.0\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"14.5\" fill=\"#0e7490\" font-weight=\"700\" text-anchor=\"middle\">TRAIN</text>\n",
+       "<rect x=\"641.5\" y=\"270.5\" width=\"150\" height=\"66\" rx=\"14\" fill=\"#0f172a\" opacity=\"0.06\"/>\n",
+       "<rect x=\"640\" y=\"268\" width=\"150\" height=\"66\" rx=\"14\" fill=\"#fffbeb\" stroke=\"#fde68a\" stroke-width=\"1.5\"/>\n",
+       "<text x=\"715.0\" y=\"306.0\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"14.5\" fill=\"#b45309\" font-weight=\"700\" text-anchor=\"middle\">TEST</text>\n",
+       "<path d=\"M 300 183 H 638\" fill=\"none\" stroke=\"#2563eb\" stroke-width=\"2.2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" marker-end=\"url(#arrowBlue)\"/>\n",
+       "<path d=\"M 226 303 H 638\" fill=\"none\" stroke=\"#ea7317\" stroke-width=\"2.2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" marker-end=\"url(#arrowOrange)\"/>\n",
+       "<text x=\"806\" y=\"187\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"11\" fill=\"#0e7490\" font-weight=\"400\" text-anchor=\"start\">whole source A</text>\n",
+       "<text x=\"806\" y=\"305\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"11\" fill=\"#b45309\" font-weight=\"400\" text-anchor=\"start\">whole source B</text>\n",
+       "</svg>"
+      ],
+      "text/plain": [
+       "<IPython.core.display.SVG object>"
+      ]
+     },
+     "metadata": {},
+     "output_type": "display_data"
+    },
+    {
+     "data": {
+      "image/svg+xml": [
+       "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"980\" height=\"500\" viewBox=\"0 0 980 500\">\n",
+       "<rect width=\"980\" height=\"500\" fill=\"#ffffff\"/>\n",
+       "<defs>\n",
+       "<marker id=\"arrow\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#94a3b8\"/></marker>\n",
+       "<marker id=\"arrowBlue\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#2563eb\"/></marker>\n",
+       "<marker id=\"arrowOrange\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#ea7317\"/></marker>\n",
+       "<marker id=\"arrowGreen\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#16a34a\"/></marker>\n",
+       "<marker id=\"arrowSlate\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#475569\"/></marker>\n",
+       "<marker id=\"arrowMute\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#64748b\"/></marker>\n",
+       "<marker id=\"arrowPurple\" markerWidth=\"9\" markerHeight=\"9\" refX=\"6.5\" refY=\"4\" orient=\"auto\"><path d=\"M0.5,0.5 L8,4 L0.5,7.5 z\" fill=\"#7c3aed\"/></marker>\n",
+       "</defs>\n",
+       "<text x=\"490.0\" y=\"34\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"21\" fill=\"#1a2733\" font-weight=\"700\" text-anchor=\"middle\">The evaluation firewall</text>\n",
+       "<text x=\"490.0\" y=\"57\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"13.5\" fill=\"#64748b\" font-weight=\"400\" text-anchor=\"middle\">Choose everything on inner folds; touch each outer fold exactly once</text>\n",
+       "<text x=\"70\" y=\"100\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"12.5\" fill=\"#1a2733\" font-weight=\"700\" text-anchor=\"start\">Outer folds (grouped by source)</text>\n",
+       "<rect x=\"70\" y=\"112\" width=\"150\" height=\"48\" rx=\"10\" fill=\"#eff6ff\" stroke=\"#bfdbfe\" stroke-width=\"1.4\"/>\n",
+       "<text x=\"145.0\" y=\"134\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"12\" fill=\"#2563eb\" font-weight=\"700\" text-anchor=\"middle\">fold 0</text>\n",
+       "<text x=\"145.0\" y=\"151\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"10.5\" fill=\"#64748b\" font-weight=\"400\" text-anchor=\"middle\">train</text>\n",
+       "<rect x=\"232\" y=\"112\" width=\"150\" height=\"48\" rx=\"10\" fill=\"#eff6ff\" stroke=\"#bfdbfe\" stroke-width=\"1.4\"/>\n",
+       "<text x=\"307.0\" y=\"134\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"12\" fill=\"#2563eb\" font-weight=\"700\" text-anchor=\"middle\">fold 1</text>\n",
+       "<text x=\"307.0\" y=\"151\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"10.5\" fill=\"#64748b\" font-weight=\"400\" text-anchor=\"middle\">train</text>\n",
+       "<rect x=\"394\" y=\"112\" width=\"150\" height=\"48\" rx=\"10\" fill=\"#fff7ed\" stroke=\"#fed7aa\" stroke-width=\"1.4\"/>\n",
+       "<text x=\"469.0\" y=\"134\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"12\" fill=\"#ea7317\" font-weight=\"700\" text-anchor=\"middle\">fold 2</text>\n",
+       "<text x=\"469.0\" y=\"151\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"10.5\" fill=\"#64748b\" font-weight=\"400\" text-anchor=\"middle\">TEST (once)</text>\n",
+       "<rect x=\"556\" y=\"112\" width=\"150\" height=\"48\" rx=\"10\" fill=\"#eff6ff\" stroke=\"#bfdbfe\" stroke-width=\"1.4\"/>\n",
+       "<text x=\"631.0\" y=\"134\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"12\" fill=\"#2563eb\" font-weight=\"700\" text-anchor=\"middle\">fold 3</text>\n",
+       "<text x=\"631.0\" y=\"151\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"10.5\" fill=\"#64748b\" font-weight=\"400\" text-anchor=\"middle\">train</text>\n",
+       "<rect x=\"718\" y=\"112\" width=\"150\" height=\"48\" rx=\"10\" fill=\"#eff6ff\" stroke=\"#bfdbfe\" stroke-width=\"1.4\"/>\n",
+       "<text x=\"793.0\" y=\"134\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"12\" fill=\"#2563eb\" font-weight=\"700\" text-anchor=\"middle\">fold 4</text>\n",
+       "<text x=\"793.0\" y=\"151\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"10.5\" fill=\"#64748b\" font-weight=\"400\" text-anchor=\"middle\">train</text>\n",
+       "<rect x=\"71.5\" y=\"202.5\" width=\"470\" height=\"150\" rx=\"14\" fill=\"#0f172a\" opacity=\"0.06\"/>\n",
+       "<rect x=\"70\" y=\"200\" width=\"470\" height=\"150\" rx=\"14\" fill=\"#f5f3ff\" stroke=\"#ddd6fe\" stroke-width=\"1.6\"/>\n",
+       "<text x=\"305\" y=\"226\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"13\" fill=\"#7c3aed\" font-weight=\"700\" text-anchor=\"middle\">Inner folds: pick everything here</text>\n",
+       "<circle cx=\"96\" cy=\"250\" r=\"3.5\" fill=\"#7c3aed\"/>\n",
+       "<text x=\"108\" y=\"254\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"11.5\" fill=\"#1a2733\" font-weight=\"400\" text-anchor=\"start\">mask ratio, learning rate, update budget</text>\n",
+       "<circle cx=\"96\" cy=\"272\" r=\"3.5\" fill=\"#7c3aed\"/>\n",
+       "<text x=\"108\" y=\"276\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"11.5\" fill=\"#1a2733\" font-weight=\"400\" text-anchor=\"start\">pooling, probe C, PCA dims</text>\n",
+       "<circle cx=\"96\" cy=\"294\" r=\"3.5\" fill=\"#7c3aed\"/>\n",
+       "<text x=\"108\" y=\"298\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"11.5\" fill=\"#1a2733\" font-weight=\"400\" text-anchor=\"start\">checkpoint selection, EMA schedule</text>\n",
+       "<circle cx=\"96\" cy=\"316\" r=\"3.5\" fill=\"#7c3aed\"/>\n",
+       "<text x=\"108\" y=\"320\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"11.5\" fill=\"#1a2733\" font-weight=\"400\" text-anchor=\"start\">SSL sees only inner-training sources (no labels)</text>\n",
+       "<rect x=\"581.5\" y=\"202.5\" width=\"330\" height=\"150\" rx=\"14\" fill=\"#0f172a\" opacity=\"0.06\"/>\n",
+       "<rect x=\"580\" y=\"200\" width=\"330\" height=\"150\" rx=\"14\" fill=\"#f0fdf4\" stroke=\"#bbf7d0\" stroke-width=\"1.6\"/>\n",
+       "<text x=\"745\" y=\"226\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"13\" fill=\"#16a34a\" font-weight=\"700\" text-anchor=\"middle\">Then, once per fold</text>\n",
+       "<circle cx=\"606\" cy=\"250\" r=\"3.5\" fill=\"#16a34a\"/>\n",
+       "<text x=\"618\" y=\"254\" font-family=\"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"11.5\" fill=\"#1a2733\" font-weight=\"400\" text-anchor=\"start\">refit frozen recipe on all outer-train</text>\n",
+       "<circle cx=\"606\" cy=\"272\" r=\"3.5\" fill=\"#16a34a\"/>\n",

exec
/bin/zsh -lc 'd=$(mktemp -d /tmp/ms-nbs.XXXXXX); python scripts/scripts_build_notebooks.py --output-dir "$d" >/dev/null; .venv/bin/python - "$d" <<'"'PY'
import json,sys
from pathlib import Path
a=json.load(open(Path(sys.argv[1])/'06_capstone_rf_vs_sjepa.ipynb'))
b=json.load(open('06_capstone_rf_vs_sjepa.ipynb'))
print(len(a['cells']),len(b['cells']))
for i,(x,y) in enumerate(zip(a['cells'],b['cells'])):
 sx=''.join(x['source']); sy=''.join(y['source'])
 if sx "'!= sy:
  print('"'\\nCELL',i,x['cell_type'],y['cell_type'])
  import difflib
  print(''.join(difflib.unified_diff(sx.splitlines(True),sy.splitlines(True),fromfile='gen',tofile='nb'))[:5000])
 for o in y.get('outputs',[]):
  if o.get('output_type')=='error': print('ERROR OUTPUT cell',i,o)
PY" in /Users/pmui/dev/alexpose/experiments/multiple-sclerosis
 succeeded in 0ms:
18 18

exec
/bin/zsh -lc ".venv/bin/python - <<'PY'
import json
p='06_capstone_rf_vs_sjepa.ipynb'; d=json.load(open(p))
for i,c in enumerate(d['cells']):
 print(i,c['cell_type'],c.get('execution_count'),''.join(c['source']).splitlines()[0][:80] if c['source'] else '')
 for o in c.get('outputs',[]):
  if 'text' in o: print('  OUT', ''.join(o['text'])[:800].replace('\\n',' | '))
  elif o.get('output_type')=='error': print(' ERR',o)
PY
printf '\\n-- current capstone artifact --\\n'; cat artifacts/capstone_results.json" in /Users/pmui/dev/alexpose/experiments/multiple-sclerosis
 succeeded in 0ms:
0 markdown None <a href="https://colab.research.google.com/github/your-org/alexpose/blob/main/ex
1 markdown None # 06 - Capstone: Random Forest vs S-JEPA, on identical folds, with controls
2 code 11 # --- Setup: install dependencies (Colab installs; local usually already has the
  OUT all light dependencies already present | 
3 code 12 # --- Make `sjepa` and `ambient` importable, locally and in Colab ---
  OUT experiment dir: /Users/pmui/dev/alexpose/experiments/multiple-sclerosis | repo root     : /Users/pmui/dev/alexpose | 
4 code 13 # --- Paths and profile (reads the root .env if python-dotenv is present) ---
  OUT SJEPA_PROFILE = laptop | SJEPA_SMOKE = 0 | 
5 code 14 from IPython.display import SVG, display
6 markdown None ## The frozen result is produced by a script, not the notebook
7 code 15 import json
  OUT Frozen R1 (1000 updates, seed 42, all 5 folds, pooled OOF): |   S-JEPA          : macro-F1 0.438 | acc 0.447 | PD-recall 0.235 |   Random Forest   : macro-F1 0.667 | acc 0.660 | PD-recall 0.588 |   effective rank per fold: [8.8, 9.5, 8.2, 8.8, 7.7] (all >> 1 -> no collapse) | 
8 markdown None ## Shortcut controls: the bar S-JEPA has to clear
9 code 16 e0_path = ARTIFACT_DIR / 'eval' / 'g1' / 'E0_results.json'
  OUT Shortcut controls on g1 (best of logreg/rf, fold-mean macro-F1): |   mean_pose       : 0.654 |   mean_std_pose   : 0.694 |   visibility_only : 0.611 |   duration_acq    : 0.410 |   body_proportion : 0.424 | E0 Random Forest pooled macro-F1: 0.667 | 
10 markdown None ## Reproduce the mechanism live (one fold, small budget)
11 code 17 import numpy as np, torch, os
12 code 18 fold0 = registry['folds'][0]
  OUT live fold-0 demo (budget=60 updates): RF f1=0.915 | S-JEPA f1=0.644 | eff_rank=7.7 | (The frozen 5-fold pooled numbers above are the ones to cite, not this one fold.) | 
13 markdown None ## Confusion of the frozen S-JEPA run
14 code 19 import numpy as np, matplotlib.pyplot as plt, seaborn as sns
15 markdown None ## A combined scoreboard
16 code 20 import pandas as pd
  OUT saved capstone_results.json | 
17 markdown None ## What this does and does not show

-- current capstone artifact --
{
  "frozen_run": "runs/r1_g1_1k_s42/results.json",
  "scoreboard": {
    "Random Forest (paired)": 0.6665728492353881,
    "S-JEPA (R1, 1k updates)": 0.43840245775729647,
    "control: mean_pose": 0.6535449735449734,
    "control: mean_std_pose": 0.6942857142857142,
    "control: visibility_only": 0.6113227513227513,
    "control: duration_acq": 0.4103848003848003,
    "control: body_proportion": 0.42444444444444435,
    "chance (3 classes)": 0.3333333333333333
  }
}
exec
/bin/zsh -lc "grep -n 'SJEPA_SMOKE' .env .env.example ../../.env 2>/dev/null || true; git diff -U3 -- scripts/notebook_content.py | grep -n -C3 \"SJEPA_SMOKE\"; grep -R \"SJEPA_SMOKE\" -n README.md docs scripts sjepa *.ipynb | head -100" in /Users/pmui/dev/alexpose/experiments/multiple-sclerosis
 succeeded in 0ms:
.env:20:SJEPA_SMOKE=0
.env.example:20:SJEPA_SMOKE=0
../../.env:92:SJEPA_SMOKE=0
238--        "print('final loss (mean of last 5 steps):',",
239--        "      round(sum(state.losses[-5:]) / min(5, len(state.losses)), 4))",
240-+        "# A small update budget keeps the notebook fast; raise it for stronger features.",
241:+        "UPDATES = 60 if __import__('os').environ.get('SJEPA_SMOKE') else 800",
242-+        "state = train_sjepa_v2(model, ds_all, cfg, total_updates=UPDATES, device=device,",
243-+        "                       mask_ratio=0.6, log_every=max(1, UPDATES // 4))",
244-+        "save_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl.pt', model, cfg, train_state=state,",
--
392--        "save_checkpoint(ARTIFACT_DIR / 'sjepa_finetuned_3class.pt', model, cfg,",
393--        "                extra={'stage': 'finetune_3class'})",
394-+        "ds_train = SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride)",
395:+        "MORE = 40 if __import__('os').environ.get('SJEPA_SMOKE') else 400",
396-+        "state = train_sjepa_v2(model, ds_train, cfg, total_updates=MORE, device=device,",
397-+        "                       mask_ratio=0.6, log_every=max(1, MORE // 4))",
398-+        "save_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl_continued.pt', model, cfg, train_state=state,",
--
451--        "if spread > 1e-3:",
452--        "    print('Features have healthy spread. Fine-tuned model saved.')",
453--        "else:",
454:-        "    print('Spread is small. With the full profile (not SJEPA_SMOKE) it opens up;',",
455--        "          'in a 2-epoch smoke run this is expected.')",
456-+        "## Regime 2 - balanced supervised adaptation (labels only in the head)\n",
457-+        "Now we use the labels honestly: freeze the encoder and fit a class-balanced logistic head on "
--
855--        "## Confusion matrices side by side\n",
856--        "Averaged over folds, where does each model confuse the conditions?\n",
857-+        "# label-free S-JEPA on this fold's training sources",
858:+        "UPDATES = 60 if os.environ.get('SJEPA_SMOKE') else 500",
859-+        "model = build_model(cfg, device=device, repaired=True)",
860-+        "ds = SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride)",
861-+        "state = train_sjepa_v2(model, ds, cfg, total_updates=UPDATES, device=device, mask_ratio=0.6)",
README.md:84:SJEPA_SMOKE=0          # set to 1 for a near-instant test run of every notebook
README.md:174:SJEPA_SMOKE=1 uv run python sjepa/tests/test_smoke.py
docs/06-0803-FINAL_REPORT.md:121:- R1: `SJEPA_SMOKE=0 .venv/bin/python scripts/scripts_r1_repaired.py --total-updates 1000 --seed 42 --output-dir artifacts/runs/r1_g1_1k_s42`
docs/06-0803-FINAL_REPORT.md:122:- Tests: `SJEPA_SMOKE=1 .venv/bin/python -m pytest sjepa/tests -p no:cov -q`
docs/06-0803-FINAL_REPORT.md:179:SJEPA_SMOKE=1 .venv/bin/python -m pytest sjepa/tests -p no:cov -q            # 15 tests
docs/06-0803-FINAL_REPORT.md:181:SJEPA_SMOKE=0 .venv/bin/python scripts/scripts_r1_repaired.py \
Binary file scripts/__pycache__/notebook_content.cpython-314.pyc matches
Binary file scripts/__pycache__/notebook_content.cpython-312.pyc matches
scripts/scripts_build_notebooks.py:132:        "# in your .env for a larger model, or SJEPA_SMOKE=1 for a near-instant test run.",
scripts/scripts_build_notebooks.py:135:        "      '| SJEPA_SMOKE =', os.environ.get('SJEPA_SMOKE', '0'))",
scripts/notebook_content.py:395:        "UPDATES = 60 if __import__('os').environ.get('SJEPA_SMOKE') else 800",
scripts/notebook_content.py:495:        "MORE = 40 if __import__('os').environ.get('SJEPA_SMOKE') else 400",
scripts/notebook_content.py:829:        "UPDATES = 60 if os.environ.get('SJEPA_SMOKE') else 500",
sjepa/config.py:13:A third switch, ``SJEPA_SMOKE=1``, shrinks whichever profile you picked down to a
sjepa/config.py:162:    2. If ``smoke`` is True (argument) or ``SJEPA_SMOKE`` is set, shrink to the
sjepa/config.py:181:    use_smoke = smoke if smoke is not None else _env_flag("SJEPA_SMOKE")
sjepa/tests/test_correctness.py:8:Run:  SJEPA_SMOKE=1 python -m pytest sjepa/tests/test_correctness.py -q
Binary file sjepa/tests/__pycache__/test_smoke.cpython-314-pytest-9.0.2.pyc matches
Binary file sjepa/tests/__pycache__/test_correctness.cpython-312-pytest-9.1.1.pyc matches
Binary file sjepa/tests/__pycache__/test_train_v2.cpython-312.pyc matches
Binary file sjepa/tests/__pycache__/test_train_v2.cpython-312-pytest-9.1.1.pyc matches
Binary file sjepa/tests/__pycache__/test_train_v2.cpython-314-pytest-9.0.2.pyc matches
Binary file sjepa/tests/__pycache__/test_correctness.cpython-314-pytest-9.0.2.pyc matches
Binary file sjepa/tests/__pycache__/test_smoke.cpython-312-pytest-9.1.1.pyc matches
sjepa/tests/test_smoke.py:5:by SJEPA_SMOKE so it is fast), and checks the properties that matter:
sjepa/tests/test_smoke.py:13:Run:  SJEPA_SMOKE=1 python -m pytest sjepa/tests/test_smoke.py -q
sjepa/tests/test_smoke.py:14:Or:   SJEPA_SMOKE=1 python sjepa/tests/test_smoke.py
sjepa/tests/test_smoke.py:134:    os.environ.setdefault("SJEPA_SMOKE", "1")
sjepa/tests/test_train_v2.py:12:Run:  SJEPA_SMOKE=1 python -m pytest sjepa/tests/test_train_v2.py -q
Binary file sjepa/__pycache__/config.cpython-314.pyc matches
Binary file sjepa/__pycache__/config.cpython-312.pyc matches
00_overview_and_video_gallery.ipynb:123:    "# in your .env for a larger model, or SJEPA_SMOKE=1 for a near-instant test run.\n",
00_overview_and_video_gallery.ipynb:126:    "      '| SJEPA_SMOKE =', os.environ.get('SJEPA_SMOKE', '0'))"
01_pose_extraction_from_raw_video.ipynb:109:    "# in your .env for a larger model, or SJEPA_SMOKE=1 for a near-instant test run.\n",
01_pose_extraction_from_raw_video.ipynb:112:    "      '| SJEPA_SMOKE =', os.environ.get('SJEPA_SMOKE', '0'))"
02_anatomical_mask_and_tokenization.ipynb:109:    "# in your .env for a larger model, or SJEPA_SMOKE=1 for a near-instant test run.\n",
02_anatomical_mask_and_tokenization.ipynb:112:    "      '| SJEPA_SMOKE =', os.environ.get('SJEPA_SMOKE', '0'))"
03_sjepa_model_and_pretrain_normal.ipynb:116:    "# in your .env for a larger model, or SJEPA_SMOKE=1 for a near-instant test run.\n",
03_sjepa_model_and_pretrain_normal.ipynb:119:    "      '| SJEPA_SMOKE =', os.environ.get('SJEPA_SMOKE', '0'))"
03_sjepa_model_and_pretrain_normal.ipynb:210:    "UPDATES = 60 if __import__('os').environ.get('SJEPA_SMOKE') else 800\n",
04_progressive_finetune_ms_pd_vicreg.ipynb:117:    "# in your .env for a larger model, or SJEPA_SMOKE=1 for a near-instant test run.\n",
04_progressive_finetune_ms_pd_vicreg.ipynb:120:    "      '| SJEPA_SMOKE =', os.environ.get('SJEPA_SMOKE', '0'))"
04_progressive_finetune_ms_pd_vicreg.ipynb:198:    "MORE = 40 if __import__('os').environ.get('SJEPA_SMOKE') else 400\n",
05_representation_visualization.ipynb:114:    "# in your .env for a larger model, or SJEPA_SMOKE=1 for a near-instant test run.\n",
05_representation_visualization.ipynb:117:    "      '| SJEPA_SMOKE =', os.environ.get('SJEPA_SMOKE', '0'))"
06_capstone_rf_vs_sjepa.ipynb:123:      "SJEPA_PROFILE = laptop | SJEPA_SMOKE = 0\n"
06_capstone_rf_vs_sjepa.ipynb:143:    "# in your .env for a larger model, or SJEPA_SMOKE=1 for a near-instant test run.\n",
06_capstone_rf_vs_sjepa.ipynb:146:    "      '| SJEPA_SMOKE =', os.environ.get('SJEPA_SMOKE', '0'))"
06_capstone_rf_vs_sjepa.ipynb:487:    "UPDATES = 60 if os.environ.get('SJEPA_SMOKE') else 500\n",

exec
/bin/zsh -lc "nl -ba 06_capstone_rf_vs_sjepa.ipynb | sed -n '25,48p'; nl -ba 03_sjepa_model_and_pretrain_normal.ipynb | sed -n '204,215p'; nl -ba 04_progressive_finetune_ms_pd_vicreg.ipynb | sed -n '186,204p'; nl -ba 04_progressive_finetune_ms_pd_vicreg.ipynb | sed -n '236,250p'; nl -ba scripts/notebook_content.py | sed -n '390,400p'" in /Users/pmui/dev/alexpose/experiments/multiple-sclerosis
 succeeded in 0ms:
    25	    "\n",
    26	    "> **Spoiler, stated up front.** On this tiny, already-inspected, source-grouped collection the S-JEPA scores *below* both the Random Forest and the nuisance controls. That is the expected, plan-anticipated outcome of removing the shortcuts and the label leak that inflated an earlier number, and it is reported as a negative result, not hidden.\n"
    27	   ]
    28	  },
    29	  {
    30	   "cell_type": "code",
    31	   "execution_count": 11,
    32	   "id": "06_capstone_rf_vs_sjepa-02",
    33	   "metadata": {},
    34	   "outputs": [
    35	    {
    36	     "name": "stdout",
    37	     "output_type": "stream",
    38	     "text": [
    39	      "all light dependencies already present\n"
    40	     ]
    41	    }
    42	   ],
    43	   "source": [
    44	    "# --- Setup: install dependencies (Colab installs; local usually already has them) ---\n",
    45	    "import importlib, importlib.util, subprocess, sys, os\n",
    46	    "\n",
    47	    "IN_COLAB = 'google.colab' in sys.modules\n",
    48	    "\n",
   204	   "execution_count": null,
   205	   "outputs": [],
   206	   "source": [
   207	    "from sjepa.train_v2 import train_sjepa_v2, save_checkpoint_v2\n",
   208	    "\n",
   209	    "# A small update budget keeps the notebook fast; raise it for stronger features.\n",
   210	    "UPDATES = 60 if __import__('os').environ.get('SJEPA_SMOKE') else 800\n",
   211	    "state = train_sjepa_v2(model, ds_all, cfg, total_updates=UPDATES, device=device,\n",
   212	    "                       mask_ratio=0.6, log_every=max(1, UPDATES // 4))\n",
   213	    "save_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl.pt', model, cfg, train_state=state,\n",
   214	    "                   extra={'stage': 'ssl_all_sources'})\n",
   215	    "print('EMA half-life (steps):', round(state.ema_half_life_steps, 1))"
   186	   "source": [
   187	    "from sjepa.config import get_config\n",
   188	    "from sjepa.models import build_model, pick_device\n",
   189	    "from sjepa.train_v2 import train_sjepa_v2, load_checkpoint_v2, save_checkpoint_v2\n",
   190	    "from sjepa.data import SequenceWindowDataset\n",
   191	    "\n",
   192	    "cfg = get_config()\n",
   193	    "device = pick_device()\n",
   194	    "model = build_model(cfg, device=device, repaired=True)\n",
   195	    "load_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl.pt', model, map_location=device)\n",
   196	    "\n",
   197	    "ds_train = SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride)\n",
   198	    "MORE = 40 if __import__('os').environ.get('SJEPA_SMOKE') else 400\n",
   199	    "state = train_sjepa_v2(model, ds_train, cfg, total_updates=MORE, device=device,\n",
   200	    "                       mask_ratio=0.6, log_every=max(1, MORE // 4))\n",
   201	    "save_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl_continued.pt', model, cfg, train_state=state,\n",
   202	    "                   extra={'stage': 'ssl_continuation_fold0'})\n",
   203	    "print('continued SSL: final effective rank', round(state.eff_rank[-1], 1))"
   204	   ],
   236	    "        x = torch.from_numpy(w).float().to(device)\n",
   237	    "        with torch.no_grad():\n",
   238	    "            V.append(m.embed(x, tm).mean(0).cpu().numpy())\n",
   239	    "        Y.append(r.label)\n",
   240	    "    return np.stack(V), Y"
   241	   ],
   242	   "id": "04_progressive_finetune_ms_pd_vicreg-11"
   243	  },
   244	  {
   245	   "cell_type": "markdown",
   246	   "metadata": {},
   247	   "source": [
   248	    "## Regime 2 - balanced supervised adaptation (labels only in the head)\n",
   249	    "\n",
   250	    "Now we use the labels honestly: freeze the encoder and fit a class-balanced logistic head on the training embeddings, then score the held-out videos. The scaler and the head are fit on **training data only**. We do this on top of both the notebook-03 checkpoint and the SSL-continued one, so we can see whether extra unlabeled training moved the probe at all.\n",
   390	    )]
   391	    c += [code(
   392	        "from sjepa.train_v2 import train_sjepa_v2, save_checkpoint_v2",
   393	        "",
   394	        "# A small update budget keeps the notebook fast; raise it for stronger features.",
   395	        "UPDATES = 60 if __import__('os').environ.get('SJEPA_SMOKE') else 800",
   396	        "state = train_sjepa_v2(model, ds_all, cfg, total_updates=UPDATES, device=device,",
   397	        "                       mask_ratio=0.6, log_every=max(1, UPDATES // 4))",
   398	        "save_checkpoint_v2(ARTIFACT_DIR / 'sjepa_ssl.pt', model, cfg, train_state=state,",
   399	        "                   extra={'stage': 'ssl_all_sources'})",
   400	        "print('EMA half-life (steps):', round(state.ema_half_life_steps, 1))",

exec
/bin/zsh -lc "git show HEAD:experiments/multiple-sclerosis/06_capstone_rf_vs_sjepa.ipynb | sed -n '25,45p'; git show HEAD:experiments/multiple-sclerosis/03_sjepa_model_and_pretrain_normal.ipynb | sed -n '25,40p'; git ls-files artifacts/eval/g1 artifacts/runs/r1_g1_1k_s42 | head" in /Users/pmui/dev/alexpose/experiments/multiple-sclerosis
 succeeded in 0ms:
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "06_capstone_rf_vs_sjepa-02",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-08-02T21:59:37.282443Z",
     "iopub.status.busy": "2026-08-02T21:59:37.282286Z",
     "iopub.status.idle": "2026-08-02T21:59:37.288954Z",
     "shell.execute_reply": "2026-08-02T21:59:37.288472Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "all light dependencies already present\n"
     ]
    }
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "03_sjepa_model_and_pretrain_normal-02",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-08-02T21:59:15.251794Z",
     "iopub.status.busy": "2026-08-02T21:59:15.251611Z",
     "iopub.status.idle": "2026-08-02T21:59:15.258946Z",
     "shell.execute_reply": "2026-08-02T21:59:15.258134Z"
    }
   },
   "outputs": [],
   "source": [
artifacts/eval/g1/COMPLETED.json
artifacts/eval/g1/E0_RF_oof.json
artifacts/eval/g1/E0_results.json
artifacts/eval/g1/fold_registry.json
artifacts/eval/g1/run_manifest.json
artifacts/runs/r1_g1_1k_s42/COMPLETED.json
artifacts/runs/r1_g1_1k_s42/oof.json
artifacts/runs/r1_g1_1k_s42/results.json
artifacts/runs/r1_g1_1k_s42/run_manifest.json

codex
Normal runs unintentionally use smoke-sized training budgets, and the new fold-based evaluation reuses a checkpoint pretrained on its test clips. These issues invalidate intended training behavior and part of the reported evaluation.

Full review comments:

- [P1] Parse the smoke flag instead of testing string truthiness — /Users/pmui/dev/alexpose/experiments/multiple-sclerosis/scripts/notebook_content.py:395-395
  With the provided `.env` setting `SJEPA_SMOKE=0`, `os.environ.get(...)` returns the truthy string `"0"`, so normal runs use only 60 updates instead of 800. The same defect affects the 40/400 and 60/500 selectors later, silently weakening every generated checkpoint.

- [P1] Restrict pretraining to the selected fold's training records — /Users/pmui/dev/alexpose/experiments/multiple-sclerosis/scripts/notebook_content.py:379-380
  When notebooks 03–04 are run sequentially, this checkpoint is trained on every record, including fold 0's later test clips, before notebook 04 loads it and reports held-out probe scores. Labels being ignored does not prevent test-covariate leakage, so the claimed leakage-safe comparison is invalid; create the checkpoint from `train_recs` for the evaluated fold.
Normal runs unintentionally use smoke-sized training budgets, and the new fold-based evaluation reuses a checkpoint pretrained on its test clips. These issues invalidate intended training behavior and part of the reported evaluation.

Full review comments:

- [P1] Parse the smoke flag instead of testing string truthiness — /Users/pmui/dev/alexpose/experiments/multiple-sclerosis/scripts/notebook_content.py:395-395
  With the provided `.env` setting `SJEPA_SMOKE=0`, `os.environ.get(...)` returns the truthy string `"0"`, so normal runs use only 60 updates instead of 800. The same defect affects the 40/400 and 60/500 selectors later, silently weakening every generated checkpoint.

- [P1] Restrict pretraining to the selected fold's training records — /Users/pmui/dev/alexpose/experiments/multiple-sclerosis/scripts/notebook_content.py:379-380
  When notebooks 03–04 are run sequentially, this checkpoint is trained on every record, including fold 0's later test clips, before notebook 04 loads it and reports held-out probe scores. Labels being ignored does not prevent test-covariate leakage, so the claimed leakage-safe comparison is invalid; create the checkpoint from `train_recs` for the evaluated fold.
