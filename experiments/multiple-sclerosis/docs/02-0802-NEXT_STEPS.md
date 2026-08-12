# Next steps: making the S-JEPA comparison scientifically meaningful

_Research and implementation plan, reviewed 2026-08-02_

## Executive decision

The next experiment should **not** be the existing `gpu` profile. The present result is constrained first by correctness, sampling, preprocessing, and evaluation problems; increasing model size would make those problems more expensive without making the result more trustworthy.

The highest-upside path is:

1. repair the predictor's missing target-position information and replace the permanent anatomical mask;
2. rebuild a strictly label-free, source-balanced S-JEPA baseline on every outer-training source for a meaningful number of optimizer updates;
3. replace double mean pooling with multi-clip temporal aggregation and add genuine supervised encoder adaptation;
4. remove acquisition shortcuts and preserve clinically meaningful motion;
5. only then scale with external gait/pathological-motion data, followed by a carefully controlled RGB V-JEPA fusion experiment.

This order matters. The current `0.570 +/- 0.112` macro-F1 is **not a clean estimate of the attainable accuracy of S-JEPA**. Two implementation choices currently prevent the pretext task from learning the intended joint-specific predictions, while low-order pose and acquisition controls reveal substantial shortcut risk. The immediate goal is therefore a valid representation, not a more optimistic headline number.

There is no peer-reviewed study in the literature reviewed for this plan that directly applies S-JEPA or V-JEPA to normal/MS/PD gait classification. Recommendations below distinguish published findings from hypotheses that must be tested here.

## What the notebooks have established

The notebook series has done several important things correctly. It creates a reproducible video-to-pose path, keeps clips with the same `source_id` in the same fold, trains a new S-JEPA model inside each capstone fold, compares against a strong hand-engineered Random Forest, and reports fold-to-fold dispersion rather than a single favorable split. The honest negative result is useful: with the current implementation and data, the RF is stronger. Fold standard deviation is descriptive dispersion, not a confidence interval.

The executed capstone run has the following operating point:

| Item | Current value | Consequence |
|---|---:|---|
| Usable clips | 47 from about 35 `source_id` values | Very small effective sample size |
| Clips / distinct sources | normal 19/16, MS 11/11, PD 17/8 | PD has only eight independent groups |
| Training windows | 481 total: normal 92, MS 167, PD 222 | Window counts do not represent independent evidence |
| Windows per source | 1 to about 75 | Long sources can dominate optimization |
| Input | 32 frames, nominally 15 fps, 33 joints, 3 channels | Only about 2.1 seconds of nominal context |
| Channels | normalized `x`, `y`, and visibility | The third channel is confidence, not physical depth |
| Tokens | 8 time groups x 33 joints = 264 | Four input frames per token |
| Target | 12 fixed shoulder/lower-body joints, 96 tokens | The same joints are hidden on every update |
| Laptop encoder | 3 layers, width 96, 4 heads | About 0.398M trainable parameters |
| Fold-local training | 10 pretraining epochs + 7 continuation epochs | Only roughly 30 SSL and 84–98 continuation updates per fold |
| Downstream model | frozen 96-D mean embedding + balanced logistic regression | No discriminative classification loss, although normal selection and class-aware VICReg already consume diagnosis labels |

Headline results from `06_capstone_rf_vs_sjepa.ipynb`:

| Metric | Random Forest | Current S-JEPA probe |
|---|---:|---:|
| Accuracy | 0.656 +/- 0.093 | 0.573 +/- 0.097 |
| Macro-F1 | 0.668 +/- 0.096 | 0.570 +/- 0.112 |

The pooled S-JEPA out-of-fold confusion counts are:

| True class | Predicted normal | Predicted MS | Predicted PD |
|---|---:|---:|---:|
| normal | 12 | 3 | 4 |
| MS | 0 | 9 | 2 |
| PD | 5 | 6 | 6 |

PD recall is therefore only `6/17 = 0.353`, with PD-to-MS the main error. The per-fold S-JEPA macro-F1 ranges from 0.451 to 0.746, which is consistent with high split sensitivity and the fact that three test folds contain only one independent PD source.

### Reflection by notebook

| Notebook | What it contributes | What should change next |
|---|---|---|
| `00_overview_and_video_gallery.ipynb` | Makes the small, heterogeneous video set inspectable | Add a manifest of participant, site/domain, task, view, frame rate, duration, and quality; do not equate a YouTube source automatically with one participant |
| `01_pose_extraction_from_raw_video.ipynb` | Produces reusable 33-landmark caches | Re-extract with timestamp-accurate sampling, temporal tracking, explicit missingness, short-gap interpolation, and world/weak-depth landmarks where available |
| `02_anatomical_mask_and_tokenization.ipynb` | Encodes clinical prior knowledge | Convert the fixed clinical mask into a *sampling bias or loss weight*, not permanent removal of the same joints from context |
| `03_sjepa_model_and_pretrain_normal.ipynb` | Demonstrates the EMA teacher/predictor loop | It currently pretrains on all normal data before the later holdout is defined; use fold-local, all-source, label-free pretraining for classification, and reserve normal-only training for an explicit anomaly-detection branch |
| `04_progressive_finetune_ms_pd_vicreg.ipynb` | Enforces a grouped train/test split | Replace the misleading class-aware VICReg stage with either strict SSL or an explicitly supervised objective; the current stage is not supervised fine-tuning in the usual sense |
| `05_representation_visualization.ipynb` | Makes embeddings inspectable | Never use test-label silhouette to select a model; add layerwise, temporal-shuffle, static-pose, confidence-only, and domain probes |
| `06_capstone_rf_vs_sjepa.ipynb` | Trains inside each fold and compares the two branches fairly at clip level | Remove epoch quartering, balance sources, use nested selection and several seeds, aggregate multiple clips intelligently, and perform actual encoder fine-tuning |

`notebook_content.py` is the durable source for the notebooks. Future notebook edits should be made there and regenerated, rather than editing only the `.ipynb` files.

## Blocking defects to repair before interpreting another score

### 1. Hidden targets have no positional identity in the predictor

`SJEPAEncoder.forward_context()` scatters encoded visible tokens back into the full sequence and fills every hidden location with the same learned `mask_token`. `Predictor` then applies a transformer without adding spatial or temporal position embeddings. A transformer is permutation-equivariant, so hidden slots with identical inputs and identical context remain identical. A synthetic audit produced exactly zero standard deviation across predicted target positions.

This means the model cannot make a different latent prediction for a left ankle at one time and a right shoulder at another time. That is a structural failure, not a tuning issue.

Required repair:

- add predictor-dimensional joint and temporal position embeddings to every predictor slot, with each hidden token constructed from `mask_token + joint_position + time_position`;
- keep the target indices explicit when multiple masks are sampled;
- initially use learned factorized joint/time embeddings for a controlled repair; compare 3-D rotary position embeddings only after the baseline works;
- add a regression test asserting that target predictions vary across target indices for a non-degenerate input.

Promotion gate: `std(pred[:, target_idx, :], dim=target-position)` must be finite and non-zero, and permuting target position IDs must change the corresponding predictions.

### 2. The fixed mask starves the online encoder of the joints used downstream

The view encoder never receives shoulders, hips, knees, ankles, heels, or foot indices as context. Direct gradient inspection shows zero gradient for those joints' spatial-position-embedding rows. Shared projection and transformer weights still update from other joints and affect the EMA teacher, but the permanently hidden joint-position rows remain at initialization. Downstream `embed()` then pools exactly these target tokens.

Clinical knowledge should influence *how often* a region is predicted, not make that region absent from the student forever.

Required repair:

- use stochastic connected graph-time blocks spanning limbs/trunk and contiguous temporal intervals;
- ensure every joint is context on some updates and target on others;
- enforce at least one lower-body/contralateral cue in context when a leg region is targeted;
- mix uniform anatomical masks with high-motion and low-motion masks so reduced motion is not ignored;
- optionally oversample clinically relevant lower-body targets by 1.5–2x, while retaining context coverage;
- sample two masks per sequence where memory permits.

Implement per-example masks explicitly as `(batch, masks, tokens)`. For an initial vectorized version, keep a fixed target count, expand the batch across masks, pad variable context lengths, and pass a key-padding mask through attention. Do not silently reuse one `(tokens,)` mask across every sample as the current API does.

Initial mask-ratio search: `0.40, 0.60, 0.75`. Include `0.90` only as a literature-faithfulness ablation; the S-JEPA value was validated with much larger action datasets, not 35 gait sources.

Promotion gates:

- every joint is visible in at least 20% and targeted in at least 10% of sampled masks over an epoch;
- every joint's spatial position embedding receives non-zero gradient over a mask bank;
- mask histograms and connected-region examples are saved with each run.

### 3. The class-aware VICReg description and behavior disagree

The current implementation subtracts each class mean and applies the variance floor and covariance penalty to the residuals. This does **not** compact a class or separate class centers. It is invariant to the location of the class centers and explicitly rewards within-class standard deviation up to the variance floor. Singleton classes in a batch contribute zero residual variance. The notebook text saying that it “keeps each condition compact” is therefore incorrect.

Required repair:

- remove class-aware VICReg from the strict SSL baseline;
- if standard VICReg is retained, apply it to a disposable projection head across two views and ablate it against no auxiliary loss;
- for diagnosis-aware training, name it supervised adaptation and use balanced cross-entropy plus either supervised contrastive, center, or prototypical loss as a separately controlled experiment;
- apply every label-using loss only to the labeled fraction in label-efficiency studies.

### 4. The teacher barely moves at the current update budget

The capstone quarters the nominal epoch counts. With only a few batches per epoch, the normal-only stage has roughly 30 optimizer steps. An EMA ramp of 0.996 toward 1.0 across so few steps leaves the target close to initialization. Copying the paper's even slower `0.9999 -> 1.0` schedule would make this worse unless the run has tens of thousands of updates.

Required repair:

- define schedules in optimizer updates, not epochs;
- report EMA half-life, teacher/student cosine distance, and target feature drift;
- use a responsive schedule such as `0.990 -> 0.9995` for a 5k-step local run, then tune by teacher lag rather than copying a paper constant;
- never use `0.9999` for a short run.

### 5. Pretraining and pooling discard the strongest available signal

For a three-class representation, normal-only pretraining throws away most fold-local unlabeled motion. It is defensible for one-class abnormality detection, as in FSGait, but that is a different scientific question. At inference, the model mean-pools target tokens within every window and then mean-pools all windows. Contextual token features may still encode phase or asymmetry, but double averaging can obscure their location, dispersion, and brief abnormalities.

Required repair:

- pretrain label-free on **all records in the current training partition**, uniformly by source;
- expose token features and intermediate layers rather than only `embed(...).mean()`;
- compare mean+standard-deviation pooling, fixed regional pooling, learned-query attention, and source-level multiple-instance learning;
- use 2–4 uniformly spaced temporal clips per source during training and evaluation;
- retain localized prediction-residual summaries as an exploratory feature, because a brief abnormal event can disappear under a global mean.

### 6. Staged training currently resets state at every call

`train_sjepa()` creates a new centered loss, optimizer, LR schedule, EMA schedule, and local step counter on every call. Checkpoints omit the running center, optimizer, scheduler/global step, and sampler state. The normal-to-all-source transition is therefore a restart with weights, not a continuous staged optimization, and future domain adaptation would have the same problem.

Required repair:

- make the running center and global optimizer/update state part of a serializable training state;
- save and restore optimizer, scaler, scheduler, EMA schedule position, RNG, sampler, and center when a stage is meant to continue;
- provide an explicit `reset_optimizer/reset_center` ablation when a deliberate restart is desired;
- test that a save/resume run matches an uninterrupted run within numerical tolerance.

## Data and evaluation risks that can masquerade as accuracy

### Frame-rate/domain leakage

All 12 audited raw MS videos are 60 fps and square 1080x1080, while normal and PD videos span roughly 15–30 fps and more heterogeneous sources; 11 MS clips survive into the current cache. The loader computes an integer stride with `round(source_fps / target_fps)`: 24 fps becomes about 12 fps and 25 fps becomes about 12.5 fps, but both caches are labeled 15 fps. This distorts cadence and makes acquisition characteristics class-correlated.

The cached mean visibility also differs by class (approximately MS 0.920, normal 0.862, PD 0.870). A read-only order-insensitive probe using each joint's temporal coordinate mean and standard deviation reached about `0.623 +/- 0.119` fold-mean macro-F1. Temporal standard deviation contains motion amplitude, so this is neither a purely static nor a purely domain probe. It does show that low-order moments are already competitive and that each shortcut control must be specified and rerun reproducibly with its features, estimator, regularization, and fold registry saved.

Required changes:

- resample by timestamps to exactly 15 fps and persist original timestamps, source fps, actual sampled fps, and frame indices;
- use MediaPipe video/tracking mode rather than independent image inference when possible;
- rename the third channel `visibility`; do not call it `z` or treat it as depth;
- make `xy` the primary model input and use confidence for missingness, quality weighting, or a separately ablated confidence branch;
- build a weak-depth/world-landmark branch and a stronger lifted-3D/SMPL branch separately;
- train a domain/view/fps classifier from learned embeddings; high domain predictability blocks a clinical interpretation.

### Normalization removes useful gait physics and amplifies noise

Per-frame pelvis centering removes root progression/walking speed, and per-frame torso scaling can turn pose jitter into large normalized excursions. Some audited normalized coordinates have maxima in the tens. `clean_sequence()` currently interpolates arbitrary-length gaps despite its short-gap description. Short clips are padded by repeating the final frame without an attention padding mask.

Required changes:

- estimate a robust sequence-level scale from the median valid torso/hip geometry, clip pathological scales, and log outliers;
- retain root translation/velocity, cadence, and physical duration as a separately ablated stream or side channel; these can themselves encode camera geometry, clip editing, and task protocol unless calibrated;
- interpolate only short gaps, initially at most 3 frames at 15 fps, and carry an explicit validity mask;
- reject or down-weight sequences with poor foot/ankle coverage;
- supply an attention padding mask so repeated padding cannot be learned as motion;
- preserve left/right asymmetry and distal foot/shank angles;
- track/crop the intended person and label camera view and walking direction;
- segment steady walking bouts or gait cycles and exclude starts, stops, turns, demonstrations, and unrelated footage in a separately reported whole-clip ablation.

### `source_id` is not yet guaranteed to be `participant_id`

A source is currently a video/YouTube identifier; some sources may show multiple people, and one participant could conceivably appear across sources. Before calling the split participant-disjoint, create explicit `participant_id`, `source_id`, `site/domain`, `walking_task`, `view`, and `clip_interval` fields and audit near-duplicates/reposts.

### The label-efficiency cell is not a valid semi-supervised result

The current bonus cell uses one non-stratified grouped split whose small test set is highly imbalanced, samples labeled records without stratification, and uses labels during representation training through normal selection and class-aware VICReg. It therefore does not isolate the effect of probe-label fraction.

The corrected protocol must use all outer-training videos only as unlabeled SSL input, restrict every label-aware operation to the sampled labeled subset, use multiple stratified group-level subset draws, and evaluate on the same locked outer folds.

## What the literature says—and what transfers to this project

### Skeleton and clinical-gait self-supervision

| Primary source | Published finding | Translation for this project |
|---|---|---|
| [S-JEPA, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) | Uses factorized spatial/temporal embeddings on encoder inputs, target-output masking, stochastic motion-aware masking at 90%, 120-frame inputs, 1,200 pretraining epochs, effective batch 256, and EMA `0.9999 -> 1.0`; target-output masking beat target-input masking by 7.2 points. Evaluation includes both frozen linear probing and supervised encoder fine-tuning. | Reproduce target-output masking and stochastic masks, then match optimizer *updates*, not paper epochs. Predictor-specific target positions are a mathematically necessary repair to this repository, not an explicitly documented paper component. The paper predicts latents from same-modality 3-D joints; ours is a 2-D `x,y`-plus-visibility variant. |
| [MAMP, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.html) | Predicting first temporal joint differences is stronger than coordinate reconstruction for skeleton action representation. | Add a velocity target; acceleration and future-latent targets are separate hypotheses. Balance high- and low-motion regions because abnormally low movement is clinically informative. |
| [GaitForeMer, MICCAI 2022](https://doi.org/10.1007/978-3-031-16452-1_13) | On 54 participants under leave-one-subject-out evaluation, NTU RGB+D 3-D motion-forecasting pretraining followed by PD four-class MDS-UPDRS severity adaptation reached F1 0.76 versus 0.60 from scratch. | External motion pretraining plus genuine disease-specific encoder adaptation is a higher-priority scaling path than training a large model from scratch here; this was severity estimation, not PD diagnosis. |
| [SSL of Gait-Based Biomarkers, 2023 preprint/workshop paper](https://arxiv.org/abs/2307.16321) | Contrastive temporal crops learned clinically useful gait features. Omitting `[CLS]` was significantly associated with better downstream classification; next-step prediction improved SSL loss, while noise/forgetful masking merely appeared unhelpful. | Select with nested downstream validation, retain a no-CLS/pooled baseline, and avoid treating lower SSL loss as better classification. Evidence is preprint/workshop-level. |
| [FSGait, ACCV 2024](https://openaccess.thecvf.com/content/ACCV2024/html/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.html) | Learns posture reconstruction and temporal prediction from normal gait for anomaly detection, but its evaluation is small and partly synthetic/non-clinical. | It motivates a conceptual one-class branch, not an expectation of better MS/PD accuracy. Keep normal-only learning separate from the default three-way representation stage. |
| [CARE-PD, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/bedc73979a95be7727af0c9a99c675ce-Paper-Datasets_and_Benchmarks_Track.pdf) and [dataset DOI](https://doi.org/10.5683/SP3/TWIKMK) | Harmonizes 8,477 walks from 362 participants across nine cohorts/eight sites. Fine-tuning generic motion models on CARE-PD's 2-D-to-3-D task substantially improved motion error and downstream UPDRS severity macro-F1; this was not S-JEPA pretraining. | This is the strongest available external clinical-motion adaptation candidate. PD-to-MS transfer is plausible, not proven; adapt fold-locally and keep site/domain controls. |
| [GaitPT scaling study, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37340) | Pretraining transformer variants on a private corpus of 2.7M in-the-wild skeleton walks shows predictable identity-recognition gains from more data and compute; larger models are not always the best use of a fixed compute budget. | It supports scaling data and optimization before architecture but is not currently an actionable public initialization. Biometric identity can reward anthropometric shortcuts, so any transfer must pass the temporal controls below. |
| [General Feature Prediction, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Towards_Efficient_General_Feature_Prediction_in_Masked_Skeleton_Modeling_ICCV_2025_paper.html) | Hierarchical local-to-global feature targets improve masked skeleton modeling efficiently. | A worthwhile later ablation after the basic predictor/mask is correct; do not add it while debugging the baseline. |

Recent clinical work also argues for longer, gait-aware observations. A [normal/tandem markerless MS study (TNSRE 2025)](https://doi.org/10.1109/TNSRE.2025.3589765) analyzed 18 cycles per person from 25 people with MS and 25 controls using three-camera 3-D pose; tandem heel-to-toe distance exposed low-EDSS differences absent in normal gait. A [single-camera 2-D MS study (MSARD 2026)](https://doi.org/10.1016/j.msard.2026.107285) studied 20 people with MS and associated normalized stride measures and shank/foot elevation-angle range with EDSS after confidence filtering and at least three cycles. These findings support longer or cycle-aligned windows, task labels, and distal-joint quality control; they do not establish a JEPA configuration.

Three additional sources shape the evaluation design. [Kaur et al., JBHI 2023](https://doi.org/10.1109/JBHI.2022.3208077) directly evaluates MS/PD/control gait classification under new-subject and new-task conditions. An [FG 2024 clinical motion-encoder benchmark](https://arxiv.org/abs/2405.17817) reports materially lower F1 under leave-one-subject-out than ordinary random splitting, reinforcing participant-disjoint evaluation. [The Paradox of Motion, FG 2024](https://arxiv.org/abs/2402.08320) shows that static pose and anthropometry can be strong shortcuts in gait identity recognition, motivating the order-insensitive/body-proportion controls here. [DPPD, AAAI 2026](https://doi.org/10.1609/aaai.v40i12.37927) uses personalized normal reconstruction and static/dynamic pathology residuals; it motivates a separate residual baseline but is neither SSL nor an MS study.

### Lessons from video JEPA

| Primary source | Published finding | Translation for gait |
|---|---|---|
| [V-JEPA](https://arxiv.org/abs/2404.08471) and [official JEPA repository](https://github.com/facebookresearch/jepa) | Predicts masked latent video regions from context with an EMA target and spatial multi-block masks repeated through the clip. In frozen K400/SSv2 evaluation, an attentive probe and multiple temporal clips substantially outperform average pooling and a single clip. | Connected anatomical graph masks are an analogy; adding bounded temporal spans is a new gait-specific hypothesis, not a literal reproduction of V-JEPA masking. Test an attentive multi-clip readout rather than assuming the video result transfers. |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) and [official repository](https://github.com/facebookresearch/vjepa2) | Uses 3-D RoPE over video time/height/width, longer clips/progressive training, and a four-layer attentive probe. A late-layer combination helped the reported Diving-48/Jester ablations. | Skeleton joint/time RoPE and layer mixing are transfer hypotheses—the paper's “3-D” axes are not 3-D anatomy. Test one-third, middle, and final skeleton layers independently before mixing them. |
| [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) | Adds dense prediction on visible and masked tokens plus deep self-supervision. Naive visible-context prediction creates a copying shortcut and sharply hurts global classification; distance weighting, coefficient warm-up, and multilevel prediction are needed together to recover it. | Treat this as a later local-feature branch and reproduce the whole anti-shortcut recipe if tested; it must not delay the core repairs. |
| [Interpreting Physics in Video World Models, 2026](https://arxiv.org/abs/2602.07050) | Speed/acceleration magnitude is decodable early; directional and possible/impossible information emerges around one-third depth, with physics-task performance peaking in middle layers. | Testing analogous layer behavior in skeleton gait is a transfer hypothesis; probe one-third, middle, and final features rather than assuming the final layer is best. |
| [V-JEPA intuitive physics](https://arxiv.org/abs/2502.11831) | Maximum surprise was stronger for single-video IntPhys classification, while average surprise was stronger for matched-pair evaluation. | Compare mean, variance, upper quantiles, and maximum per-limb residuals for intermittent gait abnormalities. No gait benefit is established; this is an exploratory transfer hypothesis. |

The clearest immediate video result concerns V-JEPA's frozen K400/SSv2 readout: average pooling materially understates that particular representation. Whether the same is true for skeleton gait must be tested. An official frozen RGB V-JEPA checkpoint can later provide a potentially complementary appearance/motion branch, but only after person cropping/background controls. Otherwise it may classify acquisition domain, clothing, resolution, or source rather than gait.

## Prioritized experimental program

### Phase 0 — Freeze the scientific unit and create shortcut baselines

Before changing the network:

1. Build `data_manifest.csv` with participant, source, clip interval, class, task, view, site/domain, original fps/resolution, duration, pose coverage, and extraction version.
2. Resolve participant identity and duplicates. Lock a participant-disjoint fold registry in JSON.
3. After identity resolution, prefer four outer folds if the eight PD source groups are verified as distinct participants and can place at least two PD participants in each test fold; use three grouped inner folds for model selection. If five folds are retained for continuity, report how many independent participants of each class occur in every fold.
4. Re-run the Random Forest and every shortcut baseline on this same registry; a result on different folds is not a paired comparison.
5. Save pooled out-of-fold predictions, not just fold averages.
6. Establish the following grouped controls on exactly the same folds:
   - duration/fps/resolution/visibility only;
   - mean pose and pose mean+standard deviation;
   - body-proportion only;
   - learned embedding after temporal frame shuffle and time reversal;
   - source/domain/view classifier;
   - nearest-source retrieval.

Output: `artifacts/eval/fold_registry.json`, a versioned data manifest, and a control-results table.

### Phase 1 — Repair S-JEPA while changing as little else as possible

First make two mechanical changes separately on inner-development data: `E1` adds only predictor position identity, and `E2` adds only variable masks plus the new mask API. These runs establish invariants; do not use outer-test performance to decide whether the correctness fixes remain.

Then create a cumulative `R1_repaired32` experiment that retains the 32-frame, 3-layer/96-width encoder:

- predictor joint/time position embeddings;
- stochastic graph-time masks, initially ratio 0.60 and two masks per sample;
- all sources in the current training partition for SSL (inner-training during selection, then all outer-training sources for the frozen refit);
- no class-aware VICReg;
- source-uniform batches;
- a learning curve at 300, 1,000, and 3,000 optimizer updates, effective batch 128 through gradient accumulation;
- AdamW, learning rate `3e-4`, weight decay `0.04`, 10% warm-up, cosine decay;
- EMA chosen by update half-life, initially `0.990 -> 0.9995`;
- frozen balanced linear probe using the existing mean pooling, so readout changes remain a later ablation.

Run one seed first to verify mechanics, then three seeds in inner folds. With only about 400 local windows, these budgets imply heavy reuse; record source exposures, nearest-source retrieval, and train/validation curves, and stop extending the curve when memorization rises without downstream improvement. Freeze `R1` before outer evaluation. Because `R1` also changes data use, loss, sampling, and optimization, its score measures the repaired *bundle* and cannot attribute a gain solely to the two structural bugs.

### Phase 2 — Build the accuracy-oriented local model

After Phase 1 passes, use the following as the accuracy-oriented search envelope. Introduce its window, readout, and layer changes sequentially according to the ablation ladder rather than switching every field at once:

```yaml
experiment: R2_local_gait_jepa
data:
  fps: 15                         # exact timestamp resampling
  channels: [x, y]                # visibility is reliability, not geometry
  window_frames: 64               # 4.27 seconds
  window_stride_for_eval: 32
  clips_per_source_per_epoch: 4
  padding_mask: true
tokenizer:
  frame_group: 4
  streams: [joints]
model:
  encoder_depth: 4
  encoder_dim: 128
  encoder_heads: 4
  predictor_depth: 3
  predictor_dim: 96
  position: factorized_joint_time_in_encoder_and_predictor
mask:
  strategy: connected_graph_time_blocks
  ratios_to_select_inner_cv: [0.40, 0.60, 0.75]
  masks_per_sequence: 2
  clinical_target_oversampling: 1.5
ssl:
  records: all_current_training_partition_unlabeled
  loss: centered_sharpened_latent_ce
  class_aware_vicreg: false
  update_learning_curve: [1000, 3000, 5000]
  extend_to_10000_only_if_inner_curve_improves: true
optimizer:
  name: AdamW
  effective_batch: 128
  lr_to_select_inner_cv: [0.0001, 0.0003, 0.0005]
  weight_decay: 0.04
  warmup_fraction: 0.10
  schedule: cosine
  grad_clip: 1.0
ema:
  start: 0.990
  end: 0.9995
readout:
  clips_per_source: 4
  token_pool_ablation: [mean_std, regional_mean_std, learned_queries]
  layer_ablation: [one_third, middle, final, learned_mix]
  probe: balanced_logistic
  probe_C_to_select_inner_cv: [0.01, 0.1, 1.0]
  optional_PCA_dims_to_select_inner_cv: [16, 32, none]
```

Keep the inner search small: choose the mask ratio first with fixed learning rate, then the learning rate for the winning mask. Select pooling/layer and probe regularization in a later pass; do not cross every knob. Fit scaling and PCA on inner-training records only. Record updates, examples seen per source, and teacher half-life in every run manifest.

The 64-frame model is intentionally moderate. The existing 8-layer/256-width GPU profile is roughly 26x the laptop model's parameter count and has a high source-memorization/overfitting risk when trained only on 35 sources. A `6 x 192` or `8 x 256` encoder becomes a defensible experiment in Phase 5, where the number of pretraining walks is orders of magnitude larger; the size choice remains an empirical hypothesis.

### Phase 3 — Preserve gait dynamics in the input and output

Add one component at a time to `R2`:

1. **Motion stream:** first temporal differences with valid-time masks, directly motivated by MAMP; add second differences only as a separate acceleration hypothesis.
2. **Bone/angle stream:** hip-knee-ankle, foot/shank, trunk, and left/right relational features. Avoid hand-coding the final answer; tokenize these alongside joints.
3. **Root stream:** pelvis displacement, root velocity, cadence, duration, and scale as explicit side information rather than losing them in framewise normalization. Require calibrated camera/task metadata and root-only/domain controls because this stream can be a shortcut.
4. **3-D skeleton branch:** first audit MediaPipe weak `z`/world landmarks, then compare a lifted 3-D or SMPL representation. The current visibility channel is not a substitute for the S-JEPA paper's 3-D skeleton input/target.
5. **Longer context:** compare 64 and 96 frames or 2–6 detected gait cycles. Do not resize all walks to the same duration without retaining original cadence.
6. **Temporal readout:** start with one learned query over token features, as in the V-JEPA probe, followed by attention/MIL over clips. Then test four region-specific queries as a new gait hypothesis. Include fixed left-leg, right-leg, trunk, and global regional summaries as a low-variance baseline.
7. **Prediction loss:** after the centered/sharpened S-JEPA baseline is stable, compare it with V-JEPA's plain L1 latent regression while holding masks, encoder, and update budget fixed. LayerNorm-normalized L1 and Smooth-L1 are additional local ablations, not V-JEPA reproductions.

Safe default augmentations are temporal crop, small in-plane camera transforms, mild coordinate noise, and realistic short joint dropout. Pace jitter, left/right mirroring, and aggressive rotation must be separate ablations because they can erase bradykinesia, asymmetry, or view information. Apply 3-D rotations only to true 3-D coordinates.

### Phase 4 — Add genuine supervised adaptation

The current “fine-tune” continues the pretext loss and then freezes the encoder. Compare three clearly named downstream regimes:

1. frozen encoder + balanced linear/attentive probe;
2. unfreeze the last one or two encoder blocks;
3. full encoder fine-tuning only after external pretraining.

Initial partial-fine-tuning configuration:

- balanced cross-entropy, label smoothing `0.05`;
- head learning rate `3e-4`, encoder learning rate `1e-5` to `3e-5`;
- layer-wise learning-rate decay `0.75`;
- label-aware class-then-source-balanced batches and clip-level augmentation;
- early stopping on inner-fold macro-F1 with a fixed maximum update budget;
- optionally add supervised contrastive or center loss at one small weight, never both initially.

The encoder/predictor SSL checkpoint must remain label-free. Supervised loss begins only in this phase, making frozen versus adapted results interpretable and making the label-efficiency experiment valid.

### Phase 5 — Scale data before scaling the network

The most plausible route to a large, genuine gain is:

```text
generic skeleton motion/gait pretraining
    -> CARE-PD pathological-gait adaptation
    -> fold-local unlabeled adaptation on this dataset
    -> supervised normal/MS/PD adaptation
```

Concrete sequence:

1. Map BlazePose-33, CARE-PD/SMPL, and external datasets to a common 17- or 22-joint topology; retain dataset-specific missingness masks.
2. Train/adapt the selected skeleton objective on CARE-PD if its license and intended use permit this project; CARE-PD does not provide a ready S-JEPA checkpoint. Never include a participant that overlaps an evaluation cohort.
3. Compare public, reproducible initialization paths based on NTU RGB+D/GaitForeMer-style motion forecasting, accessible walking data, and CARE-PD. Treat GaitPT as scaling evidence unless its private corpus or a compatible released checkpoint becomes available.
4. Use unlabeled outer-training videos for fold-local domain adaptation, without labels or test sources.
5. Only with external data, test `6 x 192` and then `8 x 256`; keep predictor narrower than the encoder.
6. Report in-domain, cross-domain, and leave-one-domain-out performance because clinical transfer can lose substantial accuracy across sites.

External-data provenance, licenses, skeleton mapping, and overlap checks must be stored with the checkpoint.

### Phase 6 — Separately reported S-JEPA + Random Forest fusion

The handcrafted branch already outperforms the current learned representation and encodes clinically meaningful angles/statistics. After a valid pure S-JEPA result exists, test a low-capacity inner-validated fusion of calibrated source-level probabilities, or concatenate standardized RF features with the frozen learned embedding under strong regularization. Compare RF-only, S-JEPA-only, and fused results on the same registry. This is a pragmatic accuracy branch, not evidence that S-JEPA alone improved.

### Phase 7 — Optional frozen RGB V-JEPA branch

This is a complementary branch, not the first fix to skeleton S-JEPA:

- use an official frozen V-JEPA 2/2.1 checkpoint;
- crop or segment the walking person and test masked/blurred backgrounds;
- sample multiple temporal clips at native timing;
- train only an attentive probe initially;
- compare RGB-only, skeleton-only, and late fusion of their source-level logits;
- require that performance survive background masking and domain/view controls.

Late fusion may improve the production classifier, but that is an engineering hypothesis and must be reported separately from the pure S-JEPA result.

## Ablation ladder and promotion rules

Screen each row against the immediately previous row on identical **inner-development folds** and training seeds. Outer folds are not a promotion leaderboard: evaluate only frozen, pre-registered phase winners there. A final ablation table may evaluate pre-registered variants on outer folds, but those results must not trigger another round of selection.

| ID | Change from previous row | Hypothesis | Promotion rule |
|---|---|---|---|
| E0 | Reproduce current capstone | Establish a deterministic reference | Predictions and metrics match within tolerance |
| E1 | Predictor position repair only | Hidden joint/time predictions become identifiable | Correctness tests pass; no collapse |
| E2 | Variable graph-time masks | All joints learn both context and target roles | Coverage/gradient gates pass and inner macro-F1 does not regress |
| E3 | Remove class-aware VICReg only | The current label-aware residual variance objective hurts transfer | Positive paired inner-fold delta or simpler equivalent result |
| E4 | Use all current-training-partition sources as unlabeled SSL input | More diverse fold-local motion improves transfer | Positive paired inner-fold delta over E3 |
| E5 | Source-uniform sampler only | Long videos should not dominate | Equal exposure test passes; inner result does not regress |
| E6 | 300/1k/3k/5k update learning curve | More optimization helps until source memorization dominates | Best inner point precedes or withstands shortcut/memorization rise |
| E7 | 64-frame context only | More gait cycles improve pathology signal | Inner macro-F1 and PD recall improve |
| E8 | Multiple temporal clips, retaining mean pooling | Temporal coverage matters | Beats single-clip mean on inner folds |
| E9 | Mean+std/regional pooling only | Dispersion and laterality matter | Beats E8 on inner folds |
| E10 | Learned/regional attentive readout only | Averaging obscures localized cues | Beats E9 across most inner seeds |
| E11 | Layer choice/mix only | Intermediate layers retain useful dynamics | A predeclared layer beats final-only in inner folds |
| E12 | Last-block supervised adaptation | Disease-specific features require encoder adaptation | Positive paired inner delta without calibration collapse |
| E13 | Motion, bone, then root streams separately | Explicit dynamics improve pathology discrimination | Temporal shuffle causes a larger, not smaller, drop |
| E14 | 3-D/world/lifted target | Depth and joint angles reduce view dependence | Improves inner ordinary and leave-one-view/domain validation |
| E15 | CARE-PD/external motion adaptation | Data scale and clinical motion transfer drive a larger gain | Beats local-only E14 in inner validation |
| E16 | S-JEPA + RF feature/logit fusion | Clinical handcrafted and learned features are complementary | Gain over both branches in inner folds; report separately |
| E17 | Frozen RGB V-JEPA late fusion | Pixels add complementary motion/appearance cues | Gain survives inner background/domain controls |

Suggested decision thresholds—not promised outcomes:

- **Mechanically valid:** all position, mask coverage, gradient, padding, timing, and split tests pass.
- **Promising local representation:** inner macro-F1 improves by at least 0.05 over a rerun of E0 and over the strongest reproducible low-order/acquisition control, with improvement in at least four of five seeds. This is a promotion rule, not an outer-test claim.
- **Competitive with the current RF:** after freezing the configuration, its paired pooled OOF macro-F1 is at least as high as the RF rerun on the new registry, with a target PD recall of at least 0.60 and participant-bootstrap deltas reported. Do not compare the future pooled score with today's mean-of-fold RF `0.668`.
- **Credible temporal gait signal:** temporal shuffle/reversal materially degrades the promoted model, while fps/domain prediction and order-insensitive pose moments do not explain its full score.

Because the dataset is tiny, a data-cleaning correction may initially lower accuracy by removing shortcuts. That is scientific progress even if it delays the competitive threshold.

## Evaluation protocol for every promoted experiment

1. Split by verified participant, not merely clip or nominal source.
2. Use a locked grouped outer fold registry and grouped inner folds for selection.
3. Within each inner split, SSL sees only inner-training sources; after selecting a configuration, refit it on all outer-training sources. Never use outer-test videos even unlabeled for the inductive result.
4. Repeat training with at least three seeds for screening and five for a final result.
5. Tune only in inner folds; never choose masks, checkpoints, layers, or pooling from outer-test labels or test silhouette.
6. Aggregate windows to clip and then participant. Give each participant equal weight.
7. Save OOF class probabilities and report:
   - macro-F1 as primary;
   - balanced accuracy and ordinary accuracy;
   - per-class precision/recall/F1, especially PD recall;
   - normalized and count confusion matrices;
   - one-vs-rest AUROC/AUPRC where estimable;
   - calibration/Brier score;
   - participant-level bootstrap confidence intervals and paired deltas to RF/current S-JEPA.
8. Report model-selection variance separately from fold composition variance.
9. For label efficiency, use several group-stratified labeled-subset draws at 10%, 25%, 50%, and 100%; all other outer-training clips may be used only unlabeled. Enforce at least two training groups per class (and enough groups for the chosen inner split); omit a fraction that cannot meet this support rather than silently rounding it up.
10. Freeze a final configuration before any one-time external holdout or cross-domain test.

## Required diagnostics and tests

Add these before launching a long run:

- `test_hidden_target_predictions_have_position_identity`
- `test_every_joint_receives_context_gradient_over_mask_bank`
- `test_mask_bank_meets_context_and_target_coverage`
- `test_per_example_multimasks_and_context_padding_are_respected`
- `test_source_sampler_is_uniform_despite_window_count`
- `test_timestamp_resampling_hits_requested_times`
- `test_padding_is_excluded_from_attention_and_pooling`
- `test_participant_groups_are_disjoint`
- `test_visibility_is_not_named_or_used_as_depth`
- `test_save_resume_preserves_center_optimizer_schedule_and_predictions`
- embedding per-dimension standard deviation, covariance, singular values, and effective rank
- teacher/student distance and EMA target drift
- per-joint gradient and attention summaries
- class-agnostic source/domain/view probe
- single-frame/body-proportion, order-insensitive pose-moment, time-shuffle, time-reversal, confidence-only, and root-only controls

Loss decreasing and teacher/student weights differing are necessary but insufficient. A collapsed or shortcut representation can pass both current sanity checks.

## Implementation map

| Area | Main files | Intended change |
|---|---|---|
| Configuration | `sjepa/config.py` | Separate correctness/local/transfer profiles; schedules in steps; data, mask, positional, and readout settings |
| Token positions | `sjepa/tokenizer.py`, `sjepa/models.py` | Predictor joint/time positions, feature access by layer/token, padding masks, fine-tunable encoder |
| Masking | `sjepa/masking.py` | Per-example `(B,M,N)` connected graph-time masks, context padding/attention masks, coverage accounting |
| Objectives/state | `sjepa/losses.py`, `sjepa/train.py` | Remove mislabeled class-aware VICReg baseline; persist center/optimizer/scheduler/global step/RNG; gradient accumulation; teacher diagnostics; explicit supervised adaptation |
| Data | `sjepa/data.py`, `scripts_extract_all.py` | Timestamp resampling, metadata, short-gap cleaning, validity/padding masks, source-uniform sampling, optional world/3-D channels |
| Evaluation | `sjepa/eval.py`, `scripts_capstone_check.py` | Locked nested group folds, OOF probabilities, participant aggregation, bootstrap deltas, shortcut controls |
| Tests | `sjepa/tests/` | All invariants listed above |
| Notebooks | `notebook_content.py`, then regenerated notebooks | Explain repaired method honestly and make each experiment/config auditable |

Every run should save a machine-readable manifest containing git revision, data-cache version, participant fold IDs, config, seed, number of optimizer updates, effective batch, examples/source exposures, mask coverage, checkpoint-selection rule, and OOF predictions. Continuation checkpoints must also contain the running center, optimizer, scheduler/global step, EMA position, RNG, and sampler state.

## Directions to defer

Do not prioritize these until the repaired local baseline through E12 is complete:

- the current large GPU profile trained from scratch;
- a 90% mask simply because the action-recognition paper used it;
- V-JEPA 2.1 dense/deep auxiliary losses;
- complex fusion before a valid pure skeleton baseline;
- t-SNE/UMAP aesthetics as evidence of separability;
- test-set silhouette or manual inspection for model selection;
- further label-aware regularizers without a clean cross-entropy baseline;
- claims that normal-only pretraining is inherently preferable for three-class classification.

## Immediate next run checklist

The next concrete milestone is `R1_repaired32`, not a full literature-inspired rewrite:

- [ ] lock and export the current source-group folds and OOF baseline for reproducibility, without calling them participant-disjoint until identity is verified;
- [ ] add predictor spatial/temporal positions and its regression test;
- [ ] add a variable mask bank and gradient/coverage tests;
- [ ] remove class-aware VICReg;
- [ ] persist the center, optimizer/schedule, global step, EMA position, RNG, and sampler state across true continuation stages;
- [ ] pretrain on all sources in each current training partition with a source-uniform sampler;
- [ ] trace 300/1,000/3,000-step learning curves with a responsive EMA, source-exposure counts, and collapse/memorization diagnostics;
- [ ] run single-frame/body-proportion, order-insensitive moment, visibility, time-shuffle, and domain controls;
- [ ] freeze the cumulative R1 configuration before outer evaluation;
- [ ] promote to 64 frames only if the repaired baseline is mechanically sound.

That cumulative experiment will show whether a repaired local recipe closes a useful portion of the gap, although it will not identify one cause by itself. If it does, proceed through the pre-registered pooling and partial-adaptation ablations. If it does not, avoid blind local model scaling and prioritize preprocessing plus external clinical-motion adaptation, where the literature suggests the largest credible gain should come from.
