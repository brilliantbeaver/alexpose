# Evidence base and common experimental protocol

This document separates executed evidence, literature facts, and proposed experiments. The separation matters because the foundation notebooks contain a working historical curriculum, but not a clean estimate of unseen-source clinical classification. Every code cell in notebooks 00 through 06 has a saved execution count and no saved Python error. They are still historical transcripts, not a verified current rerun. Several saved paths point to older machines or `gavd5`, current result artifacts are absent, and the repository README requires new state-hash, pooling, and missingness-control runs before reusing the classifier readouts.

## 1. What S-JEPA predicts

A joint-embedding predictive architecture hides some inputs and predicts their internal representations. It does not need to reconstruct pixels or coordinates. In S-JEPA, a student encoder sees visible joint-time tokens, a slowly updated teacher encoder sees the full sequence, and a predictor estimates teacher latents at the hidden positions. The original S-JEPA paper evaluates skeletal action recognition on clean 3D datasets and reports strong transfer, but it does not study clinical gait, in-the-wild 2D pose, calibrated uncertainty, or counterfactual output ([S-JEPA paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf)). No author-linked official checkpoint was verified, so this repository uses its own implementation.

The distinction between a representation learner and a world model is practical. The current local model predicts masked latent tokens inside one observed window. It is not a generative gait simulator and it does not infer forces. Proposals may still use its conditional prediction as a measurement, but they must not describe it as a physical simulator.

## 2. Notebook 00 to 06 synthesis

### Notebook 00: model contract

`00_sjepa_from_first_principles.ipynb` defines a 64-frame sequence, four-frame temporal patches, and 33 BlazePose joints, giving 528 possible tokens. Only shoulders and lower-body joints, indices 11, 12, and 23 through 32, are eligible for masking. An executed sanity check produced prediction and target shapes `(2, 57, 32)`, finite loss, and zero target-encoder gradients. The realized global mask fraction was 0.216 because 60 percent masking applies only to the eligible subset.

**Reusable evidence:** the student, EMA teacher, predictor, mask contract, and anti-collapse checks already work.

**Boundary:** the notebook is a mechanism check, not downstream evidence.

### Notebook 01: historical cohort

`01_gavd_manifest_and_youtube.ipynb` builds a 96-sequence manifest from 18 videos: 12 normal, 9 Parkinsonian, 12 stroke, 47 myopathic, and 16 cerebral-palsy sequences. The normal sequences come from one source, Parkinsonian from two, stroke from three, cerebral palsy from two, and myopathic from ten.

**Reusable evidence:** source videos are cached once and original frame indices are preserved.

**Boundary:** class and source are heavily entangled. The historical cohort cannot support independent source-held five-class claims.

### Notebook 02: pose extraction

`02_extract_and_watch_skeletons.ipynb` extracts MediaPipe poses within annotated person boxes and preserves missing frames instead of compressing time. All 96 locked files passed its historical training-readiness audit. The extant pose files cover 22,233 frames. Mean any-pose coverage is 0.992, with one stroke sequence at 0.569. The executed preview reported lower visibility at heels than at shoulders, hips, knees, and ankles. Detector behavior is therefore part of the observation, not neutral preprocessing.

**Reusable evidence:** bounding-box guided extraction, validity arrays, video overlays, and provenance can be reused.

**Boundary:** pose missingness can itself predict labels. Every new model needs a missingness-only control.

### Notebook 03: restricted masking

`03_neurologic_keypoint_masking.ipynb` verifies that only the 12 declared shoulder and lower-body joints can be hidden. In an executed batch, 59.5 percent of valid eligible tokens were masked, which was 21.6 percent of the full token grid. Every forbidden token remained visible.

**Reusable evidence:** anatomically restricted, validity-aware mask sampling is implemented.

**Boundary:** a configured 60 percent mask is not 60 percent of all tokens. Reports must state both eligible and global fractions.

### Notebook 04: label-aware staged training

`04_pretrain_sjepa_on_normal.ipynb` uses 12 canonical normal sequences plus 63 accepted windows from 17 additional YouTube videos. Only normal receives this separate auto-bounding-box acquisition path. Stage 0 trains for 300 epochs on 75 normal sequences across 18 videos. Four later stages add Parkinsonian, stroke, myopathic, and cerebral-palsy data for 75 epochs each with balanced replay. The loss combines JEPA, a VICReg term, and condition compactness and separation terms. The condition terms use stage labels after Stage 0.

The final executed augmented checkpoint has fingerprint `13069dac0e6d655a6f0aa3c2c2f43fd66530a69d65e167da2794c129a0db2772`. Its last-stage JEPA loss was 0.5584 and feature standard deviation was 0.4135.

**Reusable evidence:** curriculum checkpoints, balanced replay, explicit fingerprints, and representation diagnostics are useful engineering.

**Boundary:** after Stage 0 this is supervised, label-aware representation training. Any evaluation on those same sequences is transductive.

### Notebook 05: weak latent geometry

`05_inspect_latent_motion.ipynb` pools each sequence into a 384-dimensional vector. Its mean masked-token cosine of 0.597 comes from only the first four canonical clips, all normal and already exposed to the encoder. It is not a held-out prediction result. Feature geometry over all 96 sequences has mean within-condition cosine distance 0.118, minimum between-centroid distance 0.027, and cosine silhouette 0.028. Those values do not show clean class geometry. A normal-anchor distance separates the exposed labels descriptively, but the checkpoint was trained with those labels.

**Reusable evidence:** token-level and pooled diagnostics, nearest neighbors, and normal-anchor calculations can become baselines.

**Boundary:** normal distance is not severity, and exposed-class geometry is not generalization.

### Notebook 06: descriptive readouts and leakage audit

`06_capstone_health_condition_classifiers.ipynb` fits Random Forest readouts on frozen 384-dimensional features. On a sequence-stratified, video-confounded split, the all-96 readout reached 0.724 accuracy and 0.754 macro-F1. The exact historical split reached 0.667 accuracy and 0.698 macro-F1. A missingness-only model reached 0.414 and 0.388 on the all-96 split. An 82-feature handcrafted model reached 0.762 accuracy and 0.728 macro-F1 on the historical split.

The notebook explicitly records that every classifier test sequence was seen during label-aware representation training. Its missingness control uses 97 aggregate features, not the complete joint-by-time missingness pattern. Its later source-held classifier folds still use that exposed encoder, so they are not a clean estimate of representation generalization. The notebook's honest label-blind source-held lane was not run.

**Reusable evidence:** the notebook correctly names the leakage and supplies strong shortcut baselines.

**Boundary:** none of its headline readouts is an unseen-source, label-blind encoder result.

## 3. Active repository results that constrain the proposals

The active Core11 transfer study is more pessimistic than the historical notebooks. On the strict 90-frame, no-padding cohort, raw Core11 reached mean macro-F1 0.423. The EMA paired shared/no-cross encoder reached 0.234 and the EMA reflection-equivariant encoder reached 0.245. Randomly initialized controls ranged from 0.334 to 0.537. The split also shared source videos, so even the raw advantage is descriptive. These results rule out “freeze S-JEPA, add a linear head” as a meaningful new contribution.

The laterality program built a careful AMASS corruption benchmark. SG-JEPA improved over correction-first S-JEPA, but an otherwise identical model with fixed 50/50 path uncertainty matched it. Side-sensitive normalized error was 0.8671 for SG-JEPA and 0.8668 for the uniform control. Side-insensitive error was 0.6696 and 0.6690. Lower is better. The proposed informative-probability mechanism is therefore unsupported, and the sealed test remains unopened.

All 24-participant StrokePIG frozen-representation probes had negative held-out R-squared. This is further evidence that training loss reduction and plausible latent plots do not establish useful transfer.

## 4. Full GAVD audit

The checked-in manifest reproduces 1,874 unique sequences, 348 source videos, and 458,116 annotated frames. It contains no participant ID, official split, severity score, affected side, verified diagnosis, or clinic-versus-wild field. The binary `dataset` field and `gait_pat` field disagree for 37 sequences from five sources: those rows say `gait_pat=normal` and `dataset=Abnormal Gait`. One source contains both cerebral-palsy and generic-abnormal sequences.

The full observational taxonomy is:

| `gait_pat` | Sequences | Videos |
| --- | ---: | ---: |
| abnormal | 767 | 117 |
| normal | 291 | 32 |
| exercise | 234 | 98 |
| myopathic | 188 | 30 |
| style | 104 | 3 |
| stroke | 76 | 19 |
| cerebral palsy | 64 | 11 |
| parkinsons | 47 | 11 |
| prosthetic | 39 | 8 |
| antalgic | 35 | 10 |
| inebriated | 23 | 8 |
| pregnant | 6 | 2 |

The central confound is stronger than class imbalance: 347 of 348 source videos contain exactly one `gait_pat` label. One source has two. GAVD therefore lacks within-source counterexamples for almost every label. This motivates SourceSwap. Profile-ID decoding is a clean diagnostic on paired, same-motion AMASS replicas. On real GAVD, source identity may also encode a person's legitimate motion, so the audit instead decodes explicit nuisance attributes conditional on raw kinematics.

Camera view is useful for stratification, not paired multiview supervision. Many source videos contain more than one view label, but only a handful of annotated intervals overlap across different views. Eleven sequence intervals change view internally and must be split, assigned frame-level view, or excluded from view-specific tests. The release also has gait-event labels on only 758 of 458,116 frames, so phase must be estimated without event supervision. Nine sources have incompatible annotation geometries. They are excluded from the primary analysis and restored only in a sensitivity analysis.

The GAVD abstract and result table reverse the model-name order, but together report binary accuracy in the 92 to 94 percent range for Kinetics-pretrained TSN and SlowFast. That closes the headline task regardless of which number is attached to which model ([GAVD paper](https://arxiv.org/pdf/2407.04190)). The same paper reports large view dependence. The proposals target a six-presentation task and measurements beyond class accuracy.

## 5. What recent work already occupies

- [GaitForeMer](https://arxiv.org/abs/2207.00106) shows that motion forecasting pretraining can improve few-shot Parkinsonian severity estimation. A generic forecasting-transfer curve is not new.
- [Zero-shot Gait Classification with Diffusion Models](https://openreview.net/forum?id=L5xyzjMCwd) uses Human Motion Diffusion Model denoising error as a Parkinsonian gait score. A scalar motion-prior surprise score is not new.
- [GaitDynamics](https://www.nature.com/articles/s41551-025-01565-8) provides a public generative prior for laboratory kinematics and predicted ground-reaction forces. It is a scorer or teacher here, never GAVD ground truth.
- [GaitEncoder](https://www.medrxiv.org/content/10.64898/2026.07.07.26357479v1) provides a public 16-dimensional clinical gait latent and a deviation-from-mean-unimpaired score. Static normative distance is occupied.
- [CARE-PD](https://arxiv.org/abs/2510.04312) provides a public multi-site Parkinsonian SMPL benchmark. Generic cross-site severity probing is occupied, but it is useful external validation for proposal 4.
- [GAITGen](https://openaccess.thecvf.com/content/WACV2026/html/Adeli_GAITGen_Disentangled_Motion-Pathology_Impaired_Gait_Generative_Model_--_Bringing_Motion_WACV_2026_paper.html) generates severity-conditioned impaired gait. Severity-conditioned generation is occupied and its checkpoint is not publicly released.
- [SleepFM](https://www.nature.com/articles/s41591-025-04133-4) shows the value of large frozen representations, leave-one-modality-out alignment, and small downstream heads. This portfolio transfers the adaptation pattern, not its sleep model.
- [ControlNet](https://arxiv.org/html/2302.05543v3) supplies the frozen-base, zero-initialized residual pattern used by several proposals.
- [Goal Force](https://arxiv.org/pdf/2601.05848) and [Masked Visual Actions](https://arxiv.org/html/2607.19343v1) show how explicit controls or masked entities can turn a large video prior into a queryable model. Their released adapters are public, but Masked Visual Actions used eight H200 GPUs for about four days. “15 hours” describes its video data, not training time.
- [Sufficient Input Subsets](https://proceedings.mlr.press/v89/carter19a.html) and [TimePNS](https://arxiv.org/abs/2607.21573) already cover minimal sufficient and counterfactually necessary explanations. That object was removed from this portfolio.
- [Action Motifs](https://openaccess.thecvf.com/content/CVPR2026/html/Kinoshita_Action_Motifs_Self-Supervised_Hierarchical_Representation_of_Human_Body_Movements_CVPR_2026_paper.html) already learns motion atoms and motifs with masked latent prediction. A motif dictionary was removed.
- Robust and equivariant conformal methods already aggregate scores across label-preserving transformations. Transformation-stable prediction sets remain a possible evaluation layer, not a proposal.
- [V-JEPA 2](https://arxiv.org/abs/2506.09985) supplies a public frozen video representation and demonstrates post-training with limited interaction data. It remains an admissible RGB baseline, not a gait expert or a dependency of the seven proposals.

## 6. Common data contract

### Primary inferential cohort

Use six observed presentation labels: normal, myopathic, stroke, cerebral palsy, parkinsons, and antalgic. The target is defined from `gait_pat`, so its value wins over the binary field in the primary analysis. Excluding the nine dual-geometry sources leaves 291 normal sequences from 32 sources, 155 myopathic from 28, 76 stroke from 19, 56 cerebral-palsy from 10, 47 parkinsons from 11, and 35 antalgic from 10. Five fixed folds with at least two held-out sources from every class are therefore feasible. As a label-conflict sensitivity analysis, remove all five sources containing the 37 disputed normal rows.

Lock these outer folds before learning observation profiles or extracting any fold-dependent statistics. No sequence or profile from a held-out source may train or select an encoder, adapter, normalizer, source generator, phase rule, covariance model, query policy, gate, or threshold for that fold. Nonlearned test-time preprocessing may inspect the current test clip, but it cannot update shared parameters.

### Secondary cohorts

- `abnormal`: open-set and generic-presentation stress test only.
- `prosthetic`: descriptive body-model failure audit only.
- `exercise`, `style`, `inebriated`, and `pregnant`: out-of-taxonomy stress tests.
- CARE-PD: external severity validation for proposal 4.
- AMASS: known-camera, known-corruption, and known-edit validation for every mechanism.

### Pose contract

No full-corpus pose manifest exists yet. The to-be-built manifest must record source video, sequence interval, frame index, bounding box, 2D landmarks, confidence, validity, extractor version, source checksum, and locked outer fold. The primary 2D route remains available even when a 3D lift fails. Any 3D proposal records the lift, body model, retargeting residual, and alternate-lift disagreement.

The active Core11 bridge outputs pelvis plus bilateral hip, knee, ankle, heel, and forefoot points. It subtracts the pelvis every frame, divides by robust bilateral leg length, and resamples to 30 Hz. The pelvis coordinate is therefore identically zero and translation, absolute scale, and original sampling rate are unavailable to the encoder. Core11 also has no shoulder or trunk token. Proposed measurements must use observable hip and leg geometry or explicitly add a separately validated input route.

### Metrics

Primary label metric: macro average precision across the six labels, pooled at source level. Secondary metrics: macro-F1, class-balanced accuracy, and sequence-level results with a source-cluster bootstrap. Each proposal also has a mechanism metric that can fail even when label accuracy rises.

### Shortcut model

Fit one strong shortcut-only model inside every training fold using duration, first and last frame, box area, foreground area, centroid drift, source dimensions, view, confidence summaries, missingness summaries, cadence, camera motion, and a static background embedding. Foreground area and static background require a locked person-segmentation and background-sampling pipeline recorded in the full manifest. The proposed representation must add information conditionally and must not simply improve nuisance recovery.

## 7. Shared training and compute rule

The conservative local anchor is 3 H100-hours for one 100-epoch JEPA run. A 25-epoch JEPA-equivalent trainable branch costs about 0.75 H100-hours on one GPU. Five folds and three seeds therefore cost 11.25 H100-hours per trainable branch. Independent folds, seeds, and adapter arms can run across eight GPUs, but GPU-hours are never divided by eight. Frozen-model inference is benchmarked on day 1 rather than converted from another GPU type, and every proposal states a maximum query count and GPU-hour stop limit.

Each proposal states:

1. one best two-week experiment with a decisive primary endpoint;
2. a numerical or structural stop rule;
3. the maximum trainable component;
4. the public checkpoint path or local checkpoint used;
5. an endpoint that remains useful under a null result.

## 8. Claim language

Use “observed presentation association,” not diagnosis. Use “model response,” not patient response. Use “predicted force,” not measured force. Use “unseen source,” not unseen participant. Use “continuous expression axis,” not clinical severity, unless CARE-PD supplies an actual severity label. These phrases are part of the experimental contract.
