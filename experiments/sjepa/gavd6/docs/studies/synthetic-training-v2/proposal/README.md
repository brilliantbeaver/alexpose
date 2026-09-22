# Correct pose errors without erasing gait

*Research proposal · Updated 20 September 2026 · Study design; all illustrations are conceptual.*

## Introduction

Blur and partial obstruction can make pose-tracking software report movement that did not occur. Using nearby frames can correct these errors, but excessive smoothing may erase movement amplitude, timing or left–right coordination.

We study paired synthetic supervision: a body model supplies projected reference joints, while frozen image-based pose estimators supply imperfect tracks from matching rendered images. These references are **synthetic anatomical proxies**, not independently verified image-keypoint annotations.

The question is: **can paired synthetic motion train a temporal skeleton JEPA to restore imperfect pose tracks while preserving movement better than direct denoising?** The task is offline 2D sequence restoration using the full observed window. Clinical utility requires separate evidence.

![Three conceptual trajectories distinguish noisy observations, faithful restoration and oversmoothing.](images/01-preserve-motion.svg)

*Figure 1. A smoother trajectory can still be wrong. Restoration must retain reference movement amplitude and timing.*

The proposed contribution is a controlled test of **restoration versus motion distortion**, including whether latent prediction adds value beyond equally informed coordinate learning. One alternative explanation to test is that fitting a coordinate readout corrects systematic differences between projected body-model joints and image-estimator landmarks. Calibration controls will help separate that possibility from the value of temporal representation learning.

## Methodology

**1. Construct matched observations.** Screen AMASS motion using declared geometry and alternating-leg-motion heuristics, then audit locomotion, overlays and anatomical correspondence before substantive gait-preservation or transfer claims. Render the same SMPL-H motion, timestamps and frontal camera under clean, blurred and lower-body-obstructed conditions. Reserve blur-plus-obstruction for development evaluation. Use frozen HRNet-W32, RTMPose-m and ViTPose-Base estimators to extract 12 common body joints, native scores and missing-observation flags. Each window contains 64 samples at 25 Hz, spanning 2.52 seconds.

Train on HRNet and RTMPose tracks from training people. Exclude ViTPose from fitting and evaluate it on development people; this exclusion does not certify its prior training exposure as untouched. Keep all windows and render variants from a person in the same split. Keep projected reference joints separate from inference inputs, and compute whole-window input normalization from observed tracks only.

![Matched renders share motion and camera; estimated tracks are inputs and projected joints are privileged targets.](images/02-paired-data.svg)

*Figure 2. Paired-data design. Change observation conditions while holding movement fixed. Automated screening supports feasibility checks; locomotion, overlay and anatomical audits are required for stronger evidence.*

**2. Learn restoration representations.** An online temporal encoder processes imperfect tracks. A predictor matches masked features from an exponential-moving-average (EMA) teacher receiving same-view clean projections. This uses **privileged synthetic supervision**. After pretraining, freeze the encoder and fit a temporal coordinate readout on training pairs. Deployment uses observed tracks only; missing joints receive prediction queries without revealing target validity.

![Separate paths show paired JEPA pretraining and deployment through a frozen encoder and coordinate readout.](images/03-paired-jepa.svg)

*Figure 3. The teacher guides training; it is absent at deployment. The candidate is a local S-JEPA-inspired adaptation, not a new official S-JEPA release.*

**3. Measure accuracy and preservation.** Coordinate error is Euclidean pixel distance divided by the reference-box diagonal, with missing predictions penalized. The scale is independent of model predictions, not an independent anatomical annotation. Motion endpoints assess 0.20-second displacement, signed horizontal ankle separation, its demeaned RMS amplitude, and supported positive-maxima timing. These are image-plane quantities, not metric stride length or heel strikes. Missing support cannot count as preservation success. Examine clean and corrupted inputs separately, and keep synthetic hidden-joint scores separate from independently annotated real-visible scores.

Balance repeated variants and windows within motions, then motions within people. Estimate paired uncertainty by resampling people with motions nested within them while retaining render variants together. Report each extractor separately, and distinguish training-seed variability from sample uncertainty. Save predictions, references, masks, timestamps and group identifiers so every metric can be reconstructed.

## Experiments

**1. Validate the paired-data pipeline.** Use an initial feasibility configuration of two training people, two development people, two windows per person and seed 17. Evaluate four rendering conditions and three extractors on development people. The resulting 48 records per method are correlated observations from two evaluation people, not 48 independent samples. Check timing alignment, split integrity, missing-observation handling and separation of inference inputs from privileged targets. Automated screening alone cannot establish gait preservation or anatomical accuracy.

**2. Compare objectives and simpler explanations.** Compare paired JEPA with coordinate pretraining followed by a separately fitted frozen-encoder readout (`coordinate`), end-to-end coordinate restoration (`direct`), and a fitted readout on an untrained frozen encoder (`initialized`). Ordinary JEPA predicts observed-track latent targets; paired JEPA uses aligned synthetic targets; shuffled JEPA changes the pretraining pairing. All three JEPA arms subsequently fit coordinate readouts. Keep shuffled donors within training people and nuisance strata, exclude the same source window, and retain natural pairs for evaluation.

Include unchanged tracks, fixed filters at strengths 0, 1 and 2 (`filter0` is unchanged), a SmoothNet-style temporal MLP adaptation, and `static`. The static control removes neighboring coordinates after shared whole-window normalization while retaining auxiliary channels; it does not remove every source of cross-frame information. Treat the SmoothNet-style model as a local adaptation rather than an official reproduction.

For the paired-JEPA versus coordinate-pretraining contrast, match training pairs, masks, update counts, seeds, encoder/readout capacity, clean-label access and tuning opportunities. Separately compare practical methods at equal total compute, including pretraining and readout fitting; equal update counts do not establish equal training cost.

![Coordinate pretraining, paired JEPA and practical denoisers share inputs and reference-based evaluation.](images/04-fair-comparison.svg)

*Figure 4. Comparison design. Coordinate pretraining tests whether latent prediction adds value; the initialized readout tests whether pretraining is needed. A separate comparison matches total compute.*

**3. Test calibration.** Fit a pooled per-joint constant offset and a small affine/ridge correction using training people and training extractors only. Apply them unchanged to development people and held-family ViTPose. Compare coordinate and motion errors with initialized, coordinate, direct and paired JEPA to test how much restoration can be explained by a simple coordinate correction. Any anatomical-convention explanation requires independent validation.

**4. Establish attainable timing support.** Score reference coordinates against themselves. Separate reference-ineligible records from incomplete predictions and peak-count mismatches; report missed and extra peaks alongside timing error and coverage. Test whether the frontal view provides sufficient horizontal ankle-separation variation for the declared timing endpoint. Unsupported timing must remain missing evidence rather than being scored as zero error.

**5. Inspect motion and convergence.** Plot reference, unchanged, calibrated and neural ankle trajectories for the same windows selected without looking at method performance, separating clean and corrupted conditions. Inspect displacement error, amplitude ratios, per-joint residuals and each training phase’s loss history. These checks are intended to distinguish calibration, insufficient optimization, weak observability and motion distortion.

**6. Repeat and test transfer.** After validating the data and endpoints, expand the number of independent people and repeat the strongest controls and candidate with seeds 17/29/43. Use independent anatomical review and dense real-video temporal references to evaluate transfer and motion preservation. Calibrate decision margins on development data and annotation repeatability before freezing the method, metrics and protocol for independent confirmation. Unknown identity reservations or prior exposure cannot be treated as cleared. Declare any revised metric or camera design as a new development experiment.

![Evidence progresses from pair validation to method comparison, repeats, real development, protocol freezing and independent confirmation.](images/05-evidence-workflow.svg)

*Figure 5. Planned evidence sequence. Repeated-seed comparisons, real temporal validation and independent confirmation follow pair and endpoint validation. A useful simpler method can proceed even if JEPA adds no benefit.*

*Study detail: [research plan](../../../../notes/prompts/03_improvement_plan.md) · [development protocol](../protocol.md) · [HAIC execution guide](../../../../slurm/synthetic-training-v2/README.md).*
