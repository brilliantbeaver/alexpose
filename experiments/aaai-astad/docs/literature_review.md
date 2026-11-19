# Reading S-JEPA carefully before studying gait

This review was checked against primary papers and available official code on November 14, 2026. It separates published methods from the gait experiments proposed here. A classifier trained on this cohort predicts an annotated gait category. It does not establish a person's diagnosis.

## 1. The local background and a naming trap

The requested `/Users/pmui/vaults/worldmodels/gait/skeleton-jepa/background` path was absent. The corresponding material was found and read at `/Users/pmui/dev/worldmodels/gait/skeleton-jepa/background/s-jepa-mamp-literature-review.md`, together with its `ieee-reference-audit.md`. The local review explains the distinction between a mask, which chooses a question, and a target, which defines an answer. It also warns that action recognition results do not establish clinical gait validity. Those distinctions guide this tutorial.

Two unrelated papers now use **S-JEPA**. This project uses **Skeleton S-JEPA** for Abdelfattah and Alahi's ECCV 2024 skeleton method. It uses **stochastic-latent JEPA** when discussing the November 2026 image method by Giakoumoglou and colleagues. Neither should be confused with speech or spectral methods sharing the acronym. The [Skeleton S-JEPA project](https://sjepa.github.io/) currently links a paper and an illustrated method summary. A matching official skeleton code repository was not established during this review. Our implementation is therefore a paper-based adaptation, not a claim of an exact official reproduction.

## 2. What the published methods do

### Skeleton S-JEPA

The [ECCV paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) groups adjacent frames of one joint into a token. Its complete skeleton enters a target transformer. The selected target features therefore retain context from the whole sequence. A transformed, masked view enters the trainable view encoder. The predictor receives visible features and placeholders at missing positions. It predicts features rather than coordinates.

The published loss matches centered, sharpened target distributions with predicted distributions using cross-entropy. The target encoder has no gradients and follows an exponential moving average, or EMA, of the view encoder. The paper uses motion-aware masking, 120-frame inputs, four-frame tokens, an eight-layer encoder, a five-layer predictor and width 256. Its teacher and prediction temperatures are 0.06 and 0.1. After pretraining, its downstream representation comes from the target encoder. These action-recognition experiments do not establish disease recognition or clinical calibration.

### MAMP

[Masked Motion Predictors are Strong 3D Action Representation Learners](https://arxiv.org/pdf/2308.07092) trains an encoder and decoder to predict measured temporal displacement at masked skeleton positions. Its targets come from coordinate differences. It has no learned EMA target encoder. Motion also guides masking: larger movement receives greater masking probability. These are separate uses of motion. Changing the mask does not change the meaning of the target.

The [official MAMP implementation](https://github.com/maoyunyao/MAMP/blob/main/model_mamp/transformer.py) contains both `random_masking` and `motion_aware_random_masking`. Its uniform branch ranks independent uniform random numbers and retains a fixed number of tokens. The motion-aware branch combines motion probabilities with Gumbel noise. `forward_encoder` uses the uniform branch when `motion_aware_tau <= 0`. Its decoder restores missing positions with placeholders and predicts coordinate-difference targets. This supports a precise description of our experiment: **uniform MAMP-style masking with Skeleton S-JEPA latent targets**. It would be inaccurate to describe uniform masking as the default published MAMP strategy.

### Stochastic positional embeddings, or StoP

The supplied OpenReview URL returned a browser verification page. The corresponding published primary source is [Stochastic positional embeddings improve masked image modeling, ICML 2024](https://proceedings.mlr.press/v235/bar24a.html). StoP makes the positional conditioning of masked image tokens stochastic. Its aim is to reduce excessive dependence on precise target locations. It does not shuffle anatomical labels and does not create a learned conditional distribution over missing motion.

The [official code](https://github.com/amirbar/StoP/blob/9380e4c85e172b6dcfb6610049300a983f735df2/src/deit.py) projects visible features and Gaussian noise through the same predictor projection. In compact notation, visible predictor tokens are `position + A(context)`, while target requests are `position + A(noise)`. Noise enters target requests, leaving target-encoder inputs unchanged. Sharing `A` makes suppressing all noise costly because it also suppresses context. The code's `noise_var` argument is passed as a standard deviation. A skeleton adaptation can retain a learned mask bias and fixed joint identity while adding this projected noise. It must describe those differences explicitly.

### Stochastic latent variables

[A Stochastic Latent Variable Generalization of Joint-Embedding Predictive Architectures](https://arxiv.org/html/2611.09302v1) appeared on November 6, 2026. It learns a posterior `q(z | context, target)` for training and a prior `p(z | context)` for inference. A sampled latent conditions the predictor. The objective combines latent prediction with `KL(q || p)`. The paper studies masked images, not gait. Its uncertainty and representation-quality findings are evidence for that experimental setting, not a guarantee for this cohort. Its visual completions require an additional decoder, and repeated uncertainty samples require additional predictor passes.

The [official implementation](https://github.com/giakoumoglou/sjepa/blob/e851ccb15297f21f54738bad6a63ce04830488ba/src/models/vision_transformer.py) uses diagonal Gaussian prior and posterior networks and injects a projected latent as a prefix token. Its ordinary inference path uses the prior mean. An uncertainty experiment therefore needs an explicit prior-sampling path. Supplying test targets to the posterior would leak the answers. Merely adding Gaussian noise to an existing predictor is a noise-injection ablation, not this learned probabilistic method.

### The Nature paper

[Skeleton motion topology-masked prediction and contrastive learning for self-supervised human action recognition](https://www.nature.com/articles/s41598-026-39330-9), published February 10, 2026, is a separate method. It combines topology and motion in masking, contrastive learning, stronger augmentations, and trajectory-guided feature dropping. Its experiments concern skeleton action recognition. It is useful related work, but neither evidence that our uniform sampler is topology-aware nor a description of Skeleton S-JEPA. Its motion-based masking is outside the main protocol required here.

### VICReg and the introductory JEPA walkthrough

[VICReg](https://arxiv.org/abs/2105.04906) combines agreement between paired views, a lower bound on each embedding dimension's standard deviation, and a penalty on off-diagonal covariance. These terms discourage constant and redundant representations. They do not repel health-condition clusters or guarantee a clinically meaningful latent space. A supervised classifier can learn category separation later. If supervised contrastive loss is introduced, it must be identified as additional label supervision and compared separately.

The supplied [JEPA walkthrough](https://www.thecprogramminglanguage.com/jepa.html) is useful pedagogical background. We do not treat its broad explanations as proof that EMA alone prevents all collapse. The tutorial's claims about a particular algorithm follow the relevant original paper and its executable checks.

## 3. Walk through the tutorial's learning problem

The following explanation is the design of this tutorial, informed by the local background review. It is not a claim that every implementation choice reproduces the ECCV experiment.

The laptop implementation uses 32-frame windows by default, four-frame tokens, the explicit 16-joint allowlist, two coordinate channels, width 64, a two-layer view encoder and a one-layer predictor. It adds clip-level projected VICReg and uses a constant EMA coefficient of 0.996. Its downstream probe pools the **online view encoder**, which is an intentional difference from the paper's target-encoder readout. The target encoder remains in the checkpoint and supplies the latent anomaly-score reference. A target-encoder probe would be a useful separately named ablation. The code's positional-noise projection has no bias and retains a learned mask bias, so it is described as a StoP-inspired skeleton adaptation.

Imagine a short sequence containing a person's pelvis, legs and arms. The pose estimator has measured 33 landmarks. We retain only the landmark IDs listed in the tutorial's sorted allowlist. A tensor `X` has dimensions `batch × time × allowed joints × coordinates`. Confidence values describe measurement quality; they are not disease labels.

1. **Make a token grid.** Group a small fixed number of adjacent frames for each allowed joint. A token includes both a joint identity and a time position. With 32 frames, 16 allowed joints and four frames per token, there are `8 × 16 = 128` tokens. Other frame counts must preserve the declared token boundaries and joint allowlist.
2. **Select targets uniformly.** Choose a fixed-size subset without replacement from the full 128-position grid. Every position has equal marginal probability, including a position whose measurement is missing. Neither displacement, velocity nor validity enters this draw. Refresh masks during training and keep an explicit random generator for reproducibility. Positional noise uses a separate random stream, so enabling it does not alter the mask schedule.
3. **Keep the answer branch unmasked.** The target encoder receives all observed allowed-joint tokens, including observations at randomly hidden positions. Measurement-invalid tokens are excluded from attention. Select target outputs at sampled and observed locations only after encoding, and detach them from autograd.
4. **Keep answers out of the question.** The view encoder processes only visible observed tokens. Predictor placeholders carry position and learned parameters, never hidden joint coordinates or teacher features. All hidden query positions remain present even when a target measurement is missing. Otherwise future missingness itself could alter the forecasting context.
5. **Predict and compare.** The predictor estimates features at sampled locations, but only observed targets contribute to the loss and running target center. In the centered cross-entropy variant, apply the teacher's running center and temperature before comparing distributions.
6. **Update the learner.** Backpropagation changes the view encoder and predictor. Then update the target encoder by EMA. The teacher is excluded from the optimizer.
7. **Measure representation health.** Track standard deviation, covariance and effective rank across independent examples. A changing coordinate pattern inside a single clip can make token-level variance look healthy even when all clips receive the same pooled embedding. Inspect both token and clip levels.
8. **Keep a normal reference.** Save a checkpoint trained only on normal training sources. Continuing training on abnormal sources produces a separate checkpoint. It must not overwrite the normal reference used for anomaly detection.

The temperature-scaled distributions above have one entry per latent feature dimension. They are not probabilities of Parkinson's disease, stroke, or another health condition. The classifier's class probabilities arise later from supervised labels.

For a teacher parameter `theta_t`, a view parameter `theta_v`, and EMA coefficient `m`, the update is:

```text
theta_t_new = m * theta_t_old + (1 - m) * theta_v_new
```

For example, if `theta_t_old = 0.50`, `theta_v_new = 0.70`, and `m = 0.99`, the updated target parameter is `0.502`. This explains why a teacher can change smoothly while the view encoder learns more quickly.

## 4. Three independent kinds of randomness

| Mechanism | What changes? | What remains fixed? | Question answered |
| --- | --- | --- | --- |
| Uniform masking | Which allowed joint-time tokens are hidden | Joint identities and the target construction | Can visible context predict different missing parts? |
| StoP-style positional conditioning | The predictor's masked-position embeddings | Anatomical index and teacher targets | Does mild location uncertainty improve robustness? |
| Learned stochastic latent variables | A conditional latent supplied to the predictor | The observed context and requested target positions | Which different completions are plausible from this context? |

These mechanisms can be combined, but introducing all of them at once makes the result difficult to explain. Establish the uniform-mask deterministic predictor first. Add positional conditioning as one ablation. Add learned prior and posterior networks only after the main result and uncertainty evaluation are reliable.

For gait, the following remain hypotheses to test. Small positional perturbations might reduce sensitivity to pose localization error. Excessive perturbations might erase subtle asymmetry. A conditional latent might represent ambiguity from occlusion or an incompletely observed gait cycle. It might instead be ignored, encode target information that the prior cannot reproduce, or turn pose failures into misleading uncertainty. Measure these possibilities rather than naming all stochastic variation epistemic uncertainty.

A useful stochastic-latent evaluation keeps each context and mask fixed, draws several samples from the learned prior, and compares predictive variation with held-out latent prediction error. Report the prediction sample count, the extra runtime, the association measure and confidence interval, and behavior under controlled pose corruption. High variance is not itself an abnormal-gait score. Low variance is not evidence that a prediction is correct.

## 5. Completion, forecasting, and anomaly detection

Random masks can leave observations both before and after a missing token. That task is **masked completion**. It does not establish forecasting. A future-prediction claim requires a separate experiment where the prediction branch sees only a time prefix and targets lie strictly later. Window normalization, interpolation, feature extraction and context encoding must also respect that time boundary. A whole-window scale or smoother can leak future information even when the attention mask looks causal.

The main anomaly task asks whether a new sequence differs from the learned normal reference. Use the unchanged normal-only checkpoint, define a sequence score, and select its decision threshold from allowed validation data. Two understandable candidates are mean latent prediction discrepancy over fixed evaluation masks and distance from the normal training embedding distribution. Scores can be high because of unusual motion, unusual camera view, poor tracking, or occlusion. Explanation plots should show these alternatives alongside the source clip.

Five-class recognition is a different question. After normal pretraining, continue representation learning on training sources from the selected conditions, preserving normal replay. The prediction loss is label-free, but normal membership, curriculum stages and class-balanced sampling use the dataset labels. That selection supervision should be disclosed. A frozen linear probe then measures accessible class information. A supervised encoder fine-tuning experiment changes the training regime and should receive its own row in the results. Neither experiment should use test sources for label-free pretraining.

## 6. Gait-specific choices that need visible justification

**Timing:** Preserve the source FPS and timestamps. A fixed frame count obtained by stretching every video to the same duration can remove cadence differences. Fixed-duration windows on a common sampling rate are easier to interpret. The shared corpus excludes a source whose frame rate is below the target rate, with a recorded `source_fps_below_target` reason. Avoiding upsampling prevents a prefix interpolation bracket from reaching into the future half. Changing the target rate changes eligibility and must be declared before model comparison. Short sequences need a documented policy rather than silent cyclic repetition.

**Geometry:** Hip centering reduces image translation, and a body scale can reduce distance effects. Those operations also remove absolute location and size. Image coordinates, detector-relative depth and world-coordinate estimates have different meanings. The paper must name the coordinate system actually used. Monocular estimates do not establish metric stride length without validation.

**Laterality:** Preserve left and right landmark identities. Random horizontal flipping needs a deliberate anatomical relabeling rule and may change the meaning of laterality features. It is not a harmless default for an asymmetry study.

**Missingness:** Invalid coordinates need a recorded validity mask. The implementation retains them as NaN. It resamples only between adjacent observed source frames, without bridging a missing-frame gap. Four consecutive XY observations must all be finite for a joint patch to be valid. A retained window needs at least 75% valid patches overall and at least 50% in each half. Numerical zero replacement happens only at the linear projection; invalid attention keys, target losses, teacher-center statistics and pooled features are explicitly excluded. This is different from treating zeros as observed joints. The quality rule was revised after a technical extraction pilot, before classifier performance was examined. Its validity geometry can still predict condition or viewpoint, so the capstone includes a pose-availability control and further nuisance audits remain necessary.

**Forecast admission:** Scale uses only observed torso measurements from the first half of each window. Admission nevertheless requires sufficient future observations to score the forecast. This is retrospective forecasting on evaluable futures, not a rule for admitting an unseen future in a live system.

**Quality control:** A subject can leave the frame, turn, or share the video with another person. Examine a source-video gallery and pose overlay before trusting any classification score. The same accepted sequences must enter the Random Forest and JEPA comparisons.

## 7. What is new enough to test?

Applying JEPA to skeletons is established prior work. A credible workshop contribution could instead examine whether a normal-first skeleton representation detects departures from normal gait under source-held-out evaluation, and whether continued training improves recognition without destroying the original normal reference. Uniform anatomical masking, positional stochasticity, and a reproducible failure analysis can form controlled parts of that study.

Treat expected improvements as hypotheses. The deliverable is a reproducible experiment and an honest interpretation of its measured results. Neither this review nor a successful software smoke test establishes that JEPA outperforms the Random Forest.
