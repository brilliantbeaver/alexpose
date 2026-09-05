# Build the Geometry In, Don't Hope It Emerges: A Reflection-Symmetry Audit of a Skeleton World Model

**Anonymous submission draft — NeurIPS 2026 Workshop on Physical World AI: Geometry, Characteristics, and Multimodal Sensing. Under double-blind review; do not distribute.**

> **Evidence status.** The registered laterality v2.1 run is complete. All five outer folds, five optimization seeds, and both training variants finished; the report was computed on real derived-pose data, not on the synthetic smoke profile; and the two hard integrity gates (every registered fold and seed present, and exact output oddness for the constructed read-out) pass. The results below are therefore the study's empirical findings, and they are reported without any post-hoc change to the frozen protocol. An earlier transductive experiment, in which an encoder was evaluated on sequences that had also entered its training, is superseded; its numbers are not evidence for this design and are omitted. Two things remain open. Submission and artifact release are still blocked because the institutional ethics determination, the data-use review, and the derived-pose release review are unresolved. And every claim keeps the narrow boundary described in Section 10.

## Abstract

The human body has an approximate left–right structure. Reflecting a pose horizontally and exchanging anatomically corresponding left and right landmarks produces another valid pose representation, so any signed left-versus-right motion quantity should reverse sign under that operation. We use this exact geometric fact to audit what a self-supervised skeleton world model actually represents. The model is a Skeleton Joint-Embedding Predictive Architecture (S-JEPA) trained on pose sequences from the Gait Abnormality in Video Dataset (GAVD); the audit target is a coordinate-derived comparison of left- and right-side motion, not a diagnosis or a validated clinical measurement.

Our protocol is built to survive scrutiny. The quality-controlled cohort holds 625 pose sequences from 93 source videos, split at the source-video level into five outer folds so that no clip, mirror, or augmentation from a test video can reach the matching training set. A fresh encoder is trained for each of the fifty registered fold-and-seed combinations, four inner folds pick the ridge read-out penalty using only outer-training sources, and 2,000 source-cluster bootstrap resamples give the uncertainty. Across every registered checkpoint the trained representation does not predict the laterality contrast better than an untrained encoder of the same architecture (learned minus untrained source-balanced $R^2 = -0.02$, 95% CI $[-0.04, 0.00]$), it is not reflection-equivariant, and training moves the representation slightly *away* from the approximate equivariance already present at initialization ($q$ increases by $0.03$, 95% CI $[0.02, 0.05]$). Reflection augmentation does not repair any of this. The only route to the correct sign-reversing behavior is to build it into the read-out analytically, which guarantees exact output antisymmetry but adds no learned content — the same construction on a random encoder does at least as well. For a skeleton world model, then, bilateral symmetry is something you install, not something you get for free from prediction.

## 1. Introduction

A world model earns its name by predicting or reconstructing parts of its input without being handed the downstream label. For a model of an articulated body, physical geometry offers a direct way to ask what such a model has actually captured. The human skeleton has a bilateral structure, and one clean instance of it is reflection: flip the body horizontally and swap each named left landmark with its right partner. This is a group action on the pose representation, and it acts predictably on any quantity that measures a left-versus-right difference. If a representation has internalized the geometry of the moving body, that structure should be visible in how the representation transforms.

This paper turns that expectation into a measurement. We ask whether an S-JEPA representation, trained with no access to a laterality label, carries usable information about a signed left-versus-right motion contrast, and whether that information behaves correctly when the input is mirrored. Three statements are easy to run together and worth keeping apart. A mathematical wrapper can force a read-out to change sign; this is a construction and proves only that the wrapper was implemented correctly. A trained encoder followed by a probe can show useful sign-reversing behavior; this needs empirical support. The encoder tokens themselves can satisfy a direct reflection-equivariance test; this is a stronger and separate empirical question. We measure all three.

The workshop's Physical Geometry theme asks for geometry-aware world models and for honest understanding of articulated, deformable scenes. Our contribution sits there, and it also speaks to the cross-cutting call for evaluation protocols and responsible deployment. Concretely:

- We frame a reusable **geometry audit** for articulated-body world models around a known group action, and we separate analytic output oddness, probe-level sign behavior, and intrinsic token equivariance so that a wrapper's guarantee is never mistaken for a learned property.
- We register a **leakage-controlled evaluation protocol**: source-level nested cross-validation, fold-local representation training, uncertainty that keeps optimization-seed variation apart from source-sampling variation, and decision thresholds frozen before any result was seen.
- We report a **fully powered null**. Self-supervised training on skeleton sequences does not produce the laterality contrast, does not yield reflection-equivariant tokens, and in fact degrades the equivariance present at random initialization; reflection augmentation does not change the verdict.
- We show the **constructive alternative**. An odd/even parity decomposition applied in the read-out guarantees exact antisymmetric outputs, but the guarantee comes from the wrapper rather than from learning, since it holds just as well on an untrained encoder.

The last point is the paper's title. If you want a skeleton model to respect a physical symmetry of the body, the reliable move is to build the symmetry into the architecture or the read-out. Waiting for a generic predictive objective to discover it, at least at this scale and on this target, did not work.

## 2. Concepts and terminology

A **pose sequence** is a run of video frames represented by estimated body-landmark coordinates rather than by pixels. Each sequence carries 33 BlazePose landmarks, and each landmark has a horizontal, a vertical, and a depth coordinate. A **source video** is the original recording that one or more pose sequences were extracted from. Because a single video can yield several sequences, splitting sequences independently would let closely related material land on both sides of a train/test divide.

An **encoder** turns a pose sequence into learned numeric features. A **read-out**, or probe, is a simpler model fitted on those features to predict the audit target. The two have different training procedures. The encoder follows a fixed, pre-registered schedule with no early-stopping validation set. The read-out uses inner cross-validation to choose its ridge penalty.

**Cross-validation** rotates which data are held out. An **outer fold** measures final held-out performance; an **inner fold** makes a training-only choice without consulting the outer test set. **Stratification** keeps the dataset-annotation counts roughly balanced across folds. Those annotations serve two limited purposes — balancing the split, and one explicitly named confounding control — and never enter the training objective as prediction labels.

## 3. Dataset inventory and cohort construction

GAVD is the Gait Abnormality in Video Dataset [ranjan2025gavd, gavdRepo2026]. Its official distribution supplies annotations and public video links rather than raw media. This study works on derived pose archives and redistributes no source video or identity-bearing frame.

The frozen inventory contains 666 annotation files describing 103 source videos. A matching derived-pose archive exists for 642 annotations; the other 24 have none. Pre-specified quality-control rules then retain 625 pose sequences from 93 source videos and exclude 17. A sequence can be excluded when the target cannot be computed, when too few of the selected landmarks are valid, or when too few complete four-frame model patches remain. These rules were fixed before any model result was inspected, and inclusion never depends on whether the target is positive, negative, large, or small.

Every retained sequence carries provenance: the source annotation, the pose-archive fingerprint, the extraction version, the pose-model identity, and the visibility threshold. A SHA-256 content digest binds the cohort table and arrays together, and later stages reject any cohort whose content or provenance drifts from the frozen protocol.

## 4. Coordinate-derived laterality target

The target compares movement on the two sides of the pose representation, using five anatomical pairs: shoulders, knees, ankles, heels, and foot-index landmarks. The hips define the pelvis-centered coordinate frame but stay out of the target, because pelvis centering makes their two velocity magnitudes algebraically identical.

For each pair the calculation keeps only **paired-valid transitions**, meaning both the left and the right landmark are observed at both ends of the same frame transition. Let $m_{L,k}$ and $m_{R,k}$ be the median left and right speeds for pair $k$. The normalized contrast and the target are

$$
c_k = \frac{m_{L,k}-m_{R,k}}{m_{L,k}+m_{R,k}+\epsilon},
\qquad y=\operatorname{mean}_k c_k,
$$

where $\epsilon$ prevents division by zero. Every registered pair must supply at least eight shared valid transitions, and all five pairs must be usable. Positive $y$ means relatively more left-side motion under this formula and negative $y$ means the reverse; the meaning is arithmetic, not clinical. Across the cohort $y$ is small and roughly centered, ranging over about $\pm 0.2$ with a standard deviation near $0.06$, which already signals that the signal to be recovered is subtle.

Short missing stretches may be interpolated for encoder input, but interpolated coordinates never enter the target. Invalid coordinates come with a validity mask and cannot act as numeric placeholders. Automated tests confirm three properties before any result is read: changing the values under invalid entries leaves $y$ unchanged, mirroring the coordinates and mask negates $y$, and mirroring twice restores the input.

## 5. Training, validation, and testing design

### 5.1 Why the source video is the splitting unit

All sequences from one source video go into the same fold, and mirrored inputs and augmentations inherit that assignment. No clip, mirror, or augmentation from an outer-test source can reach the matching outer-training set. GAVD supplies no persistent person identifier, and we do not infer one, so two different videos could in principle show the same unidentified person. The split therefore supports held-out-source-video evaluation, which is a weaker and more honest claim than held-out-person evaluation.

### 5.2 Five outer folds

The 93 eligible source videos are divided into five stratified outer folds. In each fold roughly four fifths of the sources train the encoder and fit the final read-out, and the remaining fifth form the untouched test set. Sequence counts vary because videos contribute different numbers of sequences.

| Outer fold | Training sources | Test sources | Training sequences | Test sequences |
|---:|---:|---:|---:|---:|
| 0 | 74 | 19 | 436 | 189 |
| 1 | 74 | 19 | 443 | 182 |
| 2 | 74 | 19 | 553 | 72 |
| 3 | 75 | 18 | 548 | 77 |
| 4 | 75 | 18 | 520 | 105 |

This is not one fixed 80/20 split. Each source is an outer-test source exactly once and an outer-training source in the other four folds, so combining the out-of-fold predictions yields one held-out prediction for every eligible sequence.

### 5.3 Four inner validation folds

Within each outer-training set the sources are divided again into four inner folds. One inner fold holds 18 or 19 validation sources while the other 55 to 57 fit a candidate read-out, and the validation role rotates until every outer-training source has served once. Because sources differ in size, an inner fitting partition holds 266 to 450 sequences and an inner validation partition holds 80 to 197; the exact memberships live in the split artifact.

These inner folds choose the ridge penalty and nothing else. They do not select an encoder checkpoint, stop encoder training, or alter the registered schedule. After the penalty is chosen the read-out is refitted on all 74 or 75 outer-training sources and then applied once to the corresponding 18 or 19 outer-test sources. Feature scaling, the target scale, and the neutral band are all estimated from outer-training data only. The nesting matters — if the outer-test targets helped pick the penalty, the final estimate would no longer describe untouched data.

## 6. Fold-local representation training

The representation model is an S-JEPA trained with a VICReg regularizer. It takes 64 time steps of 33 landmarks, groups time into four-frame patches, and uses a 96-dimensional embedding, a four-layer encoder, a two-layer predictor, and four attention heads. During self-supervised training a fraction of the sequence representation (0.6) is masked and the model predicts the hidden latent content against an exponential-moving-average target encoder. Neither the laterality target nor the dataset annotation enters this objective.

Training is fold-local. For each combination of outer fold, optimization seed, and variant, a new encoder starts from a recorded seed-matched initialization and sees only that fold's outer-training sources. Sampling draws a source video uniformly and then one of its sequences, so a video with many sequences does not dominate. The paper profile registers five seeds (42 through 46) and two variants. The `vanilla` variant never reflects a training sample; the `reflection_augmented` variant reflects a sample with probability 0.5. Within a seed the two variants share initialization, source draws, masks, geometric views, optimizer schedule, and update count, and reflection decisions use a separate random stream, so the pair stays matched. The design is $5$ folds $\times\, 5$ seeds $\times\, 2$ variants $=50$ fold-local encoders, each trained for the fixed 300-epoch schedule with AdamW at learning rate $10^{-3}$, weight decay $0.05$, and batch size 20.

## 7. Held-out evaluation

For a given fold and seed, feature extraction uses the encoder that never saw that fold's test sources. The primary lane fits a free linear ridge read-out to the learned representation. Additional lanes answer distinct questions:

- an identically structured untrained encoder gives a paired initialization floor;
- a target-component self-consistency oracle checks that the saved pair contrasts reconstruct the target — it is a verification, not a learned baseline;
- pooled and missingness features test measured nuisance explanations;
- constructed one-pass and two-pass read-outs test whether antisymmetry can be guaranteed analytically;
- direct token comparisons test whether the encoder itself is reflection-equivariant.

For the parity constructions we split each feature into an odd and an even part under reflection, $z^-(x) = (z(x) - z(Mx))/\sqrt{2}$ and $z^+(x) = (z(x) + z(Mx))/\sqrt{2}$, and give the learned and paired-initial encoders equal-dimension lanes: single/free, two-pass odd/free, two-pass odd/zero-origin, and two-pass even/free. The odd/zero-origin lane is the constructed repair; its feature parity and output oddness are imposed by design and are never treated as emergence.

The principal measure is source-balanced $R^2$, the fraction of target variation explained after each source video is given equal total weight. Mean absolute error, source-level sign accuracy, and mirror errors are secondary. Each checkpoint is scored on its own, and metrics and squared mirror errors are formed within a checkpoint before the registered seeds are averaged, so signed errors cannot cancel across seeds.

For the direct representation test, let $Z(x)$ and $Z(Mx)$ be target-encoder tokens and let $S$ apply the full 33-joint anatomical permutation while leaving latent channels alone. On the tokens valid in both aligned views the strict error is

$$
q=\frac{\lVert Z(Mx)-S\,Z(x)\rVert_C^2}{\lVert Z(Mx)\rVert_C^2+\lVert S\,Z(x)\rVert_C^2},
$$

with no channel fitting, rotation, sign search, centering, or read-out allowed. A value of zero is exact equivariance, unrelated equal-energy representations sit near one, and a collapsed zero-energy representation is rejected. Each learned checkpoint is paired with its own initial target encoder, and the two variants share that initialization for every fold and seed.

## 8. Results

The registered run supports none of the emergence claims and passes only the construction's own integrity checks. We report the primary lane and its key comparisons; the pattern is the same for the reflection-augmented variant and for a high-coverage sensitivity subset (600 sequences from 91 sources, restricted to at least 0.9 authorized coverage), where the primary source-balanced $R^2$ is again near $0.06$.

**Predictive utility does not clear the floor.** The learned free read-out reaches a source-balanced $R^2$ of $0.06$ (95% CI $[-0.03, 0.13]$), so its lower bound sits below the registered minimum of zero. More telling is the paired comparison against the untrained encoder: training changes held-out $R^2$ by $-0.02$ (95% CI $[-0.04, 0.00]$), a difference that is slightly negative and consistent with no gain (Figure 2a). The self-consistency oracle recovers the target almost perfectly ($R^2 \approx 1.00$), which confirms the target is reconstructible from its own pair contrasts and that the failure lies with the learned features rather than the metric. Measured nuisances (visibility and missingness, acquisition and extraction version, dataset annotation, and their concatenation) explain essentially nothing on their own, and adding them to the learned features leaves held-out $R^2$ unchanged.

**Reflection augmentation does not help.** On the primary lane the reflection-augmented encoder differs from vanilla by $0.00$ in source-balanced $R^2$ (95% CI $[-0.01, 0.01]$). The augmentation that most directly encodes the symmetry into training has no measurable effect on the recovered signal.

**The construction works, but the encoder is not why.** The constructed odd/zero-origin read-out produces exactly antisymmetric outputs for every prediction and seed, so its output-oddness gate passes at the $10^{-6}$ tolerance. Its absolute predictive value is still weak ($R^2 = 0.04$, 95% CI $[-0.04, 0.11]$), and, crucially, the same construction on an untrained encoder does at least as well: the learned-minus-untrained difference under the odd/zero construction is $-0.06$ (95% CI $[-0.10, -0.02]$), and under the odd/free construction it is $-0.07$ (95% CI $[-0.11, -0.03]$). Both intervals exclude zero on the negative side. Whatever modest structure the odd lanes exploit is a property of the parity wrapper applied to the anatomical pairs, not of the learned representation. The one place training nudges anything is the even (symmetric) part, where it adds a small amount ($+0.02$, 95% CI $[0.01, 0.04]$) — the opposite of what a laterality signal, which lives in the odd part, would require.

**The native read-out is far from antisymmetric.** Left to itself, the learned free read-out is not sign-reversing: its normalized native output antisymmetry error is $0.22$ (95% CI $[0.19, 0.24]$), well above the registered margin of $0.10$, and barely different from the untrained encoder's error. Exact antisymmetry appears only when it is imposed.

**Training degrades reflection equivariance.** The strict token test is the sharpest result. The learned representation is not equivariant — its error is $q = 0.11$ (95% CI $[0.10, 0.14]$), above the $0.10$ margin. The untrained initialization is closer to equivariant at $q = 0.08$ (95% CI $[0.07, 0.09]$), and the learned-minus-initial difference is $+0.03$ (95% CI $[0.02, 0.05]$), so self-supervised training moves the tokens away from the approximate equivariance they started with (Figure 2b). Reflection augmentation shrinks this damage a little ($q = 0.11$, learned-minus-initial $+0.02$, 95% CI $[0.01, 0.04]$) but does not reach the margin and does not beat the initialization. Figure 3 places the learned representation against the frozen thresholds on both axes at once: it misses the predictive-utility gate and both geometric-fidelity margins, while the built-in construction is the only lane that reaches exact symmetry.

Table 1 collects the decisions. Every empirical gate registered in the protocol is unmet, and the one gate that passes — exact output oddness — is a statement about the construction, not about what the encoder learned.

| Registered gate | Requirement | Outcome |
|---|---|---|
| Native probe predictive utility | learned $R^2$ lower CI $> 0$ | not met ($0.06$, CI $[-0.03, 0.13]$) |
| Native training content | learned $-$ untrained $R^2$ lower CI $> 0$ | not met ($-0.02$, CI $[-0.04, 0.00]$) |
| Native output symmetry | error upper CI $< 0.10$ | not met ($0.22$, CI $[0.19, 0.24]$) |
| Strict token equivariance | learned $q$ upper CI $< 0.10$ | not met ($0.11$, CI $[0.10, 0.14]$) |
| Equivariance training effect | learned $-$ initial $q$ upper CI $< 0$ | not met ($+0.03$, CI $[0.02, 0.05]$) |
| Learned increment over nuisances | lower CI $> 0$ | not met ($0.05$, CI $[-0.02, 0.12]$) |
| Constructed predictive utility | learned $R^2$ lower CI $> 0$ | not met ($0.04$, CI $[-0.04, 0.11]$) |
| Constructed training content | learned $-$ untrained $R^2$ lower CI $> 0$ | not met ($-0.06$, CI $[-0.10, -0.02]$) |
| Constructed output oddness | error $\leq 10^{-6}$, every seed | met (by construction) |

## 9. Uncertainty and decision rules

Two kinds of variation stay separate. Optimization-seed variation asks how results shift across the five registered initializations; source-sampling uncertainty asks how the estimate shifts when the 93 observed videos are resampled, using 2,000 bootstrap repetitions drawn at the source-video level so that every sequence and registered seed from a sampled source travels together. These intervals are conditional on the already fitted cross-validation pipeline. They describe uncertainty over the observed source collection, not over all people or all recording conditions, and they do not propagate new-seed, split-allocation, or full-retraining uncertainty.

Every threshold lives in `config/protocol.json` and was frozen before results were seen. Native probe behavior would require useful absolute prediction, a native output error below the margin, and a positive learned-minus-untrained difference. Strict representation behavior would require a token error below the margin together with an improvement over the paired initialization. The protocol also fixes a null-result policy: a failed diagnostic is reported as is and does not license retuning. That policy is what makes the results in Section 8 usable rather than a starting point for search.

## 10. Limitations

The split prevents overlap at the source-video level but cannot guarantee that people are distinct across videos, so the supported claim is post-development, within-GAVD cross-validated performance on held-out source videos. It says nothing about unseen people, another dataset, clinical validity, prevalence, diagnosis, or treatment.

The target captures one signed left-versus-right motion contrast. It is not the full kinematic structure of walking and has not been validated as a clinical gait index. The strict equivariance test fixes a particular joint permutation and identity channel action, so a different notion of equivariance could give a different number. All of these findings remain conditional on the BlazePose landmark schema, the pose-extraction system, the preprocessing, the architecture, the registered seeds, and the measured controls. A larger model, a different objective, or a denser laterality target might behave differently, and the null we report is specific to the configuration we registered. Unmeasured acquisition and missingness effects may remain.

## 11. Ethics and data use

Public availability of video links is not authorization for research use or redistribution. The suite stores no raw video and no identity-bearing frame, but derived poses, source identifiers, embeddings, predictions, and checkpoints can still be linkable or sensitive. The institutional ethics determination, the data-use review, and the derived-pose release review are all unresolved, and submission and artifact release stay blocked until each carries a genuine dated internal reference. A careful statistical design does not substitute for those determinations.

## 12. Reproducibility

The notebooks run in numeric order. Notebook 00 freezes the protocol and reports governance status. Notebook 01 audits cohort construction and the target. Notebook 02 builds the source-level outer and inner folds. Notebook 03 trains the fold-local encoders. Notebook 04 fits read-outs and produces held-out predictions. Notebook 05 aggregates the registered checkpoints and uncertainty estimates. Notebook 06 is a separate gate for a possible external dataset, which the GAVD review cannot authorize on its own.

Every number in Section 8 traces to report artifacts under `artifacts/paper/protocol_<digest>/report/` that carry matching protocol, context, cohort, split, checkpoint, and implementation digests, and the figures are regenerated from those same artifacts by `docs/figures/make_v21_figures.py`. The split assignments live in `artifacts/paper/protocol_<digest>/splits/source_splits.json`, the registered rules in `config/protocol.json`, and the interpretation boundary in `PROTOCOL.md` and `RUNBOOK.md`.
