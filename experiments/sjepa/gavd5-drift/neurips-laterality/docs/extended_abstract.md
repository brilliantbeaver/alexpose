# Build the Geometry In, Don't Hope It Emerges: A Reflection-Symmetry Audit of a Skeleton World Model

**Anonymous submission draft — NeurIPS 2026 Workshop on Physical World AI: Geometry, Characteristics, and Multimodal Sensing.**

> **Evidence status.** The registered laterality v2.1 run is complete: all five outer folds, five optimization seeds, and both training variants finished on real derived-pose data, and the two hard integrity gates (every registered fold and seed present, exact output oddness for the constructed read-out) pass. The results below are the study's empirical findings, reported without any post-hoc change to the frozen protocol. Earlier transductive estimates, computed on sequences that had also entered training, are superseded and omitted. Submission and artifact release stay blocked while the institutional ethics determination, the data-use review, and the derived-pose release review are unresolved.

## Motivation

A human pose representation has an approximate left–right structure. Reflect a skeleton horizontally and exchange its anatomical left and right landmarks, and you get another valid pose representation; any signed left-versus-right motion quantity should reverse sign under that operation. This is a group action, and it gives a direct handle on what a geometry-aware world model has captured about the articulated body. We use it to audit a self-supervised Skeleton Joint-Embedding Predictive Architecture (S-JEPA). The model learns by predicting hidden latent content in pose sequences from the Gait Abnormality in Video Dataset (GAVD); it never sees the laterality target or a dataset annotation during representation training.

The target is derived from coordinates and is not a diagnosis or a validated clinical gait measure. For the left and right shoulders, knees, ankles, heels, and foot-index points, it compares median motion speed only on frame transitions where both sides are observed at both endpoints, then averages five normalized pair contrasts. Mirroring negates this target, changing invalid-coordinate placeholders leaves it unchanged, and mirroring twice restores the input. Across the cohort the target spans roughly ±0.2 with a standard deviation near 0.06, so the signal to recover is subtle.

We keep three questions apart. A mathematical wrapper can force a read-out to change sign, which proves only that the wrapper is correct. A trained encoder with a probe can show useful sign-reversing behavior, which needs empirical support. The encoder tokens can satisfy a direct reflection-equivariance test, which is stronger still. The audit measures all three, so a wrapper's guarantee is never read as a learned property.

## Cohort and source-level split

The GAVD inventory contains 666 annotation files for 103 source videos. A matching pose archive is available for 642 annotations. Pre-specified quality-control rules retain 625 pose sequences from 93 source videos and exclude 17; inclusion depends on target computability and pose coverage, never on the sign or size of the target. The split is created at the source-video level because one video can yield several related sequences, and every sequence, mirror, and augmentation inherits its source's fold — so no excerpt from a test video reaches the matching training set. GAVD supplies no persistent person identifier, so this supports a held-out-video claim rather than a held-out-person one.

| Outer fold | Training sources | Test sources | Training sequences | Test sequences |
|---:|---:|---:|---:|---:|
| 0 | 74 | 19 | 436 | 189 |
| 1 | 74 | 19 | 443 | 182 |
| 2 | 74 | 19 | 553 | 72 |
| 3 | 75 | 18 | 548 | 77 |
| 4 | 75 | 18 | 520 | 105 |

Sequence counts vary because videos contribute different numbers of sequences, and every source video is tested exactly once.

## Training, validation, and evaluation

A fresh encoder is trained for every outer fold, seed, and variant: five folds, five optimization seeds (42–46), and two variants — vanilla and reflection-augmented — for 50 fold-local encoders, each following the fixed 300-epoch schedule on its outer-training sources with no early-stopping validation set. Within a seed the variants share initialization, source draws, masks, views, and optimizer schedule, with reflection on a separate random stream, so the pair stays matched. Validation is reserved for the linear ridge read-out: four inner folds inside each outer-training set select the ridge penalty and nothing else, after which the read-out is refitted on all outer-training sources and applied once to the untouched outer-test sources, with all fitted preprocessing estimated from outer-training data only.

The primary measure is source-balanced $R^2$, which gives every source video equal total weight. Comparison lanes include a paired untrained encoder (an initialization floor), a target-component self-consistency oracle (a verification, not a learned baseline), measured nuisance features, analytically antisymmetric odd/even read-outs, and a direct encoder-token reflection test. That strict test compares target-encoder tokens under the full anatomical permutation with no channel fitting, rotation, sign search, or read-out; zero is exact equivariance. Five seeds describe optimization variation, and 2,000 source-video bootstrap resamples — keeping each video's sequences and seeds together — describe uncertainty over the observed source collection, conditional on the fitted pipeline.

## Results

The registered run supports none of the emergence claims and passes only the construction's own integrity check. The same pattern holds for the reflection-augmented variant and for a high-coverage sensitivity subset (600 sequences from 91 sources at ≥0.9 authorized coverage), where the primary $R^2$ is again near 0.06.

- **Predictive utility does not clear the floor.** The learned free read-out reaches $R^2 = 0.06$ (95% CI $[-0.03, 0.13]$), lower bound below the registered minimum of zero. Against the untrained encoder, training changes held-out $R^2$ by $-0.02$ (95% CI $[-0.04, 0.00]$). The self-consistency oracle recovers the target almost perfectly ($R^2 \approx 1.00$), so the failure is in the learned features, not the metric. Measured nuisances add nothing.
- **Reflection augmentation does not help.** On the primary lane the reflection-augmented encoder differs from vanilla by $0.00$ (95% CI $[-0.01, 0.01]$).
- **The construction works, but the encoder is not why.** The odd/zero-origin read-out is exactly antisymmetric for every seed (oddness gate met at $10^{-6}$), yet its absolute value is weak ($R^2 = 0.04$, 95% CI $[-0.04, 0.11]$), and the same construction on an untrained encoder does at least as well (learned minus untrained $-0.06$, 95% CI $[-0.10, -0.02]$). Training nudges only the even part ($+0.02$, 95% CI $[0.01, 0.04]$), opposite to where a laterality signal lives.
- **Training degrades reflection equivariance.** The learned token error is $q = 0.11$ (95% CI $[0.10, 0.14]$), above the $0.10$ margin, while the untrained initialization is closer at $q = 0.08$ (95% CI $[0.07, 0.09]$); the learned-minus-initial difference is $+0.03$ (95% CI $[0.02, 0.05]$). The native free read-out is far from antisymmetric (output error $0.22$, 95% CI $[0.19, 0.24]$). Reflection augmentation reduces the damage a little ($q = 0.11$, learned minus initial $+0.02$) without reaching the margin.

Figure 1 shows both learned-minus-untrained effects: no predictive gain on any read-out, and a rise in token error after training. For this configuration and target, bilateral symmetry is something you install in the read-out, not something a generic predictive objective discovers.

## Claim boundary, ethics, and status

The strongest supported claim is post-development, within-GAVD cross-validated performance on held-out source videos. The protocol does not support diagnosis, clinical validity, prevalence, treatment effects, or generalization to unidentified people in new videos, and the null is specific to the registered architecture, seeds, target, and controls. Folder names are dataset annotations, not diagnoses. Submission and artifact release stay blocked until the institutional ethics determination, the data-use review, and the derived-pose release review each carry a genuine dated internal reference. The project redistributes no raw video or identity-bearing frame, and derived artifacts remain non-public unless completed reviews permit release.
