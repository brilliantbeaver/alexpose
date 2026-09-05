# Build the Geometry In, Don't Hope It Emerges: A Reflection-Symmetry Audit of a Skeleton World Model

**Anonymous submission draft — NeurIPS 2026 Workshop on Physical World AI: Geometry, Characteristics, and Multimodal Sensing.**

> **Evidence status.** This extended abstract describes the registered laterality v2.1 protocol. The previous transductive estimates are superseded and are not results from this design. The complete paper run and governance reviews are still pending, so no v2.1 performance conclusion is reported.

## Motivation

A human pose representation has a natural left–right structure. If a skeleton is reflected horizontally and its anatomical left and right landmarks are exchanged, a signed left-versus-right motion quantity should reverse sign. We use that simple geometric fact to test a self-supervised Skeleton Joint-Embedding Predictive Architecture (S-JEPA). The model learns by predicting hidden latent content in pose sequences; it does not receive the laterality target or dataset annotation during representation training.

The target is derived from coordinates and is not a diagnosis or validated clinical gait measurement. For the left and right shoulders, knees, ankles, heels, and foot-index points, it compares median motion speed only on frame transitions where both sides are observed at both endpoints. The five normalized pair contrasts are averaged. Mirroring must negate this target, changing invalid coordinate placeholders must leave it unchanged, and mirroring twice must restore the input.

## Cohort and split

The Gait Abnormality in Video Dataset (GAVD) inventory contains 666 annotation files for 103 source videos. A matching pose archive is available for 642 annotations. Pre-specified quality-control rules retain 625 pose sequences from 93 source videos and exclude 17 sequences. Inclusion depends on target computability and pose coverage, never on the sign or size of the target.

The split is created at the source-video level because one video can yield several related sequences. Every sequence, mirror, and augmentation from a source inherits that source's fold. This prevents an excerpt from a test video from entering the corresponding training set.

Five outer cross-validation folds rotate the held-out test sources:

| Outer fold | Training sources | Test sources | Training sequences | Test sequences |
|---:|---:|---:|---:|---:|
| 0 | 74 | 19 | 436 | 189 |
| 1 | 74 | 19 | 443 | 182 |
| 2 | 74 | 19 | 553 | 72 |
| 3 | 75 | 18 | 548 | 77 |
| 4 | 75 | 18 | 520 | 105 |

Sequence counts vary because source videos contribute different numbers of sequences. Every source video is tested exactly once. Because GAVD does not supply persistent person identifiers, this is held-out-video evaluation, not held-out-person evaluation.

## Training, validation, and evaluation

A fresh encoder is trained for every outer fold, seed, and variant. The registered paper design contains five folds, five optimization seeds, and two variants—vanilla and reflection-augmented—for 50 encoders in total. Each encoder follows the fixed 300-epoch schedule and uses all of its outer-training sources. It has no early-stopping validation set.

Validation is reserved for the linear ridge read-out. Four inner folds are built entirely inside each outer-training set. In each inner round, 55–57 sources fit a candidate read-out and 18–19 sources validate its ridge penalty. After selection, the read-out is refitted on all 74–75 outer-training sources and evaluated once on the untouched 18–19 outer-test sources. Feature scaling, target scaling, and other fitted preprocessing use outer-training data only.

The inner fitting partitions contain 266–450 sequences and the rotating inner validation partitions contain 80–197 sequences. The variation occurs because source videos contribute different numbers of sequences. None of these inner partitions includes an outer-test source.

The primary measure is source-balanced $R^2$, which gives every source video equal total weight. Comparison lanes include a paired untrained encoder, a target-component self-consistency oracle, measured nuisance features, analytically antisymmetric read-outs, and direct encoder-token reflection tests. The oracle verifies target reconstruction and is not a learned baseline. Exact sign change from a constructed wrapper proves a mathematical property of the wrapper; evidence that training learned useful symmetry additionally requires positive held-out prediction and improvement over the paired initialization.

Five seeds describe optimization variation. Separately, 2,000 source-video bootstrap resamples describe uncertainty associated with the observed source collection while keeping each video's sequences together. These intervals are conditional on the fitted cross-validation pipeline and do not imply population or unseen-person generalization.

## Claim boundary, ethics, and status

The strongest possible claim is post-development, within-GAVD cross-validated performance on held-out source videos. The protocol does not support diagnosis, clinical validity, prevalence, treatment effects, or generalization to unidentified people in new videos. Folder names are dataset annotations, not diagnoses.

No v2.1 empirical conclusion will be reported until all registered folds, seeds, variants, evaluations, and lineage checks are complete. Submission and artifact release also remain blocked until the institutional ethics determination, data-use review, and derived-pose release review contain genuine dated internal references. The project redistributes no raw video or identity-bearing frames, and derived artifacts remain non-public unless completed reviews permit release.
