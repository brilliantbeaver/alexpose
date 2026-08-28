# Study 01 legacy entry: repaired fixed-reflection JEPA transfer on GAVD

> **Status:** retained fixed-reflection baseline and engineering study. It is not
> the program's proposed representation-learning contribution; see
> [Study 02](../studies/02-semantic-gauge-predictive-representations/).
>
> **Detailed protocol:** [METHODS_GAVD_ICLR.md](../methods/METHODS_GAVD_ICLR.md)
> **Training and evaluation runbook:** [08-downstream-evaluation.md](../tutorials/08-downstream-evaluation.md)

## Decision and boundaries

This baseline study asks one narrow question:

> After removing cross-branch target leakage, does bilateral-equivariant JEPA outperform ordinary JEPA, mirror augmentation, and matched unconstrained pairing on video-disjoint GAVD gait-pattern classification?

The primary label, normal versus clinician-annotated abnormal gait, is unchanged by anatomical reflection. The downstream prediction must therefore be **reflection-invariant**. This is a representation-transfer study from AMASS to monocular pose data. It is not participant-disjoint validation, clinical diagnosis, metric 3-D transfer, or a claim about signed force asymmetry.

The retained clinical signed-force proposal in [README_FORCE_FUTURE.md](./README_FORCE_FUTURE.md) remains a Study 03 validation route. Its `odd_output` baseline is correct for a signed right-minus-left target, but is not valid for GAVD's reflection-invariant labels. This study instead uses an `even_output` control.

## What is currently wrong

The existing `repaired-jepa-seed7-v2` checkpoints are useful diagnostics, not paper results.

- The paired JEPA uses one token-index mask for both an original skeleton and its reflected counterpart. A masked physical joint can remain visible in the other branch at its swapped left/right index, allowing cross-attention to copy the target.
- Invalid joints are zero-filled but not carried as attention validity, so missingness can become a pose shortcut.
- The run contains only configured seed 7, omits the ordinary S-JEPA arm, has no AMASS test evaluation, and chooses an endpoint from one validation draw.
- The legacy GAVD cache has only 18 source videos, with 12 normal sequences from one video. It is source-confounded and must not appear in a paper result.

## The repaired model

For a reflection permutation `p`, masks must close over the two-branch orbit:

```text
mask_B[t, p(j)] = mask_A[t, j]
valid_B[t, p(j)] = valid_A[t, j]
```

Use this rule for coordinates, validity, motion scores, tube/block masks, and paired random draws. The encoder and predictor must accept branch-specific masks; a synthetic copyability test must show that mirrored counterparts, temporal neighbours, and overlapping receptive fields cannot reveal a target through an unintended route.

Every attention path receives an explicit validity/padding mask plus validity/confidence embeddings. Invalid coordinates and invalid motion differences never contribute to a target loss.

## Pretraining comparisons

Train the following arms with matched parameter counts where possible, identical AMASS source windows, paired seeds, and both exposure-matched and compute-matched reports:

| Arm | Question |
| --- | --- |
| `standard_sjepa` | Does ordinary latent prediction work? |
| `standard_mirror_aug` | Does reflected exposure alone help? |
| `paired_shared_no_cross` | Does paired encoding help without communication? |
| `paired_unconstrained` | Does cross-branch fusion help without an exact constraint? |
| `reflection_equivariant` | Does exact bilateral equivariance add value? |

Separate MAMP's two changes rather than naming them as one intervention:

| Condition | Latent/motion target | Sampler |
| --- | --- | --- |
| `U` | S-JEPA latent target | uniform orbit-closed tube/block |
| `S` | S-JEPA latent target | motion-aware orbit-closed tube/block |
| `T` | masked displacement target | uniform orbit-closed tube/block |
| `M` | latent + displacement, fixed normalized weight | motion-aware orbit-closed tube/block |

Train the 5 x 4 pilot at three paired seeds. Select with validation identities only, then confirm the selected condition across all arms at five paired seeds. Monitor feature rank, teacher entropy, finite norms, covariance, train/validation gap, target-copyability, and commutation; raw JEPA loss from separately moving teachers does not rank architectures.

## GAVD downstream evaluation

Build `GAVD-VideoGroup-v2` from the full [official GAVD annotation release](https://github.com/Rahmyyy/GAVD), not the legacy cache.

1. Freeze the annotation snapshot, source-video registry, extraction configuration, quality thresholds, class definition, and group splits before inspecting model outcomes.
2. Use the official sequence boxes and a single versioned pose extractor/tracker for every retrievable video. Record source IDs, file hashes, timing, pose quality, missingness, extraction failures, and attrition by class/view. Do not add manually selected normal videos.
3. Keep every sequence, crop, window, and reflected copy from a source video in the same split. Deduplicate URLs/files; where reliable, add a source-family-held-out stress split.
4. Use nested `StratifiedGroupKFold` by source video. Aggregate windows to sequences and then one video-level prediction. The primary endpoint is video-level balanced accuracy and macro-F1 for normal versus abnormal.

GAVD does not provide reliable participant identifiers. Results are **video-disjoint**, not participant-disjoint. Fine-grained pathology labels are exploratory gait-pattern tags, not medical diagnoses.

## The downstream controls that decide the result

Every classifier is reflection-invariant because the label is reflection-invariant.

- `even_output`: for a single-stream encoder, average the class logits for `x` and `Mx`.
- paired models: use the even representation or symmetrize logits from both branch orders. Apply the identical wrapper to `paired_unconstrained` and `reflection_equivariant`.
- compare random and scratch encoders; raw kinematic features; a matched non-JEPA masked-motion baseline; static-pose, temporally shuffled, root-only, and pose-quality/missingness-only inputs.

For each encoder, report three distinct regimes: AMASS-only frozen encoder plus fold-trained linear head; unlabeled target-**train** adaptation then frozen head; and supervised target-train fine-tuning. Only the first is AMASS-only transfer; none may access held-out videos.

## Success and failure

The architecture claim requires a pre-registered practical advantage of `reflection_equivariant` over both `paired_unconstrained` and `even_output`, across five paired seeds, with all GAVD folds and nuisance controls reported. A gain matched by static pose, pose quality, or unconstrained fusion is not evidence that equivariance improved motion representation.

If the repaired model fails, report the null: output symmetrization, augmentation, or unconstrained pairing was sufficient for this task. A later participant-held-out force study is needed before making the stronger clinical/asymmetry claim.
