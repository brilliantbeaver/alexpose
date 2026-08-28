# Fixed-reflection downstream evaluation and baseline roadmap

## Read this first

> **Study 01 only.** This roadmap governs the fixed-known-reflection baseline.
> It does not test the semantic-gauge contribution; see
> [Study 02](../studies/02-semantic-gauge-predictive-representations/).

**Current status:** `repaired-jepa-seed7-v2` is engineering evidence, not a
paper result. It now contains a standard-S-JEPA seed-7 artifact, but the saved
aggregate summary contains only that newest run; there is still no paired
multi-seed comparison or untouched AMASS test evaluation. Its near-zero
reflection residual confirms fixed-reflection wiring, not useful gait semantics.

**Immediate baseline question:**

> Does repaired bilateral-equivariant JEPA improve **video-disjoint, clinician-annotated gait-pattern transfer** beyond mirror augmentation, unconstrained pairing, and ordinary JEPA?

This is not a claim of participant-disjoint evaluation, clinical diagnosis, or metric motion capture. OUMVLP-Mesh is deferred while its licence is unavailable. The immediate downstream cohort is a full, fresh GAVD re-extraction: `GAVD-VideoGroup-v2`.

## 1. Blockers and required repairs

| Blocker | Why it invalidates the comparison | Required repair |
| --- | --- | --- |
| **Mirrored-target leakage** | Both paired branches receive the same token-index mask. After reflection, an original masked left joint can remain visible as the mirrored branch's right-index version of the *same physical joint*. Cross-attention can copy instead of predict. | Use an **orbit-closed mask**: `mask_B[t, p(j)] = mask_A[t, j]`, where `p` swaps left/right joints. Apply the same rule to validity, motion scores, tube/block masks, and random draws. Add a synthetic copyability unit test. |
| **Invalid joints are treated as zeros** | Padding and failed pose detections can act as plausible coordinates and create a missingness shortcut. | Pass validity/confidence embeddings and attention padding masks through every encoder/predictor path. Mask invalid coordinate and motion targets. |
| **Pilot is not a fair result** | The current stored arms each have one seed; the newest `standard_sjepa` run replaced the aggregate summary; `evaluate_test=false`; raw losses compare moving teachers. | Treat v2 as diagnostic. Run matched seeds, fixed updates, fixed multi-draw validation, one final AMASS test opening, and a sealed manifest. |
| **Current GAVD cache is source-confounded** | It contains 96 sequences from 18 videos; 12 normal sequences come from one video. Sequence-level splits leak video identity. | Retire it from paper tables. Rebuild the full official GAVD annotation release with video-grouped splits and a single extraction pipeline. |

## 2. Corrected pretraining protocol

### Inputs and invariants

- Keep `core11-v1` for the immediate GAVD transfer study; freeze joint order, left/right permutation, body frame, frame rate, normalization, and validity semantics in a versioned contract.
- Use two explicit inputs: `relative` (body-centred positions, bones, velocities) and `relative+root` (adds normalized root velocity/cadence). The latter must beat a root-only control.
- Match **visible-token counts**, not a copied mask ratio. With 176 Core11 tokens, compare 32, 64, and 96 visible valid tokens.
- Use an unmasked EMA teacher and monitor feature rank, teacher entropy, finite norms, train/validation gap, and covariance. Sweep EMA start `{0.999, 0.9995, 0.9999}`; the current 0.996 horizon is too short to assume it is appropriate.

### Architecture controls

| Arm | Isolates |
| --- | --- |
| `standard_sjepa` | ordinary one-view latent prediction |
| `standard_mirror_aug` | value of seeing reflected examples |
| `paired_shared_no_cross` | value of simultaneous paired encoding without communication |
| `paired_unconstrained` | value of paired cross-attention without an exact constraint |
| `reflection_equivariant` | value of exact bilateral equivariance |

All paired arms must use orbit-closed masks. Log parameters, visible/target tokens, views, FLOPs, updates, throughput, peak memory, GPU-hours, and failures. Compare both exposure-matched and compute-matched settings. For GAVD's reflection-invariant label, compare downstream classifiers with an `even_output` wrapper: average single-stream logits for a sequence and its reflection; use even features or branch-order-symmetrized logits for both paired arms.

### Objective and mask controls

Keep MAMP's two ideas separate: motion-aware sampling and masked motion prediction. Cross every architecture arm with:

| Condition | Target | Mask sampler |
| --- | --- | --- |
| `U` | S-JEPA latent | uniform orbit-closed tube/block |
| `S` | S-JEPA latent | motion-aware orbit-closed tube/block |
| `T` | displacement/motion only | uniform orbit-closed tube/block |
| `M` | latent + displacement, normalized fixed weight | motion-aware orbit-closed tube/block |

Define motion from valid in-clip coordinates, mask invalid differences, and use the same motion head in every relevant cell. Verify that reflection permutes both motion-energy maps and sampled masks. Tube/block masks need temporal exclusion tests: gait is periodic, so nearby frames and bilateral counterparts can otherwise reveal the answer.

## 3. Training order and gates

1. **Repair and test.** Implement orbit-closed masks, validity attention masks, copyability diagnostics, and the run manifest. A small smoke run must pass all properties before scale-up.
2. **Pilot.** Run the full 5 x 4 matrix with three paired seeds and matched source windows/masks where mathematically possible. Select only on validation identities and retain every trial.
3. **Confirm.** Train the chosen configuration across all architecture arms at five paired seeds, fixed updates, and a preregistered final-EMA or smoothed multi-draw selection rule. Open AMASS test once, after configuration freeze.
4. **Stop if pretext fails.** Do not scale a run with residual copyability, invalid-token leakage, collapse/rank/entropy warnings, or unsealed provenance.

Every run manifest records Git revision/diff, command, package/CUDA versions, source and split hashes, effective configuration, RNG/scheduler state, checkpoint hash, and metrics.

## 4. `GAVD-VideoGroup-v2`: immediate downstream cohort

The [GAVD annotations](https://github.com/Rahmyyy/GAVD) cover 1,874 gait sequences linked to 452 public videos. The release contains annotations and links rather than raw video; retrieve only in accordance with platform terms and applicable ethics requirements. This is a monocular, non-metric pose transfer benchmark—not a replacement for an estimated-SMPL or clinical cohort.

### Build it once, uniformly

1. Snapshot the official annotations and resolve every sequence to a canonical video ID/URL, label, frames, boxes, and view. Quarantine conflicting labels and alignment failures.
2. Retrieve only the official linked videos; record file hash, FPS, duration, resolution, retrieval status, and failure reason. Do not add manual normal videos.
3. Use official boxes and one versioned pose extractor/tracker for every class. Record confidence, missingness, track continuity, and box–track agreement. Reject changed/misaligned videos instead of silently re-timing them.
4. Deduplicate URLs/files. Keep every sequence, crop, pose window, and augmentation from one video in the same split. If reliable uploader/source-family metadata exists, use a source-family-held-out stress split too.
5. Cap training windows per video and aggregate windows to sequence then **one video-level prediction**. Report extraction attrition by class and view.

GAVD does not provide reliable participant IDs. Call this protocol **video-disjoint**; do not infer identities from faces or claim participant-disjointness.

### Frozen task and evaluation

- **Primary task:** normal versus clinician-annotated abnormal, scored at video level.
- **Secondary task:** normal versus pathological versus non-pathological abnormal, only if class support after retrieval is adequate.
- **Exploratory:** fine-grained gait-pattern labels, only with sufficient independent videos. These labels are not verified medical diagnoses.
- Use nested `StratifiedGroupKFold` by video ID for outer and inner splits. All tuning, normalization, target-train SSL adaptation, and fine-tuning occur inside the outer-training fold.
- Primary metrics: balanced accuracy and macro-F1; add AUROC/AUPRC for binary classification, out-of-fold confusion matrices, and group-bootstrap confidence intervals. Do not headline sequence-level accuracy.

Evaluate three separate rows: (1) AMASS-only frozen encoder plus fold-trained linear head, (2) unlabeled target-**train** adaptation plus frozen head, and (3) supervised target-train fine-tuning. The latter two are not zero-shot and must never see held-out videos.

## 5. Controls that decide the claim

Use the identical downstream protocol for every JEPA arm and for:

- random encoder and scratch supervised model;
- raw positions/bones/velocities and matched non-JEPA masked-motion baseline;
- static-pose-only and temporally shuffled/phase-scrambled inputs;
- pose confidence/missingness/bbox-only features; and
- root-only input where root motion is available.

`even_output` is the correct output-level symmetry baseline for normal-versus-abnormal classification; the clinical force proposal's `odd_output` applies only to a signed right-minus-left target. A gain that disappears under video grouping, is matched by static pose, pose quality, or `even_output`, is not motion-representation evidence. Reflection residuals are property tests; to demonstrate a meaningful equivariant representation, also report reflection consistency, parity-channel activity, and degradation under controlled temporal gaps, limb occlusion, left/right swaps, and pose noise.

## 6. What the paper may claim

If the corrected models outperform `paired_unconstrained` and `standard_sjepa` with `even_output` across five seeds and the video-disjoint GAVD protocol, the paper can claim that bilateral-equivariant JEPA improves **gait-pattern transfer from AMASS to an in-the-wild monocular pose cohort**, subject to the listed stress tests.

It may not claim clinical diagnosis, participant-disjoint generalization, metric 3-D transfer, or a general advantage of equivariance if the controls fail. OUMVLP-Mesh/Gait3D remain valuable future identity-retrieval confirmation once data access and separate SMPL-24 contracts are available; participant-held-out kinetic data remains necessary for clinical/asymmetry claims.

## References

- [MAMP, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.html)
- [S-JEPA, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf)
- [GAVD](https://github.com/Rahmyyy/GAVD) and [paper](https://arxiv.org/abs/2407.04190)
- [The Paradox of Motion, FG 2024](https://arxiv.org/abs/2402.08320)
