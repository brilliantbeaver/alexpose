# Study 01 methodology: repaired fixed-reflection JEPA and video-disjoint transfer

> **Companion proposal:** [README_GAVD_ICLR.md](../proposals/README_GAVD_ICLR.md)
> **Execution roadmap:** [08-downstream-evaluation.md](../tutorials/08-downstream-evaluation.md)
>
> **Status:** this is a baseline-study protocol. The semantic-gauge method is
> specified separately in [Study 02](../studies/02-semantic-gauge-predictive-representations/).

## 1. Scope and preregistered claim

This protocol evaluates AMASS-pretrained skeleton JEPAs on `GAVD-VideoGroup-v2`. Its primary claim is restricted to **video-disjoint transfer of clinician-annotated normal-versus-abnormal gait patterns** from AMASS to a monocular pose cohort.

Do not claim participant-disjoint generalization, medical diagnosis, metric 3-D transfer, clinical force prediction, or real-camera invariance. GAVD's linked public videos lack reliable participant IDs, its poses are estimated, and its labels are gait-pattern annotations rather than verified diagnoses.

Before training, commit the code revision, dirty-worktree state, AMASS and GAVD manifests, joint schema, split manifest, seeds, metrics, effect threshold, tuning bounds, and extraction configuration. Never change one after inspecting an outer-test outcome.

## 2. Input and reflection contract

Each input has coordinates `x[B,T,J,3]`, visibility/confidence `v[B,T,J]`, and a JEPA training mask `m[B,P,J]`; visibility and training masks are distinct. `core11-v1` contains pelvis and paired hip, knee, ankle, heel, and forefoot joints. Freeze joint order, temporal patch size, body frame, frame rate, normalization, bilateral permutation `p`, and missingness semantics in a versioned contract.

An anatomical reflection must:

1. swap all bilateral joint IDs and their validity/confidence values;
2. negate only the mediolateral coordinate; and
3. preserve forward, vertical, and time axes.

Test `M(M(x,v)) = (x,v)` on real examples. Preserve transforms so visual overlays can detect a globally mirrored but internally consistent pipeline.

### 2.1 Orbit-closed masking

For paired arms, branch B is the anatomical reflection of A. A physical target must be unavailable in both coordinate descriptions:

```text
m_B[t, p(j)] = m_A[t, j]
v_B[t, p(j)] = v_A[t, j]
```

Apply this closure before selecting visible student tokens and before selecting teacher targets. Motion energy, tube/block selection, and stochastic draws obey the same permutation. The predictor takes `(state_A, state_B, m_A, m_B)`, not a single mask reused for both branches.

Run a copyability test with independent synthetic token values. Test direct mirror counterparts, adjacent time tokens, neighbouring joints, tube/block geometry, and the cross-attention receptive field. The test fails if a masked target has an allowed identical or deterministically recoverable counterpart in the other branch.

### 2.2 Validity-aware attention

Every encoder and predictor attention call receives a padding/attention mask derived from `v`; it also receives a learned validity/confidence embedding. Invalid coordinates are zero only as storage, never as a semantic coordinate. Invalid targets and invalid displacement differences are excluded from every loss and every pooling operation.

## 3. JEPA training protocol

Use an unmasked EMA teacher, contextual latent targets, target centering, and a fixed update budget. Report effective configuration, not only requested configuration. Match visible valid tokens (pilot 32, 64, and 96), AMASS source-window exposures, update counts, seeds, and tuning opportunities. Report parameter count, FLOPs, throughput, peak memory, and GPU-hours separately for exposure-matched and compute-matched comparisons.

Sweep the EMA start `{0.999, 0.9995, 0.9999}` toward 1.0. Monitor teacher entropy/perplexity, feature variance and effective rank, covariance, finite norms, and train/validation gaps. A model is ineligible if it fails health thresholds, exact-geometry checks, or the copyability test; a single moving-teacher KL is not an architecture ranking.

### 3.1 Architecture arms

| Arm | Required comparison purpose |
| --- | --- |
| `standard_sjepa` | one-view latent-prediction reference |
| `standard_mirror_aug` | reflected examples without paired fusion |
| `paired_shared_no_cross` | paired input without cross-branch communication |
| `paired_unconstrained` | matched cross-branch fusion without weight tying |
| `reflection_equivariant` | exact branch-swap equivariance throughout |

For `reflection_equivariant`, numerical commutation must hold through online encoder, target encoder, predictor, and masked prediction. This is an implementation property, not downstream evidence. `paired_unconstrained` is the closest architectural control.

### 3.2 Objective arms

| Condition | Student target | Sampler |
| --- | --- | --- |
| `U` | S-JEPA latent target | uniform orbit-closed tube/block |
| `S` | S-JEPA latent target | motion-aware orbit-closed tube/block |
| `T` | valid masked displacement only | uniform orbit-closed tube/block |
| `M` | latent and displacement | motion-aware orbit-closed tube/block |

In `T` and `M`, displacement is a fixed lagged in-clip coordinate difference. Normalize it from training data only, mask invalid terms, use a matched motion head, and report its loss separately. For `M`, set the combined-loss weight by a preregistered normalized-gradient rule or equal-budget development grid. Motion-aware sampling uses visibility-weighted, clipped motion energy and logs its temperature and induced token distribution.

Run all 20 cells at three paired seeds. Choose the condition using validation identities only, then run every architecture arm at five paired confirmation seeds. Open the AMASS test once after the configuration is frozen; use final EMA or a preregistered multi-draw rule, never a lucky single draw at schedule end.

## 4. `GAVD-VideoGroup-v2` construction

Snapshot the [official GAVD annotations](https://github.com/Rahmyyy/GAVD). The source release provides annotations and public-video links; retrieve videos only in compliance with platform terms and applicable ethics requirements. Do not redistribute videos.

For every annotation sequence, record canonical source-video ID/URL, annotation rows, label, frames, boxes, view, retrieval state, file hash, duration, frame rate, resolution, extractor version, tracking configuration, confidence, missingness, track continuity, and box–track agreement. Report every retrieval and extraction failure by class and view.

Use only official sequence boxes and a single frozen pose extractor/tracker for normal and abnormal material. Do not add hand-picked normal videos, switch extractors by source, or silently retime changed videos. Quarantine duplicate, changed, ambiguous, conflicting, or misaligned videos. Cap training windows per video to prevent a prolific source from dominating the fold.

The GAVD pose bridge is non-metric and must remain separate from any future SMPL-24 benchmark. Retain raw mapping/quality metadata but never provide source ID, filename, URL, boxes, or quality measures as JEPA classifier features.

## 5. Split, task, and leakage rules

The atomic group is canonical source video. All sequences, windows, crops, reflected copies, normalization statistics, and target-train SSL exposures from a video remain in one fold. Deduplicate exact files/URLs; where reliable uploader or original-source metadata exists, generate an additional source-family-held-out stress split.

Freeze the task after availability/label auditing but before examining model predictions:

| Role | Target | Requirement |
| --- | --- | --- |
| Primary | normal vs clinician-annotated abnormal | adequate independent video support in every fold |
| Secondary | normal vs pathological vs non-pathological abnormal | sufficient support after retrieval attrition |
| Exploratory | named gait patterns | sufficient independent videos; no diagnostic wording |

Use outer and inner `StratifiedGroupKFold` partitions by source video. Fit preprocessing statistics, pooling choices, classifiers, calibration, label subsets, and all hyperparameters exclusively in each outer-training set. Aggregate window embeddings to sequence then video before scoring. A source-family split is a robustness analysis, not a replacement for the primary video-grouped protocol.

## 6. Frozen encoder evaluation

For each fixed JEPA checkpoint and outer fold:

1. run the frozen encoder on training and held-out video windows with valid-token pooling;
2. fit a linear classification head only on outer-training video embeddings, selecting its hyperparameters in inner group CV;
3. aggregate held-out window predictions to sequence then video; and
4. save fold-level predictions, groups, quality flags, and selection traces.

Report out-of-fold video-level balanced accuracy and macro-F1; for binary classification add AUROC, AUPRC, calibration, and group-bootstrap confidence intervals. Do not headline sequence-level metrics. Report class/view attrition and every completed outer fold.

The label is reflection-invariant. Use a reflection-invariant readout for every model:

- `even_output` for a single-stream encoder averages logits from `x` and `M(x)`;
- paired encoders use the even representation or average logits under both branch orders; and
- `paired_unconstrained` and `reflection_equivariant` receive the same output wrapper.

`even_output` is the correct output-level baseline here. The signed-force study's `odd_output` baseline belongs only to a reflection-odd target.

Report these regimes in separate rows: **AMASS-only frozen encoder**, **unlabeled outer-train GAVD adaptation then frozen head**, and **supervised outer-train fine-tuning**. The last two are not zero-shot and cannot access held-out videos, including through SSL.

## 7. Baselines, falsification, and decision rule

Use the same folds, pooling, tuning budget, and head protocol for all arms and for random encoder, scratch supervised model, raw kinematics, matched non-JEPA masked motion, static pose, temporal shuffle/phase scramble, root-only, and pose-confidence/missingness/bbox-only controls.

Primary architecture comparisons are `reflection_equivariant` versus `paired_unconstrained` and versus `standard_sjepa` with `even_output`. To claim a specific internal-equivariance advantage, it must exceed both closest controls by the preregistered practical threshold across five paired seeds; report seed-level paired deltas and group-bootstrap intervals. A result reproduced by static pose, pose quality, or generic paired fusion is a null for the motion-representation claim.

Report reflection consistency, parity-channel rank/energy, and performance degradation under fixed temporal gaps, joint dropout, limb occlusion, pose noise, and left/right swaps. A deliberate swap is a falsification: a laterality-aware representation should change predictably, not necessarily remain accurate.

## 8. Pass/fail order

| Gate | Required before proceeding |
| --- | --- |
| Architecture | orbit-closed copyability test, validity-aware attention, and commutation checks pass |
| Pretraining | sealed manifest, healthy representations, matched pilot and confirmation seeds |
| GAVD cohort | official-only uniform extraction, video-grouped manifest, audited attrition, adequate class support |
| Transfer | frozen encoder beats closest controls at video level with all folds reported |
| Motion interpretation | static and nuisance controls do not explain the advantage |

Failure at any gate narrows the conclusion; it does not justify a weaker split, a post-hoc task, or a clinical claim. The participant-held-out force protocol remains the future route to signed asymmetry or clinical-performance evidence.
