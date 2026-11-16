# Independent implementation review

The fresh extraction and dynamics implementations were reviewed before any new real outer-fold performance was inspected. The reviewer did not write either implementation. The initial scientific critique and pre-outcome design review are recorded separately in [initial_adversarial_review.md](initial_adversarial_review.md) and [design_adversarial_review.md](design_adversarial_review.md).

The inspected model implementation is accepted for the declared real-data run. This disposition establishes software behavior and protocol agreement. It does not establish that the model predicts useful dynamics, that any comparison is positive, or that GAVD supports clinical generalization. Real-cohort reconstruction and final-result review remain separate acceptance steps.

## Reviewed identities

| Artifact | SHA256 |
| --- | --- |
| `worldmodels/restart_data.py` | `ebbff76426b5f576690f7bd95051a77781e5bcdd18f72db23e9bfdbdf6bb6376` |
| `tests/test_restart_data.py` | `3e50b1a6f56bba88750bd0325ebe80bc4c7dac29829827e053e37d889ee7765e` |
| `worldmodels/restart_experiment.py` | `226b276ae3d0759624aec9ae6acff59d0cf282b795bc309db814c4f49fe8259c` |
| `tests/test_restart_experiment.py` | `92000e283a457354150cb22f70c0e470000f7d00f303f29f37c0adcfedad88c7` |
| `docs/restart/protocol.md` at implementation acceptance | `e7c331238c3ae61237336c11b36526afb20952ae31da51d9374962e98eafd299` |

Material changes to these artifacts reopen the affected checks. The actual real-run identity must match the accepted implementation or have a documented subsequent review.

## Findings resolved before fitting

**The temporal control required a substantive correction.** A single fixed feature permutation can be absorbed by linear regression. Moving the permutation before the eight-position selection additionally changes which observations the model receives. The coordinator identified this problem before model outcomes. The revised implementation first selects the same eight observations, then rotates the first seven opposite-leg coordinate and validity pairs by a stable window-specific offset in one through six. It preserves the latest observation, own-leg inputs, target, anchor and input dimension. Independent tests verify those invariants. This control disrupts relative time alignment while preserving selected values; it does not constitute a physiological intervention or establish a neural mechanism.

**Masked fitting needed to preserve the declared observation weights.** The initial implementation used hierarchical window weights but omitted normalization for the number of observed target coordinates. A window with more visible targets would therefore contribute more total coordinate-loss mass. Following review, each window's target-loss weight is multiplied by the output dimension divided by its observed target-coordinate count. The per-output masked solve now corresponds to equal total loss mass per hierarchically weighted window. Missing targets remain excluded. The weighted ridge solution is independently compared with scikit-learn's solver.

**Input identity verification was strengthened.** The initial runner recorded hashes but did not compare them with the preparation record. The accepted runner checks the saved array and metadata hashes and the 24-frame, 16-prefix, nominal-10-Hz contract before fitting. A complete reviewer replay confirms that a corrupted preparation hash is rejected.

**Target support and leg aggregation were made explicit.** The full-joint 70% quality threshold alone does not guarantee sufficient distal-leg targets. The runner requires at least four observed joint-time targets per leg by the four-step horizon and at least eight by the eight-step horizon, with prefix observation support for each target joint. Every method receives the same retained windows and mask. Each leg's observed-coordinate error is averaged first, and the two legs receive equal weight in the window score. The protocol now declares both rules and the runner saves excluded rows.

## Data adapter assessment

The adapter requests nearest native frames on a nominal 10 Hz grid separately within contiguous annotation spans. It decodes a source once, calls the shared Ambient RGB pose estimator on requested frames, records native frame numbers and decoder timestamps, and converts one-based annotation frames to zero-based decoded indexes exactly once. Identical source/frame extraction requests are shared, while per-sequence annotation gates are applied separately. Duplicate source/frame window tuples are retained only once.

Cache identities include annotation, media, pose weights, extraction implementation and support modules, normalization, gate and exclusion identities. The adapter compares current videos against the media-manifest hash where one is available. It invalidates unresolved annotation geometry, preserves missingness, and uses prefix-only torso scaling. Source progress records distinguish actual new model calls from cache reuse.

No blocking defect was found in these inspected paths. Full-image first-person detection and pelvis containment still cannot guarantee the intended person's identity in overlapping scenes. The sampling grid is nominally, rather than exactly, uniform in physical time. The adapter records actual intervals and grid error so this limitation can be quantified. Neither successful extraction nor cache consistency establishes anatomical truth.

## Model and evaluation assessment

All learned forecast fits use normal sources outside the current outer fold. Inner splits operate on source IDs, and feature scaling is fitted inside each inner training fold. Each arm and leg chooses ridge strength using equal-source validation error. Shrinkage variants additionally select beta on the same permitted inner data. Unshrunk arms remain primary and the secondary zero-beta choice means persistence, not successful learned dynamics.

Inference features use only the visible prefix. Forward filling cannot take a later observation into an earlier prefix position. Future values and future validity do not affect checkpoint predictions. Normalization is body-relative and uses a shared pelvis reference, so the own-leg name does not imply raw unilateral anatomy. The shifted arm is fitted independently; inference-only disruptions of the bilateral fit are also saved as complementary controls.

Training and reporting preserve source, sequence and window relationships. Nested folds and model seeds do not become independent patients. The source bootstrap conditions on the fixed, overlapping cross-validation fits, as the protocol states. Any secondary pooled out-of-fold AUROC retains the additional assumption that scores from different fitted fold models are comparable.

## Checks actually executed

The reviewer independently ran the data and model test modules together:

```bash
PYTHONDONTWRITEBYTECODE=1 experiments/aaai-worldmodels/.venv/bin/python -m pytest \
  -c experiments/aaai-worldmodels/pyproject.toml \
  experiments/aaai-worldmodels/tests/test_restart_data.py \
  experiments/aaai-worldmodels/tests/test_restart_experiment.py \
  -q -p no:cacheprovider
```

The result was **14 passed in 1.43 seconds**. These checks cover sampling and gaps, future-invariant scaling and features, target masks, strict geometry handling, cache identity, weighted ridge arithmetic, hierarchical aggregation, source folds, and the corrected temporal control.

A separate complete synthetic replay used twenty generated windows from ten source groups, five outer folds and three inner folds. It completed in **12.23 seconds**. The reviewer verified all eighty saved checkpoint predictions against the saved forecast array, then changed hidden future coordinates and validity and obtained exactly unchanged predictions. The replay also verified normal-only training membership, inner and outer source isolation, source-score reconstruction, every output-manifest hash, and rejection of a corrupted input identity. This was a software test using synthetic inputs, not real empirical evidence. Its machine-readable record is `artifacts/restart_review/synthetic_replay.json`; the replay source and artifacts are retained in the same scratch directory.

The independent real-input replay script is prepared at `artifacts/restart_review/audit_prepared.py`. It will reconstruct retained windows from raw source caches and annotation gates, check source and annotation hashes, and verify frame/time identities and exclusions after full extraction completes. Its result must be recorded before treating the fresh cohort as accepted evidence.
