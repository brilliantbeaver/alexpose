# Accuracy review of `wacv-v1-revision.tex`

Reviewed 16 October 2026 against the seven top-level notebooks, their retained outputs, the saved experiment artifacts, and the implementation called by the notebooks. All manuscript line numbers below refer to SHA-256 `67a5ed3f42f3a01b40f458e8ee0e71d00f0f3516655390604cf4a796b86385b3`.

## Overall assessment

**No material numerical error or contradiction with the executed methods was found in the revised manuscript.** Its main positive claim is supported: a frozen S-JEPA representation, trained on 24 source recordings per fold and paired with a supervised linear classifier, exceeds the specified corrected joint-angle Random Forest on held-out collection-label prediction under both reported weighting schemes. Mean pose remains close and performs better under equal clip weighting, as the paper already states.

The manuscript distinguishes this descriptive result from the additional questions of how much pretraining helps, whether temporal order matters, and whether the features measure neurological impairment. Those qualifications should remain. The available evidence supports a positive small-data representation-learning case study without requiring a claim of general superiority or clinical validation.

One inference should be softened: line 229 treats the visibility baseline's score as demonstrating label-associated information. The score is verified, but the experiment does not establish that stronger interpretation. The principal supporting-documentation problem is in the notebooks: several narrative cells retain older results beside newer executed outputs. The manuscript uses the current results correctly. Three additional wording refinements would improve precision in the training discussion; none changes a reported result or requires another training run.

## Verified classification results

The primary execution is `capstone-20261009T060955236580Z`, under `artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/`. Notebook 06, cell 11, prints this path, and cell 14 displays its summary. Cell numbers in this review are one-based, counting both code and Markdown cells.

| Method | Source-weighted macro-F1 | Clip-weighted macro-F1 | Correct clips |
|---|---:|---:|---:|
| S-JEPA + linear classifier | 0.456624 | 0.397034 | 36/88 |
| Corrected joint-angle RF | 0.325184 | 0.320940 | 30/88 |
| Mean pose + linear classifier | 0.451893 | 0.409840 | 37/88 |
| Visibility + linear classifier | 0.317565 | 0.318057 | 29/88 |
| Majority class | 0.259398 | 0.202125 | 23/88 |
| Earlier RF with duplicated ankle-range field | 0.411263 | 0.397316 | 37/88 |

The earlier RF row comes from the separate 21 September execution and is correctly distinguished in the paper. The main table at lines 194–203 rounds all values correctly. The source-weighted differences are **0.131440** versus corrected RF, **0.045361** versus earlier RF, and **0.004731** versus mean pose. Therefore the paper's differences of 0.131, 0.045 and approximately 0.005 are correct; subtracting already-rounded scores would give the wrong first two differences.

Other result statements also check out:

- S-JEPA exceeds corrected RF in three of five folds, and mean pose exceeds S-JEPA in four. S-JEPA's fold scores range from 0.217267 to 0.779598 (line 210).
- Source-weighted recalls are 39.6%, 42.9% and 56.9% for S-JEPA and 27.1%, 23.1% and 51.0% for corrected RF. S-JEPA's PD precision is 38.6% (line 212).
- S-JEPA correctly classifies 10 Normal, 8 MS and 18 PD clips; 13 MS clips are assigned to PD. These raw counts are correctly distinguished from source-weighted percentages.
- Correcting the RF feature changes 12 predictions. The two executions produce identical S-JEPA predicted labels despite small numerical differences in learned weights and embeddings. They are correctly described as the same-seed executions, rather than independent replications (line 207).
- All 88 clips have exactly one held-out prediction, their labels and source IDs agree with the registry, and training, validation and test source IDs are disjoint within each fold.

The independent recalculation in [results-verification.json](evidence/results-verification.json) checks every saved metric, per-class statistic and confusion-matrix entry. A separate reviewer also reconstructed the source weights using exact rational arithmetic; see [results-review.md](results-review.md).

## One inference to revise

**Location:** main manuscript line 229, Discussion, “Scope of the evidence,” final sentence beginning “The visibility-only score.”

**Priority:** limited-scope correction to the interpretation, with no numerical change.

The text says that the visibility score of 0.318 “shows that detector outputs contain label-associated information.” The retained predictions establish the measured classification score. There is no shuffled-label or other reference analysis establishing that the score reflects a reproducible association beyond chance variation. The following qualification discusses the possible source of such information, while leaving its existence asserted. Reporting the observation and keeping that interpretation open is more defensible.

**Replace the final sentence of the paragraph with:**

```latex
The visibility-only classifier reaches source-weighted macro-F1 of 0.318. Whether this reflects a reproducible association with the collection labels, and whether movement or recording conditions contribute to that association, remains unresolved.
```

**Evidence:** notebook 06, cell 14, and the October `oof.json` reproduce visibility source-weighted macro-F1 of 0.3175646279. The comparison is fitted and evaluated in `sjepa/full_experiment.py:194–199`; the executed notebooks contain no corresponding shuffled-label analysis. This correction does not imply that visibility contains no useful information, and does not alter the observed S-JEPA–RF ranking. No significance test needs to be added to the main paper to make this wording accurate.

This issue was raised by the fresh Codex challenge review and verified against the manuscript, current predictions and executed comparison code. The other review passes accepted the broader framing; the final recommendation retains the narrower wording because it states exactly what was measured.

## Additional wording refinements

These are precision improvements, not corrections to the central result. They can be made without adding statistical jargon or weakening the observed comparison with RF.

### 1. Make the continuation conclusion explicitly about validation

**Location:** [main manuscript](../wacv-v1-revision.tex), line 217, Results, “Learning during training.”

The phrase “does not consistently improve label prediction” follows validation results, so its intended meaning is understandable. However, the saved experiment evaluates the test predictions only for the selected stage; it does not provide a paired test comparison of both stages. Naming validation removes that ambiguity.

**Replace the paragraph with:**

```latex
The continuation stage has a lower final training-batch loss in every fold, but improves source-weighted validation macro-F1 in only folds 3 and 4. The initial stage is retained in folds 0--2. Lower final training-batch loss therefore does not consistently coincide with better validation classification in this experiment; the validation step determines which stage is used for each test fold.
```

**Evidence:** notebook 06, cell 11; the October `results.json` selections; `sjepa/full_experiment.py:163–183`. Continuation validation scores are 0.276105, 0.250462, 0.257210, 0.332239 and 0.559710. Initial scores are 0.293979, 0.490618, 0.324930, 0.266008 and 0.375309.

### 2. Describe the measured training behavior directly

**Locations:** main manuscript lines 215 and 226, where the text says the model “fits” its feature-prediction task.

The loss values and effective ranks are correct. The target encoder and the centering term change during training, so decreasing loss is measured against evolving targets. Effective rank measures variation within sampled training batches. The existing qualification about an untrained encoder already prevents a major overclaim; direct descriptions of these observations would make the evidence easier to assess.

**Replace only the first sentence of line 215 with:**

```latex
Training-batch feature-prediction loss decreases while the sampled target-encoder features retain variation across inputs.
```

**Replace only the second sentence of line 226, beginning “The model fits,” with:**

```latex
The training records show decreasing feature-prediction loss and nonconstant features on sampled training batches, while the frozen-feature classifier measures information associated with the collection labels.
```

The remaining numerical sentences in line 215 can stay. The fold-0 first/last 50-update mean losses are 3.858368 and 1.022220, and the final effective rank is 12.134889. The initial-stage endpoint ranks across folds span 10.895738–12.134889, agreeing with the rounded range in the manuscript.

**Evidence:** notebook 03, cells 15 and 19; October run diagnostics; `sjepa/train_v2.py:215–239`; `sjepa/losses.py:82–92`. The diagnostic averages all target output tokens, whereas the classifier averages a fixed subset of output positions. The supplement already states this distinction correctly.

### 3. Name what the proposed untrained comparison would measure

**Location:** main manuscript line 226, sentence beginning “A matched untrained encoder would establish.”

“Establish how much of that information is gained” is stronger and less concrete than the proposed comparison warrants. The observable quantity would be a difference in predictive performance under a matched evaluation.

**Replace that sentence with:**

```latex
A matched untrained-encoder comparison would estimate the effect of pretraining on held-out label prediction under the same evaluation procedure.
```

This preserves the current positive result and states the next scientific question precisely. No untrained-encoder result is available in the executed comparison.

## Supporting notebook corrections

These inconsistencies should be resolved when preparing a reproducible notebook release. **They do not justify changing the manuscript back to older scores.**

| Notebook location | Stale narrative | Current evidence |
|---|---|---|
| Notebook 06, Markdown cell 12 | Continuation validation F1 of 0.344 in fold 1 and 0.284 in fold 3; runtime of 216 seconds | Executed cell 11 gives 0.250 and 0.332, respectively, and 257.4 seconds. Stage choices remain the same. |
| Notebook 06, Markdown cell 15 | RF described as the current extractor at source F1 0.411 and clip F1 0.397 | Executed cell 14 gives corrected RF values of 0.325184 and 0.320940. The 0.411 result belongs to the earlier implementation. |
| Notebook 06, Markdown cells 9 and 18 | The ankle correction is described as outstanding; RF is said to recognize 5 of 29 MS clips | The correction is present in the October execution, whose RF predictions recognize 4 of 29 MS clips. |
| Notebook 06, Markdown cells 19–20 | Discussion still treats the RF correction as outstanding or retains the older comparison | The October artifacts and present manuscript document the completed correction and retain the earlier score separately. |
| Notebook 03, Markdown cell 18 | Says only the final rank was printed and other values were graph estimates; quotes the earlier rank endpoint | Executed cell 19 contains exact first/last 50-update summaries. Current cell 15 gives 12.134888648986816; both the older and current endpoints round to 12.13. |

For future maintenance, build result tables and interpretive summaries from the selected run's saved metadata, or label historical commentary explicitly. Notebook filenames containing `pretrain_normal` and `vicreg` also refer to older stages of development; the current execution uses all three training-label groups without a condition-label or VICReg term in the encoder loss.

## Methods and inference checks

| Manuscript claim | Verification |
|---|---|
| 91 raw clips, 88 usable clips, 41 sources; 15 padded clips; 358 windows | Counts match notebook outputs, registry and cached arrays. Label-specific counts in Table 1 also agree. |
| 24 fitting sources and 45–58 fitting clips per fold | Matches all five registry partitions. The additional 8–9 validation sources are properly disclosed at line 148; total development uses 32–33 sources. |
| Source separation | Confirmed for identified source IDs. Participant independence cannot be verified and is not claimed. |
| Pose normalization and temporal windows | Saved normalized arrays agree exactly with recomputation from cleaned cached arrays for all 88 clips. Integer-stride sampling, complete 32-frame windows, stride 16 and repeat-last-frame padding match the code. |
| Anatomical masking | A fresh 5,000-mask check with seed 42 gives mean hidden fraction 0.6320439394. The stated 63.2% is correct. The 1.5 factor applies to region-selection weights, as the manuscript explains. |
| Classification readout | The full input enters the target encoder. The fixed seed-zero mask selects 161 of 264 output positions for averaging. It is not an input mask at classification time. |
| Training objective and continuation | Encoder fitting ignores collection labels. Both stages execute 800 and 400 updates, respectively; the continuation restarts optimizer, center and random-number state. The manuscript correctly distinguishes this from uninterrupted training. |
| Baseline construction | The corrected RF has 15 varying joint-angle fields in an 82-field container; mean pose and visibility each have 66 features. Training-only fitting and differences in inputs are disclosed accurately. |
| Practical representation utility | Supported for the trained encoder plus supervised readout on held-out collection labels. The experiment does not isolate the benefit caused by pretraining. |
| Temporal and clinical interpretation | Adequately bounded by the mean-pose comparison, uncertain participant identity, unverified diagnoses and potential recording cues. Training-only visualizations in notebook 05 do not establish clinical clusters, and the manuscript does not use them that way. |

The code-content digest for the current `sjepa` and `ambient` implementation matches the October run's provenance. Detailed checks and exact code anchors are recorded in [methods-review.md](methods-review.md) and [inference-review.md](inference-review.md).

## Facts outside the notebook verification boundary

The notebooks establish the inventory and processing of the supplied clips. They do not independently document every historical curation action in line 58: which YouTube search terms were used, how titles determined labels, which walking segments were manually selected, or which footage was cropped. These are author-reported collection procedures. Retain them if they reflect the actual collection process, and preserve the collection records that support them. Their absence from the numerical notebooks is an audit limit, not evidence that the statements are false.

The current pose-extraction defaults support the described MediaPipe setup, but the reused pose caches do not record an extraction-time model-asset hash and SDK version. The exact historical extraction environment therefore remains less directly verifiable than the downstream training configuration. Similarly, the recorded source partitions establish separation during the executed evaluation; they do not establish that earlier development decisions were made without inspecting this collection's test results. The paper's interpretation as an internal case study should remain.

The capstone checkpoints do not retain full per-update loss histories. The first/last 50-update summaries can therefore be checked against notebook 03's saved numerical output and its matching checkpoint payload, but cannot be independently regenerated from a retained loss vector. This is a provenance limitation rather than a numerical discrepancy.

No new literature review or independent clinical validation was performed in this notebook-focused review. No encoder or classifier was retrained. The manuscript, its PDF and the notebooks were left unchanged.

## Review record

The review covered all seven notebooks: 124 cells, including 55 executed code cells. Saved text outputs were exported with cell numbers; original file hashes appear in [input-manifest.json](evidence/input-manifest.json). Checks covered both saved full executions, all 88 pose-cache hashes, feature changes, checkpoint provenance, predictions, source partitions and every saved classification metric. Three parallel reviewers examined numerical results, methods and scientific inference. A separate read-only Codex challenge review inspected an immutable evidence snapshot; its final disposition is recorded in [review-disposition.md](review-disposition.md).

All review files are under `docs/wacv2027-v1/review-2026-10-16/`. The full notebook text exports retain older commentary deliberately so that the discrepancies above remain inspectable.
