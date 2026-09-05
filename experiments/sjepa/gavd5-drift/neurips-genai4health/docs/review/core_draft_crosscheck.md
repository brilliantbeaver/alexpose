# Adversarial crosscheck of the GenAI4Health drafts

Review date: September 5, 2026. Reviewed sources: `genai4health_paper_draft.tex` and `genai4health_extended_abstract.tex`, current notebook 00–07 source and numerical outputs, relevant notebook-08 preprocessing/pooling code, checked normal-anchor artifacts, and the local evidence builder. No manuscript was edited by this reviewer.

Reviewed paper SHA-256: `be8063d08ab6db7e8611670cdd9fbcf7c2cf76b4160af0081bafdf35aec9a290`.

Reviewed extended abstract SHA-256: `bb15cf26fbe88d8aa7f4dff76ba9fa83053ff83c1901fb4ad83c3f006508126b`.

## Assessment

The central numerical claims and current implementation description agree with the inspected evidence. The drafts correctly distinguish a retrospective case study from confirmatory generalization, clinical validation, architecture-wide inferiority, and an implemented health agent. They explicitly disclose the readout aggregation mismatch, annotation-informed training order, incomplete experiment grid, incomplete runtime configuration, and limits of hash/access-log evidence.

No numerical blocker was found in the paper or extended abstract. Two methods details should be made explicit because they affect reproducibility of the new central normal-anchor result. Additional wording recommendations prevent readers from interpreting registry entries as external preregistration or coordinate cosine as retained function.

## Corrections to make

### 1. Name the anchor diagnostic's preprocessing variant

Paper appendix, preprocessing paragraph (reviewed line 157), says only the temporal diagnostic has a different interpolation path. Notebook 08 also calls `interpolate_low_visibility(..., max_gap=4)` before `center_and_scale` and temporal resizing. The current notebook-04/06 training/readout path does not use that short-gap interpolation. Because anchor weighting is now a central finding, its variant belongs in the explicit methods.

Suggested replacement: “The normal-anchor and temporal diagnostics use a separate preprocessing path that interpolates short internal low-visibility gaps before normalization.”

This does not invalidate the weighting comparison: both reported means are calculated from exactly the same cached vectors and differ only in aggregation weights. It prevents an incorrect inference that the classification and anchor analyses differ solely in output pooling.

### 2. Specify the 12-joint anchor pooling domain

Paper appendix, feature definitions (reviewed line 169), says the anchor audit uses a 64-dimensional validity-weighted pooled EMA vector. Notebook 08's `target_embeddings` pools only `tokens[:, :, MASK_KEYPOINTS]`, using validity weights for those same 12 joints. It does not pool all 33 joints for the anchor vector.

Suggested replacement: “The normal-anchor audit instead pools valid EMA tokens over the 12 selected joints to produce a 64-dimensional vector.”

No numerical change is required.

## Wording improvements

- In both abstracts, prefer “embedding cosine” or “coordinate cosine” to “retention cosine.” The fuller paper correctly explains that cosine is not functional retention; the abstract should preserve that distinction on its own.
- In both abstracts, identify the 64 clips as the **normal-validation subset**. There are 131 total validation clips; the 64 count is correct only for the normal-annotated subset.
- “Five folds are registered” is technically compatible with entries in a split registry, but can sound like independently timestamped preregistration. Prefer “five folds are specified in the frozen source registry.” The existing caveat about absent immutable preregistration is correct.
- Figure `source_weighting_and_drift` caption should say “local source aliases,” rather than “non-identifying local source aliases.” Aliasing avoids printing source IDs; it does not independently establish anonymity or resistance to linkage through public counts.

## Numerical and method claims checked

| Claim | Verified evidence | Result |
|---|---|---|
| Raw cohort 666 sequences / 103 sources | Raw manifest | Correct |
| Metadata-public 657 / 100 | Public manifest | Correct |
| Decode eligible 655 / 98 | Decoded eligibility flag in manifest | Correct |
| Pose eligible 639 / 97 | Authoritative fold-0 QC ledger | Correct |
| Post-QC train/validation/test 377/59, 131/18, 131/20 | QC role aggregation | Correct |
| Category-specific table counts | Raw/public/decode/QC ledgers | Correct |
| Test source support 7, 2, 3, 6, 2 | Saved source predictions, stated category order | Correct |
| Macro-F1 .292 / .251 / .441 | Independently recomputed from source predictions | Correct rounding |
| Balanced accuracy .257 / .248 / .443 | Same predictions | Correct rounding |
| Accuracies .300 / .300 / .500 | Same predictions | Correct |
| Raw-minus-latent macro-F1 .148 | Exact difference .148088578089 | Correct rounding |
| All lanes miss all three stroke sources | Per-lane confusion matrices | Correct |
| Normal validation weighting .889 versus .701 | Checked cached similarities; source counts 60,1,1,1,1 | Correct |
| Dominant source 93.75% clip weight, 20% source weight | 60/64 and 1/5 | Correct |
| Dominant source cosine .905 | Checked source mean .904993951321 | Correct rounding |
| Test-normal cosine .850 | Checked mean .849632382393 on seven sources | Correct rounding; not a repair gain |
| Temporal R² values, including negatives | Current fold-local temporal CSV | Correct rounding |
| Smooth-L1 + .10 variance + .01 covariance | Notebook 04 and checkpoint objective | Correct |
| Annotation-informed schedule with no primary label loss | Notebook 04 and checkpoint | Correct |
| Four-frame coordinate means, 528 tokens, width64, 2 layers | Producer model and saved config | Correct |
| Online/predictor/projector parameter counts | Saved tensor-state counts | 70,528 / 50,368 / 8,320, total129,216 |
| 20 epochs × five stages | 100-row training history | Correct |
| Selected epochs7,0,0,4,0 | All five stage checkpoints | Correct; zero-based is disclosed |
| Uncertified defaults rather than certified runtime settings | Notebook defaults and missing serialization | Correct framing |
| C values1,1,10; refit on77 development sources | Notebook06 selection code and contracts | Correct |
| Validation means versus test mean probabilities | Notebook06 implementation | Correctly disclosed |
| No test in training access log | Saved checkpoint plus IDs | Correct; does not prove historical secrecy |
| Legacy geometry limitations | 655 locked QC entries lack resolution-safe fields | Correct |

The new core verification script independently recomputes the manifest/split fingerprints and all three source readout metrics, hashes every stage checkpoint, and inspects component parameter counts. It complements the existing evidence builder, whose cohort derivation begins with saved census tables and whose weighting calculations begin with the advanced audit's checked per-source details.

## Omissions considered but not required for this position paper

The manuscript does not need every implementation caveat in the main text. Its current appendix appropriately contains batch-weighted checkpoint selection, changing validation mask seeds, best-epoch weight restoration without optimizer rewind, incomplete runtime settings, absence of attention padding masks, and normalized-coordinate differences rather than physical velocities. The central comparison remains descriptive and exposes its aggregation mismatch. These limits are substantial for a performance paper, but the present bounded audit does not hide them.

The fixed four-frame averaging already limits fine temporal information. Adding that as a possible explanation is optional and should remain a hypothesis, since no matched architectural ablation exists. Likewise, low missingness-only performance cannot establish the causal mechanism of the learned representation. The manuscript correctly avoids that inference.

The appendix's optional group-loss weight is code-defined, not a reported experimental result, which is appropriate. Describing source grouping as the available unit rather than verified patient independence is also appropriate. No before/after numerical “leakage penalty” is claimed from incomparable historical runs.

## Verification tool added

[verify_core_artifacts.py](../reproducibility/verify_core_artifacts.py) is self-contained within the repository layout, can be invoked from any working directory, and writes nothing. It uses safe tensor-only checkpoint loading (`weights_only=True`) and resolves stage files by basename instead of machine-specific absolute paths. It passed on the current artifacts. Example from the experiment root:

```powershell
.venv/Scripts/python.exe neurips-genai4health/docs/reproducibility/verify_core_artifacts.py
```

The result proves consistency of the particular artifacts checked. It does not rerun training or certify model validity, historical test secrecy, patient independence, or clinical utility.
