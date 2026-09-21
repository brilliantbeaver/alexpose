# Adversarial review and revision record

This record accompanies [the research synthesis](../../docs/17-research-review-and-directions.md). The review covers all seven top-level notebooks, the current `video-data-full` inventory, the retained full-data capstone results, and the proposed research directions. It distinguishes corrections to the writeup from experimental changes that require new data or training.

## Independent review

The requested `codex:adversarial-review` command ran through the installed Codex plugin in read-only mode. Its focus was the new synthesis, evidence script, and figures, with the existing notebooks and implementation treated as evidence. The review challenged dataset accounting, model-selection boundaries, numerical claims, uncertainty, geometry, symmetry, novelty, and the strength of the conclusions. The user's request to revise the document authorized the subsequent edits.

The [first review output is preserved verbatim](2026-09-20-research-adversarial-pass1.md). It returned **needs-attention**, with one medium-severity finding. It reproduced the empirical calculations, including the retained scores, source-resampling interval, embedding statistics, and diagnostic image, but identified an important limitation in the interpretation of masked-feature learning.

## Finding and response

**Finding: reflection can expose original target coordinates through visible opposite-side slots.** The augmentation swaps left/right joint data while the training mask remains indexed to the original slots. For an asymmetric mask, the transformed coordinates of an original target joint can therefore enter the context through the opposite joint. The reviewer requested an explicit distinction from train/test leakage, a reproducible audit, and a corrected comparison before attributing lower loss to inference of hidden movement.

**Writeup revision.** The introduction now flags this limitation. The masking section explains the information path with a left-ankle/right-ankle example, distinguishes it from crossing a source split, and limits what the training-loss decrease can establish. The methodological interpretation and research priorities require a no-reflection or consistently mapped-mask comparison, with new fold-specific training and provenance. Saved classification scores remain descriptions of the implemented procedure; no corrected-model performance is claimed.

**Reproducible verification.** The [evidence script](../../scripts/review_research_evidence.py) checks the actual augmentation using a synthetic input: an original left-ankle x coordinate of 0.25 appears as −0.25 in the reflected right slot. It also draws 512 masks with seed 42 from the current sampler. Conditional on reflection, 17,931 of 85,409 target slots (21.0%) expose their original coordinates through a visible opposite-side slot. Consistently permuting the mask closes this specific path for all those masks. These are diagnostic draws, not a replay of the retained training history, and the check does not measure the effect on learned features or classification. It also does not implement the full context/target correspondence change in production training.

The finding is addressed in the research document and reproduction evidence. The underlying training change and its controlled evaluation remain explicit work for the next experiment.

**Independent re-review.** A focused second invocation of `codex:adversarial-review` checked this revision against the augmentation and training code and independently reproduced the 512-mask audit. It returned **approve**, with no material findings. Its [output is preserved verbatim](2026-09-20-research-adversarial-pass2.md). This approval concerns the accuracy and completeness of the correction to the writeup; it does not certify a repaired training procedure or clinical validity.

## Other corrections made while checking the evidence

The synthesis incorporates the following qualifications from the systematic local audit:

- Dataset accounting distinguishes 91 raw clips, 88 usable clips, 41 source recordings, and 358 available windows. Sources are recording IDs, with participant identity and clinical labels unverified.
- The shape audit identifies the third channel as detector visibility, describes trimming, interpolation, padding, and omitted trailing frames, and limits geometric preservation to operations that actually preserve it.
- Nominal metadata imply approximately 11.88–15.00 sampled frames per second despite a stored rate of 15. Unlimited internal-gap interpolation and missing provenance prevent stronger timing claims.
- Five outer source folds each use one inner validation holdout. The encoder, scaler, probe, feature selection, and checkpoint choice have explicit fitting boundaries. Source-uniform encoder sampling is distinguished from class-weighted clip-level head fitting.
- Notebook 03 trains on all three collection labels' training sources. Notebook 04 performs restarted label-free continuation and a frozen probe; its legacy filename does not establish normal-only pretraining or a VICReg experiment.
- Pooled source-weighted F1, pooled clip-weighted F1, and the average of fold F1 scores are reported separately. Fold standard deviation is not presented as a confidence interval.
- The retrospective source bootstrap is labeled a descriptive sensitivity interval conditional on fixed predictions. It does not include retraining, selection, seed, or split uncertainty and does not prove superiority or equivalence.
- The RF extractor duplicates right-ankle range in the left-ankle field. Rebuilt features and refitted small RF models reproduce the existing predictions, but a corrected baseline needs fresh results.
- Mean pose, visibility, and learned features differ in input information, pooling, dimensionality, and selection opportunities. Their comparison does not isolate a causal temporal-learning effect.
- Synthetic symmetry illustrations are marked as synthetic; no per-condition asymmetry measurement is attributed to this dataset. Proposed bilateral methods and forecasting tasks are separated from completed experiments and established prior work.

## Verification and remaining boundaries

The reproduction script verifies the saved aggregate metrics from all 88 held-out prediction rows, reconstructs the five RF prediction sets, checks the fold 0 training-embedding membership, recomputes silhouettes, reads all 91 raw-video containers, and regenerates the review figures. Its [JSON record](2026-09-20-research-evidence.json) includes input, notebook, and reviewed-code hashes. Neural models and pose extraction were not rerun.

The existing split and correctness suites passed: **30 tests** with `.venv/bin/python -m pytest sjepa/tests/test_splits.py sjepa/tests/test_correctness.py -q`. These tests support the checked implementation properties; their success does not establish clinical validity or eliminate the newly documented training-objective issue.

All local links in the synthesis and this response record resolve, and the reviewed notebook and code hashes match the evidence record. The seven illustrations and a rendered version of the document were inspected for legibility. Scoped whitespace checks passed. The README links to the new synthesis; existing user edits and retained experimental outputs were preserved.

The substantive remaining work is experimental: correct and rerun the reflection and RF comparisons, retain timestamps and missingness in a new cache version, establish participant and label provenance, and evaluate declared controls and independent data. Those limits remain visible in the synthesis because editing a research document cannot resolve them.
