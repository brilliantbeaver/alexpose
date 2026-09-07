# Adversarial crosscheck of the GenAI4Health drafts

## Current review: 2026-09-06

Both TeX drafts were read against all source cells and saved outputs in notebooks 07–09 and 05a–05d. This review also recomputed the weighting comparison from the retained per-source numerical supplement and compared it with the earlier cache-level verification. Manuscript, notebook, figure, and upstream artifact files were not edited by this reviewer. PDF layout and external citations remain outside this review's scope.

### Decision and rationale

The same-records weighting comparison is defensible and is the clearest advanced-notebook result for a submission distinct from laterality. Its contribution is a concrete example of how an evaluation summary can conceal concentration in one source. The mathematical distinction between clip and source means is established statistical practice; this experiment should not claim to invent grouped evaluation or discover a new universal bias correction. Its workshop relevance is the practical consequence for interpreting a movement representation that might later supply a health assistant.

The lead writer's revised hierarchy is supported: use **0.89 versus 0.70** as the primary example, retain **60/64 clips (about 94%) versus 20% of weight** to explain it, and use the source-held-out classification comparison as supporting evidence. Remove the temporal-probe table and cumulative drift curve from submission-facing appendices. They are fully retained in the research review, but introduce weakly grounded physiological targets and additional causal qualifications without improving the central argument.

### Required evidence corrections before release

| Issue in the inspected draft/package | Evidence from this review | Required handling |
|---|---|---|
| Present-tense claim that local recalculation verifies all checkpoints and cached embeddings | The expected checkpoint, evaluation, and split directories are absent here; the full artifact script fails at its first required sidecar. | Describe preserved run records and independently reproducible summary arithmetic. Do not claim a fresh checkpoint-level verification today. |
| Numerical-supplement verifier fails its checksum gate | Both CSV values are unchanged; current LF versus recorded CRLF bytes exactly explain the mismatch. | Packaging owner must make provenance portable and rerun the actual verifier. |
| "Initial checkpoint" can imply random weights | Cosines compare the validation-selected, trained normal-only checkpoint with the final trained checkpoint. | Use "normal-only reference" and "final model" consistently. |
| Multiple notebook names, local digests, and code-gate details distract from the paper | These identify local implementation states rather than scientific contributions. | Keep them in review/reproducibility records; describe the method and remaining evidence requirements directly in the manuscript. |

The first two issues change what this review can certify, but do not alter the numerical comparison. From the retained source rows, equal-clip cosine is **0.8890614295**, equal-source cosine is **0.7010576725**, and the difference is **0.1880037570**. These agree with the previously checked float32 cache aggregation within `1e-7`. Three or more decimal places add no substantive interpretation for a five-source illustration.

### Adversarial interpretation checks

- The comparison changes only weights on the same per-clip similarities. It does not change the model, data, reference checkpoint, preprocessing, or scoring target. It identifies an aggregation effect, not an effect of removing leakage.
- Cosine summarizes similarity of model-coordinate vectors across checkpoints. It is not accuracy, a percentage of retained clinical function, disease severity, or a person's longitudinal decline. A rotation can change the coordinate comparison while preserving information, and uninformative constant vectors can remain stable.
- Equal source weighting and equal clip weighting answer different questions. Neither is established as equal patient weighting. The five observed uploads cannot supply a population truth against which a bias magnitude is estimated.
- The larger source-versus-clip difference is specific to the five-source validation example. In the earlier checked test output, source and clip means are 0.849632 and 0.853572, respectively. Do not imply a universal inflation factor.
- The records contain one completed fold and seed. A five-fold registry is a plan for assignment, not five completed model evaluations. There is no evaluated repair intervention or matched consolidation comparison.
- No valid future-prediction result is available. The forecasting section should state a proposed past-only evaluation requirement without presenting the notebook's blocked gate as an experimental failure. Latent discrepancy is not calibrated uncertainty, and observational forecasting would still not establish action-conditioned planning or treatment response.
- The method does not evaluate a health agent or clinician trust. Relevance to those workshop themes must be argued as an upstream evidence requirement and a future testable use, not as an implemented healthcare capability.

### Why temporal results remain outside the submission

The four-bin readout yields peak-position R² 0.318 versus 0.173 for mean/std pooling, but lacks an untrained encoder or direct-pose comparison. The target is the normalized array position of maximum ankle separation, without a visibility filter; the displacement-ratio target is not energy; the circular-shift lag ignores stored frame indices and is not a validated gait phase. All three lag R² values are negative. None tests an unseen future.

The pooling sanity check permutes tokens after they have been encoded with time positions and full-clip context. It correctly verifies a symmetry of the averaging operation, but does not show that mean/std features have lost all temporal information. Test features are prepared before ridge selection; the fitting code does not use test labels for that selection. Calling early materialization itself label leakage would be inaccurate. These distinctions are important for method development and are now recorded in `advanced_notebook_audit.md`, rather than imposed on a short-paper reader.

### Handoff

No new experiment changes the earlier scientific conclusions. The lead writer has accepted the simplified evidence hierarchy and owns manuscript revisions. The core-evidence reviewer owns the portable supplement/provenance repair. This is a review of the drafts inspected before those revisions; final layout, the regenerated package, and any new manuscript language require the lead agent's final verification. The review below is preserved as the earlier checkpoint-backed assessment, not a claim that those unavailable inputs were rechecked today.

## Earlier draft review: 2026-09-05

The original review of `genai4health_paper_draft.tex` and `genai4health_extended_abstract.tex` follows. It emphasized normal-reference cosine, source weighting, forecasting, temporal probes, consolidation, and clinical interpretation. Its references to then-current artifact directories and suggested wording describe the state inspected on that date; the current review above supersedes them where availability or presentation has changed.

## Verdict

The centerpiece is numerically correct and its interpretation is appropriately bounded. Both drafts clearly distinguish validation-normal weighting from test readout results, describe a single completed fold and seed, withhold clinical/agentic/forecasting claims, and avoid promoting the archived reflection or AnchorGuard results. No numerical change is needed to the main weighting result. Two method-description corrections are recommended before finalizing; two small wording refinements would improve precision.

## Required method corrections

1. **Identify the actual reference checkpoint.** The paper's retention paragraph currently says embeddings are measured “before and after the cumulative curriculum.” The reference is the validation-selected checkpoint after normal-only Stage 0, not an untrained encoder before the entire curriculum. Replace with “at the selected normal-only Stage-0 and final Stage-4 checkpoints.” The extended abstract's “initial normal-only and final checkpoints” is already substantially clearer. In the appendix's sole-candidate sentence, also prefer “normal-only Stage-0 reference” to “initial checkpoint.”

2. **Disclose the normal-anchor preprocessing path as well as the temporal path.** Appendix B correctly says the training/readout path has no short-gap interpolation, then names only the temporal diagnostic as using a different path. Notebook 08, which supplies the main weighting finding, also uses that alternate preprocessing: threshold 0.45, interpolation of internal gaps up to four frames, original validity retained, and then centering/scaling/resizing. Suggested addition: “The temporal and normal-anchor diagnostics use a separate path that interpolates short internal gaps up to four frames while retaining the original validity mask. The weighting comparison holds this diagnostic preprocessing and its cached vectors fixed.” This is a disclosure correction, not a confound in the same-vector comparison.

## Additional precision edits

- The temporal appendix names `energy_ratio` and `peak_phase` as pose-derived targets. The former is the log ratio of mean lower-joint frame displacement between the second and first halves of a clip; the latter is the normalized index of maximum ankle separation. They are not physical energy or validated gait-cycle phase. Call these “notebook target names” or add that they are heuristic clip statistics without established biomechanical validity.
- Figure 2's “non-identifying local source aliases” need not make a privacy assertion. “Local source aliases” suffices. The plot uses aggregate counts and names A–E, and the public manuscript need not display source identifiers.

All four suggestions were messaged to the lead writer or are recorded here; no manuscript edits were made by this reviewer.

## Verified weighting claims

| Claim in draft | Source value | Judgment |
|---|---:|---|
| Same 64 normal-validation clips | 64 sequences, 5 uploads | Correct |
| Equal clip cosine 0.889 | 0.889061450958252 | Correct rounding |
| Equal source cosine 0.701 | 0.7010577321052551 | Correct rounding |
| One upload contributes 60 clips | 60, with all four others contributing 1 each | Correct |
| Dominant clip weight 93.75% | 60/64 | Correct |
| Dominant source weight 20% | 1/5 | Correct |
| Dominant source cosine 0.905 | 0.9049939513206482 | Correct rounding |
| Test source cosine 0.850 on seven normal uploads | 0.8496323823928833, 56 sequences, 7 uploads | Correct |
| Other source cosines 0.755, 0.698, 0.603, 0.543 | 0.7553919554, 0.6981571913, 0.6033803225, 0.5433649421 | Correct rounding/order |
| Only averaging weights change in the main comparison | Same 64-dimensional cached Stage-0/final EMA sequence embeddings, same per-sequence cosines | Correct |

The equations compute the average of per-sequence cosines with either clip or source weights. They do not silently compute cosine of an average vector. No fitted feature standardization, whitening, Procrustes alignment, or reference change occurs between the two aggregations. The generated `evidence/validation_normal_source_weighting.csv` agrees with the independently audited JSON, including the source-alias order A–E. `reproducibility/build_evidence.py` computes the clip value as a clip-count-weighted average of source means and the source value as their unweighted average; that is mathematically the same comparison, up to floating-point rounding.

The drafts appropriately describe a change of estimand and an aggregation effect. They do not call it an identified leakage effect, a patient-level estimate, an accuracy percentage, a population bias estimate, or proof that source weighting is universally preferable. This matters because the test clip and source means are much closer (0.853572 and 0.849632); the validation example illustrates a possible strong effect rather than a universal inflation factor.

## Consolidation and retention claims

The observed final candidate set in `work/artifacts/real/checkpoints` consists of one final objective bundle, `sjepa_outer_fold_0_seed_42_jepa_vicreg`, and its five stage checkpoints. There is no other final objective bundle or current fold-local repair result. Notebook 08 defaults to that single objective as its candidate list, and the saved report supplies no score from an alternative objective. Within the paper's stated scope, “sole available candidate” and “no current fold-local consolidation comparison” are supported. For maximal specificity, “sole available final objective for fold 0/seed 42” is preferable to any unqualified assertion about the whole project.

The final test cosine compares the final representation against the trained normal-only reference on the same seven test uploads. It is not a comparison between repair and no repair, and the fact that it exceeds the five-source validation mean does not indicate improvement after an intervention. The drafts explicitly state these boundaries.

The rotation/collapse counterexamples are mathematically sound. A post-training orthogonal change of latent basis can preserve task information while changing cross-checkpoint coordinate cosine; a correspondingly adapted linear readout can restore identical predictions. A constant representation can have cosine 1 to itself without useful variation. Thus high cosine is neither sufficient for functional retention nor is low cosine necessary evidence of information loss. The paper does not improperly equate representation change with health change, forgetting, or clinical decline.

## Forecasting and world-model claims

The current notebook and fold-evaluation directory have no qualifying future-trained checkpoint or surprise report. The two explicit code objections in both drafts are correct:

- `future_mask` hides only the selected 12 joints in the suffix and leaves other future joint coordinates available to attention.
- `copy_last_cosine` obtains its earlier token from the target encoder applied to the full unmasked clip; noncausal attention makes that token future-dependent.

The main draft correctly distinguishes information used to encode a future scoring target from information entering a predictor's context. A full target encoder is not inherently a violation when used only to define the held-out target; copying its earlier contextualized token into a baseline is the specific issue. Whole-clip preprocessing adds another causal boundary, accurately mentioned in the main paper. Neither draft treats the missing experiment as a measured null or treats the historical surprise AUROCs as current evidence.

Calling the current model non-generative in the sense relevant to these drafts is appropriately qualified by describing masked feature prediction and the lack of an implemented clinical agent. The drafts do not infer action conditioning, counterfactual dynamics, intervention response, motion synthesis, or calibrated uncertainty from the JEPA name.

## Temporal and clinical claims

All nine appendix temporal values exactly match rounding of `temporal_readout_seed_42.csv`:

| Target | Mean/std | Signed moment | Time bins |
|---|---:|---:|---:|
| `peak_phase` | 0.173011 | 0.052344 | 0.317745 |
| `energy_ratio` | 0.105126 | 0.053575 | 0.176175 |
| `phase_lag` | −0.070535 | −0.053989 | −0.052171 |

All rows have 20 test sources. The reported negative phase-lag results and the statement that negative probe scores do not prove information absence are correct. Test features are encoded before ridge selection in the notebook; the fitting code uses validation scores for selection and does not select alpha from test labels. The paper accurately separates these statements, rather than calling early feature materialization proof of label leakage. The diagnostic remains supplementary, observational, and nonforecasting.

Clinical wording stays within the evidence. The drafts describe future ambient-health applications as proposals, state that diagnoses were not independently adjudicated, and avoid claiming patient-disjoint sampling, disease progression, medical benefit, clinical calibration, clinician trust, or safe autonomous care. The observed missingness score is treated as a reason to investigate alternative signal pathways, not a causal proof that the encoder uses a shortcut. “Missingness may partly reflect real movement difficulty” is explicitly a possibility, not a finding established by these data.

## Reproducibility and manuscript strength

The position is stronger because it centers a verified arithmetic contrast while clearly acknowledging the small model, one fold/seed, legacy pose geometry, validation/test readout aggregation mismatch, and incomplete runtime serialization. The sole-candidate and missing-forecast statements are supported by current directory contents, not inferred from old artifact names. The claim-to-evidence table is consistent with this audit.

The original audit's complete checkpoint and cached-cosine checks remain the numerical provenance. This review adds independent comparison against the new manuscript text, exported weighting CSV, figure-building calculations, current artifact inventory, and temporal CSV. No new training, predictions, source-data transformations, or manuscript mutations were performed. After the two method clarifications above, there is no remaining blocking mathematical or evidentiary issue in the reviewed scope.
