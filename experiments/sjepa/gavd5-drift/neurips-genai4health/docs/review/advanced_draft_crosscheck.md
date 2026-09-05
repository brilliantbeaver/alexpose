# Adversarial crosscheck of the GenAI4Health drafts

Reviewed on 2026-09-05: `genai4health_paper_draft.tex` and `genai4health_extended_abstract.tex`, with emphasis on normal-reference cosine, source weighting, forecasting, temporal probes, consolidation, and clinical interpretation. Source files were read without editing. This review supplements `advanced_notebook_audit.md` and its independently checked numerical JSON. PDF layout and external citations were outside this crosscheck.

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
