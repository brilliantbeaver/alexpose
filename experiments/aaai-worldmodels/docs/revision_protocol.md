# Development-blind revision protocol

Protocol identity: `medwm-gait-revision-v1`. This document declares a revision before the reviewer inspects final test performance. It is a research plan and acceptance contract, not a manuscript draft or a statement of successful results. The coordinator must record its SHA256 and the final run configuration before releasing test metrics.

## 1. What was inspected, and what remains sealed

The literature reviewer read only these permitted fields from `pilot-comparison-v2/results.json`: `validation_selection`, `representation_diagnostics`, `support`, `steps_per_stage`, `n_windows`, `seed`, `epochs_parameter`, `versions`, and `status`. The reviewer also read the first and last training-history entries in `adaptation_stages.json`. No test classifier metrics, test anomaly metrics, source predictions, or paired test comparisons were inspected for this revision decision.

The v2 code has already computed test results. They are therefore **computed but evaluation-blind to this reviewer**, rather than a newly sampled untouched population. Anyone who has already seen them must disclose that exposure. Do not regenerate a more favorable split or silently treat this as a new independent test dataset. If any author used those results to select revisions, label the revised study exploratory and reserve independent data for confirmation.

Permitted development evidence is unfavorable but informative:

- All ten RF/probe validation settings had binary balanced accuracy 0.5 under the five-class-derived abnormal probability and fixed 0.5 threshold.
- The normal model had mean source-level feature standard deviation 0.125 for the online encoder and 0.058 for the teacher. Their effective ranks were 2.56 and 3.48 across 19 normal training sources. These are low-dimensional representations, not proof of total collapse or proof of useful clinical structure.
- The pilot retained 494 windows. Normal training had 19 sources; normal validation had six. Test support included only one cerebral-palsy source. These counts limit inference regardless of how many windows are generated.
- Each initial stage used 95 updates. The pilot's 16-frame fixed-stride observations and weak coordinate loss were designed as implementation checks, not evidence of useful long-horizon prediction.

## 2. What could be new enough to matter

Generic normal-gait prediction and uncertainty-based skeleton anomaly detection already have substantial precedent. The literature does not support a claim that combining these ideas is itself a first.

| Existing method | Verified overlap | Consequence for this project |
| --- | --- | --- |
| GaitForeMer, MICCAI 2022 | Transformer motion-forecasting pretraining followed by gait-impairment prediction. [Primary paper](https://arxiv.org/abs/2207.00106) | Forecast pretraining for clinical gait is established. |
| FSGait, ACCV 2024 | Normal-only gait reconstruction and temporal prediction, with sequence/frame/joint outputs. [Primary proceedings](https://openaccess.thecvf.com/content/ACCV2024/html/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.html) | Normal-reference predictive gait localization is established. |
| MoCoDAD, ICCV 2023 | Multimodal future skeleton generation for anomaly detection. [Primary paper](https://arxiv.org/abs/2307.07205) | Diverse skeletal futures for anomaly scoring are established. |
| SeeKer, ICCV 2025 | Conditional keypoint densities and detector-confidence-aware anomaly scores. [Primary paper](https://arxiv.org/abs/2506.18368) | Uncertainty and pose-confidence handling are established. |
| GenGait, 2026 preprint v2 | Normative masked-transformer reconstruction and joint inconsistency detection. [Primary preprint](https://arxiv.org/abs/2604.01997v2) | Masked normal-gait modeling and localized deviations are established. |

These are related-work checks, not reproduced baselines. The CVF and PMC endpoints sometimes returned access challenges; indexed primary-paper method text and accessible arXiv versions were used. No benchmark superiority over these methods can be claimed without implementing a matched comparison.

The strongest feasible contribution is an **audited medical-gait evaluation of what predictive surprise measures**: whether a small normal-motion model improves on persistence, whether it uses temporal context, and whether stochastic predictive uncertainty adds information beyond pose quality. A transparent negative result would still inform medical-world-model evaluation. A classifier improvement alone would not establish a world-model contribution.

## 3. Freeze two distinct revisions

### Revision A: correct the supervised binary endpoint

Train a directly balanced binary Random Forest and binary logistic probes with labels `normal` and `selected abnormal`. Keep five-class classification as a separate secondary task. Equal weighting of five classes gives normal only one fifth of the total class weight, so summing four abnormal probabilities and thresholding at 0.5 is not equivalent to training a balanced binary decision rule. This is a plausible explanation of the flat validation results; it must be tested, not assumed.

Use the identical v2 observations, representations, source assignments, and saved checkpoints for this endpoint correction. RF candidates remain depths 4 and 8. The executed development probe grid uses C values 0.01, 0.1, 1.0, and 10.0. Imputers and scalers fit on training observations only. Give each binary class equal total weight and each source equal total weight within that class. Select by validation source-level binary balanced accuracy at probability 0.5, then validation macro-F1; exact ties select the first declared candidate. Do not tune a threshold on final test labels.

The initial literature proposal had a smaller probe grid and no macro-F1 tie-break. The coordinator executed the four-value grid and tie-break above before final test inspection and sealed `binary-development-v1/selection.json`. This is a transparent development amendment, not a claim that the smaller initial proposal was followed. The expanded probe budget must be disclosed when comparing it with the two-depth RF search. Development RF balanced accuracy was 0.631 and the mixed-encoder probe was 0.619; these are validation findings only and provide no test-generalization evidence.

Before final test release, the mixed-adaptation JEPA probe is designated the primary JEPA comparator because it has the highest observed development binary balanced accuracy among the four JEPA representations. RF is its primary comparator. Other JEPA readouts remain secondary, with no best-test-model selection.

Primary supervised endpoint: source-level binary balanced accuracy. Secondary: AUROC, average precision with prevalence, sensitivity, specificity, confusion matrix, and five-class metrics clearly separated. Compare direct binary versus five-class-derived predictions on the same source rows. This is a methodological correction, not a new medical-world-model architecture.

### Revision B: test normal-motion prediction rather than only classification

**H1, primary:** a normal-only residual predictor improves future relative-articulation error over persistence on held-out normal sources, and its predictions benefit from ordered past context.

**H2, secondary:** a matched conditional latent-variable predictor improves predictive distribution quality or selective prediction over the deterministic residual model, beyond simple observed-prefix pose-quality rankings, without a material loss of mean-prediction accuracy.

The second hypothesis is secondary because the available source count is small. Failure must remain visible; diverse samples alone do not support it. Stochastic positional jitter is deferred from this minimal experiment. Introducing both stochastic mechanisms simultaneously would obscure attribution and increase the experiment budget.

## 4. New observation contract

Use the original source partition and selected five-label scope. New data preparation has its own identity and cannot overwrite v2 artifacts. Process 32 selected frames per window, with 16 observed context frames and 16 future frames, retaining the existing native source-frame stride of two. This is a **fixed-frame forecast**, not a fixed 15 Hz or fixed-duration forecast. Record actual sampled FPS and horizon seconds for every window. Report their distributions by split and label and include sample rate in acquisition controls.

A fixed 15 Hz time grid would make horizon interpretation stronger, but it is not established by the accepted native-stride implementation. Do not relabel the new data as 15 Hz. If time resampling is later added, create a separate declared method identity and rerun all comparators on its identical cohort.

Keep the sixteen approved joints, image-plane units, prefix-derived torso scale, and explicit missingness. Prediction remains body-relative articulation. The normalization caveat for uniformly masked completion remains unchanged. Future-only predictions must not use future positions, future validity, future confidence, or future normalization statistics as inputs.

Keep full-image pose extraction. Annotation-guided cropping would alter estimator scale/context/confidence and introduce another treatment. Instead, add a predeclared annotation-box pelvis-containment check to reject obvious person mismatches. Both hips must pass the declared confidence threshold before forming the pelvis center. Apply annotation-to-decoded coordinate scaling only when both image dimensions are finite/positive and the relative aspect-ratio discrepancy is at most 5%. For incompatible geometry, record `unresolved_geometry` and preserve the manual-review gate; do not invent a coordinate correspondence. A pelvis outside the valid scaled annotation box invalidates that frame. The containment rule is not a guarantee of correct identity when people overlap.

Record per-frame containment status, prefix/future selected-joint confidence, valid fractions, nominal/decoded times, and before/after source attrition. Exclude confirmed wrong-person cases by documented inspection rules before outcome access. All compared methods use the resulting common cohort. Do not replace a failed test source or relax a quality threshold to recover a desired metric.

## 5. Matched prediction arms and compute

Use three seeds, **27, 41, and 73**, with equal source sampling and identical normal training exposure. Pretraining is uniform masked prediction on normal training sources only. For each seed, create one shared pretrained state, then initialize these forecast arms from it:

| Arm | Coordinate prediction | Additional stochastic mechanism |
| --- | --- | --- |
| `direct_det` | Direct future-coordinate readout. | None. |
| `residual_det` | Last observed per-joint articulation plus a learned correction. | None. |
| `residual_slv` | Same residual form. | Eight-dimensional conditional latent with training posterior and context-only inference prior. |

Initialize coordinate correction heads at zero consistently across arms. The initial residual prediction must equal persistence numerically. Copy every shared encoder/teacher/predictor parameter exactly. Independent initialization or a different pretraining history would confound the stochasticity comparison.

Coordinator-approved fixed laptop budget, amended from the initial 160/240 suggestion before revision validation: width 48, two encoder layers, one predictor layer, batch size eight distinct sources, AdamW learning rate 0.0003, EMA 0.99; 200 uniform-pretraining updates followed by 300 forecast updates per arm. Pretraining uses latent CE and MSE with variance/covariance regularization and coordinate weight zero. Forecasting uses common CE weight 1, latent MSE weight 1, variance weight 1, covariance weight 0.04, and coordinate weight 1. The stochastic arm adds KL weight 0.01. Uniform mask ratio is 0.5. These are declared new settings, not validated optimal hyperparameters or faithful published training schedules.

The coordinator/model author may amend this compute budget **before validation performance is inspected**, using only measured runtime/memory or numerical feasibility. Record the exact amendment, reason, timestamp, and new configuration hash. After validation inspection, further tuning requires a numbered development amendment and full reporting. Test outcomes cannot motivate an amendment.

Use dedicated mask randomness and identical source batches across matched arms. Store all seeds and each parent checkpoint identity. Keep final-step checkpoints, rather than choosing a different training step for each model from final test behavior. Report representation spread/rank, loss components, gradient failures, KL, and whether sampled predictions actually vary. Low rank is a diagnostic; do not optimize a rank threshold until it tells a favorable story.

## 6. Common targets, baselines, and controls

The predictor hides **all future tokens**. The primary coordinate loss/scoring support is future-observed joints that were observed at least once in the past, which is the common support available to persistence. Restricting loss support must not expose excluded future coordinates to the encoder. Score only observed outcomes, with no imputed target treated as measured evidence.

Persistence repeats each joint's last observed prefix value. Evaluate the learned and persistence predictions on exactly the same coordinates. An untrained model records the architecture-only reference. A static-prefix control replaces the visible past with its last observed articulation. A temporal-shuffle control permutes the first fifteen context frames while preserving the final context frame, joint identities, and validity correspondence.

For temporal controls, keep the persistence anchor from the original prefix. Otherwise a joint missing at the final prefix frame can acquire a different “last observed” value after shuffling. If the implementation cannot hold the anchor fixed, restrict that particular control comparison to joints observed at the final context frame and apply the same narrower mask to both original and controlled predictions. Record the chosen solution and support size.

Compute future error at cumulative horizons 4, 8, and 16 sampled frames. The sixteen-frame endpoint is primary; the shorter endpoints are descriptive. Report the actual seconds represented by each horizon. Average coordinates within a window, windows within sequence, sequences within source, then sources equally. Do not average all windows directly.

Primary H1 effect for each source is `persistence MSE minus residual_det MSE`; positive means improvement. Report the source-mean difference, paired source-bootstrap interval, all three seed results, and source support. Do not count seeds times sources as independent patients. Temporal-control effects are `controlled-input MSE minus original-input MSE` on matched support. Improvement over persistence without sensitivity to temporal ordering supports a useful short-horizon estimator, but weakens a claim of learning richer motion dynamics.

## 7. Uncertainty must compete with acquisition quality

Use **16 independent context-prior samples** per stochastic prediction, never the target-informed posterior. Use their mean for MSE. Fix the number of samples and randomization before evaluation; do not select the best sample against the true future.

The primary H2 distribution metric is an energy score on the common valid future-coordinate vector. Let distance be Euclidean distance divided by the square root of the number of observed coordinate dimensions. Compute the mean sample-to-outcome distance minus one half the mean sample-to-sample distance over distinct sample pairs. A deterministic forecast has zero second term. Aggregate window/sequence/source as above. This evaluates sampled distribution quality without pretending that a heuristic variance is a normalized likelihood.

The latent arm has extra prior/posterior parameters. Report parameter counts and runtime; this three-arm experiment does not isolate the effect of parameter count from the stochastic mechanism.

Report predictive dispersion versus future error using source-level Spearman association and selective risk at retained fractions 50%, 75%, and 100%. Compare deferral by predictive dispersion with deferral by each simple prefix-only nuisance ranking: missing-joint fraction and negative mean observed confidence. These are uncertainty controls, not disease classifiers. Future confidence may be used only for a clearly labeled retrospective confound audit, never as an online uncertainty input. If a descriptive partial-rank correlation is added, report the nuisance-design rank and return an unavailable result when fewer than three residual degrees of freedom remain. Six normal validation sources cannot support a heavily adjusted nuisance model.

Also fit a training-only, directly balanced binary quality-only classifier using declared acquisition features, including sample FPS and prefix quality. Any use of future quality must appear as a separate retrospective comparator. Use the same source partition and validation budget as the motion classifiers. Good quality-only classification indicates acquisition confounding; it must not be suppressed because it challenges a motion interpretation.

MSE noninferiority for the stochastic arm is a **descriptive development tolerance**, predeclared as no more than 10% worse than the deterministic residual arm on normal validation source-mean MSE. It is not a formal noninferiority trial with this small sample. Dispersion/error correlation or nominal interval coverage alone is insufficient to claim calibrated medical uncertainty.

## 8. Development criteria and one final release

Before any final test metrics are opened, write a development report containing every arm/seed's permitted training and validation outputs, cohort hashes, exclusions, and completed control checks.

| Question | Predeclared development criterion | If it fails |
| --- | --- | --- |
| Is execution valid? | Finite losses/predictions, disjoint sources, target isolation, shared support, complete provenance, and all declared seeds completed. | Fix software defects under a new run identity; preserve failed artifacts. |
| Does residual prediction improve persistence? | Mean normal-validation gain across seeds is positive, with positive gain in at least two of three seeds. | Report no development support for H1; do not claim useful dynamics merely from training loss. |
| Does time ordering matter? | Prespecified temporal-control error increases on average, with sign and support reported for every seed. | Limit interpretation to static/postural or local-smoothness information. |
| Does latent uncertainty help? | Lower normal-validation energy score and MSE within the 10% descriptive tolerance; report selective-risk comparison against prefix quality. | Retain H2 as a negative or inconclusive result; do not select favorable samples or discard seeds. |
| Is binary evaluation aligned? | Direct binary models and five-class-derived models use identical source rows with validation-only settings. | Correct the endpoint implementation before final release. |

These criteria decide which scientific statements are warranted, not which unfavorable arms are hidden. All frozen arms should receive final evaluation after the coordinator records the manifest, including arms that fail development criteria. Freeze preprocessing, source membership, model configurations, checkpoint steps, metric definitions, uncertainty sample count, threshold rules, and code hashes together. The binary seal must also pin window metadata, reused encoder/old-classifier bytes, protocol SHA256, and runtime dependency versions, not merely the array file and top-level runner source. The coordinator can then authorize the single final evaluation as part of the user's existing task authorization.

Final inference uncertainty uses paired source resampling with 2,000 bootstrap replicates and a fixed seed. Report per-training-seed estimates separately, and bootstrap the source-wise mean effect across the three seeds for the displayed paired interval. Its uncertainty is conditional on the retained sources and cannot establish independent-patient or external-site coverage. Preserve the uninformative 5% conformal threshold when normal calibration support remains too small.

## 9. Conditions for a paper draft

Do not draft empirical results until the real-data notebook executions, revised model runs, source accounting, controls, and final review have finished. Then match the paper's claims to the evidence. A controlled failure of predictive surprise, an acquisition-confound finding, or a binary-endpoint correction can support a useful methodological report, but should not be repackaged as a successful clinical classifier.

Clinical person alignment, unknown participant overlap, annotation validity, native-frame-rate variation, small condition-specific source counts, and lack of external patient validation remain limitations. The scope is observed gait dynamics under source shift. No treatment-response, disease-progression simulation, causal neurological explanation, or prospective clinical decision benefit is tested by this design.
