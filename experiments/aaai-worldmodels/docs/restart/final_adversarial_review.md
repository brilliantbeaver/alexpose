# Independent review of the completed fresh experiment

The completed experiment supports a useful, bounded positive result: a fitted bilateral observation model predicts normal-source articulation better than persistence on this internal cohort. It does not establish that bilateral history is better than a compact current-posture transition model. The latter has lower error, and the incremental benefit over own-leg history remains uncertain. A strong manuscript should retain this distinction rather than treating a favorable anomaly score as proof of neurological dynamics.

The reviewer advised on design and implementation invariants before outcomes, including the handling of missing-target weights and the optional shrinkage sensitivity. The reviewer did not implement the experiment, produce its original result files or author the manuscript. This assessment independently replayed the saved real inputs and outputs after the coordinator authorized access to the completed fixed results. It is independent of those artifact authors, rather than a claim that the reviewer had no earlier role in design critique.

## Evidence accepted

The reviewed real-run result file has SHA256 `f14a6e533338c96da99b5085cb231b9de5cc47f7fe50a4c118225c42dc1fa1e8`. Its implementation is `226b276ae3d0759624aec9ae6acff59d0cf282b795bc309db814c4f49fe8259c`, and its frozen protocol is `e7c331238c3ae61237336c11b36526afb20952ae31da51d9374962e98eafd299`. These identities match the implementation accepted before fitting. The preparation record has SHA256 `cc828d9a2f5cee879ed954c26a4fa98a65c357b76c3ebd4aca0131be84adebdc`.

The full extraction made 46,783 recorded pose-model calls across 641 selected annotation sequences from 96 sources. It retained 2,818 windows from 456 sequences and 91 sources. A common distal-target support rule removed three windows before fold assignment, leaving 2,815 windows and the same 456 sequences and 91 sources. The normal forecasting endpoint has **29 source groups**, not 2,815 independent observations. All five outer folds completed, with only normal sources outside each fold contributing to its forecast fits.

Independent reconstruction reproduced every prepared window and mask exactly from the saved sequence poses and annotation gates. It checked all 96 source-cache hashes, raw-to-sequence equality, current annotation hashes, 38,562 gated frames belonging to retained sequences, native-frame and timestamp mappings, duplicate-window exclusion and the two historical whole-source exclusions. This verifies stored construction, not the anatomical accuracy or identity of every pose.

The completed model replay checked all 94 artifact-manifest hashes and all 80 saved checkpoints. Every checkpoint reproduced its saved float32 forecast coordinates exactly after the documented storage conversion. The largest float32 rounding difference from the full-precision prediction was approximately `1.19e-7`. All learned window errors were recalculated directly from absolute predicted and observed coordinates. Changing every hidden future coordinate to NaN and invalidating the future left every checkpoint prediction unchanged.

The reviewer independently reconstructed all outer-training feature scalers and verified every inner and outer source-membership ledger and every recorded hyperparameter argmin. Ten prespecified output coordinates, one per bilateral leg and fold, were separately refitted with scikit-learn's weighted Ridge solver. Their maximum prediction difference was approximately `6.2e-15`. Source aggregation, all paired source means and bootstrap intervals, and all five anomaly AUROCs were independently recomputed. The AUROC check used direct positive-negative pair comparisons rather than the study's ranking function.

Machine-readable evidence and replay scripts are retained under `artifacts/restart_review/`: `prepared_input_audit.json`, `real_replay.json`, `audit_prepared.py` and `replay_real.py`. Earlier independent software checks and the synthetic full-pipeline replay are documented in [implementation_review.md](implementation_review.md).

## Results and permitted claims

All errors below concern source-average, pelvis-relative 2D articulation over the complete eight-sample forecast. Gains are reference error minus bilateral error. The intervals resample source groups conditional on the already fitted cross-validation models.

| Comparison | Gain | 95% source-bootstrap interval | Interpretation |
| --- | ---: | --- | --- |
| Persistence versus bilateral | 0.040630 | [0.023993, 0.059111] | The fitted model improves on persistence in this exploratory cohort. |
| Own history versus bilateral | 0.001112 | [-0.002525, 0.005244] | The incremental benefit of opposite-leg history is uncertain. |
| Independently fitted shifted history versus bilateral | 0.005909 | [0.001407, 0.011336] | Correct alignment is useful relative to this declared matched control. |
| Static posture versus bilateral | -0.008996 | [-0.015894, -0.002068] | The separately fitted current-posture transition model is stronger. |

Persistence MSE is 0.121262 and bilateral MSE is 0.080632. Their difference is 33.5% of the persistence mean; bilateral error is lower on 23 of 29 normal sources. These are defensible positive results. The static model's MSE is 0.071636. Because it predicts a separate output at each future time, its name refers to its input context, not to a frozen output. Its success supports the usefulness of a compact current-state observation model while weakening a claim that richer bilateral temporal history is necessary.

The matched shifted control preserves the same selected values, observation indicators, current anchor and nominal capacity. Its window-specific permutation tests alignment. It also disrupts smooth temporal structure and is not a biological intervention. The positive matched-control contrast does not independently establish neural coupling. The complete predeclared dynamics hierarchy fails because of the static comparator, and this failure must remain visible in the abstract, results, discussion and slides.

The prospective shrinkage sensitivity does not improve the bilateral result. Its MSE is 0.083779, with beta 0.5 selected in five of ten leg/fold fits and beta 1 in the other five. It remains a transparent negative optimization sensitivity. There is no justification for replacing the original unshrunk primary model with a more favorable post hoc variant.

The secondary excess-error score, bilateral MSE minus persistence MSE, has AUROC 0.7931 with interval [0.6802, 0.8960] and average precision 0.8646. This is a favorable association with the selected dataset annotations. Raw bilateral error has AUROC 0.3042, persistence error 0.1935, bilateral-minus-own error 0.5495 and prefix missingness 0.4266. All directions remain as declared. Larger raw forecast error is therefore not a reliable general synonym for abnormal gait in this sample. No diagnosis accuracy, clinically calibrated threshold or treatment benefit follows from these rankings.

## Systemic critique and concrete improvements

| Priority | Limitation that remains | Concrete next step |
| --- | --- | --- |
| High | Source IDs are not verified patient identities. Related uploads, repeated participants or recording settings may cross groups. | Build an external cohort with participant and site identifiers. Keep all visits and clips from one participant together, and evaluate a frozen model on independent participants and sites. |
| High | Prior GAVD results informed redevelopment. New folds do not undo this exposure. | Keep the current analysis explicitly exploratory. Use a new prospectively declared external cohort for confirmation, with no outcome-driven changes to the current run. |
| High | Selection differs substantially by annotation. Normal sequences retained after duration and quality rules are 141/287 selected for extraction, while myopathic sequences are 168/183. | Report per-label attrition and its causes. In a future study, prospectively assess duration, view, activity and visibility matching. Do not remove inconvenient sources after seeing forecast errors. |
| High | Full-image detection and pelvis containment cannot guarantee subject identity throughout every expanded interval. | Have blinded reviewers inspect trajectory discontinuities and representative complete intervals using an outcome-independent identity rubric. Retain the current uncertainty and record any future data correction under a new identity. |
| High | No action, longitudinal disease state or intervention outcome is observed. | Frame this as an observation-transition component. To study medical counterfactuals, collect repeated clinical measurements with measured interventions, treatment timing and appropriate causal design. |
| Medium | All forecast fitting ultimately depends on 29 normal sources, with only 23 or 24 in each outer training fold. | Prefer parsimonious models and report parameter-selection variability. Add independent normal participants before escalating model capacity or claiming stable clinical uncertainty. |
| Medium | The source bootstrap conditions on fixed overlapping fits and uncertain source independence. | In a larger confirmatory cohort, evaluate retraining variability with a declared repeated grouped procedure and a truly independent final test. Keep the present conditional intervals labeled. |
| Medium | Pooled out-of-fold anomaly ranking assumes comparable score scales across separately fitted models. | State this assumption now. In a new protocol, assess training-only score calibration or foldwise ranking without using abnormal held-out outcomes to choose the method. |
| Medium | A weak prefix-missingness ranking does not eliminate camera, view, activity or pose-estimator shortcuts. | Add prospectively declared nuisance measurements and matched acquisition analyses. Keep future-quality analyses explicitly retrospective. |
| Medium | The new full-duration cohort has no matched neural or published modern comparator. | Do not claim superiority to JEPA, GenGait or the state of the art. A subsequent controlled comparison should use the same observations, source folds, target support and tuning budget. Historical JEPA metrics cannot fill this role because its cohort and task differ. |
| Medium | The within-repository ASTAD study already investigates own/opposite-leg prediction. | State the methodological overlap and identify this study's distinct time-grid, coverage, control and audit contributions. Avoid an unsupported first-method claim. |

GAVD is sufficient for a small exploratory source-level comparison and for testing whether simple controls overturn a tempting interpretation. It is insufficient by itself for patient-level, site-level or clinical deployment generalization. Processing the available annotated duration substantially improves coverage, but cannot repair absent participant identities or unverified diagnoses. The actual complete forecast interval ranges from 0.7674 to 0.8342 seconds, and maximum window-grid error is about 0.0412 seconds; the nominal-time qualification is necessary.

## Notebooks and delivery status

The reviewer inspected the current execution evidence for all six restart notebooks. All thirty nonempty code cells have execution counts, no saved error outputs occur, structural validation passes, and current notebook source, executed output and input hashes match the execution record. This is an independent inspection of saved execution evidence, not a second complete notebook execution. Notebook 11 includes a fresh real pose call, and notebook 13 includes a separately labeled didactic real fit. Those demonstrations are not additional held-out efficacy studies.

The notebook audit is recorded in `artifacts/restart_review/notebook_audit.json`. Historical tutorial executions remain separate from the fresh six-notebook sequence. The draft manuscript receives a separate artifact-specific review under `docs/draft01/reviews/independent_review.md`, including rendering and submission-readiness checks.

No unresolved software or arithmetic finding blocks reporting the fixed real experiment. The scientific limitations above remain material and are part of the result. Strong reporting preserves the useful positive forecasting gain, the stronger current-posture comparator, the uncertain bilateral increment, and every declared secondary outcome together.
