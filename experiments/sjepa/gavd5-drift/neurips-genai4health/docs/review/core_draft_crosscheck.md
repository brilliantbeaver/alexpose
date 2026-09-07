# Core evidence crosscheck for the GenAI4Health revision

Updated September 6, 2026. The final read-only pass reviewed both complete LaTeX drafts of *Before Health Agents Interpret Movement: Lessons from a Gait Representation Study*. This report distinguishes present-day numerical checks from the earlier checkpoint review. No manuscript, source notebook, or scientific artifact was edited by this reviewer.

## Overall assessment

The rewritten paper and companion abstract pass the final core evidence check. All numerical and method claims agree with the retained records and inspected implementation after the three corrections documented below. No remaining numerical, methodological-description, or clinical-claim blocker was found. This is a consistency judgment within the stated exploratory position-paper scope, not a claim of experimental completeness or likely acceptance.

The main recorded results remain consistent with the current notebooks: pose summaries classify 10 of 20 held-out uploads correctly, learned features classify 6, and missingness alone classifies 6. Macro-F1 is 0.44, 0.29, and 0.25, respectively. All 60 saved source predictions match the portable supplement. The raw-to-pose-eligible cohort progression and the 59 / 18 / 20 source roles were independently reconstructed from the annotation files, retained manifest, and QC ledger.

These findings support an exploratory evaluation case and a position about evidence needed by future health systems. They do not demonstrate a new clinical world model, agent performance, a new JEPA architecture, or a generally superior classifier. The companion abstract should retain that distinction without becoming a catalogue of implementation details.

## Checks completed for this revision

| Claim | Current source of verification | Assessment |
|---|---|---|
| Selected inventory: 666 clips / 103 sources | Fresh scan of annotation folders | Confirmed |
| Public metadata: 657 / 100 | Retained metadata manifest and saved notebook 01 | Confirmed |
| Recorded decoded cohort: 655 / 98 | Locked QC rows and saved notebook 01 | Confirmed as recorded eligibility, not fresh decoding |
| Pose eligible: 639 / 97 | Retained QC ledger and 0.50 coverage criterion | Confirmed |
| Train / validation / test clips: 377 / 131 / 131 | QC ledger grouped by role | Confirmed |
| Train / validation / test sources: 59 / 18 / 20 | QC ledger grouped by source | Confirmed; source sets do not overlap |
| Test class counts: 7, 2, 3, 6, 2 | Complete notebook 06 predictions | Confirmed for normal, Parkinson's, stroke, myopathic, cerebral palsy |
| Pose-summary accuracy / balanced accuracy / macro-F1 | Independent score reconstruction | 0.50 / 0.44 / 0.44 |
| Learned-feature accuracy / balanced accuracy / macro-F1 | Independent score reconstruction | 0.30 / 0.26 / 0.29 |
| Missingness accuracy / balanced accuracy / macro-F1 | Independent score reconstruction | 0.30 / 0.25 / 0.25 |
| All three readouts miss all three stroke-labelled sources | Complete notebook 06 predictions | Confirmed; small descriptive subgroup only |
| Smooth-L1 + 0.10 variance + 0.01 covariance | Notebook 04 executed source; recorded configuration | Confirmed method description |
| Four-frame means, width 64, two layers, pooled MLP predictor | Notebook 04 and preserved run metadata | Consistent; original tensor checkpoint unavailable today |
| Five specified outer folds, one recorded evaluated fold/seed | Notebook source and saved outputs | Correct distinction |
| Readout refit on 77 development sources | Notebook 06 selection code and role counts | Confirmed |
| Mean features in selection, mean probabilities at test | Notebook 06 code | Confirmed unresolved mismatch |
| 655 locked caches lack newer geometry fields | Current QC ledger | Confirmed; no retrospective correction implied |
| Same-vector weighting gives 0.89 versus 0.70 | Five retained source summary rows | Arithmetic confirmed; cached vectors themselves unavailable today |
| One source supplies 60 / 64 normal-validation clips | Retained weighting rows | Confirmed: about 94% clip weight versus 20% source weight |
| All 60 supplement prediction rows agree with notebook | Complete saved-output comparison | Confirmed |
| Original final/stage checkpoint bytes agree with metadata | September 5 historical review only | Not reverified September 6; model files are absent here |

## Scientific wording requirements

**State what is measured.** Use “dataset annotations” rather than diagnoses, “source videos” rather than patients, and “normalized pose summaries” rather than calibrated physical kinematics. Cross-checkpoint cosine describes embedding-coordinate similarity, not preserved clinical function. Classification and weighting are two separate analyses; one is not evidence that the other explains an error.

**Keep the baseline comparison descriptive.** “The pose-summary classifier had the higher score in this split” is supported. General JEPA inferiority, a statistically established gap, and causal shortcut use are not. The two classifiers with 6 correct sources need not make the same mistakes. The source-grouped design is valuable, while the small number of test sources and the selection/test aggregation mismatch limit the inference.

**Report the trained method, not the teaching architecture.** Notebook 00 now explicitly explains the difference. The actual recorded learner averages four frames per patch, retains masked placeholders, uses a pooled MLP predictor, and compares latent vectors with Smooth-L1. The full two-view VICReg objective is absent. The loss omits condition labels, but annotations determine the cumulative training order.

**Avoid enlarging the evidence through terminology.** No clinical trust, useful functional retention, causal forecasting, or health-agent outcome was evaluated. “Predictive” refers to masked latent prediction here. Clinical and agentic workflows can motivate the position, but should remain proposed applications.

**Distinguish arithmetic reproducibility from training reproduction.** The current small supplement reconstructs scores from predictions and means from source summaries. It does not require model files. Original checkpoint hashes and model metadata remain historical records; claiming to have freshly verified those absent files would be inaccurate.

## Precision and result presentation

Two decimal places are sufficient for the main macro-F1 and cosine results. Report integer numerators and denominators for accuracy where helpful: 10 / 20 and 6 / 20. Use 60 / 64 clips with “about 94%” when describing concentration. The underlying values remain at machine precision in the CSVs for numerical checking.

The approximate macro-F1 difference is 0.15. Avoid presenting 0.148 as if three decimals conveyed meaningful between-run precision when no repeated-run uncertainty is available. Exact chosen hyperparameters, layer counts, and the 0.10 / 0.01 loss weights are controlled settings rather than noisy observations and need not be rounded into different values.

A two-panel illustration of the weighting calculation is more useful to the position than a gallery of latent plots. Source counts and the three classifier results can be shown in compact tables. The training-loss trajectory does not establish clinical learning because successive stages change the data population and target model.

## Numerical supplement changes and limits

The previous standalone verifier failed solely because its recorded checksums described CRLF files and the current copies use LF. Both historical checksums were reproduced by converting the current files to CRLF in memory. The values were not altered.

The updated verifier hashes canonical LF bytes, retains the historical raw digests separately, and prints a narrow “numerical checks” success message. The refresh script compares all 60 predictions with notebook 06 and all five weighting rows with the retained evidence, without rewriting either CSV. Its optional full-check mode fails clearly when the original checkpoint and protocol files are absent.

This resolves the portability failure without pretending to restore missing research artifacts. It does not verify patient independence, a complete historical record of test access, clinical validity, or new experimental efficacy.

## Outstanding experimental issues

The revised manuscript should acknowledge the same substantive constraints: one fold and model seed, no random-encoder control, incomplete original runtime settings, old crop-geometry records, different preparation in the anchor/temporal diagnostics, and inconsistent selection/test aggregation. A new matched evaluation would be informative but would be a post hoc continuation after the current test results have been inspected. These issues cannot be repaired by stronger wording or by recalculating the same scores.

## Final adversarial manuscript pass

Both rewritten LaTeX drafts were read in full, including tables, equations, captions, the proposed evidence record, counterarguments, and appendices. This pass checked the following points against the code and retained numerical records:

- The central comparison holds the same normal-annotated validation clips fixed and compares their EMA-encoder features after normal-only training and the final selected training stage. It does not compare visits, patients, original versus mirrored clips, or an untrained versus trained encoder.
- Both averaging formulas use per-video means of per-clip cosines. The source means round to 0.90, 0.76, 0.70, 0.60, and 0.54. The 60 / 64 clip concentration gives about 94% versus 20% weight, reproducing the 0.89 and 0.70 averages. Neither scalar is presented as a percentage, clinical score, or functional-retention estimate.
- The classification table reports 10 / 20, 6 / 20, and 6 / 20 correctly classified videos with balanced accuracies 0.44, 0.26, and 0.25. Category denominators and the three stroke-label errors are correct. An additional correct prediction in either two-video category would change that category's recall by one half.
- The 639 / 97 cohort, 59 / 18 / 20 video roles, 77-video classifier refit, and 377 / 131 / 131 clip roles agree with the recorded QC cohort. The manuscript does not treat upload grouping as patient independence or confuse these counts with the laterality experiment.
- The prepared 64-frame inputs, four-frame coordinate means, 33-landmark context, two Transformer blocks, width 64, four attention heads, pooled MLP predictor, validity rules, selected target joints, and variance/covariance weights match the primary notebook-04 implementation. The teaching Transformer predictor and centered cross-entropy are not imported into the method description.
- Feature dimensions are consistent with their definitions: 64 for selected-joint similarity pooling, 256 for the learned classification summary, 144 for normalized pose summaries, and 97 for availability summaries. The two preparation paths are distinguished, and the paper does not call normalized frame differences physical velocities.
- The training schedule is annotation-informed, retains prior categories, and does not use a condition target in the primary encoder loss. Validation selects the model; logistic-regression selection and subsequent refitting are described separately. The absence of a random-encoder baseline and the unresolved aggregation mismatch remain visible.
- The proposed evidence record and assistant comparison are clearly future work. No experimental result is claimed for documentation improving clinical trust, assistant reasoning, forecasting, planning, patient outcomes, or safety. Known reporting guidance is credited rather than presented as a newly invented general framework.
- The reanalysis is described as reconstruction from retained predictions and source summaries. Neither draft claims that missing checkpoints or full embedding caches were loaded or revalidated during the current revision.

Three method-description corrections were requested and confirmed in the final source:

| Issue found during final reading | Confirmed correction |
|---|---|
| Clip-QC coverage could be read as requiring both finite coordinates and visibility, while the executed eligibility gate counts visibility alone | The appendix now states the visibility-only clip rule and then distinguishes the finite-coordinate requirement for valid encoder observations |
| “Averaging clip predictions” could imply hard-label voting | Main paper and companion explicitly say that testing averages the category probabilities predicted for the clips |
| The 0.60 mask fraction was presented as executed although it is an overridable, unrecorded default | The unsupported runtime number was removed; the minimum-eligible-count masking mechanism remains explained |

The cosine definition was also improved to describe directional alignment rather than undifferentiated coordinate agreement. Optional abstract identification of the 64 clips as a normal-validation subset was suggested, but its omission does not change a claim: the body defines that subset unambiguously and neither abstract identifies the 64 clips as the entire cohort.

The stated GaitForeMer counterexample is supported by its primary paper: its pretraining combines motion forecasting with activity prediction and its study reports improved gait-severity estimation. The rewritten paper keeps this separate from the present experiment. [GaitForeMer](https://pmc.ncbi.nlm.nih.gov/articles/PMC9635991/)

The descriptions of prior reporting practice agree with the primary sources: Model Cards address intended use and evaluation conditions; V3 separates verification, analytical validation, and clinical validation; DECIDE-AI addresses early clinical evaluation, safety, and human factors. [Model Cards](https://arxiv.org/abs/1810.03993), [V3 framework](https://www.nature.com/articles/s41746-020-0260-4), [DECIDE-AI](https://www.nature.com/articles/s41591-022-01772-9)

The repository's MIT License and its separate policy for externally hosted video were checked directly. The manuscripts acknowledge annotation reuse while leaving the authors' project-specific institutional and data-use status unresolved, without inventing approval, exemption, or consent. [GAVD repository](https://github.com/Rahmyyy/GAVD), [MIT License](https://github.com/Rahmyyy/GAVD/blob/main/LICENSE)

The portable numerical verifier and current-record checker pass after the final textual corrections. PDF layout and submission-rule compliance are separate checks performed by the coordinating agent. No observed number was changed to resolve a review criticism.
