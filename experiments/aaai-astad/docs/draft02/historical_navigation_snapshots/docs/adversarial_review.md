# Adversarial methodology and tutorial review

> Historical foundation record. This document describes the earlier five-condition experiment. The revised twelve-folder study is documented in [the current protocol](context_study_protocol.md), [real results](context_study_results.md), and [critical review](context_study_review.md). Its manuscript is in [draft01](draft01/README.md).

Reviewed November 14, 2026. This review inspected the tutorial and vector-graphic generators, README, source documentation, and current data, model, training, workflow and evaluation modules. The reviewer authored the literature review and paper plan, so this is an internal implementation review rather than independent external peer review. Implementation fixes were made by the responsible agents and then checked again. This document records the final inspected behavior and the research evidence still required.

## Concrete findings and their resolution

| Finding from the first review | Why it mattered | Resolution inspected |
| --- | --- | --- |
| CLI pilot downloads and extraction selected different recordings. | A fresh installation could download one source and attempt another. | Extraction rows now come from source IDs in the download report before the sequence limit is applied. |
| Audit and Notebook 01 preview omitted the configured window settings. | Changing the target frame rate could make audit and training use different data. | Both paths now pass frame count, stride, frame rate and frame-number base consistently. |
| Notebook 01 retained an empty preview selection after downloading its first media. | The first run could skip the newly available overlay and extraction. | The notebook refreshes its video selection after the download cell. |
| SPE consumed the random stream used by subsequent masks. | The advertised controlled mask schedule changed when positional noise was enabled. | Training and model loss now use a separate positional-noise generator. The mask-schedule regression test checks successive masks and generator states. |
| Graphics described an ordinary 95th percentile, validation of every curriculum stage, and repeated masks for forecasting. | Those descriptions differed from the implemented calibration, stage selection and suffix scoring. | Graphics now describe the finite-sample quantile, normal versus final checkpoint, and one future-suffix mask. |
| Forecasting had only a single-example heatmap. | The cohort-level anomaly call evaluated masked completion alone. | Notebook 05 and the CLI now score masked and forecasting models separately, save source tables and metrics, and calculate source-cluster ranking intervals. |
| Explanation and capstone outputs could be shown beside an unrelated opening clip. | Readers could inspect the wrong motion when interpreting a residual or prediction. | `show_window_context` identifies the actual sequence, source and timestamps and displays its local segment when available. Synthetic examples explicitly have no observed source video. |
| Prefix resampling could interpolate from a future raw frame when the source frame rate was low. | A causal model on the final tensor does not make earlier preprocessing causal. | The shared corpus now excludes sources below the target frame rate, records the reason and source FPS, and tests raw-future perturbations at an accepted source rate. |
| Several diagrams implied causation or a sequence between independent components. | VICReg terms and the three stochastic mechanisms are separate design components. | Those connecting arrows were removed. VICReg's title now says that it discourages collapse and redundancy, rather than guaranteeing informative features. |

The full-cohort extraction run and subsequent numerical research results are separate from these software corrections. This review does not convert an unfinished cohort run into empirical validation.

## Review of the revised missing-observation path

The initial complete-case rule accepted no windows from an eight-sequence extraction pilot. That feasibility failure motivated a documented preprocessing revision before real classifier performance was examined. The final rule requires at least 75% fully observed four-frame joint patches overall and at least 50% in each half. Missing coordinates remain NaN. Resampling joins adjacent observed source frames only and does not fill missing-frame gaps. Sources below the target frame rate are excluded rather than upsampled. Pelvis centering is per frame and torso scale comes only from the observed first half.

The model's validity handling was inspected at each point where a placeholder could become unintended evidence:

1. The uniform sampler still selects a fixed number of positions from the entire allowed joint-time grid. It receives no coordinates, velocities, labels or validity mask.
2. The tokenizer uses numerical zeros only inside invalid patches before projection. Those patches are excluded as attention keys in both view and target encoders. Invalid encoder outputs are cleared.
3. The predictor excludes invalid visible positions, while retaining every hidden positional query regardless of whether its future target is observed. This avoids passing future availability into prediction through a changing query structure.
4. Cross-entropy, teacher-center updates, pooled embeddings and residual scores use valid observations only. Invalid attributions remain NaN with zero scoring coverage.
5. A row with no observed context or no observed targets raises an explicit error. It does not silently receive a zero anomaly score.

The regression tests challenge hidden-coordinate access, missing-target losses, invalid-patch finite remnants, future missingness, EMA gradients and batch-invariant scoring. These tests support specific implementation properties. They do not establish that missingness is irrelevant to the learned representation. Attention can still depend on which observed joints remain.

The capstone's nuisance control uses valid-token fractions by joint and temporal patch, together with camera-view one-hot features fitted on training rows only. Unknown held-out views are ignored by the fitted encoder rather than changing its categories. This checks several likely shortcuts. It does not replace measured nuisance-control results, performance stratified by view and quality, or audits of participant identity and recording context. Those analyses remain required before a robustness claim. The forecast cohort also requires future observation coverage, so it is retrospective evaluation on evaluable futures rather than a live admission policy.

## Evidence boundaries that must remain visible

The recorded follow-up to the original eight-sequence pilot retained 12 windows from six sequences: seven normal windows, three stroke windows and two myopathic windows. Parkinson's and cerebral palsy contributed no accepted windows in that limited pilot. These historical pilot counts are recorded in `artifacts/pilot_validity_recheck.json`; they are not the final counts of any later full-cohort run.

Normal-only training and residual scoring on this limited real subset establish that the partial-observation model can execute on extracted poses. They do not establish five-class accuracy, adequate class coverage, useful anomaly ranking or superiority over Random Forest. A later full-cohort run must report its own fresh retained counts, exclusions, source split and held-out results. No synthetic score may be substituted for that evidence.

The implementation predicts latent features, including a separately trained future suffix. It does not decode a visible future skeleton trajectory. An additional coordinate readout and reconstruction evaluation are required for such a claim. The capstone processes annotated held-out gait sequences; arbitrary unannotated video still requires a reviewed person crop and time interval.

The learned stochastic-latent prior/posterior architecture is reviewed and deferred. The implemented optional mechanism is StoP-inspired positional noise. It is not a posterior, a calibrated uncertainty estimator or demonstrated diverse motion generation. Causal clinical explanations and clinical responses are not implemented.

## Scientific claims supported by inspection

| Property | Evidence in the implementation | Remaining boundary |
| --- | --- | --- |
| Current annotations define the selected corpus. | `build_manifest` scans the five current folders and hashes CSV bytes. | Annotation counts do not establish media availability or accepted poses. |
| Masking uses the exact 16-joint subset. | `JOINTS`, `JOINT_INDICES` and the numerical table agree; other model joint counts are rejected. | This is a declared anatomical design choice, not an optimal clinical subset. |
| The normal reference survives adaptation. | Progressive training copies the supplied anchor and protects checkpoint identity. | Labels still select normal membership, curriculum stages and balanced sampling. The prediction objective alone is label-free. |
| Known source groups remain separated. | Assignment precedes windows; training rejects a source spanning splits. | Distinct source IDs do not prove distinct participants. |
| Downstream fitting and selection use permitted data. | Training-only imputation and scaling precede probes; validation chooses RF depth, probe regularization and encoder stage. | Repeated user-directed experimentation can still overfit validation. |
| Methods share the evaluated cohort. | Exported prediction tables align sequence IDs, sources, labels and ordering. | The classical descriptors and learned patches use different summary mechanisms. |
| Statistical intervals preserve known clusters. | Classification and anomaly bootstrap draws carry complete sources; paired classifier differences reuse the same draw. | Intervals omit unknown cross-source participant dependence and training-seed uncertainty. |
| Estimands are distinguished. | Classification reports source-weighted point estimates and explicitly named sequence-weighted bootstrap differences. | They are different summaries and must not be presented as the same quantity. |
| Small calibration support is exposed. | The finite-sample order statistic returns infinity when its required rank exceeds normal-source count. | A 5% finite threshold needs at least 19 exchangeable normal calibration observations; source IDs do not establish exchangeability. |
| Adaptations are disclosed. | Documentation separates the 2024 skeleton method, uniform MAMP-style masking, StoP and the 2026 stochastic-latent method. | This is not an exact reproduction of an upstream benchmark. |

## Visual and language audit

All 14 current SVGs were rendered with `rsvg-convert` and visually inspected again after the corrections. No text overflow, clipped labels, excessive line crossings or text-arrow overlap was observed. The figures remain readable at half their native width. Independent components are now shown without misleading flow arrows. The HTML slide deck has 16 slides with local SVG references and no network CDN. A PDF export is included; the SVG review alone does not establish every browser's print-layout behavior.

The README was checked against the implementation after it was written. Its normal/progressive distinction, real-data limitations, root `.env` decision, `python-dotenv` aliases, source counts, joint subset, objective, calibration rule and deferred stochastic-latent scope are explicit. A search of current Markdown, Python, notebook, HTML and SVG text found no em dashes.

## Disposition

The concrete implementation and tutorial mismatches identified in this review have been addressed. The project is suitable for executing the documented internal research protocol. This is not a finding that a robust gait classifier or a publication-ready empirical result has already been established. Full retained-cohort evaluation, nuisance controls, adequate uncertainty and appropriately limited claims remain necessary before submitting a paper.

## Final integration follow-up

Full extraction and real model runs subsequently completed. The dated [data audit](data_audit.md) and [results](results.md) supersede the earlier pilot-only status. The host also checked a held-out video inference path against the capstone probabilities and found agreement within 1e-6. A late notebook integration check found that the forecasting tutorial could overwrite the CLI summary without its bootstrap fields. The notebook now writes a separate forecast-notebook report; the full CLI uncertainty record was restored and the affected notebook was re-executed. The normal-space baseline has a regression test proving that its covariance fit receives only normal training sources. These were host integration checks, not additional external peer review.
