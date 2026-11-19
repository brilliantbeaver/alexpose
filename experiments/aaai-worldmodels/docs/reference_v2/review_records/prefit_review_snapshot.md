# Independent review of the reference-aware gait study

The reviewer did not author the new cohort, models, masks, evaluation, runner, transfer code, notebooks, manuscript, or slides. This document records an independent inspection of the actual artifacts. Earlier author summaries and cross-reviews supplied context, but the checks below used the code and inputs directly. Acceptance applies only to the exact identities in the associated acceptance records. A changed artifact reopens its affected checks.

## Prefit scientific and implementation review

The internal study is an exploratory reuse of GAVD after earlier outcomes were known. Its primary candidate is fixed as posture-plus-velocity ridge. The matched posture calibration and low-motion score are retained as essential competing explanations. Masking is a separate coordinate-forecasting experiment, with a fixed architecture and objective, common target support, three optimization seeds, and a shared normal-only duration-selection rule. This distinction prevents abnormal-label outcomes from choosing the primary detector.

The model input boundary accepts exactly sixteen prefix observations. Future coordinates and observation availability enter training loss or retrospective scoring through separate arguments. Hidden values are erased before causal filling and feature scaling. Protected shoulders, hips, and latest joint anchors preserve the coordinate reference under artificial masking. Motion-aware masks can reveal information through the mask pattern; the implementation acknowledges this and retains pattern-only and randomized controls. Their presence does not itself demonstrate that the adaptive-mask channel is harmless.

The declared error first averages observed coordinates within each leg, then weights the legs equally. It averages windows within recordings/sequences, sequences within sources, and sources equally. Current GAVD components are singleton source IDs. The runner refuses multi-source components rather than silently changing the hierarchy. External sources instead represent coded participants, with recordings as sequences, so their observed errors and prefix covariates are aggregated to participants before the nonlinear tail transformation. Each participant then receives the mean of fifteen fixed source-trained calibration-component scores, separately for the primary and posture families.

Within each outer training pool, the calibration sources C are excluded from the final forecast R fit, the nuisance model, all nuisance out-of-fit predictors, and their nested regularization choices. C and outer sources are scored by the same final R predictor and nuisance model. No result is selected from a favorable rotation or score direction. The out-of-fit nuisance-response predictors have smaller training sets than the final R predictors; this is an explicit modeling approximation, and small C groups yield coarse ranks. Neither cross-fitting nor averaging creates a conformal or clinical coverage guarantee.

The primary within-fold AUROC difference excludes cross-fold score pairs. Its paired bootstrap resamples source components within fold and binary-label strata. It preserves the comparison pair and does not treat windows, pair entries, or model seeds as independent subjects. The conditional intervals omit full retraining uncertainty and unknown participant overlap. Two alternate source partitions separately characterize forecasting sensitivity, without being interpreted as independent studies or as full retraining intervals for the anomaly endpoint.

## Checks independently executed before real outer fitting

The reviewer ran the seven focused v2 test modules with the experiment's Python 3.12 interpreter. The initial run returned **57 passed in 19.63 seconds**. This included the author's full synthetic staged runner replay, freeze/configuration refusal, artifact-corruption detection, weighted model arithmetic, common-support aggregation, calibration provenance, source isolation, and participant-transfer hierarchy. These tests are software evidence, not real model-performance evidence.

A separate reviewer-written script is retained at `artifacts/reference-v2-independent-review/prefit_checks.py`, with its result in `prefit_checks.json`. It performed the following checks without inspecting a new efficacy score.

- It loaded the 2,815 retained real windows from 91 sources, verified 29 normal source groups, checked source-fold isolation, and independently recomputed the common distal-target support rule.
- It recomputed prefix-motion rates with explicit time/joint loops across every retained window. The maximum discrepancy from the evaluation API was approximately `1.78e-15`.
- It checked all fifteen real R/C/outer schedules and every intermediate nuisance-predictor membership for calibration-source exclusion.
- It inspected 637 combinations of one real window per source and seven mask policies using actual decoder timestamps. Exact matched budgets, natural support, reference protection, and latest anchors all passed.
- It fitted the three predictor modes only on synthetic normal-group fixtures. Fixed-mask predictions were exactly invariant to NaNs in hidden observations and extreme values in naturally missing observations. All three modes refused a full future-containing window at their prefix boundary.
- It independently expanded multinomial bootstrap draws into explicit source resamples and calculated AUROC with scikit-learn. All 200 paired bootstrap draws produced the same primary interval as the matrix implementation, with zero endpoint discrepancy.
- It checked that unequal numbers of windows per external recording still give recordings equal participant weight.

The reviewer also independently verified all fourteen cohort-manifest artifacts, all 96 raw source-pose cache hashes, and all 96 actual source-video hashes. Those checks appear in `data_hash_check.json`. Matching hashes establish provenance consistency, not verified person identity or clinical diagnosis.

## Prefit issues and required reporting boundaries

The cross-review process corrected unequal weighting of inner validation fold means; current selection pools one held-out error per normal source. Real-run configuration guards now bind the declared grids and policies to the accepted freeze. The final cohort-identity guard compares both the cohort identity and membership hashes with the accepted freeze before loading observations or fitting. Its refusal test changes the membership file and asserts that the data loader is never reached. The actual default cohort has independently verified identity and membership hashes.

No causal-input, masked-loss, paired-bootstrap, or participant-before-tail mathematical defect was found in the inspected implementation. The implementation's scope must remain clear in the manuscript: it forecasts relative projected articulation, uses labels supplied by the datasets, and scores abnormalities after the observed suffix arrives. It does not establish neural mechanisms, patient-level diagnosis, treatment response, or prospective clinical usefulness.

The external extraction release and tiny timing pilot are pipeline evidence only. Final acceptance still requires the completed extraction attrition, actual frame/cache reconstruction, observation QA, frozen transfer outputs, core source results, manuscript, notebook execution, and rendered-slide review. A positive wording request does not change those evidential requirements.

## Separate fixed robustness and transfer dependencies

The reviewer also inspected the separately released robustness and transfer modules against the final protocol. The robustness code fixes window-identity-derived randomness before outcomes, preserves reference and latest-anchor support under corruption, retains common observed future targets, and computes the same three-seed mean prediction endpoint as the main mask study. The source-label probe remains inside normal training sources and explicitly discloses that the forecasting model already saw these sequences and that annotation spans can overlap. It is a representation diagnostic, not participant validation. The nuisance matching rule uses permitted normal-training covariate scales and publishes every chosen pair and distance. Absence of a caliper cannot guarantee adequate similarity.

The transfer code retains the fifteen fixed GAVD calibration components in each of the primary and posture families. It aggregates error and prefix covariates within recordings and coded participants before applying each component tail transformation. It performs no external fitting or calibration. The reviewer checked the exact implementation order and independently checked an unequal-recording-window-count example. External identity codes are dataset metadata, and video hashes cannot establish the absence of an unknown repeated GAVD participant.

Two focused robustness tests passed in 1.98 seconds. The final integration rerun and exact accepted hashes appear in `prefit_acceptance.json`. No new gait-efficacy outcomes were inspected in this prefit review.

## Final results and publication review

This section will be completed after the declared runs and generated research artifacts exist. No final empirical or manuscript acceptance is implied by the prefit checks above.
