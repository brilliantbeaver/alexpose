# Adversarial review of the fresh design

This review examines the proposed `medwm-dynamics-audit-v1` protocol before new outer-fold outcomes are available. The reviewer did not write the protocol or implementation. The review is conditional: software and execution acceptance require inspection of the implemented method and saved outputs.

The design is a constructive response to the old study. It expands observation coverage, uses an approximately common physical-time grid, fits a transparent low-capacity transition model, and tests the contribution of the opposite leg with an independently fitted control of the same capacity. These choices improve the chance of detecting a reproducible signal while making a null result interpretable. They cannot guarantee favorable results or workshop acceptance.

## Required interpretation

Own-leg prediction is not a novel idea within this repository. The sibling ASTAD study already investigates local and relational leg prediction. The new manuscript should state that overlap and identify the distinct contribution as a fuller-coverage, fixed-time observation-transition audit with explicit falsification criteria and provenance. It should not claim to have invented bilateral prediction or neural coupling analysis.

The held-out units are source videos from a previously examined dataset. New grouped folds prevent training on a held-out source within a fit, but they do not erase earlier methodological exposure. The correct description is exploratory source-grouped cross-validation. This is neither fresh patient-disjoint external validation nor a prospective neurological prediction study.

The transition target is relative 2D articulation over about 0.8 seconds. The model has no treatment action, longitudinal disease state, clinical outcome or physical calibration. It can contribute an observable dynamics component and a validation framework to medical world models. It cannot support counterfactual treatment simulation or localization of a neural lesion.

## High-value pre-outcome improvement

A persistence shrinkage option can reduce prediction variance without increasing model complexity substantially:

\[
\widehat y = a + \beta\widehat r,\qquad
\beta\in\{0,0.25,0.5,1\},
\]

where `a` is the latest observed own-joint coordinate and `r` is the ridge-predicted residual. Jointly choose ridge strength and shrinkage using only the inner normal-source validation folds. Apply the same grid to every learned arm, with a declared tie rule favoring smaller shrinkage. The outer-held-out observations must not select the grid, model, threshold or score direction.

The zero option provides an interpretable persistence fallback. If it dominates, the result is evidence against useful learned temporal gain in this setting. It must not be renamed a successful learned dynamics model merely because the model now avoids large errors. If feasible, report the fixed `beta=1` version as a secondary sensitivity, with no outcome-dependent decision about including it. This recommendation is a pre-outcome proposal. Its adoption, final grid and protocol hash must be recorded before fitting and scoring the outer folds.

## Implementation acceptance conditions

| Issue | Required check |
| --- | --- |
| Common time grid | Save chosen native frame numbers and decoded timestamps. Report actual interval and horizon distributions. Nearest-frame sampling is nominally uniform; it is not exactly uniform in physical time. Do not describe interpolated or copied values as new observed targets. |
| Prefix information boundary | Changing any future coordinate, future confidence or future gate outcome must leave every model input and predicted coordinate unchanged. Eligibility can depend on observed future targets, but that selection must be reported separately. |
| Missing observations | Forward fill only inside the visible prefix and retain observation indicators. No missing training target becomes an observed zero. Fit only with permitted normal-training sources. |
| Shared reference | Centering uses both hips. Own-leg therefore means own distal-leg history after a shared pelvic reference, as already acknowledged in the protocol. |
| Matched temporal control | Shift opposite-leg coordinates and validity together while preserving the most recent observation. Keep own-leg history, target, anchor and target mask fixed. Fit the shifted model from scratch with the same parameterization and tuning budget. |
| Shift interpretation | Circular wrapping introduces a boundary discontinuity and can alter the relationship between missingness and time. This is a timing-disruption test, not a physiological intervention or an isolated proof of neural coordination. The separately fitted static model is a necessary complementary control. |
| Training hierarchy | Weight sources equally, sequences equally within a source, and windows equally within a sequence. If the two legs become separate training rows, prevent that transformation from silently changing source weights or counting one window twice as independent evidence. |
| Transform fitting | Fit scalers and any imputation statistics within each inner training fold. Refit with the chosen parameters using the full outer training set before outer prediction. Validation and abnormal sources cannot influence these transforms. |
| Target support | Use exactly the same observed target coordinates and prefix-known-joint support for all methods. State how both legs and partially visible joints are combined within each window. |
| Model selection | Select by source-aggregated normal validation loss. Do not let long videos or plentiful windows dominate the selected ridge strength. Save every candidate validation score and the selected parameters. |
| Data reuse | Deduplicate raw extraction requests by source/frame while keeping sequence and person-gate lineage. Overlapping sequences remain dependent and belong to one source fold. Report overlap rather than treating extra windows as independent sample growth. |
| Acquisition sensitivity | Use past-only quality information for any prospective abstention score. Future quality can define an explicitly retrospective sensitivity, but cannot choose a deployed threshold or conceal poor results. |

## Primary claims and constructive success criteria

The primary contrasts should be identifiable directly in the saved protocol and final result schema. Bilateral forecast error versus persistence tests useful prediction. Bilateral versus own-leg history tests incremental information but also changes capacity. Bilateral versus independently fitted shifted opposite history supplies the key matched-capacity contrast. Bilateral versus a separately fitted static model tests whether temporal history contributes beyond posture.

A dynamics claim requires the prespecified hierarchy to succeed. A favorable classification score cannot rescue a failed temporal test. A small positive point estimate whose interval spans zero is an uncertain estimate, not a reliable improvement. Confidence intervals condition on the fixed cross-validation fits and source grouping; they do not capture all uncertainty from training-set selection, repeated participants or a new clinical population.

Nested selection is appropriate for the modest candidate set. The inner folds contain relatively few normal sources, so selected parameters may be unstable. Report selection frequencies and per-source effects. Repeated model seeds or overlapping folds must not be counted as independent clinical replications. Avoid expanding the model menu after inspecting outer errors. A subsequent methodological change should receive a new identity and be reported as further exploratory development.

## Secondary ranking cautions

Raw forecast error, excess error above persistence and bilateral-minus-own error answer different questions. Keep their fixed larger-is-abnormal directions, including when rankings are below chance. Scores obtained from different outer-fold models are pooled out-of-fold observations. Their comparability is an assumption because fitted intercepts, regularization and residual scales differ. Report foldwise results when support permits and consider a declared training-only scale calibration sensitivity. Do not use abnormal outer outcomes to choose calibration, signs or thresholds.

The weakest route to a favorable paper would be to select the best secondary AUROC and omit the failed primary mechanism checks. The strongest route is an explicit result, positive or negative, explaining which evidence supports an observation-transition model and which plausible alternatives remain. The user’s preference for useful positive findings is compatible with careful design, stronger training-only regularization and better observation coverage. It does not change the evidential threshold after outcomes are known.

## Review status

The protocol is scientifically reasonable for an exploratory workshop study subject to the conditions above. No implementation or fresh-run acceptance is implied. The historical implementation review is in [initial_adversarial_review.md](initial_adversarial_review.md). The next review must inspect the actual code, fold membership, target support, input and output hashes, saved candidate-selection records, source-level results, manuscript claims and rendered artifacts.
