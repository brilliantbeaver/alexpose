# Med-WM submission scope and evidence plan

The [First Workshop on Medical World Models at AAAI-27](https://med-wm.github.io/aaai27-mwm/) emphasizes patient dynamics, prediction, simulation, uncertainty, and clinical decision support. A gait classification comparison is more convincing for this venue when accompanied by a carefully delimited future-prediction question.

## Verified submission information

| Item | Workshop page at retrieval |
| --- | --- |
| Submission deadline | November 20, 2026, Anywhere on Earth, replacing November 15. |
| Full paper | 6–8 pages, excluding references and appendices. |
| Short, poster, or position paper | 2–4 pages, excluding references and appendices. |
| Format | AAAI two-column PDF, maximum 50 MB. |
| Supplement | Optional single PDF or ZIP, maximum 50 MB. |
| Review and archival status | Single-blind and non-archival. |
| Submission portal | [OpenReview MEDWM](https://openreview.net/group?id=AAAI.org/2027/Workshop/MEDWM). |

The website labels the extension as a November 16 update, although the workspace date is November 15. The difference may reflect time zones; the site does not establish its announcement time zone. Both displayed dates are preserved. OpenReview returned its loading shell, so the live invitation deadline was not independently confirmed. Recheck the portal before submission. The workshop explicitly permits a position paper supported by literature and a concrete evaluation plan without completed experiments. [Official call](https://med-wm.github.io/aaai27-mwm/)

## Proposed paper framing

This is our recommendation, not a claim made by the workshop or the source papers:

**Working question:** Can normal-motion predictive representations distinguish source-independent gait deviations, and when does stochastic prediction express useful uncertainty instead of acquisition noise?

Use “predictive gait representation” or “a component toward a medical world model” until future-only prediction has been demonstrated. Randomly masking joint-time tokens permits context from both sides of the missing observation. That establishes completion ability, not forecasting. A future-only task supplies past observations to the context encoder and reserves later observations solely for targets and retrospective evaluation.

GAVD provides observational video labels rather than treatment assignments, longitudinal patient trajectories, or verified clinical outcomes. This study therefore cannot support claims of treatment-response simulation, causal neurological inference, or deployment-ready medical decision support. A paper can still contribute a reproducible evaluation protocol, a source-aware dataset audit, and evidence about representation learning under realistic pose noise.

## A practical sequence near the deadline

1. Freeze the cohort, split file, method identity, and primary metrics before inspecting final test predictions. Preserve the exact source counts and pose failures.
2. Complete a small real-data pilot and estimate runtime. A synthetic smoke test verifies code paths but supplies no clinical performance evidence.
3. Prioritize the matched RF versus deterministic Skeleton-JEPA comparison and a simple future-prediction baseline. Reserve model capacity and compute budgets explicitly.
4. Add one stochasticity comparison only when the deterministic pipeline is stable. Report unfinished ablations as planned work.
5. Draft around the evidence that actually exists. If completed experiments are insufficient, a position paper can argue for source-independent, uncertainty-aware evaluation and provide the implemented protocol as supporting material.

## Evidence needed for each intended claim

| Intended statement | Minimum supporting artifact |
| --- | --- |
| The pipeline is runnable. | Fresh environment command, notebook execution record, dependency versions, and failures. |
| The dataset is reasonably sized. | Eligible source and sequence counts per condition after all exclusions, with attrition reasons. |
| JEPA improves classification. | Same held-out sources, same input availability, paired metrics and source-level uncertainty against RF. |
| Normal-reference scores detect deviations. | Normal-only training, declared threshold selection, held-out normal specificity and abnormal sensitivity. |
| The model predicts future movement. | Past-only inputs, declared horizon, leakage checks, and persistence/temporal baseline results. |
| Stochasticity improves uncertainty. | Explicit prior samples, error-dispersion analysis, calibration or selective-risk measurements. |
| Explanations are useful. | Localized residual examples, acquisition confound checks, and carefully bounded interpretation. |
| The result generalizes clinically. | External patient/site validation, which GAVD source grouping alone does not provide. |

The paper should include negative results and explain which observations would disprove the proposed mechanism. A successful submission cannot be guaranteed, and no current tutorial or synthetic result should be described as a completed clinical study.

## Rechecked on November 16, 2026

The official workshop page was opened again on November 16. It still lists November 20, 2026, Anywhere on Earth, and explicitly dates the extension announcement November 16. The submission lengths remain 6–8 pages for full papers and 2–4 pages for short, poster, or position papers, excluding references and appendices. Reviewing remains single-blind and the workshop non-archival. This new retrieval resolves the earlier workspace-date discrepancy in the historical note above. The live OpenReview invitation deadline still requires checking before actual submission. [Official workshop call](https://med-wm.github.io/aaai27-mwm/)

The completed empirical work now supports a narrower framing: a controlled case study that distinguishes forecasting skill, anomaly ranking, and uncertainty quality. Consult [the final scientific review](final_scientific_review.md) for the accepted empirical limits before drafting. A high classification or anomaly-ranking score must not be presented as evidence of useful patient dynamics or clinical readiness.
