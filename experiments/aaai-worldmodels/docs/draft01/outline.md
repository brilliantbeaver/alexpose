# Manuscript structure and evidence contract

Working title: **Testing Bilateral Predictive Information in a Video-Based Gait Observation Model**.

Target a short empirical workshop paper if the final evidence remains a compact internal benchmark. Expand to a full paper only if the complete experiment and its controls provide enough distinct material. This is a recommendation, not a submission decision or a prediction of acceptance.

| Section | Argument | Evidence needed before finalization |
| --- | --- | --- |
| Abstract | Medical world models need observation dynamics that survive simple controls. | Exact retained sources, primary paired effects, and uncertainty. |
| Introduction | Coordinated gait is a useful observable layer, but classification alone does not establish dynamics. | Clinical primary sources and a clear distinction from disease simulation. |
| Related work | Explain skeletal prediction, normative modeling, and bilateral coordination without first-method claims. | Verified references in `references.bib`. |
| Data and observation contract | Explain source video organization, Ambient extraction, fixed-time windows, missingness, and person gates. | Input hashes, attrition table, timing distribution, and known exclusions. |
| Method | Define persistence, own-leg, bilateral, shifted-context, static, and damped-velocity comparisons. | Exact code, fold-local transforms, parameter selection, and shared support. |
| Evaluation | Separate normal forecast skill, bilateral information, timing specificity, and secondary abnormality association. | Frozen metric definitions, fold manifest, source aggregation, and uncertainty caveats. |
| Results | Report every prespecified primary comparison and failed hypothesis. | Generated source-level tables and plots, with no hand-entered unverified numbers. |
| Discussion | State what the evidence says about a motor-observation layer and what it cannot say about neurology. | Failure analysis, acquisition controls, and historical-test exposure statement. |
| Limitations | Explain source versus patient grouping, label validity, 2D measurements, and missing external validation. | Data audit and independent review. |

Keep two main figures: a vector diagram of the common observation contract and controls, and a paired source-effect plot. A compact table should show raw annotation counts, media eligibility, pose retention, and independent source counts. Additional per-label and per-fold tables belong in the appendix or reproducibility report.

The initial manuscript deliberately contains no new empirical values. The final manuscript should be generated from accepted run outputs and should keep this initial version as the pre-results narrative. Citation keys match `references.bib`. The coordinator owns the final manuscript, compiled PDF, readiness report, and response to independent review.
