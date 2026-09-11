# FMTS V2 adversarial review

| Perspective | Objection and severity | Evidence | Correction in V3 | Residual limitation |
|---|---|---|---|---|
| Temporal researcher | Moderate: source holdout might be read as temporal generalization. | Source folds are stratified video partitions; no chronological test. | State that chronology, participant independence and untouched development are untested. | New source/person/temporal splits require new data or experiments. |
| Evidence auditor | Major: file-hash checks can be mistaken for content deduplication. | Recorded 625-archive replay and source-membership audit; no complete reupload/person audit. | Separate integrity and split checks from contamination checks in Appendix A. | Acquisition versions and person IDs are unavailable. |
| Statistical reviewer | Moderate: conditional intervals omit selection and split uncertainty. | Bootstrap seed 812 resamples saved predictions, without retraining or selection correction. | Name the omitted sources of uncertainty and preserve exploratory status. | No familywise or development-selection adjustment. |
| Editor | Moderate: the registered hypothesis and post-inspection extension are awkwardly juxtaposed. | PROTOCOL.md and extension module docstrings. | Connect the null across studies while distinguishing registration from exploratory questions. | Shared development cohort limits confirmation. |

V2 moved all five paired contrasts into the main text. The table was typeset with ordinary 10-point text and short headings, rather than reduced fonts. V3 adds inferential boundaries without claiming new evidence.
