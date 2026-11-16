# Interpretation clarifications after the frozen fit

These notes clarify reporting. They do not change the frozen protocol, fitted models, score directions, cohorts, or primary endpoints.

The horizon-eight mean squared error averages all eight nominal future samples from 0.1 through 0.8 seconds after the final prefix observation. It is not the error only at the final sample. The prefix contains 16 observations and spans 1.5 seconds between its endpoints. The future suffix contains eight observations. Quality is checked separately in the prefix and suffix, which are unequal lengths.

Full extraction retains 2,818 windows. The common distal-target support gate removes three windows, leaving 2,815 windows from the same 456 sequences and 91 source IDs for every model. The complete cohort contains 29 normal source IDs. This is source support, not a verified independent patient count.

The `static` arm has a static input context. It is a learned, horizon-specific prediction conditioned on the current bilateral pose, and its output can change across future times. It should be described as the current-pose learned transition or static-context predictor. Persistence is the separate comparator that repeats the last observed articulation. The current-pose learned transition outperforms the bilateral-history predictor, which limits the claim that a long bilateral history is necessary.

The primary bilateral-history model improves on persistence, and its matched temporal-control difference is positive. Its improvement over own-leg history remains uncertain. Shrinkage does not improve the unshrunk bilateral result. These findings remain together in the manuscript and teaching materials; the preferred positive result does not replace the unsuccessful comparisons.

The prespecified excess-error score is bilateral forecast error minus persistence error. Its positive association with abnormal dataset annotations is secondary exploratory evidence. Pooling scores from different outer folds assumes that scores from separately fitted models are comparable. Its source-bootstrap interval conditions on these fixed fits and cannot establish clinical patient-population significance. It does not establish diagnosis, intervention response, or a neurological mechanism.
