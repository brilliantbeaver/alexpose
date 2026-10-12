# Technical method details accompanying the readable manuscript

The top-level notebooks define this experiment; `notebook-audit.md` maps the details to cells and implementation. These equations restate the evaluated training procedure, rather than describing a new run.

For predicted feature vector u, reference vector z, and running center c, p = softmax(u / 0.10) and q = softmax((z - c) / 0.06). For each training window, average -sum_d q_d log(p_d) over its target token positions; average those per-window losses equally across the batch. The reference encoder does not receive loss gradients. Once per batch, update c to 0.9 c + 0.1 times the mean of all masked reference vectors. The center therefore weights target positions equally, whereas the final loss weights windows equally. The reference encoder is an exponential moving average of the context encoder, with cosine momentum from .996 toward 1 over each stage.

Training uses the exact architecture and settings shown in the manuscript's settings table. The stochastic masks, distinct feature readout mask, geometric transforms, source-uniform window sampling, and restarted continuation state are documented in `notebook-audit.md`. Stage 2 preserves network weights but starts a new optimizer, center, schedules and seeded sampling streams; it is not one uninterrupted 1,200-update call.

The exact source split is shuffled `StratifiedKFold` applied to one row for each sorted unique source. Five outer folds use seed 42. Within the outer development set, the first of four stratified splits with seed 43+k forms validation. The registry in `reproducibility.md` fixes membership. Scaling and label-classifier fitting use training clips only, and validation selects between the two trained stages before test feature extraction.

The recorded frame rate 15 is the requested extraction target, not a retained per-clip elapsed-time guarantee. Coordinates are converted from MediaPipe image fractions to pixels, cleaned and interpolated, then centered on the hip midpoint and divided by torso length for the encoder and mean-pose control. See `claim-ledger.md` for the distinct cached-coordinate branches and their limits.

The original reference list was fully checked for bibliographic identity in `reference-audit.md`; retained claim support is mapped in `venue-and-literature.md`. No new disease mechanism, patient identity, diagnostic label, or permission fact is supplied by these technical details.
