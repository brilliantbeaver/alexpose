# Frozen Core11 GAVD probe — current evidence boundary

**Artifact:** `work/artifacts/gavd_core11_frozen_probe/`
**Code:** `uv run gavd6 gavd evaluate-core11` and
`src/gavd6_sjepa/research_directions/reflection_equivariance/gavd_core11_probe_evaluation.py`

The saved probe evaluates frozen 256-dimensional representations on the 96-row
GAVD cohort. Its only available split is sequence-stratified and shares a mean
of 11 source videos between train and test; grouped generalization is blocked
because at least one class has fewer than two source videos. It is therefore a
within-corpus diagnostic, not an unseen-video result.

On the strict 90-frame, no-short-clip-padding cohort, the raw Core11 baseline
has mean macro-F1 0.423. The EMA paired shared/no-cross model has 0.234 and the
EMA reflection-equivariant model has 0.245. Randomly initialized controls reach
macro-F1 values from 0.334 to 0.536 across seeds. These observations do not
support a representation advantage for the frozen fixed-reflection variants.

The artifact tables, predictions, and adapter audit should be retained together;
their duplicate `strict90_*` summary names are preserved for provenance. A new
comparison requires a cohort with sufficient independent source videos before
it can make a grouped-generalization claim.
