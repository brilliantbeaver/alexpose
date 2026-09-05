# Numerical supplement

This supplement reproduces the two headline calculations in *Before Gait Models Inform Care: Evidence Boundaries for Predictive Health AI*. It contains no video, pose trajectory, direct video identifier, author information, or local filesystem path.

Run `python verify.py` with Python 3.9 or later. Only the standard library is used, no network is accessed, and no files are written.

- `test_source_predictions.csv`: one true dataset annotation and saved predicted annotation per test source for each of three readouts; 20 consistent source aliases and 60 rows.
- `normal_validation_weighting.csv`: five source aliases, their clip counts, and mean per-clip cross-checkpoint cosine. These aliases are separate from the test aliases.
- `provenance.json`: input checksums, original manifest/split/checkpoint digests, recorded model configuration and loss, and expected scores.
- `verify.py`: independently computes accuracy, balanced accuracy, macro-F1, equal-source and equal-clip cosine, and the unrounded macro-F1 difference.

The five source means reproduce cosine to float32 numerical tolerance. Clip weighting and source weighting change only aggregation of the same stored embeddings. The model remains fixed, and no model is retrained here.

These are one-fold descriptive results. They do not supply patient identity, uncertainty across training runs, complete training configuration, clinical validation, or future-prediction results. Source aliases do not establish person anonymity. Dataset annotations are not diagnoses established by this project. Consult the manuscript for the validation/test aggregation mismatch, different diagnostic preprocessing, and legacy pose geometry.
