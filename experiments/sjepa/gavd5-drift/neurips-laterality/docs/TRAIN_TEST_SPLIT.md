# GAVD training, validation, and testing split

This page is the canonical human-readable account of the registered laterality
v2.1 split. The machine-readable authority is
[`../artifacts/paper/protocol_6f7baefbda07/splits/source_splits.json`](../artifacts/paper/protocol_6f7baefbda07/splits/source_splits.json),
derived from the accepted cohort in
[`../artifacts/paper/protocol_6f7baefbda07/cohort/manifest.csv`](../artifacts/paper/protocol_6f7baefbda07/cohort/manifest.csv).

## Accepted GAVD cohort

Quality control retains **625 pose sequences from 93 unique source videos**.
Splitting operates on source-video IDs, not sequences. Every sequence, mirror,
and augmentation from a source inherits that source's assignment.

| GAVD dataset annotation | Source videos | Pose sequences |
|---|---:|---:|
| Cerebral palsy | 9 | 58 |
| Myopathic | 28 | 183 |
| Normal | 29 | 270 |
| Parkinson's | 9 | 39 |
| Stroke | 18 | 75 |
| **Total** | **93** | **625** |

These names are dataset annotations used to stratify source counts; they are not
diagnoses and do not enter the encoder objective as labels.

The accepted sequences carry extraction provenance from three archive
generations: 94 sequences from `gavd3_pose_v2_video_mode`, 1 from
`gavd4_pose_v2_video_mode`, and 530 from `gavd5_pose_v2_video_mode`. These are
provenance categories, not separate train/test strata. Some source-video IDs
occur in more than one generation, so generation-specific distinct-source
counts must not be added to obtain the 93-source cohort total.

This extraction-generation label is distinct from the archive metadata schema.
A cache carrying the `gavd5_pose_v3_split_provenance` schema was refreshed to
attach split provenance without recomputing its pose coordinates; its required
`cache_origin_version` records which of the three generations actually produced
the tensor. Absolute `source_csv` strings likewise record the extraction host
and may use Windows or POSIX separators. Validation compares only the terminal
condition directory and annotation filename, not the host-specific path prefix.

## Five outer train/test folds

The 93 sources are assigned to five stratified outer folds. For a given fold,
the encoder and final read-out may use only the outer-training sources; the
outer-test sources remain sealed until final evaluation.

| Outer fold | Training sources | Training sequences | Test sources | Test sequences |
|---:|---:|---:|---:|---:|
| 0 | 74 | 436 | 19 | 189 |
| 1 | 74 | 443 | 19 | 182 |
| 2 | 74 | 553 | 19 | 72 |
| 3 | 75 | 548 | 18 | 77 |
| 4 | 75 | 520 | 18 | 105 |

For every row, training plus test equals the full 93-source, 625-sequence
cohort. The test-source sets are mutually disjoint and collectively contain all
93 sources exactly once; likewise, every accepted sequence is tested exactly
once per cross-validation pass. That pass is repeated for each of five seeds
and two variants: each source belongs to the training partition in four outer
fold rotations (40 training jobs) and to the test partition in one rotation
(10 evaluation jobs). Consequently, counts must not be summed across folds or
jobs as if they represented distinct data.

Sequence totals vary sharply even when source counts differ by at most one
because GAVD source videos yield unequal numbers of accepted sequences. The
protocol intentionally balances and resamples source videos rather than
breaking a source across folds to balance sequence counts.

## Four inner read-out folds

Only the current outer-training set is divided into four inner folds:

- with 74 outer-training sources, an inner fit uses 55 or 56 sources and
  validates on 18 or 19;
- with 75 outer-training sources, an inner fit uses 56 or 57 sources and
  validates on 18 or 19;
- across the registered inner partitions, fitting uses 266–450 sequences and
  validation uses 80–197 sequences.

The inner folds select only the ridge penalty. They do not select or stop the
encoder. After selection, the read-out is refitted on all 74 or 75
outer-training sources and applied once to the corresponding 18 or 19
outer-test sources. Scaling, imputation, target scaling, and neutral-band
calibration are also fitted using outer-training data only.

## Leakage controls and claim boundary

The split implementation and artifact loaders enforce the following:

1. Outer train and test source IDs do not overlap.
2. Every cohort source and sequence is assigned to exactly one outer test fold.
3. No outer-test source can reach an optimizer batch.
4. Inner training and validation partitions contain only outer-training sources.
5. Saved evaluation rows must exactly match the declared outer-test sources and
   sequences.

This establishes held-out-**source-video** evaluation within GAVD. GAVD does
not provide persistent person identifiers, so it does not establish
held-out-person or external-dataset generalization.

## Where the split is used

- [`02_source_level_splits.ipynb`](../02_source_level_splits.ipynb) constructs
  and audits the source assignments.
- [`03_fold_local_training.ipynb`](../03_fold_local_training.ipynb) trains a
  new encoder using only each fold's outer-training sources.
- [`04_held_out_evaluation.ipynb`](../04_held_out_evaluation.ipynb) selects the
  ridge penalty inside the outer-training set, refits there, and evaluates the
  untouched outer-test sources.
- [`../laterality/training.py`](../laterality/training.py),
  [`../laterality/evaluation.py`](../laterality/evaluation.py), and
  [`../laterality/splitting.py`](../laterality/splitting.py) contain the
  executable enforcement checks.
