# Full-sequence observation contract

The restart uses every available annotation span in the five requested categories after the existing media and whole-source exclusions. It estimates two-dimensional articulation from downloaded videos through Ambient's `SequenceKeypointExtractor.extract_from_image`. The reference source directory remains `data/videos-gavd`. The adapter does not copy or download videos.

The implementation is [restart_data.py](../../worldmodels/restart_data.py). Run it from `experiments/aaai-worldmodels` with this experiment's Python 3.12 environment:

```bash
.venv/bin/python -m worldmodels.restart_data --output artifacts/runs/restart-full10hz-v1
```

Completed output directories are immutable. An interrupted extraction can resume only when the request, annotations, video bytes, pose model, Ambient extractor, supporting extractor modules, sampling adapter, normalization module and gating module match the recorded identities. Source caches are checked against those identities before reuse. This study creates fresh pose calls instead of inheriting poses from the former bounded-prefix studies.

## Frames and time

GAVD frame numbers are one-based. OpenCV indexes are zero-based. The adapter converts once when mapping a decoded index to its annotation frame number. Each contiguous annotated span gets a nominal 10 Hz grid anchored at its first native frame. Each grid point selects the nearest native frame, with ties rounded upward. The adapter does not interpolate skeletons. It rejects source rates below 10 Hz instead of repeating frames. Different annotation spans never share a window across an annotation gap.

Sampling uses the source's measured nominal FPS because annotation frames index that native sequence. This is not presentation-time resampling. Raw decoder presentation timestamps and nominal times are both saved. Every window records its actual duration, each future target's time after the last context observation, native frame intervals, and maximum deviation from a nominal uniform grid. Windows with missing or nonmonotonic presentation timestamps are rejected.

A window contains 24 observations. The first 16 form the context, and the last eight form the future. Adjacent windows advance by eight observations. Thus the nominal observed span is 2.3 seconds and the final prediction horizon is 0.8 seconds after the final context observation. Overlapping windows remain dependent observations of one source. The `sample_fps` column is the requested nominal 10 Hz rate; `fps` is the native decoder rate.

## Person selection and missingness

Ambient returns its first full-image pose estimate. The existing experiment gate tests whether the estimated pelvis lies inside the rescaled annotated person box. The restart invalidates all joints for unresolved boxes or aspect-ratio mismatches, in addition to missing pelvis and outside-box outcomes. This is more conservative than the older extended adapter. Box containment still cannot prove identity when people overlap, and a source may contain multiple participants. The original two whole-source identity exclusions remain in force.

Raw poses contain all 33 BlazePose joints in image pixels. Model inputs use only the declared 16 joints, indices 11–16 and 23–32. Confidence below 0.5 or nonfinite coordinates mark an observation as missing. Each frame is centered on its observed pelvis. The scale is the median valid torso length from the context prefix alone. Future coordinates cannot alter context scale or context coordinates. This target measures relative articulation, not world-space displacement, physical speed, or measured three-dimensional anatomy.

Each accepted window must have at least 70% valid joint observations separately in its prefix and its future. A future joint is valid for scoring only if that joint had at least one valid prefix observation. This common support lets persistence and learned models share the same target mask. Missing values are replaced with zero only after the Boolean validity mask is retained. No interpolation is counted as observed evidence.

## Output schema

| Artifact | Contract |
|---|---|
| `windows.npz` | `x`: float32 `[N,24,16,2]`; `valid`: Boolean `[N,24,16]`. The row order matches `windows.csv`. |
| `windows.csv` | One row per retained window, with `window_id`, `source_id`, `sequence_id`, `label`, frame and time arrays, dimensions, quality summaries and view annotations. |
| `source_poses/<source_id>.npz` | One raw pose per unique requested native source frame. Includes raw `xy`, confidence, frame numbers, decoded indexes, timestamps, nominal times and complete provenance. |
| `sequence_poses/<sequence_id>.npz` | Every sampled observation in the sequence, including raw and gated confidence, gate reasons, annotation-span lengths and the raw-source cache fingerprint. |
| `inventory.csv` and `inventory.json` | Fresh annotation inventory with media eligibility and the historical seed-27 partition mapping. |
| `selected_sequences.csv` | Candidate sequences after all existing exclusions. |
| `excluded_sequences.csv` | Media and whole-source exclusions, including their reasons. |
| `attrition.csv` | Sampled frames, candidate and retained windows, and rejection counts per sequence. |
| `person_gate_frames.csv` | Every sequence-frame gate decision. Overlapping annotations can refer to the same source frame. |
| `duplicate_windows.csv` | Rejected exact duplicate source/frame-number tuples, with the retained sequence ID. |
| `source_progress.json` | Per-source model calls, timings, video and pose-cache hashes, and resumption status. |
| `request.json` | Frozen data, implementation, pose-model, environment and sampling identities. |
| `preparation.json` | Final counts, artifact fingerprints and execution status. It is written only after completion. |

The `window_id` is a SHA-256 fingerprint of the source ID and complete native frame tuple. Exact duplicate windows are kept once, in deterministic sequence-ID order. Partly overlapping annotations remain source-dependent and are documented separately in the inventory audit. Averaging and uncertainty must operate at source level, with equal source contribution. The same source must remain in one fold across all its sequences, windows and labels.

The `split` and `original_split` fields preserve the historical partition for traceability. They do not certify a fresh holdout. Earlier test outcomes have been inspected, so any new cross-validation or redesigned analysis on these sources must be described as exploratory. A final independent generalization claim requires previously untouched participants and recording settings, with verified diagnoses and a defensible patient grouping key.
