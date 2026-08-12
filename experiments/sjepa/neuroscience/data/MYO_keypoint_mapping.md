# Myopathy Feature to BLAZEPOSE_33 Keypoint Mapping

This file maps the Myopathy features extracted from the provided `## Myopathy` CSV to BLAZEPOSE_33 keypoints used by the GAVD JEPA pipeline.

## Analysis Basis

- The `Priority` and `Feature` columns were extracted to `penny/neuroscience/data/MYO_features.csv`.
- Source-of-truth feature math is in `ambient/pose/joint_angles.py`, `ambient/analysis/feature_extractor.py`, `ambient/analysis/symmetry_analyzer.py`, and `ambient/classification/features.py`.
- BLAZEPOSE_33 lower-body and trunk landmarks used here:
  - `11 LEFT_SHOULDER`, `12 RIGHT_SHOULDER`
  - `23 LEFT_HIP`, `24 RIGHT_HIP`
  - `25 LEFT_KNEE`, `26 RIGHT_KNEE`
  - `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`
- Implementation caveat: `trunk_lean_angle` is a proxy in `GaitFeatureVector`; the current fallback derives it from `shoulder_symmetry_index`, which traces to `11 LEFT_SHOULDER` and `12 RIGHT_SHOULDER`. A true trunk-lean implementation would normally use shoulder-to-hip alignment, but that exact computation is not defined in the current repo.

## Task #2 Table

This table includes every Myopathy feature from the CSV, including excluded and medium-priority rows.

| Feature | BLAZEPOSE_33 region | Keypoints include |
|---|---|---|
| `left_hip_mean` | Left sagittal hip angle chain | `11 LEFT_SHOULDER`, `23 LEFT_HIP`, `25 LEFT_KNEE`. |
| `right_hip_mean` | Right sagittal hip angle chain | `12 RIGHT_SHOULDER`, `24 RIGHT_HIP`, `26 RIGHT_KNEE`. |
| `left_knee_mean` | Left sagittal knee angle chain | `23 LEFT_HIP`, `25 LEFT_KNEE`, `27 LEFT_ANKLE`. |
| `right_knee_mean` | Right sagittal knee angle chain | `24 RIGHT_HIP`, `26 RIGHT_KNEE`, `28 RIGHT_ANKLE`. |
| `walking_speed_ms` | Excluded - whole-body motion | Current code converts `walking_speed_pixels_per_sec` or fallback `velocity_mean`; these are computed from center-of-mass or all-keypoint motion, not a single anatomical region. |
| `step_length_cv` | Bilateral ankle / step-width proxy | `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. In `GaitFeatureVector`, this field is currently populated from `step_width_std`, so it behaves as ankle separation variability, not true step length CV. |
| `stride_length_m` | Excluded - mixed COM/ankle fallback | Medium-priority feature. Primary computation uses center-of-mass motion and cadence; fallback uses `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. Excluded because it is not uniquely region-specific. |
| `trunk_lean_angle` | Bilateral shoulder/trunk proxy | Current repo fallback uses `shoulder_symmetry_index`, which traces to `11 LEFT_SHOULDER`, `12 RIGHT_SHOULDER`. |
| `positional_symmetry_score` | Excluded - multi-region symmetry aggregate | Aggregates positional symmetry across multiple unrelated pairs: shoulders `11/12`, elbows `13/14`, wrists `15/16`, hips `23/24`, knees `25/26`, ankles `27/28`, heels `29/30`, and foot indices `31/32`. |
| `postural_sway_area` | Excluded - center of mass / whole body | Convex hull or bounding-box area of the center-of-mass proxy from all available BLAZEPOSE_33 landmarks. |

## Task #1 Grouped View

Only high-priority (`H`) features are included in the region groups.

### Bilateral Trunk, Hip, and Knee Angle Chain

Keypoints:

- `11 LEFT_SHOULDER`
- `12 RIGHT_SHOULDER`
- `23 LEFT_HIP`
- `24 RIGHT_HIP`
- `25 LEFT_KNEE`
- `26 RIGHT_KNEE`
- `27 LEFT_ANKLE`
- `28 RIGHT_ANKLE`

Features:

- `left_hip_mean` - left hip angle from shoulder, hip, and knee.
- `right_hip_mean` - right hip angle from shoulder, hip, and knee.
- `left_knee_mean` - left knee angle from hip, knee, and ankle.
- `right_knee_mean` - right knee angle from hip, knee, and ankle.
- `trunk_lean_angle` - repo fallback is a shoulder-symmetry proxy; true trunk lean is not directly implemented.

### Bilateral Ankle Separation Proxy

Keypoints:

- `27 LEFT_ANKLE`
- `28 RIGHT_ANKLE`

Features:

- `step_length_cv` - currently populated from `step_width_std`; ankle separation variability proxy.

## Excluded (Non-Region-Specific)

Only high-priority (`H`) features are listed here unless otherwise noted.

- `walking_speed_ms` - whole-body/center-of-mass or all-keypoint motion statistic.
- `positional_symmetry_score` - aggregate across multiple upper- and lower-body symmetry regions.
- `postural_sway_area` - center-of-mass proxy from all available keypoints.
- `stride_length_m` - medium-priority (`M`) row; mixed center-of-mass/cadence computation with an ankle-distance fallback.

## Task #3 De-Duped Keypoint Table

This table lists only the BLAZEPOSE_33 landmarks that participate in high-priority, region-specific feature mappings. Excluded whole-body, center-of-mass, multi-region aggregate, and medium-priority features are intentionally not assigned to individual keypoints.

| BLAZEPOSE_33 index | Keypoint | Features involved |
|---:|---|---|
| 11 | `LEFT_SHOULDER` | `left_hip_mean`; `trunk_lean_angle` proxy |
| 12 | `RIGHT_SHOULDER` | `right_hip_mean`; `trunk_lean_angle` proxy |
| 23 | `LEFT_HIP` | `left_hip_mean`; `left_knee_mean` |
| 24 | `RIGHT_HIP` | `right_hip_mean`; `right_knee_mean` |
| 25 | `LEFT_KNEE` | `left_hip_mean`; `left_knee_mean` |
| 26 | `RIGHT_KNEE` | `right_hip_mean`; `right_knee_mean` |
| 27 | `LEFT_ANKLE` | `left_knee_mean`; `step_length_cv` proxy |
| 28 | `RIGHT_ANKLE` | `right_knee_mean`; `step_length_cv` proxy |

## Alias and Proxy Notes

- `step_length_cv` is not currently an independent true step-length CV in `GaitFeatureVector`; it is populated from `step_width_std`.
- `trunk_lean_angle` is not directly computed as trunk orientation in the current feature pipeline. It reads `trunk_lean` if supplied upstream, otherwise falls back to `shoulder_symmetry_index * 10`.
- `walking_speed_ms`, `positional_symmetry_score`, `postural_sway_area`, and `stride_length_m` are clinically meaningful Myopathy features, but they are not single-region features under the current repo implementation and the stated grouping rules.
