# Cerebral Palsy Feature to BLAZEPOSE_33 Keypoint Mapping

This file maps the Cerebral Palsy features extracted from the provided `## Cerebral Palsy` CSV to BLAZEPOSE_33 keypoints used by the GAVD JEPA pipeline.

## Analysis Basis

- The `Priority` and `Feature` columns were extracted to `penny/neuroscience/data/CP_features.csv`.
- All rows in the provided Cerebral Palsy CSV are marked `H`, so the high-priority filter includes every listed feature.
- BLAZEPOSE_33 lower-body and trunk landmarks used here:
  - `11 LEFT_SHOULDER`, `12 RIGHT_SHOULDER`
  - `23 LEFT_HIP`, `24 RIGHT_HIP`
  - `25 LEFT_KNEE`, `26 RIGHT_KNEE`
  - `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`
  - `31 LEFT_FOOT_INDEX`, `32 RIGHT_FOOT_INDEX`
- Source-of-truth feature math is in `ambient/pose/joint_angles.py`, `ambient/analysis/feature_extractor.py`, `ambient/analysis/symmetry_analyzer.py`, and `ambient/classification/features.py`.
- Implementation caveat: `ambient.pose.joint_angles.JointAngleCalculator` defines BLAZEPOSE_33 ankle angles with `FOOT_INDEX` landmarks. `ambient.analysis.feature_extractor.FeatureExtractor` has BLAZEPOSE_33 landmark mappings, but its generic BLAZEPOSE joint-angle path currently returns no joint angles if used directly.

## Task #2 Table

Only high-priority (`H`) features are included. In this CSV, all extracted features are high priority.


| Feature            | BLAZEPOSE_33 region                 | Keypoints include                                                                                                                                                                                                                                                       |
| ------------------ | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `left_knee_mean`   | Left sagittal knee angle chain      | `23 LEFT_HIP`, `25 LEFT_KNEE`, `27 LEFT_ANKLE`.                                                                                                                                                                                                                         |
| `right_knee_mean`  | Right sagittal knee angle chain     | `24 RIGHT_HIP`, `26 RIGHT_KNEE`, `28 RIGHT_ANKLE`.                                                                                                                                                                                                                      |
| `left_ankle_mean`  | Left sagittal ankle angle chain     | `25 LEFT_KNEE`, `27 LEFT_ANKLE`, `31 LEFT_FOOT_INDEX`.                                                                                                                                                                                                                  |
| `right_ankle_mean` | Right sagittal ankle angle chain    | `26 RIGHT_KNEE`, `28 RIGHT_ANKLE`, `32 RIGHT_FOOT_INDEX`.                                                                                                                                                                                                               |
| `left_hip_mean`    | Left sagittal hip angle chain       | `11 LEFT_SHOULDER`, `23 LEFT_HIP`, `25 LEFT_KNEE`.                                                                                                                                                                                                                      |
| `right_hip_mean`   | Right sagittal hip angle chain      | `12 RIGHT_SHOULDER`, `24 RIGHT_HIP`, `26 RIGHT_KNEE`.                                                                                                                                                                                                                   |
| `knee_asymmetry`   | Bilateral knee angle asymmetry      | `23 LEFT_HIP`, `25 LEFT_KNEE`, `27 LEFT_ANKLE`, `24 RIGHT_HIP`, `26 RIGHT_KNEE`, `28 RIGHT_ANKLE`. Derived from `left_knee_mean` and `right_knee_mean`, so it is not independent.                                                                                       |
| `pelvic_tilt_mean` | Pelvis/hip alignment proxy          | Current feature-vector code reads `pelvic_tilt_asymmetry` if present, otherwise uses `hip_distance_symmetry_index` as a proxy. Repo-traceable proxy keypoints are `23 LEFT_HIP`, `24 RIGHT_HIP`, with body-center context from `11 LEFT_SHOULDER`, `12 RIGHT_SHOULDER`. |
| `walking_speed_ms` | Excluded - whole-body motion        | Current code converts `walking_speed_pixels_per_sec` or fallback `velocity_mean`; these are computed from center-of-mass or all-keypoint motion, not a single anatomical region.                                                                                        |
| `stride_length_m`  | Excluded - mixed COM/ankle fallback | Primary computation uses center-of-mass motion and cadence; fallback uses `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. Excluded because it is not uniquely region-specific.                                                                                                       |


## Task #1 Grouped View

Only high-priority (`H`) features are included.

### Bilateral Trunk-Pelvis-Lower-Limb Angle Chain

Keypoints:

- `11 LEFT_SHOULDER`
- `12 RIGHT_SHOULDER`
- `23 LEFT_HIP`
- `24 RIGHT_HIP`
- `25 LEFT_KNEE`
- `26 RIGHT_KNEE`
- `27 LEFT_ANKLE`
- `28 RIGHT_ANKLE`
- `31 LEFT_FOOT_INDEX`
- `32 RIGHT_FOOT_INDEX`

Features:

- `left_hip_mean` - left hip angle from shoulder, hip, and knee.
- `right_hip_mean` - right hip angle from shoulder, hip, and knee.
- `left_knee_mean` - left knee angle from hip, knee, and ankle.
- `right_knee_mean` - right knee angle from hip, knee, and ankle.
- `left_ankle_mean` - left ankle angle from knee, ankle, and foot index.
- `right_ankle_mean` - right ankle angle from knee, ankle, and foot index.
- `knee_asymmetry` - derived from `left_knee_mean` and `right_knee_mean`; not independent.
- `pelvic_tilt_mean` - pelvis/hip alignment proxy using hip pair, with shoulder/hip center-line context in the repo fallback.

## Excluded (Non-Region-Specific)

Only high-priority (`H`) features are included.

- `walking_speed_ms` - whole-body/center-of-mass or all-keypoint motion statistic.
- `stride_length_m` - mixed center-of-mass/cadence computation with an ankle-distance fallback, so it is not uniquely tied to one BLAZEPOSE_33 region.

## Task #3 De-Duped Keypoint Table

This table lists only the BLAZEPOSE_33 landmarks that participate in high-priority, region-specific feature mappings. Excluded whole-body and mixed-region features are intentionally not assigned to individual keypoints.


| BLAZEPOSE_33 index | Keypoint           | Features involved                                                         |
| ------------------ | ------------------ | ------------------------------------------------------------------------- |
| 11                 | `LEFT_SHOULDER`    | `left_hip_mean`; `pelvic_tilt_mean` proxy context                         |
| 12                 | `RIGHT_SHOULDER`   | `right_hip_mean`; `pelvic_tilt_mean` proxy context                        |
| 23                 | `LEFT_HIP`         | `left_hip_mean`; `left_knee_mean`; `knee_asymmetry`; `pelvic_tilt_mean`   |
| 24                 | `RIGHT_HIP`        | `right_hip_mean`; `right_knee_mean`; `knee_asymmetry`; `pelvic_tilt_mean` |
| 25                 | `LEFT_KNEE`        | `left_hip_mean`; `left_knee_mean`; `left_ankle_mean`; `knee_asymmetry`    |
| 26                 | `RIGHT_KNEE`       | `right_hip_mean`; `right_knee_mean`; `right_ankle_mean`; `knee_asymmetry` |
| 27                 | `LEFT_ANKLE`       | `left_knee_mean`; `left_ankle_mean`; `knee_asymmetry`                     |
| 28                 | `RIGHT_ANKLE`      | `right_knee_mean`; `right_ankle_mean`; `knee_asymmetry`                   |
| 31                 | `LEFT_FOOT_INDEX`  | `left_ankle_mean`                                                         |
| 32                 | `RIGHT_FOOT_INDEX` | `right_ankle_mean`                                                        |


## Alias and Proxy Notes

- `knee_asymmetry` is a derived feature: `abs(left_knee_mean - right_knee_mean)`. It should be interpreted as a summary of the two knee-angle means, not as an independent computation.
- `pelvic_tilt_mean` is currently a proxy in `GaitFeatureVector`: it reads `pelvic_tilt_asymmetry` if supplied by upstream analysis, otherwise falls back to `hip_distance_symmetry_index * 5`.
- `walking_speed_ms` and `stride_length_m` are clinically meaningful CP gait features, but the repo implementation does not make them single-region features. They are therefore excluded from keypoint-region groups under the stated rules.

