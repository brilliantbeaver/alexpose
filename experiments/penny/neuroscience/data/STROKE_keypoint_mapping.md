# Stroke Feature to BLAZEPOSE_33 Keypoint Mapping

This file maps the Stroke features extracted from the provided `## Stroke` CSV to BLAZEPOSE_33 keypoints used by the GAVD JEPA pipeline.

## Analysis Basis

- The `Priority` and `Feature` columns were extracted to `penny/neuroscience/data/STROKE_features.csv`.
- Source-of-truth feature math is in `ambient/pose/joint_angles.py`, `ambient/analysis/feature_extractor.py`, `ambient/analysis/temporal_analyzer.py`, `ambient/analysis/symmetry_analyzer.py`, and `ambient/classification/features.py`.
- BLAZEPOSE_33 landmarks referenced by region-specific Stroke features:
  - `11 LEFT_SHOULDER`, `12 RIGHT_SHOULDER`
  - `23 LEFT_HIP`, `24 RIGHT_HIP`
  - `25 LEFT_KNEE`, `26 RIGHT_KNEE`
  - `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`
  - `31 LEFT_FOOT_INDEX`, `32 RIGHT_FOOT_INDEX`
- Important implementation caveat: `TemporalAnalyzer` cycle and phase logic still hardcodes COCO ankle indices `15` and `16`. In BLAZEPOSE_33 those are `LEFT_WRIST` and `RIGHT_WRIST`. The intended anatomical mapping for temporal gait features is `27 LEFT_ANKLE` and `28 RIGHT_ANKLE`.

## Task #2 Table

This table includes every Stroke feature from the CSV, including excluded and non-high-priority rows.

| Feature | BLAZEPOSE_33 region | Keypoints include |
|---|---|---|
| `hip_asymmetry` | Bilateral hip angle asymmetry | Medium-priority feature. `11 LEFT_SHOULDER`, `23 LEFT_HIP`, `25 LEFT_KNEE`, `12 RIGHT_SHOULDER`, `24 RIGHT_HIP`, `26 RIGHT_KNEE`. Derived from `left_hip_mean` and `right_hip_mean`; not independent. |
| `knee_asymmetry` | Bilateral knee angle asymmetry | `23 LEFT_HIP`, `25 LEFT_KNEE`, `27 LEFT_ANKLE`, `24 RIGHT_HIP`, `26 RIGHT_KNEE`, `28 RIGHT_ANKLE`. Derived from `left_knee_mean` and `right_knee_mean`; not independent. |
| `ankle_asymmetry` | Bilateral ankle angle asymmetry | `25 LEFT_KNEE`, `27 LEFT_ANKLE`, `31 LEFT_FOOT_INDEX`, `26 RIGHT_KNEE`, `28 RIGHT_ANKLE`, `32 RIGHT_FOOT_INDEX`. Derived from `left_ankle_mean` and `right_ankle_mean`; not independent. |
| `left_hip_range` | Left sagittal hip angle chain | Low-priority feature. `11 LEFT_SHOULDER`, `23 LEFT_HIP`, `25 LEFT_KNEE`. |
| `right_hip_range` | Right sagittal hip angle chain | Low-priority feature. `12 RIGHT_SHOULDER`, `24 RIGHT_HIP`, `26 RIGHT_KNEE`. |
| `left_knee_range` | Left sagittal knee angle chain | Medium-priority feature. `23 LEFT_HIP`, `25 LEFT_KNEE`, `27 LEFT_ANKLE`. |
| `right_knee_range` | Right sagittal knee angle chain | Medium-priority feature. `24 RIGHT_HIP`, `26 RIGHT_KNEE`, `28 RIGHT_ANKLE`. |
| `left_ankle_range` | Left sagittal ankle angle chain | Low-priority feature. `25 LEFT_KNEE`, `27 LEFT_ANKLE`, `31 LEFT_FOOT_INDEX`. |
| `right_ankle_range` | Right sagittal ankle angle chain | Low-priority feature. `26 RIGHT_KNEE`, `28 RIGHT_ANKLE`, `32 RIGHT_FOOT_INDEX`. |
| `walking_speed_ms` | Excluded - whole-body motion | Current code converts `walking_speed_pixels_per_sec` or fallback `velocity_mean`; these are computed from center-of-mass or all-keypoint motion, not a single anatomical region. |
| `cadence_steps_min` | Excluded - mixed temporal/COM fallback | Current feature vector reads temporal cadence when present, otherwise falls back to COM frequency-derived `estimated_cadence`. Excluded because it is not uniquely region-specific. |
| `stride_length_m` | Excluded - mixed COM/ankle fallback | Low-priority feature. Primary computation uses center-of-mass motion and cadence; fallback uses `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. |
| `step_width_m` | Bilateral ankle separation | Medium-priority feature. `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. |
| `stance_percentage` | Bilateral ankle phase timing | Medium-priority feature. Intended BLAZEPOSE_33 keypoints: `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`; current temporal code has the COCO-index caveat above. |
| `swing_percentage` | Bilateral ankle phase timing | Medium-priority feature. Intended BLAZEPOSE_33 keypoints: `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`; current temporal code has the COCO-index caveat above. |
| `double_support_percentage` | Bilateral ankle phase timing | Intended BLAZEPOSE_33 keypoints: `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`; current temporal code has the COCO-index caveat above. |
| `stance_swing_ratio` | Bilateral ankle phase timing | Intended BLAZEPOSE_33 keypoints: `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`; current temporal code has the COCO-index caveat above. |
| `stride_length_si` | Bilateral ankle trajectory asymmetry | `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. In the feature-vector fallback, it is computed from left/right ankle total path length asymmetry. |
| `stance_time_si` | Bilateral ankle cycle timing | `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. Derived from cycle-duration/stance timing asymmetry when available. |
| `swing_time_si` | Bilateral ankle cycle timing | `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. Derived from step-frequency/swing timing symmetry proxy when available. |
| `hip_angle_si` | Bilateral hip angle asymmetry | Medium-priority feature. `11 LEFT_SHOULDER`, `23 LEFT_HIP`, `25 LEFT_KNEE`, `12 RIGHT_SHOULDER`, `24 RIGHT_HIP`, `26 RIGHT_KNEE`. |
| `knee_angle_si` | Bilateral knee angle asymmetry | Medium-priority feature. `23 LEFT_HIP`, `25 LEFT_KNEE`, `27 LEFT_ANKLE`, `24 RIGHT_HIP`, `26 RIGHT_KNEE`, `28 RIGHT_ANKLE`. |
| `ankle_angle_si` | Bilateral ankle angle asymmetry | Medium-priority feature. `25 LEFT_KNEE`, `27 LEFT_ANKLE`, `31 LEFT_FOOT_INDEX`, `26 RIGHT_KNEE`, `28 RIGHT_ANKLE`, `32 RIGHT_FOOT_INDEX`. |
| `velocity_mean` | Excluded - whole-body kinematics | Medium-priority feature. Computed as mean motion speed across all keypoints. |
| `velocity_std` | Excluded - whole-body kinematics | Low-priority feature. Computed as velocity variability across all keypoints. |
| `jerk_mean` | Excluded - whole-body kinematics | Low-priority feature. Computed from acceleration changes across all keypoints. |
| `stride_time_cv` | Bilateral ankle cycle timing | Medium-priority feature. Intended BLAZEPOSE_33 keypoints: `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. Literal alias/proxy of `step_regularity_cv`; not independent. |
| `step_length_cv` | Bilateral ankle / step-width proxy | Medium-priority feature. `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. Currently populated from `step_width_std`, so it behaves as ankle separation variability, not true step length CV. |
| `trunk_lean_angle` | Bilateral shoulder/trunk proxy | Medium-priority feature. Current repo fallback uses `shoulder_symmetry_index`, which traces to `11 LEFT_SHOULDER`, `12 RIGHT_SHOULDER`. |
| `pelvic_tilt_mean` | Pelvis/hip alignment proxy | Low-priority feature. Current fallback uses `hip_distance_symmetry_index`, tracing most directly to `23 LEFT_HIP`, `24 RIGHT_HIP`, with shoulder/hip center-line context. |
| `com_movement_mean` | Excluded - center of mass / whole body | All available BLAZEPOSE_33 landmarks are averaged into a center-of-mass proxy. |
| `com_movement_std` | Excluded - center of mass / whole body | All available BLAZEPOSE_33 landmarks are averaged into a center-of-mass proxy. |
| `com_stability_index` | Excluded - center of mass / whole body | All available BLAZEPOSE_33 landmarks are averaged into a center-of-mass proxy. |
| `postural_sway_area` | Excluded - center of mass / whole body | Convex hull or bounding-box area of the center-of-mass proxy from all available BLAZEPOSE_33 landmarks. |
| `step_width_std` | Bilateral ankle separation | Low-priority feature. `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. |
| `step_width_range` | Bilateral ankle separation | Low-priority feature. `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. |
| `ankle_distance_asymmetry` | Bilateral ankle trajectory asymmetry | Low-priority feature. `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. |
| `cycle_duration_asymmetry` | Bilateral ankle cycle timing | Medium-priority feature. Intended BLAZEPOSE_33 keypoints: `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`; current temporal code has the COCO-index caveat above. |
| `double_support_duration_mean` | Bilateral ankle phase timing | Medium-priority feature. Intended BLAZEPOSE_33 keypoints: `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`; current temporal code has the COCO-index caveat above. |
| `stance_duration_mean` | Bilateral ankle phase timing | Medium-priority feature. Intended BLAZEPOSE_33 keypoints: `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`; current temporal code has the COCO-index caveat above. |
| `swing_duration_mean` | Bilateral ankle phase timing | Medium-priority feature. Intended BLAZEPOSE_33 keypoints: `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`; current temporal code has the COCO-index caveat above. |
| `phase_asymmetry` | Bilateral ankle phase timing | Medium-priority feature. Intended BLAZEPOSE_33 keypoints: `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`; current temporal code has the COCO-index caveat above. |
| `overall_symmetry_index` | Excluded - multi-region symmetry aggregate | Aggregates symmetry indices across multiple unrelated pairs: shoulders `11/12`, elbows `13/14`, wrists `15/16`, hips `23/24`, knees `25/26`, ankles `27/28`, heels `29/30`, and foot indices `31/32`. |
| `positional_symmetry_score` | Excluded - multi-region symmetry aggregate | Aggregates positional symmetry across multiple unrelated pairs: shoulders, elbows, wrists, hips, knees, ankles, heels, and foot indices. |
| `movement_symmetry_score` | Excluded - multi-region symmetry aggregate | Aggregates movement symmetry/correlation across multiple unrelated pairs: shoulders, elbows, wrists, hips, knees, ankles, heels, and foot indices. |
| `temporal_symmetry_score` | Bilateral ankle cycle timing | `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. `SymmetryAnalyzer` temporal symmetry focuses on left/right ankle vertical motion and uses the BLAZEPOSE_33 mapping correctly. |

## Task #1 Grouped View

Only high-priority (`H`) features are included in the region groups.

### Bilateral Knee Angle Chain

Keypoints:

- `23 LEFT_HIP`
- `24 RIGHT_HIP`
- `25 LEFT_KNEE`
- `26 RIGHT_KNEE`
- `27 LEFT_ANKLE`
- `28 RIGHT_ANKLE`

Features:

- `knee_asymmetry` - derived from `left_knee_mean` and `right_knee_mean`; not independent.

### Bilateral Ankle Angle Chain

Keypoints:

- `25 LEFT_KNEE`
- `26 RIGHT_KNEE`
- `27 LEFT_ANKLE`
- `28 RIGHT_ANKLE`
- `31 LEFT_FOOT_INDEX`
- `32 RIGHT_FOOT_INDEX`

Features:

- `ankle_asymmetry` - derived from `left_ankle_mean` and `right_ankle_mean`; not independent.

### Bilateral Ankle Trajectory and Phase Timing

Keypoints:

- `27 LEFT_ANKLE`
- `28 RIGHT_ANKLE`

Features:

- `double_support_percentage` - intended ankle-derived gait-phase timing.
- `stance_swing_ratio` - intended ankle-derived stance/swing timing ratio.
- `stride_length_si` - left/right ankle path-length asymmetry fallback for stride-length symmetry.
- `stance_time_si` - ankle-derived temporal asymmetry proxy.
- `swing_time_si` - ankle-derived temporal asymmetry proxy.
- `temporal_symmetry_score` - ankle-based temporal symmetry aggregate from `SymmetryAnalyzer`.

## Excluded (Non-Region-Specific)

Only high-priority (`H`) features are listed here.

- `walking_speed_ms` - whole-body/center-of-mass or all-keypoint motion statistic.
- `cadence_steps_min` - mixed temporal-cycle/center-of-mass fallback feature.
- `com_movement_mean` - center-of-mass proxy from all available keypoints.
- `com_movement_std` - center-of-mass proxy from all available keypoints.
- `com_stability_index` - center-of-mass proxy from all available keypoints.
- `postural_sway_area` - center-of-mass sway area from all available keypoints.
- `overall_symmetry_index` - aggregate across multiple upper- and lower-body symmetry regions.
- `positional_symmetry_score` - aggregate across multiple upper- and lower-body positional symmetry regions.
- `movement_symmetry_score` - aggregate across multiple upper- and lower-body movement symmetry regions.

## Task #3 De-Duped Keypoint Table

This table lists only the BLAZEPOSE_33 landmarks that participate in high-priority, region-specific feature mappings. Excluded whole-body, center-of-mass, multi-region aggregate, and non-high-priority features are intentionally not assigned to individual keypoints.

| BLAZEPOSE_33 index | Keypoint | Features involved |
|---:|---|---|
| 23 | `LEFT_HIP` | `knee_asymmetry` |
| 24 | `RIGHT_HIP` | `knee_asymmetry` |
| 25 | `LEFT_KNEE` | `knee_asymmetry`; `ankle_asymmetry` |
| 26 | `RIGHT_KNEE` | `knee_asymmetry`; `ankle_asymmetry` |
| 27 | `LEFT_ANKLE` | `knee_asymmetry`; `ankle_asymmetry`; `double_support_percentage`; `stance_swing_ratio`; `stride_length_si`; `stance_time_si`; `swing_time_si`; `temporal_symmetry_score` |
| 28 | `RIGHT_ANKLE` | `knee_asymmetry`; `ankle_asymmetry`; `double_support_percentage`; `stance_swing_ratio`; `stride_length_si`; `stance_time_si`; `swing_time_si`; `temporal_symmetry_score` |
| 31 | `LEFT_FOOT_INDEX` | `ankle_asymmetry` |
| 32 | `RIGHT_FOOT_INDEX` | `ankle_asymmetry` |

## Alias and Proxy Notes

- `hip_asymmetry`, `knee_asymmetry`, and `ankle_asymmetry` are derived summaries of left/right mean joint angles, not independent computations.
- `stride_time_cv` and `step_regularity_cv` trace to the same underlying computation. `TemporalAnalyzer` sets `stride_time_cv = step_regularity_cv`.
- `step_length_cv` is not currently an independent true step-length CV in `GaitFeatureVector`; it is populated from `step_width_std`.
- Temporal cycle and phase features are anatomically ankle-based, but the current `TemporalAnalyzer` implementation should be updated to accept `keypoint_format` so BLAZEPOSE_33 uses `27/28` instead of hardcoded COCO `15/16`.
