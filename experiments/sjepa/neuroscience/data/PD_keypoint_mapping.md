# Parkinson's Feature to BLAZEPOSE_33 Keypoint Mapping

This file maps the Parkinson's features extracted from the provided `## Parkinsons` CSV to BLAZEPOSE_33 keypoints used by the GAVD JEPA pipeline.

## Analysis Basis

- The GAVD JEPA notebooks process skeleton clips as `T x 33 x C` MediaPipe/BlazePose sequences.
- The `Priority` and `Feature` columns were extracted to `penny/neuroscience/data/PD_features.csv`.
- All rows in the provided Parkinson's CSV are marked `H`, so the high-priority filter includes every listed feature.
- Source-of-truth feature math is in `ambient/analysis/feature_extractor.py`, `ambient/analysis/temporal_analyzer.py`, `ambient/analysis/symmetry_analyzer.py`, and `ambient/classification/features.py`.
- Important implementation caveat: `TemporalAnalyzer` cycle and phase logic still hardcodes COCO ankle indices `15` and `16`. In BLAZEPOSE_33 those are `LEFT_WRIST` and `RIGHT_WRIST`. The intended anatomical mapping for those temporal gait features is `27 LEFT_ANKLE` and `28 RIGHT_ANKLE`.

## BLAZEPOSE_33 Lower-Limb Landmarks Used

- `23 LEFT_HIP`, `24 RIGHT_HIP`
- `25 LEFT_KNEE`, `26 RIGHT_KNEE`
- `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`
- `29 LEFT_HEEL`, `30 RIGHT_HEEL`
- `31 LEFT_FOOT_INDEX`, `32 RIGHT_FOOT_INDEX`

## Task #2 Table

Only high-priority (`H`) features are included. In this CSV, all extracted features are high priority.

| Feature | BLAZEPOSE_33 region | Keypoints include |
|---|---|---|
| `stride_length_m` | Excluded - mixed COM/ankle fallback | Primary computation uses center-of-mass motion from all available BLAZEPOSE_33 landmarks; fallback uses `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. Excluded because it is not uniquely region-specific. |
| `stride_time_cv` | Bilateral ankle cycle timing | Intended BLAZEPOSE_33 keypoints: `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. Note: this is a literal alias/proxy of `step_regularity_cv` in `TemporalAnalyzer`; it should not be treated as an independent feature. |
| `step_length_cv` | Bilateral ankle / step-width proxy | `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. In `GaitFeatureVector`, this field is currently populated from `step_width_std`, so it behaves as ankle separation variability, not true step length CV. |
| `stride_velocity_cv` | Excluded - whole-body velocity proxy | Current `GaitFeatureVector` populates this from `velocity_std`, computed across all keypoints, not from the temporal analyzer's stride-velocity CV. Excluded as non-region-specific. |
| `com_movement_mean` | Excluded - center of mass / whole body | All available BLAZEPOSE_33 landmarks are averaged into a center-of-mass proxy. |
| `com_movement_std` | Excluded - center of mass / whole body | All available BLAZEPOSE_33 landmarks are averaged into a center-of-mass proxy. |
| `com_stability_index` | Excluded - center of mass / whole body | All available BLAZEPOSE_33 landmarks are averaged into a center-of-mass proxy. |
| `postural_sway_area` | Excluded - center of mass / whole body | Convex hull or bounding-box area of the center-of-mass proxy from all available BLAZEPOSE_33 landmarks. |
| `stride_length_si` | Bilateral ankle trajectory asymmetry | `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. In the feature-vector fallback, it is computed from left/right ankle total path length asymmetry. |
| `phase_asymmetry` | Bilateral ankle gait-phase timing | Intended BLAZEPOSE_33 keypoints: `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. Current phase extraction has the COCO-index caveat described above. |
| `overall_symmetry_index` | Excluded - multi-region symmetry aggregate | Aggregates symmetry indices across multiple unrelated pairs: shoulders `11/12`, elbows `13/14`, wrists `15/16`, hips `23/24`, knees `25/26`, ankles `27/28`, heels `29/30`, and foot indices `31/32`. |
| `hip_symmetry_index` | Bilateral hip trajectory | `23 LEFT_HIP`, `24 RIGHT_HIP`. |
| `knee_symmetry_index` | Bilateral knee trajectory | `25 LEFT_KNEE`, `26 RIGHT_KNEE`. |
| `ankle_symmetry_index` | Bilateral ankle trajectory | `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. |
| `cycle_duration_asymmetry` | Bilateral ankle cycle timing | Intended BLAZEPOSE_33 keypoints: `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. Current cycle detection has the COCO-index caveat described above. |
| `movement_symmetry_score` | Excluded - multi-region symmetry aggregate | Aggregates movement symmetry/correlation across multiple unrelated pairs: shoulders, elbows, wrists, hips, knees, ankles, heels, and foot indices. |
| `temporal_symmetry_score` | Bilateral ankle cycle timing | `27 LEFT_ANKLE`, `28 RIGHT_ANKLE`. `SymmetryAnalyzer` temporal symmetry focuses on left/right ankle vertical motion and uses the BLAZEPOSE_33 mapping correctly. |

## Task #1 Grouped View

Only high-priority (`H`) features are included.

### Bilateral Hip Trajectory

Keypoints:

- `23 LEFT_HIP`
- `24 RIGHT_HIP`

Features:

- `hip_symmetry_index` - left/right hip movement-speed symmetry from the feature extractor.

### Bilateral Knee Trajectory

Keypoints:

- `25 LEFT_KNEE`
- `26 RIGHT_KNEE`

Features:

- `knee_symmetry_index` - left/right knee movement-speed symmetry from the feature extractor.

### Bilateral Ankle Trajectory and Cycle Timing

Keypoints:

- `27 LEFT_ANKLE`
- `28 RIGHT_ANKLE`

Features:

- `stride_time_cv` - alias/proxy of `step_regularity_cv`; not independent.
- `step_length_cv` - currently populated from `step_width_std`; ankle separation variability proxy.
- `stride_length_si` - left/right ankle path-length asymmetry fallback for stride-length symmetry.
- `phase_asymmetry` - intended stance/swing phase asymmetry from ankle-derived gait cycles.
- `ankle_symmetry_index` - left/right ankle movement-speed symmetry.
- `cycle_duration_asymmetry` - left/right cycle-duration asymmetry from ankle-derived gait cycles.
- `temporal_symmetry_score` - ankle-based temporal symmetry aggregate from `SymmetryAnalyzer`.

## Excluded (Non-Region-Specific)

Only high-priority (`H`) features are included.

- `stride_length_m` - primary computation uses center-of-mass movement from all keypoints, with an ankle fallback; not uniquely tied to one keypoint region.
- `stride_velocity_cv` - current feature-vector field is populated from whole-body `velocity_std`, not the temporal stride-velocity CV.
- `com_movement_mean` - center-of-mass proxy from all available keypoints.
- `com_movement_std` - center-of-mass proxy from all available keypoints.
- `com_stability_index` - center-of-mass proxy from all available keypoints.
- `postural_sway_area` - sway area of the center-of-mass proxy from all available keypoints.
- `overall_symmetry_index` - aggregate across multiple upper- and lower-body symmetry regions.
- `movement_symmetry_score` - aggregate across multiple upper- and lower-body movement-symmetry regions.

## Task #3 De-Duped Keypoint Table

This table lists only the BLAZEPOSE_33 landmarks that participate in high-priority, region-specific feature mappings. Excluded center-of-mass, whole-body, and multi-region aggregate features are intentionally not assigned to individual keypoints.

| BLAZEPOSE_33 index | Keypoint | Features involved |
|---:|---|---|
| 23 | `LEFT_HIP` | `hip_symmetry_index` |
| 24 | `RIGHT_HIP` | `hip_symmetry_index` |
| 25 | `LEFT_KNEE` | `knee_symmetry_index` |
| 26 | `RIGHT_KNEE` | `knee_symmetry_index` |
| 27 | `LEFT_ANKLE` | `stride_time_cv` alias/proxy of `step_regularity_cv`; `step_length_cv` proxy from `step_width_std`; `stride_length_si`; `phase_asymmetry`; `ankle_symmetry_index`; `cycle_duration_asymmetry`; `temporal_symmetry_score` |
| 28 | `RIGHT_ANKLE` | `stride_time_cv` alias/proxy of `step_regularity_cv`; `step_length_cv` proxy from `step_width_std`; `stride_length_si`; `phase_asymmetry`; `ankle_symmetry_index`; `cycle_duration_asymmetry`; `temporal_symmetry_score` |

## Alias and Proxy Notes

- `stride_time_cv` and `step_regularity_cv` trace to the same underlying computation. `TemporalAnalyzer` sets `stride_time_cv = step_regularity_cv`, and `GaitFeatureVector` reads `step_regularity_cv` when filling `stride_time_cv`.
- `step_length_cv` is not currently an independent true step-length CV in `GaitFeatureVector`; it is populated from `step_width_std`.
- `stride_velocity_cv` is not currently an independent true stride-velocity CV in `GaitFeatureVector`; it is populated from whole-body `velocity_std`.
- Temporal cycle and phase features are anatomically ankle-based, but the current `TemporalAnalyzer` implementation should be updated to accept `keypoint_format` so BLAZEPOSE_33 uses `27/28` instead of hardcoded COCO `15/16`.
