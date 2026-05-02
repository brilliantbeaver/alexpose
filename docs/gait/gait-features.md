# Gait feature reference (82 features)

This document explains the **82 gait features** produced by `ambient.classification.features.GaitFeatureVector`.

You can use it as:

- A **tutorial** for reading a feature vector
- A **reference** when you are debugging feature extraction
- A **guide** for understanding what feature values mean in real gait

## Table of contents

- [What the input is](#what-the-input-is)
- [Quick glossary](#quick-glossary)
- [Common value ranges, and why they are approximate](#common-value-ranges-and-why-they-are-approximate)
- [Feature groups](#feature-groups)
- [1) Core joint angles (15 features)](#1-core-joint-angles-15-features)
- [2) Spatiotemporal parameters (4 features)](#2-spatiotemporal-parameters-4-features)
- [3) Temporal phase features (4 features)](#3-temporal-phase-features-4-features)
- [4) Symmetry indices (6 features)](#4-symmetry-indices-6-features)
- [5) Kinematic features (9 features)](#5-kinematic-features-9-features)
- [6) Variability metrics (3 features)](#6-variability-metrics-3-features)
- [7) Postural features (2 features)](#7-postural-features-2-features)
- [8) Extended joint angles (6 features)](#8-extended-joint-angles-6-features)
- [9) Extended temporal features (4 features)](#9-extended-temporal-features-4-features)
- [10) Stability features (4 features)](#10-stability-features-4-features)
- [11) Extended stride features (5 features)](#11-extended-stride-features-5-features)
- [12) Extended symmetry features (6 features)](#12-extended-symmetry-features-6-features)
- [13) Advanced temporal features (8 features)](#13-advanced-temporal-features-8-features)
- [14) Advanced symmetry features (4 features)](#14-advanced-symmetry-features-4-features)
- [15) Enhanced kinematic features (2 features)](#15-enhanced-kinematic-features-2-features)
- [Practical tutorials](#practical-tutorials)

## What the input is

The feature extractor starts from a **pose sequence over time** (a series of frames). From that it computes joint angles, timing, symmetry, stability, and movement statistics.

Some values are in **real units** (meters, seconds) and some are in **pixel units** (pixels per second). Pixel values depend on camera zoom and resolution, so treat them as **relative** unless you calibrate your setup.

## Quick glossary

- **Left (L) and Right (R)**: body sides.
- **Mean**: average value across the sequence.
- **Range**: \(max - min\) across the sequence.
- **Std**: standard deviation across the sequence. Higher means more variability.
- **Asymmetry**: difference between left and right. Higher means less symmetric.
- **SI (Symmetry Index)**: a normalized percent difference between left and right.
- **CV (Coefficient of Variation)**: \(std / mean\). Higher means less consistent.

## Common value ranges, and why they are approximate

Gait features vary with:

- Speed (slow walk vs fast walk)
- Age, height, and leg length
- Camera view and scaling (for pixel based values)
- Health condition and compensation patterns

So the ranges below are **rules of thumb**. They are meant to help you detect values that are clearly wrong, like a stride length of 10 meters, or a stance percentage of 5 percent.

## Feature groups

The 82 features are organized as:

1. Core joint angles (15)
2. Spatiotemporal parameters (4)
3. Temporal phase features (4)
4. Symmetry indices (6)
5. Kinematic features (9)
6. Variability metrics (3)
7. Postural features (2)
8. Extended joint angles (6)
9. Extended temporal features (4)
10. Stability features (4)
11. Extended stride features (5)
12. Extended symmetry features (6)
13. Advanced temporal features (8)
14. Advanced symmetry features (4)
15. Enhanced kinematic features (2)

Total: 15 + 4 + 4 + 6 + 9 + 3 + 2 + 6 + 4 + 4 + 5 + 6 + 8 + 4 + 2 = **82**.

---

## 1) Core joint angles (15 features)

These are angle statistics for hip, knee, and ankle. The goal is to capture typical posture and how much each joint moves during walking.

### Mean joint angles (6)

#### `left_hip_mean`, `right_hip_mean` (degrees)

- **Meaning**: average hip flexion angle over the sequence.
- **Typical range**: often around **10 to 40 degrees**, depending on how the angle is defined and the walking speed.
- **How to read it**:
  - Higher mean can indicate more hip flexion posture, or a crouched gait.
  - Very large magnitude values (over 180) usually mean angle calculation is wrong.
- **Example**:
  - L = 25°, R = 24° looks symmetric.
  - L = 40°, R = 15° suggests a strong side difference or a tracking issue.

#### `left_knee_mean`, `right_knee_mean` (degrees)

- **Meaning**: average knee flexion angle over the sequence.
- **Typical range**: often **10 to 60 degrees** depending on definition and phase coverage.
- **How to read it**:
  - Higher mean can suggest a more flexed knee posture, like crouch gait.
- **Example**:
  - L = 20°, R = 22° is mild and symmetric.
  - L = 55°, R = 20° is a large asymmetry.

#### `left_ankle_mean`, `right_ankle_mean` (degrees)

- **Meaning**: average ankle angle over the sequence.
- **Typical range**: depends heavily on the definition. A common healthy walking pattern has ankle dorsiflexion and plantarflexion around neutral, so mean values often sit in a moderate band like **-20 to +20 degrees**.
- **Example**:
  - L = 5°, R = 7° is typical.
  - L = -40°, R = 10° can be real in some conditions, but can also indicate bad keypoints.

### Left right asymmetry (absolute difference) (3)

These are absolute differences between left and right mean angles.

#### `hip_asymmetry` (degrees)

- **Meaning**: \(|left_hip_mean - right_hip_mean|\)
- **Typical range**: **0 to 10 degrees** for many healthy walks, but can be higher in pathology or if the camera view hides one side.
- **Example**:
  - L = 25°, R = 24° → hip_asymmetry = 1°
  - L = 40°, R = 15° → hip_asymmetry = 25°

#### `knee_asymmetry` (degrees)

- **Meaning**: \(|left_knee_mean - right_knee_mean|\)
- **Typical range**: often **0 to 10 degrees** in symmetric gait.

#### `ankle_asymmetry` (degrees)

- **Meaning**: \(|left_ankle_mean - right_ankle_mean|\)
- **Typical range**: often **0 to 10 degrees**.

### Range of motion (ROM) (6)

Range is computed as \(max - min\) for the angle time series.

#### `left_hip_range`, `right_hip_range` (degrees)

- **Meaning**: how much the hip moves over the sequence.
- **Typical range**: often **20 to 60 degrees** depending on speed and angle definition.
- **Example**:
  - left hip goes from 10° to 45° → range = 35°

#### `left_knee_range`, `right_knee_range` (degrees)

- **Meaning**: knee ROM across the sequence.
- **Typical range**: often **30 to 80 degrees**.
- **Example**:
  - knee goes from 5° to 65° → range = 60°

#### `left_ankle_range`, `right_ankle_range` (degrees)

- **Meaning**: ankle ROM across the sequence.
- **Typical range**: often **15 to 50 degrees**.

---

## 2) Spatiotemporal parameters (4 features)

These features describe speed and step geometry. They are among the most clinically meaningful features.

### `walking_speed_ms` (meters per second)

- **Meaning**: estimated forward walking speed.
- **Typical range**:
  - Slow: **0.4 to 0.8 m/s**
  - Comfortable adult walk: **1.0 to 1.4 m/s**
  - Fast walk: **1.5 to 2.0 m/s**
- **Common failure mode**: if pixel to meter calibration is wrong, this can be too small or too large.
- **Example**:
  - 1.2 m/s is a typical comfortable pace.
  - 0.2 m/s is very slow and can be real for severe impairment, or a calibration error.

### `cadence_steps_min` (steps per minute)

- **Meaning**: step rate.
- **Typical range**:
  - Slow: **70 to 90 steps/min**
  - Typical: **95 to 125 steps/min**
  - Fast: **125 to 160 steps/min**
- **Example**:
  - 110 steps/min is typical.
  - 20 steps/min is almost surely a detection error.

### `stride_length_m` (meters)

- **Meaning**: distance between successive contacts of the same foot.
- **Typical range**: often **0.9 to 1.6 m** for adults at normal speed.
- **Example**:
  - 1.3 m is typical.
  - 0.2 m suggests shuffling steps, or a scaling problem.

### `step_width_m` (meters)

- **Meaning**: side to side distance between the feet during steps.
- **Typical range**: roughly **0.05 to 0.20 m**.
- **How to read it**:
  - Higher values can indicate a wider base of support, often used for stability.
  - Very high values (over 0.5 m) are usually wrong.
- **Example**:
  - 0.10 m is typical.
  - 0.25 m is noticeably wide.

---

## 3) Temporal phase features (4 features)

These features describe how a gait cycle is divided into stance and swing.

### `stance_percentage` (percent of gait cycle)

- **Meaning**: percent of the gait cycle where the foot is on the ground.
- **Typical range**: **55 to 70 percent**.
- **Example**:
  - 62 percent stance is common.
  - 30 percent stance is suspicious and may indicate phase inversion.

### `swing_percentage` (percent of gait cycle)

- **Meaning**: percent of the gait cycle where the foot is in the air.
- **Typical range**: **30 to 45 percent**.
- **Example**:
  - 38 percent swing is common.

### `double_support_percentage` (percent of gait cycle)

- **Meaning**: percent of time when both feet are on the ground.
- **Typical range**: often **10 to 30 percent**.
- **How to read it**:
  - Higher values often indicate cautious walking and reduced stability.
- **Example**:
  - 18 percent is typical.
  - 40 percent suggests very cautious gait or a segmentation issue.

### `stance_swing_ratio` (unitless)

- **Meaning**: \(stance\_percentage / swing\_percentage\)
- **Typical range**: around **1.2 to 2.0**, often near **1.5**.
- **Example**:
  - stance = 60, swing = 40 → ratio = 1.5
  - stance = 70, swing = 30 → ratio = 2.33 (often seen in cautious gait)

---

## 4) Symmetry indices (6 features)

These use a normalized Symmetry Index (SI):

\[
SI = \frac{L - R}{0.5(L + R)} \times 100
\]

### How to interpret SI

- **Magnitude \(|SI|\)**: how asymmetric the gait is
- **Sign**:
  - Positive means left is larger than right
  - Negative means right is larger than left

A useful rule of thumb is:

- \(|SI| < 12\%\): often seen in healthy walking
- \(|SI| > 16\%\): often seen in pathological or compensatory gait

#### `stride_length_si` (percent)

- **Meaning**: asymmetry of stride length.
- **Example**:
  - L = 1.20 m, R = 1.16 m
  - SI = (1.20 - 1.16) / 0.5(1.20 + 1.16) * 100 ≈ 3.4%

#### `stance_time_si` (percent)

- **Meaning**: asymmetry of stance time (how long each side stays in stance).
- **Example**:
  - L stance = 0.70 s, R stance = 0.60 s → SI ≈ 15.4%

#### `swing_time_si` (percent)

- **Meaning**: asymmetry of swing time.
- **Example**:
  - L swing = 0.40 s, R swing = 0.46 s → SI ≈ -13.9%

#### `hip_angle_si`, `knee_angle_si`, `ankle_angle_si` (percent)

- **Meaning**: normalized asymmetry of the mean joint angle for each joint.
- **Example**:
  - left knee mean = 30°, right knee mean = 27°
  - knee_angle_si = (30 - 27) / 0.5(30 + 27) * 100 ≈ 10.5%

---

## 5) Kinematic features (9 features)

These are motion statistics. In this implementation they are **pixel based** because they come from keypoint movement in the image plane.

### How to interpret pixel based kinematics

- They are great for **relative comparisons** within the same camera setup.
- They are less reliable for absolute comparisons across different cameras.

#### `velocity_mean` (pixels per second)

- **Meaning**: average speed of keypoint motion.
- **Typical range**: depends on resolution. For a stable camera and a walking person, values like **50 to 400 px/s** are common.
- **Example**:
  - 80 px/s could be slow or far from the camera.
  - 300 px/s could be fast or close to the camera.

#### `velocity_std` (pixels per second)

- **Meaning**: variability of velocity.
- **How to read it**:
  - Low std means smooth, consistent motion.
  - High std can indicate irregular motion or tracking jitter.
- **Example**:
  - mean = 200, std = 20 → stable motion
  - mean = 200, std = 120 → very variable motion

#### `velocity_max`, `velocity_min` (pixels per second)

- **Meaning**: peak and minimum observed velocities.
- **How to read it**:
  - Very high max can happen during fast limb swings, but can also be keypoint glitches.
  - A min near 0 is common because some keypoints can pause briefly.

#### `acceleration_mean` (pixels per second squared)

- **Meaning**: average acceleration magnitude.
- **How to read it**:
  - Higher values indicate more rapid changes in velocity.

#### `acceleration_std` (pixels per second squared)

- **Meaning**: variability of acceleration.
- **Example**:
  - high std often indicates jerky movement or noisy tracking.

#### `acceleration_max` (pixels per second squared)

- **Meaning**: peak acceleration.
- **Common failure mode**: very large spikes can indicate missed detections.

#### `jerk_mean` (pixels per second cubed)

- **Meaning**: average jerk, which measures how quickly acceleration changes.
- **How to read it**:
  - Higher jerk is often associated with less smooth movement.

#### `jerk_std` (pixels per second cubed)

- **Meaning**: variability of jerk.
- **How to read it**:
  - High jerk variability can indicate tremor like motion, irregular stepping, or jitter in keypoints.

---

## 6) Variability metrics (3 features)

These describe consistency. Higher variability often correlates with instability or neurological impairment, but it can also come from segmentation errors.

#### `stride_time_cv` (unitless)

- **Meaning**: coefficient of variation for stride or step timing.
- **Typical range**:
  - Very consistent: **0.01 to 0.03**
  - Mild variability: **0.03 to 0.06**
  - High variability: **> 0.06**
- **Example**:
  - mean stride time = 1.0 s, std = 0.02 s → CV = 0.02

#### `step_length_cv` (unitless)

- **Meaning**: coefficient of variation for step length.
- **Typical range**: similar order of magnitude as `stride_time_cv` if computed correctly.
- **Important note**: in this pipeline it is currently filled using a proxy (`step_width_std`). That means the name reads like step length, but the value behaves more like width variability.

#### `stride_velocity_cv` (unitless)

- **Meaning**: coefficient of variation for walking speed.
- **Typical range**:
  - Consistent: **0.02 to 0.05**
  - Variable: **> 0.08**
- **Important note**: in this pipeline it is currently filled using a proxy (`velocity_std`). That makes it reflect pixel velocity variability unless calibrated.

---

## 7) Postural features (2 features)

These capture upper body and pelvis alignment.

#### `trunk_lean_angle` (degrees)

- **Meaning**: how much the trunk leans forward or sideways.
- **Typical range**:
  - Small lean: **0 to 10 degrees**
  - Larger lean: **10 to 25 degrees**
- **Example**:
  - 5° often looks upright.
  - 20° suggests forward lean or compensation.

#### `pelvic_tilt_mean` (degrees)

- **Meaning**: average pelvic tilt related measure.
- **Typical range**: often **0 to 15 degrees** depending on definition.
- **Example**:
  - 3° is small.
  - 18° is large and may indicate pelvic compensation or a compute proxy.

---

## 8) Extended joint angles (6 features)

These are standard deviations of joint angles.

#### `left_hip_std`, `right_hip_std` (degrees)

- **Meaning**: variability of hip angle across the sequence.
- **Typical range**: often **3 to 20 degrees**.
- **How to read it**:
  - Low std means repeatable hip motion.
  - High std can indicate irregular gait, speed changes, or angle noise.

#### `left_knee_std`, `right_knee_std` (degrees)

- **Meaning**: variability of knee angle.
- **Typical range**: often **5 to 25 degrees**.

#### `left_ankle_std`, `right_ankle_std` (degrees)

- **Meaning**: variability of ankle angle.
- **Typical range**: often **3 to 20 degrees**.

---

## 9) Extended temporal features (4 features)

These describe sequence length and periodicity.

#### `sequence_length` (frames)

- **Meaning**: number of frames in the sequence.
- **Typical range**: depends on clip length. For a 5 second clip at 30 fps it is about **150 frames**.
- **Example**:
  - 300 frames at 30 fps is 10 seconds.

#### `duration_seconds` (seconds)

- **Meaning**: clip duration.
- **Typical range**: depends on your dataset, often **2 to 20 seconds**.

#### `dominant_frequency` (Hz)

- **Meaning**: main movement frequency in the sequence. For walking, it often correlates with step rhythm.
- **Typical range**:
  - Typical walking step frequency is around **1 to 2.5 Hz**.
- **Example**:
  - 2.0 Hz means a repeating pattern about twice per second.

#### `fps` (frames per second)

- **Meaning**: sampling rate used for timing calculations.
- **Typical range**: usually **24, 30, or 60**.
- **Example**:
  - fps = 30 means each frame is about 0.033 seconds.

---

## 10) Stability features (4 features)

These estimate balance and stability using center of mass like motion patterns.

#### `com_movement_mean` (pixels or normalized units)

- **Meaning**: average center of mass movement magnitude.
- **How to read it**:
  - Higher can indicate more body sway or larger movement.
  - It can also increase when the person is closer to the camera.

#### `com_movement_std` (pixels or normalized units)

- **Meaning**: variability of center of mass movement.
- **Example**:
  - low std suggests steady movement.
  - high std suggests swaying or instability, or noisy tracking.

#### `com_stability_index` (unitless)

- **Meaning**: a stability score where larger values typically indicate more stable movement, depending on implementation.
- **How to sanity check**:
  - If it is negative or extremely large, it often indicates a compute problem.

#### `postural_sway_area` (pixels squared or normalized area)

- **Meaning**: area covered by center of mass sway.
- **How to read it**:
  - Larger area means more sway.

---

## 11) Extended stride features (5 features)

These focus on foot placement and foot trajectory.

#### `step_width_std` (meters)

- **Meaning**: variability of step width.
- **Typical range**: often **0.00 to 0.05 m** for steady walking.
- **Example**:
  - 0.01 m is consistent.
  - 0.08 m is very variable.

#### `step_width_range` (meters)

- **Meaning**: \(max - min\) step width across steps.
- **Typical range**: often **0.00 to 0.15 m**.

#### `left_ankle_total_distance`, `right_ankle_total_distance` (pixels)

- **Meaning**: total path length traveled by each ankle keypoint across the sequence.
- **How to read it**:
  - Larger values can indicate longer stride, more steps, or being closer to the camera.
- **Example**:
  - If left is 4000 px and right is 4100 px, they are similar.

#### `ankle_distance_asymmetry` (unitless or percent like measure)

- **Meaning**: how different the left and right ankle traveled distances are.
- **How to read it**:
  - Near 0 means symmetric.
  - Large values can indicate limping, unequal step lengths, or tracking failure on one side.

---

## 12) Extended symmetry features (6 features)

These are symmetry indices for upper and lower body joints. They are usually treated as percent type symmetry scores.

#### `shoulder_symmetry_index`, `elbow_symmetry_index`, `wrist_symmetry_index` (percent like)

- **Meaning**: left right symmetry for upper limb motion.
- **How to read it**:
  - Low value indicates symmetric arm swing.
  - High value suggests reduced arm swing on one side, or tracking asymmetry.
- **Example**:
  - shoulder = 5% looks symmetric.
  - wrist = 30% suggests one arm is not swinging or not detected well.

#### `hip_symmetry_index`, `knee_symmetry_index`, `ankle_symmetry_index` (percent like)

- **Meaning**: left right symmetry for lower limb motion at each joint.
- **How to read it**:
  - Similar to SI, larger magnitude means less symmetric.

---

## 13) Advanced temporal features (8 features)

These describe gait cycle timing in seconds.

#### `cycle_count` (count)

- **Meaning**: number of gait cycles detected.
- **Typical range**:
  - A 5 to 10 second clip often contains **3 to 12 cycles** depending on speed.
- **Example**:
  - cycle_count = 0 usually indicates cycle detection failed.

#### `left_cycle_duration_mean`, `right_cycle_duration_mean` (seconds)

- **Meaning**: average cycle duration for left and right sides.
- **Typical range**: roughly **0.8 to 1.4 seconds** for typical walking.
- **Example**:
  - left = 1.05 s, right = 1.02 s is symmetric.

#### `cycle_duration_asymmetry` (unitless or percent like)

- **Meaning**: how different left and right cycle durations are.
- **How to read it**:
  - Values near 0 mean symmetric.
  - High values mean one side cycles more slowly, or detection is noisy.

#### `double_support_duration_mean` (seconds)

- **Meaning**: mean time spent in double support.
- **Typical range**: often **0.10 to 0.30 s** in typical walking.
- **Example**:
  - 0.25 s suggests cautious gait or slow speed.

#### `stance_duration_mean` (seconds)

- **Meaning**: mean stance phase duration.
- **Typical range**: often **0.55 to 0.90 s**.

#### `swing_duration_mean` (seconds)

- **Meaning**: mean swing phase duration.
- **Typical range**: often **0.35 to 0.60 s**.

#### `phase_asymmetry` (unitless or percent like)

- **Meaning**: asymmetry across phases, usually capturing stance and swing differences between sides.
- **Example**:
  - low value means similar timing on both sides.
  - high value means timing differences that can match limping patterns.

---

## 14) Advanced symmetry features (4 features)

These summarize symmetry across different aspects.

#### `overall_symmetry_index` (unitless or percent like)

- **Meaning**: a single score summarizing symmetry.
- **How to read it**:
  - near 0 indicates high symmetry.
  - larger means more asymmetry.

#### `positional_symmetry_score` (unitless)

- **Meaning**: symmetry of body positions, like left right alignment.
- **Example**:
  - low score means positions match well across sides.

#### `movement_symmetry_score` (unitless)

- **Meaning**: symmetry of motion patterns, like mirrored trajectories.
- **Example**:
  - high score can indicate one side moves differently, even if average posture is similar.

#### `temporal_symmetry_score` (unitless)

- **Meaning**: symmetry of timing, like phase durations and cycle timing.
- **Example**:
  - high temporal asymmetry can indicate unequal stance time or step timing.

---

## 15) Enhanced kinematic features (2 features)

These are pixel based versions of speed and stride length. They are helpful when you want raw image plane measures without meter conversion.

#### `walking_speed_pixels_per_sec` (pixels per second)

- **Meaning**: walking speed measured directly in the image plane.
- **How to read it**:
  - Larger can mean faster walking, or being closer to the camera.
- **Example**:
  - 150 px/s vs 300 px/s shows a big difference within the same setup.

#### `estimated_stride_length_pixels` (pixels)

- **Meaning**: stride length in pixels.
- **How to read it**:
  - Useful for comparing stride length within one camera view.
- **Example**:
  - 90 px is shorter than 140 px in the same setup.

---

## Practical tutorials

### Tutorial 1: sanity checking one sample

If a sample is healthy and steady, you often see:

- `walking_speed_ms` around 1.0 to 1.4
- `cadence_steps_min` around 95 to 125
- `stance_percentage` around 55 to 70, `swing_percentage` around 30 to 45
- symmetry indices \(|SI|\) below about 12 percent
- modest variability: `stride_time_cv` around 0.01 to 0.04

If you see:

- `fps` = 0
- `cycle_count` = 0
- `stance_percentage` + `swing_percentage` far from 100
- joint angles beyond 180 degrees

then timing detection or angle estimation likely failed.

### Tutorial 2: reading a limp pattern

A common limp signature is:

- high `stance_time_si` or `phase_asymmetry`
- high `cycle_duration_asymmetry`
- increased `double_support_percentage` or `double_support_duration_mean`

Example:

- left cycle duration mean = 1.20 s
- right cycle duration mean = 0.95 s
- stance_time_si = 22%
- double support percentage = 28%

This combination suggests one side spends longer in stance and the person is using more double support, both common compensation strategies.

### Tutorial 3: detecting tracking jitter

If keypoints are noisy you often see:

- high `velocity_std`, `acceleration_std`, and `jerk_std`
- unusually high `velocity_max` spikes
- unstable stability metrics like very large `postural_sway_area`

Example:

- velocity_mean = 200 px/s
- velocity_std = 160 px/s
- velocity_max = 2500 px/s

That max value is likely a detection glitch, not true movement.

