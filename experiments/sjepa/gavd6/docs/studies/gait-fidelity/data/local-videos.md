# What the multiple-sclerosis videos can contribute

> **Scope in the revised study:** this is a dated feasibility audit for a deferred extension. These videos and clinical candidates do not enter `jepa-response-01`. The [active data specification](README.md) reuses the completed parent AMASS bundle; availability statements below retain their original audit dates.

[Gait Fidelity](../README.md) · [Local video gallery](video-gallery.html) · [Clinical dataset audit](clinical-candidates.md)

**Audited September 21, 2026.** The local collection is useful for developing and testing how pose restoration handles difficult real images. It contains walking aids, changes in viewpoint, small or partly cropped people, and recordings with several people. A reviewed subset could support measurements of visible joint positions, anatomical side assignment, and clearly observable gait events. The present files do not provide the independent references needed to establish preservation of clinical asymmetry, affected side, or treatment response.

**Storage:** this audit verified the files on your Mac at `/Users/theodoremui/dev/alexpose/experiments/multiple-sclerosis/video-data-full`. A HAIC copy of this exact collection has not been established. Shared source IDs with GAVD on HAIC do not prove that the same clips are present there; see the [availability inventory](availability.md).

The MS, PD, and Normal names below are labels supplied with the collection. This audit did not infer diagnoses from the videos. The [source project](../../../../../../multiple-sclerosis/README.md) also distinguishes its condition labels from clinically verified diagnoses.

![The local videos support development and visible-reference annotation; clinical measurement claims require independent reference evidence.](../images/07-video-evidence.svg)

*Figure 1. Assign each dataset a role according to its reference evidence. Adding blur or obstruction to a real video provides a controlled observation change, while the original video's hidden joint locations and clinical measurements can remain unknown.*

## Inventory and inspection scope

All 91 MP4s in [video-data-full](../../../../../../multiple-sclerosis/video-data-full) are hydrated video files, rather than Git LFS pointers. Every file was SHA-256 hashed and probed; decoding frame timestamps produced 16,788 video frames without reported errors. Presentation timestamps increased monotonically. These checks establish readable files and usable timing metadata, without proving the accuracy of the original capture clock or the absence of edits.

| Collection label | Raw clips | Filename source IDs | Existing pose caches | Summed clip duration | Median duration | Duration range |
|---|---:|---:|---:|---:|---:|---:|
| MS | 30 | 13 | 29 | 248.17 s | 7.38 s | 3.10–30.00 s |
| PD | 35 | 12 | 35 | 190.36 s | 4.70 s | 0.90–16.72 s |
| Normal | 26 | 16 | 24 | 114.09 s | 4.27 s | 1.43–7.77 s |
| Total | 91 | 41 | 88 | 552.61 s | 4.77 s | 0.90–30.00 s |

The collection occupies 388,687,508 bytes, or 388.69 decimal MB. Its summed duration is 9.21 minutes of clips; overlaps could make the duration of unique footage smaller. Nine clips are shorter than two seconds, and 33 are shorter than four seconds. Whether a clip contains enough complete gait events must be determined from its contents and the proposed measurement. Short recordings cannot be assumed to support repeated-cycle variability.

The decoded frame cadence is approximately 23.976 fps in five clips, 25 fps in one, 29.97 fps in 48, 30 fps in 34, and 59.94 fps in three. Raster dimensions range from 110×222 to 1920×1080, with some portrait recordings up to 1080×1920. The most common sizes are 1280×720 (44 clips), 1920×1080 (23), and 640×360 (15). All MS clips have a 30 fps cadence, with 16 at 1920×1080 and 14 at 640×360. These acquisition differences may correlate with collection labels; a metadata-only control should assess their predictive value using the same source-grouped partitions as the model.

There are no byte-identical MP4s according to SHA-256. This does not rule out overlapping cuts, recompressed copies, reposts, or repeated participants. Source `tsOMPBS277Q` supplies 14 of the 30 raw MS clips and 13 of the 29 cached MS clips. Source `pFLC9C-xH8E` supplies seven PD clips, so a random clip split would be particularly unsuitable.

**There is already substantial source overlap with GAVD.** Comparing the exact filename source IDs against this repository's [GAVD video manifest](../../../../manifests/gavd/gavd_full_videos.csv) and [sequence manifest](../../../../manifests/gavd/gavd_full_sequences.csv) finds **28 of 41 sources, covering 61 of 91 clips**: all 16 Normal sources, three MS sources, and nine PD sources. The [overlap record](../evidence/gavd-source-overlap.json) lists the matches. This verifies shared source IDs, not exact frame overlap or membership in a particular historical training/test split. Cross-check those histories and reservations before using this collection alongside GAVD or gavd5-drift. It cannot be assumed to be an independent external dataset merely because it lives in another experiment folder.

The visual screen selected the lexicographically first clip from each source, then added any omitted extraction exclusions, the shortest clip, and the smallest raster. This produced **43 midpoint frames covering all 41 source IDs**, displayed on six contact sheets. A second purposeful check inspected **six clips at four times each**, at 10%, 35%, 65%, and 90% of their duration, on two additional sheets. The audit therefore inspected eight sheets, not eight complete video playbacks. Full continuous visual review, gait-event annotation, anatomical labeling, and estimates of the prevalence of each failure mode remain to be done.

The [raw video inventory](../evidence/video-inventory.csv), [summary](../evidence/video-summary.json), [decoded timestamp audit](../evidence/timestamps-audit.json), and [pose-cache inventory](../evidence/cache-inventory.json) retain the numerical evidence. Frame-rate values from container headers and measured presentation-time spacing are recorded separately because they need not be identical.

## What the sampled frames show

Open the [local gallery](video-gallery.html) to inspect the original clips. It reads the local sibling collection; its role is local review, rather than redistribution of the footage. Selection and annotation should be recorded before model scores are inspected.

| Clip | Observation in the sampled frames | Implication for Gait Fidelity |
|---|---|---|
| `MS/3T0BfK9HOzU.mp4` | Rear view with a rollator, another person nearby, and a large PRE overlay | Review aid occlusion and target-person tracking |
| `MS/Ivxdl6r2z_o.mp4` | Frontal view with two walking aids and another person nearby | Keep valid unusual movement while recording occlusion and identity uncertainty |
| `MS/WvoNYV6nZtM.mp4` | A 30-second approach with an aid and a substantial change in apparent person size | Preserve physical time and distinguish camera scale from movement |
| `MS/zOxtPrKySB8.mp4` | Partly cropped upper body and a nearby assistant | Record validity separately for each joint instead of treating the whole skeleton as reliable |
| `MS/tsOMPBS277Q_P1.mp4` | A walker, timers and text, with lateral/rear views across a turn | Retain detector failure as an outcome; this clip was excluded from the existing pose cache |
| `Normal/JD1AGVpftps_P1_02.mp4` | A small person within a long corridor view | Measure person size in pixels; nominal video resolution does not describe joint detail |
| `Normal/tUT8Fh1zGKA.mp4` | Side-view walking against a black background | Verify capture or rendering provenance before treating a folder label as evidence of real clinical capture |
| `PD/_Wn9oYGpRdM_P1.mp4` | A 110×222 raster, another person nearby, and feet becoming cropped during approach | Distal-joint and contact measurements have limited image support |
| `PD/pFLC9C-xH8E_P1.mp4` | A caption describes left arm swing | Record the embedded text as a possible label cue; it is not an independent clinical examination or dense annotation |
| `PD/pFLC9C-xH8E_P4_01.mp4` | Only 0.9009 seconds of movement | Do not count it as evidence of repeated-cycle variability |

Several source IDs share similar rooms, framing, timers, and PRE overlays. Some pairs also have similar scene and clothing appearances. These observations justify checking source provenance and related footage, without establishing participant identity from appearance. The visible variation makes this collection useful for finding tracking failures, but also makes condition classification vulnerable to recording cues.

## Reuse the videos, rebuild the measurement cache

The existing caches were prepared for representation learning and condition classification. They contain 7,643 frames across 88 clips, with a median of 68.5 frames per clip and a range of 14–450. Every cache records `fps=15`, and every saved coordinate is finite. The schema contains only `keypoints`, `keypoints_norm`, `fps`, `source_id`, `label`, and `clip_name`. It retains neither original timestamps nor flags identifying missing, interpolated, or padded observations.

Four details in [the extraction code](../../../../../../multiple-sclerosis/sjepa/data.py) matter for movement measurement:

1. **Sampling does not produce a common 15 fps clock.** `load_video_sequence` retains every `round(src_fps/15)`-th frame. For 23.976 or 25 fps footage, this selects every second frame, around 11.99 or 12.5 observations per second. The decoded native spacing is about 41.708 or 40 ms, so the selected frames are roughly 83.416 or 80 ms apart, rather than 66.667 ms. Six files have these cadences. This follows the retained sampling code and decoded video timing; the original OpenCV extraction was not rerun during this audit. A nominal cached rate cannot recover the original time grid.
2. **Interpolation is not restricted to short gaps.** Although `clean_sequence` describes short-gap interpolation, it fills all interior gaps without a maximum duration. It interpolates visibility as well as coordinates and trims empty ends without retaining their source-frame offset. Finite cached values therefore cannot distinguish observed movement from imputation. Using these values as reference truth would reward agreement with the interpolation procedure.
3. **Normalization changes with every frame.** `normalize_sequence` subtracts the current pelvis position and divides by the current torso length. This representation can be useful for classification, while scale changes can alter the amplitude and timing of movement quantities. The cleaned pixel-coordinate field is more useful for diagnostic overlays, but its absent timestamps and missingness flags still prevent a complete physical-time audit.
4. **Joint names do not establish anatomical validity.** MediaPipe supplies 33 landmarks, including heels and foot-index points absent from the current body12 synthetic model. A mapping between them must declare which landmarks are lost, how left and right correspond, and which measurements remain supported. An ankle-only representation cannot automatically supply validated foot-contact events.

For example, `Normal/PAeh4qBwsUk.mp4`, `PD/M-_cogKwXK4_P1_04.mp4`, both `PD/bmi1hYOnTHs` clips, and `PD/v1SoZ_S31pk.mp4` have the approximately 23.976 fps cadence. `Normal/tUT8Fh1zGKA.mp4` has a 25 fps cadence. Their existing caches all declare 15 fps. Keep these as explicit timing checks when building the new cache, rather than assuming that an output array index always represents the same elapsed time.

The three existing exclusions are `Normal/JD1AGVpftps_P1_02.mp4`, `Normal/diCVwltkV5M_P1_01.mp4`, and `MS/tsOMPBS277Q_P1.mp4`. The retained message is only “too few valid frames.” The extraction path requires at least 30% fully finite x-coordinate frames and at least eight frames after cleaning, but the log does not identify which criterion failed in each case. Preserve that uncertainty, as the [existing split audit](../../../../../../multiple-sclerosis/docs/11-full-data-splits.md) does.

For Gait Fidelity, create a new versioned cache from the raw videos and retain:

- MP4 hash, decoded frame index, original presentation timestamp, shot ID, and selected person track;
- unmodified estimator coordinates and confidence, with separate availability, visibility, and reference-validity fields;
- derived interpolation, its bounded physical-time gap rule, and an explicit flag for every imputed or padded value;
- input-only normalization parameters and inverse transforms, so predictions can be returned to the declared measurement coordinates.

The original detector outputs, derived model inputs, and independent reference annotations need separate fields. A cleaned detector sequence is an input candidate, not an anatomical reference.

## A practical experiment using this collection

First establish annotation feasibility on a source-grouped development sample spanning the observed viewpoints, person sizes, aids, and occlusions. Two reviewers, blinded to model identity and model predictions, can annotate visible joint locations, anatomical side where resolvable, and initial contacts only where the image supports them. Record uncertainty and disagreement before adjudication. Agreement between reviewers does not make a hidden joint observable, and difficult clips should remain in the failure and coverage accounting.

Once this protocol is fixed, compare unchanged tracks, temporal filtering, direct coordinate restoration, and the proposed masking variants on the same windows. Use the same references, elapsed times, eligibility rules, and missingness policy for every method. Report position agreement and supported gait quantities alongside missed events, extra events, and coverage. Apply the [measurement design](../methods/evaluation.md) rather than substituting the MS/PD/Normal label for a movement measurement.

A useful extension adds controlled blur or obstruction to those reviewed original videos. Annotations of visible joints in the original footage can remain references for the corrupted counterpart, provided frame geometry and timing are preserved. This measures response to a known observation change. It does not reveal originally hidden joints or turn the original pose estimator into ground truth. Pairing a pathological-looking clip with a healthy-looking clip would not provide a matched physical intervention.

![Related clips and windows remain within one source group; verified person or repost relationships join sources before partitioning.](../images/09-source-splits.svg)

*Figure 2. The minimum boundary is the source recording. Verified person, session, or repost links can require a larger group. Repeated windows, artificial occlusions, and additional cameras do not create additional independent people.*

Use verified person/recording/repost groups when available; otherwise keep known sources intact and state the unresolved identity limitation. The suffix `_P...` is not evidence of a separate participant. The existing [five source-grouped folds](../../../../../../multiple-sclerosis/artifacts/eval/full-v1/fold_registry.json) are useful precedents, but these recordings have already been inspected during development. A new partition of the same familiar collection does not create an untouched external cohort. Confirmatory clinical measurement claims need separately reserved, independently referenced participants.

## Reference evidence and rights still needed

The [mapping document](../../../../../../multiple-sclerosis/mapping-data/ms-pd-mapping.md) explains possible disease-related gait features; it does not supply patient-level contact times, severity scores, or affected-side records for these clips. The audited local material did not provide independently verified affected-side examinations, synchronized motion capture or force measurements, calibrated cameras, per-frame manual joint truth, or a complete clip-to-original time crosswalk. A baked-in caption or timer is worth recording as source information, while its clinical meaning and timing still require verification.

The GAVD source matches provide useful links to existing sequence and camera-view annotations, without filling those reference gaps. The three matching MS sources are labeled broadly `abnormal` in the examined GAVD gait-pattern field; that is not independent confirmation of the more specific MS label. A GAVD camera view such as `left side` describes viewpoint, not the clinically affected side of the person.

The local Git LFS storage guide describes how to retrieve files, not source-specific permissions for reuse or redistribution. Resolve original source provenance and applicable terms before publishing footage or adopting a new data-use role. The local gallery can support review while those records are assembled.

The immediate contribution is a carefully documented real-video stress test and a feasible route to independent visible-reference annotation. Whether the proposed restoration method preserves meaningful bilateral gait changes remains an experiment to run, with the clinical references and statistical boundaries specified in [the full study](../README.md).
