<!-- iteration02-execution-status -->
> This is the preserved design or feasibility record. Consult the [implemented protocol](protocol.md), [verified results](results.md), and [external execution report](external_replication_results.md) for current execution status.

# Data and pipeline advice for the second research iteration

This is a read-only feasibility review of the refreshed annotations, retained poses, completed study artifacts, and ingestion code. It was performed on November 15, 2026. No video was downloaded, no model was trained, and no historical result was changed for this review. The proposed cohort counts below are availability calculations, not new detection results.

## The most useful immediate change

Use a separately declared lower-limb readout for the next main experiment, while retaining the existing 16-joint model input and uniform masking allowlist. Keep arm-related analysis as an additional observable-subset experiment.

This recommendation follows a concrete data problem. All five normal sources rejected by the current 12-joint profile rule fail only elbow or wrist support. Every one has at least four distinct eligible target patches for each of the eight current lower-leg readout joints. The missing arm observations should not disqualify an otherwise measurable lower-limb trajectory, and their absence must not be interpreted as a neurological arm deficit.

Applying the existing availability table's four-patch minimum to only joints 25–32 yields a candidate cohort of **304 source videos, including 28 normal sources, 1,381 sequences and 9,442 windows**. The current published draft's cohort remains **251 sources, including 23 normal sources, 1,303 sequences and 8,950 windows**. The candidate rule retains 53 additional sources without lowering visibility, changing pose estimation, or changing what joints the model may mask. It selects a different endpoint and observation population, so it needs its own frozen protocol and results. Its apparent sample gain is not a new result about performance.

All 28 candidate normal sources already appear in the earlier study index. The extra five are newly eligible for this readout, not newly independent normal participants. That distinction must appear in the README, new notebooks, slides and manuscript.

| Folder | Current profile sources | Candidate lower-limb sources |
| --- | ---: | ---: |
| Normal | 23 | 28 |
| Parkinsons | 9 | 10 |
| Stroke | 14 | 15 |
| Cerebral palsy | 10 | 10 |
| Myopathic | 25 | 27 |
| Abnormal | 92 | 103 |
| Antalgic | 8 | 9 |
| Exercise | 59 | 87 |
| Inebriated | 2 | 5 |
| Pregnant | 2 | 2 |
| Prosthetic | 6 | 7 |
| Style | 2 | 2 |
| Distinct union | 251 | 304 |

One source has two abnormal category memberships, so summing category rows does not give the distinct union. The focal five-class candidate subset has 90 distinct sources: 28 normal, 10 Parkinsons, 15 stroke, 10 cerebral palsy and 27 myopathic. These remain small condition-specific samples.

If the next question concerns arm-leg relationships, do not quietly give it the 304-source denominator. Report its own arm-supported population and compare all relevant methods on identical supported sources. A lower-limb primary and a separately labeled arm-leg secondary are clearer than silently changing the score dimension for each person.

## What the fresh inventory actually contains

A new read through `build_study_manifest` confirmed 1,874 sequence CSVs, 458,116 annotation rows, 348 distinct source IDs and 32 normal source IDs. The annotation inventory is fully represented relative to the pinned public catalog; it is not fully represented by usable videos or measurable gait.

The existing acquisition ledger records 332 decodable sources. For normal videos specifically, 31 are present and one failed acquisition. Four normal sources contribute no accepted 32-frame pose window: one has missing media/poses, one contains only too-short annotated sequences, and two fail observation quality. The five additional sources lost at the current profile gate fail arm support, as described above.

The realistic recovery order is therefore:

1. Reuse all currently valid lower-limb observations under a new declared readout.
2. Retry the one missing normal source under a bounded, logged acquisition rule. Its recovery would add observations, not a never-explored participant population.
3. Inspect the two poor-validity normal sources with frame/bounding-box overlays before considering a new pose configuration. If a crop or person-selection problem is demonstrated, apply the same correction policy to every affected category and write a new pose-cache version.
4. Treat the too-short source as short rather than fabricating extra frames. Shorter-window sensitivity can be declared separately, but does not create additional gait cycles.

Do not choose recovery attempts because a source has an inconvenient anomaly score. Do not lower confidence thresholds, interpolate through missing detections, or splice separate annotations merely to increase the normal count.

## Long-cycle modeling has a measurable cost

The completed model uses 32 observations at 15 Hz, with stride 16. The first-to-last sample span is 31/15, approximately 2.07 seconds. A four-frame token spans three sample intervals and covers a nominal 4/15-second patch. These windows support short movement completion; they do not guarantee two or more complete gait cycles.

A read-only rewindowing probe retained the existing confidence threshold, adjacent-frame interpolation, prefix torso normalization, four-frame tokens, 75% overall validity and 50% validity per half. It used stride 16 for every row below. Counts are before the joint-specific profile gate and are upper bounds for a more demanding cycle analysis.

| Samples at 15 Hz | First-to-last span | Accepted windows | Accepted sequences | Sources | Normal sources | Normal windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2.07 seconds | 9,474 | 1,389 | 311 | 28 | 482 |
| 48 | 3.13 seconds | 8,226 | 1,248 | 303 | 27 | 367 |
| 64 | 4.20 seconds | 7,039 | 966 | 284 | 23 | 284 |
| 96 | 6.33 seconds | 5,368 | 664 | 231 | 12 | 179 |
| 128 | 8.47 seconds | 4,219 | 438 | 162 | 9 | 129 |

A six-to-eight-second primary would sharply reduce the normal population before any phase-quality gate. With 12 normals, four-fold evaluation with five reference normals would leave only four fitting normals per fold. That design would not solve the learning problem in the current draft.

Prefer a short-window main experiment, with phase uncertainty explicitly represented or reported, and a matched 64-frame sensitivity if the scientific question requires longer context. Keep a real-time representation beside any phase-aligned representation. Resampling every stride to an identical phase length can remove cadence, freezing, asymmetry or variable stance duration, which may themselves be relevant observations. The next method should not treat a phase estimate as an observed neurological state.

The 135 currently retained normal sequences have a median accepted-window coverage span of 3.13 seconds; 25% have only a single 2.07-second window. This span is not proof of uninterrupted high-confidence measurements or complete cycles. A cycle notebook must display the exact accepted interval and its missingness.

## An unused annotation resource can check phase estimation

The fresh CSVs contain **751 initial-contact labels**: 383 right and 368 left contacts, across 80 sequences from 11 source videos. There are seven other nonempty event labels. No initial-contact annotations occur in the normal folder in this snapshot.

| Annotated category | Sources with any nonempty gait event | Nonempty event rows |
| --- | ---: | ---: |
| Abnormal | 6 | 359 |
| Exercise | 1 | 33 |
| Myopathic | 2 | 40 |
| Parkinsons | 1 | 56 |
| Style | 1 | 270 |

There are no consecutive same-foot contact labels one frame apart in this inventory. The median same-foot contact spacing is 45 source frames, but source FPS varies, so 45 must not be treated as seconds or a universal gait period. Each contact must be mapped using its video's actual FPS and the verified one-based frame convention.

These event annotations provide an auxiliary check of a fixed phase/contact estimator. Report timing error in seconds, event matching tolerance, missed contacts, unmatched detections, per-source coverage and source-level uncertainty. Do not present 751 contacts as 751 independent people. Do not assume that sparse labels imply every unlabeled frame is a negative contact.

The annotations must not become a hidden source of test tuning. A fixed rule can be evaluated descriptively against them without tuning. If choosing a phase detector or its thresholds using these labels, allocate a separate source-disjoint auxiliary development set and disclose that the phase component used abnormal-source event supervision. Do not then claim the whole pipeline was learned only from normal videos. The same source must not choose phase settings and evaluate the final detector in the same fold.

## What the existing ingestion pipeline guarantees

The next implementation should reuse these functions rather than copy an older notebook's ad hoc ingestion.

| Existing component | Relevant behavior | Next-iteration use |
| --- | --- | --- |
| `study_data.build_study_manifest` | Reads all 12 current condition folders, excludes `source/`, verifies schema and sequence/source consistency, hashes CSV bytes and creates prior-use flags. | Build the new inventory from current CSVs. Use historical indices only for exposure/provenance flags. |
| `data.find_video` | Searches complete source-ID files under `data/videos-gavd/<source_id>/` and the legacy flat location, requiring a decodable first frame. | Keep one source recording per ID. First-frame success does not prove that every annotated frame decodes. |
| `scripts/ingest_study.py` | Gives normal sources first download opportunity, then stable source-ID order; records bounded timeouts and failures. | Retain bounded normal-first acquisition and a complete failure ledger. |
| `ambient.utils.youtube_cache.download_youtube_urls` | Supplies the actual downloader used by the study wrapper. | Keep downloads and logs under `data/videos-gavd`; do not invent a second notebook-specific downloader. |
| `batch.batch_extract` | Uses process-local Ambient extractors and verifies cache provenance. | Reuse valid poses; write a new versioned cache if extraction behavior changes. |
| `data.extract_sequence` | Uses GAVD per-frame boxes, scales annotation dimensions to decoded video dimensions, adds a 5% margin, decodes source frame number minus one, restores full-frame pixel coordinates. | Inspect person selection, box scaling and source-frame alignment before a new extraction policy. |
| `ambient.pose.keypoint_extractor.SequenceKeypointExtractor` | Uses MediaPipe image detection and returns the first detected pose in the selected crop. | Do not describe the current pipeline as a validated temporal tracker or guaranteed person identity. |
| `data._resample` | Interpolates only between adjacent valid source frames; preserves exact observations and does not bridge frame gaps. | Preserve missingness. A new anti-aliasing or resampling method is a registered preprocessing change. |
| `data.normalize_window` | Centers each frame on pelvis and uses a torso scale from the first half of the window. | Reuse for comparability. Note that poor hip observations can invalidate a whole frame and centering removes global translation. |
| `study_data.make_outer_folds` / `fold_metadata` | Freeze source roles with normal-only fitting/reference data and abnormal auxiliary data. | Reuse for a new candidate cohort, with a separate identity and role table. Do not overwrite historical folds. |
| `media.show_window_context` | Displays the actual source and interval associated with a prediction. | Use beside every inspected model explanation. Never show an unrelated attractive gait clip. |
| `media.annotated_frame` | Shows an annotated source frame and its rescaled box. | Add a first/middle/last frame review rather than checking only a single frame. |

Pose caches store raw 33-joint coordinates, confidence, decoded-frame flags, original frame numbers, FPS and annotation/video/model/configuration hashes. The model selects only the 16 permitted joints. `load_windows` uses `raw_xy` and its own prefix-based normalization, not the older sequence-normalized `xy` array. A next implementation must not accidentally use `xy` and thereby change the body reference or introduce full-sequence scale information.

Do not infer clinical stride length, walking speed or ground clearance in centimeters from these monocular projected coordinates. The current pelvis centering removes absolute progression, and torso normalization yields relative image units. Bone-angle and timing proxies need a view-qualified interpretation.

## The exact joint rule remains unchanged

All model masking must be uniform over the declared joint-time token grid, independent of velocity and motion intensity. Missing observations are filtered after the mask is drawn, without redrawing preferentially around fast joints. Additional readout choices do not authorize masking additional joints.

| BLAZEPOSE_33 index | Keypoint name | Features involved |
| --- | --- | --- |
| 11 | Left shoulder | Torso reference, projected trunk orientation and upper-arm relationship. |
| 12 | Right shoulder | Torso reference, projected trunk orientation and upper-arm relationship. |
| 13 | Left elbow | Optional arm-swing and arm-leg relationship readouts. |
| 14 | Right elbow | Optional arm-swing and arm-leg relationship readouts. |
| 15 | Left wrist | Optional arm-swing and arm-leg relationship readouts. |
| 16 | Right wrist | Optional arm-swing and arm-leg relationship readouts. |
| 23 | Left hip | Shared pelvis reference and projected left-leg geometry. |
| 24 | Right hip | Shared pelvis reference and projected right-leg geometry. |
| 25 | Left knee | Lower-limb target and projected knee-angle proxy with hip and ankle. |
| 26 | Right knee | Lower-limb target and projected knee-angle proxy with hip and ankle. |
| 27 | Left ankle | Lower-limb target, relative foot trajectory and timing proxy. |
| 28 | Right ankle | Lower-limb target, relative foot trajectory and timing proxy. |
| 29 | Left heel | Lower-limb target and local foot configuration. |
| 30 | Right heel | Lower-limb target and local foot configuration. |
| 31 | Left foot index | Lower-limb target and local foot configuration. |
| 32 | Right foot index | Lower-limb target and local foot configuration. |

The eight-joint lower-limb readout uses indices 25–32. The shoulders and hips remain model inputs and masking candidates. Arm targets remain eligible during representation training if that is the declared objective, even when the main source score uses lower-limb targets only. That distinction should be explicit in the notebook's tensor diagram.

## View, identity and activity require new accounting

Eleven annotation sequences contain more than one `cam_view` value. Existing window metadata stores the set of views across the whole sequence. That field is adequate as a coarse nuisance descriptor but is not an exact window-level view. Before any view-specific score or phase model, join the original frame-level annotations to the actual window times. Define how to handle transitions and unknown views before inspecting model rankings.

The candidate normal cohort has source memberships across right-side, left-side, front and back views, but these memberships overlap across sources. Its 482 windows include 205 right-side, 115 front, 83 left-side, 69 back, nine from a mixed front/right-side sequence and one from an `other` view. Do not add source-view counts as though they were independent subjects.

GAVD includes exercise and style videos. Some may show movements other than steady walking. Keep the authoritative binary label rule, but require a separately reported focal-condition evaluation and a fixed, label-blind walking-interval review if making a walking-specific claim. Do not remove an exercise source because it is easy to classify or difficult to interpret after scoring.

Check duplicate footage through video fingerprints and temporal overlap. Group confirmed duplicate recordings together before a new source split. Similar normalized pose trajectories are not proof of duplicate people. Do not use face recognition to invent participant identities. Repeated people across separate recordings remain an unresolved dependence unless the dataset supplies reliable identity information or an appropriate manual provenance review establishes a link.

## Notebook design for the next iteration

The exact numbering can follow the coordinator's chosen methodology. The following teaching sequence makes the data-dependent decisions inspectable.

| Proposed notebook | Main question | Required outputs before moving on |
| --- | --- | --- |
| `12_observation_population_and_lower_limb_support.ipynb` | Which gait relations are actually measurable in this corpus? | Fresh inventory; current versus candidate source tables; reason for every newly eligible or excluded source; numerical joint table; actual video and box overlays; a saved new cohort identity. |
| `13_phase_and_known_dependency_checks.ipynb` | Does the proposed timing or relation estimator recover a known signal? | Clearly labeled synthetic processes; phase ambiguity examples; source-disjoint auxiliary event validation if used; 32/48/64-frame availability comparison; no invented gait cycles. |
| `14_nested_context_and_physical_readouts.ipynb` | Does added context improve prediction under a learner that retains the own-context solution? | Exact input-access diagram; normal-only development checks; observable-coordinate targets; linear/harmonic baseline; meaningful checkpoint and target-support assertions. |
| `15_source_matched_controls_and_explanations.ipynb` | Does a score respond to the intended relationship rather than a view or observation change? | Anatomically constrained edits or surrogates; exact paired target support; original/edited video skeletons; units; source-level paired changes; abstentions. |
| `16_frozen_evaluation_and_condition_comparison.ipynb` | Does the frozen method improve the declared endpoint on the available sources? | Paired source-level comparisons on identical cohorts; all-folder and focal-family results; separate supervised classifiers; uncertainty; explicit historical source reuse. |

Each notebook should begin with local `uv` instructions and root `.env` loading through the requested aliases. It should load its prerequisites explicitly from an immutable artifact directory, or state the exact command that creates them. It should run independently in a fresh kernel. Code cells that only read a completed experiment must be described as inspection, not model training. A teaching example with synthetic arrays must never enter a real-data results table.

## A workable sequence of scientific gates

1. Freeze the new observation population using labels, availability and the declared readout, before new model scores are computed. Preserve the 251-source historical study intact.
2. Establish that the new estimator recovers known added-context information on a synthetic process, while preserving missingness and source grouping. A successful shape check is not this test.
3. Require useful normal-development prediction under the declared context conditions. The improvement must be measured on normal observations excluded from fitting, with the small reference population reported. Do not reuse a calibration sample to select many model variants without acknowledging that adaptation.
4. Check coordinate targets and a matched simple predictor before attributing gains to JEPA. Keep random encoders with fitted heads and simple motion features as necessary controls.
5. Freeze one main model recipe, one primary score and score direction, one cohort, and all exclusion rules. At most a small, declared normal-development choice should remain. Extra seeds measure algorithm variation, not new participants.
6. Report every eligible held-out source, including difficult or unscorable cases. If a source lacks the new readout, report coverage and compare common-source performance separately. Do not hide abstentions inside an AUROC denominator.
7. Only then revise manuscript claims. A better all-folder score without a comparable focal-condition result may reflect activity separation. A better small-sample point estimate without informative uncertainty does not establish superiority.

## How sufficient is GAVD?

These scores are editorial judgments, not acceptance probabilities.

| Question | Score / 5 | Assessment |
| --- | ---: | --- |
| Can the catalog support a broad internal observational study? | 4 | It includes 12 categories, hundreds of videos and auditable frame-level annotations. |
| Can the current normal data support a rich new deep model trained from scratch? | 2 | Even the candidate cohort has only 28 previously explored normal sources. Window count does not replace participant diversity. |
| Can it support a careful low-capacity relation or phase study? | 3 | It may, if the estimator has known-signal checks, short-window coverage, shared baselines and conservative conclusions. |
| Can it establish participant-independent generalization? | 1 | Source IDs do not identify independent people, and all normal sources overlap prior development. |
| Can it establish clinical diagnostic validity? | 1 | Folder labels are not revalidated diagnoses, and relevant clinical measurements are absent. |

The README should say that GAVD is sufficient for a bounded exploratory study of observable movement under recording variation, but insufficient by itself for a generalizable clinical classifier or a validated neurological world model. Stronger external claims need independently identified normal participants and relevant abnormal populations from a different acquisition setting. Recovering the five arm-occluded normals is worthwhile, but does not change that conclusion.

## Reproducibility of this advice

The inventory was recomputed using `build_study_manifest` against current `data/gavd`. The source/readout calculation used the completed primary run's `profile_eligibility.csv` and `window_index_before_profile_qc.csv`, requiring every `joint_25_distinct_patches` through `joint_32_distinct_patches` to be at least four. The duration probe reloaded existing pose bytes and repeated `_resample`, `normalize_window` and the existing token-validity test at each listed window length, without writing any new cache. Its 32-frame counts exactly reproduced the saved pre-profile cohort. The event census read the current CSV `gait_event` fields directly.

Relevant evidence is available in [the dataset inventory](../../../../data/gavd/README.md), [the completed study results](../context_study_results.md), [the observation code](../../aaai_gait/data.py), [the source-role code](../../aaai_gait/study_data.py), [the real execution record](../current_execution_validation.md), and [the current submission limits](../draft01/submission_readiness.md).

## Published-baseline support check

A final complete-case probe on the expanded pose corpus found **1,664 fully observed 32-frame windows from 326 sequences and 130 sources**, but only **11 windows from five normal sources**. Fully observed here means finite normalized XY for all 16 permitted joints across every frame, under the existing visibility threshold and normalization.

This is larger than the earlier abnormal complete-case subset, but the normal bottleneck is unchanged. It cannot support the current four-fold design with five separate normal references per fold and remaining normal fitting sources. Do not force STG-NF or another complete-input density model onto zero-filled missing coordinates and call it a faithful, same-observation reproduction. A graph/layout adaptation or observation-aware density model requires separate correctness checks and accurate attribution. A tiny five-normal complete-case demonstration can check code execution, but cannot serve as the principal published-method comparison for this study.

The exact probes are executable through [probe_data.py](probe_data.py). Their counts, parameters, source-code and artifact hashes, fresh annotation identity, and every pose-file hash are recorded in [data_feasibility_probe.json](data_feasibility_probe.json). The script writes only that adjacent report and does not modify the historical experiment. Run it from the existing `uv` environment when reproducing the availability assessment.
