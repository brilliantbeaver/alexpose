# Current data audit and evaluation boundaries

> Historical foundation record. This document describes the earlier five-condition experiment. The revised twelve-folder study is documented in [the current protocol](context_study_protocol.md), [real results](context_study_results.md), and [critical review](context_study_review.md). Its manuscript is in [draft01](draft01/README.md).

The experiment reads the current CSV files under `data/gavd` directly. It does not load the older `multiple-sclerosis` manifests, pickled feature caches, or the five upstream CSVs in `data/gavd/source`. Reading both the split CSVs and their source copies would count annotations twice. The local [GAVD README](../../../data/gavd/README.md) records the complete annotation snapshot and upstream provenance.

## What the refreshed inventory contains

A fresh scan on November 14, 2026 found the following selected records. These are annotation counts before video availability, pose extraction, or quality exclusions.

| Source gait label | Sequences | Annotated frames | Source recordings |
| --- | ---: | ---: | ---: |
| normal | 291 | 41,340 | 32 |
| parkinsons | 47 | 10,426 | 11 |
| stroke | 76 | 33,934 | 19 |
| cerebral palsy | 64 | 20,346 | 11 |
| myopathic | 188 | 34,595 | 30 |
| **Selected union** | **666** | **140,641** | **103** |

The repeated `stroke` item in the requested list contributes one class. The source folder is literally `cerebral palsy`, with a space. Labels come from `gait_pat`. The broader `dataset` field can say `Abnormal Gait` even for a sequence whose `gait_pat` is `normal`, so it cannot substitute for the class label.

The full local dataset contains 1,874 sequences, 458,116 annotated frames and 348 distinct YouTube IDs. The selected cohort is therefore a well defined subset, not all of GAVD. None of the selected 103 recordings spans two selected labels in this snapshot, although the evaluation code supports mixed-label recordings. The portable SHA256 identity of selected sequence IDs, labels, source IDs and current CSV byte hashes is:

```text
e1a781f3113406f692a67dab7608c6d1fae135b546882996a4a5533e4d898d1f
```

This is an audit value, not a constant that overrides future files. `build_manifest()` recomputes it on every run. A changed annotation byte changes the identity. Cache files carry annotation and video hashes, model-weight hash, frame-index convention, confidence threshold, crop margin and extractor-version identifier. The window corpus identity includes every accepted pose file hash and the normalization, time sampling and window configuration. Checkpoints should bind to this corpus identity as well as the split identity.

## Is the cohort large enough to establish generalizability?

It supports an internal feasibility experiment, but it cannot by itself establish clinical or population generalizability. The 666 sequences are not 666 independent participants. Parkinsonian and cerebral palsy examples each come from only 11 source recordings. Overlapping windows increase optimization examples, not independent sample size. A recording identifier also cannot establish that the same person never appears in another recording or compilation.

YouTube footage can contain teaching demonstrations, edited clips, walking aids, changes in viewpoint, compression artifacts, mirrored footage and repeated presentations of a person. Clinical labels, severity, medication state, demographics and acquisition protocol are not uniformly verified. GAVD annotation availability does not imply permission to redistribute source media, and the local dated availability report checked public responses rather than actual decoding. Downloads and decoding therefore get their own outcome ledger. These limitations require careful claims, rather than an assertion that sufficient frame count solves generalization.

Describe the task as recognizing **source gait-pattern labels** and detecting deviations from the learned normal reference within this selected collection. Do not claim disease diagnosis, causal explanations or detection of every abnormal gait. A patient's gait can be ordinary in a short video, and a deviation can result from factors outside the selected conditions.

The four nonnormal labels form a useful, interpretable first study, but they do not have one shared etiology. Myopathic gait concerns muscle-related impairment and should not be described as the same central neurological disorder as stroke or Parkinson's disease. Their membership is a prespecified practical cohort choice. It does not prove that a learned boundary follows neuroscience rather than filming conditions. Other GAVD patterns should remain outside the primary five-class evaluation. A subsequent open-set study could hold out one complete condition from representation adaptation and supervised fitting, then test whether the normal-only detector flags it. Such an experiment needs fresh splits and a separate prespecified report.

A stronger paper needs an independent collection with verified participant IDs and clinical labels, or a clearly limited internal source-group study that directly measures these uncertainty sources. Until external evaluation exists, keep external validity as an unresolved question.

## Frozen train, validation and test membership

`grouped_split()` assigns complete source recordings before pose windows are created. It uses class counts to approach a 60/20/20 split and never sees model predictions. For sources that each have one label, class-specific source quotas guarantee coverage. Repeated random candidate assignments choose the best sequence-count balance within those quotas. Mixed-label sources remain together. A split with missing classes raises an error rather than silently reporting an easier task.

The default seed 42, applied to the current annotation inventory, gives:

| Label | Train sequences / sources | Validation sequences / sources | Test sequences / sources |
| --- | ---: | ---: | ---: |
| normal | 173 / 20 | 57 / 6 | 61 / 6 |
| parkinsons | 29 / 7 | 9 / 2 | 9 / 2 |
| stroke | 42 / 11 | 13 / 4 | 21 / 4 |
| cerebral palsy | 39 / 7 | 15 / 2 | 10 / 2 |
| myopathic | 120 / 18 | 30 / 6 | 38 / 6 |

These counts precede pose quality exclusions. Keep the assignment fixed when reporting availability and extraction attrition. Do not repeatedly resplit until an appealing test score appears. If an entire class disappears from a partition after QC, the experiment stops. Report the failure and revise the acquisition/QC protocol before opening test predictions. Repeated training seeds on one frozen split estimate optimization variation. Repeated grouped folds estimate a different source-sampling uncertainty and must retrain all encoders inside each fold.

## How the existing alexpose code is reused

The `multiple-sclerosis/sjepa/data.py` tutorial demonstrates `ambient.pose.keypoint_extractor.SequenceKeypointExtractor.extract_from_image`. It obtains 33 pixel-coordinate landmarks and their MediaPipe visibility values. That tutorial processes prepared local clips without per-frame GAVD boxes. Our `extract_sequence()` reuses the same ambient extractor, but reads every current sequence's per-frame bbox and source-video frame number.

The repository's `ambient/gavd/gavd_processor.py` and `ambient/pose/keypoint_extractor.py` interpret GAVD frame numbers as one-based absolute source positions. The tutorial still makes `frame_number_base=1` an explicit recorded choice and asks the reader to inspect the first annotated frame against the bbox. We seek to `frame_num - 1`, not to a clip-relative offset. The CSV's `vid_info` dimensions describe the annotation image, so bbox coordinates scale to the actual downloaded dimensions before cropping. Landmark pixel coordinates translate back from the crop to the full image.

There is a separate temporal-alignment limitation. The CSV's `vid_info` field does not record the frame rate used when annotations were created. A downloaded rendition could have a different frame rate, or the hosted video could have been edited. Scaling bbox coordinates and reading the current FPS cannot prove that an absolute frame number still refers to the same instant. Inspect sampled bbox overlays and known events against the actual rendition before treating extracted sequences as paper evidence. Do not invent an original FPS or silently rescale annotation frame numbers. The current pipeline records actual FPS and preserves original frame numbers, but automatic original-rendition alignment is not established.

A bbox crop reduces the chance of selecting a bystander, but it does not guarantee person identity. The ambient wrapper selects the first returned pose inside its crop. Overlay inspection is required when the box contains more than one person or loses the annotated walker. Confidence is a quality signal, not a calibrated error probability.

Source videos are downloaded once under `data/videos-gavd/<source_id>/<source_id>.<ext>`. Existing flat files at `data/videos-gavd/<source_id>.<ext>` are reused only if a frame decodes. No media is copied into condition folders. One recording may supply several sequences, so this source-centered layout avoids duplicate downloads and preserves grouping. Download reports keep failures visible. Source hashes detect accidental substitution or changed media when extraction runs.

## Time, missing poses and normalization

The pose cache retains all 33 detected landmarks for visual audit, the actual source FPS, original frame numbers, confidence and decode flags. Model windows use exactly the 16 approved joints, with XY coordinates only. Confidence never becomes a model input or RF feature. Missing or low-confidence coordinates become NaN. Sampling uses original frame times and linearly interpolates onto 15 Hz only between adjacent valid original frames. A source below the target FPS is excluded with the reason `source_fps_below_target`; the workflow never upsamples slower source videos. This keeps a prefix sample from using a raw source frame beyond the half-window forecast boundary. The original source FPS remains in window metadata. A lower prespecified target FPS is a possible separate acquisition protocol, not an automatic per-video adjustment. It never bridges missing annotation frames. A token contains four consecutive frames of one selected joint. It is valid only when all eight XY values are finite. A 32-frame window is accepted when at least 75% of its 128 tokens are valid, and at least 50% of the tokens in each temporal half are valid. Missing coordinates remain NaN until the model projection boundary. The model excludes invalid tokens from attention keys, pooled embeddings and target losses. The mask sampler still chooses uniformly across the declared token grid, independent of motion and confidence, while loss accounting ignores invalid targets. This preserves the requested uniform masking experiment without inventing unobserved motion.

The first implementation required every joint in every frame to be observed. Its pilot rejected all windows, so the final implementation replaced complete-case exclusion with explicit token validity. The threshold change is a documented preprocessing decision made during a technical pilot, before fitting or examining classifier performance. It is not a retrospective accuracy optimization. Report retained counts and missing-token fractions by class, source and camera view, and investigate attrition before interpreting accuracy.

Each model window centers every frame on its pelvis and divides coordinates by median torso length calculated from the first half of that window only. Changing future coordinates therefore cannot change the normalized observed prefix. This supports the tutorial's half-window forecasting diagnostic. A shorter observed prefix would need its own normalization rule. The cache's whole-sequence normalized XY is available for visualization only and is not used to construct training windows. Image coordinates provide projected motion rather than calibrated 3D movement. Framewise centering removes camera translation and also removes absolute forward travel. The system cannot infer walking speed in meters per second from these coordinates.

## What the exp5 comparison means

The exp5 notebooks use `ambient.pose.joint_angles.get_joint_angles`, `GaitFeatureVector` and a Random Forest classifier. They examine different feature sets and older cached inputs. Their recorded scores cannot be numerically compared with a newly restricted cohort and refreshed split. Both RF and S-JEPA must be rerun on the exact same accepted sequences and split membership.

`build_ambient_features()` provides the exact 15 core angle fields from the current ambient implementation. It explicitly requests `feature_groups=['core_angles']`. Calling `GaitFeatureVector.from_joint_angles()` only calculates that core information. Reporting its default broader array as 82 fully measured features would include fields that were never computed.

The main `build_features()` baseline uses 36 observed-value summaries on the same normalized windows as S-JEPA: six angle means, three mean-angle asymmetries, six angle ranges, six angle standard deviations, wrist and ankle speed/excursion summaries, ankle separation statistics and projected torso-orientation variability. Its first 15 fields follow the exp5 angle construction, and its additional fields are stated explicitly. Each statistic uses available observations. A speed difference needs both adjacent coordinates. A feature with no valid observation remains NaN. A training-only median imputer handles absent statistics and degenerate angles, with a recorded zero fallback for any training column that is entirely missing. This fallback is computational and carries no physiological meaning. The RF uses balanced class weights; optional source weights keep a recording with many windows from dominating training. No training routine invokes ungrouped internal cross-validation. Any RF hyperparameter search must use the frozen validation sources or grouped inner folds confined to training sources.

## Outcomes and uncertainty

First average window probabilities within each annotated sequence. Keep two differently labeled sequences separate even when they share a recording. Report five-class macro F1, balanced accuracy, the explicit class-order confusion matrix and per-class recall. Derive an abnormal score as `1 - P(normal)` for the supervised classifier. The normal-only latent-surprise detector is a separate experiment and gets its own threshold and ranking metrics.

Sequence-weighted scores answer how well sequences are classified in this particular inventory. Also report source-balanced scores with weights inverse to the number of evaluated sequences per source. Resample complete source recordings for confidence intervals. A paired RF-versus-S-JEPA bootstrap must reuse the same sampled recordings. Report how many bootstrap replicates lacked a class and were skipped. These intervals cannot account for unidentified repeated people across recordings, and tiny two-recording test classes remain fragile.

Normal-only anomaly calibration must use normal validation recordings. The default split contains six such recordings. A distribution-free 5% finite-sample upper quantile then becomes infinity, because fewer than 19 independent calibration observations cannot support that cutoff. The helper exposes this result instead of manufacturing a nominal 5% false-positive guarantee. Ranking AUROC and average precision remain estimable on the test set. A looser prespecified calibration level or a larger independent normal calibration collection changes the question and must be stated. Even with enough recordings, exchangeability is an assumption that GAVD does not establish.

## Executed technical checks

The synthetic tests exercise current-file hashing, duplicate frame rejection, source separation, class coverage, bbox/frame mapping, stale-cache refusal, time-gap rejection, causally safe prefix normalization, feature FPS dependence, train-only imputation, sequence aggregation and paired bootstrap behavior. Their passing results are software evidence, not clinical performance.

A real-media extraction check uses an already downloaded source file and the current full MediaPipe model. It preserves the results under `artifacts/pilot_extraction.json` and `artifacts/pilot_extractions.json`. These are technical checks on a few available clips, not an evaluation cohort. All five first-pass class examples decoded, but every proposed window failed the strict visibility rule. Three further normal clips from newly downloaded recordings also produced no accepted windows, including one with 97.1% visible joint-frame detections. The eight-sequence pilot therefore identified a material limitation in the initial complete-case rule. A 16-joint window can fail because of a few occluded landmarks. The follow-up artifact is `artifacts/pilot_normal_followup.json`.

The final validity-aware preprocessing was then applied to exactly those same cached poses, without new example selection or relaxed visibility thresholds. It accepts 12 windows from six sequences. These include seven normal windows from four recordings, three stroke windows and two myopathic windows. The Parkinsonian and cerebral palsy pilot sequences remain excluded. The recheck is recorded in `artifacts/pilot_validity_recheck.json`. This establishes usable finite-observation inputs for technical integration checks, but it does not establish five-class cohort coverage or classifier performance. Full extraction and a fresh class/split attrition report remain required.

The first cerebral palsy pilot frame was inspected with its scaled bbox. The box encloses a small person in an outdoor poolside scene. A single frame cannot establish gait semantics or a clinical label, and no semantic exclusion was made from this observation. A later five-source contact sheet at `artifacts/visual_review/bbox_contact.png` showed the boxes enclosing the intended visible person in those sampled frames. This is a limited alignment spot check, not a validation of every annotated segment or source label.

## Missingness and camera-view controls

A model can still exploit missingness even when confidence is never an explicit input. Invalid attention positions, an imputed classical feature, or a camera-specific occlusion pattern can identify a source or condition. Report a nuisance-only baseline trained on the same source split using token-valid fractions by joint and temporal patch, plus camera-view one-hot fields when available. Fit all nuisance preprocessing on training sources and choose settings on validation sources. High nuisance-baseline performance weakens a claim that the learned representation captures gait dynamics.

Also compare performance after stratifying by camera view and after restricting both methods to a common quality band. Report how many sources survive each restriction. A temporal-shuffle diagnostic should keep the same missingness geometry when testing motion dependence, or else it changes two factors at once. These controls need their own measured results before a paper can claim robustness to pose failures. The eight-sequence pilot is too small and lacks class/split coverage to provide those results.

## Executed media acquisition

On November 14, 2026 (November 15 UTC), a single bounded download pass attempted every missing recording in the selected cohort. Eighteen selected recordings were already locally decodable. Eighty-one more were downloaded successfully into source-specific folders under `data/videos-gavd`. The resulting inventory contains 99 decodable recordings out of the selected 103, covering 654 annotated sequences and 137,271 annotated frames before pose QC. Decodable means the first video frame opens successfully. It does not imply that every annotated frame is available or that the content/identity has been manually verified.

| Label | Decodable recordings | Sequences with decodable source | Annotated frames with decodable source |
| --- | ---: | ---: | ---: |
| normal | 31 / 32 | 288 / 291 | 40,921 / 41,340 |
| parkinsons | 11 / 11 | 47 / 47 | 10,426 / 10,426 |
| stroke | 18 / 19 | 75 / 76 | 32,930 / 33,934 |
| cerebral palsy | 10 / 11 | 60 / 64 | 19,002 / 20,346 |
| myopathic | 29 / 30 | 184 / 188 | 33,992 / 34,595 |
| **Total** | **99 / 103** | **654 / 666** | **137,271 / 140,641** |

The unavailable sources are `YjRoLtP1di0` (stroke, reported private), `sf5X4YYkWUA` (cerebral palsy, reported private), and `yULxvDc9e8c` (myopathic, unavailable with no more specific established cause). The downloaded `EHx08gSNfxQ` normal file did not decode and remains excluded. Their 12 sequences stay visible in the master annotation inventory. The complete source outcome ledger is `artifacts/media_inventory.csv`, with a copy at `data/videos-gavd/download_report.csv`; it includes timestamps, source hashes, and error evidence. Per-source download logs remain under `artifacts`.

The full extraction runner in `aaai_gait/batch.py` uses four isolated spawn processes, each retaining one ambient pose extractor. It attempts the complete 666-sequence manifest, immediately records the 12 unavailable-source sequences, and verifies or creates pose caches for the other 654. Its incremental report is `artifacts/pose_extraction_report.csv`. It is safe to resume after interruption because caches must match current annotations, video bytes, model weights and extractor settings. The final `artifacts/full_extraction_summary.json` records accepted window counts and class/split coverage; its existence marks completion of the current recorded full-run script, not a diagnosis or a model-performance result.

## Completed extraction and final eligible cohort

The full four-worker extraction completed in 706 seconds on the local machine. It created 652 new pose files, verified two existing files from the worker smoke check, and recorded 12 sequences whose source recording was unavailable or undecodable. No worker or extraction exception occurred. Of the 137,271 annotated frames associated with decodable source files, 137,041 frames decoded successfully. The remaining 230 belonged to one myopathic sequence, `cljwrvc82004b3n6ljbuorcr7`, in source `n93bgWhLZk4`. Its source FPS was also below the target, so it was excluded before window construction.

The final fixed preprocessing rules retain **2,504 windows from 450 annotated sequences and 91 source recordings**. All five labels remain in each split. These windows overlap, so 2,504 must never be reported as an independent participant count. The exact sequence/window index is `artifacts/full_window_index.csv`.

| Label | Train windows / sequences / sources | Validation windows / sequences / sources | Test windows / sequences / sources |
| --- | ---: | ---: | ---: |
| normal | 225 / 79 / 17 | 139 / 39 / 6 | 118 / 17 / 5 |
| parkinsons | 136 / 22 / 7 | 32 / 6 / 2 | 16 / 7 / 1 |
| stroke | 611 / 37 / 10 | 10 / 2 / 1 | 158 / 21 / 4 |
| cerebral palsy | 230 / 33 / 6 | 60 / 14 / 2 | 100 / 9 / 2 |
| myopathic | 380 / 101 / 16 | 105 / 27 / 6 | 184 / 36 / 6 |
| **Total** | **1,582 / 272 / 56** | **346 / 88 / 17** | **576 / 90 / 18** |

There are 216 fully excluded sequences. Twelve have no usable source media, seven have source FPS below 15, 55 are too short to form a 32-frame window at the fixed sampling rate, and 142 have no window satisfying the token-validity gate. A further 135 retained sequences lose some proposed windows to QC. Those 135 are part of the retained 450 and must not be added to the fully excluded count. The exact reasons appear in `artifacts/full_window_exclusions.csv`. The low-FPS sources are `eBvzFkcvScg` (stroke, 10 FPS) and `n93bgWhLZk4` (myopathic, about 14.886 FPS). They were excluded by the previously specified no-upsampling policy.

The final corpus identity is:

```text
0e67e2108cff126263c5f9a156e1adb74902ae87e8d8a71214a95b60cb28d683
```

This identity covers pose bytes and preprocessing. The full extraction summary is `artifacts/full_extraction_summary.json`. Neither the counts nor that identity are classifier-performance results.

Missingness remains a measurable nuisance in this retained cohort. The average valid-token fraction is 90.68%, with median 90.625% and minimum 75%, as required by the gate. These are window-level descriptive quantities. The means differ across labels:

| Label | Mean valid-token fraction |
| --- | ---: |
| normal | 87.25% |
| parkinsons | 93.02% |
| stroke | 91.63% |
| cerebral palsy | 91.77% |
| myopathic | 90.77% |

The differences make the missingness-only control essential. They do not establish whether a condition, camera view, body size, image resolution, occlusion, pose-estimator behavior, or another factor caused the difference. QC may preferentially remove difficult views or severe impairments, so retained examples cannot stand in for every original case.

Two limitations remain especially consequential. The test set contains only **one Parkinsonian source recording**, and validation contains only **one stroke source recording**. The split and thresholds were kept fixed, and these small counts are reported rather than hidden by resplitting. A cluster bootstrap cannot recover between-recording Parkinsonian test variability from one recording. Stroke-specific hyperparameter selection is also fragile. The current experiment can support an internal feasibility comparison on this retained collection, while stronger claims need additional independent sources and repeated grouped evaluation or an external dataset. The original-FPS alignment uncertainty described above remains unresolved beyond the limited visual spot checks.

The actual extraction environment is recorded in `artifacts/extraction_environment.json`: Python 3.14.6, NumPy 2.2.6, pandas 2.3.3, OpenCV 4.12.0.88 and MediaPipe 0.10.31 on macOS arm64. This was the repository's existing local environment. The notebook installation instructions separately specify the supported `uv` environment in `pyproject.toml`. Do not describe a recommended interpreter version as the interpreter used for this recorded extraction. The report preserves this distinction so an exact reproduction can match versions rather than assuming that every supported environment yields identical pose bytes.
