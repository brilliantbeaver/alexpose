# Full-sequence temporal gait experiments: execution specification

**Status: DESIGN CONTRACT — implementation pending.** This document specifies the code, notebook, configuration and HAIC launcher work proposed by [47-improvement-plan.md](47-improvement-plan.md). Commands labelled **AFTER IMPLEMENTATION** are acceptance targets for that work; the `temporal_gait` modules and `temporal-gait` launchers described below do not currently exist. No HAIC data search, transfer, training or Slurm submission was performed for this document.

The experiment uses the full GAVD recordings and all declared walking bouts already stored on HAIC. The operator supplies their exact manifest paths. “Full sequence” means retaining each complete walking bout and its original timeline, then drawing indexed windows throughout that bout. It does not mean putting an entire recording into one transformer call, treating nonwalking video as labelled gait, or admitting held-out sources to self-supervised training.

## 1. Follow the existing execution style, with stronger data contracts

The structural template is the existing [motion-preservation package](../../gavd6/src/gavd6_sjepa/research_directions/motion_preservation/workflow.py), [configuration](../../gavd6/src/gavd6_sjepa/research_directions/motion_preservation/config.py), [notebook runner](../../gavd6/scripts/research_directions/motion_preservation/execute_notebook.py), [Slurm common setup](../../gavd6/slurm/motion-preservation/common.sh) and [submission wrapper](../../gavd6/slurm/motion-preservation/submit.sh). Keep importable numerical code in `src`, orchestration in `workflow.py`, short explanatory notebooks and thin launchers.

Do not copy the motion-preservation GAVD stress adapter unchanged. It defaults to eight sequences and a 3.2-second initial window, treats GAVD as an observational gallery, and can discover reservation files automatically. Those are properties of its existing study, not the requirements of this one. Do not invoke the shared GAVD download job or infer raw-video locations from repository defaults.

Useful existing code to review and selectively reuse:

| Existing implementation | Reuse decision |
| --- | --- |
| `future_prediction/video_pose.py::decode_exact_window` | Its explicit zero-based frame checks are useful for verified constant-frame-rate sources. Its returned average FPS does not establish presentation timestamps for variable-frame-rate media; add a PTS-aware decoder rather than deriving every timestamp from `frame / average_fps`. |
| `source_scaling/cohort.py::source_groups` | Reuse its connected-component logic for explicit participant-to-recording links. No participant ID may be inferred from a recording or walking-bout ID. |
| `source_scaling/cohort.py::reserve_sources` | Reuse the principle that historical exposure and linked sources constrain reservation. Do not import its unrelated source-scaling policy or reassign old confirmation recordings casually. |
| `source_scaling/availability.py` | Reuse frozen exact file resolution, a documented available cohort, and changed-input checks. Replace discovery with supplied `video_path` values. File path/size/mtime are change detectors, not cryptographic proof of video identity. |
| `future_prediction/contracts.py` and `source_scaling/data.py` | Reuse stable IDs, hashes, write-once configuration and stage receipts after checking their signatures and study assumptions. |
| `motion_preservation/execute_notebook.py` | Preserve the launcher's interpreter, source-notebook protection, cell-by-cell saves, failure metadata, kernel cleanup and refusal to overwrite executed notebooks. |
| `latent_laterality/protocol.py` | Reuse explicit arm declarations and uniform-posterior/paired controls when testing a laterality mechanism. Its old scientific gates do not authorize the new study. |

The existing [latent-laterality launcher guidance](../../gavd6/slurm/latent-laterality/README.md) reports that the seed-7 AMASS v2 uniform control reproduced the claimed SG-JEPA gain and confirmation stopped. Preserve that result and its sealed test split. The new workflow must not silently submit that study's jobs 15/16.

## 2. Proposed study layout

```text
gavd6/
  src/gavd6_sjepa/research_directions/temporal_gait/
    __init__.py
    config.py                 typed configuration, resolution and validation
    contracts.py              immutable identities, receipts, role permissions
    manifests.py              exact inputs, exposure and grouped split audit
    video.py                  complete-bout decode, PTS, crop/pose alignment
    windows.py                physical-time windows, masks and source sampler
    preprocessing.py          causal normalization, derivatives, uncertainty
    models.py                 shared encoder/predictor and explicit study arms
    masking.py                target/context masks and leakage assertions
    objectives.py             masked/future objectives and optional dense loss
    video_teacher.py          optional released frozen video-model adapters
    training.py               single-task training, EMA, checkpoints, resume
    information_audit.py      raw/simple/oracle/nuisance diagnostic controls
    evaluation.py             fixed readouts, perturbations, source aggregation
    statistics.py             paired group bootstrap, intervals, multiplicity
    plots.py                  tables/figures from saved predictions
    workflow.py               one function per notebook stage
  scripts/research_directions/temporal_gait/
    execute_notebook.py        same execution contract as motion preservation
    run_stage.py               non-notebook CLI for one declared stage/task
    validate_run.py            inventory/contracts/receipts validation
    build_notebooks.py         reproducible notebook source generation if used
  notebooks/temporal_gait/
    00_inventory_and_split.ipynb
    01_full_bout_timing_and_windows.ipynb
    02_information_and_baseline_audit.ipynb
    03_time_faithful_masked_jepa.ipynb
    04_causal_future_jepa.ipynb
    05_optional_dense_and_video_transfer.ipynb
    06_development_comparison.ipynb
    07_locked_calibration_and_test.ipynb
    08_aggregate_and_claim_audit.ipynb
    README.md
  slurm/temporal-gait/
    common.sh
    submit.sh
    inventory.sbatch
    prepare-bouts.sbatch
    audit-baselines.sbatch
    train-masked.sbatch
    train-future.sbatch
    cache-video-teacher.sbatch
    train-extensions.sbatch
    evaluate-development.sbatch
    calibrate.sbatch
    evaluate-test.sbatch
    aggregate.sbatch
    pilot.example.json
    confirmatory.example.json
    README.md
  tests/temporal_gait/
    test_manifests.py
    test_timing_and_windows.py
    test_causality.py
    test_masking_and_objectives.py
    test_training_resume.py
    test_evaluation.py
    test_launchers_and_notebooks.py
  docs/studies/temporal-gait/
    README.md
    protocol/protocol.md
    execution/haic.md
    results/README.md
```

Register the new study in the existing repository navigation only after implementation. Do not change historical notebook outputs or manuscript claims as a side effect of scaffolding this package.

## 3. Inputs: exact HAIC locations and explicit scientific roles

Require the following paths in JSON or explicit `TG_*` environment variables. Empty, nonexistent, ambiguous or incompatible required inputs stop before media decoding. Do not `glob`, `rglob`, crawl HAIC, search sibling output directories, download videos, guess extensions, or silently substitute the legacy extracted-clip cache.

| Input | Required fields and interpretation |
| --- | --- |
| Video manifest, `TG_VIDEO_MANIFEST` | `video_id`, absolute `video_path`; unique recording IDs and a single chosen full source file per ID. Optional existing content digest and producer metadata are retained. |
| Complete walking-bout manifest, `TG_SEQUENCE_MANIFEST` | `sequence_id`, `video_id`, start/end frame or start/end PTS, declared indexing/base/inclusivity, annotation provenance and walking-bout status. A legacy `first_frame,last_frame` manifest is interpreted as one-based inclusive only when its schema explicitly says so. |
| Existing source reservation, `TG_RESERVATION` | Explicit historical roles and protected sources. The implementation accepts a versioned schema or a documented adapter, not arbitrary interpretation of a `split` column. Existing protected sources remain excluded unless a separately documented protocol legitimately opens them. |
| Historical exposure inventory, `TG_EXPOSURE_MANIFEST` | `video_id`, study/run identifier, exposure type, date or `unknown`; distinguish previously trained-on, used for model selection, manually inspected and test-only evaluated. Unknown exposure is not certified fresh confirmation. |
| Optional participant/duplicate registry, `TG_GROUP_REGISTRY` | Explicit `video_id,participant_id` or verified same-source/duplicate links; provenance for each link. Merge linked recordings before splitting. Without reliable identities, all claims use recording/source-group disjointness. |
| Optional pose manifest, `TG_POSE_MANIFEST` | Exact pose-cache path, `sequence_id`, extractor/version/checkpoint, joint order, coordinate frame, units, source frame IDs/PTS, confidence and observed/missing mask. Existing pose caches are reusable only after timing and provenance checks. |
| Optional labels, `TG_LABEL_MANIFEST` | Endpoint name, value, unit, annotation reference, label quality and permitted use. A gait-type or diagnosis label is not automatically a clinically validated anatomical laterality label. |
| Optional video-model assets | Exact author-code location, commit, checkpoint path/hash and preprocessing/model identifier. Never replace a missing released teacher with a randomly initialized network or synthetic outputs. |

The sequence manifest must enumerate **all supplied walking bouts**, not merely the first annotation interval per video. Record a reconciliation table: declared recordings, declared bouts, eligible bouts, decoded bouts, exclusions by reason, physical minutes and independent source groups. If annotations describe sparse frames rather than complete continuous walking intervals, stop the full-bout claim and require an explicit upstream bout manifest; do not invent continuity across gaps. Maintain separate label-blind SSL eligibility and endpoint-specific evaluation support: inability to compute the laterality score or a particular forecast horizon must not automatically discard an otherwise usable training bout.

Split before fitting preprocessing, selecting windows for evaluation, loading labels for tuning, training, or caching learned teacher targets. Metadata needed to enumerate protected sources may be read for the inventory, but test pixels, test poses, test-derived crops and test labels stay closed until the test stage. Freeze the allocation after inventory, retaining sufficiently supported source groups for training, development and untouched test. A distinct calibration partition is needed only for a declared uncertainty analysis; primary point-forecast evaluation need not consume a calibration set. The historical 93-source laterality cohort is development-exposed and ineligible for fresh confirmation. Keep protected historical confirmation sources as an additional excluded role unless explicitly designated by the approved new protocol. If no untouched groups remain, run development-only source-held-out cross-validation and omit a new-confirmation claim. A numerical train/test percentage must not imply fresh data exist before this audit.

Group assignments are immutable across modalities, windows, seeds and arms. Source availability cannot trigger rerandomization. Report unavailable media against the original roster; changing the actual study cohort requires a new cohort version. Training uses all eligible training-source bouts. Supervised decoders/readouts and direct forecasting baselines use the same permitted training sources and endpoint support. Development selects architecture/loss/hyperparameters/checkpoints; an optional calibration partition serves a separately declared uncertainty analysis only, not exclusive readout training. Test evaluates the locked selection once. Unlabelled test-source pretraining is transductive exposure and is outside the primary inductive protocol.

## 4. Complete-bout timeline and window contract

1. Preserve original source frame IDs, PTS/time base, decoded timestamps, bout boundaries, missing frames, discontinuities and tracking identities. Require strictly increasing timestamps within contiguous segments. Log variable-frame-rate status, dropped frames and decoder failures. Duplicate PTS or unexplained timestamp resets are exclusions, not silently repaired time.
2. Extract or validate pose throughout each complete authorised walking bout. Retain 33-joint observations in storage, with 2D image coordinates as the primary measurement and estimated depth as a separately labelled channel. Keep raw image coordinates and the unmodified pose/mask/confidence arrays alongside model-specific derived features. Do not silently convert a 33-joint stream to the motion-preservation study's 22-joint schema or time-normalize a variable-length bout into 64 points.
3. Use a 2.56-second prefix at 25 Hz, represented by 64 samples, and forecast queries at `b + h` for `h ∈ {0.25, 0.50, 1.00}` seconds; 0.50 seconds is primary. The forecast issue time is `b`; the prefix is half-open `[b - 2.56, b)`. Its 64 bin-left query timestamps are `b - 2.56 + i/25`, `i = 0,…,63`, so the latest model query is `b - 0.04`. Report this 40 ms query-to-issue offset and actual observation ages; do not redefine the horizon relative to the last observed frame. The 1.28-second/32-sample short-prefix analysis and 5.12-second/128-sample extension remain separate conditions on reported common support. Two-sample patches represent 80 ms bins; a four-sample/160 ms patch is a control.
4. Separate the **observable endpoint** from the **latent target interval**. Score observed coordinates at the nearest original frame to `b + h` within a fixed initial tolerance of ±0.020 seconds, breaking an exact tie toward the earlier frame; no qualifying observation means missing endpoint support. Save the actual target timestamp and deviation, but do not give either to the predictor. For the first future-feature recipe, independently encode a two-sample target interval `(b + h - 0.08, b + h]`, with nominal queries `b + h - 0.04` and `b + h`. This is a declared 80 ms target interval ending at each horizon, not an additional target duration or a hidden forecast gap. At the shortest horizon both target queries are strictly after `b`; every selected target observation must remain after `b` and within its query tolerance. Use target-valid masks only in teacher/loss/scoring branches. For the four-sample patch control, preserve the target interval and use explicitly masked structural padding if necessary; do not extend the interval to satisfy tensor shape. A longer target interval is a separately registered ablation, not a silent change to horizon difficulty.
5. Start windows throughout every eligible contiguous bout. Training uses a deterministic source-balanced sampler: draw an eligible recording uniformly, then one of its bouts, then a valid context endpoint, with logged probabilities and a coverage schedule. Duplicate/participant-connected groups constrain split and batch composition; group-uniform sampling would be a different weighting choice and must be labelled separately. Evaluation uses a fixed 0.50-second endpoint stride and explicitly includes an eligible final window so the end is represented. Overlapping windows remain correlated; they are not independent subjects. Save coverage and unrepresented boundary time; record eligible sources and bouts separately for every prefix/horizon. Paired comparisons use identical examples and scored joints, with all-eligible coverage reported separately.
6. For a 25 Hz model grid, use timestamps to select samples with a declared maximum age (initially 0.08 seconds, or two model intervals). A causal grid point uses the latest available observation at or before that point and within the declared prefix; future observations may not fill a context gap. Preserve actual observation timestamps and `age_s`, mark repeated/held samples, and mask stale or missing values. Do not invent 25 Hz measurements from lower-rate video. Keep 12.5/25 Hz sensitivity, with past-only antialiasing and declared delay if filtering is used before downsampling; the filter cannot cross `b`. A fixed-grid baseline must not hide artificial zero velocities caused by repeated observations.
7. In the separate velocity-input ablation, derivatives use true elapsed time: backward velocity `(x_t - x_prev)/(timestamp_t - timestamp_prev)`, with a validity mask, maximum gap and uncertainty. Acceleration uses valid past differences. Centered differences and bidirectional smoothing are prohibited in the causal arm. The initial attribution model keeps positions, observed validity and elapsed time/observation age; adding velocity channels is not bundled into the first preprocessing contrast. Physical-time sinusoidal/continuous positional encodings receive seconds rather than stretched sequence indices.
8. Padding, structural missingness, confidence, observed status and JEPA masking are separate fields. Labels, corruption seeds, target coordinates and target visibility are not student inputs. Every requested horizon has a fixed predictor query even when its future observations are absent; data-dependent future masks may mask loss/scoring after prediction, but may not alter predictor queries, attention masks or horizon embeddings. Statistics such as standardization parameters are fitted on training data only.
9. Save each indexed row with `window_id,video_id,source_group,sequence_id,role,context_start_s,context_end_s,horizon_s,target_query_s,latent_target_start_s,latent_target_end_s,actual_target_timestamp_s,target_deviation_s,frame_index_digest,timestamp_digest,preprocessing_contract_id,eligibility_reason`. Student-facing and target/scoring-only fields are explicitly separated by the loader interface. A window ID includes its timing and input contract. Moving a source, changing a decoder, or changing stride must not silently relabel an existing cache.

“Complete-bout coverage” concerns input retention and systematic sampling. It does not require using every frame equally often during optimization. Report distinct groups, videos, bouts, physical hours and effective training exposure alongside frame/window counts.

## 5. Causality and laterality boundaries

For a forecast issued at `b`, every student input and preprocessing state must be a function of permitted observations at times `< b`, consistent with the half-open prefix. This includes tracked person selection, crop trajectory, pelvis centering, body scale, camera stabilization, visibility/uncertainty estimates, normalization and interpolation. A crop that uses the union of boxes over context plus future, a scale estimated from the full bout, or a smoother using future frames violates the contract even if the transformer itself has a causal attention mask.

Use a fixed prefix-derived origin and scale for each forecast and express all its future predictions/targets in that same frame. The origin may use a declared observed prefix pelvis reference; the primary evaluation scale is the robust prefix body length defined in section 10, with no future-derived fallback. Model-input standardization may use training-only constants but must be inverted consistently for scoring. Crop from current/past boxes with a declared past-only tracker. Preserve the pelvis trajectory as a separate motion channel when root-centering removes it. Running pelvis-relative features may be a separate input stream, but they do not authorize recentering target positions using future pelvis coordinates. For claims of per-token causality, normalization at token `t` may only use information through `t`; a statistic across the whole context is allowed only for the separate “forecast available at prefix end” claim. State which claim each arm makes.

Missing-data interpolation must not read across the forecast boundary. In the strict online arm, interior gap filling also uses past information only. Offline denoising may be evaluated separately with an explicit noncausal label. A precomputed pose or crop cache from a future-aware extractor is ineligible for a strict causal claim unless that future dependence is removed or measured as a separate arm.

For masked JEPA, the EMA teacher may encode the declared full training prefix to generate stopped-gradient targets, consistent with a bidirectional representation-learning objective. This arm has no observations at or after `b`, even though it can attend bidirectionally within the prefix. Student crop/scale/centering and any data-dependent normalization must use only information explicitly available to that student; a masked coordinate must not be reconstructed indirectly from a preprocessing statistic derived from the teacher's complete input. For causal future JEPA, the student receives context only; each future target interval is encoded independently in a separate teacher execution. Do not contextualize prefix and future together and then slice the embeddings. Allowing the teacher to see the future does not authorize future pixels, target statistics, or teacher state to enter student preprocessing. Loss is normalized within each example/horizon and then across its valid horizons; missing targets are recorded without changing the student's fixed query structure. Checkpoint selection and diagnostics use the permitted role only.

For masked objectives, choose per-example ragged masks with a denominator of valid eligible tokens, beginning with realized target fractions 0.50 and 0.75 on development data. Record realized fraction, valid targets, visible context and immediate-neighbor visibility. Enforce minimum visible-context and target counts per example; do not lower the whole batch to the least-observed sample's mask count. Random and structured controls must match realized target count and exposure. Confidence-weighting, teacher target validity and visible-token objectives each need an explicit separate mask contract.

Horizontal reflection, anatomical landmark renaming and unilateral tracking corruption are different transformations. Preserve left/right landmark semantics in the dataset schema. Any reflection augmentation must transform sign-sensitive labels/targets and paired masks consistently. A globally unanchored left/right chart does not supply anatomical side truth. Report gauge-invariant gait properties separately from side-sensitive properties, and reserve anatomical-side claims for independently verified references.

Required counterfactual tests alter all future pixels/poses/boxes/confidence after `b`, then assert bitwise-identical or explicitly tolerance-bounded student inputs and outputs. When perturbing future timestamps, keep the mutated timestamps strictly `> b`; moving future data into the past would test a different input history. Independently test that a frame exactly at `b` is excluded from the half-open prefix. Add a test that changes held-out labels and verifies no training/preprocessing artifacts change. Tests must intercept the decoder/label loader to prove unopened test media remain unopened during development.

## 6. Stages, outputs and decision gates

| Stage / notebook | Workflow entry point and access | Required outputs / gate |
| --- | --- | --- |
| 00 inventory/split | `inventory_and_freeze(cfg)`; supplied manifests and metadata only | `config/resolved.json`, `config/input-contract.json`, `config/source-reservation.csv`, `config/exposure-audit.csv`, `reports/inventory.json`, `receipts/00.json`; incompatible/missing sources reported explicitly. |
| 01 timing/windows | `prepare_bouts(cfg, roles=("train","development"))`; authorised pixels/poses only | Full-bout timing/pose shards, `manifests/bouts.parquet`, `manifests/ssl-windows.parquet`, `manifests/evaluation-support.parquet`, exclusion and coverage reports, alignment samples and receipt. Training/development caches physically separate from later roles. |
| 02 information audit | `audit_information(cfg)`; train/development | Raw kinematic, continuity, persistence/constant-velocity, random encoder, label/nuisance-only diagnostic controls; signed/unsigned target checks; audit of side identifiability, frame order, time stretching and confidence shortcuts. Gate proceeds only if endpoints and timing are measurable. |
| 03 masked training | `train_task(cfg, task_id, family="masked")`; training inputs, development selection | One immutable condition/seed task per process; checkpoints, optimizer/EMA/RNG/sampler state, loss and representation-health traces, compute and receipt. Include reconstructed legacy configuration and time-faithful matched-capacity baseline. |
| 04 causal training | `train_task(cfg, task_id, family="future")`; same frozen training cohort | Same outputs with causal context/target identity, physical horizon and counterfactual causality audit. Compare against persistence and constant-velocity baselines on identical examples. |
| 05 optional extensions | `cache_video_teacher(...)` and `train_task(..., family="extension")`; train/development | Separate frozen **prefix** video-feature comparator branch and dense/deep-supervision arm. First compare pose-only, RGB-only and late fusion with matched readouts; do not begin with video distillation. Explicit released-model, preprocessing and known/unknown pretraining-overlap contracts are mandatory. |
| 06 development | `evaluate_development(cfg)` | Out-of-sample per-window and per-source predictions, paired seed summaries, effective-rank/variance diagnostics, perturbation tests, compute and `decisions/development.json`. A negative result is a valid `stop` decision. |
| 07 lock/optional calibration | `lock_and_calibrate(cfg)`; selected architecture/weights/readout fixed in `locks/selection.json` before reading any calibration data | If an uncertainty analysis is declared, prepare its separate inputs and fit only predeclared uncertainty/calibration parameters. Otherwise record `not_required` without opening a calibration partition. Write `locks/analysis.json` binding checkpoint, trained readout, endpoint, aggregation, configuration and code identities. Do not fit an encoder or new decoder here. |
| 07 test | `evaluate_test(cfg)`; test opens only after a valid lock | Prepare and evaluate test inputs using locked choices; create `test/opened.json` before access, raw predictions, failure/exclusion ledger and `receipts/test.json`. Never recalibrate after seeing results. |
| 08 aggregate | `aggregate_and_audit_claims(cfg)`; saved predictions only | Source-level tables, paired confidence intervals, all-arm seed results, plots, compute ledger, claim-to-artifact mapping and final status. A development-only report must remain clearly development-only. |

Development DAG:

```text
00 freeze → 01 full bouts/timing → 02 information audit
                                      ↓ passing measurement gate
                         ┌────────────┴────────────┐
                      03 masked                04 future
                         └────────────┬────────────┘
                                06 comparison
                                      ↓ selected optional question
                    05 dense OR frozen-video cache → extension fit
                                      ↓
                         06 final development comparison
                                      ↓ frozen selection/analysis
                   07 lock/optional calibration → 07 test → 08 aggregate
```

`pilot` may automate 00→01→02 and seed-42 03/04→06 **only** if a machine-readable gate in stage 02 passes. Its first mechanism comparison applies historical versus time-faithful preparation to identical raw windows on a common legacy-source subset, keeping four-sample patches, coordinate channels, capacity, augmentations, masks and optimization fixed. New window endpoints are not the historical whole-clip R² estimand. Exact historical replay is a separate arm on original selected clips; the clock-channel and two-sample-patch changes in the revised JSON below are subsequent named contrasts. Full-bout indexing remains available; cohort expansion is a subsequent matched comparison, not bundled with that initial contrast. Scheduler success alone is not a scientific gate. `develop` expands promising settings to seeds 42,43,44; `confirm` executes the frozen finalist contrast at seeds 42–46 if feasible. `calibrate` (which also freezes a no-calibration analysis) and `test` are explicit separate commands. No pilot/extension command opens test or treats successful shell execution as scientific success.

## 7. Initial configuration contract

All values below are **proposed revised development settings**, not tuned optima or a substitute for the exact historical reproduction configuration. The main plan governs initial capacity, timing, seed lists and the final arm list. Keep width 96 / encoder depth 4 / predictor depth 2 / four heads for attribution; width 192 is a later capacity experiment. Use a typed versioned schema, reject unknown keys, nonpositive times, impossible masks and inconsistent model dimensions, and save the fully resolved configuration before producing artifacts. The implementation's `pilot.example.json` must contain both a referenced immutable historical recipe and this separate revised recipe; `pilot` first dispatches the controlled historical-recipe contrast.

```json
{
  "schema_version": "temporal-gait-v1",
  "mode": "real",
  "device": "cuda",
  "run_root": "/ABSOLUTE/HAIC/OUTPUT/temporal-gait/pilot-01",
  "video_manifest": "/ABSOLUTE/HAIC/MANIFESTS/full-videos.csv",
  "sequence_manifest": "/ABSOLUTE/HAIC/MANIFESTS/all-walking-bouts.csv",
  "reservation_manifest": "/ABSOLUTE/HAIC/MANIFESTS/existing-source-reservation.csv",
  "exposure_manifest": "/ABSOLUTE/HAIC/MANIFESTS/historical-exposure.csv",
  "pose_manifest": null,
  "group_registry": null,
  "label_manifest": null,
  "data": {
    "selection": "all_eligible_authorized_bouts",
    "source_discovery": false,
    "timestamp_policy": "source_pts",
    "model_fps": 25.0,
    "context_seconds": 2.56,
    "context_samples": 64,
    "prefix_interval": "half_open_left_closed_right_open",
    "query_alignment": "bin_left",
    "query_to_issue_offset_seconds": 0.04,
    "short_context_seconds": 1.28,
    "long_context_extension_seconds": 5.12,
    "forecast_horizon_seconds": [0.25, 0.5, 1.0],
    "primary_horizon_seconds": 0.5,
    "endpoint_timestamp_tolerance_seconds": 0.02,
    "latent_target_interval_seconds": 0.08,
    "latent_target_samples": 2,
    "latent_target_alignment": "ending_at_horizon_query",
    "evaluation_stride_seconds": 0.5,
    "maximum_observation_age_seconds": 0.08,
    "sampler": "source_then_bout_then_endpoint",
    "student_preprocessing": "past_only"
  },
  "model": {
    "stored_pose_joints": 33,
    "coordinate_dimensions": 2,
    "embedding_dim": 96,
    "encoder_layers": 4,
    "attention_heads": 4,
    "predictor_layers": 2,
    "predictor_dim": 96,
    "positional_encoding": "physical_time",
    "patch_samples": 2,
    "patch_control_samples": 4,
    "input_channels": ["coordinates", "observed_mask", "elapsed_seconds", "age_seconds"],
    "confidence_input": false
  },
  "training": {
    "engineering_seeds": [42],
    "development_seeds": [42, 43, 44],
    "final_seeds": [42, 43, 44, 45, 46],
    "checkpoint_updates": [0, 300, 1200, 3600, 10800],
    "pilot_max_updates": 1200,
    "maximum_development_updates": 10800,
    "batch_size": 64,
    "gradient_accumulation_steps": 1,
    "optimizer": "adamw",
    "learning_rate": 0.0003,
    "weight_decay": 0.05,
    "warmup_fraction": 0.05,
    "learning_rate_schedule": "cosine_after_warmup",
    "gradient_clip_norm": 1.0,
    "ema_start": 0.99,
    "ema_end": 0.9999,
    "ema_start_control": 0.999,
    "precision": "bf16_if_supported",
    "masked_target_fraction": 0.5,
    "masked_target_fraction_control": 0.75,
    "mask_fraction_denominator": "per_example_valid_eligible_tokens",
    "baseline_objective": "centered_ce_plus_vicreg",
    "vicreg_multiplier": 0.05,
    "teacher_stop_gradient": true,
    "distributed": false
  },
  "evaluation": {
    "point_estimate_unit": "recording",
    "within_recording_weighting": "equal_bouts_then_equal_windows_within_bout",
    "split_and_bootstrap_unit": "source_duplicate_group",
    "primary_metric": "mean_euclidean_2d_error_prefix_body_length_units",
    "coordinate_system": "original_full_frame_pixels_aspect_ratio_restored",
    "scored_joint_ids": [25, 26, 27, 28, 29, 30, 31, 32],
    "minimum_valid_bilateral_pairs": 3,
    "normalization_scale": "median_prefix_shoulder_hip_knee_ankle_chain_length",
    "minimum_valid_chain_observations_per_side": 8,
    "minimum_scale_pixels": 20.0,
    "minimum_scale_frame_diagonal_fraction": 0.02,
    "bootstrap_repetitions": 2000,
    "open_test": false
  },
  "extensions": {"dense_supervision": false, "frozen_video_teacher": false}
}
```

Use optimizer-update budgets rather than an arbitrary epoch count: save initialization and updates 300,1,200,3,600,10,800, extending only after a development decision and measured pilot resources. Freeze the total LR/EMA schedule horizon before step 1; stopping at a pilot checkpoint may later resume the same prespecified schedule, but a cosine run designed to end at update 1,200 cannot be silently extended to 10,800 as an identical condition. The historical bridge retains its original 1,200 updates, batch 20, constant learning rate and EMA 0.999; the revised schedule above is a separate contrast. For revised training, aim for 32–64 distinct source draws per batch when the cohort permits it. Log effective batch size, actual covariance batch size, optimizer updates, unique source exposure, valid-target/token exposure, parameter count, wall time and GPU hours. Gradient accumulation does not reproduce large-batch VICReg covariance if the statistic is computed per microbatch. Report both exposure-matched and compute-matched comparisons where relevant.

Keep optional confidence and velocity-input channels as explicit ablations, not unexamined defaults. Report pose-estimation quality separately. The initial 25 Hz / 2.56-second / 64-sample setting preserves the small-model comparison while recording physical time; 12.5 Hz, short/long prefixes and width 192 are separate measured conditions. E3 adds visible/context loss and intermediate-layer losses as a factorial study, beginning with layers 2 and 4 and normalized coefficients. Motion auxiliary targets get JEPA-only, auxiliary-only and shuffled-target controls; the old uniform-posterior result is a reason to test mechanism controls, not evidence that a generic extra loss works.

Precedence: explicit `TG_*` overrides → supplied JSON → existing run `config/resolved.json` on resume → documented defaults for nonpath fields only. Existing immutable outputs may be reused only when all scientifically relevant resolved values match. No default supplies a real HAIC data path. A changed CLI/environment override must be visible in a config diff and cannot silently mutate the saved run.

## 8. HAIC launch contract and commands

Use `mind` / `hai` as defaults because the existing motion-preservation scripts use them; expose `TG_ACCOUNT` / `TG_PARTITION` overrides. Confirm resource availability with the operator's HAIC environment during implementation. One H100 per training task is the initial convention. Requesting four GPUs without a tested distributed launch and distributed sampler does not make the implementation distributed.

| Job | Initial scheduler request | Intended work |
| --- | --- | --- |
| Inventory | 4 CPU, 16 GB, 30 min, no GPU | Resolve supplied manifests and freeze contracts. |
| Prepare bouts | 8 CPU, 64 GB, 12 h; GPU only for a GPU pose backend | Decode/alignment/cache; shard by recording so two tasks cannot overwrite one bout cache. |
| Information audit | 8 CPU, 32 GB, 2 h; GPU only if needed for encoder features | Existing cache and simple baselines. |
| Masked/future/extension training | 1 H100, 8 CPU, 48–64 GB, 12 h | One condition/seed per array task. |
| Frozen video teacher cache | 1 H100, 8 CPU, 64 GB, 12 h | Optional expensive encoder pass, independently cached. |
| Development/calibration/test | 1 H100, 8 CPU, 48 GB, 8 h if encoding; CPU job for saved features | Fixed readouts and predictions, with role checks. |
| Aggregate | 8 CPU, 32 GB, 2 h, no GPU | Bootstrap and report from saved arrays. |

These are initial allocation limits, **not measured runtime or cost estimates**. Full video decode and frozen video-model inference may dominate. The first authorised run must measure minutes of footage per GPU-hour, windows per second, memory, cache sizes and per-stage wall time before expanding the arm/seed grid. Do not promise an H100-hour total before that measurement.

**AFTER IMPLEMENTATION — operator supplies exact locations:**

```bash
cd /ABSOLUTE/PATH/TO/gavd6
export GAVD6_ROOT="$PWD"
export TG_PYTHON="$GAVD6_ROOT/.venv/bin/python"
export TG_RUN_ROOT="/ABSOLUTE/HAIC/OUTPUT/temporal-gait/pilot-01"
export TG_CONFIG="$GAVD6_ROOT/slurm/temporal-gait/pilot.example.json"
export TG_VIDEO_MANIFEST="/ABSOLUTE/HAIC/MANIFESTS/full-videos.csv"
export TG_SEQUENCE_MANIFEST="/ABSOLUTE/HAIC/MANIFESTS/all-walking-bouts.csv"
export TG_RESERVATION="/ABSOLUTE/HAIC/MANIFESTS/existing-source-reservation.csv"
export TG_EXPOSURE_MANIFEST="/ABSOLUTE/HAIC/MANIFESTS/historical-exposure.csv"
# Set these only when matching explicit assets exist:
# export TG_POSE_MANIFEST="/ABSOLUTE/HAIC/MANIFESTS/full-bout-poses.csv"
# export TG_GROUP_REGISTRY="/ABSOLUTE/HAIC/MANIFESTS/verified-source-groups.csv"
# export TG_LABEL_MANIFEST="/ABSOLUTE/HAIC/MANIFESTS/validated-endpoints.csv"
```

The implementer must add any actual dependencies to `gavd6/pyproject.toml` and lock them. Document the supported CUDA/PyTorch/interpreter combination; do not change the existing shared environment spec experimentally during a research job. A PTS-aware decoder and released video teacher may require additional pinned packages. An optional extra name is not considered available until implemented.

**AFTER IMPLEMENTATION — inventory and planned development pipeline:**

```bash
"$TG_PYTHON" scripts/research_directions/temporal_gait/execute_notebook.py \
  --notebook 00 --run-root "$TG_RUN_ROOT" --config "$TG_CONFIG" \
  --output-dir "$TG_RUN_ROOT/notebook_runs/inventory-preview"

"$TG_PYTHON" scripts/research_directions/temporal_gait/validate_run.py \
  --run-root "$TG_RUN_ROOT" --phase inventory

bash slurm/temporal-gait/submit.sh pilot --dry-run
bash slurm/temporal-gait/submit.sh pilot
```

The dry run validates flags/configuration and prints shell-escaped `sbatch` commands, complete dependencies, role access and array mappings. It may create only declared output/log directories, matching the existing convention; it must not submit, decode media, hash terabytes of video, fit a model or open test data. Every actual submission logs command arguments, job IDs and dependencies in `logs/submissions.tsv`.

**AFTER IMPLEMENTATION — explicit branches and decisions:**

```bash
# Valid individual stages:
bash slurm/temporal-gait/submit.sh prepare --dry-run
bash slurm/temporal-gait/submit.sh audit --dry-run
bash slurm/temporal-gait/submit.sh masked --dry-run
bash slurm/temporal-gait/submit.sh future --dry-run
bash slurm/temporal-gait/submit.sh evaluate --dry-run

# Optional branch: requires an extension decision and exact released assets.
bash slurm/temporal-gait/submit.sh cache-video --dry-run
bash slurm/temporal-gait/submit.sh extensions --dry-run

# Development replication follows a passing pilot decision.
bash slurm/temporal-gait/submit.sh develop --dry-run
# Final replication and locking follow a selected development comparison.
bash slurm/temporal-gait/submit.sh confirm --dry-run
bash slurm/temporal-gait/submit.sh calibrate --dry-run
bash slurm/temporal-gait/submit.sh test --dry-run
bash slurm/temporal-gait/submit.sh aggregate --dry-run
```

The command names above are requirements for the proposed `submit.sh`; the real help output must match them. Dropping `--dry-run` submits the selected stage only after its prerequisite receipts and scientific decision are verified. `develop` selects seeds 42,43,44 for promising settings; `confirm` selects seeds 42–46 for the frozen finalist comparison, never automatic test access. All lists and candidate task identities must be frozen before the first fit; the immutable task grid includes the later rows, and `pilot` selects only seed-42 rows. Compatible completed tasks may be reused with verified receipts, avoiding retraining the same condition/seed under a new name. Do not edit seed lists or training recipes in place to expand an existing run. If final development checks change the selected arm, create a new documented selection before calibration/test; do not continue calling the abandoned contrast confirmatory.

Use `TG_DEPENDENCY=123456` or `afterok:123456:123457` for an explicit prerequisite. Chained jobs use `--parsable --export=ALL --kill-on-invalid-dep=yes --chdir="$GAVD6_ROOT"`. Evaluation of an array depends on the **entire array**. Validate numeric returned job IDs. Produce `manifests/task-grid.csv` before submission with exact `task_id,condition_id,seed,config_digest,output_dir`; an array task selects one row and never launches the whole grid internally. Log arrays with `%A_%a`.

`common.sh` resolves the checkout/run root, validates the selected interpreter and supplied JSON, sets `PYTHONUNBUFFERED=1`, `MPLBACKEND=Agg` and thread limits no larger than allocated CPUs. It must not export a default mode/device over saved JSON. Notebooks use the chosen interpreter through an isolated temporary kernelspec; do not rely on a stale user Jupyter kernel.

## 9. Resume, provenance and completion

Proposed output contract:

```text
$TG_RUN_ROOT/
  config/                 resolved settings, split/exposure/input/code contracts
  manifests/              complete bouts, windows, task grid and exclusions
  cache/train/            pose/timing/features for training sources
  cache/development/      development inputs and optional teacher targets
  cache/calibration/      created only after selection lock
  cache/test/             created only during explicit locked test evaluation
  models/<condition>/<seed>/
  predictions/<role>/<condition>/<seed>/
  reports/                audit, timing, coverage, representation health, results
  decisions/              measurement and development scientific gates
  locks/                  immutable analysis/calibration/checkpoint identities
  receipts/               completed stage artifacts and hashes
  test/opened.json         records first test access and the governing lock
  notebook_runs/           executed notebook copies, including failed attempts
  logs/                   stdout/stderr, submissions.tsv and resource records
```

Save git commit plus the actual bytes/hash of relevant dirty source, notebook, launcher, config and dependency files; a commit ID alone cannot identify this currently dirty workspace. Hash manifests, protocol, checkpoints, preprocessing definitions and generated caches. Record explicit source file path/size/mtime and any supplied or obtained content digest; do not mislabel stat checks as a SHA-256 verification. Where source hashes are unavailable, document the weaker file identity and preserve hashed decoded frame/timing artifacts. Avoid rehashing the entire video corpus for every checkpoint or array task.

Write a stage receipt only after output validation completes, with input receipt IDs, resolved configuration digest, code/dependency identity, permitted roles, generated artifact paths/hashes, sample/source counts, runtime and status. Partial output files do not establish completion. Use atomic finalization and exclusive task ownership; a killed process must not leave a success receipt.

Resume only if input/config/code identities and sample order agree. Save model, EMA teacher, optimizer, learning-rate schedule, AMP scaler if present, epoch/step, Python/NumPy/Torch CPU/CUDA RNG state and sampler position. Log whether resumption is exact or a documented restart. Changing an endpoint, mask schedule, frame timing, teacher, preprocessing, split or arm requires a new condition/run identity. A source path relocation can be accepted only through a validated relocation record preserving the intended source identity.

Executed notebooks never overwrite source notebooks or previous executed copies. Use a unique attempt directory for a rerun:

```bash
# AFTER IMPLEMENTATION: a second notebook attempt with unchanged science inputs.
export TG_NOTEBOOK_OUTPUT_DIR="$TG_RUN_ROOT/notebook_runs/audit-attempt-02"
bash slurm/temporal-gait/submit.sh audit --dry-run
```

Changing the notebook attempt directory alone does not authorize regeneration of incompatible caches. Read receipts to determine whether a stage is complete, failed, unavailable or scientifically stopped. A failure after test access leaves `test/opened.json` intact; a technical rerun must use the same locked choices and record what was repaired. It cannot become a new opportunity to tune on test.

## 10. Evaluation and handoff acceptance

Train supervised probes/forecast decoders and direct forecasting baselines on the same training-source labels and supported endpoints, and choose hyperparameters/checkpoints on development. Fit scalers inside the appropriate training fold. Optional calibration serves a separately declared uncertainty analysis only; it cannot fit a new decoder or select an encoder.

For the new primary forecast endpoint, restore coordinates to original full-frame pixels with their correct aspect ratio before computing Euclidean distances. Use joint IDs `[25,26,27,28,29,30,31,32]` (knees, ankles, heels and foot-index points in the declared 33-joint schema); a target is scoreable only when at least three of the four bilateral pairs have both joints observed. Compare methods on exactly the same valid pairs/joints. Do not score an unmatched single side to increase coverage.

Normalize every target error by a robust **prefix body length**, not body width: for each valid side and original prefix time, compute `||shoulder−hip|| + ||hip−knee|| + ||knee−ankle||` in full-frame pixels; take the median over those valid side/time observations. Left-chain joint IDs are `[11,23,25,27]`, right-chain IDs `[12,24,26,28]`. Require at least eight valid chain observations at distinct original source times on each side and a scale of at least `max(20 pixels, 0.02 × frame diagonal)`. Held/repeated samples cannot manufacture eight independent observations. The scale uses prefix data only and remains fixed for every horizon from that prefix. Freeze these pilot eligibility thresholds before test, log failures and common-support coverage, and never rewrite the historical laterality score's normalization under this new metric.

Compute each window's mean normalized Euclidean error across its supported target joints; average windows equally within each bout, bouts equally within each recording, then recordings equally. Verified duplicate/source groups remain the split and bootstrap clusters; they do not silently replace recording weights in the point estimate. Group-balanced or duration/window-weighted estimands may be reported as named sensitivity analyses.

Use paired source/duplicate-group bootstraps for comparisons; retain every member recording and every method's predictions within each sampled group, then recompute the declared equal-recording estimator. Report seeds separately and the paired method difference; five optimization seeds are not five independent patient cohorts. The primary endpoint is 0.50-second mean Euclidean 2D forecast error, with ±0.020-second target tolerance and common joint/window support. Report context-feature decoder and predicted-future-feature decoder results separately, recording which distribution trained each decoder. Observed-future features are privileged diagnostics, not deployable forecasts. Include persistence, prefix-fitted robust constant velocity and periodic extrapolation, direct past-pose ridge, small supervised forecasting model, initialized encoder, time-faithful masked JEPA, and within-source/cross-source mismatched future-target controls. Choose the strongest comparator on development and freeze it before test. Report positive and negative results, coverage, exclusions, smallest group counts, missing endpoint support and confidence intervals; no errors on interpolated targets count as observed errors. Preregister one primary endpoint/contrast; mark additional horizon, context, channel, view and severity comparisons exploratory or apply the declared multiplicity procedure.

The implementation is ready for HAIC only when:

- The input adapter accepts explicitly supplied full-video and complete-bout manifests, rejects discovery/fallback, and proves protected sources stay closed.
- Timing tests cover one-based/inclusive annotation conversion, variable frame rate, gaps, repeated observations, final bout windows and no time stretching.
- Causal future perturbations cannot change student context; EMA targets have no gradients; target tokens cannot leak through input masks, padding, confidence, normalization or caches.
- Source-group split tests cover multiple bouts per recording, linked/duplicate recordings, optional real participant IDs and unavailable media without reassignment.
- A small clearly labelled synthetic fixture tests shapes, loss execution, resume and stage gates. Its scores never enter real-data results.
- Every notebook runs from a clean kernel under the chosen interpreter, saves on failure, delegates substantive computation to importable code and has consistent links/commands.
- Launchers pass shell syntax checks, a mocked `sbatch` test verifies full-array dependencies and task mapping, and dry runs make no data/model/test access.
- Real HAIC execution remains a separate operator action with exact paths supplied; numerical claims are updated only from completed auditable artifacts.

At handoff, provide the implemented source/tests, both example JSON files, notebook guide, tested commands, expected outputs, measured pilot resources and a claim-to-evidence table. Until then, this document is a specification for those deliverables, not evidence that any new experiment ran.
