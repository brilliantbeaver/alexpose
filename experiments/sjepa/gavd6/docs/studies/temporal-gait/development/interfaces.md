# Frozen implementation boundary — 2026-09-15

Status: local implementation, not a real-data result. The coordinator owns config,
data, workflow, registry, and integration. Other owners must not edit those files.

## Data interface

`windows.WindowDataset` is a dataclass containing `arrays: dict[str,np.ndarray]`
and `records: list[dict]`, with `__len__`, `subset(indices)`, `save(path)`,
`load(path)`. No pickle or object arrays. Each row has:

- `context`: float32 [N,64,33,2], prefix-normalized full-frame detector pixels;
- `context_valid`: bool [N,64,33]; unobserved values are zero;
- `context_times`: float32 [N,64,33], actual times relative to issue;
- `context_age`: float32 [N,64,33], zero for unobserved;
- `query_times`: float32 [N,64], [-2.56,…,-.04];
- `future`: float32 [N,H,2,33,2], separate teacher interval in the PREFIX origin/scale;
- `future_valid`: bool [N,H,2,33];
- `future_times`: float64 [N,H,2,33], original matched PTS inside the strict teacher interval;
- `endpoint`: float32 [N,H,33,2], original-time nearest-observed query;
- `endpoint_valid`: bool [N,H,33];
- `endpoint_times`: float64 [N,H,33], absolute matched PTS (NaN if absent);
- `origin`: float64 [N,2]; `scale`: float64 [N];
- `scale_valid`: bool [N]. SSL eligibility does not require scale/endpoint validity.

Records have `window_id`, `video_id`, `sequence_id`, `group_id`, `role`,
`issue_time`, `bout_start`, `bout_end`, `mode`, `scale_reason`, audited
`historical_laterality_member` and separately labeled prefix measurement diagnostics.
The matching `index_*` context arrays are the common-window index-resize control.
Context only is a dict of tensors with context/context_valid/context_times/
context_age/query_times. Model APIs must never accept the dataset or target dict.
Fixed horizon queries come from config. Missing future information is not input.

## Configuration fields

`RunConfig`: mode (`real`/`synthetic`), run_root, video_manifest,
sequence_manifest, pose_manifest, identity_manifest, exposure_manifest,
reservation_manifest, split_manifest (explicit role per video); device,
seeds, pilot_seeds, updates, checkpoint_updates, batch_size, hidden_dim,
encoder_depth, predictor_depth, heads, patch_size, clock_channels,
learning_rate, weight_decay, ema_start, ema_end, mask_fraction,
prefix_seconds, grid_hz, horizons, target_interval_seconds,
endpoint_tolerance_seconds, max_observation_age_seconds, window_stride_seconds,
bootstrap_samples, ridge_alphas, precision, min_relative_improvement,
arms, schema_version. Root may add fields needed for safety.
`from_env(config_path=None)` loads only explicit TG_CONFIG; contradictory TG_*
overrides fail. `root`, `to_dict`, `scientific_dict`, `validate(check_input_paths=False)`.
Final config adds explicit cohort scope, objective/augmentation/schedule controls
and runtime-only `resume_from`; the current dataclass is the executable schema.
Synthetic configs explicitly opt into tiny fixtures; never a fallback.

## Model/training owner

`models.JEPA(cfg)`: `.encode(context, state='online'|'teacher'|'initialized')`
returns [B,D]; `.predict_future(context, state='online'|'teacher'|'initialized')`
returns [B,H,D]. Must document exact readouts. Training may expose richer private
helpers. `training.train_condition(cfg, task, dataset, output_dir)` returns a
JSON-compatible checkpoint record. `training.load_model(cfg, checkpoint_path)`.
Task keys: `task_id`, `arm` (masked_index/masked/future/future_wrong_source/
future_wrong_time), `seed`, `fold` (0 for fixed source-held-out protocol).
Checkpoints preserve initialization, online, teacher, optimizer, RNG and draws.
Training uses source→bout→window sampling. Prepared-feature masking is explicitly
not raw-observation withholding; hidden feature tokens must not enter student.

## Evaluation owner

`evaluation.evaluate_condition(cfg, task, train_dataset, eval_dataset,
checkpoint_path, output_dir)` returns JSON-compatible artifact paths/status.
Fits all decoders/direct predictors on matched train sources only. Emits per-window
per-horizon per-method predictions and metadata sufficient to replay intervals.
`evaluation.compare_conditions(cfg, records, output_dir)` selects on development
only, validates complete grid; no test selection. Owner can add a baseline-only
entry point for E0. All-invalid examples remain explicit, not zero-error.

## Notebook/HPC boundary

`workflow.run_stage(cfg, stage, task_id=None, role='development', phase='pilot') -> dict`.
Stage names: inventory, prepare, audit, masked, future, extensions, cache-video,
evaluate, calibrate, test, aggregate. `workflow.plan_tasks(cfg, phase)` -> immutable
list of task rows; phases pilot/develop/confirm. Dry-run CLI validates configs and
prints task/stage plans without opening manifests/media, training or submission.
E3/video are explicit gated extensions, not silently executed or declared ready.

Canonical notebooks are output-free, synthetic execution requires explicit
TG_MODE=synthetic configuration. No real training merely by rendering a notebook.

## Acceptance

Owners report changed files, evidence, executed tests, failures and assumptions.
Use unittest with gavd6/.venv/bin/python and PYTHONPATH=src. Tests must exercise
arithmetic, mutation boundaries, gradients, partitions and incompatible resumes,
not simply restate constants. No HAIC path discovery, no raw-data downloads.
