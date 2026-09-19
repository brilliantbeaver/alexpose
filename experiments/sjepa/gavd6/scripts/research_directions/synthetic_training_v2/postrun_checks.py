#!/usr/bin/env python3
"""Submit or run CPU diagnostics against verified, immutable source results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import atomic_json, sha256_file

CHECKS = ('all', 'calibration', 'timing', 'curves')
COMPARATORS = ('unchanged', 'initialized', 'coordinate', 'direct', 'paired_jepa')


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def analysis_code():
    paths = [Path(__file__), *sorted((Path(__file__).parent / 'diagnostics').glob('*.py')),
             ROOT / 'slurm/synthetic-training-v2/postrun-checks.sbatch']
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def source_files(source):
    """Only consumed artifacts; mutable Slurm logs are deliberately excluded."""
    paths = [source / name for name in ('identity.json', 'effective-config.json', 'report.md')]
    paths += sorted((source / 'data/bundle').glob('*'))
    paths += sorted((source / 'predictions').glob('*'))
    paths += sorted((source / 'fits').glob('*/training.json'))
    paths += sorted((source / 'evaluation').glob('*.csv'))
    return {str(path.relative_to(source)): sha256_file(path) for path in paths if path.is_file()}


def _records(records, seed):
    return [{**record, 'source_person_id': record['person_id'],
             'person_id': record['canonical_person_id'], 'seed': int(seed)} for record in records]


def _joint_residuals(prediction, inputs, targets, records, method):
    """Per-record/joint signed errors and corrections; no outcome-based selection."""
    import numpy as np
    rows = []
    joints = ('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
              'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
              'right_knee', 'left_ankle', 'right_ankle')
    for index, record in enumerate(records):
        for joint, name in enumerate(joints):
            reference = targets['visible'][index, :, joint] & targets['valid'][index, :, joint]
            finite = np.isfinite(prediction[index, :, joint]).all(-1)
            supported = reference & finite
            observed = inputs['observed'][index, :, joint]
            correction_support = supported & observed
            row = {**record, 'method': method, 'joint': name,
                   'reference_frames': int(reference.sum()), 'scored_frames': int(supported.sum()),
                   'missing_predictions': int((reference & ~finite).sum()),
                   'correction_frames': int(correction_support.sum())}
            for axis, label in enumerate(('x', 'y')):
                error = prediction[index, :, joint, axis] - targets['xy'][index, :, joint, axis]
                correction = prediction[index, :, joint, axis] - inputs['xy'][index, :, joint, axis]
                row[f'bias_{label}_px'] = float(error[supported].mean()) if supported.any() else float('nan')
                row[f'mae_{label}_px'] = float(abs(error[supported]).mean()) if supported.any() else float('nan')
                row[f'correction_{label}_px'] = float(correction[correction_support].mean()) if correction_support.any() else float('nan')
            rows.append(row)
    return rows


def analyze_source(source, output, verification, *, checks='all', ridge=1e-3,
                   tolerance_s=.12, max_windows=4):
    """Analyze a verified source. CLI callers must use check_results first.

    Kept separate from scheduler integration to test the complete numeric path
    with fixture artifacts. Such tests retain their fixture evidence label.
    """
    import numpy as np
    import pandas as pd
    from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig
    from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import TrackBundle
    from gavd6_sjepa.research_directions.synthetic_training_v2.evaluation import evaluate_predictions, aggregate_metrics
    from diagnostics.calibration import fit_calibrations, apply_calibration
    from diagnostics.plots import select_windows, plot_trajectories, inspect_training
    from diagnostics.timing import timing_diagnostics

    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Diagnostic output must be separate from the original source result tree.')
    if checks not in CHECKS or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('Select a known check and a finite positive, fixed ridge penalty.')
    if not np.isfinite(tolerance_s) or tolerance_s <= 0 or not 1 <= max_windows <= 8:
        raise ValueError('Use a finite positive peak tolerance and 1 to 8 plotted windows.')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'analysis.json').exists():
        raise FileExistsError('This diagnostic attempt already started; use a new output directory.')
    before = source_files(source)
    cfg = RunConfig.load(source / 'effective-config.json')
    bundle = TrackBundle.load(source / 'data/bundle')
    bundle.validate(cfg.held_extractor)
    train, dev = bundle.subset('train'), bundle.subset('development')
    train_people = {r['canonical_person_id'] for r in train.records}
    dev_people = {r['canonical_person_id'] for r in dev.records}
    if not train_people or not dev_people or train_people & dev_people:
        raise ValueError('Calibration requires nonempty, disjoint training and development people.')
    selection = select_windows(dev.records, max_windows=max_windows)
    atomic_json(output / 'selected-windows.json', dict(selection=selection,
        policy='Sorted people, one sorted window per person before second windows; metadata only, before calibration/scoring.',
        plot_seed=int(cfg.seeds[0]), numeric_seeds=list(cfg.seeds)))
    code_before = analysis_code()
    atomic_json(output / 'analysis.json', dict(started_utc=now(), source=str(source),
        source_identity=read(source / 'identity.json')['signature'], verification=verification,
        evidence_status=bundle.evidence_status, checks=checks, ridge=ridge,
        peak_tolerance_seconds=tolerance_s, seeds=list(cfg.seeds), selected_windows=selection,
        source_sha256=before, analysis_code_sha256=code_before,
        scope='Post-hoc development diagnostics; no new scientific gate or confirmation claim.'))

    # Training labels are the only labels passed to fitting. Development labels
    # first enter the scoring path after both pooled models have been frozen.
    models = fit_calibrations(train, cfg.held_extractor, ridge=ridge)
    atomic_json(output / 'calibration-models.json', models)
    calibrated = {name: apply_calibration(model, dev.inputs) for name, model in models.items()}
    predictions_dir = output / 'predictions'
    predictions_dir.mkdir()
    for method, prediction in calibrated.items():
        np.savez_compressed(predictions_dir / f'{method}.npz', prediction=prediction)
    metrics, timing, residuals = [], [], []
    plot_predictions = {}
    for seed in cfg.seeds:
        records = _records(dev.records, seed)
        predictions = dict(calibrated)
        for method in COMPARATORS:
            path = source / 'predictions' / f'{method}-{seed}.npz'
            meta = read(path.with_suffix('.json'))
            if meta['records'] != records or meta['method'] != method or meta['seed'] != seed:
                raise ValueError(f'Prediction records are misaligned: {path}')
            with np.load(path, allow_pickle=False) as saved:
                for key, expected in (('input_xy', dev.inputs['xy']), ('target_xy', dev.targets['xy']),
                                      ('timestamps', dev.inputs['timestamps']),
                                      ('target_valid', dev.targets['valid']), ('target_visible', dev.targets['visible']),
                                      ('eval_scale', dev.targets['eval_scale'])):
                    if not np.array_equal(saved[key], expected, equal_nan=True):
                        raise ValueError(f'Prediction reference/input differs from retained bundle: {path}: {key}')
                predictions[method] = saved['prediction'].copy()
        for method, prediction in predictions.items():
            metrics.append(evaluate_predictions(prediction, dev.targets, dev.inputs['timestamps'], records,
                           method=method, evidence_status=bundle.evidence_status))
            residuals.extend(_joint_residuals(prediction, dev.inputs, dev.targets, records, method))
            if checks in ('all', 'timing'):
                timing.append(timing_diagnostics(prediction, dev.targets, dev.inputs['timestamps'], records,
                              method=method, evidence_status=bundle.evidence_status, tolerance_s=tolerance_s))
        if checks in ('all', 'timing'):
            timing.append(timing_diagnostics(dev.targets['xy'], dev.targets, dev.inputs['timestamps'], records,
                          method='reference_oracle', evidence_status=bundle.evidence_status, tolerance_s=tolerance_s))
        if seed == cfg.seeds[0]:
            plot_predictions = predictions
    frame = pd.concat(metrics, ignore_index=True)
    frame.to_csv(output / 'per-window.csv', index=False)
    measures = ('visible_nle', 'lower_limb_nle', 'missing_rate', 'displacement_nle',
                'ankle_separation_mae', 'amplitude_error', 'amplitude_ratio', 'event_timing_mae_s')
    summary = pd.concat([aggregate_metrics(frame, metric) for metric in measures], ignore_index=True)
    summary.to_csv(output / 'per-person-balanced-summary.csv', index=False)
    strata = []
    for variant, group in frame.groupby('variant', sort=True):
        for metric in measures:
            values = aggregate_metrics(group, metric)
            values['variant'] = variant
            strata.append(values)
    pd.concat(strata, ignore_index=True).to_csv(output / 'by-condition.csv', index=False)
    pd.DataFrame(residuals).to_csv(output / 'per-joint-residuals.csv', index=False)
    if timing:
        event_frame = pd.concat(timing, ignore_index=True)
        event_frame.to_csv(output / 'timing-per-window.csv', index=False)
        keys = ['method', 'extractor', 'seed', 'variant', 'reference_status', 'prediction_status']
        event_frame.groupby(keys, dropna=False).size().rename('records').reset_index().to_csv(
            output / 'timing-support.csv', index=False)
        _timing_summary(event_frame).to_csv(output / 'timing-summary.csv', index=False)
    if checks in ('all', 'curves'):
        images = output / 'images'
        images.mkdir()
        plot_trajectories(images, dev.inputs, dev.targets, dev.records,
                          {f'{name}_seed{cfg.seeds[0]}': value for name, value in plot_predictions.items()}, selection)
        inspect_training(source, output)
    if source_files(source) != before or analysis_code() != code_before:
        raise RuntimeError('Source evidence or diagnostic code changed during analysis; this attempt is not certified.')
    _report(output, summary, bundle.evidence_status, checks)
    return dict(status='POSTRUN_CHECKS_COMPLETE', checks=checks, output=str(output),
                report=str(output / 'report.md'), evidence_status=bundle.evidence_status,
                source_unchanged=True, training_people=len(train_people), development_people=len(dev_people))


def _timing_summary(frame):
    """Descriptive event totals, with incomplete predictions in the denominator."""
    import numpy as np
    import pandas as pd
    rows = []
    for keys, group in frame.groupby(['method', 'extractor', 'seed'], sort=True):
        eligible = group.loc[group.reference_eligible]
        matched = int(eligible.matched_timing_denominator.sum())
        reference = int(eligible.reference_event_denominator.sum())
        predicted = int(eligible.prediction_event_denominator.sum())
        complete = int(eligible.complete_prediction_window_denominator.sum())
        timed = eligible.loc[eligible.matched_timing_denominator > 0]
        rows.append(dict(zip(('method', 'extractor', 'seed'), keys)) | dict(
            records=len(group), reference_eligible=len(eligible), reference_ineligible=len(group) - len(eligible),
            original_supported=int(group.original_support.sum()), complete_predictions=complete,
            incomplete_predictions=len(eligible) - complete,
            count_mismatch_records=int(eligible.prediction_status.eq('event_count_mismatch').sum()),
            reference_events=reference, matched_events=matched,
            missed_events=reference - matched if len(eligible) else np.nan,
            known_extra_events=float(eligible.extra_event_count.sum()) if complete else np.nan,
            extra_count_unknown_records=len(eligible) - complete,
            event_recall=matched / reference if reference else np.nan,
            known_event_precision=matched / predicted if predicted else np.nan,
            conditional_timing_mae_s=float((timed.matched_timing_mae_s * timed.matched_timing_denominator).sum() / matched)
                if matched else np.nan))
    return pd.DataFrame(rows)


def _report(output, summary, evidence_status, checks):
    import pandas as pd

    def number(value):
        return 'unsupported' if pd.isna(value) else f'{value:.6f}'

    rows = ['# Synthetic training v2: post-run checks', '', f'Evidence: **{evidence_status}**. Checks: **{checks}**.', '',
        'These are post-hoc development diagnostics. Original predictions, metrics and scientific gates are unchanged.', '',
        'Calibration fits use training people and training extractors only. The same pooled correction is applied to every development extractor. '
        'Joint offsets and ridge affine residuals use input-only window normalization; absent input joints remain missing. '
        'The offset has 24 parameters: one normalized x/y translation per joint. Its pixel correction is constant within each window, '
        'but scales across windows with the input-derived scale. It preserves raw pixel displacement. '
        'The affine residual has 72 parameters and can change motion as well as bias. '
        'The fixed ridge penalty is recorded in `analysis.json`; development labels do not select its value.', '',
        '| Method | Extractor | Seed | Visible normalized error | People |', '| --- | --- | ---: | ---: | ---: |']
    for row in summary.loc[summary.metric.eq('visible_nle')].itertuples():
        rows.append(f'| {row.method} | {row.extractor} | {row.seed} | {row.value:.8f} | {row.people} |')
    rows += ['', 'Full coordinate/motion scores and support: `per-window.csv`, `per-person-balanced-summary.csv`, '
             '`by-condition.csv`. Signed joint errors and corrections: `per-joint-residuals.csv`.', '',
             'Joint bias/MAE are conditional on finite predictions; read their `scored_frames` and `missing_predictions` counts. '
             'The primary coordinate and displacement metrics retain penalties for missing predictions.', '',
             'Calibrations are deterministic and fitted once. Repeated rows across neural training seeds are the same '
             'calibration predictions, not independent calibration fits. Unsupported motion aggregates remain missing; '
             'supported-only averages must not replace the declared population.']
    motion = summary.loc[summary.metric.isin(('displacement_nle', 'amplitude_error', 'amplitude_ratio'))].pivot(
        index=['method', 'extractor', 'seed'], columns='metric', values='value')
    rows += ['', '| Method | Extractor | Seed | 0.20 s displacement error | Amplitude error | Amplitude ratio |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for (method, extractor, seed), values in motion.iterrows():
        rows.append(f'| {method} | {extractor} | {seed} | {number(values.displacement_nle)} | '
                    f'{number(values.amplitude_error)} | {number(values.amplitude_ratio)} |')
    rows += ['', 'Displacement and amplitude errors use reference-box-diagonal units; lower is better. Amplitude ratio is predicted/reference RMS. '
             'A per-window ratio of one matches RMS amplitude; aggregate ratios can conceal attenuation and amplification. '
             'Matching amplitude alone does not establish waveform or timing preservation. An unsupported aggregate is not zero error. '
             'Read `by-condition.csv` to separate clean and corrupted variants before interpreting an overall gain.']
    if checks in ('all', 'timing'):
        rows += ['', '## Timing eligibility and coverage', '',
            '`timing-per-window.csv` separates reference eligibility, original equal-peak-count support, '
            'and exploratory tolerance-matched events. `reference_oracle` uses reference coordinates as predictions '
            'and establishes the attainable support under the original reference rules. `timing-support.csv` counts failure reasons.', '',
            'The added matcher maximizes one-to-one ordered matches within the fixed tolerance and then minimizes absolute time error. '
            'Missed/extra events and coverage accompany matched-event timing error. Missing predictions are failures, '
            'not successful zero errors. Reference-ineligible rows are not scored. These are 2D ankle-separation peaks, not heel strikes.', '',
            'The initial camera is frontal. Check reference amplitude, peak counts and bounding-box scale variation before '
            'interpreting absent timing support as model distortion. Framewise scale normalization can change ankle-separation '
            'amplitude/timing even when a constant per-joint pixel correction preserves displacement.']
        events = pd.read_csv(output / 'timing-summary.csv')
        rows += ['', '| Extractor | Seed | Oracle-eligible / all records | Reference-ineligible |',
                 '| --- | ---: | ---: | ---: |']
        for event in events.loc[events.method.eq('reference_oracle')].itertuples():
            rows.append(f'| {event.extractor} | {event.seed} | {event.reference_eligible} / {event.records} | {event.reference_ineligible} |')
        rows += ['', '| Method | Extractor | Seed | Original support | Incomplete predictions | Count mismatches | Missed peaks | Known extra peaks | Matched / reference peaks | Conditional MAE (s) |',
                 '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for event in events.loc[~events.method.eq('reference_oracle')].itertuples():
            rows.append(f'| {event.method} | {event.extractor} | {event.seed} | {event.original_supported} / {event.reference_eligible} | '
                        f'{event.incomplete_predictions} | {event.count_mismatch_records} | {number(event.missed_events)} | '
                        f'{number(event.known_extra_events)} | {event.matched_events} / {event.reference_events} | {number(event.conditional_timing_mae_s)} |')
        rows += ['', 'Counts are correlated track records/events, not independent people. Missing predictions count all eligible reference events as missed; '
                 'their extra-peak counts are unknown, so known extras/precision are conditional on complete predictions. '
                 'The matched timing MAE is event-pooled, conditional on matched events, and cannot be interpreted without coverage; '
                 'these descriptive event totals are not person-balanced estimates. '
                 'Original support uses equal peak counts with no tolerance; tolerance-based matching is a separate post-hoc diagnostic. '
                 'Per-condition reasons, amplitude and scale variation are retained in `timing-per-window.csv`.']
    if checks in ('all', 'curves'):
        rows += ['', '## Trajectories and fitting history', '',
            'Plots in `images/` use the metadata-only windows in `selected-windows.json`, with clean/corrupted conditions '
            'and extractors in separate files. Plots use the first declared seed; numeric tables cover every seed. '
            'Training summaries and histories are in `training-summary.csv` and `training-history.csv`. '
            'Loss scales differ across objectives and phases; curves alone do not certify convergence or motion fidelity.']
    rows += ['', 'All source hashes and analysis settings are retained in `analysis.json`. '
             'These diagnostics do not establish anatomical accuracy, clinical validity, causality of a calibration mechanism, '
             'or a benefit of JEPA. A simple control matching a neural gain is evidence that this gain need not require temporal pretraining.', '']
    (output / 'report.md').write_text('\n'.join(rows))


def submit(work, *, checks, ridge, tolerance_s, max_windows):
    from haic import state_for
    state = state_for(work)
    if not state.get('source_config'):
        raise ValueError('No configured source run exists; finish the automated experiment first.')
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:6]
    output = work / 'diagnostics' / attempt
    output.mkdir(parents=True, exist_ok=False)
    request = dict(work=str(work), output=str(output), checks=checks, ridge=ridge,
                   analysis_code_sha256=analysis_code(),
                   tolerance_s=tolerance_s, max_windows=max_windows, created_utc=now())
    atomic_json(output / 'request.json', request)
    command = ['sbatch', '--parsable', '--no-requeue', '--export=ALL', '--nodes=1', '--ntasks=1',
               '--cpus-per-task=4', '--mem=16G', '--time=00:30:00',
               f'--account={state["account"]}', f'--partition={state["partition"]}',
               f'--chdir={ROOT}', f'--job-name=stv2-checks-{attempt[-6:]}',
               f'--output={output}/slurm-%j.out', str(ROOT / 'slurm/synthetic-training-v2/postrun-checks.sbatch'),
               str(output / 'request.json')]
    atomic_json(output / 'submission.json', dict(status='submitting', command=command,
                request_sha256=sha256_file(output / 'request.json')))
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('SBATCH_') and key not in {'SLURM_CLUSTERS', 'SLURM_HINT'}}
    env.update(STV2_ROOT=str(ROOT), STV2_PYTHON=sys.executable, STV2_WORK=str(work))
    try:
        value = subprocess.check_output(command, env=env, text=True).strip().split(';')[0]
        if not value.isdigit():
            raise RuntimeError(f'Uncertain sbatch response {value!r}; inspect {output / "submission.json"} before resubmitting.')
    except Exception as error:
        atomic_json(output / 'submission.json', dict(status='submission_uncertain', command=command,
                    request_sha256=sha256_file(output / 'request.json'), error=str(error)))
        raise
    atomic_json(output / 'submission.json', dict(status='submitted', job_id=value, command=command,
                request_sha256=sha256_file(output / 'request.json')))
    print(f'CPU diagnostics job: {value}\nOutput: {output}\nLog: {output}/slurm-{value}.out\n'
          'No GPU requested. Original experiment results remain unchanged.')
    return output


def run_request(path):
    from check_results import check_results
    path = Path(path).resolve(strict=True)
    request = read(path)
    work, output = Path(request['work']).resolve(), path.parent
    if output != Path(request['output']).resolve() or not output.is_relative_to(work / 'diagnostics'):
        raise ValueError('Request output must lie in this run\'s diagnostics directory.')
    submission = read(output / 'submission.json')
    if submission['request_sha256'] != sha256_file(path):
        raise ValueError('Diagnostic request changed after submission.')
    atomic_json(output / 'status.json', dict(status='running', started_utc=now()))
    try:
        if request['analysis_code_sha256'] != analysis_code():
            raise ValueError('Diagnostic code changed after submission; submit a new attempt after this job ends.')
        verified = check_results(work)
        if request['analysis_code_sha256'] != analysis_code():
            raise ValueError('Diagnostic code changed while verifying source results.')
        result = analyze_source(Path(verified['results']), output, verified,
                    checks=request['checks'], ridge=request['ridge'],
                    tolerance_s=request['tolerance_s'], max_windows=request['max_windows'])
        if request['analysis_code_sha256'] != analysis_code():
            raise ValueError('Diagnostic code changed during execution; this attempt is not certified.')
    except Exception as error:
        atomic_json(output / 'status.json', dict(status='failed', error=f'{type(error).__name__}: {error}', updated_utc=now()))
        raise
    atomic_json(output / 'status.json', dict(result, updated_utc=now()))
    print(json.dumps(result, indent=2))


def status(work):
    attempts = sorted((work / 'diagnostics').glob('*/request.json'))
    if not attempts:
        print('No post-run diagnostics have been submitted.')
        return
    output = attempts[-1].parent
    print(f'Latest diagnostics: {output}')
    saved = output / 'status.json'
    print(json.dumps(read(saved), indent=2) if saved.exists() else 'Analysis has not started.')
    if not (output / 'submission.json').exists():
        print('Submission is being recorded; repeat status shortly.')
        return
    submission = read(output / 'submission.json')
    print('Submission:', submission['status'])
    if 'job_id' in submission:
        accounting = subprocess.check_output(['sacct', '-n', '-P', '-j', submission['job_id'],
                        '--format=JobIDRaw,State,ExitCode,Elapsed,AllocTRES'], text=True)
        print(accounting.strip() or 'Scheduler accounting has not appeared yet.')
        for line in accounting.splitlines():
            fields = line.strip().split('|')
            if len(fields) < 3 or fields[0] != submission['job_id']:
                continue
            state = fields[1].split()[0].rstrip('+')
            if state in {'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'CANCELLED', 'NODE_FAIL',
                         'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}:
                print(f'DIAGNOSTICS_FAILED: scheduler reports {state} ({fields[2]}). '
                      'Any saved running status is stale; inspect this attempt\'s log.')
            elif state == 'COMPLETED' and (not saved.exists() or read(saved).get('status') != 'POSTRUN_CHECKS_COMPLETE'):
                print('DIAGNOSTICS_INCOMPLETE: scheduler ended but no successful analysis status was saved.')
        log = output / f'slurm-{submission["job_id"]}.out'
        print('Log:', log)
        if log.exists():
            from collections import deque
            with log.open() as stream:
                print(''.join(deque(stream, maxlen=20)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('submit', 'run', 'status'))
    parser.add_argument('--work', type=Path, default=os.environ.get('STV2_WORK'))
    parser.add_argument('--request', type=Path)
    parser.add_argument('--checks', choices=CHECKS, default='all')
    parser.add_argument('--ridge', type=float, default=1e-3)
    parser.add_argument('--tolerance-s', type=float, default=.12)
    parser.add_argument('--max-windows', type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == 'run':
        if args.request is None:
            parser.error('run requires --request.')
        run_request(args.request)
    else:
        if args.work is None:
            parser.error('Source the saved session.env or supply --work.')
        work = args.work.expanduser().resolve(strict=True)
        if args.command == 'status':
            status(work)
        else:
            import math
            if not math.isfinite(args.ridge) or args.ridge <= 0 or not math.isfinite(args.tolerance_s) or args.tolerance_s <= 0:
                parser.error('Ridge and event tolerance must be finite and positive.')
            if not 1 <= args.max_windows <= 8:
                parser.error('Select 1 to 8 windows.')
            submit(work, checks=args.checks, ridge=args.ridge, tolerance_s=args.tolerance_s, max_windows=args.max_windows)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, AssertionError, subprocess.SubprocessError) as error:
        print(f'POSTRUN_CHECKS: {type(error).__name__}: {error}', file=sys.stderr)
        raise SystemExit(1)
