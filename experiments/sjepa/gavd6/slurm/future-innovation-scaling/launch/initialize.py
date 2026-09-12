"""Automate metadata preparation, synthetic calibration and source reservation.

This launcher freezes its own initialization provenance into each new study.
It never encodes video or writes to the parent.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

CHECKOUT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CHECKOUT / 'src'))
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import (
    read_json, write_json, sha256_file, stage_lock)
from gavd6_sjepa.research_directions.future_innovation.fi_cache_reuse import read_parent
from gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_cohort import (
    freeze, read_study, fingerprint, reserve_sources)
from gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_data import seal, verify_seal
from gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_readiness import (
    development_media, require_development_media, original_inputs)
import pandas as pd


def settings(env):
    checkout = Path(env.get('GAVD6_ROOT', CHECKOUT)).resolve()
    def path(name, default=None):
        value = env.get(name) or default
        if not value:
            raise ValueError(f'Set {name} before running the source learning curve')
        p = Path(value)
        return (checkout / p).resolve() if not p.is_absolute() else p.resolve()
    root, parent = path('FI_RUN_ROOT'), path('FI_PARENT_ROOT')
    if root == parent or root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError('FI_RUN_ROOT must be separate from gate-v2; choose a new sibling output directory')
    full = path('GAVD_FULL_ROOT', checkout / 'data/gavd_full')
    return dict(checkout=checkout, root=root, parent=parent, full=full,
                launcher=checkout / 'slurm/future-innovation-scaling/launch')


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f'Required file is missing: {path}')
    return path


def source_ids(path):
    frame = pd.read_csv(require_file(path), usecols=['video_id'], dtype=str)
    if frame.empty or frame.video_id.isna().any() or frame.video_id.str.strip().eq('').any():
        raise ValueError(f'Exposure inventory has missing recording identities: {path}')
    return set(frame.video_id)


def input_plan(cfg, env):
    """Resolve documented defaults without silently dropping known exposures."""
    directories = [cfg['full'] / 'manifests', cfg['checkout'] / 'manifests/gavd']
    manifest_dir = next((p for p in directories if (p / 'gavd_full_sequences.csv').exists()
                         or (p / 'gavd_full_videos.csv').exists()), directories[0])
    sequences = require_file(manifest_dir / 'gavd_full_sequences.csv')
    videos = require_file(manifest_dir / 'gavd_full_videos.csv')
    bundled = cfg['launcher'] / 'known-exposure.csv'
    provenance = read_json(require_file(cfg['launcher'] / 'known-exposure.json'))
    if sha256_file(require_file(bundled)) != provenance['csv_sha256']:
        raise ValueError('Bundled exposure inventory checksum mismatch')
    exposed = source_ids(bundled)
    if len(exposed) != provenance['recordings']:
        raise ValueError('Bundled exposure inventory count mismatch')
    paths = {bundled.resolve(), require_file(cfg['parent'] / 'manifests/gate-windows.csv')}
    for gate in sorted(cfg['parent'].parent.glob('gate-*')):
        if not gate.is_dir() or gate.resolve() == cfg['root']:
            continue
        manifest = gate / 'manifests/gate-windows.csv'
        if manifest.exists():
            paths.add(manifest.resolve())
        elif (gate / 'config/cohort-contract.json').exists():
            raise FileNotFoundError(f'Historical gate cohort is missing its exposure manifest: {manifest}')
    for value in env.get('FI_INSPECTED_MANIFESTS', '').split(os.pathsep):
        if value:
            paths.add(require_file((cfg['checkout'] / value).resolve()))
    for p in paths:
        exposed.update(source_ids(p))
    participant = env.get('FI_PARTICIPANT_REGISTRY')
    participant = require_file((cfg['checkout'] / participant).resolve()) if participant else None
    return dict(sequences=sequences, videos=videos, exposures=sorted(paths),
                source_ids=sorted(exposed), participants=participant, bundled_provenance=provenance)


def check_study(cfg):
    read_study(cfg['root'], cfg['parent'], require_software=True)
    verify_seal(cfg['root'], 'reports/cohort-audit-complete.json')
    if (cfg['root'] / 'launch/complete.json').exists():
        verify_seal(cfg['root'], 'launch/complete.json')


def check_new_root(cfg):
    if ((cfg['root'] / 'config').exists()
            and not (cfg['root'] / 'config/study.json').exists()
            and not (cfg['root'] / 'launch/inputs.json').exists()):
        raise ValueError('FI_RUN_ROOT already contains another experiment; choose a new sibling directory')


def preflight(cfg, env, mode):
    check_new_root(cfg)
    parent = read_json(require_file(cfg['parent'] / 'config/run-contract.json'))
    if parent.get('protocol') != 'direct-v2':
        raise ValueError('FI_PARENT_ROOT must contain the completed direct-v2 gate, not gate-v1')
    frozen = (cfg['root'] / 'config/study.json').exists()
    staging = cfg['root'] / 'launch/staging' / cfg['root'].name
    recovering = (staging / 'reports/cohort-audit-complete.json').exists()
    if frozen and (cfg['root'] / 'reports/cohort-audit-complete.json').exists():
        check_study(cfg)
    elif mode not in ('all', 'prepare'):
        raise ValueError('No frozen study at FI_RUN_ROOT. Run submit.sh all to initialize it automatically.')
    elif not recovering:
        input_plan(cfg, env)
    required = {'compute': 'data/cohort-complete.json',
                'fit': 'manifests/plan-complete.json', 'report': 'manifests/plan-complete.json'}
    if mode in required:
        verify_seal(cfg['root'], required[mode])
    if mode in ('all', 'prepare'):
        annotation_root = Path(env.get('FI_ANNOTATION_ROOT') or cfg['full'] / 'annotations/GAVD/data')
        annotations = [require_file(annotation_root / f'GAVD_Clinical_Annotations_{i}.csv') for i in range(1, 6)]
        for name in ('FI_POSE_MODEL', 'FI_TEACHER_CHECKPOINT'):
            if not env.get(name):
                raise ValueError(f'Set {name} using the same value as the gate-v2 run')
            require_file(Path(env[name]))
        for name, default in (('FI_VIDEO_ROOT', cfg['full'] / 'youtube/all'), ('VJEPA2_ROOT', None)):
            p = Path(env.get(name) or default) if env.get(name) or default else None
            if p is None or not p.is_dir():
                raise FileNotFoundError(f'Required directory is missing: {name}={p}')
        original_inputs(cfg['parent'], annotations, Path(env['FI_POSE_MODEL']), Path(env['FI_TEACHER_CHECKPOINT']))
        # Preview the exact metadata-only reservation for a new study; do not
        # freeze it here or let local video availability change its assignment.
        bound = cfg['root'] if frozen else staging if recovering else None
        if bound is not None:
            roster = pd.read_csv(bound / 'config/source-reservation.csv', dtype={'video_id': str})
            videos = bound / 'config/full-videos.csv'
        else:
            plan = input_plan(cfg, env)
            sequences = pd.read_csv(plan['sequences'], dtype={'video_id': str, 'sequence_id': str})
            participants = pd.read_csv(plan['participants'], dtype=str) if plan['participants'] else None
            roster = reserve_sources(sequences, plan['source_ids'], participants)
            videos = plan['videos']
        media = development_media(roster, videos, env.get('FI_VIDEO_ROOT') or cfg['full'] / 'youtube/all')
        require_development_media(media)
        print(f'Development media: {len(media)}/{len(media)} recording files found; decoding and pose eligibility still pending.')
        print('Original annotation, pose-model and teacher-checkpoint checksums match gate-v2.')
    print(f'Study: {cfg["root"]}\nParent: {cfg["parent"]}')
    print('Frozen study will be verified and reused.' if frozen else
          'Initialization will calibrate and freeze the source reservation before pose processing.')


def launcher_files(cfg):
    return {p.name: sha256_file(p) for p in sorted(cfg['launcher'].iterdir())
            if p.is_file() and p.suffix in ('.py', '.sh', '.sbatch', '.csv', '.json')}


def capture_inputs(cfg, env):
    directory = cfg['root'] / 'launch'
    receipt = directory / 'inputs.json'
    if receipt.exists():
        record = read_json(receipt)
        if record['launcher'] != launcher_files(cfg) or record['software'] != fingerprint():
            raise ValueError('Implementation changed during initialization; choose a new FI_RUN_ROOT')
        for relative, digest in record['artifacts'].items():
            if sha256_file(require_file(cfg['root'] / relative)) != digest:
                raise ValueError(f'Initialization input changed: {relative}')
        return record
    plan = input_plan(cfg, env)
    inputs = directory / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    for key, name in (('sequences', 'full-sequences.csv'), ('videos', 'full-videos.csv'),
                      ('participants', 'participants.csv')):
        if plan[key]:
            shutil.copyfile(plan[key], inputs / name)
    pd.DataFrame({'video_id': plan['source_ids']}).to_csv(inputs / 'inspected-recordings.csv', index=False)
    software_dir = directory / 'software'
    software_dir.mkdir(exist_ok=True)
    launcher = launcher_files(cfg)
    for name in launcher:
        shutil.copyfile(cfg['launcher'] / name, software_dir / name)
    files = [*inputs.iterdir(), *software_dir.iterdir()]
    record = dict(version='source-curve-launch-v1', launcher=launcher, software=fingerprint(),
                  artifacts={str(p.relative_to(cfg['root'])): sha256_file(p) for p in files},
                  manifest_sources={key: dict(path=str(plan[key]), sha256=sha256_file(plan[key]))
                                    for key in ('sequences', 'videos', 'participants') if plan[key]},
                  exposure_sources=[dict(path=str(p), sha256=sha256_file(p), recordings=len(source_ids(p)))
                                    for p in plan['exposures']],
                  bundled_provenance=plan['bundled_provenance'],
                  participants_supplied=plan['participants'] is not None)
    write_json(receipt, record)
    return record


def calibrate(cfg, runner=subprocess.run):
    directory = cfg['root'] / 'launch/calibration'
    directory.mkdir(parents=True, exist_ok=True)
    for completed in sorted(directory.glob('attempt-*/calibration.json')):
        result = read_json(completed)
        if result.get('passed') is not True or result.get('software') != fingerprint():
            raise ValueError(f'Calibration failed or used different code: {completed}. Choose a new study after fixing the cause.')
        return completed
    # Partial attempts are kept for diagnosis; the deterministic check restarts.
    i = 1
    while (directory / f'attempt-{i:03d}').exists():
        i += 1
    attempt = directory / f'attempt-{i:03d}'
    runner([sys.executable, str(cfg['checkout'] / 'scripts/research_directions/future_innovation/calibrate_source_learning_curve.py'),
            '--output-root', str(attempt)], check=True, cwd=cfg['checkout'])
    result = require_file(attempt / 'calibration.json')
    calibrated = read_json(result)
    if calibrated.get('status') != 'synthetic_calibration_only' or calibrated.get('passed') is not True or calibrated.get('software') != fingerprint():
        raise ValueError(f'Passing calibration under the current implementation is required: {result}')
    return result


def publish(cfg, staging):
    """Publish frozen directories atomically; resume a partially published freeze."""
    read_study(staging, cfg['parent'], require_software=True)
    verify_seal(staging, 'reports/cohort-audit-complete.json')
    # Copy to temporary sibling directories so the complete staging tree remains
    # available across a crash after publishing any one directory.
    for name in ('config', 'manifests', 'reports'):
        source, dest = staging / name, cfg['root'] / name
        if dest.exists():
            expected = {str(p.relative_to(source)): sha256_file(p) for p in source.rglob('*') if p.is_file()}
            actual = {str(p.relative_to(dest)): sha256_file(p) for p in dest.rglob('*') if p.is_file()}
            if expected != actual:
                raise ValueError(f'Conflicting partially initialized directory: {dest}')
        else:
            pending = cfg['root'] / 'launch' / f'publish-{name}'
            shutil.copytree(source, pending, dirs_exist_ok=True)
            pending.rename(dest)
    check_study(cfg)


def initialize(cfg, env, runner=subprocess.run):
    root = cfg['root']
    check_new_root(cfg)
    with stage_lock(root, 'scaling-initialize'), stage_lock(root, 'scaling-freeze'):
        staging = root / 'launch/staging' / root.name
        if ((root / 'config/study.json').exists() and (root / 'reports/cohort-audit-complete.json').exists()
                and ((root / 'launch/complete.json').exists() or not (root / 'config/launcher.json').exists())):
            check_study(cfg)
            print('Reusing the frozen source reservation; no calibration or reassignment needed.')
            return
        read_parent(cfg['parent'])  # Check inherited arrays and audits before expensive calibration.
        record = capture_inputs(cfg, env)
        if not (staging / 'reports/cohort-audit-complete.json').exists():
            calibration = calibrate(cfg, runner)
            if record['launcher'] != launcher_files(cfg) or record['software'] != fingerprint():
                raise ValueError('Implementation changed during calibration; choose a new study')
            # An interrupted freeze is retained; start another complete staging
            # tree with the same basename so its run_id equals the final run_id.
            if staging.exists():
                i = 1
                while staging.with_name(f'{root.name}-interrupted-{i}').exists():
                    i += 1
                staging.rename(staging.with_name(f'{root.name}-interrupted-{i}'))
            (staging / 'config').mkdir(parents=True)
            write_json(staging / 'config/launcher.json', record)
            inputs = root / 'launch/inputs'
            freeze(staging, cfg['parent'], inputs / 'full-sequences.csv', inputs / 'full-videos.csv',
                   [inputs / 'inspected-recordings.csv'],
                   cfg['checkout'] / 'docs/studies/future-innovation/source-learning-curve-protocol.md',
                   calibration, inputs / 'participants.csv' if record['participants_supplied'] else None)
        publish(cfg, staging)
        seal(root, 'launch/complete.json', [root / 'launch/inputs.json',
             *[root / relative for relative in record['artifacts']], root / 'config/launcher.json'])
        print(f'Source reservation frozen: {root / "config/source-reservation.csv"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'check', 'initialize', 'status', 'verify'])
    parser.add_argument('--mode', choices=['all', 'prepare', 'compute', 'fit', 'report'], default='all')
    args = parser.parse_args()
    try:
        cfg = settings(os.environ)
        if args.command in ('preflight', 'check'):
            preflight(cfg, os.environ, args.mode)
        elif args.command == 'initialize':
            initialize(cfg, os.environ)
        elif args.command == 'status':
            if not (cfg['root'] / 'config/study.json').exists():
                print('Study is not initialized yet. Run submit.sh all; initialization is its first CPU job.')
            else:
                from gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_cli import status
                print(json.dumps(status(cfg['root'], cfg['parent']), indent=2))
        else:
            from gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_training import report
            print(json.dumps(report(cfg['root'], cfg['parent'], verify=True), indent=2))
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print(f'Source learning curve: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
