"""Read-only, relocated direct-v2 cache inheritance for a separate direct-v3 run.

No teacher import, raw-video access, path rewriting, or writes into the parent.
The embedded cache binding is always checked against the original parent run.
"""
from dataclasses import asdict
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .fi_contracts import (check_run, code_fingerprint, read_json, runtime_versions,
                           sha256_file, stable_key, write_json, write_once_json)
from .fi_cohort import validate_cohort
from .fi_feature_cache import cache_binding
from .fi_joint_models import JointModelContract, baseline_schema, skeleton_schema, POLICIES, BINS, CHANNELS
from .fi_readiness import verify_readiness_tables
from .fi_validity_audits import target_variance_checks

PROTOCOL = 'direct-v3'


def checked_file(root, relative, digest):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or sha256_file(path) != digest:
        raise ValueError(f'Artifact checksum/path mismatch: {relative}')
    return path


def snapshot(root):
    root = Path(root).resolve()
    return {str(p.relative_to(root)): {'sha256': sha256_file(p), 'mtime_ns': p.stat().st_mtime_ns,
                                      'size': p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file()}


def verify_snapshot(root, saved):
    if snapshot(root) != saved:
        raise ValueError('Parent artifacts changed (contents, modification time or inventory)')


def read_parent(root):
    """Verify the original contract and each available inherited evidence item."""
    root = Path(root).resolve()
    run = read_json(root / 'config/run-contract.json')
    if run.get('protocol') != 'direct-v2':
        raise ValueError('Cache inheritance requires a direct-v2 parent')
    for name, digest in run['config_sha256'].items(): checked_file(root, f'config/{name}', digest)
    from .fi_contracts import FRAME, DIRECT_ARMS, VJEPA_COMMIT
    if read_json(root/'config/frame-contract.json') != asdict(FRAME):
        raise ValueError('Incompatible inherited temporal/crop/token boundary')
    controls = read_json(root/'config/control-contract.json')
    if tuple(controls['arms']) != DIRECT_ARMS or controls['shuffle_block'] != 4 or controls['shuffle_seed'] != 260905 or controls['audit_count'] != 3:
        raise ValueError('Incompatible inherited controls')
    teacher = read_json(root/'config/teacher-contract.json')
    if (teacher['commit'] != VJEPA_COMMIT or teacher['builder'] != 'vjepa2_1_vit_base_384'
            or teacher['embedding_dim'] != 768 or teacher['target_is_full_clip_contextual'] is not True
            or teacher['synthetic'] is not run['synthetic']):
        raise ValueError('Incompatible inherited teacher contract')
    cohort_contract = read_json(root / 'config/cohort-contract.json')
    checked_file(root, 'config/run-contract.json', cohort_contract['run_contract_sha256'])
    checked_file(root, 'manifests/gate-windows.csv', cohort_contract['manifest_sha256'])
    if run['manifest_sha256'] != cohort_contract['manifest_sha256']:
        raise ValueError('Parent cohort/run identity mismatch')
    cohort = pd.read_csv(root / 'manifests/gate-windows.csv')
    validate_cohort(cohort, run)
    contract = read_json(root / 'config/cache-contract.json')
    binding = cache_binding(root)
    if contract['binding'] != binding: raise ValueError('Parent cache binding mismatch')
    checked_file(root, 'manifests/cache-index.csv', contract['index_sha256'])
    checked_file(root, 'config/nuisance-schema.json', contract['schema_sha256'])
    projection = read_json(root / 'config/projection-contract.json')
    pp = checked_file(root, 'config/projection-256.npy', projection['sha256'])
    p = np.load(pp, allow_pickle=False)
    if p.shape != (768, 256) or not np.isfinite(p).all(): raise ValueError('Invalid inherited projection')
    names, _ = baseline_schema(read_json(root / 'config/nuisance-schema.json'))
    index = pd.read_csv(root / 'manifests/cache-index.csv')
    if (not index.window_id.is_unique or set(index.window_id) != set(cohort.window_id)
            or {p.stem for p in (root / 'teacher-cache').glob('*.npz')} != set(cohort.window_id)):
        raise ValueError('Duplicated, missing or unexpected parent cache window')
    index = index.set_index('window_id')
    arrays = {name: [] for name in ('baseline', 'person', 'background', 'skeleton', 'matching')}
    for window in cohort.window_id:
        row = index.loc[window]
        path = checked_file(root, f'teacher-cache/{window}.npz', row.sha256)
        receipt = read_json(path.with_suffix('.json'))
        if (Path(row.path).name != path.name or Path(row.path).parent.name != 'teacher-cache'
                or row.binding != binding or receipt['window_id'] != window
                or receipt['sha256'] != row.sha256 or receipt['binding'] != binding
                or not np.isfinite(receipt['geometry_max_abs']) or receipt['geometry_max_abs'] < 0):
            raise ValueError('Parent cache receipt identity/schema mismatch')
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {*arrays, 'window_id', 'binding'} or str(payload['window_id']) != window or str(payload['binding']) != binding:
                raise ValueError('Parent payload schema/binding mismatch')
            for name in arrays:
                v = payload[name]
                shape = {'baseline': (len(names),), 'person': (256,), 'background': (256,), 'skeleton': (32, 33, 4)}.get(name)
                if (not np.issubdtype(v.dtype, np.number) or not np.isfinite(v).all()
                        or (shape is not None and v.shape != shape) or (name == 'matching' and (v.ndim != 1 or not len(v) or (not run['synthetic'] and v.shape != (11,))))):
                    raise ValueError(f'Invalid inherited {name} shape or values')
                arrays[name].append(v)
    arrays = {k: np.stack(v) for k, v in arrays.items()}
    readiness = read_json(root / 'qc/readiness-summary.json')
    if (readiness.get('protocol') != 'direct-v2' or readiness['binding'] != stable_key(
            sha256_file(root / 'config/cache-contract.json'), sha256_file(root / 'config/cohort-contract.json'))
            or set(readiness['artifacts']) != {'qc/teacher-stability.csv', 'qc/causal-leakage.csv'}):
        raise ValueError('Inherited readiness binding/files mismatch')
    for name, digest in readiness['artifacts'].items(): checked_file(root, name, digest)
    planned_path = 'manifests/audit-windows.csv'
    checked_file(root, planned_path, cohort_contract['artifacts'][planned_path])
    plan = pd.read_csv(root / planned_path)
    count = read_json(root / 'config/control-contract.json')['audit_count']
    if len(plan) != count or not plan.window_id.is_unique or not set(plan.window_id) <= set(cohort.window_id):
        raise ValueError('Invalid inherited audit plan')
    stable, leakage = verify_readiness_tables(root, plan, count)
    checks = {'data_contract_valid': True, 'input_audit_complete': True,
              'teacher_stable': bool(stable.passed.all()), 'causal_leakage_absent': bool(leakage.passed.all()),
              'target_variance_valid': all(target_variance_checks(cohort, arrays, JointModelContract(), targets=('person',)).values())}
    if readiness['checks'] != checks or readiness['passed'] is not all(checks.values()) or readiness.get('pixel_selectivity_evaluated') is not False:
        raise ValueError('Inherited readiness flags disagree with arithmetic')
    if not all(checks.values()): raise ValueError('Parent retained measurement checks failed')
    return run, cohort, arrays, checks


def initialize_cached_run(root, parent_root, protocol_document, calibration, model=None):
    root, parent = Path(root).resolve(), Path(parent_root).resolve()
    if root == parent or root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError('Child and parent must be separate, nonnested run roots')
    model = model or JointModelContract()
    if (root / 'config/run-contract.json').exists() and (root / 'qc/readiness-summary.json').exists():
        load_reused_cache(root)
        if read_json(root / 'config/parent-lineage.json')['parent_root'] != str(parent):
            raise ValueError('Resume parent differs')
        return
    before = snapshot(parent)
    run, cohort, arrays, checks = read_parent(parent)
    calibration = Path(calibration)
    calibrated = read_json(calibration)
    if calibrated.get('passed') is not True or calibrated.get('model_contract') != asdict_json(model):
        raise ValueError('Passing calibration under this exact model contract is required')
    for d in ('config', 'manifests', 'models', 'predictions', 'qc', 'reports', 'logs'):
        (root / d).mkdir(parents=True, exist_ok=True)
    inherited = ('config/frame-contract.json', 'config/teacher-contract.json', 'config/nuisance-contract.json',
                 'config/control-contract.json', 'config/nuisance-schema.json', 'config/projection-contract.json',
                 'config/projection-256.npy', 'config/thresholds.json', 'config/uv.lock',
                 'manifests/gate-windows.csv', 'manifests/audit-windows.csv',
                 'qc/teacher-stability.csv', 'qc/causal-leakage.csv')
    for relative in inherited:
        destination = root / relative
        if destination.exists() and sha256_file(destination) != sha256_file(parent / relative):
            raise ValueError('Interrupted initialization has incompatible inherited file')
        if not destination.exists(): shutil.copyfile(parent / relative, destination)
    names, kinds = skeleton_schema()
    config = {'model-contract.json': asdict(model),
              'runtime-contract.json': runtime_versions(),
              'preprocessing-contract.json': {'input_version': model.input_version, 'target_version': model.target_version,
                  'policies': POLICIES, 'minimum_observed_sources': 2, 'unsupported_transform': 'exact_zero',
                  'target_scale_floor': 1e-8, 'target_mask': 'training_variance > 1e-10; intersect outer folds'},
              'feature-contract.json': {'version': model.feature_version, 'bins': BINS, 'channels': CHANNELS,
                  'names': names, 'kinds': kinds, 'units': 'prefix-normalized body scale; velocities per frame',
                  'missing': 'coordinates/confidence are NaN without valid observations; velocities require adjacent valid frames within bin'},
              'protocol-contract.json': {'name': PROTOCOL, 'primary': 'joint-ridge-v1', 'development': True,
                  'parent_protocol': 'direct-v2', 'seed_amendment': 'one deterministic fit; stochastic seed criteria inapplicable; all effect/control and paired bootstrap thresholds retained',
                  'candidate_types': ['baseline_only', 'joint_ridge'], 'calibration_sha256': sha256_file(calibration),
                  'protocol_document_sha256': sha256_file(protocol_document)},
              'parent-lineage.json': {'parent_root': str(parent), 'parent_run_id': run['run_id'],
                  'parent_protocol': run['protocol'], 'teacher_evidence': 'reused; no teacher or pixel audit rerun',
                  'raw_alignment_evidence': 'inherited cohort receipts; raw media not required or reopened',
                  'parent_config_sha256': run['config_sha256'],
                  'identities': {name: sha256_file(parent / name) for name in ('config/run-contract.json', 'config/cohort-contract.json', 'config/cache-contract.json', 'config/projection-contract.json', 'config/projection-256.npy', 'qc/readiness-summary.json', 'manifests/cache-index.csv')},
                  'parent_snapshot': before}}
    # Resume an initialization interrupted after some immutable config writes.
    # Runtime/code identity belongs to the first initialization attempt.
    if (root/'config/runtime-contract.json').exists():
        config['runtime-contract.json'] = read_json(root/'config/runtime-contract.json')
    for name, payload in config.items(): write_once_json(root / 'config' / name, payload)
    for source, relative in ((Path(protocol_document), 'config/frozen-protocol.md'), (calibration, 'config/calibration.json')):
        if not (root / relative).exists(): shutil.copyfile(source, root / relative)
    confignames = [p.name for p in sorted((root / 'config').iterdir()) if p.name not in ('run-contract.json','cohort-contract.json','cache-contract.json')]
    child = {**run, 'protocol': PROTOCOL, 'run_id': root.name, 'code_sha256': code_fingerprint(),
             'change_reason': 'Calibrated deterministic development repair after inspected direct-v2 STOP',
             'config_sha256': {name: sha256_file(root / 'config' / name) for name in confignames},
             'development': True, 'cache_reused': True}
    if (root/'config/run-contract.json').exists():
        frozen = read_json(root/'config/run-contract.json')
        child['code_sha256'] = frozen['code_sha256']
    write_once_json(root / 'config/run-contract.json', child)
    write_once_json(root / 'config/cohort-contract.json', {
        'run_contract_sha256': sha256_file(root / 'config/run-contract.json'),
        'manifest_sha256': sha256_file(root / 'manifests/gate-windows.csv'),
        'artifacts': {r: sha256_file(root / r) for r in inherited if r.startswith('manifests/')},
        'evidence': 'parent cohort and alignment reused unchanged'})
    write_once_json(root / 'config/cache-contract.json', {'reuse': 'verified-parent-v1',
        'parent_lineage_sha256': sha256_file(root / 'config/parent-lineage.json'),
        'binding': stable_key(sha256_file(root / 'config/run-contract.json'), sha256_file(root / 'config/parent-lineage.json'))})
    write_once_json(root / 'qc/readiness-summary.json', {'protocol': PROTOCOL, 'passed': True, 'checks': checks,
        'binding': stable_key(sha256_file(root / 'config/cache-contract.json'), sha256_file(root / 'config/cohort-contract.json')),
        'teacher_evidence': 'reused', 'parent_readiness_sha256': sha256_file(parent / 'qc/readiness-summary.json'),
        'artifacts': {p: sha256_file(root / p) for p in ('qc/teacher-stability.csv', 'qc/causal-leakage.csv')}})
    verify_snapshot(parent, before)
    load_reused_cache(root)


def asdict_json(model):
    import json
    return json.loads(json.dumps(asdict(model)))


def load_reused_cache(root):
    root = Path(root).resolve()
    run = read_json(root / 'config/run-contract.json')
    if run['protocol'] != PROTOCOL: raise ValueError('Wrong cached repair protocol')
    for name, digest in run['config_sha256'].items(): checked_file(root, f'config/{name}', digest)
    model = JointModelContract(**read_json(root / 'config/model-contract.json'))
    if not run['synthetic'] and (model != JointModelContract()):
        # JSON arrays deserialize as lists: compare serialized contracts.
        if asdict_json(model) != asdict_json(JointModelContract()): raise ValueError('Real direct-v3 model contract is frozen')
    lineage = read_json(root / 'config/parent-lineage.json')
    parent = Path(lineage['parent_root'])
    verify_snapshot(parent, lineage['parent_snapshot'])
    for p, digest in lineage['identities'].items(): checked_file(parent, p, digest)
    original, cohort, arrays, _ = read_parent(parent)
    if original['run_id'] != lineage['parent_run_id'] or original['synthetic'] is not run['synthetic']:
        raise ValueError('Parent run or synthetic identity changed')
    cohort_contract = read_json(root / 'config/cohort-contract.json')
    checked_file(root, 'config/run-contract.json', cohort_contract['run_contract_sha256'])
    checked_file(root, 'manifests/gate-windows.csv', cohort_contract['manifest_sha256'])
    if sha256_file(root / 'manifests/gate-windows.csv') != sha256_file(parent / 'manifests/gate-windows.csv'):
        raise ValueError('Child cohort differs from parent')
    contract = read_json(root / 'config/cache-contract.json')
    if (contract.get('reuse') != 'verified-parent-v1' or contract['parent_lineage_sha256'] != sha256_file(root / 'config/parent-lineage.json')
            or contract['binding'] != stable_key(sha256_file(root / 'config/run-contract.json'), sha256_file(root / 'config/parent-lineage.json'))):
        raise ValueError('Child cache lineage changed')
    return cohort, arrays


def verify_reused_readiness(root):
    root = Path(root)
    cohort, arrays = load_reused_cache(root)
    summary = read_json(root / 'qc/readiness-summary.json')
    if (summary['protocol'] != PROTOCOL or summary['binding'] != stable_key(sha256_file(root / 'config/cache-contract.json'), sha256_file(root / 'config/cohort-contract.json'))
            or summary['teacher_evidence'] != 'reused' or set(summary['artifacts']) != {'qc/teacher-stability.csv', 'qc/causal-leakage.csv'}):
        raise ValueError('Child readiness binding/schema changed')
    parent = Path(read_json(root / 'config/parent-lineage.json')['parent_root'])
    if summary['parent_readiness_sha256'] != sha256_file(parent / 'qc/readiness-summary.json'):
        raise ValueError('Inherited readiness identity changed')
    for p, digest in summary['artifacts'].items():
        checked_file(root, p, digest)
        if digest != sha256_file(parent / p): raise ValueError('Inherited audit table changed')
    if summary['checks'] != read_json(parent / 'qc/readiness-summary.json')['checks'] or summary['passed'] is not True:
        raise ValueError('Child readiness checks changed')
    return summary
