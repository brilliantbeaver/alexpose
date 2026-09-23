"""One-way declaration and evaluation of previously unexposed AMASS people.

Preparation and metric evaluation are separate from fitting. A declaration binds
all planned original-test people, frozen fitted checkpoints, training calibration
and metric settings before the loader opens their motion arrays.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os
import tempfile

from .common import atomic_json, read_json, sha256, utc_now, digest


def lock_confirmation(cfg, output, *, reviewed_by, evidence, exposure_ledger=None):
    from .cohort import load_cohort
    from .scheduler import _verify_frozen, verify_completed
    from .training import load_model
    output = Path(output)
    if output.exists():
        raise FileExistsError('Confirmation declarations are immutable; do not overwrite an opened test protocol')
    if cfg.get('fixture') or not str(reviewed_by).strip() or not str(evidence).strip():
        raise ValueError('Source confirmation needs a named exposure reviewer and supporting evidence')
    _verify_frozen(cfg)
    work = Path(cfg['work']); plan = read_json(work / 'plan.json'); ledger = read_json(work / 'ledger.json')
    if not (work / 'evaluation/summary.json').is_file() or read_json(work / 'evaluation/summary.json').get('status') != 'DEVELOPMENT_MATRIX_COMPLETE':
        raise ValueError('Complete and review development evaluation before declaring confirmation')
    cohort = read_json(cfg['cohort']['plan_path'])
    rows = [r for r in cohort['records'] if r['split'] == 'confirmation']
    if not rows or (not exposure_ledger and any(r['exposure'] != 'unexposed_verified' or r['reserved'] is not False for r in rows)):
        raise PermissionError('All predeclared original-test people require verified unexposed history and explicit nonreservation')
    fits, hashes = [], {}
    for phase in plan['phases']:
        if phase['phase'] == 'pretrain':
            continue
        verify_completed(ledger['completed'][phase['phase_id']])
        result = ledger['completed'][phase['phase_id']]['result']
        checkpoint = str(Path(result['checkpoint']).resolve())
        if sha256(checkpoint) != result['checkpoint_sha256']:
            raise RuntimeError('A fitted checkpoint changed before confirmation declaration')
        _, payload = load_model(checkpoint)
        if payload['signature'].get('evidence_status') not in {'technical-source-screen', 'automated-source-screen', 'audited-source'}:
            raise PermissionError('Confirmation requires an explicitly source-trained checkpoint')
        hashes[checkpoint] = result['checkpoint_sha256']
        fits.append(dict(method=phase['recipe']['recipe_id'], seed=phase['seed'], checkpoint=checkpoint))
    calibration = work / 'evaluation/calibration.json'
    lock = dict(schema='gf-confirmation-lock-v1', created_utc=utc_now(), reviewed_by=str(reviewed_by), evidence=str(evidence),
                cohort_identity=cohort['identity'], cohort_files=cohort['files'],
                person_ids=sorted({r['canonical_person_id'] for r in rows}),
                checkpoint_hashes=hashes, fits=fits, seeds=list(cfg['seeds']),
                measurement=deepcopy(cfg['measurement']), evaluation=deepcopy(cfg['evaluation']),
                model=deepcopy(cfg['model']), calibration=str(calibration.resolve()), calibration_sha256=sha256(calibration),
                config_sha256=sha256(work / 'config.json'), plan_sha256=sha256(work / 'plan.json'),
                claim_scope='Synthetic image-plane motion restoration on unexposed people; no clinical validity claim')
    if exposure_ledger is not None:
        lock['exposure_ledger'] = str(Path(exposure_ledger).resolve())
        lock['exposure_ledger_sha256'] = sha256(exposure_ledger)
    # Verify the identical lock contract used by preparation, without opening
    # raw test arrays; remove an unpublished lock if this validation fails.
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, lock)
    check = deepcopy(cfg); check['cohort']['confirmation_lock'] = str(output)
    try:
        load_cohort(check, partition='confirmation', require_available=False)
    except Exception:
        output.unlink()
        raise
    return dict(status='CONFIRMATION_PROTOCOL_LOCKED', lock=str(output), sha256=sha256(output), people=len(lock['person_ids']), fits=len(fits))


def _evaluate_confirmation(cfg, bundle_path, lock_path, output):
    from .cohort import load_cohort
    from .data import load_dataset
    from .evaluation import summarize_predictions, write_baseline_predictions
    from .training import predict
    from .scheduler import _verify_frozen
    _verify_frozen(cfg)
    lock_path = Path(lock_path); lock = read_json(lock_path)
    if lock.get('schema') != 'gf-confirmation-lock-v1':
        raise ValueError('Unknown confirmation declaration')
    output = Path(output)
    work = Path(cfg['work'])
    for path, expected in [(work / 'config.json', lock['config_sha256']), (work / 'plan.json', lock['plan_sha256']),
                           (Path(lock['calibration']), lock['calibration_sha256'])]:
        if sha256(path) != expected:
            raise RuntimeError('Frozen configuration, phase plan or training calibration changed')
    for checkpoint, expected in lock['checkpoint_hashes'].items():
        if sha256(checkpoint) != expected:
            raise RuntimeError('Frozen confirmation checkpoint changed')
    local = deepcopy(cfg); local['cohort']['confirmation_lock'] = str(lock_path)
    local['measurement'] = lock['measurement']; local['evaluation'] = lock['evaluation']; local['seeds'] = lock['seeds']
    expected, _, identity = load_cohort(local, partition='confirmation', require_available=False)
    data = load_dataset(bundle_path, allow_confirmation=True)
    if data.evidence_status not in {'technical-source-screen', 'audited-source'}:
        raise PermissionError('Source confirmation cannot use fixture or automated-development evidence')
    if {r['split'] for r in data.records} != {'confirmation'} or not data.provenance.get('confirmation_admitted'):
        raise PermissionError('Evaluation requires separately admitted confirmation records only')
    expected_families = set(expected.source_family_id)
    actual_families = {r['source_family_id'] for r in data.records}
    if not actual_families.issubset(expected_families) or {r['canonical_person_id'] for r in data.records} - set(lock['person_ids']):
        raise ValueError('Confirmation records do not belong to the locked population')
    if data.provenance.get('cohort_identity') != identity:
        raise ValueError('Prepared confirmation bundle differs from the frozen cohort')
    if data.provenance.get('confirmation_lock_sha256') != sha256(lock_path):
        raise ValueError('Prepared confirmation bundle is not bound to this declaration')
    output.mkdir(parents=True, exist_ok=True)
    calibration = read_json(lock['calibration'])
    def exports():
        for fit in lock['fits']:
            path = output / f"{fit['method']}-seed-{fit['seed']}.npy"
            predictions = predict(data, fit['checkpoint'], local, output=path)
            yield fit['method'], fit['seed'], predictions
        for name, prediction in write_baseline_predictions(data, calibration, output).items():
            for seed in lock['seeds']:
                yield name, seed, prediction
    person, coverage, comparisons = summarize_predictions(data, exports(), local, output)
    for comparison in comparisons.values():
        comparison['evaluation_split']='confirmation'
        comparison['lock_sha256']=sha256(lock_path)
        if comparison.get('status')=='descriptive_development_estimate':
            comparison['status']='descriptive_locked_confirmation_estimate'
    atomic_json(output/'comparisons.json',comparisons)
    # Evidence reflects the actual retained cohort. Missing/failed source families
    # cannot be concealed by complete-looking conditional metrics.
    planned_people = set(lock['person_ids'])
    retained_people = set(person.canonical_person_id)
    missing_families = sorted(expected_families - actual_families)
    complete = not missing_families and retained_people == planned_people
    summary = dict(status='CONFIRMATION_EVALUATED', independent_confirmation=True,
                   evidence_status=data.evidence_status, clinical_validation=False,
                   planned_people=len(planned_people), evaluated_people=len(retained_people),
                   planned_windows=len(expected_families), evaluated_windows=len(actual_families),
                   missing_source_families=missing_families, complete_planned_population=complete,
                   lock_sha256=sha256(lock_path), bundle_manifest_sha256=sha256(Path(bundle_path) / 'manifest.json'),
                   claim_scope=lock['claim_scope'],
                   statistical_interpretation='Paired person/seed intervals on reference-supported measurements; inspect coverage and prespecified effect margins before claiming benefit')
    atomic_json(output / 'summary.json', summary)
    return summary | dict(output=str(output))


def evaluate_confirmation(cfg, bundle_path, lock_path, output):
    """Publish completed evaluation atomically; retain failed staging for diagnosis."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Confirmation output is immutable; preserve the first complete evaluation')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-staging-', dir=output.parent))
    try:
        result = _evaluate_confirmation(cfg, bundle_path, lock_path, staging)
        if output.exists():
            raise FileExistsError('Another evaluator published confirmation output')
        os.rename(staging, output)
        return result | dict(output=str(output))
    except Exception as exc:
        atomic_json(staging/'failure.json', dict(status='unpublished', error=str(exc), destination=str(output)))
        raise
