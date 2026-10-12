"""Protected, outcome-independent AMASS confirmation for the readout repair study.

Planning inspects metadata only. Original test identities never become training or
validation identities; a separate real exposure review and complete checkpoint
inventory are required before any source preparation. Reduced repeated conditions
do not increase the number of independent people.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .common import atomic_json, code_identity, digest, read_json, sha256, utc_now, verify_code

PLAN_SCHEMA = 'gf-repair-cohort-v1'
LOCK_SCHEMA = 'gf-repair-confirmation-lock-v1'
PANEL = dict(movement_levels_deg=[0., 5., 15.], physical_states=['original', 'mirrored'],
             observations=['clear', 'occluded'], naming=['correct', 'global_swap', 'temporary_swap'],
             extractor_families=['vitpose', 'rtmpose'])


def _identity(value):
    return digest({k: v for k, v in value.items() if k != 'identity'})


def _publish(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise FileExistsError(f'Immutable repair artifact differs: {path}')
    else:
        atomic_json(path, value)
    return path


def _fixture(config):
    value = config.get('mode') == 'fixture'
    if 'fixture' in config and bool(config['fixture']) != value:
        raise ValueError('Fixture flag and preparation mode disagree')
    return value


def _check_deadline(config, *, enforce=True):
    value = config.get('repair', {}).get('deadline_utc')
    if not value:
        raise ValueError('Repair confirmation requires an explicit deadline_utc')
    deadline = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if deadline.tzinfo is None:
        raise ValueError('Repair deadline must have an explicit timezone')
    if enforce and not _fixture(config) and datetime.now(timezone.utc) >= deadline:
        raise TimeoutError('Repair confirmation deadline has passed')
    return deadline


def scientific_config(config):
    """Settings bound before test access, excluding only operational locations."""
    value = {key: deepcopy(config.get(key)) for key in
             ('mode', 'fixture', 'seed', 'seeds', 'model', 'training', 'measurement', 'evaluation', 'held_extractor')}
    value['data'] = {k: deepcopy(v) for k, v in config.get('data', {}).items()
                     if k not in {'shard_index', 'num_shards', 'resume_preparation_paths', 'partition'}}
    excluded = {'cohort_plan', 'confirmation_lock', 'parent_binding', 'parent_work',
                'source_work', 'output', 'work', 'readout_binding', 'prediction_receipts'}
    value['repair'] = {k: deepcopy(v) for k, v in config.get('repair', {}).items() if k not in excluded}
    return value


def _parent_plan(config):
    from .cohort import plan_cohort
    if config.get('cohort', {}).get('preset', 'named_walking') != 'named_walking':
        raise ValueError('Slim confirmation requires the named_walking parent cohort')
    fresh = plan_cohort(config)
    path = config.get('cohort', {}).get('plan_path')
    if path and Path(path).is_file():
        saved = read_json(path)
        if saved.get('identity') != fresh['identity']:
            raise ValueError('Parent cohort changed since its frozen plan')
        a = [{k: v for k, v in r.items() if k != 'available'} for r in saved['records']]
        b = [{k: v for k, v in r.items() if k != 'available'} for r in fresh['records']]
        if a != b or saved['rules'] != fresh['rules']:
            raise ValueError('Parent cohort records were tampered with')
    return fresh


def _fixture_rows(config, windows_per_person):
    n = int(config.get('repair', {}).get('fixture_confirmation_people', 3))
    if n < 2:
        raise ValueError('Fixture confirmation needs at least two generated identities')
    samples, hz = int(config['data']['samples']), float(config['data']['hz'])
    rows = []
    for person in range(n):
        name = f'repair-confirmation-fixture-p{person}'
        for window in range(windows_per_person):
            motion = f'{name}/generated-walk-{window}'
            motion_hash = digest(['repair-confirmation-generator-v1', motion])
            start = float(window * samples / hz)
            rows.append(dict(person_id=name, canonical_person_id=name, original_split='test', split='confirmation',
                role='final', relative_path=motion, raw_path='', source_dataset='software_fixture',
                subject_id_candidate=name, motion_id=motion, motion_hash=motion_hash,
                historical_motion_sha256=motion_hash, start_s=start, end_s=start + (samples-1)/hz,
                parent_audited_start_s=start, source_family_id=digest([motion_hash, start, samples, hz])[:24],
                exposure='software_fixture', reserved=False, locked=True, available=True,
                motion_label='analytic_software_fixture', label_source='software_fixture',
                locomotion_status='fixture', audit_reviewer='analytic_generator',
                audit_evidence='software_fixture_only', audit_date='not_human_evidence', review_mode='fixture'))
    return rows


def plan_slim_cohort(parent_config, output=None, *, windows_per_person=2, seed=731):
    """Select distinct raw motions by metadata hash, retaining every test person.

    Insufficient distinct motions fail the plan rather than silently dropping a
    participant. No raw motion contents or model outputs are read here.
    """
    windows_per_person = int(windows_per_person)
    if windows_per_person < 2:
        raise ValueError('At least two distinct-motion windows per person are required')
    config = deepcopy(parent_config)
    fixture = _fixture(config)
    if fixture:
        candidates = _fixture_rows(config, windows_per_person)
        parent_identity = digest(['repair-generated-confirmation-v1', candidates])
        files = {}
    else:
        parent = _parent_plan(config)
        candidates = [deepcopy(r) for r in parent['records'] if r['split'] == 'confirmation']
        parent_identity, files = parent['identity'], parent['files']
    if not candidates:
        raise ValueError('The approved parent has no confirmation candidate people')
    groups = defaultdict(list)
    motion_people = {}
    for row in candidates:
        if row['original_split'] != 'test' or row['split'] != 'confirmation' or not row.get('locked'):
            raise ValueError('Confirmation requires preserved, locked original-test identities')
        person = row['canonical_person_id']
        if motion_people.setdefault(row['motion_hash'], person) != person:
            raise ValueError('Raw content crosses confirmation identities')
        groups[person].append(row)
    selected = []
    for person, rows in sorted(groups.items()):
        motion_groups = defaultdict(list)
        for row in rows:
            motion_groups[row['motion_hash']].append(row)
        if len(motion_groups) < windows_per_person:
            raise ValueError(f'{person} has fewer than {windows_per_person} distinct raw motions; no person was dropped')
        ranked = sorted(motion_groups, key=lambda h: digest([seed, 'motion', person, h]))
        for motion_hash in ranked[:windows_per_person]:
            row = min(motion_groups[motion_hash], key=lambda r: digest([seed, 'window', person, r['source_family_id']]))
            selected.append({k: v for k, v in row.items() if k != 'available'})
    cameras = deepcopy(config.get('data', {}).get('cameras',
        [{'id': 'oblique', 'azimuth_deg': 45.}, {'id': 'side', 'azimuth_deg': 90.}]))
    if len(cameras) != 2 or len({c['id'] for c in cameras}) != 2:
        raise ValueError('Slim panel requires exactly two predeclared cameras')
    specs = config.get('preparation', {}).get('estimators', [])
    if not fixture:
        if any(sum(s['family'] == family for s in specs) != 1 for family in PANEL['extractor_families']):
            raise ValueError('Slim source panel requires exactly one vitpose and one rtmpose estimator')
    panel = dict(deepcopy(PANEL), cameras=cameras, samples=int(config['data']['samples']), hz=float(config['data']['hz']))
    n = len(selected)
    result = dict(schema=PLAN_SCHEMA, fixture=fixture, parent_cohort_identity=parent_identity,
        parent_config=config, authority_files=files, seed=int(seed), windows_per_person=windows_per_person,
        selection='hash_ranked_distinct_raw_motions_then_window; no replacement after reference QC',
        person_ids=sorted(groups), source_family_ids=sorted(r['source_family_id'] for r in selected),
        records=selected, panel=panel,
        summary=dict(planned_people=len(groups), planned_windows=n,
            candidate_windows=len(candidates), source_windows=dict(sorted(Counter(r['source_dataset'] for r in selected).items())),
            source_people={source: len({r['canonical_person_id'] for r in selected if r['source_dataset'] == source})
                           for source in sorted({r['source_dataset'] for r in selected})},
            rendered_clips=n*3*2*2*2, rendered_frames=n*3*2*2*2*panel['samples'],
            extraction_frame_passes=n*3*2*2*2*panel['samples']*2,
            observation_records=n*4*2*2*2*2*3, independent_confirmation=False,
            evidence_boundary='Software fixture only' if fixture else 'Named-walking candidates; exposure and source QC are not yet verified'))
    result['identity'] = _identity(result)
    if output is not None:
        path = Path(output)
        if path.suffix != '.json':
            path = path/'manifest.json'
        _publish(path, result)
    return result


def _read_plan(config):
    path = Path(config['repair']['cohort_plan']).resolve()
    plan = read_json(path)
    if plan.get('schema') != PLAN_SCHEMA or plan.get('identity') != _identity(plan):
        raise ValueError('Slim cohort plan identity changed')
    for source, expected in plan['authority_files'].items():
        if sha256(source) != expected:
            raise ValueError(f'Slim cohort authority changed: {source}')
    fresh = plan_slim_cohort(plan['parent_config'], windows_per_person=plan['windows_per_person'], seed=plan['seed'])
    if fresh != plan:
        raise ValueError('Slim cohort population or selection changed')
    if plan['fixture'] != _fixture(config):
        raise PermissionError('Fixture and source confirmation plans cannot be interchanged')
    if int(config['data']['samples']) != plan['panel']['samples'] or float(config['data']['hz']) != plan['panel']['hz']:
        raise ValueError('Repair sampling clock differs from the frozen source-window plan')
    return path, plan


def _exposure(plan, path):
    from .cohort import _read, _reservations
    if plan['fixture']:
        if path is not None:
            raise ValueError('Software fixture exposure cannot be represented as a human exposure ledger')
        return {p: dict(reserved=False, exposure='software_fixture') for p in plan['person_ids']}
    if path is None:
        raise PermissionError('Source confirmation needs a real separately reviewed exposure ledger')
    table = _read(path, {'person_id', 'canonical_person_id', 'original_split', 'reserved', 'exposure'})
    if set(table.canonical_person_id) != set(plan['person_ids']):
        raise PermissionError('Exposure ledger must cover exactly all planned confirmation people')
    authorities = pd.DataFrame([dict(identity=p, split='test') for p in plan['person_ids']])
    reviewed = _reservations(path, authorities)
    for row in plan['records']:
        if row['exposure'] not in {'unknown', 'unexposed_verified'} or row['reserved'] is True:
            raise PermissionError('Previously recorded exposure or reservation cannot be erased')
    if any(v['exposure'] != 'unexposed_verified' or v['reserved'] is not False for v in reviewed.values()):
        raise PermissionError('Every source confirmation identity needs explicit nonreservation and unexposed_verified history')
    return reviewed


def _fits(checkpoint_fits, fixture):
    from .training import load_model
    from .repair_evaluation import EXPECTED_FITS, METHODS, VARIANTS
    fits, seen = [], set()
    for supplied in checkpoint_fits:
        fit = deepcopy(supplied)
        method, seed = str(fit['method']), int(fit['seed'])
        if (method, seed) in seen:
            raise ValueError('Duplicate method/seed in confirmation checkpoint inventory')
        seen.add((method, seed))
        checkpoint = str(Path(fit['checkpoint']).resolve())
        expected = fit.get('sha256', fit.get('checkpoint_sha256'))
        if not expected or sha256(checkpoint) != expected:
            raise ValueError('Frozen checkpoint hash is missing or changed')
        _, payload = load_model(checkpoint)
        signature = payload.get('signature', {})
        if signature.get('seed') != seed:
            raise ValueError('Declared fit seed differs from the completed checkpoint')
        if not fixture or method in METHODS:
            phase, encoder, policy, objective, variant = (signature.get(k) for k in
                ('phase', 'encoder', 'policy', 'objective', 'representation_variant'))
            actual = None
            if phase == 'end_to_end' and encoder == 'direct' and policy is None and objective == 'base' and variant is None:
                actual = 'P-direct-none-base'
            elif phase == 'readout' and encoder == 'paired_jepa' and policy == 'graph_time' and variant in VARIANTS:
                if signature.get('repair', {}).get('format') == 'gait-fidelity-readout-repair-v1' and objective in {'scalar_low', 'dense_change'}:
                    actual = f'R-repair-{variant}-{objective}'
                elif not signature.get('repair') and objective in {'base', 'paired_change'}:
                    actual = f'F-response-{variant}-graph_time-{objective}'
            if actual != method:
                raise ValueError('Declared method differs from the completed checkpoint objective or architecture')
        evidence = signature.get('evidence_status')
        permitted = {'fixture-tested'} if fixture else {'technical-source-screen', 'automated-source-screen', 'audited-source'}
        if evidence not in permitted:
            raise PermissionError('Checkpoint source/fixture evidence is incompatible with confirmation')
        fit.update(method=method, seed=seed, checkpoint=checkpoint, sha256=expected)
        fits.append(fit)
    if not fits:
        raise ValueError('Confirmation requires a nonempty complete fitted checkpoint inventory')
    if not fixture and seen != EXPECTED_FITS:
        raise ValueError('Source confirmation requires all fixed 27 method/seed fits before any test access')
    return sorted(fits, key=lambda r: (r['method'], r['seed']))


def lock_slim_confirmation(config, checkpoint_fits, *, exposure_ledger, reviewed_by, evidence, output):
    """Freeze people, references, statistical choices, code, and completed fits."""
    _check_deadline(config)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Confirmation locks are immutable; preserve the original declaration')
    if not str(reviewed_by).strip() or not str(evidence).strip():
        raise ValueError('A named reviewer and concrete evidence are required')
    path, plan = _read_plan(config)
    _exposure(plan, exposure_ledger)
    protocol = deepcopy(config.get('repair', {}).get('statistical_protocol'))
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError('Freeze the statistical_protocol before confirmation')
    fits = _fits(checkpoint_fits, plan['fixture'])
    root = str(Path(config.get('code_root', Path(__file__).resolve().parents[4])).resolve())
    code = code_identity(root)
    if not code:
        raise ValueError('Confirmation code identity cannot be empty')
    scientific = scientific_config(config)
    lock = dict(schema=LOCK_SCHEMA, created_utc=utc_now(), fixture=plan['fixture'],
        independent_confirmation=not plan['fixture'], reviewed_by=str(reviewed_by), evidence=str(evidence),
        cohort_plan=str(path), cohort_plan_sha256=sha256(path), cohort_identity=plan['identity'],
        parent_cohort_identity=plan['parent_cohort_identity'], person_ids=plan['person_ids'],
        source_family_ids=plan['source_family_ids'], fit_manifest=fits,
        checkpoint_hashes={f['checkpoint']: f['sha256'] for f in fits},
        statistical_protocol=protocol, scientific_config=scientific,
        config_scientific_identity=digest(scientific), code_root=root, code=code,
        exposure_ledger=str(Path(exposure_ledger).resolve()) if exposure_ledger is not None else None,
        exposure_ledger_sha256=sha256(exposure_ledger) if exposure_ledger is not None else None,
        claim_scope='Software fixture only' if plan['fixture'] else
        'Projected synthetic motion restoration on reviewed unexposed original-test people; no clinical validity claim')
    lock['identity'] = _identity(lock)
    atomic_json(output, lock)
    return dict(status='FIXTURE_CONFIRMATION_PROTOCOL_LOCKED' if plan['fixture'] else 'REPAIR_CONFIRMATION_PROTOCOL_LOCKED',
                lock=str(output), sha256=sha256(output), people=len(plan['person_ids']), fits=len(fits))


def verify_slim_lock(config, lock_path=None, *, enforce_deadline=True):
    """Metadata/weight checks only; called again immediately before source access."""
    _check_deadline(config, enforce=enforce_deadline)
    path, plan = _read_plan(config)
    lock_path = Path(lock_path or config['repair']['confirmation_lock']).resolve()
    lock = read_json(lock_path)
    if lock.get('schema') != LOCK_SCHEMA or lock.get('identity') != _identity(lock):
        raise PermissionError('Repair confirmation declaration identity changed')
    if lock['fixture'] != plan['fixture'] or lock['cohort_identity'] != plan['identity'] or lock['cohort_plan_sha256'] != sha256(path):
        raise PermissionError('Confirmation declaration does not bind this slim cohort')
    if lock['person_ids'] != plan['person_ids'] or lock['source_family_ids'] != plan['source_family_ids']:
        raise PermissionError('Confirmation declaration changed the protected population')
    if scientific_config(config) != lock['scientific_config'] or digest(scientific_config(config)) != lock['config_scientific_identity']:
        raise PermissionError('Scientific configuration changed after confirmation was locked')
    verify_code(lock['code_root'], lock['code'])
    for checkpoint, expected in lock['checkpoint_hashes'].items():
        if sha256(checkpoint) != expected:
            raise PermissionError('Frozen confirmation checkpoint changed')
    checked = _fits(lock['fit_manifest'], plan['fixture'])
    if checked != lock['fit_manifest'] or {f['checkpoint']: f['sha256'] for f in checked} != lock['checkpoint_hashes']:
        raise PermissionError('Confirmation fit identity or checkpoint inventory changed')
    ledger = lock['exposure_ledger']
    if ledger is not None and sha256(ledger) != lock['exposure_ledger_sha256']:
        raise PermissionError('Reviewed confirmation exposure ledger changed')
    _exposure(plan, ledger)
    return lock


def load_slim_cohort(config, *, available_only=True):
    """Verified source rows for the preparation backend; never alters roles."""
    authority = config.get('_repair_lock_config', config)
    lock = verify_slim_lock(authority)
    _, plan = _read_plan(authority)
    reviewed = _exposure(plan, lock['exposure_ledger'])
    rows, excluded = [], []
    for index, original in enumerate(plan['records']):
        if '_repair_shard_index' in config and index % config['_repair_num_shards'] != config['_repair_shard_index']:
            continue
        row = dict(original, **reviewed[original['canonical_person_id']])
        if available_only and not plan['fixture'] and not Path(row['raw_path']).is_file():
            excluded.append(dict(source_family_id=row['source_family_id'], canonical_person_id=row['canonical_person_id'],
                reason='planned_raw_motion_unavailable', raw_path=row['raw_path']))
        else:
            rows.append(row)
    return pd.DataFrame(rows, columns=list(plan['records'][0])), excluded, plan['identity']


def _fixture_bundle(plan, rows, seed):
    """Generate separate held identities directly, with no training-bundle input."""
    from .data import TrackBundle, SWAP, apply_naming
    samples, hz = plan['panel']['samples'], plan['panel']['hz']
    inputs, targets, records = defaultdict(list), defaultdict(list), []
    for row in rows:
        family = row['source_family_id']
        times = row['start_s'] + np.arange(samples)/hz
        individuality = int(digest(row['canonical_person_id'])[:6], 16)/0xffffff
        phase = 2*np.pi*(.9+.3*individuality)*(times-times[0]) + int(digest(family)[:4], 16)/0xffff
        for physical in PANEL['physical_states']:
            for camera in plan['panel']['cameras']:
                flatten = .85 if camera == plan['panel']['cameras'][0] else 1.
                for state, magnitude in [('baseline',0.), ('no_change',0.), ('knee_flexion',5.), ('knee_flexion',15.)]:
                    xy = np.zeros((samples,12,2), np.float32)
                    for side in (0,1):
                        sign = 1 if side == 0 else -1
                        xy[:,side] = [300+sign*35,90]
                        xy[:,2+side] = xy[:,side]+[sign*20,55]
                        xy[:,4+side] = xy[:,2+side]+[sign*10,55]
                        xy[:,6+side] = [300+sign*20,230]
                        hip = .25*np.sin(phase+side*np.pi)
                        knee = .5+(.3+(np.deg2rad(magnitude) if side else 0.))*np.sin(phase+side*np.pi+.5)
                        xy[:,8+side] = xy[:,6+side]+75*np.stack([flatten*np.sin(hip),np.cos(hip)],-1)
                        xy[:,10+side] = xy[:,8+side]+75*np.stack([flatten*np.sin(hip-knee),np.cos(hip-knee)],-1)
                    if physical == 'mirrored':
                        xy = xy[:,SWAP].copy(); xy[...,0] = 600-xy[...,0]
                    for observation in PANEL['observations']:
                        for extractor in PANEL['extractor_families']:
                            extractor_id = 'fixture-source' if extractor == 'vitpose' else f'fixture-{extractor}'
                            rng = np.random.default_rng(int(digest([seed,family,physical,camera['id'],magnitude,observation,extractor])[:8],16))
                            raw = xy+rng.normal(0,1.,xy.shape).astype(np.float32)
                            observed = np.ones((samples,12),bool)
                            if observation == 'occluded':
                                observed[samples//3:samples//2,8:] = False
                            raw[~observed] = np.nan
                            confidence = np.where(observed,.9,np.nan).astype(np.float32)
                            for naming in PANEL['naming']:
                                inp = apply_naming(dict(xy=raw,confidence=confidence,observed=observed,timestamps=times),naming)
                                for key,value in inp.items(): inputs[key].append(value)
                                for key,value in dict(xy=xy,valid=np.ones_like(observed),visible=observed,
                                    eval_scale=np.full(samples,350.,np.float32)).items(): targets[key].append(value)
                                record = dict(row, window_id=family, target_kind='software_fixture',
                                    # The established dataset contract requires
                                    # this enum for every confirmation split.
                                    # Here it means freshly generated identities,
                                    # never a statement about human exposure.
                                    exposure='unexposed_verified', exposure_scope='generated_software_fixture_only',
                                    physical_state=physical,camera_id=camera['id'],movement_state=state,
                                    movement_magnitude=magnitude,magnitude_deg=magnitude,movement_level_deg=magnitude,
                                    naming=naming,observation=observation,extractor=extractor_id,
                                    extractor_family=extractor,student_id=extractor_id,box_source='analytic_fixture',
                                    pair_id=digest([family,physical,camera['id'],observation,naming,extractor])[:24],
                                    endpoint='baseline' if state=='baseline' else 'intervention',held_intervention=magnitude==15.,
                                    variant=f'{physical}-{camera["id"]}-{state}-{magnitude:g}-{observation}-{naming}')
                                records.append(record)
    return TrackBundle({k:np.stack(v) for k,v in inputs.items()}, {k:np.stack(v) for k,v in targets.items()},
        records, 'fixture-tested', dict(hz=hz,samples=samples,fixture=True,confirmation_admitted=True,
                                      evidence_boundary='Generated software identities; no human evidence'))


def prepare_slim(config, output, *, shard_index=0, num_shards=1):
    """Prepare one locked confirmation shard and report every retained/missing unit."""
    lock = verify_slim_lock(config)
    _, plan = _read_plan(config)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Slim preparation output is immutable; choose a new attempt')
    if not 0 <= int(shard_index) < int(num_shards):
        raise ValueError('Invalid slim preparation shard index/count')
    assigned = [r for i,r in enumerate(plan['records']) if i % int(num_shards) == int(shard_index)]
    lock_sha = sha256(config['repair']['confirmation_lock'])
    provenance = dict(repair_confirmation_lock_sha256=lock_sha, repair_cohort_identity=plan['identity'],
        source_selection='repair_confirmation', fixture=plan['fixture'], confirmation_admitted=True,
        shard_index=int(shard_index), num_shards=int(num_shards), frozen_people=plan['person_ids'],
        identity=digest([plan['identity'], lock_sha, num_shards]))
    if plan['fixture']:
        from .data import save_dataset
        output.mkdir(parents=True)
        if assigned:
            data = _fixture_bundle(plan, assigned, int(config.get('seed',731))+100003)
            data.provenance.update(provenance)
            path = save_dataset(data, output/'bundle', storage='npy')
        else:
            path = None
    else:
        from .preparation import prepare
        source = deepcopy(config)
        source['cohort'] = dict(source.get('cohort',{}), confirmation_lock=config['repair']['confirmation_lock'])
        source['data'] = dict(source['data'], source_selection='repair_confirmation',partition='confirmation',
            movement_levels_deg=plan['panel']['movement_levels_deg'],held_level_deg=15.,
            cameras=plan['panel']['cameras'],shard_index=0,num_shards=1)
        source['preparation'] = deepcopy(plan['parent_config']['preparation'])
        source['preparation']['estimators'] = [s for s in source['preparation']['estimators'] if s['family'] in PANEL['extractor_families']]
        # The loader verifies the original scientific configuration, not these
        # implementation-only overrides used by the existing render backend.
        source['_repair_lock_config'] = config
        source['_repair_shard_index'] = int(shard_index)
        source['_repair_num_shards'] = int(num_shards)
        path = prepare(source, output)
        if Path(path).suffix == '.json':
            path = None
    retained_records, exclusions = [], []
    if path is not None:
        meta_path = Path(path)/'manifest.json'
        meta = read_json(meta_path)
        meta['provenance'].update(provenance)
        atomic_json(meta_path,meta)
        retained_records = meta['records']
    coverage_path = output/'coverage.json'
    if coverage_path.is_file(): exclusions = read_json(coverage_path).get('exclusions',[])
    available = {r['source_family_id'] for r in plan['records'] if plan['fixture'] or Path(r['raw_path']).is_file()}
    retained = {r['source_family_id'] for r in retained_records}
    intended = {r['source_family_id'] for r in assigned}
    planned_people = {r['canonical_person_id'] for r in assigned}
    retained_people = {r['canonical_person_id'] for r in retained_records}
    # A source loader filters unavailable windows before its standard sharding;
    # it must not shift another shard's assignments. Check this defensively.
    if not retained.issubset(intended):
        raise ValueError('Prepared family escaped its original immutable shard assignment')
    receipt = dict(schema='gf-repair-confirmation-preparation-v1', status='complete' if path is not None else 'no_work',
        bundle=str(path) if path is not None else None,
        bundle_manifest_sha256=sha256(Path(path)/'manifest.json') if path is not None else None, **provenance,
        planned_people=len(planned_people), retained_people=len(retained_people),
        missing_people=sorted(planned_people-retained_people), planned_windows=len(intended),
        retained_windows=len(retained), missing_source_families=sorted(intended-retained),
        unavailable_source_families=sorted(intended-available), exclusions=exclusions,
        complete_planned_population=retained==intended,
        independent_confirmation=False, reference_qc_only=True,
        note='Preparation does not establish a model effect or increase independent participant count')
    atomic_json(output/'repair-preparation-status.json',receipt)
    return receipt
