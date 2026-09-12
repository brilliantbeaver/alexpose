"""Frozen source reservations and nested source-subset plans for scaling v1.

This module does not encode videos or fit predictors. Participant IDs are accepted
only through an explicit registry; a sequence or recording is never called a person.
"""
from dataclasses import asdict
from pathlib import Path
import math
import shutil

import numpy as np
import pandas as pd

from ..future_innovation.fi_cache_reuse import read_parent, snapshot, verify_snapshot, checked_file
from ..future_innovation.fi_contracts import read_json, write_json, write_once_json, sha256_file, stable_key, runtime_versions
from ..future_innovation.fi_cohort import assign_source_folds
from ..future_innovation.fi_joint_models import JointModelContract

PROTOCOL = 'source-learning-curve-v1'
POLICY = dict(protocol=PROTOCOL, nominal_source_sizes=[40, 80, 160],
              subset_seeds=[261201, 261202, 261203], reservation_seed=261101,
              confirmation_fraction=0.2, outer_folds=5, inner_folds=3,
              penalty_reference_windows=40, bootstrap_repetitions=2000,
              bootstrap_seed=260905, all_endpoint_once=True)


def fingerprint():
    directory = Path(__file__).parent
    checkout=Path(__file__).resolve().parents[4]
    from ..future_innovation.fi_contracts import code_fingerprint
    paths=[*directory.glob('*.py'),
           * (checkout/'slurm/future-innovation-scaling').glob('*.sh'),
           * (checkout/'slurm/future-innovation-scaling').glob('*.sbatch'),
           * (checkout/'scripts/research_directions/future_innovation').glob('*source_learning_curve.py')]
    return {'historical_future_innovation':code_fingerprint(),
            **{str(p.relative_to(checkout)):sha256_file(p) for p in sorted(paths)}}


def source_groups(sources, participants=None):
    """Connected components of explicit participant/source links (many-to-many)."""
    sources = sorted(set(map(str, sources)))
    parent = {s: s for s in sources}
    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    known = {}
    if participants is not None:
        if not {'video_id', 'participant_id'} <= set(participants):
            raise ValueError('Participant registry needs video_id and participant_id')
        for row in participants.fillna('').itertuples(index=False):
            source, person = str(row.video_id), str(row.participant_id).strip()
            if source not in parent:
                raise ValueError('Participant registry contains an unknown source')
            if not person:
                continue
            if person in known:
                a, b = find(source), find(known[person])
                parent[max(a, b)] = min(a, b)
            known[person] = source
    components = {}
    for s in sources:
        components.setdefault(find(s), []).append(s)
    return {s: stable_key('source-component', *component)[:20]
            for component in components.values() for s in component}


def reserve_sources(sequences, exposed, participants=None, policy=None):
    policy = policy or POLICY
    if not sequences.sequence_id.is_unique or sequences[['sequence_id', 'video_id']].isna().any().any():
        raise ValueError('Missing or duplicate sequence identities')
    groups = source_groups(sequences.video_id, participants)
    exposed = set(map(str, exposed))
    exposed_groups = {groups[s] for s in exposed if s in groups}
    unexposed = sorted(set(groups.values()) - exposed_groups,
                       key=lambda g: stable_key('confirmation', g, seed=policy['reservation_seed']))
    reserved = set(unexposed[:math.ceil(len(unexposed) * policy['confirmation_fraction'])])
    development = set(groups.values()) - reserved
    folds = assign_source_folds(development, policy['outer_folds'])
    return pd.DataFrame([dict(video_id=s, source_group=groups[s],
                             previously_exposed=s in exposed,
                             exposure_group=groups[s] in exposed_groups,
                             role='confirmation' if groups[s] in reserved else 'development',
                             outer_fold=-1 if groups[s] in reserved else folds[groups[s]],
                             annotated_sequences=int((sequences.video_id.astype(str) == s).sum()))
                         for s in sorted(groups)])


def make_plan(cohort, roster, policy=None):
    """Same held-out rows at every size; prefixes contain whole source groups."""
    policy = policy or POLICY
    if not cohort.window_id.is_unique:
        raise ValueError('Duplicate eligible windows')
    roles = roster.set_index('video_id')
    if not set(cohort.video_id) <= set(roles.index) or any(roles.loc[v, 'role'] != 'development' for v in cohort.video_id):
        raise ValueError('Confirmation or unregistered source entered development')
    cohort = cohort.drop(columns=['outer_fold', 'source_group'], errors='ignore').merge(
        roster[['video_id', 'source_group', 'outer_fold']], on='video_id', validate='many_to_one')
    fits, entries = {}, []
    for fold in range(policy['outer_folds']):
        held = cohort[cohort.outer_fold == fold]
        pool = cohort[cohort.outer_fold != fold]
        if held.video_id.nunique() < 2 or pool.source_group.nunique() < 6:
            raise ValueError('Insufficient eligible groups for held-out matching / three inner folds')
        groups = {g: sorted(set(rows.video_id)) for g, rows in pool.groupby('source_group')}
        for seed in policy['subset_seeds']:
            order = sorted(groups, key=lambda g: stable_key('training-subset', fold, g, seed=seed))
            for size in [*policy['nominal_source_sizes'], 'all']:
                item = dict(outer_fold=fold, subset_seed=seed, size=str(size), available_sources=int(pool.video_id.nunique()))
                if size != 'all' and pool.video_id.nunique() < size:
                    entries.append({**item, 'status': 'unavailable', 'reason': 'fewer eligible outer-training sources than requested'})
                    continue
                chosen = []
                for group in order:
                    chosen.extend(groups[group])
                    if size != 'all' and len(chosen) >= size:
                        break
                selected = pool[pool.video_id.isin(chosen)]
                identity = stable_key('fit', fold, *sorted(selected.window_id))[:24]
                fit = dict(fit_id=identity, outer_fold=fold, train_windows=sorted(selected.window_id),
                           test_windows=sorted(held.window_id), train_sources=sorted(set(selected.video_id)),
                           test_sources=sorted(set(held.video_id)), train_groups=sorted(set(selected.source_group)),
                           test_groups=sorted(set(held.source_group)), train_count=len(selected))
                if len(fit['train_groups']) < 6:
                    raise ValueError('Source grouping leaves too few groups for inner fitting')
                if identity in fits and fits[identity] != fit:
                    raise ValueError('Conflicting subset identity')
                fits[identity] = fit
                entries.append({**item, 'status': 'planned', 'fit_id': identity,
                                'actual_sources': len(chosen), 'actual_windows': len(selected)})
    return dict(protocol=PROTOCOL, fits=fits, entries=entries,
                common_evaluation_windows=sorted(cohort.window_id),
                common_evaluation_sources=sorted(set(cohort.video_id)))


def freeze(root, parent, sequence_manifest, video_manifest, inspected_manifests,
           protocol_document, calibration, participant_registry=None, video_root=None, resolution_manifest=None):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if root == parent or root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError('Study and parent roots must be separate and nonnested')
    if (root / 'config/study.json').exists():
        read_study(root, parent)
        raise ValueError('Study already frozen; use status/prepare/plan/run to resume')
    calibrated=read_json(calibration)
    if calibrated.get('status')!='synthetic_calibration_only' or calibrated.get('passed') is not True or calibrated.get('software')!=fingerprint():
        raise ValueError('Passing source learning-curve calibration under the current implementation is required')
    parent_before = snapshot(parent)
    parent_run, old, _, _ = read_parent(parent)
    sequences = pd.read_csv(sequence_manifest, dtype={'video_id': str, 'sequence_id': str})
    videos = pd.read_csv(video_manifest, dtype={'video_id': str})
    if not videos.video_id.is_unique or set(sequences.video_id) != set(videos.video_id):
        raise ValueError('Full manifests disagree on source identities')
    candidates = pd.read_csv(parent / 'manifests/candidates.csv')
    candidate_contract = read_json(parent / 'config/candidates-contract.json')
    checked_file(parent, 'manifests/candidates.csv', candidate_contract['manifest_sha256'])
    if not set(candidates.sequence_id) <= set(sequences.sequence_id):
        raise ValueError('Expanded manifest does not include parent candidates')
    if not inspected_manifests:
        raise ValueError('An explicit historical exposure inventory is required')
    exposed = set(old.video_id)
    inputs = {'full-sequences.csv': Path(sequence_manifest), 'full-videos.csv': Path(video_manifest)}
    inputs['calibration.json']=Path(calibration)
    for i, path in enumerate(inspected_manifests):
        frame = pd.read_csv(path, usecols=['video_id'], dtype=str)
        if frame.video_id.isna().any():
            raise ValueError('Missing source identity in exposure registry')
        exposed.update(frame.video_id)
        inputs[f'exposure-{i}.csv'] = Path(path)
    participants = None if participant_registry is None else pd.read_csv(participant_registry, dtype=str)
    if participants is not None:
        inputs['participants.csv'] = Path(participant_registry)
    roster = reserve_sources(sequences, exposed, participants)
    for d in ('config', 'manifests', 'reports', 'logs', 'models', 'predictions', 'qc', 'boxes', 'poses', 'frames', 'teacher-cache'):
        (root / d).mkdir(parents=True, exist_ok=True)
    for name, path in inputs.items():
        shutil.copyfile(path, root / 'config' / name)
    shutil.copyfile(protocol_document, root / 'config/frozen-protocol.md')
    roster.to_csv(root / 'config/source-reservation.csv', index=False)
    availability = None
    if video_root is not None:
        from .fi_scaling_availability import freeze_available
        availability = freeze_available(root, video_root, resolution_manifest=resolution_manifest or video_manifest)
        amendment = Path(__file__).resolve().parents[4]/'docs/studies/future-innovation/source-learning-curve-available-cohort-protocol.md'
        shutil.copyfile(amendment, root/'config/frozen-cohort-amendment.md')
    copied = ('teacher-contract.json', 'frame-contract.json', 'nuisance-contract.json', 'control-contract.json',
              'nuisance-schema.json', 'projection-contract.json', 'projection-256.npy', 'thresholds.json')
    for name in copied:
        shutil.copyfile(parent / 'config' / name, root / 'config' / name)
    write_json(root / 'config/policy.json', POLICY)
    write_json(root / 'config/model-contract.json', asdict(JointModelContract()))
    write_json(root / 'config/parent-snapshot.json', parent_before)
    write_json(root / 'config/software.json', fingerprint())
    write_json(root / 'config/runtime.json', runtime_versions())
    sealed = {p.name: sha256_file(p) for p in sorted((root / 'config').iterdir()) if p.is_file()}
    study = dict(protocol=PROTOCOL, run_id=root.name, parent_root=str(parent), synthetic=parent_run['synthetic'],
                 config_sha256=sealed, input_provenance={n: {'path': str(p.resolve()), 'sha256': sha256_file(p)} for n,p in inputs.items()},
                 parent_run_sha256=sha256_file(parent / 'config/run-contract.json'))
    if availability is not None:
        study['cohort_policy'] = availability['cohort_policy']
    write_once_json(root / 'config/study.json', study)
    # Generic run metadata lets the original candidate builder validate inputs.
    # The scaling CLI owns dispatch; old gate commands never accept this protocol.
    write_once_json(root / 'config/run-contract.json', dict(protocol=PROTOCOL, synthetic=study['synthetic'],
        config_sha256={**sealed, 'study.json': sha256_file(root / 'config/study.json')},
        code_sha256=stable_key(*sealed.values()), cohort_size=None, minimum_sources=25, source_cap=None,
        inputs_sha256={}, input_paths={}, manifest_sha256=None))
    annotations = sequences[['sequence_id', 'video_id', 'first_frame', 'last_frame', 'n_annotated_frames']].copy()
    annotations['annotation_length_compatible']=(annotations.last_frame-annotations.first_frame+1>=64)&(annotations.n_annotated_frames>=64)
    annotations['candidate_status'] = np.where(annotations.sequence_id.isin(candidates.sequence_id), 'parent_candidate', 'no_parent_candidate')
    annotations['eligibility_status'] = np.where(annotations.sequence_id.isin(old.sequence_id), 'parent_verified', 'not_processed')
    cohort_contract=read_json(parent/'config/cohort-contract.json')
    checked_file(parent,'manifests/exclusions.csv',cohort_contract['artifacts']['manifests/exclusions.csv'])
    exclusions = pd.read_csv(parent / 'manifests/exclusions.csv')
    failed = set(exclusions.loc[exclusions.stage == 'pose_eligibility', 'sequence_id'])
    annotations.loc[annotations.sequence_id.isin(failed), 'eligibility_status'] = 'parent_pose_failed'
    annotations = annotations.merge(roster, on='video_id', validate='many_to_one')
    annotations.to_csv(root / 'manifests/initial-eligibility.csv', index=False)
    known = 0 if participants is None else participants.participant_id.dropna().replace('', np.nan).nunique()
    report = dict(protocol=PROTOCOL, status='awaiting_expanded_processing', measurement_complete=False,
                  annotated_sequences=len(sequences), recordings=videos.video_id.nunique(),
                  parent_candidates=len(candidates), parent_candidate_recordings=candidates.video_id.nunique(),
                  annotation_length_compatible=int(annotations.annotation_length_compatible.sum()),
                  annotation_length_note='Necessary span/count conditions only; contiguous boxes and pose eligibility require processing',
                  verified_eligible_sequences=len(old), verified_eligible_recordings=old.video_id.nunique(),
                  eligibility_counts=annotations.eligibility_status.value_counts().to_dict(),
                  explicitly_identified_participants=known, total_participants=None,
                  participant_status='unknown; source IDs are not participant IDs' if participants is None else 'partial explicit registry',
                  exposed_recordings=int(roster.previously_exposed.sum()),
                  reservation=roster.groupby('role').agg(recordings=('video_id','size'), sequences=('annotated_sequences','sum')).to_dict('index'),
                  confirmation_claim='reserved relative to supplied exposure registries; participant independence unestablished')
    if availability is not None:
        report['availability'] = availability
    write_json(root / 'reports/cohort-audit.json', report)
    write_json(root/'reports/cohort-audit-complete.json',{'artifacts':{name:sha256_file(root/name) for name in
               ('reports/cohort-audit.json','manifests/initial-eligibility.csv','config/source-reservation.csv')}})
    verify_snapshot(parent, parent_before)
    return report


def read_study(root, parent_override=None, *, require_software=False):
    root = Path(root)
    study = read_json(root / 'config/study.json')
    if study['protocol'] != PROTOCOL:
        raise ValueError('Wrong learning-curve protocol')
    for name, digest in study['config_sha256'].items():
        checked_file(root, 'config/' + name, digest)
    if read_json(root / 'config/policy.json') != POLICY:
        raise ValueError('Unsupported scaling policy; use a new version for amendments')
    if require_software and read_json(root / 'config/software.json') != fingerprint():
        raise ValueError('Fitting code changed after freeze; create a new study')
    if 'availability-contract.json' in study['config_sha256'] or study.get('cohort_policy') is not None:
        from .fi_scaling_availability import COHORT_POLICY, verify_available
        if study.get('cohort_policy') != COHORT_POLICY or 'availability-contract.json' not in study['config_sha256']:
            raise ValueError('Inconsistent available-cohort version/binding')
        verify_available(root)
    parent = Path(parent_override or study['parent_root']).resolve()
    verify_snapshot(parent, read_json(root / 'config/parent-snapshot.json'))
    return study, parent
