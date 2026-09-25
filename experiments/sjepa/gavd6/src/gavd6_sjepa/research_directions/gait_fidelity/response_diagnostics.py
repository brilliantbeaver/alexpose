"""Training-person diagnostics for the separately frozen response follow-up.

Slots retain their time/joint order. Reference labels are diagnostic targets,
never encoder inputs. Parent feature export runs its recorded interpreter and
source checkout, and verifies its original frozen run and checkpoint receipt.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path
import subprocess

import numpy as np

from .common import atomic_json, digest, read_json, sha256


def diagnostic_panel(bundle, *, max_families_per_person=3, seed=17):
    """Select one nonzero, non-held pair per hash-selected training family."""
    from .training import paired_indices
    if not 1 <= max_families_per_person <= 3:
        raise ValueError('Diagnostic panel permits one to three families per person')
    candidates = {}
    for a, b in paired_indices(bundle.records):
        first, second = bundle.records[a], bundle.records[b]
        if second['movement_state'] == 'no_change' or not float(second['movement_level_deg']):
            continue
        if first['split'] != 'train' or second['split'] != 'train' or second.get('held_intervention', False):
            raise PermissionError('Diagnostic references must be non-held training examples')
        key = (str(first['canonical_person_id']), str(first['source_family_id']))
        identifier = [seed, key, first['pair_id'], second['movement_state'], second['movement_level_deg']]
        candidates.setdefault(key, []).append((digest(identifier), int(a), int(b)))
    people = sorted({key[0] for key in candidates})
    rows = []
    for person in people:
        families = sorted((family for who, family in candidates if who == person),
                          key=lambda family: digest([seed, person, family]))[:max_families_per_person]
        for family in families:
            _, a, b = min(candidates[person, family])
            rows.append(dict(canonical_person_id=person, source_family_id=family,
                             endpoint_indices=[a, b], pair_id=bundle.records[a]['pair_id'], split='train'))
    if not rows:
        raise ValueError('No nonzero training pairs for representation diagnostics')
    panel = dict(schema='gait-fidelity-response-panel-v1', seed=seed, rows=rows,
                 maximum_families_per_person=max_families_per_person,
                 selection='hash-ranked training families, then hash-ranked nonzero pair; no feature or outcome selection')
    panel['identity'] = digest(panel)
    return panel


def person_folds(people, *, seed=17):
    """Exactly three deterministic folds, without splitting a person."""
    people = np.asarray(people, dtype=str)
    unique = sorted(set(people), key=lambda value: digest([seed, value]))
    if len(unique) < 3:
        raise ValueError('Three-fold person cross-validation needs at least three people')
    assignment = {person: index % 3 for index, person in enumerate(unique)}
    return np.asarray([assignment[person] for person in people], dtype=np.int64)


def person_ridge_probe(features, target, people, *, alpha=.01, seed=17):
    """Fixed mean-loss ridge with standardization learned inside each train fold.

    The objective is mean((Xw+b-y)^2) + alpha*||w||^2. A dual solve avoids
    constructing a slot-by-feature Gram matrix for the small frozen panel.
    """
    if alpha != .01:
        raise ValueError('The registered diagnostic ridge alpha is fixed at .01')
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    people = np.asarray(people, dtype=str)
    if features.ndim < 2 or len(features) != len(target) or target.shape != people.shape:
        raise ValueError('Probe features, scalar targets and people must align')
    x = features.reshape(len(features), -1)
    if not np.isfinite(x).all() or not np.isfinite(target).all():
        raise ValueError('Probe cannot silently exclude nonfinite diagnostic examples')
    folds = person_folds(people, seed=seed)
    predicted = np.empty_like(target)
    receipts = []
    for fold in range(3):
        train, test = folds != fold, folds == fold
        mean, scale = x[train].mean(0), x[train].std(0)
        scale = np.where(scale > 1e-12, scale, 1.)
        xt, xv = (x[train] - mean) / scale, (x[test] - mean) / scale
        y_mean = target[train].mean()
        dual = np.linalg.solve(xt @ xt.T + (train.sum() * alpha) * np.eye(train.sum()),
                               target[train] - y_mean)
        predicted[test] = xv @ (xt.T @ dual) + y_mean
        receipts.append(dict(fold=fold, train_people=sorted(set(people[train])),
                             test_people=sorted(set(people[test])), training_rows=int(train.sum()),
                             testing_rows=int(test.sum()), standardization='training fold only',
                             training_target_mean=float(y_mean)))
    error, zero_error = (predicted - target) ** 2, target ** 2
    per_person = [dict(canonical_person_id=person,
                       mse=float(error[people == person].mean()),
                       zero_change_mse=float(zero_error[people == person].mean()))
                  for person in sorted(set(people))]
    mse = float(np.mean([row['mse'] for row in per_person]))
    baseline = float(np.mean([row['zero_change_mse'] for row in per_person]))
    return dict(alpha=alpha, objective='mean squared error plus alpha times squared coefficient norm',
                folds=receipts, people=len(per_person), rows=len(target), mse=mse,
                zero_change_mse=baseline, improvement=baseline-mse,
                predictions=predicted.tolist(), fold_indices=folds.tolist(), per_person=per_person,
                interpretation='Training-person feasibility diagnostic; no development, confirmation or clinical inference')


def feature_statistics(features):
    """Across-example variance at each fixed endpoint/time/joint/feature slot."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 4 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError('Features must be finite [pairs,2,ordered_slots,width]')
    delta = values[:, 1] - values[:, 0]
    return dict(pairs=len(values), slots=values.shape[2], width=values.shape[3],
                fixed_slot_variance=float(values.var(axis=0).mean()),
                fixed_slot_delta_variance=float(delta.var(axis=0).mean()),
                mean_delta_l2=float(np.linalg.norm(delta.reshape(len(delta), -1), axis=1).mean()),
                mean_delta_energy=float(np.square(delta).mean()),
                variance_axes='examples only, preserving endpoint/time/joint/feature slots')


def export_features(checkpoint, bundle_path, panel_path, cfg, output):
    """Export using only the importing checkout's established public model API.

    This function is deliberately self-contained: its exact source is copied
    into a child-owned driver executed by the recorded parent interpreter.
    """
    from pathlib import Path
    import numpy as np
    import torch
    from gavd6_sjepa.research_directions.gait_fidelity.common import atomic_json, digest, read_json, sha256
    from gavd6_sjepa.research_directions.gait_fidelity.data import load_dataset
    from gavd6_sjepa.research_directions.gait_fidelity.masking import sample_mask
    from gavd6_sjepa.research_directions.gait_fidelity.measurements import knee_excursion, reference_support
    from gavd6_sjepa.research_directions.gait_fidelity.training import load_model, normalize_batch, _take, _tensors
    panel = read_json(panel_path)
    expected = dict(panel); expected.pop('identity')
    if digest(expected) != panel['identity']:
        raise RuntimeError('Diagnostic panel identity changed')
    bundle = load_dataset(bundle_path)
    pairs = np.asarray([row['endpoint_indices'] for row in panel['rows']], dtype=np.int64)
    for pair, record in zip(pairs, panel['rows']):
        for index in pair:
            row = bundle.records[index]
            if (row['split'] != 'train' or row.get('held_intervention', False)
                    or str(row['canonical_person_id']) != record['canonical_person_id']
                    or str(row['source_family_id']) != record['source_family_id']):
                raise PermissionError('Diagnostic panel does not match training-only bundle rows')
    model, payload = load_model(checkpoint, device=cfg.get('device', 'cpu'), require_final=False)
    if not hasattr(model, 'encoder'):
        raise ValueError('Representation export requires an encoder')
    settings = payload['signature']['model']
    patch_size = settings['patch_size']
    arrays, normalization, mask_receipts = {}, {}, {}
    for mode in ('masked', 'deployment', 'no_change'):
        chosen = pairs if mode != 'no_change' else np.repeat(pairs[:, :1], 2, axis=1)
        rng = np.random.default_rng(panel['seed'] + (313 if mode == 'no_change' else 211))
        parts = {'encoder': [], 'predictor': [], 'teacher': []}
        origins, scales, masks = [], [], []
        for selected in chosen:
            raw = _take(bundle.inputs, selected)
            hidden = None
            if mode != 'deployment':
                hidden, receipt = sample_mask(raw['observed'], 'graph_time', rng=rng,
                    fraction=float(cfg.get('training', {}).get('mask_fraction', .5)),
                    patch_size=patch_size, return_receipt=True)
                masks.append(receipt)
            normalized, origin, scale, fallback = normalize_batch(raw, hidden, patch_size=patch_size)
            device = cfg.get('device', 'cpu')
            inputs = _tensors(normalized, device)
            hidden_tensor = None if hidden is None else torch.as_tensor(hidden, dtype=torch.bool, device=device)
            valid = torch.as_tensor(bundle.targets['valid'][selected], dtype=torch.bool, device=device)
            reference = torch.as_tensor((bundle.targets['xy'][selected] - origin[:, None, None])
                                        / scale[:, None, None, None], dtype=torch.float32, device=device)
            teacher_inputs = dict(xy=torch.where(valid[..., None], reference, 0), observed=valid,
                                  confidence=valid.float(), timestamps=inputs['timestamps'])
            with torch.inference_mode():
                encoded = model.encoder(inputs, hidden_tensor)
                parts['encoder'].append(encoded.float().cpu().numpy())
                if hasattr(model, 'predictor'):
                    parts['predictor'].append(model.predictor(encoded).float().cpu().numpy())
                    parts['teacher'].append(model.teacher(teacher_inputs).float().cpu().numpy())
            origins.append(origin.tolist()); scales.append(scale.tolist())
        for branch, values in parts.items():
            if values:
                arrays[mode + '_' + branch] = np.stack(values)
        normalization[mode] = dict(origins=origins, scales=scales)
        mask_receipts[mode] = masks
    flat = pairs.reshape(-1)
    truth, valid = bundle.targets['xy'][flat], bundle.targets['valid'][flat]
    measure = cfg['measurement']
    fixed = reference_support(truth, valid, groups=np.arange(len(flat)).reshape(-1, 2),
                              min_segment_px=measure['min_segment_px'])
    result = knee_excursion(truth, valid, support=fixed, timestamps=bundle.inputs['timestamps'][flat],
                            min_segment_px=measure['min_segment_px'], min_frames=measure['min_frames'],
                            min_fraction=measure['min_coverage'])
    if not result['supported'].all():
        raise ValueError('Hash-selected diagnostic panel contains unsupported references; no replacement sampling')
    arrays['reference_delta_A'] = np.diff(result['asymmetry'].reshape(-1, 2), axis=1).ravel()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'features.npz', **arrays)
    metadata = dict(schema='gait-fidelity-response-features-v1', panel=panel,
                    checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha256(checkpoint),
                    checkpoint_signature_sha256=digest(payload['signature']),
                    arrays=str((output / 'features.npz').resolve()), arrays_sha256=sha256(output / 'features.npz'),
                    slot_order='time patch, body12 joint, feature; no pooling or permutation',
                    normalization=normalization, masks=mask_receipts, training_only=True,
                    no_change='identical baseline observations, independently drawn masks and context normalization')
    atomic_json(output / 'features.json', metadata)
    return metadata


def analyze_feature_export(export_directory, output=None):
    source = Path(export_directory)
    metadata = read_json(source / 'features.json')
    if not metadata.get('training_only') or any(row['split'] != 'train' for row in metadata['panel']['rows']):
        raise PermissionError('Feature diagnostics only admit training references')
    if sha256(metadata['arrays']) != metadata['arrays_sha256']:
        raise RuntimeError('Retained feature arrays changed')
    people = [row['canonical_person_id'] for row in metadata['panel']['rows']]
    result = dict(schema='gait-fidelity-response-feature-diagnostics-v1', statistics={}, probes={},
                  panel_identity=metadata['panel']['identity'], training_only=True, probe_result='not_assessed')
    with np.load(metadata['arrays'], allow_pickle=False) as arrays:
        result['unsupported_branches'] = [f'{mode}_{branch}' for mode in ('masked', 'deployment', 'no_change')
            for branch in ('encoder', 'predictor', 'teacher') if f'{mode}_{branch}' not in arrays.files]
        for key in sorted(set(arrays.files) - {'reference_delta_A'}):
            features = arrays[key]
            result['statistics'][key] = feature_statistics(features)
            if not key.startswith('no_change_'):
                if len(set(people)) >= 3:
                    result['probes'][key] = person_ridge_probe(features[:, 1]-features[:, 0],
                                                              arrays['reference_delta_A'], people)
                else:
                    result['probes'][key] = dict(status='insufficient_training_people_for_three_folds')
    teacher = result['probes'].get('masked_teacher', {})
    if 'improvement' in teacher:
        result['probe_result'] = 'positive_training_person_probe' if teacher['improvement'] > 0 else 'no_positive_training_person_probe'
    result['probe_interpretation'] = 'Descriptive only; probes never select models, admit or stop fitting, or establish nonlinear impossibility'
    if output is not None:
        atomic_json(output, result)
    return result


def export_parent_features(parent_cfg, completed, bundle_path, panel_path, output, *, phase=None):
    """Execute a child-owned driver through the original parent code boundary."""
    from .response_evaluation import verified_response_completion
    result = verified_response_completion(completed, phase)
    checkpoint = result['checkpoint']
    if sha256(checkpoint) != result['checkpoint_sha256']:
        raise RuntimeError('Parent checkpoint identity changed')
    output = Path(output).resolve(); parent_work = Path(parent_cfg['work']).resolve()
    if output.is_relative_to(parent_work):
        raise PermissionError('Parent exports must be written under the child run')
    output.mkdir(parents=True, exist_ok=True)
    request = dict(config=parent_cfg, completed=completed, checkpoint=checkpoint,
                   bundle=str(bundle_path), panel=str(Path(panel_path).resolve()), output=str(output))
    atomic_json(output / 'request.json', request)
    driver = ("import json,sys\nfrom pathlib import Path\n"
              "request=json.loads(Path(sys.argv[1]).read_text())\n"
              "sys.path.insert(0,str(Path(request['config']['code_root'])/'src'))\n"
              "from gavd6_sjepa.research_directions.gait_fidelity.scheduler import _verify_frozen,verify_completed\n"
              "from gavd6_sjepa.research_directions.gait_fidelity import training as parent_training\n"
              "assert Path(parent_training.__file__).resolve().is_relative_to(Path(request['config']['code_root']).resolve())\n"
              "_verify_frozen(request['config']);verify_completed(request['completed'])\n"
              + inspect.getsource(export_features)
              + "\nexport_features(request['checkpoint'],request['bundle'],request['panel'],request['config'],request['output'])\n")
    path = output / 'parent-export.py'; path.write_text(driver)
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    with (output / 'export.log').open('w') as log:
        subprocess.run([parent_cfg['python'], '-I', '-B', str(path), str(output / 'request.json')],
                       cwd=parent_cfg['code_root'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata = read_json(output / 'features.json')
    metadata['parent_boundary'] = dict(code_root=parent_cfg['code_root'], interpreter=parent_cfg['python'],
        original_completion=completed, driver_sha256=sha256(path), parent_frozen_sha256=sha256(parent_work/'frozen.json'))
    atomic_json(output / 'features.json', metadata)
    return metadata


def run_response_diagnostics(cfg, output, *, include_child=True):
    from .data import load_dataset
    from .followup import verify_parent
    from .response_evaluation import verified_response_completion
    verify_parent(cfg)
    binding = cfg['followup']['parent_binding']
    bundle = load_dataset(binding['bundle'])
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    panel = diagnostic_panel(bundle)
    atomic_json(output / 'panel.json', panel)
    parent_cfg = read_json(Path(binding['parent_work'])/'config.json')
    parent_plan = {phase['phase_id']: phase for phase in read_json(Path(binding['parent_work'])/'plan.json')['phases']}
    exports = []
    selected = set()
    for phase_id, phase in binding['baseline_phases'].items():
        if (phase['recipe']['encoder'] == 'initialized'
                and phase['recipe']['readout_or_training_objective'] == 'paired_change'):
            selected.add(phase_id)
        for dependency in phase['depends_on']:
            if binding['parent_completed'].get(dependency, {}).get('result', {}).get('phase') == 'pretrain':
                selected.add(dependency)
    for phase_id in sorted(selected):
        completed = binding['parent_completed'][phase_id]
        destination = output / ('parent-' + phase_id)
        export_parent_features(parent_cfg, completed, binding['bundle'], output/'panel.json', destination, phase=parent_plan[phase_id])
        exports.append((phase_id, destination, 'parent'))
    if include_child:
        state = read_json(Path(cfg['work'])/'ledger.json')
        for phase in read_json(Path(cfg['work'])/'plan.json')['phases']:
            if phase['phase'] != 'pretrain':
                continue
            phase_id = phase['phase_id']
            completed = state['completed'][phase_id]
            result = verified_response_completion(completed, phase)
            destination = output / ('child-' + phase_id)
            export_features(result['checkpoint'], binding['bundle'], output/'panel.json', cfg, destination)
            exports.append((phase_id, destination, 'child'))
    if not exports:
        raise ValueError('No verified parent JEPA checkpoints were available for diagnostics')
    summaries = []
    for phase_id, destination, origin in exports:
        result = analyze_feature_export(destination, destination/'diagnostics.json')
        summaries.append(dict(phase_id=phase_id, origin=origin, diagnostics=str(destination/'diagnostics.json'),
                              probe_result=result['probe_result']))
    result = dict(status='TRAINING_ONLY_DIAGNOSTICS_COMPLETE', panel_identity=panel['identity'],
                  exports=summaries, training_only=True, independent_confirmation=False, clinical_validation=False)
    atomic_json(output/'diagnostics-summary.json', result)
    return result
