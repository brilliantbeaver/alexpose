"""Notebook-sized stages for response-conditioned synthetic teaching.

Source fitting and deployment never read human target coordinates. Real scoring
is a separate stage using an already frozen teacher and saved predictions.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from .config import RunConfig
from . import data
from .annotations import prepare_gavd_panel, import_gavd_annotations
from .estimators import load_estimator
from .measurements import diagnostic_summary, frame_scores, landmark_errors, paired_recording_interval
from .selectors import FEATURE_VIEWS, LessonSelector, chosen_errors, selection_regret
from .trials import (adapt, context_feature_paths, make_episode_features, predict_dataset,
                     read_npz, reference_arrays, run_source_student, student_spec, write_json)


def inventory(cfg: RunConfig):
    """Read manifests and report asset readiness without loading GPU models."""
    from ..motion_preservation.motion_data import load_amass_manifest, load_gavd_manifest
    cfg.validate()
    if not (cfg.root / 'config.json').exists():
        cfg.save()
    amass = load_amass_manifest(cfg.amass_manifest_dir, cfg.amass_root, cfg.seed)
    gavd = load_gavd_manifest(cfg.gavd_manifest_dir, cfg.gavd_video_root)
    assets = {name: getattr(cfg, name) for name in (
        'amass_root', 'body_model_root', 'render_texture_dir', 'render_background_dir',
        'render_uv_path', 'coco_image_root', 'coco_annotations_json', 'gavd_video_root',
        'gavd_reservation_csv', 'context_repo', 'context_checkpoint', 'image_checkpoint')}
    rows = [dict(asset=name, path=path or '', exists=bool(path) and Path(path).exists())
            for name, path in assets.items()]
    for student in cfg.students:
        for field in ('config', 'checkpoint', 'head_checkpoint'):
            if student.get(field):
                rows.append(dict(asset=f"{student['student_id']}:{field}", path=student[field],
                                 exists=Path(student[field]).is_file()))
    availability = pd.DataFrame(rows)
    availability.to_csv(cfg.root / 'asset_inventory.csv', index=False)
    return dict(amass=amass, gavd=gavd, assets=availability, students=pd.DataFrame(cfg.students))


def prepare_data(cfg: RunConfig):
    """Render full-body lessons, prepare COCO replay, and cache frozen context."""
    if (cfg.root / 'selectors/frozen.joblib').exists():
        raise RuntimeError('The teacher is frozen. Keep its lesson library or use a new run root.')
    if any((cfg.root / 'source').glob('*/outcomes.csv')):
        raise RuntimeError('Source outcomes already depend on this data library. Use a new run to rebuild it.')
    if not (cfg.root / 'config.json').exists():
        cfg.save()
    synthetic = data.prepare_amass_library(cfg)
    replay = data.prepare_coco_replay(cfg)
    index = data.load_pose_manifest(synthetic)
    # Finish shared GPU extraction before the launcher releases the student array.
    for split in ('train', 'validation'):
        contexts = index.loc[index.role.eq(f'{split}_context')].reset_index(drop=True)
        context_feature_paths(cfg, contexts, split)
    return dict(synthetic=str(synthetic), replay=str(replay),
                counts=index.groupby(['role', 'domain_id']).size().rename('frames').reset_index())


def source_trials(cfg: RunConfig, student_id=None):
    """Run one source student per array job, or all source students locally."""
    if (cfg.root / 'selectors/frozen.joblib').exists():
        raise RuntimeError('Source fitting is frozen. Use a new run root for additional training trials.')
    ids = [student_id] if student_id else [s['student_id'] for s in cfg.source_students]
    if not ids:
        raise ValueError('Configure train and validation students before running trials.')
    return pd.concat([run_source_student(cfg, sid) for sid in ids], ignore_index=True)


def _source_episodes(cfg):
    tables = []
    for student in cfg.source_students:
        path = cfg.root / 'source' / student['student_id'] / 'outcomes.csv'
        if not path.exists():
            raise FileNotFoundError(f'Complete source trials for {student["student_id"]}: {path}')
        table = pd.read_csv(path)
        expected = 'train' if student['role'] == 'train' else 'validation'
        if set(table.split) != {expected} or set(table.student_id) != {student['student_id']}:
            raise ValueError('A source outcome file has the wrong student or fitting role.')
        tables.append(table)
    outcomes = pd.concat(tables, ignore_index=True)
    lessons = sorted(data.load_pose_manifest(cfg.root / 'data/synthetic.csv', roles='lesson').lesson_id.unique())
    actions = ['replay', *lessons]
    episodes = []
    for (sid, split, domain, budget), group in outcomes.loc[outcomes.budget.gt(0)].groupby(
            ['student_id', 'split', 'domain_id', 'budget'], sort=True):
        if group.action.duplicated().any():
            raise ValueError('Each source intervention should have one outcome per domain.')
        errors = group.set_index('action').error
        if not set(actions) <= set(errors.index):
            raise ValueError(f'Incomplete candidate outcomes for {sid}/{domain}/{budget}.')
        features = read_npz(cfg.root / 'source' / sid / 'features' / f'{domain}_{budget}.npz')
        episodes.append(dict(student_id=sid, split=split, domain_id=domain, budget=int(budget),
                             features=features, errors=errors.reindex(actions).to_numpy(float),
                             extra=errors.to_dict()))
    if not episodes or {e['split'] for e in episodes} != {'train', 'validation'}:
        raise ValueError('Fit requires independent train and validation student checkpoints.')
    return lessons, episodes, outcomes


def _feature_matrix(episodes):
    return [e['features'] for e in episodes]


def fit_selectors(cfg: RunConfig):
    """Fit source-only teachers and choose budget/comparator before real labels."""
    output = cfg.root / 'selectors'
    if (output / 'frozen.joblib').exists():
        raise FileExistsError('A frozen teacher already exists; use a new run for changed fitting.')
    if (cfg.root / 'evaluation').exists():
        raise RuntimeError('Real outcomes already exist. A new method needs a new development run.')
    lessons, episodes, outcomes = _source_episodes(cfg)
    train = [e for e in episodes if e['split'] == 'train']
    valid = [e for e in episodes if e['split'] == 'validation']
    gains = np.stack([e['errors'][0] - e['errors'][1:] for e in train])
    views = [name for name, keys in FEATURE_VIEWS.items()
             if all(all(key in e['features'] for key in keys) for e in episodes)]
    candidates = [('nearest', k) for k in cfg.selector_neighbors] + [('ridge', a) for a in cfg.selector_ridge]
    fitted, rows = {}, []
    for view in views:
        for kind, parameter in candidates:
            selector = LessonSelector(view, kind, parameter, lessons).fit(_feature_matrix(train), gains)
            key = (view, kind, float(parameter))
            fitted[key] = selector
            for budget in cfg.adaptation_steps:
                selected = [e for e in valid if e['budget'] == budget]
                actions = selector.select(_feature_matrix(selected))
                errors = np.stack([e['errors'] for e in selected])
                selected_errors = chosen_errors(actions, errors, lessons)
                regrets = selection_regret(actions, errors, lessons)
                for e, action, error, regret in zip(selected, actions, selected_errors, regrets):
                    rows.append(dict(view=view, kind=kind, parameter=float(parameter), budget=budget,
                                     student_id=e['student_id'], domain_id=e['domain_id'],
                                     action=action, error=float(error), regret=float(regret)))
    validation = pd.DataFrame(rows)
    mean = validation.groupby(['view', 'kind', 'parameter', 'budget'], as_index=False).agg(
        error=('error', 'mean'), regret=('regret', 'mean'))
    if 'full' not in views or 'source_progress' not in views:
        raise ValueError('Full and source-progress inputs must exist before evaluating the main claim.')
    best_full = mean.loc[mean.view.eq('full')].sort_values(['error', 'budget', 'kind', 'parameter']).iloc[0]
    budget = int(best_full.budget)
    chosen, models = {}, {}
    for view in views:
        best = mean.loc[mean.view.eq(view) & mean.budget.eq(budget)].sort_values(['error', 'kind', 'parameter']).iloc[0]
        key = (view, str(best.kind), float(best.parameter))
        models[view] = fitted[key]
        chosen[view] = dict(kind=str(best.kind), parameter=float(best.parameter), validation_error=float(best.error))
    # Retain a strictly matched information comparison as well as each view's
    # strongest source-tuned implementation. Only its input fields differ.
    matched_key = ('source_progress', str(best_full.kind), float(best_full.parameter))
    models['source_progress_matched'] = fitted[matched_key]
    matched = mean.loc[mean.view.eq('source_progress') & mean.kind.eq(best_full.kind)
                       & mean.parameter.eq(best_full.parameter) & mean.budget.eq(budget)].iloc[0]
    chosen['source_progress_matched'] = dict(kind=str(best_full.kind), parameter=float(best_full.parameter),
                                             validation_error=float(matched.error))
    comparator_pool = ['before', 'after', 'source_progress', 'magnitude', 'weakness', 'domain']
    primary = min(comparator_pool, key=lambda name: chosen[name]['validation_error'])
    source_at_budget = [e for e in train if e['budget'] == budget]
    fixed_gains = np.mean([e['errors'][0] - e['errors'][1:] for e in source_at_budget], axis=0)
    fixed = 'replay' if fixed_gains.max() <= 0 else lessons[int(fixed_gains.argmax())]
    selected = dict(budget=budget, primary_comparator=primary, best_fixed=fixed,
                    methods=chosen, lessons=lessons, context_kind=cfg.context_kind,
                    omitted_feature_views=sorted(set(FEATURE_VIEWS) - set(views)),
                    n_train_students=len({e['student_id'] for e in train}),
                    n_validation_students=len({e['student_id'] for e in valid}))
    output.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output / 'validation_predictions.csv', index=False)
    mean.to_csv(output / 'validation_summary.csv', index=False)
    outcomes.to_csv(output / 'source_outcomes.csv', index=False)
    write_json(output / 'selection.json', selected)
    joblib.dump(dict(models=models, selection=selected, config=asdict(cfg)), output / 'frozen.joblib')
    return dict(selection=selected, validation=mean.sort_values('error'))


def _frozen(cfg):
    path = cfg.root / 'selectors/frozen.joblib'
    if not path.is_file():
        raise FileNotFoundError('Fit and freeze source selectors in notebook 03 before deployment.')
    bundle = joblib.load(path)
    # These are scientific settings, not an environment provenance framework.
    for field in ('seed', 'students', 'probe_steps', 'train_batch_size', 'synthetic_fraction',
                  'learning_rate', 'weight_decay', 'context_kind', 'context_builder',
                  'context_checkpoint', 'context_checkpoint_key', 'context_frames', 'context_image_size',
                  'context_repo', 'image_checkpoint', 'missing_prediction_penalty', 'accurate_joint_threshold'):
        if getattr(cfg, field) != bundle['config'][field]:
            raise ValueError(f'{field} changed after source selection. Use the frozen configuration or a new run.')
    return bundle


def prepare_gavd(cfg):
    """Create eligible viewing-group and independent annotation materials."""
    if any((cfg.root / 'deployment').glob('*/choices.csv')):
        raise RuntimeError('Deployment choices already exist. Keep their fixed GAVD panel or use a new run.')
    result = prepare_gavd_panel(cfg)
    path = cfg.root / 'data/gavd_context.csv'
    if path.is_file():
        context_feature_paths(cfg, data.load_pose_manifest(path), 'gavd')
    return result


def _heuristic_actions(features, lessons, domain, seed):
    diagnostic = features['diagnostic_post'].reshape(len(lessons), 24)
    hard = lessons[int(diagnostic[:, :12].mean(axis=1).argmax())]
    matched = 'low_resolution' if 'low' in domain else next(
        (name for name in ('front', 'oblique', 'side') if domain.startswith(name)), 'balanced')
    if matched not in lessons:
        matched = lessons[0]
    return dict(hard_example=hard, geometry_match=matched,
                random=lessons[int(np.random.default_rng(seed).integers(len(lessons)))])


def _deploy_student(cfg, item, bundle):
    started = perf_counter()
    root = cfg.root / 'deployment' / item['student_id']
    if (root / 'choices.csv').exists():
        raise FileExistsError(f'Deployment already exists for {item["student_id"]}; retain it for evaluation.')
    root.mkdir(parents=True, exist_ok=True)
    index = data.load_pose_manifest(cfg.root / 'data/synthetic.csv')
    contexts = data.load_pose_manifest(cfg.root / 'data/gavd_context.csv')
    evaluation = data.load_pose_manifest(cfg.root / 'data/gavd_evaluation.csv')
    # Neither context nor evaluation landmark files are opened during deployment.
    context_dataset = data.PoseFrameDataset(contexts, labeled=False)
    eval_dataset = data.PoseFrameDataset(evaluation, labeled=False)
    diagnostic = data.PoseFrameDataset(index.loc[index.role.eq('diagnostic')])
    replay = data.PoseFrameDataset(cfg.root / 'data/replay.csv')
    probe = data.PoseFrameDataset(index.loc[index.role.eq('probe')])
    lesson_rows = index.loc[index.role.eq('lesson')]
    lessons, budget = bundle['selection']['lessons'], bundle['selection']['budget']
    paths = context_feature_paths(cfg, contexts, 'gavd')
    estimator = load_estimator(student_spec(item), cfg.device)
    original = estimator.head_state()
    before = predict_dataset(estimator, context_dataset, cfg.predict_batch_size)
    diag_targets, diag_visible, diag_boxes = reference_arrays(diagnostic)
    diag_pre = diagnostic_summary(predict_dataset(estimator, diagnostic, cfg.predict_batch_size),
        diag_targets, diag_visible, diag_boxes, diagnostic.index.lesson_id, lessons,
        missing_penalty=cfg.missing_prediction_penalty)
    baseline_predictions = predict_dataset(estimator, eval_dataset, cfg.predict_batch_size)
    lr = float(item.get('learning_rate', cfg.learning_rate))
    history, probe_seconds = adapt(estimator, replay, probe, cfg, cfg.probe_steps, lr, seed=cfg.seed)
    probed = estimator.head_state()
    estimator.save_head(root / 'probe_head.pt')
    after = predict_dataset(estimator, context_dataset, cfg.predict_batch_size)
    diag_post = diagnostic_summary(predict_dataset(estimator, diagnostic, cfg.predict_batch_size),
        diag_targets, diag_visible, diag_boxes, diagnostic.index.lesson_id, lessons,
        missing_penalty=cfg.missing_prediction_penalty)
    probe_predictions = predict_dataset(estimator, eval_dataset, cfg.predict_batch_size)
    choices = []
    for domain_number, (domain, group) in enumerate(contexts.groupby('domain_id', sort=True)):
        pos = group.index.to_numpy()
        features = make_episode_features(before[pos], after[pos], group, diag_pre, diag_post, history,
                                         read_npz(paths[str(domain)]), cfg, item, budget)
        selected = {name: model.select([features])[0] for name, model in bundle['models'].items()}
        selected.update(_heuristic_actions(features, lessons, str(domain), cfg.seed + domain_number))
        selected.update(best_fixed=bundle['selection']['best_fixed'], replay='replay', pooled='pooled',
                        original='original', probe='probe', full_replay='full_replay')
        for method, action in selected.items():
            choices.append(dict(student_id=item['student_id'], family=item['family'], student_role=item['role'],
                                domain_id=str(domain), method=method, action=action, budget=budget))
    choice_table = pd.DataFrame(choices)
    # All decisions are saved before applying any selected lesson or loading human references.
    choice_table.to_csv(root / 'choices_pending.csv', index=False)
    predictions = dict(original=baseline_predictions, probe=probe_predictions)
    timings = {'probe': probe_seconds}
    for action in sorted(set(choice_table.action) - {'original', 'probe'}):
        estimator.load_head_state(original if action == 'full_replay' else probed)
        lesson = None if action in {'replay', 'full_replay'} else data.PoseFrameDataset(
            lesson_rows if action == 'pooled' else lesson_rows.loc[lesson_rows.lesson_id.eq(action)])
        steps = cfg.probe_steps + budget if action == 'full_replay' else budget
        _, seconds = adapt(estimator, replay, lesson, cfg, steps, lr,
                           seed=cfg.seed if action == 'full_replay' else cfg.seed + 100)
        predictions[action] = predict_dataset(estimator, eval_dataset, cfg.predict_batch_size)
        timings[action] = seconds
    # Predict each unique chosen branch on all reference frames. This allows later
    # preselected cross-student lesson exchanges without using their labels to choose.
    for action, array in predictions.items():
        np.savez_compressed(root / f'predictions_{action}.npz', predictions=array,
                            frame_ids=evaluation.frame_id.to_numpy(str))
    choice_table.to_csv(root / 'choices.csv', index=False)
    write_json(root / 'timing.json', dict(probe_seconds=probe_seconds, adaptation_seconds=timings,
                                        total_comparison_seconds=perf_counter() - started))
    return choice_table


def deploy(cfg, student_id=None):
    """Choose using unlabeled context and cache predictions before real scoring."""
    bundle = _frozen(cfg)
    ids = [student_id] if student_id else [s['student_id'] for s in cfg.deployment_students]
    contexts = data.load_pose_manifest(cfg.root / 'data/gavd_context.csv')
    # Normally completed in prepare_gavd before array jobs, also supports local use.
    context_feature_paths(cfg, contexts, 'gavd')
    return pd.concat([_deploy_student(cfg, cfg.student(sid), bundle) for sid in ids], ignore_index=True)


def _aligned_predictions(path, frame_ids):
    payload = read_npz(path)
    ids = payload['frame_ids'].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError('Prediction frame IDs must be unique.')
    positions = {key: i for i, key in enumerate(ids)}
    missing = set(frame_ids) - positions.keys()
    if missing:
        raise ValueError(f'Prediction cache lacks {len(missing)} required reference frames.')
    return payload['predictions'][[positions[str(key)] for key in frame_ids]]


def crossover(cfg, student_id=None):
    """Apply other students' already selected lessons, without real references.

    All configured students' full-method decisions must exist first. Identical
    choices are retained. Only missing branches are trained, from each saved
    post-probe head and with exactly the original remaining training budget.
    """
    bundle = _frozen(cfg)
    all_choices = []
    for item in cfg.deployment_students:
        path = cfg.root / 'deployment' / item['student_id'] / 'choices.csv'
        if not path.is_file():
            raise FileNotFoundError(f'Complete all preselected deployments before crossover: {path}')
        all_choices.append(pd.read_csv(path).loc[lambda x: x.method.eq('full')])
    choices = pd.concat(all_choices, ignore_index=True)
    if choices.student_id.nunique() < 2:
        raise ValueError('The crossover needs at least two preselected students.')
    index = data.load_pose_manifest(cfg.root / 'data/synthetic.csv', roles='lesson')
    evaluation = data.PoseFrameDataset(cfg.root / 'data/gavd_evaluation.csv', labeled=False)
    replay = data.PoseFrameDataset(cfg.root / 'data/replay.csv')
    ids = [student_id] if student_id else [s['student_id'] for s in cfg.deployment_students]
    results = []
    for sid in ids:
        cfg.student(sid)
        root = cfg.root / 'deployment' / sid
        own = choices.loc[choices.student_id.eq(sid)].set_index('domain_id')
        exchanges = []
        for _, other in choices.loc[~choices.student_id.eq(sid)].iterrows():
            if other.domain_id not in own.index:
                raise ValueError('Crossover students must share the preselected deployment collections.')
            exchanges.append(dict(student_id=sid, other_student_id=other.student_id,
                                  domain_id=other.domain_id, own_action=own.loc[other.domain_id, 'action'],
                                  other_action=other.action, budget=int(other.budget)))
        exchange = pd.DataFrame(exchanges)
        missing = [action for action in exchange.other_action.unique()
                   if not (root / f'predictions_{action}.npz').exists()]
        if missing:
            item = cfg.student(sid)
            estimator = load_estimator(student_spec(item), cfg.device)
            estimator.load_head(root / 'probe_head.pt')
            probed = estimator.head_state()
            lr = float(item.get('learning_rate', cfg.learning_rate))
            for action in missing:
                estimator.load_head_state(probed)
                lesson = None if action == 'replay' else data.PoseFrameDataset(index.loc[index.lesson_id.eq(action)])
                adapt(estimator, replay, lesson, cfg, bundle['selection']['budget'], lr, seed=cfg.seed + 100)
                pred = predict_dataset(estimator, evaluation, cfg.predict_batch_size)
                np.savez_compressed(root / f'predictions_{action}.npz', predictions=pred,
                                    frame_ids=evaluation.index.frame_id.to_numpy(str))
        exchange.to_csv(root / 'crossovers.csv', index=False)
        results.append(exchange)
    return pd.concat(results, ignore_index=True)


def evaluate(cfg, split='early'):
    """Score frozen decisions on human references without modifying any model."""
    if split not in {'early', 'confirmation'}:
        raise ValueError('Evaluation split must be early or confirmation.')
    bundle = _frozen(cfg)
    references = import_gavd_annotations(cfg, split=split)
    dataset = data.PoseFrameDataset(references)
    targets, visible, _ = reference_arrays(dataset)
    # Independent annotated boxes determine score scale, never model crop geometry.
    reference_boxes = references[[f'reference_box_{key}' for key in ('x1', 'y1', 'x2', 'y2')]].to_numpy(float)
    root = cfg.root / 'evaluation' / split
    root.mkdir(parents=True, exist_ok=True)
    tables, intervals, damage_rows, crossover_rows = [], [], [], []
    missing = [s['student_id'] for s in cfg.deployment_students
               if not (cfg.root / 'deployment' / s['student_id'] / 'choices.csv').exists()]
    if missing:
        raise FileNotFoundError(f'Complete deployment for the configured roster before scoring: {missing}')
    for item in cfg.deployment_students:
        saved = cfg.root / 'deployment' / item['student_id']
        choices = pd.read_csv(saved / 'choices.csv')
        predictions = {action: _aligned_predictions(saved / f'predictions_{action}.npz', references.frame_id.tolist())
                       for action in choices.action.unique()}
        original_error, _, _ = landmark_errors(predictions['original'], targets, visible, reference_boxes,
                                               cfg.missing_prediction_penalty)
        for method, method_choices in choices.groupby('method', sort=True):
            arrays = np.full_like(targets, np.nan)
            for _, choice in method_choices.iterrows():
                mask = references.domain_id.eq(choice.domain_id).to_numpy()
                arrays[mask] = predictions[choice.action][mask]
            scores = frame_scores(arrays, targets, visible, reference_boxes, references,
                                  missing_penalty=cfg.missing_prediction_penalty)
            scores['method'], scores['student_id'], scores['student_role'] = method, item['student_id'], item['role']
            tables.append(scores)
            error, _, _ = landmark_errors(arrays, targets, visible, reference_boxes, cfg.missing_prediction_penalty)
            for row_pos, record in references.iterrows():
                accurate = visible[row_pos] & (original_error[row_pos] <= cfg.accurate_joint_threshold)
                damage_rows.append(dict(student_id=item['student_id'], method=method,
                    recording_id=record.recording_id, frame_id=record.frame_id,
                    initially_accurate=int(accurate.sum()),
                    error_change_on_accurate=float(np.mean(error[row_pos, accurate] - original_error[row_pos, accurate])) if accurate.any() else np.nan,
                    tail_nle=float(np.nanquantile(error[row_pos], .9)) if visible[row_pos].any() else np.nan))
        student_scores = pd.concat([t for t in tables if t.student_id.iloc[0] == item['student_id']], ignore_index=True)
        for baseline in dict.fromkeys([bundle['selection']['primary_comparator'], 'source_progress',
                                       'source_progress_matched', 'full_replay', 'pooled']):
            interval = paired_recording_interval(student_scores, 'full', baseline,
                                                  samples=cfg.bootstrap_samples, seed=cfg.seed)
            intervals.append(dict(student_id=item['student_id'], student_role=item['role'], **interval))
        if (saved / 'crossovers.csv').exists():
            exchanges = pd.read_csv(saved / 'crossovers.csv')
            for _, exchange in exchanges.iterrows():
                selected_rows = references.loc[references.domain_id.eq(exchange.domain_id)]
                positions = selected_rows.index.to_numpy()
                errors = {}
                for name, action in [('own', exchange.own_action), ('other', exchange.other_action)]:
                    if action not in predictions:
                        predictions[action] = _aligned_predictions(saved / f'predictions_{action}.npz', references.frame_id.tolist())
                    error_table = frame_scores(predictions[action][positions], targets[positions], visible[positions],
                        reference_boxes[positions], selected_rows, missing_penalty=cfg.missing_prediction_penalty)
                    errors[name] = error_table.groupby('recording_id').nle.mean()
                for recording_id in errors['own'].index:
                    crossover_rows.append(dict(student_id=item['student_id'], other_student_id=exchange.other_student_id,
                        domain_id=exchange.domain_id, recording_id=recording_id,
                        own_action=exchange.own_action, other_action=exchange.other_action,
                        own_error=float(errors['own'][recording_id]), other_error=float(errors['other'][recording_id]),
                        own_advantage=float(errors['other'][recording_id] - errors['own'][recording_id]),
                        identical_choice=bool(exchange.own_action == exchange.other_action)))
    if not tables:
        raise FileNotFoundError('No completed deployment predictions. Execute notebook 05 first.')
    frames = pd.concat(tables, ignore_index=True)
    recordings = frames.groupby(['student_id', 'student_role', 'method', 'domain_id', 'recording_id'], as_index=False).agg(
        nle=('nle', 'mean'), pixel_error=('pixel_error', 'mean'), failed_joints=('failed_joints', 'sum'),
        visible_joints=('visible_joints', 'sum'))
    summary = recordings.groupby(['student_id', 'student_role', 'method'], as_index=False).agg(
        nle=('nle', 'mean'), pixel_error=('pixel_error', 'mean'), n_recordings=('recording_id', 'nunique'))
    frames.to_csv(root / 'frame_scores.csv', index=False)
    recordings.to_csv(root / 'recording_scores.csv', index=False)
    summary.to_csv(root / 'summary.csv', index=False)
    pd.DataFrame(intervals).to_csv(root / 'paired_intervals.csv', index=False)
    pd.DataFrame(damage_rows).to_csv(root / 'accurate_joint_damage.csv', index=False)
    if crossover_rows:
        pd.DataFrame(crossover_rows).to_csv(root / 'crossover_recording_scores.csv', index=False)
    write_json(root / 'scope.json', dict(split=split, primary_comparator=bundle['selection']['primary_comparator'],
                n_recordings=int(references.recording_id.nunique()), n_collections=int(references.domain_id.nunique()),
                reference='independent visible 2D landmarks', uncertainty='recording conditional on these students and selections',
                context_kind=cfg.context_kind,
                crossover_students=sorted({r['student_id'] for r in crossover_rows}),
                configured_students=[s['student_id'] for s in cfg.deployment_students]))
    return dict(summary=summary, intervals=pd.DataFrame(intervals), recordings=recordings)


def report(cfg):
    """Plot measured source outcomes and selection regret; never synthesize gains."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source = cfg.root / 'selectors/source_outcomes.csv'
    if not source.exists():
        raise FileNotFoundError('Fit source selectors before making the source report.')
    output = cfg.root / 'reports'
    output.mkdir(parents=True, exist_ok=True)
    outcomes = pd.read_csv(source)
    validation = pd.read_csv(cfg.root / 'selectors/validation_summary.csv')
    selected = json.loads((cfg.root / 'selectors/selection.json').read_text())
    budget = selected['budget']
    matrix = outcomes.loc[outcomes.budget.eq(budget)].pivot_table(index='student_id', columns='action', values='error')
    fig, ax = plt.subplots(figsize=(max(7, .85 * len(matrix.columns)), max(3, .65 * len(matrix))))
    picture = ax.imshow(matrix.to_numpy(), aspect='auto', cmap='viridis_r')
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title(f'Measured source reference error after {budget} updates')
    fig.colorbar(picture, ax=ax, label='Normalized landmark error')
    fig.tight_layout()
    fig.savefig(output / 'source_errors.svg')
    plt.close(fig)
    domain_errors = outcomes.loc[outcomes.budget.eq(budget)].pivot_table(
        index=['split', 'student_id', 'domain_id'], columns='action', values='error')
    lesson_names = selected['lessons']
    domain_gains = domain_errors[lesson_names].rsub(domain_errors.replay, axis=0)
    domain_gains.to_csv(output / 'lesson_gains_by_student_and_domain.csv')
    choices = domain_errors[['replay', *lesson_names]].idxmin(axis=1).rename('best_action').reset_index()
    choices.to_csv(output / 'source_lesson_preferences.csv', index=False)
    variation = choices.groupby(['split', 'domain_id'], as_index=False).agg(
        n_students=('student_id', 'nunique'), n_preferred_lessons=('best_action', 'nunique'))
    variation.to_csv(output / 'source_preference_variation.csv', index=False)
    fig, ax = plt.subplots(figsize=(11, max(4, .24 * len(domain_gains))))
    limit = max(float(np.abs(domain_gains.to_numpy()).max()), 1e-8)
    picture = ax.imshow(domain_gains.to_numpy(), aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(lesson_names)), lesson_names, rotation=45, ha='right')
    labels = [f'{student}: {domain}' for _, student, domain in domain_gains.index]
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_title('Lesson gain beyond equal-budget replay, by student and setting')
    fig.colorbar(picture, ax=ax, label='Replay error minus lesson error (positive helps)')
    fig.tight_layout()
    fig.savefig(output / 'source_lesson_gains.svg')
    plt.close(fig)
    best = validation.loc[validation.budget.eq(budget)].sort_values('error').groupby('view', as_index=False).first().sort_values('regret')
    fig, ax = plt.subplots(figsize=(8, max(3, .35 * len(best))))
    ax.barh(best.view, best.regret)
    ax.set_xlabel('Source-validation selection regret (lower is better)')
    fig.tight_layout()
    fig.savefig(output / 'selection_regret.svg')
    plt.close(fig)
    lines = ['# Source experiment report', '',
             'These measurements use simulated reference labels. Real GAVD transfer is evaluated separately.', '',
             f"Selected budget: {budget}. Primary comparator: {selected['primary_comparator']}.", '',
             '![Measured source errors](source_errors.svg)', '', '![Selection regret](selection_regret.svg)', '',
             '![Lesson gains by student and domain](source_lesson_gains.svg)', '',
             'Preference differences are descriptive. Compare students on the same split/reference panel; the real crossover is a separate test.', '',
             f"Context encoder: {cfg.context_kind}. Omitted feature views: {', '.join(selected['omitted_feature_views']) or 'none'}.", '',
             'Repeated domains evaluate shared adapted checkpoints; they are not independent student interventions.']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    return dict(selection=selected, source_matrix=matrix, validation=best, report=str(output / 'README.md'))
