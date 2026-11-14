#!/usr/bin/env python3
"""Read-only audit of retained experiments, independent of sjepa scoring code.

Run from any directory with the experiment's Python environment:
  .venv/bin/python docs/wacv2027-v1/review-2026-10-15/evidence/verify_results.py

No encoder or classifier is fitted. Source-bootstrap intervals condition on the
saved predictions. NumPy is used for resampling/arrays; PyTorch only reads local
trusted checkpoints and compares tensors. The only imported project object is
the feature-name list, not its feature extractor or evaluation functions.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'sjepa').is_dir())
sys.path.insert(0, str(ROOT.parents[1]))
from ambient.classification.features import GaitFeatureVector

LABELS = ['normal', 'ms', 'pd']
SYSTEMS = ['sjepa', 'rf', 'mean_pose', 'visibility', 'majority']
OUT = Path(__file__).resolve().with_name('results-verification.json')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def metrics(rows, pred_key, equal_source):
    counts = Counter(row['source_id'] for row in rows)
    cm = [[0.0] * 3 for _ in LABELS]
    for row in rows:
        assert row['true'] in LABELS and row[pred_key] in LABELS
        cm[LABELS.index(row['true'])][LABELS.index(row[pred_key])] += (
            1 / counts[row['source_id']] if equal_source else 1)
    per_class = {}
    for i, label in enumerate(LABELS):
        tp, support, predicted = cm[i][i], sum(cm[i]), sum(row[i] for row in cm)
        per_class[label] = {
            'precision': tp / predicted if predicted else 0,
            'recall': tp / support if support else 0,
            'f1': 2 * tp / (support + predicted) if support + predicted else 0,
            'support': support,
        }
    return {
        'confusion': cm,
        'per_class': per_class,
        'accuracy': sum(cm[i][i] for i in range(3)) / sum(map(sum, cm)),
        **{'macro_' + key: statistics.mean(v[key] for v in per_class.values())
           for key in ['precision', 'recall', 'f1']},
        'labels': LABELS,
    }


def close_tree(actual, expected, path=''):
    """Check every retained scalar, class statistic, and confusion cell."""
    if isinstance(expected, dict):
        for key in expected:
            close_tree(actual[key], expected[key], path + '/' + key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            close_tree(a, b, path + '/' + str(i))
    elif isinstance(expected, (float, int)):
        assert abs(actual - expected) < 1e-12, (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def bootstrap(rows):
    """Paired label-stratified source resampling, fixed fitted predictions.

    Exactly 10,000 draws. Sources are lexicographically sorted; PCG64 seed 42.
    All 10,000 draws for Normal are made first, then MS, then PD. Each source's
    clips together carry weight one per draw; repeated sources repeat that mass.
    Percentiles use NumPy's default linear interpolation.
    """
    sources = sorted({r['source_id'] for r in rows})
    counts = Counter(r['source_id'] for r in rows)
    labels = {r['source_id']: r['true'] for r in rows}
    assert all(labels[r['source_id']] == r['true'] for r in rows)
    per_source = np.zeros((len(sources), len(SYSTEMS), 3, 3))
    for row in rows:
        for m, name in enumerate(SYSTEMS):
            per_source[sources.index(row['source_id']), m,
                       LABELS.index(row['true']), LABELS.index(row['pred_' + name])] += 1 / counts[row['source_id']]
    rng = np.random.default_rng(42)
    sampled = np.zeros((10000, len(SYSTEMS), 3, 3))
    for label in LABELS:
        ix = [i for i, source in enumerate(sources) if labels[source] == label]
        sampled += per_source[rng.choice(ix, size=(10000, len(ix)), replace=True)].sum(axis=1)
    denominator = sampled.sum(axis=-1) + sampled.sum(axis=-2)
    numerator = 2 * np.diagonal(sampled, axis1=-2, axis2=-1)
    values = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0).mean(axis=-1)
    return {
        'replicates': 10000, 'seed': 42, 'rng': 'NumPy default_rng PCG64',
        'source_order': sources, 'class_draw_order': LABELS,
        'draw_layout': 'all replicates for each class before moving to next class',
        'interval_type': '2.5th/97.5th percentile, linear interpolation; conditional on fixed predictions',
        'source_macro_f1_intervals': {name: np.percentile(values[:, i], [2.5, 97.5]).tolist()
                                       for i, name in enumerate(SYSTEMS)},
        'sjepa_minus_rf_interval': np.percentile(values[:, 0] - values[:, 1], [2.5, 97.5]).tolist(),
        'sjepa_minus_mean_pose_interval': np.percentile(values[:, 0] - values[:, 2], [2.5, 97.5]).tolist(),
    }


registry_path = ROOT / 'artifacts/eval/full-v1/fold_registry.json'
registry = read(registry_path)
records = {r['clip']: r for r in registry['inventory']['cache']}
expected_folds = {clip: f['fold'] for f in registry['folds'] for clip in f['test_clips']}
assert len(records) == len(expected_folds) == 88
assert len({r['source_id'] for r in records.values()}) == 41
assert all(sha(ROOT / 'artifacts/keypoints-full' / r['cache_file']) == r['sha256'] for r in records.values())

splits = []
for fold in registry['folds']:
    partitions = {}
    for partition in ['train', 'validation', 'test']:
        recs = [records[c] for c in fold[partition + '_clips']]
        sources = {r['source_id'] for r in recs}
        assert sources == set(fold[partition + '_sources'])
        partitions[partition] = {
            'clips': len(recs), 'sources': len(sources),
            'class_clips': dict(Counter(r['label'] for r in recs)),
            'class_sources': {lab: len({r['source_id'] for r in recs if r['label'] == lab}) for lab in LABELS},
            'windows': sum(max(1, (r['n_frames'] - 32) // 16 + 1) for r in recs),
        }
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not set(fold[a + '_sources']) & set(fold[b + '_sources'])
    splits.append({'fold': fold['fold'], **partitions})

run_paths = sorted(p.parent for p in (ROOT / 'artifacts/runs/full-v1').glob('*/*/capstone-*/results.json'))
assert len(run_paths) == 2
runs, rows_by_run, feature_roots = {}, {}, []
for path in run_paths:
    saved, rows, provenance = read(path / 'results.json'), read(path / 'oof.json'), read(path / 'provenance.json')
    assert len(rows) == 88 and Counter(r['clip'] for r in rows) == Counter({c: 1 for c in records})
    for row in rows:
        rec = records[row['clip']]
        assert (row['true'], row['source_id'], row['fold']) == (rec['label'], rec['source_id'], expected_folds[row['clip']])
    computed = {}
    for name in SYSTEMS:
        fold_scores = [metrics([r for r in rows if r['fold'] == k], 'pred_' + name, True) for k in range(5)]
        aggregates = {key: {'mean': statistics.mean(m[key] for m in fold_scores),
                            'std': statistics.pstdev(m[key] for m in fold_scores)}
                      for key in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']}
        computed[name] = {'source_weighted': metrics(rows, 'pred_' + name, True),
                          'clip_weighted': metrics(rows, 'pred_' + name, False),
                          'source_weighted_fold_mean_std': aggregates,
                          'fold_source_macro_f1': [m['macro_f1'] for m in fold_scores],
                          'correct_clips': sum(r['true'] == r['pred_' + name] for r in rows)}
        close_tree(computed[name], saved['metrics'][name], name)
    for fold in registry['folds']:
        complete = read(path / f"fold-{fold['fold']}/complete.json")
        assert complete['value']['rows'] == [r for r in rows if r['fold'] == fold['fold']]
        assert digest(complete['value']) == complete['value_sha256']
        for name, checksum in complete['files'].items():
            assert sha(path / f"fold-{fold['fold']}" / name) == checksum
    feature_identity = {k: provenance[k] for k in ['schema', 'code', 'environment', 'inputs']}
    feature_identity['fps'] = provenance['config']['target_fps']
    feature_root = ROOT / 'artifacts/cache/capstone/features' / digest(feature_identity)
    assert feature_root.is_dir()
    feature_roots.append(feature_root)
    runs[path.name] = {
        'path': relative(path), 'oof_sha256': sha(path / 'oof.json'), 'results_sha256': sha(path / 'results.json'),
        'provenance_sha256': sha(path / 'provenance.json'), 'code_sha256': provenance['code'],
        'environment': provenance['environment'], 'config': saved['config'], 'execution': saved['execution'],
        'base_updates': saved['base_updates'], 'continuation_updates': saved['continuation_updates'],
        'selections': saved['selections'], 'metrics': computed, 'bootstrap': bootstrap(rows),
        'feature_cache': relative(feature_root), 'all_saved_metrics_match': True,
        'sjepa_fold_wins_vs_rf': sum(a > b for a, b in zip(computed['sjepa']['fold_source_macro_f1'], computed['rf']['fold_source_macro_f1'])),
        'sjepa_fold_wins_vs_mean_pose': sum(a > b for a, b in zip(computed['sjepa']['fold_source_macro_f1'], computed['mean_pose']['fold_source_macro_f1'])),
    }
    rows_by_run[path.name] = {row['clip']: row for row in rows}

names = GaitFeatureVector.get_feature_names()
left, right = names.index('left_ankle_range'), names.index('right_ankle_range')
feature_changes, duplicates, arrays_by_run = [], [0, 0], [{}, {}]
for clip in sorted(records):
    values = [dict(np.load(p / (digest(clip) + '.npz'), allow_pickle=False)) for p in feature_roots]
    for i, value in enumerate(values):
        arrays_by_run[i][clip] = value['rf']
        duplicates[i] += int(value['rf'][left] == value['rf'][right])
    for kind in ['visibility', 'mean_pose']:
        assert np.array_equal(values[0][kind], values[1][kind])
    a, b = values[0]['rf'], values[1]['rf']
    changed = np.flatnonzero(a != b).tolist()
    assert changed == [left]
    feature_changes.append({'clip': clip, 'column': left, 'name': names[left], 'before': float(a[left]), 'after': float(b[left])})
for path, values in zip(run_paths, arrays_by_run):
    runs[path.name]['rf_varying_feature_names_by_fold'] = [
        [names[i] for i in np.flatnonzero(np.stack([values[c] for c in f['train_clips']]).std(axis=0) > 0)]
        for f in registry['folds']]
    assert all(len(x) == 15 for x in runs[path.name]['rf_varying_feature_names_by_fold'])

roots = {'sjepa': ROOT / 'sjepa', 'ambient': ROOT.parents[1] / 'ambient'}
code_hashes = {f'{name}/{p.relative_to(base)}': sha(p) for name, base in roots.items()
               for p in sorted(base.rglob('*.py')) if 'tests' not in p.relative_to(base).parts}
current_code_hash = digest(code_hashes)
feature_code = (roots['ambient'] / 'classification/features.py').read_text()
correct = 'left_ankle_range=safe_get(left_ankle_stats, "range"),'
old = 'left_ankle_range=safe_get(right_ankle_stats, "range"),'
assert feature_code.count(correct) == 1
code_hashes['ambient/classification/features.py'] = hashlib.sha256(feature_code.replace(correct, old).encode()).hexdigest()
single_reversion_hash = digest(code_hashes)
assert current_code_hash == runs[run_paths[1].name]['code_sha256']
assert single_reversion_hash == runs[run_paths[0].name]['code_sha256']

model_comparisons = []
for fold in registry['folds']:
    k = fold['fold']
    for stage, updates in [('ssl', 800), ('continued', 400)]:
        paths = [p / f'fold-{k}/{stage}.pt' for p in run_paths]
        # These are trusted local experiment artifacts, not downloaded models.
        a, b = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
        for payload in [a, b]:
            context = payload['extra']['split_context']
            assert context['train_clips'] == fold['train_clips']
            assert context['train_sources'] == fold['train_sources']
            assert payload['train_state']['step'] == payload['extra']['updates_this_stage'] == updates
            assert payload['train_state']['schedule']['seed'] == 42
        comparison = {'fold': k, 'stage': stage, 'checkpoint_sha256': [sha(p) for p in paths], 'groups': {}}
        for group in ['view_encoder', 'target_encoder', 'predictor']:
            comparison['groups'][group] = {
                'tensor_count': len(a[group]),
                'exactly_equal_tensors': sum(torch.equal(a[group][n], b[group][n]) for n in a[group]),
                'maximum_absolute_difference': max((a[group][n] - b[group][n]).abs().max().item() for n in a[group]),
            }
        embeddings = [dict(np.load(p / f'fold-{k}/{stage}-embeddings.npz', allow_pickle=False)) for p in run_paths]
        comparison['embeddings'] = {part: {
            'shape': list(embeddings[0][part].shape),
            'equal': bool(np.array_equal(embeddings[0][part], embeddings[1][part])),
            'maximum_absolute_difference': float(np.max(np.abs(embeddings[0][part] - embeddings[1][part]))),
        } for part in ['train', 'validation']}
        model_comparisons.append(comparison)

old_rows, new_rows = [rows_by_run[p.name] for p in run_paths]
prediction_changes = {name: [clip for clip in sorted(records) if old_rows[clip]['pred_' + name] != new_rows[clip]['pred_' + name]] for name in SYSTEMS}

legacy_path = ROOT / 'artifacts/runs/r1_g1_1k_s42'
legacy_rows, legacy_saved = read(legacy_path / 'oof.json'), read(legacy_path / 'results.json')
legacy = {'path': relative(legacy_path), 'clips': len(legacy_rows),
          'sources': len({r['source_id'] for r in legacy_rows}), 'metrics': {}}
for name in ['sjepa', 'rf']:
    value = metrics(legacy_rows, 'pred_' + name, False)
    for key in ['macro_f1', 'accuracy', 'per_class', 'confusion']:
        close_tree(value[key], legacy_saved[name + '_pooled'][key])
    legacy['metrics'][name] = value
legacy_rf = read(ROOT / 'artifacts/eval/g1/E0_RF_oof.json')
assert len(legacy_rf) == len(legacy_rows)
assert {r['clip']: r['pred'] for r in legacy_rf} == {r['clip']: r['pred_rf'] for r in legacy_rows}
legacy['rf_identical_to_E0_predictions'] = True

nb = read(ROOT / '06_capstone_rf_vs_sjepa.ipynb')
training_cell = next((i, c) for i, c in enumerate(nb['cells']) if c.get('id') == '06_capstone_rf_vs_sjepa-08')
logs = ''.join(''.join(o.get('text', [])) for o in training_cell[1].get('outputs', []))
parsed_logs, fold, stage = [], None, None
for line in logs.splitlines():
    match = re.match(r'fold (\d+) (ssl|continued): training', line)
    if match:
        fold, stage = int(match[1]), match[2]
    match = re.match(r'step\s+(\d+) loss (\S+) lr (\S+) tau (\S+) emb_std (\S+) eff_rank (\S+)', line)
    if match:
        parsed_logs.append(dict(fold=fold, stage=stage, step=int(match[1]), loss=float(match[2]),
                                lr=float(match[3]), tau=float(match[4]), emb_std=float(match[5]), effective_rank=float(match[6])))

result = {
    'purpose': 'independent read-only reanalysis of retained results; no model fitting or source-artifact modification',
    'review_date': '2026-10-15', 'script_sha256': sha(__file__),
    'registry_path': relative(registry_path), 'registry_sha256': registry['registry_sha256'],
    'dataset_sha256': registry['dataset_sha256'], 'all_88_pose_cache_hashes_match': True,
    'splits': splits, 'runs': runs,
    'rf_correction': {'changed_column': 'left_ankle_range', 'changed_clip_count': len(feature_changes),
                      'before_after_duplicate_ankle_count': duplicates, 'details': feature_changes,
                      'current_code_hash': current_code_hash, 'single_line_reversion_code_hash': single_reversion_hash,
                      'all_other_hashed_code_unchanged': True, 'all_other_feature_values_identical': True},
    'run_comparison': {'changed_prediction_clips_by_system': prediction_changes,
                       'checkpoint_and_embedding_comparison': model_comparisons,
                       'interpretation': 'two computed same-seed executions; different tensors/embeddings, same S-JEPA hard predictions; not independent-seed replication'},
    'legacy_47_clip_generation': legacy,
    'notebook_06_training_log': {'zero_based_cell': training_cell[0], 'cell_id': training_cell[1]['id'],
                                'logged_updates_only': parsed_logs,
                                'note': 'Rounded logs every 80/40 updates, not complete loss histories. Endpoint values are in each results.json.'},
}
OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(f'PASS: all saved metrics and artifact checks; wrote {OUT}')
for name, run in runs.items():
    print(name, {m: round(v['source_weighted']['macro_f1'], 9) for m, v in run['metrics'].items()})
