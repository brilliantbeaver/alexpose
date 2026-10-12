"""Reproduce this writeup from local evidence, without modifying source outputs.

Run from gavd6: MPLCONFIGDIR=/tmp/stv2-mpl .venv/bin/python \
  docs/studies/synthetic-training-v2/writeups/analyze_runs.py
Seeds remain fitted repeats of the same eight people. Intervals resample people
after averaging seeds, conditional on these three fits; they are exploratory.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / 'analysis'
SOURCE = ROOT / 'outputs/full-runs'
SEED17 = SOURCE / 'updates-2000-seed-17/updates-2000-seed-17'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/research_directions/synthetic_training_v2')]
from gavd6_sjepa.research_directions.synthetic_training_v2.evaluation import evaluate_predictions
from gavd6_sjepa.research_directions.synthetic_training_v2.workflow import reconstruct_metrics
from check_results import assert_saved_metrics_match
from expansion_results import evaluation_targets
from diagnostics.timing import timing_diagnostics

spec = importlib.util.spec_from_file_location('prior_audit', OUT.parent.parent / 'results/seed17-complete-analysis-20260919/recompute.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
METRICS = prior.METRICS + ['p95_nle', 'synthetic_all_nle', 'synthetic_occluded_nle']
KEY = ['person_id', 'motion_id', 'window_id', 'variant', 'extractor']

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    # Hash all input artifacts used or checked, including model receipts, before
    # analysis. Record missing receipts explicitly instead of assuming a copy
    # is complete or using the original absolute HAIC paths.
    files = sorted(p for p in SOURCE.rglob('*') if p.is_file())
    before = {str(p.relative_to(ROOT)): sha(p) for p in files}
    audit = {'scope': 'Local downloaded evidence; no scheduler or fresh-inference certification',
             'seeds': [17, 29, 43], 'missing': ['200-update runs', 'suite scheduler accounting',
                 'seed-17 raw neural predictions and source bundle'], 'checks': []}
    checks = audit['checks']
    expanded = pd.read_csv(SEED17 / 'exploratory-per-window.csv', low_memory=False)
    base17 = expanded.query("endpoint == 'visible' and scale_policy == 'frame_reference'").copy()
    assert len(base17) == 14 * 384
    # Recheck every seed-17 expanded metric table, including missing strata.
    check, _, _ = prior.audit_aggregates(expanded, pd.read_csv(SEED17 / 'exploratory-balanced.csv'))
    checks.append(check)
    manifest17 = read(SEED17 / 'expansion-analysis.json')
    audit['seed17_manifest_keys'] = list(manifest17)
    # Manifest contains relative output-file digests in the 'files' field.
    for key, values in manifest17.items():
        if isinstance(values, dict) and values and all(isinstance(v, str) and len(v) == 64 for v in values.values()):
            if all((SEED17 / p).is_file() for p in values):
                assert all(sha(SEED17 / p) == h for p, h in values.items())
                checks.append({'table': 'seed17 manifest ' + key, 'files': len(values), 'matched': True})
    primary = [base17]
    allvalid = [expanded.query("endpoint == 'all_valid_synthetic' and scale_policy == 'window_median_reference'").copy()]
    timing17 = pd.read_csv(SEED17 / 'exploratory-timing.csv', low_memory=False)
    timings = [timing17.query("(endpoint == 'visible' and scale_policy == 'frame_reference') or (endpoint == 'all_valid_synthetic' and scale_policy == 'window_median_reference')").copy()]
    train17 = pd.read_csv(SEED17 / 'training-summary.csv')
    assert train17.status.eq('complete').all()
    assert train17.nonfinite_loss_entries.eq(0).all()
    assert train17.recorded_phase_updates.eq(train17.planned_phase_updates).all()
    trainrows = []
    for arm, g in train17.groupby('arm'):
        row = g.iloc[-1]
        trainrows.append(dict(seed=17, arm=arm, updates=int(g.recorded_phase_updates.sum()),
            seconds=row.elapsed_seconds, status=row.status,
            online_feature_std=row.last_online_features_mean_std,
            online_effective_rank=row.last_online_features_effective_rank))
    refhashes = {}
    for seed in (29, 43):
        run = SOURCE / f'updates-2000-seed-{seed}'
        cfg = read(run / 'effective-config.json')
        assert cfg['updates'] == cfg['readout_updates'] == 2000
        assert cfg['seeds'] == [seed] and cfg['resource_contrast'] == 'matched_data_steps'
        manifest = read(run / 'data/bundle/manifest.json')
        records = pd.DataFrame(manifest['records'])
        train, dev = records[records.split.eq('train')], records[records.split.eq('development')]
        assert len(train) == 576 and len(dev) == 384
        assert train.canonical_person_id.nunique() == 24 and dev.canonical_person_id.nunique() == 8
        assert not set(train.canonical_person_id) & set(dev.canonical_person_id)
        assert set(train.extractor) == {'hrnet_w32', 'rtmpose_m'}
        assert set(train.variant) == {'clean', 'blur', 'obstruction'}
        assert set(dev.variant) == {'clean', 'blur', 'obstruction', 'blur_obstruction'}
        refhashes[str(seed)] = {name: sha(run / 'data/bundle' / name) for name in ['inputs.npz', 'targets.npz', 'manifest.json']}
        receipts = {}
        for path in sorted((run / 'receipts').glob('*.json')):
            r = read(path)
            missing, mismatches = [], []
            for name, digest in r['files'].items():
                local = run / name
                if not local.is_file():
                    missing.append(name)
                elif sha(local) != digest:
                    mismatches.append(name)
            assert not mismatches, (path, mismatches)
            receipts[path.stem] = {'listed': len(r['files']), 'missing': missing, 'mismatches': mismatches}
        checks.append({'seed': seed, 'receipts': receipts})
        reconstructed = reconstruct_metrics(run)
        assert_saved_metrics_match(run / 'evaluation/per-window.csv', reconstructed)
        assert len(reconstructed) == 12 * 384
        checks.append({'seed': seed, 'raw_prediction_metric_rows_reconstructed': len(reconstructed), 'matched_saved': True})
        # Repeated baselines and reference masks verify cross-download alignment.
        for method in ('unchanged', 'filter0', 'filter1', 'filter2'):
            a = reconstructed[reconstructed.method.eq(method)].set_index(KEY).sort_index()
            b = base17[base17.method.eq(method)].set_index(KEY).sort_index()
            assert a.index.equals(b.index)
            for col in METRICS + ['visible_count', 'displacement_count', 'amplitude_count']:
                np.testing.assert_allclose(a[col], b[col], rtol=1e-11, atol=1e-13, equal_nan=True)
        # Calibration is deterministic and shared, not independently fitted by seed.
        cal = base17[base17.method.isin(['joint_offset', 'joint_affine'])].copy()
        cal['seed'] = seed
        primary.append(pd.concat([reconstructed, cal], ignore_index=True))
        for arm in cfg['arms']:
            fit = read(run / f'fits/{arm}-{seed}/training.json')
            h = pd.DataFrame(fit['history'])
            planned = 2000 if arm == 'initialized' else 4000
            assert fit['status'] == 'complete' and fit['optimizer_updates'] == fit['planned_updates'] == planned
            assert len(h) == planned and np.isfinite(h[['loss', 'gradient_norm']]).all().all()
            assert h['update'].tolist() == list(range(1, planned + 1))
            features = fit['history'][-1].get('online_features', {})
            trainrows.append(dict(seed=seed, arm=arm, updates=planned, seconds=fit['elapsed_seconds'],
                status=fit['status'], online_feature_std=features.get('mean_std'),
                online_effective_rank=features.get('effective_rank')))
        for path in sorted((run / 'predictions').glob('*.npz')):
            meta = read(path.with_suffix('.json'))
            with np.load(path, allow_pickle=False) as z:
                targets = dict(xy=z['target_xy'], valid=z['target_valid'], visible=z['target_visible'], eval_scale=z['eval_scale'])
                for endpoint, scale in [('visible', 'frame_reference'), ('all_valid_synthetic', 'window_median_reference')]:
                    target = evaluation_targets(targets, meta['records'], endpoint, scale)
                    if endpoint == 'all_valid_synthetic':
                        frame = evaluate_predictions(z['prediction'], target, z['timestamps'], meta['records'], method=meta['method'], evidence_status=meta['evidence_status'])
                        allvalid.append(frame.assign(endpoint=endpoint, scale_policy=scale))
                    t = timing_diagnostics(z['prediction'], target, z['timestamps'], meta['records'], method=meta['method'], evidence_status=meta['evidence_status'])
                    timings.append(t.assign(endpoint=endpoint, scale_policy=scale))
                    if meta['method'] == 'unchanged':
                        t = timing_diagnostics(target['xy'], target, z['timestamps'], meta['records'], method='reference_oracle', evidence_status=meta['evidence_status'])
                        timings.append(t.assign(endpoint=endpoint, scale_policy=scale))
        print(f'Verified seed {seed}: predictions, panel, training and receipt hashes', flush=True)
    assert refhashes['29'] == refhashes['43']
    # Compare available shared-bundle digests against seed-17 recorded provenance.
    sourcehash = read(SEED17 / 'analysis.json')['source_sha256']
    assert all(sourcehash['data/bundle/' + name] == digest for name, digest in refhashes['29'].items())
    audit['shared_bundle_hashes'] = refhashes
    audit['seed17_source_bundle_hashes'] = {k: v for k, v in sourcehash.items() if k.startswith('data/bundle/')}
    primary = pd.concat(primary, ignore_index=True)
    people, summary = prior.balanced(primary, METRICS)
    people.to_csv(OUT / 'per-person-primary.csv', index=False)
    summary.to_csv(OUT / 'per-seed-primary.csv', index=False)
    for seed in (29, 43):
        saved = pd.read_csv(SOURCE / f'updates-2000-seed-{seed}/evaluation/per-person-balanced-summary.csv')
        rebuilt = summary[summary.seed.eq(seed) & summary.method.isin(saved.method) & summary.metric.isin(saved.metric)]
        checks.append(prior.compare_values(rebuilt, saved, [*prior.STRATA, 'metric'], ['value'], f'seed{seed} primary summary'))
    # Each condition/window/motion is balanced before averaging fitted seeds.
    combined = summary.groupby(['method', 'extractor', 'metric']).value.agg(['mean', 'min', 'max']).reset_index()
    combined.to_csv(OUT / 'three-seed-summary.csv', index=False)
    conditions = []
    for condition, group in primary.groupby('variant'):
        _, table = prior.balanced(group, METRICS)
        conditions.append(table.assign(condition=condition))
    pd.concat(conditions).to_csv(OUT / 'per-seed-conditions.csv', index=False)
    avg = people.groupby(['method', 'extractor', 'person_id'])[METRICS].mean().reset_index()
    rng = np.random.default_rng(20260920)
    draws = rng.integers(8, size=(50000, 8))
    contrast = []
    pairs = [('direct', x) for x in ['unchanged', 'joint_affine', 'static', 'filter2', 'paired_jepa']]
    pairs += [('paired_jepa', x) for x in ['coordinate', 'initialized', 'ordinary_jepa', 'shuffled_jepa']]
    for metric in ['visible_nle', 'displacement_nle']:
        for extractor, group in avg.groupby('extractor'):
            wide = group.pivot(index='person_id', columns='method', values=metric).sort_index()
            assert len(wide) == 8
            for candidate, comparator in pairs:
                a, b = wide[candidate].to_numpy(), wide[comparator].to_numpy()
                effects = 100 * (1 - a[draws].mean(1) / b[draws].mean(1))
                lo, hi = np.quantile(effects, [.025, .975])
                contrast.append(dict(metric=metric, extractor=extractor, candidate=candidate, comparator=comparator,
                    reduction_percent=100*(1-a.mean()/b.mean()), ci95_low=lo, ci95_high=hi,
                    people_improved=int((a < b).sum()), people=8, seed_count=3, bootstrap_draws=50000,
                    uncertainty='pointwise person bootstrap after averaging three fitted seeds; no multiplicity or training-selection adjustment'))
    pd.DataFrame(contrast).to_csv(OUT / 'paired-contrasts.csv', index=False)
    av = pd.concat(allvalid, ignore_index=True)
    _, avsum = prior.balanced(av, METRICS)
    avsum.to_csv(OUT / 'all-valid-fixed-scale.csv', index=False)
    # Descriptive event counts remain separated by seed; never pool duplicated
    # reference events to imply a larger study population.
    timing = pd.concat(timings, ignore_index=True)
    ts = []
    for labels, group in timing.groupby(['seed', 'method', 'extractor', 'endpoint', 'scale_policy']):
        ts.append(dict(zip(['seed', 'method', 'extractor', 'endpoint', 'scale_policy'], labels)) | prior.timing_group(group))
    pd.DataFrame(ts).to_csv(OUT / 'timing.csv', index=False)
    pd.DataFrame(trainrows).to_csv(OUT / 'training.csv', index=False)
    assert sum(r['updates'] for r in trainrows) == 90000
    audit['training_updates'] = 90000
    audit['analysis_code_sha256'] = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    audit['input_hashes'] = before
    assert all(sha(ROOT / name) == digest for name, digest in before.items()), 'Source artifacts changed'
    audit['source_artifacts_unchanged'] = True
    (OUT / 'verification.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(combined.query("metric == 'visible_nle'").pivot(index='method', columns='extractor', values='mean').round(6).to_string())
    print('Wrote verified analysis to', OUT)

if __name__ == '__main__':
    main()
