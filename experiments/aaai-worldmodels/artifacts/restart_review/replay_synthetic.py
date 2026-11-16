"""Small complete reviewer replay, using synthetic data only."""
from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import worldmodels.restart_experiment as module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    scratch = Path(__file__).parent
    prepared = scratch / 'synthetic_input'
    output = scratch / 'synthetic_output'
    prepared.mkdir(exist_ok=True)
    rng = np.random.default_rng(91)
    rows = [dict(source_id=f'{label}{source}', sequence_id=f'{label}{source}seq{sequence}',
                 window_id=f'{label}{source}seq{sequence}window0', label=label, prefix_valid_fraction=1.0)
            for label in ('normal', 'stroke') for source in range(5) for sequence in range(2)]
    meta = pd.DataFrame(rows)
    x = rng.normal(size=(len(meta), 24, 16, 2)).astype(np.float32)
    valid = np.ones(x.shape[:-1], bool)
    np.savez_compressed(prepared/'windows.npz', x=x, valid=valid)
    meta.to_csv(prepared/'windows.csv', index=False)
    (prepared/'preparation.json').write_text(json.dumps(dict(frames=24, context_frames=16, sample_fps=10.,
        data_sha256=digest(prepared/'windows.npz'), metadata_sha256=digest(prepared/'windows.csv'))))
    with threadpool_limits(limits=1):
        result = module.run(prepared, output)
    membership = json.loads((output/'fold_membership.json').read_text())
    for fold in membership:
        assert not set(fold['train_sources']) & set(fold['heldout_sources'])
        assert all(source.startswith('normal') for source in fold['train_sources'])
        for inner in fold['inner']:
            assert not set(inner['train']) & set(inner['validation'])
            assert set(inner['train'] + inner['validation']) == set(fold['train_sources'])
    cohort = pd.read_csv(output/'cohort.csv')
    stored = np.load(output/'forecast_predictions.npz', allow_pickle=False)
    checkpoints = 0
    for path in (output/'models').glob('*.joblib'):
        checkpoint = joblib.load(path)
        fold = int(path.name.split('_')[0][4:])
        mask = cohort.outer_fold.to_numpy() == fold
        pred = module.predict_checkpoint(checkpoint, x[mask], valid[mask], cohort.loc[mask].window_id.tolist())
        arm = checkpoint['arm'] + ('_shrunk' if '_shrunk_' in path.name else '')
        leg = checkpoint['leg']
        np.testing.assert_allclose(pred, stored[arm][mask, :, leg*4:(leg+1)*4], atol=1e-6)
        changed = x[mask].copy()
        changed[:, 16:] = 10000
        changed_valid = valid[mask].copy()
        changed_valid[:, 16:] = False
        np.testing.assert_array_equal(pred, module.predict_checkpoint(checkpoint, changed, changed_valid,
                                                                     cohort.loc[mask].window_id.tolist()))
        checkpoints += 1
    assert checkpoints == 80
    errors = pd.read_csv(output/'window_predictions.csv')
    sequence = errors.groupby(['source_id', 'sequence_id', 'label', 'fold', 'arm', 'horizon'])[['mse','left_mse','right_mse']].mean().reset_index()
    source = sequence.groupby(['source_id', 'label', 'fold', 'arm', 'horizon'])[['mse','left_mse','right_mse']].mean().reset_index()
    pd.testing.assert_frame_equal(source, pd.read_csv(output/'source_scores.csv'), check_exact=False, atol=1e-12, rtol=1e-12)
    for relative, expected in json.loads((output/'artifact_hashes.json').read_text()).items():
        assert digest(output/relative) == expected
    rejected = scratch/'synthetic_bad_identity'
    prep = json.loads((prepared/'preparation.json').read_text())
    prep['data_sha256'] = 'bad'
    (prepared/'preparation.json').write_text(json.dumps(prep))
    try:
        module.run(prepared, rejected)
    except ValueError as exc:
        assert 'hash mismatch' in str(exc)
    else:
        raise AssertionError('Corrupt preparation identity was accepted')
    record = dict(status='passed', synthetic_only=True, windows=len(meta), sources=int(meta.source_id.nunique()),
                  folds=5, checkpoints_replayed=checkpoints, seconds=result['seconds'],
                  code_sha256=digest(module.__file__), audit_script_sha256=digest(__file__),
                  checks=['all source and nested folds isolated', 'normal-only training membership',
                          'all 80 saved checkpoint predictions replayed', 'hidden future invariance at checkpoint inference',
                          'source aggregation replayed', 'all output manifest hashes verified', 'corrupt input identity rejected'])
    (scratch/'synthetic_replay.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
