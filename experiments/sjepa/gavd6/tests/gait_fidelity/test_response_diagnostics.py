import tempfile
import unittest
from pathlib import Path

import numpy as np

from gavd6_sjepa.research_directions.gait_fidelity.common import atomic_json, sha256
from gavd6_sjepa.research_directions.gait_fidelity.data import fixture_bundle
from gavd6_sjepa.research_directions.gait_fidelity.response_diagnostics import (
    analyze_feature_export, diagnostic_panel, feature_statistics, person_folds, person_ridge_probe,
    export_parent_features,
)


class ResponseDiagnosticsTests(unittest.TestCase):
    def test_panel_is_training_only_hash_selected_and_capped(self):
        bundle = fixture_bundle(samples=32, people=5, windows=4)
        panel = diagnostic_panel(bundle)
        self.assertEqual(panel, diagnostic_panel(bundle))
        people = {}
        for row in panel['rows']:
            people.setdefault(row['canonical_person_id'], set()).add(row['source_family_id'])
            for index in row['endpoint_indices']:
                self.assertEqual(bundle.records[index]['split'], 'train')
            self.assertGreater(bundle.records[row['endpoint_indices'][1]]['movement_level_deg'], 0)
        self.assertTrue(all(len(families) == 3 for families in people.values()))
        # Development targets cannot affect selection.
        dev = [index for index, row in enumerate(bundle.records) if row['split'] == 'development']
        bundle.targets['xy'][dev] = -999
        self.assertEqual(panel, diagnostic_panel(bundle))

    def test_person_folds_and_train_only_mean_loss_ridge(self):
        people = np.repeat([f'p{n}' for n in range(6)], 2)
        x = np.arange(24., dtype=float).reshape(12, 2)
        x[-2:] += 1000
        y = np.sin(np.arange(12.)) + x[:, 0] * .2
        folds = person_folds(people)
        result = person_ridge_probe(x, y, people)
        for person in set(people):
            self.assertEqual(len(set(folds[people == person])), 1)
        for fold, receipt in enumerate(result['folds']):
            self.assertFalse(set(receipt['train_people']) & set(receipt['test_people']))
            train, test = folds != fold, folds == fold
            mean, scale = x[train].mean(0), x[train].std(0)
            xt, xv = (x[train]-mean)/scale, (x[test]-mean)/scale
            coefficient = np.linalg.solve(xt.T@xt + .01*train.sum()*np.eye(2), xt.T@(y[train]-y[train].mean()))
            expected = xv@coefficient + y[train].mean()
            np.testing.assert_allclose(np.asarray(result['predictions'])[test], expected, atol=1e-8)
        with self.assertRaises(ValueError):
            person_ridge_probe(x, y, people, alpha=1)

    def test_fixed_slot_variance_does_not_count_position_embeddings_as_diversity(self):
        values = np.broadcast_to(np.arange(24.).reshape(1, 2, 4, 3), (8, 2, 4, 3)).copy()
        result = feature_statistics(values)
        self.assertEqual(result['fixed_slot_variance'], 0.)
        self.assertEqual(result['fixed_slot_delta_variance'], 0.)
        self.assertGreater(result['mean_delta_l2'], 0.)

    def test_export_integrity_and_training_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            arrays = folder/'features.npz'
            rng = np.random.default_rng(17)
            values = rng.normal(size=(6, 2, 2, 3))
            np.savez(arrays, masked_teacher=values, deployment_encoder=values,
                     no_change_teacher=values, reference_delta_A=np.arange(6.))
            metadata = dict(training_only=True, arrays=str(arrays), arrays_sha256=sha256(arrays),
                panel=dict(identity='fixed', rows=[dict(split='train', canonical_person_id=f'p{i//2}') for i in range(6)]))
            atomic_json(folder/'features.json', metadata)
            result = analyze_feature_export(folder)
            self.assertIn('masked_teacher', result['probes'])
            self.assertNotIn('no_change_teacher', result['probes'])
            metadata['panel']['rows'][0]['split'] = 'development'
            atomic_json(folder/'features.json', metadata)
            with self.assertRaises(PermissionError):
                analyze_feature_export(folder)
            metadata['panel']['rows'][0]['split'] = 'train'
            metadata['arrays_sha256'] = 'bad'
            atomic_json(folder/'features.json', metadata)
            with self.assertRaises(RuntimeError):
                analyze_feature_export(folder)

    def test_parent_export_requires_original_completion_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                export_parent_features({'work': temporary}, {'result': {}}, 'bundle', 'panel', Path(temporary)/'child')


if __name__ == '__main__':
    unittest.main()
