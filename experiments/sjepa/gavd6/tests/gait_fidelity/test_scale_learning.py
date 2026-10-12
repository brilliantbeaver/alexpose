"""Scale paths exercise the same tensors as legacy fixtures without full loads."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from gavd6_sjepa.research_directions.gait_fidelity.data import (
    fixture_bundle, save_dataset, load_dataset, merge_disk_datasets, select_rows, IndexedArray,
)
from gavd6_sjepa.research_directions.gait_fidelity.training import (
    paired_indices, hierarchical_pair_groups, draw_hierarchical_pairs, train_phase,
)
from gavd6_sjepa.research_directions.gait_fidelity.evaluation import (
    fit_calibration, apply_calibration, calibration_row_weights, load_predictions, summarize_predictions,
)


class ScaleLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.bundle = fixture_bundle(samples=16, people=4)

    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_mmap_roundtrip_lazy_subset_and_stream_merge(self):
        bundle = self.bundle
        paths = []
        for split in ('train', 'development'):
            paths.append(save_dataset(bundle.subset(split), self.root / split, storage='npy'))
        merged = merge_disk_datasets(paths, self.root / 'merged')
        loaded = load_dataset(merged)
        self.assertIsInstance(loaded.inputs['xy'], np.memmap)
        subset = loaded.subset('train')
        self.assertIsInstance(subset.inputs['xy'], IndexedArray)
        np.testing.assert_array_equal(subset.inputs['xy'][:5], bundle.subset('train').inputs['xy'][:5])
        np.testing.assert_array_equal(loaded.targets['xy'], bundle.targets['xy'])
        np.testing.assert_array_equal(subset.inputs['xy'][[2, 1], :, 6:8], bundle.inputs['xy'][[2, 1], :, 6:8])
        path = merged / 'inputs-xy.npy'
        with path.open('r+b') as f:
            f.seek(-1, 2); f.write(b'X')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            load_dataset(merged)

    def test_hierarchy_does_not_overweight_prolific_people_or_motions(self):
        records = []
        # Three motions for person A, one for B; the first A motion has ten windows.
        for person, counts in [('A', [10, 1, 1]), ('B', [1])]:
            for motion, windows in enumerate(counts):
                for window in range(windows):
                    records.append(dict(canonical_person_id=person, motion_hash=f'{person}-{motion}', source_family_id=f'{person}-{motion}-{window}'))
        pairs = np.column_stack([np.arange(len(records)), np.arange(len(records))])
        draws = draw_hierarchical_pairs(hierarchical_pair_groups(records, pairs), 30000, np.random.default_rng(17))
        person_a = np.mean([records[i]['canonical_person_id'] == 'A' for i in draws])
        motion_a0 = np.mean([records[i]['motion_hash'] == 'A-0' for i in draws])
        self.assertAlmostEqual(person_a, .5, delta=.015)
        self.assertAlmostEqual(motion_a0, 1/6, delta=.012)
        weights = calibration_row_weights(records)
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertAlmostEqual(sum(w for r,w in zip(records,weights) if r['canonical_person_id']=='A'), .5)

    def test_streamed_calibration_matches_batch_sizes_and_is_training_only(self):
        train = self.bundle.subset('train')
        a = fit_calibration(train, balanced=True, batch_size=1)
        b = fit_calibration(train, balanced=True, batch_size=37)
        np.testing.assert_allclose(a['offset'], b['offset'], atol=1e-10)
        np.testing.assert_allclose(a['affine'], b['affine'], atol=1e-10)
        with self.assertRaisesRegex(ValueError, 'training people only'):
            fit_calibration(self.bundle.subset('development'))
        raw = {k:v[:3] for k,v in self.bundle.inputs.items()}
        out = apply_calibration(raw, a)
        self.assertTrue(np.isnan(out['joint_offset'][~raw['observed']]).all())

    def test_core_training_disk_predictions_and_confirmation_rejection(self):
        cfg = dict(device='cpu', fixture=False,
            model=dict(width=8, encoder_layers=1, predictor_layers=1, heads=2, patch_size=4, window_size=16),
            training=dict(sampling='person_motion', end_to_end_updates=1, batch_size=4, learning_rate=.0003),
            measurement=dict(min_frames=8, min_coverage=.5, min_segment_px=2.))
        recipe = dict(encoder='direct', pretraining_mask=None, readout_or_training_objective='paired_change', recipe_id='direct')
        receipt = train_phase(self.bundle, recipe, 'end_to_end', 17, cfg, self.root/'model')
        prediction, indices = load_predictions(receipt)
        self.assertIsInstance(prediction, np.memmap)
        self.assertEqual(len(prediction), len(indices))
        self.assertTrue(all(self.bundle.records[i]['split']=='development' for i in indices))
        bad = deepcopy(self.bundle); bad.records[0]['split'] = 'confirmation'
        with self.assertRaisesRegex(PermissionError, 'cannot enter'):
            train_phase(bad, recipe, 'end_to_end', 17, cfg, self.root/'forbidden')
        recipe['readout_or_training_objective'] = 'repaired_change'
        with self.assertRaisesRegex(ValueError, 'matched_cycles'):
            train_phase(self.bundle, recipe, 'end_to_end', 17, cfg, self.root/'wrong-control')
        recipe['readout_or_training_objective']='base'
        cfg['training'].update(sampling='matched_cycles',repaired_trials=1)
        full_receipt=train_phase(self.bundle,recipe,'end_to_end',17,cfg,self.root/'matched-source')
        self.assertTrue(full_receipt['predictions'].endswith('.npy'))
        self.assertIsInstance(load_predictions(full_receipt)[0],np.memmap)

    def test_streamed_evaluation_retains_missing_predictions_and_person_coverage(self):
        dev = self.bundle.subset('development')
        truth = np.asarray(dev.targets['xy'])
        failed = truth.copy(); failed[0,0,8] = np.nan
        cfg = dict(measurement=dict(min_frames=8, min_coverage=.5, min_segment_px=2., sign_tolerance_deg=1.),
                   evaluation=dict(primary_candidate='oracle', primary_comparator='failed', bootstrap_draws=30, bootstrap_seed=17))
        person, coverage, comparisons = summarize_predictions(dev, [('oracle',17,truth),('failed',17,failed)], cfg, self.root/'eval')
        self.assertEqual(coverage.loc[coverage.method.eq('failed'),'successful'].iloc[0], len(dev.records)-1)
        self.assertGreater(person.loc[person.method.eq('failed'),'A_error'].max(), 0.)
        self.assertEqual(comparisons['A_error']['people'], 2)
        self.assertTrue((self.root/'eval/coverage-per-person.csv').is_file())


if __name__ == '__main__': unittest.main()
