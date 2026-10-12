import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.gait_fidelity.data import fixture_bundle, SWAP
from gavd6_sjepa.research_directions.gait_fidelity.evaluation import measurement_rows
from gavd6_sjepa.research_directions.gait_fidelity.common import atomic_json, sha256
from gavd6_sjepa.research_directions.gait_fidelity.response_evaluation import (
    evaluate_response_followup, response_metrics, summarize_response_predictions, verified_response_completion)


def configuration():
    return dict(measurement=dict(min_segment_px=2., min_frames=16, min_coverage=.8, sign_tolerance_deg=1.),
                data=dict(held_level_deg=10), evaluation=dict(primary_candidate='candidate', primary_comparator='control',
                bootstrap_draws=50, bootstrap_seed=731))


class ResponseEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = fixture_bundle(samples=32, people=5, windows=2).subset('development')

    def test_oracle_and_reversed_sides_preserve_reference_direction_population(self):
        cfg = configuration()
        oracle = measurement_rows(self.bundle, self.bundle.targets['xy'], cfg, 'oracle', 17)
        correct = response_metrics(oracle, cfg)
        eligible = correct.response_direction_eligible
        self.assertTrue(eligible.any())
        self.assertEqual(correct.response_error.max(), 0.)
        self.assertTrue(correct.loc[eligible, 'response_direction_correct'].all())
        swapped = measurement_rows(self.bundle, np.asarray(self.bundle.targets['xy'])[:, :, SWAP], cfg, 'swapped', 17)
        reversed_rows = response_metrics(swapped, cfg)
        np.testing.assert_array_equal(eligible, reversed_rows.response_direction_eligible)
        self.assertFalse(reversed_rows.loc[eligible, 'response_direction_correct'].any())
        np.testing.assert_allclose(reversed_rows.response_magnitude_error, 0., atol=1e-10)

    def test_missing_prediction_counts_as_wrong_response_direction(self):
        cfg = configuration()
        predicted = np.asarray(self.bundle.targets['xy']).copy()
        changed = next(index for index, row in enumerate(self.bundle.records) if row['movement_level_deg'] == 10)
        predicted[changed, 0, 8] = np.nan
        failed = response_metrics(measurement_rows(self.bundle, predicted, cfg, 'missing', 17), cfg)
        selected = failed[~failed.prediction_success & failed.reference_eligible]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0].response_error, 720.)
        self.assertTrue(selected.iloc[0].response_direction_eligible)
        self.assertFalse(selected.iloc[0].response_direction_correct)
        self.assertEqual(selected.iloc[0].direction_accuracy, 0.)
        self.assertTrue(selected.iloc[0].held_intervention)

    def test_streamed_primary_comparison_uses_people_and_seeds(self):
        cfg = configuration()
        truth = np.asarray(self.bundle.targets['xy'])
        exports = [(name, seed, truth if name == 'candidate' else truth[:, :, SWAP])
                   for name in ('candidate', 'control') for seed in (17, 29, 43)]
        with tempfile.TemporaryDirectory() as temporary:
            people, coverage, comparisons = summarize_response_predictions(self.bundle, iter(exports), cfg, temporary)
            self.assertEqual(len(people), 2*2*3)
            self.assertEqual(comparisons['response_error']['people'], 2)
            self.assertEqual(comparisons['response_error']['seeds'], 3)
            self.assertGreater(comparisons['response_error']['improvement'], 0.)
            self.assertEqual(people.loc[people.method.eq('candidate'), 'response_error'].max(), 0.)
            self.assertTrue((coverage.direction_eligible > 0).all())
            conditions = pd.read_csv(Path(temporary)/'response-by-condition-person.csv')
            for field in ('camera_id', 'naming', 'observation', 'held_intervention', 'left_excursion_error'):
                self.assertIn(field, conditions)
            self.assertTrue((Path(temporary)/'response-comparisons.json').is_file())

    def test_training_references_cannot_enter_evaluation(self):
        train = fixture_bundle(samples=32, people=3, windows=2).subset('train')
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(PermissionError):
            summarize_response_predictions(train, [], configuration(), temporary)

    def test_cached_evaluation_still_verifies_child_prediction_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            phases, completed = [], {}
            truth = np.asarray(self.bundle.targets['xy'])
            for variant in ('jepa_delta_v1', 'jepa_endpoint_v1', 'coordinate_delta_v1'):
                for seed in (17, 29, 43):
                    phase_id = f'{variant}-{seed}'
                    phase = dict(phase_id=phase_id, phase='readout', seed=seed,
                                 recipe=dict(recipe_id=variant, representation_variant=variant))
                    phases.append(phase)
                    path = work/(phase_id+'.npz')
                    np.savez_compressed(path, xy=truth, indices=np.arange(len(truth)))
                    result = dict(recipe_id=variant, phase='readout', seed=seed,
                                  predictions=str(path), predictions_sha256=sha256(path))
                    receipt = work/(phase_id+'.json')
                    atomic_json(receipt, dict(result=result, artifacts={str(path):sha256(path)}))
                    completed[phase_id] = dict(receipt=str(receipt), sha256=sha256(receipt), result=result)
            output = work/'evaluation'; output.mkdir()
            summary = output/'response-summary.json'; atomic_json(summary, {'status':'cached'})
            receipt = output/'complete.json'
            atomic_json(receipt, dict(result={'summary':str(summary)}, artifacts={str(summary):sha256(summary)}))
            completed['evaluation'] = dict(receipt=str(receipt), sha256=sha256(receipt), result={'summary':str(summary)})
            atomic_json(work/'ledger.json', dict(completed=completed))
            atomic_json(work/'plan.json', dict(phases=phases))
            cfg = dict(configuration(), study_kind='jepa_response_followup', work=str(work),
                       followup=dict(parent_binding=dict(bundle='unused', baseline_phases={})))
            with patch('gavd6_sjepa.research_directions.gait_fidelity.scheduler._verify_frozen'), \
                 patch('gavd6_sjepa.research_directions.gait_fidelity.followup.verify_parent'), \
                 patch('gavd6_sjepa.research_directions.gait_fidelity.data.load_dataset', return_value=self.bundle):
                self.assertEqual(evaluate_response_followup(cfg), {'status':'cached'})
                changed = Path(completed[phases[0]['phase_id']]['result']['predictions'])
                with changed.open('ab') as stream:
                    stream.write(b'changed')
                with self.assertRaises(RuntimeError):
                    evaluate_response_followup(cfg)

    def test_hashed_receipt_rejects_stale_ledger_result_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary)/'receipt.json'
            atomic_json(receipt, dict(result=dict(recipe_id='right'), artifacts={}))
            stale = dict(receipt=str(receipt), sha256=sha256(receipt), result=dict(recipe_id='wrong'))
            with self.assertRaisesRegex(RuntimeError, 'ledger result'):
                verified_response_completion(stale)


if __name__ == '__main__':
    unittest.main()
