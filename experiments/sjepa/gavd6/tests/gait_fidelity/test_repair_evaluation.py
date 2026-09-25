"""Scientific pairing, failure accounting, split and publication guarantees."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import t

from gavd6_sjepa.research_directions.gait_fidelity.common import atomic_json, digest, read_json, sha256
from gavd6_sjepa.research_directions.gait_fidelity.data import fixture_bundle, select_rows, TrackBundle, SWAP
from gavd6_sjepa.research_directions.gait_fidelity.repair_evaluation import (
    EXPECTED_FITS, METHODS, PRIMARY_CANDIDATE, PRIMARY_COMPARATOR, SEEDS,
    _check_deadline, _confirmation, _prediction_receipts, _verify_prediction, evaluate_repair,
    paired_comparison, power_sensitivity, response_failure_decomposition, waveform_failure_decomposition)


def configuration():
    return dict(fixture=True, seeds=list(SEEDS),
                measurement=dict(min_segment_px=2., min_frames=8, min_coverage=.8, sign_tolerance_deg=1.),
                data=dict(held_level_deg=10.),
                repair=dict(statistical_protocol=dict(primary_candidate=PRIMARY_CANDIDATE,
                    primary_comparator=PRIMARY_COMPARATOR, primary_metric='waveform_error',
                    primary_extractor='fixture-source', seeds=list(SEEDS), bootstrap_draws=20, bootstrap_seed=731)))


def people_table(effects=(1., 2., 3., 4.), *, candidate='candidate', comparator='control'):
    rows = []
    for person, effect in enumerate(effects):
        for seed in SEEDS:
            for method, value in ((candidate, 10.), (comparator, 10. + effect)):
                rows.append(dict(method=method, seed=seed, canonical_person_id=f'p{person}', waveform_error=value))
    return pd.DataFrame(rows)


class RepairPairingTests(unittest.TestCase):
    def test_source_deadline_is_explicit_and_fixture_is_exempt(self):
        with self.assertRaises(TimeoutError):
            _check_deadline(dict(fixture=False, repair=dict(deadline_utc='2000-01-01T00:00:00Z')))
        with self.assertRaisesRegex(ValueError, 'explicit ISO'):
            _check_deadline(dict(fixture=False, repair={}))
        with self.assertRaisesRegex(ValueError, 'timezone'):
            _check_deadline(dict(fixture=False, repair=dict(deadline_utc='2100-01-01T00:00:00')))
        _check_deadline(dict(fixture=True, repair=dict(deadline_utc='2000-01-01T00:00:00Z')))

    def test_person_averaged_t_interval_not_person_seed_pseudoreplication(self):
        table = people_table()
        result = paired_comparison(table, 'candidate', 'control', 'waveform_error', split='development', draws=20)
        self.assertEqual(result['people'], 4)
        self.assertEqual(result['paired_person_seed_cells'], 12)
        self.assertEqual(result['improvement'], 2.5)
        half = t.ppf(.975, 3) * np.std([1., 2., 3., 4.], ddof=1) / 2
        np.testing.assert_allclose(result['person_averaged_t_ci95'], [2.5-half, 2.5+half])
        self.assertEqual(result['per_seed_improvement'], {str(s): 2.5 for s in SEEDS})
        self.assertFalse(result['noninferiority_established'])

    def test_effect_sign_reverses_and_confirmation_label_remains(self):
        result = paired_comparison(people_table(), 'control', 'candidate', 'waveform_error', split='confirmation', draws=20)
        self.assertEqual(result['improvement'], -2.5)
        self.assertEqual(result['evaluation_split'], 'confirmation')
        self.assertEqual(result['crossed_bootstrap']['status'], 'descriptive_locked_confirmation_estimate')
        self.assertEqual(result['crossed_bootstrap']['evaluation_split'], 'confirmation')

    def test_missing_seed_and_duplicate_cell_are_not_valid_inference(self):
        table = people_table()
        missing = table[table.seed.ne(43)]
        self.assertEqual(paired_comparison(missing, 'candidate', 'control', 'waveform_error',
            split='development', draws=20)['status'], 'incomplete_person_seed_pairing')
        with self.assertRaisesRegex(ValueError, 'Duplicate person/seed'):
            paired_comparison(pd.concat([table, table.iloc[:1]]), 'candidate', 'control', 'waveform_error',
                              split='development', draws=20)

    def test_undefined_reference_support_is_not_silently_dropped(self):
        table = people_table()
        table.loc[0, 'waveform_error'] = np.nan
        result = paired_comparison(table, 'candidate', 'control', 'waveform_error', split='development', draws=20)
        self.assertEqual(result['status'], 'insufficient_reference_support')
        self.assertNotIn('person_averaged_t_ci95', result)

    def test_failure_decomposition_retains_720_degree_penalty(self):
        table = pd.DataFrame(dict(reference_eligible=[True, True, False], prediction_success=[True, False, False],
            reference_change=[2., -5., np.nan], response_error=[3., 720., np.nan]))
        result = response_failure_decomposition(table)
        self.assertEqual(result.response_error.mean(), 361.5)
        self.assertEqual(result.response_failure_contribution.mean(), 360.)
        self.assertEqual(result.response_success_contribution.mean(), 1.5)
        self.assertEqual(result.response_success_conditional_error.mean(), 3.)
        self.assertEqual(result.response_failure_rate.mean(), .5)
        self.assertTrue(np.isnan(result.iloc[2].response_failure_rate))

    def test_waveform_decomposition_distinguishes_180_degree_failure_penalty(self):
        table = pd.DataFrame(dict(reference_eligible=[True, True, False], prediction_success=[True, False, False],
                                  waveform_error=[4., 180., np.nan]))
        result = waveform_failure_decomposition(table)
        self.assertEqual(result.waveform_error.mean(), 92.)
        self.assertEqual(result.waveform_failure_contribution.mean(), 90.)
        self.assertEqual(result.waveform_success_contribution.mean(), 2.)
        self.assertEqual(result.waveform_success_conditional_error.mean(), 4.)

    def test_power_uses_this_development_sd_and_does_not_reopen_confirmation(self):
        comparison = paired_comparison(people_table(), 'candidate', 'control', 'waveform_error', split='development', draws=20)
        result = power_sensitivity(comparison, split='development')
        self.assertEqual(result['paired_person_sd'], np.std([1., 2., 3., 4.], ddof=1))
        powers = [r['power_two_sided_alpha_05'] for r in result['scenarios'] if r['hypothetical_improvement_deg'] == .5]
        self.assertTrue(all(a < b for a, b in zip(powers, powers[1:])))
        self.assertEqual(power_sensitivity(comparison, split='confirmation')['status'], 'not_computed_on_confirmation')


class RepairEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        full = fixture_bundle(samples=16, people=3, windows=2).subset('development')
        # Keep complete paired panels on one camera/physical state/naming; the
        # two observation states are necessary for nuisance contrasts.
        cls.bundle = select_rows(full, [i for i, r in enumerate(full.records)
                                       if r['camera_id'] == 'side' and r['physical_state'] == 'original'
                                       and r['naming'] == 'correct'])

    def exports(self, bundle=None, *, fail_candidate=False):
        bundle = self.bundle if bundle is None else bundle
        truth = np.asarray(bundle.targets['xy'])
        for method in METHODS:
            for seed in SEEDS:
                value = truth.copy()
                if method == PRIMARY_COMPARATOR:
                    value = value[:, :, SWAP]
                if fail_candidate and method == PRIMARY_CANDIDATE:
                    idx = next(i for i, row in enumerate(bundle.records) if row['movement_level_deg'] == 10.)
                    value[idx, 0, 8] = np.nan
                yield method, seed, value

    def test_complete_fixture_stream_reports_real_cluster_count_and_failed_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'evaluation'
            receipt = evaluate_repair(self.bundle, self.exports(fail_candidate=True), configuration(), output)
            summary = read_json(output / 'summary.json')
            self.assertEqual(summary['fits'], 27)
            self.assertEqual(summary['evaluated_people'], 2)
            self.assertFalse(summary['independent_confirmation'])
            self.assertFalse(summary['noninferiority_established'])
            people = pd.read_csv(output / 'per-person.csv')
            self.assertEqual(len(people), 27 * 2)
            candidate = people[people.method.eq(PRIMARY_CANDIDATE)]
            self.assertGreater(candidate.response_failure_contribution.max(), 0.)
            self.assertGreater(candidate.waveform_failure_contribution.max(), 0.)
            np.testing.assert_allclose(people.response_error,
                people.response_failure_contribution + people.response_success_contribution, atol=1e-8)
            np.testing.assert_allclose(people.waveform_error,
                people.waveform_failure_contribution + people.waveform_success_contribution, atol=1e-8)
            comparisons = read_json(output / 'comparisons.json')
            self.assertEqual(comparisons['primary']['waveform_error']['people'], 2)
            self.assertEqual(comparisons['primary']['waveform_error']['seeds'], 3)
            self.assertTrue((output / 'report.md').is_file())
            for path, expected in receipt['artifacts'].items():
                self.assertEqual(sha256(path), expected)
            self.assertEqual(sha256(output / 'complete.json'), receipt['sha256'])
            with self.assertRaises(FileExistsError):
                evaluate_repair(self.bundle, [], configuration(), output)

    def test_primary_extractor_does_not_silently_pool_secondary_extractor(self):
        original = self.bundle
        records = deepcopy(original.records) + [dict(r, extractor='second', extractor_family='secondary') for r in original.records]
        bundle = TrackBundle({k: np.concatenate([np.asarray(v), np.asarray(v)]) for k, v in original.inputs.items()},
                             {k: np.concatenate([np.asarray(v), np.asarray(v)]) for k, v in original.targets.items()},
                             records, original.evidence_status, original.provenance)
        truth = np.asarray(bundle.targets['xy'])
        n = len(original.records)
        def exports():
            for method in METHODS:
                for seed in SEEDS:
                    value = truth.copy()
                    if method == PRIMARY_COMPARATOR:
                        value[:n] = value[:n, :, SWAP]
                    elif method == PRIMARY_CANDIDATE:
                        value[n:] = value[n:, :, SWAP]
                    yield method, seed, value
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'evaluation'
            evaluate_repair(bundle, exports(), configuration(), output)
            comparisons = read_json(output / 'comparisons.json')
            self.assertGreater(comparisons['primary']['waveform_error']['improvement'], 0.)
            self.assertAlmostEqual(comparisons['secondary_all_extractors']['waveform_error']['improvement'], 0., places=9)

    def test_incomplete_fit_set_never_publishes_complete_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'evaluation'
            with self.assertRaisesRegex(ValueError, 'Incomplete repair fit set'):
                evaluate_repair(self.bundle, [next(self.exports())], configuration(), output)
            self.assertFalse(output.exists())
            staging = next(Path(temporary).glob('.evaluation-staging-*'))
            self.assertTrue((staging / 'failure.json').is_file())
            self.assertFalse((staging / 'complete.json').exists())

    def test_mid_evaluation_deadline_never_publishes_partial_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'evaluation'
            with patch('gavd6_sjepa.research_directions.gait_fidelity.repair_evaluation._check_deadline',
                       side_effect=[None, None, None, TimeoutError('frozen deadline')]), \
                 self.assertRaisesRegex(TimeoutError, 'frozen deadline'):
                evaluate_repair(self.bundle, self.exports(), configuration(), output)
            self.assertFalse(output.exists())
            staging = next(Path(temporary).glob('.evaluation-staging-*'))
            self.assertTrue((staging / 'failure.json').is_file())
            self.assertFalse((staging / 'complete.json').exists())

    def test_mixed_or_wrong_split_is_rejected_without_relabeling(self):
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(PermissionError):
            evaluate_repair(self.bundle, [], configuration(), Path(temporary) / 'evaluation', split='confirmation')
        self.assertEqual({r['split'] for r in self.bundle.records}, {'development'})

    def test_receipt_array_and_ordered_population_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'predictions.npy'
            truth = np.asarray(self.bundle.targets['xy'])
            np.save(path, truth, allow_pickle=False)
            row = dict(method=PRIMARY_CANDIDATE, seed=17, split='development', predictions=str(path),
                       predictions_sha256=sha256(path), checkpoint_sha256='a'*64, record_identity=digest(self.bundle.records))
            _verify_prediction(truth, row, truth.shape, None)
            with self.assertRaisesRegex(ValueError, 'Supplied predictions'):
                _verify_prediction(truth+1., row, truth.shape, None)
            cfg = configuration()
            cfg['repair']['prediction_receipts'] = [dict(row, method=m, seed=s) for m, s in sorted(EXPECTED_FITS)]
            self.assertEqual(len(_prediction_receipts(cfg, self.bundle, 'development')), 27)
            cfg['repair']['prediction_receipts'][0]['record_identity'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'population identity'):
                _prediction_receipts(cfg, self.bundle, 'development')
            with path.open('ab') as stream:
                stream.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'array changed'):
                _verify_prediction(truth, row, truth.shape, None)

    def test_confirmation_lock_tampering_is_detected_before_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'lock.json'
            lock = dict(schema='gf-repair-confirmation-lock-v1', person_ids=['p'])
            lock['identity'] = digest(lock)
            lock['person_ids'] = ['tampered']
            atomic_json(path, lock)
            with self.assertRaisesRegex(ValueError, 'lock identity changed'):
                _confirmation(configuration(), self.bundle, path)

    def test_confirmation_fit_set_and_owner_mapping_are_enforced(self):
        bundle = TrackBundle(self.bundle.inputs, self.bundle.targets,
            [dict(r, split='confirmation', original_split='test') for r in self.bundle.records],
            self.bundle.evidence_status, deepcopy(self.bundle.provenance))
        with tempfile.TemporaryDirectory() as temporary:
            path, plan_path = Path(temporary) / 'lock.json', Path(temporary) / 'plan.json'
            plans = {r['source_family_id']: dict(r) for r in bundle.records}
            atomic_json(plan_path, dict(records=list(plans.values())))
            fits = [dict(method=m, seed=s, checkpoint=f'{m}-{s}', sha256='a'*64) for m, s in sorted(EXPECTED_FITS)]
            lock = dict(schema='gf-repair-confirmation-lock-v1', fit_manifest=fits,
                person_ids=sorted({r['canonical_person_id'] for r in bundle.records}),
                source_family_ids=sorted(plans), cohort_plan=str(plan_path), cohort_identity='cohort',
                checkpoint_hashes={r['checkpoint']: r['sha256'] for r in fits})
            lock['identity'] = digest(lock)
            atomic_json(path, lock)
            bundle.provenance.update(repair_confirmation_lock_sha256=sha256(path), repair_cohort_identity='cohort')
            with patch('gavd6_sjepa.research_directions.gait_fidelity.repair_cohort.verify_slim_lock', return_value=lock):
                declared, _ = _confirmation(configuration(), bundle, path)
                self.assertEqual(len(declared['fit_manifest']), 27)
                bundle.records[0]['canonical_person_id'] = bundle.records[-1]['canonical_person_id']
                with self.assertRaisesRegex(ValueError, 'family identity'):
                    _confirmation(configuration(), bundle, path)
            lock['fit_manifest'] = fits[:-1]
            lock['identity'] = digest({k:v for k,v in lock.items() if k != 'identity'})
            atomic_json(path, lock)
            with patch('gavd6_sjepa.research_directions.gait_fidelity.repair_cohort.verify_slim_lock', return_value=lock), \
                 self.assertRaisesRegex(ValueError, '27-fit'):
                _confirmation(configuration(), bundle, path)


if __name__ == '__main__':
    unittest.main()
