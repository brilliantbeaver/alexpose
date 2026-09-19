"""Post-run diagnostics must preserve evidence and separate fit/evaluation data."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import atomic_json, sha256_file
from gavd6_sjepa.research_directions.synthetic_training_v2.data import fixture_bundle
from gavd6_sjepa.research_directions.synthetic_training_v2.workflow import _write_prediction

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'scripts/research_directions/synthetic_training_v2'
sys.path.insert(0, str(SCRIPTS))
import postrun_checks as checks


def source_fixture(parent):
    """Saved numeric fixture, explicitly not a certified source experiment."""
    cfg = RunConfig.fixture('source-fixture', str(parent))
    source = cfg.root
    atomic_json(source / 'effective-config.json', cfg.as_dict())
    atomic_json(source / 'identity.json', {'signature': 'unit-test-fixture'})
    (source / 'report.md').write_text('Retained original fixture report.\n')
    bundle = fixture_bundle()
    bundle.save(source / 'data/bundle')
    dev = bundle.subset('development')
    for method in checks.COMPARATORS:
        prediction = dev.inputs['xy'].copy() if method == 'unchanged' else dev.targets['xy'].copy()
        if method == 'paired_jepa':
            # No ankle oscillation: complete predictions but wrong peak counts.
            prediction[:, :, 10:12] = prediction[:, :1, 10:12]
        _write_prediction(source / 'predictions', method, 17, prediction, dev)
    atomic_json(source / 'fits/paired_jepa-17/training.json', {
        'arm': 'paired_jepa', 'seed': 17, 'status': 'complete', 'optimizer_updates': 2,
        'planned_updates': 2, 'phases': [{'name': 'pretrain', 'updates': 2}],
        'history': [{'phase': 'pretrain', 'phase_update': 1, 'loss': 1.},
                    {'phase': 'pretrain', 'phase_update': 2, 'loss': .5}],
    })
    return source


class PostrunIntegrationTests(unittest.TestCase):
    def test_all_checks_keep_evidence_and_report_censored_and_mismatched_timing(self):
        with tempfile.TemporaryDirectory() as folder:
            source = source_fixture(folder)
            before = checks.source_files(source)
            output = Path(folder) / 'diagnostics'
            result = checks.analyze_source(source, output, {'scope': 'fixture-test'}, max_windows=1)
            self.assertEqual(result['status'], 'POSTRUN_CHECKS_COMPLETE')
            self.assertEqual(result['evidence_status'], 'fixture-tested')
            self.assertEqual(checks.source_files(source), before)
            models = checks.read(output / 'calibration-models.json')
            self.assertEqual(set(models), {'joint_offset', 'joint_affine'})
            records = pd.read_csv(output / 'timing-per-window.csv')
            oracle = records.loc[records.method.eq('reference_oracle')]
            self.assertTrue(oracle.reference_eligible.any())
            self.assertTrue((~oracle.reference_eligible).any())
            self.assertTrue((oracle.original_support == oracle.reference_eligible).all())
            self.assertTrue(records.loc[records.method.eq('paired_jepa')].prediction_status.eq('event_count_mismatch').any())
            report = (output / 'report.md').read_text()
            for label in ('Oracle-eligible', 'Reference-ineligible', 'Count mismatches', 'Known extra peaks',
                          'Amplitude ratio', '0.20 s displacement', 'conditional on finite', '24 parameters'):
                self.assertIn(label, report)
            self.assertTrue(list((output / 'images').glob('trajectory-*.svg')))
            self.assertTrue(list((output / 'images').glob('training-*.svg')))
            self.assertTrue((output / 'training-history.csv').is_file())
            with self.assertRaises(FileExistsError):
                checks.analyze_source(source, output, {})

    def test_rejects_output_inside_source_and_prediction_reference_misalignment(self):
        with tempfile.TemporaryDirectory() as folder:
            source = source_fixture(folder)
            with self.assertRaisesRegex(ValueError, 'separate'):
                checks.analyze_source(source, source / 'extra', {})
            path = source / 'predictions/paired_jepa-17.npz'
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: value.copy() for key, value in data.items()}
            arrays['target_xy'][0, 0, 0, 0] += 1
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, 'differs from retained bundle'):
                checks.analyze_source(source, Path(folder) / 'output', {}, checks='calibration')

    def test_reference_cannot_enter_calibration_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            source = source_fixture(folder)
            from diagnostics.calibration import fit_calibrations
            seen = []
            def fit(train, held, ridge):
                self.assertTrue(all(r['split'] == 'train' for r in train.records))
                seen.extend(r['canonical_person_id'] for r in train.records)
                return fit_calibrations(train, held, ridge)
            with patch('diagnostics.calibration.fit_calibrations', side_effect=fit):
                checks.analyze_source(source, Path(folder) / 'out', {}, checks='calibration')
            dev_people = {r['canonical_person_id'] for r in fixture_bundle().subset('development').records}
            self.assertTrue(seen)
            self.assertFalse(set(seen) & dev_people)


class PostrunSchedulerTests(unittest.TestCase):
    def test_submit_uses_cpu_and_removes_scheduler_input_overrides(self):
        with tempfile.TemporaryDirectory(prefix='postrun spaces ') as folder:
            state = {'source_config': str(Path(folder) / 'source.json'), 'account': 'test-account', 'partition': 'test-partition'}
            overrides = {'SBATCH_GRES': 'gpu:h100:1', 'SBATCH_ARRAY_INX': '1-10',
                         'SLURM_CLUSTERS': 'foreign', 'SLURM_HINT': 'nomultithread'}
            with patch('haic.state_for', return_value=state), patch.dict(os.environ, overrides), \
                    patch.object(checks.subprocess, 'check_output', return_value='12345;cluster\n') as call:
                output = checks.submit(Path(folder), checks='all', ridge=.001, tolerance_s=.12, max_windows=4)
            command = call.call_args.args[0]
            environment = call.call_args.kwargs['env']
            self.assertFalse(any('gres=' in item or 'gpu' in item for item in command))
            self.assertIn('--cpus-per-task=4', command)
            self.assertFalse(set(overrides) & set(environment))
            self.assertEqual(environment['STV2_PYTHON'], sys.executable)
            self.assertEqual(command[-1], str(output / 'request.json'))
            submission = checks.read(output / 'submission.json')
            self.assertEqual(submission['job_id'], '12345')
            self.assertEqual(submission['request_sha256'], sha256_file(output / 'request.json'))

    def test_uncertain_submit_retains_recovery_details(self):
        with tempfile.TemporaryDirectory() as folder:
            state = {'source_config': 'source.json', 'account': 'a', 'partition': 'p'}
            with patch('haic.state_for', return_value=state), \
                    patch.object(checks.subprocess, 'check_output', return_value='lost-response'):
                with self.assertRaisesRegex(RuntimeError, 'submission.json'):
                    checks.submit(Path(folder), checks='all', ridge=.001, tolerance_s=.12, max_windows=4)
            saved = next(Path(folder).glob('diagnostics/*/submission.json'))
            self.assertEqual(checks.read(saved)['status'], 'submission_uncertain')

    def request(self, work):
        output = work / 'diagnostics/attempt'
        request = output / 'request.json'
        atomic_json(request, dict(work=str(work), output=str(output), checks='all', ridge=.001,
                                  tolerance_s=.12, max_windows=1, analysis_code_sha256=checks.analysis_code()))
        atomic_json(output / 'submission.json', dict(status='submitted', job_id='12345',
                                                     request_sha256=sha256_file(request)))
        return request

    def test_verification_failure_prevents_analysis_and_records_reason(self):
        with tempfile.TemporaryDirectory() as folder:
            request = self.request(Path(folder))
            with patch('check_results.check_results', side_effect=ValueError('Run identity changed')), \
                    patch.object(checks, 'analyze_source') as analyze:
                with self.assertRaisesRegex(ValueError, 'identity'):
                    checks.run_request(request)
            analyze.assert_not_called()
            self.assertEqual(checks.read(request.parent / 'status.json')['status'], 'failed')

    def test_changed_request_or_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            request = self.request(Path(folder))
            with patch.object(checks, 'analysis_code', return_value={'changed': 'hash'}), \
                    patch('check_results.check_results') as verify:
                with self.assertRaisesRegex(ValueError, 'code changed'):
                    checks.run_request(request)
            verify.assert_not_called()
            request.write_text(request.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'request changed'):
                checks.run_request(request)

    def test_code_change_during_verification_or_analysis_prevents_success(self):
        for changed_during in ('verification', 'analysis'):
            with self.subTest(changed_during=changed_during), tempfile.TemporaryDirectory() as folder:
                request = self.request(Path(folder))
                initial = checks.analysis_code()
                sequence = [initial, {'changed': 'hash'}] if changed_during == 'verification' else [initial, initial, {'changed': 'hash'}]
                with patch.object(checks, 'analysis_code', side_effect=sequence), \
                        patch('check_results.check_results', return_value={'results': str(Path(folder) / 'source')}), \
                        patch.object(checks, 'analyze_source', return_value={'status': 'POSTRUN_CHECKS_COMPLETE'}) as analyze:
                    with self.assertRaisesRegex(ValueError, 'code changed'):
                        checks.run_request(request)
                self.assertEqual(analyze.call_count, 0 if changed_during == 'verification' else 1)
                self.assertEqual(checks.read(request.parent / 'status.json')['status'], 'failed')

    def test_status_reconciles_scheduler_killed_worker_without_submission(self):
        with tempfile.TemporaryDirectory() as folder:
            request = self.request(Path(folder))
            atomic_json(request.parent / 'status.json', {'status': 'running'})
            stdout = io.StringIO()
            with patch.object(checks.subprocess, 'check_output', return_value='12345|TIMEOUT|0:0|30:00|cpu=4\n') as call, \
                    contextlib.redirect_stdout(stdout):
                checks.status(Path(folder))
            self.assertEqual(call.call_args.args[0][0], 'sacct')
            self.assertIn('DIAGNOSTICS_FAILED', stdout.getvalue())
            self.assertIn('stale', stdout.getvalue())

    def test_bash_wrapper_forwards_selected_check_and_spaced_work(self):
        with tempfile.TemporaryDirectory(prefix='shell path ') as folder:
            folder = Path(folder)
            capture = folder / 'arguments.json'
            stub = folder / 'python stub'
            stub.write_text('#!/bin/sh\nexec ' + sys.executable + ' -c \'import json,os,sys; '
                            'open(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\' "$@"\n')
            stub.chmod(0o700)
            environment = dict(os.environ, STV2_ROOT=str(ROOT), STV2_WORK=str(folder),
                               STV2_PYTHON=str(stub), CAPTURE=str(capture))
            for option in ('all', 'calibration', 'timing', 'curves', 'status'):
                subprocess.run(['bash', str(ROOT / 'slurm/synthetic-training-v2/postrun-checks.sh'),
                                option, '--work', str(folder)], env=environment, check=True)
                args = json.loads(capture.read_text())
                self.assertEqual(args[-2:], ['--work', str(folder)])
                self.assertEqual(args[1], 'status' if option == 'status' else 'submit')
                if option != 'status':
                    self.assertEqual(args[2:4], ['--checks', option])


if __name__ == '__main__':
    unittest.main()
