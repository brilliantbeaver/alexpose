"""Freeze existing study identities and the ordinary worker launch contract."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from gavd6_sjepa.research_directions.gait_fidelity.spec import build_plan


class ResponseCompatibilityTests(unittest.TestCase):
    def test_existing_core_and_full_plans_retain_their_exact_identity(self):
        # These are the pre-follow-up plans, independently read from repository HEAD.
        expected = {
            'core': 'e542099c5df1924b32870b6cdc700d78f6dc9d3f10f72aa7bfd08892c1e7166a',
            'full': 'f4e3f3e502cf8cb2c83ed4433abca856e1fc92048510b44c943cb595eb86beec',
        }
        for experiment_set, identity in expected.items():
            self.assertEqual(build_plan(experiment_set=experiment_set)['identity'], identity)

    def test_ordinary_workers_exec_the_original_module_without_deadline_supervision(self):
        path = Path(__file__).resolve().parents[2] / 'slurm/gait-fidelity/worker.py'
        spec = importlib.util.spec_from_file_location('response_compatibility_worker', path)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / 'config.json').write_text(json.dumps({'fixture': True}))
            args = [str(path), str(work), 'original-phase', str(work / 'attempt')]
            with patch.object(sys, 'argv', args), patch.object(worker.os, 'execv', side_effect=SystemExit(0)) as execute:
                with patch.object(worker, 'supervise') as supervise:
                    with self.assertRaises(SystemExit):
                        worker.main()
                    supervise.assert_not_called()
            execute.assert_called_once_with(sys.executable, [sys.executable, '-u', '-m',
                'gavd6_sjepa.research_directions.gait_fidelity', 'worker', '--work', str(work),
                '--phase-id', 'original-phase', '--attempt', str(work / 'attempt')])


if __name__ == '__main__':
    unittest.main()
