"""Budget bookkeeping fixtures; these do not execute a CUDA operator or job."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig
from gavd6_sjepa.research_directions.synthetic_training_v2.runtime import budgeted_gpu_stage

class BudgetScopeTests(unittest.TestCase):
    def test_concurrent_scope_cannot_reserve_same_remaining_budget(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Path(d)/'prior.json';ledger.write_text(json.dumps({'scope_authorized':True,'entries':[]}))
            cfg=RunConfig(run_id='budget-fixture',output_root=d,mode='source',bundle='not-opened',device='cuda',
                authorized_gpu_hours=1,projected_gpu_hours=.001,cost_ledger=str(ledger))
            with patch('signal.setitimer'),budgeted_gpu_stage(cfg,'fixture-outer'):
                with self.assertRaisesRegex(ValueError,'already running'):
                    with budgeted_gpu_stage(cfg,'fixture-inner'):self.fail('concurrent scope entered')
            rows=[json.loads(p.read_text()) for p in (cfg.root/'costs').glob('*.json')]
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['status'],'completed')
    def test_failed_attempt_counts_against_future_scope(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Path(d)/'prior.json';ledger.write_text(json.dumps({'scope_authorized':True,'entries':[]}))
            cfg=RunConfig(run_id='failed-budget-fixture',output_root=d,mode='source',bundle='not-opened',device='cuda',
                authorized_gpu_hours=1,projected_gpu_hours=.001,cost_ledger=str(ledger))
            with patch('signal.setitimer'),self.assertRaises(RuntimeError):
                with budgeted_gpu_stage(cfg,'fixture-failure'):raise RuntimeError('injected fixture failure')
            row=json.loads(next((cfg.root/'costs').glob('*.json')).read_text())
            self.assertEqual(row['status'],'failed');self.assertGreaterEqual(row['gpu_seconds'],0)
