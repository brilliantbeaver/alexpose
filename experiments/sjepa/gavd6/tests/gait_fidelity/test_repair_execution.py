"""Exercise failure paths of repair Slurm submission and immutable stage reuse."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from gavd6_sjepa.research_directions.gait_fidelity.common import atomic_json, read_json, sha256
from gavd6_sjepa.research_directions.gait_fidelity import repair, repair_execution as execution
from gavd6_sjepa.research_directions.gait_fidelity.scheduler import _state, reserve, update_attempt


class RepairExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.work = Path(self.temporary.name)
        self.cfg = dict(work=str(self.work), fixture=False, code_root=str(Path(__file__).resolve().parents[2]),
            repair=dict(deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=24)).isoformat()),
            resources=dict(max_jobs=4,gpu_hours=48.,account='mind',partition='hai',gpu='h100:1',
                           cpu_workers=4,memory='64G',prepare_wall_minutes=240,max_attempts=2))
        self.phase=dict(phase_id='fit-jepa_delta_v1-dense_change-17',allocation_minutes=60)

    def tearDown(self):
        self.temporary.cleanup()

    def test_ambiguous_submit_reserves_before_external_effect_and_cannot_duplicate(self):
        def submit(command, **kwargs):
            state=_state(self.work)
            self.assertEqual(len(state['attempts']),1)
            self.assertEqual(state['attempts'][0]['status'],'reserved')
            self.assertTrue((Path(state['attempts'][0]['path'])/'submission.json').exists())
            return SimpleNamespace(returncode=1,stdout='',stderr='connection lost after acceptance')
        with patch.object(execution.subprocess,'run',side_effect=submit) as called:
            with self.assertRaisesRegex(RuntimeError,'unresolved'):
                execution._submission(self.cfg,self.phase)
            with self.assertRaisesRegex(RuntimeError,'unresolved attempt'):
                execution._submission(self.cfg,self.phase)
            self.assertEqual(called.call_count,1)
        self.assertEqual(_state(self.work)['attempts'][0]['reserved_gpu_hours'],1.)

    def test_worker_receipt_and_resources(self):
        response=SimpleNamespace(returncode=0,stdout='312;cluster\n',stderr='')
        with patch.object(execution.subprocess,'run',return_value=response) as called:
            self.assertEqual(execution._submission(self.cfg,self.phase),'312')
        command=called.call_args.args[0]
        self.assertIn('--gres=gpu:h100:1',command)
        self.assertIn('--time=60',command)
        self.assertTrue(any(v.endswith('repair-worker.sbatch') for v in command))
        self.assertEqual(_state(self.work)['attempts'][0]['status'],'submitted')

    def test_cutoff_refuses_before_allocating(self):
        self.cfg['repair']['deadline_utc']=(datetime.now(timezone.utc)+timedelta(minutes=20)).isoformat()
        with patch.object(execution.subprocess,'run') as called:
            with self.assertRaisesRegex(RuntimeError,'Insufficient time'):
                execution._submission(self.cfg,self.phase)
        called.assert_not_called()
        self.assertEqual(_state(self.work)['attempts'],[])

    def test_budget_includes_failed_allocations(self):
        self.cfg['resources']['gpu_hours']=1.2
        old=reserve(self.work,'failed',1.,limit=1.2,max_jobs=4)
        update_attempt(self.work,old['id'],dict(status='failed',allocated_gpu_hours=.5))
        with patch.object(execution.subprocess,'run') as called:
            with self.assertRaisesRegex(RuntimeError,'budget exhausted'):
                execution._submission(self.cfg,self.phase)
        called.assert_not_called()

    def test_cutoff_cancels_only_owned_numeric_active_ids(self):
        for phase,status,job in [('own','submitted','412'),('unknown','reserved',None),('finished','complete','413')]:
            item=reserve(self.work,phase,1.,limit=48,max_jobs=4)
            update_attempt(self.work,item['id'],dict(status=status,job_id=job))
        with patch.object(execution.subprocess,'run',return_value=SimpleNamespace(returncode=0,stderr='')) as called:
            result=execution.cancel_owned(self.cfg)
        called.assert_called_once_with(['scancel','412'],capture_output=True,text=True,check=False)
        self.assertEqual(result['cancelled_job_ids'],['412'])
        self.assertEqual(len(result['unresolved_job_names']),1)
        self.assertEqual(_state(self.work)['attempts'][0]['status'],'submitted')

    def test_pending_status_is_not_reported_as_training(self):
        self.cfg['repair']['confirmation_lock']=str(self.work/'confirmation/lock.json')
        item=reserve(self.work,'fit',1.,limit=48,max_jobs=4)
        update_attempt(self.work,item['id'],dict(status='running',job_id='123',scheduler_state='PENDING'))
        self.assertEqual(repair.status(self.cfg)['active'][0]['status'],'PENDING')

    def test_confirmation_evaluation_waits_for_all_shards(self):
        phases=execution.phase_plan(self.cfg,'confirmation')
        self.assertEqual(len(phases),5)
        self.assertEqual(phases[-1]['depends_on'],[p['phase_id'] for p in phases[:-1]])
        self.assertEqual(phases[-1]['allocation_minutes'],120)

    def test_fixture_never_submits_source_jobs(self):
        self.cfg['fixture']=True
        with patch.object(execution.subprocess,'run') as called:
            with self.assertRaisesRegex(ValueError,'Fixtures'):
                execution.launch(self.cfg)
            with self.assertRaisesRegex(ValueError,'Local execution'):
                execution.run(self.cfg,local=False)
        called.assert_not_called()

    def test_finished_benchmark_readonly_after_cutoff(self):
        artifact=self.work/'result.json'; atomic_json(artifact,dict(fixture=True))
        receipt=self.work/'complete.json'
        atomic_json(receipt,dict(artifacts={str(artifact):sha256(artifact)},result=dict(benchmark='done')))
        value=dict(receipt=str(receipt),sha256=sha256(receipt),result=dict(benchmark='done'))
        atomic_json(self.work/'ledger.json',dict(attempts=[],completed={'benchmark':value}))
        self.cfg['repair']['deadline_utc']='2000-01-01T00:00:00Z'
        self.assertEqual(execution.finished_stage(self.cfg,'benchmark'),{'benchmark':'done'})
        artifact.write_text('tampered')
        with self.assertRaisesRegex(RuntimeError,'artifact changed'):
            execution.finished_stage(self.cfg,'benchmark')

    def test_exposure_worksheet_does_not_authorize_unknown_people(self):
        plan=self.work/'plan.json'; parent=self.work/'parent.json'; bundle=self.work/'bundle'
        atomic_json(plan,dict(person_ids=['test-a','test-b']))
        atomic_json(parent,dict(preparation={'reservation_csv':str(self.work/'missing.csv')}))
        atomic_json(bundle/'manifest.json',dict(records=[dict(canonical_person_id='test-b')]))
        self.cfg['repair'].update(cohort_plan=str(plan),parent_config=str(parent),parent_bundle=str(bundle))
        value=repair.exposure_audit(self.cfg)
        self.assertFalse(value['ready'])
        self.assertFalse(value['worksheet_is_reviewed_evidence'])
        self.assertEqual([p['status'] for p in value['people']],['review_required','known_exposure_or_conflict'])
        self.assertNotIn('unexposed_verified',Path(value['review_worksheet']).read_text())

    def test_discovered_reservations_and_training_roles_cannot_be_erased(self):
        plan=self.work/'plan.json'; parent=self.work/'parent.json'; bundle=self.work/'bundle'
        ledger=self.work/'historical.csv'
        ledger.write_text('person_id,canonical_person_id,original_split,reserved,exposure\na,a,test,true,unknown\nb,b,train,false,unexposed_verified\n')
        atomic_json(plan,dict(person_ids=['a','b']))
        atomic_json(parent,dict(preparation={'reservation_csv':str(ledger)}))
        atomic_json(bundle/'manifest.json',dict(records=[]))
        self.cfg['repair'].update(cohort_plan=str(plan),parent_config=str(parent),parent_bundle=str(bundle))
        value=repair.exposure_audit(self.cfg)
        self.assertEqual([p['status'] for p in value['people']],['known_exposure_or_conflict']*2)

    def test_source_confirmation_cli_must_use_accounted_worker(self):
        with patch.object(repair,'load_repair',return_value=self.cfg),patch.object(repair,'evaluate') as evaluate:
            with self.assertRaisesRegex(ValueError,'launch.*confirmation'):
                repair.main(['evaluate','--work',str(self.work),'--split','confirmation'])
        evaluate.assert_not_called()


if __name__=='__main__':
    unittest.main()
