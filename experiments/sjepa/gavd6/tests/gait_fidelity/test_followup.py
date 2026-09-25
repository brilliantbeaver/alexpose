"""Dependency integrity, recipe separation and hard allocation admission checks."""
from datetime import datetime, timezone, timedelta
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from gavd6_sjepa.research_directions.gait_fidelity.common import atomic_json, read_json, sha256
from gavd6_sjepa.research_directions.gait_fidelity.config import initialize, load_config
from gavd6_sjepa.research_directions.gait_fidelity.data import fixture_bundle, save_dataset
from gavd6_sjepa.research_directions.gait_fidelity.followup import (
    BUDGETS, DEADLINE, initialize_followup, inspect_parent, verify_parent, category,
    allocation_minutes, admit_matrix, budget_usage, deadline_cancellations)
from gavd6_sjepa.research_directions.gait_fidelity.scheduler import reserve, update_attempt, _state
from gavd6_sjepa.research_directions.gait_fidelity.spec import build_plan, build_response_plan, RESPONSE_VARIANTS


def retained_parent(folder):
    """Metadata fixture: verify binding behavior without presenting dummy files as fits."""
    work = Path(folder)/'parent'
    cfg = initialize(work, fixture=True, experiment_set='core')
    bundle = save_dataset(fixture_bundle(samples=32), work/'bundle')
    config_hash = sha256(work/'config.json')
    # This metadata fixture has no executable parent snapshot. Integration tests
    # separately run real CPU training and freeze the actual project source.
    atomic_json(work/'frozen.json', dict(config_sha256=config_hash,
                                       plan_sha256=sha256(work/'plan.json'), code={}))
    completed = {'prepare': dict(result=dict(bundle=str(bundle)),
                                bundle_manifest_sha256=sha256(bundle/'manifest.json'))}
    for phase in read_json(work/'plan.json')['phases']:
        root = work/'retained'/phase['phase_id']
        root.mkdir(parents=True)
        artifact = root/'checkpoint.pt'
        artifact.write_text('metadata-only integrity fixture')
        result = dict(checkpoint=str(artifact), updates=1 if phase['phase'] != 'end_to_end' else 2)
        receipt = root/'complete.json'
        atomic_json(receipt, dict(phase_id=phase['phase_id'], config_sha256=config_hash,
                                 artifacts={str(artifact):sha256(artifact)}, result=result))
        completed[phase['phase_id']] = dict(receipt=str(receipt), sha256=sha256(receipt), result=result)
    report = work/'report.md'; report.write_text('Software metadata fixture, no scientific result.')
    receipt = work/'evaluation/complete.json'
    atomic_json(receipt, dict(phase_id='evaluation', config_sha256=config_hash,
                             artifacts={str(report):sha256(report)}, result=dict(report=str(report))))
    completed['evaluation'] = dict(receipt=str(receipt),sha256=sha256(receipt),result=dict(report=str(report)))
    atomic_json(work/'ledger.json', dict(attempts=[],completed=completed))
    return work, cfg


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.parent, self.parent_cfg = retained_parent(self.root)
        self.child = self.root/'child'

    def tearDown(self):
        self.temporary.cleanup()

    def test_separate_matrix_and_no_fictitious_preparation(self):
        before = {str(p):sha256(p) for p in self.parent.rglob('*') if p.is_file()}
        cfg = initialize_followup(self.child,parent_work=self.parent)
        self.assertEqual(cfg,load_config(self.child))
        plan = read_json(self.child/'plan.json')
        self.assertEqual(plan['counts'],dict(recipes=3,final_fits=9,pretraining_phases=9,optimization_phases=18))
        self.assertEqual({r['representation_variant'] for r in plan['recipes']},set(RESPONSE_VARIANTS))
        self.assertFalse((self.child/'ledger.json').exists())
        self.assertFalse(any('prepare' in p['depends_on'] for p in plan['phases']))
        self.assertFalse({p['phase_id'] for p in plan['phases']} & {p['phase_id'] for p in build_plan()['phases']})
        self.assertEqual(before,{str(p):sha256(p) for p in self.parent.rglob('*') if p.is_file()})
        self.assertEqual(cfg['training']['pretraining_updates'],1)

    def test_parent_must_finish_before_child_exists(self):
        state=read_json(self.parent/'ledger.json');state['completed'].pop('evaluation')
        atomic_json(self.parent/'ledger.json',state)
        with self.assertRaisesRegex(RuntimeError,'complete parent'):
            initialize_followup(self.child,parent_work=self.parent)
        self.assertFalse(self.child.exists())

    def test_parent_active_reservations_block_setup(self):
        state=read_json(self.parent/'ledger.json');state['attempts']=[dict(status='accounting_pending')]
        atomic_json(self.parent/'ledger.json',state)
        with self.assertRaisesRegex(RuntimeError,'unresolved'):
            inspect_parent(self.parent)

    def test_no_child_inside_parent(self):
        with self.assertRaises(ValueError):
            initialize_followup(self.parent/'child',parent_work=self.parent)

    def test_parent_arrays_and_amendment_bound(self):
        amendment=self.parent/'admissions/training-min-two-windows/decision.json'
        atomic_json(amendment,dict(excluded=['person-old']))
        cfg=initialize_followup(self.child,parent_work=self.parent)
        self.assertIn(str(amendment.resolve()),cfg['followup']['parent_binding']['files'])
        original=amendment.read_bytes();amendment.write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'dependency changed'):
            verify_parent(cfg)
        amendment.write_bytes(original)
        array=self.parent/'bundle/inputs.npz';array.write_bytes(array.read_bytes()+b'changed')
        with self.assertRaisesRegex(RuntimeError,'dependency changed'):
            verify_parent(cfg)

    def test_missing_parent_checkpoint_and_changed_receipt_rejected(self):
        state=read_json(self.parent/'ledger.json')
        value=next(v for k,v in state['completed'].items() if k.startswith('pretrain'))
        Path(value['result']['checkpoint']).write_text('tampered')
        with self.assertRaisesRegex(RuntimeError,'dependency changed'):
            inspect_parent(self.parent)

    def test_ledger_result_cannot_override_hashed_parent_receipt(self):
        state=read_json(self.parent/'ledger.json')
        value=next(v for k,v in state['completed'].items() if k.startswith('pretrain'))
        value['result']['checkpoint']='/some/other/checkpoint.pt'
        atomic_json(self.parent/'ledger.json',state)
        with self.assertRaisesRegex(RuntimeError,'ledger phase/result'):
            inspect_parent(self.parent)

    def test_parent_models_require_receipts(self):
        state=read_json(self.parent/'ledger.json')
        value=next(v for k,v in state['completed'].items() if k.startswith('pretrain'))
        value.pop('receipt')
        atomic_json(self.parent/'ledger.json',state)
        with self.assertRaisesRegex(RuntimeError,'completion receipt'):
            inspect_parent(self.parent)

    def test_confirmation_never_admitted(self):
        manifest=self.parent/'bundle/manifest.json';data=read_json(manifest)
        data['records'][0]['split']='confirmation';atomic_json(manifest,data)
        with self.assertRaises(PermissionError):
            inspect_parent(self.parent)

    def test_budget_fields_and_updates_cannot_be_edited(self):
        cfg=initialize_followup(self.child,parent_work=self.parent)
        cfg['training']['pretraining_updates']=2;atomic_json(self.child/'config.json',cfg)
        with self.assertRaisesRegex(ValueError,'actual parent update'):
            load_config(self.child)

    def test_inherited_scientific_settings_cannot_be_retuned_before_freeze(self):
        cfg=initialize_followup(self.child,parent_work=self.parent)
        cfg['training']['mask_fraction']=.1;atomic_json(self.child/'config.json',cfg)
        with self.assertRaisesRegex(ValueError,'optimizer, masks'):
            load_config(self.child)

    def test_source_inherits_selected_profile_not_requested_updates(self):
        cfg=read_json(self.parent/'config.json')
        cfg.update(fixture=False,mode='source',device='cuda',code_root=str(self.root/'old-release'))
        cfg['training'].update(pretraining_updates=2000,readout_updates=2000,end_to_end_updates=4000)
        atomic_json(self.parent/'config.json',cfg)
        config_hash=sha256(self.parent/'config.json')
        frozen=read_json(self.parent/'frozen.json');frozen['config_sha256']=config_hash
        atomic_json(self.parent/'frozen.json',frozen)
        state=read_json(self.parent/'ledger.json')
        for value in state['completed'].values():
            if 'receipt' in value:
                receipt=read_json(value['receipt']);receipt['config_sha256']=config_hash
                atomic_json(value['receipt'],receipt);value['sha256']=sha256(value['receipt'])
        result=dict(budget=dict(selected=dict(updates=dict(pretraining_updates=1,readout_updates=1,end_to_end_updates=2))))
        profile=self.parent/'profile/complete.json'
        atomic_json(profile,dict(phase_id='profile',config_sha256=config_hash,artifacts={},result=result))
        state['completed']['profile']=dict(receipt=str(profile),sha256=sha256(profile),result=result)
        atomic_json(self.parent/'ledger.json',state)
        child=initialize_followup(self.child,parent_work=self.parent,
            deadline_utc=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat())
        self.assertEqual(child['training']['pretraining_updates'],1)
        self.assertEqual(len(child['followup']['parent_binding']['baseline_phases']),15)

    def test_category_caps_and_failed_allocations_remain_charged(self):
        work=self.root/'accounting'
        first=reserve(work,'followup-profile',2,limit=48,max_jobs=8,
                      budget_category='profile',category_limits=BUDGETS)
        update_attempt(work,first['id'],dict(status='failed',allocated_gpu_hours=1.75))
        self.assertEqual(category('followup-profile',1),'recovery')
        reserve(work,'followup-profile',2,limit=48,max_jobs=8,
                budget_category='recovery',category_limits=BUDGETS)
        with self.assertRaisesRegex(RuntimeError,'profile allocation'):
            reserve(work,'other-profile',1,limit=48,max_jobs=8,
                    budget_category='profile',category_limits=BUDGETS)
        usage=budget_usage(_state(work))
        self.assertEqual(usage['profile'],1.75);self.assertEqual(usage['recovery'],2.)

    def test_no_half_budget_fallback_and_deadline_admission(self):
        cfg=initialize_followup(self.child,parent_work=self.parent)
        cfg['fixture']=False
        now=datetime(2026,9,23,tzinfo=timezone.utc)
        timings={f'{v}:{p}':dict(fixed_seconds=1,optimization_seconds=1,profile_updates=1)
                 for v in RESPONSE_VARIANTS for p in ('pretrain','readout')}
        good=admit_matrix(cfg,timings,now=now)
        self.assertTrue(good['admitted'])
        self.assertEqual(good['actual_updates']['pretraining_updates'],1)
        late=admit_matrix(cfg,timings,now=datetime(2026,9,24,23,tzinfo=timezone.utc))
        self.assertFalse(late['admitted'])
        for t in timings.values():t['optimization_seconds']=9000
        self.assertFalse(admit_matrix(cfg,timings,now=now)['admitted'])

    def test_fixed_queue_allowance_and_diagnostic_allocation(self):
        cfg=initialize_followup(self.child,parent_work=self.parent);cfg['fixture']=False
        deadline=datetime(2026,9,25,1,tzinfo=timezone.utc)
        self.assertEqual(allocation_minutes(cfg,'followup-diagnostics',now=deadline-timedelta(hours=7)),240)
        self.assertEqual(allocation_minutes(cfg,'followup-diagnostics',1,now=deadline-timedelta(hours=7)),240)
        self.assertEqual(category('followup-diagnostics',1),'recovery')
        with self.assertRaises(RuntimeError):
            allocation_minutes(cfg,'fit',now=deadline-timedelta(hours=3))
        self.assertEqual(allocation_minutes(cfg,'fit',now=deadline-timedelta(hours=5)),120)

    def test_complete_three_seed_matrix_required(self):
        with self.assertRaises(ValueError):
            build_response_plan([17])

    def test_deadline_cancels_only_retained_numeric_child_ids(self):
        cfg=initialize_followup(self.child,parent_work=self.parent);cfg['fixture']=False
        owned=reserve(self.child,'fit',2,limit=48,max_jobs=8,budget_category='fits',category_limits=BUDGETS)
        update_attempt(self.child,owned['id'],dict(status='submitted',job_id='1234'))
        pending=reserve(self.child,'another',2,limit=48,max_jobs=8,budget_category='fits',category_limits=BUDGETS)
        now=datetime(2026,9,25,1,tzinfo=timezone.utc)
        with patch('gavd6_sjepa.research_directions.gait_fidelity.followup.subprocess.run',
                   return_value=subprocess.CompletedProcess([],0,'','')) as call:
            result=deadline_cancellations(cfg,_state(self.child),now=now)
        self.assertEqual(result,['1234'])
        self.assertEqual(call.call_args.args[0],['scancel','1234'])
        self.assertTrue(next(a for a in _state(self.child)['attempts'] if a['id']==owned['id'])['deadline_cancel_requested'])
        self.assertFalse(next(a for a in _state(self.child)['attempts'] if a['id']==pending['id']).get('deadline_cancel_requested',False))


class WorkerDeadlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[2]/'slurm/gait-fidelity/worker.py'
        spec=importlib.util.spec_from_file_location('response_worker_supervisor',path)
        cls.worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.worker)

    def test_delayed_queued_worker_does_not_start_work(self):
        with tempfile.TemporaryDirectory() as folder:
            marker=Path(folder)/'started';receipt=Path(folder)/'deadline.json'
            code=f'from pathlib import Path;Path({str(marker)!r}).write_text("started")'
            result=self.worker.supervise([sys.executable,'-c',code],
                deadline=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),receipt=receipt)
            self.assertEqual(result,124);self.assertFalse(marker.exists());self.assertTrue(receipt.exists())

    def test_supervisor_kills_group_even_when_term_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt=Path(folder)/'deadline.json'
            started=time.monotonic()
            result=self.worker.supervise([sys.executable,'-c',
                'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)'],
                deadline=(datetime.now(timezone.utc)+timedelta(seconds=5.3)).isoformat(),
                receipt=receipt,grace_seconds=.05)
            self.assertEqual(result,124);self.assertLess(time.monotonic()-started,3.)
            self.assertTrue(receipt.is_file())

    def test_original_core_workers_keep_exec_behavior(self):
        with tempfile.TemporaryDirectory() as folder:
            atomic_json(Path(folder)/'config.json',dict(study_kind='original'))
            with patch.object(sys,'argv',['worker.py',folder,'phase','attempt']), \
                 patch.object(self.worker.os,'execv',side_effect=SystemExit(0)) as call:
                with self.assertRaises(SystemExit):self.worker.main()
            self.assertEqual(call.call_args.args[0],sys.executable)
            self.assertIn('gavd6_sjepa.research_directions.gait_fidelity',call.call_args.args[1])

    def test_descendant_is_killed_when_leader_exits_on_term(self):
        with tempfile.TemporaryDirectory() as folder:
            pidfile=Path(folder)/'descendant.pid'
            heartbeat=Path(folder)/'heartbeat'
            child=('import signal,time;from pathlib import Path;'
                   'signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                   f'p=Path({str(heartbeat)!r})\n'
                   'for _ in range(500):\n p.open("a").write(".");time.sleep(.01)\n')
            leader=('import subprocess,sys,time;from pathlib import Path;'
                    f'p=subprocess.Popen([sys.executable,"-c",{child!r}]);'
                    f'Path({str(pidfile)!r}).write_text(str(p.pid));time.sleep(20)')
            self.worker.supervise([sys.executable,'-c',leader],
                deadline=(datetime.now(timezone.utc)+timedelta(seconds=5.4)).isoformat(),
                receipt=Path(folder)/'deadline.json',grace_seconds=.05)
            self.assertTrue(pidfile.is_file())
            pid=int(pidfile.read_text())
            try:
                self.assertTrue(heartbeat.is_file(),'The TERM-ignoring descendant never started')
                size=heartbeat.stat().st_size
                time.sleep(.15)
                self.assertEqual(heartbeat.stat().st_size,size,
                                 'A descendant continued executing after the deadline process-group kill')
            finally:
                try:os.kill(pid,9)
                except (ProcessLookupError,PermissionError):pass


if __name__=='__main__':
    unittest.main()
