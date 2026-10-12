"""Preparation timing uses opened development only and cannot overpromise resources."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from gavd6_sjepa.research_directions.gait_fidelity.common import atomic_json, read_json, sha256
from gavd6_sjepa.research_directions.gait_fidelity.data import fixture_bundle, load_dataset, save_dataset
from gavd6_sjepa.research_directions.gait_fidelity.repair_cohort import plan_slim_cohort
from gavd6_sjepa.research_directions.gait_fidelity.repair_profile import (
    plan_benchmark, benchmark_preparation, admit_confirmation, audit_expansion,
)


class RepairProfileTest(unittest.TestCase):
    def setUp(self):
        temporary=TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()
        self.parent=dict(mode='fixture',fixture=True,data=dict(samples=16,hz=25.))
        parent_path=self.root/'parent.json';atomic_json(parent_path,self.parent)
        bundle=save_dataset(fixture_bundle(samples=16,people=4),self.root/'parent-bundle')
        self.config=deepcopy(self.parent)
        self.config.update(work=str(self.root/'work'),asset_root=str(self.root),
            resources=dict(max_jobs=4,gpu_hours=48.,prepare_wall_minutes=240),
            repair=dict(parent_config=str(parent_path),parent_bundle=str(bundle),
                cohort_plan=str(self.root/'cohort.json'),benchmark_plan=str(self.root/'benchmark-plan.json'),
                deadline_utc='2100-01-01T00:00:00Z',preparation_safety_factor=1.5,
                queue_allowance_hours=1.,evaluation_allowance_hours=2.,
                binding_files={str(parent_path):sha256(parent_path),str(bundle/'manifest.json'):sha256(bundle/'manifest.json')}))
        plan_slim_cohort(self.parent,self.config['repair']['cohort_plan'])

    def test_benchmark_plan_is_exact_retained_development_and_immutable(self):
        with patch('numpy.load',side_effect=AssertionError('Planning must not read arrays')):
            plan=plan_benchmark(self.config)
        self.assertEqual(len(plan['records']),2)
        self.assertEqual({r['split'] for r in plan['records']},{'development'})
        self.assertEqual({r['original_split'] for r in plan['records']},{'validation'})
        self.assertFalse(plan['protected_confirmation_opened'])
        self.assertEqual(plan,plan_benchmark(self.config))
        modified=read_json(self.config['repair']['benchmark_plan']);modified['records'][0]['split']='confirmation'
        atomic_json(self.config['repair']['benchmark_plan'],modified)
        with self.assertRaises(FileExistsError):plan_benchmark(self.config)

    def test_bound_metadata_tamper_and_test_targets_are_rejected(self):
        path=Path(self.config['repair']['parent_bundle'])/'manifest.json'
        value=read_json(path);value['records'][0]['split']='confirmation';atomic_json(path,value)
        with self.assertRaisesRegex(PermissionError,'changed'):plan_benchmark(self.config)
        self.config['repair']['binding_files'][str(path)]=sha256(path)
        with self.assertRaisesRegex(PermissionError,'confirmation targets'):plan_benchmark(self.config)

    def test_fixture_runs_actual_small_pipeline_and_never_calls_it_hardware_evidence(self):
        result=benchmark_preparation(self.config,self.root/'measured')
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['retained_windows'],2)
        self.assertGreater(result['elapsed_seconds'],0)
        self.assertEqual(result['timing_evidence'],'software_fixture_not_HAIC_runtime')
        data=load_dataset(result['bundle'])
        self.assertEqual({r['split'] for r in data.records},{'development'})
        self.assertEqual({r['extractor_family'] for r in data.records},{'rtmpose','vitpose'})
        self.assertEqual({r['movement_magnitude'] for r in data.records},{0.,5.,15.})
        self.assertFalse(data.provenance['confirmation_admitted'])
        self.assertTrue(admit_confirmation(self.config,result)['admitted'])
        receipt=read_json(result['receipt']);receipt['elapsed_seconds']=.00001;atomic_json(result['receipt'],receipt)
        with self.assertRaisesRegex(PermissionError,'receipt changed'):admit_confirmation(self.config,result)

    def source_admission(self,*,remaining_hours=10,used=0,complete=True,setup=60,maximum=120,state=None):
        config=deepcopy(self.config);config.update(mode='source',fixture=False)
        config['repair']['deadline_utc']=(datetime.now(timezone.utc)+timedelta(hours=remaining_hours)).isoformat()
        plan=dict(records=[{}]*28,person_ids=list(range(14)))
        benchmark=dict(status='complete' if complete else 'incomplete',retained_windows=2 if complete else 1,
            maximum_family_seconds=maximum,setup_seconds=setup)
        with patch('gavd6_sjepa.research_directions.gait_fidelity.repair_profile._verify_benchmark_result',return_value=benchmark),\
             patch('gavd6_sjepa.research_directions.gait_fidelity.repair_profile._read_plan',return_value=(None,plan)),\
             patch('gavd6_sjepa.research_directions.gait_fidelity.scheduler._state',
                   return_value=state or dict(attempts=[dict(reserved_gpu_hours=used)])):
            return admit_confirmation(config,{})

    def test_admission_counts_setup_each_worker_longest_shard_and_reserves(self):
        result=self.source_admission()
        self.assertTrue(result['admitted'])
        self.assertEqual(result['windows_in_longest_shard'],7)
        self.assertAlmostEqual(result['projected_preparation_gpu_hours'],1.5)
        self.assertAlmostEqual(result['projected_longest_shard_hours'],.375)
        self.assertAlmostEqual(result['required_wall_hours'],7+4/60)
        self.assertEqual(result['worker_reservation_gpu_hours'],18)
        self.assertEqual(result['projected_evaluation_gpu_hours'],2)
        self.assertEqual(result['projected_additional_gpu_hours'],3.5)
        self.assertFalse(result['scientific_result_guaranteed'])

    def test_admission_refuses_deadline_short_budget_incomplete_and_large_family(self):
        for kwargs,reason in [
            ({'remaining_hours':-1},'experiment_deadline_passed'),
            ({'remaining_hours':3},'insufficient_deadline_for_preparation_queue_and_evaluation_reserve'),
            ({'used':47},'insufficient_remaining_gpu_allocation'),
            ({'used':40},'insufficient_cap_for_all_declared_worker_reservations'),
            ({'complete':False},'benchmark_did_not_complete_both_declared_source_windows'),
            ({'maximum':2000},'longest_projected_shard_exceeds_worker_time_limit')]:
            result=self.source_admission(**kwargs)
            self.assertFalse(result['admitted']);self.assertIn(reason,result['reasons'])

    def test_resume_does_not_charge_completed_or_already_reserved_phases_twice(self):
        completed={}
        for index in range(4):
            path=self.root/f'completed-shard-{index}.json'
            atomic_json(path,dict(phase_id=f'confirmation-shard-{index:02d}',artifacts={}))
            completed[f'confirmation-shard-{index:02d}']=dict(receipt=str(path),sha256=sha256(path))
        state=dict(completed=completed,attempts=[dict(phase_id='confirmation-shard-00',status='complete',
            reserved_gpu_hours=4.,allocated_gpu_hours=.5)])
        result=self.source_admission(remaining_hours=4,state=state)
        self.assertTrue(result['admitted']);self.assertEqual(result['remaining_preparation_shards'],0)
        self.assertEqual(result['projected_preparation_gpu_hours'],0)
        self.assertEqual(result['worker_reservation_gpu_hours'],2)
        self.assertAlmostEqual(result['required_wall_hours'],3+2/60)
        state['attempts'].append(dict(phase_id='confirmation-evaluate',status='running',reserved_gpu_hours=2.))
        result=self.source_admission(remaining_hours=3,state=state)
        self.assertTrue(result['admitted'])
        self.assertEqual(result['charged_or_reserved_gpu_hours'],2.5)
        self.assertEqual(result['worker_reservation_gpu_hours'],0)
        self.assertEqual(result['projected_additional_gpu_hours'],0)
        self.assertAlmostEqual(result['required_wall_hours'],2+2/60)

    def test_cmu_capacity_uses_raw_inventory_not_the_excluded_eligible_cohort(self):
        directory=self.root/'manifests/amass';directory.mkdir(parents=True)
        pd.DataFrame([dict(source_dataset='CMU',relative_path='CMU/01/01_01.npz',subject_id_candidate='CMU::01',
            num_frames=250,mocap_framerate=25)]).to_csv(directory/'amass_raw_inventory.csv',index=False)
        pd.DataFrame(columns=['source_dataset','relative_path','subject_id_candidate','num_frames','mocap_framerate']).to_csv(
            directory/'amass_raw_inventory_eligible.csv',index=False)
        result=audit_expansion(self.config)
        self.assertEqual(result['cmu']['motions'],1)
        self.assertEqual(result['cmu']['candidate_folders'],1)
        self.assertIsNone(result['cmu']['verified_independent_people'])
        self.assertFalse(result['cmu']['ready'])

    def test_missing_expansion_assets_remain_not_ready_and_do_not_change_protocol(self):
        before=deepcopy(self.config)
        result=audit_expansion(self.config)
        self.assertFalse(result['cmu']['ready']);self.assertFalse(result['gavd']['ready'])
        self.assertIsNone(result['cmu']['verified_independent_people'])
        self.assertFalse(result['media_decoded']);self.assertFalse(result['raw_motion_arrays_opened'])
        self.assertEqual(self.config,before)


class SourceBenchmarkTest(unittest.TestCase):
    def test_actual_source_adapter_uses_two_open_windows_and_both_estimators(self):
        from tests.gait_fidelity.test_source_pipeline import SourcePipelineTest,_Body,_load_estimator,_motion,_render
        from gavd6_sjepa.research_directions.gait_fidelity.cohort import plan_cohort
        with TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();helper=SourcePipelineTest();helper.root=root
            parent=helper.make_source_configuration();parent['fixture']=False
            inv_path=root/'manifests/amass_raw_inventory_eligible.csv'
            inventory=pd.read_csv(inv_path);inventory['source_dataset']='test';inventory['motion_id']='normal_walk'
            inventory['sha256']=[sha256(root/'amass'/r) for r in inventory.relative_path]
            for index in range(2):
                relative=f'held/walk-{index}.npz';raw=root/'amass'/relative;raw.parent.mkdir(exist_ok=True);raw.write_bytes(relative.encode())
                inventory.loc[len(inventory)]=dict(relative_path=relative,subject_id_candidate='held',status='ok',num_frames=128,
                    mocap_framerate=25,source_dataset='test',motion_id='normal_walk',sha256=sha256(raw))
            inventory.to_csv(inv_path,index=False)
            for filename,row in [('amass_subject_registry.csv',dict(subject_id_candidate='held',identity='held',identity_audit_status='approved',excluded=False)),
                                 ('amass_subject_splits.csv',dict(identity='held',split='test'))]:
                path=root/'manifests'/filename;frame=pd.read_csv(path);pd.concat([frame,pd.DataFrame([row])]).to_csv(path,index=False)
            parent['cohort']=dict(preset='named_walking',plan_path=str(root/'parent-cohort/manifest.json'))
            full=plan_cohort(parent,root/'parent-cohort')
            # Only metadata is needed for the timing selection; this explicit
            # software adapter receipt represents previously retained windows.
            prior=root/'retained';prior.mkdir()
            atomic_json(prior/'manifest.json',dict(records=[r for r in full['records'] if r['split']!='confirmation']))
            parent_path=root/'parent.json';atomic_json(parent_path,parent)
            config=deepcopy(parent);config.update(work=str(root/'work'),resources=dict(max_jobs=4,gpu_hours=48,prepare_wall_minutes=240))
            config['repair']=dict(parent_config=str(parent_path),parent_bundle=str(prior),deadline_utc='2100-01-01T00:00:00Z',
                cohort_plan=str(root/'slim.json'),benchmark_plan=str(root/'benchmark.json'),
                binding_files={str(parent_path):sha256(parent_path),str(prior/'manifest.json'):sha256(prior/'manifest.json')})
            plan_slim_cohort(parent,config['repair']['cohort_plan'])
            opened=[]
            def checked_motion(row,*args,**kwargs):
                opened.append(row['split']);self.assertEqual(row['split'],'development')
                return _motion(row,*args,**kwargs)
            with ExitStack() as stack:
                for target,replacement in (
                    ('gavd6_sjepa.research_directions.synthetic_training_v2.runtime.require_haic_runtime',lambda:{'test_backend':True}),
                    ('gavd6_sjepa.research_directions.motion_preservation.motion_data.load_motion',checked_motion),
                    ('gavd6_sjepa.research_directions.motion_preservation.body_geometry.SMPLHBody',_Body),
                    ('gavd6_sjepa.research_directions.synthetic_training.estimators.load_estimator',_load_estimator),
                    ('gavd6_sjepa.research_directions.gait_fidelity.preparation._render_fixed',_render)):
                    stack.enter_context(patch(target,replacement))
                result=benchmark_preparation(config,root/'timed-source')
            self.assertEqual(opened,['development','development'])
            self.assertEqual(result['status'],'complete')
            self.assertEqual(len(result['family_seconds']),2)
            self.assertGreaterEqual(result['elapsed_seconds'],sum(result['family_seconds'].values()))
            data=load_dataset(result['bundle'])
            self.assertEqual(data.provenance['source_selection'],'repair_benchmark')
            self.assertEqual({r['locomotion_status'] for r in data.records},{'metadata_candidate'})
            self.assertEqual({r['movement_magnitude'] for r in data.records},{0.,5.,15.})
            self.assertEqual({r['extractor_family'] for r in data.records},{'rtmpose','vitpose'})
            self.assertTrue(admit_confirmation(config,result)['admitted'])


if __name__=='__main__':unittest.main()
