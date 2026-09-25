"""Recovery changes only common training eligibility and retains frozen lineage."""
import copy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from gavd6_sjepa.research_directions.gait_fidelity.common import atomic_json, read_json, sha256
from gavd6_sjepa.research_directions.gait_fidelity.config import initialize, freeze
from gavd6_sjepa.research_directions.gait_fidelity.data import fixture_bundle, select_rows, save_dataset, load_dataset
from gavd6_sjepa.research_directions.gait_fidelity.preparation import merge_datasets
from gavd6_sjepa.research_directions.gait_fidelity.scheduler import _merge_preparation, _verify_frozen

TOOL = Path(__file__).resolve().parents[2] / 'tools/gait-fidelity/admit_training_cohort.py'
spec = importlib.util.spec_from_file_location('gf_training_admission', TOOL)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class TrainingAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name).resolve() / 'study'
        cfg = initialize(self.work, fixture=True, experiment_set='core')
        cfg.update(fixture=False, mode='source', device='cuda')
        cfg['data'].update(source_selection='full_manifest', num_shards=2)
        atomic_json(self.work/'config.json', cfg)
        freeze(cfg)
        self.cfg = cfg
        full = fixture_bundle(people=5, windows=2, samples=8)
        self.parent = select_rows(full, [i for i, r in enumerate(full.records)
            if r['source_family_id'] != 'fixture-p0-w1'])
        families = sorted({r['source_family_id'] for r in self.parent.records})
        ledger = dict(attempts=[], completed={})
        self.paths = []
        for index in range(2):
            phase = f'prepare-shard-{index:02d}'
            attempt = self.work/'attempts'/phase/'original'
            selected = set(families[index::2])
            part = select_rows(self.parent, [i for i,r in enumerate(self.parent.records) if r['source_family_id'] in selected])
            part.provenance = copy.deepcopy(full.provenance)
            part.provenance.update(identity='same-frozen-source', source_selection='full_manifest',
                shard_index=index, num_shards=2, frozen_people=[f'fixture-p{i}' for i in range(5)],
                source_families=sorted(selected), exclusions=[], assets={})
            path = save_dataset(part, attempt/'prepared/bundle', storage='npy')
            self.paths.append(path)
            receipt = attempt/'complete.json'
            atomic_json(receipt, dict(phase_id=phase, config_sha256=sha256(self.work/'config.json'),
                result=dict(bundle=str(path)), artifacts={str(p):sha256(p) for p in path.iterdir()}))
            ledger['attempts'].append(dict(phase_id=phase, status='complete', path=str(attempt)))
            ledger['completed'][phase] = dict(receipt=str(receipt), sha256=sha256(receipt), result=dict(bundle=str(path)))
        atomic_json(self.work/'ledger.json', ledger)

    def test_recovery_retains_parents_and_development_and_resumes_frozen_coordinator(self):
        with self.assertRaisesRegex(ValueError, 'fewer than two training source families'):
            merge_datasets(self.paths, self.work/'data/bundle')
        before = {p:sha256(p) for path in self.paths for p in path.iterdir()}
        immutable = {p:sha256(self.work/p) for p in ('config.json','plan.json','frozen.json')}
        _, parts, proposal = recovery.inspect(self.work)
        self.assertEqual(proposal['excluded_training_people'], {'fixture-p0':['fixture-p0-w0']})
        self.assertEqual(proposal['before']['train'], dict(people=3, windows=5))
        self.assertEqual(proposal['after']['train'], dict(people=2, windows=4))
        result = recovery.apply_admission(self.work, parts, proposal)
        self.assertEqual(result['status'], 'ADMITTED_BUNDLE_READY')
        merged = load_dataset(result['bundle'])
        self.assertFalse(any(r['canonical_person_id']=='fixture-p0' for r in merged.records))
        expected = {tuple(sorted(r.items())):i for i,r in enumerate(self.parent.records) if r['split']=='development'}
        seen = set()
        for i,row in enumerate(merged.records):
            if row['split']!='development': continue
            key=tuple(sorted(row.items()));seen.add(key);j=expected[key]
            for group in ('inputs','targets'):
                for name, value in getattr(merged, group).items():
                    np.testing.assert_array_equal(value[i],getattr(self.parent,group)[name][j])
        self.assertEqual(seen,set(expected))
        self.assertEqual(before,{p:sha256(p) for p in before})
        self.assertEqual(immutable,{p:sha256(self.work/p) for p in immutable})
        self.assertEqual(recovery.apply_admission(self.work,parts,proposal)['status'], 'ADMITTED_BUNDLE_ALREADY_PRESENT')
        _verify_frozen(self.cfg)
        _merge_preparation(self.cfg)
        entry = read_json(self.work/'ledger.json')['completed']['prepare']
        self.assertEqual(entry['result']['counts']['source_families'],8)
        self.assertTrue((self.work/'data/viewer.html').is_file())
        with self.assertRaisesRegex(ValueError,'already admitted'):
            recovery.inspect(self.work)

    def test_no_admission_after_training_starts(self):
        ledger=read_json(self.work/'ledger.json')
        ledger['attempts'].append(dict(phase_id='profile',status='failed'))
        atomic_json(self.work/'ledger.json',ledger)
        with self.assertRaisesRegex(ValueError,'before profiling'):
            recovery.inspect(self.work)

    def test_changed_source_manifest_is_rejected(self):
        (self.paths[0]/'manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Source manifest differs'):
            recovery.inspect(self.work)

    def test_existing_unrelated_combined_bundle_is_preserved(self):
        _, parts, proposal = recovery.inspect(self.work)
        dest=self.work/'data/bundle';dest.mkdir(parents=True)
        atomic_json(dest/'manifest.json',dict(provenance={}))
        before=sha256(dest/'manifest.json')
        with self.assertRaisesRegex(ValueError,'different combined bundle'):
            recovery.apply_admission(self.work,parts,proposal)
        self.assertEqual(before,sha256(dest/'manifest.json'))

    def test_explicit_person_must_match_preview_before_apply(self):
        args = ['admit_training_cohort.py', '--work', str(self.work), '--apply',
                '--expected-singleton', 'wrong-person']
        with patch('sys.argv',args), patch.object(recovery,'require_finished_coordinator'), \
             patch.object(recovery,'apply_admission') as apply:
            with self.assertRaisesRegex(ValueError,'exact previewed people'):
                recovery.main()
            apply.assert_not_called()
        self.assertFalse((self.work/'admissions').exists())

    def test_active_coordinator_is_rejected(self):
        atomic_json(self.work/'control/coordinator.json',dict(state='submitted',job_id='123'))
        with patch.object(recovery.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='123\n')):
            with self.assertRaisesRegex(ValueError,'queued or running'):
                recovery.require_finished_coordinator(self.work)


if __name__=='__main__':
    unittest.main()
