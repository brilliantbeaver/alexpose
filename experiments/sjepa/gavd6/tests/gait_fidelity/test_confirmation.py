"""Confirmation plumbing uses analytic tensors; no human evidence is generated."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from gavd6_sjepa.research_directions.gait_fidelity.common import atomic_json, read_json, sha256
from gavd6_sjepa.research_directions.gait_fidelity.confirmation import evaluate_confirmation, lock_confirmation
from gavd6_sjepa.research_directions.gait_fidelity.data import fixture_bundle, save_dataset
from gavd6_sjepa.research_directions.gait_fidelity.training import train_phase
from gavd6_sjepa.research_directions.gait_fidelity.evaluation import fit_calibration


class ConfirmationTests(unittest.TestCase):
    def test_frozen_models_and_calibration_evaluate_separate_locked_population(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            # The contract is simulated using the analytic fixture. This is an
            # execution test; its source-style labels are never research output.
            bundle = fixture_bundle(samples=16, people=4)
            bundle.evidence_status = 'technical-source-screen'
            for row in bundle.records:
                row.update(review_mode='technical_geometry', locomotion_status='metadata_candidate')
            cfg = dict(work=str(root), device='cpu', fixture=False,
                cohort=dict(plan_path=str(root/'cohort.json')),
                model=dict(width=8,encoder_layers=1,predictor_layers=1,heads=2,patch_size=4,window_size=16),
                training=dict(sampling='person_motion',end_to_end_updates=1,batch_size=4,learning_rate=.0003),
                seeds=[17], measurement=dict(min_frames=8,min_coverage=.5,min_segment_px=2.,sign_tolerance_deg=1.),
                evaluation=dict(primary_candidate='candidate',primary_comparator='control',bootstrap_draws=30,bootstrap_seed=17))
            recipe = dict(encoder='direct',pretraining_mask=None,readout_or_training_objective='base',recipe_id='candidate')
            receipt = train_phase(bundle,recipe,'end_to_end',17,cfg,root/'fit')
            atomic_json(root/'config.json',cfg); atomic_json(root/'plan.json',{})
            atomic_json(root/'calibration.json',fit_calibration(bundle.subset('train'),balanced=True))
            checkpoint = receipt['checkpoint']
            lock = dict(schema='gf-confirmation-lock-v1',cohort_identity='planned',
                checkpoint_hashes={checkpoint:receipt['checkpoint_sha256']},
                fits=[dict(method=name,seed=17,checkpoint=checkpoint) for name in ('candidate','control')],
                seeds=[17],measurement=cfg['measurement'],evaluation=cfg['evaluation'],model=cfg['model'],
                calibration=str(root/'calibration.json'),calibration_sha256=sha256(root/'calibration.json'),
                config_sha256=sha256(root/'config.json'),plan_sha256=sha256(root/'plan.json'),
                claim_scope='analytic execution test only')
            confirmation = bundle.subset('development')
            for row in confirmation.records:
                row.update(split='confirmation',original_split='test',exposure='unexposed_verified',reserved=False)
            lock['person_ids'] = sorted({r['canonical_person_id'] for r in confirmation.records})
            atomic_json(root/'lock.json',lock)
            confirmation.provenance = dict(confirmation.provenance,cohort_identity='planned',confirmation_admitted=True,
                                            confirmation_lock_sha256=sha256(root/'lock.json'))
            path = save_dataset(confirmation,root/'confirmation-data',storage='npy')
            expected = pd.DataFrame([dict(source_family_id=family) for family in sorted({r['source_family_id'] for r in confirmation.records})])
            with patch('gavd6_sjepa.research_directions.gait_fidelity.scheduler._verify_frozen'), patch(
                    'gavd6_sjepa.research_directions.gait_fidelity.cohort.load_cohort',return_value=(expected,[],'planned')):
                result = evaluate_confirmation(cfg,path,root/'lock.json',root/'evaluation')
                self.assertTrue(result['complete_planned_population'])
                self.assertEqual(result['evaluated_people'],2)
                self.assertTrue((root/'evaluation/comparisons.json').is_file())
                with self.assertRaisesRegex(FileExistsError,'immutable'):
                    evaluate_confirmation(cfg,path,root/'lock.json',root/'evaluation')
                atomic_json(root/'calibration.json',{'changed':True})
                with self.assertRaisesRegex(RuntimeError,'calibration changed'):
                    evaluate_confirmation(cfg,path,root/'lock.json',root/'second')
                self.assertFalse((root/'second').exists())
                self.assertTrue(list(root.glob('.second-staging-*/failure.json')))

    def test_fixture_cannot_declare_source_confirmation(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError,'Source confirmation'):
                lock_confirmation({'fixture':True},Path(temporary)/'lock.json',reviewed_by='reviewer',evidence='fixture')


if __name__ == '__main__': unittest.main()
