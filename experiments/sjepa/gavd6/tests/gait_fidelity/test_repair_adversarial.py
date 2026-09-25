"""Independent review: reject scientifically mislabeled confirmation inputs."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import tempfile
import unittest
from unittest.mock import patch

from gavd6_sjepa.research_directions.gait_fidelity.common import sha256
from gavd6_sjepa.research_directions.gait_fidelity.repair_cohort import _fits, lock_slim_confirmation
from gavd6_sjepa.research_directions.gait_fidelity.repair_profile import admit_confirmation


class RepairAdmissionAdversarialTests(unittest.TestCase):
    def _admission(self, *, remaining_hours, used):
        config = dict(mode='source', fixture=False, work='/unused/adversarial-test',
            resources=dict(max_jobs=4, gpu_hours=48., prepare_wall_minutes=240),
            repair=dict(deadline_utc=(datetime.now(timezone.utc) + timedelta(hours=remaining_hours)).isoformat(),
                preparation_safety_factor=1.5, queue_allowance_hours=1., evaluation_allowance_hours=2.))
        plan = dict(records=[{}] * 28, person_ids=list(range(14)))
        benchmark = dict(status='complete', retained_windows=2,
            maximum_family_seconds=120., setup_seconds=60.)
        with patch('gavd6_sjepa.research_directions.gait_fidelity.repair_profile._verify_benchmark_result', return_value=benchmark), \
             patch('gavd6_sjepa.research_directions.gait_fidelity.repair_profile._read_plan', return_value=(None, plan)), \
             patch('gavd6_sjepa.research_directions.gait_fidelity.scheduler._state',
                   return_value=dict(attempts=[dict(reserved_gpu_hours=used)])):
            return admit_confirmation(config, {})

    def test_admission_reserves_evaluation_after_preparation_allocations(self):
        # 31 already charged + four 4-hour workers = 47, leaving only one
        # GPU-hour for the mandatory two-hour evaluation allocation.
        result = self._admission(remaining_hours=10., used=31.)
        self.assertFalse(result['admitted'])

    def test_admission_requires_declared_slurm_allocation_to_fit_cutoff(self):
        # A short measured workload cannot make a declared four-hour Slurm
        # allocation fit when only 3.5 hours remain before the frozen cutoff.
        result = self._admission(remaining_hours=3.5, used=0.)
        self.assertFalse(result['admitted'])

    def test_same_seed_wrong_objective_checkpoint_cannot_be_called_dense(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'scalar.pt'
            checkpoint.write_bytes(b'completed-scalar-checkpoint')
            fit = dict(method='R-repair-jepa_delta_v1-dense_change', seed=17,
                       checkpoint=str(checkpoint), sha256=sha256(checkpoint))
            signature = dict(seed=17, evidence_status='technical-source-screen',
                phase='readout', encoder='paired_jepa', policy='graph_time',
                representation_variant='jepa_delta_v1', objective='scalar_low',
                repair=dict(format='gait-fidelity-readout-repair-v1'))
            with patch('gavd6_sjepa.research_directions.gait_fidelity.training.load_model',
                       return_value=(None, dict(signature=signature))), self.assertRaises(ValueError):
                _fits([fit], fixture=False)

    def test_source_test_cannot_open_before_all_fixed_fits_are_declared(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'dense.pt'; checkpoint.write_bytes(b'completed-dense-checkpoint')
            fit = dict(method='R-repair-jepa_delta_v1-dense_change', seed=17,
                       checkpoint=str(checkpoint), sha256=sha256(checkpoint))
            signature = dict(seed=17, evidence_status='technical-source-screen',
                phase='readout', encoder='paired_jepa', policy='graph_time',
                representation_variant='jepa_delta_v1', objective='dense_change',
                repair=dict(format='gait-fidelity-readout-repair-v1'))
            config = dict(mode='source', fixture=False, repair=dict(deadline_utc='2100-01-01T00:00:00Z',
                statistical_protocol=dict(primary_candidate=fit['method'], primary_metric='waveform_error')))
            plan_path = root / 'plan.json'; plan_path.write_text('{}')
            plan = dict(fixture=False, identity='cohort', parent_cohort_identity='parent',
                        person_ids=['new-person'], source_family_ids=['new-family'])
            exposure = root / 'exposure.csv'; exposure.write_text('reviewed source exposure')
            with patch('gavd6_sjepa.research_directions.gait_fidelity.repair_cohort._read_plan',
                       return_value=(plan_path, plan)), \
                 patch('gavd6_sjepa.research_directions.gait_fidelity.repair_cohort._exposure'), \
                 patch('gavd6_sjepa.research_directions.gait_fidelity.training.load_model',
                       return_value=(None, dict(signature=signature))), self.assertRaises(ValueError):
                lock_slim_confirmation(config, [fit], exposure_ledger=exposure,
                    reviewed_by='reviewer', evidence='adversarial software test', output=root / 'lock.json')
            self.assertFalse((root / 'lock.json').exists())


if __name__ == '__main__':
    unittest.main()
