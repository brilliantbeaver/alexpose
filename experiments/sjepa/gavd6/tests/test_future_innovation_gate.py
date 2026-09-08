import unittest

from gavd6_sjepa.research_directions.future_innovation.fi_gate_decision import (
    decide_gate,
)


class FutureInnovationGateTests(unittest.TestCase):
    def valid(self):
        return {
            "delta_r2_real": 0.06,
            "delta_r2_time_shuffle": 0.02,
            "delta_r2_clip_mismatch": 0.005,
            "delta_r2_background_target": 0.02,
            "delta_r2_no_skeleton": 0.005,
            "motion_to_background_change_ratio": 3.0,
            "person_edit_direction_fraction": 0.9,
            "bootstrap_positive_fraction": 0.95,
            "seed_real_gains": [0.06, 0.06, 0.06],
            **{
                key: True
                for key in (
                    "data_contract_valid",
                    "evaluation_contract_valid",
                    "controls_complete",
                    "target_audit_complete",
                    "target_variance_valid",
                    "teacher_stable",
                    "causal_leakage_absent",
                    "capacity_control_clear",
                )
            },
        }

    def test_only_complete_stable_point_pass_advances_without_adapter_authority(self):
        result = decide_gate(self.valid())
        self.assertEqual(result["decision"], "ADVANCE")
        self.assertTrue(result["allow_full_experiment"])
        self.assertFalse(result["allow_adapter_training"])

    def test_every_validity_failure_stops(self):
        for key in (
            "data_contract_valid",
            "evaluation_contract_valid",
            "controls_complete",
            "target_audit_complete",
            "target_variance_valid",
            "teacher_stable",
            "causal_leakage_absent",
        ):
            for value in (False, None, 1):
                with self.subTest(key=key, value=value):
                    self.assertEqual(
                        decide_gate({**self.valid(), key: value})["decision"], "STOP"
                    )

    def test_point_failures_and_direction_consistency(self):
        for change in (
            {"delta_r2_time_shuffle": 0.04},
            {"delta_r2_clip_mismatch": 0.02},
            {"delta_r2_background_target": 0.04},
            {"motion_to_background_change_ratio": 1.0},
            {"person_edit_direction_fraction": 0.7},
            {"delta_r2_real": 0.04, "seed_real_gains": [0.04, 0.04, 0.04]},
        ):
            self.assertEqual(
                decide_gate({**self.valid(), **change})["decision"], "STOP"
            )

    def test_instability_and_unresolved_capacity_are_inconclusive(self):
        for change in (
            {"bootstrap_positive_fraction": 0.7},
            {"capacity_control_clear": False},
            {"seed_real_gains": [0.16, 0.01, 0.01]},
        ):
            self.assertEqual(
                decide_gate({**self.valid(), **change})["decision"], "INCONCLUSIVE"
            )

    def test_missing_seed_nonfinite_and_inconsistent_aggregation_stop(self):
        for change in (
            {"seed_real_gains": [0.06, 0.06]},
            {"seed_real_gains": [0.07, 0.07, 0.07]},
            {"delta_r2_no_skeleton": float("nan")},
            {"delta_r2_real": float("inf")},
            {"bootstrap_positive_fraction": 1.1},
        ):
            self.assertEqual(
                decide_gate({**self.valid(), **change})["decision"], "STOP"
            )
