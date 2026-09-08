"""Independent regressions for recorded masking controls and cache integrity."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laterality.geometry import FULL_MIRROR_PAIRS
from laterality_extensions.comparative_masks import (
    MaskBudget, MaskPolicy, coverage_summary, reflect_sample,
)
from laterality_extensions.comparative_training import (
    _digest_array, load_comparison, masks_for_batch, save_comparison,
    train_comparison,
)
from laterality_extensions.masked_learning import LearningSettings, load_learning_dataset


class IndependentTrainingReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.data = load_learning_dataset()
        cls.settings = LearningSettings(steps=2, reflection_probability=1.0)
        cls.conditions = {
            "whole_trajectories": MaskPolicy("whole_trajectory"),
            "fixed_individual_set": MaskPolicy("random_subset", subset_seed=31),
        }
        cls.budgets = {"whole_trajectories": MaskBudget(trajectories=2)}
        cls.result = train_comparison(
            cls.data, cls.settings, cls.conditions, budgets=cls.budgets,
            matched_to="whole_trajectories", checkpoint_steps=(1, 2),
        )

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.prior_threads)

    def saved_fixture(self, directory):
        destination = Path(directory) / "complete_comparison"
        save_comparison(self.result, destination)
        return destination

    def read_manifest(self, destination):
        return json.loads((destination / "manifest.json").read_text())

    def write_manifest(self, destination, manifest):
        (destination / "manifest.json").write_text(json.dumps(manifest, allow_nan=False))

    def write_payload(self, destination, name, payload, manifest):
        path = destination / f"{name}.pt"
        torch.save(payload, path)
        manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_manifest(destination, manifest)

    def test_recorded_coverage_describes_the_reflected_training_observations(self):
        self.assertTrue(self.result["reflection_schedule"].all())
        permutation = np.arange(33)
        for left, right in FULL_MIRROR_PAIRS:
            permutation[[left, right]] = permutation[[right, left]]
        for step, rows in enumerate(self.result["source_schedule"]):
            masks, original_coverage = masks_for_batch(
                self.data, rows, self.settings, self.conditions, budgets=self.budgets,
                matched_to="whole_trajectories", step=step,
            )
            for name in self.conditions:
                for batch_row, dataset_row in enumerate(rows):
                    valid = self.data.valid[dataset_row].reshape(-1, 4, 33).all(axis=1)
                    _, reflected_valid, reflected_mask = reflect_sample(
                        self.data.xyz[dataset_row], valid, masks[name][batch_row],
                    )
                    expected = coverage_summary(reflected_mask, reflected_valid)
                    recorded = self.result["runs"][name]["coverage"][step][batch_row]
                    for key, value in expected.items():
                        self.assertEqual(recorded[key], value, msg=f"{name}: {key}")
                    self.assertTrue(recorded["reflected"])
                    original = original_coverage[name][batch_row]
                    for key in ("eligible_landmarks", "selected_landmarks"):
                        if key in original:
                            self.assertEqual(recorded[key], sorted(map(int, permutation[original[key]])))

    def test_structural_fields_cannot_be_ignored_for_a_scattered_budget(self):
        with self.assertRaisesRegex(ValueError, "Scattered masking"):
            masks_for_batch(
                self.data, self.data.train_rows[:3], self.settings,
                {"gait": MaskPolicy("gait"), "uniform": MaskPolicy("uniform")},
                budgets={"uniform": MaskBudget(hidden_count=3, trajectories=2)},
            )
        with self.assertRaisesRegex(ValueError, "reference budget differs"):
            masks_for_batch(
                self.data, self.data.train_rows[:3], self.settings, self.conditions,
                budgets={**self.budgets, "fixed_individual_set": MaskBudget(hidden_count=1)},
                matched_to="whole_trajectories",
            )

    def test_settings_and_policy_metadata_must_match_the_requested_identity(self):
        mutations = (
            lambda run: run["settings"].update(seed=999),
            lambda run: run["policy"].update(subset_seed=999),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as directory:
                destination = self.saved_fixture(directory)
                manifest = self.read_manifest(destination)
                mutate(manifest["runs"]["fixed_individual_set"])
                self.write_manifest(destination, manifest)
                with self.assertRaisesRegex(ValueError, "metadata disagree"):
                    load_comparison(destination, self.result["identity"])

    def test_pairing_is_recomputed_instead_of_trusting_saved_true_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = self.saved_fixture(directory)
            manifest = self.read_manifest(destination)
            self.assertTrue(all(manifest["pairing"].values()))
            manifest["runs"]["fixed_individual_set"]["view_digest"] = "different geometric views"
            self.write_manifest(destination, manifest)
            with self.assertRaisesRegex(ValueError, "controls did not pass"):
                load_comparison(destination, self.result["identity"])

    def test_mask_coverage_must_agree_with_recorded_counts_and_reflections(self):
        for key, value in (("hidden_tokens", 999), ("reflected", False), ("context_tokens", 0)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                destination = self.saved_fixture(directory)
                manifest = self.read_manifest(destination)
                manifest["runs"]["whole_trajectories"]["coverage"][0][0][key] = value
                self.write_manifest(destination, manifest)
                with self.assertRaisesRegex(ValueError, "coverage and target counts disagree"):
                    load_comparison(destination, self.result["identity"])

    def test_missing_or_partial_declared_checkpoints_fail_with_updated_checksum(self):
        for mutation in ("missing_update", "missing_parameter", "wrong_shape", "nonfinite", "final_disagreement"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                destination = self.saved_fixture(directory)
                name = "whole_trajectories"
                payload = torch.load(destination / f"{name}.pt", weights_only=True)
                manifest = self.read_manifest(destination)
                if mutation == "missing_update":
                    del payload["checkpoints"][1]
                else:
                    key = next(iter(payload["checkpoints"][1]))
                    if mutation == "missing_parameter":
                        del payload["checkpoints"][1][key]
                    elif mutation == "wrong_shape":
                        payload["checkpoints"][1][key] = torch.zeros(999)
                    elif mutation == "nonfinite":
                        payload["checkpoints"][1][key] = payload["checkpoints"][1][key].clone()
                        payload["checkpoints"][1][key].flatten()[0] = float("nan")
                    else:
                        payload["model"][key] = payload["model"][key] + 0.01
                self.write_payload(destination, name, payload, manifest)
                with self.assertRaises(ValueError):
                    load_comparison(destination, self.result["identity"])

    def test_test_sources_cannot_enter_a_checksum_consistent_saved_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = self.saved_fixture(directory)
            manifest = self.read_manifest(destination)
            schedule_path = destination / "source_schedule.npy"
            schedule = np.load(schedule_path, allow_pickle=False)
            schedule[0, 0] = self.data.test_rows[0]
            np.save(schedule_path, schedule, allow_pickle=False)
            manifest["files"][schedule_path.name] = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
            for run in manifest["runs"].values():
                run["source_draw_digest"] = _digest_array(schedule)
            self.write_manifest(destination, manifest)
            with self.assertRaisesRegex(ValueError, "test video|declared.*schedule|schedule.*declared"):
                load_comparison(destination, self.result["identity"])

    def test_training_schedule_still_must_match_its_declared_sampling_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = self.saved_fixture(directory)
            manifest = self.read_manifest(destination)
            schedule_path = destination / "source_schedule.npy"
            schedule = np.load(schedule_path, allow_pickle=False)
            replacement = next(row for row in self.data.train_rows if row != schedule[0, 0])
            schedule[0, 0] = replacement
            np.save(schedule_path, schedule, allow_pickle=False)
            manifest["files"][schedule_path.name] = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
            for run in manifest["runs"].values():
                run["source_draw_digest"] = _digest_array(schedule)
            self.write_manifest(destination, manifest)
            with self.assertRaises(ValueError):
                load_comparison(destination, self.result["identity"])

    def test_reflection_schedule_must_match_its_declared_sampling_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = self.saved_fixture(directory)
            manifest = self.read_manifest(destination)
            path = destination / "reflection_schedule.npy"
            reflections = np.load(path, allow_pickle=False)
            reflections[0, 0] = False  # The declared reflection probability is one.
            np.save(path, reflections, allow_pickle=False)
            manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for run in manifest["runs"].values():
                run["coverage"][0][0]["reflected"] = False
            self.write_manifest(destination, manifest)
            with self.assertRaises(ValueError):
                load_comparison(destination, self.result["identity"])

    def test_nonfinite_saved_projector_fails_even_with_updated_file_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = self.saved_fixture(directory)
            name = "whole_trajectories"
            payload = torch.load(destination / f"{name}.pt", weights_only=True)
            key = next(iter(payload["projector"]))
            payload["projector"][key] = payload["projector"][key].clone()
            payload["projector"][key].flatten()[0] = float("nan")
            self.write_payload(destination, name, payload, self.read_manifest(destination))
            with self.assertRaises(ValueError):
                load_comparison(destination, self.result["identity"])


if __name__ == "__main__":
    unittest.main()
