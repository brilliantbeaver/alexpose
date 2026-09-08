"""Integration checks for the real-input boundary and complete fold/seed workflow."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laterality_extensions.motion_gavd import (
    audit_training_masks, collect_gavd_grid, gavd_plan, grid_status, locked_pose_view,
    prepare_gavd_inputs, readout_contrasts, run_gavd_grid, study_inputs,
)


class InputTests(unittest.TestCase):
    def test_expanded_cache_requires_exact_registered_inventory_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            poses = root / "poses" / "normal"
            poses.mkdir(parents=True)
            np.savez(poses / "a.npz", extraction_version="v2", sequence=np.zeros((4, 33, 4)))
            np.savez(poses / "b.npz", extraction_version="gavd5_pose_v3_split_provenance", cache_origin_version="v2")
            np.savez(poses / "new.npz", extraction_version="gavd5_pose_v3_split_provenance")
            originals = {p.name: p.read_bytes() for p in poses.iterdir()}
            contract = {"pose_archive_count": 2,
                        "pose_inventory_sha256": hashlib.sha256(b"normal/a\nnormal/b\n").hexdigest()}
            context = SimpleNamespace(pose_root=poses.parent, artifact_root=root / "artifacts", protocol={"data": {
                "conditions": ["normal"], "inventory_contract": contract,
                "extraction_provenance": {"extraction_version_counts": {"v2": 2}}}})
            copied = locked_pose_view(context, log=None)
            self.assertEqual({p.name for p in copied.glob("normal/*.npz")}, {"a.npz", "b.npz"})
            self.assertEqual(locked_pose_view(context, log=None), copied)
            self.assertEqual(originals, {p.name: p.read_bytes() for p in poses.iterdir()})
            contract["pose_inventory_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "Cannot recover"):
                locked_pose_view(context, log=None)

    def test_partial_cohort_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cohort").mkdir()
            retained = root / "cohort/metadata.json"
            retained.write_text("retained partial input")
            with patch("laterality_extensions.motion_gavd.load_context", return_value=SimpleNamespace(artifact_root=root)):
                with self.assertRaisesRegex(RuntimeError, "Partial cohort"):
                    prepare_gavd_inputs(log=None)
            self.assertEqual(retained.read_text(), "retained partial input")

    def test_real_mode_is_explicit_and_never_falls_back(self):
        with patch("laterality_extensions.motion_gavd.prepare_gavd_inputs", side_effect=FileNotFoundError("GAVD missing")):
            with self.assertRaisesRegex(FileNotFoundError, "GAVD missing"):
                study_inputs(log=None)
        with self.assertRaises(ValueError):
            study_inputs(mode="typo", log=None)

    def test_audit_visits_only_declared_training_clips_and_is_label_blind(self):
        inputs = study_inputs(mode="synthetic", folds=(0, 1), seeds=(42, 46), log=None)
        first = audit_training_masks(inputs, experiments=("motion", "regions"), log=None)
        for (fold, seed), group in first["per_clip"].groupby(["fold", "seed"]):
            data = inputs["datasets"][fold]
            self.assertEqual(set(group.sequence_id), set(data.sequence_ids[data.train_rows]))
            self.assertFalse(set(group.source_id) & set(data.test_sources))
            self.assertEqual(len(group), len(data.train_rows) * 5)
        inputs["datasets"] = {f: replace(d, targets=d.targets + 1000) for f, d in inputs["datasets"].items()}
        second = audit_training_masks(inputs, experiments=("motion", "regions"), log=None)
        pd.testing.assert_frame_equal(first["per_clip"], second["per_clip"])


class CompleteGridTests(unittest.TestCase):
    def test_all_five_folds_and_seeds_train_test_cache_and_reopen_without_training(self):
        # Actual tiny optimization in 50 paired jobs: 125 encoders. Generated
        # fixtures keep this test cheap; production uses the same orchestration.
        torch.set_num_threads(1)
        with patch.dict(os.environ, {"LATERALITY_RESEARCH_CPU_THREADS": "1"}), tempfile.TemporaryDirectory() as temporary:
            inputs = study_inputs(mode="synthetic", log=None)
            self.assertEqual(len(inputs["census"]), 25)
            self.assertEqual(inputs["expected"].sequence_id.nunique(), len(inputs["datasets"][0].xyz))
            plan = gavd_plan(inputs, device="cpu", output_dir=temporary)
            self.assertEqual((plan["training_runs"], plan["optimizer_updates"]), (125, 125))
            missing = collect_gavd_grid(plan, inputs, log=None)
            self.assertEqual(missing["missing_jobs"], 50)
            self.assertNotIn("per_seed", missing)
            result = run_gavd_grid(plan, inputs, enabled=True, log=None)
            self.assertEqual(result["status"], "Complete")
            self.assertEqual(len(result["jobs"]), 50)
            predictions = result["predictions"]
            self.assertTrue(predictions.synthetic.all())
            keys = ["experiment", "condition", "representation", "seed", "sequence_id"]
            self.assertFalse(predictions.duplicated(keys).any())
            cohort_size = len(inputs["datasets"][0].xyz)
            self.assertEqual(len(predictions), 5 * 8 * 5 * cohort_size)
            self.assertTrue(result["per_seed"].evaluated_clips.eq(cohort_size).all())
            for job in result["jobs"].itertuples():
                data = inputs["datasets"][job.fold]
                schedule = np.load(Path(job.training_directory) / "source_schedule.npy")
                self.assertTrue(set(data.source_ids[schedule.ravel()]) <= set(data.train_sources))
                subset = predictions[(predictions.fold == job.fold) & (predictions.seed == job.seed)]
                self.assertEqual(set(subset.source_id), set(data.test_sources))
            with patch("laterality_extensions.motion_gavd.train_mask_study", side_effect=AssertionError("must not train")), \
                 patch("laterality_extensions.motion_gavd.evaluate_motion_readouts", side_effect=AssertionError("must reuse readouts")):
                reopened = collect_gavd_grid(plan, inputs, log=None)
            pd.testing.assert_frame_equal(predictions, reopened["predictions"])
            self.assertEqual(len(readout_contrasts(result["per_seed"])), 5 * 5 * 4)
            first_job = Path(result["jobs"].iloc[0].training_directory)
            checkpoint = first_job / "uniform.pt"
            checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "Corrupt training artifact"):
                collect_gavd_grid(plan, inputs, log=None)

    def test_real_plan_inherits_full_recipe_and_refuses_changed_workload(self):
        inputs = study_inputs(mode="synthetic", folds=(0,), seeds=(42,), log=None)
        # Inspect the real recipe without opening or pretending to train real input.
        from laterality_extensions.motion_structured_training import plan_mask_study
        real = plan_mask_study()
        self.assertEqual(real["training_runs"], 125)
        self.assertEqual(real["optimizer_updates"], 150000)
        self.assertEqual(real["settings"]["steps"], 1200)
        self.assertEqual(real["settings"]["embed_dim"], 96)
        plan = gavd_plan(inputs)
        plan["workload"].loc[0, "seed"] = 44
        with self.assertRaisesRegex(ValueError, "Workload"):
            grid_status(plan, inputs)


if __name__ == "__main__":
    unittest.main()
