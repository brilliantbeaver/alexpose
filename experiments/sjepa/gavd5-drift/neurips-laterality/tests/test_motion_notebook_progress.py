"""Progress observers preserve results, cache reuse and failure boundaries."""
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notebook_progress import (
    NotebookTaskProgress, audit_training_masks_with_progress, collect_gavd_grid_with_progress,
    evaluate_retained_motion_with_progress, grid_status_with_progress, run_gavd_grid_with_progress,
    run_notebook_task, study_inputs_with_progress,
)
from laterality_extensions import motion_gavd as workflow
from laterality_extensions import motion_structured_training as training
from laterality_extensions import motion_readout as readout


def display_recorder():
    progress = NotebookTaskProgress("Test task", "stage", refresh_seconds=0)
    progress.events = []
    progress._publish = lambda **kwargs: progress.events.append({
        "status": progress.status, "completed": progress.completed_units,
        "active": dict(progress.active) if progress.active else None,
    })
    return progress


class MotionNotebookProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.inputs = workflow.study_inputs(mode="synthetic", folds=(0, 1), seeds=(42, 46), log=None)

    def test_input_and_mask_wrappers_preserve_values_and_expose_nested_progress(self):
        progress = display_recorder()
        original_load = workflow.load_learning_dataset
        original_masks = workflow.paired_study_masks
        loaded = study_inputs_with_progress(mode="synthetic", folds=(0, 1), seeds=(42, 46), progress=progress)
        pd.testing.assert_frame_equal(loaded["memberships"], self.inputs["memberships"])
        self.assertEqual(progress.completed_units, 3)
        self.assertIs(workflow.load_learning_dataset, original_load)
        raw = workflow.audit_training_masks(self.inputs, experiments=("motion", "regions"), batch_size=7, log=None)
        progress = display_recorder()
        wrapped = audit_training_masks_with_progress(self.inputs, experiments=("motion", "regions"),
                                                     batch_size=7, progress=progress)
        for name in ("per_clip", "summary"):
            pd.testing.assert_frame_equal(raw[name], wrapped[name])
        self.assertIs(workflow.paired_study_masks, original_masks)
        self.assertEqual(progress.completed_units, 9)
        active = [event["active"] for event in progress.events if event["active"]]
        self.assertTrue(any("fold 1" in row["label"] and "seed 46" in row["label"] for row in active))
        self.assertTrue(any(0 < row["completed_steps"] < row["total_steps"] for row in active))
        self.assertTrue(progress.finished)

    def test_failure_and_keyboard_interrupt_restore_observed_functions(self):
        original_scores = workflow.motion_scores
        progress = display_recorder()
        interrupted = Mock(side_effect=KeyboardInterrupt("stop mask audit"))
        with patch.object(workflow, "paired_study_masks", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                audit_training_masks_with_progress(self.inputs, progress=progress)
            self.assertIs(workflow.paired_study_masks, interrupted)
        self.assertIs(workflow.motion_scores, original_scores)
        self.assertTrue(progress.finished)
        self.assertIn("KeyboardInterrupt", progress.error)
        self.assertLess(progress.completed_units, progress.total_units)

        failing_load = Mock(side_effect=ValueError("invalid cohort"))
        progress = display_recorder()
        with patch.object(workflow, "load_learning_dataset", failing_load):
            with self.assertRaisesRegex(ValueError, "invalid cohort"):
                study_inputs_with_progress(mode="synthetic", folds=(0,), progress=progress)
            self.assertIs(workflow.load_learning_dataset, failing_load)
        self.assertIn("invalid cohort", progress.error)

    def test_disabled_and_incomplete_grids_do_not_train_or_claim_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = workflow.study_inputs(mode="synthetic", folds=(0,), seeds=(42,), log=None)
            plan = workflow.gavd_plan(inputs, output_dir=folder)
            original_status, original_jobs = workflow.grid_status, workflow._jobs
            with patch.object(workflow, "train_mask_study", side_effect=AssertionError("must not train")):
                progress = display_recorder()
                result = run_gavd_grid_with_progress(plan, inputs, progress=progress, enabled=False)
                self.assertIn("disabled", result["status"])
                self.assertEqual(progress.computed_units, 1)
                self.assertGreater(progress.skipped_units, 0)
                self.assertIn("disabled", progress.status)
                progress = display_recorder()
                result = collect_gavd_grid_with_progress(plan, inputs, progress=progress)
                self.assertEqual(result["missing_jobs"], 2)
                self.assertIsNotNone(progress.blocked_reason)
                self.assertLess(progress.completed_units, progress.total_units)
            self.assertIs(workflow.grid_status, original_status)
            self.assertIs(workflow._jobs, original_jobs)
            progress = display_recorder()
            status = grid_status_with_progress(plan, inputs, progress=progress)
            self.assertEqual(len(status), 2)
            self.assertIn("2/2 jobs missing", progress.status)

    def test_training_readouts_and_cached_collection_are_unchanged(self):
        with patch.dict(os.environ, {"LATERALITY_RESEARCH_CPU_THREADS": "1"}), tempfile.TemporaryDirectory() as folder:
            inputs = workflow.study_inputs(mode="synthetic", folds=(0,), seeds=(42,), log=None)
            raw_plan = workflow.gavd_plan(inputs, device="cpu", output_dir=Path(folder) / "raw")
            wrapped_plan = workflow.gavd_plan(inputs, device="cpu", output_dir=Path(folder) / "wrapped")
            raw = workflow.run_gavd_grid(raw_plan, inputs, enabled=True, log=None)
            originals = {name: getattr(workflow, name) for name in (
                "train_mask_study", "_evaluation", "grid_status", "_jobs", "aggregate_motion_study")}
            original_masks, original_fit = training.paired_study_masks, readout.fit_source_readout
            progress = display_recorder()
            wrapped = run_gavd_grid_with_progress(wrapped_plan, inputs, progress=progress, enabled=True)
            for name in ("predictions", "selection", "diagnostics", "predictor_diagnostics", "per_seed", "summary", "paired_intervals"):
                pd.testing.assert_frame_equal(raw[name], wrapped[name])
            self.assertEqual(progress.completed_units, 8)
            self.assertEqual(progress.reused_units, 2)
            self.assertEqual(progress.new_candidate_units, progress.computed_units)
            active = [event["active"] for event in progress.events if event["active"]]
            self.assertTrue(any("Paired optimizer update 1/1" in row["detail"] for row in active))
            self.assertTrue(any("select ridge" in row["detail"] for row in active))
            self.assertTrue(any("Predictor correspondence" in row["detail"] for row in active))
            for name, original in originals.items():
                self.assertIs(getattr(workflow, name), original)
            self.assertIs(training.paired_study_masks, original_masks)
            self.assertIs(readout.fit_source_readout, original_fit)
            with patch.object(workflow, "train_mask_study", side_effect=AssertionError("must not train")), \
                 patch.object(workflow, "evaluate_motion_readouts", side_effect=AssertionError("must not refit")):
                progress = display_recorder()
                cached = collect_gavd_grid_with_progress(wrapped_plan, inputs, progress=progress)
            pd.testing.assert_frame_equal(wrapped["predictions"], cached["predictions"])
            self.assertEqual(progress.reused_units, 2)
            self.assertEqual(progress.completed_units, 4)
            self.assertEqual(progress.new_candidate_units, progress.computed_units)

            corrupt = Path(wrapped["jobs"].iloc[0].training_directory) / "uniform.pt"
            corrupt.write_bytes(corrupt.read_bytes() + b"corrupt")
            progress = display_recorder()
            with self.assertRaisesRegex(RuntimeError, "Corrupt training"):
                collect_gavd_grid_with_progress(wrapped_plan, inputs, progress=progress)
            self.assertIn("Corrupt training", progress.error)
            self.assertIs(workflow.grid_status, originals["grid_status"])
            self.assertIs(workflow._jobs, originals["_jobs"])

    def test_training_failure_restores_optimizer_callback_observers(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = workflow.study_inputs(mode="synthetic", folds=(0,), seeds=(42,), log=None)
            plan = workflow.gavd_plan(inputs, output_dir=folder)
            original_masks, original_eval = training.paired_study_masks, workflow._evaluation
            progress = display_recorder()
            interrupted = Mock(side_effect=KeyboardInterrupt("stop training"))
            with patch.object(workflow, "train_mask_study", interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    run_gavd_grid_with_progress(plan, inputs, progress=progress, enabled=True)
                self.assertIs(workflow.train_mask_study, interrupted)
            self.assertIs(training.paired_study_masks, original_masks)
            self.assertIs(workflow._evaluation, original_eval)
            self.assertIn("KeyboardInterrupt", progress.error)

    def test_optional_task_skip_and_result_passthrough(self):
        function = Mock(return_value=object())
        progress = display_recorder()
        self.assertIsNone(run_notebook_task(function, progress=progress, label="Optional readouts", enabled=False))
        function.assert_not_called()
        self.assertEqual(progress.skipped_units, 1)
        result = run_notebook_task(function, 3, progress=progress, label="Control", setting=4)
        self.assertIs(result, function.return_value)
        function.assert_called_once_with(3, setting=4)
        self.assertEqual(progress.skipped_units, 0)
        self.assertTrue(progress.finished)

    def test_retained_readout_reports_fitting_and_restores_observers(self):
        payload = object()
        original_fit = Mock(return_value="fitted")

        def evaluate(directory):
            self.assertEqual(directory, "retained-directory")
            readout.fit_source_readout("features", "targets", "sources")
            return payload

        progress = display_recorder()
        with patch.object(readout, "fit_source_readout", original_fit), \
             patch.object(readout, "evaluate_retained_comparison", side_effect=evaluate):
            result = evaluate_retained_motion_with_progress("retained-directory", progress=progress)
            self.assertIs(result, payload)
            self.assertIs(readout.fit_source_readout, original_fit)
        self.assertTrue(any(event["active"] and "select ridge" in event["active"]["detail"]
                            for event in progress.events))


if __name__ == "__main__":
    unittest.main()
