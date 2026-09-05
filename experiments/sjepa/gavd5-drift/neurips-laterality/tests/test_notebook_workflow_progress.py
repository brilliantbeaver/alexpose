"""Tests for notebook-only evaluation, reporting, and gate progress."""

from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from notebook_progress import (  # noqa: E402
    NotebookTaskProgress,
    aggregate_and_save_with_progress,
    evaluate_selected_with_progress,
)


class NotebookTaskProgressTests(unittest.TestCase):
    def test_eta_ignores_cached_units_and_uses_completed_work(self):
        now = [0.0]
        progress = NotebookTaskProgress(
            "Evaluation",
            "job",
            refresh_seconds=0.0,
            clock=lambda: now[0],
        )
        progress._publish = Mock()  # type: ignore[method-assign]
        progress.start(4, cached_candidate_units=1)
        progress.start_unit(1, "cached", candidate_cached=True)
        progress.complete_unit(reused=True, duration_seconds=0.01)
        self.assertIsNone(progress._seconds_per_unit())

        progress.start_unit(2, "computed")
        progress.complete_unit(duration_seconds=10.0)
        self.assertAlmostEqual(progress._remaining_seconds(), 20.0)
        self.assertAlmostEqual(progress._progress_fraction(), 0.5)
        self.assertIn("50.0%", progress.as_html())

    def test_blocked_workflow_explains_that_it_was_not_run(self):
        progress = NotebookTaskProgress("External gate", "stage")
        progress._publish = Mock()  # type: ignore[method-assign]
        progress.start(1)
        progress.block("governance is unresolved")
        rendered = progress.as_html()
        self.assertIn("Blocked / not run", rendered)
        self.assertIn("governance is unresolved", rendered)

    def test_optional_skip_finishes_preflight_without_claiming_computation(self):
        progress = NotebookTaskProgress("External gate", "check")
        progress._publish = Mock()  # type: ignore[method-assign]
        progress.start(2)
        progress.start_unit(1, "discover configuration")
        progress.complete_unit(duration_seconds=0.01)

        progress.finish_skipped("optional study was not configured")

        self.assertEqual(progress.completed_units, 2)
        self.assertEqual(progress.computed_units, 1)
        self.assertEqual(progress.skipped_units, 1)
        self.assertEqual(progress._progress_fraction(), 1.0)
        rendered = progress.as_html()
        self.assertIn("100.0%", rendered)
        self.assertIn("Not configured / not run", rendered)
        self.assertIn("skipped 1", rendered)
        self.assertIn("optional study was not configured", rendered)
        self.assertIn("#b7791f", rendered)

    def test_terminal_block_can_account_for_an_unrunnable_check(self):
        progress = NotebookTaskProgress("External gate", "check")
        progress._publish = Mock()  # type: ignore[method-assign]
        progress.start(2)
        progress.start_unit(1, "discover configuration")
        progress.complete_unit(duration_seconds=0.01)

        progress.block("one required path is missing", account_for_remaining=True)

        self.assertEqual(progress.completed_units, 2)
        self.assertEqual(progress.skipped_units, 1)
        self.assertEqual(progress._progress_fraction(), 1.0)
        rendered = progress.as_html()
        self.assertIn("100.0%", rendered)
        self.assertIn("Blocked / not run", rendered)
        self.assertIn("one required path is missing", rendered)

    def test_unit_context_marks_an_exception_as_failed(self):
        progress = NotebookTaskProgress("Audit", "stage")
        progress._publish = Mock()  # type: ignore[method-assign]
        progress.start(1)
        with self.assertRaisesRegex(ValueError, "bad input"):
            with progress.unit(1, "validate"):
                raise ValueError("bad input")
        self.assertEqual(progress.status, "Stopped because a stage failed")
        self.assertIn("ValueError: bad input", progress.error or "")


class NotebookWorkflowWrapperTests(unittest.TestCase):
    def test_evaluation_wrapper_preserves_job_order_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            context = SimpleNamespace(
                variants=("vanilla",),
                folds=(0,),
                seeds=(42, 43),
                artifact_root=Path(directory),
                profile="smoke",
            )
            calls: list[tuple[int, int, str]] = []

            def fake_evaluate(context, cohort, splits, fold, seed, variant):
                calls.append((fold, seed, variant))
                return pd.DataFrame({"seed": [seed], "variant": [variant]})

            progress = NotebookTaskProgress("Evaluation", "job")
            progress._publish = Mock()  # type: ignore[method-assign]
            with patch(
                "laterality.evaluation.evaluate_fold",
                side_effect=fake_evaluate,
            ):
                result = evaluate_selected_with_progress(
                    context,
                    object(),
                    object(),
                    progress=progress,
                )

        self.assertEqual(calls, [(0, 42, "vanilla"), (0, 43, "vanilla")])
        self.assertEqual(result["seed"].tolist(), [42, 43])
        self.assertEqual(progress.completed_units, 2)
        self.assertEqual(progress.status, "Held-out evaluation complete")

    def test_reporting_wrapper_restores_every_decorated_function(self):
        from laterality import reporting

        phase_counts = {
            "load_selected_evaluations": 1,
            "seed_average_predictions": 1,
            "metric_table": 2,
            "optimization_seed_table": 1,
            "bootstrap_table": 1,
            "checkpoint_bootstrap_table": 1,
            "native_symmetry_bootstrap_table": 1,
            "native_symmetry_seed_table": 1,
            "representation_equivariance_bootstrap_table": 1,
            "representation_equivariance_seed_table": 1,
            "_write_overview_figure": 1,
            "atomic_write_json": 1,
        }
        replacements = {name: Mock(name=name) for name in phase_counts}

        def fake_aggregate(context, cohort, splits):
            for name, count in phase_counts.items():
                for _ in range(count):
                    getattr(reporting, name)()
            return {"summary": "unchanged"}

        progress = NotebookTaskProgress("Report", "stage")
        progress._publish = Mock()  # type: ignore[method-assign]
        with ExitStack() as stack:
            for name, replacement in replacements.items():
                stack.enter_context(patch.object(reporting, name, replacement))
            stack.enter_context(
                patch.object(
                    reporting,
                    "aggregate_and_save",
                    side_effect=fake_aggregate,
                )
            )
            report = aggregate_and_save_with_progress(
                SimpleNamespace(profile="smoke"),
                object(),
                object(),
                progress=progress,
            )
            for name, replacement in replacements.items():
                self.assertIs(getattr(reporting, name), replacement)

        self.assertEqual(report, {"summary": "unchanged"})
        self.assertEqual(progress.completed_units, sum(phase_counts.values()))
        self.assertEqual(progress.status, "Statistical report complete")


if __name__ == "__main__":
    unittest.main()
