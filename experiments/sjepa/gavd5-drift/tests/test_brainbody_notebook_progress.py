from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "neurips-brain-body"
MODULE_PATH = NOTEBOOK_DIR / "notebook_progress.py"
SPEC = importlib.util.spec_from_file_location("brainbody_notebook_progress", MODULE_PATH)
progress_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(progress_module)
NotebookTaskProgress = progress_module.NotebookTaskProgress


class NotebookTaskProgressTests(unittest.TestCase):
    def test_reports_steps_eta_and_completion(self) -> None:
        now = [0.0]
        progress = NotebookTaskProgress(
            "Pipeline",
            "stage",
            refresh_seconds=0.0,
            clock=lambda: now[0],
        )
        progress._publish = Mock()  # type: ignore[method-assign]
        progress.start(2, profile="smoke")
        with progress.unit(1, "Prepare", total_steps=2):
            now[0] = 1.0
            progress.update_unit(completed_steps=1)
            self.assertAlmostEqual(progress._progress_fraction(), 0.25)
            now[0] = 2.0
        self.assertAlmostEqual(progress._remaining_seconds(), 2.0)
        with progress.unit(2, "Evaluate"):
            now[0] = 5.0
        progress.complete(status="Pipeline complete")
        self.assertEqual(progress.completed_units, 2)
        self.assertEqual(progress.status, "Pipeline complete")
        self.assertTrue(progress.finished)

    def test_marks_failures_and_optional_skips(self) -> None:
        progress = NotebookTaskProgress("Gate", "stage")
        progress._publish = Mock()  # type: ignore[method-assign]
        progress.start(3)
        with self.assertRaises(RuntimeError):
            with progress.unit(1, "Validate"):
                raise RuntimeError("bad contract")
        self.assertEqual(progress.error, "RuntimeError: bad contract")
        self.assertTrue(progress.finished)

        progress.start(3)
        with progress.unit(1, "Discover"):
            pass
        progress.finish_skipped("checkpoint unavailable")
        self.assertEqual(progress.completed_units, 3)
        self.assertEqual(progress.computed_units, 1)
        self.assertEqual(progress.skipped_units, 2)
        self.assertEqual(progress.skipped_reason, "checkpoint unavailable")

    def test_revises_discovered_plan_and_accounts_for_retries(self) -> None:
        progress = NotebookTaskProgress("Extract", "sequence")
        progress._publish = Mock()  # type: ignore[method-assign]
        progress.start(1)
        with progress.unit(1, "Validate protocol"):
            pass

        progress.revise_plan(
            4,
            cached_candidate_units=2,
            note="Two sequence caches were discovered after validation.",
        )
        progress.start_unit(2, "sequence-a", candidate_cached=True)
        progress.complete_unit(outcome="reused")
        progress.start_unit(3, "sequence-b")
        progress.retry_unit(
            attempt=1,
            max_attempts=2,
            detail="Transient decoder failure; reopening the video.",
        )
        progress.complete_unit(outcome="failed")
        with progress.unit(4, "Audit cache"):
            pass

        self.assertEqual(progress.completed_units, 4)
        self.assertEqual(progress.reused_units, 1)
        self.assertEqual(progress.failed_units, 1)
        self.assertEqual(progress.retry_events, 1)
        self.assertIn("failed 1", progress.as_html())

    def test_every_notebook_has_documented_terminal_progress(self) -> None:
        notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
        self.assertEqual(len(notebooks), 14)
        for path in notebooks:
            with self.subTest(notebook=path.name):
                payload = json.loads(path.read_text(encoding="utf-8"))
                markdown = "\n".join(
                    "".join(cell.get("source", []))
                    for cell in payload["cells"]
                    if cell.get("cell_type") == "markdown"
                )
                code = "\n".join(
                    "".join(cell.get("source", []))
                    for cell in payload["cells"]
                    if cell.get("cell_type") == "code"
                )
                self.assertIn("How to read the progress display", markdown)
                self.assertIn(
                    "from notebook_progress import NotebookTaskProgress", code
                )
                self.assertIn(".start(", code)
                self.assertIn(".unit(", code)
                self.assertTrue(
                    any(
                        marker in code
                        for marker in (".complete(", ".finish_skipped(", ".block(")
                    )
                )


if __name__ == "__main__":
    unittest.main()
