"""Tests for behavior-neutral training progress and notebook ETA reporting."""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from laterality.config import canonical_json_digest, load_context  # noqa: E402
from laterality.data import prepare_cohort  # noqa: E402
from laterality.splitting import build_source_splits  # noqa: E402
from laterality.training import (  # noqa: E402
    _approved_implementation_compatibility,
    _state_digest,
    implementation_digest,
    load_checkpoint,
    train_fold,
    train_selected,
)
from notebook_progress import NotebookTrainingProgress  # noqa: E402


class NotebookProgressTests(unittest.TestCase):
    def test_eta_moves_from_historical_timing_to_recent_epoch_timing(self):
        now = [0.0]
        progress = NotebookTrainingProgress(
            refresh_seconds=0.0,
            clock=lambda: now[0],
        )
        progress._publish = Mock()  # type: ignore[method-assign]
        progress(
            {
                "event": "run_started",
                "profile": "paper",
                "device": "cpu",
                "total_jobs": 2,
                "cached_candidate_jobs": 1,
                "new_candidate_jobs": 1,
                "epochs_per_job": 10,
                "updates_per_epoch": 2,
            }
        )
        progress(
            {
                "event": "job_completed",
                "job_index": 1,
                "completed_jobs": 1,
                "checkpoint_reused": True,
                "variant": "vanilla",
                "fold": 0,
                "seed": 42,
                "epochs": 10,
                "historical_training_seconds": 200.0,
                "device": "cpu",
                "reused_jobs": 1,
                "trained_jobs": 0,
            }
        )
        self.assertAlmostEqual(progress._remaining_seconds(), 200.0)
        self.assertIn("50.0%", progress.as_html())

        progress(
            {
                "event": "job_started",
                "job_index": 2,
                "total_jobs": 2,
                "variant": "reflection_augmented",
                "fold": 0,
                "seed": 42,
                "epochs": 10,
                "total_optimizer_updates": 20,
                "device": "cpu",
            }
        )
        for epoch in range(1, 6):
            now[0] += 10.0
            progress(
                {
                    "event": "epoch_completed",
                    "job_index": 2,
                    "total_jobs": 2,
                    "variant": "reflection_augmented",
                    "fold": 0,
                    "seed": 42,
                    "epoch": epoch,
                    "epochs": 10,
                    "optimizer_updates": 2 * epoch,
                    "total_optimizer_updates": 20,
                    "mean_total_loss": 1.0 / epoch,
                    "epoch_seconds": 10.0,
                    "job_elapsed_seconds": 10.0 * epoch,
                    "device": "cpu",
                }
            )

        # Five recent 10-second epochs supersede the 20-second historical rate.
        self.assertAlmostEqual(progress._remaining_seconds(), 50.0)
        rendered = progress.as_html()
        self.assertIn("75.0%", rendered)
        self.assertIn("Epoch <strong>5/10</strong>", rendered)
        self.assertIn("optimizer updates 10/20", rendered)

    def test_reusing_a_display_object_starts_with_clean_state(self):
        progress = NotebookTrainingProgress(refresh_seconds=0.0)
        progress._publish = Mock()  # type: ignore[method-assign]
        start = {
            "event": "run_started",
            "profile": "smoke",
            "device": "cpu",
            "total_jobs": 1,
            "cached_candidate_jobs": 0,
            "new_candidate_jobs": 1,
            "epochs_per_job": 1,
            "updates_per_epoch": 1,
        }
        progress(start)
        progress(
            {
                "event": "job_completed",
                "job_index": 1,
                "completed_jobs": 1,
                "checkpoint_reused": False,
                "epochs": 1,
                "historical_training_seconds": 2.0,
                "device": "cpu",
                "reused_jobs": 0,
                "trained_jobs": 1,
            }
        )
        progress(start)
        self.assertEqual(progress._progress_fraction(), 0.0)
        self.assertEqual(progress.completed_jobs, 0)
        self.assertEqual(progress.historical_epoch_seconds, [])


class TrainingEventTests(unittest.TestCase):
    @staticmethod
    def _context(directory: str):
        return SimpleNamespace(
            variants=("vanilla",),
            folds=(0,),
            seeds=(7,),
            artifact_root=Path(directory),
            profile="smoke",
            profile_config={"epochs": 2, "batch_size": 10},
        )

    @staticmethod
    def _checkpoint() -> dict:
        return {
            "epochs": 2,
            "optimizer_updates": 2,
            "history": [
                {"epoch": 1, "mean_total_loss": 1.0},
                {"epoch": 2, "mean_total_loss": 0.5},
            ],
            "source_draw_counts": {"source-a": 2},
            "wall_seconds": 4.0,
            "device": "cpu",
        }

    def test_selected_training_emits_hierarchical_events(self):
        events: list[dict] = []

        def fake_train_fold(
            context,
            cohort,
            splits,
            fold,
            seed,
            variant,
            *,
            progress_callback,
        ):
            progress_callback(
                {
                    "event": "job_started",
                    "variant": variant,
                    "fold": fold,
                    "seed": seed,
                    "epochs": 2,
                    "total_optimizer_updates": 2,
                    "device": "cpu",
                }
            )
            for epoch in (1, 2):
                progress_callback(
                    {
                        "event": "epoch_completed",
                        "variant": variant,
                        "fold": fold,
                        "seed": seed,
                        "epoch": epoch,
                        "epochs": 2,
                        "optimizer_updates": epoch,
                        "total_optimizer_updates": 2,
                        "mean_total_loss": 1.0 / epoch,
                        "epoch_seconds": 2.0,
                        "job_elapsed_seconds": 2.0 * epoch,
                        "device": "cpu",
                    }
                )
            progress_callback(
                {
                    "event": "checkpoint_saving",
                    "variant": variant,
                    "fold": fold,
                    "seed": seed,
                    "epoch": 2,
                    "epochs": 2,
                    "device": "cpu",
                }
            )
            return self._checkpoint()

        with tempfile.TemporaryDirectory() as directory:
            context = self._context(directory)
            with (
                patch("laterality.training.train_fold", side_effect=fake_train_fold),
                patch("laterality.training.resolve_device", return_value=torch.device("cpu")),
            ):
                summaries = train_selected(
                    context,
                    object(),
                    {"folds": [{"train_sources": ["source-a"]}]},
                    progress_callback=events.append,
                )

        self.assertEqual(
            [event["event"] for event in events],
            [
                "run_started",
                "job_started",
                "epoch_completed",
                "epoch_completed",
                "checkpoint_saving",
                "job_completed",
                "run_completed",
            ],
        )
        self.assertEqual(events[1]["job_index"], 1)
        self.assertEqual(events[2]["total_jobs"], 1)
        self.assertFalse(summaries[0]["checkpoint_reused"])

    def test_a_broken_display_does_not_abort_training(self):
        def broken_callback(event):
            raise RuntimeError("front end disconnected")

        with tempfile.TemporaryDirectory() as directory:
            context = self._context(directory)
            with (
                patch(
                    "laterality.training.train_fold",
                    return_value=self._checkpoint(),
                ),
                patch("laterality.training.resolve_device", return_value=torch.device("cpu")),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                summaries = train_selected(
                    context,
                    object(),
                    {"folds": [{"train_sources": ["source-a"]}]},
                    progress_callback=broken_callback,
                )

        self.assertEqual(len(summaries), 1)
        self.assertEqual(len(caught), 1)
        self.assertIn("progress display failed", str(caught[0].message))

    def test_progress_callback_cannot_perturb_training_randomness(self):
        def consume_randomness(event):
            if event["event"] == "epoch_completed":
                random.random()
                np.random.random(100)
                torch.rand(100)

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                "os.environ",
                {
                    "LATERALITY_ARTIFACT_ROOT": directory,
                    "LATERALITY_DEVICE": "cpu",
                },
                clear=False,
            ):
                context = load_context(profile="smoke")
                # Two epochs are needed so a callback after epoch one could, in the
                # absence of RNG isolation, change the stochastic views in epoch two.
                context.protocol["training"]["smoke"]["epochs"] = 2
                cohort = prepare_cohort(context)
                split_config = context.protocol["splits"]
                splits = build_source_splits(
                    cohort.table,
                    context.protocol["data"]["conditions"],
                    int(split_config["outer_folds"]),
                    int(split_config["inner_folds"]),
                    int(split_config["seed"]),
                )
                splits["split_digest"] = canonical_json_digest(splits)
                plain = train_fold(
                    context,
                    cohort,
                    splits,
                    0,
                    7,
                    "vanilla",
                    reuse_valid=False,
                )
                reported = train_fold(
                    context,
                    cohort,
                    splits,
                    0,
                    7,
                    "vanilla",
                    reuse_valid=False,
                    progress_callback=consume_randomness,
                )

        self.assertEqual(plain["initial_state_digest"], reported["initial_state_digest"])
        self.assertEqual(plain["model_state_digest"], reported["model_state_digest"])
        self.assertEqual(plain["history"], reported["history"])


class CheckpointCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _checkpoint(expected: dict) -> dict:
        model_state = {"weight": torch.tensor([1.0])}
        return {
            **expected,
            "lineage_digest": canonical_json_digest(expected),
            "objective_inputs": ["coordinates", "validity"],
            "sampling": "source_uniform_then_sequence_uniform",
            "rng_streams": ["sampling", "mask", "reflection"],
            "initial_target_state": {},
            "initial_state_digest": _state_digest({}),
            "model_state": model_state,
            "model_state_digest": _state_digest(model_state),
        }

    def test_only_reviewed_progress_only_digest_is_accepted(self):
        current_digest = implementation_digest()
        expected = {
            "schema": "test",
            "implementation_digest": current_digest,
            "train_sources": ["a"],
            "forbidden_test_sources": ["b"],
        }
        legacy_digests = {
            "5fd4492da01dbf01bca7e1f79f3f4c3c0009287352f0b53b20299715111ee6b7",
            "8db585368a424942c78f852f17b82436c8745eb398dcf6c2bd06defb8b340a45",
            "4e6108a04e486e587e807d3660763fd254b46cf446bf5bbba9eec899eb062a04",
        }
        approved = _approved_implementation_compatibility()
        self.assertEqual(
            approved,
            {(legacy_digest, current_digest) for legacy_digest in legacy_digests},
        )
        legacy_digest = min(legacy_digests)
        legacy_expected = {**expected, "implementation_digest": legacy_digest}
        legacy_checkpoint = self._checkpoint(legacy_expected)
        unknown_expected = {**expected, "implementation_digest": "unknown-change"}
        unknown_checkpoint = self._checkpoint(unknown_expected)

        with patch("laterality.training._expected_lineage", return_value=expected):
            with patch(
                "laterality.training.torch.load", return_value=legacy_checkpoint
            ):
                loaded = load_checkpoint(
                    Path("unused.pt"), None, None, {}, 0, 0, "vanilla"
                )
            self.assertIs(loaded, legacy_checkpoint)

            with patch(
                "laterality.training.torch.load", return_value=unknown_checkpoint
            ):
                with self.assertRaisesRegex(RuntimeError, "lineage mismatch"):
                    load_checkpoint(
                        Path("unused.pt"), None, None, {}, 0, 0, "vanilla"
                    )


if __name__ == "__main__":
    unittest.main()
