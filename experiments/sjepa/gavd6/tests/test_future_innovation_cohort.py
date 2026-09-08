import tempfile
import unittest
from pathlib import Path

from gavd6_sjepa.research_directions.future_innovation.fi_cohort import (
    deterministic_window_start,
    select_eligible,
    validate_cohort,
)
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import (
    stage_lock,
    write_once_json,
)


class FutureInnovationCohortTests(unittest.TestCase):
    def test_duplicate_stage_writers_are_rejected_and_lock_is_released(self):
        with tempfile.TemporaryDirectory() as directory:
            with stage_lock(directory, "fold-0"):
                with (
                    self.assertRaisesRegex(ValueError, "Another job"),
                    stage_lock(directory, "fold-0"),
                ):
                    self.fail("duplicate lock acquired")
                with stage_lock(directory, "fold-1"):
                    pass
            with stage_lock(directory, "fold-0"):
                pass

    def candidates(self):
        return [
            {
                "window_id": f"w{i}",
                "sequence_id": f"seq{i}",
                "video_id": f"source{i // 4}",
            }
            for i in range(200)
        ]

    def test_deterministic_selection_source_caps_and_folds(self):
        candidates = self.candidates()
        selected = select_eligible(candidates)
        self.assertTrue(selected.equals(select_eligible(list(reversed(candidates)))))
        validate_cohort(selected)
        self.assertEqual(len(selected), 50)
        self.assertLessEqual(selected.groupby("video_id").size().max(), 2)
        self.assertEqual(selected.groupby("video_id").outer_fold.nunique().max(), 1)

    def test_small_cohort_and_source_overlap_fail(self):
        with self.assertRaisesRegex(ValueError, "exactly 50"):
            select_eligible(self.candidates()[:40])
        cohort = select_eligible(self.candidates())
        pair = cohort[cohort.video_id.duplicated(keep=False)].index[:1]
        self.assertTrue(len(pair))
        cohort.loc[pair, "outer_fold"] = (cohort.loc[pair, "outer_fold"] + 1) % 5
        with self.assertRaisesRegex(ValueError, "multiple folds"):
            validate_cohort(cohort)

    def test_window_start_and_no_short_sequence(self):
        self.assertEqual(deterministic_window_start(10, 73, "a"), 10)
        self.assertIn(deterministic_window_start(10, 100, "a"), range(10, 38))
        with self.assertRaises(ValueError):
            deterministic_window_start(10, 72, "a")

    def test_frozen_contract_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            write_once_json(path, {"seed": 7})
            write_once_json(path, {"seed": 7})
            with self.assertRaises(ValueError):
                write_once_json(path, {"seed": 8})
            with self.assertRaises(ValueError):
                write_once_json(Path(directory) / "nan.json", {"score": float("nan")})
