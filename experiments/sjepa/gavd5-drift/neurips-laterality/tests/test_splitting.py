"""Contract tests for source-level outer and nested read-out splits."""
from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from laterality.splitting import build_source_splits  # noqa: E402


CONDITIONS = (
    "normal",
    "parkinsons",
    "stroke",
    "myopathic",
    "cerebralpalsy",
)
SOURCE_COUNTS = {
    "normal": 7,
    "parkinsons": 6,
    "stroke": 8,
    "myopathic": 7,
    "cerebralpalsy": 6,
}


def uneven_cohort() -> pd.DataFrame:
    """Build sources whose clip counts differ by almost two orders of magnitude."""
    rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(CONDITIONS):
        for source_index in range(SOURCE_COUNTS[condition]):
            video_id = f"{condition}-source-{source_index:02d}"
            if source_index == 0:
                sequence_count = 61 + 7 * condition_index
            else:
                sequence_count = 1 + ((source_index + condition_index) % 4)
            for sequence_index in range(sequence_count):
                rows.append(
                    {
                        "sequence_id": f"{video_id}-sequence-{sequence_index:03d}",
                        "video_id": video_id,
                        "condition": condition,
                    }
                )
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=1776).reset_index(drop=True)


def inner_held_out_sources(inner_fold: dict) -> set[str]:
    """Accommodate the conventional validation/test names for inner hold-outs."""
    for key in ("validation_sources", "val_sources", "test_sources"):
        if key in inner_fold:
            return set(map(str, inner_fold[key]))
    raise AssertionError(
        "inner read-out fold needs validation_sources, val_sources, or test_sources"
    )


class SourceSplitTests(unittest.TestCase):
    def setUp(self):
        self.cohort = uneven_cohort()
        self.source_condition = (
            self.cohort[["video_id", "condition"]]
            .drop_duplicates()
            .set_index("video_id")["condition"]
            .to_dict()
        )
        self.all_sources = set(self.source_condition)
        self.result = build_source_splits(
            self.cohort,
            conditions=CONDITIONS,
            outer_folds=3,
            inner_folds=2,
            seed=20260904,
        )

    def test_outer_folds_are_disjoint_complete_and_source_balanced(self):
        folds = self.result["folds"]
        self.assertEqual(len(folds), 3)
        self.assertEqual({int(fold["fold"]) for fold in folds}, {0, 1, 2})

        test_appearances: Counter[str] = Counter()
        test_sizes: list[int] = []
        per_condition: dict[str, list[int]] = {condition: [] for condition in CONDITIONS}
        for fold in folds:
            train_sources = set(map(str, fold["train_sources"]))
            test_sources = set(map(str, fold["test_sources"]))
            self.assertFalse(train_sources & test_sources)
            self.assertEqual(train_sources | test_sources, self.all_sources)
            self.assertTrue(train_sources)
            self.assertTrue(test_sources)
            test_appearances.update(test_sources)
            test_sizes.append(len(test_sources))
            for condition in CONDITIONS:
                count = sum(
                    self.source_condition[source] == condition for source in test_sources
                )
                self.assertGreater(count, 0)
                per_condition[condition].append(count)

        self.assertEqual(set(test_appearances), self.all_sources)
        self.assertTrue(all(count == 1 for count in test_appearances.values()))
        self.assertLessEqual(max(test_sizes) - min(test_sizes), 1)
        for condition, counts in per_condition.items():
            with self.subTest(condition=condition, counts=counts):
                self.assertLessEqual(max(counts) - min(counts), 1)

    def test_inner_folds_partition_only_the_outer_training_sources(self):
        for outer in self.result["folds"]:
            outer_train = set(map(str, outer["train_sources"]))
            outer_test = set(map(str, outer["test_sources"]))
            inner_folds = outer["inner_readout_folds"]
            self.assertEqual(len(inner_folds), 2)
            held_out_appearances: Counter[str] = Counter()

            for inner in inner_folds:
                inner_train = set(map(str, inner["train_sources"]))
                inner_held_out = inner_held_out_sources(inner)
                self.assertFalse(inner_train & inner_held_out)
                self.assertEqual(inner_train | inner_held_out, outer_train)
                self.assertFalse((inner_train | inner_held_out) & outer_test)
                held_out_appearances.update(inner_held_out)

                for condition in CONDITIONS:
                    self.assertTrue(
                        any(
                            self.source_condition[source] == condition
                            for source in inner_train
                        )
                    )
                    self.assertTrue(
                        any(
                            self.source_condition[source] == condition
                            for source in inner_held_out
                        )
                    )

            self.assertEqual(set(held_out_appearances), outer_train)
            self.assertTrue(
                all(count == 1 for count in held_out_appearances.values())
            )

    def test_sequence_multiplicity_and_row_order_do_not_change_assignment(self):
        one_row_per_source = (
            self.cohort.sort_values("video_id")
            .drop_duplicates("video_id")
            .sample(frac=1.0, random_state=99)
            .reset_index(drop=True)
        )
        collapsed_result = build_source_splits(
            one_row_per_source,
            conditions=CONDITIONS,
            outer_folds=3,
            inner_folds=2,
            seed=20260904,
        )

        actual = {
            int(fold["fold"]): set(map(str, fold["test_sources"]))
            for fold in self.result["folds"]
        }
        collapsed = {
            int(fold["fold"]): set(map(str, fold["test_sources"]))
            for fold in collapsed_result["folds"]
        }
        self.assertEqual(actual, collapsed)

    def test_conflicting_annotations_for_one_source_are_rejected(self):
        bad = self.cohort.copy()
        conflict = bad.iloc[[0]].copy()
        conflict["sequence_id"] = "conflicting-sequence"
        original = str(conflict.iloc[0]["condition"])
        conflict["condition"] = next(c for c in CONDITIONS if c != original)
        bad = pd.concat([bad, conflict], ignore_index=True)

        with self.assertRaises(ValueError):
            build_source_splits(
                bad,
                conditions=CONDITIONS,
                outer_folds=3,
                inner_folds=2,
                seed=20260904,
            )


if __name__ == "__main__":
    unittest.main()
