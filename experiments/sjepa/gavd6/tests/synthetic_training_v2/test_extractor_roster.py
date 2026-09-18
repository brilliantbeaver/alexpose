"""A claimed held extractor must actually be represented in the source roster."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from gavd6_sjepa.research_directions.synthetic_training_v2.preparation import (
    held_family, prepare_source,
)


class ExtractorRosterTests(unittest.TestCase):
    def setUp(self):
        self.specs = [dict(student_id="rtmpose_m", family="rtmpose"),
                      dict(student_id="vitpose_base", family="vitpose")]

    def test_student_id_and_family_resolve_the_same_held_group(self):
        self.assertEqual(held_family(self.specs, "vitpose"), "vitpose")
        self.assertEqual(held_family(self.specs, "vitpose_base"), "vitpose")

    def test_missing_or_typo_held_family_is_rejected(self):
        for held in ("vitpsoe", "", None):
            with self.subTest(held=held), self.assertRaisesRegex(ValueError, "held extractor"):
                held_family(self.specs, held)

    def test_ambiguous_family_or_student_name_is_rejected(self):
        specs = [dict(student_id="vitpose", family="rtmpose"),
                 dict(student_id="vitpose_base", family="vitpose")]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            held_family(specs, "vitpose")

    def test_empty_duplicate_or_incomplete_roster_is_rejected(self):
        for specs in ([], self.specs + [self.specs[0]],
                      [dict(student_id="empty", family=""), *self.specs]):
            with self.subTest(specs=specs), self.assertRaises(ValueError):
                held_family(specs, "vitpose")

    def test_holding_every_family_leaves_no_training_extractor(self):
        with self.assertRaisesRegex(ValueError, "training extractor"):
            held_family(self.specs[1:], "vitpose")

    def test_source_rejects_missing_held_family_before_asset_hashing_or_gpu(self):
        table = pd.DataFrame([dict(canonical_person_id="train-person", split="train"),
                              dict(canonical_person_id="dev-person", split="development")])
        rejected = pd.DataFrame(columns=["exclusion_reason"])
        scope = SimpleNamespace(device="cuda", held_extractor="vitpsoe",
                                require_gpu_scope=lambda: None)
        config = dict(scope_config="scope", reservation_csv="reservation", manifest_dir="registry",
                      amass_root="source", locomotion_audit="audit", estimators=self.specs)
        with tempfile.TemporaryDirectory() as directory, \
             patch("gavd6_sjepa.research_directions.synthetic_training_v2.config.RunConfig.load", return_value=scope), \
             patch("gavd6_sjepa.research_directions.synthetic_training_v2.preparation.audited_motion_windows", return_value=(table, rejected)), \
             patch("gavd6_sjepa.research_directions.synthetic_training_v2.preparation.preparation_provenance", side_effect=AssertionError("Hashed assets")), \
             patch("gavd6_sjepa.research_directions.synthetic_training_v2.runtime.require_haic_runtime", side_effect=AssertionError("GPU check")):
            with self.assertRaisesRegex(ValueError, "held extractor"):
                prepare_source(config, Path(directory) / "output", Path(directory))


if __name__ == "__main__":
    unittest.main()
