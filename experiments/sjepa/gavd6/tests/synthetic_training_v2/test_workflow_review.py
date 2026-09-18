"""Independent reviewer checks for provenance and artifact-derived claims.

These tests inspect the workflow/evaluation boundary, not model internals.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import atomic_json, sha256_file
from gavd6_sjepa.research_directions.synthetic_training_v2.data import fixture_bundle
from gavd6_sjepa.research_directions.synthetic_training_v2.evaluation import aggregate_metrics
from gavd6_sjepa.research_directions.synthetic_training_v2.workflow import (
    _evaluate, _receipt, _write_prediction, reconstruct_metrics,
)


class WorkflowReviewerTests(unittest.TestCase):
    def _receipt_fixture(self, root):
        cfg = RunConfig.fixture("receipt-review", root)
        cfg.root.mkdir()
        atomic_json(cfg.root / "identity.json", {"signature": "current-run"})
        (cfg.root / "artifact.txt").write_text("unchanged bytes")
        return cfg, {"stage": "audit", "identity": "current-run", "result": {},
                     "files": {"artifact.txt": sha256_file(cfg.root / "artifact.txt")}}

    def test_foreign_run_receipt_cannot_certify_current_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, receipt = self._receipt_fixture(folder)
            receipt["identity"] = "other-run"
            atomic_json(cfg.root / "receipts/audit.json", receipt)
            with self.assertRaisesRegex(ValueError, "identity|Identity|run"):
                _receipt(cfg, "audit")

    def test_wrong_stage_receipt_cannot_satisfy_prerequisite(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, receipt = self._receipt_fixture(folder)
            receipt["stage"] = "jepa"
            atomic_json(cfg.root / "receipts/audit.json", receipt)
            with self.assertRaisesRegex(ValueError, "stage|Stage|identity|Identity"):
                _receipt(cfg, "audit")

    def test_transitive_prerequisite_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, receipt = self._receipt_fixture(folder)
            atomic_json(cfg.root / "receipts/audit.json", receipt)
            atomic_json(cfg.root / "receipts/data.json", {"stage": "data", "identity": "current-run", "result": {}, "files": {}})
            (cfg.root / "artifact.txt").write_text("changed after data completed")
            with self.assertRaisesRegex(ValueError, "Stale|corrupt|artifact|hash"):
                _receipt(cfg, "data")

    def test_saved_predictions_reconstruct_balanced_metric_and_canonical_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle = fixture_bundle(17).subset("development")
            # A constant 2-pixel x displacement has a known error independently
            # of the fixture's coordinate dynamics and input missingness.
            prediction = bundle.targets["xy"].copy()
            prediction[..., 0] += 2
            _write_prediction(Path(folder) / "predictions", "known-offset", 17, prediction, bundle)
            actual = reconstruct_metrics(folder)
            reference = []
            for i, record in enumerate(bundle.records):
                visible = bundle.targets["visible"][i]
                error = 2 / bundle.targets["eval_scale"][i]
                supported_frames = visible.any(axis=1)
                reference.append(float(error[supported_frames].mean()))
                self.assertEqual(actual.iloc[i].person_id, record["canonical_person_id"])
            np.testing.assert_allclose(actual.visible_nle, reference, rtol=1e-7, atol=1e-9)
            summary = aggregate_metrics(actual)
            for row in summary.itertuples():
                selected = actual.loc[actual.extractor.eq(row.extractor)]
                by_window = selected.groupby(["person_id", "motion_id", "window_id"]).visible_nle.mean()
                by_motion = by_window.groupby(level=["person_id", "motion_id"]).mean()
                expected = by_motion.groupby(level="person_id").mean().mean()
                self.assertAlmostEqual(row.value, expected, places=12)

    def test_favorable_fixture_predictions_cannot_authorize_scientific_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = RunConfig.fixture("fixture-gate-review", folder)
            cfg.root.mkdir()
            bundle = fixture_bundle(17)
            bundle.save(cfg.root / "data/bundle")
            dev = bundle.subset("development")
            for method, shift in (("paired_jepa", 0.1), ("coordinate", 3.), ("direct", 3.),
                                  ("smoothnet", 3.), ("initialized", 3.)):
                prediction = dev.targets["xy"].copy()
                prediction[..., 0] += shift
                _write_prediction(cfg.root / "predictions", method, 17, prediction, dev)
            result = _evaluate(cfg)
            self.assertEqual(result["gates"]["B"]["status"], "insufficient_evidence")
            self.assertEqual(result["gates"]["real_transfer"]["status"], "insufficient_evidence")
            saved = json.loads((cfg.root / "evaluation/gates.json").read_text())
            self.assertEqual(saved, result["gates"])


if __name__ == "__main__":
    unittest.main()
