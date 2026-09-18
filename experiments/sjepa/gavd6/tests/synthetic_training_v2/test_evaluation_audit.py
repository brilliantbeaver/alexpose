"""Challenge arithmetic, support, sampling and evidence-level boundaries."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.synthetic_training_v2.audit import reconstruct_pilot
from gavd6_sjepa.research_directions.synthetic_training_v2.evaluation import (
    aggregate_metrics, evaluate_predictions, paired_cluster_bootstrap, scientific_gate,
)

ROOT = Path(__file__).resolve().parents[2]


def fixture(batch=2, length=64):
    time = np.broadcast_to(np.arange(length) / 25, (batch, length)).copy()
    xy = np.zeros((batch, length, 12, 2))
    xy[:, :, 10, 0] = np.sin(2 * np.pi * time)
    mask = np.ones(xy.shape[:-1], bool)
    target = dict(xy=xy, valid=mask.copy(), visible=mask.copy(),
                  eval_scale=np.ones((batch, length)) * 100)
    rows = [dict(person_id=f"p{i}", motion_id="m0", window_id="w0",
                 variant="clean", extractor="e0", split="development", seed=1,
                 target_kind="synthetic_proxy") for i in range(batch)]
    return xy.copy(), target, time, rows


class EvaluationTests(unittest.TestCase):
    def test_missing_prediction_penalty_and_hidden_reference_separation(self):
        pred, target, time, rows = fixture()
        pred[0, 0, 0] = np.nan
        target["visible"][0, :, 1] = False
        result = evaluate_predictions(pred, target, time, rows)
        self.assertEqual(result.loc[0, "missing_count"], 1)
        self.assertAlmostEqual(result.loc[0, "visible_nle"], 1 / (64 * 11))
        self.assertEqual(result.loc[0, "synthetic_occluded_count"], 64)
        rows[0]["target_kind"] = "real_visible_annotation"
        result = evaluate_predictions(pred, target, time, rows)
        self.assertTrue(np.isnan(result.loc[0, "synthetic_occluded_nle"]))
        self.assertEqual(result.loc[0, "synthetic_occluded_count"], 0)

    def test_target_invalid_and_zero_support_do_not_become_zero_error(self):
        pred, target, time, rows = fixture()
        target["valid"][0] = False
        target["visible"][0] = False
        target["xy"][0] = np.nan
        result = evaluate_predictions(pred, target, time, rows)
        self.assertTrue(np.isnan(result.loc[0, "visible_nle"]))
        self.assertEqual(result.loc[0, "visible_count"], 0)
        summary = aggregate_metrics(result)
        self.assertTrue(np.isnan(summary.iloc[0].value))
        self.assertEqual(summary.iloc[0].status, "insufficient_evidence")

    def test_physical_displacement_and_attenuation(self):
        pred, target, time, rows = fixture()
        pred[:, :, 10, 0] *= 0.5
        result = evaluate_predictions(pred, target, time, rows)
        self.assertTrue((result.displacement_nle > 0).all())
        np.testing.assert_allclose(result.amplitude_ratio, 0.5)
        np.testing.assert_allclose(result.event_timing_mae_s, 0)
        self.assertTrue((result.event_count >= 2).all())
        # Arbitrary-duration stretching cannot masquerade as 0.2s support.
        result = evaluate_predictions(pred, target, time * 1.11, rows)
        self.assertEqual(result.loc[0, "displacement_count"], 0)
        self.assertTrue(np.isnan(result.loc[0, "displacement_nle"]))

    def test_shape_mask_time_and_scale_fail_closed(self):
        pred, target, time, rows = fixture()
        for bad in (np.zeros_like(time), np.full_like(time, np.nan)):
            with self.assertRaises(ValueError):
                evaluate_predictions(pred, {**target, "eval_scale": bad}, time, rows)
        with self.assertRaises(ValueError):
            evaluate_predictions(pred, target, time[:, ::-1], rows)
        with self.assertRaises(ValueError):
            evaluate_predictions(pred, {**target, "valid": target["valid"].astype(float)}, time, rows)
        with self.assertRaises(ValueError):
            evaluate_predictions(pred, target, time, [rows[0], rows[0]])

    def test_nested_balance_does_not_count_variants_as_people(self):
        rows = []
        for person, error, n in (("p0", 0.0, 10), ("p1", 1.0, 1)):
            for variant in range(n):
                rows.append(dict(person_id=person, motion_id="m", window_id="w",
                                 variant=str(variant), extractor="e", split="development",
                                 seed=1, method="a", evidence_status="source-run", visible_nle=error))
        summary = aggregate_metrics(pd.DataFrame(rows))
        self.assertEqual(summary.iloc[0].value, 0.5)
        self.assertEqual(summary.iloc[0].people, 2)

    def test_bootstrap_exact_pairs_and_seed_separation(self):
        pred, target, time, rows = fixture(4)
        a = evaluate_predictions(pred + 1, target, time, rows, method="a")
        b = evaluate_predictions(pred + 2, target, time, rows, method="b")
        frame = pd.concat([a, b], ignore_index=True)
        result = paired_cluster_bootstrap(frame, "a", "b", draws=50)
        self.assertEqual(result["groups"], 4)
        self.assertEqual(len(result["matched_keys"]), 4)
        self.assertAlmostEqual(result["relative_improvement"], 0.5)
        self.assertTrue(all(x["improvement"] > 0 for x in result["draws"]))
        # Row order does not change pairing or RNG behavior.
        self.assertEqual(result, paired_cluster_bootstrap(frame.iloc[::-1], "a", "b", draws=50))
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap(frame.iloc[:-1], "a", "b")
        bad = frame.copy()
        bad.loc[0, "seed"] = 2
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap(bad, "a", "b")

    def test_fixture_and_missing_preservation_cannot_pass(self):
        args = dict(relative_improvement=0.1, ci95=[0.05, 0.15],
                    preservation_ok=True, clean_retention_ok=True)
        self.assertEqual(scientific_gate(evidence_status="fixture-tested", **args)["status"], "insufficient_evidence")
        self.assertEqual(scientific_gate(evidence_status="source-run", **args)["status"], "pass")
        self.assertEqual(scientific_gate(evidence_status="source-run", **{**args, "preservation_ok": False})["status"], "fail")
        self.assertEqual(scientific_gate(evidence_status="source-run", **{**args, "preservation_ok": None})["status"], "insufficient_evidence")


class PilotAuditTests(unittest.TestCase):
    def test_reconstruct_historical_arithmetic_and_rounding_tie(self):
        report = reconstruct_pilot(ROOT)
        self.assertEqual(report["counts"]["source_rows"], 2304)
        self.assertEqual(report["counts"]["selector_decisions"], 6912)
        self.assertEqual(report["counts"]["selector_configurations"], 144)
        self.assertEqual(report["full_vs_replay"], dict(improved=37, harmed=10, tied=1, replay_selected=0))
        table = {row["method"]: row["error"] for row in report["table"]}
        self.assertAlmostEqual(table["full"], 0.02702647999854117, places=14)
        self.assertAlmostEqual(table["domain"], 0.02678073984210087, places=14)
        self.assertAlmostEqual(report["oracle"]["additional_relative_reduction_percent"], 0.04057177311832, places=10)

    def test_duplicate_and_altered_arithmetic_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            relative = Path("notebook_runs/synthetic-training")
            for sub in ("run-02-v1/source", "run-03-v1/selectors"):
                shutil.copytree(ROOT / relative / sub, repo / relative / sub)
            path = repo / relative / "run-02-v1/source/hrnet_w32/outcomes.csv"
            original = path.read_text()
            path.write_text(original + original.splitlines()[1] + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                reconstruct_pilot(repo)
            path.write_text(original)
            path = repo / relative / "run-03-v1/selectors/validation_summary.csv"
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["error"] = str(float(rows[0]["error"]) + 0.01)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "arithmetic"):
                reconstruct_pilot(repo)

    def test_export_fresh_identity_and_historical_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            report = reconstruct_pilot(ROOT, folder)
            self.assertEqual(json.loads((Path(folder) / "pilot-audit.json").read_text()), report)
            with self.assertRaises(FileExistsError):
                reconstruct_pilot(ROOT, folder)
        with self.assertRaises(ValueError):
            reconstruct_pilot(ROOT, ROOT / "notebook_runs/synthetic-training/new-audit")


if __name__ == "__main__":
    unittest.main()
