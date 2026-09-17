"""Independent model-owner regressions for coordinator-owned data/workflow code.

No model/training implementation is reviewed here. Media paths are deliberately
nonexistent; spies fail if metadata-only or partition guards access contents.
"""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gavd6_sjepa.research_directions.temporal_gait.config import RunConfig
from gavd6_sjepa.research_directions.temporal_gait.contracts import atomic_json
from gavd6_sjepa.research_directions.temporal_gait.fixtures import fixture_bout
from gavd6_sjepa.research_directions.temporal_gait.manifests import SCHEMA, inventory
from gavd6_sjepa.research_directions.temporal_gait.windows import make_window
from gavd6_sjepa.research_directions.temporal_gait.information_audit import measurement_differences
from gavd6_sjepa.research_directions.temporal_gait import workflow


PACKAGE = "gavd6_sjepa.research_directions.temporal_gait"


def cfg_fixture(root):
    return RunConfig(mode="synthetic", device="cpu", run_root=str(root),
                     seeds=[42], pilot_seeds=[42], development_seeds=[42],
                     updates=2, checkpoint_updates=[0, 2], batch_size=2,
                     hidden_dim=16, encoder_depth=1, predictor_depth=1,
                     direct_updates=2, bootstrap_samples=20)


class IndependentReviewRegressionTests(unittest.TestCase):
    def test_teacher_interval_and_primary_endpoint_are_separate(self):
        cfg = cfg_fixture("/tmp/temporal-gait-test-not-written")
        pose, bout = fixture_bout()
        # A valid native clock puts the closest endpoint 3ms AFTER b+h.
        pose.times = pose.times + .003
        pose.validate()
        arrays, _ = make_window(pose, cfg, bout, 3.)
        h = cfg.horizons.index(.5)
        self.assertAlmostEqual(arrays["endpoint_times"][h, 25], 3.503)
        self.assertGreater(arrays["endpoint_times"][h, 25], 3.5)
        actual = arrays["future_times"][h][arrays["future_valid"][h]]
        self.assertTrue(np.all(actual > 3.42))
        self.assertTrue(np.all(actual <= 3.5))
        self.assertAlmostEqual(arrays["future_times"][h, -1, 25], 3.493)
        self.assertFalse(np.array_equal(arrays["future"][h, -1, 25], arrays["endpoint"][h, 25]))

    def test_training_status_translates_to_receipt_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = cfg_fixture(directory)
            with patch(PACKAGE + ".workflow.verify_receipt", return_value={}), \
                 patch(PACKAGE + ".workflow.read_json", return_value={"measurement_gate": True}), \
                 patch(PACKAGE + ".workflow.load_role", return_value="only_train"), \
                 patch(PACKAGE + ".training.train_condition", return_value={"status": "trained"}), \
                 patch(PACKAGE + ".workflow.receipt", side_effect=lambda *a, **kw: kw) as written:
                result = workflow.train_task(cfg, 0, "masked")
                self.assertEqual(result["status"], "complete")
                self.assertEqual(written.call_args.kwargs["roles"], ("train",))
            with patch(PACKAGE + ".workflow.verify_receipt", return_value={}), \
                 patch(PACKAGE + ".workflow.read_json", return_value={"measurement_gate": True}), \
                 patch(PACKAGE + ".workflow.load_role", return_value="only_train"), \
                 patch(PACKAGE + ".training.train_condition", return_value={"status": "interrupted"}), \
                 patch(PACKAGE + ".workflow.receipt", side_effect=lambda *a, **kw: kw):
                self.assertEqual(workflow.train_task(cfg, 0, "masked")["status"], "incomplete")

    def test_same_task_uses_same_lock_across_execution_phases(self):
        names = []

        @contextmanager
        def lock(root, name):
            names.append(name)
            yield

        cfg = cfg_fixture("/tmp/temporal-gait-test-not-written")
        with patch(PACKAGE + ".workflow.stage_lock", side_effect=lock), \
             patch(PACKAGE + ".workflow.verify_run", return_value="identity"), \
             patch(PACKAGE + ".workflow.train_task", return_value={}):
            for phase in ("pilot", "develop", "confirm"):
                workflow.run_stage(cfg, "masked", task_id=0, phase=phase)
        self.assertEqual(len(names), 3)
        self.assertEqual(len(set(names)), 1)

    def test_missing_test_open_marker_blocks_cache_and_prepare_before_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = cfg_fixture(directory)
            with patch(PACKAGE + ".workflow._verify_analysis_lock", return_value={}), \
                 patch(PACKAGE + ".windows.WindowDataset.load", side_effect=AssertionError("cache opened")) as loader, \
                 patch(PACKAGE + ".fixtures.dataset", side_effect=AssertionError("test synthesized")) as fixture:
                with self.assertRaises(ValueError):
                    workflow.load_role(cfg, "test")
                with self.assertRaises(ValueError):
                    workflow._prepare_role(cfg, "test")
                loader.assert_not_called()
                fixture.assert_not_called()

    def test_stale_test_open_marker_blocks_all_test_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = cfg_fixture(directory)
            marker = cfg.root / "test/opened.json"
            correct = dict(analysis_sha256="current-analysis", identity="current-run",
                           mode="synthetic", status="opened_before_access")
            with patch(PACKAGE + ".workflow._verify_analysis_lock", return_value={}), \
                 patch(PACKAGE + ".workflow.verify_run", return_value="current-run"), \
                 patch(PACKAGE + ".workflow.sha256_file", return_value="current-analysis"), \
                 patch(PACKAGE + ".windows.WindowDataset.load", side_effect=AssertionError("cache opened")) as loader, \
                 patch(PACKAGE + ".fixtures.dataset", side_effect=AssertionError("test synthesized")) as fixture:
                for field in correct:
                    atomic_json(marker, {**correct, field: "stale"})
                    with self.subTest(field=field):
                        with self.assertRaisesRegex(ValueError, "Stale"):
                            workflow.load_role(cfg, "test")
                        with self.assertRaisesRegex(ValueError, "Stale"):
                            workflow._prepare_role(cfg, "test")
                loader.assert_not_called()
                fixture.assert_not_called()
                atomic_json(marker, correct)
                self.assertEqual(workflow.verify_test_open(cfg), correct)

    def test_measurement_summary_equal_video_bout_weight_and_no_future_reads(self):
        def row(video, bout, timed, indexed):
            return dict(video_id=video, sequence_id=bout,
                        diagnostic_raw_signed_speed_contrast=0.,
                        diagnostic_timegrid_signed_speed_contrast=timed,
                        diagnostic_indexgrid_signed_speed_contrast=indexed)

        records = [row("a", "long", 1., 2.) for _ in range(20)]
        records.extend([row("a", "short", 3., 2.), row("b", "only", 0., 1.),
                        row("c", "unsupported", None, 0.)])
        # No arrays attribute exists: a future-target read would fail this test.
        summary = measurement_differences(SimpleNamespace(records=records))
        means = summary["video_equal_means"]
        self.assertEqual(summary["matched_windows"], 22)
        self.assertEqual(summary["excluded_windows"], 1)
        self.assertEqual(summary["videos"], 2)
        self.assertEqual(means["timegrid_absolute_difference"], 1.)
        self.assertEqual(means["indexgrid_absolute_difference"], 1.5)
        self.assertEqual(means["time_minus_index_absolute_difference"], -.5)
        self.assertFalse(summary["used_future_targets"])
        self.assertFalse(summary["used_for_training_gate_or_model_selection"])
        self.assertIn("not duration-weighted", summary["definition"])

    def test_same_partition_links_inventory_never_opens_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = [dict(video_id=v, video_path=f"/NEVER_OPEN/{v}.mp4", sha256=v * 64,
                           is_full_video=True) for v in ("a", "b", "c")]
            payloads = {
                "video": dict(rows=videos), "sequence": dict(rows=[]), "pose": dict(rows=[]),
                "identity": dict(verified_links_complete=True, rows=[
                    dict(video_id=v, participant_id="same-person", provenance="verified_fixture")
                    for v in ("a", "b")]),
                "split": dict(rows=[dict(video_id=v, role=role, reason="frozen_fixture")
                                    for v, role in (("a", "train"), ("b", "train"), ("c", "test"))]),
                "exposure": dict(historical_laterality_93_checked=True, audit_provenance="fixture",
                                 rows=[dict(video_id=v, status="certified_unexposed", provenance="fixture",
                                            historical_laterality_member=False) for v in ("a", "b", "c")]),
                "reservation": dict(rows=[dict(video_id=v, protected=False, reason="fixture")
                                          for v in ("a", "b", "c")]),
            }
            paths = {}
            for kind, body in payloads.items():
                path = root / f"{kind}.json"
                atomic_json(path, dict(schema_version=SCHEMA, kind=kind, **body))
                paths[kind + "_manifest"] = str(path)
            cfg = RunConfig(run_root=str(root / "run"), **paths)
            with patch(PACKAGE + ".video.probe_pts", side_effect=AssertionError("video opened")) as video, \
                 patch(PACKAGE + ".video.load_pose", side_effect=AssertionError("pose opened")) as pose, \
                 patch("numpy.load", side_effect=AssertionError("NPZ opened")) as npz, \
                 patch("subprocess.run", side_effect=AssertionError("decoder launched")) as decoder:
                report = inventory(cfg)
                video.assert_not_called()
                pose.assert_not_called()
                npz.assert_not_called()
                decoder.assert_not_called()
            groups = {v["video_id"]: v["group_id"] for v in report["videos"]}
            self.assertEqual(groups["a"], groups["b"])
            self.assertNotEqual(groups["a"], groups["c"])
            self.assertEqual(report["declared_videos"], 3)

    def test_real_config_without_explicit_manifests_fails_before_read(self):
        with patch(PACKAGE + ".manifests.read_json", side_effect=AssertionError("metadata read")) as reader:
            with self.assertRaises(ValueError):
                inventory(RunConfig(run_root="/tmp/temporal-gait-test-not-written"))
            reader.assert_not_called()

    def test_unsupported_cohort_scope_cannot_be_silently_substituted(self):
        cfg = cfg_fixture("/tmp/temporal-gait-test-not-written")
        with self.assertRaisesRegex(ValueError, "cohort scope"):
            replace(cfg, cohort_scope="whatever_is_available").validate()


if __name__ == "__main__":
    unittest.main()
