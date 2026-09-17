import copy
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gavd6_sjepa.research_directions.temporal_gait.config import RunConfig
from gavd6_sjepa.research_directions.temporal_gait.contracts import atomic_json, digest, freeze_run, verify_run, sha256_file
from gavd6_sjepa.research_directions.temporal_gait.fixtures import fixture_bout, dataset
from gavd6_sjepa.research_directions.temporal_gait.information_audit import assert_future_boundary, audit_datasets
from gavd6_sjepa.research_directions.temporal_gait.manifests import inventory, groups, SCHEMA
from gavd6_sjepa.research_directions.temporal_gait.preprocessing import context_input, nearest_endpoint, prefix_geometry, signed_speed_contrast
from gavd6_sjepa.research_directions.temporal_gait.video import load_pose
from gavd6_sjepa.research_directions.temporal_gait.windows import prepare_bout, issue_times, make_window, WindowDataset
from gavd6_sjepa.research_directions.temporal_gait.workflow import plan_tasks, run_stage


def fixture_config(root):
    return RunConfig(mode="synthetic", run_root=str(root), device="cpu", seeds=[42], pilot_seeds=[42], development_seeds=[42], updates=2, checkpoint_updates=[0, 2], hidden_dim=16, encoder_depth=1, predictor_depth=1, direct_updates=2, bootstrap_samples=20, batch_size=2)


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = fixture_config("/tmp/temporal-gait-unit-not-executed")
        self.pose, self.bout = fixture_bout()

    def test_half_open_bin_left_and_original_time_endpoint(self):
        arrays, record = make_window(self.pose, self.cfg, self.bout, 2.56)
        self.assertAlmostEqual(float(arrays["query_times"][-1]), -.04, places=7)
        self.assertAlmostEqual(float(arrays["context_times"][arrays["context_valid"]].max()), -.04, places=7)
        self.assertAlmostEqual(arrays["endpoint_times"][0, 25], 2.81)
        # 2.81 is NOT a 25-Hz grid point; input grid rounding would yield 2.80 or2.84.
        self.assertNotAlmostEqual(arrays["endpoint_times"][0, 25] / .04, round(2.81 / .04))
        self.assertTrue(np.all(arrays["context_times"][arrays["context_valid"]] < 0))

    def test_future_mutation_including_boundary_and_model_prediction(self):
        from gavd6_sjepa.research_directions.temporal_gait.models import JEPA
        import torch
        report = assert_future_boundary(self.pose, self.cfg, 2.56)
        self.assertEqual(report["status"], "passed")
        changed = copy.deepcopy(self.pose)
        changed.coords[changed.times >= 2.56] *= 1000
        changed.observed[changed.times >= 2.56] = False
        keys = ("context", "context_valid", "context_times", "context_age", "query_times")
        def student(p):
            values = context_input(p, self.cfg, 2.56)
            return {k: torch.as_tensor(values[k])[None] for k in keys}
        torch.manual_seed(7)
        model = JEPA(self.cfg).eval()
        with torch.no_grad():
            torch.testing.assert_close(model.predict_future(student(self.pose)), model.predict_future(student(changed)), rtol=0, atol=0)

    def test_full_bout_final_window_and_no_horizon_eligibility_filter(self):
        data, coverage = prepare_bout(self.pose, self.cfg, self.bout)
        self.assertEqual(data.records[-1]["issue_time"], 6.)
        self.assertEqual(coverage["covered_context_seconds"], 6.)
        self.assertFalse(data.arrays["endpoint_valid"][-1].any())
        self.assertTrue(data.arrays["context_valid"][-1].any())
        long = issue_times(0., 600., self.cfg)
        self.assertGreater(len(long), 1000)
        self.assertEqual(len(issue_times(0., 2., self.cfg)), 0)
        self.assertEqual(len(issue_times(0., 2.56, self.cfg)), 1)

    def test_missing_vfr_nonmonotone_and_stale_observation(self):
        pose = copy.deepcopy(self.pose)
        pose.times[1] = pose.times[0]
        with self.assertRaisesRegex(ValueError, "nonmonotone"):
            pose.validate()
        pose = copy.deepcopy(self.pose)
        pose.times[1] = np.nan
        with self.assertRaises(ValueError):
            pose.validate()
        pose = copy.deepcopy(self.pose)
        pose.observed[(pose.times > 2.3) & (pose.times < 2.56), 25] = False
        result = context_input(pose, self.cfg, 2.56)
        self.assertFalse(result["context_valid"][-1, 25])
        pose = copy.deepcopy(self.pose)
        pose.times += np.arange(len(pose.times)) ** 2 * 1e-8
        _, coverage = prepare_bout(pose, self.cfg, self.bout)
        self.assertTrue(coverage["vfr"])

    def test_nearest_observed_frame_tie_and_visibility_not_cherry_picked(self):
        coords, valid, times = nearest_endpoint(self.pose, 2.815, 2.56, .02)
        self.assertAlmostEqual(times[25], 2.81)
        pose = copy.deepcopy(self.pose)
        pose.observed[281, 25] = False
        _, valid, times = nearest_endpoint(pose, 2.81, 2.56, .02)
        self.assertFalse(valid[25])
        self.assertTrue(np.isnan(times[25]))

    def test_scale_is_pooled_chain_median_not_mean_of_side_medians(self):
        coords = np.zeros((10, 33, 2))
        valid = np.ones((10, 33), bool)
        # left ten100px lengths; right eight300px; pooled median100, mean of medians200.
        coords[:, 27, 1] = 100
        coords[:, 28, 1] = 300
        valid[-2:, 28] = False
        _, scale, ok, _ = prefix_geometry(coords, valid, 640, 480)
        self.assertTrue(ok)
        self.assertEqual(scale, 100.)
        valid[-3:, 28] = False
        self.assertFalse(prefix_geometry(coords, valid, 640, 480)[2])

    def test_translation_time_dilation_reflection_reversal_measurement(self):
        pose = self.pose
        base = signed_speed_contrast(pose.coords, pose.observed, pose.times)
        self.assertAlmostEqual(base, signed_speed_contrast(pose.coords + 1234, pose.observed, pose.times), places=12)
        self.assertAlmostEqual(base, signed_speed_contrast(pose.coords, pose.observed, pose.times * 2), places=12)
        self.assertAlmostEqual(base, signed_speed_contrast(pose.coords[::-1], pose.observed[::-1], pose.times), places=12)
        reflected = pose.coords.copy()
        reflected[..., 0] *= -1
        # Coordinate reflection alone preserves speed contrast; semantic side swap changes sign.
        self.assertAlmostEqual(base, signed_speed_contrast(reflected, pose.observed, pose.times), places=12)
        observed = pose.observed.copy()
        for l, r in ((25, 26), (27, 28), (29, 30), (31, 32)):
            reflected[:, [l, r]] = reflected[:, [r, l]]
            observed[:, [l, r]] = observed[:, [r, l]]
        self.assertAlmostEqual(-base, signed_speed_contrast(reflected, observed, pose.times), places=12)


class ManifestAndResumeTests(unittest.TestCase):
    def test_transitive_person_duplicate_grouping(self):
        links = [{"video_id": "a", "participant_id": "p", "provenance": "verified"},
                 {"video_id": "b", "participant_id": "p", "duplicate_id": "d", "provenance": "verified"},
                 {"video_id": "c", "duplicate_id": "d", "provenance": "verified"}]
        g = groups(["a", "b", "c", "d"], links)
        self.assertEqual(g["a"], g["c"])
        self.assertNotEqual(g["a"], g["d"])

    def make_manifests(self, root):
        videos = [{"video_id": v, "video_path": f"/NOT_ACCESSED/{v}.mp4", "sha256": digest(v), "is_full_video": True} for v in ("a", "b", "c")]
        payloads = {"video": {"rows": videos}, "sequence": {"rows": []}, "pose": {"rows": []},
                    "identity": {"verified_links_complete": True, "rows": []},
                    "split": {"rows": [{"video_id": v, "role": role, "reason": "fixed"} for v, role in zip(("a", "b", "c"), ("train", "development", "test"))]},
                    "exposure": {"historical_laterality_93_checked": True, "audit_provenance": "fixture", "rows": [{"video_id": v, "status": "certified_unexposed", "provenance": "fixture", "historical_laterality_member": False} for v in ("a", "b", "c")]},
                    "reservation": {"rows": [{"video_id": v, "protected": False, "reason": "fixture"} for v in ("a", "b", "c")]}}
        config = {}
        for kind, value in payloads.items():
            path = root / (kind + ".json")
            atomic_json(path, {"schema_version": SCHEMA, "kind": kind, **value})
            config[kind + "_manifest"] = str(path)
        return RunConfig(run_root=str(root / "run"), **config), payloads

    def test_inventory_metadata_only_and_no_fresh_exposed_or_linked_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, payloads = self.make_manifests(root)
            with patch("gavd6_sjepa.research_directions.temporal_gait.video.probe_pts", side_effect=AssertionError("media opened")):
                result = inventory(cfg)
            self.assertEqual(result["declared_videos"], 3)
            exposure = {"schema_version": SCHEMA, "kind": "exposure", **payloads["exposure"]}
            exposure["rows"][-1]["status"] = "unknown"
            atomic_json(cfg.exposure_manifest, exposure)
            with self.assertRaisesRegex(ValueError, "untouched"):
                inventory(cfg)
            exposure["rows"][-1]["status"] = "certified_unexposed"
            atomic_json(cfg.exposure_manifest, exposure)
            atomic_json(cfg.identity_manifest, {"schema_version": SCHEMA, "kind": "identity", "verified_links_complete": True,
                        "rows": [{"video_id": v, "duplicate_id": "same", "provenance": "verified"} for v in ("a", "c")]})
            with self.assertRaisesRegex(ValueError, "partitions"):
                inventory(cfg)

    def test_config_fail_closed_unknown_overrides_and_no_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            cfg = fixture_config(temp)
            atomic_json(path, {**cfg.to_dict(), "silent_cap": 8})
            with self.assertRaisesRegex(ValueError, "Unknown"):
                RunConfig.from_env(path)
            atomic_json(path, cfg.to_dict())
            with patch.dict(os.environ, {"TG_MODE": "real"}):
                with self.assertRaisesRegex(ValueError, "Contradictory"):
                    RunConfig.from_env(path)
            with self.assertRaisesRegex(ValueError, "requires"):
                replace(cfg, mode="real").validate()
            with self.assertRaises(ValueError):
                dataset(replace(cfg, mode="real"), "train")

    def test_resume_changed_code_config_manifest_and_grid_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture_config(temp)
            freeze_run(cfg)
            verify_run(cfg)
            with self.assertRaisesRegex(ValueError, "Resume rejected"):
                verify_run(replace(cfg, learning_rate=.123))
            with patch("gavd6_sjepa.research_directions.temporal_gait.contracts.code_fingerprint", return_value={"changed": "code"}):
                with self.assertRaises(ValueError):
                    verify_run(cfg)
            tasks = plan_tasks(cfg, "pilot")
            self.assertEqual(len(tasks), 5)
            self.assertEqual(tasks, plan_tasks(cfg, "confirm"))

    def test_split_and_loader_all_derivatives_inherit(self):
        cfg = fixture_config("/tmp/temporal-gait-no-write")
        train, _ = dataset(cfg, "train")
        dev, _ = dataset(cfg, "development")
        self.assertTrue(audit_datasets(train, dev, cfg)["measurement_gate"])
        dev.records[0]["group_id"] = train.records[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit_datasets(train, dev, cfg)


if __name__ == "__main__":
    unittest.main()
