"""Fault-injection regressions for the independent Codex implementation review."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from gavd6_sjepa.research_directions.temporal_gait.config import RunConfig
from gavd6_sjepa.research_directions.temporal_gait.contracts import atomic_json, read_json, sha256_file, verify_run
from gavd6_sjepa.research_directions.temporal_gait import contracts, evaluation, workflow
from gavd6_sjepa.research_directions.temporal_gait.windows import WindowDataset

PACKAGE = "gavd6_sjepa.research_directions.temporal_gait"


def config(root):
    return RunConfig(mode="synthetic", device="cpu", run_root=str(root), seeds=[42], pilot_seeds=[42], development_seeds=[42],
                     arms=["masked"], updates=1, checkpoint_updates=[0, 1], batch_size=2, hidden_dim=8, heads=2,
                     encoder_depth=1, predictor_depth=1, direct_updates=1, ridge_alphas=[1.], bootstrap_samples=8)


class WorkflowIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def prepared(self, root, trained=False):
        cfg = config(root)
        for stage in ("inventory", "prepare", "audit"):
            workflow.run_stage(cfg, stage)
        if trained:
            workflow.run_stage(cfg, "masked", task_id=0)
        return cfg

    def locked(self, root, with_test_cache=False):
        """Actual tiny synthetic pipeline, not mocked lock/receipt preconditions."""
        cfg = self.prepared(root, trained=True)
        workflow.run_stage(cfg, "evaluate")
        workflow.run_stage(cfg, "calibrate")
        contracts.verify_receipt(cfg, "00")
        contracts.verify_receipt(cfg, "07-lock")
        if with_test_cache:
            self.write_open_marker(cfg)
            outputs = workflow._prepare_role(cfg, "test")
            contracts.receipt(cfg, "07-data", outputs, roles=("test",))
            # Establish an actually usable cache before corrupting a dependency.
            self.assertGreater(len(workflow.load_role(cfg, "test")), 0)
        return cfg

    def write_open_marker(self, cfg):
        atomic_json(cfg.root / "test/opened.json", {
            "analysis_sha256": sha256_file(cfg.root / "locks/analysis.json"),
            "identity": verify_run(cfg), "mode": cfg.mode,
            "status": "opened_before_access"})

    def test_modified_cache_rejected_before_training_and_development_model_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.prepared(tmp, trained=True)
            cache = cfg.root / "cache/train/windows.npz"
            data = WindowDataset.load(cache)
            data.arrays["context"][0, 0, 0, 0] += 1
            data.save(cache)
            with patch(PACKAGE + ".training.train_condition", side_effect=AssertionError("training reached")), \
                 patch(PACKAGE + ".evaluation.evaluate_condition", side_effect=AssertionError("model reached")):
                with self.assertRaisesRegex(ValueError, "Changed output"):
                    workflow.load_role(cfg, "train")
                with self.assertRaisesRegex(ValueError, "Changed output"):
                    workflow.run_stage(cfg, "evaluate")
            # Removing a final task receipt is not required to prove the training
            # reader boundary: no new optimizer can bypass load_role's receipt.

    def test_protocol_and_retained_snapshot_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.prepared(tmp)
            real_hash = contracts.sha256_file
            def changed(path):
                if str(path).endswith("docs/studies/temporal-gait/protocol/protocol.md"):
                    return "f" * 64
                return real_hash(path)
            with patch(PACKAGE + ".contracts.sha256_file", side_effect=changed), \
                 patch(PACKAGE + ".windows.WindowDataset.load", side_effect=AssertionError("cache opened")):
                with self.assertRaisesRegex(ValueError, "Resume rejected"):
                    workflow.load_role(cfg, "train")
            snapshot = cfg.root / "config/governing-documents/protocol.md"
            contracts.atomic_bytes(snapshot, b"changed protocol interpretation\n")
            with self.assertRaisesRegex(ValueError, "snapshot"):
                verify_run(cfg)

    def test_incomplete_grid_never_becomes_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.prepared(tmp)
            result = workflow.run_stage(cfg, "evaluate")
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["missing_tasks"], [0])
            self.assertFalse((cfg.root / "receipts/06-pilot.json").exists())

    def test_summary_write_failure_and_pre_receipt_failure_recover_without_overwrite(self):
        for failure_point in ("summary", "receipt"):
            with self.subTest(failure_point=failure_point), tempfile.TemporaryDirectory() as tmp:
                cfg = self.prepared(tmp, trained=True)
                original_json, original_receipt = evaluation._json, workflow.receipt
                def fail_json(path, payload):
                    if Path(path).name == "development-selection.json":
                        raise RuntimeError("injected summary interruption")
                    return original_json(path, payload)
                def fail_receipt(*args, **kwargs):
                    if args[1] == "06-pilot":
                        raise RuntimeError("injected before receipt")
                    return original_receipt(*args, **kwargs)
                with patch(PACKAGE + ".evaluation._json", side_effect=fail_json if failure_point == "summary" else original_json), \
                     patch(PACKAGE + ".workflow.receipt", side_effect=fail_receipt if failure_point == "receipt" else original_receipt):
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        workflow.run_stage(cfg, "evaluate")
                original = cfg.root / "evaluation/pilot/summary/paired-source-scores.json"
                saved_hash = sha256_file(original)
                self.assertFalse((cfg.root / "receipts/06-pilot.json").exists())
                result = workflow.run_stage(cfg, "evaluate")
                self.assertEqual(result["status"], "complete")
                self.assertEqual(sha256_file(original), saved_hash)
                self.assertFalse(result["details"]["ready_for_expansion"])

    def test_locked_test_summary_retry_retains_original_lock_and_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.prepared(tmp, trained=True)
            workflow.run_stage(cfg, "evaluate")
            workflow.run_stage(cfg, "calibrate")
            lockhash = sha256_file(cfg.root / "locks/analysis.json")
            original_json = evaluation._json
            def fail(path, payload):
                if Path(path).name == "locked-test-summary.json":
                    raise RuntimeError("injected locked summary interruption")
                return original_json(path, payload)
            with patch(PACKAGE + ".evaluation._json", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    workflow.run_stage(cfg, "test")
            openhash = sha256_file(cfg.root / "test/opened.json")
            preserved = cfg.root / "test/summary/paired-source-scores.json"
            preserved_hash = sha256_file(preserved)
            with patch(PACKAGE + ".evaluation._fit_baselines", side_effect=AssertionError("test refit")):
                result = workflow.run_stage(cfg, "test")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(sha256_file(preserved), preserved_hash)
            self.assertEqual(sha256_file(cfg.root / "locks/analysis.json"), lockhash)
            self.assertEqual(sha256_file(cfg.root / "test/opened.json"), openhash)
            records = read_json(cfg.root / "test/conditions.json")
            self.assertTrue(all(r["fitted_reused"] for r in records))

    def test_historical_overlap_final_rejected_before_test_access(self):
        cfg = replace(config("/tmp/never-written-temporal-gait"), mode="real", cohort_scope="historical_overlap")
        with patch(PACKAGE + ".workflow._prepare_role", side_effect=AssertionError("test accessed")):
            with self.assertRaisesRegex(ValueError, "development-only"):
                workflow.lock_and_calibrate(cfg)
            with self.assertRaisesRegex(ValueError, "may not open"):
                workflow.evaluate_test(cfg)

    def test_mutated_inventory_blocks_public_test_before_open_marker_or_preparation(self):
        for field, value in (("role", "test"),
                             ("video_path", "/SYNTHETIC-NEVER-OPEN/post-freeze-source.mp4")):
            with self.subTest(field=field), tempfile.TemporaryDirectory(prefix="SYNTHETIC-inventory-integrity-") as tmp:
                cfg = self.locked(tmp)
                inventory_path = cfg.root / "reports/inventory.json"
                changed = read_json(inventory_path)
                self.assertEqual(changed["videos"][0]["role"], "train")
                changed["videos"][0][field] = value
                atomic_json(inventory_path, changed)
                # A valid 07-lock does not authenticate the derived inventory.
                contracts.verify_receipt(cfg, "07-lock")
                with self.assertRaisesRegex(ValueError, "Changed output: reports/inventory.json"):
                    contracts.verify_receipt(cfg, "00")
                with patch(PACKAGE + ".workflow._prepare_role", side_effect=AssertionError("test preparation reached")) as prepare:
                    with self.assertRaisesRegex(ValueError, "Changed output: reports/inventory.json"):
                        workflow.run_stage(cfg, "test")
                    prepare.assert_not_called()
                self.assertFalse((cfg.root / "test/opened.json").exists())
                self.assertFalse((cfg.root / "receipts/07-data.json").exists())

    def test_mutated_inventory_blocks_direct_prepare_for_every_role(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC-direct-prepare-integrity-") as tmp:
            cfg = self.locked(tmp)
            self.write_open_marker(cfg)
            inventory_path = cfg.root / "reports/inventory.json"
            changed = read_json(inventory_path)
            changed["videos"][0]["video_path"] = "/SYNTHETIC-NEVER-OPEN/changed-path.mp4"
            atomic_json(inventory_path, changed)
            with patch(PACKAGE + ".fixtures.dataset", side_effect=AssertionError("fixture preparation reached")) as fixture, \
                 patch(PACKAGE + ".video.probe_pts", side_effect=AssertionError("video opened")) as video, \
                 patch(PACKAGE + ".video.load_pose", side_effect=AssertionError("pose opened")) as pose:
                for role in ("train", "development", "test"):
                    with self.subTest(role=role):
                        with self.assertRaisesRegex(ValueError, "Changed output: reports/inventory.json"):
                            workflow._prepare_role(cfg, role)
                fixture.assert_not_called()
                video.assert_not_called()
                pose.assert_not_called()

    def test_changed_analysis_with_matching_open_marker_cannot_bypass_lock_receipt(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC-analysis-lock-integrity-") as tmp:
            cfg = self.locked(tmp, with_test_cache=True)
            analysis_path = cfg.root / "locks/analysis.json"
            changed = read_json(analysis_path)
            changed["selection"]["selected_arm"] = "future"
            atomic_json(analysis_path, changed)
            # Matching these two mutable artifacts is insufficient: the original
            # immutable 07-lock output digest is the authority being exercised.
            self.write_open_marker(cfg)
            contracts.verify_receipt(cfg, "00")
            contracts.verify_receipt(cfg, "07-data")
            with self.assertRaisesRegex(ValueError, "Changed output: locks/analysis.json"):
                contracts.verify_receipt(cfg, "07-lock")
            with patch(PACKAGE + ".windows.WindowDataset.load", side_effect=AssertionError("cache opened")) as cache, \
                 patch(PACKAGE + ".fixtures.dataset", side_effect=AssertionError("fixture prepared")) as fixture, \
                 patch(PACKAGE + ".video.probe_pts", side_effect=AssertionError("video opened")) as video, \
                 patch(PACKAGE + ".video.load_pose", side_effect=AssertionError("pose opened")) as pose:
                readers = {
                    "analysis_helper": lambda: workflow._verify_analysis_lock(cfg),
                    "open_helper": lambda: workflow.verify_test_open(cfg),
                    "cache_reader": lambda: workflow.load_role(cfg, "test"),
                    "prepare_reader": lambda: workflow._prepare_role(cfg, "test"),
                }
                for name, reader in readers.items():
                    with self.subTest(reader=name):
                        with self.assertRaisesRegex(ValueError, "Changed output: locks/analysis.json"):
                            reader()
                cache.assert_not_called()
                fixture.assert_not_called()
                video.assert_not_called()
                pose.assert_not_called()

    def test_missing_lock_receipt_blocks_all_test_helpers_despite_valid_marker_and_cache(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC-missing-lock-receipt-") as tmp:
            cfg = self.locked(tmp, with_test_cache=True)
            original = cfg.root / "receipts/07-lock.json"
            original.rename(cfg.root / "receipts/SYNTHETIC-withheld-07-lock.json")
            contracts.verify_receipt(cfg, "00")
            contracts.verify_receipt(cfg, "07-data")
            with patch(PACKAGE + ".windows.WindowDataset.load", side_effect=AssertionError("cache opened")) as cache, \
                 patch(PACKAGE + ".fixtures.dataset", side_effect=AssertionError("fixture prepared")) as fixture, \
                 patch(PACKAGE + ".video.probe_pts", side_effect=AssertionError("video opened")) as video, \
                 patch(PACKAGE + ".video.load_pose", side_effect=AssertionError("pose opened")) as pose:
                readers = {
                    "analysis_helper": lambda: workflow._verify_analysis_lock(cfg),
                    "open_helper": lambda: workflow.verify_test_open(cfg),
                    "cache_reader": lambda: workflow.load_role(cfg, "test"),
                    "prepare_reader": lambda: workflow._prepare_role(cfg, "test"),
                }
                for name, reader in readers.items():
                    with self.subTest(reader=name):
                        with self.assertRaisesRegex(FileNotFoundError, "07-lock.json"):
                            reader()
                cache.assert_not_called()
                fixture.assert_not_called()
                video.assert_not_called()
                pose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
