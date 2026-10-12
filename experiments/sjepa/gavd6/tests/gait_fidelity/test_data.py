"""Numerical data contracts, with no downloaded/licensed source dependencies."""
import copy
import json

import numpy as np
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from gavd6_sjepa.research_directions.gait_fidelity.data import (
    SWAP, apply_naming, fixture_bundle, load_dataset, save_dataset, validate_bundle,
)
from gavd6_sjepa.research_directions.gait_fidelity.preparation import (
    geometry_screen, knee_intervention, mirror_body, prepare, merge_datasets, select_intervals,
)
from gavd6_sjepa.research_directions.gait_fidelity.visualization import build_viewer
from gavd6_sjepa.research_directions.motion_preservation.body_geometry import demo_motion
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import atomic_json



class DataContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = fixture_bundle(samples=16)

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tmp_path = Path(self.temporary.name)

    def test_physical_time_and_roundtrip(self):
        tmp_path = self.tmp_path
        bundle = self.bundle
        path = save_dataset(bundle, tmp_path / "bundle")
        loaded = load_dataset(path)
        assert loaded.inputs["xy"].shape[1] == 16
        assert len({r["canonical_person_id"] for r in loaded.records if r["split"] == "development"}) == 2
        for k in bundle.inputs:
            np.testing.assert_array_equal(loaded.inputs[k], bundle.inputs[k])
        raw = bytearray((path / "targets.npz").read_bytes()); raw[-1] ^= 1
        (path / "targets.npz").write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_dataset(path)

    def test_failed_dataset_write_never_publishes_a_partial_bundle(self):
        folder = self.tmp_path / "bundle"
        original = np.savez_compressed
        calls = []
        def failing_write(path, **values):
            calls.append(path)
            if len(calls) == 2:
                raise OSError("Injected target-write failure")
            return original(path, **values)
        with patch("gavd6_sjepa.research_directions.gait_fidelity.data.np.savez_compressed", failing_write):
            with self.assertRaisesRegex(OSError, "Injected"):
                save_dataset(self.bundle, folder)
        self.assertFalse(folder.exists())
        staging = list(self.tmp_path.glob(".bundle-staging-*"))
        self.assertEqual(len(staging), 1)
        self.assertEqual(json.loads((staging[0] / "write-failure.json").read_text())["status"], "unpublished")
        save_dataset(self.bundle, folder)
        self.assertEqual(len(load_dataset(folder).records), len(self.bundle.records))

    def test_naming_moves_confidence_missingness_and_coordinates_together(self):
        x = np.arange(12 * 12 * 2).reshape(12, 12, 2).astype(float)
        confidence = np.arange(144).reshape(12, 12).astype(float)
        observed = np.ones((12, 12), bool); observed[5, 10] = False
        original = dict(xy=x, confidence=confidence, observed=observed, timestamps=np.arange(12) / 25)
        changed = apply_naming(original, "temporary_swap")
        for k in ("xy", "confidence", "observed"):
            np.testing.assert_array_equal(changed[k][4:8], original[k][4:8][:, SWAP])
            np.testing.assert_array_equal(changed[k][:4], original[k][:4])
        np.testing.assert_array_equal(changed["timestamps"], original["timestamps"])
        roundtrip = apply_naming(apply_naming(original, "global_swap"), "global_swap")
        np.testing.assert_array_equal(roundtrip["xy"], x)

    def test_reference_naming_tampering_fails(self):
        bundle = self.bundle
        b = copy.deepcopy(bundle)
        index = next(i for i, r in enumerate(b.records)
                     if r["naming"] == "global_swap" and r["movement_state"] == "knee_flexion")
        b.targets["xy"][index, 0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "Naming interventions"):
            validate_bundle(b)

    def test_nonuniform_clock_and_split_leakage_fail(self):
        bundle = self.bundle
        b = copy.deepcopy(bundle); b.inputs["timestamps"][0, 5] += .005
        with self.assertRaisesRegex(ValueError, "Physical clock"):
            validate_bundle(b)
        b = copy.deepcopy(bundle)
        b.records[0]["split"] = "development"
        with self.assertRaisesRegex(ValueError, "leaks across splits"):
            validate_bundle(b)

    def test_hidden_data_cannot_silently_become_interpolated(self):
        bundle = self.bundle
        b = copy.deepcopy(bundle)
        index = tuple(np.argwhere(~b.inputs["observed"])[0])
        b.inputs["xy"][index] = [0, 0]
        with self.assertRaisesRegex(ValueError, "silent interpolation"):
            validate_bundle(b)

    def test_nochange_endpoint_is_exact_and_unique(self):
        bundle = self.bundle
        groups = {}
        for i, r in enumerate(bundle.records): groups.setdefault(r["pair_id"], []).append(i)
        for indices in groups.values():
            base = [i for i in indices if bundle.records[i]["movement_state"] == "baseline"]
            zero = [i for i in indices if bundle.records[i]["movement_state"] == "no_change"]
            assert len(base) == len(zero) == 1
            np.testing.assert_array_equal(bundle.targets["xy"][base[0]], bundle.targets["xy"][zero[0]])
            np.testing.assert_array_equal(bundle.inputs["xy"][base[0]], bundle.inputs["xy"][zero[0]])

    def test_mirror_is_involution_and_preserves_bone_lengths(self):
        body, _ = demo_motion(frames=16, fps=25)
        origin = np.median(body.joints[:, 0], axis=0)
        normal = body.joints[0, 2] - body.joints[0, 1]
        mirror = mirror_body(body, origin=origin, normal=normal)
        twice = mirror_body(mirror, origin=origin, normal=normal)
        np.testing.assert_allclose(twice.joints, body.joints, atol=1e-6)
        np.testing.assert_allclose(twice.vertices, body.vertices, atol=1e-6)
        np.testing.assert_array_equal(twice.faces, body.faces)
        original_left = np.linalg.norm(body.joints[:, 1] - body.joints[:, 4], axis=1)
        mirror_right = np.linalg.norm(mirror.joints[:, 2] - mirror.joints[:, 5], axis=1)
        np.testing.assert_allclose(original_left, mirror_right, atol=1e-6)

    def test_extended_intervals_retain_two_nonoverlapping_windows(self):
        rows = [dict(relative_path="person/walk.npz", start_s=start, duration_s=30.) for start in (5., 8.)]
        selected, excluded = select_intervals(rows, 128, 25.)
        self.assertEqual(len(selected), 2)
        self.assertEqual(excluded, [])
        self.assertEqual(selected[0]["start_s"], 5.)
        self.assertAlmostEqual(selected[1]["start_s"], 10.12)
        self.assertEqual(selected[1]["parent_audited_start_s"], 8.)
        self.assertGreaterEqual(selected[1]["start_s"], selected[0]["start_s"] + 128 / 25)

    def test_geometry_screen_records_contacts_and_rejects_large_step(self):
        body, _ = demo_motion(frames=64, fps=25)
        unchanged = geometry_screen(body, body, minimum_changed_mm=0.)
        assert unchanged["pass_"]
        bad = copy.deepcopy(body); bad.joints[30, 8, 0] += 1
        result = geometry_screen(body, bad)
        assert not result["pass_"] and "joint_step_discontinuity" in result["reasons"]

    def test_source_mode_never_falls_back_to_fixture(self):
        tmp_path = self.tmp_path
        with self.assertRaises(KeyError):
            prepare(dict(mode="source"), tmp_path / "source")
        assert not (tmp_path / "source" / "bundle").exists()

    def test_fixture_prepare_and_viewer_are_self_contained(self):
        tmp_path = self.tmp_path
        path = prepare(dict(mode="fixture", data=dict(samples=16, fixture_people=3)), tmp_path / "fixture")
        b = load_dataset(path)
        assert b.evidence_status == "fixture-tested"
        viewer = build_viewer(b, tmp_path / "viewer.html", max_tracks=2)
        content = viewer.read_text()
        assert "cdn" not in content.lower()
        assert "NaN" not in content
        assert len(json.loads((tmp_path / "viewer-selection.json").read_text())["indices"]) == 2

    def test_merge_requires_all_shards_and_preserves_roles(self):
        tmp_path = self.tmp_path
        bundle = self.bundle
        paths = []
        # Whole families are the sharding unit; splitting track rows would break pairs.
        families = sorted({r["source_family_id"] for r in bundle.records})
        for shard in range(2):
            b = copy.deepcopy(bundle)
            ix = np.array([i for i, r in enumerate(b.records) if families.index(r["source_family_id"]) % 2 == shard])
            b.inputs = {k: v[ix] for k, v in b.inputs.items()}
            b.targets = {k: v[ix] for k, v in b.targets.items()}
            b.records = [b.records[i] for i in ix]
            b.provenance.update(shard_index=shard, num_shards=2, identity="same-frozen-config",
                                frozen_people=sorted({r["canonical_person_id"] for r in bundle.records}), assets={})
            paths.append(save_dataset(b, tmp_path / f"shard{shard}"))
        with self.assertRaisesRegex(ValueError, "Every declared"):
            merge_datasets(paths[:1], tmp_path / "incomplete")
        merged = merge_datasets(paths, tmp_path / "merged")
        assert len(load_dataset(merged).records) == len(bundle.records)

    def test_confirmation_references_lock_before_array_access(self):
        tmp_path = self.tmp_path
        bundle = self.bundle
        path = save_dataset(bundle, tmp_path / "bundle")
        manifest = json.loads((path / "manifest.json").read_text())
        manifest["records"][0]["split"] = "confirmation"
        atomic_json(path / "manifest.json", manifest)
        (path / "targets.npz").unlink()
        with self.assertRaisesRegex(PermissionError, "Confirmation"):
            load_dataset(path)


if __name__ == "__main__":
    unittest.main()
