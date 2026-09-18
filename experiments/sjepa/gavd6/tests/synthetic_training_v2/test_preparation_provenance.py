"""Temporary fake assets exercise source identity without body models or GPUs."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import code_identity
from gavd6_sjepa.research_directions.synthetic_training_v2.preparation import (
    audited_motion_windows, held_family, preparation_provenance, prepare_source,
)

PRODUCERS = (
    "research_directions/temporal_gait/objectives.py",
    "research_directions/temporal_gait/contracts.py",
    "research_directions/synthetic_training/rendering.py",
    "research_directions/synthetic_training/estimators.py",
    "research_directions/synthetic_training/measurements.py",
    "research_directions/motion_preservation/motion_data.py",
    "research_directions/motion_preservation/body_geometry.py",
    "data_foundations/amass_conversion.py",
    "research_directions/synthetic_training_v2/preparation.py",
)


def assets(root):
    def write(name, text="fixture bytes"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return str(path)
    for producer in PRODUCERS:
        write("src/gavd6_sjepa/" + producer, "# fake producer")
    for name in ("amass_raw_inventory_eligible.csv", "amass_subject_registry.csv", "amass_subject_splits.csv"):
        write("registry/" + name)
    write("configs/base.py", "value=1\n")
    config_path = write("configs/model.py", "_base_=['base.py']\nvalue=2\n")
    config = dict(manifest_dir=str(root / "registry"),
                  locomotion_audit=write("audit.csv"), reservation_csv=write("reservation.csv"),
                  scope_config=write("scope.json", "{}"), uv_path=write("uv.npz"),
                  texture_dir=str(root / "textures"), background_dir=str(root / "backgrounds"),
                  body_model_root=str(root / "body"), dmpl_root=str(root / "dmpl"),
                  estimators=[dict(student_id="source", family="pose-family", config=config_path,
                                   checkpoint=write("checkpoint.pth"), head_checkpoint=write("head.pth"))])
    write("textures/texture.png")
    write("backgrounds/scene.jpg")
    write("body/smplh/male/model.npz")
    write("dmpl/male/model.npz")
    table = pd.DataFrame([dict(raw_path=write("motion.npz"))])
    return config, table


def metadata():
    base = pd.DataFrame([dict(relative_path="motion.npz", original_split="validation",
                              available=True, duration_s=10., person_id="known-person")])
    audit = dict(relative_path="motion.npz", start_s=0., locomotion_status="audited_locomotion",
                 audit_reviewer="reviewer", audit_evidence="independent fixture audit",
                 audit_date="2026-09-18", exposure="unexposed_verified",
                 canonical_person_id="known-person", reserved=False)
    reservation = dict(person_id="known-person", canonical_person_id="known-person",
                       original_split="validation", reserved=True, exposure="unexposed_verified")
    return base, audit, reservation


class PreparationProvenanceTests(unittest.TestCase):
    def test_each_producer_asset_and_inherited_config_changes_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, table = assets(root)
            initial = preparation_provenance(config, table, root)
            paths = ["checkpoint.pth", "head.pth", "configs/base.py", "audit.csv", "reservation.csv",
                     "registry/amass_subject_splits.csv", "registry/amass_subject_registry.csv",
                     "registry/amass_raw_inventory_eligible.csv", "uv.npz", "textures/texture.png",
                     "backgrounds/scene.jpg", "body/smplh/male/model.npz", "dmpl/male/model.npz", "motion.npz"]
            for name in paths:
                with self.subTest(asset=name):
                    path = root / name
                    original = path.read_bytes()
                    path.write_bytes(original + b"\n# changed")
                    self.assertNotEqual(initial["identity"], preparation_provenance(config, table, root)["identity"])
                    path.write_bytes(original)
            self.assertEqual(initial["identity"], preparation_provenance(config, table, root)["identity"])
            self.assertEqual(len(initial["assets"]["estimators"]["source"]["configuration_files"]), 2)

    def test_shared_renderer_and_amass_loader_are_producers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets(root)
            initial = code_identity(root)
            for name in PRODUCERS:
                path = root / "src/gavd6_sjepa" / name
                original = path.read_text()
                path.write_text(original + "\n# semantic change")
                self.assertNotEqual(initial, code_identity(root), name)
                path.write_text(original)

    def test_reservation_cannot_be_downgraded_by_audit_and_alias_protection_propagates(self):
        base, audit, reservation = metadata()
        reservation["reserved"] = False
        alias = dict(reservation, person_id="known-alias", reserved=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([audit]).to_csv(root / "audit.csv", index=False)
            pd.DataFrame([reservation, alias]).to_csv(root / "reservation.csv", index=False)
            with patch("gavd6_sjepa.research_directions.motion_preservation.motion_data.load_amass_manifest", return_value=base):
                allowed, rejected = audited_motion_windows("registry", "source", root / "audit.csv", root / "reservation.csv")
            self.assertEqual(len(allowed), 0)
            self.assertEqual(rejected.iloc[0].exclusion_reason, "reserved_identity")

    def test_authority_mismatch_or_missing_coverage_rejected(self):
        base, audit, reservation = metadata()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([audit]).to_csv(root / "audit.csv", index=False)
            for change in (dict(person_id="unrelated"), dict(original_split="train"),
                           dict(canonical_person_id="relabeled"), dict(exposure="development_exposed")):
                with self.subTest(change=change):
                    pd.DataFrame([{**reservation, **change}]).to_csv(root / "reservation.csv", index=False)
                    with patch("gavd6_sjepa.research_directions.motion_preservation.motion_data.load_amass_manifest", return_value=base):
                        with self.assertRaises(ValueError):
                            audited_motion_windows("registry", "source", root / "audit.csv", root / "reservation.csv")

    def test_zero_eligible_exports_shortage_without_runtime_or_motion_loading(self):
        base, audit, reservation = metadata()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([audit]).to_csv(root / "audit.csv", index=False)
            pd.DataFrame([reservation]).to_csv(root / "reservation.csv", index=False)
            config = dict(scope_config="explicit", manifest_dir="registry", amass_root="source",
                          locomotion_audit=str(root / "audit.csv"), reservation_csv=str(root / "reservation.csv"))
            scope = SimpleNamespace(device="cuda", require_gpu_scope=lambda: None)
            with patch("gavd6_sjepa.research_directions.synthetic_training_v2.config.RunConfig.load", return_value=scope), \
                 patch("gavd6_sjepa.research_directions.motion_preservation.motion_data.load_amass_manifest", return_value=base), \
                 patch("gavd6_sjepa.research_directions.synthetic_training_v2.runtime.require_haic_runtime", side_effect=AssertionError("GPU preflight called")), \
                 patch("gavd6_sjepa.research_directions.motion_preservation.motion_data.load_motion", side_effect=AssertionError("Protected motion opened")):
                with self.assertRaisesRegex(ValueError, "No adequate"):
                    prepare_source(config, root / "output", root)
            report = json.loads((root / "output/preparation-status.json").read_text())
            self.assertEqual(report["status"], "insufficient_evidence")
            self.assertEqual(report["counts"]["admitted_windows"], 0)
            self.assertEqual(report["counts"]["excluded_windows"], 1)

    def test_held_student_resolves_entire_family(self):
        specs = [dict(student_id="pose_s", family="pose"), dict(student_id="pose_m", family="pose"),
                 dict(student_id="other", family="other")]
        self.assertEqual(held_family(specs, "pose_s"), "pose")
        self.assertEqual(held_family(specs, "pose"), "pose")


if __name__ == "__main__":
    unittest.main()
