"""HAIC setup reuses paths and never manufactures source-review decisions."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "stv2_haic_inputs", REPO / "scripts/research_directions/synthetic_training_v2/haic_inputs.py")
INPUTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INPUTS)


class HaicInputsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo, self.work = self.root / "checkout", self.root / "run"
        self.config = INPUTS.preparation_config(self.repo, self.work, {
            "USER": "fixture", "ST_MODEL_ROOT": str(self.root / "models"),
            "ST_BODY_MODEL_ROOT": str(self.root / "bodies"),
        })

    @staticmethod
    def write(path, content="fixture bytes"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def manifests(self):
        rows = [
            ("train", "train-person", "train.npz", 251, True, "male"),
            ("validation", "dev-person", "dev.npz", 251, True, "female"),
            ("test", "test-person", "test.npz", 251, True, "male"),
            ("train", "short-person", "short.npz", 25, True, "male"),
            ("train", "missing-person", "absent.npz", 251, False, "male"),
        ]
        folder = Path(self.config["manifest_dir"])
        folder.mkdir(parents=True)
        inventory, registry, splits = [], [], []
        for split, person, name, frames, exists, gender in rows:
            inventory.append(dict(relative_path=name, subject_id_candidate=person, status="ok",
                                  num_frames=frames, mocap_framerate=25, gender=gender))
            registry.append(dict(subject_id_candidate=person, identity=person,
                                 identity_audit_status="approved", excluded=False))
            splits.append(dict(identity=person, split=split))
            if exists:
                self.write(Path(self.config["amass_root"]) / name)
        for name, records in zip(INPUTS.MANIFEST_FILES, (inventory, registry, splits)):
            pd.DataFrame(records).to_csv(folder / name, index=False)

    def reviewed_files(self, *, training_windows=2, include_development=True):
        people = [("train-person", "train", "train.npz", training_windows)]
        if include_development:
            people.append(("dev-person", "validation", "dev.npz", 1))
        # A protected original test row verifies exclusion without raw array access.
        people.append(("test-person", "test", "test.npz", 1))
        audits, reservations = [], []
        for person, split, name, count in people:
            reservations.append(dict(person_id=person, canonical_person_id=person, original_split=split,
                                     reserved=False, exposure="fixture_review"))
            for index in range(count):
                audits.append(dict(relative_path=name, start_s=index * 2.56, locomotion_status="audited_locomotion",
                                   audit_reviewer="software fixture", audit_evidence="software fixture only",
                                   audit_date="2026-09-18", exposure="fixture_review", canonical_person_id=person))
        Path(self.config["locomotion_audit"]).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(audits).to_csv(self.config["locomotion_audit"], index=False)
        pd.DataFrame(reservations).to_csv(self.config["reservation_csv"], index=False)

    def assets(self):
        for estimator in self.config["estimators"]:
            self.write(estimator["checkpoint"])
            self.write(estimator["config"], "_base_=['base.py']\nvalue=1\n")
            self.write(Path(estimator["config"]).parent / "base.py", "base_value=2\n")
        self.write(self.config["uv_path"])
        self.write(Path(self.config["texture_dir"]) / "body.jpg")
        self.write(Path(self.config["background_dir"]) / "scene.png")
        for gender in ("male", "female"):
            self.write(Path(self.config["body_model_root"]) / "smplh" / gender / "model.npz")
            self.write(Path(self.config["dmpl_root"]) / gender / "model.npz")

    def test_configuration_inherits_all_supplied_paths_and_official_roster(self):
        env = {key: str(self.root / key.lower()) for key in (
            "ST_AMASS_ROOT", "ST_BODY_MODEL_ROOT", "ST_DMPL_ROOT", "ST_UV_PATH",
            "ST_TEXTURE_DIR", "ST_BACKGROUND_DIR", "ST_MODEL_ROOT",
        )}
        before = dict(env)
        config = INPUTS.preparation_config(self.repo, self.work, env)
        for field, variable in (
            ("amass_root", "ST_AMASS_ROOT"), ("body_model_root", "ST_BODY_MODEL_ROOT"),
            ("dmpl_root", "ST_DMPL_ROOT"), ("uv_path", "ST_UV_PATH"),
            ("texture_dir", "ST_TEXTURE_DIR"), ("background_dir", "ST_BACKGROUND_DIR"),
        ):
            self.assertEqual(config[field], env[variable])
        self.assertEqual(env, before)
        self.assertEqual(config["scope_config"], str(self.work / "config/prepare-01.json"))
        self.assertEqual([spec["student_id"] for spec in config["estimators"]], ["rtmpose_m", "hrnet_w32", "vitpose_base"])
        for spec in config["estimators"]:
            self.assertTrue(Path(spec["config"]).is_relative_to(env["ST_MODEL_ROOT"]))
            self.assertTrue(Path(spec["checkpoint"]).is_relative_to(env["ST_MODEL_ROOT"]))
            self.assertNotIn("role", spec)
        inherited = INPUTS.preparation_config(self.repo, self.work, {
            "USER": "example-user", "AMASS_ROOT": str(self.root / "existing-amass"),
            "ST_MODEL_ROOT": str(self.root / "existing-models"),
        })
        self.assertEqual(inherited["amass_root"], str(self.root / "existing-amass/extracted"))
        self.assertEqual(inherited["texture_dir"], str(self.root / "existing-models/synthetic-rendering/smplitex/textures"))
        self.assertEqual(inherited["body_model_root"], "/hai/scratch/example-user/body_models")

    def test_drafts_preserve_splits_and_leave_every_review_decision_blank(self):
        self.manifests()
        with patch("gavd6_sjepa.research_directions.motion_preservation.motion_data.load_motion",
                   side_effect=AssertionError("raw motion opened")):
            draft = INPUTS.create_drafts(self.config, self.work)
        candidates = pd.read_csv(draft / "motion-candidates.csv", keep_default_na=False)
        self.assertEqual(set(candidates.relative_path), {"train.npz", "dev.npz"})
        reservations = pd.read_csv(draft / "person-reservations.draft.csv", keep_default_na=False)
        for name in ("canonical_person_id", "reserved", "exposure"):
            self.assertTrue(reservations[name].eq("").all())
        self.assertEqual(reservations.set_index("person_id").loc["test-person", "original_split"], "test")
        audit = pd.read_csv(draft / "locomotion-audit.draft.csv", keep_default_na=False)
        self.assertEqual(list(audit.columns), list(INPUTS.AUDIT_COLUMNS))
        self.assertTrue(audit.empty)
        self.assertFalse(Path(self.config["locomotion_audit"]).exists())
        self.assertFalse(Path(self.config["reservation_csv"]).exists())

    def test_existing_drafts_return_unchanged_without_reloading_manifests(self):
        self.manifests()
        draft = INPUTS.create_drafts(self.config, self.work)
        review = draft / "locomotion-audit.draft.csv"
        review.write_text(review.read_text() + "reviewer work in progress\n")
        before = {p.name: p.read_bytes() for p in draft.iterdir()}
        with patch("gavd6_sjepa.research_directions.motion_preservation.motion_data.load_amass_manifest",
                   side_effect=AssertionError("existing draft must not be regenerated")):
            self.assertEqual(INPUTS.create_drafts(self.config, self.work), draft)
        self.assertEqual({p.name: p.read_bytes() for p in draft.iterdir()}, before)

    def test_missing_review_files_are_reported_together_with_remedy(self):
        self.manifests()
        self.assets()
        with self.assertRaises(FileNotFoundError) as caught:
            INPUTS.check_inputs(self.config, "vitpose")
        message = str(caught.exception)
        self.assertIn(self.config["locomotion_audit"], message)
        self.assertIn(self.config["reservation_csv"], message)
        self.assertIn("review-drafts", message)

    def test_complete_metadata_passes_without_loading_models_or_motion_arrays(self):
        self.manifests()
        self.assets()
        self.reviewed_files()
        with patch("gavd6_sjepa.research_directions.motion_preservation.motion_data.load_motion",
                   side_effect=AssertionError("raw motion opened")), \
             patch("gavd6_sjepa.research_directions.synthetic_training.estimators.load_estimator",
                   side_effect=AssertionError("model loaded")), \
             patch("gavd6_sjepa.research_directions.synthetic_training_v2.runtime.require_haic_runtime",
                   side_effect=AssertionError("GPU required")):
            result = INPUTS.check_inputs(self.config, "vitpose_base")
        self.assertEqual(result["counts"]["admitted_windows"], 3)
        self.assertEqual(result["counts"]["train_windows"], 2)
        self.assertEqual(result["excluded_windows"], 1)
        self.assertEqual(result["held_extractor_family"], "vitpose")

    def test_cpu_check_rejects_missing_split_or_shuffled_window_support(self):
        self.manifests()
        self.assets()
        self.reviewed_files(training_windows=1)
        with self.assertRaisesRegex(ValueError, "two distinct reviewed windows"):
            INPUTS.check_inputs(self.config, "vitpose")
        self.reviewed_files(include_development=False)
        with self.assertRaisesRegex(ValueError, "Both train and development"):
            INPUTS.check_inputs(self.config, "vitpose")

    def test_cpu_check_rejects_unknown_held_family_and_unsupported_spec_fields(self):
        self.manifests()
        self.assets()
        self.reviewed_files()
        with self.assertRaisesRegex(ValueError, "absent from the roster"):
            INPUTS.check_inputs(self.config, "typo")
        self.config["estimators"][0]["role"] = "train"
        with self.assertRaisesRegex(ValueError, "StudentSpec fields"):
            INPUTS.check_inputs(self.config, "vitpose")

    def test_inherited_config_and_exact_body_paths_are_checked(self):
        self.manifests()
        self.assets()
        self.reviewed_files()
        config = Path(self.config["estimators"][0]["config"])
        config.write_text("_base_=['missing-base.py']\n")
        with self.assertRaisesRegex(FileNotFoundError, "missing-base.py"):
            INPUTS.check_inputs(self.config, "vitpose")
        config.write_text("_base_=['base.py']\n")
        self.config["body_model_root"] = str(Path(self.config["body_model_root"]) / "smplh")
        self.assertEqual(INPUTS.check_inputs(self.config, "vitpose")["status"], "pass")
        missing = Path(self.config["dmpl_root"]) / "female/model.npz"
        missing.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "DMPL female"):
            INPUTS.check_inputs(self.config, "vitpose")

    def test_cpu_check_rejects_gender_unsupported_by_actual_amass_loader(self):
        self.manifests()
        self.assets()
        self.reviewed_files()
        inventory = Path(self.config["manifest_dir"]) / INPUTS.MANIFEST_FILES[0]
        table = pd.read_csv(inventory)
        table.loc[table.relative_path.eq("train.npz"), "gender"] = "neutral"
        table.to_csv(inventory, index=False)
        with self.assertRaisesRegex(ValueError, "source loader requires male or female"):
            INPUTS.check_inputs(self.config, "vitpose")


if __name__ == "__main__":
    unittest.main()
