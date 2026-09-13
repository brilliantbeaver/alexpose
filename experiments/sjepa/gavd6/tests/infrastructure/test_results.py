"""infrastructure / test results."""


import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from gavd6_sjepa.shared_infrastructure.result_catalog import (
    SCHEMA,
    atomic_json,
    describe,
    digest,
    identity,
    new_path,
    refresh,
    registry,
    resolve,
    select,
    snapshot,
)
from gavd6_sjepa.shared_infrastructure.result_migration import (
    integrity,
    migrate,
    migration_plan,
    rollback,
    verify,
)

# Exercise preservation, rollback, path safety and scientific-status boundaries.


def fixture(project):
    parent, child = project / "outputs/old-parent", project / "outputs/old-child"
    atomic_json(parent / "config/run-contract.json", {"protocol": "fixture", "run_id": "parent"})
    (parent / ".DS_Store").write_bytes(b"historical inventory item")
    atomic_json(child / "config/parent-lineage.json", dict(parent_root=str(parent), parent_run_id="parent",
                parent_snapshot=snapshot(parent), identities={"config/run-contract.json": digest(parent / "config/run-contract.json")}))
    atomic_json(child / "config/run-contract.json", {"protocol": "fixture", "config_sha256": {
        "parent-lineage.json": digest(child / "config/parent-lineage.json")}})
    return dict(schema=SCHEMA, entries=[dict(id=name, study="future-innovation", legacy_path=f"outputs/old-{name}",
        canonical_path=f"outputs/studies/future-innovation/{name}", kind="fixture", description="Synthetic path fixture",
        parents=[] if name == "parent" else ["parent"], migrate=True) for name in ("parent", "child")])

class OrganizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name).resolve()
        self.definitions = fixture(self.project)
        self.readers = patch("gavd6_sjepa.shared_infrastructure.result_migration.legacy_readers", return_value={})
        self.readers.start(); self.addCleanup(self.readers.stop)

    def test_catalog_does_not_write_inside_bundles_or_conflate_stop_with_failure(self):
        root = self.project / "outputs/old-child"
        atomic_json(root / "reports/gate-decision.json", {"measurement_complete": True, "decision": "STOP"})
        before = snapshot(root)
        value = refresh(self.project, self.definitions)
        row = next(r for r in value["entries"] if r["id"] == "child")
        self.assertEqual(row["execution_state"], "measurement_complete")
        self.assertEqual(row["scientific_decision"], "STOP")
        self.assertEqual(row["verification"]["status"], "not_checked")
        self.assertEqual(snapshot(root), before)
        self.assertEqual(value["unregistered_roots"], [])

    def test_inspection_pass_does_not_make_a_frozen_study_complete(self):
        root = self.project / "outputs/old-child"
        atomic_json(root / "config/study.json", {"protocol": "source-learning-curve-v1"})
        atomic_json(root / "notebook_runs/batch/execution.json", {"status": "passed"})
        row = describe(self.project, self.definitions["entries"][1])
        self.assertEqual(row["execution_state"], "frozen_awaiting_processing")
        self.assertEqual(row["scientific_decision"], "not_evaluated")

    def test_verified_copy_aliases_backups_and_unchanged_contracts(self):
        saved = {e["id"]: snapshot(self.project / e["legacy_path"]) for e in self.definitions["entries"]}
        original = (self.project / "outputs/old-child/config/parent-lineage.json").read_bytes()
        self.assertEqual(migrate(self.project, self.definitions)["status"], "plan_only")
        self.assertFalse((self.project / "outputs/studies").exists())
        result = migrate(self.project, self.definitions, apply=True)
        receipt = json.loads(Path(result["receipt"]).read_text())
        for op in receipt["operations"]:
            self.assertTrue(Path(op["source"]).is_symlink())
            self.assertEqual(snapshot(op["destination"]), saved[op["id"]])
            self.assertEqual(snapshot(op["backup"]), saved[op["id"]])
            self.assertEqual(resolve(self.project, op["id"], self.definitions), Path(op["destination"]))
        child = resolve(self.project, "child", self.definitions)
        self.assertEqual((child / "config/parent-lineage.json").read_bytes(), original)
        deps = integrity(child)["dependencies"]
        self.assertEqual(deps[0]["runtime_location"], str(resolve(self.project, "parent", self.definitions)))
        self.assertEqual(migrate(self.project, self.definitions, apply=True)["status"], "already_organized")

    def test_copy_corruption_rolls_back_without_losing_original_or_partial_copy(self):
        old = self.project / "outputs/old-parent"; before = snapshot(old)
        copy = shutil.copytree
        def damaged(source, target, *args, **kwargs):
            result = copy(source, target, *args, **kwargs)
            if Path(source).name == "old-parent":
                (Path(target) / ".DS_Store").write_bytes(b"corrupted copy")
            return result
        with patch("gavd6_sjepa.shared_infrastructure.result_migration.shutil.copytree", side_effect=damaged):
            with self.assertRaisesRegex(ValueError, "Copied bundle"):
                migrate(self.project, self.definitions, apply=True)
        self.assertEqual(snapshot(old), before)
        self.assertFalse(old.is_symlink())
        receipts = list((self.project / "outputs/.organization/relocations").glob("*/relocation.json"))
        record = json.loads(receipts[0].read_text())
        self.assertEqual(record["status"], "rolled_back")
        self.assertTrue(Path(record["operations"][0]["retained_copy"]).exists())

    def test_reader_failure_after_switch_restores_all_originals(self):
        before = snapshot(self.project / "outputs/old-parent")
        with patch("gavd6_sjepa.shared_infrastructure.result_migration.legacy_readers",
                   side_effect=[{}, ValueError("relocated reader failed")]):
            with self.assertRaisesRegex(ValueError, "relocated reader"):
                migrate(self.project, self.definitions, apply=True)
        self.assertEqual(snapshot(self.project / "outputs/old-parent"), before)
        for name in ("parent", "child"):
            self.assertFalse((self.project / f"outputs/old-{name}").is_symlink())
            self.assertFalse((self.project / f"outputs/studies/future-innovation/{name}").exists())

    def test_explicit_rollback_restores_paths_and_catalog(self):
        result = migrate(self.project, self.definitions, apply=True)
        record = rollback(self.project, Path(result["receipt"]), self.definitions)
        self.assertEqual(record["status"], "rolled_back")
        self.assertEqual(resolve(self.project, "parent", self.definitions), self.project / "outputs/old-parent")
        row = json.loads((self.project / "outputs/catalog.json").read_text())["entries"][0]
        self.assertEqual(row["location"], str(self.project / "outputs/old-parent"))

    def test_changed_parent_inventory_or_mtime_fails(self):
        parent = self.project / "outputs/old-parent"; child = self.project / "outputs/old-child"
        (parent / "new-readme.md").write_text("unrecorded addition")
        with self.assertRaisesRegex(ValueError, "Dependency"):
            integrity(child)
        (parent / "new-readme.md").unlink()
        (parent / ".DS_Store").touch()
        with self.assertRaisesRegex(ValueError, "Dependency"):
            integrity(child)

    def test_nested_symlink_and_destination_collision_are_rejected(self):
        parent = self.project / "outputs/old-parent"
        (parent / "escape").symlink_to(self.project)
        with self.assertRaisesRegex(ValueError, "Nested symlink"):
            snapshot(parent)
        (parent / "escape").unlink()
        (self.project / "outputs/studies/future-innovation/parent").mkdir(parents=True)
        with self.assertRaises(FileExistsError): migration_plan(self.project, self.definitions)

    def test_stale_verification_and_missing_required_artifact_are_visible(self):
        verify(self.project, self.definitions)
        child = self.project / "outputs/old-child"
        (child / "extra.txt").write_text("after verification")
        row = refresh(self.project, self.definitions)["entries"][1]
        self.assertEqual(row["verification"]["status"], "stale")
        atomic_json(child / "reports/complete.json", {"artifacts": {"models/missing.joblib": "a" * 64}})
        with self.assertRaisesRegex(ValueError, "checksum"):
            integrity(child)

    def test_missing_historical_raw_evidence_is_disclosed_not_numerically_verified(self):
        root = self.project / "outputs/old-parent"
        atomic_json(root / "config/candidates-contract.json", {"artifacts": {"boxes/missing.npz": "a" * 64}})
        result = integrity(root)
        self.assertEqual(len(result["unavailable_source_evidence"]), 1)
        self.assertFalse(result["numerical_reconstruction"])

    def test_identifier_and_receipt_traversal_rejected(self):
        for name in ("../elsewhere", "a/b", "x\nboom", ".."):
            with self.assertRaises(ValueError): new_path(self.project, "future-innovation", name)
        root = self.project / "outputs/old-parent"
        atomic_json(root / "reports/complete.json", {"artifacts": {"../../outside": "a" * 64}})
        with self.assertRaisesRegex(ValueError, "relative path"):
            integrity(root)

    def test_id_selection_rejects_stale_explicit_root_and_does_not_create_new_run(self):
        selected = select(self.project, 'parent', self.definitions)
        self.assertEqual(selected, self.project/'outputs/old-parent')
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            select(self.project, 'parent', self.definitions, supplied_root='outputs/old-child')
        new = select(self.project, 'new-available-dev-20260912', self.definitions, new_study='future-innovation')
        self.assertEqual(new, self.project/'outputs/studies/future-innovation/new-available-dev-20260912')
        self.assertFalse(new.exists())

    def test_alternate_haic_parent_layout_does_not_migrate_the_container(self):
        definition = dict(schema=SCHEMA, entries=[dict(id='gate-v2',study='future-innovation',kind='feasibility_gate',
            legacy_path='outputs/future-innovation', canonical_path='outputs/studies/future-innovation/gate-v2',
            alternate_paths=['outputs/future-innovation/gate-v2'],identity_file='config/run-contract.json',migrate=True)])
        gate = self.project/'outputs/future-innovation/gate-v2'
        atomic_json(gate/'config/run-contract.json',{'protocol':'direct-v2','run_id':'gate-v2'})
        self.assertEqual(resolve(self.project,'gate-v2',definition),gate)
        self.assertEqual(migration_plan(self.project,definition),[])

    def test_archived_relative_links_survive_without_editing_notebooks(self):
        (self.project/'docs').mkdir(); (self.project/'docs/protocol.md').write_text('Protocol')
        source = self.project/'outputs/old-child/notebook_runs/batch/notebook.ipynb'
        atomic_json(source, {'cells':[{'cell_type':'markdown','source':'[Protocol](../../../../docs/protocol.md)'}]})
        before = source.read_bytes()
        migrate(self.project,self.definitions,apply=True)
        notebook = resolve(self.project,'child',self.definitions)/'notebook_runs/batch/notebook.ipynb'
        self.assertEqual(notebook.read_bytes(),before)
        self.assertTrue((notebook.parent/'../../../../docs/protocol.md').exists())
        self.assertEqual(len(refresh(self.project,self.definitions)['entries']),2)


if __name__ == "__main__":
    unittest.main()
