"""Evidence-state and generation checks for the read-only teaching layer."""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest

import nbformat

from gavd6_sjepa.research_directions.future_innovation.fi_tutorial_inspection import (
    artifact_inventory,
    inspect_report,
    read_optional_table,
)


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/research_directions/future_innovation/build_future_innovation_notebooks.py"
spec = importlib.util.spec_from_file_location("fi_tutorial_builder", BUILDER_PATH)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class FutureInnovationTutorialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def decision(self, *, complete=True, synthetic=False, decision="STOP", sealed=True):
        value = {"measurement_complete": complete, "synthetic": synthetic,
                 "decision": decision, "checks": {"real_gain": False}, "metrics": {}}
        path = self.write("reports/gate-decision.json", value)
        narrative = self.root / "reports/gate-report.md"
        narrative.write_text("# Fixture report\n\nSynthetic test fixture, not a research result.\n")
        if sealed:
            self.write("reports/final-report-contract.json", {"artifacts": {
                str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (path, narrative)
            }})

    def test_missing_copy_does_not_create_run_or_infer_remote_state(self):
        missing = self.root / "unavailable"
        self.assertEqual(inspect_report(missing)["state"], "UNAVAILABLE")
        self.assertFalse(artifact_inventory(missing).present_locally.any())
        self.assertIsNone(read_optional_table(missing, "manifests/gate-windows.csv"))
        self.assertFalse(missing.exists())

    def test_incomplete_stop_is_not_a_negative_scientific_result(self):
        self.decision(complete=False, sealed=False)
        self.assertEqual(inspect_report(self.root)["state"], "INCOMPLETE")

    def test_synthetic_report_never_becomes_scientific_advance(self):
        self.decision(synthetic=True, decision="ADVANCE")
        result = inspect_report(self.root)
        self.assertEqual(result["state"], "SYNTHETIC")
        self.assertIn("cannot", result["explanation"])

    def test_complete_report_requires_both_seal_entries_and_real_data_flag(self):
        self.decision(sealed=False)
        self.assertEqual(inspect_report(self.root)["state"], "UNVERIFIED")
        self.write("reports/final-report-contract.json", {"artifacts": {}})
        self.assertEqual(inspect_report(self.root)["state"], "INVALID")
        self.decision(synthetic=None)
        self.assertEqual(inspect_report(self.root)["state"], "UNVERIFIED")

    def test_complete_negative_and_advance_reports_are_read_without_writes(self):
        for decision in ("STOP", "INCONCLUSIVE", "ADVANCE"):
            with self.subTest(decision=decision):
                self.decision(decision=decision)
                before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
                result = inspect_report(self.root)
                self.assertEqual(result["state"], f"COMPLETE / {decision}")
                self.assertTrue(result["seal_verified"])
                self.assertIsNotNone(result["report_text"])
                self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns)
                                         for p in self.root.rglob("*") if p.is_file()})

    def test_changed_report_is_not_displayed_as_valid(self):
        self.decision()
        (self.root / "reports/gate-report.md").write_text("Changed after sealing")
        result = inspect_report(self.root)
        self.assertEqual(result["state"], "INVALID")
        self.assertIsNone(result["decision"])
        self.assertIsNone(result["report_text"])

    def test_malformed_existing_evidence_is_not_treated_as_missing(self):
        self.write("reports/gate-decision.json", [])
        self.assertEqual(inspect_report(self.root)["state"], "INVALID")
        path = self.root / "broken.csv"
        path.write_text('a,b\n"unterminated')
        with self.assertRaises(ValueError):
            read_optional_table(self.root, "broken.csv")

    def test_generation_and_notebook_links(self):
        for number, name in builder.NAMES.items():
            with self.subTest(notebook=name):
                expected = builder.render(number)
                path = builder.DESTINATION / name
                actual = nbformat.read(path, as_version=4)
                self.assertEqual(json.loads(nbformat.writes(actual)), json.loads(nbformat.writes(expected)))
                self.assertTrue(all(not c.get("outputs") for c in actual.cells))
                for cell in actual.cells:
                    if cell.cell_type == "markdown":
                        for target in re.findall(r"\]\(([^)]+)\)", cell.source):
                            if "://" not in target and not target.startswith("#"):
                                self.assertTrue((path.parent / target.split("#")[0]).exists(), target)


if __name__ == "__main__":
    unittest.main()
