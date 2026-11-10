#!/usr/bin/env python3
"""Small independent regression fixtures; no network or real dataset mutation.

Run: python3 -m unittest discover -s data/gavd -p 'test_*.py'
"""

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


processor = load_script("process_gavd")
verifier = load_script("verify_processing")


class DatasetFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gavd-regression-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"
        self.output.mkdir()
        self.rows = [
            ["cross_boundary", str(number), "", "", "Abnormal Gait", "cerebral palsy",
             "{'top': 1, 'left': 2, 'height': 3, 'width': 4}",
             "{'height': 100, 'width': 200, 'mime_type': 'video/mp4'}",
             "videoid", "https://www.youtube.com/watch?v=videoid"]
            for number in range(1, 6)
        ]
        self.manifest = {"commit": "a" * 40, "files": []}
        for number, row in enumerate(self.rows, 1):
            path = self.source / f"GAVD_Clinical_Annotations_{number}.csv"
            self.write_csv(path, [row])
            content = path.read_bytes()
            self.manifest["files"].append({
                "name": path.name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "git_blob_sha": hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest(),
                "bytes": len(content),
                "rows": 1,
            })
        (self.source / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        self.result = self.output / "cerebral palsy" / "cross_boundary.csv"

    def write_csv(self, path, rows, header=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header or verifier.HEADER)
            writer.writerows(rows)

    def process(self):
        with contextlib.redirect_stdout(io.StringIO()):
            processor.process(self.source, self.output)

    def verify(self):
        return verifier.verify_processing(self.source, self.output)

    def snapshot(self):
        return {str(path.relative_to(self.output)): path.read_bytes()
                for path in self.output.rglob("*") if path.is_file()}

    def test_combines_sequence_crossing_all_parts_and_preserves_fields(self):
        self.process()
        with self.result.open(newline="", encoding="utf-8") as stream:
            actual = list(csv.reader(stream))
        self.assertEqual(actual, [verifier.HEADER] + self.rows)
        self.assertTrue(self.verify()["success"])

    def test_missing_row_fails(self):
        self.write_csv(self.result, self.rows[:-1])
        report = self.verify()
        self.assertFalse(report["success"])
        self.assertEqual(report["missing_rows"], 1)

    def test_duplicate_row_and_key_fail(self):
        self.write_csv(self.result, self.rows + [self.rows[0]])
        report = self.verify()
        self.assertFalse(report["success"])
        self.assertEqual(report["extra_rows"], 1)
        self.assertEqual(report["duplicate_seq_frame_keys"]["local"], 1)

    def test_altering_empty_field_fails(self):
        edited = [row[:] for row in self.rows]
        edited[0][3] = "unexpected annotation"
        self.write_csv(self.result, edited)
        report = self.verify()
        self.assertFalse(report["success"])
        self.assertEqual((report["missing_rows"], report["extra_rows"]), (1, 1))

    def test_every_header_is_checked(self):
        self.write_csv(self.result, self.rows, header=list(reversed(verifier.HEADER)))
        report = self.verify()
        self.assertFalse(report["success"])
        self.assertTrue(any("Header mismatch" in issue for issue in report["issues"]))

    def test_stale_alias_duplicate_fails(self):
        self.write_csv(self.result, self.rows)
        self.write_csv(self.output / "cerebralpalsy" / self.result.name, self.rows)
        report = self.verify()
        self.assertFalse(report["success"])
        self.assertEqual(report["extra_rows"], 5)
        self.assertTrue(any("Gait label" in issue for issue in report["issues"]))

    def test_source_validation_failure_leaves_existing_output_untouched(self):
        self.write_csv(self.result, self.rows[:1])
        before = self.snapshot()
        with (self.source / self.manifest["files"][-1]["name"]).open("a") as stream:
            stream.write("tampered")
        with self.assertRaises(ValueError):
            self.process()
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(self.verify()["success"])

    def test_unexpected_existing_rows_leave_output_untouched(self):
        edited = [self.rows[0][:]]
        edited[0][3] = "preserve this local annotation"
        self.write_csv(self.result, edited)
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.process()
        self.assertEqual(self.snapshot(), before)

    def test_nested_source_duplicate_inventory_fails(self):
        source = self.source / self.manifest["files"][0]["name"]
        nested = self.source / "stale" / source.name
        nested.parent.mkdir()
        nested.write_bytes(source.read_bytes())
        self.write_csv(self.result, self.rows)
        report = self.verify()
        self.assertFalse(report["success"])
        self.assertTrue(any("exactly the five" in issue for issue in report["issues"]))
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.process()
        self.assertEqual(self.snapshot(), before)

    def test_rerun_keeps_output_bytes_and_mtime(self):
        self.process()
        before_bytes = self.snapshot()
        before_mtime = self.result.stat().st_mtime_ns
        self.process()
        self.assertEqual(self.snapshot(), before_bytes)
        self.assertEqual(self.result.stat().st_mtime_ns, before_mtime)


if __name__ == "__main__":
    unittest.main()
