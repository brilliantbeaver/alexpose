"""Exercise the generated installer's Python path without HAIC or job submission."""
import base64
from contextlib import contextmanager, redirect_stdout
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/research_directions/synthetic_training_v2/package_full_experiment.py"
SPEC = importlib.util.spec_from_file_location("full_experiment_package_tests", SCRIPT)
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)
HELPER = "scripts/research_directions/synthetic_training_v2/full_experiment.py"
GUIDE = "slurm/synthetic-training-v2/README.md"
PROTOCOL = "docs/studies/synthetic-training-v2/protocol.md"
CONTRACTS = "gavd6_sjepa.research_directions.synthetic_training_v2.contracts"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FullExperimentInstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="full installer ")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "checkout"
        self.pilot = self.root / "outputs/synthetic-training-v2/source-smoke-01"
        self.pilot.mkdir(parents=True)
        (self.root / PROTOCOL).parent.mkdir(parents=True)
        (self.root / PROTOCOL).write_bytes(b"frozen protocol\n")
        (self.root / GUIDE).parent.mkdir(parents=True)
        (self.root / GUIDE).write_bytes(b"old guide\n")
        contents = {HELPER: b"VALUE = 'installed'\n", GUIDE: b"new guide\n"}
        self.package = dict(
            schema=1, scientific_code_identity="science-v1",
            allowed_paths=list(PACKAGE.PAYLOAD_PATHS),
            dependencies={PROTOCOL: digest((self.root / PROTOCOL).read_bytes())},
            files={name: dict(content=base64.b64encode(data).decode("ascii"), sha256=digest(data))
                   for name, data in contents.items()},
        )
        self.contracts = ModuleType(CONTRACTS)
        self.contracts.code_identity = Mock(return_value="science-v1")
        self.contracts.sha256_file = lambda path: digest(Path(path).read_bytes())
        self.haic = ModuleType("haic")
        self.locks = []

        @contextmanager
        def stage_lock(root, name):
            self.assertEqual(Path(root), self.pilot)
            self.locks.append(name)
            yield

        self.haic.stage_lock = stage_lock
        self.haic.state_for = Mock(return_value={"jobs": []})
        self.haic.settled = Mock(return_value={})
        self.haic.scheduler_snapshot = Mock(return_value={})
        self.haic.TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"}

    def execute(self, package=None, *, corrupt_archive=False):
        archive = gzip.compress(json.dumps(package or self.package).encode(), mtime=0)
        source = PACKAGE.INSTALLER.split("<<'PY'\n", 1)[1].rsplit("\nPY\n", 1)[0]
        source = source.replace("__PAYLOAD__", base64.b64encode(archive).decode("ascii"))
        source = source.replace("__ARCHIVE_SHA256__", "0" * 64 if corrupt_archive else digest(archive))
        output = io.StringIO()
        with patch.dict(sys.modules, {CONTRACTS: self.contracts, "haic": self.haic}), \
                patch.object(sys, "argv", ["-", str(self.root), str(self.pilot)]), \
                patch.object(sys, "path", list(sys.path)), redirect_stdout(output):
            exec(compile(source, "generated-haic-installer", "exec"), {"__name__": "__main__"})
        return output.getvalue()

    def assert_not_installed(self):
        self.assertFalse((self.root / HELPER).exists())
        self.assertEqual((self.root / GUIDE).read_bytes(), b"old guide\n")
        self.assertFalse(list(self.pilot.glob("full-experiment-install-backup-*")))

    def test_exact_payload_installs_and_existing_copies_are_backed_up(self):
        output = self.execute()
        self.assertIn("FULL_EXPERIMENT_INSTALLED", output)
        self.assertEqual((self.root / HELPER).read_bytes(), b"VALUE = 'installed'\n")
        self.assertEqual((self.root / GUIDE).read_bytes(), b"new guide\n")
        self.assertEqual((self.root / PROTOCOL).read_bytes(), b"frozen protocol\n")
        backups = list(self.pilot.glob("full-experiment-install-backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / GUIDE).read_bytes(), b"old guide\n")
        self.assertFalse((backups[0] / HELPER).exists())
        installed = json.loads((backups[0] / "installed-files.json").read_text())
        self.assertEqual(set(installed), {HELPER, GUIDE})
        self.assertIn("full-expansion-manage", self.locks)

    def test_second_identical_install_is_read_only_and_needs_no_new_scheduler_check(self):
        self.execute()
        before = {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.haic.settled.reset_mock()
        self.haic.scheduler_snapshot.reset_mock()
        output = self.execute()
        after = {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertIn("already installed", output)
        self.haic.settled.assert_not_called()
        self.haic.scheduler_snapshot.assert_not_called()

    def test_active_original_jobs_prevent_every_payload_write(self):
        self.haic.settled.side_effect = RuntimeError("Jobs still active")
        with self.assertRaisesRegex(RuntimeError, "active"):
            self.execute()
        self.assert_not_installed()

    def test_active_or_unaccounted_expansion_prevents_every_payload_write(self):
        expansion = self.pilot.parent / "full-01"
        write_json(expansion / "request.json", dict(kind="stv2-full-experiment-v1", pilot_work=str(self.pilot)))
        write_json(expansion / "submission.json", dict(job_id="12345", status="submitted"))
        for snapshot in ({}, {"12345": {"state": "PENDING"}}, {"12345": {"state": "RUNNING"}}):
            with self.subTest(snapshot=snapshot):
                self.haic.scheduler_snapshot.return_value = snapshot
                with self.assertRaisesRegex(SystemExit, "active|accounting"):
                    self.execute()
                self.assert_not_installed()

    def test_uncertain_expansion_submission_cannot_be_ignored(self):
        expansion = self.pilot.parent / "full-01"
        write_json(expansion / "request.json", dict(kind="stv2-full-experiment-v1", pilot_work=str(self.pilot)))
        write_json(expansion / "submission.json", dict(status="uncertain"))
        with self.assertRaisesRegex(SystemExit, "Unresolved"):
            self.execute()
        self.assert_not_installed()

    def test_active_diagnostic_job_prevents_replacing_its_modules(self):
        diagnostic = self.pilot / "diagnostics/attempt-01"
        write_json(diagnostic / "submission.json", dict(job_id="12346", status="submitted"))
        self.haic.scheduler_snapshot.return_value = {"12346": {"state": "RUNNING"}}
        with self.assertRaisesRegex(SystemExit, "active|accounting"):
            self.execute()
        self.assert_not_installed()

    def test_unknown_diagnostic_submission_prevents_replacing_its_modules(self):
        diagnostic = self.pilot / "diagnostics/attempt-01"
        write_json(diagnostic / "submission.json", dict(status="submission_uncertain"))
        with self.assertRaisesRegex(SystemExit, "Unresolved|uncertain"):
            self.execute()
        self.assert_not_installed()

    def test_terminal_failed_diagnostic_allows_install_without_rewriting_results(self):
        diagnostic = self.pilot / "diagnostics/attempt-01"
        write_json(diagnostic / "submission.json", dict(job_id="12346", status="submitted"))
        write_json(diagnostic / "status.json", dict(status="failed", error="retained evidence"))
        before = (diagnostic / "status.json").read_bytes()
        self.haic.scheduler_snapshot.return_value = {"12346": {"state": "FAILED"}}
        self.execute()
        self.assertEqual((diagnostic / "status.json").read_bytes(), before)

    def test_revision_dependency_and_archive_mismatches_fail_before_writes(self):
        self.contracts.code_identity.return_value = "different-scientific-code"
        with self.assertRaisesRegex(SystemExit, "scientific code"):
            self.execute()
        self.assert_not_installed()
        self.contracts.code_identity.return_value = "science-v1"
        self.package["dependencies"][PROTOCOL] = "0" * 64
        with self.assertRaisesRegex(SystemExit, "dependency differs"):
            self.execute()
        self.assert_not_installed()
        with self.assertRaisesRegex(SystemExit, "payload checksum"):
            self.execute(corrupt_archive=True)
        self.assert_not_installed()

    def test_unlisted_and_parent_traversal_payloads_are_rejected(self):
        entry = self.package["files"][HELPER]
        self.package["files"]["src/unlisted.py"] = entry
        with self.assertRaisesRegex(SystemExit, "Unsafe payload path"):
            self.execute()
        self.assert_not_installed()
        del self.package["files"]["src/unlisted.py"]
        self.package["files"]["../escape.py"] = entry
        self.package["allowed_paths"].append("../escape.py")
        with self.assertRaisesRegex(SystemExit, "Unsafe payload path"):
            self.execute()
        self.assertFalse((self.base / "escape.py").exists())
        self.assert_not_installed()

    def test_external_parent_symlink_is_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        (self.root / "scripts").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(SystemExit, "Unsafe|[Ss]ymlink"):
            self.execute()
        self.assertEqual(list(outside.iterdir()), [])
        self.assert_not_installed()

    def test_internal_parent_symlink_cannot_redirect_allowed_writes(self):
        internal = self.root / "src/preserved-science"
        internal.mkdir(parents=True)
        (self.root / "scripts").symlink_to(internal, target_is_directory=True)
        with self.assertRaisesRegex(SystemExit, "Unsafe|[Ss]ymlink"):
            self.execute()
        self.assertEqual(list(internal.iterdir()), [])
        self.assert_not_installed()


if __name__ == "__main__":
    unittest.main()
