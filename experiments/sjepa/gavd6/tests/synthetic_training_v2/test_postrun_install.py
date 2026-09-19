"""Execute the README's scoped install in disposable local Git repositories.

Only the HAIC session.env location is substituted. A fake Python executable
captures the resulting CLI call, so these tests cannot submit scheduler jobs.
"""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest

from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import code_identity


ROOT = Path(__file__).resolve().parents[2]
NESTED = Path("experiments/sjepa/gavd6")
FILES = (
    "scripts/research_directions/synthetic_training_v2/diagnostics/__init__.py",
    "scripts/research_directions/synthetic_training_v2/diagnostics/calibration.py",
    "scripts/research_directions/synthetic_training_v2/diagnostics/timing.py",
    "scripts/research_directions/synthetic_training_v2/diagnostics/plots.py",
    "scripts/research_directions/synthetic_training_v2/postrun_checks.py",
    "slurm/synthetic-training-v2/postrun-checks.sh",
    "slurm/synthetic-training-v2/postrun-checks.sbatch",
)
SHARED_CODE = (
    "research_directions/temporal_gait/objectives.py",
    "research_directions/temporal_gait/contracts.py",
    "research_directions/synthetic_training/rendering.py",
    "research_directions/synthetic_training/estimators.py",
    "research_directions/synthetic_training/measurements.py",
    "research_directions/motion_preservation/motion_data.py",
    "research_directions/motion_preservation/body_geometry.py",
    "data_foundations/amass_conversion.py",
)


@unittest.skipUnless(shutil.which("git") and shutil.which("bash"), "Git and Bash are required")
class PostrunInstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="stv2 install with spaces ")
        self.addCleanup(self.temporary.cleanup)
        self.temp = Path(self.temporary.name)
        readme = (ROOT / "slurm/synthetic-training-v2/README.md").read_text()
        match = re.search(r"\*\*4\. On HAIC:.*?```bash\n(.*?)\n```", readme, re.S)
        self.assertIsNotNone(match, "Step 4 needs an executable installation block")
        self.block = match.group(1)
        self.haic_source = 'source "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training-v2/source-smoke-01/session.env"'
        self.assertEqual(self.block.count(self.haic_source), 1)

    def command(self, args, cwd, check=True, env=None):
        return subprocess.run(args, cwd=cwd, env=env, check=check, text=True, capture_output=True)

    def git(self, root, *args):
        return self.command(["git", *args], root).stdout.strip()

    def prepare(self, *, omitted=(), tracked_missing=False):
        remote = self.temp / "local origin"
        remote.mkdir()
        self.git(remote, "init", "-b", "main")
        self.git(remote, "config", "user.name", "Fixture")
        self.git(remote, "config", "user.email", "fixture@example.invalid")
        origin_root = remote / NESTED
        scientific = list((ROOT / "src/gavd6_sjepa/research_directions/synthetic_training_v2").glob("*.py"))
        scientific += [ROOT / "src/gavd6_sjepa" / name for name in SHARED_CODE]
        for source in scientific:
            destination = origin_root / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        stable = {
            "docs/studies/synthetic-training-v2/protocol.md": "original protocol\n",
            "scripts/research_directions/synthetic_training_v2/automated_run.py": "original controller\n",
        }
        for name, contents in stable.items():
            destination = origin_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(contents)
        profile = origin_root / "slurm/synthetic-training-v2/study.env"
        profile.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / profile.relative_to(origin_root), profile)
        (origin_root / FILES[-2]).write_text("# old tracked diagnostic wrapper\n")
        if tracked_missing:
            for name in omitted:
                destination = origin_root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text("# previously tracked diagnostic\n")
        self.git(remote, "add", ".")
        self.git(remote, "commit", "-m", "baseline")
        local = self.temp / "haic checkout"
        self.command(["git", "clone", str(remote), str(local)], self.temp)
        root = local / NESTED
        (root / FILES[-2]).write_text("# locally modified tracked diagnostic wrapper\n")
        for name in FILES[:-2]:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("# local diagnostic version to back up\n")
        # Preserve a local scientific edit as well as unchanged protected files.
        science_path = root / "src/gavd6_sjepa/research_directions/synthetic_training_v2/config.py"
        science_path.write_text(science_path.read_text() + "\n# pre-existing local scientific edit\n")
        for name in FILES:
            destination = origin_root / name
            if name in omitted:
                if destination.exists():
                    destination.unlink()
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        self.git(remote, "add", "--all")
        self.git(remote, "commit", "-m", "publish diagnostics")
        work = root / "outputs/synthetic-training-v2/source smoke"
        work.mkdir(parents=True)
        for name, contents in (("source-01/report.md", "retained report\n"), ("control.json", '{"ledger":"unchanged"}\n')):
            path = work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        capture = self.temp / "python invocation.json"
        python = self.temp / "python capture"
        python.write_text("#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\n"
                          "Path(os.environ['STV2_TEST_CAPTURE']).write_text(json.dumps({'argv':sys.argv[1:], 'cwd':os.getcwd()}))\n")
        python.chmod(0o755)
        session = work / "session.env"
        session.write_text("\n".join("export " + key + "=" + shlex.quote(str(value)) for key, value in
                           dict(STV2_ROOT=root, STV2_WORK=work, STV2_PYTHON=python).items()) + "\n")
        protected = [root / name for name in stable] + [work / "source-01/report.md", work / "control.json"]
        before = {str(path): path.read_bytes() for path in protected}
        return dict(local=local, root=root, work=work, session=session, capture=capture,
                    scientific_identity=code_identity(root), protected=before,
                    head=self.git(local, "rev-parse", "HEAD"), index=self.git(local, "ls-files", "--stage"))

    def execute(self, case):
        block = self.block.replace(self.haic_source, "source " + shlex.quote(str(case["session"])))
        return self.command(["bash", "-c", block], case["root"], check=False,
                            env=dict(os.environ, STV2_TEST_CAPTURE=str(case["capture"])))

    def assert_protected(self, case):
        self.assertEqual(code_identity(case["root"]), case["scientific_identity"])
        self.assertEqual(self.git(case["local"], "rev-parse", "HEAD"), case["head"])
        self.assertEqual(self.git(case["local"], "ls-files", "--stage"), case["index"])
        self.assertEqual({path: Path(path).read_bytes() for path in case["protected"]}, case["protected"])

    def test_full_publication_installs_only_scoped_files_and_preserves_backups(self):
        case = self.prepare()
        result = self.execute(case)
        self.assertEqual(result.returncode, 0, result.stderr)
        call = json.loads(case["capture"].read_text())
        self.assertEqual(Path(call["argv"][0]).resolve(), (case["root"] / FILES[4]).resolve())
        self.assertEqual(call["argv"][1:], ["submit", "--checks", "all"])
        self.assertEqual(Path(call["cwd"]).resolve(), case["root"].resolve())
        backups = list(case["work"].glob("diagnostics-install-backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / FILES[-2]).read_text(), "# locally modified tracked diagnostic wrapper\n")
        self.assertEqual((backups[0] / FILES[0]).read_text(), "# local diagnostic version to back up\n")
        for name in FILES:
            self.assertEqual((case["root"] / name).read_bytes(), (ROOT / name).read_bytes())
        self.assert_protected(case)

    def check_missing_publication(self, omitted, *, tracked_missing=False):
        case = self.prepare(omitted=omitted, tracked_missing=tracked_missing)
        before = {name: (case["root"] / name).read_bytes() for name in FILES if (case["root"] / name).is_file()}
        result = self.execute(case)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(case["capture"].exists(), "Incomplete publication must not invoke the launcher")
        self.assertEqual(list(case["work"].glob("diagnostics-install-backup-*")), [])
        self.assertEqual({name: (case["root"] / name).read_bytes() for name in before}, before)
        self.assert_protected(case)

    def test_missing_individual_helper_fails_before_restore_or_launch(self):
        self.check_missing_publication((FILES[3],))

    def test_deleted_previously_tracked_helper_fails_before_restore_or_launch(self):
        self.check_missing_publication((FILES[3],), tracked_missing=True)

    def test_missing_diagnostic_directory_fails_before_restore_or_launch(self):
        self.check_missing_publication(FILES[:4])


if __name__ == "__main__":
    unittest.main()
