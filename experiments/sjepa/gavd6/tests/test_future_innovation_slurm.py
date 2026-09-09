import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "slurm/future-innovation"


class FutureInnovationSlurmTests(unittest.TestCase):
    def test_shell_syntax(self):
        for script in sorted(SCRIPTS.iterdir()):
            if script.suffix in (".sh", ".sbatch"):
                completed = subprocess.run(
                    ["bash", "-n", str(script)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_common_helper_allows_run_root_inside_checkout(self):
        env = {
            **os.environ,
            "GAVD6_ROOT": str(PROJECT),
            "FI_RUN_ROOT": str(PROJECT / "outputs/future-innovation/gate-v1"),
        }
        completed = subprocess.run(
            [
                "bash",
                "-c",
                'mkdir() { :; }; source "$1"',
                "_",
                str(SCRIPTS / "fi-common.sh"),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_each_job_routes_to_existing_cli_and_outer_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binaries = root / "bin"
            binaries.mkdir()
            log = root / "commands.jsonl"
            mock = binaries / "uv"
            mock.write_text(
                "#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ['FI_TEST_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            )
            mock.chmod(0o755)
            run = root / "run"
            (run / "qc").mkdir(parents=True)
            (run / "qc/alignment-review.json").write_text("{}")
            annotation = root / "dataset/annotations/GAVD/data"
            annotation.mkdir(parents=True)
            for part in range(1, 6):
                (annotation / f"GAVD_Clinical_Annotations_{part}.csv").touch()
            env = {
                **os.environ,
                "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                "GAVD6_ROOT": str(PROJECT),
                "FI_RUN_ROOT": str(run),
                "GAVD_FULL_ROOT": str(root / "dataset"),
                "FI_TEST_LOG": str(log),
                "VJEPA2_ROOT": str(root / "teacher"),
                "FI_TEACHER_CHECKPOINT": str(root / "checkpoint.pt"),
                "FI_POSE_MODEL": str(root / "pose.task"),
                "FI_CHANGE_REASON": "test",
                "SLURM_ARRAY_TASK_ID": "3",
            }
            for script in sorted(SCRIPTS.glob("*.sbatch")):
                completed = subprocess.run(
                    ["bash", str(script)],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            from gavd6_sjepa.command_line_interface import COMMANDS

            for call in calls:
                self.assertEqual(
                    call[:5],
                    [
                        "run",
                        "--no-sync",
                        "python",
                        "-m",
                        "gavd6_sjepa.command_line_interface",
                    ],
                )
                self.assertIn((call[5], call[6]), COMMANDS)
                self.assertEqual(call[-2:], ["--run-root", str(run)])
            fit = next(call for call in calls if call[6] == "run-gate")
            self.assertIn("3", fit)

    def test_dependency_submission_and_absolute_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binaries = root / "bin"
            binaries.mkdir()
            log = root / "jobs.jsonl"
            mock = binaries / "sbatch"
            mock.write_text(
                "#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\np=Path(os.environ['FI_TEST_LOG'])\nn=len(p.read_text().splitlines()) if p.exists() else 0\nwith p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\nprint(100+n)\n"
            )
            mock.chmod(0o755)
            run = root / "run"
            (run / "qc").mkdir(parents=True)
            (run / "qc/alignment-review.json").write_text("{}")
            env = {
                **os.environ,
                "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                "GAVD6_ROOT": str(PROJECT),
                "FI_RUN_ROOT": str(run),
                "FI_TEST_LOG": str(log),
            }
            for phase in ("prepare", "compute"):
                completed = subprocess.run(
                    ["bash", str(SCRIPTS / "submit-fi-pipeline.sh"), phase],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 6)
            self.assertIn("--dependency=afterok:100", calls[1])
            self.assertIn("--dependency=afterok:102", calls[3])
            self.assertIn("--dependency=afterok:103", calls[4])
            self.assertIn("--dependency=afterany:104", calls[5])
            for call in calls:
                self.assertIn("--kill-on-invalid-dep=yes", call)
                self.assertTrue(
                    any(arg.startswith(f"--output={run}/logs/") for arg in call)
                )
