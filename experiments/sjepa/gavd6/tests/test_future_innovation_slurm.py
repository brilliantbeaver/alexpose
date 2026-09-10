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

    def test_common_helper_exports_absolute_paths(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "outputs") as directory:
            run = Path(directory)
            env = {
                **os.environ,
                "GAVD6_ROOT": str(PROJECT),
                "FI_RUN_ROOT": str(run.relative_to(PROJECT)),
            }
            completed = subprocess.run(
                ["bash", "-c", 'source "$1"; bash -c \'printf "%s\\n" "$FI_RUN_ROOT"\'',
                 "_", str(SCRIPTS / "fi-common.sh")],
                cwd="/tmp", env=env, capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.splitlines()[-1], str(run.resolve()))

    def test_compute_requires_completed_preparation_before_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "GAVD6_ROOT": str(PROJECT), "FI_RUN_ROOT": directory}
            completed = subprocess.run(
                ["bash", str(SCRIPTS / "submit-fi-pipeline.sh"), "compute"],
                env=env, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("No completed cohort", completed.stderr)

    def test_invalid_phase_does_not_require_environment(self):
        env = {
            k: v for k, v in os.environ.items() if k not in ("GAVD6_ROOT", "FI_RUN_ROOT")
        }
        completed = subprocess.run(
            [
                "bash",
                str(SCRIPTS / "submit-fi-pipeline.sh"), "invalid",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("prepare|compute|all", completed.stderr)

    def test_each_job_routes_to_existing_cli_and_outer_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            binaries = root / "bin"
            binaries.mkdir()
            log = root / "commands.jsonl"
            mock = binaries / "uv"
            mock.write_text(
                "#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ['FI_TEST_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            )
            mock.chmod(0o755)
            run = root / "run"
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
            for script in sorted(SCRIPTS.glob("0[1-6]-*.sbatch")):
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
            (run / "config").mkdir()
            (run / "config/run-contract.json").write_text("{}")
            for key in ("GAVD_FULL_ROOT", "VJEPA2_ROOT", "FI_TEACHER_CHECKPOINT", "FI_POSE_MODEL", "FI_CHANGE_REASON"):
                env.pop(key)
            completed = subprocess.run(
                ["bash", str(SCRIPTS / "01-build-gate-cohort.sbatch")],
                env=env, capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(log.read_text().splitlines()[-1])[6], "build-cohort")

    def test_dependency_submission_and_absolute_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            binaries = root / "bin"
            binaries.mkdir()
            log = root / "jobs.jsonl"
            mock = binaries / "sbatch"
            mock.write_text(
                "#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\np=Path(os.environ['FI_TEST_LOG'])\nn=len(p.read_text().splitlines()) if p.exists() else 0\nwith p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\nprint(100+n)\n"
            )
            mock.chmod(0o755)
            run = root / "run"
            (run / "config").mkdir(parents=True)
            (run / "config/cohort-contract.json").write_text("{}")
            env = {
                **os.environ,
                "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                "GAVD6_ROOT": str(PROJECT),
                "FI_RUN_ROOT": str(run),
                "FI_TEST_LOG": str(log),
            }
            for phase in ("prepare", "compute", "all"):
                completed = subprocess.run(
                    ["bash", str(SCRIPTS / "submit-fi-pipeline.sh"), phase],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 12)
            self.assertIn("--dependency=afterok:100", calls[1])
            self.assertIn("--dependency=afterok:102", calls[3])
            self.assertIn("--dependency=afterok:103", calls[4])
            self.assertIn("--dependency=afterany:102:103:104", calls[5])
            self.assertIn("--dependency=afterok:107", calls[8])
            self.assertIn("--dependency=afterany:108:109:110", calls[11])
            self.assertIn(f"--output={run}/logs/teacher-%j.out", calls[2])
            self.assertIn(f"--output={run}/logs/fit-%A_%a.out", calls[4])
            for call in calls:
                self.assertIn("--kill-on-invalid-dep=yes", call)
                self.assertIn("--export=ALL", call)
                self.assertTrue(
                    any(arg.startswith(f"--output={run}/logs/") for arg in call)
                )
