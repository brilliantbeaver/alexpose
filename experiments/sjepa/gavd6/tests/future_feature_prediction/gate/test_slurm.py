"""future feature prediction / gate / test slurm."""


import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.support import REPO_ROOT

PROJECT = REPO_ROOT

SCRIPTS = PROJECT / "slurm/future-prediction"

class FutureInnovationNotebookSlurmTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "calls.jsonl"
        self.run = self.root / "run with spaces"
        self.env = {**os.environ, "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
                    "GAVD6_ROOT": str(PROJECT), "FI_RUN_ROOT": str(self.run),
                    "FI_TEST_LOG": str(self.log), "SLURM_ARRAY_TASK_ID": "3"}

    def mock_command(self, name, extra):
        path = self.bin / name
        path.write_text("#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\n"
                        "p=Path(os.environ['FI_TEST_LOG'])\n"
                        "n=len(p.read_text().splitlines()) if p.exists() else 0\n"
                        "with p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n" + extra)
        path.chmod(0o755)

    def command(self, script, *args):
        return subprocess.run(["bash", str(SCRIPTS / script), *args], env=self.env,
                              cwd="/tmp", capture_output=True, text=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_all_notebook_jobs_route_to_executor_with_real_mode(self):
        self.mock_command("uv", "")
        self.env["FI_TUTORIAL_MODE"] = "teach"
        for relative in ("notebooks/setup.sbatch", "notebooks/cohort.sbatch", "notebooks/teacher-cache.sbatch", "notebooks/fit.sbatch", "notebooks/report.sbatch"):
            script = SCRIPTS / relative
            result = self.command(str(script.relative_to(SCRIPTS)))
            self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 5)
        for number, args in enumerate(calls):
            self.assertEqual(args[:3], ["run", "--no-sync", "python"])
            self.assertEqual(args[3], str(PROJECT / "scripts/research_directions/future_prediction/execute_notebook.py"))
            self.assertEqual(args[4:10], ["--notebook", f"{number:02}", "--mode", "execute", "--run-root", str(self.run)])
        self.assertEqual(calls[2][-2:], ["--device", "cuda"])
        self.assertEqual(calls[3][-2:], ["--outer-fold", "3"])
        fit = (SCRIPTS / "notebooks/fit.sbatch").read_text()
        self.assertIn("#SBATCH --array=0-4%5", fit)
        self.assertNotIn("--gres", fit)
        self.assertIn("#SBATCH --gres=gpu:h100:1", (SCRIPTS / "notebooks/teacher-cache.sbatch").read_text())

    def test_dependency_graph_and_report_waits_for_all_predecessors(self):
        self.mock_command("sbatch", "print(str(100+n)+';cluster')\n")
        (self.run / "config").mkdir(parents=True)
        (self.run / "config/cohort-contract.json").write_text("{}")
        for phase in ("prepare", "compute", "all"):
            result = self.command("submit-notebooks.sh", phase)
            self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 10)
        expected = {1: "afterok:100", 3: "afterany:102", 4: "afterany:102:103",
                    6: "afterok:105", 7: "afterok:106", 8: "afterany:107", 9: "afterany:105:106:107:108"}
        for index, dependency in expected.items():
            self.assertIn(f"--dependency={dependency}", calls[index])
        for index in (0, 2, 5):
            self.assertFalse(any(a.startswith("--dependency=") for a in calls[index]))
        self.assertIn(f"--output={self.run}/logs/notebook-fit-%A_%a.out", calls[8])
        for call in calls:
            self.assertIn("--kill-on-invalid-dep=yes", call)
            self.assertIn("--export=ALL", call)
            self.assertIn(f"--chdir={PROJECT}", call)
            self.assertTrue(Path(call[-1]).is_absolute())
        receipts = (self.run / "logs/notebook-submissions.tsv").read_text().splitlines()
        self.assertEqual(len(receipts), 10)
        folders = list((self.run / "notebook_runs").iterdir())
        self.assertEqual(len(folders), 3)  # One shared folder per submission, not per notebook.
        self.assertTrue(all(p.is_dir() and p.name.startswith("haic-") for p in folders))

    def test_cached_cli_and_notebook_paths_request_no_gpu(self):
        self.mock_command("sbatch", "print(str(100+n)+';cluster')\n")
        self.env.update(FI_PARENT_ROOT=str(self.root/'parent'),FI_CALIBRATION=str(self.root/'calibration.json'))
        for script,args,count in (("submit-repair.sh",(),3),("submit-notebooks.sh",("cached",),5)):
            result=self.command(script,*args)
            self.assertEqual(result.returncode,0,result.stderr)
        calls=self.calls()
        self.assertEqual(len(calls),8)
        self.assertIn('--dependency=afterok:100',calls[1])
        self.assertIn('--dependency=afterany:100:101',calls[2])
        self.assertTrue(calls[5][-1].endswith('notebooks/reuse-cache.sbatch'))
        for call in calls:
            self.assertNotIn('--gres',Path(call[-1]).read_text())
        self.mock_command('uv','')
        result=self.command('initialize-repair.sbatch')
        self.assertEqual(result.returncode,0,result.stderr)
        result=self.command('notebooks/reuse-cache.sbatch')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(self.calls()[-1][-2:],['--device','cpu'])

    def test_submission_failure_stops_downstream_jobs_and_preserves_receipts(self):
        self.mock_command("sbatch", "print(100+n)\nsys.exit(1 if n==1 else 0)\n")
        result = self.command("submit-notebooks.sh", "all")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("submission failed at cohort", result.stderr)
        self.assertEqual(len(self.calls()), 2)
        self.assertEqual(len((self.run / "logs/notebook-submissions.tsv").read_text().splitlines()), 1)

    def test_compute_missing_cohort_and_invalid_phase_never_submit(self):
        self.mock_command("sbatch", "raise AssertionError('must not submit')\n")
        for phase, message in (("compute", "No completed cohort"), ("invalid", "prepare|compute|all")):
            result = self.command("submit-notebooks.sh", phase)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)
        self.assertFalse(self.log.exists())

    def test_failed_notebook_executor_propagates_to_slurm(self):
        self.mock_command("uv", "sys.exit(7)\n")
        result = self.command("notebooks/report.sbatch")
        self.assertEqual(result.returncode, 7)


class FutureInnovationSlurmTests(unittest.TestCase):
    def test_shell_syntax(self):
        for script in sorted(SCRIPTS.rglob("*")):
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
                 "_", str(SCRIPTS / "common.sh")],
                cwd="/tmp", env=env, capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.splitlines()[-1], str(run.resolve()))

    def test_compute_requires_completed_preparation_before_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "GAVD6_ROOT": str(PROJECT), "FI_RUN_ROOT": directory}
            completed = subprocess.run(
                ["bash", str(SCRIPTS / "submit-pipeline.sh"), "compute"],
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
                str(SCRIPTS / "submit-pipeline.sh"), "invalid",
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
            for relative in ("build-cohort.sbatch", "extract-poses.sbatch", "cache-teacher.sbatch", "audit-targets.sbatch", "fit-models.sbatch", "build-report.sbatch"):
                script = SCRIPTS / relative
                completed = subprocess.run(
                    ["bash", str(script)],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            from gavd6_sjepa.cli import COMMANDS

            for call in calls:
                self.assertEqual(
                    call[:5],
                    [
                        "run",
                        "--no-sync",
                        "python",
                        "-m",
                        "gavd6_sjepa.cli",
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
                ["bash", str(SCRIPTS / "build-cohort.sbatch")],
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
                    ["bash", str(SCRIPTS / "submit-pipeline.sh"), phase],
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


if __name__ == "__main__":
    unittest.main()
