"""Exercise notebook Slurm routing and dependencies with local command stand-ins."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "slurm/future-innovation"


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
        for script in sorted(SCRIPTS.glob("1[0-4]-*.sbatch")):
            result = self.command(script.name)
            self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 5)
        for number, args in enumerate(calls):
            self.assertEqual(args[:3], ["run", "--no-sync", "python"])
            self.assertEqual(args[3], str(PROJECT / "scripts/research_directions/future_innovation/execute_future_innovation_notebook.py"))
            self.assertEqual(args[4:10], ["--notebook", f"{number:02}", "--mode", "execute", "--run-root", str(self.run)])
        self.assertEqual(calls[2][-2:], ["--device", "cuda"])
        self.assertEqual(calls[3][-2:], ["--outer-fold", "3"])
        fit = (SCRIPTS / "13-notebook-03-fit.sbatch").read_text()
        self.assertIn("#SBATCH --array=0-4%5", fit)
        self.assertNotIn("--gres", fit)
        self.assertIn("#SBATCH --gres=gpu:h100:1", (SCRIPTS / "12-notebook-02-teacher.sbatch").read_text())

    def test_dependency_graph_and_report_waits_for_all_predecessors(self):
        self.mock_command("sbatch", "print(str(100+n)+';cluster')\n")
        (self.run / "config").mkdir(parents=True)
        (self.run / "config/cohort-contract.json").write_text("{}")
        for phase in ("prepare", "compute", "all"):
            result = self.command("submit-fi-notebooks.sh", phase)
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

    def test_submission_failure_stops_downstream_jobs_and_preserves_receipts(self):
        self.mock_command("sbatch", "print(100+n)\nsys.exit(1 if n==1 else 0)\n")
        result = self.command("submit-fi-notebooks.sh", "all")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("submission failed at cohort", result.stderr)
        self.assertEqual(len(self.calls()), 2)
        self.assertEqual(len((self.run / "logs/notebook-submissions.tsv").read_text().splitlines()), 1)

    def test_compute_missing_cohort_and_invalid_phase_never_submit(self):
        self.mock_command("sbatch", "raise AssertionError('must not submit')\n")
        for phase, message in (("compute", "No completed cohort"), ("invalid", "prepare|compute|all")):
            result = self.command("submit-fi-notebooks.sh", phase)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)
        self.assertFalse(self.log.exists())

    def test_failed_notebook_executor_propagates_to_slurm(self):
        self.mock_command("uv", "sys.exit(7)\n")
        result = self.command("14-notebook-04-report.sbatch")
        self.assertEqual(result.returncode, 7)


if __name__ == "__main__":
    unittest.main()
