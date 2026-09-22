"""Run the complete controller loop with fake compute and real scope artifacts."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/research_directions/synthetic_training_v2/full_experiment.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("stv2_full_worker_tests", SCRIPT)
FULL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FULL)


class FullExperimentWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / "full-01"
        self.output.mkdir()
        self.request = dict(
            output=str(self.output), python=sys.executable, pilot_work=str(self.root / "pilot-01"),
            pilot_bundle=str(self.root / "pilot-01/paired/bundle"),
            allocation_seconds=7200, authorized_gpu_hours=3.,
            prior_entries=[dict(stage="previous_allocations", gpu_seconds=200.)],
            preparation=dict(review_mode="automated_development"),
            train_people=24, development_people=8, windows_per_person=4, max_candidates=512,
            reservation_sources=[str(self.root / "known-people.csv")],
            limits=dict(preparation_seconds=600., fit_stage_seconds={"200": 300., "2000": 600.}),
            updates=[200, 2000], seeds=[17, 29, 43],
            batch_size=64, model={}, held_extractor="vitpose",
        )
        self.request_path = self.output / "request.json"
        FULL.write_new(self.request_path, self.request)
        FULL.write_new(self.output / "submission.json", dict(token="test-token", job_id="777", status="submitted"))
        self.commands, self.summaries = [], {}

    def run_fake_worker(self, *, fail_analysis=None):
        analysis_count = 0

        def fake_report(configs, diagnostics, output):
            output = Path(output)
            # Preserve report_suite's actual immutability rule: a second call
            # with the same directory must fail rather than overwrite history.
            output.mkdir(parents=True, exist_ok=False)
            self.assertEqual(len(configs), len(diagnostics))
            self.assertTrue(all((Path(path) / "analysis.json").is_file() for path in diagnostics))
            text = f"# Mock summary\n\nVerified analyses: {len(configs)}\n"
            (output / "report.md").write_text(text)
            self.summaries[output] = text

        def fake_execute(allocation, phase, command):
            nonlocal analysis_count
            self.commands.append((phase, list(command)))
            allocation.update(phase)
            if phase.startswith("fit_"):
                config_path = Path(command[command.index("--config") + 1])
                config = RunConfig.load(config_path)
                config.require_gpu_scope()
                config.root.mkdir(parents=True, exist_ok=False)
                FULL.write_new(config.root / "mock-fit.json", dict(updates=config.updates, seed=config.seed))
            elif phase.startswith("analyze_"):
                analysis_count += 1
                if analysis_count == fail_analysis:
                    raise RuntimeError("Simulated second analysis failure")
                diagnostic = Path(command[command.index("--output") + 1])
                FULL.write_new(diagnostic / "analysis.json", dict(status="mock_analysis_completed"))
            elif phase == "refresh_partial_summary":
                # Execute the real CLI report branch, including its selection
                # of a new immutable summary and replacement of the root view.
                self.assertEqual(FULL.main(command[2:]), 0)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"SLURM_JOB_ID": "777", "SLURM_JOB_NAME": "stv2-full-test-token"}))
            stack.enter_context(patch.object(FULL, "assert_request_unchanged"))
            stack.enter_context(patch.object(FULL, "verify_panel"))
            stack.enter_context(patch("haic_inputs.check_inputs"))
            stack.enter_context(patch.object(FULL.Allocation, "execute", new=fake_execute))
            stack.enter_context(patch.object(FULL.Allocation, "elapsed", new=lambda allocation: 90. + 10. * len(self.commands)))
            stack.enter_context(patch("expansion_results.report_suite", side_effect=fake_report))
            process = stack.enter_context(patch.object(FULL.subprocess, "Popen"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            try:
                FULL.run_request(self.request_path, FULL.digest(self.request))
            finally:
                process.assert_not_called()

    def test_six_scope_worker_preserves_each_summary_and_cumulative_cost_snapshot(self):
        original_request = self.request_path.read_bytes()
        self.run_fake_worker()
        expected = [(updates, seed) for seed in (17, 29, 43) for updates in (200, 2000)]
        completed = FULL.read(self.output / "completed-scopes.json")
        self.assertEqual(completed["expected_scopes"], 6)
        self.assertEqual(len(completed["configs"]), 6)
        configs = [RunConfig.load(path) for path in completed["configs"]]
        self.assertEqual([(config.updates, config.seed) for config in configs], expected)
        self.assertTrue(all(config.readout_updates == config.updates and config.seeds == [config.seed] for config in configs))
        self.assertEqual(len([phase for phase, _ in self.commands if phase.startswith("fit_")]), 6)
        self.assertEqual(len([phase for phase, _ in self.commands if phase.startswith("analyze_")]), 6)
        self.assertEqual(len([phase for phase, _ in self.commands if phase == "refresh_partial_summary"]), 6)
        self.assertEqual([path.name for path in self.summaries], [f"completed-{index:02d}" for index in range(1, 7)])
        for path, expected_text in self.summaries.items():
            self.assertEqual((path / "report.md").read_text(), expected_text)
        root_report = (self.output / "report.md").read_text()
        self.assertIn("Completed **6 of 6**", root_report)
        self.assertIn("Verified analyses: 6", root_report)
        latest = FULL.read(self.output / "latest-summary.json")
        self.assertEqual(latest["completed_scopes"], 6)
        self.assertEqual(Path(latest["output"]).name, "completed-06")
        previous_elapsed = 0.
        for config in configs:
            ledger = FULL.read(config.cost_ledger)
            self.assertEqual(len(ledger["entries"]), 2)
            self.assertEqual(ledger["entries"][0], self.request["prior_entries"][0])
            current = ledger["entries"][1]["gpu_seconds"]
            self.assertGreater(current, previous_elapsed)
            self.assertAlmostEqual(config.measured_gpu_hours, (200. + current) / 3600)
            previous_elapsed = current
            config.require_gpu_scope()
        status = FULL.read(self.output / "status.json")
        self.assertEqual(status["status"], "worker_complete")
        self.assertNotEqual(status["status"], "FULL_EXPERIMENT_COMPLETE")
        self.assertEqual(status["completed_scopes"], 6)
        self.assertIn("Slurm COMPLETED", status["scheduler"])
        self.assertFalse(status["confirmation_opened"])
        self.assertEqual(self.request_path.read_bytes(), original_request)
        screening = next(command for phase, command in self.commands if phase == "screen_new_people")
        self.assertIn("--exclude-bundle", screening)
        self.assertEqual(screening[screening.index("--reservation-csv") + 1], self.request["reservation_sources"][0])

    def test_second_analysis_failure_preserves_first_completed_scope_and_stops(self):
        with self.assertRaisesRegex(RuntimeError, "second analysis failure"):
            self.run_fake_worker(fail_analysis=2)
        completed = FULL.read(self.output / "completed-scopes.json")
        self.assertEqual(len(completed["configs"]), 1)
        self.assertEqual(completed["expected_scopes"], 6)
        first = RunConfig.load(completed["configs"][0])
        self.assertEqual((first.updates, first.seed), (200, 17))
        self.assertTrue((first.root / "mock-fit.json").is_file())
        self.assertTrue((Path(completed["diagnostics"][0]) / "analysis.json").is_file())
        self.assertEqual(list(self.summaries), [self.output / "summaries/completed-01"])
        self.assertIn("Completed **1 of 6**", (self.output / "report.md").read_text())
        self.assertEqual(self.commands[-1][0], "analyze_updates-2000-seed-17")
        self.assertEqual(len([phase for phase, _ in self.commands if phase.startswith("fit_")]), 2)
        self.assertEqual(len([phase for phase, _ in self.commands if phase.startswith("analyze_")]), 2)
        self.assertFalse((self.output / "config/updates-0200-seed-29.json").exists())
        status = FULL.read(self.output / "status.json")
        self.assertEqual(status["status"], "failed")
        self.assertIn("second analysis failure", status["error"])
        self.assertIn("completed-scopes.json", status["completed_results"])


if __name__ == "__main__":
    unittest.main()
