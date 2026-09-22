"""Full-study orchestration tests; no scheduler or licensed source assets used."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/research_directions/synthetic_training_v2/full_experiment.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("stv2_full_experiment_tests", SCRIPT)
FULL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FULL)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FullExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.pilot = self.root / "pilot-01"
        self.output = self.root / "full-01"
        self.bundle = self.pilot / "paired/bundle"
        self.prepared = self.bundle.parent
        self.config = RunConfig(run_id="source-01", output_root=str(self.pilot),
                                mode="source", device="cuda", bundle=str(self.bundle),
                                held_extractor="vitpose", authorized_gpu_hours=4.)
        self.config_path = self.pilot / "config/source.json"
        write(self.config_path, self.config.as_dict())
        write(self.config.root / "identity.json", {"signature": "retained"})
        write(self.bundle / "manifest.json", {"schema": "coco-body12-xy-v1", "records": []})
        selected = self.pilot / "inputs/automated-01"
        selected.mkdir(parents=True)
        (selected / "locomotion-audit.csv").write_text("relative_path,start_s\n")
        (selected / "person-reservations.csv").write_text("person_id,canonical_person_id,original_split,reserved,exposure\n")
        global_reservations = self.pilot / "inputs/registry-reservations.csv"
        global_reservations.write_text("person_id,canonical_person_id,original_split,reserved,exposure\n")
        draft = self.pilot / "inputs/review-drafts/person-reservations.draft.csv"
        draft.parent.mkdir()
        draft.write_text("person_id,canonical_person_id,original_split,reserved,exposure\n")
        write(selected / "screen.json", {"reservation_sources": [{"path": str(global_reservations),
                                                                  "sha256": FULL.sha256_file(global_reservations)}]})
        self.preparation = {"review_mode": "automated_development",
                            "locomotion_audit": str(selected / "locomotion-audit.csv"),
                            "reservation_csv": str(selected / "person-reservations.csv")}
        write(self.prepared / "preparation-provenance.json",
              {"configuration": self.preparation})
        self.prepare_job = dict(job_id="1001", phase="prepare", stage="prepare", gpu=True,
                                output=str(self.prepared))
        self.state = dict(source_config=str(self.config_path), jobs=[self.prepare_job],
                          account="test-account", partition="test-partition",
                          prior_entries=[dict(stage="prior_setup", gpu_seconds=100.)])
        self.snapshot = {"1001": dict(state="COMPLETED", exit_code="0:0", gpu_seconds=99.,
                                      allocation="cpu=8,gres/gpu=1,gres/gpu:h100=1")}
        self.args = FULL.parser().parse_args([
            "launch", "--pilot-work", str(self.pilot), "--name", "full-01", "--gpu-hours", "2"])

    @contextlib.contextmanager
    def launch_context(self, *, scheduler_result=None, scheduler_error=None):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("check_results.check_results", return_value={"status": "verified"}))
            stack.enter_context(patch.object(FULL.haic, "state_for", return_value=self.state))
            stack.enter_context(patch.object(FULL.haic, "settled", return_value=self.snapshot))
            stack.enter_context(patch("haic_inputs.check_inputs"))
            stack.enter_context(patch.object(FULL, "estimate_limits", return_value=dict(
                preparation_seconds=600., fit_stage_seconds={"200": 300., "2000": 300.})))
            stack.enter_context(patch.object(FULL, "implementation_files", return_value={"test-only": "immutable"}))
            scheduler = stack.enter_context(patch.object(
                FULL.subprocess, "run", side_effect=scheduler_error,
                return_value=scheduler_result or subprocess.CompletedProcess([], 0, stdout="12345;test\n", stderr="")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yield scheduler

    def test_dry_run_creates_no_files_and_submits_no_job(self):
        before = {str(path.relative_to(self.root)): path.read_bytes()
                  for path in self.root.rglob("*") if path.is_file()}
        self.args.dry_run = True
        with self.launch_context() as scheduler:
            self.assertEqual(FULL.launch(self.args), self.output)
            scheduler.assert_not_called()
        after = {str(path.relative_to(self.root)): path.read_bytes()
                 for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(self.output.exists())

    def test_success_records_request_before_submission_and_scrubs_sbatch_overrides(self):
        with self.launch_context() as scheduler, patch.dict(os.environ, {"SBATCH_GRES": "gpu:h100:8"}):
            FULL.launch(self.args)
        request = FULL.read(self.output / "request.json")
        submission = FULL.read(self.output / "submission.json")
        self.assertEqual(submission["status"], "submitted")
        self.assertEqual(submission["job_id"], "12345")
        self.assertEqual(request["updates"], [200, 2000])
        self.assertEqual(request["seeds"], [17, 29, 43])
        self.assertFalse(request["confirmation_opened"])
        self.assertEqual(request["evidence_status"], "automated-source-screen")
        self.assertEqual(request["allocation_seconds"], 7200)
        self.assertAlmostEqual(request["authorized_gpu_hours"], 2 + 200 / 3600)
        self.assertEqual(len(request["reservation_sources"]), 3)
        self.assertTrue(set(request["reservation_sources"]) <= set(request["pilot_files"]))
        self.assertNotIn("SBATCH_GRES", scheduler.call_args.kwargs["env"])
        self.assertIn("--no-requeue", scheduler.call_args.args[0])
        self.assertIn("--gres=gpu:h100:1", scheduler.call_args.args[0])

    def test_existing_request_is_never_overwritten_or_resubmitted(self):
        write(self.output / "request.json", {"sentinel": "preserve"})
        before = (self.output / "request.json").read_bytes()
        with self.launch_context() as scheduler:
            FULL.launch(self.args)
            scheduler.assert_not_called()
        self.assertEqual((self.output / "request.json").read_bytes(), before)

    def test_existing_directory_without_request_is_not_adopted(self):
        write(self.output / "retained.json", {"sentinel": "preserve"})
        with self.launch_context() as scheduler, self.assertRaises(FileExistsError):
            FULL.launch(self.args)
        scheduler.assert_not_called()
        self.assertEqual(FULL.read(self.output / "retained.json"), {"sentinel": "preserve"})

    def test_uncertain_scheduler_response_is_retained_and_retry_does_not_resubmit(self):
        with self.launch_context(scheduler_error=subprocess.TimeoutExpired("sbatch", 90)) as scheduler:
            with self.assertRaises(subprocess.TimeoutExpired):
                FULL.launch(self.args)
            first_request = (self.output / "request.json").read_bytes()
            record = FULL.read(self.output / "submission.json")
            self.assertEqual(record["status"], "uncertain")
            self.assertTrue(record["token"])
            FULL.launch(self.args)
            self.assertEqual(scheduler.call_count, 1)
        self.assertEqual((self.output / "request.json").read_bytes(), first_request)

    def test_bad_submission_id_retains_uncertainty(self):
        result = subprocess.CompletedProcess([], 0, stdout="unexpected text\n", stderr="")
        with self.launch_context(scheduler_result=result), self.assertRaisesRegex(ValueError, "sbatch"):
            FULL.launch(self.args)
        self.assertEqual(FULL.read(self.output / "submission.json")["status"], "uncertain")

    def test_insufficient_budget_and_candidate_bound_fail_before_submit(self):
        for name, value in (("gpu_hours", 0.1), ("max_candidates", 4),
                            ("windows_per_person", 1), ("train_people", 1)):
            with self.subTest(argument=name):
                args = FULL.parser().parse_args([
                    "launch", "--pilot-work", str(self.pilot), "--name", "full-01", "--gpu-hours", "2"])
                setattr(args, name, value)
                with self.launch_context() as scheduler, self.assertRaises((ValueError, PermissionError)):
                    FULL.launch(args)
                scheduler.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_prior_plus_new_allocation_cannot_exceed_study_ceiling(self):
        self.state["prior_entries"] = [dict(stage="prior_scope", gpu_seconds=47 * 3600)]
        with self.launch_context() as scheduler, self.assertRaisesRegex(PermissionError, "48"):
            FULL.launch(self.args)
        scheduler.assert_not_called()
        self.assertFalse(self.output.exists())

    def previous_attempt(self, name="earlier-01", job_id="2001", *, pilot=None):
        folder = self.root / name
        write(folder / "request.json", dict(kind="stv2-full-experiment-v1",
              pilot_work=str(pilot or self.pilot), prior_entries=[dict(gpu_seconds=999999.)]))
        write(folder / "submission.json", dict(status="uncertain" if job_id is None else "submitted", job_id=job_id))
        return folder

    def test_previous_failed_and_successful_allocations_count_once_without_their_snapshots(self):
        self.previous_attempt("earlier-01", "2001")
        self.previous_attempt("earlier-02", "2002")
        snapshot = {"2001": dict(state="FAILED", exit_code="1:0", gpu_seconds=120., allocation="gres/gpu=1,gres/gpu:h100=1"),
                    "2002": dict(state="COMPLETED", exit_code="0:0", gpu_seconds=600., allocation="gres/gpu=1,gres/gpu:h100=1")}
        with patch.object(FULL.haic, "scheduler_snapshot", return_value=snapshot):
            entries = FULL.previous_expansion_costs(self.pilot, self.output)
        self.assertEqual(len(entries), 2)
        self.assertEqual(sum(row["gpu_seconds"] for row in entries), 722.)
        self.assertEqual({row["state"] for row in entries}, {"FAILED", "COMPLETED"})

    def test_previous_active_or_unaccounted_allocation_blocks_submission(self):
        self.previous_attempt()
        for snapshot in ({}, {"2001": dict(state="RUNNING", exit_code="0:0", gpu_seconds=120., allocation="gres/gpu=1")}):
            with self.subTest(snapshot=snapshot), patch.object(FULL.haic, "scheduler_snapshot", return_value=snapshot):
                with self.assertRaisesRegex(ValueError, "active or awaiting accounting"):
                    FULL.previous_expansion_costs(self.pilot, self.output)

    def test_previous_uncertain_submission_blocks_even_without_job_id(self):
        self.previous_attempt(job_id=None)
        with patch.object(FULL.haic, "scheduler_snapshot") as scheduler:
            with self.assertRaisesRegex(ValueError, "Unresolved"):
                FULL.previous_expansion_costs(self.pilot, self.output)
            scheduler.assert_not_called()

    def test_previous_allocation_must_have_one_consistently_declared_gpu(self):
        self.previous_attempt()
        for resources in ("gres/gpu=2", "cpu=8", "gres/gpu=1,gres/gpu:h100=2"):
            with self.subTest(resources=resources):
                snapshot = {"2001": dict(state="COMPLETED", exit_code="0:0", gpu_seconds=120., allocation=resources)}
                with patch.object(FULL.haic, "scheduler_snapshot", return_value=snapshot):
                    with self.assertRaises(ValueError):
                        FULL.previous_expansion_costs(self.pilot, self.output)

    def test_other_pilot_and_current_output_are_not_double_counted(self):
        self.previous_attempt("another-pilot-expansion", "2001", pilot=self.root / "another-pilot")
        self.previous_attempt("full-01", "2002")
        with patch.object(FULL.haic, "scheduler_snapshot", return_value={}) as scheduler:
            self.assertEqual(FULL.previous_expansion_costs(self.pilot, self.output), [])
            scheduler.assert_called_once_with([])

    def test_missing_prior_reservation_file_is_not_silently_forgotten(self):
        sources = FULL.read(Path(self.preparation["locomotion_audit"]).parent / "screen.json")["reservation_sources"]
        Path(sources[0]["path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            FULL.reservation_sources(self.pilot, self.preparation)

    def test_changed_prior_reservation_file_requires_explicit_reconciliation(self):
        sources = FULL.read(Path(self.preparation["locomotion_audit"]).parent / "screen.json")["reservation_sources"]
        Path(sources[0]["path"]).write_text("silently changed knowledge")
        with self.assertRaises(ValueError):
            FULL.reservation_sources(self.pilot, self.preparation)

    def allocation(self):
        request = dict(output=str(self.output), allocation_seconds=7200,
                       prior_entries=[dict(stage="prior", gpu_seconds=100.)],
                       batch_size=64, model={}, held_extractor="vitpose", authorized_gpu_hours=3.)
        return FULL.Allocation(request)

    def test_scope_costs_are_cumulative_snapshots_not_sum_of_previous_scopes(self):
        allocation = self.allocation()
        with patch.dict(os.environ, {"SLURM_JOB_ID": "777"}), patch.object(allocation, "remaining", return_value=5000):
            with patch.object(allocation, "elapsed", return_value=120.):
                first = allocation.scope("updates-0200-seed-17", self.bundle, 200, 17, 300.)
            with patch.object(allocation, "elapsed", return_value=600.):
                second = allocation.scope("updates-2000-seed-17", self.bundle, 2000, 17, 300.)
        a, b = RunConfig.load(first), RunConfig.load(second)
        self.assertAlmostEqual(a.measured_gpu_hours, 220 / 3600)
        self.assertAlmostEqual(b.measured_gpu_hours, 700 / 3600)
        self.assertEqual(len(FULL.read(b.cost_ledger)["entries"]), 2)
        a.require_gpu_scope()
        b.require_gpu_scope()

    def test_scope_refuses_shortened_budget_and_existing_files(self):
        allocation = self.allocation()
        with patch.dict(os.environ, {"SLURM_JOB_ID": "777"}), patch.object(allocation, "remaining", return_value=500):
            with self.assertRaises(PermissionError):
                allocation.scope("updates-0200-seed-17", self.bundle, 200, 17, 300.)
        self.assertFalse(self.output.exists())
        with patch.dict(os.environ, {"SLURM_JOB_ID": "777"}), patch.object(allocation, "remaining", return_value=5000):
            allocation.scope("updates-0200-seed-17", self.bundle, 200, 17, 300.)
            with self.assertRaises(FileExistsError):
                allocation.scope("updates-0200-seed-17", self.bundle, 200, 17, 300.)

    def panel(self):
        records = []
        for split, prefix, variants, extractors in (("train", "t", 3, 2), ("development", "v", 4, 3)):
            for person in (f"{prefix}1", f"{prefix}2"):
                for window in ("first", "second"):
                    for variant in range(variants):
                        for extractor in range(extractors):
                            records.append(dict(split=split, canonical_person_id=person,
                                                window_id=f"{person}-{window}", variant=variant, extractor=extractor))
        return SimpleNamespace(records=records, evidence_status="automated-source-screen", validate=lambda held: None)

    def test_panel_verification_requires_actual_people_and_exact_record_counts(self):
        request = dict(held_extractor="vitpose", pilot_bundle=str(self.bundle), output=str(self.output),
                       train_people=2, development_people=2, windows_per_person=2)
        old = SimpleNamespace(records=[dict(canonical_person_id="old-person")])
        panel = self.panel()
        with patch.object(FULL.TrackBundle, "load", side_effect=[panel, old]):
            FULL.verify_panel(self.root / "new-bundle", request)
        evidence = FULL.read(self.output / "panel-verification.json")
        self.assertEqual(evidence["independent_development_people"], 2)
        self.assertEqual(evidence["physical_windows"], 8)
        self.assertEqual(evidence["track_records"], 72)
        panel.records.pop()
        with patch.object(FULL.TrackBundle, "load", side_effect=[panel, old]), self.assertRaisesRegex(ValueError, "counts"):
            FULL.verify_panel(self.root / "new-bundle", request)

    def test_panel_refuses_old_people_and_wrong_evidence_label(self):
        request = dict(held_extractor="vitpose", pilot_bundle=str(self.bundle), output=str(self.output),
                       train_people=2, development_people=2, windows_per_person=2)
        for name, old in (("overlap", SimpleNamespace(records=[dict(canonical_person_id="v1")])),
                          ("label", SimpleNamespace(records=[dict(canonical_person_id="old-person")]))):
            with self.subTest(case=name):
                panel = self.panel()
                if name == "label":
                    panel.evidence_status = "fixture-tested"
                with patch.object(FULL.TrackBundle, "load", side_effect=[panel, old]), self.assertRaises(ValueError):
                    FULL.verify_panel(self.root / "new-bundle", request)

    def test_status_needs_successful_scheduler_exit_in_addition_to_worker_complete(self):
        write(self.output / "request.json", dict(prior_entries=[dict(gpu_seconds=100.)]))
        write(self.output / "submission.json", dict(job_id="12345", status="submitted"))
        write(self.output / "status.json", dict(status="worker_complete"))
        args = SimpleNamespace(pilot_work=self.pilot, name="full-01")
        cases = [(None, "worker_complete"),
                 (dict(state="RUNNING", exit_code="0:0", gpu_seconds=500.), "worker_complete"),
                 (dict(state="COMPLETED", exit_code="0:0", gpu_seconds=500.), "FULL_EXPERIMENT_COMPLETE"),
                 (dict(state="COMPLETED", exit_code="1:0", gpu_seconds=500.), "allocation_failed"),
                 (dict(state="TIMEOUT", exit_code="0:0", gpu_seconds=500.), "allocation_failed")]
        for scheduler, expected in cases:
            with self.subTest(scheduler=scheduler), contextlib.redirect_stdout(io.StringIO()):
                snapshot = {} if scheduler is None else {"12345": scheduler}
                with patch.object(FULL.haic, "scheduler_snapshot", return_value=snapshot):
                    result = FULL.status(args)
                self.assertEqual(result["status"], expected)
                if scheduler and scheduler["state"] in FULL.haic.TERMINAL:
                    self.assertAlmostEqual(result["total_recorded_gpu_hours"], 600 / 3600)

    def test_worker_refuses_login_node_execution(self):
        request = dict(output=str(self.output), python=sys.executable)
        write(self.output / "request.json", request)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(PermissionError, "Slurm"):
            FULL.run_request(self.output / "request.json", FULL.digest(request))

    def test_worker_rejects_changed_request_before_starting(self):
        request = dict(output=str(self.output), python=sys.executable)
        write(self.output / "request.json", request)
        expected = FULL.digest(request)
        write(self.output / "request.json", dict(request, allocation_seconds=999999))
        with patch.dict(os.environ, {"SLURM_JOB_ID": "777"}), self.assertRaisesRegex(ValueError, "Request changed"):
            FULL.run_request(self.output / "request.json", expected)
        self.assertFalse((self.output / "status.json").exists())


if __name__ == "__main__":
    unittest.main()
