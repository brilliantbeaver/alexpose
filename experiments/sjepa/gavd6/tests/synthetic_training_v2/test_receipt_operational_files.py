"""Slurm bookkeeping may change after immutable stage evidence is recorded."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import atomic_json
from gavd6_sjepa.research_directions.synthetic_training_v2.workflow import _receipt, run_stage


REPO = Path(__file__).resolve().parents[2]
WORKFLOW = "gavd6_sjepa.research_directions.synthetic_training_v2.workflow"


def write_stage_evidence(cfg, stage, repo, identity, *, resume=False):
    # Keep the real run initialization, stage locking, receipt creation and
    # dependency validation; isolate this regression from scientific fitting.
    atomic_json(cfg.root / "scientific" / f"{stage}.json", {"stage": stage})
    if stage == "audit":
        atomic_json(cfg.root / "scientific/submission.json", {"evidence": "preserved"})
        atomic_json(cfg.root / "scientific/logs/diagnostics.json", {"evidence": "preserved"})
        atomic_json(cfg.root / "costs/attempt.json", {"gpu_seconds": 1.0})
    return {"stage": stage, "status": "completed"}


class OperationalReceiptTests(unittest.TestCase):
    def test_slurm_log_and_submission_updates_allow_next_stage_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = RunConfig.fixture("slurm-receipt", folder)
            log = cfg.root / "logs/stv2-stage-101.out"
            log.parent.mkdir(parents=True)
            log.write_text("job started\n")
            submission = cfg.root / "submission.json"
            atomic_json(submission, [{"stage": "audit", "job_id": "101"}])

            with patch(f"{WORKFLOW}._stage", side_effect=write_stage_evidence) as execute:
                run_stage(cfg, "audit", REPO)
                # run.py prints after run_stage has saved the receipt. The
                # launcher can also still be appending newly submitted jobs.
                with log.open("a") as stream:
                    stream.write("audit completed\n")
                atomic_json(submission, [
                    {"stage": "audit", "job_id": "101"},
                    {"stage": "data", "job_id": "102"},
                ])
                result = run_stage(cfg, "data", REPO)
                self.assertEqual(result["status"], "completed")

                log.write_text("rotated scheduler log\n")
                atomic_json(submission, [{"stage": "report", "job_id": "110"}])
                self.assertEqual(run_stage(cfg, "data", REPO, resume=True), result)
                self.assertEqual(execute.call_count, 2)

            for stage in ("audit", "data"):
                files = _receipt(cfg, stage)["files"]
                self.assertNotIn("submission.json", files)
                self.assertFalse(any(Path(name).parts[0] == "logs" for name in files))

    def test_scientific_artifact_mutations_still_reject_dependent_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = RunConfig.fixture("scientific-receipt", folder)
            with patch(f"{WORKFLOW}._stage", side_effect=write_stage_evidence) as execute:
                run_stage(cfg, "audit", REPO)
                run_stage(cfg, "data", REPO)
                protected = (
                    "scientific/audit.json",
                    "scientific/submission.json",
                    "scientific/logs/diagnostics.json",
                    "costs/attempt.json",
                    "effective-config.json",
                )
                receipt = _receipt(cfg, "audit")
                for name in protected:
                    with self.subTest(artifact=name):
                        self.assertIn(name, receipt["files"])
                        path = cfg.root / name
                        original = path.read_bytes()
                        path.write_text(json.dumps({"changed": True}))
                        with self.assertRaisesRegex(ValueError, "Stale/corrupt stage artifact"):
                            run_stage(cfg, "data", REPO, resume=True)
                        path.write_bytes(original)
                self.assertEqual(execute.call_count, 2)
                _receipt(cfg, "data")


if __name__ == "__main__":
    unittest.main()
