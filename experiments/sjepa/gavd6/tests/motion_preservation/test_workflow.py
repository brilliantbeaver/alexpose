"""motion preservation / test workflow."""


from pathlib import Path
from dataclasses import replace
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.motion_preservation import (
    preservation_metrics as metrics,
    repair_models as learning,
    workflow,
)
from gavd6_sjepa.research_directions.motion_preservation.body_geometry import (
    demo_motion,
    estimate_bone_lengths,
)
from gavd6_sjepa.research_directions.motion_preservation.config import RunConfig

# Research boundaries across preparation, fitting and held-out evaluation.


class ResearchBoundaryTests(unittest.TestCase):
    def test_explicit_missing_configuration_does_not_start_another_experiment(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(os.environ, {"MP_CONFIG": str(Path(root) / "missing.json")}, clear=True):
                with self.assertRaisesRegex(FileNotFoundError, "MP_CONFIG"):
                    RunConfig.from_env()

    def test_momask_requires_its_trained_sample_rate(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = RunConfig(run_root=root, mode="real", fps=30.)
            with self.assertRaisesRegex(ValueError, "20 FPS"):
                workflow.cache_predictions(cfg)

    def test_final_is_not_constructed_before_calibration(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=RunConfig(run_root=root,mode="demo",device="cpu")
            with self.assertRaisesRegex(RuntimeError,"training and calibration"):
                workflow.build_pairs(cfg,roles=("final",))
            self.assertFalse((Path(root)/"cases.csv").exists())

    def test_test_people_cannot_choose_operating_points(self):
        with self.assertRaisesRegex(ValueError,"calibration people"):
            metrics.lock_operating_points(pd.DataFrame({"role":["final"]}))

    def test_conversion_and_prior_diagnostics_are_not_projected(self):
        body,_=demo_motion(frames=16)
        raw=body.joints
        distorted=raw.copy(); distorted[:,10,1]+=.1
        case=dict(raw=raw,prior=distorted,bridge=distorted,confidence=np.ones(raw.shape[:2]),
                  observed=np.ones(raw.shape[:2]),raw_transport=np.ones(raw.shape[:2]),
                  prior_transport=np.ones(raw.shape[:2]),flow_valid=np.ones(raw.shape[:2]),
                  bone_lengths=estimate_bone_lengths(raw))
        predicted=learning.baseline_candidates(case)
        np.testing.assert_array_equal(predicted["prior_unprojected"],distorted)
        np.testing.assert_array_equal(predicted["conversion_only_unprojected"],distorted)
        self.assertGreater(np.max(abs(predicted["prior"]-distorted)),.01)

    def test_complete_demo_keeps_final_and_ambiguous_inputs_honest(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=RunConfig(run_root=root,mode="demo",device="cpu",prior_id="demo_smoothing",
                          prior_backend="demo_smoothing",flow_backend="demo_farneback",
                          max_motions_per_role=2,duration_s=.8,image_size=48,epochs=1,
                          seeds=[17],feature_modes=["full","coordinates"],bootstrap_samples=30)
            workflow.build_pairs(cfg)
            workflow.cache_predictions(cfg)
            index,cases=workflow._load_cases(cfg,"train")
            for person in index.person_id.unique():
                ids=index.index[index.person_id.eq(person) & index.fixture.eq("ambiguous")]
                self.assertEqual(len(ids),2)
                a,b=(cases[i] for i in ids)
                np.testing.assert_array_equal(a["features"],b["features"])
                self.assertFalse(np.array_equal(a["truth"],b["truth"]))
            workflow.train_gate(cfg)
            workflow.calibrate(cfg)
            training=(Path(root)/"models/training.json").read_bytes()
            locked=(Path(root)/"calibration/locked.json").read_bytes()
            result=workflow.evaluate(cfg)
            self.assertEqual(result["decision"]["status"],"demo_only_no_research_decision")
            self.assertEqual(result["decision"]["noise_removal_scope"], "observed_joints")
            legacy = json.loads(locked)
            legacy.pop("noise_removal_scope")
            (Path(root)/"calibration/locked.json").write_text(json.dumps(legacy))
            with self.assertRaisesRegex(RuntimeError, "previous all-joint repair metric"):
                workflow.require_fitted(cfg)
            (Path(root)/"calibration/locked.json").write_bytes(locked)
            # Good average repair cannot hide a comparator that misses the
            # target on the cases where event preservation actually matters.
            summary = result["summary"].copy()
            primary = result["decision"]["primary"]
            comparator = result["decision"]["calibration_selected_comparator"]
            summary.loc[summary.method.isin([primary, comparator]), "noise_removal"] = .30
            summary.loc[summary.method.eq(primary), "event_and_noise_removal"] = .25
            summary.loc[summary.method.eq(comparator), "event_and_noise_removal"] = .20
            with patch.object(metrics, "summarize", return_value=summary):
                mixed = workflow.evaluate(cfg)["decision"]
            self.assertTrue(mixed["minimum_repair_met"])
            self.assertFalse(mixed["same_clip_event_and_noise_repair"])
            with self.assertRaisesRegex(ValueError, "Demo results"):
                workflow.evaluate(replace(cfg, mode="real"))
            with self.assertRaisesRegex(ValueError, "Demo cases"):
                workflow._load_cases(replace(cfg, mode="real"), "development")
            self.assertNotIn("final",pd.read_csv(Path(root)/"cases.csv").role.unique())
            workflow.build_pairs(cfg,roles=("final",))
            workflow.cache_predictions(cfg,roles=("final",))
            final=workflow.evaluate(cfg,split="final")
            self.assertEqual(final["decision"]["mode"],"demo")
            self.assertEqual(training,(Path(root)/"models/training.json").read_bytes())
            self.assertEqual(locked,(Path(root)/"calibration/locked.json").read_bytes())
            with self.assertRaisesRegex(RuntimeError,"final evaluation"):
                workflow.train_gate(cfg)


if __name__ == "__main__":
    unittest.main()
