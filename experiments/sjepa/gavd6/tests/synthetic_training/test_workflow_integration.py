"""CPU integration fixtures, never scientific pose-estimation results.

Only the external pretrained estimator is replaced. Dataset loading, measured
errors, source trials, selector fitting, deployment, annotation import and
recording-level evaluation all execute their actual study implementations.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
import joblib

from gavd6_sjepa.research_directions.synthetic_training.config import RunConfig
from gavd6_sjepa.research_directions.synthetic_training.data import PoseFrameDataset
from gavd6_sjepa.research_directions.synthetic_training.measurements import paired_recording_interval
from gavd6_sjepa.research_directions.synthetic_training.rendering import KEYPOINT_NAMES
from gavd6_sjepa.research_directions.synthetic_training.selectors import feature_vector
from gavd6_sjepa.research_directions.synthetic_training import trials, workflow


class TinyTrainableEstimator:
    """An explicitly non-scientific linear RGB estimator for workflow testing."""

    def __init__(self, spec, device="cpu"):
        self.spec = spec
        self.head = torch.nn.Linear(3, 24)
        with torch.no_grad():
            self.head.weight.fill_(0.05)
            self.head.bias.fill_(6 + (sum(map(ord, spec.student_id)) % 3))

    @staticmethod
    def inputs(images):
        return torch.tensor(np.array([image.mean(axis=(0, 1)) / 255 for image in images]), dtype=torch.float32)

    def predict(self, images, *, boxes=None, batch_size=32):
        with torch.no_grad():
            return self.head(self.inputs(images)).numpy().reshape(-1, 12, 2)

    def make_optimizer(self, lr, weight_decay=0):
        return torch.optim.SGD(self.head.parameters(), lr=lr, weight_decay=weight_decay)

    def train_batch(self, images, keypoints, visible, optimizer, *, boxes=None):
        pred = self.head(self.inputs(images)).reshape(-1, 12, 2)
        target = torch.from_numpy(np.nan_to_num(keypoints).astype(np.float32))
        mask = torch.from_numpy(visible).unsqueeze(-1)
        loss = ((pred - target).square() * mask).sum() / (2 * mask.sum())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def head_state(self):
        return {name: value.detach().clone() for name, value in self.head.state_dict().items()}

    def load_head_state(self, state):
        self.head.load_state_dict(state)

    def save_head(self, path):
        torch.save(self.head_state(), path)

    def load_head(self, path):
        self.load_head_state(torch.load(path, weights_only=True))


class StudyFixture:
    def __init__(self, root):
        self.root = root
        (root / "data").mkdir()
        self.targets = {}
        self.serial = 0
        self.cfg = RunConfig(
            run_root=str(root), device="cpu", context_kind="simple", context_frames=4,
            probe_steps=1, adaptation_steps=[1, 2], train_batch_size=4,
            synthetic_fraction=0.5, learning_rate=0.1, predict_batch_size=4,
            selector_neighbors=[1], selector_ridge=[10.0], bootstrap_samples=30,
            gavd_annotations_csv=str(root / "gavd-{split}-annotations.csv"),
            students=[dict(student_id=name, family=family, role=role, config="unused.py", checkpoint="unused.pt")
                      for name, family, role in (("source_a", "family_a", "train"),
                                               ("validation_b", "family_b", "validation"),
                                               ("held_c", "family_c", "held"))],
        )
        self.cfg.validate()

    def frame(self, role, *, domain="support", lesson="", recording=None, frame=0, labeled=True):
        self.serial += 1
        uid = f"fixture_{self.serial}"
        clip = recording or f"{role}_{domain}_{lesson}"
        path = self.root / "data" / f"{uid}.png"
        rgb = np.full((24, 32, 3), [30 + self.serial % 80, 50, 120], np.uint8)
        Image.fromarray(rgb).save(path)
        points = np.tile([10 + self.serial % 3, 12 + self.serial % 2], (12, 1)).astype(np.float32)
        visible = np.ones(12, bool)
        visible[-1] = False
        points[-1] = np.nan
        label_path = self.root / "data" / f"{uid}.npz"
        if labeled:
            np.savez_compressed(label_path, keypoints=points, visible=visible)
        self.targets[uid] = (points, visible)
        return dict(frame_id=uid, clip_id=clip, frame_index=frame, image_path=str(path),
                    label_path=str(label_path) if labeled else "", label_index=0,
                    bbox_x1=1, bbox_y1=1, bbox_x2=31, bbox_y2=23,
                    width=32, height=24, role=role, domain_id=domain, lesson_id=lesson,
                    person_id=f"person_{role}_{domain}", motion_id=clip, recording_id=clip)

    def prepare(self):
        rows = [self.frame("probe", frame=i) for i in range(2)]
        for role in ("lesson", "diagnostic"):
            for lesson in ("front", "low_resolution"):
                rows.extend(self.frame(role, lesson=lesson, frame=i) for i in range(2))
        for split in ("train", "validation"):
            for role in ("context", "reference"):
                for domain in ("front_high", "side_low"):
                    rows.extend(self.frame(f"{split}_{role}", domain=domain, frame=i) for i in range(2))
        pd.DataFrame(rows).to_csv(self.root / "data/synthetic.csv", index=False)
        pd.DataFrame([self.frame("replay", frame=i) for i in range(3)]).to_csv(self.root / "data/replay.csv", index=False)
        context, references = [], []
        for domain in ("front_high", "side_low"):
            context.extend(self.frame("context", domain=domain, frame=i, labeled=False) for i in range(2))
            for role, count in (("early", 2), ("confirmation", 1)):
                for recording in range(count):
                    references.extend(self.frame(role, domain=domain, recording=f"real_{role}_{domain}_{recording}",
                                                 frame=i, labeled=False) for i in range(2))
        self.reference_index = pd.DataFrame(references)
        pd.DataFrame(context).to_csv(self.root / "data/gavd_context.csv", index=False)
        self.reference_index.to_csv(self.root / "data/gavd_evaluation.csv", index=False)

    def annotate(self, split):
        rows = []
        for row in self.reference_index.loc[self.reference_index.role.eq(split)].itertuples():
            points, visible = self.targets[row.frame_id]
            for j, landmark in enumerate(KEYPOINT_NAMES):
                rows.append(dict(frame_id=row.frame_id, landmark=landmark, x=points[j, 0], y=points[j, 1],
                                 visible=visible[j], box_x1=0, box_y1=0, box_x2=30, box_y2=20,
                                 annotator="human A", reviewer="human B" if split == "confirmation" else ""))
        # Human row order intentionally differs from image-manifest order.
        pd.DataFrame(rows).sample(frac=1, random_state=42).to_csv(
            self.root / f"gavd-{split}-annotations.csv", index=False)


class WorkflowIntegrationTests(unittest.TestCase):
    def test_actual_trials_fit_deploy_crossover_and_annotation_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = StudyFixture(Path(directory))
            fixture.prepare()
            cfg = fixture.cfg
            loader = lambda spec, device: TinyTrainableEstimator(spec, device)
            with patch.object(trials, "load_estimator", side_effect=loader), \
                    patch.object(workflow, "load_estimator", side_effect=loader), \
                    patch.object(workflow, "import_gavd_annotations", side_effect=AssertionError("Target-label leakage")):
                source = workflow.source_trials(cfg)
                self.assertEqual(set(source.student_id), {"source_a", "validation_b"})
                self.assertTrue(np.isfinite(source.error).all())
                fitted = workflow.fit_selectors(cfg)
                self.assertIn(fitted["selection"]["budget"], [1, 2])
                self.assertIn("source_progress", fitted["selection"]["methods"])
                frozen = joblib.load(cfg.root / "selectors/frozen.joblib")
                self.assertEqual(frozen["models"]["full"].kind, frozen["models"]["source_progress_matched"].kind)
                self.assertEqual(frozen["models"]["full"].parameter, frozen["models"]["source_progress_matched"].parameter)
                reported = workflow.report(cfg)
                self.assertTrue(Path(reported["report"]).is_file())
                choices = workflow.deploy(cfg)
                self.assertEqual(choices.student_id.nunique(), 3)
                self.assertFalse((cfg.root / "gavd-early-annotations.csv").exists())
                exchange = workflow.crossover(cfg)
                self.assertEqual(len(exchange), 3 * 2 * 2)
                # Exercise crossover's missing-branch path, not only its table export.
                one = exchange.iloc[0]
                missing_path = cfg.root / "deployment" / one.student_id / f"predictions_{one.other_action}.npz"
                missing_path.unlink()
                workflow.crossover(cfg, student_id=one.student_id)
                self.assertTrue(missing_path.is_file())
                with self.assertRaisesRegex(RuntimeError, "frozen"):
                    workflow.source_trials(cfg, student_id="source_a")
            # Reordered caches must align by frame ID. A failed visible prediction
            # stays in the accuracy denominator and receives the declared penalty.
            prediction_path = cfg.root / "deployment/held_c/predictions_original.npz"
            saved = trials.read_npz(prediction_path)
            saved["predictions"][0, 0] = np.nan
            np.savez_compressed(prediction_path, frame_ids=saved["frame_ids"][::-1],
                                predictions=saved["predictions"][::-1])
            fixture.annotate("early")
            held_choices = cfg.root / "deployment/held_c/choices.csv"
            pending = held_choices.with_name("choices_not_ready.csv")
            held_choices.rename(pending)
            try:
                with self.assertRaisesRegex(FileNotFoundError, "configured roster"):
                    workflow.evaluate(cfg, split="early")
                self.assertFalse((cfg.root / "evaluation/early/summary.csv").exists())
            finally:
                pending.rename(held_choices)
            evaluated = workflow.evaluate(cfg, split="early")
            self.assertEqual(evaluated["summary"].student_id.nunique(), 3)
            self.assertTrue(evaluated["summary"].n_recordings.eq(4).all())
            self.assertTrue(evaluated["intervals"].n_recordings.eq(4).all())
            self.assertTrue(np.isfinite(evaluated["summary"].nle).all())
            failures = evaluated["recordings"].loc[
                lambda table: table.student_id.eq("held_c") & table.method.eq("original"), "failed_joints"]
            self.assertEqual(failures.sum(), 1)
            frame_scores = pd.read_csv(cfg.root / "evaluation/early/frame_scores.csv")
            frame_id = str(saved["frame_ids"][0])
            keypoints, visible = fixture.targets[frame_id]
            expected = np.linalg.norm(saved["predictions"][0] - keypoints, axis=-1) / np.sqrt(30**2 + 20**2)
            expected[0] = cfg.missing_prediction_penalty
            actual = frame_scores.loc[frame_scores.student_id.eq("held_c") & frame_scores.method.eq("original")
                                      & frame_scores.frame_id.eq(frame_id), "nle"].iloc[0]
            self.assertAlmostEqual(actual, float(expected[visible].mean()), places=6)
            crossed = pd.read_csv(cfg.root / "evaluation/early/crossover_recording_scores.csv")
            self.assertEqual(len(crossed), 3 * 2 * 4)
            np.testing.assert_allclose(crossed.own_advantage, crossed.other_error - crossed.own_error)
            identical = crossed.loc[crossed.identical_choice]
            self.assertTrue(identical.own_advantage.eq(0).all())
            # Confirmation coordinates are absent and were never needed for early scoring.
            self.assertFalse((cfg.root / "gavd-confirmation-annotations.csv").exists())
            fixture.annotate("confirmation")
            confirmed = workflow.evaluate(cfg, split="confirmation")
            self.assertTrue(confirmed["summary"].n_recordings.eq(2).all())
            self.assertTrue(confirmed["intervals"].n_domains.eq(2).all())

    def test_source_progress_excludes_all_target_before_change_information(self):
        base = {key: np.arange(3, dtype=np.float32) for key in
                ("context", "target_post", "target_pre", "target_delta", "diagnostic_pre",
                 "diagnostic_post", "loss_history", "descriptors", "budget")}
        changed = {**base, "target_pre": np.full(3, 900, np.float32),
                   "target_delta": np.full(3, -700, np.float32)}
        np.testing.assert_array_equal(feature_vector(base, "source_progress"),
                                      feature_vector(changed, "source_progress"))
        self.assertFalse(np.array_equal(feature_vector(base, "full"), feature_vector(changed, "full")))

    def test_bootstrap_is_invariant_to_repeating_frames_in_a_recording(self):
        scores = pd.DataFrame([
            dict(recording_id="a", domain_id="one", method="full", nle=0.1),
            dict(recording_id="a", domain_id="one", method="baseline", nle=0.2),
            dict(recording_id="b", domain_id="one", method="full", nle=0.5),
            dict(recording_id="b", domain_id="one", method="baseline", nle=0.3),
        ])
        expanded = pd.concat([scores, *[scores.loc[scores.recording_id.eq("a")]] * 20], ignore_index=True)
        one = paired_recording_interval(scores, "full", "baseline", samples=100, seed=1)
        many = paired_recording_interval(expanded, "full", "baseline", samples=100, seed=1)
        self.assertEqual(one, many)


if __name__ == "__main__":
    unittest.main()
