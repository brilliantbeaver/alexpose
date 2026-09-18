"""Released OpenMMLab metadata remains loadable with Torch's restricted loader."""

from collections import OrderedDict
import importlib.util
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gavd6_sjepa.research_directions.synthetic_training.estimators import _load_pose_checkpoint

try:
    from mmengine.logging import HistoryBuffer
except ImportError:
    HistoryBuffer = None


class _UnexpectedMetadata:
    """A class absent from the inspected official checkpoint metadata."""


@unittest.skipIf(HistoryBuffer is None, "MMEngine is required for checkpoint compatibility checks")
class PoseCheckpointLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # MMEngine imports Torch distributed helpers that register their own
        # types; measure our call's scope after that one-time initialization.
        import mmengine.runner  # noqa: F401

    def save_checkpoint(self, payload, path, zip_format=True):
        # The released HRNet checkpoints store only the HistoryBuffer instance
        # dictionary. Newer MMEngine __getstate__ additionally serializes
        # callback functions; those are intentionally outside this allowlist.
        with patch.object(HistoryBuffer, "__getstate__", lambda history: history.__dict__):
            torch.save(payload, path, _use_new_zipfile_serialization=zip_format)

    def checkpoint(self, model):
        return {
            "state_dict": OrderedDict(("module." + key, value.clone())
                                      for key, value in model.state_dict().items()),
            "meta": {"dataset_meta": {
                "keypoint_colors": np.array([[51, 153, 255]], dtype=np.uint8),
                "sigmas": np.array([0.026], dtype=np.float32),
            }},
            # HRNet releases include message_hub metric histories in addition
            # to the NumPy arrays shared by RTMPose and ViTPose checkpoints.
            "message_hub": {"loss": HistoryBuffer(
                log_history=np.array([0.5], dtype=np.float64),
                count_history=np.array([1], dtype=np.int64)),
                "best_score": np.float64(0.8)},
        }

    def test_legacy_and_zip_metadata_load_without_global_allowlist_changes(self):
        original = torch.nn.Linear(2, 3)
        before = torch.serialization.get_safe_globals()
        with tempfile.TemporaryDirectory() as directory:
            for zip_format in (False, True):
                with self.subTest(zip_format=zip_format):
                    path = Path(directory) / "pose.pth"
                    self.save_checkpoint(self.checkpoint(original), path, zip_format)
                    with self.assertRaises(pickle.UnpicklingError):
                        torch.load(path, map_location="cpu", weights_only=True)
                    restored = torch.nn.Linear(2, 3)
                    payload = _load_pose_checkpoint(restored, path)
                    for key, value in original.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
                    self.assertEqual(payload["meta"]["dataset_meta"]["keypoint_colors"].dtype, np.uint8)
                    self.assertEqual(payload["message_hub"]["loss"].current(), 0.5)
                    self.assertEqual(payload["message_hub"]["best_score"], 0.8)
                    self.assertCountEqual(torch.serialization.get_safe_globals(), before)
                    with self.assertRaises(pickle.UnpicklingError):
                        torch.load(path, map_location="cpu", weights_only=True)

    def test_unexpected_metadata_stays_restricted_and_scope_is_cleaned_up(self):
        before = torch.serialization.get_safe_globals()
        model = torch.nn.Linear(2, 3)
        payload = self.checkpoint(model)
        payload["unexpected"] = _UnexpectedMetadata()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unexpected.pth"
            self.save_checkpoint(payload, path)
            with self.assertRaises(pickle.UnpicklingError):
                _load_pose_checkpoint(model, path)
        self.assertCountEqual(torch.serialization.get_safe_globals(), before)

    def test_missing_weights_remain_a_strict_failure(self):
        model = torch.nn.Linear(2, 3)
        payload = self.checkpoint(model)
        del payload["state_dict"]["module.bias"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.pth"
            self.save_checkpoint(payload, path)
            with self.assertRaisesRegex(RuntimeError, "bias"):
                _load_pose_checkpoint(model, path)


@unittest.skipUnless(importlib.util.find_spec("mmpretrain") is not None,
                     "MMPreTrain is required for the real ViT checkpoint checks")
class ViTPoseCheckpointLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mmpretrain.models.backbones import VisionTransformer
        cls.backbone_type = VisionTransformer

    def model(self, with_cls_token=False):
        model = torch.nn.Module()
        model.backbone = self.backbone_type(
            arch=dict(embed_dims=8, num_layers=1, num_heads=2, feedforward_channels=16),
            img_size=16, patch_size=8, with_cls_token=with_cls_token, out_type="featmap")
        model.head = torch.nn.Conv2d(8, 3, 1)
        return model.eval()

    def test_unused_legacy_token_loads_without_changing_predictions_or_file(self):
        from mmengine.runner import load_checkpoint
        original = self.model()
        inputs = torch.randn(2, 3, 16, 16)
        state = original.state_dict()
        state["backbone.cls_token"] = torch.randn(1, 1, 8)
        # MMPreTrain already handles the legacy extra position-embedding row;
        # the token compatibility hook must preserve that behavior.
        state["backbone.pos_embed"] = torch.cat(
            [torch.randn(1, 1, 8), state["backbone.pos_embed"]], dim=1)
        # Exercise MMEngine's normal module-prefix removal too.
        payload = {"state_dict": OrderedDict(("module." + k, v) for k, v in state.items())}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-vitpose.pth"
            torch.save(payload, path)
            before = path.read_bytes()
            restored = self.model()
            hook_count = len(restored.backbone._load_state_dict_pre_hooks)
            with self.assertRaisesRegex(RuntimeError, "backbone.cls_token"):
                load_checkpoint(restored, str(path), strict=True)
            loaded = _load_pose_checkpoint(restored, path)
            self.assertIn("module.backbone.cls_token", loaded["state_dict"])
            self.assertEqual(path.read_bytes(), before)
            self.assertIsNone(restored.backbone.cls_token)
            self.assertEqual(len(restored.backbone._load_state_dict_pre_hooks), hook_count)
            with torch.inference_mode():
                expected = original.head(original.backbone(inputs)[0])
                actual = restored.head(restored.backbone(inputs)[0])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with self.assertRaisesRegex(RuntimeError, "backbone.cls_token"):
                load_checkpoint(restored, str(path), strict=True)

    def test_other_mismatches_still_fail_and_temporary_hook_is_removed(self):
        for mismatch in ("missing_head", "unexpected_backbone", "wrong_shape"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                model = self.model()
                state = model.state_dict()
                state["backbone.cls_token"] = torch.randn(1, 1, 8)
                if mismatch == "missing_head":
                    del state["head.bias"]
                    error_key = "head.bias"
                elif mismatch == "unexpected_backbone":
                    state["backbone.unexpected"] = torch.zeros(1)
                    error_key = "backbone.unexpected"
                else:
                    state["head.weight"] = torch.zeros(1)
                    error_key = "head.weight"
                path = Path(directory) / "invalid.pth"
                torch.save({"state_dict": state}, path)
                hook_count = len(model.backbone._load_state_dict_pre_hooks)
                with self.assertRaisesRegex(RuntimeError, error_key):
                    _load_pose_checkpoint(model, path)
                self.assertEqual(len(model.backbone._load_state_dict_pre_hooks), hook_count)

    def test_enabled_class_token_is_loaded_and_required(self):
        original = self.model(with_cls_token=True)
        with torch.no_grad():
            original.backbone.cls_token.fill_(0.375)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token-enabled.pth"
            torch.save({"state_dict": original.state_dict()}, path)
            restored = self.model(with_cls_token=True)
            _load_pose_checkpoint(restored, path)
            torch.testing.assert_close(restored.backbone.cls_token, original.backbone.cls_token)
            state = original.state_dict()
            del state["backbone.cls_token"]
            torch.save({"state_dict": state}, path)
            with self.assertRaisesRegex(RuntimeError, "backbone.cls_token"):
                _load_pose_checkpoint(restored, path)


if __name__ == "__main__":
    unittest.main()
