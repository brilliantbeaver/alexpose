"""Validate asset setup boundaries without downloading research data."""
import hashlib
import importlib.util
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/research_directions/synthetic_training/prepare_render_assets.py"
SPEC = importlib.util.spec_from_file_location("st_render_asset_setup", SCRIPT)
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


class AssetSetupTests(unittest.TestCase):
    def test_backgrounds_exclude_every_annotated_image_and_missing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i in range(5):
                (root / f"{i}.jpg").touch()
            payload = {"categories": [{"id": 1, "name": "person"}],
                       "images": [{"id": i, "file_name": f"{i}.jpg"} for i in range(6)],
                       "annotations": [{"image_id": 0, "keypoints": [0] * 51},
                                       {"image_id": 1, "iscrowd": 1, "keypoints": [0] * 51}]}
            selected = setup.background_candidates(payload, root, 3, 17)
            self.assertEqual({item["id"] for item in selected}, {2, 3, 4})
            self.assertEqual(selected, setup.background_candidates(payload, root, 3, 17))
            with self.assertRaisesRegex(ValueError, "found 3"):
                setup.background_candidates(payload, root, 4, 17)

    def test_wrong_coco_annotation_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "person_keypoints"):
            setup.background_candidates({"categories": [{"id": 2, "name": "car"}]}, Path("/tmp"), 1, 17)

    def test_download_publishes_valid_png_and_preserves_existing_file(self):
        data = BytesIO()
        Image.new("RGB", (64, 64), "red").save(data, format="PNG")
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "texture.png"
            with patch.object(setup, "urlopen", return_value=BytesIO(data.getvalue())) as request:
                self.assertTrue(setup.download("https://example.invalid/texture.png", target))
                self.assertFalse(setup.download("https://example.invalid/texture.png", target))
                request.assert_called_once()
            self.assertEqual(target.read_bytes(), data.getvalue())
            self.assertFalse(target.with_name(target.name + ".part").exists())

    def test_bad_download_never_becomes_an_available_asset(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "texture.png"
            with patch.object(setup, "urlopen", side_effect=lambda *a, **k: BytesIO(b"not a PNG")), patch.object(setup.time, "sleep"):
                with self.assertRaises(Exception):
                    setup.download("https://example.invalid/texture.png", target)
            self.assertFalse(target.exists())

    def test_existing_uv_with_wrong_hash_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "smpl_uv.obj"
            target.write_bytes(b"modified")
            with patch.object(setup, "urlopen") as request:
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    setup.download("https://example.invalid/uv.obj", target,
                                   expected_hash=hashlib.sha256(b"expected").hexdigest())
                request.assert_not_called()
            self.assertEqual(target.read_bytes(), b"modified")


if __name__ == "__main__":
    unittest.main()
