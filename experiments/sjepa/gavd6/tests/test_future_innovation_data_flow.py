"""Run real CSV/video/pose/cache interfaces with a lightweight injected detector/teacher."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.future_innovation.fi_cohort import (
    build_candidates,
    extract_and_freeze,
    load_cohort,
)
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import (
    initialize_run, read_json,
)
from gavd6_sjepa.research_directions.future_innovation.fi_feature_cache import (
    cache_teacher,
    load_cache,
)
from gavd6_sjepa.research_directions.future_innovation.fi_validity_audits import (
    require_audits,
    run_audits,
)
from gavd6_sjepa.research_directions.future_innovation.fi_video_pose import (
    normalize_skeleton,
)


class FutureInnovationDataFlowTests(unittest.TestCase):
    def test_manifest_decode_pose_freeze_and_cache_interfaces(self):
        self._data_flow("legacy-v1", "central")

    def test_direct_cache_and_readiness_with_no_background_pixels(self):
        self._data_flow("direct-v2", "full")

    def test_direct_cache_and_readiness_with_no_shared_background_flow(self):
        self._data_flow("direct-v2", "alternating")

    def _data_flow(self, protocol, box_pattern):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            youtube = directory / "youtube"
            (youtube / "all").mkdir(parents=True)
            video = directory / "source.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 64)
            )
            self.assertTrue(writer.isOpened())
            for t in range(64):
                writer.write(np.full((64, 64, 3), 50 + t, dtype=np.uint8))
            writer.release()
            sources = [f"source-{i:04d}" for i in range(25)]
            for source in sources:
                os.link(video, youtube / "all" / f"{source}.avi")
            # Existing cache resolver supports .mp4/.mkv/.webm/.mov/.m4v. The
            # container is detected by OpenCV; name the MJPEG fixtures .mp4.
            for path in (youtube / "all").glob("*.avi"):
                path.rename(path.with_suffix(".mp4"))
            sequence_rows, annotation_rows = [], []
            box = {"left": 25.6, "top": 12.8, "width": 12.8, "height": 38.4}
            info = {"width": 64, "height": 64}
            for i in range(50):
                source = sources[i // 2]
                sequence_rows.append(
                    {
                        "sequence_id": f"seq-{i}",
                        "video_id": source,
                        "first_frame": 1,
                        "last_frame": 64,
                        "cam_view": "front",
                        "gait_pattern_annotation": "NOT_USED",
                    }
                )
                for frame in range(1, 65):
                    if box_pattern == "full":
                        box = {"left": 0, "top": 0, "width": 64, "height": 64}
                    elif box_pattern == "alternating":
                        box = {"left": 0 if frame % 2 else 31, "top": 0, "width": 33, "height": 64}
                    annotation_rows.append(
                        {
                            "seq": f"seq-{i}",
                            "frame_num": frame,
                            "bbox": str(box),
                            "vid_info": str(info),
                            "id": source,
                        }
                    )
            sequences = directory / "sequences.csv"
            videos = directory / "videos.csv"
            annotations = directory / "annotations.csv"
            model = directory / "fake-model.task"
            model.write_text("synthetic model fixture")
            pd.DataFrame(sequence_rows).to_csv(sequences, index=False)
            pd.DataFrame({"video_id": sources}).to_csv(videos, index=False)
            pd.DataFrame(annotation_rows).to_csv(annotations, index=False)
            root = directory / "run"
            initialize_run(
                root,
                protocol=protocol,
                sequence_manifest=sequences,
                video_manifest=videos,
                annotations=[annotations],
                pose_model=model,
                vjepa_root=directory,
                checkpoint=model,
                change_reason="synthetic data interfaces",
                synthetic=True,
                youtube_dir=youtube,
            )
            build_candidates(root, sequences, videos, [annotations], youtube)

            def detector(video, rows, boxes, fps, model_path):
                self.assertEqual([r["frame_num"] for r in rows], list(range(1, 65)))
                raw = np.full((32, 33, 4), 0.5, dtype=np.float32)
                raw[..., 3] = 0.9
                raw[:, 11:13, 1] = 0.3
                history, scale = normalize_skeleton(raw, boxes[:32])
                return raw, history, scale

            with patch(
                "gavd6_sjepa.research_directions.future_innovation.fi_cohort.extract_history",
                side_effect=detector,
            ):
                extract_and_freeze(root)
            cohort = load_cohort(root, verify_artifacts=True)
            self.assertEqual(len(cohort), 50)
            self.assertEqual(cohort.source_first_frame.unique().tolist(), [0])
            frozen = (root / "config/run-contract.json").read_bytes()
            build_candidates(root, sequences, videos, [annotations], youtube)
            with patch(
                "gavd6_sjepa.research_directions.future_innovation.fi_cohort.extract_history",
                side_effect=AssertionError("must reuse frozen poses"),
            ):
                extract_and_freeze(root)
            self.assertEqual((root / "config/run-contract.json").read_bytes(), frozen)
            class Teacher:
                def verify_geometry(self, video):
                    return 0.0

                def encode_past_context(self, video):
                    # Small memory footprint: broadcast one vector over tokens.
                    return np.broadcast_to(
                        np.arange(768, dtype=np.float32), (16 * 576, 768)
                    )

                def encode_full_target(self, video):
                    return np.broadcast_to(
                        np.arange(768, dtype=np.float32), (32 * 576, 768)
                    )

            cache_teacher(root, Teacher())
            loaded, arrays = load_cache(root)
            self.assertEqual(loaded.window_id.tolist(), cohort.window_id.tolist())
            self.assertEqual(arrays["person"].shape, (50, 256))
            self.assertEqual(arrays["skeleton"].shape, (50, 32, 33, 4))
            self.assertGreater(arrays["baseline"].shape[1], 3 * 768)
            # Completed cache is reused without teacher inference.
            with patch.object(
                Teacher,
                "encode_full_target",
                side_effect=AssertionError("must reuse cache"),
            ), patch(
                "gavd6_sjepa.research_directions.future_innovation.fi_feature_cache.fixed_projection",
                side_effect=AssertionError("must reuse saved projection across runtimes"),
            ):
                cache_teacher(root, Teacher())
            # A constant fake target fails the variance check, after completing
            # either protocol's actual teacher calls and retained measurements.
            if protocol == "legacy-v1":
                self.assertFalse(run_audits(root, Teacher()))
                self.assertEqual(len(pd.read_csv(root / "qc/target-sensitivity.csv")), 10)
                self.assertEqual(len(list(root.glob("qc/pixel-edit-contact-sheets/*.jpg"))), 10)
            else:
                from gavd6_sjepa.research_directions.future_innovation.fi_readiness import run_readiness
                self.assertFalse(run_readiness(root, Teacher()))
                summary = read_json(root / "qc/readiness-summary.json")
                self.assertTrue(summary["checks"]["teacher_stable"])
                self.assertTrue(summary["checks"]["causal_leakage_absent"])
                self.assertFalse(summary["checks"]["target_variance_valid"])
                self.assertFalse((root / "qc/target-sensitivity.csv").exists())
                schema = read_json(root / "config/nuisance-schema.json")
                nuisance = arrays["baseline"][:, schema["context_embedding_columns"]:]
                columns = schema["columns"]
                self.assertTrue(np.isfinite(nuisance).all())
                for name in ("background_flow_pair_fraction", "background_flow_pixel_fraction"):
                    np.testing.assert_array_equal(nuisance[:, columns.index(name)], 0)
                rgb_support = nuisance[:, columns.index("background_rgb_pixel_fraction")]
                if box_pattern == "full":
                    np.testing.assert_array_equal(rgb_support, 0)
                else:
                    self.assertTrue(np.all(rgb_support > 0))
            with self.assertRaisesRegex(ValueError, "Validity audit failed"):
                require_audits(root)
            with (root / "manifests/candidates.csv").open("a") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "Candidate order changed"):
                build_candidates(root, sequences, videos, [annotations], youtube)
            with Path(cohort.iloc[0].pose_path).open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "Cohort artifact changed"):
                extract_and_freeze(root)
