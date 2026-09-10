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
    initialize_run,
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
            ):
                cache_teacher(root, Teacher())
            # The same complete audit pipeline rejects a constant fake target
            # before training and preserves all ten intervention reports/sheets.
            self.assertFalse(run_audits(root, Teacher()))
            sensitivity = pd.read_csv(root / "qc/target-sensitivity.csv")
            self.assertEqual(len(sensitivity), 10)
            self.assertEqual(
                len(list(root.glob("qc/pixel-edit-contact-sheets/*.jpg"))), 10
            )
            with self.assertRaisesRegex(ValueError, "Validity audit failed"):
                require_audits(root)
