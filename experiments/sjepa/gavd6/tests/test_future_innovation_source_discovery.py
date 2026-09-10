"""Adversarial cache discovery: nested files, explicit paths, ambiguity and gaps."""

from pathlib import Path
import tempfile
import unittest
import os

import pandas as pd

from gavd6_sjepa.research_directions.future_innovation.fi_source_inventory import discover_sources
from gavd6_sjepa.research_directions.future_innovation.fi_cohort import build_candidates, deterministic_window_start
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import initialize_run


class FutureInnovationSourceDiscoveryTests(unittest.TestCase):
    def test_nested_cache_and_manifest_path_and_missing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            nested = root / "cache/condition"
            nested.mkdir(parents=True)
            (nested / "source-a.AVI").write_bytes(b"video")
            explicit = root / "original title.mp4"
            explicit.write_bytes(b"video")
            table = pd.DataFrame({"video_id": ["source-a", "source-b", "source-c"],
                                  "video_path": [None, "original title.mp4", None]})
            found = discover_sources(table, root / "videos.csv", root / "cache")
            self.assertEqual(found["source-a"]["path"], (nested / "source-a.AVI").resolve())
            self.assertEqual(found["source-b"]["path"], explicit.resolve())
            self.assertIn("not found", found["source-c"]["error"])

    def test_ambiguous_ids_are_not_silently_assigned_and_symlinks_deduplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            a, b = root / "a", root / "b"
            a.mkdir(); b.mkdir()
            (a / "source.mp4").write_bytes(b"original")
            (b / "source.mp4").symlink_to(a / "source.mp4")
            os.link(a / "source.mp4", b / "source.mkv")
            table = pd.DataFrame({"video_id": ["source"]})
            self.assertIsNotNone(discover_sources(table, root / "videos.csv", root)["source"]["path"])
            (b / "source.webm").write_bytes(b"different export")
            found = discover_sources(table, root / "videos.csv", root)
            self.assertIn("Ambiguous", found["source"]["error"])
            table["video_path"] = str(a / "source.mp4")
            self.assertEqual(discover_sources(table, root / "videos.csv", root)["source"]["path"], a / "source.mp4")

    def test_additional_roots_use_exact_ids_and_ignore_partial_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cache, additional = root / "cache", root / "external"
            cache.mkdir(); additional.mkdir()
            (additional / "abc.mp4").write_bytes(b"video")
            (additional / "abc-longer.mp4").write_bytes(b"wrong source")
            (cache / "abc.mp4.part").write_bytes(b"partial")
            table = pd.DataFrame({"video_id": ["abc", "abc-long"]})
            found = discover_sources(table, root / "manifest.csv", cache, [additional])
            self.assertEqual(found["abc"]["path"], additional / "abc.mp4")
            self.assertIsNone(found["abc-long"]["path"])

    def fixture(self, root, *, gap=False, short=False):
        cache = root / "storage"
        (cache / "all").mkdir(parents=True)
        source_ids = [f"video-{i}" for i in range(25)]
        for video_id in source_ids:
            (cache / "all" / f"{video_id}.mp4").write_bytes(b"source fixture; discovery only")
        sequences, frames = [], []
        missing_frame = deterministic_window_start(0, 159, "sequence-0") + 33
        for i in range(50):
            length = 160 if gap and i == 0 else 32 if short and i == 0 else 64
            sequences.append({"sequence_id": f"sequence-{i}", "video_id": source_ids[i // 2],
                              "first_frame": 1, "last_frame": length})
            for frame in range(1, length + 1):
                if gap and i == 0 and frame == missing_frame:
                    continue
                frames.append({"seq": f"sequence-{i}", "frame_num": frame, "id": source_ids[i // 2],
                               "bbox": str({"left": 10, "top": 10, "width": 50, "height": 80}),
                               "vid_info": str({"width": 100, "height": 100})})
        sequence_path, video_path, annotation_path = (root / name for name in ("sequences.csv", "videos.csv", "annotations.csv"))
        pd.DataFrame(sequences).to_csv(sequence_path, index=False)
        pd.DataFrame({"video_id": source_ids}).to_csv(video_path, index=False)
        pd.DataFrame(frames).to_csv(annotation_path, index=False)
        model = root / "model"
        model.write_text("fixture")
        run = root / "run"
        initialize_run(run, sequence_manifest=sequence_path, video_manifest=video_path,
                       annotations=[annotation_path], pose_model=model, vjepa_root=root,
                       checkpoint=model, synthetic=True, youtube_dir=cache)
        return (run, sequence_path, video_path, [annotation_path], cache), missing_frame

    def test_gap_does_not_discard_another_complete_annotation_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, missing = self.fixture(Path(temporary).resolve(), gap=True)
            build_candidates(*args)
            candidates = pd.read_csv(args[0] / "manifests/candidates.csv")
            row = candidates.set_index("sequence_id").loc["sequence-0"]
            self.assertEqual(len(candidates), 50)
            self.assertFalse(row.source_first_frame + 1 <= missing <= row.source_last_frame + 1)
            self.assertEqual(row.source_last_frame - row.source_first_frame, 63)

    def test_incomplete_discovery_can_retry_after_cached_source_arrives(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = self.fixture(Path(temporary).resolve())
            path = args[-1] / "all/video-0.mp4"
            path.rename(path.with_suffix(".part"))
            with self.assertRaisesRegex(ValueError, "Only 48 candidates"):
                build_candidates(*args)
            self.assertFalse((args[0] / "config/candidates-contract.json").exists())
            path.with_suffix(".part").rename(path)
            build_candidates(*args)
            self.assertEqual(len(pd.read_csv(args[0] / "manifests/candidates.csv")), 50)

    def test_short_annotated_sequence_is_not_padded_to_force_a_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = self.fixture(Path(temporary).resolve(), short=True)
            with self.assertRaisesRegex(ValueError, "Only 49 candidates"):
                build_candidates(*args)
            exclusions = pd.read_csv(args[0] / "manifests/exclusions.csv")
            self.assertEqual(exclusions.reason.tolist(), ["Sequence shorter than 64 frames"])
            self.assertTrue(pd.read_csv(args[0] / "manifests/source-availability.csv").available.all())


if __name__ == "__main__":
    unittest.main()
