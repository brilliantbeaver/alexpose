"""Unit tests for the full-GAVD YouTube manifest/downloader helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from gavd6_sjepa.data_foundations import gavd_video_download_pipeline as download_gavd_full


class DownloadGavdFullTests(unittest.TestCase):
    def write_manifest(self, directory: Path) -> Path:
        path = directory / "gavd_full_sequences.csv"
        pd.DataFrame(
            [
                {"video_id": "abcdefghijk", "url": "https://youtu.be/abcdefghijk", "last_frame": 20, "source_height": 720},
                {"video_id": "abcdefghijk", "url": "https://www.youtube.com/watch?v=abcdefghijk", "last_frame": 42, "source_height": 720},
                {"video_id": "lmnopqrstuv", "url": "https://youtu.be/lmnopqrstuv", "last_frame": 11, "source_height": 480},
            ]
        ).to_csv(path, index=False)
        return path

    def test_load_unique_videos_uses_global_video_id_and_max_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            videos = download_gavd_full.load_unique_videos(self.write_manifest(Path(temporary)))
        self.assertEqual(videos["video_id"].tolist(), ["abcdefghijk", "lmnopqrstuv"])
        first = videos.iloc[0]
        self.assertEqual(first.required_last_frame, 42)
        self.assertEqual(first.sequences, 2)

    def test_sharding_is_deterministic_and_nonoverlapping(self) -> None:
        videos = pd.DataFrame(
            {
                "video_id": ["a", "b", "c", "d", "e"],
                "url": ["u"] * 5,
                "required_last_frame": [1] * 5,
                "source_height": [720] * 5,
                "sequences": [1] * 5,
            }
        )
        first = download_gavd_full.select_videos(
            videos, video_ids=None, shard_count=2, shard_index=0, max_videos=None
        )
        second = download_gavd_full.select_videos(
            videos, video_ids=None, shard_count=2, shard_index=1, max_videos=None
        )
        self.assertEqual(first["video_id"].tolist(), ["a", "c", "e"])
        self.assertEqual(second["video_id"].tolist(), ["b", "d"])

    def test_invalid_youtube_id_is_rejected_before_file_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_manifest(Path(temporary))
            table = pd.read_csv(path)
            table.loc[0, "video_id"] = "../../not-a-video"
            table.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "invalid YouTube IDs"):
                download_gavd_full.load_unique_videos(path)

    def test_format_selection_preserves_notebook_policy(self) -> None:
        selector = download_gavd_full.video_format_selector(480)
        self.assertIn("bestvideo[ext=mp4][vcodec^=avc1][height<=480][fps<=30]", selector)
        self.assertEqual(
            download_gavd_full.video_format_selector(None, hls_only=True),
            "best[protocol^=m3u8][height<=720][fps<=30]",
        )

    def test_cache_path_ignores_partial_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "all"
            folder.mkdir()
            (folder / "abcdefghijk.mp4.part").touch()
            self.assertEqual(
                download_gavd_full.cached_video_path(Path(temporary), "abcdefghijk"),
                folder / "abcdefghijk.mp4",
            )


if __name__ == "__main__":
    unittest.main()
