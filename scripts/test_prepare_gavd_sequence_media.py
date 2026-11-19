"""Adversarial local fixtures for acquisition/extraction; never contact YouTube.

Run: python3 -m unittest discover -s scripts -p test_prepare_gavd_sequence_media.py
"""

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import prepare_gavd_sequence_media as media


class AcquisitionRecoveryTests(unittest.TestCase):
    """Mock downloaders and decoders so recovery needs no external programs."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        root_patch = patch.object(media, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.sources = self.root / "sources"
        self.folder = self.sources / "normal"
        self.folder.mkdir(parents=True)
        self.canonical = self.folder / "abcdefghijk.mp4"
        self.staged = self.folder / ".acquiring/abcdefghijk/abcdefghijk.mp4"
        info = dict(height=48, width=64, mime_type="video/mp4")
        self.source = dict(source_key="normal/abcdefghijk", gait_pat="normal", video_id="abcdefghijk",
                           url="https://www.youtube.com/watch?v=abcdefghijk", annotation_video_info=info,
                           annotation_video_infos=[info], required_last_frame=2,
                           sequence_ids=["sequence"], status="pending")
        self.metadata = dict(decoded_frames=12, width=64, height=48, pix_fmt="yuv420p")
        self.downloader = MagicMock()
        self.downloader.__enter__.return_value = self.downloader
        youtube = ModuleType("yt_dlp")
        youtube.__path__ = []
        youtube.YoutubeDL = MagicMock(return_value=self.downloader)
        version = ModuleType("yt_dlp.version")
        version.__version__ = "fixture-version"
        youtube.version = version
        shared = ModuleType("ambient.utils.youtube_cache")
        shared.YtDlpConfig = SimpleNamespace
        shared._build_output_template = lambda folder, scheme: str(folder / "%(id)s.%(ext)s")
        shared._build_ydl_options = lambda template, config: dict(
            outtmpl=template, format=config.format_selector, extractor_args=config.extractor_args)
        modules_patch = patch.dict(sys.modules, {
            "ambient": ModuleType("ambient"), "ambient.utils": ModuleType("ambient.utils"),
            "ambient.utils.youtube_cache": shared, "yt_dlp": youtube, "yt_dlp.version": version,
        })
        modules_patch.start()
        self.addCleanup(modules_patch.stop)
        runtime_patch = patch.object(media.shutil, "which", return_value=None)
        runtime_patch.start()
        self.addCleanup(runtime_patch.stop)

    def downloaded_info(self):
        return dict(id="abcdefghijk", ext="mp4", format_id="fixture-format", fps=12, height=48, width=64)

    def test_corrupt_canonical_source_is_preserved_before_fresh_download(self):
        self.canonical.write_bytes(b"corrupt source")

        def decode(path):
            if path.read_bytes() == b"corrupt source":
                raise RuntimeError("synthetic decode corruption")
            return self.metadata.copy()

        def download(url, download):
            self.assertFalse(self.canonical.exists())
            self.staged.write_bytes(b"recovered source")
            return self.downloaded_info()

        self.downloader.extract_info.side_effect = download
        with patch.object(media, "probe", side_effect=decode):
            result = media.acquire(self.source, self.sources)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["acquisition"], "downloaded")
        self.assertEqual(self.canonical.read_bytes(), b"recovered source")
        self.assertEqual([attempt["strategy"] for attempt in result["attempts"]], ["cached_validation", "default"])
        preserved = self.root / result["attempts"][0]["preserved_path"]
        self.assertEqual(preserved.parent.resolve(), (self.folder / ".invalid").resolve())
        self.assertEqual(preserved.read_bytes(), b"corrupt source")
        self.assertEqual(result["sha256"], media.sha256(self.canonical))
        self.downloader.extract_info.assert_called_once_with(self.source["url"], download=True)

    def test_invalid_staged_download_is_quarantined_before_next_strategy(self):
        calls = []
        quarantined = self.staged.parent / ".invalid/default-abcdefghijk.mp4"

        def download(url, download):
            calls.append(url)
            if len(calls) == 1:
                self.staged.write_bytes(b"invalid first strategy")
            else:
                self.assertFalse(self.staged.exists())
                self.assertEqual(quarantined.read_bytes(), b"invalid first strategy")
                self.staged.write_bytes(b"valid second strategy")
            return self.downloaded_info()

        def decode(path):
            if path.read_bytes() == b"invalid first strategy":
                raise RuntimeError("synthetic invalid staged source")
            return self.metadata.copy()

        self.downloader.extract_info.side_effect = download
        with patch.object(media, "probe", side_effect=decode):
            result = media.acquire(self.source, self.sources)
        self.assertEqual(result["status"], "available")
        self.assertEqual(self.canonical.read_bytes(), b"valid second strategy")
        self.assertEqual(quarantined.read_bytes(), b"invalid first strategy")
        self.assertEqual([(row["strategy"], row["status"]) for row in result["attempts"]],
                         [("default", "failed"), ("embedded", "downloaded")])
        self.assertIn("synthetic invalid staged source", result["attempts"][0]["error"])
        self.assertEqual(self.downloader.extract_info.call_count, 2)

    def test_pending_manifest_recovers_available_source_sidecar_acquisition_evidence(self):
        annotations = self.root / "annotations"
        outputs = self.root / "outputs"
        csv_path = annotations / "normal/sequence.csv"
        csv_path.parent.mkdir(parents=True)
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(media.HEADER)
            for frame in (1, 2):
                writer.writerow(["sequence", frame, "", "", "Normal Gait", "normal", "{}",
                                 repr(self.source["annotation_video_info"]), "abcdefghijk", self.source["url"]])
        self.canonical.write_bytes(b"valid source")
        available = {**self.source, "status": "available", "path": media.relative(self.canonical),
                     "bytes": self.canonical.stat().st_size, "sha256": media.sha256(self.canonical),
                     "probe": self.metadata, "downloaded_at": "2026-11-15T00:00:00+00:00",
                     "selected_format": self.downloaded_info(), "attempts": [{"strategy": "original-download"}]}
        media.save_json(self.folder / ".provenance/abcdefghijk.json", available)
        media.save_json(outputs / "manifest.json", {"sources": [self.source], "sequences": []})
        reused = {key: value for key, value in available.items()
                  if key in ("status", "path", "bytes", "sha256", "probe")}
        reused.update(acquisition="reused_canonical", attempts=[])

        def extracted(sequence, source, old_sequences):
            return {**sequence, "status": "completed", "bytes": 1}

        arguments = ["prepare_gavd_sequence_media.py", "--annotation-root", str(annotations),
                     "--source-root", str(self.sources), "--output-root", str(outputs), "--workers", "1"]
        with patch.object(media, "CATEGORIES", ("normal",)), patch.object(sys, "argv", arguments), \
                patch.object(sys, "path", sys.path[:]), patch.object(media, "acquire", return_value=reused), \
                patch.object(media, "extract", side_effect=extracted), \
                patch.object(media.subprocess, "check_output", return_value="ffmpeg fixture\n"), \
                patch("builtins.print"):
            self.assertEqual(media.main(), 0)
        resumed = json.loads((outputs / "manifest.json").read_text())["sources"][0]
        for field in ("downloaded_at", "selected_format", "attempts"):
            self.assertEqual(resumed[field], available[field])
        self.downloader.extract_info.assert_not_called()


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
class MediaPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.root_patch = patch.object(media, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.source_path = self.root / "sources/normal/abcdefghijk.mp4"
        self.source_path.parent.mkdir(parents=True)
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            "testsrc2=size=64x48:rate=12", "-frames:v", "12", "-c:v", "libx264",
            "-crf", "0", "-pix_fmt", "yuv420p", str(self.source_path),
        ], capture_output=True, check=True)
        self.source = dict(path="sources/normal/abcdefghijk.mp4", sha256=media.sha256(self.source_path),
                           probe=media.probe(self.source_path), status="available")

    def sequence(self, start=4, end=7):
        return dict(sequence_id="sequence", gait_pat="normal", video_id="abcdefghijk",
                    annotation_sha256="fixture-annotation", start_frame=start, end_frame=end,
                    expected_frames=end - start + 1,
                    output_path=f"outputs/normal/abcdefghijk-{start}-{end}.mp4", status="pending")

    def pixels(self, path):
        raw = subprocess.check_output([
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an",
            "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-f", "rawvideo", "-",
        ])
        frame_bytes = 64 * 48 * 3 // 2
        self.assertEqual(len(raw) % frame_bytes, 0)
        return [raw[i:i + frame_bytes] for i in range(0, len(raw), frame_bytes)]

    def test_middle_range_has_both_exact_endpoints_and_every_interior_frame(self):
        seq = self.sequence()
        result = media.extract(seq, self.source, {})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["probe"]["decoded_frames"], 4)
        source_frames = self.pixels(self.source_path)
        self.assertEqual(self.pixels(self.root / seq["output_path"]), source_frames[3:7])
        self.assertNotEqual(source_frames[2], source_frames[3])
        self.assertNotEqual(source_frames[6], source_frames[7])

    def test_single_first_and_last_frame_ranges_are_inclusive(self):
        frames = self.pixels(self.source_path)
        for number in (1, 12):
            with self.subTest(frame=number):
                seq = self.sequence(number, number)
                result = media.extract(seq, self.source, {})
                self.assertEqual(result["probe"]["decoded_frames"], 1)
                self.assertEqual(self.pixels(self.root / seq["output_path"]), [frames[number - 1]])

    def test_short_source_is_rejected_without_publishing_output(self):
        seq = self.sequence(11, 13)
        with self.assertRaisesRegex(RuntimeError, "requires 13"):
            media.extract(seq, self.source, {})
        self.assertFalse((self.root / seq["output_path"]).exists())

    def test_unchanged_completed_output_is_reused_without_reencoding(self):
        seq = self.sequence()
        first = media.extract(seq, self.source, {})
        path = self.root / seq["output_path"]
        old_mtime = path.stat().st_mtime_ns
        with patch.object(media.subprocess, "run", side_effect=AssertionError("Unexpected reencoding")):
            again = media.extract(seq, self.source, {seq["sequence_id"]: first})
        self.assertEqual(again["processing"], "reused_verified_hash")
        self.assertEqual(again["sha256"], first["sha256"])
        self.assertEqual(path.stat().st_mtime_ns, old_mtime)

    def test_clip_sidecar_recovers_completed_output_after_missing_manifest_checkpoint(self):
        seq = self.sequence()
        first = media.extract(seq, self.source, {})
        path = self.root / seq["output_path"]
        sidecar = path.parent / ".provenance/sequence.json"
        self.assertEqual(json.loads(sidecar.read_text())["sha256"], first["sha256"])
        for checkpoint in ({}, {seq["sequence_id"]: seq}):
            with self.subTest(checkpoint=bool(checkpoint)):
                with patch.object(media.subprocess, "run", side_effect=AssertionError("Unexpected reencoding")):
                    recovered = media.extract(seq, self.source, checkpoint)
                self.assertEqual(recovered["processing"], "reused_verified_hash")
                self.assertEqual(recovered["sha256"], first["sha256"])

    def test_interruption_between_sidecar_and_output_publication_is_recoverable(self):
        seq = self.sequence()
        path = self.root / seq["output_path"]
        original_replace = Path.replace

        def interrupt_publication(current, target):
            if current.name.endswith(".partial.mp4"):
                raise RuntimeError("synthetic interruption before publication")
            return original_replace(current, target)

        with patch.object(Path, "replace", interrupt_publication):
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                media.extract(seq, self.source, {})
        self.assertFalse(path.exists())
        self.assertTrue((path.parent / ".provenance/sequence.json").exists())
        recovered = media.extract(seq, self.source, {})
        self.assertEqual(recovered["status"], "completed")
        self.assertEqual(self.pixels(path), self.pixels(self.source_path)[3:7])

    def test_chapter_bearing_source_produces_clean_video_only_output(self):
        chapters = self.root / "chapters.ffmetadata"
        chapters.write_text(";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=500\ntitle=First\n"
                            "[CHAPTER]\nTIMEBASE=1/1000\nSTART=500\nEND=1000\ntitle=Second\n")
        remuxed = self.root / "source-with-chapters.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-i", str(self.source_path), "-i", str(chapters),
            "-map", "0:v:0", "-map_metadata", "1", "-map_chapters", "1", "-c:v", "copy", str(remuxed),
        ], capture_output=True, check=True)
        remuxed.replace(self.source_path)
        self.source.update(sha256=media.sha256(self.source_path), probe=media.probe(self.source_path))
        source_chapters = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(self.source_path),
        ]))
        self.assertEqual(len(source_chapters["chapters"]), 2)
        seq = self.sequence()
        result = media.extract(seq, self.source, {})
        self.assertEqual(result["status"], "completed")
        output = self.root / seq["output_path"]
        inspected = subprocess.run([
            "ffprobe", "-v", "error", "-show_chapters", "-show_streams", "-of", "json", str(output),
        ], capture_output=True, text=True, check=True)
        self.assertEqual(inspected.stderr, "")
        metadata = json.loads(inspected.stdout)
        self.assertEqual(metadata["chapters"], [])
        self.assertEqual([stream["codec_type"] for stream in metadata["streams"]], ["video"])
        self.assertEqual(self.pixels(output), self.pixels(self.source_path)[3:7])

    def test_existing_output_with_changed_annotation_is_preserved_and_rejected(self):
        seq = self.sequence()
        first = media.extract(seq, self.source, {})
        changed = {**seq, "annotation_sha256": "changed"}
        with self.assertRaisesRegex(RuntimeError, "matching provenance"):
            media.extract(changed, self.source, {seq["sequence_id"]: first})
        self.assertEqual(media.sha256(self.root / seq["output_path"]), first["sha256"])

    def test_existing_output_with_changed_source_is_preserved_and_rejected(self):
        seq = self.sequence()
        first = media.extract(seq, self.source, {})
        with self.assertRaisesRegex(RuntimeError, "matching provenance"):
            media.extract(seq, {**self.source, "sha256": "changed"}, {seq["sequence_id"]: first})
        self.assertEqual(media.sha256(self.root / seq["output_path"]), first["sha256"])

    def test_corrupted_completed_output_is_not_reused(self):
        seq = self.sequence()
        first = media.extract(seq, self.source, {})
        path = self.root / seq["output_path"]
        path.write_bytes(b"damaged clip")
        with self.assertRaisesRegex(RuntimeError, "matching provenance"):
            media.extract(seq, self.source, {seq["sequence_id"]: first})
        self.assertEqual(path.read_bytes(), b"damaged clip")

    def test_encoder_failure_cannot_publish_a_completed_clip(self):
        seq = self.sequence()
        failed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="synthetic encoder failure")
        with patch.object(media.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "synthetic encoder failure"):
                media.extract(seq, self.source, {})
        self.assertFalse((self.root / seq["output_path"]).exists())

    def test_unavailable_source_blocks_every_associated_sequence(self):
        video_info = dict(height=48, width=64, mime_type="video/mp4")
        src = dict(source_key="normal/abcdefghijk", gait_pat="normal", video_id="abcdefghijk",
                   url="https://www.youtube.com/watch?v=abcdefghijk", annotation_video_info=video_info,
                   annotation_video_infos=[video_info], required_last_frame=7,
                   sequence_ids=["sequence"], status="pending")
        unavailable = dict(status="unavailable", error="private", attempts=[])
        with patch.object(media, "acquire", return_value=unavailable), patch.object(media, "extract") as extract:
            record, sequences = media.process_source(src, [self.sequence()], self.root / "sources", {}, {})
        extract.assert_not_called()
        self.assertEqual(record["status"], "unavailable")
        self.assertEqual(sequences[0]["status"], "blocked_source")
        self.assertEqual(sequences[0]["expected_frames"], 4)

    def write_annotation(self, frames=(4, 5, 6, 7), sequence_id="sequence", height=48, width=64):
        path = self.root / "annotations/normal" / f"{sequence_id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(media.HEADER)
            for frame in frames:
                writer.writerow([sequence_id, frame, "right side", "", "Normal Gait", "normal", "{}",
                                 repr(dict(height=height, width=width, mime_type="video/mp4")),
                                 "abcdefghijk", "https://www.youtube.com/watch?v=abcdefghijk"])

    def test_source_keeps_multiple_annotation_dimension_declarations(self):
        self.write_annotation(frames=(1, 2), sequence_id="first", height=48, width=64)
        self.write_annotation(frames=(4, 5), sequence_id="second", height=96, width=128)
        with patch.object(media, "CATEGORIES", ("normal",)):
            sequences, sources = media.inventory(self.root / "annotations", self.root / "outputs")
        self.assertEqual(len(sequences), 2)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["required_last_frame"], 5)
        self.assertEqual([seq["annotation_video_info"]["height"] for seq in sequences], [48, 96])
        self.assertEqual(sources[0]["annotation_video_info"]["height"], 96)
        self.assertEqual(len(sources[0]["annotation_video_infos"]), 2)

    def test_inventory_rejects_frame_gaps_and_output_name_collisions(self):
        with patch.object(media, "CATEGORIES", ("normal",)):
            self.write_annotation(frames=(4, 6, 7))
            with self.assertRaisesRegex(ValueError, "contiguous"):
                media.inventory(self.root / "annotations", self.root / "outputs")
            self.write_annotation()
            self.write_annotation(sequence_id="other")
            with self.assertRaisesRegex(ValueError, "collision"):
                media.inventory(self.root / "annotations", self.root / "outputs")


if __name__ == "__main__":
    unittest.main()
