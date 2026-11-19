"""Small adversarial media fixtures; no network or real data mutations.

Run: python3 -m unittest discover -s scripts -p test_verify_gavd_sequence_media.py
"""

import csv
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from verify_gavd_sequence_media import HEADER, compare_decoded, decode_hashes, discover_sequences, verify


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
class MediaVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.annotations = self.root / "annotations"
        self.sources = self.root / "sources"
        self.outputs = self.root / "outputs"
        for directory in (self.annotations, self.sources, self.outputs):
            (directory / "normal").mkdir(parents=True)
        self.source = self.sources / "normal/abcdefghijk.mp4"
        self.output = self.outputs / "normal/abcdefghijk-4-7.mp4"
        self.run_ffmpeg("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=12", "-frames:v", "12",
                        "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", str(self.source))
        self.write_csv()

    def tearDown(self):
        self.temporary.cleanup()

    def run_ffmpeg(self, *args):
        subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True, check=True)

    def write_csv(self, frames=(4, 5, 6, 7), name="sequence"):
        path = self.annotations / "normal" / f"{name}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADER)
            for frame in frames:
                writer.writerow([name, frame, "right side", "", "Normal Gait", "normal", "{}",
                                 "{'height': 48, 'width': 64, 'mime_type': 'video/mp4'}",
                                 "abcdefghijk", "https://www.youtube.com/watch?v=abcdefghijk"])

    def make_clip(self, start=3, end=7):
        self.run_ffmpeg("-i", str(self.source), "-vf",
                        f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS",
                        "-an", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p",
                        "-fps_mode", "passthrough", str(self.output))

    def test_inclusive_slice_is_verified_pixel_for_pixel(self):
        self.make_clip()
        report = verify(self.annotations, self.sources, self.outputs, ("normal",), jobs=1)
        self.assertTrue(report["complete"])
        self.assertEqual(report["summary"]["verified_frames"], 4)
        self.assertEqual(report["sequences"][0]["decoded_frames"], 4)

    def test_same_count_but_one_frame_shift_is_rejected(self):
        self.make_clip(start=4, end=8)
        source, output = decode_hashes(self.source), decode_hashes(self.output)
        issues = compare_decoded(source, output, 4, 7)
        self.assertTrue(any("Pixel hashes differ" in issue for issue in issues))
        self.assertFalse(any("frame count" in issue for issue in issues))

    def test_omitted_last_inclusive_frame_is_rejected(self):
        self.make_clip(end=6)
        issues = compare_decoded(decode_hashes(self.source), decode_hashes(self.output), 4, 7)
        self.assertTrue(any("frame count" in issue for issue in issues))

    def test_changed_playback_speed_is_rejected_even_when_pixels_match(self):
        self.run_ffmpeg("-i", str(self.source), "-vf",
                        "trim=start_frame=3:end_frame=7,setpts=(PTS-STARTPTS)*2",
                        "-an", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p",
                        "-fps_mode", "passthrough", str(self.output))
        issues = compare_decoded(decode_hashes(self.source), decode_hashes(self.output), 4, 7)
        self.assertTrue(any("frame times differ" in issue for issue in issues))
        self.assertFalse(any("Pixel hashes differ" in issue for issue in issues))

    def test_clip_without_source_cannot_pass_alignment(self):
        self.make_clip()
        self.source.unlink()
        report = verify(self.annotations, self.sources, self.outputs, ("normal",), jobs=1)
        self.assertFalse(report["complete"])
        self.assertEqual(report["summary"]["source_statuses"], {"missing_source": 1})
        self.assertEqual(report["summary"]["sequence_statuses"], {"invalid_output": 1})

    def test_csv_gaps_and_duplicate_output_mappings_are_rejected(self):
        self.write_csv(frames=(4, 6, 7))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            discover_sequences(self.annotations, ("normal",))
        self.write_csv()
        self.write_csv(name="another_sequence")
        with self.assertRaisesRegex(ValueError, "same output"):
            discover_sequences(self.annotations, ("normal",))

    def test_missing_and_unexpected_outputs_make_report_incomplete(self):
        shutil.copyfile(self.source, self.outputs / "normal/unexpected.mp4")
        report = verify(self.annotations, self.sources, self.outputs, ("normal",), jobs=1)
        self.assertFalse(report["complete"])
        self.assertEqual(report["summary"]["sequence_statuses"], {"missing_output": 1})
        self.assertEqual(report["summary"]["unexpected_outputs"], 1)


if __name__ == "__main__":
    unittest.main()
