"""Check report identity binding without media, network, or formatting snapshots.

Run: python3 -m unittest discover -s scripts -p test_document_gavd_sequence_media.py
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import document_gavd_sequence_media as documentation


class DocumentationInputIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        root_patch = patch.object(documentation, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.manifest_path = self.root / "manifest.json"
        self.verification_path = self.root / "verification.json"
        sequence = dict(sequence_id="sequence", gait_pat="normal", video_id="abcdefghijk",
                        start_frame=4, end_frame=7, expected_frames=4,
                        annotation_sha256="same-annotation-hash", status="completed", bytes=123,
                        output_path="outputs/normal/abcdefghijk-4-7.mp4",
                        probe=dict(decoded_frames=4, width=64, height=48))
        source = dict(gait_pat="normal", video_id="abcdefghijk", status="available", bytes=999,
                      path="sources/normal/abcdefghijk.mp4",
                      probe=dict(decoded_frames=12, width=64, height=48))
        self.manifest = dict(completed_at="2026-11-15T00:00:00+00:00",
                             categories=list(documentation.CATEGORIES),
                             sequences=[sequence], sources=[source])
        verified_sequence = {key: sequence[key] for key in (
            "sequence_id", "gait_pat", "video_id", "start_frame", "end_frame", "expected_frames", "bytes")}
        verified_sequence.update(status="verified", csv_sha256=sequence["annotation_sha256"],
                                 decoded_frames=4, width=64, height=48,
                                 output=str(self.root / sequence["output_path"]))
        verified_source = dict(gait_pat="normal", video_id="abcdefghijk", status="verified_decode", bytes=999,
                               decoded_frames=12, width=64, height=48,
                               path=str(self.root / source["path"]))
        self.verification = dict(finished_at="2026-11-15T00:01:00+00:00",
                                 categories=list(documentation.CATEGORIES),
                                 sequences=[verified_sequence], sources=[verified_source])

    def load(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.verification_path.write_text(json.dumps(self.verification))
        return documentation.load_inputs(self.manifest_path, self.verification_path)

    def test_matching_artifacts_accept_relative_preparation_and_absolute_verified_paths(self):
        manifest, verification, checks = self.load()
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(verification, self.verification)
        self.assertEqual(checks["sequence"]["status"], "verified")

    def test_identical_annotation_identity_does_not_allow_another_output_root(self):
        self.verification["sequences"][0]["output"] = str(
            self.root / "another-output-root/normal/abcdefghijk-4-7.mp4")
        with self.assertRaisesRegex(ValueError, "different output file"):
            self.load()

    def test_identical_annotation_identity_does_not_allow_another_source_root(self):
        self.verification["sources"][0]["path"] = str(
            self.root / "another-source-root/normal/abcdefghijk.mp4")
        with self.assertRaisesRegex(ValueError, "different source file"):
            self.load()


if __name__ == "__main__":
    unittest.main()
