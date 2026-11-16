"""Independent input replay. Does not read fitted models or outcome reports."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from worldmodels.restart_data import strict_gate, supported_window, window_identity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    prepared = PROJECT / "artifacts/runs/restart-full10hz-v1"
    report = json.loads((prepared / "preparation.json").read_text())
    assert digest(prepared / "windows.npz") == report["data_sha256"]
    assert digest(prepared / "windows.csv") == report["metadata_sha256"]
    with np.load(prepared / "windows.npz", allow_pickle=False) as archive:
        x, valid = archive["x"], archive["valid"]
    meta = pd.read_csv(prepared / "windows.csv")
    inventory = pd.read_csv(prepared / "selected_sequences.csv").set_index("sequence_id")
    progress = json.loads((prepared / "source_progress.json").read_text())
    for source in progress["sources"]:
        assert digest(prepared / "source_poses" / f'{source["source_id"]}.npz') == source["source_cache_sha256"]
    assert len(meta) == len(x) == len(valid) == report["n_windows"]
    assert meta.window_id.is_unique
    assert not set(meta.source_id) & {"CMls0LwK5OU", "eBvzFkcvScg"}
    sequence_count = 0
    frame_count = 0
    for sequence, group in meta.groupby("sequence_id"):
        with np.load(prepared / "sequence_poses" / f"{sequence}.npz", allow_pickle=False) as archive:
            pose = {k: archive[k].copy() for k in archive.files}
        lineage = json.loads(str(pose["provenance_json"]))
        source_path = prepared / "source_poses" / f'{lineage["source_id"]}.npz'
        assert digest(source_path) == lineage["source_cache_sha256"]
        assert digest(REPO / inventory.loc[sequence].csv_path) == lineage["csv_sha256"]
        rows = pd.read_csv(REPO / inventory.loc[sequence].csv_path, keep_default_na=False).set_index("frame_num")
        lookup = {int(number): i for i, number in enumerate(pose["frame_numbers"])}
        with np.load(source_path, allow_pickle=False) as archive:
            source_lookup = {int(number): i for i, number in enumerate(archive["frame_numbers"])}
            indices = [source_lookup[int(number)] for number in pose["frame_numbers"]]
            np.testing.assert_array_equal(pose["xy"], archive["xy"][indices])
            np.testing.assert_array_equal(pose["raw_confidence"], archive["confidence"][indices])
            np.testing.assert_array_equal(pose["timestamp_seconds"], archive["timestamp_seconds"][indices])
        width, height = map(int, pose["image_dimensions"])
        for i, number in enumerate(pose["frame_numbers"]):
            row = rows.loc[int(number)]
            confidence, reason = strict_gate(pose["xy"][i], pose["raw_confidence"][i], row.bbox, row.vid_info, width, height)
            np.testing.assert_array_equal(confidence, pose["confidence"][i])
            assert reason == pose["gate_reason"][i]
        for i, row in group.iterrows():
            numbers = json.loads(row.frame_numbers)
            indices = [lookup[int(number)] for number in numbers]
            reconstructed_x, reconstructed_valid, reason = supported_window(pose["xy"][indices], pose["confidence"][indices])
            assert not reason
            np.testing.assert_array_equal(reconstructed_x, x[i])
            np.testing.assert_array_equal(reconstructed_valid, valid[i])
            np.testing.assert_allclose(pose["timestamp_seconds"][indices], json.loads(row.timestamp_seconds), atol=0, rtol=0)
            assert row.window_id == window_identity(row.source_id, numbers)
            assert np.all(np.diff(json.loads(row.timestamp_seconds)) > 0)
        sequence_count += 1
        frame_count += len(pose["frame_numbers"])
    output = dict(status="passed", windows=len(x), sources=int(meta.source_id.nunique()),
                  retained_sequences_replayed=sequence_count, gated_frames_replayed=frame_count,
                  raw_source_hashes_verified=len(progress["sources"]),
                  data_sha256=digest(prepared / "windows.npz"), metadata_sha256=digest(prepared / "windows.csv"),
                  audit_script_sha256=digest(__file__),
                  checks=["source cache hashes", "raw-to-sequence array equality", "annotation hashes",
                          "person gate exact replay", "window normalization and support exact replay",
                          "timestamps and frame identities", "duplicate window exclusion", "whole-source exclusions"],
                  limitation="This replay verifies stored input construction. It does not establish anatomical accuracy or unique patient identity.")
    destination = Path(__file__).with_name("prepared_input_audit.json")
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
