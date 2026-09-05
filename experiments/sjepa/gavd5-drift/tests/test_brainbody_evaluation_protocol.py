from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "neurips-brain-body" / "evaluation_protocol.py"
SPEC = importlib.util.spec_from_file_location("evaluation_protocol", MODULE_PATH)
protocol = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(protocol)


def synthetic_manifest() -> pd.DataFrame:
    rows = []
    for condition in protocol.CONDITIONS:
        for video_number in range(10):
            video_id = f"{condition[:2]}{video_number:09d}"[:11]
            for sequence_number in range(1 + video_number % 3):
                rows.append(
                    {
                        "condition": condition,
                        "video_id": video_id,
                        "sequence_id": f"{condition}-{video_number}-{sequence_number}",
                        "first_frame": 1,
                        "last_frame": 64,
                        "n_annotated_frames": 64,
                    }
                )
    return pd.DataFrame(rows)


def test_nested_registry_is_source_disjoint_and_reproducible():
    manifest = synthetic_manifest()
    first = protocol.build_nested_split_registry(manifest)
    second = protocol.build_nested_split_registry(manifest.sample(frac=1, random_state=9))
    assert protocol.split_fingerprint(first) == protocol.split_fingerprint(second)
    assert set(first["role"]) == {"train", "validation", "test"}
    assert first.loc[first["role"].eq("test"), "video_id"].value_counts().eq(1).all()
    outer_test = first.loc[first["role"].eq("test")]
    total_sources = outer_test.groupby("outer_fold")["video_id"].nunique()
    assert int(total_sources.max() - total_sources.min()) <= 1
    per_condition = outer_test.pivot_table(
        index="outer_fold",
        columns="condition",
        values="video_id",
        aggfunc="nunique",
        fill_value=0,
    )
    assert ((per_condition.max() - per_condition.min()) <= 1).all()


def test_expand_split_assigns_every_sequence_once():
    manifest = synthetic_manifest()
    registry = protocol.build_nested_split_registry(manifest)
    expanded = protocol.expand_split(manifest, registry, outer_fold=0)
    assert len(expanded) == len(manifest)
    assert expanded["sequence_id"].is_unique
    assert not expanded["role"].isna().any()


def test_attach_pose_qc_replaces_placeholders_and_namespaces_pose_path():
    split = pd.DataFrame(
        [
            {
                "sequence_id": "s1",
                "video_id": "v1",
                "role": "train",
                "outer_fold": 0,
                "pose_qc_eligible": "pending_notebook_02",
                "path": "video-v1.mp4",
            },
            {
                "sequence_id": "s2",
                "video_id": "v2",
                "role": "validation",
                "outer_fold": 0,
                "pose_qc_eligible": "pending_notebook_02",
                "path": "video-v2.mp4",
            },
        ]
    )
    qc = pd.DataFrame(
        [
            {
                "sequence_id": "s1",
                "video_id": "v1",
                "pose_qc_eligible": True,
                "in_locked_manifest": True,
                "outer_fold": 0,
                "split_role": "train",
                "manifest_sha256": "manifest",
                "split_sha256": "split",
                "status": "ready",
                "path": "pose-s1.npz",
            },
            {
                # An inventory extra can share a metadata-public key but is not
                # permitted to attach because it failed the decoded lock.
                "sequence_id": "s2",
                "video_id": "v2",
                "pose_qc_eligible": False,
                "in_locked_manifest": False,
                "outer_fold": None,
                "split_role": None,
                "manifest_sha256": None,
                "split_sha256": None,
                "status": "unexpected",
                "path": "old-pose-s2.npz",
            },
        ]
    )
    attached = protocol.attach_pose_qc_eligibility(
        split,
        qc,
        outer_fold=0,
        manifest_sha256="manifest",
        split_sha256="split",
    )
    assert attached["pose_qc_eligible"].tolist() == [True, False]
    assert attached["pose_qc_recorded"].tolist() == [True, False]
    assert attached.loc[0, "pose_path"] == "pose-s1.npz"
    assert pd.isna(attached.loc[1, "pose_path"])
    assert attached["path"].tolist() == ["video-v1.mp4", "video-v2.mp4"]
    assert not any(column.endswith(("_x", "_y")) for column in attached.columns)


def test_attach_pose_qc_rejects_lineage_or_role_mismatch():
    split = pd.DataFrame(
        [{"sequence_id": "s1", "video_id": "v1", "role": "train", "outer_fold": 0}]
    )
    qc = pd.DataFrame(
        [
            {
                "sequence_id": "s1",
                "video_id": "v1",
                "pose_qc_eligible": True,
                "in_locked_manifest": True,
                "outer_fold": 0,
                "split_role": "test",
                "manifest_sha256": "manifest",
                "split_sha256": "split",
                "status": "ready",
                "path": "pose-s1.npz",
            }
        ]
    )
    try:
        protocol.attach_pose_qc_eligibility(
            split,
            qc,
            outer_fold=0,
            manifest_sha256="manifest",
            split_sha256="split",
        )
    except ValueError as error:
        assert "role disagrees" in str(error)
    else:
        raise AssertionError("Role mismatch should be rejected")

    qc.loc[0, "split_role"] = "train"
    qc.loc[0, "manifest_sha256"] = "wrong"
    try:
        protocol.attach_pose_qc_eligibility(
            split,
            qc,
            outer_fold=0,
            manifest_sha256="manifest",
            split_sha256="split",
        )
    except ValueError as error:
        assert "manifest hash" in str(error)
    else:
        raise AssertionError("Manifest hash mismatch should be rejected")


class PoseQCAttachmentUnittestBridge(unittest.TestCase):
    """Keep the contract tests runnable when only the stdlib runner is present."""

    def test_placeholder_and_path_collision_contract(self):
        test_attach_pose_qc_replaces_placeholders_and_namespaces_pose_path()

    def test_lineage_and_role_mismatch_contract(self):
        test_attach_pose_qc_rejects_lineage_or_role_mismatch()


def test_actual_raw_manifest_census():
    manifest = protocol.scan_sequence_manifest(ROOT / "data-gavd")
    assert len(manifest) == 666
    assert manifest["video_id"].nunique() == 103
    assert manifest["url"].nunique() == 103
    assert int(manifest["n_annotated_frames"].sum()) == 140_641
    assert manifest.groupby("condition")["sequence_id"].nunique().to_dict() == {
        "cerebralpalsy": 64,
        "myopathic": 188,
        "normal": 291,
        "parkinsons": 47,
        "stroke": 76,
    }


def test_dated_public_cohort_has_twenty_test_sources_per_fold():
    manifest = protocol.scan_sequence_manifest(ROOT / "data-gavd")
    snapshot = pd.read_csv(
        ROOT
        / "neurips-brain-body"
        / "docs"
        / "youtube_validity_2026-09-04.csv"
    )
    assert len(snapshot) == 103
    assert not snapshot["url_status"].eq("probe_error").any()
    public = protocol.apply_availability_snapshot(manifest, snapshot)
    assert len(public) == 657
    assert public["video_id"].nunique() == 100
    assert public.groupby("condition")["video_id"].nunique().to_dict() == {
        "cerebralpalsy": 10,
        "myopathic": 29,
        "normal": 32,
        "parkinsons": 11,
        "stroke": 18,
    }

    registry = protocol.build_nested_split_registry(public)
    outer_test = registry.loc[registry["role"].eq("test")]
    sources_per_fold = outer_test.groupby("outer_fold")["video_id"].nunique()
    assert sources_per_fold.to_dict() == {0: 20, 1: 20, 2: 20, 3: 20, 4: 20}

    per_condition = outer_test.pivot_table(
        index="outer_fold",
        columns="condition",
        values="video_id",
        aggfunc="nunique",
        fill_value=0,
    )
    assert ((per_condition.max() - per_condition.min()) <= 1).all()
