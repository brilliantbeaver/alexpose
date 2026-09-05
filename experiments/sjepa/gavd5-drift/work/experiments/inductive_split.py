"""Inductive redesign, step 1: build & persist the source-disjoint outer CV split.

WHY. The transductive laterality study pretrains one S-JEPA on ALL evaluated
sequences (fingerprint 7d13841a), so no lane can claim generalization. The fix is
a source-disjoint outer split over which the ENTIRE 5-stage curriculum is retrained
fold-locally (see pretrain_fold_local.py / inductive_harness.py) and the probe is
scored out-of-sample (inductive_probe.py). This module only builds the split.

COHORT RULE (reproduces the canonical 626 / 93-video modeled cohort exactly).
  pose_records_from_cache() over all 5 conditions -> prepare_sequence -> keep a
  sequence iff its 12-keypoint (MASK_KEYPOINTS) validity coverage >= 0.50. This
  DROP is applied to EVERY condition. Verified: it yields precisely the 626
  sequence_ids stored in sjepa_curriculum_final.pt (normal 270, parkinsons 41,
  stroke 74, myopathic 183, cerebralpalsy 58; 93 unique source videos).

SPLIT. StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED),
  groups = video_id (the independent unit = source video), stratify = condition.
  Train/test are therefore source-video-disjoint. NO person IDs exist and identity
  inference is forbidden, so this is generalization to unseen SOURCE VIDEOS, not
  unseen individuals. Folder labels are dataset annotations, not diagnoses.
"""
import sys
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _probe_common import (  # noqa: E402
    ART, CONDITIONS, MASK_KEYPOINTS, pose_records_from_cache, prepare_sequence,
)

FRAMES = 64
MIN_COVERAGE = 0.50
N_SPLITS = 5
SPLIT_SEED = 20260904
FOLDS_PATH = ART / "inductive_folds.json"

EXPECTED_TOTAL = 626
EXPECTED_VIDEOS = 93
EXPECTED_PER_CONDITION = {
    "normal": 270, "parkinsons": 41, "stroke": 74, "myopathic": 183, "cerebralpalsy": 58,
}


def assemble_modeled_cohort(frames=FRAMES, min_coverage=MIN_COVERAGE):
    """Deterministic list of {sequence_id, video_id, condition, coverage} for the
    modeled cohort. Coverage drop applied to EVERY condition. Order follows
    pose_records_from_cache (condition-major, then glob-sorted filenames)."""
    cohort = []
    for r in pose_records_from_cache():
        _, valid = prepare_sequence(r["raw"], frames)          # valid: [frames, 33]
        coverage = float(valid[:, MASK_KEYPOINTS].mean())
        if coverage >= min_coverage:
            cohort.append({
                "sequence_id": r["sequence_id"],
                "video_id": r["video_id"],
                "condition": r["condition"],
                "coverage": coverage,
            })
    return cohort


def reconcile_against_canonical(cohort):
    """Assert the assembled cohort is exactly the canonical 626 the encoder trained
    on, and return the canonical dataset_fingerprint (7d13841a...)."""
    import torch
    ck = torch.load(ART / "sjepa_curriculum_final.pt", map_location="cpu", weights_only=False)
    canon = set(map(str, ck["sequence_ids"]))
    got = {c["sequence_id"] for c in cohort}
    extra, missing = sorted(got - canon), sorted(canon - got)
    assert not extra and not missing, (
        f"assembled cohort != canonical 626: extra={extra[:5]} missing={missing[:5]}")
    return str(ck["dataset_fingerprint"])


def build_folds(verbose=True, reconcile=True):
    from sklearn.model_selection import StratifiedGroupKFold

    cohort = assemble_modeled_cohort()
    per_cond = {c: sum(1 for x in cohort if x["condition"] == c) for c in CONDITIONS}
    n_videos = len({x["video_id"] for x in cohort})
    assert len(cohort) == EXPECTED_TOTAL, f"{len(cohort)} != {EXPECTED_TOTAL}"
    assert n_videos == EXPECTED_VIDEOS, f"{n_videos} != {EXPECTED_VIDEOS}"
    assert per_cond == EXPECTED_PER_CONDITION, per_cond

    canonical_fp = reconcile_against_canonical(cohort) if reconcile else None

    seq = np.array([c["sequence_id"] for c in cohort])
    vid = np.array([c["video_id"] for c in cohort])
    cond = np.array([c["condition"] for c in cohort])

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    folds = []
    for f, (tr, te) in enumerate(sgkf.split(seq, cond, groups=vid)):
        train_vids, test_vids = sorted(set(vid[tr].tolist())), sorted(set(vid[te].tolist()))
        # HARD GUARANTEES:
        assert not (set(train_vids) & set(test_vids)), f"fold {f}: train/test video overlap"
        assert set(cond[tr].tolist()) == set(CONDITIONS), (
            f"fold {f}: train missing conditions {set(CONDITIONS) - set(cond[tr].tolist())}")
        tr_counts = {c: int((cond[tr] == c).sum()) for c in CONDITIONS}
        te_counts = {c: int((cond[te] == c).sum()) for c in CONDITIONS}
        tr_vid_counts = {c: len(set(vid[tr][cond[tr] == c].tolist())) for c in CONDITIONS}
        te_vid_counts = {c: len(set(vid[te][cond[te] == c].tolist())) for c in CONDITIONS}
        folds.append({
            "fold": f,
            "train_video_ids": train_vids,
            "test_video_ids": test_vids,
            "train_sequence_ids": sorted(seq[tr].tolist()),
            "test_sequence_ids": sorted(seq[te].tolist()),
            "n_train_seq": int(len(tr)), "n_test_seq": int(len(te)),
            "n_train_videos": len(train_vids), "n_test_videos": len(test_vids),
            "train_condition_seq_counts": tr_counts,
            "test_condition_seq_counts": te_counts,
            "train_condition_video_counts": tr_vid_counts,
            "test_condition_video_counts": te_vid_counts,
        })

    # every sequence is a test row exactly once (partition check)
    all_test = [s for fold in folds for s in fold["test_sequence_ids"]]
    assert len(all_test) == EXPECTED_TOTAL and len(set(all_test)) == EXPECTED_TOTAL, \
        "test folds are not a clean partition of the cohort"

    manifest = {
        "schema": "inductive_folds/v1",
        "split": "StratifiedGroupKFold(groups=video_id, stratify=condition)",
        "n_splits": N_SPLITS,
        "split_seed": SPLIT_SEED,
        "min_coverage": MIN_COVERAGE,
        "frames": FRAMES,
        "total_sequences": EXPECTED_TOTAL,
        "total_videos": EXPECTED_VIDEOS,
        "per_condition_sequences": per_cond,
        "canonical_fingerprint": canonical_fp,
        "independent_unit": "source video (video_id); NOT the individual",
        "cohort": [
            {"sequence_id": c["sequence_id"], "video_id": c["video_id"],
             "condition": c["condition"]} for c in cohort
        ],
        "folds": folds,
    }
    FOLDS_PATH.write_text(json.dumps(manifest, indent=2))

    if verbose:
        print(f"cohort: {len(cohort)} sequences / {n_videos} videos  (canonical fp "
              f"{canonical_fp[:8] if canonical_fp else 'n/a'})")
        print(f"per condition: {per_cond}")
        print(f"\nStratifiedGroupKFold(n_splits={N_SPLITS}, seed={SPLIT_SEED}) "
              f"grouped by video_id, stratified by condition:")
        header = "fold |            " + "  ".join(f"{c[:5]:>11s}" for c in CONDITIONS)
        print(header)
        for fold in folds:
            def cell(c):
                return f"{fold['test_condition_seq_counts'][c]:>2d}s/{fold['test_condition_video_counts'][c]:>2d}v"
            row = "  ".join(f"{cell(c):>11s}" for c in CONDITIONS)
            print(f"  {fold['fold']}  | test: {row}   (Σ {fold['n_test_seq']}s/{fold['n_test_videos']}v)")
        print(f"\nEvery train fold contains all 5 conditions: OK")
        print(f"Train/test source-video-disjoint per fold: OK")
        print(f"Test folds partition the 626 cohort exactly: OK")
        print(f"wrote {FOLDS_PATH}")
    return manifest


def load_folds():
    return json.loads(FOLDS_PATH.read_text())


if __name__ == "__main__":
    build_folds()
