#!/usr/bin/env python3
"""Create a conservative, evidence-annotated AMASS subject registry.

This is a time-bounded audit for the seven-source inventory currently used by
gavd6. It deliberately excludes rows whose person-level identity cannot be
resolved safely from the available source metadata.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


AMASS_POSTER = "https://files.is.tue.mpg.de/black/papers/amass_iccv_poster.pdf"
ACCAD_SOURCE = (
    "https://accad.ohio-state.edu/research/motion-lab/mocap-system-and-data"
)
CMU_SOURCE = "https://mocap.cs.cmu.edu/"

EXPECTED_CANDIDATES = {
    "ACCAD": 20,
    "BioMotionLab_NTroje": 111,
    "CMU": 102,
    "EKUT": 4,
    "Eyes_Japan_Dataset": 12,
    "KIT": 55,
    "MPI_HDM05": 4,
}

DIRECT_ID_SOURCES = {
    "BioMotionLab_NTroje",
    "EKUT",
    "Eyes_Japan_Dataset",
    "KIT",
    "MPI_HDM05",
}

REQUIRED_REGISTRY_COLUMNS = {
    "source_dataset",
    "subject_id_candidate",
    "audited_subject_id",
    "identity",
    "identity_audit_status",
    "known_downstream_overlap",
    "excluded",
    "exclusion_reason",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def source_local_id(source: str, candidate: str) -> str:
    prefix = f"{source}::"
    if not candidate.startswith(prefix):
        raise ValueError(f"candidate {candidate!r} does not start with {prefix!r}")
    local_id = candidate[len(prefix) :].strip()
    if not local_id:
        raise ValueError(f"candidate {candidate!r} has a blank local ID")
    return local_id


def accad_decision(local_id: str) -> tuple[str, str, bool, str, str]:
    if local_id.startswith("Female1"):
        return (
            "Female1",
            "approved",
            False,
            "",
            "Official ACCAD metadata groups General, Gestures, Running, and Walking under Female 1.",
        )
    if local_id.startswith("Male1"):
        return (
            "Male1",
            "approved",
            False,
            "",
            "Official ACCAD metadata groups General, Running, and Walking under Male 1.",
        )
    if local_id.startswith("Male2") or local_id == "MartialArtsWalksTurns_c3d":
        return (
            "Male2",
            "approved",
            False,
            "",
            "Official ACCAD metadata groups General, martial arts, Running, Walking, and Walks/Turns under Male 2.",
        )
    return (
        local_id,
        "excluded_unresolved_identity",
        True,
        "Conservative exclusion: ACCAD person identity is not established by the available source label.",
        "Excluded rather than guessing during the time-bounded audit.",
    )


def decide_row(source: str, candidate: str) -> dict[str, object]:
    local_id = source_local_id(source, candidate)
    common_note = (
        "No source-level overlap with the downstream datasets named in the local methodology was found. "
        "known_downstream_overlap=false means no known match, not independently proven person-level disjointness."
    )

    if source == "ACCAD":
        audited_id, status, excluded, reason, source_note = accad_decision(local_id)
        return {
            "audited_subject_id": audited_id,
            "identity": f"{source}::{audited_id}",
            "identity_audit_status": status,
            "known_downstream_overlap": "false",
            "excluded": str(excluded).lower(),
            "exclusion_reason": reason,
            "audit_method": "official-source alias reconciliation",
            "audit_evidence": ACCAD_SOURCE,
            "audit_notes": f"{source_note} {common_note}",
        }

    if source == "CMU":
        return {
            "audited_subject_id": local_id,
            "identity": candidate,
            "identity_audit_status": "excluded_unverified_aliases",
            "known_downstream_overlap": "false",
            "excluded": "true",
            "exclusion_reason": (
                "Conservative source exclusion: CMU states that one person may appear under multiple "
                "subject numbers, and no authoritative person crosswalk was completed."
            ),
            "audit_method": "source-level quarantine",
            "audit_evidence": f"{CMU_SOURCE} | {AMASS_POSTER}",
            "audit_notes": (
                "The local inventory has 102 CMU candidate folders while the AMASS poster reports 96 CMU "
                f"subjects. Excluded to prevent person leakage. {common_note}"
            ),
        }

    if source in DIRECT_ID_SOURCES:
        return {
            "audited_subject_id": local_id,
            "identity": candidate,
            "identity_audit_status": "approved",
            "known_downstream_overlap": "false",
            "excluded": "false",
            "exclusion_reason": "",
            "audit_method": "source-scoped folder ID reconciliation",
            "audit_evidence": AMASS_POSTER,
            "audit_notes": (
                "The number of source-scoped candidate IDs matches the AMASS-published subject count for "
                f"this source. {common_note}"
            ),
        }

    raise ValueError(f"no frozen audit rule for source {source!r}")


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    for path in (args.output, args.summary):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    registry = pd.read_csv(args.registry, keep_default_na=False, dtype=str)
    inventory = pd.read_csv(args.inventory, keep_default_na=False, dtype=str)
    missing = REQUIRED_REGISTRY_COLUMNS - set(registry.columns)
    if missing:
        raise ValueError(f"registry missing columns: {sorted(missing)}")

    source_counts = registry.groupby("source_dataset").size().to_dict()
    if source_counts != EXPECTED_CANDIDATES:
        raise ValueError(
            f"registry source counts changed; expected {EXPECTED_CANDIDATES}, got {source_counts}"
        )
    if registry.duplicated(["source_dataset", "subject_id_candidate"]).any():
        raise ValueError("registry contains duplicate source/candidate rows")

    ok_inventory = inventory.loc[inventory["status"].eq("ok")].copy()
    inventory_keys = set(
        map(
            tuple,
            ok_inventory[["source_dataset", "subject_id_candidate"]].drop_duplicates().to_numpy(),
        )
    )
    registry_keys = set(
        map(tuple, registry[["source_dataset", "subject_id_candidate"]].to_numpy())
    )
    if inventory_keys != registry_keys:
        raise ValueError("registry and valid inventory subject keys do not match exactly")

    audited = registry.copy()
    decisions = [
        decide_row(row.source_dataset, row.subject_id_candidate)
        for row in audited.itertuples(index=False)
    ]
    for column in decisions[0]:
        audited[column] = [decision[column] for decision in decisions]
    audited["audit_scope"] = "time-bounded-source-identity-v1"

    expected_identity = (
        audited["source_dataset"] + "::" + audited["audited_subject_id"]
    )
    if not audited["identity"].eq(expected_identity).all():
        raise ValueError("audited identity formula mismatch")
    excluded = audited["excluded"].eq("true")
    approved = audited["identity_audit_status"].eq("approved")
    if ((~approved) & (~excluded)).any():
        raise ValueError("every non-approved row must be excluded")
    if (excluded & audited["exclusion_reason"].str.strip().eq("")).any():
        raise ValueError("every excluded row must have a reason")

    eligible_candidates = set(audited.loc[approved & ~excluded, "subject_id_candidate"])
    sequence_summary = (
        ok_inventory.assign(
            eligible=ok_inventory["subject_id_candidate"].isin(eligible_candidates)
        )
        .groupby("source_dataset")
        .agg(
            total_sequences=("relative_path", "size"),
            eligible_sequences=("eligible", "sum"),
        )
    )
    registry_summary = audited.assign(eligible=approved & ~excluded).groupby(
        "source_dataset"
    ).agg(
        candidate_rows=("subject_id_candidate", "size"),
        unique_audited_identities=("identity", "nunique"),
        eligible_candidate_rows=("eligible", "sum"),
        excluded_candidate_rows=("excluded", lambda values: values.eq("true").sum()),
    )
    eligible_identity_counts = (
        audited.loc[approved & ~excluded].groupby("source_dataset")["identity"].nunique()
    )
    registry_summary["eligible_identities"] = eligible_identity_counts.reindex(
        registry_summary.index, fill_value=0
    )
    summary = registry_summary.join(sequence_summary).reset_index()
    summary["excluded_sequences"] = (
        summary["total_sequences"] - summary["eligible_sequences"]
    )
    summary["audit_scope"] = "time-bounded-source-identity-v1"

    atomic_write_csv(audited, args.output)
    atomic_write_csv(summary, args.summary)

    print(f"wrote audited registry: {args.output}")
    print(f"wrote audit summary: {args.summary}")
    print(f"eligible candidate rows: {int((approved & ~excluded).sum())}")
    print(f"eligible identities: {audited.loc[approved & ~excluded, 'identity'].nunique()}")
    print(f"eligible sequences: {int(ok_inventory['subject_id_candidate'].isin(eligible_candidates).sum())}")
    print(f"excluded candidate rows: {int(excluded.sum())}")


if __name__ == "__main__":
    main()
