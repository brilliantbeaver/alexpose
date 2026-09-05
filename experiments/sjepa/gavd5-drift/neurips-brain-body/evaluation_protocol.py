"""Leakage-resistant data and split utilities for the BrainBodyFM notebooks.

The independent unit in GAVD is the source YouTube video, not a CSV clip.  This
module deliberately keeps availability auditing, split construction, and split
expansion outside model code so every notebook uses the same frozen contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


CONDITIONS = (
    "normal",
    "parkinsons",
    "stroke",
    "myopathic",
    "cerebralpalsy",
)
SPLIT_VERSION = "brainbody-source-constrained-v2"
DEFAULT_SPLIT_SEED = 20260904
_BALANCE_WEIGHT = 1_000_000.0


def _first_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            return next(csv.DictReader(handle))
        except StopIteration as exc:
            raise ValueError(f"Empty sequence CSV: {path}") from exc


def _frame_span(path: Path) -> tuple[int, int, int]:
    frame = pd.read_csv(path, usecols=["seq", "frame_num"])
    if frame.empty:
        raise ValueError(f"Empty sequence CSV: {path}")
    sequence_ids = frame["seq"].dropna().astype(str).unique()
    if len(sequence_ids) != 1 or sequence_ids[0] != path.stem:
        raise ValueError(
            f"{path} must contain exactly its filename sequence id; got "
            f"{sequence_ids.tolist()}"
        )
    numbers = pd.to_numeric(frame["frame_num"], errors="raise").astype(int)
    return int(numbers.min()), int(numbers.max()), int(numbers.nunique())


def scan_sequence_manifest(data_root: Path | str) -> pd.DataFrame:
    """Read all five GAVD folders without making an availability decision."""
    data_root = Path(data_root)
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        folder = data_root / condition
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing GAVD condition folder: {folder}")
        for path in sorted(folder.glob("*.csv")):
            first = _first_csv_row(path)
            required = {"seq", "frame_num", "id", "url"}
            missing = required.difference(first)
            if missing:
                raise ValueError(f"{path} is missing columns {sorted(missing)}")
            first_frame, last_frame, n_frames = _frame_span(path)
            video_id = str(first["id"]).strip()
            if len(video_id) != 11:
                raise ValueError(f"Bad YouTube id {video_id!r} in {path}")
            rows.append(
                {
                    "condition": condition,
                    "sequence_id": path.stem,
                    "video_id": video_id,
                    "url": str(first["url"]).strip(),
                    "csv_path": str(path.resolve()),
                    "first_frame": first_frame,
                    "last_frame": last_frame,
                    "n_annotated_frames": n_frames,
                    "cam_view": first.get("cam_view", ""),
                    "dataset_annotation": first.get("dataset", ""),
                    "gait_pattern_annotation": first.get("gait_pat", ""),
                }
            )
    manifest = pd.DataFrame(rows).sort_values(
        ["condition", "video_id", "sequence_id"]
    ).reset_index(drop=True)
    if manifest["sequence_id"].duplicated().any():
        duplicate = manifest.loc[
            manifest["sequence_id"].duplicated(False), "sequence_id"
        ].tolist()
        raise ValueError(f"Duplicate sequence ids: {duplicate[:10]}")
    label_counts = manifest.groupby("video_id")["condition"].nunique()
    if int(label_counts.max()) != 1:
        bad = label_counts[label_counts > 1].index.tolist()
        raise ValueError(f"Source videos cross condition folders: {bad}")
    return manifest


def apply_availability_snapshot(
    manifest: pd.DataFrame,
    snapshot: Path | str | pd.DataFrame,
    accepted_statuses: Iterable[str] = ("public",),
) -> pd.DataFrame:
    """Filter through a dated URL snapshot, never a hard-coded dead-id list.

    A ``public`` metadata result only establishes URL resolvability.  Download,
    decoded-frame-span, and pose-quality gates remain later, separate filters.
    """
    status = (
        snapshot.copy()
        if isinstance(snapshot, pd.DataFrame)
        else pd.read_csv(snapshot)
    )
    required = {"video_id", "url_status", "checked_at_utc"}
    missing = required.difference(status.columns)
    if missing:
        raise ValueError(f"Availability snapshot is missing {sorted(missing)}")
    if status["video_id"].duplicated().any():
        raise ValueError("Availability snapshot has duplicate video ids")
    accepted = set(accepted_statuses)
    merged = manifest.merge(
        status[[c for c in status.columns if c != "condition"]],
        on="video_id",
        how="left",
        validate="many_to_one",
    )
    if merged["url_status"].isna().any():
        missing_ids = sorted(merged.loc[merged["url_status"].isna(), "video_id"].unique())
        raise ValueError(f"Snapshot has no status for {missing_ids}")
    return merged.loc[merged["url_status"].isin(accepted)].reset_index(drop=True)


def stable_frame_sha256(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    columns = list(columns)
    stable = frame.loc[:, columns].sort_values(columns).reset_index(drop=True)
    payload = stable.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest_fingerprint(manifest: pd.DataFrame) -> str:
    return stable_frame_sha256(
        manifest,
        [
            "condition",
            "video_id",
            "sequence_id",
            "first_frame",
            "last_frame",
            "n_annotated_frames",
        ],
    )


def source_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Collapse clips to the independent source-video unit."""
    return (
        manifest.groupby(["condition", "video_id"], as_index=False)
        .agg(
            sequences=("sequence_id", "nunique"),
            annotated_frames=("n_annotated_frames", "sum"),
            first_frame=("first_frame", "min"),
            last_frame=("last_frame", "max"),
        )
        .sort_values(["condition", "video_id"])
        .reset_index(drop=True)
    )


def _stable_cost(seed: int, *parts: object) -> float:
    """Return a deterministic sub-unit tie-break cost.

    The integer-program objectives put a weight of one million on a one-clip
    balance improvement.  These costs therefore choose reproducibly among
    equally balanced allocations without changing the scientific objective.
    """
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) / 2**32


def _constraint_matrix(
    constraints: list[list[tuple[int, float]]],
    n_variables: int,
) -> lil_matrix:
    matrix = lil_matrix((len(constraints), n_variables), dtype=float)
    for row_number, terms in enumerate(constraints):
        for variable, coefficient in terms:
            matrix[row_number, variable] = coefficient
    return matrix


def _solve_binary_assignment(
    objective: np.ndarray,
    integrality: np.ndarray,
    constraints: list[list[tuple[int, float]]],
    lower: list[float],
    upper: list[float],
    binary_variables: int,
) -> np.ndarray:
    bounds = Bounds(
        np.zeros(len(objective), dtype=float),
        np.concatenate(
            [
                np.ones(binary_variables, dtype=float),
                np.full(len(objective) - binary_variables, np.inf),
            ]
        ),
    )
    matrix = _constraint_matrix(constraints, len(objective)).tocsr()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(matrix, lower, upper),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Source-fold allocation failed: {result.message}")
    binary = result.x[:binary_variables]
    if not np.all(np.isclose(binary, np.round(binary), atol=1e-6)):
        raise RuntimeError("Source-fold allocation returned a fractional assignment")
    return np.round(binary).astype(int)


def _partition_one_condition(
    sources: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    """Partition one condition into source-count-balanced local bins.

    Exact source quotas are hard constraints.  The MILP then minimizes the
    total absolute deviation of sequence counts from their per-bin target.
    """
    sources = sources.sort_values("video_id").reset_index(drop=True)
    n_sources = len(sources)
    base, remainder = divmod(n_sources, n_splits)
    quotas = [base + int(index < remainder) for index in range(n_splits)]

    n_binary = n_sources * n_splits
    deviation_start = n_binary
    n_variables = n_binary + n_splits
    objective = np.zeros(n_variables, dtype=float)
    objective[deviation_start:] = _BALANCE_WEIGHT
    for source_index, row in sources.iterrows():
        for local_bin in range(n_splits):
            objective[source_index * n_splits + local_bin] = _stable_cost(
                seed, row["condition"], row["video_id"], local_bin
            )

    constraints: list[list[tuple[int, float]]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(terms: list[tuple[int, float]], low: float, high: float) -> None:
        constraints.append(terms)
        lower.append(low)
        upper.append(high)

    for source_index in range(n_sources):
        add(
            [
                (source_index * n_splits + local_bin, 1.0)
                for local_bin in range(n_splits)
            ],
            1.0,
            1.0,
        )
    for local_bin, quota in enumerate(quotas):
        add(
            [
                (source_index * n_splits + local_bin, 1.0)
                for source_index in range(n_sources)
            ],
            float(quota),
            float(quota),
        )

    sequence_target = float(sources["sequences"].sum()) / n_splits
    for local_bin in range(n_splits):
        load = [
            (
                source_index * n_splits + local_bin,
                float(sources.loc[source_index, "sequences"]),
            )
            for source_index in range(n_sources)
        ]
        deviation = deviation_start + local_bin
        # load - deviation <= target
        add([*load, (deviation, -1.0)], -np.inf, sequence_target)
        # target - load <= deviation
        add(
            [
                *((variable, -coefficient) for variable, coefficient in load),
                (deviation, -1.0),
            ],
            -np.inf,
            -sequence_target,
        )

    assignment = _solve_binary_assignment(
        objective=objective,
        integrality=np.concatenate(
            [np.ones(n_binary, dtype=int), np.zeros(n_splits, dtype=int)]
        ),
        constraints=constraints,
        lower=lower,
        upper=upper,
        binary_variables=n_binary,
    ).reshape(n_sources, n_splits)
    if not np.all(assignment.sum(axis=1) == 1):
        raise RuntimeError("Each source must be assigned to exactly one local bin")
    result = sources.copy()
    result["local_bin"] = assignment.argmax(axis=1)
    return result


def _align_condition_bins(
    local: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> dict[tuple[str, int], int]:
    """Align condition-local bins while balancing total sources and clips."""
    summary = (
        local.groupby(["condition", "local_bin"], as_index=False)
        .agg(source_videos=("video_id", "nunique"), sequences=("sequences", "sum"))
        .sort_values(["condition", "local_bin"])
        .reset_index(drop=True)
    )
    conditions = sorted(summary["condition"].unique())
    expected_bins = set(range(n_splits))
    for condition in conditions:
        actual_bins = set(summary.loc[summary["condition"].eq(condition), "local_bin"])
        if actual_bins != expected_bins:
            raise RuntimeError(f"Condition {condition} has incomplete local bins")

    n_conditions = len(conditions)
    n_binary = n_conditions * n_splits * n_splits
    deviation_start = n_binary
    n_variables = n_binary + n_splits
    objective = np.zeros(n_variables, dtype=float)
    objective[deviation_start:] = _BALANCE_WEIGHT

    lookup = summary.set_index(["condition", "local_bin"])

    def variable(condition_index: int, local_bin: int, fold: int) -> int:
        return (condition_index * n_splits + local_bin) * n_splits + fold

    for condition_index, condition in enumerate(conditions):
        for local_bin in range(n_splits):
            for fold in range(n_splits):
                objective[variable(condition_index, local_bin, fold)] = _stable_cost(
                    seed, "align", condition, local_bin, fold
                )

    constraints: list[list[tuple[int, float]]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(terms: list[tuple[int, float]], low: float, high: float) -> None:
        constraints.append(terms)
        lower.append(low)
        upper.append(high)

    for condition_index in range(n_conditions):
        for local_bin in range(n_splits):
            add(
                [
                    (variable(condition_index, local_bin, fold), 1.0)
                    for fold in range(n_splits)
                ],
                1.0,
                1.0,
            )
        for fold in range(n_splits):
            add(
                [
                    (variable(condition_index, local_bin, fold), 1.0)
                    for local_bin in range(n_splits)
                ],
                1.0,
                1.0,
            )

    total_sources = int(summary["source_videos"].sum())
    source_floor, source_remainder = divmod(total_sources, n_splits)
    source_ceiling = source_floor + int(source_remainder > 0)
    sequence_target = float(summary["sequences"].sum()) / n_splits
    for fold in range(n_splits):
        source_terms: list[tuple[int, float]] = []
        sequence_terms: list[tuple[int, float]] = []
        for condition_index, condition in enumerate(conditions):
            for local_bin in range(n_splits):
                index = variable(condition_index, local_bin, fold)
                row = lookup.loc[(condition, local_bin)]
                source_terms.append((index, float(row["source_videos"])))
                sequence_terms.append((index, float(row["sequences"])))
        add(source_terms, float(source_floor), float(source_ceiling))
        deviation = deviation_start + fold
        add([*sequence_terms, (deviation, -1.0)], -np.inf, sequence_target)
        add(
            [
                *((index, -coefficient) for index, coefficient in sequence_terms),
                (deviation, -1.0),
            ],
            -np.inf,
            -sequence_target,
        )

    assignment = _solve_binary_assignment(
        objective=objective,
        integrality=np.concatenate(
            [np.ones(n_binary, dtype=int), np.zeros(n_splits, dtype=int)]
        ),
        constraints=constraints,
        lower=lower,
        upper=upper,
        binary_variables=n_binary,
    ).reshape(n_conditions, n_splits, n_splits)
    mapping: dict[tuple[str, int], int] = {}
    for condition_index, condition in enumerate(conditions):
        for local_bin in range(n_splits):
            mapping[(condition, local_bin)] = int(
                assignment[condition_index, local_bin].argmax()
            )
    return mapping


def _source_fold_table(
    rows: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    sources = source_table(rows)
    label_counts = sources.groupby("video_id")["condition"].nunique()
    if not label_counts.empty and int(label_counts.max()) > 1:
        bad = sorted(label_counts[label_counts > 1].index)
        raise ValueError(f"Source videos cross conditions: {bad[:10]}")
    per_class_sources = (
        sources.groupby("condition")["video_id"]
        .nunique()
        .reindex(CONDITIONS, fill_value=0)
    )
    if int(per_class_sources.min()) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} source videos per condition; got "
            f"{per_class_sources.to_dict()}"
        )
    local = pd.concat(
        [
            _partition_one_condition(
                sources.loc[sources["condition"].eq(condition)],
                n_splits=n_splits,
                seed=seed,
            )
            for condition in CONDITIONS
        ],
        ignore_index=True,
    )
    mapping = _align_condition_bins(local, n_splits=n_splits, seed=seed)
    local["fold"] = [
        mapping[(row.condition, int(row.local_bin))]
        for row in local.itertuples(index=False)
    ]
    return local.sort_values(["fold", "condition", "video_id"]).reset_index(drop=True)


def _fold_assignments(
    rows: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return row indices from a constrained source-level fold allocation."""
    fold_table = _source_fold_table(rows, n_splits=n_splits, seed=seed)
    fold_by_video = fold_table.set_index("video_id")["fold"]
    row_folds = rows["video_id"].map(fold_by_video)
    if row_folds.isna().any():
        raise RuntimeError("A sequence has no source-level fold assignment")
    indices = np.arange(len(rows), dtype=int)
    return [
        (indices[row_folds.to_numpy() != fold], indices[row_folds.to_numpy() == fold])
        for fold in range(n_splits)
    ]


def build_nested_split_registry(
    manifest: pd.DataFrame,
    *,
    outer_splits: int = 5,
    inner_splits: int = 4,
    seed: int = DEFAULT_SPLIT_SEED,
) -> pd.DataFrame:
    """Create outer test and fold-local validation assignments by source video.

    Allocation operates on one row per ``video_id``. Exact per-condition source
    quotas differ by at most one; constrained optimization then balances clip
    counts without letting one upload masquerade as many independent samples.
    """
    rows = manifest.sort_values(["condition", "video_id", "sequence_id"]).reset_index(
        drop=True
    )
    sources = source_table(rows).set_index("video_id")
    records: list[dict[str, object]] = []
    outer = _fold_assignments(rows, outer_splits, seed)
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer):
        development = rows.iloc[outer_train_idx].reset_index(drop=True)
        inner = _fold_assignments(development, inner_splits, seed + outer_fold + 1)
        chosen_inner = outer_fold % inner_splits
        inner_train_idx, validation_idx = inner[chosen_inner]
        role_ids = {
            "train": set(development.iloc[inner_train_idx]["video_id"]),
            "validation": set(development.iloc[validation_idx]["video_id"]),
            "test": set(rows.iloc[outer_test_idx]["video_id"]),
        }
        for role, video_ids in role_ids.items():
            for video_id in sorted(video_ids):
                source = sources.loc[video_id]
                records.append(
                    {
                        "split_version": SPLIT_VERSION,
                        "split_seed": seed,
                        "outer_fold": outer_fold,
                        "inner_splits": inner_splits,
                        "inner_validation_fold": chosen_inner,
                        "role": role,
                        "condition": rows.loc[
                            rows["video_id"].eq(video_id), "condition"
                        ].iloc[0],
                        "video_id": video_id,
                        "sequences": int(source["sequences"]),
                        "annotated_frames": int(source["annotated_frames"]),
                    }
                )
    registry = pd.DataFrame(records).sort_values(
        ["outer_fold", "role", "condition", "video_id"]
    ).reset_index(drop=True)
    validate_split_registry(registry, manifest, outer_splits=outer_splits)
    return registry


def validate_split_registry(
    registry: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    outer_splits: int = 5,
) -> None:
    all_ids = set(manifest["video_id"])
    for fold in range(outer_splits):
        part = registry.loc[registry["outer_fold"].eq(fold)]
        role_sets = {
            role: set(part.loc[part["role"].eq(role), "video_id"])
            for role in ("train", "validation", "test")
        }
        if set.union(*role_sets.values()) != all_ids:
            raise AssertionError(f"Fold {fold} does not cover every source")
        if any(
            role_sets[a] & role_sets[b]
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ):
            raise AssertionError(f"Fold {fold} has source overlap between roles")
        for role in role_sets:
            classes = set(part.loc[part["role"].eq(role), "condition"])
            if classes != set(CONDITIONS):
                raise AssertionError(f"Fold {fold} {role} lacks classes: {classes}")
        manifest_sources = source_table(manifest)
        test_ids = role_sets["test"]
        development = manifest_sources.loc[
            ~manifest_sources["video_id"].isin(test_ids)
        ]
        validation = part.loc[part["role"].eq("validation")]
        if part["inner_splits"].nunique() != 1:
            raise AssertionError(f"Fold {fold} has inconsistent inner-split metadata")
        inner_splits = int(part["inner_splits"].iloc[0])
        chosen_inner = int(part["inner_validation_fold"].iloc[0])
        if inner_splits < 2 or not 0 <= chosen_inner < inner_splits:
            raise AssertionError(
                f"Fold {fold} has invalid inner split {chosen_inner}/{inner_splits}"
            )
        for condition in CONDITIONS:
            development_count = int(
                development["condition"].eq(condition).sum()
            )
            validation_count = int(
                validation["condition"].eq(condition).sum()
            )
            floor = development_count // inner_splits
            ceiling = floor + int(development_count % inner_splits > 0)
            if validation_count not in {floor, ceiling}:
                raise AssertionError(
                    f"Fold {fold} validation source imbalance for {condition}: "
                    f"{validation_count} not in {{{floor}, {ceiling}}}"
                )
        validation_total = int(validation["video_id"].nunique())
        development_total = int(development["video_id"].nunique())
        validation_floor = development_total // inner_splits
        validation_ceiling = validation_floor + int(
            development_total % inner_splits > 0
        )
        if validation_total not in {validation_floor, validation_ceiling}:
            raise AssertionError(
                f"Fold {fold} validation has {validation_total} sources; expected "
                f"{validation_floor} or {validation_ceiling}"
            )
    test_counts = registry.loc[registry["role"].eq("test"), "video_id"].value_counts()
    if set(test_counts.index) != all_ids or not test_counts.eq(1).all():
        raise AssertionError("Every source must be outer-test exactly once")
    outer_test = registry.loc[registry["role"].eq("test")]
    total_by_fold = outer_test.groupby("outer_fold")["video_id"].nunique()
    if int(total_by_fold.max() - total_by_fold.min()) > 1:
        raise AssertionError(
            f"Outer folds have imbalanced source counts: {total_by_fold.to_dict()}"
        )
    per_condition = outer_test.pivot_table(
        index="outer_fold",
        columns="condition",
        values="video_id",
        aggfunc="nunique",
        fill_value=0,
    )
    spread = per_condition.max(axis=0) - per_condition.min(axis=0)
    if bool((spread > 1).any()):
        raise AssertionError(
            f"Outer per-condition source counts differ by more than one: "
            f"{spread.to_dict()}"
        )


def split_fingerprint(registry: pd.DataFrame) -> str:
    return stable_frame_sha256(
        registry,
        [
            "split_version",
            "split_seed",
            "outer_fold",
            "inner_splits",
            "inner_validation_fold",
            "role",
            "condition",
            "video_id",
        ],
    )


def expand_split(
    manifest: pd.DataFrame,
    registry: pd.DataFrame,
    outer_fold: int,
) -> pd.DataFrame:
    assignments = registry.loc[
        registry["outer_fold"].eq(int(outer_fold)), ["video_id", "role"]
    ]
    expanded = manifest.merge(
        assignments, on="video_id", how="inner", validate="many_to_one"
    )
    expanded["outer_fold"] = int(outer_fold)
    return expanded


def _strict_boolean(series: pd.Series, *, label: str) -> pd.Series:
    """Parse an audit boolean without turning unknown strings into ``False``."""

    normalized = series.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes"}
    false_values = {"false", "0", "no"}
    unknown = ~normalized.isin(true_values | false_values)
    if unknown.any():
        values = sorted(series.loc[unknown].astype(str).unique())
        raise ValueError(f"{label} has non-boolean values: {values[:10]}")
    return normalized.isin(true_values)


def attach_pose_qc_eligibility(
    split_rows: pd.DataFrame,
    pose_qc: pd.DataFrame,
    *,
    outer_fold: int,
    manifest_sha256: str,
    split_sha256: str,
) -> pd.DataFrame:
    """Attach one authoritative fold-local pose-QC ledger to split rows.

    Notebook 01 intentionally carries a ``pending_notebook_02`` placeholder and
    video-download fields such as ``path``. Notebook 02 later emits the real QC
    boolean and a *pose* path. A generic pandas merge would suffix these names,
    remove the expected ``pose_qc_eligible`` column, and blur video paths with
    pose paths. This function replaces all earlier QC placeholders, namespaces
    the ledger fields, and validates lineage before any row can become eligible.
    """

    key_columns = ["sequence_id", "video_id"]
    split_required = {*key_columns, "role", "outer_fold"}
    qc_required = {
        *key_columns,
        "pose_qc_eligible",
        "in_locked_manifest",
        "outer_fold",
        "split_role",
        "manifest_sha256",
        "split_sha256",
        "status",
        "path",
    }
    missing_split = split_required.difference(split_rows.columns)
    missing_qc = qc_required.difference(pose_qc.columns)
    if missing_split:
        raise ValueError(f"Split rows are missing {sorted(missing_split)}")
    if missing_qc:
        raise ValueError(f"Pose-QC ledger is missing {sorted(missing_qc)}")
    if split_rows["sequence_id"].duplicated().any():
        raise ValueError("Split rows contain duplicate sequence_id values")
    if pose_qc["sequence_id"].duplicated().any():
        raise ValueError("Pose-QC ledger contains duplicate sequence_id values")

    expected_fold = int(outer_fold)
    split_folds = pd.to_numeric(split_rows["outer_fold"], errors="raise").astype(int)
    if not split_folds.eq(expected_fold).all():
        raise ValueError(
            f"Split rows contain an outer fold other than {expected_fold}"
        )

    qc = pose_qc.copy()
    qc["in_locked_manifest"] = _strict_boolean(
        qc["in_locked_manifest"], label="in_locked_manifest"
    )
    qc["pose_qc_eligible"] = _strict_boolean(
        qc["pose_qc_eligible"], label="pose_qc_eligible"
    )
    # Files outside the locked decoded cohort remain visible in notebook 02's
    # inventory but must never attach to a metadata-public split row.
    qc = qc.loc[qc["in_locked_manifest"]].copy()
    qc_folds = pd.to_numeric(qc["outer_fold"], errors="raise").astype(int)
    if not qc_folds.eq(expected_fold).all():
        raise ValueError(f"Pose-QC ledger does not belong to outer fold {expected_fold}")
    if not qc["manifest_sha256"].astype(str).eq(str(manifest_sha256)).all():
        raise ValueError("Pose-QC manifest hash differs from the frozen manifest")
    if not qc["split_sha256"].astype(str).eq(str(split_sha256)).all():
        raise ValueError("Pose-QC split hash differs from the frozen registry")
    eligible_not_ready = qc["pose_qc_eligible"] & ~qc["status"].astype(str).eq(
        "ready"
    )
    if eligible_not_ready.any():
        bad = qc.loc[eligible_not_ready, "sequence_id"].astype(str).tolist()
        raise ValueError(f"Pose-QC marks non-ready rows eligible: {bad[:10]}")
    eligible_without_path = qc["pose_qc_eligible"] & qc["path"].fillna("").astype(
        str
    ).str.strip().eq("")
    if eligible_without_path.any():
        bad = qc.loc[eligible_without_path, "sequence_id"].astype(str).tolist()
        raise ValueError(f"Pose-QC eligible rows lack a pose path: {bad[:10]}")

    split_keys = split_rows[key_columns].drop_duplicates()
    orphaned = qc[key_columns].merge(
        split_keys,
        on=key_columns,
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if orphaned["_merge"].ne("both").any():
        bad = orphaned.loc[
            orphaned["_merge"].ne("both"), "sequence_id"
        ].astype(str).tolist()
        raise ValueError(f"Pose-QC locked rows are absent from split rows: {bad[:10]}")

    optional_columns = [
        "neurologic_observed_fraction",
        "pose_frame_coverage",
        "resolution_safe_geometry",
        "crop_geometry_version",
        "eligibility_stage",
        "error",
    ]
    selected = qc[
        [
            *key_columns,
            "pose_qc_eligible",
            "path",
            "status",
            "split_role",
            *[column for column in optional_columns if column in qc],
        ]
    ].rename(
        columns={
            "path": "pose_path",
            "status": "pose_qc_status",
            "split_role": "pose_qc_split_role",
            "error": "pose_qc_error",
            "eligibility_stage": "pose_qc_eligibility_stage",
        }
    )

    left = split_rows.copy()
    # Replacement, never coalescence: pending or stale upstream QC fields have
    # no authority after the fold-specific ledger exists.
    qc_owned = {
        "pose_qc_eligible",
        "pose_path",
        "pose_qc_status",
        "pose_qc_split_role",
        "pose_qc_recorded",
        "pose_qc_error",
        "pose_qc_eligibility_stage",
        "neurologic_observed_fraction",
        "pose_frame_coverage",
        "resolution_safe_geometry",
        "crop_geometry_version",
    }
    left = left.drop(columns=[column for column in qc_owned if column in left])
    attached = left.merge(
        selected,
        on=key_columns,
        how="left",
        validate="one_to_one",
        indicator="pose_qc_merge",
    )
    attached["pose_qc_recorded"] = attached["pose_qc_merge"].eq("both")
    attached = attached.drop(columns="pose_qc_merge")
    role_mismatch = attached["pose_qc_recorded"] & attached["role"].ne(
        attached["pose_qc_split_role"]
    )
    if role_mismatch.any():
        bad = attached.loc[role_mismatch, "sequence_id"].astype(str).tolist()
        raise ValueError(f"Pose-QC role disagrees with frozen split: {bad[:10]}")
    attached["pose_qc_eligible"] = attached["pose_qc_eligible"].fillna(
        False
    ).astype(bool)
    return attached


def write_split_bundle(
    manifest: pd.DataFrame,
    registry: pd.DataFrame,
    output_dir: Path | str,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "eligible_sequence_manifest.csv"
    registry_path = output_dir / "source_split_registry.csv"
    manifest.to_csv(manifest_path, index=False)
    registry.to_csv(registry_path, index=False)
    contract = {
        "split_version": SPLIT_VERSION,
        "manifest_sha256": manifest_fingerprint(manifest),
        "split_sha256": split_fingerprint(registry),
        "sequences": int(manifest["sequence_id"].nunique()),
        "source_videos": int(manifest["video_id"].nunique()),
        "outer_folds": int(registry["outer_fold"].nunique()),
        "inner_folds": int(registry["inner_splits"].iloc[0]),
        "primary_unit": "source_video",
        "allocation": (
            "exact per-condition source quotas plus constrained absolute-deviation "
            "balancing of sequence counts"
        ),
        "roles": ["train", "validation", "test"],
        "test_use": "once, after preprocessing/model/checkpoint selection is frozen",
    }
    (output_dir / "source_split_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    return contract
