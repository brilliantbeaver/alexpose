from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .artifacts import atomic_write_json
from .config import ExperimentContext, canonical_json_digest
from .data import PreparedCohort


def source_table(cohort_table: pd.DataFrame, conditions: Iterable[str]) -> pd.DataFrame:
    required = {"sequence_id", "video_id", "condition"}
    missing = required - set(cohort_table.columns)
    if missing:
        raise ValueError(f"Cohort table lacks columns: {sorted(missing)}")
    allowed = set(map(str, conditions))
    observed = set(cohort_table["condition"].astype(str))
    if not observed <= allowed:
        raise ValueError(f"Unexpected dataset annotations: {sorted(observed - allowed)}")
    if observed != allowed:
        raise ValueError(f"Configured annotations absent from cohort: {sorted(allowed - observed)}")
    condition_count = cohort_table.groupby("video_id")["condition"].nunique()
    if (condition_count != 1).any():
        raise ValueError("Each source video must have exactly one dataset annotation")
    table = (
        cohort_table.groupby("video_id", as_index=False)
        .agg(condition=("condition", "first"), sequence_count=("sequence_id", "size"))
        .sort_values("video_id")
        .reset_index(drop=True)
    )
    if table["video_id"].duplicated().any():
        raise AssertionError("Source table is not one row per video")
    return table


def _condition_counts(table: pd.DataFrame, ids: list[str], conditions: list[str]) -> dict[str, int]:
    selected = table[table["video_id"].isin(ids)]
    counts = selected["condition"].value_counts()
    return {condition: int(counts.get(condition, 0)) for condition in conditions}


def build_source_splits(
    cohort_df: pd.DataFrame,
    conditions: Iterable[str],
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> dict[str, Any]:
    condition_order = list(map(str, conditions))
    sources = source_table(cohort_df, condition_order)
    minimum_class_count = int(sources["condition"].value_counts().min())
    if minimum_class_count < outer_folds:
        raise ValueError(
            f"Need at least {outer_folds} sources in every annotation; minimum is {minimum_class_count}"
        )

    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    folds: list[dict[str, Any]] = []
    for fold, (train_index, test_index) in enumerate(
        outer.split(np.zeros(len(sources)), sources["condition"])
    ):
        train_table = sources.iloc[train_index].sort_values("video_id").reset_index(drop=True)
        test_table = sources.iloc[test_index].sort_values("video_id").reset_index(drop=True)
        train_ids = train_table["video_id"].astype(str).tolist()
        test_ids = test_table["video_id"].astype(str).tolist()
        if set(train_ids) & set(test_ids):
            raise AssertionError("Outer train/test sources overlap")

        inner_minimum = int(train_table["condition"].value_counts().min())
        n_inner = min(int(inner_folds), inner_minimum)
        if n_inner < 2:
            raise ValueError("Each outer training set needs at least two sources per annotation")
        inner = StratifiedKFold(
            n_splits=n_inner,
            shuffle=True,
            random_state=seed + 1009 * (fold + 1),
        )
        inner_payload: list[dict[str, Any]] = []
        validation_counter: Counter[str] = Counter()
        for inner_fold, (inner_train_index, validation_index) in enumerate(
            inner.split(np.zeros(len(train_table)), train_table["condition"])
        ):
            inner_train = sorted(train_table.iloc[inner_train_index]["video_id"].astype(str))
            validation = sorted(train_table.iloc[validation_index]["video_id"].astype(str))
            if set(inner_train) & set(validation):
                raise AssertionError("Inner train/validation sources overlap")
            if set(inner_train) | set(validation) != set(train_ids):
                raise AssertionError("Inner partition does not cover the outer training sources")
            validation_counter.update(validation)
            inner_payload.append(
                {
                    "fold": inner_fold,
                    "train_sources": inner_train,
                    "validation_sources": validation,
                }
            )
        if validation_counter != Counter({source: 1 for source in train_ids}):
            raise AssertionError("Every outer-training source must validate exactly once")

        train_sequences = sorted(
            cohort_df.loc[cohort_df["video_id"].isin(train_ids), "sequence_id"].astype(str)
        )
        test_sequences = sorted(
            cohort_df.loc[cohort_df["video_id"].isin(test_ids), "sequence_id"].astype(str)
        )
        folds.append(
            {
                "fold": fold,
                "train_sources": sorted(train_ids),
                "test_sources": sorted(test_ids),
                "train_sequence_ids": train_sequences,
                "test_sequence_ids": test_sequences,
                "train_source_counts": _condition_counts(sources, train_ids, condition_order),
                "test_source_counts": _condition_counts(sources, test_ids, condition_order),
                "inner_readout_folds": inner_payload,
            }
        )

    all_sources = set(sources["video_id"].astype(str))
    test_counter = Counter(source for fold in folds for source in fold["test_sources"])
    if test_counter != Counter({source: 1 for source in all_sources}):
        raise AssertionError("Every source must appear in exactly one outer test fold")
    for condition in condition_order:
        counts = [fold["test_source_counts"][condition] for fold in folds]
        if max(counts) - min(counts) > 1:
            raise AssertionError(f"Outer source counts are imbalanced for {condition}: {counts}")
    return {
        "schema": "neurips_laterality_source_splits/v2",
        "seed": int(seed),
        "outer_folds": int(outer_folds),
        "inner_folds_requested": int(inner_folds),
        "source_count": len(sources),
        "sequence_count": len(cohort_df),
        "conditions": condition_order,
        "source_census": _condition_counts(sources, sorted(all_sources), condition_order),
        "folds": folds,
    }


def split_path(artifact_root: Path) -> Path:
    return artifact_root / "splits" / "source_splits.json"


def save_splits(
    context: ExperimentContext,
    cohort: PreparedCohort,
    splits: dict[str, Any],
) -> Path:
    content = dict(splits)
    content["protocol_digest"] = context.protocol_digest
    content["context_digest"] = context.context_digest
    content["cohort_digest"] = cohort.cohort_digest
    content["split_digest"] = canonical_json_digest(splits)
    return atomic_write_json(split_path(context.artifact_root), content)


def load_splits(context: ExperimentContext, cohort: PreparedCohort) -> dict[str, Any]:
    path = split_path(context.artifact_root)
    payload = json.loads(path.read_text())
    if payload.get("schema") != "neurips_laterality_source_splits/v2":
        raise RuntimeError("Unsupported split manifest")
    if payload.get("protocol_digest") != context.protocol_digest:
        raise RuntimeError("Split protocol digest mismatch")
    if payload.get("context_digest") != context.context_digest:
        raise RuntimeError("Split profile/effective-context digest mismatch")
    if payload.get("cohort_digest") != cohort.cohort_digest:
        raise RuntimeError("Split cohort digest mismatch")
    digest_payload = {
        key: value
        for key, value in payload.items()
        if key
        not in {"protocol_digest", "context_digest", "cohort_digest", "split_digest"}
    }
    if canonical_json_digest(digest_payload) != payload.get("split_digest"):
        raise RuntimeError("Split manifest content digest mismatch")
    expected_sources = set(cohort.table["video_id"].astype(str))
    counter = Counter(source for fold in payload["folds"] for source in fold["test_sources"])
    if counter != Counter({source: 1 for source in expected_sources}):
        raise RuntimeError("Split source coverage is invalid")
    return payload


def get_fold(splits: dict[str, Any], fold: int) -> dict[str, Any]:
    matches = [entry for entry in splits["folds"] if int(entry["fold"]) == int(fold)]
    if len(matches) != 1:
        raise ValueError(f"Split manifest has {len(matches)} entries for fold {fold}")
    return matches[0]
