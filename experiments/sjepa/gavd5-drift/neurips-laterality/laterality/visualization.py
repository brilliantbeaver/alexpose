"""Privacy-preserving inline diagnostics for the ordered notebook suite."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import ExperimentContext


def _profile_label(context: ExperimentContext) -> str:
    return (
        "PAPER PROFILE — EMPIRICAL OUTPUT"
        if context.is_paper
        else "SYNTHETIC SMOKE — NON-EVIDENTIARY"
    )


def _finish(figure, context: ExperimentContext):
    figure.text(
        0.995,
        0.005,
        _profile_label(context),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#8b1e3f" if not context.is_paper else "#264653",
        weight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.98))
    return figure


def governance_figure(
    context: ExperimentContext,
    governance_payload: dict[str, Any],
):
    import matplotlib.pyplot as plt

    reviews = (
        ("Ethics determination", "ethics_determination"),
        ("Data-use review", "data_use_review"),
        ("Derived-pose release", "derived_pose_release_review"),
    )
    resolved = [
        governance_payload.get(key, {}).get("status") == "resolved"
        and bool(governance_payload.get(key, {}).get("reference"))
        and bool(governance_payload.get(key, {}).get("date"))
        for _, key in reviews
    ]
    figure, axis = plt.subplots(figsize=(8.5, 2.6))
    y = np.arange(len(reviews))
    axis.barh(
        y,
        np.ones(len(reviews)),
        color=["#2a9d8f" if value else "#c44536" for value in resolved],
    )
    axis.set_yticks(y, [label for label, _ in reviews])
    axis.set_xlim(0.0, 1.0)
    axis.set_xticks([])
    axis.invert_yaxis()
    for index, value in enumerate(resolved):
        axis.text(
            0.5,
            index,
            "RESOLVED" if value else "BLOCKED",
            ha="center",
            va="center",
            color="white",
            weight="bold",
        )
    axis.set_title("Submission governance gate (status only; references suppressed)")
    for spine in axis.spines.values():
        spine.set_visible(False)
    return _finish(figure, context)


def cohort_figure(context: ExperimentContext, cohort) -> Any:
    import matplotlib.pyplot as plt

    attrition = cohort.attrition
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    counts = [
        int(attrition["input_sequences"]),
        int(attrition["accepted_sequences"]),
        int(attrition["excluded_sequences"]),
    ]
    axes[0].bar(
        ["Input poses", "QC eligible", "Excluded"],
        counts,
        color=["#457b9d", "#2a9d8f", "#c44536"],
    )
    for index, value in enumerate(counts):
        axes[0].text(index, value, str(value), ha="center", va="bottom")
    axes[0].set_ylabel("Sequences")
    axes[0].set_title("Locked cohort attrition")

    target = cohort.table["target"].to_numpy(dtype=np.float64)
    axes[1].hist(target, bins=min(24, max(8, int(np.sqrt(len(target))))), color="#6d597a")
    axes[1].axvline(0.0, color="black", linewidth=1.0)
    axes[1].set_xlabel("Paired-valid motion contrast")
    axes[1].set_ylabel("Sequences")
    axes[1].set_title("Coordinate-derived target (not a clinical measure)")
    return _finish(figure, context)


def split_figure(context: ExperimentContext, splits: dict[str, Any]):
    import matplotlib.pyplot as plt

    conditions = list(splits["conditions"])
    frame = pd.DataFrame(
        [
            {"fold": int(fold["fold"]), **fold["test_source_counts"]}
            for fold in splits["folds"]
        ]
    ).set_index("fold")
    figure, axis = plt.subplots(figsize=(9.5, 4.2))
    frame[conditions].plot(
        kind="bar",
        stacked=True,
        ax=axis,
        color=["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"],
    )
    axis.set_xlabel("Outer fold")
    axis.set_ylabel("Held-out source videos")
    axis.set_title("Outer-test composition by dataset annotation")
    axis.legend(title="Dataset annotation", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    return _finish(figure, context)


def training_figure(
    context: ExperimentContext,
    training_summaries: Sequence[dict[str, Any]],
):
    import matplotlib.pyplot as plt

    history_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    for summary in training_summaries:
        for item in summary["history"]:
            history_rows.append(
                {
                    "variant": summary["variant"],
                    "epoch": int(item["epoch"]),
                    "loss": float(item["mean_total_loss"]),
                }
            )
        balance_rows.append(
            {
                "variant": summary["variant"],
                "fold": int(summary["fold"]),
                "seed": int(summary["seed"]),
                "draw_ratio": float(summary["maximum_source_draws"])
                / float(summary["minimum_source_draws"]),
            }
        )
    history = pd.DataFrame(history_rows)
    balance = pd.DataFrame(balance_rows)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    colors = {"vanilla": "#457b9d", "reflection_augmented": "#e76f51"}
    for variant, group in history.groupby("variant", sort=True):
        aggregate = group.groupby("epoch")["loss"].agg(["mean", "min", "max"])
        color = colors.get(str(variant), None)
        axes[0].plot(aggregate.index, aggregate["mean"], label=variant, color=color)
        axes[0].fill_between(
            aggregate.index,
            aggregate["min"],
            aggregate["max"],
            color=color,
            alpha=0.16,
        )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean total loss")
    axes[0].set_title("Fold/seed training histories")
    axes[0].legend(title="Variant")

    for index, (variant, group) in enumerate(balance.groupby("variant", sort=True)):
        axes[1].scatter(
            np.full(len(group), index),
            group["draw_ratio"],
            label=variant,
            alpha=0.8,
            color=colors.get(str(variant), None),
        )
    variants = sorted(balance["variant"].unique())
    axes[1].set_xticks(np.arange(len(variants)), variants, rotation=15, ha="right")
    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("Maximum / minimum source draws")
    axes[1].set_title("Source-sampling balance diagnostic")
    return _finish(figure, context)


def evaluation_figure(
    context: ExperimentContext,
    evaluations: pd.DataFrame,
):
    import matplotlib.pyplot as plt

    primary_lane = str(context.protocol["evaluation"]["primary_lane"])
    selected = evaluations[evaluations["lane"] == primary_lane].copy()
    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for (variant, seed), frame in selected.groupby(["variant", "seed"], sort=True):
        for state in ("learned", "initial"):
            per_source = (
                pd.DataFrame(
                    {
                        "video_id": frame["video_id"].astype(str),
                        "error": frame[f"{state}_strict_equivariance_error"],
                    }
                )
                .groupby("video_id", as_index=False)["error"]
                .mean()
            )
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "state": state,
                    "error": float(per_source["error"].mean()),
                }
            )
        learned_source = frame.groupby("video_id")[
            "learned_strict_equivariance_error"
        ].mean()
        initial_source = frame.groupby("video_id")[
            "initial_strict_equivariance_error"
        ].mean()
        for source in learned_source.index:
            source_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "learned": float(learned_source.loc[source]),
                    "initial": float(initial_source.loc[source]),
                }
            )
    summary = pd.DataFrame(rows)
    paired = pd.DataFrame(source_rows)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    pivot = summary.groupby(["variant", "state"])["error"].mean().unstack()
    pivot[[column for column in ("initial", "learned") if column in pivot]].plot(
        kind="bar", ax=axes[0], color=["#8d99ae", "#2a9d8f"]
    )
    axes[0].axhline(
        float(
            context.protocol["evaluation"]["representation_equivariance"][
                "maximum_error_margin"
            ]
        ),
        color="#c44536",
        linestyle="--",
        label="registered margin",
    )
    axes[0].set_ylabel("Strict token equivariance error")
    axes[0].set_xlabel("")
    axes[0].set_title("Held-out learned versus paired initialization")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend()

    for variant, group in paired.groupby("variant", sort=True):
        axes[1].scatter(group["initial"], group["learned"], alpha=0.45, label=variant)
    maximum = float(max(paired[["initial", "learned"]].max()))
    axes[1].plot([0.0, maximum], [0.0, maximum], color="black", linestyle="--")
    axes[1].set_xlabel("Initial-encoder error")
    axes[1].set_ylabel("Learned-encoder error")
    axes[1].set_title("Paired source/checkpoint diagnostic")
    axes[1].legend(title="Variant")
    return _finish(figure, context)


def report_dashboard(
    context: ExperimentContext,
    report: dict[str, Any],
    *,
    output_path=None,
):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    checkpoint = report["checkpoint_bootstrap"]
    predictive = checkpoint[
        checkpoint["comparison_type"].isin(["absolute_primary", "absolute_constructed"])
    ]
    labels = [
        "Native single/free" if value == "absolute_primary" else "Constructed odd/zero"
        for value in predictive["comparison_type"]
    ]
    estimates = predictive["estimate"].to_numpy(dtype=float)
    lower = estimates - predictive["ci95_low"].to_numpy(dtype=float)
    upper = predictive["ci95_high"].to_numpy(dtype=float) - estimates
    axes[0, 0].bar(labels, estimates, color=["#457b9d", "#6d597a"][: len(labels)])
    axes[0, 0].errorbar(
        np.arange(len(labels)), estimates, yerr=np.vstack((lower, upper)), fmt="none", color="black"
    )
    axes[0, 0].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 0].set_ylabel("Mean-checkpoint source-balanced R²")
    axes[0, 0].set_title("Absolute predictive utility gates")
    axes[0, 0].tick_params(axis="x", rotation=15)

    representation = report["representation_bootstrap"]
    representation = representation[
        representation["comparison_type"].isin(
            [
                "absolute_learned_strict_equivariance",
                "absolute_initial_strict_equivariance",
            ]
        )
    ].copy()
    representation["label"] = (
        representation["variant_a"].astype(str)
        + " / "
        + representation["state_a"].astype(str)
    )
    axes[0, 1].bar(
        representation["label"],
        representation["estimate"],
        color=["#2a9d8f" if state == "learned" else "#8d99ae" for state in representation["state_a"]],
    )
    axes[0, 1].errorbar(
        np.arange(len(representation)),
        representation["estimate"],
        yerr=np.vstack(
            (
                representation["estimate"] - representation["ci95_low"],
                representation["ci95_high"] - representation["estimate"],
            )
        ),
        fmt="none",
        color="black",
    )
    axes[0, 1].axhline(
        float(
            context.protocol["evaluation"]["representation_equivariance"][
                "maximum_error_margin"
            ]
        ),
        color="#c44536",
        linestyle="--",
    )
    axes[0, 1].set_ylabel("Strict token error")
    axes[0, 1].set_title("Direct representation audit")
    axes[0, 1].tick_params(axis="x", rotation=20)

    native = report["native_symmetry_bootstrap"]
    native = native[native["comparison_type"] == "absolute_native_learned_symmetry"]
    axes[1, 0].bar(native["variant_a"], native["estimate"], color="#f4a261")
    axes[1, 0].errorbar(
        np.arange(len(native)),
        native["estimate"],
        yerr=np.vstack(
            (
                native["estimate"] - native["ci95_low"],
                native["ci95_high"] - native["estimate"],
            )
        ),
        fmt="none",
        color="black",
    )
    axes[1, 0].axhline(
        float(
            context.protocol["evaluation"]["decision_rules"][
                "native_symmetry_normalized_error_margin"
            ]
        ),
        color="#c44536",
        linestyle="--",
    )
    axes[1, 0].set_ylabel("Per-checkpoint normalized output error")
    axes[1, 0].set_title("Native output symmetry (no seed cancellation)")
    axes[1, 0].tick_params(axis="x", rotation=15)

    diagnostics = report["summary"]["empirical_diagnostics"]
    gate_items = [
        (key, value)
        for key, value in diagnostics.items()
        if isinstance(value, bool)
    ]
    gate_labels = [key.replace("_", " ") for key, _ in gate_items]
    gate_values = [1.0 for _ in gate_items]
    axes[1, 1].barh(
        np.arange(len(gate_items)),
        gate_values,
        color=["#2a9d8f" if value else "#c44536" for _, value in gate_items],
    )
    axes[1, 1].set_yticks(np.arange(len(gate_items)), gate_labels, fontsize=8)
    axes[1, 1].set_xlim(0.0, 1.0)
    axes[1, 1].set_xticks([])
    axes[1, 1].invert_yaxis()
    for index, (_, value) in enumerate(gate_items):
        axes[1, 1].text(
            0.5,
            index,
            "PASS" if value else "NOT SUPPORTED",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            weight="bold",
        )
    axes[1, 1].set_title("Registered empirical diagnostics")
    figure.suptitle("Laterality evidence dashboard", weight="bold")
    _finish(figure, context)
    if output_path is not None:
        destination = str(output_path)
        figure.savefig(destination, format="svg" if destination.endswith(".svg") else None)
    return figure


def external_gate_figure(context: ExperimentContext, status: dict[str, Any]):
    import matplotlib.pyplot as plt

    validated = str(status.get("status", "")).startswith("manifest contract validated")
    figure, axis = plt.subplots(figsize=(8.5, 2.2))
    axis.barh(
        [0],
        [1],
        color="#2a9d8f" if validated else "#c44536",
    )
    axis.text(
        0.5,
        0,
        "MANIFEST PREREQUISITES VALIDATED — EVALUATION NOT RUN"
        if validated
        else "EXTERNAL EVALUATION BLOCKED / NOT RUN",
        ha="center",
        va="center",
        color="white",
        weight="bold",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("Optional subject-indexed external gate")
    for spine in axis.spines.values():
        spine.set_visible(False)
    return _finish(figure, context)


__all__ = [
    "cohort_figure",
    "evaluation_figure",
    "external_gate_figure",
    "governance_figure",
    "report_dashboard",
    "split_figure",
    "training_figure",
]
