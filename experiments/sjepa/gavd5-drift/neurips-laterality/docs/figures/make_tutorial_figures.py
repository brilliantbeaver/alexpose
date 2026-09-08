"""Reproduce the tutorial masking figure from the completed private predictions.

Run from the repository root:
    .venv/bin/python -B neurips-laterality/docs/figures/make_tutorial_figures.py

This reads saved results and the registered cohort without training or inference.
Only aggregate scores and training seeds enter the generated public assets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parents[1]
sys.path.insert(0, str(SUITE_ROOT))

from laterality_extensions.masked_learning import (  # noqa: E402
    load_learning_dataset,
    masking_implementation_digest,
    summarize_cross_fitted_predictions,
)

FOLDS = tuple(range(5))
SEEDS = tuple(range(42, 47))
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260907
VARIANTS = {"gait_targets": {"mask_policy": "gait"},
            "uniform_targets": {"mask_policy": "uniform"}}
REPRESENTATIONS = {"learned_encoder", "initial_encoder", "raw_pose", "training_mean"}
CONDITIONS = {
    "gait_pretraining": ("gait_targets", "learned_encoder"),
    "uniform_pretraining": ("uniform_targets", "learned_encoder"),
    "no_pretraining": ("gait_targets", "initial_encoder"),
    "direct_pose": ("gait_targets", "raw_pose"),
    "training_mean": ("gait_targets", "training_mean"),
}


def load_predictions() -> tuple[pd.DataFrame, dict]:
    """Require the complete declared grid, intact files, and exact held-out rows."""
    artifact_root = SUITE_ROOT / "artifacts" / "research_extensions" / "masking"
    manifests = sorted(artifact_root.glob("exploratory_real_*/manifest.json"))
    if len(manifests) != len(FOLDS) * len(SEEDS):
        raise ValueError("Expected exactly 25 retained real masking comparisons")
    datasets = {fold: load_learning_dataset(real=True, fold=fold) for fold in FOLDS}
    implementation = masking_implementation_digest()
    frames, completed, common_settings = [], set(), None
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("complete") is not True or manifest.get("synthetic") is not False
                or manifest.get("variants") != VARIANTS):
            raise ValueError("Expected complete real-data gait/uniform comparisons")
        if manifest.get("pairing") != {
            "same_initialization": True, "same_source_draws": True,
            "same_hidden_token_counts": True,
        }:
            raise ValueError("The saved comparison failed its pairing checks")
        expected_files = {"metrics.csv", "training.csv"} | {
            f"{variant}{suffix}" for variant in VARIANTS
            for suffix in (".pt", "_predictions.csv")
        }
        if set(manifest["files"]) != expected_files:
            raise ValueError("Unexpected or incomplete saved file inventory")
        for name, expected_digest in manifest["files"].items():
            with (manifest_path.parent / name).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected_digest:
                raise ValueError("A saved comparison file failed digest validation")

        runs = manifest["runs"]
        settings = runs["gait_targets"]["settings"]
        key = (settings["fold"], settings["seed"])
        if key not in {(fold, seed) for fold in FOLDS for seed in SEEDS} or key in completed:
            raise ValueError("Unexpected or duplicate fold/seed comparison")
        completed.add(key)
        shared = {name: value for name, value in settings.items()
                  if name not in {"fold", "seed", "mask_policy"}}
        if common_settings is None:
            common_settings = shared
        elif shared != common_settings:
            raise ValueError("Training settings differ across the completed grid")
        dataset = datasets[key[0]]
        test = dataset.test_rows
        paired_frames = {}
        history = pd.read_csv(manifest_path.parent / "training.csv")
        for variant, changes in VARIANTS.items():
            run = runs[variant]
            if run["settings"] != {**settings, **changes}:
                raise ValueError("The masking arms differ in undeclared settings")
            if (run["implementation_digest"] != implementation
                    or run["cohort_digest"] != dataset.cohort_digest
                    or run["split_digest"] != dataset.split_digest):
                raise ValueError("Saved implementation, cohort, or split lineage differs")
            partition = run["source_partition"]
            if (partition["train_sources"] != list(dataset.train_sources)
                    or partition["test_sources"] != list(dataset.test_sources)):
                raise ValueError("Saved source roles differ from the registered partition")
            for field in ("initial_state_digest", "source_draw_digest", "hidden_token_counts"):
                if run[field] != runs["gait_targets"][field]:
                    raise ValueError("Saved arms do not share the declared pairing")
            if history.loc[history.variant == variant, "step"].tolist() != list(range(1, 1201)):
                raise ValueError("Expected 1,200 completed updates in every arm")
            frame = pd.read_csv(manifest_path.parent / f"{variant}_predictions.csv",
                                dtype={"source_id": str, "sequence_id": str})
            if set(frame.representation) != REPRESENTATIONS:
                raise ValueError("The saved prediction controls are incomplete")
            for _, rows in frame.groupby("representation"):
                if (rows.sequence_id.tolist() != dataset.sequence_ids[test].astype(str).tolist()
                        or rows.source_id.tolist() != dataset.source_ids[test].astype(str).tolist()
                        or not np.allclose(rows.target, dataset.targets[test], rtol=0, atol=1e-12)
                        or not np.isfinite(rows.prediction).all()):
                    raise ValueError("Held-out prediction coverage or values do not match")
            paired_frames[variant] = frame
            frames.append(frame.assign(fold=key[0], seed=key[1], variant=variant))
        for representation in REPRESENTATIONS - {"learned_encoder"}:
            control_rows = [frame.loc[frame.representation == representation].reset_index(drop=True)
                            for frame in paired_frames.values()]
            pd.testing.assert_frame_equal(*control_rows)
    if completed != {(fold, seed) for fold in FOLDS for seed in SEEDS}:
        raise ValueError("The declared fold/seed grid is incomplete")
    if (common_settings["steps"] != 1200 or common_settings["ridge_alpha"] != 1.0
            or common_settings["ema_momentum"] != 0.999):
        raise ValueError("The tutorial's declared update budget, ridge, or EMA setting changed")
    return pd.concat(frames, ignore_index=True), common_settings


def paired_source_bootstrap(predictions: pd.DataFrame) -> dict:
    """Resample complete videos jointly across arms and the five fixed fitted seeds."""
    learned = predictions.loc[predictions.representation == "learned_encoder"].copy()
    reference = learned.loc[(learned.seed == SEEDS[0]) & (learned.variant == "gait_targets")]
    sources = sorted(reference.source_id.unique())
    truth = reference.assign(target_square=reference.target ** 2).groupby("source_id")
    target_mean = truth.target.mean().reindex(sources).to_numpy()
    target_square = truth.target_square.mean().reindex(sources).to_numpy()
    errors = learned.assign(squared_error=(learned.target - learned.prediction) ** 2,
                            absolute_error=abs(learned.target - learned.prediction))
    source_errors = errors.groupby(["variant", "seed", "source_id"])[
        ["squared_error", "absolute_error"]].mean()
    means = {variant: np.stack([source_errors.loc[(variant, seed)].reindex(sources).to_numpy()
                               for seed in SEEDS]).mean(axis=0)
             for variant in VARIANTS}
    source_difference = means["gait_targets"] - means["uniform_targets"]
    count = len(sources)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(count, np.repeat(1 / count, count),
                              size=BOOTSTRAP_REPLICATES) / count
    target_variance = weights @ target_square - (weights @ target_mean) ** 2
    r2_difference = -(weights @ source_difference[:, 0]) / target_variance
    mae_difference = weights @ source_difference[:, 1]
    return {
        "status": "Exploratory paired percentile bootstrap of saved held-out predictions",
        "replicates": BOOTSTRAP_REPLICATES,
        "random_seed": BOOTSTRAP_SEED,
        "resampling_unit": "Source video; all its clips, arms, and training seeds move together",
        "estimand": "Mean per-seed pooled metric for gait targets minus uniform targets",
        "conditioning": "Five fitted training seeds and the existing source split; no refitting",
        "excluded_uncertainty": "New training runs, new source splits, and analysis selection",
        "r2_ci95": np.quantile(r2_difference, [0.025, 0.975]).tolist(),
        "mae_ci95": np.quantile(mae_difference, [0.025, 0.975]).tolist(),
    }


def make_summary(predictions: pd.DataFrame, settings: dict) -> dict:
    scores = summarize_cross_fitted_predictions(predictions)
    if not ((scores.test_sources == 93) & (scores.test_sequences == 625) & (scores.folds == 5)).all():
        raise ValueError("Expected all 625 clips and 93 sources in every pooled score")
    for representation in ("raw_pose", "training_mean"):
        baseline = predictions.loc[(predictions.variant == "gait_targets")
                                   & (predictions.representation == representation)]
        rows = [baseline.loc[baseline.seed == seed, ["sequence_id", "target", "prediction"]]
                .sort_values("sequence_id").reset_index(drop=True) for seed in SEEDS]
        for repeated in rows[1:]:
            pd.testing.assert_frame_equal(rows[0], repeated)
    conditions = {}
    for name, (variant, representation) in CONDITIONS.items():
        rows = scores.loc[(scores.variant == variant) & (scores.representation == representation)]
        rows = rows.sort_values("seed")
        if rows.seed.tolist() != list(SEEDS):
            raise ValueError("A scored condition is missing a training seed")
        conditions[name] = {
            "mean_r2": float(rows.r2.mean()), "mean_mae": float(rows.mae.mean()),
            "seed_sd_r2": float(rows.r2.std(ddof=1)),
            "seed_sd_mae": float(rows.mae.std(ddof=1)),
            "per_seed": rows[["seed", "r2", "mae"]].to_dict(orient="records"),
        }
    gait = conditions["gait_pretraining"]
    uniform = conditions["uniform_pretraining"]
    differences = [{"seed": a["seed"], "r2": a["r2"] - b["r2"], "mae": a["mae"] - b["mae"]}
                   for a, b in zip(gait["per_seed"], uniform["per_seed"])]
    return {
        "schema": "neurips_laterality_tutorial_masking_summary/v1",
        "evidence": "Completed exploratory real-data grid; distinct from the registered study",
        "coverage": {"folds": list(FOLDS), "seeds": list(SEEDS), "comparisons": 25,
                     "trained_encoders": 50, "clips": 625, "source_videos": 93},
        "evaluation": "Frozen EMA teacher; training-source-only preprocessing and fixed ridge alpha=1",
        "aggregation": "Pool held-out folds per seed, weight each source video equally, then average seed metrics",
        "baseline_note": "Direct pose and training-mean predictions repeat identically across training seeds",
        "shared_training_settings": {name: value for name, value in settings.items()
                                     if name not in {"device", "confirm_real_run"}},
        "conditions": conditions,
        "gait_minus_uniform": {
            "mean_r2": gait["mean_r2"] - uniform["mean_r2"],
            "mean_mae": gait["mean_mae"] - uniform["mean_mae"],
            "per_seed": differences,
            "source_bootstrap": paired_source_bootstrap(predictions),
        },
    }


def make_figure(summary: dict) -> None:
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.2, 2.65))
    fig.subplots_adjust(left=0.32, right=0.98, bottom=0.26, top=0.78)
    rows = [("gait_pretraining", "Gait-target pretraining", "#0072B2"),
            ("uniform_pretraining", "Uniform-target pretraining", "#009E73"),
            ("no_pretraining", "No pretraining", "#667085")]
    for y, (condition, _, color) in zip((2, 1, 0), rows):
        values = summary["conditions"][condition]
        # Fixed vertical offsets expose all five seeds and their nearby mean.
        # Only horizontal position encodes a measured quantity.
        ax.scatter([row["r2"] for row in values["per_seed"]],
                   y + np.linspace(0.02, 0.22, len(SEEDS)),
                   s=29, color=color, alpha=0.8, edgecolors="white", linewidths=0.4, zorder=3)
        ax.scatter(values["mean_r2"], y - 0.18, marker="D", s=60, color=color,
                   edgecolors="#20252b", linewidths=0.9, zorder=4)
    ax.axvline(0, color="#525b66", linewidth=1.0)
    ax.set(xlim=(-1.05, 0.055), ylim=(-0.50, 2.50),
           yticks=(2, 1, 0), yticklabels=[row[1] for row in rows],
           xticks=np.arange(-1.0, 0.01, 0.2), xlabel="Source-balanced R² (higher is better)")
    ax.set_xticklabels(["−1.0", "−0.8", "−0.6", "−0.4", "−0.2", "0"])
    ax.grid(axis="x", color="#e8ebef", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.tick_params(axis="x", length=3, color="#9aa1aa")
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#b8bec6")
    legend = [Line2D([], [], marker="o", linestyle="none", color="#667085", markersize=4.5,
                     label="One training seed"),
              Line2D([], [], marker="D", linestyle="none", color="#667085", markersize=6,
                     markeredgecolor="#20252b", label="Mean of five seeds")]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 1.12),
              ncol=2, frameon=False, fontsize=9, handletextpad=0.4, columnspacing=1.6)
    for extension in ("svg", "pdf"):
        fig.savefig(HERE / f"tutorial_masking_results.{extension}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    predictions, settings = load_predictions()
    summary = make_summary(predictions, settings)
    make_figure(summary)
    (HERE / "tutorial_masking_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("Validated 25 paired comparisons and all 625 clips from 93 source videos.")
    print("Wrote tutorial_masking_results.svg, tutorial_masking_results.pdf, and tutorial_masking_summary.json.")


if __name__ == "__main__":
    main()
