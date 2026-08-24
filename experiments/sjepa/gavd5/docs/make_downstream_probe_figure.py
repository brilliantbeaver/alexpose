"""Plot accuracy and macro-F1 for the latest five-class downstream probes.

Run notebook 06 first. This script reads its fingerprint-bound artifacts and
refuses incomplete, mixed-checkpoint, or non-real runs. It intentionally keeps
the binary Lane C task out of the chart because its scores are not comparable
with the five-class probes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]

# Keep these values identical to docs/make_figures.py, the project's canonical
# Matplotlib figure generator.
COLORS = {
    "navy": "#17324D",
    "blue": "#3977A8",
    "teal": "#3D8B7D",
    "green": "#72A66A",
    "gold": "#D9A441",
    "orange": "#D97745",
    "red": "#B94E48",
    "purple": "#7562A8",
    "ink": "#25384A",
    "muted": "#64788A",
    "paper": "#F7F5EF",
    "grid": "#D9E0E6",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "figure.dpi": 180,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def default_artifact_dir() -> Path:
    configured = Path(
        os.getenv("GAVD_ARTIFACT_DIR", ROOT / "work" / "artifacts")
    ).expanduser()
    return configured if configured.name == "real" else configured / "real"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Downstream probe contract failed: {message}")


def load_probe_scores(artifacts: Path) -> tuple[pd.DataFrame, dict, str]:
    contract_path = artifacts / "classifier_contract.json"
    metrics_path = artifacts / "classifier_metrics.csv"
    lane_c_path = artifacts / "lane_c_video_disjoint_metrics.csv"
    for path in (contract_path, metrics_path, lane_c_path):
        require(path.is_file(), f"missing {path}")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(contract.get("mode") == "real", "only real-run scores may be plotted")
    require(contract.get("curriculum_complete") is True, "curriculum is incomplete")
    require(
        contract.get("conditions_seen")
        == ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"],
        "unexpected curriculum order",
    )
    require(contract.get("include_augmented_normal") is True, "latest run is not augmented")
    require(contract.get("feature_count") == 384, "expected 384 frozen-encoder features")

    checkpoint_name = contract.get("encoder_checkpoint", "")
    checkpoint_path = artifacts / checkpoint_name
    require(checkpoint_path.is_file(), f"missing checkpoint {checkpoint_path}")

    metrics = pd.read_csv(metrics_path)
    lane_c = pd.read_csv(lane_c_path)
    canonical_path = artifacts / "sequence_embeddings.parquet"
    augmented_path = artifacts / "augmented_normal_embeddings.parquet"
    require(canonical_path.is_file(), f"missing {canonical_path}")
    require(augmented_path.is_file(), f"missing {augmented_path}")
    canonical = pd.read_parquet(canonical_path)
    augmented = pd.read_parquet(augmented_path)
    require(len(canonical) == 96, "canonical embedding table does not have 96 rows")
    require(len(augmented) == 63, "augmented embedding table does not have 63 rows")
    require(augmented["video_id"].nunique() == 17, "augmented table does not have 17 videos")
    fingerprint = str(contract["checkpoint_fingerprint"])
    for name, frame in (("canonical", canonical), ("augmented", augmented)):
        require(
            set(frame["checkpoint_fingerprint"].astype(str)) == {fingerprint},
            f"{name} embeddings used another checkpoint fingerprint",
        )
    latent_columns = sorted(column for column in canonical if column.startswith("latent_"))
    require(len(latent_columns) == 384, "canonical embedding width is not 384")
    embedding_digest = hashlib.sha256()
    for row in canonical.sort_values("sequence_id").itertuples(index=False):
        embedding_digest.update(str(row.sequence_id).encode("utf-8"))
        values = np.asarray(
            [getattr(row, column) for column in latent_columns], dtype=np.float32
        )
        embedding_digest.update(np.ascontiguousarray(values).tobytes())
    require(
        embedding_digest.hexdigest() == contract.get("embedding_corpus_sha256"),
        "canonical embeddings do not match the contract hash",
    )
    run_files = [
        canonical_path,
        augmented_path,
        metrics_path,
        lane_c_path,
        contract_path,
    ]
    run_mtimes = [path.stat().st_mtime for path in run_files]
    require(
        max(run_mtimes) - min(run_mtimes) <= 120.0,
        "probe artifacts were not generated in one notebook run",
    )
    require(set(lane_c["encoder"].astype(str)) == {checkpoint_name}, "Lane C used another encoder")
    require(set(lane_c["n_sequences"].astype(int)) == {159}, "Lane C is not the 159-row run")

    a1_rows = metrics.loc[metrics["task"].eq("five_class_all_sequences")]
    a2_rows = metrics.loc[metrics["task"].eq("five_class_exp5_exact")]
    require(len(a1_rows) == 1, "expected one A1 five-class row")
    require(len(a2_rows) == 1, "expected one A2 five-class row")
    a1 = a1_rows.iloc[0]
    a2 = a2_rows.iloc[0]
    lane_five = lane_c.loc[
        lane_c["task"].eq("five_class_classifier_video_disjoint_encoder_transductive")
    ]
    require(len(lane_five) == 1, "expected one five-class Lane C row")
    lane_five = lane_five.iloc[0]

    require(a1["split"] == contract["splits"]["all_sequences"], "A1 split mismatch")
    require(a2["split"] == contract["splits"]["exp5_exact"], "A2 split mismatch")
    require(int(lane_five["n_folds"]) == 2, "five-class Lane C must use two folds")
    require(
        int(lane_five["test_seqs_seen_in_representation_training"]) == 159,
        "Lane C representation-exposure audit changed",
    )

    scores = pd.DataFrame(
        [
            {
                "probe": "All 96 examples\ndivided at random",
                "accuracy": float(a1["accuracy"]),
                "macro_f1": float(a1["macro_f1"]),
            },
            {
                "probe": "Matched comparison\n47 for training, 21 for testing",
                "accuracy": float(a2["accuracy"]),
                "macro_f1": float(a2["macro_f1"]),
            },
            {
                "probe": "Videos kept separate\naverage of two splits",
                "accuracy": float(lane_five["accuracy_mean"]),
                "macro_f1": float(lane_five["macro_f1_mean"]),
            },
            {
                "probe": "Videos kept separate\nall predictions combined",
                "accuracy": float(lane_five["accuracy_pooled_oof"]),
                "macro_f1": float(lane_five["macro_f1_pooled_oof"]),
            },
        ]
    )
    require(
        np.isfinite(scores[["accuracy", "macro_f1"]].to_numpy()).all(),
        "scores contain missing or infinite values",
    )
    require(
        scores[["accuracy", "macro_f1"]].stack().between(0.0, 1.0).all(),
        "scores fall outside [0, 1]",
    )
    return scores, contract, file_sha256(checkpoint_path)


def plot_probe_scores(scores: pd.DataFrame, contract: dict, checkpoint_sha: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.1, 3.15))
    x = np.arange(len(scores))
    width = 0.34
    accuracy_bars = ax.bar(
        x - width / 2,
        scores["accuracy"],
        width,
        color=COLORS["blue"],
        label="Accuracy (correct identifications)",
    )
    f1_bars = ax.bar(
        x + width / 2,
        scores["macro_f1"],
        width,
        color=COLORS["teal"],
        label="F1 score (balances mistakes and misses)",
    )

    ax.set_xticks(x, scores["probe"])
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Performance (higher is better)")
    ax.set_title("Five-condition classification using learned movement features")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.24)
    ax.set_axisbelow(True)

    for group in (accuracy_bars, f1_bars):
        for bar in group:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.022,
                f"{value:.1%}",
                ha="center",
                fontsize=6.8,
                color=COLORS["ink"],
            )

    fig.suptitle(
        "How accurately were walking conditions identified?",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
    )
    fig.text(
        0.5,
        0.045,
        "The first two tests can place clips from the same video on both sides. "
        "The last two separate videos, but those videos were still used earlier to learn the movement patterns.",
        ha="center",
        fontsize=6.4,
        color=COLORS["red"],
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.78)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=default_artifact_dir())
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()

    artifacts = args.artifact_dir.expanduser().resolve()
    figure_dir = args.figure_dir.expanduser().resolve()
    scores, contract, checkpoint_sha = load_probe_scores(artifacts)
    figure = plot_probe_scores(scores, contract, checkpoint_sha)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        figure.savefig(figure_dir / f"downstream_probe_results.{suffix}", bbox_inches="tight")
    figure.savefig(
        figure_dir / "downstream_probe_results.png",
        bbox_inches="tight",
        dpi=240,
    )
    plt.close(figure)

    print(scores.to_string(index=False))
    print(f"checkpoint_sha256={checkpoint_sha}")
    print(f"wrote {figure_dir / 'downstream_probe_results.png'}")


if __name__ == "__main__":
    main()
