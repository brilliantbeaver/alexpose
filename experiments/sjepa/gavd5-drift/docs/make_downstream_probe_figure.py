#!/usr/bin/env python
"""Plot only current, lineage-checked downstream results.

The stale 159-row Lane C artifact is deliberately excluded.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "work" / "artifacts" / "real"
OUT = ROOT / "docs" / "figures"
EXPECTED_FP = "7d13841aceac9eda843d43ca8434193e294d2fa10a48b6c6d21f6413a6e457e2"
EXPECTED_SHA = "64008d77689cefa4beb51a0dcf5ed6cae743454134c163e9087f66510af4e7ad"
EXPECTED_FILES = {
    "classifier_contract.json": "bd7ee48accb7f31e51b2fffe6e8789a20fcf487ab68db18b45f21615bb8f45e0",
    "classifier_metrics.csv": "0202618fbb199473d56dbb4c9ae3815ae396704403d85c3b41380524044b7f1c",
    "missingness_only_classifier_metrics.csv": "e4ec206c1264052bda261f7f49bf905daaaf39ba04668ea41b728aafd286c389",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Downstream figure contract failed: {message}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for name, expected in EXPECTED_FILES.items():
        require(file_sha256(ART / name) == expected, f"{name} content hash changed")
    require(file_sha256(ART / "sjepa_curriculum_final.pt") == EXPECTED_SHA,
            "checkpoint file SHA mismatch")
    contract = json.loads((ART / "classifier_contract.json").read_text())
    require(contract["checkpoint_fingerprint"] == EXPECTED_FP, "wrong checkpoint fingerprint")
    require(contract.get("include_augmented_normal") is False, "added-normal data is enabled")
    require(contract.get("encoder_checkpoint") == "sjepa_curriculum_final.pt", "wrong checkpoint")
    all_train = set(contract["all_sequences_train_sequence_ids"])
    all_test = set(contract["all_sequences_test_sequence_ids"])
    exact_train = set(contract["exp5_exact_train_sequence_ids"])
    exact_test = set(contract["exp5_exact_test_sequence_ids"])
    require(len(all_train) == 438 and len(all_test) == 188 and not (all_train & all_test),
            "all-row split identity changed")
    require(len(all_train | all_test) == 626, "all-row split no longer covers 626 rows")
    require(len(exact_train) == 47 and len(exact_test) == 21 and not (exact_train & exact_test),
            "exact split identity changed")
    metrics = pd.read_csv(ART / "classifier_metrics.csv")
    missing = pd.read_csv(ART / "missingness_only_classifier_metrics.csv")

    all_row = metrics.loc[metrics["task"].eq("five_class_all_sequences")].iloc[0]
    exact = metrics.loc[metrics["task"].eq("five_class_exp5_exact")].iloc[0]
    miss = missing.loc[missing["lane"].eq("all_sequences")].iloc[0]
    labels = ["Model variant\nall 626", "Model variant\nexact 47/21", "Missingness only\nall 626"]
    accuracy = [all_row.accuracy, exact.accuracy, miss.accuracy]
    macro_f1 = [all_row.macro_f1, exact.macro_f1, miss.macro_f1]
    expected_accuracy = [0.9202127659574468, 0.8571428571428571, 0.44148936170212766]
    expected_f1 = [0.8985120869742677, 0.8607142857142858, 0.35468163310268574]
    require(np.allclose(accuracy, expected_accuracy, atol=1e-12, rtol=0), "accuracy scores changed")
    require(np.allclose(macro_f1, expected_f1, atol=1e-12, rtol=0), "macro-F1 scores changed")

    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    x = np.arange(3)
    width = 0.34
    bars_a = ax.bar(x - width / 2, accuracy, width, color="#3977A8", label="Accuracy")
    bars_f = ax.bar(x + width / 2, macro_f1, width, color="#3D8B7D", label="Macro-F1")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Score")
    ax.set_title("Current five-class probes are descriptive, not held-out performance")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    for bars in (bars_a, bars_f):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018,
                    f"{bar.get_height():.3f}", ha="center", fontsize=7.5)
    fig.text(0.5, 0.01, "All classifier test rows were used by the label-aware encoder; Lane C is excluded as stale.",
             ha="center", color="#B94E48", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"downstream_probe_results.{suffix}", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote downstream_probe_results.svg/.pdf/.png")


if __name__ == "__main__":
    main()
