"""Build the revised, GAVD-only signed-laterality audit notebook.

Run this file from the repository root.  It intentionally creates a new
notebook and never modifies nb_05a_signed_laterality_probe.ipynb.
"""

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "nb_05a_gavd_signed_laterality_audit.ipynb"
CELL_NUMBER = 0


def markdown(text: str) -> dict:
    global CELL_NUMBER
    CELL_NUMBER += 1
    return {"cell_type": "markdown", "id": f"md-{CELL_NUMBER:02d}", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    global CELL_NUMBER
    CELL_NUMBER += 1
    return {"cell_type": "code", "id": f"code-{CELL_NUMBER:02d}", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


cells = [
markdown(r'''
# Notebook 05a (revised): GAVD signed-laterality representation audit

**This is not a clinical experiment.** It audits whether a historically exposed hybrid S-JEPA checkpoint retains an **input-derived signed coordinate excursion** in GAVD, and whether source-video-held-out readouts behave coherently under anatomical reflection and fixed sensor-frame yaw.

The independent unit is a **source video**, never a clip or a participant. GAVD has no participant identifiers, the historical encoder has been exposed to its lineage, and its target comes from the pose coordinates supplied to the model. Therefore every real result is labelled:

`transductive · source-video-grouped · signed coordinate excursion · hybrid JEPA checkpoint`

The stroke cohort remains the future decisive clinical experiment. This notebook intentionally adds no stroke or MoVi loader.
'''),
markdown(r'''
## 0. Frozen protocol and execution contract

All scientific operations live in `signed_laterality_gavd_protocol.py` and are independently tested in `tests/test_signed_laterality_gavd_protocol.py`; notebook cells orchestrate them but do not redefine them.

`GAVD_AUDIT_MODE=real` is fail-closed. It requires all of the following:

- `GAVD_AUDIT_ARTIFACT_ROOT`, e.g. the directory ending in `cache/artifacts`;
- `GAVD_AUDIT_CHECKPOINT`, an explicit checkpoint path;
- `GAVD_AUDIT_RUN_ID`, a new immutable output run name; and
- `GAVD_AUDIT_CHECKPOINT_SHA256`, the expected complete SHA-256 digest.

The only other mode is explicit `smoke`, which uses synthetic wiring data and watermarks every artifact. There is no automatic fallback from real data to smoke data.
'''),
code(r'''
from pathlib import Path
import json, os, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
while not (ROOT / "signed_laterality_gavd_protocol.py").is_file() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if not (ROOT / "signed_laterality_gavd_protocol.py").is_file():
    raise FileNotFoundError("Run the notebook from within the gavd5 repository.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from signed_laterality_gavd_protocol import (
    AUDIT_LABELS, AuditConfig, SJEPAGait, anatomical_reflection, build_result_bundle,
    coordinate_reference_features, encode_tokens, file_sha256, load_frozen_encoder,
    left_right_label_shuffle, load_gavd_records, make_smoke_records, mirror_metrics, nested_group_rankings,
    nuisance_features, prepare_for_encoder, regression_metrics, run_grouped_arm,
    signed_right_minus_left_excursion, source_balanced_regression_metrics,
    source_balanced_mirror_metrics, source_bootstrap_mean, source_group_folds, source_manifest,
    target_permutation, validate_rankings, yaw_rotate,
    laterality_features,
)

MODE = os.getenv("GAVD_AUDIT_MODE", "smoke").strip().lower()
if MODE not in {"real", "smoke"}:
    raise ValueError("GAVD_AUDIT_MODE must be 'real' or 'smoke'.")
ARTIFACT_ROOT = os.getenv("GAVD_AUDIT_ARTIFACT_ROOT")
CHECKPOINT_ENV = os.getenv("GAVD_AUDIT_CHECKPOINT")
RUN_ID = os.getenv("GAVD_AUDIT_RUN_ID")
EXPECTED_SHA = os.getenv("GAVD_AUDIT_CHECKPOINT_SHA256")
INPUT_FRAME = os.getenv("GAVD_AUDIT_INPUT_FRAME", "canonical_body")

if MODE == "real":
    if not all((ARTIFACT_ROOT, CHECKPOINT_ENV, RUN_ID, EXPECTED_SHA)):
        raise ValueError("Real mode requires artifact root, checkpoint, run ID, and expected checkpoint SHA-256.")
    CHECKPOINT_PATH = Path(CHECKPOINT_ENV).expanduser().resolve()
    actual_sha = file_sha256(CHECKPOINT_PATH)
    if actual_sha != EXPECTED_SHA.lower():
        raise ValueError("Checkpoint SHA-256 does not match the frozen expected identity.")
    OUTPUT_DIR = ROOT / "work" / "idea5_signed_laterality_audit" / RUN_ID
else:
    CHECKPOINT_PATH = None
    actual_sha = "SYNTHETIC / NOT GAVD"
    OUTPUT_DIR = ROOT / "work" / "idea5_signed_laterality_audit" / "smoke"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = AuditConfig(
    artifact_root=ARTIFACT_ROOT, mode=MODE, checkpoint_name=CHECKPOINT_PATH.name if CHECKPOINT_PATH else "synthetic",
    output_run_id=RUN_ID or "smoke", input_frame=INPUT_FRAME,
)
CONFIG.validate()
print(" | ".join(AUDIT_LABELS.values()))
print(f"mode={MODE}; output={OUTPUT_DIR}; input_frame={INPUT_FRAME}")
if MODE == "smoke":
    print("SYNTHETIC / NOT GAVD — all tables and figures below are plumbing checks only.")
'''),
markdown(r'''
## 1. Run protocol tests before any inference

The test suite verifies double reflection, odd/even targets, mask/confidence and bone preservation, source-disjoint folds, nested rankings, yaw wiring, and exact Arm-C oddness. It runs outside the notebook so correctness does not depend on cell order.
'''),
code(r'''
import subprocess
completed = subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_signed_laterality_gavd_protocol.py", "-q"],
    cwd=ROOT, capture_output=True, text=True,
)
print(completed.stdout + completed.stderr)
if completed.returncode:
    raise RuntimeError("Protocol tests failed; do not continue to data/model inference.")
'''),
markdown(r'''
## 2. Bind a frozen encoder and state its limitation

Real mode records the exact checkpoint hash and its embedded lineage metadata. The checkpoint is not required to be a particular historical fingerprint, but it must be explicitly selected and hashed. A changed fingerprint is a new audit, not an unnoticed continuation.

The canonical-body transform is the revised protocol default: pelvis-centered, robust leg-length scaled, and oriented by within-clip anatomical geometry. It is recorded because it may differ from the historical encoder's training preprocessing. If body-frame geometry is not credible for a particular cache, use the explicit `historical_center_scale` sensitivity policy and state that yaw is unavailable—not a silent camera-axis assumption.
'''),
code(r'''
if MODE == "real":
    encoder, encoder_config, checkpoint_metadata = load_frozen_encoder(CHECKPOINT_PATH)
    if checkpoint_metadata["mode"] != "real" or not checkpoint_metadata["curriculum_complete"]:
        raise ValueError("Selected checkpoint is not a completed real-data historical curriculum checkpoint.")
    if encoder_config["joints"] != 33 or encoder_config["coordinate_dim"] != 3:
        raise ValueError("Checkpoint does not use the frozen BlazePose-33 / xyz schema.")
    frames = CONFIG.frame_count or encoder_config["frames"]
    checkpoint_metadata = {**checkpoint_metadata, "path": str(CHECKPOINT_PATH), "sha256": actual_sha}
else:
    import torch
    torch.manual_seed(CONFIG.seed)
    encoder_config = {"frames": 64, "joints": 33, "coordinate_dim": 3, "segment_length": 4,
                      "embed_dim": 32, "encoder_depth": 1, "predictor_depth": 1, "heads": 4}
    model = SJEPAGait(**encoder_config).eval()
    encoder = model.target_encoder
    frames = encoder_config["frames"]
    checkpoint_metadata = {"path": "SYNTHETIC / NOT GAVD", "sha256": actual_sha,
                           "dataset_fingerprint": "synthetic random encoder"}

print(json.dumps(checkpoint_metadata, indent=2, default=str))
'''),
markdown(r'''
## 3. Freeze the primary and sensitivity cohorts

The primary audit is the 96-sequence canonical extraction cohort. The sensitivity audit appends only verified augmented-normal files and carries provenance in every row. It is **not** a normal-versus-abnormal analysis: the additional data are normal-only and follow a different acquisition/clip route.

The loader uses `manifest.csv` and `augmented_normal_pose_coverage.csv` as membership manifests, verifies each NPZ's sequence and source IDs, and never follows the stale absolute `csv_path` fields.
'''),
code(r'''
if MODE == "real":
    cohort_specs = [("canonical", "PRIMARY: common canonical extraction path"),
                    ("sensitivity_full", "SENSITIVITY ONLY: canonical + augmented normal with provenance")]
else:
    cohort_specs = [("smoke", "SYNTHETIC / NOT GAVD")]

def load_cohort(name):
    return make_smoke_records(CONFIG.seed) if name == "smoke" else load_gavd_records(ARTIFACT_ROOT, name)

for name, label in cohort_specs:
    records = load_cohort(name)
    table = pd.DataFrame([record.manifest_row() for record in records])
    print(f"\n{name}: {label}\n", table.groupby(["provenance", "condition"], dropna=False).size())
    print(f"sources={table.video_id.nunique()}, sequences={len(table)}")
'''),
markdown(r'''
## 4. Generate source-grouped OOF audit results

For each cohort, all source-video folds are fixed before fitting and written beside the results. Fit weights sum to one per source so a long source cannot dominate the head. The canonical-normal set has only one source, so no condition-level inference or stratified condition split is attempted.

Arms are descriptive GAVD audit controls, not a causal clinical comparison:

- **A:** historical frozen encoder + ordinary one-view head;
- **B:** same head trained with sign-aware reflected pairs;
- **C:** exact odd, zero-bias projected readout; its parity is a wiring assertion;
- **D:** parameter-matched unconstrained two-view readout with shared scalar `q` and only `a,b,c` extra parameters;
- **E:** expectedly strong raw-coordinate reference;
- **F:** fixed-seed random frozen-encoder floor;
- **G:** exactly reflection-invariant side-agnostic and acquisition-nuisance controls.
'''),
code(r'''
from collections import defaultdict
import torch

def audit_one_cohort(cohort_name, records):
    groups = np.asarray([record.video_id for record in records])
    folds = source_group_folds(records, CONFIG.outer_folds)
    # Reserved frozen artifact: GAVD has no participant-level label-budget analysis.
    rankings = nested_group_rankings(groups, CONFIG.seed)
    validate_rankings(rankings, groups)
    manifest = source_manifest(records, folds)
    (OUTPUT_DIR / f"{cohort_name}_source_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUTPUT_DIR / f"{cohort_name}_source_rankings.json").write_text(json.dumps(rankings, indent=2))

    y_historical = np.asarray([-signed_right_minus_left_excursion(record.raw) for record in records])
    y = -y_historical  # all new reporting: positive = right excursion greater than left
    prepared = [prepare_for_encoder(record.raw, frames, INPUT_FRAME) for record in records]
    prepared_mirror = [prepare_for_encoder(anatomical_reflection(record.raw), frames, INPUT_FRAME) for record in records]
    xyz = np.stack([item[0] for item in prepared])
    xyz_mirror = np.stack([item[0] for item in prepared_mirror])
    # Per-record transform gates execute before any model feature is consumed.
    for record in records:
        mirrored = anatomical_reflection(record.raw)
        if not np.allclose(anatomical_reflection(mirrored), record.raw, atol=2e-6):
            raise AssertionError(f"Double reflection failed: {record.sequence_id}")
        if not np.isclose(signed_right_minus_left_excursion(mirrored), -signed_right_minus_left_excursion(record.raw)):
            raise AssertionError(f"Odd target failed: {record.sequence_id}")

    token = encode_tokens(encoder, xyz)
    token_mirror = encode_tokens(encoder, xyz_mirror)
    learned, learned_mirror = np.stack([laterality_features(x) for x in token]), np.stack([laterality_features(x) for x in token_mirror])

    torch.manual_seed(CONFIG.seed + 1)
    random_encoder = SJEPAGait(**encoder_config).eval().target_encoder
    random_token = encode_tokens(random_encoder, xyz)
    random_token_mirror = encode_tokens(random_encoder, xyz_mirror)
    random_features, random_features_mirror = np.stack([laterality_features(x) for x in random_token]), np.stack([laterality_features(x) for x in random_token_mirror])

    raw_features = np.stack([coordinate_reference_features(record.raw[..., :3]) for record in records])
    raw_features_mirror = np.stack([coordinate_reference_features(anatomical_reflection(record.raw)[..., :3]) for record in records])
    shuffled_raw = [left_right_label_shuffle(record.raw, CONFIG.seed + index) for index, record in enumerate(records)]
    shuffled_features = np.stack([laterality_features(x) for x in encode_tokens(encoder, np.stack([
        prepare_for_encoder(raw, frames, INPUT_FRAME)[0] for raw in shuffled_raw]))])
    shuffled_features_mirror = np.stack([laterality_features(x) for x in encode_tokens(encoder, np.stack([
        prepare_for_encoder(anatomical_reflection(raw), frames, INPUT_FRAME)[0] for raw in shuffled_raw]))])
    # Symmetrization across actual encoder outputs, not pooling, makes G-side invariant by construction.
    side_invariant = (learned + learned_mirror) / 2.0
    if not np.allclose(side_invariant, (learned_mirror + learned) / 2.0, atol=1e-12):
        raise AssertionError("Side-agnostic control is not reflection invariant.")
    nuisance = np.stack([nuisance_features(record.raw) for record in records])
    nuisance_mirror = np.stack([nuisance_features(anatomical_reflection(record.raw)) for record in records])
    if not np.allclose(nuisance, nuisance_mirror, atol=1e-12):
        raise AssertionError("Nuisance feature contains left/right information.")

    arms = {
        "A_standard": ("A", learned, learned_mirror),
        "B_sign_aware": ("B", learned, learned_mirror),
        "C_exact_odd": ("C", learned, learned_mirror),
        "D_two_view_unconstrained": ("D", learned, learned_mirror),
        "E_coordinate_reference": ("E", raw_features, raw_features_mirror),
        "F_random_encoder": ("F", random_features, random_features_mirror),
        "G_left_right_shuffle": ("A", shuffled_features, shuffled_features_mirror),
        "G_side_invariant": ("G_side_agnostic", side_invariant, side_invariant),
        "G_nuisance": ("G_nuisance", nuisance, nuisance_mirror),
    }
    rows, oof = [], {}
    for name, (arm, original, reflected) in arms.items():
        result = run_grouped_arm(arm, original, reflected, y, groups, folds, CONFIG.inner_folds)
        oof[name] = (result, original, reflected)
        metrics = {**regression_metrics(y, result.prediction),
                   **source_balanced_regression_metrics(y, result.prediction, groups),
                   **mirror_metrics(result.prediction, result.mirrored_prediction),
                   **source_balanced_mirror_metrics(result.prediction, result.mirrored_prediction, groups),
                   "mean_alpha": float(np.mean(result.alpha_by_fold))}
        rows.append({"arm": name, **metrics})
    summary = pd.DataFrame(rows).sort_values("arm")
    if not np.allclose(oof["C_exact_odd"][0].mirrored_prediction, -oof["C_exact_odd"][0].prediction, atol=1e-10):
        raise AssertionError("Arm C failed exact OOF oddness wiring assertion.")

    # Destructive target permutation: it preserves source grouping but destroys target-feature pairing.
    permuted_y = target_permutation(y, groups, CONFIG.seed + 17)
    permuted = run_grouped_arm("A", learned, learned_mirror, permuted_y, groups, folds, CONFIG.inner_folds)
    permutation_row = {"permuted_target_oof_r2": regression_metrics(y, permuted.prediction)["r2_untruncated"]}

    # Fixed yaw test: inputs rotate after canonical preprocessing; targets do not change or get recomputed.
    yaw_predictions = {}
    yaw_rows = []
    if INPUT_FRAME == "canonical_body":
      for degrees in (-30.0, 30.0):
        yaw = np.stack([yaw_rotate(item, degrees) for item in xyz])
        yaw_mirror = np.stack([yaw_rotate(item, degrees) for item in xyz_mirror])
        yaw_features = np.stack([laterality_features(x) for x in encode_tokens(encoder, yaw)])
        yaw_features_mirror = np.stack([laterality_features(x) for x in encode_tokens(encoder, yaw_mirror)])
        for name in ("A_standard", "C_exact_odd", "D_two_view_unconstrained"):
            arm, _, _ = arms[name]
            yaw_predictions[(name, degrees)] = run_grouped_arm(
                arm, learned, learned_mirror, y, groups, folds, CONFIG.inner_folds,
                evaluation_original=yaw_features, evaluation_mirrored=yaw_features_mirror,
            ).prediction
      for name in ("A_standard", "C_exact_odd", "D_two_view_unconstrained"):
        clean = oof[name][0].prediction
        for degrees in (-30.0, 30.0):
            drift = np.abs(yaw_predictions[(name, degrees)] - clean) / (np.abs(yaw_predictions[(name, degrees)]) + np.abs(clean) + 1e-8)
            yaw_rows.append({"arm": name, "degrees": degrees, "source_balanced_yaw_drift": source_bootstrap_mean(records, drift)["estimate"],
                             "source_balanced_yaw_sign_flip_rate": source_bootstrap_mean(records, (yaw_predictions[(name, degrees)] * clean < 0).astype(float))["estimate"]})
    else:
      yaw_rows.append({"arm": "not_evaluated", "degrees": None, "status": "input frame lacks a saved canonical vertical axis"})
    bootstrap = {name: source_bootstrap_mean(records, np.abs(y - result.prediction)) for name, (result, _, _) in oof.items()}
    oof_table = pd.DataFrame({"sequence_id": [record.sequence_id for record in records], "video_id": groups,
                              "target_right_minus_left": y, **{name: result.prediction for name, (result, _, _) in oof.items()}})
    return summary, pd.DataFrame(yaw_rows), permutation_row, bootstrap, oof_table, y, oof, manifest
'''),
markdown(r'''
## 5. Tables, geometry plots, and result bundle

All decodability and reflection values below are out-of-fold. Metrics are descriptive; no legacy `R²` threshold, raw-coordinate percentage, or mirror-slope band determines “success.” Source-balanced values are the principal summaries. The raw-coordinate reference is expected to be strong because the target is coordinate-derived.
'''),
code(r'''
all_bundles = {}
for cohort_name, cohort_label in cohort_specs:
    records = load_cohort(cohort_name)
    summary, yaw_table, permutation, bootstrap, oof_table, y, oof, manifest = audit_one_cohort(cohort_name, records)
    print(f"\n{cohort_name}: {cohort_label}")
    display(summary.round(4))
    display(yaw_table.round(4))
    print("falsification:", permutation)
    print("descriptive source-block bootstrap MAE intervals:", bootstrap)

    result_dict = {row.arm: {key: float(value) for key, value in row.drop(labels="arm").items()} for _, row in summary.iterrows()}
    result_dict["falsification"] = permutation
    bundle = build_result_bundle(CONFIG, records, checkpoint_metadata, result_dict)
    bundle["cohort_role"] = cohort_label
    bundle["yaw"] = yaw_table.to_dict(orient="records")
    bundle["source_bootstrap_absolute_error"] = bootstrap
    bundle["oof_predictions_path"] = str(OUTPUT_DIR / f"{cohort_name}_oof_predictions.csv")
    bundle["frozen_protocol_artifacts"] = {
        "source_manifest": str(OUTPUT_DIR / f"{cohort_name}_source_manifest.json"),
        "source_manifest_sha256": file_sha256(OUTPUT_DIR / f"{cohort_name}_source_manifest.json"),
        "reserved_source_rankings": str(OUTPUT_DIR / f"{cohort_name}_source_rankings.json"),
        "reserved_source_rankings_sha256": file_sha256(OUTPUT_DIR / f"{cohort_name}_source_rankings.json"),
    }
    bundle["result_interpretation"] = (
        "A positive result indicates retention of an input-derived coordinate diagnostic by a historically "
        "exposed representation. It does not establish clinical laterality, affected-side recovery, diagnosis, "
        "participant generalization, or parity-aware clinical efficacy."
    )
    (OUTPUT_DIR / f"{cohort_name}_result_bundle.json").write_text(json.dumps(bundle, indent=2, default=str))
    summary.to_csv(OUTPUT_DIR / f"{cohort_name}_oof_summary.csv", index=False)
    oof_table.to_csv(OUTPUT_DIR / f"{cohort_name}_oof_predictions.csv", index=False)
    all_bundles[cohort_name] = bundle

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    standard = oof["A_standard"][0]
    axes[0].scatter(y, standard.prediction, alpha=.75)
    axes[0].axline((0, 0), slope=1, color="black", ls="--")
    axes[0].set(xlabel="right-minus-left signed coordinate excursion", ylabel="OOF Arm-A prediction",
                title="Source-grouped decodability")
    axes[1].scatter(standard.prediction, standard.mirrored_prediction, alpha=.75)
    axes[1].axline((0, 0), slope=-1, color="black", ls="--")
    axes[1].set(xlabel="OOF prediction on x", ylabel="same OOF head on M(x)", title="Reflection geometry")
    if MODE == "smoke":
        fig.text(.5, .5, "SYNTHETIC / NOT GAVD", ha="center", va="center", fontsize=24, alpha=.18, rotation=25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{cohort_name}_oof_audit.png", dpi=160)
    plt.show()
'''),
markdown(r'''
## 6. Licensed conclusion

This notebook may conclude only that the explicitly named historical checkpoint retains—or does not retain—a **coordinate-derived**, source-video-grouped signed signal under this frozen preprocessing and audit protocol. Arm C's exact oddness is a wiring property, not evidence that the encoder is reflection-equivariant. The canonical/sensitivity contrast is about provenance robustness, never disease differences.

The next study adds the participant-disjoint stroke cohort and independent force target; it is where parity-aware learning, low-label recovery, and clinical usefulness can be tested.
'''),
]

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
