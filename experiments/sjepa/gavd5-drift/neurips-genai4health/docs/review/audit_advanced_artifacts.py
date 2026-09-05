"""Read-only cross-check of notebook 08 artifacts and notebook 05b simulation.

Run from the experiment root with .venv/Scripts/python.exe.
This neither trains models nor writes upstream experiment artifacts.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "work/artifacts/real"
CHECKPOINTS = ART / "checkpoints"
sidecar_path = CHECKPOINTS / "sjepa_outer_fold_0_seed_42_jepa_vicreg.json"
sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
manifest = pd.read_csv(ART / "evaluation_protocol/eligible_sequence_manifest.csv")
sequence_to_video = manifest.set_index("sequence_id")["video_id"]
reference_hashes = {}
results = {"checkpoint_checks": [], "cached_cosines": [], "final_source_details": {}, "simulation_prediction_r2": []}

entries = [*sidecar["stage_checkpoints"], {
    "path": sidecar["checkpoint_path"],
    "sha256": sidecar["checkpoint_sha256"],
    "stage": "final",
}]
for entry in entries:
    path = CHECKPOINTS / Path(entry["path"]).name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == entry["sha256"], path
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("contract_version", "mode", "outer_fold", "model_seed",
                "manifest_sha256", "split_sha256", "split_version",
                "role_video_ids", "role_sequence_ids"):
        assert checkpoint[key] == sidecar[key], (path, key)
    fit = set(checkpoint["encoder_fit_video_ids"])
    selection = set(checkpoint["checkpoint_selection_video_ids"])
    roles = {key: set(value) for key, value in checkpoint["role_video_ids"].items()}
    assert fit <= roles["train"]
    assert selection <= roles["validation"]
    assert checkpoint["test_tensors_loaded"] is False
    reference_hashes[entry["stage"]] = digest
    results["checkpoint_checks"].append({
        "stage": entry["stage"], "sha256": digest,
        "fit_sources": len(fit), "selection_sources": len(selection),
        "roles": {key: len(value) for key, value in roles.items()},
        "hash_and_role_checks": "PASS",
    })

cache_dir = ART / "fold_evaluation/outer_fold_0/embedding_cache"
report = json.loads((cache_dir.parent / "normal_anchor_drift_seed_42.json").read_text())
for role in ("train", "validation", "test"):
    reference_path = next(cache_dir.glob(f"normal_{role}_stage_00_*.npz"))
    with np.load(reference_path, allow_pickle=False) as data:
        reference = data["embeddings"].copy()
        ref_meta = json.loads(data["metadata_json"].item())
    assert ref_meta["checkpoint_sha256"] == reference_hashes[0]
    for path in sorted(cache_dir.glob(f"normal_{role}_stage_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            embeddings = data["embeddings"].copy()
            meta = json.loads(data["metadata_json"].item())
        assert meta["sequence_ids"] == ref_meta["sequence_ids"]
        assert meta["role"] == role
        assert meta["manifest_sha256"] == sidecar["manifest_sha256"]
        assert meta["split_sha256"] == sidecar["split_sha256"]
        stage_number = int(path.name.split("_stage_")[1].split("_")[0])
        assert meta["checkpoint_sha256"] == reference_hashes[
            "final" if stage_number == 99 else stage_number
        ]
        cosines = torch.nn.functional.cosine_similarity(
            torch.as_tensor(reference), torch.as_tensor(embeddings), dim=1
        ).numpy()
        video_ids = sequence_to_video.loc[meta["sequence_ids"]].to_numpy()
        sources = pd.DataFrame({"video_id": video_ids, "cosine": cosines}).groupby("video_id")["cosine"].mean()
        value = float(sources.mean())
        if stage_number == 99:
            details = pd.DataFrame({"video_id": video_ids, "cosine": cosines}).groupby("video_id").agg(
                sequences=("cosine", "size"), mean_cosine=("cosine", "mean")
            ).reset_index()
            assert np.isclose(np.average(details["mean_cosine"], weights=details["sequences"]), cosines.mean())
            results["final_source_details"][role] = details.to_dict(orient="records")
            metric_name = f"{role}_source_equal_anchor_cosine"
            assert np.isclose(value, report["final_test"][metric_name], atol=1e-7)
        else:
            expected = [row for row in report["development_drift"]
                        if row["role"] == role and row["stage"] == stage_number]
            if expected:
                assert np.isclose(value, expected[0]["source_equal_anchor_cosine"], atol=1e-7)
        results["cached_cosines"].append({
            "role": role, "stage": stage_number,
            "sequences": len(embeddings), "sources": len(sources),
            "source_equal_mean": value,
            "sequence_equal_mean": float(cosines.mean()),
            "source_min": float(sources.min()), "source_max": float(sources.max()),
            "saved_metric_check": "PASS",
        })

# Independently reproduce notebook 05b's illustrative scatter generator.
# It uses a squared-correlation variance formula, although plot labels say R2.
for index, requested in enumerate((0.44, 0.41, 0.12, 0.47)):
    rng = np.random.default_rng(42 + index)
    truth = rng.normal(0, 1, 40)
    predicted = truth + rng.normal(0, np.sqrt((1 - requested) / requested), 40)
    prediction_r2 = 1 - np.square(truth - predicted).sum() / np.square(truth - truth.mean()).sum()
    results["simulation_prediction_r2"].append({
        "figure_label_r2": requested,
        "actual_prediction_r2": float(prediction_r2),
        "actual_squared_correlation": float(np.corrcoef(truth, predicted)[0, 1] ** 2),
    })

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, help="Optional derived JSON report path within this review directory.")
args = parser.parse_args()
if args.output:
    output = args.output.resolve()
    if output.parent != Path(__file__).resolve().parent:
        raise ValueError("Derived output must be in the same review directory as this script.")
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(json.dumps(results, indent=2))
