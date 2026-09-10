"""Read-only independent audit of the retained Notebook 18 grid.

Run from any working directory. Prints JSON; creates no experiment artifacts.
The source bootstrap uses source-level sufficient statistics and shares each
resample across clips, conditions and seeds. New trained-minus-initial intervals
are exploratory analyses added on 2026-09-09, conditional on fitted models.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "artifacts/motion_structured/grids/292443b0fab5339f5da7ca566a85d6172ffc5b64abe5febf2546681a0152ff57"


def source_score(group):
    w = 1.0 / group.groupby("source_id")["target"].transform("size").to_numpy()
    y, p = group.target.to_numpy(), group.prediction.to_numpy()
    ybar = np.average(y, weights=w)
    return (1.0 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - ybar) ** 2),
            np.average(np.abs(y - p), weights=w))


def source_bootstrap(first, reference, seed=812, repetitions=2000):
    keys = ["seed", "sequence_id", "source_id", "fold", "target"]
    paired = first[keys + ["prediction"]].merge(reference[keys + ["prediction"]],
        on=keys, suffixes=("_first", "_reference"), validate="one_to_one")
    assert len(paired) == 3125
    paired["se_first"] = (paired.target - paired.prediction_first) ** 2
    paired["se_reference"] = (paired.target - paired.prediction_reference) ** 2
    paired["ae_difference"] = ((paired.target - paired.prediction_first).abs()
                              - (paired.target - paired.prediction_reference).abs())
    paired["target_squared"] = paired.target ** 2
    src = paired.groupby(["seed", "source_id"])[
        ["target", "target_squared", "se_first", "se_reference", "ae_difference"]].mean()
    sources = np.array(sorted(paired.source_id.unique()))
    seeds = sorted(paired.seed.unique())
    arrays = np.stack([src.loc[s].reindex(sources).to_numpy() for s in seeds])
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(sources), size=(repetitions, len(sources)), replace=True)
    mean = arrays[:, draws, :].mean(axis=2)
    denom = mean[:, :, 1] - mean[:, :, 0] ** 2
    delta_r2 = ((mean[:, :, 3] - mean[:, :, 2]) / denom).mean(axis=0)
    delta_mae = mean[:, :, 4].mean(axis=0)
    observed = arrays.mean(axis=1)
    r2 = ((observed[:, 3] - observed[:, 2]) / (observed[:, 1] - observed[:, 0] ** 2)).mean()
    return {"difference_r2": float(r2), "r2_ci95": np.quantile(delta_r2, [.025, .975]).tolist(),
            "difference_mae": float(observed[:, 4].mean()),
            "mae_ci95": np.quantile(delta_mae, [.025, .975]).tolist(),
            "repetitions": repetitions, "bootstrap_seed": seed}


def main():
    manifest = json.loads((GRID / "manifest.json").read_text())
    assert manifest["complete"]
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((GRID / name).read_bytes()).hexdigest() == digest, name
    predictions = pd.read_csv(GRID / "predictions.csv")
    keys = ["experiment", "condition", "representation", "observation", "seed"]
    assert len(predictions) == 125000
    assert predictions.available.all() and not predictions.synthetic.any()
    assert not predictions.duplicated(keys + ["sequence_id"]).any()
    assert predictions.groupby("sequence_id").source_id.nunique().eq(1).all()
    assert predictions.groupby("source_id").fold.nunique().eq(1).all()
    rows = []
    for key, group in predictions.groupby(keys):
        assert len(group) == 625 and group.source_id.nunique() == 93
        r2, mae = source_score(group)
        rows.append(dict(zip(keys, key)) | {"r2": r2, "mae": mae})
    computed = pd.DataFrame(rows)
    saved = pd.read_csv(GRID / "per_seed.csv")
    joined = computed.merge(saved, on=keys, validate="one_to_one", suffixes=("_audit", "_saved"))
    errors = {metric: float((joined[f"{metric}_audit"] - joined[f"{metric}_saved"]).abs().max())
              for metric in ("r2", "mae")}
    assert errors["r2"] < 1e-12 and errors["mae"] < 1e-12
    jobs = pd.read_csv(GRID / "jobs.csv")
    assert len(jobs) == 50 and not jobs.duplicated(["experiment", "fold", "seed"]).any()
    histories = []
    for job in jobs.itertuples():
        directory = Path(job.training_directory)
        for path in directory.glob("*_training.csv"):
            h = pd.read_csv(path)
            assert np.array_equal(h.step.to_numpy(), np.arange(1, 1201))
            histories.append({"experiment": job.experiment, "arm": path.stem.removesuffix("_training"),
                "initial_prediction_loss": float(h.masked_prediction_loss.iloc[0]),
                "final_prediction_loss": float(h.masked_prediction_loss.iloc[-1]),
                "initial_total_loss": float(h.loss.iloc[0]), "final_total_loss": float(h.loss.iloc[-1])})
    assert len(histories) == 125
    masks, learned = {}, {}
    for experiment, condition in [("motion", "mamp_motion"), ("motion", "robust_motion"),
                                  ("regions", "connected_region")]:
        first = predictions[(predictions.experiment == experiment) & (predictions.condition == condition)
                            & (predictions.representation == "pretrained_teacher__mean_motion")]
        reference = predictions[(predictions.experiment == experiment) & (predictions.condition == "uniform")
                                & (predictions.representation == "pretrained_teacher__mean_motion")]
        masks[f"{experiment}/{condition}-uniform"] = source_bootstrap(first, reference)
    for experiment, condition in [("motion", "uniform"), ("motion", "mamp_motion"),
                                  ("motion", "robust_motion"), ("regions", "uniform"),
                                  ("regions", "connected_region")]:
        base = predictions[(predictions.experiment == experiment) & (predictions.condition == condition)]
        first = base[base.representation == "pretrained_teacher__mean_motion"]
        reference = base[base.representation == "initial_online__mean_motion"]
        learned[f"{experiment}/{condition}"] = source_bootstrap(first, reference)
    diagnostic = pd.read_csv(GRID / "predictor_diagnostics.csv")
    diagnostic["gap"] = diagnostic.mismatched_target_mse - diagnostic.matched_target_mse_on_control_clips
    mismatch = diagnostic.groupby(["experiment", "condition"]).gap.agg(
        rows="size", mean="mean", positive=lambda s: int((s > 0).sum())).reset_index()
    report = {"grid": str(GRID), "grid_file_hashes_verified": len(manifest["files"]),
        "prediction_rows": len(predictions), "pooled_rows_recomputed": len(computed),
        "largest_absolute_score_discrepancy": errors, "histories_verified": len(histories),
        "arm_optimizer_updates": 1200 * len(histories),
        "training_history_means": pd.DataFrame(histories).groupby(["experiment", "arm"]).mean().reset_index().to_dict("records"),
        "recomputed_mask_intervals": masks,
        "exploratory_new_trained_minus_initial_intervals": learned,
        "predictor_correspondence": mismatch.to_dict("records")}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
