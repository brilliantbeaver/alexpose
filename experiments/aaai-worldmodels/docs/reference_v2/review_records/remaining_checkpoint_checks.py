"""Close the saved-checkpoint chain for legacy, untrained, and pattern arms.

This independent replay performs inference only and never instantiates the
mutating study runner or fits a model.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import sys
import time

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / "experiments/aaai-worldmodels"
sys.path.insert(0, str(PROJECT))
from worldmodels.v2_data import load_cohort
from worldmodels import v2_models as vm
from worldmodels import restart_experiment as legacy

CORE = PROJECT / "artifacts/runs/reference-v2-core-v1"
OUT = PROJECT / "artifacts/reference-v2-independent-review"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    x, valid, meta, times = load_cohort()
    ledger = json.loads((CORE / "model_fit_ledger.json").read_text())
    selections = json.loads((CORE / "selection.json").read_text())
    refs = dict(np.load(CORE / "refs_predictions.npz", allow_pickle=False))
    masks = dict(np.load(CORE / "masks_predictions.npz", allow_pickle=False))
    counts = {"legacy": 0, "untrained_probe": 0, "pattern_only": 0}
    records = []
    for row in ledger.values():
        for role in row["roles"]:
            match = re.fullmatch(r"outer(\d+):ref_(own|bilateral|shifted|static):refit", role)
            probe = re.fullmatch(r"outer(\d+):untrained_probe:refit", role)
            pattern = re.fullmatch(r"outer(\d+):M4_pattern_only_seed27", role)
            if not (match or probe or pattern):
                continue
            fold = int((match or probe or pattern).group(1))
            ii = np.flatnonzero(meta.outer_fold.eq(fold))
            checkpoint = joblib.load(CORE / row["checkpoint"])
            assert sha(CORE / row["checkpoint"]) == row["checkpoint_sha256"]
            training = set(checkpoint["training_source_ids"])
            expected_training = set(meta.loc[meta.outer_fold.ne(fold) & meta.label.eq("normal"), "source_id"])
            assert training == expected_training
            if match:
                arm = match.group(2)
                candidates = [v for v in selections.values() if v.get("mode") == arm and set(v["training_sources"]) == training]
                assert len(candidates) == 1
                alphas = candidates[0]["selected_alphas"]
                legs = []
                for leg, alpha in enumerate(alphas):
                    state = dict(arm=arm, leg=leg, beta=1.0, model=checkpoint["legs"][leg][float(alpha)])
                    legs.append(legacy.predict_checkpoint(state, x[ii, :16], valid[ii, :16], meta.iloc[ii].window_id))
                prediction = np.concatenate(legs, axis=2).astype(np.float32)
                saved = refs[arm][ii]
                kind = "legacy"
            elif probe:
                features, anchor, _ = vm.causal_features(x[ii, :16], valid[ii, :16], times[ii, :16])
                model = vm.ResidualMLP(832, 32)
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                with torch.no_grad():
                    representation = model.encoder(torch.as_tensor(((features - checkpoint["input_mean"]) / checkpoint["input_scale"]).astype(np.float32))).numpy()
                ridge = checkpoint["ridge"]
                residual = (((representation - ridge["mean"]) / ridge["scale"]) @ ridge["coef"] + ridge["intercept"]).reshape(-1, 8, 8, 2)
                prediction = (residual + anchor[:, None]).astype(np.float32)
                saved = refs["untrained_probe"][ii]
                kind = "untrained_probe"
            else:
                prediction = vm.predict_forecaster(checkpoint, x[ii, :16], valid[ii, :16], times[ii, :16])
                saved = masks["mlp_pattern_only"][ii]
                kind = "pattern_only"
            np.testing.assert_array_equal(prediction, saved)
            counts[kind] += 1
            records.append(dict(role=role, windows=len(ii), checkpoint_sha256=row["checkpoint_sha256"], bitwise_equal=True))
    assert counts == dict(legacy=20, untrained_probe=5, pattern_only=5)
    # Rebuild all seed-mean mask forecasts without accessing the saved means.
    seed_means = 0
    for key in masks:
        if "_seed" in key or key == "mlp_pattern_only":
            continue
        seeds = [masks[f"{key}_seed{seed}"] for seed in (27, 41, 73)]
        np.testing.assert_array_equal(np.mean(seeds, axis=0).astype(np.float32), masks[key])
        seed_means += 1
    assert seed_means == 7
    # Independently reconstruct latest-point persistence and damped velocity.
    target_joints = (8, 10, 12, 14, 9, 11, 13, 15)
    anchor = np.zeros((len(x), 8, 2), dtype=float)
    velocity = np.zeros_like(anchor)
    latest = np.repeat(times[:, 15, None], 8, axis=1)
    for i in range(len(x)):
        for output, joint in enumerate(target_joints):
            known = np.flatnonzero(valid[i, :16, joint])
            if len(known):
                anchor[i, output] = x[i, known[-1], joint]
                latest[i, output] = times[i, known[-1]]
            if len(known) > 1:
                a, b = known[-2:]
                velocity[i, output] = (x[i, b, joint] - x[i, a, joint]) / (times[i, b] - times[i, a])
    persistence = np.repeat(anchor[:, None], 8, axis=1)
    dt = times[:, 15, None, None] + np.arange(1, 9)[None, :, None] * .1 - latest[:, None]
    damped = persistence + .5 * dt[..., None] * velocity[:, None]
    np.testing.assert_array_equal(persistence.astype(np.float32), refs["persistence"])
    np.testing.assert_array_equal(damped.astype(np.float32), refs["damped_velocity"])
    result = dict(identity="reference-v2-independent-remaining-checkpoint-replay-v1", status="passed", reviewed_at_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__), core_manifest_sha256=sha(CORE / "artifact_hashes.json"), trajectory_replays=counts, exact_seed_mean_forecasts=seed_means, exact_simple_reference_forecasts=2, windows=len(x), records=records, seconds=time.monotonic()-started, limitation="Inference replay of saved states; no retraining or clinical identity validation.")
    (OUT / "remaining_checkpoint_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "trajectory_replays", "exact_seed_mean_forecasts", "exact_simple_reference_forecasts", "seconds")}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
