"""Synthetic mechanism probes at gate-v2 dimensions; never GAVD evidence.

The ordinary condition calls unchanged production training. Zero initialization
is a process-local intervention and leaves production code/contracts untouched.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from diagnose_future_innovation_fits import null_weights, row_basis
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import (
    ModelContract, code_fingerprint,
)
from gavd6_sjepa.research_directions.future_innovation import fi_residual_models as models
from gavd6_sjepa.shared_infrastructure.artifact_io_operations import sha256_file


class ZeroOutputHead(models.SkeletonResidualHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)


def planted_skeleton_probe(n_train, n_test, baseline_dim, width, updates, seed):
    """An easy paired positive control, with X=0 to isolate skeleton learning.

    The signal is constant over time: this does not validate shuffle sensitivity
    or recovery in the presence of high-dimensional competing baseline inputs.
    """
    rng = np.random.default_rng(811)
    signal = rng.uniform(-1, 1, size=n_train + n_test)
    skeleton = np.zeros((len(signal), 32, 33, 4), dtype=np.float32)
    skeleton[..., 0] = signal[:, None, None]
    skeleton[..., 2:] = 1
    x = np.zeros((len(signal), baseline_dim))
    target = signal[:, None] * rng.normal(size=(1, 256))
    result = []
    for arm in ("real-skeleton", "no-skeleton"):
        history = skeleton.copy()
        if arm == "no-skeleton":
            history[..., :3] = 0
        with patch.object(models, "SkeletonResidualHead", ZeroOutputHead):
            head, _ = models.train_head(
                history[:n_train], x[:n_train], target[:n_train],
                np.arange(n_train).astype(str), np.ones(256, dtype=bool),
                seed=seed, weight_decay=0.1, updates=[max(updates)],
                model_contract=ModelContract(width=width),
            )
        prediction = models.predict_head(head, history[n_train:], x[n_train:], "cpu")
        result.append({"arm": arm, "seed": seed,
                       "heldout_zero_correction_mse": float(np.mean(target[n_train:] ** 2)),
                       "heldout_fitted_mse": float(np.mean((target[n_train:] - prediction) ** 2))})
    return result


def run_probe(*, seeds=(7, 19, 31), n_train=40, n_test=256, baseline_dim=2382,
              width=64, updates=(25, 50, 100, 200)):
    torch.set_num_threads(1)
    rng = np.random.default_rng(260911)
    n = n_train + n_test
    x = rng.normal(size=(n, baseline_dim))
    x = (x - x[:n_train].mean(0)) / x[:n_train].std(0)
    skeleton = rng.normal(size=(n, 32, 33, 4)).astype(np.float32)
    skeleton[..., 2:] = 1
    residual = np.zeros((n, 256))
    ids = np.arange(n_train).astype(str)
    valid = np.ones(256, dtype=bool)
    contract = ModelContract(width=width)
    basis, _, _ = row_basis(x[:n_train])
    results = []
    for seed in seeds:
        for initialization in ("production", "zero_output"):
            intervention = (patch.object(models, "SkeletonResidualHead", ZeroOutputHead)
                            if initialization == "zero_output" else nullcontext())
            with intervention:
                head, history = models.train_head(
                    skeleton[:n_train], x[:n_train], residual[:n_train], ids, valid,
                    seed=seed, weight_decay=0.1, updates=updates, model_contract=contract,
                    validation={"skeleton": skeleton[n_train:], "x": x[n_train:],
                                "residual": residual[n_train:], "weights": np.ones(n_test)},
                )
            correction = models.predict_head(head, skeleton[n_train:], x[n_train:], "cpu")
            w_x = head.output.weight.detach().numpy()[:, width:]
            w_null = null_weights(w_x, basis)
            null_correction = x[n_train:] @ w_null.T
            results.append({
                "seed": seed, "initialization": initialization,
                "training_history": history,
                "heldout_residual_mse": float(np.mean(correction ** 2)),
                "heldout_null_component_mse": float(np.mean(null_correction ** 2)),
                "heldout_mse_after_null_removal": float(np.mean((correction - null_correction) ** 2)),
                "null_removal_max_training_change": float(np.max(np.abs(x[:n_train] @ w_null.T))),
                "parameters": sum(p.numel() for p in head.parameters()),
            })
    # A noisy linear teacher demonstrates why in-sample ridge errors can be tiny
    # even though held-out error remains large. No residual head is fitted here.
    teacher = x[:, :12] @ rng.normal(size=(12, 256)) / np.sqrt(12)
    teacher += rng.normal(size=(n, 256)) * 0.5
    ridge = []
    for alpha in contract.ridge_alphas:
        baseline = models.fit_baseline(x[:n_train], teacher[:n_train], ids, ids,
                                       alpha, contract.target_variance_tolerance)
        error = baseline.y_scaler.transform(teacher) - baseline.predict(x)
        ridge.append({"alpha": alpha,
                      "training_residual_mse": float(np.mean(error[:n_train] ** 2)),
                      "heldout_residual_mse": float(np.mean(error[n_train:] ** 2))})
    return {"synthetic": True, "evidence_scope": "Mechanism reproduction, not GAVD performance",
            "code_sha256": code_fingerprint(),
            "script_sha256": sha256_file(Path(__file__)),
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "n_train": n_train, "n_test": n_test, "baseline_dim": baseline_dim,
            "training_rank": len(basis), "null_dimensions": baseline_dim - len(basis),
            "zero_residual_probes": results, "noisy_teacher_ridge_probe": ridge,
            "planted_skeleton_probe_scope": "Easy sample-pairing positive control; X=0; no temporal-order signal",
            "planted_skeleton_probe": planted_skeleton_probe(
                n_train, n_test, baseline_dim, width, updates, seeds[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    result = run_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved synthetic mechanism probes to {args.output}")


if __name__ == "__main__":
    main()
