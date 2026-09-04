#!/usr/bin/env python3
"""Run one CPU update-budget smoke of the detached SG-JEPA objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gavd6_sjepa.research_directions.reflection_equivariance.amass_core11_training_pipeline import core11_train_config
from gavd6_sjepa.research_directions.reflection_equivariance.jepa_model_architecture import build_model
from .laterality_gauge_training_pipeline import GaugeWindowDataset, fit_sg
from .laterality_corruption_inference import (
    SequenceGaugeConfig,
    TwoStateDurationModel,
    calibrate_duration_temperature,
)
from .laterality_sequence_benchmarking import (
    _continuity_logits,
    _fit_continuity_head,
    _identity_partition,
    make_synthetic_examples,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = SequenceGaugeConfig(
        clean_probability=0.20,
        global_probability=0.0,
        local_probability=0.60,
        repeated_probability=0.20,
    )
    examples = make_synthetic_examples(
        identities=12, frames=64, seed=args.seed, config=config
    )
    fit_examples, calibration_examples, validation_examples = _identity_partition(examples)
    head = _fit_continuity_head(fit_examples, config)
    duration = TwoStateDurationModel(max_duration=8)
    calibration_logits = _continuity_logits(head, calibration_examples, config)
    temperature = calibrate_duration_temperature(
        duration,
        calibration_logits,
        [example.path for example in calibration_examples],
    )
    train = GaugeWindowDataset(
        fit_examples,
        _continuity_logits(head, fit_examples, config),
        duration,
        temperature,
        mode="uncorrected",
        config=config,
    )
    validation = GaugeWindowDataset(
        validation_examples,
        _continuity_logits(head, validation_examples, config),
        duration,
        temperature,
        mode="uncorrected",
        config=config,
    )
    model = build_model(
        core11_train_config("smoke"), "reflection_equivariant", args.seed
    )
    result = fit_sg(
        model,
        train,
        validation,
        torch.device("cpu"),
        seed=args.seed,
        output_dir=output,
        epochs=1,
        uniform=False,
    )
    payload = {
        "synthetic_smoke": True,
        "test_split_evaluated": False,
        "temperature": temperature,
        **result,
    }
    (output / "smoke_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
