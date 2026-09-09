"""Bounded, real-GAVD throughput probe; never writes scientific checkpoints.

Run from the repository root with .venv-cuda/Scripts/python.exe. Timings exclude
initial CUDA warmup and the final snapshot. Use --profile for a Chrome trace.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch

SUITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_ROOT))
from laterality_extensions.masked_learning import LearningSettings, resolve_learning_device
from laterality_extensions.motion_gavd import gavd_plan, study_inputs
from laterality_extensions.motion_structured_training import train_mask_study


def benchmark(*, device="cuda", precision="fp32", steps=12, warmup=2,
              experiment="motion", output=None, profile=False):
    if steps < warmup + 3 or warmup < 1:
        raise ValueError("Need at least one warmup and three subsequent updates")
    device = str(resolve_learning_device(device))
    inputs = study_inputs(mode="gavd", folds=(0,), seeds=(42,), create_missing=False, log=None)
    plan = gavd_plan(inputs, device=device, precision=precision)
    settings = replace(LearningSettings(**plan["settings"]), steps=steps, confirm_real_run=True)
    ticks = []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    def progress(event):
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elif device == "mps":
            torch.mps.synchronize()
        ticks.append(time.perf_counter())
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    from contextlib import nullcontext
    profiler = torch.profiler.profile(activities=activities) if profile else nullcontext()
    started = time.perf_counter()
    with profiler:
        result = train_mask_study(inputs["datasets"][0], settings, experiment=experiment,
            precision=precision, progress=progress)
    wall = time.perf_counter() - started
    # Last callback includes copying final weights; omit it from steady-state time.
    intervals = np.diff(ticks)[warmup - 1:-1]
    first = next(iter(result["runs"].values()))
    record = {"purpose": "real GAVD performance validation; not a scientific result",
        "identity": result["identity"], "paired_controls": result["pairing"],
        "warmup_updates": warmup, "measured_updates": len(intervals), "wall_seconds": wall,
        "preparation_seconds": first["preparation_seconds"],
        "paired_step_seconds": intervals.tolist(),
        "median_paired_step_seconds": float(np.median(intervals)),
        "encoder_updates_per_second": len(result["runs"]) / float(np.median(intervals)),
        "resident_input_mib": first["resident_input_bytes"] / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.startswith("cuda") else None,
        "history": {name: run["history"].to_dict("records") for name, run in result["runs"].items()}}
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if profile:
            profiler.export_chrome_trace(str(output.with_suffix(".trace.json")))
            output.with_suffix(".profile.txt").write_text(profiler.key_averages().table(
                sort_by="self_cuda_time_total" if device.startswith("cuda") else "self_cpu_time_total", row_limit=30))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--experiment", choices=("motion", "regions"), default="motion")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    arguments = parser.parse_args()
    warnings.filterwarnings("ignore", message="enable_nested_tensor")
    result = benchmark(**vars(arguments))
    print(json.dumps({key: value for key, value in result.items() if key not in {"identity", "history"}}, indent=2))
