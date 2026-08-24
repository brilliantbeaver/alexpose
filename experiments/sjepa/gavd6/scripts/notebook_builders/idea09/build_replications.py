"""Build opt-in replication launchers for the full-96 GaitParity protocols.

Run: uv run python scripts/notebook_builders/idea09/build_replications.py
It only writes notebooks; it never starts a training run.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = ROOT / "notebooks" / "experiments" / "idea09_reflection_equivariance"


def markdown(source: str) -> dict:
    source = source.strip("\n") + "\n"
    return {
        "cell_type": "markdown",
        "id": hashlib.sha256(("md:" + source).encode()).hexdigest()[:12],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    source = source.strip("\n") + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": hashlib.sha256(("code:" + source).encode()).hexdigest()[:12],
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook(profile: str, device: str, run_id: str) -> dict:
    title = "CPU" if profile == "cpu" else "CUDA GPU"
    profile_summary = (
        "64 frames, stride 64, 8-frame segments, width 32, two encoder layers, "
        "one predictor layer, batch size 8, and six epochs."
        if profile == "cpu"
        else "64 frames, stride 32, 4-frame segments, width 96, four encoder layers, "
        "two predictor layers, batch size 32, and 100 epochs."
    )
    cells = [
        markdown(f"""# Full-GAVD GaitParity replication — {title}

This launcher reproduces the strongest supported **{profile}** protocol on the locked
96-sequence GAVD pose cache. It executes 09c (contract), 09d (training), and 09e
(audit) for seeds 7, 19, and 31 under both exposure- and compute-matched allocation.

Configuration: {profile_summary}

This is local, transductive representation/geometry evidence only. It is not clinical
or held-out-video generalization evidence.
"""),
        markdown("""## Before running

1. Use the `python3` kernel created by `uv sync`.
2. Ensure a complete real pose cache exists: 12 normal, 9 Parkinson's, 12 stroke,
   47 myopathic, and 16 cerebral-palsy sequences.
3. Set `RUN_PIPELINE = True` only when ready. Leaving it `False` performs no training.

The launcher writes the executed child notebooks beside each run's artifacts so the
exact contract, training, and audit outputs are retained.
"""),
        code(f'''from pathlib import Path
import os
import sys

STARTUP_REVISION = "ide-safe-v2"
KERNEL_START_DIR = Path.cwd().resolve()

def find_notebook_root(start=None):
    start = Path(start or Path.cwd()).expanduser().resolve()
    relative_path = Path("experiments") / "sjepa" / "gavd6"
    candidates = []
    override = os.getenv("GAIT_PARITY_PROJECT_DIR")
    if override:
        candidates.append(Path(override).expanduser().resolve())
    for base in (start, *start.parents):
        candidates.extend((base, base / relative_path))
    for candidate in dict.fromkeys(candidates):
        if ((candidate / "gait_parity_jepa.py").is_file()
                and (candidate / "notebooks" / "experiments" / "idea09_reflection_equivariance"
                     / "01_encoder_contract.ipynb").is_file()):
            return candidate
    searched = "\\n - ".join(str(path) for path in dict.fromkeys(candidates))
    raise FileNotFoundError(
        "Could not locate experiments/sjepa/gavd6. "
        "Set GAIT_PARITY_PROJECT_DIR to that directory.\\n"
        f"Searched:\\n - {{searched}}"
    )

PROJECT_DIR = find_notebook_root()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from gait_parity_jepa import EXPECTED_COUNTS, load_gavd_records

RUN_PIPELINE = False  # Explicit opt-in. Change to True to start the 18 trainings.
RUN_ID = "{run_id}"
PROFILE = "{profile}"
DEVICE = "{device}"
SEEDS = "7,19,31"
MATCHING_REGIMES = ("exposure", "compute")

# Use a local gavd6 cache by default. Set this environment variable to a mounted,
# immutable 96-sequence cache if it lives elsewhere.
POSE_DIR = Path(os.getenv(
    "GAIT_PARITY_POSE_DIR",
    PROJECT_DIR / "work" / "artifacts" / "real" / "poses",
)).expanduser().resolve()

print({{
    "startup_revision": STARTUP_REVISION,
    "kernel_start_dir": str(KERNEL_START_DIR),
    "project_dir": str(PROJECT_DIR),
    "run_pipeline": RUN_PIPELINE,
    "run_id": RUN_ID,
    "profile": PROFILE,
    "device": DEVICE,
    "seeds": SEEDS,
    "matching_regimes": MATCHING_REGIMES,
    "pose_dir": str(POSE_DIR),
}})
'''),
        code('''if RUN_PIPELINE:
    if DEVICE == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("This replication launcher requires a CUDA-enabled PyTorch host.")
    if not POSE_DIR.is_dir():
        raise FileNotFoundError(f"Full pose cache is missing: {POSE_DIR}")
    records = load_gavd_records(POSE_DIR)
    counts = {condition: sum(record["condition"] == condition for record in records)
              for condition in EXPECTED_COUNTS}
    assert counts == EXPECTED_COUNTS, counts
    print(f"Validated {len(records)} locked GAVD sequences: {counts}")
else:
    print("Dry configuration only — no pose files loaded and no training started.")
'''),
        markdown("""## Execute the frozen three-notebook pipeline

For each matching regime, the launcher injects only the declared replication environment
into the child notebook kernels. `09c` writes the immutable contract, `09d` trains and
checkpoints the three variants for every seed, and `09e` audits representation health and
reflection geometry. The child notebooks are copied with their executed outputs into the
matching run directory.
"""),
        code('''if RUN_PIPELINE:
    import nbformat
    from nbclient import NotebookClient

    child_notebooks = (
        "03_gavd_contract.ipynb",
        "04_gavd_training.ipynb",
        "05_gavd_audit.ipynb",
    )
    base_environment = {
        "GAIT_PARITY_MODE": "real",
        "GAIT_PARITY_PROFILE": PROFILE,
        "GAIT_PARITY_DEVICE": DEVICE,
        "GAIT_PARITY_RUN_ID": RUN_ID,
        "GAIT_PARITY_SEEDS": SEEDS,
        "GAIT_PARITY_POSE_DIR": str(POSE_DIR),
    }

    original_environment = {
        key: os.environ.get(key)
        for key in (*base_environment, "GAIT_PARITY_MATCHING")
    }
    try:
        for matching in MATCHING_REGIMES:
            os.environ.update(base_environment)
            os.environ["GAIT_PARITY_MATCHING"] = matching
            output_dir = (PROJECT_DIR / "work" / "artifacts" / "gait_parity" /
                          "real" / RUN_ID / matching)
            executed_dir = output_dir / "executed_notebooks"
            executed_dir.mkdir(parents=True, exist_ok=True)
            print(f"\\n=== {matching}: 09c → 09d → 09e ===")
            for name in child_notebooks:
                notebook = nbformat.read(
                    PROJECT_DIR / "notebooks" / "experiments" /
                    "idea09_reflection_equivariance" / name,
                    as_version=4,
                )
                client = NotebookClient(
                    notebook,
                    timeout=None,
                    kernel_name="python3",
                    resources={"metadata": {"path": str(PROJECT_DIR)}},
                )
                client.execute()
                nbformat.write(notebook, executed_dir / name)
                print(f"completed {name}")
    finally:
        for key, value in original_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
else:
    print("Set RUN_PIPELINE = True in the configuration cell to execute this protocol.")
'''),
        markdown("""## Expected output and interpretation

Successful execution produces two training manifests containing eighteen model runs (three
variants, three seeds, and two matching regimes), checkpoints and histories for all seeds, and a checkpoint-health /
geometry audit. A passing audit demonstrates local feasibility and the declared reflection
geometry. It does not establish clinical validity, unseen-video generalization, or a
performance advantage without a predeclared held-out-video evaluation.
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


for profile, device, name, run_id in [
    ("cpu", "cpu", "06_cpu_replication.ipynb", "gavd96-cpu-strong-v1"),
    ("gpu", "cuda", "07_gpu_replication.ipynb", "gavd96-gpu-strong-v1"),
]:
    path = NOTEBOOK_DIR / name
    path.write_text(json.dumps(build_notebook(profile, device, run_id), indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
