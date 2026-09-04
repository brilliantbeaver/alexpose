"""Build the thin AMASS Core11 training notebook.

Run: uv run python scripts/research_directions/reflection_equivariance/build_reflection_amass_training_notebook.py
The generated notebook delegates to the package runner and is dry by default.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "notebooks" / "experiments" / "idea09_reflection_equivariance" / "08_amass_core11_training.ipynb"


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


cells = [
    markdown(r'''# Train the three JEPA variants on AMASS Core11

This notebook is a small launcher for the production training code in
`src/gavd6_sjepa`. It builds the 64-frame/32-stride Core11 window index and trains
the standard, paired-unconstrained, and reflection-equivariant variants.

Training is dry by default. Configure a smoke run before setting the opt-in flag:

```bash
export AMASS_RUN_ROOT=/path/to/amass-run
export AMASS_PROFILE=smoke
export AMASS_DEVICE=cuda
export AMASS_RUN_ID=amass-core11-smoke-v1
export AMASS_RUN_TRAINING=1
```
'''),
    markdown("## Locate and import the shared training package"),
    code(r'''
from pathlib import Path
import os
import sys


def find_notebook_root(start=None):
    start = Path(start or Path.cwd()).expanduser().resolve()
    candidates = []
    override = os.getenv("GAIT_PARITY_PROJECT_DIR")
    if override:
        candidates.append(Path(override).expanduser().resolve())
    relative_path = Path("experiments") / "sjepa" / "gavd6"
    for base in (start, *start.parents):
        candidates.extend((base, base / relative_path))
    for candidate in dict.fromkeys(candidates):
        if (candidate / "src" / "gavd6_sjepa" / "research_directions"
                / "reflection_equivariance" / "amass_training_entrypoint.py").is_file():
            return candidate
    searched = "\n - ".join(str(path) for path in dict.fromkeys(candidates))
    raise FileNotFoundError(
        "Could not locate experiments/sjepa/gavd6. "
        "Set GAIT_PARITY_PROJECT_DIR to that directory.\n"
        f"Searched:\n - {searched}"
    )


PROJECT_DIR = find_notebook_root()
SOURCE_DIR = PROJECT_DIR / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from gavd6_sjepa.research_directions.reflection_equivariance.amass_training_entrypoint import main

print({
    "project_dir": str(PROJECT_DIR),
    "run_training": os.getenv("AMASS_RUN_TRAINING", "0"),
    "profile": os.getenv("AMASS_PROFILE", "smoke"),
    "device": os.getenv("AMASS_DEVICE", "cuda"),
    "run_root": os.getenv("AMASS_RUN_ROOT"),
    "output_dir": os.getenv("AMASS_OUTPUT_DIR"),
})
'''),
    markdown("## Run the shared entrypoint"),
    code("main()"),
    markdown(r'''## Outputs

A training run writes a window index, run configuration, per-variant loss histories,
checkpoints, and a compact training manifest. Validation and test identities are indexed
but are not used by the optimizer.
'''),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {NOTEBOOK}")
