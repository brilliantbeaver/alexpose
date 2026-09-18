# Paired synthetic restoration notebooks

These nine notebooks explain and execute the development workflow through the study modules. They evaluate offline 2D body-12 track restoration and the additional value of paired JEPA over coordinate learning. They do not provide clinical validation or open confirmation labels.

| Notebook | Workflow stage | Question |
|---|---|---|
| [00 Artifact audit](00_artifact_audit.ipynb) | `audit` | What does the retained pilot establish? |
| [01 Paired data](01_paired_data.ipynb) | `data` | Are pairs, masks, times and source groups valid? |
| [02 Image adaptation](02_image_adaptation.ipynb) | `adaptation` | What evidence is missing for the separate image-model gate? |
| [03 Information ladder](03_information_ladder.ipynb) | `information` | What can unchanged tracks and filters achieve? |
| [04 Coordinate versus JEPA](04_coordinate_vs_jepa.ipynb) | `direct`, `jepa` | Does the latent objective add value beyond coordinate supervision? |
| [05 Evaluation](05_evaluation.ipynb) | `evaluate` | Do accuracy gains preserve supported motion? |
| [06 Optional gates](06_optional_gates.ipynb) | `optional` | Is there evidence to reopen personalization or add video? |
| [07 Development snapshot](07_development_snapshot.ipynb) | `freeze` | Can development artifacts be recorded while confirmation stays closed? |
| [08 Evidence report](08_evidence_report.ipynb) | `report` | Can claims be reconstructed from saved predictions and receipts? |

Read the [protocol](../../docs/studies/synthetic-training-v2/protocol.md), [primary-source literature ledger](../../docs/studies/synthetic-training-v2/literature.md) and [independent protocol review](../../docs/studies/synthetic-training-v2/protocol-review.md) first. SmoothNet, PoseBERT, DeciWatch and MotionBERT already establish major parts of temporal refinement and corrupted-pose reconstruction. The practical MLP here is **SmoothNet-style**, not an exact reproduction. The controlled question is whether aligned synthetic targets and the latent objective add measurable value without erasing timing, amplitude or side-specific variation.

## Configuration and execution

Every notebook requires `STV2_CONFIG` pointing to a JSON file accepted by `RunConfig.load`. There is no implicit default configuration. Use a unique `run_id` whenever configuration, protocol, source manifests or tracked study code change. Run from inside the repository, or set `GAVD6_ROOT` to the repository root. Relative bundle and artifact paths resolve from that root. Keep configurations outside the canonical notebook directory.

From the repository root, choose an existing interpreter explicitly. The examples use `.venv/bin/python`; source HAIC execution must instead use the verified Torch **2.6.0+cu124** and torchvision **0.21.0+cu124** environment. Do not run an unqualified root `uv sync` on HAIC.

Create a small CPU fixture configuration in `/tmp`, choosing a fresh run ID if needed:

```sh
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
from pathlib import Path
from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig
cfg = RunConfig.fixture("notebook-fixture-001")
Path("/tmp/stv2-fixture.json").write_text(json.dumps(cfg.as_dict(), indent=2) + "\n")
PY
```

Execute the stage graph, then execute notebooks with the same explicit configuration and interpreter:

```sh
.venv/bin/python scripts/research_directions/synthetic_training_v2/run.py \
  --config /tmp/stv2-fixture.json --stage all
.venv/bin/python scripts/research_directions/synthetic_training_v2/execute_notebooks.py \
  --config /tmp/stv2-fixture.json --python .venv/bin/python
```

The notebook executor supplies `STV2_CONFIG` and preserves the canonical notebooks. With Jupyter, export `STV2_CONFIG` in the launching environment and select the same interpreter's kernel. Run notebooks in table order; each fresh kernel loads artifact receipts, not another notebook's variables. Repeating a completed stage with the same identity verifies artifact hashes and reuses its result. Outputs and executed copies belong under the run's artifact directory.

The fixture is an analytic software case with tiny capacity and short fits. It can test the path from data contracts to saved predictions and report reconstruction. It cannot rank scientific methods or make a scientific gate pass. Stage 00 separately reconstructs retained historical source aggregates.

## Source runs and evidence boundaries

A source configuration uses `mode: "source"` and an audited `bundle` path, plus explicit model, arms, seeds and compute scope. The same commands accept that configuration. Source preparation is a prerequisite: the notebooks do not obtain licenses, render motions, run pose extractors, certify anatomy or create independent temporal annotations. Large source fitting requires those reviews and the measured all-stage GPU scope; the default authorized GPU budget is zero. No notebook submits jobs.

Gate A remains a recorded pending image-adaptation comparison. Optional personalization and video branches are recorded without fitting. The current evaluation leaves decisive Gate B and real transfer insufficient until their missing repeated-seed, preservation, reference and margin evidence exists. The `freeze` stage in notebook 07 writes **only a development snapshot**. Independent confirmation requires a separate explicit operation, which is unavailable here.

Inspect `report.md`, `identity.json`, `effective-config.json`, `environment.json`, `receipts/`, `fits/`, `predictions/` and `evaluation/` under the configured run root. Reports rebuild from prediction-level evidence; person groups, rather than render variants or frames, are the independent units. CPU timings do not estimate H100 throughput. Equal updates do not establish equal total compute.

## Canonical notebook maintenance

Edit [the builder](../../scripts/research_directions/synthetic_training_v2/build_notebooks.py), then regenerate output-free notebooks:

```sh
.venv/bin/python scripts/research_directions/synthetic_training_v2/build_notebooks.py
.venv/bin/python -m unittest tests.synthetic_training_v2.test_notebooks -v
```

Cell IDs, ordering and serialized content are deterministic. Notebook tests check nbformat schema, executable syntax, local links, stage ordering, explicit configuration and exact regeneration. Scientific computation lives under `src/gavd6_sjepa/research_directions/synthetic_training_v2/`; notebooks contain setup, stage calls and concise displays.
