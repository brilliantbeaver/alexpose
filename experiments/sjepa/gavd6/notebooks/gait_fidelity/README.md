# Gait Fidelity tutorials

These thirteen notebooks derive the calculations used by the HAIC implementation. Seven numbered tutorials and experiments A–E cover the core/full study; experiment F covers the separate JEPA response follow-up. Equations lead into small NumPy and PyTorch examples, with assertions that compare the visible calculation against production code or retained results. Start with the [HAIC guide](../../slurm/gait-fidelity/README.md) for a source experiment, or open notebook 00 locally to create a small CPU software fixture.

See [environment and readiness](ENVIRONMENT.md) for all required and automatically supplied variables, compatible interpreters, source-data prerequisites and the points where HAIC notebook execution must wait for Slurm jobs.

| Notebook | What you do |
| --- | --- |
| [00 · Start here](00_start_here.ipynb) | Trace the full pipeline, inspect settings and reconstruct the count of shared training phases. |
| [01 · Data and references](01_data_and_references.ipynb) | Audit full AMASS identity/duration selection and GAVD recording groups, project reference joints, and derive input-only normalization. |
| [02 · Masking and controls](02_masking_and_controls.ipynb) | Pack four-frame joint tokens, implement all five mask samplers and measure hidden-token exposure. |
| [03 · Architecture and experiment matrix](03_experiment_matrix.ipynb) | Follow token embeddings, Transformer features and residual coordinate prediction; inspect the saved core/full recipe set. |
| [04 · Losses, updates and execution](04_run_and_monitor.ipynb) | Calculate coordinate, feature and movement losses; inspect a scratch optimizer update and teacher averaging; launch the shared run. |
| [05 · Evaluation and visualization](05_evaluate_and_visualize.ipynb) | Reconstruct synthetic motion measurements and derive the training-only, recording-weighted GAVD label probe. |
| [06 · Verification and uncertainty](06_verify_and_write.ipynb) | Rebuild person-level summaries and the paired people/seed bootstrap, verify receipts and qualify conclusions. |

Read the five experiment tutorials after 00–06. Each contains a worked calculation for its comparison, followed by recipes and saved outputs from the central run. New source runs default to ten core recipes (30 final models); `--experiment-set full` selects all 34 recipes (102 models). Tutorials for omitted groups label their calculations as educational and show no fitted results. The counts below describe the full protocol.

| Experiment tutorial | Worked calculation | Models |
| --- | --- | ---: |
| [A · Masking and change supervision](experiments/A_masking_and_change.ipynb) | Within-person/seed gains and encoder–loss interaction | 36 |
| [B · Mask structure controls](experiments/B_mask_structure_controls.ipynb) | Joint hiding probabilities and realized interval-length matching | 24 |
| [C · Practical benchmarks](experiments/C_practical_benchmarks.ipynb) | Training-only offset/affine fitting and interpolation/triangular filtering | 12 |
| [D · Pretraining information](experiments/D_pretraining_information.ipynb) | A matched, bijective shuffled-reference assignment | 12 |
| [E · Pairing and label controls](experiments/E_pairing_and_label_controls.ipynb) | Recomputed change labels and complete-cycle minibatches | 18 |

The separate [F · JEPA response coupling](experiments/F_jepa_response.ipynb)
tutorial derives the new feature and coordinate residual losses, checks their
gradients and reads the child run's results. Its mathematical examples run
without a source run and do not train models. The [follow-up HAIC guide](../../slurm/gait-fidelity/JEPA_RESPONSE.md)
creates nine new final models across three seeds, after the core finishes.
Source the child's session before starting its kernel; keep A–E attached to the
parent core/full session.

By default the notebooks create `outputs/gait-fidelity/tutorial-fixture`. Its generated data and tiny training budget check software execution; they are excluded from scientific conclusions. When `GF_WORK` names an initialized source run, the notebooks use that saved configuration and route GPU work through Slurm. An explicitly requested but missing run stops rather than silently creating a fixture in its place.

Open notebooks from the checkout or its isolated release. For an existing HAIC run, source its `session.env` **before starting the notebook kernel** so the kernel inherits `GF_ROOT`, `GF_WORK` and `GF_PYTHON`. You can also inspect downloaded tables locally without launching source stages; source preflight and submission still require the HAIC assets and scheduler.

The teaching examples use scratch arrays and models. The canonical prepare, launch, evaluate and verify cells retain responsibility for saved experimental outputs. The examples do not change the recipe matrix, training schedule, data split or scientific claims. Standard attention and optimizer implementations remain library primitives; their scientifically relevant inputs, outputs and update rules are shown in the notebooks.

The notebook sources are retained in `build_notebooks.py` and `lesson_*.py`; regenerate with `python notebooks/gait_fidelity/build_notebooks.py`. The lesson files are **build-time sources**: their equations and code are embedded directly into each notebook, so reading a notebook does not require opening those files. Saved notebooks contain no outputs or private data. The execution checker retains populated copies separately.

To execute all 12 tutorials with a fresh CPU fixture and retain their outputs:

```bash
.venv/bin/python notebooks/gait_fidelity/execute_tutorials.py \
  --work outputs/gait-fidelity/tutorial-check \
  --output outputs/gait-fidelity/tutorial-check-notebooks
```

The runner uses real Jupyter kernels, retains executed notebook copies and writes an execution receipt. It refuses a source study so that a tutorial check cannot submit cluster work.

The [earlier validation receipt](../../docs/studies/gait-fidelity/records/transparent-tutorials-20260922.json) describes the previous 94-cell tutorial revision. The full-manifest revision adds population audits, scalable aggregation and GAVD calculations; use the [current validation record](../../slurm/gait-fidelity/validation/README.md) and [execution receipt](../../slurm/gait-fidelity/validation/tutorial-execution.json) when assessing this version. Historical exact-output comparisons do not apply to changed donor sampling or data-weighting protocols. HAIC/H100 execution remains a separate validation step.
