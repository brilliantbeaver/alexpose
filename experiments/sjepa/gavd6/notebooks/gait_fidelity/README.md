# Gait Fidelity tutorials

These fifteen notebooks connect the gait-fidelity calculations to the completed experiments. **To learn the algorithm, work through 00–06, then F and G; to interpret the latest results, read [07 · Completed-study walkthrough](07_completed_study_walkthrough.ipynb) alongside them.** Notebook 07 reads `outputs/iclr` and visualizes walking core, JEPA response and readout repair without initializing a run. The [algorithm guide](docs/ALGORITHM_GUIDE.md) maps each scientific operation to its equation, visible code and verification. The [visualization guide](docs/VISUAL_WALKTHROUGH.md) explains the result panels.

Seven numbered tutorials (00–06) and experiments A–E cover the core/full implementation. F derives paired residual pretraining and G derives the completed frozen-readout repair. Equations lead into NumPy/PyTorch calculations and checks against production, including gradients and optimizer updates. The multiple-sclerosis notebooks informed the stepwise exposition and data-access explanations; gait-fidelity retains its own body-12 restoration algorithm, splits and metrics. Start with the [HAIC guide](../../slurm/gait-fidelity/README.md) for a source experiment, or use the CPU execution command below for a local teaching run.

See [environment and readiness](docs/ENVIRONMENT.md) for all required and automatically supplied variables, compatible interpreters, source-data prerequisites and the points where HAIC notebook execution must wait for Slurm jobs.

The folder separates the reading sequence from its supporting files:

```text
gait_fidelity/
├── README.md                     # Reading order and commands
├── 00_start_here.ipynb … 07_…    # Main tutorial sequence
├── experiments/                 # Experiment notebooks A–G
├── docs/                        # Algorithm, environment and visualization guides
└── scripts/
    ├── build_notebooks.py        # Generate the notebooks in this folder
    ├── execute_tutorials.py      # Validate the CPU tutorial workflow
    ├── execute_walkthrough.py   # Execute the downloaded-evidence walkthrough
    ├── tutorial_helpers.py      # Runtime paths and canonical study commands
    └── lessons/                 # Prose and code embedded by the builder
```

Run the commands below from the repository root. The scripts locate their inputs
relative to their own files; generated results remain under `outputs/`. Python
entry points previously in this folder now live in `scripts/`.

| Notebook | What you do |
| --- | --- |
| [00 · Start here](00_start_here.ipynb) | Trace the full pipeline, inspect settings and reconstruct the count of shared training phases. |
| [01 · Data and references](01_data_and_references.ipynb) | Reconstruct condition counts and paired endpoints; check identity boundaries; implement naming swaps, reference projection and input-only normalization. |
| [02 · Masking and controls](02_masking_and_controls.ipynb) | Pack four-frame joint tokens, implement all five mask samplers and measure hidden-token exposure. |
| [03 · Architecture and experiment matrix](03_experiment_matrix.ipynb) | Expand channel packing, positions, attention, predictor stack and coordinate readout; check forward values and gradients. |
| [04 · Losses, updates and execution](04_run_and_monitor.ipynb) | Implement person→motion→window→pair sampling, coordinate/feature/regularization losses, optimizer and teacher updates; launch the selected study. |
| [05 · Evaluation and visualization](05_evaluate_and_visualize.ipynb) | Derive geometry, fixed support, waveform/response/direction metrics and failure costs; explicitly fit response-regression and GAVD-classification probes. |
| [06 · Verification and uncertainty](06_verify_and_write.ipynb) | Rebuild hierarchical means, crossed people/seed bootstrap and repair's person-level t interval; verify retained comparisons and scope. |
| [07 · Completed-study walkthrough](07_completed_study_walkthrough.ipynb) | Trace the completed experiments from paired data through representation and readout changes to results, failure accounting and person-level uncertainty. Uses downloaded evidence, not a fixture. |

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
without a source run, use fresh scratch models, and do not modify saved study
checkpoints or launch source fits. The [follow-up HAIC guide](../../slurm/gait-fidelity/JEPA_RESPONSE.md)
describes the response experiment after the core finishes. The completed response-02 plan has nine new pretraining fits and eighteen final readouts (three variants × two output objectives × three seeds); notebook 07 reads these counts from its saved plan.
Source the child's session before starting its kernel; keep A–E attached to the
parent core/full session.

[G · Readout repair](experiments/G_readout_repair.ipynb) exposes differentiable knee geometry, linear percentile interpolation, scalar/dense/geometry reductions, initial gradient-energy calibration, and matched frozen-readout updates. It checks values, coordinate gradients, optimizer states, failure cases, and all six saved source-calibration coefficients. The completed repair fits twelve new readouts over the response encoders. G's scratch calculations need no checkpoint or initialized study.

By default the original tutorials create `outputs/gait-fidelity/tutorial-fixture`. Its generated data and tiny training budget check software execution; they are excluded from scientific conclusions. When `GF_WORK` names an initialized source run, those notebooks use that saved configuration and route GPU work through Slurm. An explicitly requested but missing run stops rather than silently creating a fixture in its place. Notebook 07 instead reads `outputs/iclr`, ignores `GF_WORK` and `GF_ROOT`, and stops if its evidence packet is absent or fails its transfer-hash checks.

Open notebooks from the checkout or its isolated release. **00–06 and A–E require a core/full session**, not a response or repair session. For an existing HAIC core run, source its `session.env` **before starting the notebook kernel** so the kernel inherits `GF_ROOT`, `GF_WORK` and `GF_PYTHON`. Selecting a follow-up session in those notebooks now raises a clear error. F/G are standalone mathematical tutorials with optional saved-result readers; 07 reviews the compact evidence. The local `outputs/gait-fidelity` root contains the downloaded core report/configuration, not the complete source-run bundle needed by 00–06.

The teaching examples use scratch arrays and models. The canonical prepare, launch, evaluate and verify cells retain responsibility for saved experimental outputs. The examples do not change the recipe matrix, training schedule, data split or scientific claims. Standard attention and optimizer implementations remain library primitives; their scientifically relevant inputs, outputs and update rules are shown in the notebooks.

The notebook sources are retained in `scripts/build_notebooks.py` and `scripts/lessons/lesson_*.py`; regenerate with `.venv/bin/python notebooks/gait_fidelity/scripts/build_notebooks.py`. Edit those sources when changing a lesson, then regenerate the notebooks. The lesson files are **build-time sources**: their equations and code are embedded directly into each notebook, so reading a notebook does not require opening those files. `scripts/tutorial_helpers.py` supplies run selection and canonical study commands. Saved notebooks contain no outputs or private data. The execution checker retains populated copies separately.

To execute 00–06 and A–G with the latest core sampling protocol on generated CPU data, retaining populated notebooks and HTML copies:

```bash
.venv/bin/python notebooks/gait_fidelity/scripts/execute_tutorials.py \
  --experiment-set core \
  --work outputs/gait-fidelity/tutorial-transparent-20260925/work \
  --output outputs/gait-fidelity/tutorial-transparent-20260925/notebooks \
  --html
```

The runner uses the invoking interpreter in real Jupyter kernels and writes an execution receipt. It refuses a source study so that a tutorial check cannot submit cluster work. Omitting `--experiment-set` retains the original full-matrix fixture default; selecting `full` requires a separate work directory and exercises optional controls. Both modes execute the mathematical examples for omitted groups while identifying those groups as unfitted. Notebook 07 has its own read-only command: `.venv/bin/python notebooks/gait_fidelity/scripts/execute_walkthrough.py --export-figures`.

The [September 25 refresh record](../../docs/studies/gait-fidelity/reviews/tutorial-refresh-20260925.md) records this revision's execution, parity checks and independent adversarial review. The [earlier validation receipt](../../docs/studies/gait-fidelity/records/transparent-tutorials-20260922.json) describes the previous 94-cell revision; its counts should not be used for this version. Production source and downloaded evidence are preserved. CPU checks do not validate the HAIC/H100 runtime or reproduce source-study scores.
