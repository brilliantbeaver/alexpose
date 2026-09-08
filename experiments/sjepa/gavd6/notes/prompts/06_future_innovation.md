**Role**: You are an expert AI/ML research scientist and technical communicator

**Task**: For the document in `proposals-03/02-future-innovation-distillation.md`, ultrathink on how to simplify the language and explanations to make the proposal, much, much easier to understand. Start from first principles and fully explain all concepts and how they inter-relate with each other. You should be writing for an advanced high school audience with a basic familiarity in machine learning.

---

**Role**: You are a world models expert specializing in Joint Embedding Predictive Architecture (JEPA)

**Task**: You are to carefully and systematically writeup a practical guide for the "48 hour decision gate" experiment in `notes/world-model-extensions/proposals-03/02-future-innovation-distillation.md`.

Ultrathink on how to use code-based explanations to walk me through setting up this experiment step-by-step. In addition, thoughtfully explain the purpose of each step and how it is relevant to the study as a whole.

---

**Role**: You are an expert in world models, JEPA, self-supervised learning, and large-scale ML experimentation on Slurm/HPC systems

**Task**: Fully implement the experiment specified in `notes/future-innovation-distillation/experiment-0-guide.md`. Your goal is to produce a complete, clean, and runnable implementation, including all necessary HAIC Slurm scripts under: `slurm/future-innovation/`.

You are to:

* Read the experiment guide completely and understand its hypothesis, experimental conditions, datasets, model, training procedure, baselines, ablations, evaluation, and expected outputs.

* Inspect the existing repository before coding. Identify and reuse existing datasets, models, training infrastructure, configs, utilities, environment conventions, and Slurm patterns. Do not invent APIs or duplicate existing functionality.

* Implement the entire experiment end-to-end. Create or modify all necessary files. The implementation should be modular, reproducible, computationally efficient, and consistent with the existing codebase.

* Build coherent Slurm orchestration in slurm/future-innovation/. Scripts should correctly handle environment setup, paths, GPUs/CPUs/memory, logging, checkpoints, seeds, experiment configuration, and job dependencies/arrays where appropriate.

* Validate thoroughly. Check Python and shell/Slurm syntax, imports, configuration paths, interfaces, and the complete execution flow:

Slurm → entrypoint → config → data → model → training → checkpoint → evaluation → outputs

Run lightweight smoke tests where feasible (imports, model construction, forward/loss pass, miniature training/evaluation). Fix any issues you find.

* Check scientific correctness. Ensure the implementation faithfully follows the guide, avoids data leakage, uses fair comparisons, preserves train/validation/test separation, and implements ablations and controls correctly.

* Perform a final code review for unnecessary duplication, brittle paths, dead code, inefficiencies, hidden assumptions, and reproducibility issues.
