# Future Innovation tutorial maintenance

The five notebooks have one editable source:
[`build_future_innovation_notebooks.py`](build_future_innovation_notebooks.py).
Their scientific operations remain in `src/gavd6_sjepa/research_directions/future_innovation/`.
The small `fi_tutorial_inspection.py` module reads existing artifacts without
starting a pipeline stage. `fi_notebook_workflow.py` supplies the execution mode
by streaming the existing CLI in the kernel's Python environment. No new
training implementation or dependency is added.

From the `gavd6` root, use the existing environment. On macOS, do not synchronize
the HAIC CUDA lock over a working development environment.

```bash
.venv/bin/python scripts/research_directions/future_innovation/build_future_innovation_notebooks.py --only 00 01 02 03 04
.venv/bin/python scripts/research_directions/future_innovation/build_future_innovation_notebooks.py --check
.venv/bin/python scripts/research_directions/future_innovation/verify_future_innovation_tutorials.py
```

The verifier and HAIC jobs share `execute_future_innovation_notebook.py`, which
runs each notebook in its own kernel using the invoking Python environment.
The verifier saves executed copies together in a `notebooks/` subfolder of a
new verification bundle under ignored `work/artifacts/notebook_runs/future_innovation/`. It never
overwrites source notebooks. `--mode inspect --run-root /path/to/run` checks the
read-only path. An absent run directory is a supported inspection state.
`--pipeline-smoke` additionally exercises the execution mode on a separate
synthetic cache, including incomplete failure and sealed resume. It calls the
production CLI for all folds and controls using the existing reduced smoke
configuration, with no real teacher inference. It prebuilds the synthetic cohort
and audits; real notebook 01 extraction still needs actual media and MediaPipe.

Builders refuse to overwrite notebooks containing outputs. Save executed
copies outside the source tree and use `--check-sources` to compare their cell
sources while allowing outputs. `--only` limits rebuilding to selected lessons.

Focused checks:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_future_innovation_tutorials.py'
```

The wider future-innovation tests cover the reused scientific operations and
notebook execution/submission adapters. Synthetic cached features and fabricated
teacher audits cannot establish real checkpoint behavior. Use `execute` mode
and the [notebook HAIC jobs](../../../slurm/future-innovation/NOTEBOOKS.md) for the
full real experiment, including initialization, cohort and poses, teacher
features and validity, complete nested fitting, scoring and report generation.
Production execution requires an explicit run root and never falls back to
teaching data. Stage locks, scientific contracts, source splits and thresholds
remain owned by the existing pipeline.

## Extending the lessons

Use one question per notebook. Explain the example and consequential choices
before the code, then interpret its output and name the next decision. Reuse
existing functions; separate changes to scientific protocols from changes to
the teaching layer. End new execution work with an artifact-based interpretation
pass and update the study overview, rather than leaving its status at “to run”.

## Initial teaching-layer verification, 10 September 2026

For the subsequent full execution-mode checks, see
[notebook execution validation](../../../slurm/future-innovation/VALIDATION.md#notebook-execution-path--2026-09-10):
169 repository tests, fresh teaching/inspection kernels, and synthetic
execution/failure/resume checks through the shared notebook runner.

- All five teaching notebooks passed in separate Python 3.12.10 kernels on
  macOS ARM64. Initial wall times, including startup, were 7.17, 1.17, 1.04,
  3.02 and 0.94 seconds (13.34 seconds total). These are local measurements,
  not a runtime guarantee or a benchmark of the real experiment.
- All five passed read-only inspection of the partial local `gate-v1` copy.
  It contains overlays but no completed scientific report. Tests separately
  covered absent, incomplete, synthetic, complete, unsealed and altered reports.
- The full repository suite ran 156 tests in 37.901 seconds: 154 passed and
  two optional official V-JEPA integration tests were skipped because their
  external source checkout was not configured.
- Generated sources, local document links and `git diff --check` passed.
  Four rendered teaching figures were inspected. The cohort notebook was
  regenerated and rerun after fixing overlapping timeline labels; its focused
  source tests also passed.

Executed copies and machine-readable timings are retained under the ignored
`work/artifacts/notebook_runs/future_innovation/` directory. No real experiment
or Slurm job was launched during this tutorial verification.
