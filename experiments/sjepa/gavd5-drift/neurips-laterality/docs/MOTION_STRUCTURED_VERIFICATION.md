# Verification of notebooks 15–18

Verified on 8 September 2026 with Python 3.12.10 and PyTorch 2.13.0+cpu on
Windows. The extension's 13 tests pass. All four source notebooks validate,
match their editable tutorial builders and execute in separate fresh kernels.
Seventeen code cells execute successfully, producing four SVG figures and
four PNG previews. All four figures were visually inspected for readable
labels, legends and axes.

The final teaching execution is retained locally at
`executed/motion_structured/synthetic_nayf51or/`, including `verification.json`.
Source notebooks remain output-free. Execution disables real-data validation,
real training and retained-encoder reanalysis, regardless of inherited flags.
The verifier confirms that 57 protected notebook, tutorial, model, protocol and
configuration files remain unchanged. Git also shows no modifications to
notebooks 00–14 or their training implementations.

## Checks that passed

- MAMP log weights agree with a direct translation of the official PyTorch
  implementation on fully observed inputs. Repeated sampling preferentially
  hides moving tokens; stationary fallback, missing-transition isolation,
  reflection and infeasible budgets are checked.
- Structured masks retain their declared intervals or full trajectories,
  and their scattered references match realized per-clip target counts.
- Training preserves paired initial states, source draws, geometric views and
  target counts. Changing labels and outer-test coordinates leaves trained
  weights unchanged. Teachers receive no gradient. Fixed hidden values cannot
  enter the dense predictor pathway.
- Completed runs are reused exactly, and corrupted artifacts are rejected.
  The enabled real-run orchestration is exercised with a mocked synthetic
  dataset, including evaluation, persistence and repeat execution. Its output
  remains explicitly synthetic. A misreported workload is rejected.
- A known amplitude contrast is recovered by motion-sensitive summaries on
  generated held-out sources. The mean summary agrees numerically with the
  previous implementation. Test-source changes cannot alter ridge selection.
- Aggregation rejects duplicate and incomplete predictions. Retained Notebook
  12 comparisons can be reanalysed without encoder training, preserving every
  original artifact byte in the test fixture.
- The real plan displays 125 encoders and 150,000 optimizer updates using the
  tracked 1,200-update recipe. Its disabled path does not train. The notebook
  builder now writes explicit UTF-8, avoiding Windows locale corruption of
  notebook JSON containing mathematical text.

Commands used from the repository root:

```text
.venv/Scripts/python.exe -m unittest discover -s neurips-laterality/tests -p test_motion_structured.py -v
.venv/Scripts/python.exe neurips-laterality/scripts/build_research_notebooks.py --check --only 15 16 17 18
.venv/Scripts/python.exe neurips-laterality/scripts/verify_motion_notebooks.py --execute
git diff --check
```

## Broader suite and local prerequisites

The full suite runs 221 tests: 219 pass, with two existing environment-dependent
failures in untouched files:

1. `test_real_training_requires_enablement` in `test_comparative_training.py`
   cannot load the expected complete reference grid. Its real-setting helper
   requires local artifacts absent from this checkout. The new planner reads
   the tracked numerical recipe and does not have this prerequisite merely to
   display a workload.
2. `test_only_reviewed_progress_only_digest_is_accepted` in
   `test_training_progress.py` compares raw source-byte hashes with a retained
   compatibility manifest. The Windows checkout has CRLF line endings in
   core files. Its current digest begins `0f497023`; computing the same digest
   with LF bytes gives `bc186e81`, exactly the manifest's reviewed value.
   No protected source or compatibility manifest was changed to suppress this
   failure.

A separate read-only real-input preflight fails because
`artifacts/paper/protocol_6f7baefbda07/cohort/metadata.json` is absent. No cohort
was rebuilt and no real-data grid was launched. Local model artifacts are also
required to reanalyse the retained encoders. GPU execution and real-data
performance remain unverified. The generated runs support implementation and
evaluation checks, with no new empirical or clinical conclusion.

The tutorial's empirical results and PDF remain the previous evidence version;
its Markdown adds navigation to this extension. The
[source review and research specification](MOTION_STRUCTURED_MASKING.md)
explains what to run once compatible local artifacts are available.
