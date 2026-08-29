# Repository research artifacts

This directory contains small, Git-tracked research records that need to stay
with their related methods. Large, machine-specific, or rerunnable outputs
belong under `work/artifacts/` or the configured AMASS run root.

- `research/`: compact, durable research bundles retained with their related
  methods. These are not run caches.

Executed notebook copies belong in the ignored
`work/artifacts/notebook_runs/` directory. Run artifacts should use their
experiment's explicit, versioned contract whenever one exists.

Source notebooks always live under [`notebooks/`](../notebooks/README.md).
