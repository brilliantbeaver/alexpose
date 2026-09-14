# Reading historical execution notebooks

Saved execution notebooks are evidence records. Their Markdown, code, metadata and outputs together determine the file hash. Moving the repository changed some relative navigation paths; editing those notebooks would invalidate the hashes used by earlier analyses.

Use the maintained links below while reading those files. The originals remain unchanged. The [navigation registry](notebook-navigation.json) records each exact notebook path, SHA-256 and link replacement. The documentation validator checks every registered replacement and only recognizes it while the original notebook still matches that hash. Unknown broken links and changed notebook bytes remain errors.

This repair covers **37 historical navigation links in 11 locally retained notebooks**. Ten notebooks match independent hash records; the older signed-laterality notebook has saved outputs but no enclosing seal was demonstrated. Hash matching identifies preserved files; it does not by itself validate their scientific results. These navigation links are separate from the [unavailable experiment artifacts](evidence-availability.md).

## Gate-v2 execution

The nine notebooks under `notebook_runs/gate-v2/` record an earlier direct-gate implementation. Use them to inspect that execution, and use the current guides to understand the repository. Current source code and later direct-v3 results are distinct versions.

- [Gate study and evidence](../studies/future-feature-prediction/gate/README.md)
- [Frozen direct-gate protocol](../studies/future-innovation/direct-gate-protocol.md)
- [Notebook execution guide](../../slurm/future-prediction/notebooks/README.md)
- [Slurm workflow](../../slurm/future-prediction/README.md)

## Source-scaling inspection

`notebook_runs/haic-run-02/23_source_learning_curves.ipynb` is a retained inspection. Its record says artifact integrity and numerical reconstruction were not verified; the notebook hash does not change that limit.

- [Source learning-curve protocol](../studies/future-innovation/source-learning-curve-protocol.md)
- [Available-cohort protocol](../studies/future-innovation/source-learning-curve-available-cohort-protocol.md)
- [Source-scaling execution guide](../../slurm/source-scaling/README.md)

## Older signed-laterality demonstration

`artifacts/notebook_runs/organized-2026-08-23/experiments/idea05_signed_laterality/01_probe.ipynb` contains historical executed outputs. Its [archived study design](../../notes/archive/portfolio-ideas/ideas/05-signed-laterality-decodability/README.md) supplies the missing navigation destination. A maintained source notebook describes the workflow but is not a substitute for this executed file.

## Validation behavior

The normal documentation check reports these original stale links explicitly with their verified companion destinations. Add `--strict-archive-links` to make every stale link inside an unchanged historical payload fail as well. Both modes audit the original notebook equations. Neither mode edits notebook bytes or old evidence manifests.
