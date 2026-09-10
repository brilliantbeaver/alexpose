# HAIC notebook investigation and adversarial review

Evidence: copied notebook executions from jobs 113910–113914 on 10 September
2026. Four original notebooks were consolidated byte-for-byte into
`gavd6/notebook_runs/haic-113910-113914/`. The added notebook 03 is explicitly a
blocked-run diagnostic, not a fabricated execution. Older support files are
retained outside the notebook folder under ignored `work/artifacts/`.

## Observed causes and repairs

| Stage | Observed behavior | Repair or interpretation |
| --- | --- | --- |
| 00 | Existing run validation passed | Retain immutable configuration checks. New initializations record additional storage roots and accept a full-source directory without requiring an `all/` layout. |
| 01 | Reused 1,662 candidates and 50 windows; displayed the first ten of 1,824 exclusions | Distinguish candidate/pose failures from source-cap and cohort-size selection. Resolve manifest paths and exact IDs recursively in declared storage. Report source availability separately. |
| 01, short spans | Original selector rejects fewer than 64 annotated source frames | Preserve the temporal definition. Full-video availability does not establish an annotated 64-frame person track. No padding or repeated frames are introduced. |
| 01, gaps | A deterministic start could land on an annotation gap despite another intact window | Select deterministically among intact 64-frame windows if the original start is unusable. |
| 01, retry | Incomplete discovery was frozen before raising its error | Do not seal an inventory unable to supply the cohort; allow retry after missing storage arrives. Existing frozen cohorts are not silently changed. |
| 02 | Caching passed, then `audit-teacher` returned exit 2 and only `validity_audits_passed: false` | Retain saved check names, thresholds and all per-window QC tables in the notebook before propagating failure. The underlying scientific failure is not converted to success. |
| 03 | No copied notebook; original dependency required successful 02 | Use `afterany` to retain each fold's blocked diagnostic. The production fit CLI still validates audits before fitting. |
| 04 | Incomplete STOP caused by failed validity audits | Preserve the distinction between incomplete execution and a complete negative result. Incomplete synthetic demonstrations also show incomplete status. |
| Outputs | Separate directories, source copies, font caches and duplicate JSON per notebook | One notebook-only folder per submission; fold suffixes and writer locks; provenance in notebook metadata; durable stage logs outside the folder; temporary kernel caches. |

## Adversarial checks

The review asks whether each repair could create a false scientific success,
silently change samples, lose failure evidence, or overwrite another task.

| Challenge | Required behavior and regression evidence |
| --- | --- |
| File exists outside `youtube/all` | Discover nested storage and declared manifest paths. Tests include AVI and mixed-case extensions. |
| Similar ID or unfinished download | Exact ID matching only; `.part` files and longer lookalike IDs do not match. |
| Multiple exports share an ID | Reject ambiguity unless the manifest identifies one full-source file; deduplicate symlinks/hard links to the same file. |
| Source arrives after a failed discovery | Retry succeeds without deleting a frozen scientific contract. |
| Annotation gap or too-short span | Recover another intact window; genuine short spans still fail. Selected windows remain 64 consecutive source frames. |
| Failed audit followed by fitting | Save notebooks 02 and 03 with diagnostic output and a failed status; no model checkpoints are created. |
| Negative or incomplete result | Complete synthetic STOP remains non-evidentiary; incomplete results fail execution. No thresholds change. |
| Two array tasks or run roots share an output folder | Distinct fold filenames; the lock follows the destination, rejecting duplicate writers even across run roots. |
| Long-running stage or failed cell | Preserve atomic notebook checkpoints and a durable command/output/exit log. No source notebook is overwritten. |
| Recovery after completed fitting/reporting | Reuse all five folds and the sealed report without changing scientific artifact bytes or modification times. |

The original failure path was reproduced locally using the existing explicitly
synthetic fixture with a failed audit. This verifies orchestration and evidence
presentation, not the real teacher. Repository tests also exercise encoded
video/pose/cache interfaces with injected lightweight models.

## Remaining HAIC evidence

The copied notebook 02 lacks `qc/validity-summary.json` and the three numerical
QC CSVs. Direct SSH inspection was attempted; Stanford rejected non-interactive
authentication. Therefore the individual failed real audit check is not yet
known. It would be incorrect to change the sensitivity/stability thresholds or
claim successful real inference based on this local repair.

Obtain the QC summary and CSVs from the recorded `gate-v1` directory. Determine
whether the failure is stability, leakage, variance, person/background
sensitivity or edit consistency. Any scientific protocol change needs a new
versioned run; the current data cannot be used to tune the gate until it passes.
To apply the discovery improvements, initialize `gate-v2` with the actual storage
roots before submitting the full notebook workflow described in
[the HAIC guide](../../../slurm/future-innovation/NOTEBOOKS.md).

Status: **DONE_WITH_CONCERNS** for the verified local implementation; the
specific real audit failure and actual HAIC storage coverage still require
the missing QC files and authenticated access. Validation: 178 repository
tests (176 passed, two optional skips), fresh teaching/inspection kernels,
75-checkpoint positive smoke, and zero-checkpoint failed-audit smoke. See
[the validation record](../../../slurm/future-innovation/VALIDATION.md).
