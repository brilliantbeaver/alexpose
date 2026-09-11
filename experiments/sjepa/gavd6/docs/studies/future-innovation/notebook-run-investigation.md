# HAIC notebook investigation and adversarial review

The [direct-v2 amendment](#direct-v2-50-clip-gate--11-september-2026) is the current
implementation. It uses a new run identity and preserves the legacy record below.

The first sections preserve the 113910–113914 investigation. The
[GOjuXSEB investigation](#run-haic-gojuxseb-verified-audit-rejection) below
identifies the failed check and supersedes the earlier notebook-failure
handling with an explicit, verified blocked outcome.

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


## Run haic-GOjuXSEB: verified audit rejection

Evidence inspected on 10 September 2026 (the retained jobs include 11 September
UTC timestamps): all nine original executed notebooks in
`notebook_runs/haic-GOjuXSEB/`. Their bytes and original error outputs are
preserved. Builder, executor and workflow hashes in their metadata matched the
checkout before this repair. The informational code/runtime-provenance notice
was not the error.

| Notebook | Recorded job | First error, zero-based cell | Diagnosis |
| --- | --- | --- | --- |
| 00 | 114014 | None | Existing configuration validated |
| 01 | 114015 | None | Cohort preparation succeeded |
| 02 | 114016 | 11 | Reused validity audit raised generic ValueError; wrapper raised RuntimeError |
| 03, folds 0–4 | Array 114017 | 12 in each notebook | The same rejected audit prohibited fitting |
| 04 | 114018 | 11 | No complete predictive measurement; report remained INCOMPLETE |

Notebook 02 retained the check table, summary values and three per-window QC
tables, resolving the earlier investigation's missing-check question:

- `target_sensitivity=False`: saved person/background change ratio
  **1.359831237852072**, displayed as **1.360**, against the fixed **2.0** minimum.
- `person_edit_consistency=True`: direction fraction **0.8**, or 8 of 10 audit
  windows, against the fixed **0.8** minimum.
- `teacher_stable`, `causal_leakage_absent`, `target_audit_complete`,
  `data_contract_valid`, and `target_variance_valid` were all saved as true.
  Together with edit consistency, these are the other six checks.
- All ten displayed stability and leakage differences were zero; the saved
  stability threshold was 1e-6. No encoder fitting or completed prediction
  result follows from these checks.

The ratio divides the median normalized person-edit feature change by the
median background-edit change (with the declared denominator floor). It is
not a mean of the ten per-window ratios. Per-row notebook displays are rounded,
so they were not used to invent a higher-precision reconstruction. The local
bundle lacks the original QC CSV files, cache and edit contact sheets. Thus the
real ratio and flags above are retained-output evidence, not an independently
recomputed pixel intervention. Alignment overlays alone cannot validate edits.

The result can arise from properties of the contextual teacher target, the
person/background edits, or their interaction. Person-region pooled features
still receive full-clip context through attention, and the person edits alter
appearance as well as future timing. The saved outputs do not isolate a cause.
The scientifically justified action is to stop this measurement before fitting;
changing the threshold or target to obtain a pass would require a separate,
explicitly specified run. This repair changes neither.

### Root cause and changes

The workflow conflated an expected scientific gate rejection with an execution
error. A newly completed rejected audit returned CLI exit 2, but reuse raised
an untyped ValueError and exited 1. Notebooks 02 and 03 propagated the generic
failure; 04 then required a complete predictive measurement even though the
audit correctly prevented it. Seven error outputs therefore described one
rejected measurement, rather than seven independent software failures.

`ValidityAuditRejected` now represents only a fully verified rejection. Audit
reuse verifies the frozen run/cohort/cache, required CSV and contact-sheet
hashes, planned windows and donors, boolean flags, finite nonnegative distances,
matching stability/leakage records, recomputed sensitivity/direction and
training-source variance checks. Passed and rejected audits use the same
verification path. Corruption is an ordinary error; no caller can qualify a
rejection from `passed=False`, a message string or an exit code alone.

Notebook 02 retains the audit diagnostics. Notebook 03 verifies the rejection
and skips fitting for every fold. Both finish with a structured blocked
outcome. Notebook 04 skips scoring only for a verified rejection and creates
an unsealed diagnostic STOP bound to that audit's checksum. It preserves
`measurement_complete=False`, `metrics=None`, and both advancement permissions
false. The completion helper checks the current audit, decision, narrative
and permissions again. Complete sealed results retain their existing resume
behavior. Ordinary scoring/preflight errors still produce diagnostics and
fail notebook execution.

The executor records `status=blocked`, `execution_completed=True` and
`scientific_outcome.scientific_status=validity_rejected`; the notebook output
also carries this record for interactive readers. A blocked process exits zero
to allow downstream diagnostics. This status never means that the audit passed
or that a predictive result is complete. Raw `audit-teacher` continues to return
nonzero (2) for verified rejection, including reuse. The production fitting
function continues to require passing audits.

Canonical notebook changes were made in their authoritative builder and
regenerated. Source notebooks remain output-free. All scientific fixtures
remain explicitly synthetic; the fixture generator now provides hashed QC
rows and labeled contact-sheet images so tests exercise the full verifier.

### Independent adversarial review

A separate read-only reviewer examined the original failures and then the
implementation. It challenged whether the changes could turn invalid evidence
into a scientific success, bypass fitting gates or reuse a stale report.

| Objection | Severity | Disposition and evidence |
| --- | --- | --- |
| The old validator exits before checking rejected audits' detailed evidence | High | Moved classification after full verification. Corrupt/rehashed rows, omitted hashes, changed plans, contradictory flags and missing artifacts fail normally. |
| A stale complete report or arbitrary STOP could mask today's failure | High | Bound diagnostic STOP to current verified audit; require incomplete measurement, no metrics/permissions and no final seal. Check decision/narrative hashes at notebook completion. Stale-report and runtime-failure regressions pass. |
| Preflight corruption bypasses the diagnostic report fallback | Medium | Confirmed in review and fixed. Ordinary preflight errors follow the CLI failure/report path and still raise at completion. Added a corruption-to-diagnostic regression. |
| Finite input rows can overflow a median; JSON Infinity could pass `isclose` and the threshold | Medium | Confirmed in review and fixed. Derived and saved summary measurements must be finite; saved measurements must be numeric non-boolean scalars. Non-finite training-target variances also fail. Added deliberately malformed Infinity evidence. |
| Blocked notebooks could imply completed fits in metadata or prose | Medium | Distinct blocked metadata, explicit incomplete measurement, conditional lesson03 wording and updated launcher guidance. Fresh-kernel checks cover each fold separately. |
| Local tests could be presented as repaired real teacher behavior | High | Tests are labeled synthetic or injected-model interface checks. Original real threshold remains unmet; HAIC reexecution and raw intervention review are still outstanding. |

### Reproduction and validation

The new rejected-audit regression failed against the original workflow with
`RuntimeError: run-gate failed (exit 1)` and passed after the repair. Further
regressions distinguish genuine threshold rejection from data corruption,
ordinary subprocess errors, modified evidence and stale reports.

Use the existing development environment from the `gavd6` root:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/research_directions/future_innovation/build_future_innovation_notebooks.py --check
.venv/bin/python scripts/research_directions/future_innovation/verify_future_innovation_tutorials.py --pipeline-smoke
```

The full execution smoke first requires notebook 04 to fail for missing fits,
then fits five folds × five arms × three seeds on synthetic features, seals its
synthetic report, and checks unchanged scientific files on resume. A separate
rejected-audit fixture runs 02, five distinct 03 fold kernels and 04, requiring
zero notebook error outputs, explicit blocked status, no fitted checkpoints,
and incomplete measurement with no advancement permission.

Final validation results and retained bundle paths are recorded in
[VALIDATION.md](../../../slurm/future-innovation/VALIDATION.md).
One intermediate smoke attempt was stopped by its source-synchronization guard
because lesson03 was regenerated while the verifier held the prior builder in
memory. That failed attempt is retained; final verification uses stable sources.
A sandbox-only launch failure also occurred before any cell ran because Jupyter
could not bind its local socket; execution was retried with approved access.

### Remaining scientific work

No HAIC job was submitted and no original run artifact was changed. To verify
this repair against the actual run, update the code and canonical notebooks on
HAIC and use a new notebook-output folder while retaining the same frozen
`gate-v1` data. Cached stages should be reused after verification and 02–04
should finish blocked with the original audit values. The full raw cohort,
cache and QC artifacts must remain available for these checks.

Inspect the ten original pixel-edit contact sheets, their frozen donor plan,
per-window changes, boxes and teacher configuration before proposing any
measurement repair. The audit still fails the registered criterion; successful
local notebook execution cannot authorize fitting through it or establish the
prediction hypothesis.

## Direct-v2 50-clip gate — 11 September 2026

The author requested a direct test of how much skeleton history adds beyond RGB
and nuisance inputs, removing background-quality/selectivity prerequisites. A
subsequent clarification fixed Experiment 0 at **50 clips** and reserved the
full GAVD dataset for the real experiment. The final implementation enforces that
size; a temporary 70-window development smoke is outside the final protocol and
is rejected by the current cohort validator. It is not a scientific result.

This amendment follows inspection of the legacy 1.360 sensitivity rejection.
It is not a claim that the old teacher passed its 2.0 rule. `legacy-v1` runs,
including runs without an explicit protocol field, keep their original controls
and audit behavior. New `direct-v2` roots freeze a separate protocol contract;
legacy cohorts, caches and fitted folds are not silently relabeled or imported.
The [current specification](direct-gate-protocol.md) is the authoritative design
summary; the original long experiment guide is retained as historical context.

### Implemented scientific changes

The primary comparison is real-skeleton versus the matched no-skeleton head,
which shares RGB/nuisance inputs, validity flags and parameter count. Both are
also compared with the same ridge baseline. Direct-v2 retains time-shuffle and
partition-local different-source mismatch controls, five outer/three inner source
folds, three seeds, the complete selection grid and source-balanced paired
uncertainty. It adds an explicit positive matched increment and stability rule,
so extra RGB-head capacity alone cannot justify advancement.

The gate no longer requires pixel-edit selectivity, a background-target arm or
its gain-reduction test, 90% crop retention, or 45% whole-body pose coverage.
Required observed input support, exact alignment, past-only preprocessing,
source isolation, repeatable teacher outputs and nonconstant person targets
remain. Unavailable background measurements are represented by zeros plus
prefix support indicators. This permits the scientific comparison while leaving
recording cues and contextual teacher attention as unresolved explanations.

The five notebooks retain their names and order. Notebook 02 runs three-window
repeatability/prefix-isolation checks without person/background edits; notebook
03 fits 60 final heads. Notebook 04 reports the paired estimate and 95% interval,
all arm/seed scores, criteria and the recommended next step. ADVANCE recommends
planning a separate full-GAVD comparison of trained JEPA features, matched
initial encoders and raw skeletons. It does not launch training or authorize
adapter distillation. Development sources used in the gate are not an untouched
test cohort for that subsequent experiment.

### Independent adversarial review

The independent read-only reviewer inspected the code against the saved
pre-amendment package/scripts under `work/artifacts/direct-v2-baseline/`, reproduced
edge cases, and checked the paired estimator. The review ran independently of
the implementation work; no unavailable adversarial-review plugin is claimed.

| Objection | Severity | Evidence and correction | Residual boundary |
| --- | --- | --- | --- |
| Ridge gain could come from a larger RGB head rather than skeletons | Major scientific attribution risk | Real-minus-no-skeleton is now primary and required to be positive on average, in every seed and in at least 90% of paired source draws. A regression test makes RGB-only extra capacity produce STOP. | It tests the specified heads and coordinate/confidence history; validity already enters the control. |
| Removing only the pixel audit leaves background prerequisites elsewhere | Blocking implementation defect | Reviewer reproduced a full-frame box failing `context_nuisance`. Direct pooling, background RGB and optical flow now accept absent support, with fixed-schema support features. Real decoding/cache-interface tests cover full-frame boxes and alternating boxes with no shared flow support. | Neutral values do not certify background invariance; nuisance cues remain possible. |
| Frozen-config corruption raises before notebook diagnostics | Material execution defect | `audit_summary_path` had called `check_run` outside protected preflight. It now executes inside the try block so CLI logs and diagnostic fallbacks remain reachable; non-object readiness JSON is rejected as ValueError. Regression tests verify both routes. | Corrupt inputs still fail; a convenient STOP cannot conceal execution defects. |
| A blocked STOP could accidentally permit JEPA training or carry the wrong protocol | Material integrity defect | Direct blocked decisions now require `allow_jepa_training_comparison=False`, matching protocol/run/audit, null metrics and incomplete measurement. Tampered decisions are rejected in regression tests. | Synthetic and incomplete runs never authorize advancement. |
| New direct reports lose copied-artifact provenance | Moderate reproducibility defect | Restored checkpoint, gate manifest, readiness, protocol, run/config and code identities in JSON and narrative. Complete decision/report pairs remain sealed; incomplete attempts are archived. | A checksum establishes identity, not independent scientific reproduction. |
| Full-cohort computation magnifies allocation and resumption costs | Scope concern | User clarified that the gate remains 50 clips. Contract and tests reject larger gates; full-GAVD training remains a separately planned experiment. Distance, prediction storage and bootstrap sufficient-sum improvements preserve the estimator. | Incomplete pose extraction restarts candidates; completed cohorts and teacher caches resume. No full-dataset runtime claim is made. |
| A source-bootstrap interval can be overstated as complete experimental uncertainty | Moderate interpretation risk | Report/specification clarify that resampling uses saved out-of-fold predictions and is conditional on those fits. The independent unequal-source-size check matched the original resampling estimator within 8.9×10⁻¹⁶. | Training and inner selection are not repeated in each bootstrap; 25 or more source videos remain the sampling units. |
| A 90%-positive gate can advance with a 95% interval containing zero | Scientific decision tradeoff | Both quantities are stated separately; the interval is never described as excluding zero unless it does. This preserves the specified stability rule. | ADVANCE remains a screening recommendation, not conclusive positive evidence or validated JEPA benefit. |

The final independent code review found no remaining blocking findings, with
acceptance conditional on the fresh-kernel checks recorded in
[VALIDATION.md](../../../slurm/future-innovation/VALIDATION.md). Local generated
media, injected detector/teacher tests and synthetic fitting validate the software
paths. No new real GAVD prediction finding or real pretrained-teacher reexecution
is claimed. The nine `haic-GOjuXSEB` notebooks remain the original evidence.

Final evidence acceptance: the independent reviewer checked all three retained
verification records, 50-clip/25-source direct reports, provenance and report seals,
unsealed blocked STOPs, preserved missing-score diagnostics, and the final test log.
The review closed with no remaining findings. This acceptance covers software and
synthetic verification; real direct-v2 prediction results remain unmeasured.
