# Implementation validation

## Direct-v2 50-clip gate — 2026-09-11

The author clarified the final scope: **50 clips for Experiment 0**, selected from
the full-GAVD candidate pool; the full dataset belongs to the subsequent real
experiment. New runs use direct-v2, while legacy runs preserve their five-arm
selectivity protocol. See the [current protocol](../../docs/studies/future-innovation/direct-gate-protocol.md)
and [independent review/dispositions](../../docs/studies/future-innovation/notebook-run-investigation.md#direct-v2-50-clip-gate--11-september-2026).

- Final repository suite: **207 tests in 35.102 seconds; 205 passed, 2 skipped**.
  The skipped tests require an external official V-JEPA source checkout. The
  final log is `work/artifacts/direct-v2-final-tests.log`. This is a local
  macOS/Python 3.12.10 verification, not the HAIC CUDA environment.
- Fourteen direct-protocol tests check the fixed 50-clip contract, deterministic
  selection from a larger pool, absent-background behavior, causal/readiness
  rejection and corruption, preserved diagnostic routing, blocked JEPA/protocol
  guards, matched-capacity attribution, uncertainty decisions and bootstrap
  equivalence. Legacy scientific rules remain covered by the existing tests.
- Three data-flow tests decode generated video and exercise actual manifests,
  pose/cache interfaces and teacher-check routines with an injected detector and
  teacher. Full-frame person boxes and alternating boxes without shared background
  flow now complete cache construction. The constant injected teacher fails only
  the intended person-target variance requirement in direct-v2. No pixel-edit
  evidence is fabricated as real teacher evidence.
- `verification-nzdmi50q` passed all five fresh teaching kernels, both complete
  synthetic notebook/CLI paths (**60 direct-v2 heads; 75 legacy heads**), deliberate
  missing-fit failure, and unchanged sealed resumption. Each protocol also passed
  seven rejected-readiness/audit notebook cases: 02, five 03 folds, and 04. These
  finish as explicitly blocked, with zero heads and no notebook error outputs.
- After adding conditional-on-saved-fits wording to the report and notebooks,
  `verification-sii_rgbe` passed fresh 03/04 teaching kernels and repeated the
  entire direct-v2 synthetic complete, incomplete, blocked and resume checks.
  The final report states 50 clips, the matched skeleton increment, its conditional
  interval and the full-GAVD next step. Synthetic ADVANCE is prohibited.
- `verification-od1co0n_` passed all five notebooks in read-only inspection mode
  against the completed direct-v2 synthetic report. The inspector remains able
  to read preserved artifacts generated before the final prose clarification.
- Generated sources, updated guide links, all Future Innovation shell syntax and
  `git diff --check` passed. All nine original `haic-GOjuXSEB` notebook SHA-256
  hashes match the saved pre-edit inventory.

The three bundles are under
`work/artifacts/notebook_runs/future_innovation/`. A machine-readable acceptance
summary is retained at `work/artifacts/direct-v2-verification-summary.json`.
Source notebooks remain unexecuted; executed copies and logs remain separate.
The synthetic fitting fixtures explicitly reduce width, search/update budgets
and bootstrap repetitions. They verify routing and estimation, not the full
real-data training budget or the pretrained teacher's behavior on GAVD.

Independent review found no remaining blocking code issues after correcting
missing-background nuisances, corrupt-config diagnostic routing, blocked JEPA
permissions and report provenance. The final uncertainty wording distinguishes
conditional bootstrap intervals from repeated training and preserves the separate
90%-positive decision rule. No real direct-v2 result or fresh HAIC run was produced
locally. Start a new run root to obtain that result; existing caches and legacy
thresholds are not silently relabeled.


## Verified audit rejection in haic-GOjuXSEB — 2026-09-10

- Final repository suite: **191 tests in 45.707 seconds; 189 passed and 2
  optional official V-JEPA source integrations skipped**. Thirteen new tests
  cover verified rejection, complete numeric audit validation, corrupt and
  rehashed rows, missing files/plan changes, boolean/finite-value checks,
  diagnostic-report fallback, stale reports, runtime errors and audit reuse.
- The initial rejected-audit regression reproduced the old RuntimeError before
  the repair. The final code distinguishes verified scientific rejection from
  execution errors while preserving the frozen threshold and fitting gate.
- Independent read-only adversarial review found two implementation issues:
  corruption bypassing report fallback, and Infinity passing numeric checks.
  Both were fixed with regressions; the final review found no blocking issues.
- The canonical notebooks pass the builder synchronization check. Python
  syntax checks and `git diff --check` passed. The original nine HAIC notebook
  files retain their original statuses and error outputs; a digest inventory is
  saved under `work/artifacts/haic-GOjuXSEB-original-notebooks.json`.

- Final fresh-kernel inspection and full pipeline smoke **passed** in
  `work/artifacts/notebook_runs/future_innovation/verification-9zg6grzj/`.
  All five inspection notebooks passed. The execution smoke retained 75 final
  synthetic checkpoints, a complete synthetic STOP, and unchanged scientific
  artifacts on fold/report resume. Its missing-fit scenario correctly failed
  before fitting, as asserted by the verifier.
- All **seven** rejected-audit notebooks (02, the five 03 folds, and 04) finished
  with `status=blocked`, `execution_completed=true`, incomplete predictive
  measurement and **zero cell errors or fitted checkpoints**. Their diagnostic
  STOP remains unsealed, has no predictive metrics and grants no advancement.
- All five teaching notebooks passed in fresh kernels in the intermediate
  `verification-qv7oyhnl/notebooks/` bundle before its later source-change
  mismatch. The final inspection/smoke bundle is green and all retained
  notebook/source/orchestration hashes match the final code. The earlier
  failed bundle and initial socket-denied attempt remain preserved.

Final kernel evidence includes `verification.json`, `checked-outcomes.json`,
executed notebooks and durable CLI logs in that bundle's run directories.
Full suite output: `work/artifacts/haic-GOjuXSEB-repair-tests.log`.
The [investigation](../../docs/studies/future-innovation/notebook-run-investigation.md#run-haic-gojuxseb-verified-audit-rejection)
records the exact saved real-data ratio, all checks, root cause and review
objections. These are local implementation checks. The recorded HAIC audit
still fails its fixed sensitivity criterion; no real teacher inference or
HAIC scheduling was rerun, and no run thresholds or models were changed.

## Discovery, failed-audit diagnostics and shared notebook output — 2026-09-10

- Full repository suite: **178 tests in 41.099 seconds; 176 passed, 2 optional
  official-source integrations skipped**. The final cross-run notebook-writer
  guard additionally passed the ten focused execution tests.
- All five teaching and all five inspection notebooks passed fresh kernels.
  The synthetic pipeline fit and retained all 75 residual checkpoints, scored,
  sealed its non-evidentiary STOP, and resumed without changing scientific
  artifact bytes or modification times.
- A separate failed-audit fixture saved notebooks 02, 03 and 04 with diagnostic
  outputs and failed status, creating **zero** residual checkpoints. Incomplete
  synthetic reports now remain visibly incomplete. This reproduces the observed
  HAIC orchestration failure without pretending to diagnose its unknown numeric
  audit check.
- Against the previous committed cohort implementation, the regression fixtures
  reproduced both the discarded intact window (49 candidates) and the failed
  retry after a missing source arrived (48 frozen candidates). Both pass with
  the fixes. Discovery tests also cover nested files, explicit manifest paths,
  extra roots, duplicate symlinks/hard links, ambiguous exports, partial files,
  lookalike IDs, and genuine short sequences.
- Output tests cover one shared notebook-only folder, distinct fold filenames,
  duplicate-writer rejection even across run roots, source protection, retained
  failed-cell outputs, deterministic subprocess controls, and relocated links.
  Existing shell tests check syntax and mocked Slurm dependencies.
- The four original copied HAIC notebooks were consolidated byte-for-byte;
  notebook 03 was added only as an explicitly unexecuted blocked-run diagnostic.

Local evidence is retained in `work/artifacts/future-innovation-fixes-tests.log`
and `work/artifacts/future-innovation-kernel-fixes.log`; executed integration
notebooks are under `verification-kzfbkaan/` in the ignored notebook verification
directory. See the [investigation and adversarial review](../../docs/studies/future-innovation/notebook-run-investigation.md)
for the real-run evidence still needed. No HAIC runs or scientific thresholds
were changed during this repair.

## Notebook execution path — 2026-09-10

The separate [notebook jobs](NOTEBOOKS.md) execute the same CLI stages as the
original jobs. The production scientific implementation and frozen defaults
were not changed for notebook execution.

- Full repository suite: **169 tests in 43.416 seconds; 167 passed, 2 skipped**.
  The skips remain the optional official-source integrations without
  `FI_TEST_VJEPA_ROOT`. Thirteen new tests cover notebook stage routing,
  initialization partitions and resume, full-grid/fold selection, subprocess
  failures and logs, report completion checks, partial notebook preservation,
  Slurm commands/dependencies, and submission failures.
- All five teaching notebooks and all five read-only inspection notebooks
  passed in fresh local kernels. Inspection used the partial local `gate-v1`
  copy and did not claim a completed scientific result.
- The notebook execution smoke passed on a separate synthetic cache: notebook
  04 first failed for missing fits while retaining its traceback and incomplete
  STOP report; 00 validated the run; 02 reused verified caches/audits; 03 fit all
  five folds, three seeds and five arms; 04 scored and sealed the complete
  synthetic STOP. Reexecuting fold 3 and reporting preserved the scientific
  artifacts byte-for-byte and their modification times.
- The smoke uses the existing explicitly reduced synthetic model/search/
  bootstrap contract. The execute cells do not reduce real-run settings.
  Synthetic cohort/features/audits are prebuilt, so this is not a real-media
  end-to-end experiment. Notebook 01 command routing and its failure boundary
  are tested separately, alongside the existing encoded-video/data-flow tests.

Executed copies, command receipts, timings and tracebacks are retained under
ignored `work/artifacts/notebook_runs/future_innovation/verification-*/`.
No real GAVD run or HAIC job was launched for this change. Actual MediaPipe
extraction, full H100 checkpoint inference, real-data audits and scheduling
remain to be verified on HAIC with its declared inputs.

## Resumption fixes — 2026-09-10

Command: `.venv/bin/python -W ignore::DeprecationWarning -m unittest discover -s tests`

Result: **148 tests run in 43.304 seconds; 146 passed, 2 skipped.** The skips are the optional official V-JEPA source integrations because `FI_TEST_VJEPA_ROOT` was not set. `git diff --check` also passed. The dependency deprecation-warning filter only reduces local test output; production warnings are not suppressed.

New regression coverage verifies:

- Implementation and runtime changes are recorded without modifying the original run contract or invalidating its data bindings. Actual frozen configuration corruption still fails.
- Completed candidates and poses are reused; modified candidate manifests and pose artifacts are rejected.
- Complete teacher caches and passing audits do not load the teacher again. Saved projection matrices are reused without a fresh numerical decomposition.
- An early STOP can resume through all five folds, score, and produce a sealed final report. Legacy sealed incomplete STOPs are archived first. Complete results remain immutable and altered sealed reports are rejected.
- Initialization accepts compatible runtime versions without requiring a particular Torch version string or a descriptive change reason. Runtime records do not collide when the same Slurm job is requeued.
- Invalid configuration still yields a report diagnostic and a nonzero exit code.
- Slurm scripts pass shell syntax checks and mocked submission tests for `prepare`, `compute`, and unattended `all`, including exports, canonical absolute paths, complete dependency chains, resumption without initialization-only variables, and non-array versus array log names.

The primary resume tests were run against the previous implementation and reproduced the fingerprint, eager teacher-loading, and incomplete-report failures before the fixes. The full suite retains the existing checkpoint, finite-value, source-isolation, and validity-gate tests. Synthetic tests do not authorize scientific advancement.

This validation is local. No HAIC jobs were submitted or existing HAIC artifacts changed. Real H100 inference and scheduler execution still require the HAIC environment and data.

## Original implementation — 2026-09-07

At the original implementation handoff, the real Experiment 0 had not been
executed. These checks establish implementation behavior, not evidence for the
scientific hypothesis. See the [study overview](../../docs/studies/future-innovation/)
for the latest documented local artifact status.

- **Repository regression suite:** 138 tests passed, including both official-source adapter tests; no skips in the final run.
- **Focused experiment suite:** 35 tests passed. These cover source selection/caps/folds, one-based annotation conversion, actual indexed video decoding, pixel/box geometry, past-only skeleton normalization, token masks/order, fixed projection, partition-local controls, weighted metrics/bootstrap multiplicity, checkpoint reload, complete predictions, and gate failure/stability cases.
- **Official V-JEPA integration:** used source commit `204698b45b3712590f06245fbfba32d3be539812`. Constructed the actual Hub ViT-B encoder/predictor without downloading weights. A small instance of the official V-JEPA 2.1 encoder verified pre-attention masking, exact future-pixel invariance, the actual patch-embedding flatten order, final-layer output, and official preprocessing equivalence.
- **Data flow:** synthetic CSVs and encoded video files exercised candidate construction, exact decode, an injected lightweight detector, pose normalization, 50-window cohort freeze, overlay generation, teacher-cache schema/checksums, resumption, and the complete ten-window pixel-edit audit. A constant fake teacher correctly failed the validity gate before fitting.
- **Model/report flow:** the CLI smoke run fit five outer folds × five arms × three seeds, selected from inner folds, saved and reloaded all 75 residual checkpoints and ten ridge/preprocessing objects, scored all predictions, bootstrapped sources, and wrote the decision/report. Synthetic results always forbid scientific advancement and adapter training.
- **HAIC scripts:** all shell/Slurm files passed `bash -n`. Mock submissions verified `afterok`/`afterany` dependencies, the five-fold array interface, failure cancellation, absolute log paths, and command argument forwarding. All 11 command help routes passed.
- **Static/configuration checks:** Python compilation, Ruff, `git diff --check`, and `uv lock --check` passed.

The final review checked source/inner-fold isolation, training-only preprocessing, separate background baselines, matched head capacity and seed schedules, target/reference units, bootstrap source multiplicity, missing-control failures, immutable configuration/artifact lineage, partial-stage recovery, and duplicate-writer protection. Shared MediaPipe primitives were extracted from the historical implementation so active research does not import archive modules.

Local execution used the existing macOS development environment (PyTorch 2.13.0); teacher integration dependencies were installed in a temporary directory. The HAIC environment remains explicitly pinned to PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, timm 1.0.15, and einops 0.8.1 in the updated lockfile. Full 384-pixel trained-checkpoint inference, real MediaPipe extraction/alignment, real target sensitivity, GPU resource usage, and actual Slurm scheduling still require HAIC and its full-GAVD inputs. The pipeline enforces its real-data audits before fitting.

Use [README.md](README.md) for the reproducible launch and validation commands. The local synthetic demonstration is under the ignored `outputs/future-innovation-implementation-smoke-v1/` directory; it is not a scientific result or a committed dataset.
