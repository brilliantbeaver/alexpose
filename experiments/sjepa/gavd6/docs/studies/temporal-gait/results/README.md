# Verification and result status — 2026-09-15

**Software verified; real GAVD experiments not run.** No HAIC connection, private-data discovery, GAVD download or real scheduler submission was performed. This task did not revise the historical negative result or manuscript; concurrent manuscript changes in the shared workspace were left untouched and recorded in the [evidence ledger](../development/evidence-ledger.md).

| Layer | Executed evidence | Status / limitation |
| --- | --- | --- |
| Study tests | 85 passing `unittest` cases | Software contracts only; includes real decoding of a generated tiny video, not GAVD |
| Repository integration | 6 passing layout/registry tests | Additive study registration; unrelated worktree changes preserved |
| Notebook execution | 15 successful fresh-kernel executions, all 9 canonical notebooks covered | Explicit synthetic stages plus a real-mode **plan-only** render with nonexistent manifest paths |
| Scheduler | `bash -n`, dry-run and mocked submission/dependency/resume tests | No Slurm submission; `shellcheck` unavailable locally |
| Retained artifacts | All 13 present stage/task receipts and their output hashes verified | Run identity binds code, configuration, runtime, manifests and governing documents |
| Development or sealed-test GAVD evaluation | None | Pending explicit manifests, compatible HAIC environment and measured pilot |

The [machine-readable verification summary](software-verification.json) records commands and scope. Current local execution evidence is retained under [software-20260915-03](../../../../outputs/temporal-gait/software-20260915-03/): [kernel executions and hashes](../../../../outputs/temporal-gait/software-20260915-03/verification.json), [claim audit](../../../../outputs/temporal-gait/software-20260915-03/reports/claim-audit.json), run-specific notebooks, source/window predictions, fitted readouts, initialized/online/teacher checkpoints, loss traces and bootstrap artifacts. This output directory is local research output, not a promise that generated artifacts are tracked in Git. Earlier `software-20260915-01` and `-02` remain preserved under their own pre-fix identities; neither is a compatible resume of the final code.

The 15 notebook calls took 82.147 seconds in total on this local CPU verification run (2026-09-15 16:12:40–16:14:02 UTC). This tiny-fixture timing is **not** a full-GAVD elapsed-time or GPU estimate. Synthetic selection is labeled `synthetic_only`, cannot authorize real expansion, and establishes neither real forecasting nor JEPA utility. Optional E3 and RGB cells returned explicit gated status; they did not train those comparators. The synthetic lock/test exercise tests software boundaries, not an untouched human cohort.

The [independent arithmetic replay](artifact-replay.md) and its [read-only diagnostic](artifact_replay.py) reconstruct all 882 primary source-score rows, 294 primary summaries, secondary scores and four paired comparisons from retained predictions. Repeated rows/methods are not independent observations. The selected fixture model loses to the periodic baseline; that negative toy outcome remains visible and is not a GAVD result.

## Reproduce the software checks

From `gavd6/`, using the repository's existing environment:

```bash
env MPLCONFIGDIR=/tmp/temporal-gait-matplotlib-tests PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests/temporal_gait -t . -q
env PYTHONPATH=src .venv/bin/python -m unittest tests.infrastructure.test_layout -v
.venv/bin/python scripts/research_directions/temporal_gait/verify_software.py --run-root /ABSOLUTE/NEW/EMPTY/software-verification-run
.venv/bin/python scripts/research_directions/temporal_gait/validate_run.py --config /ABSOLUTE/NEW/EMPTY/software-verification-run/config/resolved.json --verify-artifacts
```

Replace the uppercase placeholder with an explicitly chosen absolute writable directory. The verifier refuses to reuse a nonempty run, uses explicit synthetic fixtures, and never discovers private data. Jupyter kernels need local loopback sockets; a sandbox that forbids sockets requires approval or execution in the user's configured environment. Canonical notebooks remain output-free.

## Review disposition and real-run prerequisites

The independent [protocol](../../../../../gavd5-drift/notes/47-reviews/codex-protocol-review.md), [implementation](../../../../../gavd5-drift/notes/47-reviews/codex-implementation-review.md), and [implementation-fix/software-evidence](../../../../../gavd5-drift/notes/47-reviews/codex-software-evidence-review.md) reviews were performed using the installed Codex CLI. Supported defects were fixed and fault-tested: prepared-cache integrity, failed-summary recovery, explicit resume identity, governing-document binding, audit GPU allocation, final-test inventory/lock authentication and unsupported secondary-horizon handling. The last review correctly required new evidence after concurrent fixes; this final `-03` bundle supplies it. It is a coordinator-verified follow-up, not a claim that Codex reran every final test. Consult [dispositions](../development/review-dispositions.md), not an earlier verdict alone. The empirical evidence/claims gate remains pending real artifacts; software verification does not replace it.

Follow the [HAIC runbook](../execution/haic.md). Required inputs are explicit full-video, complete-walking-bout, full-bout pose/provenance, duplicate/verified-person identity, exposure, reservation and split manifests; an explicit output root; and the user's interpreter/account/partition/resource choices. All required source rosters must agree. Pose caches must preserve every source frame in each supplied complete bout, original PTS, missing detections and causal extraction provenance. This milestone validates supplied caches against supplied videos; it does not provide raw-video pose extraction.

The first historical-overlap pilot is a mechanism/development study, not a sealed test. Full-cohort expansion needs a separately configured `full_allowed` run. Architecture/seed expansion, final protocol lock and test opening are gated on development evidence, not job exit. A real CUDA/decoder pilot must establish memory and throughput before estimating full-run cost. All prepared windows are retained without a sequence cap, but the current trainer loads its allowed role's arrays into RAM; resource sizing remains an empirical prerequisite.

E3 dense × intermediate-layer objectives, released-video-checkpoint transfer, uncertainty calibration and clinical/physical-unit claims remain explicitly unimplemented. Add them only under the plan's subsequent gates. A null JEPA result, a timing/support explanation, or a stronger direct predictor is a valid scientific outcome.
