# Adversarial review dispositions

No approval score. Review findings are accepted/fixed, rejected with evidence, or unresolved. A software pass is not permission to invent empirical results.

## Gate1 — protocol

[Independent Codex protocol review](../../../../../gavd5-drift/notes/47-reviews/codex-protocol-review.md) inspected the plan **and current historical implementation**. Installed CLI help was checked. The first sandboxed invocation could not initialize; the same read-only review was then run with approved external process access. No model/reasoning override was forced; the installed default reported GPT-6 Astra. The review did not edit code or run expensive training.

| Finding | Disposition | Evidence / falsification |
| --- | --- | --- |
| Success/uncertainty rule missing before confirmation | accepted/fixed | protocol evaluation section and `workflow.lock_and_calibrate`; tests exercise improvement point/interval classifications |
| Primary across-seed estimand unspecified | accepted/fixed | `statistics.paired_group_bootstrap`: mean per-seed source errors, same group draws, ratio of error means; unequal-source/seed arithmetic tests |
| Historical raw inference/intervals cannot be reproduced from saved summaries | unresolved evidence limitation, correctly disclosed | historical ledger preserves exact aggregate matches but no synthetic reconstruction of missing raw artifacts |

## Local independent cross-review

The model owner reviewed the coordinator's data/workflow (not only their own code). Supported defects were fixed: training receipt status mapping, teacher-interval/endpoint separation, test-open identity binding, and phase-independent per-task writer locks. See [local cross-review](local-review-model-agent.md) and `test_review_regressions.py`. Additional integration checks cover artifact atomicity, explicit checkpoint/dataset identity, no test refitting, complete-grid detection and failed-attempt preservation.

## Gate2 — implementation

[Independent Codex implementation review](../../../../../gavd5-drift/notes/47-reviews/codex-implementation-review.md) returned four supported findings. The reviewer could inspect code but its scratch-dependent tests were sandbox-blocked; the coordinator reruns them in the configured local test environment. This is an evidence limit, not a passing reviewer test run.

| Finding | Disposition | Fix / verification |
| --- | --- | --- |
| P1 modified prepared cache could bypass later audit receipt | accepted/fixed | Every train/development `load_role` verifies preparation01. Test cache gets07-data receipt. Fault injection changes a coordinate after audit and rejects before model access. |
| P2 partial summary publication prevented retry | accepted/fixed | Fresh comparison-attempt directories preserve failed summaries. Injected failure during summary and before receipt; retries must preserve old files, locked choices and fitted predictors. |
| P2 resume checkpoint conflicted with immutable Slurm grid | accepted/fixed | `scientific_dict()` excludes runtime resume path consistently; submit requires one explicit task and exports/logs checkpoint. Mocked resume succeeds; changed LR is rejected. |
| P2 written protocol not bound to analysis identity | accepted/fixed | Governing protocol/decisions are hashed and snapshotted. Changed source bytes or snapshot reject resume/cache access. |
| Notebook/builder markdown differed during concurrent edits | accepted/fixed | Canonical notebooks regenerated after builder edits; source-equality tests rerun before freeze. |
| HPC cross-check: CUDA direct baseline had CPU-only audit allocation | accepted/fixed | Audit launcher now requests one H100. Regression parses the emitted real pilot DAG and verifies GPU allocations for all CUDA-consuming stages without opening nonexistent manifests or submitting jobs. New retained run `software-20260915-02` passes 77 tests and 15 fresh-kernel executions. |

Integration also found and fixed canonical-path aliases in receipt outputs and the impossibility of untouched test membership within `historical_overlap`; that scope now rejects final locking/opening early. Final counts and retained execution artifacts are in the [results status](../results/README.md).

## Gate2 follow-up — independent software evidence review

The [Codex follow-up](../../../../../gavd5-drift/notes/47-reviews/codex-software-evidence-review.md) independently passed28 scratch-free tests, exercised the real-config scheduler dry run without manifests/submissions, reconstructed882 primary source scores/294 summaries/four bootstrap contrasts, and replayed15 selected-task prediction methods from retained checkpoint/readouts without refitting. These checks used `software-20260915-02`. Its13 receipts passed before subsequent source changes correctly invalidated current-code compatibility. The CLI retained the response through `--output-last-message`; the review agent itself remained read-only.

| Follow-up finding | Disposition | Final evidence / limitation |
| --- | --- | --- |
| Derived test inventory was read without receipt00 | accepted/fixed | `_prepare_role` and `evaluate_test` authenticate00 before metadata/media use or opening marker. Actual tiny-pipeline tests mutate both role and path and assert zero reader calls. |
| Direct test helpers trusted a changed analysis plus matching opening marker | accepted/fixed | Shared analysis verifier authenticates07-lock. Changed selection and missing receipt both reject before cache/fixture/video reads; original07-data alone cannot authorize access. |
| Missing secondary horizon aborted supported primary readout | accepted/fixed | Explicit serialized `not_estimable` secondary with support metadata/NaNs; no cross-horizon borrowing. Tests cover zero/partial/heldout-only absence, strict primary failure, old-fit compatibility and full frozen-test replay. |
| Current-code verification became stale during fixes | accepted/fixed | Preserved01/02; final03 contains15 successful fresh kernels,85 passing study tests,6 layout tests and current-identity/receipt validation. The final rerun is coordinator-observed, not misattributed to Codex. |

The model owner also independently inspected the evaluator fix (not their own implementation) and ran its four focused tests: all passed. Compatibility here means legacy-shaped synthetic fits from this new study, not absent historical checkpoints. Shared-context decoders intentionally keep their original primary-selected penalty for all output heads; the repair does not introduce cross-horizon borrowing in horizon-specific decoders.

The independent numerical replay and all final execution details are linked from [results](../results/README.md). This closes the supported software findings with code and regression evidence, not an arbitrary approval score. Real-data provenance, full-cohort resources and empirical utility remain unverified; they cannot be closed by synthetic tests.

## Gate3 — empirical evidence/claims

**Pending real HAIC artifacts.** A local review can assess synthetic labels and restraint of claims, but cannot validate real source coverage, CUDA behavior, causal upstream extraction, model utility or test inference without the actual retained run. Before changing the paper, rerun the scoped Codex prompt with the exact empirical artifact manifest and changed manuscript list. Preserve the historical scientific stop and disclose any null outcome.
