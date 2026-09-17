**The four earlier fixes are present. One reproducible implementation defect and one current-evidence gap remain.** This is an E0–E2 software review, not empirical gate 3 approval.

During review, another writer changed `workflow.py` and added integrity tests. The findings below reflect that newer code. I made no implementation changes.

1. **P2 — Missing secondary-horizon support aborts an otherwise supported primary analysis.**

   **Location:** [evaluation.py:433](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/src/gavd6_sjepa/research_directions/temporal_gait/evaluation.py:433), particularly the unconditional horizon loop at line 440; the exception originates at line 301. The measurement gate checks primary support in [information_audit.py:105](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/src/gavd6_sjepa/research_directions/temporal_gait/information_audit.py:105).

   **Evidence:** An executed, in-memory diagnostic retained supported 0.50-second endpoints but removed 1.00-second support. The measurement gate returned `true`, and the shared context readout fitted successfully. Horizon-specific decoder fitting then raised `Train-grouped ridge validation has no complete observed primary support`. `_evaluate` fits every decoder unconditionally, so this exception prevents publication of the supported primary comparison.

   **Consequence:** Real bouts or missing observations can pass E0 and consume training resources, then fail development evaluation solely because an exploratory horizon lacks inner-validation support. This conflicts with the protocol’s treatment of unavailable targets as unscored observations.

   **Minimal fix:** Represent unsupported secondary readouts explicitly as unavailable and continue supported primary evaluation. Preserve missing-prediction accounting; do not manufacture secondary predictions or select penalties using development/test data.

   **Falsification test:** Give training/development adequate primary support and no 1.00-second support, including a variant where only the inner holdout lacks that support. Require primary evaluation to complete, secondary unavailability to be retained, and primary results to remain unchanged.

2. **P2 — The retained verification bundle no longer identifies the current workflow.**

   **Location:** [results/README.md:14](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/docs/studies/temporal-gait/results/README.md:14), [software-verification.json](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/docs/studies/temporal-gait/results/software-verification.json), and [config/identity.json](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/outputs/temporal-gait/software-20260915-02/config/identity.json).

   **Evidence:** Early in this review, the actual validator successfully checked all 13 receipts. After the concurrent edit, identity verification correctly rejected the run. Its recorded `workflow.py` SHA256 starts `0e150bb40887`; the reviewed current file starts `ca9b2c3e5b9a`. The summary still points to `software-20260915-02` and the coordinator-observed 77 tests.

   **Consequence:** Those artifacts remain credible evidence for their frozen implementation, but cannot establish execution of the newly added inventory/lock guards and regressions. This is expected incompatibility enforcement, not a failed historical run or evidence of corrupted predictions.

   **Minimal fix:** Freeze the final implementation, run the affected regressions and bounded software verifier in a new root, then update the verification references. Preserve `-01` and `-02`; do not rewrite their identities.

   **Falsification test:** The new bundle must pass current-code identity and receipt verification, cover the final task grid, and retain successful notebook execution and accurate test provenance.

**Earlier findings and subsequent fixes**

| Finding | What currently verifies the fix | Limits |
|---|---|---|
| Prepared-cache receipt access | [workflow.py:99](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/src/gavd6_sjepa/research_directions/temporal_gait/workflow.py:99) verifies preparation receipt `01` before train/development loading and `07-data` before test-cache loading. `test_modified_cache_rejected_before_training_and_development_model_access` changes a cached coordinate and checks rejection before model access. | I inspected that fault test; its scratch-dependent execution remains coordinator-reported. |
| Partial-summary retry | `_fresh_output` now applies to development and test summaries. `test_summary_write_failure_and_pre_receipt_failure_recover_without_overwrite` covers both development failure windows; `test_locked_test_summary_retry_retains_original_lock_and_fits` checks preserved summaries, lock/opening hashes and reused fits. | These fault-injection tests were inspected, not rerun here. |
| Explicit single-task Slurm resume | [config.py:77](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/src/gavd6_sjepa/research_directions/temporal_gait/config.py:77) canonicalizes `resume_from`; grid creation uses that representation. [submit.py:76](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/scripts/research_directions/temporal_gait/submit.py:76) requires one explicit training task. Tests cover unchanged grid bytes, exact checkpoint export/logging, scientific-setting rejection and wrong-family rejection. | Mocked submission establishes orchestration behavior, not actual Slurm or CUDA restart behavior. |
| Governing-document binding | [contracts.py:69](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/src/gavd6_sjepa/research_directions/temporal_gait/contracts.py:69) includes protocol and decisions in the fingerprint; freezing retains snapshots, and verification checks their hashes. The regression covers changed protocol identity and corrupted snapshot contents. | Binding preserves the reviewed text; it does not prove manifest attestations or scientific validity. |

The **audit GPU correction is supported**. `_fit_mlp` uses `cfg.device`; [audit-baselines.sbatch:5](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/slurm/temporal-gait/audit-baselines.sbatch:5) now requests one H100. I executed the actual real-pilot dry-run path with nonexistent manifest paths and guards prohibiting submission, manifest reads and grid writes. Its emitted audit, masked, future and evaluation launchers all request GPUs; evaluation depends on both complete arrays. This separately confirms the behavior targeted by `test_emitted_real_pilot_cuda_stages_request_accelerators`.

The inventory gap identified during review is **fixed in the newer code**: [workflow.py:49](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/src/gavd6_sjepa/research_directions/temporal_gait/workflow.py:49) authenticates inventory before preparation, and line 281 does so before public test inventory access. `_verify_analysis_lock` now verifies receipt `07-lock`, preventing an edited analysis plus matching opening marker from authorizing direct readers. Four added regression tests exercise these boundaries using actual synthetic setup. I inspected them; current retained execution evidence is still missing as described above.

**Executed checks and retained evidence**

- **28 existing tests passed**, covering timing, future-input isolation, masking gradients, independent teacher intervals, source arithmetic, notebook regeneration and shell syntax. No scratch-dependent workflow tests or fresh kernels were rerun.
- All **13 receipts and 105 distinct receipted outputs** validated before the concurrent workflow change.
- All **15 retained notebook hashes** matched; code cells had execution counts and no error outputs. Fourteen executions were synthetic, including gated extensions; the real-mode execution was plan-only. This verifies retained records, not a fresh kernel rerun.
- Independently recomputed **882 source-score rows, 294 method/horizon summaries**, development selection and **four bootstrap contrasts**, including relative-improvement intervals.
- Replayed **all 15 prediction methods for the selected synthetic test task’s 48 windows**, using retained checkpoint/readouts without refitting. Model/evaluation source hashes still matched the frozen bundle; this standalone replay does not validate the changed workflow.
- Verified synthetic train/development/test video, group and window disjointness.

The reported **77 study tests and 6 layout tests remain coordinator-observed evidence**. Their missing separately retained stdout is disclosed; I do not claim to have rerun those suites. A shell here-document attempt was sandbox-blocked before tests started; the 28-test run subsequently succeeded through a write-free invocation.

The retained selection—`masked`, `online_context_ridge`, versus `periodic`—correctly reports statistical failure, `synthetic_only`, and no expansion readiness. Its negative score is not a software failure.

**Information boundaries and handoff**

Current code supports prefix-only geometry/sampling, separate endpoint and strict teacher intervals, prepared-feature withholding, train-only fitted readouts, equal-bout/equal-video reductions, paired complete-group bootstraps and averaging seed errors rather than coordinates. Observed-future decoding remains privileged and excluded from selection. The executed mutation checks establish downstream array/model behavior; they cannot prove upstream extraction/tracking causality, complete identity links or untouched-source attestations.

Uncapped indexing and retained late windows are supported. Role-wide RAM assembly remains explicit and unmeasured at full-cohort scale. Neither synthetic timing nor launcher allocations establish capacity or throughput.

- **Software handoff:** resolve the secondary-horizon failure and refresh evidence against the final frozen code.
- **Real HAIC submission:** additionally requires explicit audited manifests, causal complete-bout exports, compatible CUDA/decoder execution and measured pilot resources.
- **Empirical paper claims:** remain pending real retained predictions, coverage, locked comparisons and independent evidence review. E3, RGB transfer, raw-pose extraction, calibrated uncertainty and exact historical replay remain legitimate disclosed deferrals.

Historical results and manuscript claims were not revised. No files were edited; filesystem restrictions prevented writing the requested review file directly.