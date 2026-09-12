# Adversarial review and resulting corrections

Two independent reviews challenged the implementation and interpretation. A parallel evidence reviewer traced the code and reproduced a reporting-verification gap with temporary synthetic artifacts. A separate **Codex CLI adversarial review** inspected the new code, protocol, notebooks, reports, mathematical claims and related work. It independently recomputed both panels' scores and primary bootstrap intervals from saved predictions without refitting. Its conclusion was that the work is useful as a development study but does not yet support a new ICLR method claim.

The original review is retained unchanged at [codex-adversarial-review.md](../../../work/artifacts/iclr-bridge-2026-09-11/codex-adversarial-review.md). Its [prompt](../../../work/artifacts/iclr-bridge-2026-09-11/codex-review-prompt.txt) and [execution log](../../../work/artifacts/iclr-bridge-2026-09-11/codex-review.log) record the actual invocation. The first sandboxed invocation failed to initialize the local app-server client (`Operation not permitted`); the approved rerun completed in read-only mode. No reviewer changed files or ran GPU work. A separate [follow-up review](../../../work/artifacts/iclr-bridge-2026-09-11/codex-followup-review.md) checked the corrections and found all four concrete findings addressed, with no unresolved correctness issue in that scope. It explicitly retained the scientific limitations and did not endorse submission readiness.

## Concrete findings and actions

| Finding | Correction | Evidence after correction |
|---|---|---|
| The original verifier checked prediction arithmetic but omitted some top-level report/seal fields and displayed candidate/training diagnostics. | Added `verification_supplement.py` as a separate post-fit module. Kept frozen fitting code, protocol and sealed run unchanged. | Six tamper tests reject coherently rehashed wrong metadata, missing seal entries, incorrect displayed gains/flags/reasons and altered training diagnostics. Real supplemental check passes for all 1,480 candidates and 40 diagnostics. |
| Real history changes confidence representation as well as coordinates, and duplicates some reference columns. | Narrowed the claim throughout the tutorial, paper and notebooks to the declared coordinate/confidence history block. The next experiment must hold confidence/validity/support fixed and include them once. | Independent [confidence-route diagnostic](../../../work/artifacts/iclr-bridge-2026-09-11/future/confidence-route-diagnostic.json) confirms 182 raw-versus-valid-conditioned differences and 31 standardized duplicate confidence columns across all 50 cached rows in each fold. No effect attribution was inferred. |
| “Read-only” was inconsistently described as requiring no fitting. | Notebook 21 now states that default verification performs CPU reconstruction refits without artifact writes. Added a separate `inspect_cached_panel` API and `RECONSTRUCT_MODELS=False` path for strict no-fit integrity inspection. | Two focused tests prohibit model loading/refitting during inspection, reject changed digests/inventory and verify no writes. Notebook 21 was re-executed after the change. |
| A tutorial sentence said a smaller, uncertain estimate was why the primary contrast was selected. | Replaced it with the methodological reason and explicit prospective timing. | Frozen protocol digest and creation record precede real fitting. The reviewer found no implementation evidence of outer-outcome selection; the error was in the explanation. |
| A validation-document link was temporarily missing, and the audit's “no top-level notebooks” statement had become stale. | Added the validation report and labeled the inventory as the pre-bridge snapshot, separately identifying notebooks 19–22. | Local-link and source/executed-cell audit passes after the final documents are present. |

Duplicated features deserve a precise interpretation. If a column is present in two blocks with penalties λ_x and λ_s, a fixed total coefficient can be divided between them. Minimizing its penalty yields effective shrinkage (1/λ_x + 1/λ_s)⁻¹. That is smaller than either penalty within that joint model, but it need not be smaller than the separately selected reference penalty. The current data do not identify how much this route changes the primary estimate. The diagnostic does not justify correcting or replacing that estimate after inspection.

The scientific design was **not amended after fitting**. Reporting and verification were strengthened, and the scope of the claims was narrowed. A new confidence-matched or reference-locked comparison would require its own prospective protocol and run. The negative/uncertain result remains valid for the feature family that was actually fitted.

## Claims retained and claims not established

The reflection-odd/time-even characterization of the laterality observable holds under the specified anatomical swap, coordinate flip and reversal of physical intervals. The conditional-expectation identity holds for nested information sets, finite second moments and a common squared-error metric. RGB complementarity is different from student-only accessibility. Exact symmetry, feature energy and low latent prediction loss do not by themselves establish useful motion information.

The saved scores, masks, source-bootstrap multiplicities and completed-stage reuse are verified. Laterality evidence in this checkout remains aggregate-only: absent raw checkpoints and predictions prevent new inference and bootstrap reconstruction. Neither synthetic calibration nor the 43-source development comparison establishes independent human-motion forecasting, beneficial student transfer, causal dynamics or agentic planning.

Novelty remains an empirical opportunity. Future privileged supervision, spectral guidance and invariant/equivariant representation splits have close precedents, including SGDD, S-JEPA, MAMP, SIE and seq-JEPA. A strong contribution would show that the proposed student-accessibility rule predicts useful transfer or justified abstention beyond matched alternatives. The current implementation creates the measurement and calibration foundation for that experiment; it does not supply the missing trained-student result.

## Reproduction

The actual adversarial invocation was:

```bash
codex exec -s read-only --ephemeral -C /Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6 -o work/artifacts/iclr-bridge-2026-09-11/codex-adversarial-review.md - < work/artifacts/iclr-bridge-2026-09-11/codex-review-prompt.txt
```

The evidence review and temporary reproduction are also retained in [the laterality audit directory](../../../work/artifacts/iclr-bridge-2026-09-11/laterality). Focused tests and the 125-test future-innovation regression suite are documented in [the validation report](03_implementation_and_validation.md). The review is an additional challenge to the evidence; it does not replace future experimental confirmation or author responsibility for the final submission.
