# Adversarial review: constraint conflicts and common execution

**Historical review.** The constraint-conflict candidate was replaced by [motion beyond keypoints](../../proposals/motion-beyond-keypoints.md). Its rank and candidate number below refer to the earlier portfolio.

Reviewed the full [archived constraint proposal](rejected-constraint-conflicts.md) and [common evidence and execution contract](../execution-contract.md). Changes were restricted to that proposal and this review log. This is a research-design review, not a proof implementation or an experiment result.

## Substantive fixes applied to Proposal 7

| Issue | Why it mattered | Revision |
| --- | --- | --- |
| A balanced feasible/conflict/unresolved pilot treated unknown like ground truth | Unknown depends on the algorithm and time budget. Selecting already solved cases biases reported coverage. | Freeze the query-generating process, retain every sampled query in the denominator, and use a separate witness/certificate reference bank for checker validation. |
| Floating-point solver status could be mistaken for proof | A numerical residual alone does not certify the exact original constraint system. | Start with an outward-rounded linear outer relaxation and exact rational checking of infeasibility certificates. Failed checks remain unknown. |
| The original full-body requirements were underspecified | Upper bone-distance constraints alone permit shortened or collapsed skeletons. | State bone-length tolerances and joint limits in the original model, then explicitly drop lower distances and nonconvex limits in the outer relaxation. |
| Feasible outer relaxation could support a false minimality claim | An outer solution may violate the original bone or joint constraints. | Every single-rule deletion needs an actual original-problem motion witness before claiming inclusion-minimality. Otherwise report only a verified conflicting subset. |
| Pairwise feasibility was asserted without a verification requirement | An apparent higher-order conflict could actually contain a trivial incompatible pair. | Require a checked original-problem witness for every pair used in this claim. |
| Matching nuisances was described as preventing shortcuts | Matching selected summaries cannot rule out remaining correlations. | Require explicit nuisance-only tests and acknowledge the limit of matching. |
| Correctness was partly attributed to learned ordering | All methods share the same independent acceptance checker. | Make certified resolution coverage and time to resolution the learned method's outcomes. Checker correctness is a shared condition, not a model improvement. |
| Comparing raw call counts hid unequal computation | One diffusion call, one solver call and one retrieval call have different costs. | Use end-to-end wall-clock budgets, record CPU/GPU resources separately, and report adaptation-cost recovery across query counts. |
| Missing inexpensive ordering baselines | A neural head could merely approximate obvious constraint slack or graph heuristics. | Add classical conflict extraction, slack/graph ordering, equal-capacity raw-coordinate and random-encoder heads. |
| Incomplete-observation novelty was not backed by the specified pilot | The pilot begins from declared initial poses and kinematic queries. | Narrow the possible contribution to that actual setting. |

The certificate route still needs implementation. For a rational linear system, the checker must verify nonnegative certificate weights, exact cancellation of variable coefficients and a strict contradictory constant. The bounds must be constructed outward so every original feasible motion remains inside the relaxation. If these conditions are not established, the result is unknown. This is a concrete implementation specification, not evidence that the proposed benchmark admits enough useful certificates.

## Remaining reasons to rank this last

[DNO](https://arxiv.org/abs/2312.11994), [retrieval-guided DNO](https://arxiv.org/abs/2605.08054) and [MIC](https://arxiv.org/abs/2607.01990) already provide substantial constrained-motion-generation capabilities. [Conflict-driven task-and-motion planning](https://arxiv.org/abs/2211.15275) already reasons about incompatible subsets. A motion domain, an explanation sentence or a shared verifier is not a sufficient novelty claim.

The difficult gap is also structural. Relaxations tight enough to certify interesting conflicts may already be cheap for classical solvers. Relaxations loose enough to be hard may return unknown even when a strong prior fails repeatedly. No amount of model confidence closes that proof gap. The one-week plan therefore correctly requires a nontrivial certificate path and a demonstrated search bottleneck before any adaptation job.

Proposal 7 should remain seventh overall, explicitly exploratory, with no training recommendation before the 48-hour gate. Its potential interface is interesting, but current evidence does not justify a high novelty or feasibility score merely because formal guarantees sound ambitious.

## Read-only review of the common contract

No substantive correction was required. The common contract distinguishes proposed budgets and thresholds from findings; keeps the confirmation cohort sealed; uses source/person grouping; separates optimization seeds from sampling uncertainty; gives whole-body AMASS and GAVD appropriate roles; and prevents automatic reuse of a large Wan recipe as a cheap baseline.

The stated compute ceiling is arithmetically correct: eight GPUs times seven days times 24 hours equals 1,344 GPU-hours. The two-pilot cap and single-flagship decision are appropriate. If the optional Proposal 2 pilot is started on day two, one of the original two pilots must first finish or stop. The table's existing cap already implies this; it should be followed operationally.

The common promise of matched pixels and future visibility should be applied to methods at the same teacher or deployed-student role. Privileged teacher information must not become accidental deployed-student access. The role-specific proposals already state that distinction, so no common-text edit was necessary.
