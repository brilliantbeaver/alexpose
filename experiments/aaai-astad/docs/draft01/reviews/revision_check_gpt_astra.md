# Revised manuscript claim-consistency check

Reviewer: GPT-6 Astra. Date: November 15, 2026. This is a read-only check of the revised manuscript, supporting sensitivity report, paired-difference figure data, README and submission-readiness assessment. I also checked saved reference artifacts and the timing-edit implementation. No models were trained and no new experimental results were generated. Final rendered-PDF layout verification remains with the coordinating author, who was rebuilding it during this check.

**Result:** no substantive remaining claim inconsistency was found in the reviewed text. The manuscript consistently presents an exploratory audit, preserves the 512-update primary analysis, and does not turn unresolved comparisons into evidence of no effect, equivalence, or biological falsification. The acknowledged empirical limits do not require another experiment before accurately reporting this bounded study.

## Small corrections before final rendering

1. **Name the latent random control explicitly in the table and figure.** In the reviewed `manuscript.md`, line 125 says “Random encoder, fitted heads,” while line 127 identifies the coordinate heads. The table caption at line 133 does not explicitly identify the first random row as latent. The paired-difference figure data also use “Random encoder, fitted heads.” Rename these to **“Random encoder, latent heads”**. Preferably add to the table caption that the unqualified own/full reference distances use latent losses. The body text already explains the distinction correctly at lines 137–139. The numerical results are unchanged.

2. **Minor copy edit:** manuscript line 65 says “a exponential-moving-average teacher.” Use “an exponential-moving-average teacher.” This has no scientific consequence.

The coordinating author was notified of the control-label correction before the final build. These are reporting edits, not reasons to reopen the protocol or switch outcomes.

## Claims checked and confirmed

- **No false absence or equivalence claim.** Lines 97 and 113 explicitly state that intervals containing zero do not establish equality or absence of benefit. The discussion and conclusion retain inconclusive language. The below-chance contrast at line 150 is not promoted to a significant reversal or reversed after inspection.
- **No biological-falsification claim.** Line 111 explicitly calls normal-prediction inadequacy a limitation of the tested learner, not a falsification of the biological hypothesis. Lines 73–75 separate fitted prediction benefit from intrinsic information, mutual information and neural coupling.
- **The primary analysis remains unchanged.** Lines 77, 113, 133 and 150 retain 512 updates as primary and label the 4,096-update run a normal-development-motivated sensitivity. Neither the favorable coordinate score nor a favorable category replaces the primary endpoint.
- **The random controls have the correct interpretation.** The new coordinate row reports 0.701 and 0.693, alongside trained-coordinate 0.686 and 0.690. The text notes common observable targets and separately fitted heads. It also states that random latent teachers change the target, preventing a clean encoder-only interpretation.
- **Source and reference dependence are accurately disclosed.** Saved `source_roles.json` files confirm 20 reference assignments using 15 distinct normal sources, pairwise overlaps of 2, 1, 1, 1, 0 and 0 sources, and disjoint fitting/reference/test sets within every fold. This agrees with line 93. All 23 normals' historical reuse remains explicit, and neither source IDs nor four folds are represented as independent participants or replications.
- **The variance-floor statement is supported.** Saved primary benefit references give latent sample variances from 0.046139 to 0.362444 and coordinate variances from 0.0000162903 to 0.00605884. The smallest regularized variances are approximately 0.062845 and 0.0000821763, respectively. Neither reaches the 1e-8 floor. These checks confirm line 89 without demonstrating that the reference estimates are precise.
- **The timing-edit interpretation is correct.** `study_controls.py` lines 41–45 select temporally offset frame slices and replace the selected limb's slice. This is a temporal shift, not a spatial translation. Manuscript line 162 correctly identifies the reported intervention effects as standardized profile-score changes, not raw coordinate MSE. A large coordinate-profile change therefore cannot be read directly as a large MSE change or a failure of translation invariance.
- **Observation selection and uncertainty remain visible.** The attrition table, conditional-interval caption, small development support, reused reference sources, and unverified participant identity are stated without claiming that a weak observation-only classifier removes all confounding.

## Readiness consistency

`submission_readiness.md` is consistent with the manuscript. It distinguishes technical preparation from scientific strength, preserves the original independent-review scores, describes the contribution as modest, and rates evidence separately for the bounded audit versus a superior detector or neurological claim. Its proposed future improvements are presented as ways to strengthen a subsequent claim, not as unfinished requirements that invalidate the current exploratory report. The README likewise preserves the primary schedule and historical-normal disclosure.

The manuscript is ready for final production checks under its present audit framing once the small control-label ambiguity is removed. No stronger detector, causal, clinical, independent-participant, or equivalence claim is supported or currently asserted.
