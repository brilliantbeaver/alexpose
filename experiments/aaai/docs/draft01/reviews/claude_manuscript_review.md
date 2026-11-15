The Salesforce SessionStart hook is irrelevant to this task. Ignoring it. This is a manuscript review, using no tools as instructed.

# ASTAD Workshop Review: "Auditing Predictive Context for Gait Anomaly Detection"

I have read the manuscript and all three supporting documents. Below is my independent audit. I treat the actual claims as written, not an inflated hypothetical.

## Bottom line up front

This is an honest audit / negative-result contribution, and it is unusually disciplined about not overclaiming. The central message, that a biologically motivated predictive-context diagnostic fails to establish an advantage over simpler controls on this cohort, is coherent and useful. The primary risk to the paper is not overclaiming but the opposite: a reader may finish uncertain what was learned. There are also several concrete mathematical and reporting errors that must be fixed, and at least one internal inconsistency in the counts that will not survive a careful reviewer.

---

## Errors and problems, by severity

### HIGH severity

**H1. Normal-source accounting across folds does not reconcile with the stated cohort.**
Offending claims: "23 normal sources" (abstract, data section); "Their test populations contain respectively 6, 6, 6 and 5 normal sources" plus "Five non-test normal sources per fold serve as development observations and the normal reference" (Evaluation section).

Why it matters: Test normals sum to 6+6+6+5 = 23. That is correct and clean, test folds partition all 23. But each fold also uses 5 non-test normals as reference, and "leaving 12 or 13 normal fitting sources." For a given fold, the arithmetic is 23 total minus test normals minus 5 reference. Fold with 6 test normals: 23 - 6 - 5 = 12 fitting. Fold with 5 test normals: 23 - 5 - 5 = 13 fitting. That reconciles. However, the 5 reference sources are described as "non-test" and "per fold," which means they are drawn from the 17 or 18 non-test normals in each fold. The paper never states whether the five reference sources are disjoint from the fitting sources (they must be, since fitting is normal-only and reference is excluded from fitting) nor whether reference sets overlap across folds. With only 23 sources and 4 folds each consuming 5 reference sources, reference sets necessarily overlap heavily across folds, which couples the four fold estimates. This is not disclosed and it weakens the claim that folds provide independent evidence.

Concrete edit: Add one sentence and ideally a small table giving, per fold, the exact identities or at least counts of test / reference / fitting normal sources, and state explicitly how much reference-source overlap exists across folds. Then qualify the between-fold interpretation: the four fold AUROCs are not independent because reference and fitting normals are reused.

**H2. The primary claim risks being internally undercut by the control that beats it, and the paper does not resolve the tension.**
Offending claims: "the primary latent profile reaches AUROC 0.641 versus 0.541 for raw full-context KL" presented as the primary contrast, while "A calibrated full-error distance reaches 0.673, exceeding the profile point estimate" and "Random encoder, fitted heads 0.651."

Why it matters: The paper's chosen primary contrast (profile vs raw full error) is the one contrast the profile "wins" on a point estimate. But the two most informative controls, the full-error reference distance (0.673) and the random encoder (0.651), both meet or exceed the profile. The honest reading is that the profile has no demonstrated advantage over a simpler reference-calibrated error, and the primary contrast is partly an artifact of comparing against the weakest comparator (raw, uncalibrated full error). The manuscript does say this in prose ("the raw-error comparison does not isolate a benefit"), which is to its credit, but the framing still leads with the one favorable number. A demanding reviewer will read the headline 0.641 vs 0.541 as mild spin.

Concrete edit: Change the primary-result framing so the headline is the null result against the strongest control, not the favorable result against the weakest. Lead Results with: "The profile does not outperform a reference-calibrated full-error distance (difference -0.031, 95% CI [-0.152, 0.096]) or a random-encoder profile." Keep the raw-error contrast but demote it. This changes no analysis and no number; it changes which number is foregrounded, which is exactly the kind of pre-registered-spirit fix the task asks for.

**H3. The bootstrap confidence interval as transcribed is implausibly precise and inconsistent with the rounding elsewhere.**
Offending claim (sensitivity doc): "conditional95%CI [-0.03508771929824558, 0.2242753623188406]."

Why it matters: A percentile bootstrap over 1,000 replicates of a weighted AUROC difference cannot have a lower bound that lands on a value with a denominator like 1/57 (−0.0350877... = −2/57 approximately, and 0.2242753... has the texture of a ratio with a small integer denominator). These look like single order-statistic values from the bootstrap distribution, which is fine, but reporting 16 significant digits on a quantity estimated from 1,000 resamples of ~23 and ~57 sources is false precision and signals the interval was copied from a float without rounding. More importantly, bootstrap with only 5–6 positive (normal) sources per fold and label-stratified resampling means the number of distinct achievable weighted-AUROC-difference values is small; the percentile endpoints are discrete and unstable. The paper should report Monte Carlo error or at least acknowledge the granularity.

Concrete edit: Round to [-0.035, 0.224] everywhere (the manuscript already does this; fix the sensitivity doc). Add a sentence that with 5–6 normal sources per fold the bootstrap distribution of the weighted AUROC is highly granular and the endpoints should be read as coarse. Optionally report the number of distinct resampled values or the Monte Carlo standard error over replicate seeds.

**H4. AUROC with 5–6 positive-class-defining normal sources per fold is being over-interpreted as a stable estimate.**
Offending claim: per-fold AUROCs "0.588, 0.646, 0.629 and 0.716" and category AUROCs reported to three decimals (e.g. cerebral palsy 0.486 on 10 abnormal sources, Parkinsons 0.565 on 9).

Why it matters: The scoring is source-level. A fold's AUROC is computed over roughly 5–6 normals against ~57 abnormals. The AUROC is then a rank statistic over 5–6 positive items (or 57, depending on which class is "positive" for the ROC, but the weighting is by normal count, so the normal side drives it). With so few sources on the limiting side, each AUROC has an enormous standard error and three-decimal reporting is not warranted. The category table is worse: a 0.486 AUROC on 10 cerebral-palsy sources against reused normals is essentially a coin flip with a confidence interval spanning most of [0,1].

Concrete edit: Report all AUROCs to two decimals. For the category table, either drop it to the appendix or add per-cell interval widths / mark the two- and three-source categories as uninterpretable (the paper already flags dependence; it should also flag that these point estimates carry no usable precision). State the effective sample size on the limiting class for the headline number.

### MEDIUM severity

**M1. The anomaly-score denominator mixes a per-region variance with a pooled mean variance in a way that is undocumented and affects results.**
Offending claim: $D(s)=\frac{1}{3}\sum_g \frac{(b_{sg}-\mu_g)^2}{\max(0.5 v_g + 0.5\bar v, 10^{-8})}$.

Why it matters: The denominator shrinks each region's variance halfway toward the mean regional variance $\bar v$. This is a reasonable James-Stein-flavored shrinkage, but (a) the 0.5/0.5 weighting is a free hyperparameter never justified or varied, (b) with only 5 reference sources, $v_g$ is a 4-degree-of-freedom variance estimate and is extremely noisy, so the shrinkage is doing real work and the leave-one-reference-out ranges (0.585–0.673 latent) likely reflect exactly this instability, and (c) the floor $10^{-8}$ interacts with squared benefit values whose scale is not reported, so it is impossible to tell if the floor ever binds. This is a hidden analyst degree of freedom.

Concrete edit: Report the scale of $b_{sg}$ and $v_g$ so readers can judge the floor and the shrinkage. Add a one-line sensitivity varying the 0.5 weight (0, 0.5, 1) on the normal reference only, decided before test inspection, or state that the 0.5 was fixed a priori and show it in the leave-one-out as the dominant sensitivity. Do not re-tune after seeing test AUROC.

**M2. "Average precision is 0.949" is reported prominently and is nearly meaningless at 91% prevalence.**
Offending claim: "Average precision is 0.949, but approximately 91% of evaluated sources are positive."

Why it matters: At 91% prevalence a constant classifier gets AP near 0.91, so 0.949 is barely above the no-skill floor. The paper does add the caveat, which is good, but it still reports AP to three decimals in every table and lists AP values (0.926 to 0.957) that are all within noise of the 0.91 baseline. Reporting AP at all here invites misreading.

Concrete edit: Either drop AP from the headline tables or always print the no-skill AP baseline (prevalence) immediately beside it in every table, so 0.949 vs baseline 0.91 is visible at a glance. Two decimals suffice.

**M3. The "ideal predictor" nesting argument is stated then immediately disclaimed, which is correct but the disclaimer undercuts the paper's own diagnostic rationale more than acknowledged.**
Offending claim: "For an ideal predictor that minimizes expected squared error, additional information therefore cannot increase the optimal risk. Our learners do not establish that bound... Negative benefit on normal coordinate targets motivates a learning-adequacy check."

Why it matters: The normal-development full-context MSE exceeds own-context MSE in all four folds (and three of four longer-run folds). This is not a minor adequacy footnote: it means the very quantity the method is built on, predictive benefit $b = \ell^o - \ell^f$, is systematically negative on normal data where the architecture should most easily exploit extra context. The entire diagnostic therefore operates in a regime where the model fails the one sanity check the authors themselves derive. The paper says this, but then still reports the profile as the primary score. The logical force of this is stronger than "a limitation": it means the diagnostic as implemented is measuring model inadequacy, not inter-joint predictive structure, and the authors should say so as the headline conclusion, not a caveat.

Concrete edit: Promote the full > own MSE finding to the first result, and state plainly that because the fitted predictor cannot exploit nested information even on normal data, any ranking signal from $D$ cannot be attributed to predictive benefit and is better explained as sensitivity to representation idiosyncrasy. This strengthens the honest-audit framing and is the single most important reframing available before Nov 20.

**M4. The whole-body-shift coordinate probe result of +0.765 median change (max 12.005) signals a normalization or leakage bug, not a finding.**
Offending claim (probes): coordinate whole_body_shift median_paired_change 0.765, max 12.005, versus latent whole_body_shift −0.037.

Why it matters: A whole-body shift, under per-frame pelvis centering, should be approximately removed by the normalization. A median coordinate-loss change of 0.765 with a max of 12 on a quantity whose own/full MSE is ~0.02–0.07 is a 10–200x blowup. Either the pelvis centering is not applied where this probe measures, or the probe perturbs before centering in a way that breaks the torso-scale prefix. This is flagged in prose as "potentially distorted limb-to-torso geometry," but a 12.0 value on an MSE scale of 0.05 is almost certainly a bug or a degenerate scaling, not geometry distortion. It should not be reported as a bland table row.

Concrete edit: Investigate whether whole-body shift is applied pre- or post-normalization. If the normalization is supposed to be shift-invariant, this probe should produce near-zero change and the large value is a bug to fix. If it is intentionally pre-normalization and interacts with the torso-scale prefix, say so and cap or winsorize the reported statistic, and explain why the coordinate target is shift-sensitive while the latent target is not. As written it undermines reader trust in the pipeline.

**M5. No multiplicity control on the matched-explanation comparisons, acknowledged but then conclusions still lean on them.**
Offending claim: "These are exploratory control comparisons without a multiple-comparison correction," while the discussion draws the conclusion that subtraction adds nothing from exactly those four uncorrected intervals.

Why it matters: Four paired intervals (full, own, untrained, ridge) all including zero is being read as evidence of no advantage. For a null conclusion this direction of error is conservative, so it is less dangerous than usual. But the asymmetry should be stated: uncorrected intervals that all cover zero make the null conclusion more credible, not less, so no correction is needed for that direction. The paper should say this explicitly rather than leaving "no correction" as an unresolved weakness.

Concrete edit: Add one sentence: because every control interval covers zero, the absence of correction does not threaten the null conclusion; correction would only matter if the paper were claiming a positive effect from one of these contrasts, which it is not.

**M6. The reference-free contrast dropping to 0.398 at 4096 updates is a below-chance result that is reported without interpretation.**
Offending claim (table): reference-free contrast 0.527 (512) and 0.398 (4096).

Why it matters: 0.398 is meaningfully below 0.5, which under source-level scoring with ~23 normals is a signal that the full-minus-own latent contrast ranks abnormals as more normal-like with longer training. Combined with the random encoder jumping to 0.714 at 4096, the longer-run results are internally incoherent as a method and actually strengthen the negative conclusion. The paper notes the random encoder but does not comment on the 0.398.

Concrete edit: Note the 0.398 and frame the 4096 column collectively as evidence against the method: longer head training raises the random-encoder control more than the profile and sends the reference-free contrast below chance, so additional training sharpens the null rather than rescuing the hypothesis.

### LOW severity

**L1.** Abstract says "251 source videos, including 23 normal sources." Elsewhere the fold description implies the evaluation is over 251 sources but the paired/category analyses use 81 or 193 sources. The abstract should name the primary evaluation N once and consistently. Minor, but a reviewer will check.

**L2.** "approximately 91% of evaluated sources are positive" — give the exact figure. 228/251 = 90.8%. Use 90.8% or "about 91%" consistently; the tables let a reader compute it exactly, so round-tripping should match.

**L3.** The abstract sentence "Longer head training does not establish a learned-encoder advantage" is true but buries that longer training makes the random control better than the method. Consider making the abstract state the stronger honest fact.

**L4.** Figure captions and the SVG references (context_real_comparison.svg, context_training_curves.svg) are fine, but the manuscript's Table 1 and the two supporting docs duplicate the same nine-row table three times with slightly different row labels ("Untrained latent profile" vs "Random encoder, fitted heads"). Unify terminology; a reviewer cross-checking will trip on the renamed row.

**L5.** "effective ranks range from 5.19 to 8.39" on a width-64 encoder with 16-joint tokens: effective rank under 9 on a 64-dim embedding is low and arguably itself an adequacy concern (the representation lives in a ~6–8 dimensional subspace). The paper treats this as merely "detects nonconstant representations." It is weak evidence of representational richness and could be noted as a further adequacy caveat.

**L6.** Coordinate latent values in the normal-validation table (own_latent 1.433, full_latent 1.208 etc.) are KL values whose scale is not anchored; a reader cannot judge whether a 0.2 difference is large. Add the uniform-baseline KL or the center-only KL as a reference point.

---

## Scores (1 to 5)

**Novelty: 2.5.** The individual ingredients (S-JEPA, normal-only gait density, conditional keypoint prediction, per-joint surprise) all exist, and the feasibility doc candidly establishes that SeeKer and STG-NF already occupy the conditional-prediction and normal-only-density spaces. The genuinely new element is the matched own/patch/full context contrast with a two-sided regional normal-reference profile and the contextual-teacher falsification control. That is a modest, well-scoped novelty. It is not a new detector. Scored as a novel audit protocol rather than a novel method.

**Rigor: 4.** This is the paper's strength. The controls are thorough (random encoder, ridge, constituent calibrated distances, matched budgets, leave-one-reference-out, observation-only features, supervised comparator), the estimand is carefully bounded, the pre-specification discipline around not promoting the 4096 run is exemplary, and the self-derived adequacy check (full > own MSE) is honestly reported. It loses a point for the hidden shrinkage hyperparameter (M1), the un-investigated probe blowup (M4), and the false-precision interval (H3), which are rigor lapses even in an otherwise careful paper.

**Evidence strength: 2.** The results are inconclusive by the authors' own correct admission. All normal sources overlap prior development, there are 23 normals total, the method does not beat its strongest controls, no clinical expert reviewed clips, and no published detector was reproduced. For an honest audit this low evidence strength is expected and is not a flaw in the work; it is a true description of what the data can support. The score reflects the state of evidence, not the authors' honesty.

**Clarity: 3.5.** The prose is precise and the claim boundaries are unusually well drawn. It loses points because the sheer density of caveats makes the takeaway hard to locate, the three near-duplicate result tables with renamed rows invite confusion, and the headline contrast (0.641 vs 0.541) is not the contrast the paper actually believes is most informative (H2).

**Workshop fit: 4.5.** ASTAD (anomaly / spatiotemporal / assistive-technology-for-disability framing, gait) is an excellent fit for a careful negative audit of a biologically motivated gait-anomaly diagnostic. Workshops are the right venue for rigorous inconclusive results and reproducible failure analyses. This is close to an ideal workshop contribution in spirit.

**Technical submission readiness: 3.** The real-data runs are complete, all twelve notebooks are run, synthetic material is segregated, and reproducibility artifacts are hashed. It is submittable. It is not polished: the duplicated tables, the unrounded interval, the un-investigated probe anomaly, and the headline framing all need a pass. None require new primary analysis.

---

## Honest audit vs strong new detector paper

These are different papers and the manuscript correctly chooses the first. A strong new detector paper would need: an independent, never-before-touched normal cohort; a demonstrated advantage over at least one reproduced published baseline (SeeKer or STG-NF) on matched observation support; the full > own MSE adequacy failure resolved so that predictive benefit is actually positive on normal data; and ideally clinical label validation. The feasibility doc is clear-eyed that none of these are achievable by Nov 20. Trying to dress this up as a detector paper would require claims the data cannot support and should not be attempted. The audit framing is the correct and the only defensible choice.

## Is the central message coherent and useful despite inconclusive results?

Yes, with one reframing. The coherent and useful message is: a plausible, biologically motivated predictive-context diagnostic, implemented carefully, fails to exploit nested context even on normal data and shows no advantage over simple reference-calibrated error or a random encoder; therefore any apparent ranking ability cannot yet be attributed to learned inter-joint structure, and specific follow-ups (synthetic ground truth with a nested architecture, pre-registration, a reproduced baseline, blinded annotation) are prerequisites before a biological reading. That is a genuine contribution: it saves the next group the cost of this dead end and specifies the gate. The reframing needed is M2/M3: lead with the adequacy failure and the null-against-strongest-control, not with the favorable raw-error contrast. The current ordering makes a strong honest paper read like a weak hedged positive one.

## How to improve without changing the primary analysis after seeing test outcomes

The primary analysis and its numbers stay fixed. Everything below is reframing, disclosure, or a pre-test-motivated check:
- Reorder Results to lead with the null and the adequacy failure (M2, M3). Reframing only.
- Fix reporting precision (H3, H4, M2, L2). Rounding only.
- Add the fold-level normal-source accounting and overlap disclosure (H1). Reporting of existing design.
- Disclose the shrinkage weight and floor scale; the only new computation is a reference-only sensitivity on the 0.5 weight, which touches no test labels (M1).
- Investigate the whole-body-shift probe (M4). This is debugging a development-set diagnostic, not the primary analysis, so it is safe.
- Add the multiplicity-direction sentence (M5) and the 0.398 interpretation (M6). Prose only.

---

## Prioritized revisions feasible before November 20

1. **Reframe the headline (H2, M3): lead with the null against the strongest control and with the full > own MSE adequacy failure.** Highest impact, zero new analysis, one afternoon of rewriting.
2. **Fix all numeric precision and the 16-digit interval (H3, H4, M2, L2).** One hour.
3. **Add per-fold normal-source accounting and cross-fold reference-overlap disclosure (H1).** Half a day; the information already exists in the run artifacts.
4. **Investigate and resolve or caveat the whole-body-shift coordinate probe blowup (M4).** Half a day of debugging on the development set; if it is a normalization bug, fixing it does not touch primary test scores.
5. **Disclose the shrinkage weight and variance floor scale, add a reference-only 0.5-weight sensitivity (M1).** Half a day, uses only reference normals, no test inspection.
6. **Unify the three duplicate tables and row labels; move the category table and AP to an appendix with baselines shown (L4, M2, L1).** One to two hours.
7. **Add the two interpretive sentences: multiplicity direction (M5) and the 0.398 / 4096 coherence note (M6).** Fifteen minutes.

Items 1, 2, 6, 7 are pure writing and are the minimum for a defensible submission. Items 3, 4, 5 strengthen it materially and are feasible in the remaining window. None alters the frozen primary comparison.

---

One meta-note for the independent check that will archive this: I have not verified any number against source artifacts (no tools used, as instructed). The arithmetic reconciliations in H1 and the AP-baseline claim in M2 are computed from the figures printed in the manuscript and should be re-derived against the run directory before adoption. The whole-body-shift concern (M4) is an inference from scale mismatch, not a confirmed bug; treat it as a flag to investigate, not a diagnosis.

I did not save anything to memory, this is a one-off review with no durable user preference to record. Say the word if you want me to note anything for future sessions.