# Review: "From Prediction Gain to Anomaly Detection: A Controlled Cross-Leg Gait Study"

## Summary of what the paper claims

The paper is a bounded negative-result/measurement study. It tests whether an opposite-leg prediction correction, over and above an own-leg baseline and an own-capacity control, adds (a) prediction information on normal recordings, and (b) detection value for unusual gait via a frozen equal-weight score S = (M+R)/2. The headline findings are: normal prediction improves (especially on the focused E_leg task and in the external lab replication), but S underperforms the motion baseline M on detection (GAVD focal 0.576 vs 0.694; external 0.892 vs 0.958), and a timing diagnostic fails prediction adequacy on its sparse matched inputs despite passing a coverage floor. I assess the paper on its own terms as a measurement study, not a detection-improvement claim.

The overall framing is unusually disciplined, and the manuscript has clearly internalized much of the existing independent audit. Most of my findings below are about residual overreach in specific passages, places where the frozen analysis is sound but the surrounding prose still claims more than the numbers support, and a few genuine correctness concerns.

---

## Confirmed contradictions and correctness issues

### C1. The abstract's first sentence overstates the direction of the headline result relative to the broad endpoint and the R-alone value
Abstract: "the frozen combination ... reduces focal AUROC from 0.694 to 0.576." This is correct arithmetic, but the abstract never states that R alone is 0.445 (focal) / 0.560 (broad), nor that M+constituent-error (0.684) and M+own-capacity sham (0.687) are both close to M. The negative result is therefore not merely "the fusion is worse than M" — it is "adding R to M in equal weight actively degrades M roughly to R's level, and the degradation is specific to R and not to the fusion mechanism" (since the sham and constituent-error fusions do not degrade M). The paper has these numbers in Table 2 but the abstract's causal story ("bilateral prediction benefit reduces ... AUROC") is weaker than what the controls actually show. **Correction:** state in the abstract that the own-capacity sham and constituent-error fusions do *not* degrade M, so the loss is attributable to R's content, not to fusing a second channel. This strengthens the result and preempts the obvious "any second channel would hurt" objection.

### C2. Results text reports "M + baseline-error distance = 0.700" but Table 2 (manuscript) omits it
The RESULTS file lists "M plus baseline-error distance | 0.700 | 0.692" as a comparator that *exceeds* M (0.694). The manuscript Table \ref{tab:detection} drops this row entirely. This is a confirmed discrepancy between the frozen evidence and the manuscript table. The omission matters: a baseline-error fusion that slightly *improves* over M (0.700 > 0.694) while the benefit-departure fusion *degrades* it (0.576) is directly relevant to the claim that "constituent-error fusions" behave differently from R. **Correction:** either restore the 0.700 row to Table 2 or state explicitly in the caption why it is excluded. As written, the manuscript's Table 2 is not a faithful subset of the frozen results, and a reader cross-checking the RESULTS file will find a missing favorable comparator.

### C3. "Capacity control" wording persists in the abstract and introduction despite the known rank gap
Abstract: "an additional own-leg capacity control." Introduction: "an own-capacity control." The frozen evidence shows the own and bilateral residuals have empirical ranks 111 vs 141 (GAVD) and 44 vs 62 (external) at equal nominal width (154 / 72 columns). The existing audit explicitly flagged this and asked for "matched nominal input width and regularization grid, with an own-capacity control," not "capacity control." The manuscript body does say "Equal widths do not equalize effective rank or inductive bias" and reports the ranks — good. But the *abstract* and *introduction* still use the bare phrase "capacity control," which a skimming reviewer will read as "effective-capacity matched." **Correction:** in the abstract and introduction, use "an additional own-leg width-and-regularization control (not an effective-capacity match)" or similar. The rank gap (111→141, 44→62) is not a rounding difference; the control under-matches the bilateral branch's effective capacity by ~27% (GAVD) and ~29% (external), and the bilateral branch has *more* effective capacity, so a prediction gain cannot be cleanly attributed to contralateral information rather than to the richer basis. This is a real limitation that the current abstract phrasing papers over.

### C4. The timing result's two offsets are described asymmetrically in a way the numbers do not support
Results: "U's negative offset has a positive raw effect of 0.000393, interval [0.000177, 0.000618]; the positive offset gives 0.000064, interval [−0.000232, 0.000368]." The negative-offset interval excludes zero; the positive-offset interval includes zero. The manuscript then says "These responses cannot establish a reliable coordination measurement." That conclusion is correct and well-supported by the adequacy failure. But the paper presents the −4 effect as if it is a near-null nuisance alongside the adequacy failure, when in fact it is a *nominally significant* conditional effect (interval excludes zero) that is being correctly discounted *only* because the predictor is inadequate on the matched support. **This is not a contradiction but it is under-explained:** a skeptical reader will ask why a conditional interval excluding zero does not count as a positive finding. The answer — that the frozen predictor cannot predict the aligned target on these ~1.75-token inputs, so any shifted-minus-aligned difference is a difference between two inadequate predictions — is the whole point of the diagnostic and deserves one explicit sentence, not just the current "a positive response alone is therefore insufficient." Make explicit that the −4 interval excludes zero and that this is exactly the case the adequacy gate is designed to catch.

---

## Nominal vs effective capacity (expanded)

Beyond C3: the paper reports feature ranks but never reports what fraction of the *prediction gain* survives when the own control is given a basis of matched effective rank. It cannot, because that analysis was (correctly) not run on the frozen protocol. The honest position — which the manuscript mostly takes — is "we cannot rule out effective capacity as the source of the prediction gain." But §"Independent participant replication" says "Equal residual width still does not equalize rank: the 72-feature own and bilateral bases have ranks 44 and 62," then immediately reports the ordered MSEs (309.838 → 206.262 → 201.038 → 190.584) as if the ordering establishes bilateral value. The ordering f+h (201.038) > f+g (190.584) is consistent with *either* contralateral information *or* the 44→62 rank advantage. **Correction:** in both the external replication and §"Controlled prediction," state plainly that the prediction gain (normal-completion improvement) is *confounded with* effective capacity and is therefore not itself part of the paper's defensible claims — the defensible claim is purely the *detection* comparison, where S and M are evaluated on the same endpoint regardless of the branch rank. This actually protects the paper: the detection result does not depend on the capacity question at all, so the capacity confound is irrelevant to the headline, and the prediction-gain narrative should be demoted to "suggestive, capacity-confounded."

---

## Calibration and source separation

### Calibration (M vs R)
The manuscript states "Dimension normalization does not make M and R equally calibrated probabilities, and R omits bilateral covariance" — good, this matches the audit. But the manuscript does not report the concrete scale mismatch the audit documented (external fold 1: median M ≈ 1.003 vs median R ≈ 0.154; GAVD U fold 1: median M ≈ 0.620 vs median R ≈ 5.804). These medians show the two channels differ in central scale by roughly an order of magnitude and *in opposite directions across datasets*. Equal-weight averaging of two channels whose normal medians differ by ~6–7× (GAVD) is a strong prior that the fusion will track whichever channel has the larger variance, and the direction of the mismatch flips between datasets. **Correction:** report at least one normal-median pair per dataset in the limitations, and state that the equal-weight mixture is not scale-balanced on the normal distribution. This reframes the negative result more precisely: S underperforms M partly because an uncalibrated equal weight lets the noisier/larger-scale R channel dominate, not necessarily because R's *ranking content* is harmful. The R-alone AUROCs (0.445 focal GAVD, 0.694 external) already show R's content is weakly-to-moderately informative, so "equal-weight fusion of an uncalibrated channel hurts" is the correct and narrower conclusion. The current text gestures at this ("may reflect score calibration") but does not commit to it with numbers.

### R's covariance omission
"R omits bilateral covariance" is stated. The audit gives the normal left–right benefit correlations (≈0.82, 0.77, 0.57, 0.54 external) and the (2,2) vs (2,−2) degeneracy example. The manuscript should keep at least the correlation range, because a diagonal distance over components correlated at 0.8 substantially mis-weights the bilateral pattern — this is a concrete reason R is miscalibrated, not a generic caveat. This is a limitation, not a contradiction, but it is specific enough to state with the numbers.

### Source separation
The source-separation story is the paper's strongest component and I found no contradiction. The role discipline (test/reference/fit normals disjoint; references never select anything; whole recordings kept in one role; grouped bootstrap) is correct and the external audit independently verified fold disjointness. Two residual concerns, both already disclosed:
- **Upload IDs ≠ participants.** Correctly and repeatedly flagged. The "unknown repeated people remain unresolved" caveat is honest. The confirmed duplicate pair already failing pose quality is a nice detail. I accept this as a stated limitation, not a defect.
- **All 28 normals were seen in prior exploration** (Introduction / protocol). This is the single most important threat to the GAVD arm and the paper is admirably explicit ("We ... do not call the analysis confirmatory"; intervals "omit ... adaptive redesign"). But the negative headline is *partly protected* by this honesty: a prior-exploration normal set biases toward *favorable* normal behavior, which would if anything inflate M and the prediction gates, not manufacture the negative S−M result. **Suggestion (not a required fix):** state explicitly that prior exposure of the normal set is a threat to the *positive* prediction-adequacy claims and the M baseline, but does not plausibly generate the *negative* detection result — so the headline is relatively robust to this threat while the supporting prediction-gain narrative is not. This directional argument strengthens the paper and costs nothing.

---

## Timing semantics

The timing diagnostic is conceptually the paper's most interesting contribution and its handling is correct: coverage floor passes, adequacy fails on matched support (24/24 outer, 72/72 internal), ~1.75 editable tokens. The separation of "raw offset response" from "prediction adequacy on exactly matched inputs" is exactly the right instrument and the conclusion ("a positive response alone is insufficient") is sound.

Two corrections:
1. Per C4, the −4 interval excludes zero and this should be stated, with the explicit argument for why it is still uninterpretable.
2. "Shifting also changes instantaneous geometry and finite-interval distributions, so even a successful effect would not isolate neural timing." This is the correct and important caveat — the shift is not a clean timing manipulation because it also perturbs geometry. Good. But then the diagnostic cannot, even in principle, measure "coordination timing" as the Conclusion implies ("A defensible relational detector must establish ... useful prediction on its exact support"). The manuscript should be clear that the timing diagnostic is an *adequacy stress test* that failed, not a *timing measurement* that returned null — the paper mostly says this but the figure caption "separates a raw offset response from prediction adequacy" is the right framing and should govern the Conclusion's wording too.

---

## Novelty and prediction adequacy

- **Novelty.** The paper correctly does *not* claim a new mask family, estimator, or dataset. The claimed contribution — a controlled separation of prediction adequacy, observation support, and detection value, with an own-capacity (width) control and exact input-access checks — is a methodological/measurement contribution. As a negative-result study this is defensible; the related-work section appropriately disclaims novelty for mask structure (VideoMAE/SLiM), conditional-surprise scoring (SeeKer/STG-NF), and representation learning (S-JEPA/MAMP). I have no novelty-inflation concern. The one place to tighten: the Introduction's "exact input-access checks" is a genuine contribution (the hidden-input invariance and zero-gate tests are real and verified), but it is buried; it is more novel than the mask-budget matching, which prior work approximates.

- **Prediction adequacy as a gate.** The internal gate is correctly described as "an operational selection rule, not a significance test" and the manuscript repeatedly refuses to call a passed gate a generalization guarantee ("an internal selection gate alone does not establish uniform-task generalization"). This is the right discipline. The disclosed weighting difference between internal selection (source/joint-weighted) and final scoring (mask-collapsed within physical targets) is handled honestly and matches the audit's medium-priority finding.

- **Missing direct-interpolation benchmark.** The audit flagged that the ridge baseline consumes interpolated features but no direct-interpolation *predictor* error is reported externally. The manuscript now says "Direct interpolation is weaker than shared ridge in every aggregate of internal held-out GAVD normals" and cites a saved comparison — this addresses the GAVD side. The external side still lacks it. **Correction:** either report the external direct-interpolation comparison or state its absence explicitly in the external limitations, matching what was done for GAVD. Do not let the GAVD sentence imply the external arm has the same benchmark.

---

## Inference and clinical claims

The paper is disciplined here. "Folder labels are not independently revalidated diagnoses," "video predictability [does not] recover neural synergies," "we do not infer a neurological cause," "source labels retained without clinical reinterpretation," and the refusal to equate prediction error with severity are all appropriate and repeated. The physiological motivation section correctly frames stroke/PD/CP/myopathy findings as *motivation for preserving amplitude/posture/timing channels*, not as mechanisms the method recovers, and explicitly notes mechanics/task constraints as alternative explanations (Kutch). I found no clinical overreach.

Two items:
1. The external arm covers only healthy + stroke, yet the GAVD arm spans four conditions. The sentence "The findings reject a detection advantage for this fixed score" (abstract) is correct, but "across two measurement systems" (Conclusion) slightly oversells the convergence: the two systems agree on the *direction* of S−M for a largely different label mix (4 conditions vs stroke-only). **Correction:** say "across two measurement systems and partly non-overlapping condition sets," so the agreement is not read as replication on the same clinical question.
2. The stroke-MSE-lower-than-healthy observation from the audit (fold 0: stroke base MSE ≈131.9 < healthy ≈204.3) directly supports "larger prediction error is not necessarily more abnormal." The manuscript states this principle but does not cite the external datum. Including it would concretely justify the design decision to *not* use raw prediction error as the score.

---

## Reasonable limitations (correctly scoped — no action required)

These are stated and appropriately bounded; I list them to distinguish them from the corrections above: five-reference benefit-scale uncertainty (with saved LORO and variance-floor sensitivities); boundary audit finding out-of-image high-confidence points in *more* normal than abnormal recordings (0.43% vs 0.24% — honestly reported as cautioning against artifact-blaming abnormals); single-view strata lacking normal support; STG-NF as an imputed non-faithful adapter (0.376 — correctly not treated as a benchmark); small JEPA probes not establishing representation limits; unmatched walking speed in the external arm; one-trial-per-participant (repeatability untested). All correctly framed as limitations, not defects.

---

## Untested suggestions (explicitly out of scope — do not run to "fix" this paper)

A rank-matched own control, a calibrated/covariance-aware fusion, an explicit anti-alias audit of the 15 Hz resampling, a repeat-trial external analysis, and a frozen non-explored normal cohort would each strengthen the program. Per the audit and the paper's own stance, every one must carry a new scientific identity and an exploratory label; none may revise the frozen scores or recipes post hoc. The manuscript's Conclusion and the protocol both say this. I flag only that the manuscript should *not* add any of these as "robustness checks" to the current identity, since doing so would contaminate the frozen negative result — keep them in a clearly separated future-work paragraph.

---

## Priority-ordered corrections preserving the frozen analysis

1. **C2 (table fidelity):** restore or explain the missing "M + baseline-error = 0.700" row; the manuscript table must be a faithful subset of frozen results.
2. **C3 (capacity wording):** replace "capacity control" with "width-and-regularization control" in abstract + introduction; report ranks there, not only in the body.
3. **C1 (attribution):** state in the abstract that the sham and constituent-error fusions do *not* degrade M, so the loss is specific to R's content.
4. **Calibration numbers:** report normal-median M vs R per dataset and the left–right benefit correlation range; reframe the negative result as "equal-weight fusion of an uncalibrated, covariance-blind channel," which the R-alone AUROCs support.
5. **C4 / timing:** state that the −4 interval excludes zero and give the one-sentence argument for why adequacy failure makes it uninterpretable.
6. **External interpolation benchmark:** report it or explicitly note its absence, parallel to the GAVD sentence.
7. **"Across two measurement systems":** qualify with the non-overlapping condition sets; attach the stroke<healthy MSE datum to the "error ≠ abnormality" claim.

None of these require refitting, rescoring, or altering the frozen protocol; all are prose, table, and number-disclosure corrections drawing on already-saved artifacts.