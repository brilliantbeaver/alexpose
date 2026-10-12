# Bounded review of the proposed next study and source note

Reviewed 28 September 2026 against `manuscript.tex`, the primary-source checks in [future-and-figure-audit.md](../notes/future-and-figure-audit.md), and the previous independent novelty audit. This reviewer did not write the manuscript. Scope is limited to the three proposed-study paragraphs and the source/scope note; layout and the completed-study results are outside this review.

## Verdict

The proposal is coherent, understandable, and appropriately conditional. It identifies a specific diagnostic rather than treating a collection of tools as the contribution. The observation-only null and independently measured movement contrast jointly defeat the always-zero shortcut. The direct-supervision comparison, whole-pair selection, separate calibration people, and warning about unsuccessful local search are valuable. The lab-specific value is prospective and does not imply clinical validity.

Small substantive clarifications are needed before finalizing. They can be made with short replacements rather than adding a new research agenda.

## Material recommendations

1. **Name the future outcome and the independent unit explicitly.** The experiment currently says “pair two walking trials” and later “each leg's reference-measured change.” Readers may carry forward the preceding definition of projected 2D knee excursion. State that there are two trials **per participant**, with **synchronized** video/reference recording, and that the proposed outcome is an agreed **anatomical 3D knee-excursion change**. This is a new measurement convention, not automatic clinical validation of the v08 scalar.

2. **Ensure the experiment includes resolvable true changes.** Two walking trials can have essentially identical reference measurements. The proposal becomes uninformative about erased motion if all between-trial contrasts are below reference repeatability. State that trial pairs must include **reference-resolvable changes**, selected by a prespecified protocol. Instructions to walk differently are insufficient to establish the magnitude or sign of a change. The same-trial digital-mask comparisons already provide exact observation-only nulls. These two types of evidence should remain distinct.

3. **Include a calibrated measurement-level uncertainty baseline.** The opening hypothesis compares with “uncertainty rules,” but the operational list contains only visibility and ensemble disagreement. A simple calibrated interval for the actual per-leg change is a stronger, highly relevant comparator, particularly because [Pace et al.](https://arxiv.org/abs/2603.26844) already study uncertainty-aware biomechanics and selective retention. Add calibrated change-uncertainty to the short baseline list; merely outperforming uncalibrated ensemble disagreement would leave a substantial alternative explanation.

4. **Keep repeated pairs and observation variants together in analysis.** Person-disjoint fitting/calibration/test sets are correct, but they do not by themselves make multiple pairs from a test participant independent. A brief statement that uncertainty is estimated **by participant, retaining all associated trial pairs and masks**, would complete the protocol. If an individual contributes many variants, do not let that person dominate the aggregate. This matters especially for the same-trial/null and between-trial/changed copies, which share all their underlying measurements.

## Clarity and qualification

- **Clarify what is fixed.** “Keep camera and body dimensions fixed initially” should mean known calibration and **person-specific dimensions held constant within a pair**. As written, it can sound like every participant has the same body or that camera nuisance variation is irrelevant. The simplest pilot can hold camera calibration fixed as well; later cross-session tests need different treatment.

- **Define retention and audit hard cases.** “Matched retention of whole trial pairs” is technically sound, but a short gloss such as “the same proportion of comparisons reported” makes it accessible to the intended HCI audience. Fix the evaluation retention levels before final testing and separately report how often difficult/atypical pairs are declined. At the same overall retention, one rule can still exclude a clinically interesting subgroup. This qualification belongs in working notes if it does not fit the brief; the manuscript should at least make the denominator clear.

- **Keep search claims one-sided.** The existing local-search limitation is correct and should survive editing. Found alternatives demonstrate ambiguity under the assumed observation/anatomical constraints; failing to find one cannot certify a correct estimate. The notes should retain independent feasibility checks, reference-exclusion audits, and optimizer sensitivity tests. A formal confidence guarantee should not be inferred from a local search.

- **Closest precedent could be better targeted.** OpenCap Monocular and SynthGait-19K establish that reconstruction and JEPA gait estimation already exist, but they are less direct precedents for the proposed uncertainty diagnostic than uncertainty-aware biomechanics. A short reference to Pace or Cotton–Sinz would make the novelty boundary stronger. [Stenum et al.](https://doi.org/10.1371/journal.pdig.0000467) also already measure within-person gait changes against references. The current text does not assert “first,” so this is a qualification improvement rather than an identified false novelty claim. The narrow contribution remains an empirical hypothesis about measurement-specific alternatives under crossed visibility and motion changes.

## Source/scope note

The note appropriately identifies v08 as the authority, separates the training-subset addition from its retained diagnostics, distinguishes the earlier report's stylistic role, dates the literature check, and acknowledges that proposed experiments have not run. Its novelty caveat is accurate and sufficient. No source-scope blocker was found within this review's remit.

The sentence about the missing completed development motion count is transparent, but this reviewer did not independently re-audit that count; the evidence reviewer should own its verification. Likewise, the training-subset addition should retain a concrete artifact pointer in the packaged evidence notes. A working-note claim of a completed literature search must not be read as verification of access to synchronized identifiable video or permission to reuse a cohort.

## Minimum acceptance check

After revision, a reader should be able to identify: the participant as the independent unit; the new 3D per-leg outcome; observation-only nulls plus reference-resolvable true changes; calibrated uncertainty as a real baseline; matched retention's denominator; and the one-sided interpretation of a successful versus failed ambiguity search. No new experiment or larger system is needed to resolve these editorial findings.

## Final revision check

Re-read the revised proposed-study section on 28 September 2026. The material findings above are resolved: the text specifies synchronized references, two trials per participant, anatomical 3D knee excursion, reference-resolvable contrasts, within-pair calibration/body dimensions, calibrated change uncertainty, person-level uncertainty, and the proportion of complete trial pairs reported. It also audits participant declines and cites the relevant uncertainty-selection precedent. The local-search and atypical-prior limitations remain explicit.

**No material remaining issue in the proposed-study section.** The full implementation must retain participant clusters when forming intervals and prespecify the tolerances/selection rules, as already required by the working notes. Those details do not require expanding this compact brief. The proposed experiment is informative about its bounded diagnostic hypothesis; it still does not establish natural-occlusion transfer, clinical benefit, or a universal novelty claim.
