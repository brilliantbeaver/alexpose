# Independent scientific and numerical review of v9

I reviewed the complete v9 source, `evidence/v9-methods.md`, the retained notebook and numerical audits, and all seven rendered PDF pages. I checked the result tables, training-set counts, paired differences and intervals against the machine-readable audit, verified the bibliography and PDF fonts, and confirmed preservation of v8 and the six recorded evaluation artifacts. The checks are summarized in `qa/v9-independent-checks.json`. No manuscript, notebook, scientific code, model or saved prediction was changed during this review, and no new experiment was run.

**The constructive framing is supported, with no numerical or scientific-interpretation blocker found.** V9 presents an observed, useful application of trained skeleton features while retaining the evidence needed to judge its limits. The paper does not claim that training itself caused the predictive information, that sequence order explains the result, or that the features are clinically validated.

## Rubric score

The same six-category rubric and weights used for v2–v8 apply. These are editorial judgments derived from the workshop scope and WACV review priorities, not official conference scores or acceptance probabilities.

| Category | Weight | Rating /5 | Weighted points | Assessment |
|---|---:|---:|---:|---|
| Workshop relevance and practical contribution | 15 | 4.0 | 12.0 | The paper now states its feasible small-data application clearly: fixed skeleton features support collection-label prediction and compare favorably with the configured RF under equal-source scoring. The underlying contribution remains an exploratory behavioral-vision study. |
| Empirical correctness and uncertainty | 25 | 4.7 | 23.5 | Primary and corrected RF results, score weighting, counts and uncertainty agree with retained evidence. Positive claims describe the observed complete procedure without inventing a causal training benefit. |
| Evaluation design and generalization scope | 20 | 3.8 | 15.2 | Source separation, validation selection, matched test groups and simple controls are appropriate for the descriptive comparison. One seed, unresolved participant overlap and unrun training/order controls remain unchanged. |
| Reproducibility and methodological completeness | 15 | 4.5 | 13.5 | Exact formulas and resampling details remain available in an explicitly named companion, alongside the existing methods and artifact audit. End-to-end historical replay still has the previously documented limitations. |
| Clinical interpretation, provenance and ethics | 10 | 4.0 | 8.0 | Useful features are defined through collection-label prediction, with no assertion of physiological meaning or clinical readiness. Identity, label, rights and consent gaps remain visible. |
| Literature, clarity and WACV presentation | 15 | 4.6 | 13.8 | The abstract, Results and Conclusion give readers a clear constructive finding. The scoring explanation is easier to follow, and uncertainty remains available without dominating the narrative. All seven pages and the retained figures are legible. |
| **Total** | **100** | | **86.0 / 100** | **A clearer constructive account of the same bounded evidence.** |

The 0.3-point change from v8 is entirely a presentation assessment: the new framing and simpler scoring section better communicate the supported result to a general research audience. No experimental, clinical or methodological score has increased. Further empirical improvement still requires additional evidence.

## Positive claims and causal boundaries

The principal statement is defensible: features obtained by training the encoder on skeleton-prediction tasks support label prediction by a supervised linear classifier on held-out source recordings. Each encoder's training pool contains 24 sources and 45–58 clips, which agrees with all five partition rows. The model's features are fixed for classifier fitting; condition labels are absent from the encoder prediction loss, while classifier fitting and training-stage selection use labels. The manuscript preserves that distinction.

The Results and Conclusion qualify the RF advantage as **observed** and tie it to equal-recording scoring. “Highest observed macro-F1” is accurate for that primary comparison. V9 defines useful features as information supporting the supplied collection labels, and explicitly states that comparing with the same network at initialization would test what training adds. Thus “trained features” describes the actual procedure without implying that the experiment has isolated the causal value of training. The absence of a matched frame-order intervention is also stated.

The RF comparison is between complete procedures. V9 makes the differences in feature construction, classifier and stage selection explicit, so its gain is not assigned to the encoder alone. Neither the trained features nor the anatomical prior are described as measuring neurological impairment, preserving clinically specific motion, or establishing diagnostic validity. The abstract presents feature learning as a complement to handcrafted gait descriptions, while the body discloses the hand-designed masking prior.

## Numerical verification

All five main result rows, three class-count rows and five partition rows agree with the retained audit. The key primary values are:

| Procedure | Equal-source macro-F1 | Equal-clip macro-F1 | Fold mean / population SD |
|---|---:|---:|---:|
| S-JEPA | 0.456624159 | 0.397033957 | 0.438067947 / 0.195980308 |
| Mean pose | 0.451892876 | 0.409840252 | 0.427332450 / 0.132209838 |
| Historical joint-angle RF | 0.411262910 | 0.397315740 | 0.395383801 / 0.097959643 |
| Visibility | 0.317564628 | 0.318057047 | 0.284918454 / 0.104069172 |
| Majority-label procedure | 0.259398496 | 0.202125202 | 0.159393939 / 0.021985887 |

The S-JEPA source-weighted lead is 0.045361249 over historical RF and 0.004731283 over mean pose. V9 correctly says that S-JEPA and historical RF both round to 0.397 under equal-clip weighting, with RF slightly higher before rounding. It also retains the 36 versus 37 correctly classified clips and the 13-clip MS recording. The paper therefore does not convert a weighted-score advantage into a false claim about raw accuracy.

Mean pose's strong result remains substantive context: its equal-clip score is higher, and it exceeds S-JEPA in four of five source-weighted folds. S-JEPA's fold range 0.217–0.780, class counts 10/24, 8/29 and 18/35, and equal-source recalls 0.396, 0.429 and 0.569 are correct. Figure 2 retains the matching fold indices and confusion percentages.

The historical RF defect is explicit in the baseline description and table footnote. The later corrected RF is kept in a separate subsection: source/clip macro-F1 0.325184104/0.320940497, with S-JEPA source difference 0.131440055. The existing verification supports all 88 predictions for each version and the 12 changed labels. Shared data, partitions and seed are stated, so the two executions are not presented as independent replication or as comparisons with all classical gait systems.

## Uncertainty without a significance-centered narrative

The exact paired values in the new companion agree with the 10,000-draw, seed-20261004 audit:

| Comparator | S-JEPA minus comparator | Conditional 2.5th–97.5th percentile interval |
|---|---:|---|
| Historical RF | 0.045361249 | [−0.133446835, 0.220113928] |
| Mean pose | 0.004731283 | [−0.160496452, 0.173998638] |
| Corrected RF, separate verification | 0.131440055 | [−0.046114011, 0.303792121] |

The S-JEPA individual interval is [0.328714413, 0.578793142]. Its main-text rounding, and every paired main-text rounding, are correct. No value from the separate 20,000-draw diagnostic has been substituted.

Moving the score formula and detailed resampling explanation into a named companion preserves reproducibility while making the body easier to read. The paper still explains whole-source resampling, paired source draws and fixed predictions. The companion supplies label-stratified counts, replacement, repeated-source weights, seed, percentile rule, overlapping training sets and unresolved participant overlap, and excludes retraining and model-selection uncertainty. The narrative reports the observed result without claiming statistical validation, general superiority, equivalence or a universal chance level for macro-F1. It does not need repeated statements about significance to remain scientifically complete.

## Remaining limits and resolved precision edits

The existing limits remain sufficient for the current claims: one training seed, uncertain identities and diagnoses, unmeasured recording cues, an untested anatomical prior, and the within-example reflection/mask correspondence defect. V9 does not assign an invented effect size to that defect. It also preserves the unprovided consent, source-rights and ethics-review records and excludes individual clinical decisions.

No additional result or experiment is required merely to support this more constructive wording. During review, the abstract's initially vague reference to an ankle-feature correction was made explicit: “We separately report the RF evaluation after correcting a duplicated ankle-range feature.” Abstract-only readers can now see the feature defect and separate verification status. The introduction also replaces an unqualified reproducibility sentence with the narrower statement that processing records and saved predictions allow the reported comparison to be checked directly.

The endpoint-training-batch discussion was shortened to the supported stage-selection facts: the initial stage is selected in groups zero through two, and continuation in groups three and four, using classification performance on separate validation recordings. This removes an unnecessary implication that the endpoint comparison estimates the benefit of validation selection. No score, partition or result changed. The abstract's opening was also shortened, restoring a clean seven-page layout after the earlier precision edits briefly pushed three reference lines onto an eighth page.

## References, layout and preservation

The manuscript cites 13 unique references, all rendered without unresolved citations. `references-v9.bib` is byte-identical to the previously verified v8 bibliography. Retained citations continue to support method background, clinical motivation and evaluation principles, without being used to authenticate the collection labels.

All seven rendered pages were inspected. The PDF uses US-letter pages and 23 embedded font subsets, with no unembedded or Type 3 fonts. The title, abstract, four tables, two figures, mathematical notation and reference list fit without observed clipping or overlap. The existing straight-arrow pipeline and matched-fold connectors remain readable. The build reports the inherited `lineno.sty` UTF-8 warning; no corresponding damaged glyph was observed. The `DRAFT` identifier is still a preparation placeholder.

V8 source and PDF hashes match the completed v8 review, and all six historical/corrected OOF, results and provenance JSON hashes match the retained numerical audit. These checks confirm artifact preservation; they do not resolve the collection's clinical or data-use provenance. This review neither submits the paper nor claims human peer review or independent model-family corroboration.

**Disposition:** v9 is suitable as the constructive next manuscript revision. The review's precision suggestions have been addressed; no numerical, scientific-framing or layout blocker remains.
