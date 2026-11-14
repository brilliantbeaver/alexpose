# Review of the supplied draft

Reviewed 2026-10-08: `../wacv2007-v1.pdf`, 20 PDF pages, with references beginning on page 16. The first page was rendered and inspected; the complete text was extracted to `original-draft-extracted.txt`. This report concerns the original document and does not endorse its cached metrics. Notebook/artifact reconciliation is performed separately for the revisions.

The draft asks a suitable question for Vision4Neuro: whether predictive skeleton learning adds useful information beyond simple pose summaries when evaluation separates web-video sources. Its most useful features are explicit source grouping, baseline comparisons, recognition of label uncertainty, and disclosure of implementation defects. The paper needs factual reconstruction before stylistic refinement because several method descriptions conflict with each other and a repeated headline number does not agree with the currently verified prediction artifact.

## Major findings and required changes

1. **Headline result provenance.** The draft repeatedly reports S-JEPA source-weighted macro-F1 0.468. The parallel artifact audit reports 0.456624 from the current verified predictions, with mean pose 0.451893. Every table, abstract number, confusion matrix and paired interval must be regenerated from one named prediction set. The supplied run hashes do not by themselves prove which predictions generated a table.
2. **Evaluation unit.** Forty-one YouTube video IDs are not forty-one verified participants. Preserve “sources” everywhere; replace “subjects,” “healthy controls” and diagnostic wording unless supported by independent metadata. Explain the target of source weighting: every source contributes equal total confusion-count mass before macro-F1 is calculated. It is neither source-level majority voting nor an average of source F1 values.
3. **Readout contradiction (§4.4).** The section describes a hypothetical linear classifier or MLP and binary controls/MS classification, although the rest of the manuscript reports three collection labels and a logistic-regression recipe. Replace it with the actual frozen feature readout and aggregation procedure from code.
4. **Mask contradiction (§5.1).** The baseline section says “16-landmark subset,” while the method describes 12 landmarks that bias regional sampling and all 33 input landmarks. State which quantity is an input joint count, a sampling-bias set and a realized masked-token count. Remove the literal editing marker “Rewrite:” in §4.1.
5. **Architecture underspecification (§4.2).** The prose duplicates preprocessing and ends with an incomplete sentence about the predictor. Specify dimensions, temporal token grouping, transformer depth/heads, predictor and pooled readout as implemented. Avoid claiming that representation learning depends “entirely” on kinematics when visibility is an input channel.
6. **Bootstrap inconsistency (§6.3).** Identical intervals [−0.121, 0.231] are attached to two different paired differences. This may be a copying error, but cannot be repaired by guessing. Recompute paired intervals from the verified out-of-fold predictions and give the exact conditional interpretation. Fold SD is descriptive and is not an independent-fold standard error.
7. **Selection and training provenance.** Separate the five-fold cached run from the standalone fold-0 training-only visualization. Report the actual selected stages; do not combine diagnostics from one run with predictions from another without labelling their provenance. An 800-step run followed by a 400-step restart is not uninterrupted 1,200-step training.
8. **Defective baselines and pretext task.** The duplicated ankle feature weakens the intended engineered baseline; reflection/mask inconsistency changes the masked-prediction task. Keep historical results explicitly identified, quantify only what can be audited and do not transfer a fixed implementation's name onto unretrained old predictions. Avoid using either defect as an untested explanation for the performance difference.
9. **Clinical and temporal interpretation.** A modest pooled F1 difference with a wide conditional interval does not establish superiority, equivalence, lack of useful temporal information, diagnostic accuracy or clinical readiness. Any inference about asymmetry, gait speed, disease biology or camera confounding requires direct tests absent here.
10. **Provenance and ethics.** The original draft does not provide clinical label adjudication, verified identity, consent, institutional approval/waiver or a data-use/redistribution determination. Record those unknowns candidly; public web access is insufficient evidence for any of them.

## Literature corrections

- §2.2 cites [41,42] for the original S-JEPA mask although those papers describe I-JEPA and V-JEPA. Use Abdelfattah and Alahi [57]. Their motion-aware sampling increases probability with motion; it is not deterministic top-motion selection.
- §2.2 calls Yan et al.'s graph-based SkeletonMAE [34] transformer-based. Correct the architecture attribution.
- §6.1 cites the spasticity study [4] for effective rank. The relevant method reference is Roy and Vetterli (EUSIPCO 2007), or define the statistic without that unrelated citation.
- §7.2 calls [35,36] “major JEPA studies”; they are DINO and MAE. Cite the appropriate feature-prediction papers and avoid claiming they explain this run's optimization behaviour.
- Creagh et al. [15] concerns smartphone ambulatory monitoring; it should not be offered as direct validation of camera pose estimation. Statements about all cited pose methods being equally sensitive or accurate require method-specific evidence.
- “Closest related work” is unnecessary without a defensible exhaustive comparison. State how each selected prior study differs in task, data or evaluation.

The reference audit separately checks all 57 original entries, including entries removed from the revised bibliography. A real reference can still be misapplied; these are separate audit dimensions.

## Organization and presentation

The Google Docs single-column document is not in the required WACV layout and has 15 content pages. Rebuild in the official 2027 kit before assessing the eight-page limit. Consolidate repeated limitations rather than restating the same negative finding across introduction, results, five discussion subsections and conclusion. A useful order is: behavioural-vision question, relevant prior evidence, data and preprocessing, model and comparison protocol, verified results, interpretation and limitations. Use one main performance table and a source-weighted confusion/paired-comparison figure; place exhaustive run diagnostics and artifact hashes in supplementary evidence.

## Baseline editorial score

This provisional assessment uses the six-category rubric in `venue-and-literature.md`. It grades the supplied draft, before notebook reconciliation, and is separate from the seven new revision scores.

| Category | Rating /5 | Weighted points |
|---|---:|---:|
| Relevance and contribution | 3.5 | 10.5/15 |
| Empirical correctness and uncertainty | 1.5 | 7.5/25 |
| Evaluation and generalization | 3.0 | 12.0/20 |
| Reproducibility | 2.5 | 7.5/15 |
| Clinical interpretation and ethics | 2.0 | 4.0/10 |
| Literature, clarity and format | 1.5 | 4.5/15 |
| **Total** | | **46.0/100** |

The largest gains must come from resolving numerical provenance and contradictory methods, followed by clearer evaluation scope and verified references. Stronger prose cannot supply missing clinical ground truth or missing ablations.
