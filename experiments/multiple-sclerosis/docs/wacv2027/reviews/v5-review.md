# Independent scientific and numerical review of v5

I reviewed `wacv2007-v5.tex`, the v4 review, the notebook audit, the numerical evidence, and the current figure generators. I inspected rendered pages 2, 3, 5, 6 and 7, including both figures, the settings and results tables, and the reference list. I checked all five main result-table rows and all three paired intervals against the declared saved run and bootstrap settings. The original study artifacts and manuscript were not edited during this review.

The principal findings are now numerically correct and substantially easier for a researcher outside machine learning to follow. The paper explains what the encoder and classifier do, introduces source separation in ordinary language, and explains F1 before giving the formula. The data support the bounded comparison stated in the abstract and conclusion. I found no new scientific or numerical defect that requires changing the reported results. The remaining substantive limits arise from the retained experiment, not from missing qualifications that can be repaired through prose alone.

## Rubric score

This is the locally defined workshop-relevant rubric, not an official conference scoring system. Ratings assess the actual evidence and manuscript; they do not predict acceptance.

| Category | Weight | Rating /5 | Weighted points | Reason |
|---|---:|---:|---:|---|
| Workshop relevance and practical contribution | 15 | 4.0 | 12.0 | The markerless behavioral-vision question is relevant, and the weighting-dependent comparison is a useful empirical observation. Clinical and methodological novelty remain bounded. |
| Empirical correctness and uncertainty | 25 | 4.7 | 23.5 | The primary and corrected-RF scores, denominators, signs and paired intervals agree with the retained evidence. The uncertainty procedure and its conditioning are described accurately. |
| Evaluation design and generalization scope | 20 | 3.8 | 15.2 | Appropriate source grouping, matched test sets and simple controls are reported. One seed, uncertain identity overlap and absent causal temporal/reflection controls still limit the study. |
| Reproducibility and methodological completeness | 15 | 4.5 | 13.5 | Source ordering, methods, historical/corrected identities, hashes and runnable analysis now have an actual companion record. Exact historical neural retraining is still limited by incomplete historical code reconstruction and nondeterministic execution. |
| Clinical interpretation, provenance and ethics | 10 | 4.0 | 8.0 | The paper correctly restricts labels and conclusions and names missing records. Verified diagnoses, identities, data-use documentation and clinical validation remain unavailable. |
| Literature, clarity and WACV presentation | 15 | 4.5 | 13.5 | The abstract, explanation of learning, settings table and paired-fold figure improve accessibility. Two inherited terms and one opportunity for a simple weighting example remain. |
| **Total** | **100** | | **85.7/100** | **Scientifically defensible pilot account; remaining revisions are targeted explanation and final consistency checks.** |

The gain over v4 reflects completed uncertainty/provenance presentation and clearer explanations. It does not reflect new participants, additional neural training seeds, or stronger evidence of temporal or clinical validity.

## Disposition of the v4 criticisms

| v4 request | v5 outcome |
|---|---|
| Align fold numbering | Resolved. Figure 2 uses registry indices 0–4; straight horizontal lines connect the two methods within a fold. The large S-JEPA advantage is visibly in fold 2. |
| Correct interpolation-reporting overclaim | Resolved. The text reports padding counts and states that interpolation frequency and gap lengths are unavailable from the retained cache. It also explains that equal frame counts need not mean equal elapsed time. |
| Introduce question and score accessibly | Substantially resolved. The abstract starts with body landmarks and average pose; Section 5.2 defines precision, recall and macro-F1. One short weighting example would help further. |
| Explain specialized learning terms | Substantially resolved. Encoder, reference/target encoder, EMA, fixed weights and linear classifier are explained. The pipeline says “label classifiers.” Its caption retains one unexplained occurrence of “probe fitting.” |
| Remove editing-history material | Substantially resolved. The “original PDF” and legacy normal-only/VICReg discussion are gone from the paper. The opening Results sentence still uses the insider name “capstone notebook,” which is unnecessary for a new reader. |
| Supply reproducibility companion and source ordering | Resolved. `evidence/reproducibility.md` exists, and sorting source IDs precedes seeded splitting in the manuscript. The record distinguishes the historical primary run from corrected verification. |
| Render BlazePose identifier and clarify adaptation | Resolved. Page 7 displays `arXiv:2006.10204`; Related Work distinguishes motion-related original S-JEPA target sampling from this region-based 2D-plus-visibility adaptation without claiming an improvement. |

## Numerical and scientific checks

The main table uses the September notebook-associated run. S-JEPA's source-weighted macro-F1 is 0.456624159, clip-weighted macro-F1 0.397033957, and fold mean/SD 0.438067947/0.195980308. Mean pose's corresponding pooled values are 0.451892876 and 0.409840252. The historical RF retains its explicit feature-defect marker. Visibility and majority values match the audit. The three-decimal table entries and the abstract's rounded 0.005 difference are consistent with these quantities.

The bootstrap reports the manuscript's declared 10,000 draws and seed 20261004, not the separate 20,000-draw sensitivity analysis retained elsewhere in the audit. S-JEPA minus mean pose is 0.004731283 with interval [−0.160496452, 0.173998638]; S-JEPA minus historical RF is 0.045361249 [−0.133446835, 0.220113928]. The corrected-RF section separately reports S-JEPA minus corrected RF as 0.131440055 [−0.046114011, 0.303792121]. The positive signs and the lower corrected-RF score are mutually consistent. The source-stratified bootstrap preserves all three source-label strata, and the paper does not claim that its intervals include retraining or full population uncertainty.

Raw correct counts 10/24 Normal, 8/29 MS and 18/35 PD sum to 36 S-JEPA successes. Source-weighted recalls 0.396, 0.429 and 0.569 correspond to different denominators from those raw clip counts, as the paper and figure caption explain. Figure 2's row-normalized matrices and fold positions agree with the notebook-primary predictions. The figure does not imply additional participants or clinically verified diagnoses.

The additional RF evidence remains a verification of an existing corrected execution and a small deterministic refit, not a newly independent neural experiment. Exactly 12 RF predictions differ between historical and corrected versions. The manuscript explains that the S-JEPA hard predictions coincide while embedding values differ and the seed/split/data remain the same. This is an accurate restriction on replication claims.

The baseline controls do not isolate the causal contribution of motion. Mean pose and visibility use different feature dimensions and full-clip summaries, whereas the learned model uses fixed-window features and stage selection. The manuscript now calls them complete-procedure comparisons and leaves the temporal contribution unresolved. No random-label/permutation baseline or untrained-encoder result is presented, and there is no claim of performance above a universal one-third macro-F1 chance level. The majority rule is a training-label-frequency baseline, not a random classifier. These absences limit what can be concluded but are not hidden successful experiments or numerical omissions.

The proposed untrained and frame-order comparisons are explicitly unrun. For a future protocol, test-only shuffling would primarily measure sensitivity to distribution shift; it would not by itself establish whether temporal learning was beneficial. This distinction should guide the future experiment, but the present manuscript does not claim to have performed such a test.

## Targeted refinements for v6

1. **Finish the terminology cleanup.** In Figure 1's caption, replace “probe fitting” with “fitting the label classifier.” In the opening Results sentence, replace “the capstone notebook” with a neutral phrase such as “the notebook-defined primary experiment.” The companion record already supplies the exact notebook and run identity.

2. **Add one concrete equal-source example.** After defining the clip weight, a short sentence can say that the MS source with 13 usable clips gives each clip weight 1/13, while a source with one clip gives that clip weight one. This makes the main weighting-dependent observation understandable without asking general readers to interpret the summation immediately. The example uses existing counts and adds no experiment.

3. **Define visibility when it first appears.** Describe it as the pose estimator's estimate of whether a landmark is visible. Preserve the distinction from disease confidence, physical certainty, and a pure measure of recording quality. “Visibility value” alone assumes familiarity with pose-estimation output.

4. **Keep the future controls specific without enlarging the current claim.** If adding detail, explain that a matched order manipulation needs a predefined training/evaluation rule. Do not retrofit a test-only shuffle or a model from another run into the current results, and do not turn the absence of an observed advantage into a claim that temporal information has no value.

5. **Perform the final artifact check after editing.** Preserve the exact numerical table/intervals and the primary/corrected separation, rebuild using the version-specific bibliography, and check the rendered captions and companion figure commands. The current `lineno.sty` UTF-8 warning is a build-environment warning; I saw no damaged manuscript glyphs on inspected pages, and the log contains no overfull or undefined-reference report.

The figures are legible at the inspected rendered scale, with balanced rectangular pipeline boxes and readable matrix values. The current paper occupies seven PDF pages, with discussion/conclusion and references on page 7; it remains within the eight-page content allowance described in the venue audit. No new experiment is needed to implement the explanation fixes above. Resolving the study's scientific limitations would require new, explicitly labeled work rather than additional polishing.
