# Independent review of draft v6

Reviewed `wacv2007-v6.tex`, the v5 review, `evidence/method-details.md`, the notebook audit, the reproducibility record, and the rendered seven-page PDF. The source figure images and full-size PDF pages 3, 4 and 6 were inspected for the user's requested straight, balanced figures and for readability of the simplified methods. This review checks the effect of the rewrite for an educated researcher outside machine learning; it does not treat easier prose as new experimental evidence. No manuscript or study artifact was changed.

The revised account preserves the essential technical distinctions while explaining the experiment more accessibly. I found no new numerical defect or substantive method contradiction introduced by moving the loss equations to the companion record. The main outcome remains a comparison of complete prediction procedures, with the contribution of temporal learning unresolved. The figures now have straight connectors, balanced rectangular panels, and readable labels without the earlier collisions.

## Rubric score

The weights are the task-specific workshop-grounded rubric, not an official WACV numerical grading scheme or a prediction of acceptance.

| Category | Weight | Rating /5 | Weighted points | Assessment |
|---|---:|---:|---:|---|
| Workshop relevance and practical contribution | 15 | 4.0 | 12.0 | The behavioral-vision question and weighting-sensitive empirical comparison remain clear and useful within the pilot's scope. |
| Empirical correctness and uncertainty | 25 | 4.7 | 23.5 | The notebook-primary results, conditional intervals, per-class errors, and separately identified RF correction remain consistent with the audited evidence. |
| Evaluation design and generalization scope | 20 | 3.8 | 15.2 | The source-separated evaluation and fixed-prediction uncertainty analysis are well explained. Rewriting has not removed the limitations of the retained design. |
| Reproducibility and methodological completeness | 15 | 4.5 | 13.5 | Simplified prose retains the masking, teacher, loss-weighting, continuation, readout, and selection details, with exact loss expressions available in the companion record. The direct pointer to that record and the submission copy's anonymity need a final cleanup. |
| Clinical interpretation, provenance and ethics | 10 | 4.0 | 8.0 | Collection labels remain distinct from diagnosis, identity uncertainty is explicit, and missing data-use records are not replaced with invented assurances. |
| Literature, clarity and WACV presentation | 15 | 4.5 | 13.5 | The weighting example, visibility definition, plain-language feature prediction, and straight matched-fold display improve comprehension. A few terms and repeated passages remain worth tightening. |
| **Total** | **100** | | **85.7 / 100** | **A defensible, readable pilot paper; final revisions should be narrow and concrete.** |

The score remains at v5 rather than increasing mechanically. The new explanation helps readers, but the evidence and generalization limits are unchanged. Remaining editorial issues fit within the same scoring band.

## What v6 resolved from the v5 review

- Figure 1 now says “label-classifier fitting,” and the Results opening uses “primary notebook-defined experiment.” The reader no longer needs to know the internal “probe” or “capstone” terminology to follow those passages.
- The source-weighting example is concrete and correct: one source's thirteen MS clips each contribute `1/13`, while a singleton source's clip contributes one. This explains the paper's central scoring distinction before the equation.
- Visibility is defined as an estimator output, with appropriate separation from clinical confidence and an independently measured quality rating. Spatial normalization is explicitly distinguished from earlier interpolation of that channel.
- The loss is described through feature distributions, centering, temperature, and batch averaging in ordinary language. Exact expressions have been retained rather than silently dropped.
- Mean pose is correctly described as retaining time spent in different positions even while discarding their order. Its competitive performance is not presented as proof that all motion information is absent.

## Technical checks of the simplified account

**The observation chain is still correct.** MediaPipe image-coordinate fractions become pixels; cleaning and interpolation precede torso normalization; the encoder and mean-pose control use body-relative coordinates while RF reads cleaned pixels. The text preserves nominal rather than exact 15-Hz sampling, unbounded interior interpolation, repeated-frame padding, and the absence of retained interpolation-gap counts.

**The mask description still matches the implementation.** It includes all 33 inputs, four-frame landmark tokens, seven overlapping regions, the 158-token budget, 36-draw limit, uniform span/start draws, clinical-region multiplier and membership correction. It correctly explains that arm joints inherit a boost through shoulder membership and that the context guarantee can be satisfied by a shoulder alone. The 512-mask coverage figures remain attributed to the notebook diagnostic rather than realized training exposure. “36-draw guard” is accurate programming shorthand but can be made plainer, as suggested below.

**The networks' roles remain distinct.** The context network's attention mask prevents target slots from supplying information to visible positions; hidden outputs are replaced with a learned placeholder before the position-aware predictor. The complete untransformed window supplies detached target features through the slowly updated reference encoder. The reflection caveat in the Discussion explains that this slot-based mask can fail to withhold the same physical joint after a bilateral permutation. These statements concern different coordinate correspondences and are not contradictory.

**Loss and center weighting survive the rewrite.** The final loss gives each window equal weight after averaging its own target positions. The center update gives every masked teacher token equal weight across the batch. `method-details.md` preserves `softmax(u/.10)`, `softmax((z-c)/.06)`, detached targets, cross-entropy, the once-per-batch `.9/.1` center update, and cosine EMA momentum. The main text does not mistake the 96 feature weights for three disease probabilities. Its “prediction error” language is tied to features, and the endpoint comparison explicitly warns that different batches are not a common evaluation set.

**Readout and selection remain faithful.** The fixed EMA encoder receives complete windows; the 161-position mask selects output vectors only; token means become window means and then clip vectors. Training-only standardization and class-balanced logistic regression remain separate from encoder learning. Class balancing does not equalize sources, and continuation retains model weights while restarting optimizer, center, schedules, and sampling streams. The source-stratified outer/first-inner-split procedure, seed formula and no-refit rule remain stated.

**The comparison has not become a claimed temporal intervention.** The baseline paragraph explicitly compares complete procedures. Differences in feature construction, readout and validation stage selection remain part of those procedures. Mean-pose competitiveness and the source/clip ranking reversal are appropriately reported as observations, without assigning a causal explanation to temporal dynamics, visibility, or the reflection defect.

**Numerical presentation is stable.** The main table still gives S-JEPA 0.457/0.397 and mean pose 0.452/0.410 under source/clip weighting, with the supported fold mean/SD and historical RF marker. The source-resampling intervals, three class-correct counts, source-weighted recalls, and corrected RF result remain those reviewed in v5. The bootstrap is labeled retrospective and conditional; saved notebook outputs are not claimed for it. No new model run, participant, independent seed, or corrected historical prediction is implied.

## Concrete refinements for v7

1. **Use “estimated” rather than “tracked” in the abstract.** The actual extractor operates in MediaPipe IMAGE mode and retains the first detected pose for each sampled frame. “Automatically tracked body landmarks” suggests an explicit temporal tracker or verified identity continuity that this pipeline does not provide. “Automatically estimated body landmarks” describes the evidence precisely without adding jargon or another limitation paragraph.

2. **Define masking terms at their first use, then simplify one implementation phrase.** Early in Section 4.1, add a short explanation such as “A mask marks token positions to hide from the context network and predict; the remaining positions provide context.” This gives a non-ML reader the meaning of target and context before the detailed rule. Replace “or a 36-draw guard” with “or after 36 region draws.” The rule and its stopping condition remain unchanged. A matching small terminology improvement is to introduce “Random Forest (RF), an ensemble of decision trees” once before later RF abbreviations.

3. **Name the exact methods record once and remove the repeated pointer.** Section 4.2 currently refers first to “the companion methods record” and then to “the companion methods audit.” The actual new file is `evidence/method-details.md`. Give that path or a stable named supplementary section at the first reference, then remove the repeated final sentence about the same technical implementation. Keep the settings table and the weighting explanation in the main text. This improves navigation and saves space without losing equations or method detail.

4. **Make the bootstrap's object explicit in the plain-language introduction.** “Repeatedly drawing whole source videos” is clarified later by the fixed-model paragraph, but can initially sound like re-extraction or retraining. Say that the procedure resamples groups of **saved predictions**, keeping all clips from each sampled source together. Retain the existing exact draw counts, shared draws across methods, per-occurrence weights and conditional limitations. No numerical recomputation is required for this wording correction.

5. **Consolidate the repeated future-controls discussion.** The first Discussion paragraph explains the untrained-encoder and frame-order checks, then the final paragraph repeats trained/untrained comparison and temporal-order testing. Keep the specific explanation once, integrate the reflection-control priority with the reflection paragraph, and reserve the closing paragraph for verified identities/diagnoses and external validation. The purpose is to reduce repetition for the intended reader, not to enlarge the list of unrun experiments or hide study limitations.

6. **Prepare an anonymous submission-facing companion copy.** The paper itself is anonymous, but `evidence/reproducibility.md`, which it directly names, currently exposes the absolute local path `/Users/pmui/dev/alexpose/experiments/multiple-sclerosis`. The internal audit may retain this truthful path. A supplement shared with double-blind reviewers should use a neutral experiment-root placeholder and be scanned for author-identifying paths or metadata. This is a concrete packaging issue in an existing referenced file, not a request to invent missing rights, consent, or clinical records. Preserve artifact hashes and relative reproduction commands in the anonymous copy.

## Figure and rendered-page assessment

Figure 1 now uses equal-width top-row boxes, equal-width lower-row boxes, straight horizontal arrows, and centered text that stays within its panels. The earlier pipeline collisions are resolved. Figure 2 uses straight horizontal lines to pair methods within each test fold, with distinct marker shapes as well as colors; it does not connect different folds or imply a time trend. The two confusion matrices use matching dimensions and readable percentages. No further visual redesign is needed for v7.

The inspected PDF pages show readable equations, tables and figure captions; no overfull box or undefined-reference report appears in the build output. The repeated `lineno.sty` UTF-8 warning is a package/build-environment issue, and no corrupted manuscript glyph was apparent on the inspected pages. The paper remains seven PDF pages. Check the final built copy after the narrow wording and companion changes, preserving the version-specific bibliography and figure identities.

No change to the primary metrics, uncertainty calculations, source partitions, or historical/corrected RF separation is requested. These revisions improve accuracy of terminology and navigation while retaining the experiment that the notebooks actually define.
