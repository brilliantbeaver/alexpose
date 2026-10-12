# Independent review of draft v3

Reviewed `wacv2007-v3.tex` and the rendered six-page PDF against all seven top-level notebooks, the active model/data implementation, the notebook-first evidence hierarchy, the results audit, and the requested v2 revisions. The contact sheet, pipeline source image, and full-size PDF page 3 were inspected. The score uses the same task-specific, workshop-grounded weights as earlier drafts; it is not an official WACV score or an acceptance probability.

| Category | Weight | Score (0–5) | Weighted points | Assessment |
|---|---:|---:|---:|---|
| Vision4Neuro relevance and practical contribution | 15 | 4.0 | 12.0 | The empirical contribution now identifies a specific result: an order-free control changes the apparent model ordering, and the small source-weighted lead is concentrated in one fold. Its scope remains an exploratory behavioral-vision study. |
| Empirical correctness and uncertainty | 25 | 4.6 | 23.0 | Notebook-associated primary metrics, class errors, paired intervals, and the separate RF correction are clearly distinguished and supported by retained predictions. No known numerical substitution remains in the reviewed text. |
| Evaluation design and generalization scope | 20 | 3.8 | 15.2 | The completed source bootstrap, fold plot, weighting definitions, and explicit conditional interpretation materially improve the analysis. The reflection defect, one seed, unknown identities, and absence of temporal interventions still limit the study itself. |
| Reproducibility and methodological completeness | 15 | 3.9 | 11.7 | The coordinate chain and region-weight correction are now correct, as are target-encoder readout and continuation semantics. The core student masking operation, actual RF feature vector, exact splitter description, and specific artifact pointer remain insufficiently specified. |
| Clinical interpretation, provenance and ethics | 10 | 4.0 | 8.0 | Collection labels, uncertain identities, and missing data-use records remain explicit. These are appropriate boundaries rather than invented assurances; no prose revision can supply the absent clinical validation. |
| Literature, clarity and WACV presentation | 15 | 3.7 | 11.1 | Results are clearer and the new fold/error figure is informative. Figure 1 has visible text collisions and box overflow in the rendered PDF, preventing a stronger presentation score. |
| **Total** | **100** | | **81.0 / 100** | **Substantial improvement; targeted method and layout corrections are still required.** |

## Assessment of the v2 critiques

1. **Completed paired uncertainty: resolved.** Section 5.2 now defines 10,000 within-label source draws, seed 20261004, equal total weight per sampled source occurrence, common draws across methods, and percentile intervals. Its caveat correctly conditions on fixed OOF predictions and does not treat dependent folds or unresolved participant overlap as independent clinical observations. The reported intervals match the numerical audit.
2. **Separate corrected RF verification: resolved.** Section 6.1 preserves the notebook RF in the main comparison, labels the correction as later verification, reports the 12 changed predictions and corrected scores, and explains that matching S-JEPA hard predictions from the same seed are not independent replication.
3. **Inspect fold variation: resolved.** The fold plot exposes the large fold-2 advantage and four disadvantages instead of relying only on the pooled mean.
4. **Class-level errors: resolved.** The text reports 10/24, 8/29, and 18/35 raw correct clips and the supported source-weighted recalls. The unsupported extra PD success from v1 is absent.
5. **Artifact identification: partly resolved.** Reanalysis is clearly labeled, and the unexecuted status of notebook 06 is stated. However, “the accompanying reproducibility record” is still an unnamed promise inside the paper: readers are not given its file, the notebook-associated run ID, or a reproduction command.
6. **Specific empirical contribution: resolved.** The Introduction now states the measured weighting reversal and one-fold concentration without claiming that the study settles the utility of temporal motion.

## Required changes for v4

### 1. Explain the actual student masking operation

Section 4.2 describes a transformed student window and a complete teacher window, then jumps to the masked-slot loss. It never tells readers how hidden inputs are excluded from student context or what the predictor receives at hidden slots. These are central elements of the evaluated method, especially because the manuscript discusses a masking defect later.

Add a concise description grounded in `sjepa/models.py:108–134`: the tokenizer linearly maps four frames × three channels to 96 features and adds learned joint/time positions; target slots are excluded as attention keys/values in the context encoder; their resulting outputs are replaced by a shared learned mask token before the predictor. The teacher processes the original complete window. Do not say the per-example implementation physically drops every hidden token before the transformer—it uses key padding and overwrites the hidden outputs. One or two sentences can convey this without expanding the section into an implementation manual.

Also specify that the center updates once per batch from the mean of masked teacher features, whereas the loss averages target tokens within each example before averaging examples. This distinguishes the implemented token-weighted center from the equal-example loss. Teacher EMA uses a cosine schedule; the teacher starts as a copy of the student. These details are available in the active loop and are worth including in an algorithm description or compact settings table.

### 2. Complete Section 4.1 with the notebook's observed mask behavior

The current region-weight formula is correct, including the average-membership correction omitted in the original draft. All 33 input landmarks and the twelve-landmark role are now correctly separated. The remaining gap is that the reader sees only a nominal 60% mask budget and a generic nonuniformity warning even though notebook 02 prints concrete coverage evidence.

Add a short sentence or compact table explicitly attributed to notebook 02 cell 13: in its seed-0 bank of 512 masks, mean target fraction is 63.3%; every joint is visible somewhere in at least 74.0% of windows, yet shoulders and hips are visible at only about 20–23% of token positions. The notebook also exposes marked time bias: left-hip context is 51.6% at the first block and 7.0% at block five. These are sampler diagnostics, not realized capstone-training exposure or evidence of improved classification. Using the notebook's 512-mask bank avoids silently substituting the new 5,000-mask audit for the user's specified source of truth.

Tighten the guarantee: a visible token from the twelve-landmark set can be a shoulder, so the sampler guarantees neither a visible hip nor any leg context anywhere in a particular window. It can also mask all joints in one time block. The current “does not guarantee leg context in every time block” is true but understates what the code permits. Add that the arms receive the clinical factor because they contain shoulders; their elbows and wrists consequently inherit the region boost. This is the practical reason that the twelve highlighted landmarks do not uniquely receive the extra targeting.

For reproducibility, either give the 36-draw guard and uniform sampling of span length/start in this section or include them in a settings record. Avoid describing 1.5 as a marginal probability multiplier. No new biological or performance claim should be attached to these corrections.

### 3. Specify what the RF actually measures

“Joint-angle summaries” is too vague for the central historical comparator, especially after the paper devotes a subsection to correcting one of its columns. Notebook 06 cell 8 supplies the exact intended vector: six left/right hip, knee, and ankle means; three absolute bilateral mean differences; and six ranges. Only 15 columns vary in this path, despite the legacy container's 82 names. In the notebook-primary run two ankle-range columns are identical, and the true left range is absent.

Add those feature counts and the seed-42 RF setting to Section 5.1. A brief supplementary note can state that joint-angle observations require geometric-mean landmark visibility at least 0.3 and undefined joint statistics become zero. It should also state that `load_raw()` contains cleaned pixel coordinates, not uncleaned video observations. This prevents readers from inferring that calibrated cadence, speed, stride length, or the full 82-field gait feature suite was evaluated.

### 4. Make the source splitter and evidence record directly reproducible

Section 5.1 should say **label-stratified splitting on one row per source**, expanded afterward to clips. The implementation is `StratifiedKFold`, not an unspecified grouping heuristic, and the inner validation set is the **first** split of the four-fold development-source splitter. Preserve the existing seed formula and one-holdout qualification.

Replace the vague “accompanying reproducibility record” reference with a concrete anonymous supplementary artifact statement. At minimum name the primary run `capstone-20260921T011305448893Z`, registry prefix `9496e61b050f`, the separate corrected run, and the supplied script/manifest filenames. The repository already has `evidence/results_audit.py` and `evidence/results-audit.json`; point to that record or package it into a named supplement with the command that regenerates manuscript numbers. This is documentation of existing work, not a request for new experiments. Retain the statement that notebook 06 contains written results but no saved executed cell outputs.

### 5. Correct Figure 1's visible collisions

In both `figures/pipeline.png` and rendered PDF page 3, “88 retained / 41 sources” extends beyond its box, “Body-relative coordinates” touches the arrows and neighboring boxes, and “Training sources: … supervised probe” collides with “Test sources: one prediction per clip.” This is visible at normal page scale, not merely a contact-sheet artifact.

Reflow each box onto three short lines and put training, validation, and test descriptions on separate rows or within separate bounded panels. Regenerate the figure and inspect its full-size PDF appearance. Do not fix the problem solely by shrinking all text until it becomes hard to read. The figure should also identify pose extraction before the pixel/body-coordinate branch clearly enough that the coordinate correction is visible as well as stated in the caption.

### 6. Remove the last coordinate ambiguity and explain reflection causally

After Equation 1, change “Visibility is unchanged” to “Spatial normalization leaves the already cleaned visibility values unchanged.” The earlier preprocessing paragraph correctly says visibility is interpolated, so the shortened current sentence can still be misread as a contradiction. This is a wording correction; the actual pixel → cleaned pixel → torso-normalized chain in v3 is otherwise correct.

The Discussion's reflection paragraph should state the consequence directly: a teacher target joint can remain visible as transformed input in the student's opposite-side context when the unpermuted mask treats that slot as context. Calling it a correspondence problem alone is less informative than explaining what is inadvertently exposed. Keep the present boundaries: this occurs within training examples, does not leak held-out sources or diagnoses, and its effect on classification is unmeasured. Do not insert the audit's synthetic exposure percentage into the main results as if it came from the notebooks.

## Scope that should remain unchanged

No additional training is needed to make these v4 corrections. The missing random-encoder, temporal-order, mask, and reflection experiments remain legitimate limitations of the study rather than editing tasks that can be claimed complete through prose. Keep the corrected RF verification separate from the notebook-primary table. Retain source-weighted clip scoring, the full-input EMA readout, source-versus-person distinction, conditional uncertainty language, and the absence of diagnostic/deployment claims.

The six-page render fits within the regular-paper limit, leaving space for the required methodological clarification. A higher score should depend on those concrete corrections and a clean figure render, rather than additional unmeasured assertions.

No manuscript, notebook, model, or saved result was changed during this review.
