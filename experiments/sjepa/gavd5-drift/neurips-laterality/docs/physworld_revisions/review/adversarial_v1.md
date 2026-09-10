# Adversarial review of paper_v1.md

Reviewed 9 September 2026 against the complete v1 manuscript, original protocol, original implementation, retained primary summary, and independently checked cohort/splits. This is a skeptical Physical World AI review. It does not independently certify the newer masking numbers, which have a separate extension evidence audit.

Version 1 is substantially more defensible than the source manuscript. It removes the universal claim that predictive learning cannot discover symmetry, distinguishes a fixed identity-channel test from all possible equivariance, attributes the historical original results to the surviving aggregate file, and states that actual forecasting and clinical validation are absent. The remaining weaknesses are mainly the specificity of its measurement, the interpretation of initial-encoder controls, and the strength of the workshop contribution.

## Changes needed before a persuasive submission

### 1. Bring the measurement-path failure into the results, not only future work

The paper now interprets a learned-versus-initial performance gap while only briefly mentioning that resizing changes the target. Notebook 07 offers direct evidence that the original target and the same formula on the prepared input disagree. The existing helper was recomputed during this review: both values are finite for 623 sequences from 92 sources; source-weighted sign agreement is 0.7041029622551362 and direct agreement R² is 0.21818988696299113.

Add a subsection before the representation comparisons, explaining the target lane versus encoder-input lane and this observed discrepancy. Present its exact estimand in the caption: weight clips equally within each source and sources equally, on the finite overlap. Call it a descriptive comparison of two calculations, not model accuracy, a statistical ceiling, or proof that a particular preprocessing operation caused information loss. This guards the causal interpretation of all later negative results and supplies a useful failure illustration.

### 2. Make “initial encoder” mean the correct baseline

An initial transformer with a fitted ridge readout is not a pipeline without learning. The landmarks already come from a pretrained pose detector, the feature summary encodes named anatomical pairs, and the ridge readout is trained using target labels. The initial encoder also has architectural priors and potentially many random features. A skeptical reviewer will object if the 0.2225 result sounds like motion can be recovered without training anything.

Add one sentence defining the control precisely: “The initialization control replaces only self-supervised encoder training; pose extraction, anatomical summaries, and training-only supervised readout fitting are retained.” Use “paired initial skeleton encoder” consistently and avoid expanding the conclusion to all representation learning.

### 3. Correct the visible-token description

Section 3 says the online encoder processes visible tokens. The original implementation allocates and processes the whole joint/time token grid, zeroes target coordinate embeddings before adding position embeddings, and includes those target positions in attention. The predictor later replaces the corresponding context features with mask tokens. This differs from an architecture that discards masked tokens from the encoder sequence.

Write “the online encoder processes the token grid with selected coordinate content hidden” and draw that actual operation. See `laterality/model.py:51` and `:165`. This matters for claims about what temporal and anatomical context the learner can exploit.

### 4. State the physical meaning and remaining coordinate distortion

The v1 sentence about non-metric inferred depth is helpful but does not fully characterize the target. The extraction mixes horizontal coordinates normalized by image width, vertical coordinates normalized by image height, and inferred depth scaled by crop width/image width. Its Euclidean norm is therefore a coordinate-derived quantity sensitive to the image geometry and detector, even after pelvis centering and bilateral scale normalization.

Add the actual target-frame construction and distinguish elapsed-time normalization from metric calibration. The displacement speed divides by `diff(frame_numbers/fps)` on the observed lane; model resampling uses relative sample index. The paper need not put raw extraction code in the main text, but an appendix should record the coordinate convention. Do not make the abstract's “physical measurement” sound like a calibrated biomechanical observable; “coordinate-derived movement measurement” would be more accurate there.

### 5. Explain what bilateral geometry contributes beyond existing skeleton methods

The existing motivation is reasonable, but a workshop reviewer can still ask why this is a Physical World AI contribution rather than an application of linear probing. Make the central test explicit: an observation transformation changes a signed measurement in a known way even when the person's movement itself is asymmetric. The evaluation checks whether a learned latent representation supports that response as well as useful prediction, and whether simple mask manipulations alter the result.

A concise contribution statement should identify this relationship and the controlled comparison as the advance. The paper should not claim a new world model from within-clip masked reconstruction. The existing scope paragraph appropriately avoids that claim and should remain visible in later revisions.

### 6. Expand the condition motivation only with justified evidence

Two cited examples—post-stroke asymmetry and Parkinsonian arm swing—cannot establish the significance of this particular target across the full annotation set. The manuscript should either state why only these examples are used, or briefly discuss the varied manifestations of bilateral coordination in myopathic and cerebral-palsy gait using authoritative clinical sources.

Do not convert those examples into empirical subgroup findings. A signed left-minus-right population mean can be near zero when different individuals have different affected sides; magnitude, spatial symmetry, temporal symmetry, and coordination are different constructs. The proposed target pools shoulder and several closely related foot/leg landmarks, so it should be described as one deliberately limited contrast.

### 7. Separate feature-summary changes from claims about temporal information

Section 4.3 already notes that the motion-sensitive summary includes observation support. Give the main additional components and dimensions in the methods or appendix. A larger feature vector changes ridge regularization and capacity as well as its access to temporal variation. Any attribution to motion needs a capacity-aware, support-only control and training-only penalty selection.

For this version, retain the useful result that the summary improves the evaluated initial-encoder readout, while avoiding “the encoder contains the missing motion information” as a general explanation. Adding the toy equal-mean/different-speed example would clarify why the summary choice matters without pretending that it proves loss from contextual tokens.

### 8. Make multiple exploration and uncertainty scope visible beside results

Section 5 correctly states that source bootstrap intervals are conditional on fixed fits and that the research trajectory used the same development cohort. Keep this discussion, but mark the later mask comparisons as exploratory near their table as well. The manuscript compares many policies, readouts, training stages, and diagnostic perturbations; nominal intervals do not automatically support a family of confirmatory discoveries.

The statement “all trained arms remain below their matched initial encoder in each of the five seeds” needs the extension audit to verify every applicable arm, readout, and seed. It should not stand in for an uncertainty interval on the aggregate contrast. Describe it as consistency across the registered seed realizations, not independent replication.

### 9. Describe a nonzero low-q degeneracy

The identity-channel limitation is now explicit, but q can also be small for a constant nonzero or otherwise insensitive representation. The original zero-energy check excludes only the all-zero case. Initialization q of 0.083 does not show that random encoders have useful geometric understanding.

Add one sentence saying that small discrepancy must be evaluated together with useful readouts and feature-variation checks. In any figure showing q, annotate the registered margin and use the full retained interval [0.095, 0.138] for learned q; failure to establish a value below 0.1 is not the same as demonstrating that the underlying expected value exceeds 0.1.

### 10. Add figures that explain a real distinction

Version 1 has tables but no actual pipeline or result illustration. The requested vector figure should make four relationships clear: all 33 landmarks enter the prepared model input; selected landmark/time coordinates are hidden without changing the allocated grid; reflection augmentation or an explicitly marked later synthetic symmetry objective introduces geometric structure in encoder training; the signed target is computed on the original observed lane and reaches only the downstream readout.

A second compact result figure can compare initial versus trained readouts under the common endpoint and show the processed/original measurement discrepancy. Any constructed trajectory figure needs “illustration, not gait data” in its caption. Do not invent per-sequence successes/failures for the historical run whose prediction rows are missing.

## Smaller corrections and verification requests

- Specify the maximum short-gap length, visibility threshold, and definition of a valid four-frame token in the appendix. The current methods are readable but not sufficiently reconstructible on these points.
- Distinguish the fixed 1,200-update budget from a full pass over the available clips. The original epoch samples one clip per source plus padding; repeated training counts do not imply all clips are shown equally often.
- Clarify the mean-summary definition. The original primary features use temporal left/right sums and differences for five pairs, which is an anatomy-informed feature construction, rather than global average pooling of the whole skeleton.
- The original source threshold and null policy are fixed computational rules, but the retained local files alone do not prove public preregistration before all exploratory work. V1 avoids the strongest historical assertion and should continue doing so.
- Make the surviving original-number JSON and each newer prediction table directly discoverable in the final artifact appendix. “Original artifacts are preserved” is ambiguous when the historical report/checkpoint chain is absent; say “existing files were left unchanged” in revision metadata and state the evidence gap in the paper.
- Expand formal references into a conventional bibliography before workshop submission, verifying authors, years, titles, and source dates. The current inline links are useful working-draft citations.

## Recommended next revision

Version 2 should focus on the data-to-target information path and make its limitations visually explicit. That revision can add the verified input-agreement diagnostic, define the initial-encoder control and the anatomical feature vector, correct the hidden-token processing description, and make the core contribution a precise test of known transformation behavior under a common coordinate-derived endpoint. These changes would strengthen the causal restraint and make the unsuccessful experiments contribute to a coherent research question rather than a list of ablations.
