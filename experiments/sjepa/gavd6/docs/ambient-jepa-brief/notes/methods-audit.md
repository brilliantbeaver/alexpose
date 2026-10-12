# Independent methodology, metrics, and figure audit

Prepared for the separate 2–3 page research brief, 28 September 2026. This is an audit and selection note, not manuscript prose. The v08 manuscript and appendices are authoritative for completed study claims.

## Material reviewed

- `docs/iclr/versions/v08/paper-v08.tex`: complete main text, abstract, figure captions, and disclosure.
- `docs/iclr/versions/v08/appendix.tex`: complete appendix.
- `docs/iclr/versions/v08/supplement/technical-details.tex`: complete detailed methods, inventories, scope, and inferential definitions.
- `paper-v08.aux`: final figure numbering and location in the compiled 16-page PDF.
- Existing v08 `method`, `study-questions`, `restoration`, and `cases` figures, inspected visually.
- Targeted implementation checks in `synthetic_training_v2/models.py` and `gait_fidelity/training.py`, `preparation.py`.
- Current `docs/ambient-jepa-writeup/PLAN.md` and `writeup/OVERVIEW.md`.
- Text extracted from the attached *HAI Internship Summer 2025 – Theodore Mui.pdf*.

## Strongest defensible scientific narrative

The contribution is an evaluation of whether JEPA-inspired representations preserve a defined movement measurement under pose restoration. It separates coordinate accuracy, time-resolved knee-angle accuracy, change in a bilateral excursion summary, and anatomical assignment. The useful finding is that their rankings disagree, while the incremental JEPA benefits remain uncertain. This supports a better measurement-validation agenda. It does not establish a successful clinical model, a physical world model, or a broadly negative result about JEPA.

For Landay's lab, the operational issue is whether an ambient mobility system confuses an observation problem with a change in a resident's movement. For Delp's lab, the issue is whether an estimated trajectory retains side-specific motion differences that a downstream biomechanical comparison needs. Those are prospective applications: neither population, clinical outcome, nor a balance assessment was evaluated.

## What is JEPA-inspired, and what is supervised

| Item | Verified implementation and implication for the brief |
|---|---|
| Input | Observed 2D coordinates, pose-estimator confidence, availability, and timestamps for 128 samples and 12 joints. RGB images are inputs to the fixed upstream pose estimators, not to the learned restoration transformer. |
| Context and target | A student processes noisy, partly masked observations; an exponential-moving-average teacher processes clean projected body-model reference joints. Teacher targets are privileged synthetic supervision, including image-hidden joints. |
| Representation objective | Centered teacher–student cross-entropy follows the S-JEPA/DINO mechanism; a VICReg term discourages collapse. It is not I-JEPA's feature-regression objective. Broadly “JEPA-inspired feature prediction” is accurate; “self-supervised learning from ordinary video alone” is inaccurate. |
| Predictive meaning | A predictor estimates masked/reference feature vectors within a fully observed time window. The transformer is bidirectional. This is complete-window pose restoration, not forecasting a future motion state. |
| Endpoint auxiliary | Match predicted and teacher features for each member of an original/edited motion pair. |
| Delta auxiliary | Match the difference between the two feature vectors. Shared errors can cancel. Features have no calibrated physical unit, and the endpoint/delta teachers evolve separately. |
| Downstream output | Retain the student encoder, discard teacher and feature predictor, and fit a small coordinate readout against clean references. This differs from S-JEPA's use of its target encoder downstream. Output corrects observed coordinates and supplies absolute coordinates for missing positions. |
| Frozen versus joint | Every representation arm freezes its encoder for readout fitting. Direct fitting jointly trains encoder and readout. These procedures have equal total update counts but different encoder adaptation and coordinate-supervised exposure. |

The brief needs a short description of context → features → reference-feature prediction → coordinate readout. The 4-layer, 4-head, width-96 architecture is reported and can be compressed to “a small transformer.” Four-frame patches yield 384 tokens. Giving the complete layer dimensions is less important than stating the privileged targets, full-window access, and frozen/joint difference.

## Data, splits, and simulated conditions

- AMASS supplies motion-capture-derived walking candidates, screened by filenames and projected geometry. This does not establish healthy or diseased gait, participant age, or clinical status.
- Fitting uses **112 people, 692 motions, and 1,645 windows**. Evaluation uses **14 different development people and 155 windows**, including **12 BioMotionLab_NTroje and two KIT people**. Those are the AMASS subsets explicitly identified for the evaluated cohort in v08. Do not imply those two datasets exhaust the training pool unless a separately verified executed roster establishes it.
- The data are resampled to 25 Hz; 128 samples span 5.08 seconds from first to last sample. “Approximately five-second windows” is suitable for the brief.
- Nominal right-knee rotations of 0°, 5°, 10°, and 15° create paired synthetic movement states. An ankle-height proxy gates the edit, and it tapers at window boundaries. These are controlled kinematic edits, not biomechanically validated disease or treatment simulations.
- Editing precedes optional physical mirroring. The same camera observes both paired states. Oblique and side camera azimuths are 45° and 90°.
- Camera fitting uses the union of original, edited, and mirrored geometry. Rendering-derived person boxes are supplied to the fixed pose estimators. The evaluated setup therefore excludes real detector failures and does not validate a camera-placement strategy for a nursing home.
- RTMPose-M, HRNet-W32, and ViTPose-base supply poses. The 15° edit and ViTPose are excluded from fitting/calibration, but evaluated on the same reused development people. These are held conditions, not a separate confirmation cohort.
- Images are clear or synthetically occluded. Implementation inspection shows a fixed image-space blocker in the lower half of the frame, with default width 15% of image width (`preparation.py:235–240`, `490`). It is not a simulated chair, physical obstacle, or complete room. Graph–time masking used in pretraining is a separate operation.
- Input naming corruption is correct, globally exchanged, or exchanged over the middle third of the sequence. It permutes coordinates/confidence/availability together; reference anatomical labels stay fixed.
- Every derived variant inherits its source-person split. Known duplicate motion content cannot cross splits. Context-only normalization excludes artificially hidden observations and references from its statistics. These precautions reduce leakage but do not establish unknown identity aliases, overlap with upstream pose-estimator training, or an untouched development history.
- No independent locked-person confirmation, real GAVD evaluation, natural-video study, Sequoias evaluation, balance outcome, or fall-risk prediction is part of v08.

## Measurements that must be explained accurately

For each leg, the image-plane knee angle uses the hip–knee and ankle–knee vectors. Excursion is its 95th minus 5th percentile over a window. Signed asymmetry is right excursion minus left excursion; the main response is the change in that asymmetry between the original and edited windows. In symbols, `q_l = P95(theta_l) − P5(theta_l)`, `A = q_R − q_L`, and `Delta A = A_b − A_a`.

This is the one equation worth retaining if space permits. It makes the scientific target interpretable and explains why a scalar alone cannot locate which leg changed. Excursion also discards temporal order. It is an image-plane quantity, not a calibrated anatomical 3D flexion angle, temporal step asymmetry, time derivative, clinical improvement score, or measure of balance. The nominal 3D edit is not its projected response magnitude.

| Metric | Meaning | Important restriction |
|---|---|---|
| Response error | Absolute error in `Delta A`, degrees. | A low value can coexist with wrong trajectories or anatomical names. Zero response supplies a valid baseline without producing any trajectory. |
| Waveform error | Mean absolute knee-angle error across both legs and reference-valid timestamps, degrees. | This is the projected angular trajectory; it does not validate kinetics, balance, or clinical usefulness. |
| Normalized localization error (NLE) | Joint Euclidean distance divided by rendered person-box diagonal. | Not a metric 3D error; training's squared normalized coordinate loss is a different quantity. |
| Geometric assignment failure | Named-versus-swapped position comparison for eligible bilateral hip, knee, and ankle pairs. | Combines wrong, ambiguous, and missing outputs; cannot be reported as a pure left–right swap rate. Eligibility differs from angle scoring; 50% is not an established chance level. |

Angular eligibility is reference-defined: finite geometry, segments at least two pixels, at least 16 supported frames, and 80% coverage. Failed eligible outputs remain in the score at **720° for response and 180° for waveform**. These are scoring costs based on metric ranges, not observed clinical errors or acceptable tolerances. At minimum, a brief reporting pooled scores needs a caption or methods sentence naming those penalties.

## Comparisons and strongest numbers

1. **Practical coordinate restoration:** unchanged inputs → jointly trained direct coordinates reduces waveform error **18.57° → 12.07°**, response error **12.69° → 7.54°**, and NLE **0.0718 → 0.0298**. Its paired response improvement is **5.14° [2.15, 8.65]** (exploratory crossed 95% interval). Reporting two of these metrics is enough for a short brief.
2. **The zero-response baseline:** **5.81°**, lower than all **16** fitted variants under the recorded pooled scoring. It produces no reconstruction. Small reference responses favor this baseline, and failures also matter. This does not establish absence of response information in model features.
3. **Core JEPA versus direct, both with change supervision:** **0.69° [−0.64, 1.98]** response gain. The comparator is not the stronger direct-coordinate arm.
4. **Delta versus endpoint, both with change supervision:** response means **9.99° versus 10.36°**, gain **0.37° [−1.11, 1.76]**. The interval leaves benefit unresolved. Coordinate-only readouts reverse the point ranking, but the interaction is also uncertain.
5. **Readout supervision:** adding scalar-response and short-segment penalties worsens waveform means across all eight families. Because two terms are added together, this is not an isolated causal result about scalar supervision. For delta/ViTPose, original → low scalar → dense gives **23.17° → 19.47° → 19.19°**. The additional dense gain is **0.28° [−0.19, 0.75]**, and its response gain **0.13° [−2.69, 2.94]** does not establish preservation of response accuracy.
6. **Anatomical assignment:** under global input naming swaps, combined failure is **81.12% endpoint / 83.40% delta / 23.29% direct-coordinate**. The post hoc combined outcome needs that definition; these are not simple percentages of leg swaps. In a three-page brief, qualitative explanation may be more useful than three additional percentages.
7. **Visibility:** exploratory strata show direct-coordinate restoration beats zero response in both clear-image edit groups and loses in both occluded groups. Participant summaries favor it over zero for **14/14** people in clear images and **1/14** under occlusion. These are not population success probabilities, and their points/ranges reflect three fitted seeds, not confidence intervals.

## Inference and optimization cautions

Conditions are averaged within windows, then motions, then people, then seeds. Repeated renders do not add participants. Recorded nonheld/held weights differ between response (2:1) and endpoint outcomes (4:1). The first two primary intervals resample people and seeds separately while preserving paired comparisons; repair averages seeds before a person-level t interval. All stages reuse 14 development people and three fitted seeds. None of the intervals accounts for adaptive reuse; documented primary comparisons do not imply preregistration.

Equal auxiliary coefficients did not match initial training influence: endpoint was calibrated to **10%** of base-gradient RMS, delta to **0.0013%**, before clipping. **11,999 of 12,000** updates across the six new pretrains were clipped. This prevents an unqualified architecture-level interpretation; it does not show that clipping or gradient imbalance caused the final ranking. A compact limitation can say “the feature comparison also had unequal initial loss influence and different evolving teachers.”

Failures account arithmetically for roughly 74% of the 0.37° endpoint–delta contrast under the 720° cost. Nevertheless, both variants' unconditional successful-error contributions exceed the zero-response score even if failures cost zero. Do not describe the entire negative result as an artifact of large failure penalties. A short brief can omit this decomposition if the scoring rule and unresolved benefit remain explicit.

## Figure selection: reuse existing v08 assets unchanged

### First choice: v08 Figure 7, `figures/study-questions.pdf`

This existing appendix result figure summarizes all three primary contrasts, makes uncertainty visible, labels the correct first comparator, and avoids a selective success story. It is roughly 11 × 5.6 inches in source geometry; at 5.5–6 inches wide it occupies 2.8–3.1 inches in height with readable labels. Use the vector PDF directly without replotting, changing markers, or relabeling effects.

Required caption context: reproduced from v08 Fig. 7; positive gain favors the candidate; first two rows are pooled response/crossed person–seed 95% intervals; third is ViTPose waveform/person-t after seed averaging; same 14 development people and three fitted seeds; no pooled effect. “Change” includes coordinate, scalar-response, and geometry supervision. Failures retain 720°/180° penalties. The brief can state the scoring rule in methods to avoid a long caption.

### Alternative if the argument emphasizes restoration tradeoffs: v08 Figure 2, `figures/restoration.pdf`

This 11 × 6.5 inch figure shows all eight families, zero response, and waveform deterioration. At full text width it is readable but costs more vertical space and introduces many control names. It is less economical beside a separate external JEPA illustration. Panel C is a useful stand-alone crop only if explicitly identified as a crop of original Fig. 2C and the reader is given the supervision legend and frozen/joint distinction. Do not redraw a compact new result chart; the current request explicitly asks for existing v08 results.

### Do not prioritize the other figures within the page budget

- Fig. 8 `cases.pdf` makes the visibility connection vivid but is tall and requires a detailed caption explaining seed ranges and exploratory status. It is valuable as a linked source, less suitable in the brief.
- Fig. 1 `method.pdf` accurately shows this study, but a second architecture diagram alongside a requested external JEPA concept diagram would repeat explanation.
- Fig. 5 laterality needs additional support/denominator explanations. Its numerical detail is secondary to the main unresolved representation question.

## Corrections and ambiguities to avoid when using the earlier overview

The existing overview is broadly faithful. Its unqualified sentence “Direct restoration beats that baseline in clear images but loses under occlusion” should explicitly call the stratification exploratory. Its balance-assessment connection should continue to describe relevance, not tested performance. Avoid interpreting dense supervision's uncertain response gain as “preserves asymmetry.” Avoid naming every feature variant S-JEPA; the completed system adapts JEPA ideas and privileged reference supervision rather than reproducing S-JEPA.

The prior internship report supplies a useful arc—problem, concrete implementation, test, negative finding, next question—but it is not authority for the current study. It discusses 14 Toronto videos; those are not this study's 14 development people. It does not document Sequoias camera placement or evaluated residents. A hypothetical chair occlusion is useful motivation only if clearly separated from the fixed synthetic blocker actually used.

No material discrepancy was found between v08's stated model/metric contract and the inspected implementation passages. Source limitations remain: complete predictions/checkpoints required to repeat the end-to-end study are absent; the available aggregate outputs support reaggregation, not fresh experimental replication. The main manuscript does not enumerate the training cohort's AMASS subdatasets; the brief should not invent that list.
