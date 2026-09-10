# Laterality manuscripts, revision scores and evidence

The recommended manuscript is [revision 8 — Bilateral Geometry for Evaluating Predictive Representations of Human Gait](physworld_revisions/paper_v8.md). The single canonical copy of each of the eight revisions is saved in the [revision folder](physworld_revisions/), and all new illustrations are in [figures](figures/physworld_figures.md). The original [paper.md](paper.md) remains unchanged. [TUTORIAL.md](TUTORIAL.md#7-how-close-is-this-to-a-strong-workshop-paper) includes the 10 September assessment of the paper's workshop fit.

This README is the consolidated record of scores, suggestions and critiques for the original paper and every revision. Detailed review texts are preserved in expandable sections below. The read-only [verification script](verify_physworld_evidence.py) and [full-precision recomputation output](physworld_evidence_recomputed.json) are also retained directly in this directory. The [final verification record](physworld_revision_verification.json) records the manuscript hashes, successful link checks and preserved review documents. Earlier manuscript/build notes are preserved at the end as a historical record.

The `physworld_revisions` folder is the sole location of the eight revised Markdown manuscripts. Its subfolders and submission artifacts are retained, while figures remain centralized in `docs/figures` and are linked with location-correct relative paths.

The revision folder also contains [LaTeX, PDF, and Overleaf instructions for all eight revisions](physworld_revisions/OVERLEAF.md). The typeset editions preserve the Markdown sources, use anonymous NeurIPS 2026 formatting, and include refreshed vector illustrations. The [landmark-selection schematic](figures/pose_landmarks_and_laterality.md) distinguishes full input shape from training and target selections.

## Assessment and scoring

The original paper scores **51.5/100** and the current revision **81.6/100**. These are reasoned editorial judgments, not calibrated acceptance probabilities. The revisions improve evidence selection, precision, narrative and presentation; they do not transform a single-cohort study into externally validated science.

Each dimension uses a 0–5 scale: 1 indicates a major deficiency, 3 a credible but limited workshop treatment, and 5 an unusually strong treatment. The weighted total is the sum of weight × dimension score / 5. Weights are: workshop fit 15%, novelty 15%, methods 15%, inference 15%, evidence 15%, interpretation 10%, clarity 10%, figures 5%. “Evidence” includes completeness and traceability; “methods” grades the explanation and controls, not the existence of an unrun experiment.

| Manuscript | Fit | Novelty | Methods | Inference | Evidence | Interpretation | Clarity | Figures | Total / 100 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Original | 3.0 | 2.0 | 3.5 | 2.5 | 2.0 | 2.0 | 3.0 | 2.5 | **51.5** |
| V1 | 3.3 | 2.5 | 3.7 | 3.6 | 3.0 | 4.0 | 3.5 | 2.5 | **65.8** |
| V2 | 3.4 | 2.5 | 4.3 | 4.0 | 3.2 | 4.3 | 3.5 | 2.5 | **70.3** |
| V3 | 3.4 | 2.8 | 4.2 | 4.2 | 3.5 | 4.4 | 3.7 | 2.5 | **73.0** |
| V4 | 3.6 | 3.0 | 4.2 | 4.2 | 3.5 | 4.5 | 4.0 | 2.5 | **75.0** |
| V5 | 3.7 | 3.0 | 4.3 | 4.3 | 3.5 | 4.5 | 4.1 | 4.0 | **77.6** |
| V6 | 3.8 | 3.0 | 4.3 | 4.3 | 3.5 | 4.6 | 4.3 | 4.4 | **78.9** |
| V7 | 3.8 | 3.0 | 4.4 | 4.4 | 3.5 | 4.7 | 4.5 | 4.5 | **80.2** |
| V8 | 4.0 | 3.0 | 4.4 | 4.4 | 3.5 | 4.7 | 4.9 | 4.5 | **81.6** |

The final scores remain moderate for novelty and evidence because the main contribution is a diagnostic result on one development cohort. Source-video grouping does not verify participant independence; the target lacks independent clinical validation; and no real-data forecast or multimodal-fusion result is available.

These weighted totals preserve the revision-stage assessments. The subsequent [v7 workshop reassessment](TUTORIAL.md#7-how-close-is-this-to-a-strong-workshop-paper) rates topical fit **4/5** and scientific maturity **3.5/5** separately, after revisiting the call's geometry and evaluation topics. Those judgments replace the tutorial's older 3/5 and 2/5 ratings; they are not a recalculation of this eight-dimension score. The reassessment removes the mistaken requirement that forecasting must succeed before the present evaluation contribution can fit Physical World AI.

## Eight saved revisions and their critiques

<a id="revision-original"></a>

### Original paper — 51.5/100

The original has a useful reflection audit and strong source-grouping intentions, but omits the latest completed comparisons. Its “fully powered null” and general claims about symmetry exceed the evidence. A fixed identity-channel token test is interpreted too broadly; target-component reconstruction is incorrectly used to locate failure inside the encoder. Input resizing, mask denominators and the limits of the raw-artifact record are insufficiently clear.

The first improvement is to make the trained-versus-initial comparison and the difference between internal feature prediction and observable laterality the central question. Retain precise effect estimates, distinguish historical summaries from recomputed predictions, and remove claims of clinical or general world-model validity. See the [full original assessment](#original-assessment).

<a id="revision-v1"></a>

### Revision 1 — evidence reframe — 65.8/100

[Read the complete v1 manuscript](physworld_revisions/paper_v1.md).

V1 replaces the universal symmetry slogan with a scoped audit and incorporates the latest unfavorable learning result. It acknowledges the missing original report chain, distinguishes synthetic reflection training from real-data comparisons, and retains a strategic abstract without a numerical inventory.

Its remaining weakness is methodological ambiguity. The reader cannot yet follow every tensor transformation or distinguish the historical reflection experiment from the latest zero-reflection grid. V2 therefore adds exact array shapes, the cross-entropy/VICReg objective, per-source evaluation weights, the actual mask denominator and readout-only inner validation. Independent reviewers also requested a clearer initial-encoder control and a measurement-agreement diagnostic. See [the v1 critique](#v1-critique), [core adversarial review](#adversarial-v1) and [extension review](#independent-v1-extension).

<a id="revision-v2"></a>

### Revision 2 — method and split audit — 70.3/100

[Read the complete v2 manuscript](physworld_revisions/paper_v2.md).

V2 distinguishes the raw target branch from the 64-frame model input, explains what shape and correspondence are preserved, and makes source-level training and readout selection explicit. It identifies where anatomy enters training and states that the signed target never supervises the encoder.

The evidence is still presented too coarsely: a range of teacher scores hides the individual controls, and a motion-summary improvement could be mistaken for a pure temporal-information gain. V3 adds all five arms, seed variation and MAE, the independently repeated input-agreement check, and separately labeled exploratory trained-minus-initial intervals. It also explains why scores from different notebook recipes cannot be read as a causal sequence of gains. See [the v2 critique](#v2-critique).

<a id="revision-v3"></a>

### Revision 3 — latest results — 73.0/100

[Read the complete v3 manuscript](physworld_revisions/paper_v3.md).

V3 shows the learning deficit for each completed arm and distinguishes it from the uncertain mask-versus-random comparisons. It adds the 623-clip/92-source preparation diagnostic and exposes the discrepancy between correct-clip prediction and the laterality readout.

Its clinical motivation and novelty positioning remain thin, and the larger summary changes both feature count and observation-support information. V4 brings in primary clinical research and recent skeleton/JEPA literature, defines the target's omitted aspects and cancellation behavior, and sets out competing explanations with tests that could weaken them. The independent reviews also require capacity-aware summaries, per-clip mask counts and explicit coordinate units. See [the v3 critique](#v3-critique), [core review](#adversarial-v3) and [extension review](#independent-v3-extension).

<a id="revision-v4"></a>

### Revision 4 — clinical and literature context — 75.0/100

[Read the complete v4 manuscript](physworld_revisions/paper_v4.md).

V4 connects bilateral geometry to different gait phenomena in stroke, Parkinson's disease, cerebral palsy and muscular disorders. It explicitly treats those as published motivations rather than newly discovered associations in GAVD. S-JEPA, MAMP, SLiM and recent geometric/predictive work constrain the novelty claim.

The manuscript still needs a precise account of the explicit reflection loss, a usable illustration and a stronger link from diagnosis of the current failure to a decisive next experiment. V5 adds the training graph, current numerical recipe, detailed feature dimensions, a synthetic future-decoding failure illustration and measurement conventions. It identifies Notebook 12 as retained-summary context. See [the v4 critique](#v4-critique).

<a id="revision-v5"></a>

### Revision 5 — adversarial revision — 77.6/100

[Read the complete v5 manuscript](physworld_revisions/paper_v5.md).

V5 integrates the training and results figures, makes the gradient path of the synthetic reflection penalty explicit, and distinguishes decoded future coordinates from the laterality task. It provides concrete next tests rather than implying an untested repair will work.

The cumulative manuscript is too expansive for the main paper. Independent review also finds that the historical reflection-augmentation improvement in token discrepancy must be retained, that its equation should use the implemented clamped denominator, and that the missing historical reports require consistent provenance language. V6 moves the detailed history, condition census and synthetic derivations into appendices, restores the positive geometric effect, and centers the latest verified grid. See [the v5 critique](#v5-critique), [provenance addendum](#adversarial-v5) and [extension review](#independent-v5-extension).

<a id="revision-v6"></a>

### Revision 6 — workshop synthesis — 78.9/100

[Read the complete v6 manuscript](physworld_revisions/paper_v6.md).

V6 follows a clearer arc: measurement and symmetry, controlled changes to hidden targets, access through the readout, and the predictor's correct-clip diagnostic. It combines the historical consistency improvement with the absence of a demonstrated laterality benefit, and separates the compact main argument from detailed appendices.

Final review identifies notation and documentation issues: a negative difference should be Δq, the audit did not regenerate checkpoint predictions, the bootstrap seed must be explicit, teacher and online tokens need distinct symbols, and the forecast table must name coordinate RMSE. V7 resolves these issues, corrects the figure panel reference, explains remaining jargon and supplies a linked bibliography. See [the v6 critique](#v6-critique), [core review](#adversarial-v6) and [extension review](#adversarial-v6-extension).

<a id="revision-v7"></a>

### Revision 7 — final revised manuscript — 80.2/100

[Read the complete v7 manuscript](physworld_revisions/paper_v7.md).

V7 supports a precise evaluation claim: trained predictors distinguish correct from mismatched clip features, yet all five trained arms have weaker tested laterality readouts than matched initialization. Motion and region masking change the available information, but no alternative establishes improved laterality recovery. The reflection-augmentation result is kept separate from predictive utility because only its summary evidence remains.

The remaining critique is empirical. The 2,890-dimensional summary mixes temporal statistics with support information, ridge penalties still reach the search boundary, the original-to-prepared target changes, and videos do not identify independent participants. A condition-specific clinical finding and a real gait-forecasting benefit remain unestablished. Rewriting cannot resolve these limits.

The next improvements should use saved encoders first: expand ridge penalties through training-source selection, isolate support and summary components, and audit the target after individual preparation steps. Then test explicit reflection training with a fixed evaluation, requiring improved observable prediction and geometric behavior together. Stronger confirmation needs an independently specified participant-separated setting; clinical interpretation also needs independent movement measurements. See [the final critique](#v7-final-critique) and [the prioritized research plan](#next-research).

The 10 September v7 update explains the choice of GAVD in [Section 6](physworld_revisions/paper_v7.md#6-scope-ethics-and-reproducibility), connecting movement diversity and recording conditions to bilateral evaluation. It documents MIT permissions and their limits using the repository's README and LICENSE at revision `a87859c881603443f200bcd640663d2c3d7a8136`. This is the documentation revision checked for this edit, not a claim about the original annotation download. Independent review prompted explicit wording about observational labels, source-video grouping and separately hosted recordings. The editorial score remains 80.2/100 because this improves provenance explanation without adding empirical evidence. The next provenance improvement is to record the annotation version and retrieval history alongside cohort exclusions; institutional determinations and derived-pose release status still require their own records.

The workshop appendix has been reduced from eight Markdown sections and two additional typeset sections to two focused sections: paired learning contrasts with their uncertainty, and measurement/training conventions with one reflection schematic. The historical results ledger, notebook inventory, repeated absolute scores, synthetic forecasting table and detailed follow-up matrix are kept in the expandable record below. The expanded pipeline and landmark-selection figures remain documentation assets. This edit improves focus without changing the empirical score. The remaining appendix retains the uncertainty assumptions and preprocessing distinctions needed to interpret the main findings.

<a id="revision-v8"></a>

### Revision 8 — plain-language workshop manuscript — 81.6/100

[Read the complete v8 manuscript](physworld_revisions/paper_v8.md).

V8 uses the same empirical evidence and two-part appendix, with a revised account of the methods and findings. Its abstract and introduction center the measured difference between successful clip-feature matching and lower movement-readout scores after training. Related Work connects masking, geometric representation, and predictive learning around that contribution. The review below documents corrections to factual wording, effect sizes, citations, and figures.

Both V7 and V8 organize the experiments by scientific question and evidence type. They do not call the reflection comparison or mask comparisons earlier, later, original, or historical experiments. This removes a chronology that readers do not need while preserving the distinctions that matter: which results come from saved predictions, which survive only as summaries, and which reflection-loss evidence is synthetic.

The higher clarity and workshop-fit scores reflect easier reading and a more direct account of the contribution. The empirical limits remain the same: one development cohort, uncertain participant independence, no external clinical validation of the movement measure, incomplete raw evidence for the reflection-augmentation comparison, and no real-data forecasting result. The most useful next work is still a training-only readout ablation, preparation-stage target checks, a real-data reflection-loss comparison with fixed evaluation, and independent participant-separated confirmation.

<a id="v8-adversarial-claim-review"></a>

#### V8 adversarial claim review and writing revision — 10 September 2026

Three independent reviews checked methods and inference, numerical results, and citations against the saved implementation, predictions, notebook outputs, and primary literature. Every manuscript section, both appendices, and all three figures were reviewed. Repeated claims in the abstract and introduction were checked against the same evidence as the corresponding Results passage. The final pass found no remaining substantive contradiction in the reviewed claims. Some results remain supported only by retained summaries; they were checked for faithful reporting, not described as independently reproduced.

The writing review found that the prior plain-language draft repeatedly announced a check, explained it, and closed with a disclaimer. V8 now develops the empirical argument in connected paragraphs, uses declarative headings, and consolidates limitations where they affect the inference. Quantitative claims replace vague descriptions of improvement or similarity. Rounded estimates follow the established precision convention; the full retained values appear below. The revision/date badge and notebook identifiers were removed from manuscript presentation. UTF-8 decoding confirmed that the source's accented names were intact; explicit UTF-8 handling prevents the garbled display produced by legacy PowerShell decoding.

The review also corrected substantive errors. Shared bilateral visibility is required at both ends of a target transition; target coordinates are already centered and scaled; the student masks projected patch vectors; and motion-mask candidates span all landmarks even though a gait-joint count defines the budget. The expanded readout uses absolute feature increments. Its ridge-boundary percentages now have their proper denominator, and the coordinate baseline is identified as a 264-value summary. The teacher-input arrow now begins at the pose input, reflecting the teacher's own patch embedding. These corrections improve the description of the existing experiments without adding empirical evidence. The recorded V8 editorial score remains 81.6/100; it is not an acceptance estimate or a claim about authorship.

<details>
<summary>Claim-to-evidence ledger and disposition of review findings</summary>

The main prediction grid is [292443b0…](../artifacts/motion_structured/grids/292443b0fab5339f5da7ca566a85d6172ffc5b64abe5febf2546681a0152ff57/). The cohort is [protocol_6f7baefbda07](../artifacts/paper/protocol_6f7baefbda07/cohort/). Numeric recomputation uses [verify_physworld_evidence.py](verify_physworld_evidence.py) and [physworld_evidence_recomputed.json](physworld_evidence_recomputed.json). Source paths below are relative to `neurips-laterality` unless linked otherwise.

| Claim or issue | Evidence checked | Disposition in V8 |
|:--|:--|:--|
| Clinical motivation | Primary articles [1–4] in the manuscript: stroke asymmetry, Parkinson's arm swing, cerebral-palsy asymmetry under speed changes, and Duchenne muscular dystrophy gait features. | Claims stay within those studies; DMD is named specifically. No disease association from this cohort is asserted. |
| Related Work | Primary papers [5–12], including official MAMP code and SLiM Semantic Tube Masking. | Synthesizes prediction targets, available context, and geometric behavior; the mask conditions are not described as full method replications. |
| Reflection and invariance | `laterality/geometry.py`, implemented pair exchange, and the signed target formula. | Explicit identities (M^2x=x) and (y(Mx)=-y(x)); exact invariance maps a clip and mirror to the same features despite opposite nonzero targets. These are mathematical consequences, not empirical discoveries. |
| 666 annotations, 642 pose archives, 625 retained clips, 93 sources | Cohort metadata, manifest, and `census.csv`. | Verified; five categories describe the selected subset, not all of GAVD. |
| Category counts | Cohort manifest: 270 normal, 183 myopathic, 75 stroke, 58 cerebral-palsy, 39 Parkinson's clips. | Verified; descriptive sample counts, without prevalence or affected-side claims. |
| Coordinate units | `notes/extract_augmented_poses.py:139–143`: depth is multiplied by crop width/full-frame width. | Corrected to depth rescaled from crop to frame-width units; metric calibration is absent. |
| Target normalization | `laterality/geometry.py:192–194`. | Pelvis centering and body-width scaling precede speed calculation. |
| Shared transitions | `laterality/geometry.py:244–250`. | Both paired landmarks must be observed at both ends of the same transition. |
| Target pairs and cancellation | `laterality/geometry.py`, target protocol, five pair contrasts and their average. | Shoulders, knees, ankles, heels, foot tips; at least eight shared transitions per pair. Hips provide centering. The average may cancel opposing pair differences. |
| Target/input distinction | `laterality/geometry.py:134–202`, `laterality/data.py`, protocol. | Target retains timestamps and observed transitions; encoder preparation fills short gaps, allows median-pelvis fallback, and resizes to 64 positions. |
| Tensor shapes and validity | `laterality/model.py`, data preparation and patch validity. | Verified 64×33×3 →16×33×12 →16×33×96, with all four observations required for a valid patch; all 33 positions remain allocated. |
| Source folds | Cohort `census.csv`, split records and `laterality/splitting.py`. | Verified source counts 19/19/19/18/18 and clip counts 189/182/72/77/105, with no source overlap. Participant independence remains unverified. |
| Encoder/readout split roles | `laterality_extensions/motion_readout.py`, comparative evaluation and saved selection tables. | Encoder training uses outer-training sources. Readout imputation/scaling are fitted within inner training splits; inner validation chooses ridge penalties. Inner-validation sources participated in unsupervised encoder training. |
| Training dimensions and budget | Protocol and 125 saved training histories. | Verified 64 frames, 33 landmarks, width 96, depths 4/2, four heads, batch 20, 1,200 updates. |
| Masking mechanism | `laterality_extensions/comparative_training.py:82`, `laterality/model.py:65–69`. | Hidden projected patch vectors are zeroed before positional information is added; raw coordinates are not the masking location. |
| Training loss and teacher | Comparative training implementation; `laterality/model.py`. | Cross-entropy over feature-channel probabilities, teacher without gradients, and VICReg on projected gait-pooled features from complete views. |
| Teacher EMA | `laterality_extensions/comparative_training.py:1005`; `laterality/training.py:539–545`. | Fixed 0.999 is scoped to motion/region runs; reflection augmentation schedules 0.999 toward 1. |
| Pairing across masks | Comparative training and saved manifests. | Shared initial weights, source draws, views, and updates; mask policies and their random streams differ, with matched hidden counts per clip. |
| Reflection augmentation and explicit loss | Reflection training, `laterality_extensions/symmetry_learning.py`, and synthetic notebook results. | Augmentation probability 0.5 is distinguished from the two-extra-pass reflection loss. GAVD mask runs contain neither. Explicit loss evidence remains synthetic. |
| Evaluation formula and uncertainty | [Recomputation script](verify_physworld_evidence.py), saved evaluation implementation and predictions. | Equal total source weight, pooled out-of-fold score per seed, five seed scores averaged. Whole-source bootstrap conditions on saved models/splits; no retraining uncertainty or equivalence claim. |
| Input agreement | Independent recomputation from cohort arrays: 623 clips/92 sources, sign 0.7041029623, calculation-agreement R² 0.2181898870. | Reports 70.4% and 0.218; no causal attribution to an individual step and no model-accuracy ceiling. |
| Reflection effects | Retained reflection summary in the historical evidence record below. | Δq −0.00843 [−0.01020, −0.00687]; prediction ΔR² +0.00408 [−0.00556, +0.01277]. Rounded consistently in the manuscript and explicitly summary-only. |
| Gait versus all-landmark masking | [Saved comparative summary](figures/tutorial_comparative_masking_summary.json), `teacher_mask_contrast`. | All-landmark minus gait ΔR² +0.0121842371 [−0.0183031387, +0.0502069001]. Raw comparison inputs are unavailable. |
| 125 encoders | Main grid manifests. | 25 fold/seed jobs×3 motion arms plus 25×2 region arms. |
| Motion candidates and budget | `laterality_extensions/motion_structured_masks.py:115,132,203–213`. | Samples across all valid landmarks; budget is floor(0.5×smallest valid gait-joint count in the batch). |
| Motion inspection | [Executed notebook 15](../executed/motion_structured/gavd_5yc3ve5h/15_motion_weighted_masking.ipynb), cell 8, and `motion_gavd.py:182`. | Mean 80.771 hidden targets, approximately 17% of valid whole-body tokens. Source-balanced enrichment: random −0.0001054, MAMP +0.01749, robust +0.03784. These are inspection draws, not a history of realized optimizer masks. |
| Region inspection | [Executed notebook 16](../executed/motion_structured/gavd_5yc3ve5h/16_structured_masking_and_context.ipynb), cell 10. | Hidden fraction 9.9%; target fraction with both temporal neighbors visible 69.9%→0; with at least one visible graph neighbor 98.8%→51.2%. Denominators corrected. |
| Three mask contrasts | Main grid predictions and recomputed paired intervals. | Every reported effect and interval agrees with the full-precision table below; no equivalence claim remains. |
| Compact and expanded readouts | `laterality_extensions/motion_readout.py:109–118`, `summary.csv`. | 960 versus 2,890 inputs; expansion includes temporal SD, mean absolute increments, and ten paired-support values. Individual contributions remain unmeasured. |
| Initial readout gain | `summary.csv`: R² 0.0708271559→0.2225435736. | Difference +0.1517164176, displayed +0.152. Increased dimension and observation support qualify interpretation. |
| Trained teachers and online encoders | All expanded-summary rows in `per_seed.csv`. | Teacher means 0.1007771974–0.1142094391; online means 0.063631652–0.080922532. All 25 teacher and 25 online comparisons have lower R² and higher MAE than matched initialization. |
| Scope of the training deficit | Compact-summary connected-region teacher R² 0.073756 exceeds initial compact R² 0.070827. | Universal all-readout language removed. The all-arms deficit is explicitly tied to the expanded summary. |
| Paired training intervals | Five intervals recomputed from saved predictions. | All below zero; Appendix A retains conditioning and lack of multiplicity adjustment. Alternative readouts and irreversible information loss remain unestablished. |
| Ridge boundary | `selection.csv`, selected expanded-summary rows. | Teacher 49/125=39.2%; online 96/125=76.8%, at penalty 10,000. Scope and denominator corrected. |
| Coordinate baseline | `laterality_extensions/masked_learning.py:421–440`, grid `summary.csv`. | R² 0.0345412064 comes from 264 coordinate/validity summary values, not flattened poses or an optimized coordinate model. |
| Predictor correspondence | `predictor_diagnostics.csv`, `comparative_evaluation.py:704`. | 375/375 trained versus 33/75 deduplicated initial evaluations; source-weighted mean squared feature-error comparisons, not per-clip success counts. |
| Feature-loss interpretation | Teacher-specific targets and saved diagnostics. | Raw losses are not ranked across different teacher feature spaces. The cause of the readout deficit is unresolved. |
| Future experiments | No completed source-separated real-data forecast or reflection-loss comparison in the retained evidence. | Explicitly proposed work: readout components, penalty extension, preparation-stage measurements, reflection-loss comparison, and past-input-only forecasting. |
| GAVD permissions | Pinned [README](https://github.com/Rahmyyy/GAVD/blob/a87859c881603443f200bcd640663d2c3d7a8136/README.md) and [MIT License](https://github.com/Rahmyyy/GAVD/blob/a87859c881603443f200bcd640663d2c3d7a8136/LICENSE). | Research annotation use, notice retention, warranty/liability terms, and separate video retrieval verified. Dataset licensing is not treated as a release right for third-party videos or derived poses. |
| Reproducibility claims | Fresh run of `verify_physworld_evidence.py`; all history hashes compared with their training manifests. | 10 grid-file hashes, 125,000 prediction rows, 200 pooled score rows, and 125 training histories verified. Maximum R² discrepancy 9.19×10⁻¹⁷. No new training or regenerated checkpoint predictions. |
| Figures | Compared SVGs, print SVGs and values with model code and numeric ledger. | Teacher receives poses through its own embedding; “Masked tokens” replaces “Masked input”; positive target means a positive mean pair contrast. Notebook identifiers removed from V8 figures. |
| Citations, encoding and metadata | All 16 primary sources, NeurIPS 2026 formatting instructions, UTF-8 source, generated bibliography and PDF metadata. | V8 owns its bibliography during conversion; GAVD surname is Ali Armin; GaitJEPA punctuation corrected; publication/preprint distinctions retained. Working-copy badge removed; official anonymous author metadata retained. |

Paired R² differences from saved predictions, shown here to ten decimal places; full values are retained in the recomputation JSON:

| Comparison | Difference | 95% source interval |
|:--|--:|:--|
| MAMP minus random | −0.0008343231 | [−0.0203716412, 0.0152958180] |
| Robust motion minus random | 0.0004848301 | [−0.0154974338, 0.0151733910] |
| Connected region minus random | −0.0086686895 | [−0.0332897030, 0.0176663663] |
| Motion random trained minus initial | −0.1088189647 | [−0.1700928599, −0.0412191374] |
| MAMP trained minus initial | −0.1096532878 | [−0.1650708929, −0.0513514559] |
| Robust motion trained minus initial | −0.1083341345 | [−0.1706396311, −0.0423071291] |
| Region random trained minus initial | −0.1130976867 | [−0.1661916517, −0.0596232756] |
| Connected region trained minus initial | −0.1217663761 | [−0.1827874953, −0.0552813437] |

All eight intervals use 2,000 paired source resamples with bootstrap seed 812. Repeated seeds are fits on the same cohort. Reflection and gait-eligibility summaries have different retained evidence and are not included in this recomputation table.

The primary-source bibliography check retained Springer’s formal 2025 S-JEPA chapter year, the cerebral-palsy article’s 2020 publication year, published MAMP/GAVD venue metadata, official seq-JEPA/SER proceedings pages, and explicit preprint/repository status for the remaining sources. GAVD’s pinned citation parses the surname as “Ali Armin”; the build now carries this correction from V8’s own Markdown into its TeX and PDF. Numeric citations remain in first-use order, with 16 cited entries and an unnumbered References heading.

</details>

#### Citation and reference review — 10 September 2026

The previous reference list mixed complete author entries with title-only records, omitted several venues and identifiers, and retained one source that v7 never cited. Descriptive links sometimes appeared after a group of claims rather than beside the claim they supported. The Markdown list also had a different order from the numbered PDF bibliography. These were presentation and traceability weaknesses, not grounds for changing the numerical results.

The [official NeurIPS 2026 author kit](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip) permits either numerical or author–year citations and any consistent reference style. It calls for an unnumbered References heading and permits 9-point reference entries. It does not mandate BibTeX or alphabetical ordering. V7 now uses one numerical sequence, ordered by first citation, with 16 cited entries in both Markdown and PDF. Clinical citations sit beside their supporting claims; authors, venue names, volume/pages or article numbers, verified DOIs and explicit preprint identifiers have been supplied consistently. The uncited 2023 DMD synergy paper was removed from v7's list but remains available to the earlier versions that cite it.

Publication status is explicit. The [Springer S-JEPA record](https://link.springer.com/chapter/10.1007/978-3-031-73411-3_21) gives 2025 in its formal chapter citation despite the ECCV 2024 venue and November 2024 online date. The [cerebral-palsy publisher record](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2019.01399/full) specifies 2020 despite the 2019 DOI component. MAMP and GAVD use published venue metadata; SLiM, V-JEPA 2.1 and Human-JEPA remain identified as preprints. GaitJEPA's acceptance is attributed to its author repository without inventing proceedings pages or a DOI. GAVD documentation uses an access date and pinned commit, rather than treating the 2026 inspection date as its publication year; the license's 2024 copyright is recorded separately.

The conversion checks citation numbering, first-use order, duplicate entries, complete metadata coverage and agreement between the cited sources and reference list. An independent review caught a Markdown escape that the math reader would misinterpret; the literal bracket syntax is tested through Pandoc. The bibliography remains embedded in each TeX file, with normal-size headings and permitted 9-point entries. All eight generated TeX/PDF/Overleaf packages were refreshed because they share the metadata catalog. V7 and V8 each have eight main pages, one reference page and two appendix pages. Rendered references and citation placement were checked, and each extracted Overleaf project compiled with matching text and pagination. No scientific result changed.

#### Numerical precision review — 10 September 2026

V7 previously gave most scores and interval bounds to four decimal places and the mean target count to three, while Figure 2 already rounded readout scores to three decimals. That inconsistency added detail without improving interpretation. The paired intervals are much wider than the last displayed decimal, and they condition on the saved fits rather than measuring all sources of uncertainty. More digits would not make the inference stronger.

The revised convention distinguishes significant digits from decimal places:

| Quantity | Display in v7 | Reason |
|:--|:--|:--|
| R² scores, substantive differences and interval bounds | Three decimal places | Keeps a common scale while removing unnecessary fourth-decimal detail. |
| Nonzero differences smaller than 0.001 | One significant digit | Preserves a small estimate's sign without printing a spurious exact zero. The retained −0.0008 and 0.0005 already have only one significant digit. |
| Average target count | 80.8 | Conveys a mean over variable counts rather than a fixed number of selected tokens. |
| Descriptive percentages | At most one decimal; whole percentages for approximate context | Retains 70.4% measurement agreement and 9.9% masking coverage, while simplifying context availability to about 70%, 99% and 51%. The existing 39% and 77% ridge summaries remain appropriate. |
| Counts, dimensions and configured values | Unchanged | Cohort sizes, diagnostic counts, training updates, resamples, thresholds, EMA values and epsilon specify the study; rounding them could describe a different experiment. |

All score displays were checked against [full-precision recomputations](physworld_evidence_recomputed.json) and the [figure ledger](figures/physworld_figure_provenance.json). In particular, the robust-motion lower bound is −0.0154974338…, which rounds directly to −0.015. Re-rounding its former −0.0155 display would incorrectly give −0.016. Its point estimate is 0.0004848301…, retained as 0.0005 under the one-significant-digit exception. Differences are calculated before rounding: subtracting two rounded scores can differ from the displayed, directly rounded paired contrast.

The convention is stated in Section 3.4 and applied to the main text, mask table, Appendix A and both Figure 2 layouts. The plotted coordinates and full-precision evidence files are unchanged. The results-only regeneration route preserves the redesigned pipeline and reflection illustrations. Refreshed vector PDFs and their validation are recorded in [the precision-update figure report](figures/learning_precision_pdf_validation.json); the submission builds and Overleaf projects use those assets. These are reporting changes, with no new training, changed result or change to the editorial score.

<details>
<summary>Detailed supporting record moved out of v7 for workshop readability</summary>

This preserves the supporting material before the 10 September appendix reduction. Section letters and notebook references below describe that earlier structure; the current manuscript has only Appendices A and B. Full historical precision, exploratory controls and local provenance remain available here without becoming additional workshop claims.

## Appendix A. Additional exploratory learning contrasts

These contrasts were calculated during the 9 September revision from the complete latest prediction grid using 2,000 paired source-video resamples with NumPy bootstrap seed 812. They were not part of the original registered reflection audit. Each compares the final teacher's motion-sensitive summary with its matched initial encoder using the same readout family, clips and seed. Intervals are marginal percentile source-bootstrap intervals conditional on the fitted models; there is no family-wise multiplicity adjustment.

| Training arm | Learned minus initial $R^2$ | 95% source interval |
|:--|--:|:--|
| Motion uniform | −0.1088 | [−0.1701, −0.0412] |
| MAMP-style motion | −0.1097 | [−0.1651, −0.0514] |
| Robust motion | −0.1083 | [−0.1706, −0.0423] |
| Region uniform | −0.1131 | [−0.1662, −0.0596] |
| Connected region | −0.1218 | [−0.1828, −0.0553] |

The [recomputation script](verify_physworld_evidence.py) checks the prediction coverage and reproduces the recorded pooled scores before calculating these contrasts. The [extension evidence audit](README.md#extension-evidence-audit) records the retained artifact paths, and the [recomputed JSON](physworld_evidence_recomputed.json) stores full-precision values and analysis settings.

## Appendix B. Competing explanations and tests that could distinguish them

| Explanation compatible with the findings | Useful next test | Outcome that would weaken it |
|:--|:--|:--|
| Preparation changes the relevant measurement | Recompute the target after each preparation stage on common valid transitions, with timestamps retained | Little discrepancy at every stage despite poor learned readouts |
| Summary choice obscures useful features | Compare mean, support-only, temporal SD, increments and their combinations on frozen initial and trained encoders | Deficit persists under adequately selected readouts with matched capacity |
| Ridge regularization is insufficient | Expand the penalty grid identically for all arms using inner training-source validation | Interior selected optima still produce a learned deficit |
| The prescribed reflection action is too restrictive | Fit a shared involutive channel map on training pairs and evaluate held-out pairs, with initial and mismatched controls | No meaningful advantage for genuine reflected pairs on held-out sources |
| Predictive targets emphasize another clip property | Compare local motion supervision or visible-token targets at matched budgets and measure a common observable outcome | Improved internal correspondence still fails to improve the common outcome |

These are proposed discriminating tests, not explanations established by the current data. Fitting a reflection map that preserves lengths would add an orthogonality assumption; the input reflection does not guarantee that learned feature channels obey it. Any new decision based on this cohort remains development work and should precede confirmation in an independently specified setting.

## Appendix C. Geometric and forecasting illustrations

![Schematic body reflection with joint exchange and sign-changing movement target, distinguished from constructed readout parity and constant-feature consistency.](figures/reflection_and_target.svg)

*Figure 3. Reflection changes the sign of the defined target through horizontal reflection and anatomical exchange. The bodies and values are schematic, not clinical examples. An odd readout can guarantee its output transformation for any encoder. The figure's generic output antisymmetrization uses one half; Appendix G gives the evaluated feature construction with $1/\sqrt2$. A constant nonzero token representation illustrates why geometric agreement alone is insufficient.*

A simple trajectory construction explains the limits of temporal averaging. Let left positions be $(0,1,0,1)$ and right positions $(0,0,1,1)$ at equal intervals. Both means are 0.5, but median speeds are respectively 1 and 0. Exchanging the trajectories reverses their contrast while retaining the means. This demonstrates a limitation of mean raw positions. An encoder could encode movement before pooling, so it does not prove a limitation of every mean-pooled representation.

Notebook 14's retained synthetic demonstration uses eight generated sources, six for training and two for testing, one seed, and four updates per arm. A decoder fitted on training-source future teacher features is frozen and applied to either observed or predicted future features.

| Future horizon | Coordinate RMSE decoded from observed future features | Coordinate RMSE decoded from predicted future features |
|:--|--:|--:|
| 0.25 seconds | 0.011 | 2.51 |
| 0.50 seconds | 0.013 | 2.61 |
| 0.75 seconds | 0.025 | 2.64 |

These are displayed coordinate RMSE values in normalized coordinate units on 24 common landmark endpoints from two synthetic test clips. They are not calibrated distances. The observed-future route uses information from the answer period and tests decoder expressiveness; the predicted route is the forecast. Its failure after four updates neither establishes real gait performance nor rules out a well-trained forecasting model. Per-example artifacts for this demonstration are absent, so no new uncertainty or finer precision is inferred.

For a future real-data comparison, prepare the input from an observed prefix only, predict future teacher features, and decode common future coordinates using a training-fitted decoder. Compare persistence, recent velocity, direct past coordinates, initial encoders and mismatched-future training at identical eligible endpoints. Observed-future features test decoder expressiveness with access to the answer period; they are not a deployable baseline. These controls distinguish the predictor's ability to forecast movement from the decoder's ability to recover coordinates when future observations are already available.

## Appendix D. Measurement conventions and information boundaries

Elapsed time is $(\mathrm{frame\ number}-\mathrm{first\ frame})/\mathrm{fps}$, and speed divides coordinate differences by positive elapsed-time increments. Target normalization uses the observed midpoint of both hips at each frame. If either hip is unavailable, that frame cannot support the target. The scale is the median of valid shoulder-pair and hip-pair widths across the clip; the implementation uses 1 when a usable scale is absent or degenerate. The target contrast uses $\epsilon=10^{-8}$.

Input normalization separately permits a median pelvis fallback, or zero when no pelvis is observed, after short-gap interpolation. It resizes values and validity to 64 positions; validity must reach 0.999 after resizing, and invalid input values are zeroed. These are engineering conventions for noisy pose estimates, rather than a recovery of anatomical metric geometry. For past-only tasks, all such references must be estimated from the permitted prefix.

Reported prepared-coordinate corruption tests remove only small gaps after normalization and interpolation. Previously removed observations can already have influenced those operations. Those results concern representation sensitivity, not performance under genuinely absent raw observations, and are excluded from the main efficacy argument.

## Appendix E. Annotation census and clinical motivation

| Local annotation | Clips / sources | Relevant published observation and implication |
|:--|--:|:--|
| Normal | 270 / 29 | Supplies the dataset reference category; a zero signed contrast is not a clinical definition of normal walking. |
| Stroke | 75 / 18 | Spatial and temporal asymmetries differ across walkers; a single speed contrast covers only part of gait behavior. [Patterson et al.](https://pubmed.ncbi.nlm.nih.gov/18226655/) |
| Parkinson's | 39 / 9 | Arm-swing asymmetry motivates preserving side identity; our shoulder points do not measure arm swing directly. [Lewek et al.](https://pubmed.ncbi.nlm.nih.gov/19945285/) |
| Cerebral palsy | 58 / 9 | Studies include unilateral and bilateral presentations and examine speed-dependent symmetry. [Gait-speed study](https://pubmed.ncbi.nlm.nih.gov/32082235/) |
| Myopathic | 183 / 28 | Dystrophy studies examine pelvic compensation and multisegment asymmetry, illustrating features beyond this target. [Longitudinal kinematics](https://pmc.ncbi.nlm.nih.gov/articles/PMC9201072/), [kinematic-synergy study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10388506/) |

These are literature-based motivations, not findings about clinical differences in our cohort. In particular, the broad myopathic annotation does not identify Duchenne muscular dystrophy. The data do not supply verified affected-side labels, standardized disease severity, or the subtype information needed to connect target sign to a particular impairment. Here, laterality means a signed comparison of estimated left- and right-side movement.

## Appendix F. Training details and explicit geometric supervision

The latest recipe uses AdamW with learning rate $10^{-3}$, weight decay 0.05, gradient clipping at 1, teacher EMA 0.999, student/teacher temperatures 0.10/0.06 and target-center EMA 0.9. CUDA BF16 computation retains FP32 weights, loss reductions and frozen evaluation. The original .6 gait-token mask and later .5 motion budget differ; neither is a fraction of the entire input. Valid region masks use six connected landmarks across eight time blocks, with equal realized counts in their paired reference. Equal updates do not establish equal runtime.

For clip $b$, define $D_b=\|Z_\theta(Mx_b)-S Z_\theta(x_b)\|_{C_b}^2$ and $E_b=\|Z_\theta(Mx_b)\|_{C_b}^2+\|S Z_\theta(x_b)\|_{C_b}^2$, where $C_b$ identifies common valid positions. Notebook 09 adds
$$
\mathcal L_{\mathrm{refl}}=\frac1B\sum_b
\frac{D_b}{\max(\operatorname{stopgrad}(E_b),10^{-12})}.
$$
This matches the implementation's detached, clamped denominator. Both numerator branches update the shared online encoder through two additional unmasked passes. The demonstration uses weight 1, 16 input frames and width 16. It has no real-GAVD result and supplies no signed target supervision. Identity action on feature channels is a modeling choice; a constant nonzero representation can satisfy it.

The [expanded pipeline](figures/training_pipeline.svg) and [figure documentation](figures/physworld_figures.md) explain all training and evaluation entry points. For the proposed extension, a fitted involutive feature-channel action or explicit even/odd channels would test another geometric assumption. Neither has a completed real-data result in this package.

## Appendix G. Historical results retained as context

The original figure JSON supplies three-decimal values, while the existing docs README retains five-decimal historical estimates and a claim of prior artifact verification. That prior verification cannot be repeated from the current checkout because the original report/prediction/checkpoint chain is absent. No unavailable value is inferred.

| Historical comparison | Estimate | Recorded 95% interval |
|:--|--:|:--|
| Learned primary $R^2$ | 0.05979 | [−0.02527, 0.12571] |
| Learned minus initial $R^2$ | −0.01798 | [−0.03851, 0.00248] |
| Reflection augmented minus vanilla $R^2$ | 0.00408 | [−0.00556, 0.01277] |
| Reflection augmented minus vanilla token $q$ | −0.00843 | [−0.01020, −0.00687] |
| Learned minus initial token $q$, vanilla | 0.03055 | [0.01576, 0.04763] |
| Learned constructed-odd $R^2$ | 0.04302 | [−0.04356, 0.11283] |
| Learned minus initial constructed-odd $R^2$ | −0.05874 | [−0.09549, −0.01740] |

The historical token test uses teacher tokens $Z_{\bar\theta}$, whereas Appendix F's training loss uses online tokens $Z_\theta$. The teacher discrepancy is the common-valid squared residual $\|Z_{\bar\theta}(Mx)-SZ_{\bar\theta}(x)\|^2_C$ divided by the sum of the two teacher-token energies. It leaves feature channels unchanged. The diagrams label the same anatomical permutation $P$; the manuscript uses $S$. Learned vanilla $q=0.114$, interval [0.095,0.138], is retained at three decimals; its interval crosses the operational 0.10 margin, while failing the upper-bound acceptance rule. A low $q$ can reflect insensitive shared features.

Let $z(x)$ be the frozen 960-dimensional bilateral summary, distinct from the token array $Z(x)$, and let $g(x)$ be a fitted scalar prediction. The odd construction uses $z^-(x)=[z(x)-z(Mx)]/\sqrt2$, no feature centering and no readout intercept. Then $g(Mx)=-g(x)$ holds algebraically, independently of its accuracy for the measured $y$. The reported performance tests the whole readout pipeline and cannot identify every source of its useful information.

Notebook 12's retained teacher $R^2$ values are −0.022633 for gait targets and −0.010449 for all-landmark targets, with initial 0.048160 and direct pose 0.129978. The all-landmark-minus-gait contrast is 0.012184, interval [−0.018303,0.050207]. Its raw predictions were unavailable for this review. Notebook 08 used an earlier implementation and fixed penalty; its more negative numbers are omitted from the main argument. Comparing scores across these recipes cannot isolate a masking or regularization effect.

## Appendix H. Latest absolute results and notebook map

| Frozen features or control | Mean $R^2$ ± seed SD | Mean absolute error |
|:--|--:|--:|
| Training-source target mean | −0.0106 ± 0.0000 | 0.046172 |
| Direct prepared-pose summary | 0.0345 ± 0.0000 | 0.044960 |
| Initial encoder, mean summary | 0.0708 ± 0.0186 | 0.044344 |
| Initial encoder, motion-sensitive summary | 0.2225 ± 0.0268 | 0.041546 |
| Motion-uniform teacher, motion-sensitive | 0.1137 ± 0.0110 | 0.043658 |
| MAMP-style teacher, motion-sensitive | 0.1129 ± 0.0306 | 0.043751 |
| Robust-motion teacher, motion-sensitive | 0.1142 ± 0.0210 | 0.043595 |
| Region-uniform teacher, motion-sensitive | 0.1094 ± 0.0088 | 0.044037 |
| Connected-region teacher, motion-sensitive | 0.1008 ± 0.0092 | 0.043756 |

Each row covers 625 clips and 93 sources. SD describes the five optimization seeds, not sampling uncertainty. The initial encoder, direct-pose, and training-mean controls are reused across arms; their repeated appearance in machine-readable tables is not independent replication. “Initial” refers only to the S-JEPA weights: the pipeline still uses a pretrained pose detector, supplied anatomical identities, feature engineering, and a supervised ridge readout.

| Notebooks | Question or role | Evidence used here |
|:--|:--|:--|
| 00–02 | Protocol, target/cohort audit and source splits | Code, cohort and splits checked; current scope is internally fixed after development |
| 03–05 | Fold-local training, held-out probes and aggregation | Historical summary; original complete report chain unavailable |
| 06 | External-subject evaluation gate | No completed external validation |
| 07 | Measurement and readout diagnostics | Prepared-target agreement recomputed on 623 clips / 92 sources |
| 08 | Earlier matched-count anatomical masking | Historical context; not pooled with later recipes |
| 09 | Explicit reflection training | Synthetic only |
| 10 | Past-only movement prediction | Implemented route; no retained real training result |
| 11–13 | Mask construction, controlled training and predictor/readout checks | Notebook 12 retained real summaries; teaching executions separately synthetic |
| 14 | Decode observed versus predicted future features | Synthetic demonstration only |
| 15–16 | Motion selection and structured-context audits | Real-data intervention checks |
| 17–18 | Paired training and motion-sensitive readouts | Complete latest grid independently recomputed |

</details>

## What the evidence supports

The latest grid contains **125,000 saved prediction records**, representing repeated evaluations of **625 clips from 93 source videos**. The audit independently recomputes 200 pooled score rows and three recorded source-bootstrap intervals, verifies ten grid-table hashes and checks 125 complete 1,200-update histories. It does not regenerate model predictions or independently rehash every checkpoint. The new trained-minus-initial intervals are explicitly exploratory, conditional on fitted models, and unadjusted marginal intervals.

The initial encoder with the enhanced summary reaches mean source-balanced R² 0.2225; the five trained teachers reach 0.1008–0.1142. All are below their matched initial control in every seed. This control still contains the pretrained pose detector, anatomical schema, feature construction and supervised ridge readout. It does not represent an entirely unlearned vision system.

The strongest positive learning diagnostic is correct-clip versus other-source hidden-feature correspondence. It can exploit contextual posture or observation properties as well as motion. The historical augmentation contrast also improves token discrepancy, Δq = −0.00843 [−0.01020, −0.00687], without a demonstrated predictive gain. These successes are retained alongside failures.

Original reflection figures and the earlier README preserve aggregate estimates, but the original report/checkpoint/prediction chain is unavailable in the present checkout. Notebook 12 also has retained summaries without its raw prediction directory. Earlier review claims of full verification are historical records, not checks repeated during this revision. The [provenance correction](#adversarial-v5) supersedes the initial audit's incomplete statement that only the rounded figure JSON survives.

## Figures and title choices

The main [training pipeline](figures/training_pipeline_compact.svg) shows named anatomical sides, masking, the online encoder, predictor, EMA teacher and source-separated readout. Its dotted reflection-loss branch is explicitly synthetic-only. The [detailed pipeline](figures/training_pipeline.svg) supplies the tensor and gradient details. The [reflection schematic](figures/reflection_and_target.svg) and [latest results figure](figures/learning_results.svg) illustrate the mathematical control, successful correspondence and unfavorable learning contrasts. All four are editable SVGs with PNG previews and [provenance](figures/physworld_figure_provenance.json).

Seven title suggestions, with rationale, are saved in [physworld_laterality_title_options.md](physworld_laterality_title_options.md). The selected title is *Bilateral Geometry for Evaluating Predictive Representations of Human Gait*. The abstract focuses on the physical-representation question and shared observable outcome rather than enumerating processing stages or numeric results.

<a id="next-research"></a>

## Prioritized improvements that require new evidence

| Priority | Concrete improvement | Decision it can support |
|:--|:--|:--|
| 1 | Broaden a common ridge grid; compare mean, support-only, SD, increments and controlled-dimensional combinations using training groups | Whether the current deficit depends on the readout/search boundary |
| 2 | Recompute the target after each preparation stage on common observations and original timestamps | Which preparation operation changes the measurement |
| 3 | Pilot an explicit reflection objective with excluded validation sources, paired initialization and measured extra compute | Whether geometry improves both observable prediction and transformation behavior |
| 4 | Confirm the selected result with verified participant separation and a compatible independent observation pipeline | Generalization beyond repeatedly inspected GAVD sources |
| 5 | Decode common future-coordinate endpoints from strictly past-only inputs and compare simple forecasts | A defensible movement-dynamics contribution |
| 6 | Add independently measured gait outcomes and affected-side metadata for a clinical question | Whether the coordinate proxy relates to clinically meaningful movement |

## Workshop fit and readiness

The revised paper primarily addresses articulated geometry and evaluation protocols. It has no additional sensing modality or demonstrated physical-property estimation. The current [Physical World AI call](https://physworld-org.github.io/physworld.github.io/cfp/) permits an eight-page long paper or four-page extended abstract, excluding references and appendices, under double-blind review. It lists 9 September 2026 for archival papers and 29 September–29 October for non-archival papers; the call does not specify a deadline time zone.

The current V7 PDF has eight main pages, one reference page and two appendix pages, for eleven pages total. The shortened supplement contains two sections and one reflection schematic; the main text retains its two figures. The NeurIPS layout was rendered and visually checked, and the Overleaf ZIP compiled in isolation. Section 6 now explains the GAVD choice and repository license. No new training was launched. Institutional ethics, data-use and derived-pose release determinations remain unresolved in the existing governance record; no submission or release was performed.

The named **codex:adversarial-review** skill was unavailable after searching the installed skill and plugin locations. Independent research reviewers provided the adversarial-review fallback documented below. The local code-regression review skill was inspected but not applied as a substitute. No claim is made that the unavailable named skill ran.

## Full review record

The following expandable sections preserve all review prose. Early reviews describe their version's state; later responses and the final scorecard supersede resolved findings. Path references are updated to the final file locations. Executable verification and numerical JSON are retained as separate files linked above.


<a id="original-assessment"></a>

<details>
<summary>Original multidimensional assessment</summary>

# Assessment of the original paper and revision contract

Reviewed 9 September 2026. Target: paper.md. Scores are reviewer judgments on a 0–5 scale, not calibrated acceptance probabilities. A score of 1 indicates a major deficiency, 3 a credible but limited workshop contribution, and 5 an unusually strong treatment. Weighted score = sum(weight × score/5). Evidence quality and presentation are scored separately so editing cannot manufacture stronger science.

| Dimension | Weight | Original score | Reason |
|:--|--:|--:|:--|
| Physical World AI relevance | 15 | 3.0 | Articulated geometry and evaluation fit; dynamics, physical intervention and multimodal evidence are absent. |
| Contribution and novelty | 15 | 2.0 | Reflection audit is useful; generic geometry/JEPA claims overlook established methods. |
| Method description and controls | 15 | 3.5 | Source grouping and paired initializations are strong; tensor preparation and actual mask denominator are underspecified. |
| Statistical inference | 15 | 2.5 | Cluster uncertainty is appropriate; “fully powered null” and categorical conclusions overstate it. |
| Evidence completeness and traceability | 15 | 2.0 | Latest completed notebooks are omitted; original report/prediction chain cannot be reproduced from the retained summary alone. |
| Interpretation and scientific restraint | 10 | 2.0 | Identity-channel failure is generalized to equivariance; self-consistency oracle is used to localize failure. |
| Clarity and narrative | 10 | 3.0 | Definitions help, but repeated slogans and defensive phrasing crowd out the intellectual sequence. |
| Figures and reproducibility communication | 5 | 2.5 | Original refers to figures without embedding them; laterality entry points and evidence status need explicit depiction. |
| **Weighted total** | **100** | **51.5/100** | Promising evaluation study requiring substantial revision. |

## Findings that materially change the paper

**1. Reframe the contribution around the latest evidence.** The current draft omits the complete anatomical, motion-weighted, and connected-region comparisons. The latest finding is that trained predictors distinguish correct from mismatched clip features while all five trained arms underperform their matched initial encoders on the tested laterality endpoint. This is a stronger empirical organizing question than a universal slogan about symmetry.

**2. Replace categorical null language with estimates and scope.** “Fully powered null” requires a justified effect size and power or equivalence analysis that the package does not supply. The recorded primary learned-minus-initial interval includes zero; that is insufficient to conclude zero benefit. Failure to meet a success gate supports a failed criterion. It does not prove the complement of the scientific hypothesis.

**3. Restrict the equivariance claim.** The token test prescribes joint exchange and identity action on feature channels. General equivariance allows another channel transformation. The text should name the tested transformation each time it interprets q. The original learned q interval also crosses the .10 margin; failing an upper-bound gate does not establish that the entire confidence interval lies above it.

**4. Correct the information-loss inference.** Reconstructing the target from its own five components checks arithmetic. It neither tests the prepared encoder input nor identifies whether preparation, encoder learning, pooling, or regularization caused poor prediction. Notebook 07's target-recomputation diagnostic is agreement between measurements, not model accuracy or an information ceiling.

**5. Be exact about preservation.** The archive has variable T×33×4 pose/visibility entries. Input preparation produces 64×33×3 coordinates and 64×33 validity, then 16×33 tokens. Short-gap interpolation, pelvis normalization and resampling change values and time discretization. Preserve provenance, anatomical correspondence, mask alignment and each clip's target association; do not claim exact physical trajectories are preserved.

**6. Explain actual masking budgets.** The base .6 fraction is applied to eligible gait tokens, with shared batch feasibility; later .5 settings also refer to a smaller pool. Neither means 60% or 50% of all body tokens. Both sides remain explicit anatomical identities. Anatomical masks are not signed laterality supervision.

**7. Separate completed data from demonstrations.** Notebook 09's explicit reflection penalty and Notebook 14's future-feature decoder have synthetic executions. Whole trajectories and temporal gaps have coverage audits but no trained real comparison. These cannot be promoted into clinical or real forecasting results.

**8. Correct evidence provenance.** The original rounded aggregate values are retained in docs/figures/v21_figure_numbers.json, but this checkout does not contain their full original report/checkpoint/prediction chain. Existing source preparation and later comparison artifacts are different evidence. State the distinction and prioritize recomputable recent results.

**9. Make the medical motivation specific and bounded.** Stroke, Parkinson's, cerebral palsy and myopathy can affect different movement features. Published clinical work motivates measurement; the local annotations do not establish affected side, subtype, severity, or a condition-specific association with this target. The target averages signed contrasts and can cancel opposing asymmetries.

**10. Situate novelty and workshop relevance honestly.** S-JEPA, MAMP and recent SLiM already address relevant representation and masking methods. Geometry-aware evaluation offers a defensible workshop contribution. A convincing claim of physical dynamics would need observable future prediction; a multimodal claim needs additional independently measured sensors.

## Seven successive revision objectives

1. Repair scope and incorporate the current evidence.
2. Make input preservation, target correspondence, split boundaries and inference explicit.
3. Strengthen the within-experiment comparisons and explain positive and negative diagnostics.
4. Ground the health motivation and novelty in primary literature; make competing hypotheses falsifiable.
5. Resolve adversarial objections about measurement, parity, adaptive development and forecasting.
6. Edit into a focused workshop manuscript with provenance-backed figures and a concrete follow-up design.
7. Address final independent review, check all claims and links, and retain a candid readiness assessment.

Each paper is saved separately. A critique of each version records addressed issues and remaining work; earlier versions are retained as an audit trail. Later scores reward clearer evidence use and better argumentation, while limitations requiring new data remain unresolved.

## Requested review capability

The named codex:adversarial-review skill was not found in the available catalog, local skills, plugin cache, or repository. The available review-agent skill addresses code regressions and is not an appropriate substitute for a manuscript review. Independent research reviewers therefore provide an explicitly labeled adversarial-review fallback. No claim is made to have executed the unavailable named skill.

## Workshop constraints checked against the current call

The workshop includes articulated geometry and evaluation protocols. Its posted format allows an eight-page long paper or four-page extended abstract, excluding references and appendices, with double-blind review. The call lists 9 September 2026 for archival papers and 29 September–29 October for non-archival papers; a deadline time zone is not supplied there. The final Markdown is a manuscript source requiring pagination in the official style before submission. [Official call](https://physworld-org.github.io/physworld.github.io/cfp/)

## Preservation record

Original paper SHA-256: C1ED8B39F242ED5CC9A6E0A9935E3A4799879E85F5DC3E4B31288C35B9C8B210.
Original tutorial SHA-256: 1944000E5D139BE64E5DB53CDB7D286EE10A006E4E53D37A5CD6D1DE2927282C.
No experiment, original paper, tutorial, or governance record is edited by this revision package.

</details>

<a id="v1-critique"></a>

<details>
<summary>Revision 1 critique and decisions</summary>

# Review of revision 1 → decisions for revision 2

Revision 1 substantially improves the original by including the latest unfavorable training result, acknowledging the unavailable primary artifact chain, and narrowing reflection conclusions. Its remaining weakness is a method description that is still too easy to misread.

| Issue | Revision decision |
|:--|:--|
| “Preserved geometry” could imply unchanged trajectories | Add stage-by-stage shapes and identify interpolation, normalization and index-based resampling. |
| JEPA description omits actual loss | Specify teacher-distribution cross-entropy plus 0.05 VICReg; distinguish diagnostic MSE. |
| General description of laterality entry conflates historical and latest training | Add entry-point/status table and state latest reflection and symmetry weights are zero. |
| Mask percentage lacks its denominator | Explain authorized gait-token pool and shared feasibility. |
| Nested CV sounds like encoder validation | Identify readout-only tuning on an encoder already pretrained using all outer-training inputs. |
| Source-balanced R² remains a name | Define its weights and denominator and clarify the single-seed OOF estimand. |
| The null is informal | Separate trained-over-initial and mask-over-random superiority questions; avoid equivalence/power claims. |

Remaining objections: the latest results need individual arm values and direct learning-contrast uncertainty; condition-specific motivation needs better sources and qualifications; the story needs a more economical organization and complete visual evidence. These are scheduled for later revisions.

Reviewer score for v1: 65.8/100. Better writing and evidence selection improve the assessment; no new efficacy result has been created.

</details>

<a id="v2-critique"></a>

<details>
<summary>Revision 2 critique and decisions</summary>

# Review of revision 2 → decisions for revision 3

Revision 2 makes the training and splitting procedure substantially easier to audit. Its main weakness is that it gives a range of trained scores without explaining the individual comparisons, and still postpones the input-measurement problem.

1. Replace the teacher range with all five trained arms, seed variation, MAE and matched controls. Identify repeated baselines and distinguish seed SD from source intervals.
2. Add the independently repeated Notebook 07 calculation with its exact overlap cohort. State that agreement after preparation is not model accuracy or a maximum attainable R².
3. Add a separate, clearly post-hoc bootstrap of trained-minus-initial scores. The original mask contrasts cannot supply uncertainty for a different estimand.
4. Explain that the initial S-JEPA baseline retains a pretrained pose detector and anatomy-informed supervised readout.
5. Correct the student description: hidden coordinate content is suppressed within the token grid; masked positions and positional embeddings remain.
6. Explain why the mask interventions are real: target-motion enrichment and lost temporal brackets are observed. Keep different mask families paired with their own budgets.
7. Include online-encoder evidence and the boundary-selected ridge penalties. Neither teacher selection nor modest penalty tuning resolves the learning deficit.
8. Avoid treating numerical changes between Notebooks 08, 12 and 18 as causal effects of masking or regularization.

Revision 3 implements these requests. It leaves the original rounded reflection results as historical context rather than relabeling them independently reproduced evidence. Independent audits of the source code and retained predictions contributed to these changes.

Reviewer score for v2: 70.3/100. Remaining work concerns literature, medical motivation, competing explanations, and editing the growing material into a focused workshop narrative.

</details>

<a id="v3-critique"></a>

<details>
<summary>Revision 3 critique and decisions</summary>

# Review of revision 3 → decisions for revision 4

Revision 3 gives the latest results enough detail to support its conclusion, including a separately labeled exploratory learning contrast. Its scientific motivation still falls short of the user's request and the workshop audience.

- Add condition-specific motivation from primary clinical research while distinguishing local annotations, published observations, and unmeasured clinical outcomes.
- Explain what the signed speed target omits: phase, stride timing, loading, cancellation across joints, and cancellation across affected sides.
- Put the normalized-image/inferred-depth convention into the main methods so “physical measurement” cannot be read as calibrated biomechanics.
- Add recent primary literature that constrains novelty: S-JEPA, MAMP, SLiM, seq-JEPA, Soft Equivariance Regularization, V-JEPA 2.1, Human-JEPA and GaitJEPA. Identify preprints and avoid transferring their gains to GAVD.
- Treat MAMP-style sampling as an adaptation of one component, with the existing JEPA objective retained.
- Explain constant-feature degeneracy in q and qualify the random encoder baseline.
- Make alternative explanations falsifiable through a table of controls and potential disconfirming outcomes.
- Incorporate the extension review's corrections: counts matched per clip rather than across the entire region batch, 2,890-dimensional enhanced summaries, explicit bootstrap scope, and measured body-neighbor availability.

Revision 4 adopts these points without adding disease-association claims. The health discussion identifies why symmetry matters while acknowledging that the five-pair target covers only a narrow subset of gait. It retains negative findings and avoids treating every diagnostic as independent evidence.

Reviewer score for v3: 73.0/100. Additional empirical strength would require a mechanism, external replication, or an observable forecasting benefit; prose revisions cannot supply these.

</details>

<a id="v4-critique"></a>

<details>
<summary>Revision 4 critique and decisions</summary>

# Review of revision 4 → decisions for revision 5

The independent core and extension reviewers agree that the central numerical claim is now defensible, but identify remaining gaps in implementation precision, evidence status and the workshop argument.

| Review finding | Action in v5 |
|:--|:--|
| Abstract implies calibrated physical measurement | Use coordinate-derived movement contrast and initial-encoder readout. |
| Target collection and training tensor descriptions are ambiguous | Retain full grid, specify loss-selected valid targets and four three-coordinate observations. |
| Explicit laterality training is requested but only verbally described | Add the actual online-token penalty, denominator handling, gradient path, extra passes and synthetic-only status. |
| Notebook 12 sounds freshly reproduced | Identify its retained-summary status and unavailable raw directory. |
| Summary dimension/capacity confound remains incompletely explained | Define 960 versus 2,890 features and motivate support-only and capacity-controlled comparisons. |
| Latest recipe is obscured by historical settings | Add actual .5 budget, region extent, EMA, temperatures and BF16/FP32 scope. |
| Clinical and geometry concepts need visual explanation | Embed the compact training pipeline, latest results and schematic reflection figure. |
| World-model relevance lacks a decisive next test | Require joint improvement in observable utility and geometric behavior; specify pilot source exclusion and independent confirmation. |
| Synthetic forecasting deserves an honest failure illustration | Add an appendix with observed-versus-predicted future decoding, actual tiny scope and no real-data claim. |
| Reconstruction conventions omitted | Add timestamps, coordinate units, pelvis fallback, body scale and epsilon in an appendix. |

This version addresses the reviewers' substantive writing requests. Research requests that require new training or a clinical dataset become explicit future tests rather than invented completed results.

Reviewer score for v4: 75.0/100. The paper now needs editorial selection: its cumulative form contains too many tables and too much historical context for the intended main-paper space.

</details>

<a id="v5-critique"></a>

<details>
<summary>Revision 5 critique and decisions</summary>

# Review of revision 5 → decisions for revision 6

Independent reviews found that v5's accumulated detail exceeded what belonged in the main argument, while one positive historical result had been omitted.

Revision 6 leads with a linked evaluation of movement measurement, target geometry, feature correspondence and downstream utility. It keeps two main figures and the three controlled mask contrasts, while moving condition counts, synthetic equations, original numerical results and absolute latest scores into appendices. Its main text is approximately 3,260 whitespace-delimited words before references and layout; this is an eight-page-intent manuscript, not a verified eight-page PDF.

Substantive corrections are as follows:

- Preserve the historical reflection-augmentation benefit in token discrepancy, −0.00843 [−0.01020,−0.00687], alongside the absence of a demonstrated prediction gain. Its omission would overstate failure.
- Distinguish the more precise historical README ledger from freshly recomputed evidence. Its earlier verification claim cannot be repeated with the original raw chain absent.
- Match the explicit reflection-loss equation to the actual detached, clamped denominator.
- State that the complete teacher can encode information already visible in context; correct-clip prediction does not isolate withheld movement.
- Keep the latest Notebook 18 grid as the sole freshly reproduced large prediction grid. Notebook 12 is retained-summary context.
- Require an observable utility benefit as well as geometric improvement in the proposed next experiment, and account for additional reflection-loss compute.

The final revision still needs notation and caption checks, a review of all linked figures, and a consolidated README with scores and concrete suggestions for each version. Research limitations remain: a single development cohort, unvalidated target, readout-boundary selection, no independent clinical or real forecasting result.

Reviewer score for v5: 77.6/100.

</details>

<a id="adversarial-v1"></a>

<details>
<summary>Independent core review of revision 1</summary>

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

</details>

<a id="independent-v1-extension"></a>

<details>
<summary>Independent extension review of revision 1</summary>

# Independent extension review of paper v1

Reviewed 9 September 2026 against notebooks 08–18 and the retained latest grid. This review concerns `paper_v1.md`; it does not overwrite that version. The extension evidence audit and its independent recomputation are the supporting record.

The first revision has a defensible central result and generally keeps its clinical, symmetry and forecasting claims within the available evidence. Its biggest remaining weakness is that the main result is more rigorous in the saved artifacts than it appears in the paper: the visible table omits useful controls and paired uncertainty, while the method description compresses distinct training recipes into one account. The following items can be addressed without training another model.

## Necessary corrections

1. **Separate the original reflection experiment from the latest masking grid.** Section 3 describes laterality entering training through reflection with probability one half and then moves to the newer encoders without declaring their distinct settings. Both latest-grid reflection flags are zero, and the trainer rejects nonzero values. Add a short explicit distinction. The forthcoming v2 is understood to address this.

2. **State the actual objective and masking behavior.** The latest experiment uses centered teacher-distribution cross-entropy plus 0.05 times the full VICReg objective. The auxiliary term includes view agreement in addition to variance/covariance. Hidden embeddings are zeroed and positional placeholders retained, so “processes visible tokens” is potentially misleading if read as visible-token pruning. The forthcoming v2 is understood to address this.

3. **Disambiguate “target prediction.”** The final sentence of §4.3 says “no alternative mask demonstrates improved target prediction.” Feature prediction demonstrably improves during training, and its targets are teacher vectors. Replace this with “no alternative mask demonstrates improved recovery of the held-out laterality score.” Use “teacher-feature prediction” and “laterality readout” consistently throughout the paper.

4. **Scope the bootstrap-count sentence.** Section 5 currently refers generically to “the 2,000-resample intervals,” after reporting several historical intervals. The latest grid and Notebook 12 use 2,000; earlier experiments have separately retained inference configurations. Write “For the latest mask comparisons and the new trained-versus-initial reanalysis, we use…” and avoid retrospectively assigning the newest procedure to historical figures.

## Additions that materially improve the argument

5. **Give the trained-versus-initial claim its own uncertainty.** The existing mask intervals answer a different comparison. Add the exploratory paired intervals now retained in `physworld_evidence_recomputed.json`. For example, motion-uniform teacher minus matched initial is ΔR² −0.108819, 95% interval [−0.170093,−0.041219]. All five arms have negative R² intervals and positive MAE intervals under source resampling. Identify these as a new 9 September analysis conditional on fitted models, with no familywise correction; do not call them registered confirmation.

6. **Restore the direct-coordinate control and complementary error metric.** The main table omits direct pose (R² 0.034541, MAE 0.0449596) and gives trained arms only as a range. Include direct pose and show MAE for initial mean-motion (0.0415461) and each trained teacher (0.0435951–0.0440373). This makes clear that initial features contain accessible predictive structure and that the trained disadvantage appears in both squared and absolute error. A table in the appendix can retain exact arm values and seed SD if the main text must remain short.

7. **State the dimensionality and regularization limitation precisely.** Mean summaries have 960 dimensions; mean-motion has 2,890 including ten support fractions, fitted from only 74–75 outer-training videos. The maximum alpha 10,000 is chosen in 49/125 teacher mean-motion fits and 96/125 online mean-motion fits, versus 0/125 initial mean-motion fits. This does not invalidate the measured paired contrast, but it limits claims about the representations under adequately explored decoding. The original and later alpha grids differ, so numerical changes between notebooks are not an isolated regularization experiment.

8. **Separate motion order, amplitude and observation support.** Section 4.3 already acknowledges support, which is helpful. Add that standard deviation is invariant to temporal ordering, while absolute increments and support are distinct components. The combined summary cannot identify which component produces the improvement. A support-only baseline and corresponding direct-coordinate ablation are the most discriminating next checks.

9. **Quantify the positive intervention audits enough to support the causal sequence.** The paper currently states that the intended geometric effects occurred but supplies no magnitude. One sentence or figure should show equal 80.771 mean targets with motion enrichment near zero for uniform and positive for the motion samplers, and the region audit's temporal brackets 0.699→0 with visible graph-neighbor fraction 0.988→0.512. Retain the denominator: approximately 17% of valid all-landmark tokens for motion and 9.9% for regions. The different budgets are why random references are distinct.

10. **Give the correspondence result an explicit scope.** There are 375 trained diagnostic rows, each formed from one fold/seed/evaluation-mask combination; all favor the matching clip on the eligible control subset. Initial controls favor matching targets in only 33/75 rows per experiment, with the same initial control repeated between experiments. These are dependent diagnostics in each model's own teacher space. A count is useful, provided it is not presented as independent replications or a cross-model ranking of raw gaps.

11. **Name the implemented MAMP comparison accurately.** “MAMP-style” is acceptable but leaves the intervention underspecified. It follows the official implementation's per-clip maximum normalization and Gumbel top-k sampling, with explicit missing-transition handling, while keeping this study's teacher-feature target. It does not reproduce MAMP's full coordinate-motion-prediction method. A short methods sentence and authoritative citation prevent a reviewer from treating the comparison as an unfair claim about the full method.

12. **Include the main failed downstream test without overloading the paper.** A brief appendix or evidence-status table can show Notebook 14's synthetic observed-future decoding versus predicted-future decoding gap, labeled four training updates, one split, two generated test sources. The existing prose correctly says real forecasting has not run. Do not let this demonstration occupy the space needed for the completed real-data experiment.

## Story and presentation

The motivation and abstract already identify an observable physical property without claiming a deployed world model. They would benefit from one clearer bridge from health motivation to the geometric task: unequal limb contributions can reflect several mechanisms, and a signed displacement score deliberately tests whether the representation retains the side distinction before attempting disease interpretation. The text should avoid implying that every condition causes the same kind of asymmetry or that available condition labels reveal the affected side.

The argument will be easier to follow if the core sequence is explicit: a reflection rule defines how the measurement should transform; the initial control tests what architecture and input already provide; different mask interventions test which missing-information tasks encourage learning; and predictor/readout disagreement exposes the unresolved task alignment. The first revision contains all four pieces, but the quantitative center of gravity should move from the historical summary toward the directly audited latest grid.

A vector figure should visually separate four places where anatomy appears: named landmark input positions, target eligibility, the shared gait-landmark regularizer, and the bilateral readout. Explicit reflection training needs a separate historical/proposed branch. Otherwise the figure could accidentally promise a symmetry-learning contribution that the strongest completed experiment does not test.

The evidence warrants a careful unfavorable result about the current recipe and linear readout. It does not warrant a general failure claim about JEPA, erasure of motion information, equivalence of masking methods, or clinical diagnostic performance. The first revision mostly respects these limits; adding direct uncertainty and the omitted controls will make that restraint look well supported rather than merely cautious.

</details>

<a id="adversarial-v3"></a>

<details>
<summary>Independent core review of revision 3</summary>

# Adversarial review of paper_v3.md

Reviewed 9 September 2026. The entire v3 manuscript was read, including its new methods table and exploratory contrast appendix. This review concentrates on inferential correctness and the connection to Physical World AI. The newer masking estimates are accepted subject to the independent extension audit; this reviewer independently checked the core cohort, split, original summary, and input-reconstruction diagnostic.

V3 resolves most of the serious objections to v1. The original-run provenance gap is prominent, its R² estimand is correctly defined, the original and prepared measurement lanes are separate, the initial-encoder control now retains its trained pose detector and fitted ridge readout, and new bootstrap analyses are explicitly exploratory. The added contrast intervals strengthen the report of poorer performance after pretraining under the evaluated readout, without establishing a general information loss. No remaining fatal statistical overclaim was found in the central tables. The following changes would improve precision and the final argument.

## 1. “Preservation of a physical measurement” still overreaches in the abstract

The abstract ends by calling the endpoint a “task-relevant physical measurement.” This quantity combines image-relative x/y with inferred depth and was not independently validated against movement measurements. The methods appropriately call it non-metric, but readers should not need to reach the limitations to qualify the abstract.

Use “coordinate-derived movement measurement” or “an anatomically defined movement contrast.” The main abstract result can remain the discrepancy between clip-specific latent prediction and downstream access to that contrast. That is a credible evaluation finding without implying calibrated physical state reconstruction.

## 2. A small q can arise from a nonzero insensitive representation

The paragraph restricting q to identity-channel transformation is good. It still presents initial q = 0.083 without mentioning that a constant nonzero encoding can be perfectly consistent under this operation while predicting nothing. The zero-energy check in the original code excludes only all-zero degeneracy.

Add a brief warning that q is interpreted jointly with feature variation and predictive utility. The relevant finding is an increase in the chosen discrepancy, not a claim that initialization understands symmetry or that training destroys geometry generally. If showing a threshold, include the learned q interval [0.095, 0.138]; it straddles 0.1 even though its upper-bound acceptance criterion fails.

## 3. Define the feature summaries and capacity difference

V3 says the motion-sensitive summary includes temporal variation, feature changes, and observation support, but the actual feature dimensions and bilateral aggregation remain absent. This matters because increasing feature dimension changes the supervised problem and its ridge penalty. The initial encoder's strong improvement could reflect several parts of that changed readout.

Give a concise formula or appendix table for the mean and motion-sensitive summaries, including their dimensions. Keep the support-feature confound and add capacity/regularization to that same sentence. A future comparison should remove support features, compare matched dimensions or controlled projections, and select every regularization choice on training sources only. The paper can acknowledge this without running further experiments during revision.

## 4. Explain the coordinate convention and time convention together

The preprocessing table is a useful addition. A reader still cannot reproduce the target's norm from the phrase “estimated coordinates.” The archive extraction normalizes x by image width, y by image height, and inferred z by width after crop correction. Elapsed time comes from frame-number increments divided by frame rate. Uniform relative-index resizing in the model lane does not preserve that timing.

An appendix should record these conventions, epsilon, the bilateral body-scale rule, and the target/input distinction when pelvis landmarks are missing. This is consequential because image aspect ratio, detector error, and missing pelvis observations can affect the endpoint and eligibility even when the mirror algebra is exact. Do not call the normalization a recovery of calibrated physical geometry.

## 5. The validity-mask table has one implementation-dependent phrase to refine

The last row of the shape table says “Gathering and padding retain each clip's own targets and common mask count.” That is not a general description of the original full-token-grid implementation, whose targets have a shared selected count and are gathered only after token processing. The word “padding” can suggest arbitrary padded target values enter the loss.

Use a direct description such as “The full joint/time grid remains allocated; target selection and validity identify the values used by the loss.” If a later trainer does pad variable target sets, describe that case separately and state how padded entries are excluded. The four-frame patch row should say “four three-coordinate observations,” rather than “four coordinates,” to make its final dimension 12 immediately clear.

## 6. Condition motivation should distinguish asymmetry constructs

The planned condition-specific motivation is worthwhile if it remains a literature-based explanation. The endpoint averages signed speed contrasts over pairs and over each clip, so opposite affected sides or opposite pair-level deviations can cancel. A near-zero y can accompany substantial temporal, spatial, phase, or absolute asymmetry. Shoulder movement is also only an indirect link to literature on arm-swing amplitude.

Make these distinctions in the motivation and target limitations. Avoid claiming that the project has discovered asymmetries associated with any listed condition. The clinical literature should justify studying bilateral structure, while this experiment tests one measurement and a representation pipeline.

## 7. Clarify the contribution through a falsifiable next comparison

The revised manuscript is honest about having no demonstrated gait forecasting, planning, or multimodal fusion. Its workshop relevance still needs a more direct articulation than an assertion that the findings could matter to future dynamics models.

State the reusable contribution as a linked evaluation: a known observation transformation, a common target that transforms predictably, a paired initialization comparison, and a distinction between feature-prediction diagnostics and measurable output utility. Then propose one decisive prospective follow-up: retain a fixed input/target path and readout evaluation, introduce an explicit reflection objective or architecture, and test whether it improves both useful prediction and transformation behavior on held-out sources. A positive q-only result would not refute the observed utility gap; improved held-out target recovery with preserved variation would be more informative. This gives the research trajectory a clear intellectual consequence without asserting that an untested repair will work.

## Smaller editorial and evidentiary refinements

- Number the new measurement-path subsection 4.1 and renumber the rest; 4.0 makes the paper look like an append-only working log.
- The abstract's “information already present at initialization” is acceptable only as access under the evaluated feature construction. “Improves the initial-encoder readout” is narrower and less likely to be interpreted as an information-theoretic finding.
- Define “enrichment” in the mask audit: it is a difference in the implemented token-motion statistic, not a percentage increase or a biological motion measure. State whether negative values are possible and keep the source-weighting rule in the appendix.
- Explain that a zero seed SD for deterministic baseline rows means the predictions were reused and did not depend on optimization seed; it is not zero sampling uncertainty. V3 already notes reuse, so a short table-caption clarification is sufficient.
- The causal language “predictor training can succeed” should stay tied to the own-source versus other-source target diagnostic. This diagnostic can be driven by static posture, acquisition, or support. V3 handles this well; later revisions should resist tightening it into evidence of learned dynamics.
- The reproducibility sentence “Original notebooks and experiment artifacts are preserved” remains ambiguous after the explicit statement that the historical report/checkpoint chain is absent. A manuscript can instead state which artifacts are available; the revision README can say existing files were left unchanged.
- Figures should use actual saved aggregates, with the initial/trained and alternative/reference pairings clear. A constructed toy trajectory needs a visible explanatory label; historical per-clip examples cannot be reconstructed without the absent prediction chain.

The planned v4 additions on condition-specific motivation, endpoint cancellation, authoritative related work, and a falsifiable next experiment are appropriate. They should supplement the current numerical evidence rather than expand the empirical claim from a small pose-representation audit into validated clinical gait modeling or a general finding about world models.

</details>

<a id="independent-v3-extension"></a>

<details>
<summary>Independent extension review of revision 3</summary>

# Independent adversarial extension review of paper v3

Reviewed 9 September 2026. This is a fresh review of `paper_v3.md` against the completed motion/region grid and the implementations. The separate v1 review remains preserved.

The central quantitative claims now match the retained evidence. The main table includes the direct-pose control and MAE; Appendix A correctly labels the new trained-versus-initial source bootstrap as exploratory and conditional on fitted models. The description of teacher-distribution cross-entropy and the gait-pooled VICReg term is technically consistent with the implementation. There is no remaining contradiction between the real training objective and the MSE predictor diagnostic.

The following corrections and additions would materially strengthen the next version.

| Priority | Location | Adversarial concern | Concrete revision |
|---|---|---|---|
| Necessary | §2 array table, final row | “Common mask count” can be read as requiring all clips in a batch to have the same number of targets. Connected-region draws can have different valid counts across clips. | Say “each clip's target count is matched across paired arms; counts may vary between clips.” State that target losses are averaged within each clip, then across clips. |
| Necessary | §5 bootstrap paragraph | The unqualified “2,000-resample intervals” follows historical reflection estimates with separate inference procedures. | Scope this count to the latest mask intervals and the new Appendix A reanalysis. Retain historical results under their original provenance. |
| Important | §3 recipe | The paragraph specifies the historical 0.6 mask parameter but gives no current 0.5 parameter or numerical precision. Readers cannot reconstruct which recipe produced the central table. | Add a compact latest-recipe statement: 1,200 updates, batch 20, current gait-derived mask fraction 0.5, constant EMA momentum 0.999, CUDA BF16 training with FP32 weights/loss reductions/evaluation. The existing architecture/AdamW values can be referenced rather than repeated. |
| Important | §4.3 learned readout | The much larger motion summary and high regularization boundary frequency are described only partly. | State mean-summary dimension 960 and mean-motion dimension 2,890 including ten support fractions. Link this to 74–75 training videos and the planned common alpha-grid expansion, without implying that expansion will necessarily fix the deficit. |
| Important | §4.3 intervention | “MAMP-style” does not define which part of the published method is tested. | Identify the official-code motion sampler convention, validity adaptation, and unchanged teacher-feature target. Make clear that the full MAMP motion-target method is not reproduced. |
| Important | §4.3 structured result | The structure intervention is less fully specified than its claim warrants. | Give the latest connected region as six graph-connected landmarks over eight of sixteen blocks, with about 9.9% valid input hidden and a separate count-matched random reference. Report visible graph-neighbor fraction 0.988→0.512 alongside temporal brackets 0.699→0. |
| Useful | §4.4 predictor diagnostic | “Useful clip correspondence” sounds semantically stronger than the measured contrast. | Prefer “sensitivity to correct clip–target pairing under this diagnostic.” Give 375/375 positive trained fold/seed/mask rows, noting their dependence; initial controls are positive in 33/75 rows per experiment and repeated across experiments. |
| Useful | Reproducibility paragraph | Appendix A links the script but not its retained machine-readable result. | Link `physworld_evidence_recomputed.json` so a reviewer can inspect exact values and provenance without running the script. |

## Bias and control interpretation

The initial baseline is now properly described as an initial S-JEPA encoder within a pipeline that already contains a pretrained pose detector, anatomical identities and a supervised ridge fit. This is an important correction: it prevents a reader from interpreting the 0.223 R² result as a system with no prior training or supervision anywhere.

The paired contrast estimates the effect of this pretraining recipe under these readout choices. A negative value is informative even when the readout is imperfect, but it is not a lower bound on the amount of motion information in the trained features. The alpha-boundary observations are therefore a limitation of interpreting representation quality, rather than a reason to remove the unfavorable result. The current wording mostly preserves that distinction.

The newer experiments use all the same development recordings as the earlier studies. Source-disjoint outer evaluation blocks a direct fitting path from test recordings into a particular fitted model, while repeated inspection can still influence which questions, features and comparisons are attempted. Appendix A labels its new analysis correctly. The next version should keep the already inspected outer cohort described as development evidence and reserve independent confirmation for another cohort or reserved setting.

The mask audits are successful manipulation checks. They show that selection moves toward higher displacement and that connected regions remove measured local context. They do not establish that those quantities capture clinically useful motion or that the predictor relies on interpolation. Equal target counts control one aspect of supervision; geometry and content-dependent sampling can still change information content. The existing text that refers to different “observable clues” is appropriately limited.

The original/prepared-target agreement diagnostic has R² near the initial-encoder score, but those numbers should not be treated as a paired comparison. The diagnostic has 623 clips/92 sources, compares two formulas, and fits no held-out predictor; the learned table has 625/93. Its current explanation that it is neither a model score nor an information ceiling is essential and should remain.

## Cross-entropy and regularizer details

The latest loss forms teacher probabilities as `softmax((target − center)/0.06)` and predicted probabilities through logits divided by 0.10. Teacher probabilities are detached. Hidden-target cross-entropy is reduced by mean targets within clip, then mean clips when counts vary. The full-input auxiliary pathway pools valid tokens from the twelve gait landmarks, projects that summary, and applies `25×view-MSE + 25×variance-penalty + covariance-penalty`, with the resulting VICReg scalar multiplied by 0.05. These details support the present high-level description.

The teacher update uses constant momentum 0.999 in this grid; the target center is updated with momentum 0.9. Separate online and EMA-teacher readouts both underperform their initial controls, so the central discrepancy is not explained by evaluating only the EMA model. One need not include every constant in the main text, but a precise supplementary recipe should preserve them.

Because the auxiliary pathway uses unmasked training views, direct hidden-content isolation applies to the masked predictor pathway. It does not mean the online encoder is forbidden from ever observing those training coordinates through another self-supervised objective. The manuscript's present full-input regularizer description handles this correctly. A pipeline diagram should preserve that distinction visually.

## Selection and emphasis

The manuscript is now strongest when it places the completed latest grid at the center and treats the earlier reflection results as the reason for the investigation. The mean-motion initial/trained contrast has directly inspectable predictions, strong controls and newly quantified conditional uncertainty. Its relationship with the positive predictor diagnostic gives the paper a substantive finding beyond a list of nonsignificant masks.

The health motivation should remain specific without implying diagnostic validation. A signed coordinate-derived displacement score tests whether side information survives the representation pathway; different neurological and muscular conditions can affect gait through different mechanisms. Condition labels are neither affected-side labels nor ground truth for this score. Explicit references can motivate why asymmetry matters, while the present experiment addresses measurement access.

The main remaining concern is methodological compression rather than unsupported results. A reviewer should be able to identify the exact source population, target, encoder recipe, training labels, matched baseline and uncertainty scope for each central claim without inferring them from the notebook sequence. Addressing the items above would close that gap while keeping the abstract free of procedural numbers.

</details>

<a id="adversarial-v5"></a>

<details>
<summary>Revision 5 review and historical-provenance correction</summary>

# Adversarial review of v5 and original-result provenance addendum

Reviewed 9 September 2026. The full `paper_v5.md`, the existing `docs/README.md`, the symmetry-loss implementation, the mean/motion feature implementation, and the reflection diagram were read. The current original-run artifact directory was checked again and still contains only cohort, inputs, splits, and the protocol snapshot.

This addendum corrects one incompleteness in the earlier core audit: the retained three-decimal figure JSON is **not the only surviving source of original-run aggregates**. The pre-existing `docs/README.md` contains five-decimal results, including a statistically distinguishable reduction in strict token error from reflection augmentation. The report/prediction/checkpoint chain remains absent in this checkout, so these are historical reported aggregates rather than results independently reproduced in this revision.

## 1. Material correction: reflection augmentation did improve the specified token score

The historical README's results table records the following comparison:

> Augmented minus vanilla token error: −0.00843 [−0.01020, −0.00687].

The source is `docs/README.md:45` as read before the root agent's planned canonical-review append. It identifies the original report row as `strict_representation_equivariance_source_bootstrap.csv`, `reflection_minus_vanilla_strict_equivariance`. The interval excludes zero. Since smaller q is better, this is a favorable effect on the **specified identity-channel token discrepancy**.

V5 reports the adverse effect of ordinary pretraining and the inconclusive predictive effect of reflection, but omits this favorable paired token effect. That omission makes the symmetry evidence unnecessarily one-sided. V6 should state the paired augmentation finding alongside the lack of established prediction gain:

“The historical report also records a reduction in strict token error from reflection augmentation, −0.00843 [−0.01020, −0.00687], while its predictive contrast remains inconclusive, +0.00408 [−0.00556, 0.01277] in R². Improved consistency under this supplied action did not establish a corresponding improvement in the movement readout.”

This does not establish intrinsic equivariance under arbitrary latent actions, successful clinical prediction, or superiority to initialization. The historical README says both trained variants remained worse than initialization on this action, and neither met the absolute acceptance criterion. The three-decimal figure JSON supports the direction of those learned-versus-initial comparisons; raw original prediction rows are unavailable for rechecking them now.

A concise mention in the abstract would improve the success/failure balance if space permits. The larger scientific point is a distinction between geometric consistency and target utility, supported by a favorable geometry-only comparison as well as unfavorable readout comparisons.

## 2. Higher-precision values recorded in the historical README

These are five-decimal **documented** values, not recovered machine-precision estimates. Do not label them full precision or pretend to recompute their intervals from the rounded figure JSON.

| Historical comparison | Estimate | 95% source-bootstrap interval |
|---|---:|---|
| Vanilla learned minus initial token error | +0.03055 | [0.01576, 0.04763] |
| Augmented minus vanilla token error | −0.00843 | [−0.01020, −0.00687] |
| Native learned predictive utility | 0.05979 | [−0.02527, 0.12571] |
| Native learned minus initial prediction | −0.01798 | [−0.03851, 0.00248] |
| Augmented minus vanilla prediction | +0.00408 | [−0.00556, 0.01277] |
| Constructed learned predictive utility | 0.04302 | [−0.04356, 0.11283] |
| Constructed learned minus initial prediction | −0.05874 | [−0.09549, −0.01740] |
| Native output antisymmetry error | 0.21548 | [0.19352, 0.23635] |

The five-decimal values agree with the rounded figure summary where both exist. The paired augmentation token-error interval cannot be inferred by subtracting the endpoints of other intervals; the historical directly reported paired comparison is the relevant source.

The README also explicitly states that the protocol was frozen internally after prior development, that no external preregistration is claimed, and that the 0.10 operational margins have no application-level calibration. Those qualifications should carry into the canonical review and final paper.

## 3. Separate historical verification from current verification

The pre-existing `docs/README.md:65` says read-only verification passed for 100,000 prediction rows, 50 jobs, and 16 lanes. That is a retained claim about an earlier manuscript revision. Its expected row count is arithmetically plausible: 625 clips × 5 seeds × 2 variants × 16 lanes = 100,000; outer-fold rotation supplies one held-out prediction per clip per seed/variant, not an additional factor of five in this total.

Current verification independently checked the retained cohort and split hashes, reproduced every split, and recomputed the input-agreement diagnostic. It **could not repeat that earlier original-run evaluation validation**, because the complete report, prediction CSVs, and checkpoints are not present. Both statements can be retained if their times and evidentiary roles are explicit. A newly added note above the old README result/verification sections should label them historical; a canonical review section can then report the current state and link the seven versions.

The README's earlier build instructions, manuscript title, YAML-header description, and eight/four-page PDF claims also describe an earlier manuscript state. They should not silently become validation claims for the seven new Markdown versions or new figures. Preserve useful history but label it accordingly.

The earlier core audit remains correct about the missing original artifacts and current verification limitations. Its statements that the only primary numeric source is the figure JSON are superseded by this addendum. The v1/v3 adversarial reviews remain records of what was known when those versions were reviewed; their chronology should not be retroactively rewritten.

## 4. Correct the displayed reflection-loss denominator

V5 displays an additive epsilon denominator and then says the implementation clamps it. The main equation should match the implementation instead of requiring the next sentence to contradict the displayed expression.

In `laterality_extensions/symmetry_learning.py`, `token_equivariance_loss` returns:

```python
(difference / energy.detach().clamp_min(1e-12)).mean()
```

Let `E_b` be the sum of squared energies of the aligned original and reflected tokens on common-valid support. The displayed denominator should be

\[
\max\{\operatorname{stopgrad}(E_b),10^{-12}\}.
\]

The code takes a per-sequence normalized residual and averages over the batch. Gradients flow through both original and reflected online-encoder outputs in the numerator; the denominator is detached. This is an additional two-pass online-encoder loss. The teacher remains outside that gradient route. V5 correctly labels the only retained efficacy demonstration as synthetic and says the loss does not exclude constant nonzero features.

## 5. Resolve the two meanings of “target” and the parity notation

The manuscript has teacher-feature prediction targets and the downstream signed target y. They enter different objectives. Its pipeline generally distinguishes them, but the parity paragraph leaves `z` undefined at the point of use. Explicitly define `z(x)` as the frozen encoder's original 960-dimensional bilateral summary in the historical parity experiment. Capital `Z` should remain the joint/time token array. The odd feature is a downstream two-pass construction; it does not redefine y and does not add an encoder training loss.

The exact readout relationship is `g(Mx) = -g(x)` for any fixed fitted weights with zero intercept and origin-preserving scaling. The target's `y(Mx) = -y(x)` is a separate algebraic fact about the observed-coordinate measurement. Satisfying the first does not show `g(x)` approximates `y(x)`.

The reflection illustration uses `g(x)=[h(x)-h(Mx)]/2`, whereas the paper's constructed feature uses division by sqrt(2). Both are valid odd projections, but they are different descriptions. Either mark the former as a general illustrative projection or show the actual zero-intercept linear readout on the sqrt(2)-scaled feature. Do not imply two differently normalized fitting procedures would produce identical ridge solutions at the same alpha. Harmonize the diagram's joint-permutation symbol `P` with the text's `S`, or explicitly state that they denote the same operation.

## 6. Remaining v5 precision issues

The central latest-grid readout and source-bootstrap descriptions now have appropriate scope. In particular, the distinction between seed variation and source uncertainty, separate random references for different mask budgets, support-feature and dimensionality changes, and adaptive reuse of the cohort are all explicit. The 960- versus 2,890-dimensional feature descriptions agree with `laterality_extensions/motion_readout.py`.

Several smaller refinements would improve the final version:

- “Coordinate depth scaled relative to crop width” is incomplete. The retained extraction stores `z = landmark.z × crop_width / image_width`; x is normalized by image width and y by image height. Appendix D should give this exact convention, since width/height anisotropy helps explain why the resulting norm is not calibrated 3D speed.
- Define the mask audit's enrichment statistic as a difference between selected and eligible token-motion scores, rather than letting its values look like percentages or physical units. Its small positive values describe the sampling manipulation, not validated biomechanical sensitivity.
- The retrospective narrative should avoid implying that the input-agreement diagnostic preceded the historical baseline training. V5 now says its question ordering is retrospective, which is an appropriate qualification.
- A constant nonzero **shared** token vector is the simple q-degeneracy example. An input-independent representation with different fixed vectors for left and right joints need not have zero q. “Can also have zero” is defensible, but “a shared constant token vector” makes the example exact.
- For the parity lane, say that the recorded learned-minus-initial comparison was adverse under that same wrapper. It does not prove no useful signal exists in either encoder or that the wrapper itself creates the predictive information.
- Main-text results and provenance should identify the original README aggregates and figure JSON as historical evidence, Notebook 12's summaries as historical context, and Notebook 18's saved prediction grid as the evidence newly recomputed in this revision.

## Recommended v6 action

Restore the favorable historical augmentation-to-q effect, fix the reflection-loss equation, define token/feature/measurement notation unambiguously, and update the original-result source note to include the README. Preserve the stronger v5 restrictions on clinical interpretation, coordinate geometry, real-data forecasting, and representation information. In the user's requested canonical `docs/README.md`, place current scores, critiques, suggestions, and remaining submission limitations in a clearly dated section, with earlier report and verification claims labeled as historical.

</details>

<a id="independent-v5-extension"></a>

<details>
<summary>Independent extension review of revision 5</summary>

# Independent adversarial review of paper v5

Reviewed 9 September 2026 against the complete motion/region grid, current implementations, `docs/README.md`, and the rendered PNG previews for the three available figures. The entire manuscript, including Appendices A–D, was inspected. Earlier versions and reviews remain preserved.

The latest numerical results and Appendix A agree with the independent recomputation. The central table correctly distinguishes initial, online and teacher encoders, retains the direct-pose control, and reports source coverage and seed variation. Cross-entropy, the full-input gait-pooled regularizer, the 0.5 gait-derived motion budget, precision and source separation are now represented accurately. The draft is scientifically stronger than v3; its main remaining weaknesses are compression, a small equation mismatch and incomplete historical balance.

## Corrections required before the next version

| Item | Finding | Concrete correction |
|---|---|---|
| Reflection-loss denominator | The displayed equation adds `epsilon_floor` to the detached energy, but the implementation clamps that energy below at `1e-12`. The following sentence acknowledges the difference, leaving the mathematical method and code inconsistent. | Write the denominator as `max(stopgrad(energy), 10^-12)` or define a clamp operator. The distinction is small numerically for ordinary inputs but should be exact in a methodology paper. |
| Figure 1 asset | V5 references `figures/training_pipeline_compact.svg`, which was absent from the figure directory at review time. The expanded diagram exists and renders. | Deliver and inspect the compact SVG before claiming all figure links are ready. This may already be in progress with the figure author. |
| Synthetic RMSE labels | Appendix C headings say “Observed-future feature RMSE” and “Predicted-future feature RMSE,” although the values measure decoded coordinate error. | Use “Coordinate RMSE from observed future features” and “Coordinate RMSE from predicted future features.” Retain the synthetic/four-update/two-test-source scope. |
| Historical consistency effect | V5 reports training's unfavorable token effect and augmentation's uncertain predictive effect, but omits the supported augmentation improvement in token discrepancy. | Preserve the historical README result: augmented minus vanilla token error −0.00843, interval [−0.01020,−0.00687], alongside predictive ΔR² 0.00408, interval [−0.00556,0.01277]. This is a useful positive consistency result and avoids presenting the history as uniform failure. |
| Local evidence scope | §8 refers broadly to “the later prediction grids,” which can imply direct access to Notebook 12's raw grid. | Name the latest motion/region grid specifically. Notebook 12 has retained completion output and numerical summaries here, while `artifacts/comparative_masking` is absent. |

## Historical evidence from the README

The higher-precision historical values in `docs/README.md` are an additional retained documentary source. They can improve the appendix's accuracy without implying a new raw-data reconstruction:

| Historical contrast | Estimate | 95% source-bootstrap interval |
|---|---:|---:|
| Vanilla learned − initial token discrepancy | 0.03055 | [0.01576,0.04763] |
| Augmented − vanilla token discrepancy | −0.00843 | [−0.01020,−0.00687] |
| Native learned predictive utility | 0.05979 | [−0.02527,0.12571] |
| Native learned − initial prediction | −0.01798 | [−0.03851,0.00248] |
| Augmented − vanilla prediction | 0.00408 | [−0.00556,0.01277] |
| Constructed learned predictive utility | 0.04302 | [−0.04356,0.11283] |
| Constructed learned − initial prediction | −0.05874 | [−0.09549,−0.01740] |
| Native output antisymmetry error | 0.21548 | [0.19352,0.23635] |

The historical README describes a completed report and verification performed at the time. The current checkout lacks that full report chain. State both facts directly: the README preserves reported estimates and earlier verification, while this revision independently reconstructs the newest grid only. Do not convert the README's historical verification paragraph into a claim that those 100,000 original predictions were checked again today.

The augmentation finding helps the story. It shows improvement on the specified geometric score without a demonstrated predictive benefit, while trained variants still trail initialization under the token test. The constructed readout has exact output parity and an unfavorable learned-versus-initial predictive contrast. These comparisons reinforce the distinction between imposed consistency and useful representation learning.

## Figure and caption review

The three available PNGs render cleanly at full resolution and contain no obvious text overlap. The expanded pipeline clearly separates training, the EMA teacher, source-separated readout and the synthetic reflection-loss extension. It correctly notes that latest motion experiments use no reflection augmentation and that the signed target does not supervise the encoder. Its density makes it more suitable as the expanded appendix illustration once the compact main figure is present.

The results figure is the most effective main-text illustration. It displays per-seed initial/trained scores and paired mask intervals, while the correspondence panel includes the 375/375 trained versus 33/75 initial comparison and warns that the rows are repeated checks. Its interpretation matches the saved artifacts. One small improvement is to draw a single marker for the seed-independent direct-pose control; five duplicate jittered seed points add no evidence and may suggest independent baseline fits. The main text already explains control reuse.

The reflection figure identifies its skeletons and numerical target values as schematic. It usefully separates anatomical exchange, an odd readout and a constant-feature failure control. Its sentence that “a positive sign identifies greater left-side motion” could be more precise: it identifies a positive average left-minus-right median-speed contrast. Pair contrasts can have mixed signs and cancel, so the sign need not describe every limb or total physical work.

Figure 2's correspondence language should preserve the contextualized-teacher limitation. Teacher vectors come from the full clip; their matched-target advantage could depend on information already visible in context. This diagnostic does not prove recovery of withheld movement. The caption and §4.5 otherwise keep its dependency and scale limitations clear.

## Structural recommendation for an eight-page main paper

The planned v6 change is sensible: the newest directly audited experiment should occupy most of the empirical section. A compact structure can preserve the intellectual sequence without turning notebook chronology into the paper's table of contents.

1. Keep the main motivation, the coordinate-derived target and its reflection law. The clinical motivation can stay in two connected paragraphs covering different mechanisms; move the detailed condition-count table to a cohort appendix.
2. Keep the input/target distinction and source-group split in the main methods, together with compact Figure 1. Move the full tensor-shape table and detailed normalization conventions to Appendix D. State the fixed model-input shape and validity rule once in the main text.
3. Keep the completed JEPA loss, shared anatomical prior and matched controls in the main methods. Move the explicit synthetic reflection-loss equation, auxiliary forward-pass details and historical recipe constants to an appendix. A short main-text sentence and the figure can still explain precisely where the proposed loss would update the encoder.
4. Retain a brief measurement/preparation diagnostic and a short bridge from the historical reflection and target-eligibility tests to the newest question. Put most numerical 00–12 history into one evidence-status appendix table, including the positive augmentation-consistency effect.
5. Keep the latest initial/trained result table, the source-bootstrap comparison and Figure 2 in the main results. The figure already displays mask contrasts, so a second full main-text table of the same three contrasts may be unnecessary; exact estimates can remain in the caption or appendix.
6. Close the empirical arc with predictor/readout disagreement and the most consequential unresolved readout controls. Keep longer alternative-mechanism and synthetic-forecast material in Appendices B and C.

This is an eight-page intent, not a verified page count. The final rendered manuscript still needs checking with the workshop template, readable figure sizes and normal margins.

## Scientific assessment

The strongest supported claim is that the trained encoders have worse accessible laterality information under the tested summaries and ridge procedures, even when the predictor acquires sensitivity to matching clip targets. The new source intervals quantify that conditional disadvantage. Their marginal, fitted-model and exploratory status is clear. The mask contrasts continue to permit modest changes in either direction and should remain inconclusive comparisons.

The health discussion now handles important target limitations: the dataset labels do not identify affected side, shoulder landmarks do not directly measure arm swing, and average signed scores can cancel across pairs or people. The paper also avoids claiming multimodal sensing, calibrated 3D motion, clinical diagnosis or successful real-data forecasting. Those boundaries should survive the compression into v6.

The evidence does not yet distinguish the contribution of missingness, movement amplitude, temporal order, feature covariance and input-preparation mismatch. The proposed low-cost controls are appropriate. The discussion should continue to present explicit reflection training as one subsequently testable hypothesis, rather than the demonstrated remedy for the measured deficit.

</details>

<a id="adversarial-v6"></a>

<details>
<summary>Independent core review of revision 6</summary>

# Final adversarial review of paper_v6.md

Reviewed 9 September 2026. The complete main text and Appendices A–H were read. This review checks scientific claims, historical versus current evidence, notation, coordinate calibration, statistical interpretation, and errors introduced by condensing and assembling the paper. It does not claim a new training run or an independent audit of every model checkpoint.

V6 has a coherent and substantially more credible argument than the starting paper. It now includes the favorable historical reflection-augmentation effect on the specified token discrepancy, while retaining the inconclusive predictive contrast. Its main evidence is the newer, recomputable readout comparison. It explains that input preparation changes the measurement path, that the initial-encoder control retains learned pose extraction and supervised readout fitting, and that source separation does not turn repeated development on these videos into an untouched confirmatory study. The central interpretation is appropriately conditional.

The following specific changes should be made for v7.

## 1. Write the negative token-error value as a difference

Section 4.1 currently writes “augmented-minus-vanilla q = −0.00843.” The discrepancy q is nonnegative, so putting a negative number directly after `q =` is an avoidable mathematical ambiguity. Write

\[
\Delta q=q_{\mathrm{augmented}}-q_{\mathrm{vanilla}}
=-0.00843,
\]

with interval [−0.01020, −0.00687]. The favorable interpretation is a reduction in the specified discrepancy, not a negative discrepancy value. Continue to identify it as a historically recorded paired result whose original prediction/checkpoint chain is absent locally.

## 2. Restore the promised bootstrap seed in Appendix A

Section 4.3 says Appendix A supplies the bootstrap seed, but Appendix A currently does not give it. The material appeared in an earlier version and was lost during condensation.

Add that these revision-time exploratory contrasts use 2,000 paired source resamples, bootstrap seed 812, and percentile 95% intervals. Keep the marginal/no-familywise-adjustment qualification and the fixed-pipeline conditioning. Distinguish this seed from the original reflection protocol's bootstrap seed and from the retained latest-grid mask intervals.

## 3. Describe validation of saved prediction rows accurately

Section 6 says the present review “independently reproduces the latest grid's 125,000 prediction rows.” No encoder inference was rerun to regenerate those rows. The review validated their coverage and used them to recompute the pooled scores and intervals.

Replace that sentence with wording such as: “The present review validates the coverage of 125,000 saved prediction rows, recomputes 200 pooled score rows and the three recorded mask intervals, and checks grid-table hashes and 125 complete update histories.” This clearly reports what was done and remains consistent with the next sentence's restriction on checkpoint verification. Do not imply that CSV hash validation authenticates unavailable original checkpoints.

## 4. Distinguish online tokens, teacher tokens, and downstream features

Appendix F defines `D_b` and `E_b` using online-encoder tokens `Z_theta` for an additional training penalty. Appendix G then defines the original audit as `q = D/E`, although the historical audit used **target/teacher encoder tokens**. The same algebra applies, but the encoder and gradient context differ.

Define the original q directly using teacher tokens `Z_bar_theta`, or state that D and E are recomputed for the frozen target encoder in the historical audit. Appendix F should retain online tokens and its correctly displayed detached, clamped denominator. There is no need to change the implemented loss.

Appendix G's lower-case `z` also remains undefined in the parity paragraph. Define it as the frozen encoder's 960-dimensional bilateral feature summary used in the original parity comparison. Capital Z denotes the joint/time token array; y denotes the observed-coordinate measurement; a fitted readout g predicts y. The identities `y(Mx) = -y(x)` and `g(Mx) = -g(x)` concern different quantities. The latter guarantees a transformation law but not prediction accuracy.

The diagrams use P for the joint permutation whereas the paper uses S. Harmonizing these symbols, or explicitly identifying them, would avoid making readers search for a second operation.

## 5. Correct the synthetic forecasting table's quantity label

Appendix C still labels its columns “Observed-future feature RMSE” and “Predicted-future feature RMSE.” The displayed numbers are errors in decoded future coordinates, not errors in the latent features themselves. Its paragraph below the table already makes this distinction correctly.

Use “Decoded-coordinate RMSE from observed future features” and “Decoded-coordinate RMSE from predicted future features,” with normalized-coordinate units in the caption. Preserve the two-synthetic-test-clip scope, the 24 common endpoints, the four-update budget, and the explicit absence of real-data forecasting evidence. Those limitations make the example useful as a pipeline illustration without granting it substantive forecasting efficacy.

## 6. Add the exact archive coordinate convention to Appendix D

The main text correctly rejects calibrated metric 3D speed, but “depth scaled relative to crop width” remains incomplete. The retained extraction code uses

```text
x = (crop_x0 + landmark.x * crop_width) / image_width
y = (crop_y0 + landmark.y * crop_height) / image_height
z = landmark.z * crop_width / image_width
```

A short appendix sentence or equation should give this convention. The important limitation is not merely a generic absence of calibration: x and y are normalized by different image dimensions, and depth is inferred. Pelvis and body-scale normalization do not recover metric geometry. The rest of Appendix D's timestamp, visibility, interpolation, and pelvis-fallback account matches the inspected original implementation.

## 7. Use “comparison jobs” when counting the 125 encoders

Section 4.2 calls the latest grid “50 paired fold/seed jobs, containing 125 encoders.” Readers may naturally interpret a paired job as two encoders, making the arithmetic seem inconsistent. The grid combines 25 motion jobs with three conditions and 25 region jobs with two conditions.

Write “50 matched comparison jobs across folds and seeds: 25 motion jobs with three conditions and 25 region jobs with two, totaling 125 encoders.” Subsequent pairwise contrasts still legitimately compare each alternative with its family reference. This is a workload clarification, not a change in the inferential unit.

## 8. Keep a few definitions close to the claims they support

The main text describes four inner folds in the original audit and three “in the extensions.” Notebook 08 used a fixed readout penalty, and some later notebooks are demonstrations rather than the same evaluated procedure. Narrow the sentence to the original audit and the later comparative/motion readout protocols, with the notebook map providing the exact scope.

The main text says motion policies increase “target-motion enrichment,” but does not define the statistic. A short appendix definition should explain that it compares the selected-token motion score with the eligible-token score under the audit's implementation. It is neither a percentage improvement nor a calibrated biological motion measurement.

Appendix A says every contrast uses the same “clips and seed,” which could sound like a single-seed analysis. Say “the same clips and each matched seed, then averaging the five seed-specific scores.” The current methods section defines that estimand correctly.

## 9. Appendix assembly and duplication check

Appendix E contains the clinical/annotation census table and a scope paragraph. It does not accidentally duplicate the old full introduction, and its clinical distinctions remain appropriate. The table deliberately uses muscular-dystrophy literature as motivation while refusing to equate the broad myopathic annotation with Duchenne disease.

Appendix H contains the detailed absolute-score table once, along with the notebook map. The main text gives selected summary values and Figure 2, so the appendix table is a reasonable supporting detail rather than an accidental duplicate. Retain it, but a short caption can point to the main result section and avoid repeating the entire initial-control explanation. The zero seed SD for deterministic baseline rows is appropriately interpretable from the reuse paragraph; one explicit phrase that it does not indicate zero source uncertainty would make this foolproof.

Appendices A and G distinguish newly computed learning contrasts from historically retained original reflection and Notebook 12 results. That separation is important and should survive canonical copies of the manuscript. The historical five-decimal values are “recorded precision,” not recovered machine precision.

## 10. Final presentation and evidence limits

The section structure now follows motivation, a known geometric hypothesis, measurement checks, controlled changes, and a next experiment that could resolve the remaining utility question. The paper avoids inferring complete loss of information or complete feature collapse. Its no-equivalence and no-power-guarantee language is appropriate. The proposed use of newly reserved or external sources acknowledges adaptive development on the current cohort.

For the final abstract, a short mention that the historical reflection augmentation improved the specified token score without an established prediction gain would improve the success/failure balance. This is optional if the abstract's emphasis remains the newer completed evidence, but the main-text historical success should remain explicit.

A source-bootstrap interval is conditional on fitted models and on treating video sources as the sampling clusters; unidentified repeated people or related uploads can undermine independence beyond that level. V6 has the correct scope restriction. Do not turn its larger evaluation-row counts into a claim of a larger independent sample.

The current word count is not a verified workshop page count. A future NeurIPS-template build with the final figures and bibliography is still needed before claiming compliance with an eight-page limit. Likewise, inline authoritative links are useful working references, but submission preparation requires a consistent bibliography and verified anonymity. These are submission-preparation tasks, separate from completion of the user's Markdown revision request.

The current governance record remains unresolved. V6 correctly says that local manuscript revision does not establish permission to submit or release derived data. This final review does not alter that status.

## Final revision priority

The essential v7 corrections are the negative-difference notation, missing bootstrap seed, saved-row verification wording, online-versus-teacher token distinction, and decoded-coordinate table headers. The coordinate convention and comparison-job arithmetic are useful precision improvements. After these edits, the paper's empirical interpretation is coherent: the observed advantages and deficits are properties of specified geometry tests and readout procedures on held-out GAVD source videos, with historical and newly recomputed evidence clearly separated.

</details>

<a id="adversarial-v6-extension"></a>

<details>
<summary>Independent extension review of revision 6</summary>

# Independent final extension review of paper v6

Reviewed 9 September 2026. The complete v6 manuscript and appendices were checked against the extension audit, independent numerical recomputation, historical README ledger and current figure previews. All twelve local-link occurrences in the manuscript resolved at the time of review, including the newly delivered compact training figure. This review requires no further model training.

The central science is now internally consistent. The latest absolute scores, source-bootstrap mask intervals and new trained-versus-initial contrasts agree with the saved predictions. Shapes, per-clip target-count pairing, variable-count loss reduction, full-input regularization, current reflection flags and clinical limitations match the implementation. The clamped reflection-loss equation now matches Notebook 09. Historical evidence is visibly separated from the directly audited latest grid, including the positive augmentation effect on token consistency.

## Corrections before v7

| Priority | Location | Remaining issue | Specific change |
|---|---|---|---|
| Necessary | §6, verification paragraph | “Independently reproduces the latest grid's 125,000 prediction rows” implies rerunning inference from checkpoints. This revision verified saved rows and recomputed aggregates; it did not generate new predictions. | Write “independently audits 125,000 saved prediction rows and recomputes 200 pooled score rows and three saved mask intervals.” Keep the separate grid-hash/history checks. |
| Necessary | Figure 2 caption | It refers to “right-hand intervals,” while the actual figure puts the intervals in the bottom panel C. | Refer to “Panel C” or “the lower panel.” Panel B at right is the correspondence count. |
| Necessary | Appendix C table | Inherited column labels call the error “feature RMSE,” although the values are coordinate error after decoding features. | Use “Coordinate RMSE from observed future features” and “Coordinate RMSE from predicted future features.” The parent agent has already identified this correction. |
| Minor | §4.3 / Appendix A | Main text says Appendix A gives the bootstrap seed, but Appendix A currently gives neither seed 812 nor 2,000 resamples explicitly. | Add one sentence identifying 2,000 paired source-video draws, NumPy bootstrap seed 812, and no retraining; link the retained `physworld_evidence_recomputed.json`. |

## Numerical and methodological checks

The latest summary is accurate: initial mean-motion R² 0.222544, trained-teacher range 0.100777–0.114209, initial mean R² 0.070827, and direct pose R² 0.034541. Initial and trained means use 625 clips/93 source videos for each seed. The five teacher-minus-initial intervals in Appendix A match the independent computation and are correctly described as exploratory marginal intervals conditional on fitted models. The historical Appendix G values match the preserved README rather than claiming newly restored raw artifacts.

The target keeps observed timestamps and common bilateral transition support; model inputs are separately interpolated, normalized and resized. The paper does not claim that this preparation preserves original timing or calibrated geometry. It states the full 64×33×3 model input, four-step patching, 16×33 token grid and 96 feature channels without conflating the synthetic 16-frame/16-channel reflection example.

Cross-entropy remains the correct description of the real-data masked objective. The shared gait-pooled VICReg branch receives full geometric views, and the teacher receives no gradients. Laterality enters through anatomical identities and supplied priors; the signed target enters only the readout. The latest grid has no reflection augmentation or explicit reflection penalty. The variable-count pathway correctly matches target counts within each paired clip while allowing different counts across clips.

Source separation and the limits of inner validation are accurately stated. The bootstrap resamples source videos jointly across clips and seeds, while fitted encoders and source partitions remain fixed. Repeated seeds and diagnostic rows do not create more independent recordings. The adaptive development history is visible, so no external preregistration or untouched confirmatory dataset is implied.

The correspondence interpretation is improved: contextualized teacher features may use information already available in the visible context, and their mismatch advantage does not establish recovery of withheld motion. This resolves the main earlier risk of overstating the positive diagnostic. The paper also preserves the possibility that another readout could access useful learned information.

## Figure and link consistency

The compact training figure is now present and readable. It shows the source groups, anatomical token grid, masked student, predictor, EMA teacher and loss, then the held-out readout. Its dotted reflection extension is labeled synthetic. The full-input regularizer is explained in the footnote and main methods. This is an appropriate main illustration; the larger diagram can remain supporting material.

Figure 2's plotted scores and mask intervals agree with the manuscript, and its dependence caveats are visible. The three panels are a useful presentation of the main finding. The seed-independent direct-pose row still displays repeated coincident seed markers; a single marker would be clearer, but the paper's explicit control-reuse statement prevents this from becoming a substantive inference error. The reflection schematic continues to identify its bodies and numbers as illustrative.

All manuscript local links currently resolve. Add the machine-readable interval output to Appendix A for a direct path to exact values. The paper does not yet have a verified final workshop-template page count; main-text word count and a two-figure layout express an eight-page intent only.

## Plain-English flow and final emphasis

The shorter main text now follows a coherent sequence from movement measurement to training intervention and then to predictor/readout disagreement. The earlier quantitative histories support that sequence from the appendices. The abstract remains strategic and avoids procedural numerical detail, while the main text carries the evidence needed to evaluate its claim.

A few terms would benefit from a brief explanation on first use. Define ridge as a linear regression readout with a weight penalty, and define equivariance as a predictable transformation of features when coordinates are transformed. In §4.1, “normalized token discrepancy q” would orient the reader before the historical q result. The methods otherwise explain the tensor and statistical notation adequately.

One dense sentence in §6 could be simplified. Instead of “The supported scope is post-development, within-GAVD performance on excluded source videos,” use “The evaluation holds out source videos within GAVD, which was used throughout development.” This states the same limitation in ordinary language. It does not imply that the held-out sources form a newly collected independent cohort.

The discussion gives a reasonable next experimental sequence using the existing checkpoints before revising training. Explicit reflection training remains one testable hypothesis after measurement and readout checks, rather than a claimed remedy. The paper should preserve this order and avoid adding more speculative mechanisms in the final version.

After the four concrete corrections above, no remaining extension-evidence error requires another experiment before delivering the seventh manuscript. A stronger empirical paper would still benefit from the proposed regularization and summary ablations, independent measurement validation and participant-separated confirmation; those are limitations of the current evidence rather than tasks accomplished by editorial revision.

</details>

<a id="core-evidence-audit"></a>

<details>
<summary>Core evidence audit: notebooks 00–07</summary>

# Independent evidence audit: notebooks 00–07 and the original paper

Audit date: 9 September 2026. This review inspected the original `docs/paper.md`, the current protocol and implementation, the canonical notebooks, retained figure data, and the saved cohort and splits. Paths below are relative to `neurips-laterality/` unless explicitly prefixed by `../`. No encoder was trained and no existing artifact was changed. Applicable writing guidance was read from the workspace `AGENT.md`.

The core experiment is a useful controlled audit of a specific pose representation, but the original manuscript turns several unsuccessful tests into stronger conclusions than the evidence permits. The strongest revised story will distinguish a known coordinate transformation, a learned representation's utility, and a downstream construction that guarantees the desired output transformation. The current workspace also has an important provenance gap: the original run's report and checkpoints are absent, although a small aggregate-number file survives. This limits what can be independently verified during this revision.

## Evidence availability and independent checks

The retained original-run directory is `artifacts/paper/protocol_6f7baefbda07/`. It contains `cohort/`, `splits/`, `inputs/`, and `protocol_snapshot.json`. It has no `report/`, original-run checkpoint directory, or original-run held-out prediction directory. A whole-workspace filename search found no `checkpoint_source_bootstrap.csv`, `strict_representation_equivariance_source_bootstrap.csv`, or `native_output_symmetry_source_bootstrap.csv`. The only canonical `05_aggregate_statistics.ipynb` has no retained code-cell outputs; the same holds for notebooks 00–07. The suite's executed directory currently contains `motion_structured/`, rather than executed copies of the original registered evaluation.

The surviving primary-number source is `docs/figures/v21_figure_numbers.json`. It supplies values rounded to three decimals, without the complete original report metadata or prediction rows. `docs/figures/make_v21_figures.py:19` identifies the now-absent report directory as the source from which those figures were generated. `docs/TUTORIAL.md:152` independently repeats two of those aggregate comparisons in prose, but is not an independent statistical replication.

The manuscript's opening claim that all fifty jobs and hard integrity gates passed is therefore a historical report claim, not a fact that this review could verify from surviving primary artifacts. A revised paper can attribute the original numbers to the retained report summary, disclose the missing prediction/checkpoint provenance, and prioritize the newer results that can be recomputed. It should not say this revision reproduced original confidence intervals or revalidated all original checkpoints.

Read-only verification using the suite's own loaders succeeded:

- `laterality.data.load_cohort` validated the cohort's manifest, array hashes, profile, and content digest.
- `laterality.splitting.load_splits` validated its stored lineage. Rebuilding the split in memory using the frozen parameters reproduced every saved outer and inner partition.
- `laterality_extensions.diagnostics.read_registered_results` returned `registered report unavailable`, with no result rows.
- `laterality.governance.submission_readiness` returned `ready: false` and listed all three reviews as unresolved.

The retained lineage is:

| Item | Digest |
|---|---|
| Protocol | `6f7baefbda07c8e6899bc5fbb82b4651995b4b0089e0a1d1b36ae04480657088` |
| Effective context | `abc7a3d26a60fd479641b2ca76f7caab16bba127aeb05f5ef2be6eb56fd9d143` |
| Cohort | `bc23447824d2bc2bbe62dfdc7df5da8db7d1e73291d1675f2556597eeeefe2e3` |

Local configuration files establish a versioned computational contract. They do not, by themselves, establish an independently timestamped public preregistration. Use “frozen post-development protocol” unless the authors can supply a dated registration record demonstrating the stronger chronology. In particular, “before any result was seen” should not conceal the earlier transductive development or the later exploratory iterations on the same sources.

## Cohort, units, and the actual shape of the data

The inventory contract records 666 annotation files, 103 original source videos, and 642 available pose archives. Quality control accepts 625 sequences from 93 sources. Three finite targets were excluded for insufficient coverage: the saved target-contract record has 628 finite targets checked, 625 retained sequences, zero maximum mirror-target residual, and zero maximum invalid-coordinate-sentinel residual.

| Dataset annotation | Sources | Sequences |
|---|---:|---:|
| Cerebral palsy | 9 | 58 |
| Myopathic | 28 | 183 |
| Normal | 29 | 270 |
| Parkinson's | 9 | 39 |
| Stroke | 18 | 75 |
| Total | 93 | 625 |

These are observed source/video and annotation counts, not independent patients or confirmed diagnoses. Sources contribute between 1 and 60 sequences, with median 4. The smallest condition groups have only nine source videos. This supports motivation by clinical gait literature but does not support discovered condition-specific asymmetry, treatment response, or diagnostic performance.

The tensor description should be explicit because the user asks whether training preserves data shape. Raw archives contain `T × 33 × 4`: three estimated coordinates plus visibility, with retained raw sequence lengths from 24 to 1,576 frames. After preprocessing, saved arrays have these exact shapes:

| Artifact | Shape | Meaning |
|---|---|---|
| `model_xyz` | `625 × 64 × 33 × 3` | Prepared model coordinates |
| `model_valid` | `625 × 64 × 33` | Separate observation-validity mask |
| `pair_contrasts` | `625 × 5` | Original-target components |
| `missingness` | `625 × 10` | Original-lane bilateral coverage summaries |
| Encoder tokens per batch | `B × 16 × 33 × 96` | Four prepared time steps per joint token |
| Main probe features per batch | `B × 960` | Five pairs, each with 96-channel differences and sums |

No honest revision should claim that preprocessing preserves the exact original trajectory or time grid. It preserves the 33-landmark schema and a consistent tensor layout, while temporal interpolation and resizing change observations. Within training, a target mask retains the allocated joint/time token axes; selected target coordinate embeddings are replaced by zero before adding position embeddings. Invalid tokens are masked in attention and zeroed at the output. The predictor replaces target-position context vectors with learned mask tokens. See `laterality/model.py:44`, `:51`, `:110`, and `:165`.

Both “gait-focused” and more broadly masked variants can receive all 33 joints. In the original run, twelve joints are eligible prediction targets and are pooled by the VICReg regularizer: shoulders, hips, knees, ankles, heels, and foot-index landmarks on both sides. The five laterality pairs omit hips. It would be wrong to illustrate the encoder input as only ten or twelve landmarks, or to imply the broad-target comparison removed every anatomical prior.

## Coordinates and target definition

The target uses pelvis-centered, body-scale-normalized coordinates only where landmarks were originally observed. `laterality/geometry.py:171` explicitly separates that lane from model interpolation. Missing target coordinates remain invalid; target pelvis centering has no fallback. Input preprocessing may interpolate a gap of at most four missing samples and use a pelvis fallback, then linearly resize by relative sample index to 64 steps. Validity is resized separately and thresholded at 0.999. The target is never recalculated on that imputed/resized input in the registered evaluation.

Target frame times are `(frame_numbers - first_frame_number) / fps`. For each bilateral pair, both landmarks must be valid at both endpoints of a transition and the elapsed time must be positive. The code divides each displacement norm by that elapsed time, takes each side's median speed on the shared support, then forms `(left - right) / (left + right + 1e-8)`. Every pair must supply at least eight transitions and all five pairs must be usable. The final target averages the five contrasts. See `laterality/geometry.py:217` and `laterality/data.py:453`.

The target's physical units need careful explanation. Retained extraction code in `../work/nb_extracts/02_extract_and_watch_skeletons.md:512` uses MediaPipe `pose_landmarks`, not calibrated world landmarks:

```text
x = (crop_x0 + landmark.x * crop_width) / image_width
y = (crop_y0 + landmark.y * crop_height) / image_height
z = landmark.z * crop_width / image_width
```

Thus the coordinate norm combines image-relative horizontal and vertical quantities with inferred depth. Image aspect ratio affects the relative x/y scaling; pelvis normalization does not calibrate a metric 3D camera frame. “Coordinate-derived motion contrast” is accurate; “physical gait speed,” “metric 3D biomechanics,” and “real-world dynamics” would overstate this measurement.

The target's mirror sign reversal remains an exact algebraic property of this representation. Horizontal sign inversion preserves coordinate norms and exchanging paired landmark identities exchanges left/right speed summaries. That property holds for symmetric and asymmetric trajectories alike; it does not require a patient's actual gait to be symmetric, or a mirrored recording to have equal prevalence in nature.

Fresh calculation from `cohort/manifest.csv` gives target mean −0.006074563028935445, sample standard deviation 0.059148206385182124, minimum −0.1948212340422903, and maximum 0.2146815707124385. These are sequence-descriptive values. The saved components reconstruct the target to maximum absolute disagreement `9.93129189996722e-17`. That is a self-consistency check, not evidence of what a learned encoder preserves.

## Split and training audit

| Outer fold | Train sources | Test sources | Train sequences | Test sequences |
|---:|---:|---:|---:|---:|
| 0 | 74 | 19 | 436 | 189 |
| 1 | 74 | 19 | 443 | 182 |
| 2 | 74 | 19 | 553 | 72 |
| 3 | 75 | 18 | 548 | 77 |
| 4 | 75 | 18 | 520 | 105 |

The split is stratified at the source-table level by dataset annotation. Every source is an outer test source exactly once and appears in the other four training folds. All clips, reflections, and other augmentations inherit the source assignment. A dataset annotation is used for split balance and one nuisance lane; it does not enter the self-supervised objective. Independent person identifiers are unavailable, so shared people or related material across source IDs remain possible.

Each outer training set has four inner readout folds: 55–57 fitting sources and 18–19 validation sources, with 266–450 and 80–197 sequences, respectively. `laterality/evaluation.py:534` calls `_fit_readout` separately inside every inner fitting partition. Imputation, weighted feature centering, and scaling are fitted there; candidate ridge penalties are scored on the corresponding inner validation sources. The selected penalty is refitted on all outer training sources before outer testing. This implementation is more precise than the paper's blanket statement that scaling is fitted on outer training data: inner selection itself also avoids leaking validation targets or feature-scaling statistics into its fitted readout.

The encoder is trained on all outer training sources, including those subsequently assigned to inner readout validation. That is the stated design for tuning a readout of a fixed, label-free representation; it is not fully nested encoder selection. The outer test sources remain excluded from both encoder updates and supervised fitting. Per-sequence pelvis/body scaling uses the observed sequence itself and no cohort-wide learned statistics, which is appropriate for full-clip representation evaluation but should not be transplanted into a past-only forecasting claim.

The original model uses a four-layer, 96-channel transformer encoder, two-layer predictor, and four heads. It starts a new model for each outer fold, seed 42–46, and variant. Its self-supervised loss is a centered, temperature-scaled cross-entropy on masked teacher features plus a VICReg term, rather than a generic unspecified latent mean-square loss. The teacher is updated by an exponential moving average of student weights. `laterality/training.py:363` provides the actual training loop, and `laterality/model.py:308` provides the loss.

Each epoch samples one sequence from every training source, then adds source-uniform draws to reach 80 samples, or four batches of 20. Three hundred epochs therefore mean 1,200 updates per encoder, with 24,000 sample draws; they do not mean 300 complete passes through all 436–553 available training sequences. Vanilla uses no reflections; reflection augmentation uses probability 0.5, with a separate RNG stream. Source sampling, masking, initialization, and update budgets are matched across variants. The registered source and implementation checks are strong design safeguards, though absent original checkpoints prevent verifying that every historical job complied.

“Mask fraction 0.6” is also easy to misstate. `uniform_authorized_mask` computes a common hidden count from 60% of the minimum number of eligible valid tokens in the current batch, capped to leave at least one eligible token visible. It does not mask 60% of all `16 × 33 = 528` allocated tokens. Bilateral semantics enter this recipe through the supplied landmark schema, eligible target joints, regularizer pooling, optional anatomical reflection, and later probe construction. The audit's signed target never supplies an encoder training label. An explicit symmetry penalty belongs to the later experimental extension and must not be illustrated as part of the original training objective.

## Exact estimand and inferential limits

The primary predictive quantity is **not** the average of 25 fold-specific R² scores. For a given seed, the implementation joins each sequence's prediction from its held-out outer fold, obtaining a full out-of-fold pass over all 625 sequences. It computes source-balanced R² on that complete pass and then averages the five seed-specific scores. Each sequence receives weight `1 / clips_in_its_source`, so each source has equal total weight. The comparison between trained and paired initial representations is computed with the same source/sequence/seed alignment. See `laterality/reporting.py:409`–`:613`.

The source bootstrap resamples 93 sources with replacement 2,000 times, carrying all corresponding sequences and registered seed predictions together. It forms each seed's metric within the resample before averaging seeds; it does not average predictions first. The alternative mean-prediction ensemble is separately stored. Confidence intervals are percentile source-resampling intervals conditional on the fitted cross-validation pipeline. They do not incorporate new optimization seeds, new split allocations, encoder refitting, model selection over the many extensions, unmeasured repeated people, or arbitrary new acquisition domains.

The original manuscript inaccurately treats the high-coverage result as though it were another primary-estimand check. `laterality/reporting.py:1180`–`:1197` builds high-coverage metrics from the **seed-mean prediction ensemble**. The high-coverage subset itself is independently verified as 600 sequences from 91 sources. Its predictive result should either be explicitly labeled ensemble sensitivity or omitted until a matching primary-estimand calculation can be recovered.

Native output error first squares `(prediction + mirrored_prediction) / (2 × outer-training target SD)` within sequence and seed, then aggregates within source and across sources/seeds before taking the square root. It is not a signed average that can cancel between seeds. The strict token statistic averages sequence-level ratios with equal total source weight. Different error constructions answer different questions and should have separate axis labels.

The declared positive-claim gates are conjunctions of useful absolute prediction, favorable comparison with initialization, and a small transformation error. Failure of these gates does not prove that the corresponding population quantity is exactly zero. The protocol has no demonstrated prospective power analysis or an equivalence margin for learned-versus-initial predictive performance. “Fully powered null,” “no effect,” and “augmentation does not help” are therefore inappropriate summaries. Prefer “the interval did not establish a predictive gain under this protocol.” The registered augmentation interval is compatible with small benefits or harms.

## Primary numbers that survive

All entries in the next table are retained three-decimal summaries from `docs/figures/v21_figure_numbers.json`, not freshly recomputed confidence intervals.

| Comparison | Estimate | 95% interval | Interpretation |
|---|---:|---|---|
| Learned primary R² | 0.060 | [−0.025, 0.126] | Positive absolute utility gate unmet |
| Learned minus initial primary R² | −0.018 | [−0.039, 0.002] | Positive training gain unestablished |
| Reflection augmented minus vanilla primary R² | 0.004 | [−0.006, 0.013] | Neither superiority nor equivalence established |
| Learned constructed odd/zero R² | 0.043 | [−0.044, 0.113] | Exact parity does not establish useful prediction |
| Learned minus initial constructed odd/zero R² | −0.059 | [−0.095, −0.017] | Adverse conditional training contrast in this lane |
| Learned minus initial odd/free R² | −0.071 | [−0.108, −0.033] | Adverse conditional training contrast in this lane |
| Learned minus initial even/free R² | 0.024 | [0.009, 0.042] | Positive exploratory contrast, not a paradox |
| Native output antisymmetry error | 0.215 | [0.194, 0.236] | Above the 0.1 registered margin |
| Strict token error, learned vanilla | 0.114 | [0.095, 0.138] | Upper-bound gate unmet; interval straddles 0.1 |
| Strict token error, initial | 0.083 | [0.075, 0.094] | Lower error under this specified test |
| Learned minus initial strict token error, vanilla | 0.031 | [0.016, 0.048] | Increase under the identity-channel test |
| Learned minus initial strict token error, reflection augmented | 0.022 | [0.008, 0.039] | Increase under the same test |

Three-decimal reporting matters: rounding the primary R² contrast's upper bound 0.002 to 0.00 or the token-error lower bound 0.095 to 0.10 can obscure whether a threshold is crossed. The learned q interval includes values below 0.1; the honest statement is failure to demonstrate an upper bound below the threshold, not conclusive evidence that the population q exceeds 0.1.

## What the strict token test does and does not establish

`laterality/symmetry.py:91` compares reflected teacher tokens to anatomically permuted original tokens with the **identity transformation in the 96 feature channels**. It allows no fitted rotation, channel permutation, sign transformation, or centering. For common-valid tokens it divides squared residual energy by the total energy of both token arrays. This is a clear, reproducible test of one chosen representation action, not a general test of every possible reflection-equivariant encoding.

A representation could store horizontal vectors in channels that should change sign on reflection, or store an action expressed in another channel basis, and fail this identity-channel check while carrying useful geometry. The revised manuscript should use “strict identity-channel joint-swap error” near every important interpretation, rather than burying the restriction in limitations.

The mathematical ratio lies between 0 and 2 when the denominator is positive. A value around 1 indicates approximately uncorrelated vectors under the particular raw-energy comparison; “unrelated representations equal one” is not universal in the presence of shared offsets. Rejecting zero-energy tensors only excludes an all-zero degeneracy. A constant nonzero, mirror-insensitive encoding can obtain zero error while retaining no useful motion. Learned positional embeddings, common token offsets, and representation sensitivity matter, so a low initial q does not establish a random model's understanding of human geometry. Pair q with held-out utility, feature-variation or rank diagnostics, and a clear account of its fixed action.

The constructed odd readout is a valid algebraic control. With `z_minus = (z(x) - z(Mx))/sqrt(2)`, no feature centering, and zero intercept, a linear map is odd. The guarantee holds for arbitrary encoders, including random or constant ones. Thus “the only route” in the abstract is an unjustified universal claim. Poor trained-minus-initial performance also does not show that every useful component comes solely from the wrapper: it compares two complete representation/probe pipelines and cannot isolate all causes of their absolute prediction.

The positive even-feature contrast is not “the opposite of what laterality requires” in the strong sense asserted by the original paper. An exactly even predictor cannot represent an exactly odd target on a mirror-balanced population except trivially, but the observed dataset need not be mirror-balanced. Even features may correlate with signed labels through source distribution or measured/unmeasured nuisance structure. This is a distributional limitation or control result, not evidence that self-supervision inherently learns the wrong physical quantity.

## Notebook 07: an independently recomputed diagnostic worth retaining

The read-only `reconstruct_processed_target` diagnostic applies the same formula to prepared model inputs, with uniform relative-time steps. It can reveal disagreement introduced by the complete measurement path, but it neither identifies a particular preprocessing operation as the cause nor estimates a decoder's best achievable performance.

Fresh calculation using the retained cohort and the existing helper gives:

| Quantity | Value |
|---|---:|
| Original input | 625 sequences / 93 sources |
| Finite original/recomputed overlap | 623 sequences / 92 sources |
| Excluded from overlap | 2 sequences |
| Direct calculation-agreement R² | 0.21818988696299113 |
| Source-balanced correlation | 0.6518182554344372 |
| Source-balanced mean absolute difference | 0.04089209804204162 |
| Original target SD on weighted overlap | 0.05984303514794512 |
| Source-balanced sign agreement | 0.7041029622551362 |
| Maximum absolute discrepancy | 0.17660128616160597 |

The 70.4% quantity weights sequences equally within each source and sources equally; it is not a percentage of source medians with matching sign. This illustration is particularly valuable because it exposes a real limitation rather than assuming that normalization and resizing preserve a motion target. Label it a newly recomputed descriptive diagnostic, not a new learned result or an information ceiling. No uncertainty interval is supplied by the existing helper.

Notebook 07 also provides a clean constructed counterexample: left positions `(0,1,0,1)` and right positions `(0,0,1,1)` have the same average position but different median speeds. Swapping the trajectories reverses their contrast while preserving the means. This demonstrates a limitation of raw time averaging; it does not prove that averaged contextual encoder features lose temporal information, because an encoder can encode motion before averaging. Reversing time preserves a median speed contrast, so successful prediction of this particular target alone cannot demonstrate knowledge of temporal direction or causal dynamics.

## Prioritized revision requests

1. Replace the universal title and conclusion with a conditional claim about geometry-aware evaluation of skeleton predictive representations. A stronger Physical World AI connection explains why known coordinate actions provide controlled tests of articulated-body representations, while acknowledging the absence of calibrated 3D dynamics, interventions, external participant validation, and multimodal sensing in this experiment.
2. Disclose original-run artifact availability and distinguish historically retained aggregates from results recomputed from prediction rows. Restore original report/checkpoint lineage before submission if available; do not fabricate full precision or infer absent confidence intervals from differences of marginal bounds.
3. State the hypothesis as whether training improves source-balanced laterality prediction and transformation consistency over the paired initialization, with useful absolute prediction required. Remove “fully powered null” and equivalence language.
4. Put preprocessing and probe limitations into the main methods/results. The newly recomputed input-agreement diagnostic and the transparent toy temporal counterexample are suitable failure illustrations, with their evidentiary limits in the captions.
5. Show the true training graph: original coordinates and validity branch into the target measurement lane and the prepared input lane; all 33 joints enter the student/teacher; anatomical selection controls eligible hidden targets; optional reflection enters before training views; the audit label is used only by the fitted readout after encoder training. Explicit symmetry-loss experiments need a separately marked branch.
6. Correct the estimand description to five full out-of-fold seed passes, distinguish the ensemble sensitivity analysis, and explain source bootstrap conditioning. Keep source counts visible when citing large evaluation-row totals in later notebooks.
7. Narrow direct q claims everywhere to the joint-permutation/identity-channel action and explain why neither parity nor low q alone proves information retention. Remove the inference that the target-component oracle locates the failure inside the encoder.
8. Use authoritative gait literature for how conditions can alter bilateral coordination, but do not derive clinical condition differences from the dataset annotations or these signed targets. A population can have substantial absolute asymmetry and near-zero average signed asymmetry because affected sides differ.
9. Retain the unresolved governance status. This audit does not justify public submission, pose release, embeddings release, or external data use; it does support local analysis and manuscript revision within the user's request.

The core methods have strong source-separation and paired-control structure. The paper will be more credible if it makes these concrete safeguards legible while treating the failed and incomplete tests as evidence that refines the next question, rather than as proof that a general predictive-learning approach cannot encode symmetry.

</details>

<a id="extension-evidence-audit"></a>

<details>
<summary>Extension evidence audit: notebooks 08–18</summary>

# Evidence audit of notebooks 08–18

Reviewed 9 September 2026. This audit inspects retained notebook outputs, implementations, the tutorial, and saved experiments. It does not retrain models or alter existing research artifacts. Paths below are relative to `neurips-laterality` unless otherwise stated.

The strongest extension finding is a discrepancy between successful optimization and useful movement representation: every trained arm learns clip correspondence under a held-out predictor diagnostic, while its motion-sensitive frozen readout performs worse than the matched initial encoder. Motion weighting changes the target distribution and connected regions remove measured local cues, but neither establishes a laterality-readout benefit. This supports a paper about testing whether geometric predictive learning preserves a physical observable. It does not support a claim of improved clinical diagnosis, successful gait forecasting, or successful explicit symmetry training on GAVD.

## What was independently verified

The latest complete grid is:

`artifacts/motion_structured/grids/292443b0fab5339f5da7ca566a85d6172ffc5b64abe5febf2546681a0152ff57`

Its manifest marks the grid complete. The audit independently verified all ten grid-table SHA-256 hashes, 125,000 prediction records, their complete pairing and coverage, the 200 seed-specific pooled R²/MAE rows, and all 125 training histories with exactly steps 1–1,200. The largest discrepancy between recomputed and saved scores was `9.194034422677078e-17` for R² and `9.71445146547012e-17` for MAE. Each experiment/arm/representation/seed contains 625 clips from 93 source videos. A source appears in only one outer test fold. The 50 paired fold/seed jobs contain 125 encoder arms and 150,000 arm-specific optimizer updates.

The companion [recomputation script](verify_physworld_evidence.py) performs these checks without importing the experimental metric code, prints its numerical audit as JSON, and writes no experimental files. It also independently reproduces the three existing source-bootstrap intervals. It checks grid-table hashes and training-history completeness; it does not claim to have independently rehashed all 125 model checkpoints and every large training manifest.

The exact output and its analysis-status annotations are retained in [physworld_evidence_recomputed.json](physworld_evidence_recomputed.json).

The audit additionally computes new paired trained-versus-initial intervals from the retained predictions. These are explicitly identified below as an exploratory analysis added on 9 September, rather than a result present in the original registered study.

## Evidence status by notebook

| Notebook(s) | Retained evidence and scope | Appropriate use in the paper |
|---|---|---|
| 08 | Current source notebook has no retained code outputs. The dated real-data result survives in `docs/figures/tutorial_masking_summary.json`: 25 fold/seed jobs, 50 encoders, fixed ridge penalty 1. | Historical masking comparison, clearly separated from the later implementations. Its raw research-masking artifacts are absent from this local workspace. |
| 09 | Symmetry-loss implementation and a default synthetic teaching experiment. The tutorial describes eight updates on generated data; the current source notebook has no inline code outputs. No retained real-data symmetry-loss comparison was found. | Explain the proposed geometric objective and its software checks; do not present a clinical or real-data training benefit. |
| 10 | Past-only forecasting implementation and documented synthetic demonstrations; current source has no inline code outputs. | Explain the information boundary and controls. Its evaluation regresses coordinates from past features, rather than decoding the predictor's forecast latent. |
| 11 | Executed generated examples for mask construction, equal counts, anatomical connectivity, motion selection and reflection of validity/masks. | Explain controlled interventions. Its generated selection counts are not GAVD performance evidence. |
| 12 | Inline output records completion of 25 real GAVD jobs and 50 encoders, alongside clearly labeled synthetic examples. `docs/figures/tutorial_comparative_masking_summary.json` retains the result snapshot. | A completed earlier real-data comparison of gait versus all-landmark target eligibility. Raw `artifacts/comparative_masking` is absent locally, so this audit does not claim a fresh reconstruction of its raw predictions. |
| 13 | Inline output is a generated three-update demonstration, including a positive control and raw-removal example. Larger real predictor and mild-corruption findings in the tutorial come from Notebook 12's saved evaluations. | Keep the synthetic software checks distinct from those empirical summaries. |
| 14 | Executed synthetic decoder comparison: eight generated clips/eight generated sources, six training/two test, four updates per arm, one seed. The real-data plan is displayed with loading/training disabled. | A failed short-run synthetic forecasting demonstration and a disciplined next experiment. No real-data forecasting gain. |
| 15 | Current source has no inline outputs, but `executed/motion_structured/gavd_5yc3ve5h/15_motion_weighted_masking.ipynb` retains the real mask audit. | Positive evidence that the sampling intervention changes motion enrichment. |
| 16 | Inline real GAVD coverage audit: 75,000 repeated mask draws over the same 625 clips/93 videos. | Positive evidence that geometries remove different local cues. Whole trajectories and interior completion have no trained downstream comparison. |
| 17–18 | The source 17 lacks inline code outputs, but 18 and the complete saved grid establish all 50 jobs/125 encoders. | The central empirical comparison. Use its raw predictions and exact scope rather than inferring completeness from workload declarations. |

There is a documentation freshness conflict: `docs/MOTION_STRUCTURED_MASKING.md` still says that new full-grid checkpoints are absent, whereas the dated 8 September notebook interpretation, Notebook 18, and actual saved grid establish completion. The manuscript should use the completed artifacts as the authority and cite dates when distinguishing earlier experimental plans.

## What the latest encoder was actually trained to do

All 33 anatomical landmarks remain in the prepared input. A clip has shape `[64,33,3]`; validity has shape `[64,33]`. Four consecutive prepared steps of one landmark form a token, giving 16 time blocks × 33 landmarks, with a 96-dimensional vector per token. A token is eligible only if all four required observations are valid. The online encoder zeros the hidden patch embeddings while retaining anatomical/time positions, so masking preserves the token array and sample identity. The teacher receives the full prepared clip. Naturally absent observations are excluded from targets and attention support.

The use of a fixed input tensor does not mean preprocessing preserves the original shape, timing or every geometric quantity. Temporal resizing changes the original number of observations; interpolation and normalization can change displacement-based measurements. The safe claim is that masking preserves the prepared joint/time indexing, bilateral identities and validity contract. The manuscript should explicitly distinguish the original target-calculation lane from the normalized, resized model-input lane.

The latest recipe uses an encoder depth of four, predictor depth of two, four attention heads, batch size 20, 1,200 updates, AdamW learning rate 0.001, weight decay 0.05, EMA momentum 0.999, and gradient clipping at 1.0. It ran on CUDA with BF16 computations and FP32 weights/loss reductions/frozen evaluation. The objective is centered teacher-distribution cross-entropy plus `0.05 × VICReg`, not a plain latent MSE. VICReg here combines agreement between geometric views, a feature-variance penalty and a covariance penalty. The geometric views use rotations within ±8 degrees and small translations. See `laterality/model.py`, `laterality_extensions/motion_structured_training.py` and the first motion-job manifest.

The trained extension is anatomy informed in three concrete places. Learned landmark embeddings distinguish left from right positions. The original gait-mask conditions select targets from shoulders, hips, knees, ankles, heels and foot tips, while the all-landmark conditions change target eligibility. The shared auxiliary regularizer pools the same twelve gait landmarks even when prediction targets can come from all 33. The subsequent readout explicitly combines five bilateral pairs—shoulders, knees, ankles, heels and foot tips—using sums and signed differences; hips are excluded from that readout. The five pairs and twelve-landmark set should not be described as neurologically validated.

Explicit reflection training is a separate intervention. In the latest motion/region grid, saved `reflection_probability=0` and `symmetry_weight=0`, and `train_mask_study` raises an error if either is nonzero. Notebook 09's proposed loss aligns anatomical token positions under reflection, without forcing the whole-body feature to be identical. A training-pipeline figure should use a distinct proposed/synthetic inset for this loss, rather than attach it to the completed GAVD training path. Side-sensitive evaluation is also not equivalent to imposing a reflection loss on the encoder.

Matched experimental arms share initial model/projector weights, source schedules, geometric views and update counts. Masks use separate controlled random streams. Regions determine a per-clip feasible count and receive a random reference at the same count; a dense reduction averages hidden-target losses within each clip and then across clips, preserving sample identities when counts vary. Both compared arms retain the full-input regularizer, so the masked predictor's input isolation should not be overstated as a prohibition on the encoder ever seeing full training clips.

## Held-out evaluation and its limits

The same five outer source partitions are reused across seeds 42–46. Their training/test clip counts are 436/189, 443/182, 553/72, 548/77 and 520/105. They have 74/19, 74/19, 74/19, 75/18 and 75/18 training/test source videos respectively. All clips from each outer test video are excluded from encoder training and readout fitting. Source identity is a recording-level grouping; there is no verified cross-video participant identity, so person-independent generalization is unestablished.

Readouts use training-source-only imputation, scaling and ridge regression. Three inner groups choose among penalties `[0.01,0.1,1,10,100,1000,10000]` by pooled source-balanced validation MSE, with smaller penalties breaking ties. Those groups tune the readout only: the encoder has already seen all outer-training sources. Choosing a whole pretraining recipe would require keeping the pilot validation sources outside that encoder's training too.

For each seed, all five outer-test folds are pooled. A clip from source s receives weight proportional to `1/n_s`, where `n_s` is the number of evaluated clips from that source. R² is computed against the weighted mean of the pooled evaluation targets, and MAE uses the same weights. The paper then averages the five seed scores. Averaging fold R² values or averaging predictions across seeds before scoring would define different estimands. The training-source-mean control is deployable; the pooled evaluation mean in the R² denominator is a metric reference.

The source bootstrap resamples 93 complete source videos with replacement, moving their clips and all paired seed predictions together. It conditions on the fitted models and existing source partitions. It excludes retraining uncertainty, uncertainty from selection after repeated inspection of this cohort, and unknown relationships between participants across videos. Seeds are repetitions of stochastic fitting on the same recordings; they do not create five independent cohorts.

## Latest reproducible movement-readout results

All rows below use 625 clips, 93 source videos and five seeds. R² and MAE are means of pooled per-seed scores; the spread shown for R² is the standard deviation across seeds.

| Representation | Mean R² ± seed SD | Mean MAE |
|---|---:|---:|
| Training-source mean | −0.010609 ± 0 | 0.0461723 |
| Direct pose summary | 0.034541 ± 0 | 0.0449596 |
| Initial encoder, mean summary | 0.070827 ± 0.018574 | 0.0443442 |
| Initial encoder, mean-motion summary | 0.222544 ± 0.026808 | 0.0415461 |
| Motion uniform teacher, mean-motion | 0.113725 ± 0.011007 | 0.0436583 |
| MAMP-convention teacher, mean-motion | 0.112890 ± 0.030590 | 0.0437511 |
| Robust-motion teacher, mean-motion | 0.114209 ± 0.021036 | 0.0435951 |
| Region uniform teacher, mean-motion | 0.109446 ± 0.008849 | 0.0440373 |
| Connected-region teacher, mean-motion | 0.100777 ± 0.009239 | 0.0437557 |

The mean-motion summary adds temporal standard deviations, mean absolute feature increments and observation-support fractions to bilateral sums/differences of means. At width 96, the mean summary has 960 features; mean-motion has 2,890, including ten support fractions. The initial encoder improves by 0.151716 R² when these components are added. Their combined addition improves trained teachers by only 0.027022–0.046815. The experiment does not isolate ordered motion from amplitude or missingness. In particular, standard deviation does not encode temporal order, and support fractions may encode recording/pose-estimation properties.

Every trained teacher has lower R² and higher MAE than its matched initial encoder under the same mean-motion summary in every seed. Final online mean-motion scores are also lower: 0.080493, 0.080923, 0.073961, 0.065700 and 0.063632 in the corresponding five-arm order. The disadvantage therefore cannot be attributed solely to choosing the EMA teacher instead of the online encoder.

The three existing paired mask contrasts are:

| Final teacher, mean-motion contrast | Mean ΔR² | 95% source-bootstrap interval |
|---|---:|---:|
| MAMP convention − motion uniform | −0.000834 | [−0.020372, 0.015296] |
| Robust motion − motion uniform | 0.000485 | [−0.015497, 0.015173] |
| Connected region − region uniform | −0.008669 | [−0.033290, 0.017666] |

These intervals include zero and permit effects in either direction. They support neither superiority nor equivalence. Region and motion references differ in mask budget and must remain separate.

The new exploratory analysis applies the same 2,000-resample source bootstrap, seed 812, directly to trained teacher versus matched initial mean-motion predictions:

| Teacher arm − matched initial | Mean ΔR² | 95% interval | Mean ΔMAE | 95% interval |
|---|---:|---:|---:|---:|
| Motion uniform | −0.108819 | [−0.170093, −0.041219] | 0.0021122 | [0.0004937, 0.0038781] |
| MAMP convention | −0.109653 | [−0.165071, −0.051351] | 0.0022050 | [0.0007061, 0.0038627] |
| Robust motion | −0.108334 | [−0.170640, −0.042307] | 0.0020490 | [0.0003668, 0.0039266] |
| Region uniform | −0.113098 | [−0.166192, −0.059623] | 0.0024912 | [0.0010502, 0.0040473] |
| Connected region | −0.121766 | [−0.182787, −0.055281] | 0.0022096 | [0.0005845, 0.0040972] |

All intervals exclude zero in the unfavorable direction, conditional on the fitted models. They are exploratory, marginal intervals without a familywise multiplicity correction. The five contrasts share controls and recordings. This direct uncertainty calculation strengthens the specific observed trained-versus-initial comparison; it does not establish that JEPA necessarily destroys movement information or that no other decoder could access it.

## What succeeded, and what remains unresolved

The archived Notebook 15 audit shows equal average target counts of 80.771 across motion arms, approximately 17% of valid all-landmark tokens. The configured 0.5 fraction derives from the smaller gait-landmark budget. Mean target-minus-eligible motion is −0.000105 for uniform, 0.01749 for MAMP convention and 0.03784 for the robust mixture. These are source-balanced summaries of repeated mask draws. The common enrichment diagnostic resembles the robust sampler's own score; it is not an independent measure of clinical movement or tracking quality.

Notebook 16 shows that connected regions reduce the fraction of targets with immediate two-sided temporal neighbors from 0.699 to zero and the fraction with a visible anatomical graph neighbor from 0.988 to 0.512. Regions hide about 9.9% of valid tokens. Whole trajectories hide about 9.5% and interior gaps about 25.3%; these two mask families have coverage audits but no trained comparison. Zero immediate brackets does not mean all temporal context is absent. The 75,000 draws are repeated uses of 625 clips, not independent observations.

All five arms substantially reduce their training objective. Mean masked prediction loss at steps 1 and 1,200 is 15.0778→0.8493 for motion uniform, 15.0418→0.8355 for MAMP, 15.0239→0.8644 for robust motion, 15.2323→0.8393 for region uniform, and 15.1253→0.9479 for connected regions. Each arm has its own moving teacher and feature coordinates, so comparing these losses does not rank semantic usefulness.

The predictor diagnostic is more informative when its own matched and mismatched targets are compared on the same control clips. Every one of the 375 trained fold/seed/evaluation-mask rows has greater error for a target taken from another source video. Mean gaps are 0.435368 for motion uniform, 0.432875 for MAMP, 0.428370 for robust motion, 0.418305 for region uniform and 0.344790 for connected regions. Each arm has 75 such rows. The initial control mean is −0.003065 and is positive in only 33/75 rows; the two experiment copies reuse that control. These diagnostic gaps live in different learned teacher spaces and should not be ranked across arms or interpreted as 375 independent replications. They establish clip-related information, which may include posture, viewpoint or recording characteristics as well as movement.

The retained standardized readout diagnostics flag none of 875 rows as nearly constant. The raw predictor token/clip diagnostics also do not flag constant representations. Complete constant-feature collapse is therefore an unsupported explanation. The standardized diagnostic is insufficient to exclude weak raw feature variation or selective loss of endpoint-relevant directions.

Regularization remains a concrete concern. The largest candidate penalty, 10,000, is selected for 49/125 teacher mean-motion fits, 96/125 online mean-motion fits and 109/125 teacher mean fits, versus 0/125 initial mean-motion fits. Several controls are repeated across arms in these counts. Boundary selection warrants a wider common training-only search; it does not establish that widening the grid will restore learning benefit.

## Earlier results that should not be blended with this grid

Notebook 12's retained summary reports teacher R² −0.022633 for gait targets and −0.010449 for all-landmark targets; initial encoder R² is 0.048160, direct pose 0.129978, and the training mean −0.010609. The all-landmark-minus-gait contrast is 0.012184 with interval [−0.018303,0.050207]. The trained encoders again trail initialization. Its ridge grid ends at 100, and its feature/implementation details differ from Notebook 18. Notebook 08's fixed-alpha values are −0.540136 for gait, −0.422757 for uniform, −0.351389 for initial, and −0.003843 for direct pose. Numerical movement across these separate experiments cannot be attributed solely to masking or regularization.

The mild prepared-coordinate corruption results from Notebook 12 remove about four tokens, generally less than 1% of valid support. Left/scattered gaps evaluate 622 clips and right gaps 621, all from 93 sources. The unaltered laterality target remains the reference. Any paired corruption effect should compare unaltered and altered predictions on the same available clips. These masks are applied after interpolation/normalization, so they test sensitivity of prepared features; a real missing-measurement claim requires removal before preparation. Notebook 13 demonstrates that stronger information boundary synthetically.

Notebook 14's observed-future decoder diagnostic yields displayed coordinate RMSE 0.011, 0.013 and 0.025 at horizons 0.25, 0.50 and 0.75 seconds. Decoding predicted future features gives 2.51, 2.61 and 2.64. Each horizon has 24 common landmark endpoints from two synthetic test clips. Four training updates do not test an adequately trained forecasting system, and access to observed future features makes the first condition a diagnostic rather than a deployable forecast. Several forecasts from one prefix are not a recursive rollout. The tutorial's 611 real clips/1,814 clip–horizon eligibility count is a preparation audit, not trained forecasting evidence.

## Concrete recommendations for the revisions

1. Center the paper on a physical-representation evaluation question: does a learned representation preserve a side-sensitive observable under sensible anatomical transformations and controlled context removal? Make the trained-versus-initial discrepancy and predictor/readout separation the empirical contribution.
2. Distinguish mathematical guarantees from learning. A sign-constrained readout can satisfy reflection by construction; zero symmetry error is then a correctness check. Its prediction accuracy and gain over a matched initial encoder remain separate tests.
3. State successive nulls as retrospective descriptions where they were not preregistered: no trained-over-initial gain; no target-eligibility advantage; no motion/region-mask advantage; no correct-clip predictor preference; no demonstrated forecasting benefit. Do not imply the whole notebook sequence was planned before inspecting results.
4. Make clinical conditions motivate different kinds of bilateral coordination, while identifying the actual endpoint as a coordinate-derived displacement contrast. There are no affected-side labels, calibrated clinical laterality ground truth, or demonstrated condition-specific diagnostic outcomes in these extension experiments.
5. Add the new trained-minus-initial uncertainty table only with its analysis date, fixed-model conditioning and exploratory status. Retain unfavorable outcomes in every version. Avoid framing a zero-spanning mask interval as evidence that all masks are equivalent.
6. Use three complementary figures: the exact training/evaluation separation; a mask audit showing a successful change in available information; and the initial/trained readout comparison with paired contrasts. A small clearly labeled synthetic future-decoding failure can belong in an appendix, without competing with the real-data result.
7. Prioritize expanded ridge regularization, support-only/direct-coordinate controls, summary-component ablations and a stagewise input/target agreement audit using existing checkpoints. These should be declared before running and preserved alongside the original scores. Reserve new sources or an external cohort for stronger confirmation because the present outer folds have repeatedly informed development.
8. Defer a larger training grid until a focused training-source pilot identifies a discriminating question. An explicit reflection penalty, changed regularizer, motion target or forecasting objective each merits its own matched comparison. No present result justifies bundling them into a claimed successful physical world model.

The multi-stage story is credible when its decisions follow the evidence: anatomical consistency checks establish the measurement contract; target-eligibility and motion/structure experiments probe the learning intervention; initial controls expose the weak learned contribution; and predictor/readout disagreement identifies the next unresolved mechanism. The paper becomes stronger by making those inferential boundaries visible.

</details>

<a id="literature-selection"></a>

<details>
<summary>Primary-source selection and rationale</summary>

# Primary-source selection for the revision

Sources were checked on 9 September 2026. The manuscript uses the literature to motivate questions and bound novelty, not to import performance gains into GAVD. Source titles and links below identify the selected primary research. Preprints are described as such unless a proceedings record was directly checked.

| Source | Use in the paper | Boundary |
|:--|:--|:--|
| [S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf), Abdelfattah and Alahi, ECCV 2024 | Establishes prior skeleton feature prediction | Our implementation and endpoint are an adaptation; action-recognition findings are not gait laterality results. |
| [Masked Motion Predictors are Strong 3D Action Representation Learners](https://arxiv.org/abs/2308.07092), Mao et al., 2023 | Motivates motion-informed sampling | The local arm adapts the official-code sampler; it does not reproduce the full motion-target method. |
| [Less is More: Compact-Token Masked Feature Prediction for Skeleton Representation Learning](https://arxiv.org/abs/2603.10648), Do et al., 2026 preprint, v3 dated 3 August | Recent connected-region masking precedent | Anatomical tubes alone are insufficient novelty. |
| [seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2f63d2963526bdd9ff1b8bcc2dc9905a-Abstract-Conference.html), Ghaemi et al., NeurIPS 2025 | Transformation-sensitive representation context | This study does not reproduce its architecture or world-model tasks. |
| [Soft Equivariance Regularization for Invariant Self-Supervised Learning](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3be6511c8f56d0dca4b5ed59fdf9b2f4-Abstract-Conference.html), Lee et al., ICLR 2026 | Supports evaluating geometry together with utility | Does not demonstrate that the local reflection loss will improve gait. |
| [V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning](https://arxiv.org/abs/2603.14482), 2026 preprint | Motivates testing dense or intermediate supervision | Larger image/video evidence; no assumed expected local gain. |
| [Human-JEPA: A Human-Centric Vision Model that Perceives and Anticipates](https://arxiv.org/abs/2608.21160), August 2026 preprint | Motivates separating encoder utility from predictor utility | RGB human-video evidence differs from pose laterality. |
| [GaitJEPA author repository](https://github.com/AVAuco/GaitJEPA) | Establishes existing JEPA-for-gait work | Silhouette recognition is a different task; no broad first-JEPA-for-gait claim. |
| [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906), Bardes et al. | Explains the regularizer | No claim that regularization excludes every form of information loss. |
| [Computer Vision for Clinical Gait Analysis: A Gait Abnormality Video Dataset](https://arxiv.org/abs/2407.04190), Ranjan et al.; associated IEEE Access DOI, 2025 | Dataset attribution | The local 625-clip subset is not the full published GAVD inventory. |
| [Gait asymmetry in community-ambulating stroke survivors](https://pubmed.ncbi.nlm.nih.gov/18226655/) | Spatial and temporal asymmetry motivation | No local stroke-effect estimate or affected-side prediction. |
| [Arm swing magnitude and asymmetry during gait in the early stages of Parkinson's disease](https://pubmed.ncbi.nlm.nih.gov/19945285/) | Motivation to preserve anatomical side identity | Shoulder landmarks do not directly measure arm swing. |
| [The Effect of Increased Gait Speed on Asymmetry and Variability in Children With Cerebral Palsy](https://pubmed.ncbi.nlm.nih.gov/32082235/) | Heterogeneity of gait symmetry | Local annotation does not resolve unilateral/bilateral subtype. |
| [Longitudinal Alterations in Gait Features in Growing Children With Duchenne Muscular Dystrophy](https://pmc.ncbi.nlm.nih.gov/articles/PMC9201072/) | Pelvic and multijoint movement beyond one signed contrast | The myopathic GAVD annotation does not identify Duchenne muscular dystrophy. |
| [Gait asymmetry in children with Duchenne muscular dystrophy: evaluated through kinematic synergies and muscle synergies of lower limbs](https://pmc.ncbi.nlm.nih.gov/articles/PMC10388506/) | Cautions against equating a simple symmetric measure with all normal movement | Motivation only; no clinical transfer claim. |

Other primary work inspected includes I-JEPA, V-JEPA 2, Group Equivariant CNNs and ASMa. They were not all included in the final manuscript because the closest skeleton and recent geometric precedents more directly explain its contribution. Omitting a tangential citation does not imply the method is novel.

The official workshop call is used only to establish topic fit, double-blind review, format and the posted submission dates. The latest manuscript primarily addresses articulated geometry and evaluation; no claim is made to contribute new multimodal sensors or physical-property estimation. [Physical World AI call](https://physworld-org.github.io/physworld.github.io/cfp/)

</details>

<a id="v6-critique"></a>

<details>
<summary>Revision 6 critique and final changes</summary>

# Review of revision 6 and changes made in revision 7

Both independent reviewers examined the complete manuscript, appendices and linked figures. The latest-grid numbers, split boundaries and training objective agreed with the audited evidence. Final corrections addressed the remaining notation and provenance issues: Δq now denotes a difference; the text distinguishes auditing saved predictions from regenerating them; Appendix A states 2,000 resamples and seed 812; historical teacher-token evaluation is separated from online-token training; scalar readout symbols are defined; the forecast table says decoded-coordinate RMSE; and the result caption identifies panel C.

The three-condition motion jobs are described as 25 × 3 encoders, with a separate 25 × 2 region grid. The description of inner tuning excludes Notebook 08's older fixed-penalty procedure. Ridge, invariance and equivariance are explained at first use. The final paper has a linked bibliography and conventional Markdown math delimiters.

Reviewer score for v6: 78.9/100. These changes improve communication and auditability without creating a new training result.

</details>

<a id="v7-final-critique"></a>

<details>
<summary>Final critique of revision 7</summary>

# Final critique of revision 7

The final manuscript earns 80.2/100 under this editorial rubric. It is a credible, carefully scoped evaluation study for an audience interested in articulated geometry and predictive representations. Its main result is a discrepancy between correct-clip feature prediction and the recovery of a signed coordinate-derived gait quantity. The final article retains positive controls and the small historical reflection-consistency improvement, as well as all unfavorable latest training comparisons.

The scientific limitation is still substantial: one development cohort, uncertain participant independence, an unvalidated coordinate target, a finite readout family, boundary-selected regularization and no established real-data forecasting. The historical original and Notebook 12 raw report chains are unavailable locally. Those limits constrain novelty and evidence scores even after the writing improves.

The next improvements should be new evidence, selected in this order: (1) wider training-only ridge selection and support-only/capacity-controlled summary ablations on saved encoders; (2) target agreement after individual preparation stages with timestamps and common observations; (3) a source-excluded explicit-reflection pilot requiring utility and geometric improvement together; (4) an independently specified participant-separated evaluation; and (5) common decoded future-coordinate endpoints under strictly past-only preparation.

The main text is approximately 3,350 whitespace-delimited words before references. Its official eight-page layout has not been verified, and governance reviews remain unresolved. The paper is a stronger workshop manuscript, not a submission-ready claim of a validated physical or clinical world model.

All actionable manuscript-level reviewer findings were incorporated. Requests requiring new experiments, clinical ground truth or institutional decisions are explicitly retained as outstanding research rather than described as completed.

</details>

<a id="historical-manuscript-notes"></a>

## Earlier manuscript and build notes

The following pre-existing README is retained as a historical snapshot. Its statements about a current title, available report directory, previous verification and PDF page counts describe that earlier revision. The current audit and seven-version scorecard above are authoritative for this work. Some older artifact files may no longer be present.

<details>
<summary>Pre-existing README content</summary>

# Manuscripts and evidence notes

The current title is *Auditing Reflection Symmetry in Self-Supervised Skeleton Representations*. The long paper and extended abstract present the completed laterality v2.1 experiment. They are internal anonymous drafts; the project governance record still blocks submission and artifact release.

## Reading and building

- [paper.md](paper.md) and [paper.pdf](paper.pdf): the full scientific account.
- `extended_abstract.md` and `extended_abstract.pdf`: the earlier shorter workshop version (files unavailable in the current checkout).
- [TUTORIAL.md](TUTORIAL.md) and [TUTORIAL.pdf](TUTORIAL.pdf): a concise explanation of the research trajectory and new notebook suite for workshop discussion.
- [COMPARATIVE_MASKING_PLAN.md](COMPARATIVE_MASKING_PLAN.md): the questions, controls, and execution stages for notebooks 11–14.
- [COMPARATIVE_MASKING_VERIFICATION.md](COMPARATIVE_MASKING_VERIFICATION.md): software verification, real-data feasibility checks, and the remaining empirical evaluation scope.
- [00_critique_and_tutorial.md](00_critique_and_tutorial.md): the detailed guide to the target, split, probes, and interpretation.
- [TRAIN_TEST_SPLIT.md](TRAIN_TEST_SPLIT.md): the canonical numeric GAVD cohort,
  outer train/test folds, inner validation ranges, and leakage guarantees.
- [figures/README.md](figures/README.md): figure provenance and regeneration instructions.

The two Markdown manuscripts are now the editable sources. Their YAML headers supply the title and abstract, and citations resolve against the existing shared bibliography at `../../docs/references.bib`. Generate the corresponding TeX and PDFs with:

```bash
bash neurips-laterality/docs/build_manuscripts.sh
```

The build uses Pandoc, Tectonic, the shared [manuscript template](manuscript_template.tex), and the existing NeurIPS 2026 style. It preserves the style's margins and type sizes. The checked PDFs have eight and four main-text pages, respectively, followed by a separate references page. Edit Markdown and rebuild to keep the text, equations, and captions synchronized.

## Workshop positioning

The [PhysWorldAI call for papers](https://physworld-org.github.io/physworld.github.io/cfp/), checked on September 5, 2026, lists a maximum of eight pages for long papers and four for extended abstracts, excluding references and appendices, with double-blind review. Its themes guide the emphasis as follows.

| Workshop area | Connection supported by this experiment |
|---|---|
| Physical Geometry | A known reflection action on articulated pose representations; direct token alignment and a signed motion target. This is the primary fit. |
| Cross-cutting evaluation | Source-disjoint representation training, paired initializations, matched parity controls, and separate tests of prediction and geometric consistency. |
| Physical Sensors | Validity-aware analysis of video-derived landmarks provides a limited sensing connection. No additional sensor or fusion system was evaluated. |
| Physical Characteristics | No measured material property, contact model, or physical dynamics result. These are possible future applications, not current contributions. |

The paper treats the S-JEPA adaptation as a representation component relevant to world models. Masked latent prediction alone does not establish future rollout or action-conditioned physical reasoning. The algebraic odd projection is attributed to established equivariance and group-averaging methods; its use as an experimental control is the contribution here.

## Results that determine the argument

All empirical figures and the manuscript results use the completed report under `../artifacts/paper/protocol_6f7baefbda07/report/`. The canonical notebooks are output-free; their explanatory cells and saved real-data artifacts were read together with the implementation. Synthetic smoke outputs and older transductive estimates are excluded.

| Finding | Estimate and 95% source-bootstrap interval | Report row |
|---|---|---|
| Vanilla learned minus initial token error | +0.03055 [0.01576, 0.04763] | `strict_representation_equivariance_source_bootstrap.csv`: `learned_minus_initial_strict_equivariance`, vanilla |
| Augmented minus vanilla token error | −0.00843 [−0.01020, −0.00687] | Same file: `reflection_minus_vanilla_strict_equivariance` |
| Native learned predictive utility | 0.05979 [−0.02527, 0.12571] | `checkpoint_source_bootstrap.csv`: `absolute_primary` |
| Native learned minus initial prediction | −0.01798 [−0.03851, 0.00248] | Same file: `primary_training_content` |
| Augmented minus vanilla prediction | +0.00408 [−0.00556, 0.01277] | Same file: `reflection_minus_vanilla_primary` |
| Constructed learned predictive utility | 0.04302 [−0.04356, 0.11283] | Same file: `absolute_constructed` |
| Constructed learned minus initial prediction | −0.05874 [−0.09549, −0.01740] | Same file: `constructed_training_content` |
| Native output antisymmetry error | 0.21548 [0.19352, 0.23635] | `native_output_symmetry_source_bootstrap.csv`: `absolute_native_learned_symmetry`, vanilla |

The paired augmentation effect corrects a material error in the earlier drafts: its improvement in token error is distinguishable from zero. Both trained variants nevertheless remain worse than initialization under the specified token action and fail the absolute acceptance criterion. Their absolute token-error intervals cross 0.10, which is why three-decimal precision matters.

The primary predictive estimand averages five seed-specific scores, each computed from pooled out-of-fold predictions. It does not average 50 separately calculated checkpoint scores or score a seed-mean prediction ensemble. The high-coverage sensitivity results are explicitly secondary ensemble estimates. Token and native-output errors have different normalizations, so the revised absolute figure gives them separate panels.

## Interpretation decisions

The revised argument distinguishes the observed training effect from the broader question of whether an encoder represents physical geometry. The strict token action fixes latent channels, and uncentered error can be low for shared or input-insensitive features. A ridge-probe failure also cannot locate a loss of information among input resizing, representation learning, pooling, and linear decoding. The target-component oracle verifies its own arithmetic and does not resolve that attribution.

Exact sign reversal follows from the representation's transformation law, including its validity masks. It does not require a physiologically symmetric person or symmetric gait. Initial features already receive the same anatomical schema and probe construction as trained features, so this is not a test of discovering geometry without supplied structure.

The protocol was frozen internally following prior development. No external preregistration is claimed. The bootstrap intervals are pointwise and conditional on the fitted pipeline. The 0.10 margins are operational choices, with no application-level calibration, and inconclusive predictive contrasts do not establish equivalence.

Read-only verification with the suite's own cohort, split, and evaluation loaders passed for all 100,000 prediction rows, 50 jobs, and 16 lanes. This checked artifact lineage, checkpoint and CSV fingerprints, source coverage and weights, and the saved token-error algebra. The experiment, protocol, trained checkpoints, and governance statuses were not changed during manuscript revision.

</details>

<a id="v8-processing-verification"></a>

## V8 processing table and notebook verification — 10 September 2026

Section 3.2 was checked against the notebook entry points, their implementation, and the saved GAVD inputs. The revision changes the two surrounding paragraphs, their heading, and the table descriptions. Its dimensions describe the completed GAVD runs, not the smaller synthetic examples. No scientific results, cohort membership, or training settings were changed.

| Detail checked | Correction or clarification | Implementation |
|:--|:--|:--|
| Missing samples and resizing | Only interior gaps of at most four samples are filled. Filled samples count as valid encoder input. Coordinates and validity are resized by sequence index; resized validity must reach 0.999. | [geometry.py](../laterality/geometry.py), `interpolate_short_gaps`, `temporal_resize`, `prepare_pose` |
| Centering and scaling | The encoder path uses a within-clip median pelvis fallback and a scale from valid shoulder and hip widths. The movement-measure path has no pelvis fallback, gap filling, or temporal resizing. | [geometry.py](../laterality/geometry.py), `pelvis_normalize`, `prepare_pose` |
| Patches, features, and masks | Four prepared steps give 12 values per landmark, projected to 96 features. All 528 positions remain in the attention grid. Prepared validity and deliberately hidden targets have separate masks; numeric target indices are optional. | [model.py](../laterality/model.py), [motion_structured_training.py](../laterality_extensions/motion_structured_training.py), [motion_runtime.py](../laterality_extensions/motion_runtime.py) |
| Identity and source exclusion | Landmark identity follows a fixed index order. IDs and movement targets are stored alongside arrays. Each fitted encoder and regression excludes that outer fold's test sources. | [data.py](../laterality/data.py), [splitting.py](../laterality/splitting.py), [comparative_evaluation.py](../laterality_extensions/comparative_evaluation.py) |

Reapplying `prepare_pose` and `paired_valid_target` to all 625 manifest-listed raw clips reproduced the saved coordinates, validity, and per-pair contrasts exactly. All 625 archive hashes, sequence IDs, and source IDs matched. The largest scalar-target difference from the CSV was approximately 9.93e-17. All invalid saved coordinates were zero. The existing five geometry contract tests also passed.

The separate source check covered all 50 saved motion/region jobs: none of their 1,200,000 sampled clip draws came from the corresponding test sources. The saved grid's [15,625 membership records](../artifacts/motion_structured/grids/292443b0fab5339f5da7ca566a85d6172ffc5b64abe5febf2546681a0152ff57/memberships.csv) likewise had no source crossing training and testing within a fold and seed.

The current pose cache contains **656 archives**, while the locked inventory specifies **642**. A normal `prepare_cohort` call correctly stops at this mismatch. The verification above used only the existing 625 accepted clips from 93 sources, all of whose archive hashes still match. The inventory must be reconciled before a fresh full-cohort rebuild; neither the inventory contract nor the saved cohort was changed here. This local reproducibility issue is recorded outside the workshop manuscript.

The generated submission files and verification hashes are recorded in the [build manifest](physworld_revisions/submission_build_manifest.json) and [revision verification record](physworld_revision_verification.json). Token projection and attention behavior were checked against code; no encoder training or checkpoint inference was performed.

## V8 discussion, scope, and ethics wording — 10 September 2026

Sections 5 and 6 were rewritten in plain language without changing the reported results or turning proposed work into completed experiments. Section 5 keeps the poorer movement estimates tied to the motion-containing feature summary and tested regression. It names the no-augmentation baseline for the reflection result, preserves the need for a real-data test of explicit reflection loss, and explains the proposed readout and forecasting checks. The movement measure is described as averaging normalized left-right median-speed differences.

Section 6 keeps the reasons for choosing GAVD, the distinction between held-out videos and unseen people, and the absence of independent clinical validation. The [pinned repository documentation](https://github.com/Rahmyyy/GAVD/blob/a87859c881603443f200bcd640663d2c3d7a8136/README.md) and [MIT License](https://github.com/Rahmyyy/GAVD/blob/a87859c881603443f200bcd640663d2c3d7a8136/LICENSE) were rechecked. The paper retains the license conditions and separates repository permissions from the reviews needed for videos and derived-pose releases. Verification counts and the distinction between checking saved outputs and rerunning inference are unchanged. The bibliography and numerical results were not revised.
