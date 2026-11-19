# Independent accessibility review

## Assignment and independence

This review assesses whether a general AI/ML researcher can understand the research question, comparisons, findings, and limits of `docs/draft03/manuscript_source.md` and generated `docs/draft03/manuscript.md`. I did not author either manuscript. My only writable artifact is this review. Numerical verification and final scientific and layout acceptance belong to the separately assigned reviewer.

The first reading below used only the abstract and introduction of both manuscript files, including the introductory figure caption. I saved this explanation before reading the scientific protocol or the remaining manuscript. The coordinator's assignment already disclosed the broad result, so this is an independent reading assessment, not a blinded test of whether a reader would discover that result unaided. Every substantive fact in the explanation below appears in the abstract or introduction.

The initial SHA-256 identities were `83e9e8fc0de1a33d3629241e0c39db087c8a8bbbcbe81c0dbe7553d80a306eca` for the source and `109bc2b222d5dde71aa7208e4b383f51e51a475f18ab0597eff8629faad93fcc` for the generated Markdown. Later material changes require a recheck.

## First reading: abstract and introduction only

**Research question.** Can a model trained on videos labelled as normal recognize unusual walking patterns beyond what simpler explanations already capture, such as how much the person moves, their current posture, the camera view, and the quality of the observed joint positions?

**Method.** The study trains small models to predict eight future observations of leg positions from sixteen earlier observations. It compares these predictions with simple references, including repeating the latest pose. It also compares six ways of hiding parts of the observed past during training, together with training on the complete past. It then tests whether a fixed score based on unexpected prediction difficulty ranks abnormal examples better than a simple score based on limited movement. Related windows from a source video stay together during evaluation, and the score is also tested on a separate dataset.

**Strongest reported finding.** A model that starts from the current posture reduces forecast error on normal sources by 40.8% compared with repeating the last pose. The paper nevertheless reports poorer abnormality ranking than the low-motion comparison, both internally and on the separate dataset. Organizing hidden inputs by anatomy or motion also does not improve forecasting over uniformly selected hidden inputs. Thus, the clearest reported result is that useful short-term prediction does not establish useful detection of unusual walking patterns.

**Contribution.** The contribution is a reproducible set of measurements, comparisons, and evaluation rules that makes this distinction testable. The introduction explicitly presents the contribution as a controlled benchmark and an empirical evaluation finding. It does not present normal-only forecasting or anatomical masking as newly invented methods.

## Full-text findings

The full-text pass read the source manuscript, generated main text and captions, `reference_v2/protocol.md`, and the text labels in the four main SVG figures. The initial full-text snapshot used source SHA-256 `000070b44f6fe3f9d37beb4e3473e0a2a3e2d7ac401f938782670019b34fbcf4`, generated Markdown SHA-256 `109bc2b222d5dde71aa7208e4b383f51e51a475f18ab0597eff8629faad93fcc`, and protocol SHA-256 `7b7560939585a6849ccf85c9e6b096d5327bbfe6e5c50ef5b314e70ddfe1f71f`. Line references below refer to those versions unless a repair is explicitly identified. The earlier explanation remains unchanged.

The coordinator made repairs during review. The first repair recheck inspected source SHA-256 `1f636a5a58f642e927c27bff4f20ae20470a7b1a8ebe7cfb7e0eab3082e7ba26`, generated Markdown SHA-256 `14f8dffad1c47ec0431ac1380afb21b7551ad4a5d907b9583305e9e999a26608`, and caption generator SHA-256 `d8753b59d3ab363e0cf88fad3dbfa63e1684e783e1ab0f48be6e77968d822774`. Recheck line references within the five findings identify those repaired versions. All five findings were closed at those identities. The later targeted final recheck below records the current identities and confirms that the repairs remain intact.

Medium findings affect the interpretation of a central comparison. Low findings affect terminology, standalone captions, or the flow of the argument. These are accessibility findings rather than independently verified numerical errors.

### AR1. Medium: identify the anomaly forecaster and its fitting population explicitly

**Evidence.** The initial source at line 83 opened with "The new candidate asks whether a source is easier or harder to predict than expected from its past." At line 89, "The matched posture model uses the same schedule" contrasted that model with "the larger state." The protocol explicitly names `ridge_posture_velocity` as the primary detector, but the initial manuscript did not directly connect its primary score to that predictor. The ranking caption at generated line 117 named the score transformation without naming its predictor. Meanwhile, the positive result at source line 111 concerned the current-posture transition.

**Expected and observed reading.** A reader should identify the predictor behind each endpoint without inferring it from "new," "matched," or "larger." The initial wording could carry the strongest current-posture forecasting result into the primary anomaly result as though they used the same predictor and fitting allocation.

**Suggested repair.** Name the posture-plus-velocity ridge at the start of the scoring subsection, identify the posture-only calibration control, and explicitly distinguish these reduced-pool calibration fits from the full-pool forecasts in Table 2. Name the predictor in the ranking caption as well.

**Recheck.** Addressed in the source and regenerated Markdown. Source line 83 now says, "The prespecified candidate uses the posture-plus-velocity ridge forecaster." Source line 89 explicitly distinguishes the forecast-fitting pools. These repairs appear at generated lines 95 and 101. The regenerated ranking caption at line 117 names the predictor and this fitting distinction. This repair preserves the fixed primary candidate and the matched posture control.

### AR2. Medium: the introductory caption implied identical past information

**Evidence.** The observation-task caption at generated lines 17 and 19 says, "All methods receive the same prefix and are scored on identical observed future coordinates." Source lines 55 and 57 describe intentionally different current-posture, own-leg, and bilateral-history inputs. The SVG itself accurately says "Current posture or history."

**Expected and observed reading.** Shared windows and target support should be distinguished from the input subsets under comparison. "The same prefix" can imply equal input information across the posture and history references, despite their central experimental difference.

**Suggested replacement.** "All methods are evaluated on the same windows and observed future coordinates, using each model's declared subset of past observations. Future availability never conditions a prediction."

**Recheck.** Addressed in the caption generator and regenerated Markdown. Builder line 313 now says, "Methods use the same evaluation windows and future target support, with their declared past inputs." This wording appears at generated lines 17 and 19.

### AR3. Low: use one name for the complete-prefix masking baseline

**Evidence.** The abstract at source line 5 calls this setting "an uncorrupted anchor." Source line 63 calls it "the suffix-only arm" while also using "target anchors" for the latest joint coordinates. The masking caption at generated line 148 calls all seven settings "training-mask policies," while the figure's clearer label is "No extra masking."

**Expected and observed reading.** The reader should distinguish the six equally budgeted prefix-corruption policies from the setting that keeps the complete observed past. "Suffix-only" does not explain which information is available, and "anchor" has two meanings within one paragraph.

**Suggested replacement.** In the abstract, use "six ways of masking the past with the same hidden-token budget, plus a baseline using the complete observed past." In the main text, use "The unmasked-prefix baseline retains every naturally observed past token and therefore has more visible training input." Describe the figure as "six masking policies and an unmasked-prefix baseline."

**Repair criterion.** Use one descriptive name consistently, and describe only the six corruption policies as having matched hidden-input budgets. Keep the protected latest-coordinate anchors explicitly defined.

**Recheck.** Addressed. The repaired abstract, introduction, and methods consistently use "unmasked prefix baseline." Generated line 150 describes six prefix-corruption policies and this baseline. The obsolete "suffix-only arm" and "uncorrupted anchor" labels are absent from the regenerated manuscript. Latest-coordinate anchors remain separately defined.

### AR4. Low: make the low-motion comparator's time support clear in standalone summaries

**Evidence.** The abstract says "a low-motion baseline." The initial ranking caption at generated line 117 says "lower observed displacement" without specifying the future segment. The main derivation at source line 79 does explain negative persistence error, while Table 3 separately lists "Low prefix motion."

**Expected and observed reading.** A reader of the abstract or figure should understand that the primary low-motion comparator uses realized future displacement from the latest past anchor. It should be readily distinguishable from the prefix-only movement comparator. The main text supplies this distinction, but a standalone summary does not.

**Suggested replacement.** In the abstract, use "a low-motion baseline based on realized displacement during the same future segment." In the ranking caption, use "negative persistence error, which assigns higher abnormality to smaller realized future displacement from the latest observed pose."

**Repair criterion.** Make the observation period explicit wherever this main comparison is summarized. Both forecast error and this baseline use the same realized future, which makes it an appropriate simple reference for the ranking endpoint. Its strong result and its historical outcome exposure should remain visible.

**Recheck.** Addressed. The regenerated abstract at line 5 explicitly names displacement over the observed future. The ranking caption at line 117 identifies smaller realized displacement over the same observed future segment. The main derivation and historical outcome-exposure qualification remain visible.

### AR5. Low: introduce the constant-prior explanation before reporting its post hoc test

**Evidence.** Source line 115 is one paragraph covering velocity, bilateral history, own-leg history, alternate partitions, the need for a posture reference, and two post hoc mean priors. "A separately declared post hoc diagnostic adds two normal-training intercept priors" introduces another comparison without first stating the competing explanation it addresses.

**Expected and observed reading.** A reader should know why an average pose is a consequential reference before processing another pair of MSE values and intervals. The current wording reports the control and its post hoc status, but its motivation arrives only in the closing inference.

**Suggested replacement.** Start a separate paragraph: "A simpler explanation is that forecasting improves because the model predicts a typical normal pose. We examined the constant-pose version of this explanation in a separately declared post hoc diagnostic, fitting a mean pose and a mean displacement using only normal outer-training sources." Retain both priors, the existing numerical comparisons, and the explicit outcome-exposed exploratory qualification.

**Repair criterion.** State the competing explanation before its test and preserve the distinction between a constant prior and a posture-conditioned predictor.

**Recheck.** Addressed. Source line 117 and regenerated line 144 introduce a separate paragraph with, "A model might improve on persistence merely by predicting an average normal pose at each future query." Both priors, their reported comparisons, and the post hoc exploratory qualification remain in that paragraph.

## Challenge to the strongest positive claim

The strongest positive claim is the reported 40.8% forecast-error reduction over persistence. Its simplest competing explanation is that a regression maps the current normalized pose toward typical later poses without learning detailed temporal organization. This explanation is compatible with a current-posture-only predictor and the reported lack of benefit from added velocity summaries. It does not require a recovered physiological state or an internal account of walking coordination.

The manuscript already challenges the most restrictive version of that explanation. The generated table reports MSE 0.10898 for a constant normal mean pose, 0.12235 for mean displacement, and 0.07174 for the current-posture model. The reported posture advantage over the mean-pose prior is 0.03724 with interval [0.02539, 0.05150]. If independently verified, that comparison supports information in the current pose beyond the tested constant prior. It leaves posture-dependent regression toward typical poses compatible with the result. These priors are explicitly post hoc and should retain that status.

The supported interpretation is useful posture-conditioned prediction under this observation and evaluation contract. The main text already limits the inference in this way. I found no need to weaken the posture reference, the low-motion control, the negative structured-mask result, or the failure of the primary anomaly score. The numerical reviewer must determine whether the reported comparisons are correct.

## What is accessible

The abstract and introduction make the research question, positive forecasting result, negative anomaly comparison, and benchmark contribution understandable without specialist gait or JEPA knowledge. The related-work section explains masking and S-JEPA in ordinary terms and clearly identifies the implemented coordinate objective. It avoids implying an official reproduction of the named representation-learning methods.

The measurement section defines a pose, the foot landmarks, normalization, the temporal grid, and the forecast horizon. The current source's explicit hip-midpoint definition makes "pelvis reference" understandable. Squared prefix-torso units and the table's multiplication by 1,000 are explained; I found no unresolved ambiguity in those units. Natural missingness, deliberately hidden inputs, observed evaluation targets, and protected latest observations are also distinguished.

The hierarchy of windows, sequences, source videos, and external coded participants is explained well. The manuscript directly states the limits of source identity, clinical labels, calibration support, fixed-fit uncertainty, and external recording differences. Its explanation of AUROC and prevalence-dependent average precision is sufficient for a general AI/ML reader. The discussion bounds the failed detector and masking findings to the tested methods and task.

The remaining abstract terminology is denser than the introduction. In particular, "medical world models" is a motivation rather than an evaluated patient simulator. The introduction and discussion make that limit clear. An optional shorter abstract phrase is "a benchmark of short-term movement prediction motivated by medical world models." This wording change is not necessary to preserve the scientific conclusion.

## Targeted final recheck

The presentation and caption edits were rechecked against the following exact SHA-256 identities. The PDF hash independently matches the current identity supplied by the coordinator.

| Artifact | SHA-256 |
| --- | --- |
| `docs/draft03/manuscript_source.md` | `ed68033808edbccfa25bca775c613aaba8602fb21f5dce8a88c589342a9ff9a6` |
| `docs/draft03/manuscript.md` | `5dc4b0d18f0e9c25b81d18de1ae7b73cbba961cfb6a0d92e8c630af2dd47877b` |
| `docs/draft03/manuscript.pdf` | `f0c4af0ebd05ee710fbd1ef2b6e26755331ebc58b48ce352ad50e3f7c4f6f760` |
| `scripts/build_next_draft03.py` | `0bd5236648063b80856bc2ea2c5f65eb12851491b5c8b23832d8ff2d6d804837` |

All five prior findings remain resolved. The primary anomaly predictor and its smaller fitting pools are explicit at generated lines 95 and 101 and in the ranking caption at line 117, preserving AR1. The observation caption at lines 17 and 19 retains the distinction between shared evaluation support and declared past inputs, preserving AR2. The revised M0 paragraph at line 75 calls the complete observed prefix a full-input reference for the six budget-matched corruption policies, preserving AR3 while removing the earlier redundant wording. The abstract and ranking caption still identify the primary low-motion comparator's observed-future displacement, preserving AR4. The separately introduced constant-pose explanation and its explicitly post hoc test remain at line 144, preserving AR5.

The shorter workflow labels retain the observation sequence and its essential distinctions: two-dimensional joints, sixteen past samples, posture or history, eight future samples, normal-source fitting, common targets, and grouped scores. The explicit 0.1 through 0.8 second horizon and 1.5 second prefix span remain visible. I found no meaning lost through shortening these labels.

The table-label text and LaTeX row mappings remain clear after the first columns were made ragged-right. The readable external score names at generated lines 277 through 285 match the score descriptions in the main text. In particular, "Forecast MSE," "Forecast minus persistence," and "Low prefix motion" state the measured quantities directly.

The Table 5 caption at generated line 287 now explicitly distinguishes 2,000 participant bootstrap draws within label strata for the individual-score intervals from 5,000 draws for the principal paired contrast. This wording makes the two uncertainty calculations distinguishable and consistent with the paired-contrast description in the main text. This is a reading check of the stated procedures, not an independent recalculation of either interval.

These edits introduce no new accessibility issue. This targeted pass inspected the source, generated Markdown, relevant LaTeX, and SVG labels. The PDF check establishes file identity only; independent scientific and rendered-layout acceptance remains with `v2_reviewer`.

## Verification and limits

The review used line-numbered local file reads, SHA-256 fingerprints, protocol comparison, and SVG text extraction. No model was trained, no scores or intervals were recomputed, and no external literature was independently verified. Rendered PDF pages were not inspected in this accessibility pass. The assigned final reviewer owns numerical reconstruction, scientific acceptance, and PDF layout checks. No measurable compute, memory, or financial cost was recorded for these lightweight reads.

The repair recheck and targeted final recheck confirmed that the five repairs propagated into and remain in generated Markdown. The hip-midpoint pelvis definition and connected-source grouping explanation also appear there. The PDF has a separate acceptance path, and this review does not claim to verify its rendering. Material edits after the identities in the targeted final recheck reopen the affected accessibility checks.

## Disposition

The manuscript is accessible to a general AI/ML researcher at the current source and generated-Markdown identities recorded in the targeted final recheck. Both medium findings and all three low findings remain addressed, and the later presentation edits introduce no new clarity issue. No accessibility change or additional experiment is required by this review. Numerical, scientific, and PDF-layout acceptance remain with the assigned final reviewer, and the coordinator owns the final disposition.
