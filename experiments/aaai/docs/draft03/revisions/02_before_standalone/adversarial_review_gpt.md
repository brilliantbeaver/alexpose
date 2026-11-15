# Independent review of the positive draft03 framing

This review examines the completed draft03 manuscript and its saved completion audit on 15 November 2026. The positive completion result is defensible. Its strength comes from paired targets, held-out evaluation units and an independent laboratory measurement. It does not establish a better anomaly detector, a newly discovered neurological mechanism, or a new masking family. I recommend keeping this framing and making the wording changes below before submission.

The reviewed manuscript had SHA-256 `ed1c45cd848ce2dd64ba16a216e9e23000692610f42dd954c3cbaa7937e1c3dd`. The reviewed draft builder had SHA-256 `4ef850e3117a55ab85371770f460cbdb43a936bb1df0821d27586bb1bf8a076f`. Later changes can legitimately change these hashes. This review did not train a model, select a new method, alter an evaluation population, or replace the original primary analysis.

## What I verified independently

I read the [manuscript](manuscript.md), [draft builder](build_positive_draft.py), [completion audit script](../../scripts/audit_normal_completion.py), [paired summary](advisory/completion_audit_v1/completion_summary.csv), [audit verification](advisory/completion_audit_v1/verification.json), saved participant/source losses, and saved bootstrap draws. I recomputed the three new paired summaries from their unit-level losses and checked the reported interval endpoints against the saved draws. I also inspected the residual fitting code and the clinical interpretation.

| Claim | Actual comparison and population | Verified result | Assessment |
|---|---|---|---|
| GAVD completion improves by 19.5%. | Uniform-trained both-leg correction versus the own-leg correction on the fixed distal-leg bank, with 28 held-out normal recording sources. | Mean MSE changes from 0.080769 to 0.065020. The absolute benefit is 0.015748, with conditional interval [0.010721, 0.021300]. | Correct. This is a ratio of mean errors on a secondary observation task. It is not a detection gain. |
| Laboratory completion improves by 5.2%. | Both-leg correction versus the own-leg correction, with 128 held-out healthy participants. | Mean MSE changes from 201.038 to 190.584 square degrees. Relative reduction is 5.200%, with interval [3.553, 6.698]%. Error decreases for 109 of 128 participants. | Correct. All four fold means favor the both-leg predictor. The interval is conditional on the recorded fits and design. |
| Anatomical masking improves completion by 7.7%. | Anatomical-trained versus uniform-trained both-leg predictors on identical distal-leg evaluation targets, with 28 normal sources. | Mean MSE changes from 0.065020 to 0.060009. Relative reduction is 7.707%, with interval [6.356, 9.003]%. Error decreases for 27 of 28 sources. | Correct for this bank and training procedure. This is a different contrast from the first two rows. |
| Anatomical masking has no conclusive advantage on the ordinary uniform bank. | The same two training policies, evaluated on the unchanged uniform task. | Relative reduction is 0.763%, with interval [−0.955, 2.584]%. Error decreases for 16 of 28 sources; one fold mean is worse. | Correct. The interval neither establishes improvement nor proves equivalence. |

The percentages use the difference of mean losses divided by the comparator mean. They do not average participant-specific percentage improvements. The paired resampling preserves the original folds and evaluation units. The source and participant counts, paired metadata checks and saved output hashes agree with the audit record. The newer paired audit resolves the earlier advisory concern that the 7.7% masking comparison was only a descriptive point estimate. The [earlier positive-framing advice](advisory/positive_framing_review.md) should be read as a pre-audit document.

The uncertainty remains conditional. The four models share training sources across folds, and the bootstrap does not repeat model fitting, source selection or exploratory design choices. GAVD recording sources are not verified distinct people. The manuscript states these limitations and should retain them.

## Changes that would make the abstract clearer

The abstract contains valid numbers but leaves their denominators implicit. A reader can easily interpret all three as improvements from adding the other leg. Only 19.5% and 5.2% are that comparison. The 7.7% figure compares training policies for the already bilateral predictor.

Replace the relevant abstract sentences with wording along these lines:

> On GAVD's predeclared distal-leg diagnostic, the uniform-trained both-leg predictor reduces mean normal completion MSE by 19.5% relative to an additional own-leg correction. In a post-outcome paired analysis of frozen laboratory predictions, the corresponding reduction is 5.2%, with a conditional 95% interval of [3.6, 6.7]%, and error decreases for 109 of 128 healthy participants. Separately, anatomical training masks reduce the bilateral predictor's MSE by 7.7% relative to uniform training on the same distal-leg targets, while the unchanged uniform evaluation task shows no conclusive advantage.

This wording states what changed in each experiment and makes the new descriptive analysis visible to readers who only see the abstract. “Error decreases for 109 participants” is more precise than “improves 109 participants,” which can sound like a health outcome. These are MSE reductions, not improvements in clinical function, classification accuracy or RMSE.

The abstract should continue to state that the original frozen fusion lowers AUROC in both datasets. Moving the negative primary result into a supplement would weaken the paper's credibility. The present draft keeps it visible.

## Two other wording issues deserve correction

First, use **numerical feature rank** or **numerical rank of the standardized feature Gram matrix** when naming the recorded ranks. The fitting code counts eigenvalues above a relative/absolute threshold. It does not estimate ridge effective degrees of freedom or a spectral effective-rank quantity. “Equal widths do not equalize effective capacity” is an appropriate caveat, but “whose effective ranks differ” is less precise as a description of the measured audit. The reported GAVD ranks of 111 versus 141 and laboratory ranks of 44 versus 62 support the narrower wording. No additional capacity experiment is needed to correct this description.

Second, the introduction says the positive result “persists against a fitted own-leg augmentation and direct interpolation.” The direct-interpolation evidence should be located explicitly. The laboratory outer-test audit compares the same healthy participants across interpolation, the retained baseline, own correction and bilateral correction. The GAVD direct-interpolation adequacy checks do not by themselves establish a new, matched distal-bank outer-test comparison. A precise alternative is: “The bilateral result is measured against a fitted own-leg augmentation. In the laboratory audit, the retained and augmented predictors also outperform the declared direct-interpolation rule.”

The introduction's example could also say that the other leg **may** help when local clues are insufficient. The current experiments establish benefit under two declared observation settings. They do not provide a general rule identifying every condition in which extra context will help.

## A clear positive-evidence paragraph

The manuscript already has enough evidence. It would benefit more from one explicit synthesis than from another optimistic claim. The following paragraph could close the completion-results subsection:

> The positive finding is that the fitted predictors can use opposite-leg observations to complete missing movement beyond the declared own-leg correction. On the GAVD distal-leg diagnostic, uniform training reduces normal-source mean MSE from 0.080769 to 0.065020, a 19.5% reduction. In the independently fitted laboratory study, the corresponding reduction is 5.2%, with a conditional 95% interval of [3.6, 6.7]%, and lower error for 109 of 128 healthy participants. A separate paired comparison shows that anatomical training improves the bilateral predictor by 7.7% on the distal-leg task, but its 0.8% improvement on the uniform task remains uncertain. The paired intervals and this emphasis are post-outcome analyses of frozen predictions. Together, these results support useful completion under the declared observation tasks, without establishing equal effective capacity, clinical benefit or improved anomaly ranking.

The phrase “independently fitted” matters. The laboratory study refits on synchronized angle measurements. It is not a demonstration that video-trained weights transfer to clinical motion capture. The two datasets should remain separate in plots and tables because their measurements and MSE units differ.

## Clinical motivation is appropriate within its current limits

The physiological argument is useful because it explains why local movement magnitude and inter-joint relationships need separate measurements. It does not need to claim that the model recovers a neurological controller.

The cited stroke study examines EMG module structure in people with hemiparesis and relates reduced independent activation to locomotor performance. It motivates investigating relationships without establishing that projected coordinate reconstruction measures the same phenomenon. [Clark and colleagues](https://pmc.ncbi.nlm.nih.gov/articles/PMC2822696/)

The cited Parkinson's study finds that acute medication can improve arm velocity and symmetry without changing its chosen bilateral coordination measure. That supports separating movement size from a relationship measure. It does not prove that all coordination measures are medication-insensitive or that the present lower-limb readout captures dopaminergic function. [Sterling and colleagues](https://pmc.ncbi.nlm.nih.gov/articles/PMC4609542/)

The warning about neurological identification is necessary. Mechanical and task constraints can produce apparently low-dimensional coordinated activity without identifying a neural grouping mechanism. The paper correctly cites this concern and does not treat a prediction gain as proof of neural synergy. [Kutch and Valero-Cuevas](https://pmc.ncbi.nlm.nih.gov/articles/PMC3342930/)

The discussion also correctly distinguishes myopathic muscle disease from a single central neurological lesion. It preserves amplitude and common time instead of independently normalizing away possible phenotypes. These are defensible design motivations. Disease-specific performance, symptom severity, medication effects, side of lesion and compensatory mechanism are not identified by the present results. The external dataset concerns healthy participants and stroke only.

## What novelty survives adversarial review

The strongest contribution is a reproducible measurement study with useful positive completion results and explicit boundaries on their interpretation. It separates an observation task, a fitted predictor comparison and a conversion into an anomaly score. The failed conversion is valuable because the positive prediction component really does work under some declared tasks.

The retained local solution, identical targets, nominal-width control, common mask budgets, paired source aggregation and independent participant replication make this more informative than reporting reconstruction error alone. However, equal width has not removed differences in feature basis, numerical rank, inductive bias or selected regularization. The safe claim is that the bilateral **predictor family** performs better under the recorded procedure. An estimate of intrinsic conditional information after all own-leg information has been exhausted would require stronger controls.

The anatomical result is task-specific specialization. The same models also reveal an uncertain uniform-bank advantage, and both banks remain visible. Masking recipes already have close precedent, so the paper should claim a controlled comparison and measured task dependence rather than invention of anatomical masking. Different valid-target exposure and normal-only recipe selection remain part of the training-policy comparison.

The title is reasonable if “when” means these declared tasks. It should not be presented as a universal theory of when the opposite leg helps. A workshop contribution is plausible on the strength of its measurement controls, transparent evidence and replication. The data do not justify a numerical acceptance probability or a claim of a new state-of-the-art system.

## Positive claims that should remain rejected

| Rejected claim | Reason |
|---|---|
| The method improves gait anomaly detection. | The frozen primary fusion loses to the preserved local channel in both datasets. |
| Anatomical masking is generally better. | The unchanged uniform task has an interval crossing zero, and encoder-mask pretraining was not tested here. |
| The model measures neural coordination or a physiological phase mechanism. | Feature capacity, body reference, mechanics and observation effects remain; the exact timing-support adequacy gates fail. |
| Lower reconstruction error makes gait more normal. | Stable compensations may be predictable, and prediction benefit does not determine pathology. |
| The method restores missing real-world occlusions. | The experiments withhold observed pose estimates or laboratory angles; they do not recover independently measured hidden motion during natural occlusion. |
| The laboratory experiment validates all four GAVD disease families. | It uses a separately fitted healthy/stroke cohort and different measurements. |
| Post-outcome intervals make the positive framing confirmatory. | They quantify conditional sampling uncertainty after the comparisons and interpretation have been inspected. |
| The negative fusion proves relational evidence cannot help. | It tests one frozen score construction and reference procedure, not every possible use of relational information. |

## Review disposition

No numerical blocker was found in the three positive headline claims. The paired audit materially improves the evidence for the masking comparison. The remaining changes are precise scientific wording: identify each comparator, disclose the post-outcome descriptive analysis in the abstract, use numerical-rank terminology, and locate the interpolation evidence correctly. The original negative primary results and failed timing gates should remain prominent. This review does not certify final pagination, venue formatting, complete bibliography verification or clinical deployment readiness.

## Follow-up verification after revision

The author applied the main wording changes. I checked the revised abstract and introduction: both-leg versus own-leg comparisons are explicit; the anatomical-versus-uniform comparison is distinct; the laboratory analysis is labeled post-outcome; the participant statement concerns lower prediction error; the rank terminology is numerical feature rank; and direct interpolation is located in the laboratory audit. The example now says that the other leg may help. The negative original primary results remain in the abstract and introduction. The reviewed revised manuscript SHA-256 is `03e2d67e4377fe425dc455d8da728b94e09c568e927116faf38e48512bdb2a11`. These wording findings are resolved. The scientific limitations and rejected claims above remain applicable.
