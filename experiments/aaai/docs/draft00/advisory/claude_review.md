# Claude advisory review

Requested model: Opus. Reported model identifiers: global.anthropic.claude-opus-4-8. This is advisory reasoning, not new experimental evidence.

# Advisory brief on the GAVD ASTAD submission

The draft review is unusually disciplined. It resists the two failure modes that sink most workshop papers written under a six-day gun: it does not pretend the trained model won, and it does not reach for clinical language the data cannot support. I am going to push against it anyway, because its central weakness is that it is *correct but unmemorable*. It converges on "run a clean controlled study, narrow the claims, maybe report a failure." That is a defensible paper and a forgettable one. The question the coordinating author should sit with is not "is this rigorous" but "will a reviewer remember the finding in a month." Below I try to find a version that is both.

## 1. Three paper concepts, ranked

### Concept A: A negative-result benchmark with a diagnosis, not just a null

**Central claim.** Under a controlled optimization budget, latent predictive discrepancy on GAVD skeletons carries no more usable abnormality signal than an untrained encoder, and we can attribute the specific *reason*: the discrepancy score is dominated by pose-observation quality and static posture rather than temporal coordination. The contribution is the attribution, demonstrated by interventions that move the suspected cause while holding the comparison fixed.

**Difference from prior art.** Asleep at the Wheel (arXiv 2608.01336) and the physical-surprise critique (arXiv 2611.01815) warn in general terms that latent prediction error confounds nuisance. A gait-specific, intervention-based decomposition on a real clinical-adjacent dataset is not the same as repeating the warning. The novelty is a *method of attribution* plus the specific finding, not the discovery that confounding exists.

**GAVD asset it uses.** The 450 retained sequences, the pose-quality metadata that already drove 32% attrition, the source-grouped splits, and the mean-pose and pose-availability controls that already exist in the notebooks. Crucially it uses the attrition itself as signal: which conditions and views were excluded.

**Decisive experiment.** Hold the teacher, data, and masks fixed. Compute the raw discrepancy AUROC, then (i) regress the score against a pose-quality covariate and report the residual AUROC, (ii) compare the ordered-context predictor against the orderless-context predictor described in the roadmap, and (iii) evaluate on a prespecified high-quality subset. If residualizing quality collapses the score toward chance and the orderless control matches the ordered one, the attribution holds.

**Failure criterion.** The attribution fails if quality residualization leaves the AUROC essentially unchanged *and* the ordered context beats orderless, because then the signal is real movement information and the "it's just nuisance" story is wrong. That would actually be a more interesting result, so this concept has an attractive property: both outcomes are publishable.

**Feasibility in one week.** High. Every ingredient exists. The orderless control is the only new code and it is a pooling change plus a permutation-invariance unit test, both specified in the roadmap.

### Concept B: The calibration-honesty contribution

**Central claim.** On uncontrolled web video of pathological gait, source-level conformal normal-only detection is *infeasible at a 5% operating point* because the number of independent normal sources is an order of magnitude below the finite-sample minimum, and we quantify exactly how much normal-source diversity GAVD would need to support a usable threshold.

**Difference from prior art.** STG-NF, FSGait, and GenGait all report operating points on datasets with enough normal subjects. Nobody has published the uncomfortable arithmetic for an in-the-wild clinical gait corpus: that with 5 or 6 normal calibration sources the finite-sample 5% threshold is literally infinite, and that 19 is a floor, not a target. The contribution is a reusable evaluation protocol and a concrete diversity budget.

**GAVD asset it uses.** The newly verified census is the whole paper. 291 normal sequences but only 32 normal source IDs, and after quality filtering 5 or 6 per split. The 245 out-of-cohort sources with no normal labels. This is the one place where the fresh scan is load-bearing.

**Decisive experiment.** Sweep the number of normal calibration sources from 2 upward, plot threshold finiteness and sensitivity against source count, and show where a defensible operating point first becomes attainable. Repeat with sequence-level versus source-level units to show the inflation that treating windows as independent would produce.

**Failure criterion.** It fails as a standalone paper if the arithmetic is the entire content, because the finite-sample rank formula is textbook. It needs the GAVD-specific diversity budget and a demonstration that common evaluation choices silently violate exchangeability to rise above a note.

**Feasibility in one week.** Very high, almost too high. The risk is that it is thin.

### Concept C: Observation-uncertainty versus movement-uncertainty separation

**Central claim.** A model with distinct observation-quality and movement-deviation outputs separates a poorly tracked ordinary gait from a well-tracked abnormal gait better than any single confidence-weighted residual.

**Difference from prior art.** SeeKer already weights by keypoint confidence; the stochastic-latent JEPA preprint already motivates learned predictive uncertainty. The claimed novelty is the *identifiable separation* of the two uncertainty sources.

**GAVD asset it uses.** This is the problem. It needs paired observations or independently checked keypoints, which the census does not provide. The 213 multi-view sources are *not verified* as the same person.

**Decisive experiment.** Train both heads, degrade observation while holding movement fixed, and show the two heads move independently.

**Failure criterion.** Fails if the movement head rises with tracking error.

**Feasibility in one week.** Low. The roadmap scores it 3/10 and that is honest. Coordinate noise is a controlled skeleton experiment, but it does not reproduce real pose-estimator failure, and without validated references you cannot demonstrate *identifiability*, only internal consistency.

**Ranking: A, then B, then C.** A is the only one that is both decisive and feasible. B is a strong *section* of A or a safety net if A's interventions do not run in time. C is a research program, not a six-day paper, and the draft already says so.

## 2. The strongest direction and where the draft aims wrong

Pursue a fused A-with-B-inside. The draft's instinct to make this "a controlled study" is right, but as written it aims too low in one specific way: it treats the negative result as a fallback ("if no specific finding is supported, narrow to an exploratory report") rather than as the headline. A negative result is only forgettable when it is a bare null. It becomes memorable when it is *explained* and when the explanation generalizes beyond your dataset. "Briefly trained JEPA lost to an untrained encoder" is a null. "On in-the-wild clinical gait, latent predictive surprise is a pose-quality detector wearing a motion-model costume, and here is the intervention that unmasks it" is a finding. The draft has all the parts for the second sentence but keeps filing it under the first.

Does the draft ask for evidence it cannot get this week? In two places, yes. It repeatedly gestures at an "independent participant-identified cohort" as the thing that would "materially increase impact." You do not have one, you are told not to collect one, and chasing it would burn the week. Drop it from the critical path and name its absence as a scope boundary, not a deficiency. Second, the "sufficient optimization" framing (the 64/256/1024/4096 update ladder) is essential as a *control* but dangerous as a *goal*. If the author treats "train longer and win" as the target, a laptop budget will not deliver convergence in six days and the paper collapses. The ladder's job is to show the result is *not merely* a 45-update artifact, so that the attribution to nuisance is credible. Frame it as a robustness control, not a performance chase.

One mechanistic idea where I think it is justified. The draft's orderless-context predictor is the right instrument, but its most load-bearing confound is that removing temporal order between patches also removes the *temporal pattern of missingness*, and GAVD missingness is almost certainly condition-correlated: more severe or more occluded gaits drop more keypoints. The clean mechanistic test is to make the orderless and ordered predictors receive *identical visibility summaries* and then evaluate on a matched-coverage subset where the per-joint missingness rate is binned equal across normal and abnormal. If the ordered-context advantage survives equal-coverage matching, it is movement; if it evaporates, your "surprise is really missingness structure" story is sharp and specific. This is the single most publishable sub-finding available to you, because missingness-as-confound in skeleton JEPA is less picked-over than generic nuisance.

## 3. The decisive experiment matrix

Keep normal-only detection and supervised classification in separate tables, with separate baselines, and never let a probe trained on all five labels be described as unsupervised.

**Normal-only detection (source-level evaluation unit, source-clustered uncertainty).**

| Model | Role |
| --- | --- |
| Normal embedding distance (existing) | Current score, baseline |
| Masked latent discrepancy, ordered context | Main object under test |
| Masked latent discrepancy, orderless context, matched visibility | Nuisance control |
| Quality-residualized discrepancy | Attribution test |
| STG-NF adapted to 16 joints | External normal-only baseline |
| SeeKer-style confidence density | External normal-only baseline |
| Coordinate-reconstruction error | Isolates value of latent targets |

The STG-NF and SeeKer adaptations are the two items most likely to overrun. If one must be cut, keep STG-NF, since a normalizing-flow normal-only baseline is exactly what a reviewer will ask for and its absence is more damaging than a missing confidence density.

**Supervised five-class (source-weighted macro-F1 primary, sequence-weighted secondary, paired uncertainty computed for the primary weighting).**

| Model | Role |
| --- | --- |
| Random Forest | Supervised reference |
| Untrained encoder + probe | The control that currently wins |
| Normal-pretrained JEPA + probe | Pretraining-benefit test |
| Mean-pose probe | Static-posture floor |
| Pose-availability/view probe | Nuisance floor |

**Evaluation-unit policy.** Fix the source as the unit before touching test data. The draft's catch about source-weighted headline numbers carrying sequence-weighted bootstrap intervals is correct and must be fixed; it is currently an apples-to-oranges uncertainty statement that a sharp reviewer will flag.

**Inspected-test-data policy.** All three splits are burned for confirmatory purposes. Treat the existing descriptive results as frozen. Every hypothesis formed after inspection (the orderless control, quality residualization, the missingness matching) is *exploratory* and must be labeled so in the paper. Do not re-partition the same sources and call it fresh. The 245 out-of-cohort sources are *not* a clean confirmatory set either, because they are "annotation references, not guaranteed usable media," and some may have been touched by earlier local projects. There is no untouched confirmatory set available this week, so the honest move is to present everything as exploratory analysis on a fixed cohort and say so plainly. That is not a weakness a reviewer punishes; concealing it is.

**What makes a negative finding informative rather than an artifact of bad training.** The negative result is scientific only if the update-budget ladder shows the null persists across budgets *and* the orderless/quality interventions localize the cause. A null at 45 updates alone proves nothing. A null that is stable from 64 to 4096 updates, plus a demonstration that the score tracks pose quality and not temporal order, is a mechanism. The decision rule: if residualizing quality collapses the AUROC toward 0.5, you have the finding regardless of absolute training quality, because you are explaining the *structure* of the score, not its magnitude.

## 4. The most serious confounds, and inconclusive versus excluded

**Missingness structure.** Addressed above; the single most important and most underexamined confound. Condition-correlated keypoint dropout can masquerade as every "signal" you find.

**Independent-people count.** 91 sources, no verified identities, one Parkinsonian test source in two seeds. This is a hard ceiling. No bootstrap manufactures people. Any claim that generalizes across patients is excluded this week.

**Attrition selection.** 32% of sequences dropped on quality. If drop rate correlates with condition or severity, your retained cohort is a biased sample and the whole detection task may be artificially easy or hard. The draft flags this; make it a figure, not a sentence.

**The inconclusive-versus-excluding distinction.** This is where the draft is sharpest and the author must hold the line. A paired interval that spans both a meaningful gain and zero is *inconclusive* and must be reported as such, not spun as a trend. An interval that lies entirely below your prespecified minimum useful gain *excludes* that gain and is a positive scientific statement. Prespecify the minimum useful gain before looking at test numbers, or the distinction evaporates. With 5 or 6 normal sources, expect most detection intervals to be inconclusive; say so without apology.

**Literature flags.** I did not and cannot verify these from here, and several carry markers of possible fabrication or drift that the author must check against the actual PDFs before citing: the two JEPA-critique preprints (arXiv 2608.01336 and 2611.01815) have arXiv IDs whose month fields (2608, 2611) correspond to Aug and Nov 2026, which is plausible for very recent work but means they are unreviewed preprints and should be cited as such; "GenGait" (arXiv 2604.01997 / 2604.01997v2) and "GaitJEPA IJCB 2026" and the stochastic-latent JEPA preprint (2611.09302) are all 2026 and all need the author to confirm they exist, say what is claimed, and are not hallucinated IDs. The multimodal GAVD paper DOI (10.1007/s44163-026-02522-3) likewise needs a direct check. Treat every 2026 citation as unverified until the author opens it. The older anchors (Skeleton S-JEPA ECCV 2024, MAMP, VICReg, StoP, STG-NF ICCV 2023, SeeKer ICCV 2025, Ren et al. NeurIPS 2019, Angelopoulos and Bates) are more plausibly real but still deserve a glance. Do not let any citation into the paper that the author has not personally opened.

## 5. Six-day plan with decision points

**Nov 14 (today, remaining).** Fix the paper question to Concept A fused with B. Audit a stratified sample of frame-timing and person-assignment on development sources. Compute per-condition and per-view attrition rates. *Decision:* if attrition is strongly condition-correlated, the cohort-bias figure becomes a core limitation and possibly a result; if timing/person alignment is materially wrong, fix and disclose before any model interpretation.

**Nov 15.** Run the update-budget ladder on development data and inspect teacher-to-online drift. Implement and unit-test the orderless-context predictor with the permutation-invariance check. *Decision:* if the null persists across budgets, proceed with the attribution paper; if training longer closes the untrained-encoder gap, the paper pivots to "the failure was the recipe" and that pivot must happen now, not on Nov 18.

**Nov 16.** Run the quality-residualization and matched-visibility/matched-coverage experiments. Stand up STG-NF on the 16-joint topology. *Decision:* if quality residualization collapses the AUROC, lock the attribution finding as the headline. If STG-NF does not run cleanly by end of day, cut it and note its absence rather than ship a broken baseline.

**Nov 17.** Finalize the one chosen mechanism (ordered-vs-orderless under equal coverage). Compute source-weighted macro-F1 with *matching* source-weighted paired intervals, replacing the mismatched sequence-weighted ones. *Decision:* if no intervention produces a sharp attribution, demote to the calibration-infeasibility paper (Concept B), which is still a real contribution and is already nearly complete.

**Nov 18.** Assemble the attrition figure, the normal-source-diversity-versus-threshold sweep, the matched-coverage comparison, and a small honest failure gallery. Freeze all numbers. *Decision:* if the intervals are all inconclusive, write the inconclusive result cleanly; do not fish for a reversed score or a lucky seed.

**Nov 19.** Write the seven pages. One page question plus closest prior art, one cohort and protocol, one model and the attribution instrument, two controlled experiments, two failure analysis and limitations. Verify every numeral against the saved artifacts. Check anonymity for double-blind.

**Nov 20.** Final source and citation check, confirm the submission-system cutoff timezone (the workshop page does not state it), submit with margin.

One closing caution. The gravest risk here is not rejection; it is writing a paper that is competent and invisible. The draft's reasonable-sounding plan is still incremental if it ships "we ran a careful study and the trained model did not help." The thing that makes this memorable is the attribution plus the missingness-structure confound, delivered as a reusable lesson for anyone applying latent-prediction anomaly detection to messy clinical video. Aim the whole week at that one sentence.
