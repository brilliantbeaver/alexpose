**Research strategy after HAIC run 02: what should a predictive teacher teach?**

This memo connects the completed source-learning-curve result to the ICLR proposal. It recommends a research program, not a revised verdict on the completed experiments. The latest result remains `development_stop`; the cached skeleton-only panel remains `no_supported_temporal_lead`; beneficial student distillation has not been measured. Literature links below were checked during this analysis. Originality assessments are targeted comparisons, not claims that an exhaustive search has established priority.

My recommendation is to develop **selection of student-accessible temporal targets, with explicit rejection of unsuitable transfer**, and make its decisive test whether a cheap measurement predicts the benefit of training a student. The latest experiment provides a stronger motivation and a substantially expanded development cache. It does not yet validate that method. The most promising additions to the existing proposal are a finite-capacity mechanism, a reference-conditioned error-reduction spectrum, tests that separate denoising from dynamics, and evaluation of transfer decisions on held-out task configurations.

**What the latest result changes.** The [retained notebook](../../../../notebook_runs/haic-run-02/23_source_learning_curves.ipynb) and [detailed analysis](../../../../notebook_runs/haic-run-02/ANALYSIS.md) report 1,403 eligible windows from 290 recordings. Training grows from 40 to 231–233 recordings per outer fit while the evaluation cohort remains fixed. RGB predictive $R^2$ increases from 0.451045 to 0.662830. Raw teacher-unit error falls approximately 39.5%, recovered from the embedded SVG rather than the absent raw-error table. Real skeletons add only 0.000357099 $R^2$ at the endpoint; the matched validity-control increment is 0.000357856. Clip mismatch adds 0.000602912.

The real-gain interval is positive, [0.000041374, 0.000669686], but the point estimate is about 140 times smaller than the frozen 0.05 requirement. The matched increment rises approximately 0.000533397 from 40 sources to all sources; the saved marginal bootstrap fractions imply at least 96.4% positive paired growth draws. This is a small development trend, not monotonic scaling or a useful distillation effect. The all-source endpoint is fitted once per outer fold, not three times independently.

The new result weakens the explanation that the *RGB-conditioned gate* failed only because it used 50 clips. It does not establish saturation, absence of nonlinear motion information, or failure of a skeleton-only student. More sources also bring more windows, so the learning curve does not isolate diversity from volume. The 256-dimensional projection and fixed summaries can still omit useful information. Source holdout remains different from participant holdout.

The local evidence is an executed inspection notebook with a saved successful numerical-verification receipt. Full remote models, predictions, and frozen implementation were not supplied locally. The notebook's SHA-256 is `7863962ae721281254aaaffab08fe92ad404c22990e82221d2a71b98b95dd2d0`. A publication figure should ultimately read the original report and bootstrap artifacts; approximate SVG-derived MSE should be replaced with exact values when those artifacts are archived.

| Evidence | What it contributes to the paper | What it cannot establish |
| --- | --- | --- |
| Laterality: matched initialization $R^2$ 0.222544; trained conditions 0.100777–0.114209 | Feature correspondence and linear access to a particular movement observable can disagree | That motion information was erased, or that all JEPA methods fail |
| Repaired 50-window RGB gate: matched increment −0.000242417 | A numerically checked negative baseline after implementation repair | Skeleton-only transferability |
| Cached 43-source posture panel: +0.001993744, interval [−0.006672679, 0.012608431] | The student's actual information boundary has been tested, but remains uncertain | A confidence-matched coordinate-motion effect or positive student benefit |
| Latest 290-source curve: strong RGB improvement, tiny skeleton increment | Much more data helps the contextual task without meeting the skeleton utility threshold | The mechanism, the best teacher subspace, or the outcome of student training |

These are complementary observations on different questions and cohorts. They should not be pooled into a single effect estimate or presented as independent replications of the same hypothesis. The large cache includes reused parent windows and expanded data from previously seen sources.

**The central distinction must stay in the paper.** Let $R$ denote RGB/nuisance inputs, $H$ skeleton history including permitted timing and quality channels, $C$ a specified function of $H$, and $Y$ the teacher target. Three quantities differ:

$$
\begin{aligned}
\Delta_{\mathrm{RGB}}
&=\mathcal{R}(R\to Y)-\mathcal{R}((R,H)\to Y),\\
\Delta_{\mathrm{history}}
&=\mathcal{R}(C\to Y)-\mathcal{R}(H\to Y),
\end{aligned}
$$

$$
\begin{aligned}
\Delta_{\mathrm{student}}
&=\mathcal{R}_{\mathrm{motion}}(\text{student without transfer})\\
&\quad-\mathcal{R}_{\mathrm{motion}}(\text{student with transfer}).
\end{aligned}
$$

Here each risk must use a fixed outcome and metric within its comparison. A teacher can encode motion already present in RGB, making the first increment small while remaining useful for a skeleton-only learner. Conversely, an RGB–skeleton interaction may be inaccessible to that learner. Even positive history accessibility need not improve a finite-capacity encoder on its downstream task. The new run measures the first quantity; the proposed method needs the second to predict the third.

Do not train the skeleton student on residuals after an RGB reference and call them student-accessible innovations. That can remove precisely the shared motion information the student should learn. The residual reference for the proposed method must be computable from the student's inputs. Also, positive temporal increment is not universally necessary for useful distillation: supervision of current posture or a sufficient current state can help a student. Our narrow claim concerns *additional temporal supervision*, not a universal criterion for all knowledge transfer.

**A more compelling paper thesis.** A working title is **“What Should a Predictive Teacher Teach? Selecting Temporal Targets for Skeleton Students.”** The prospective claim is:

> Under limited data and encoder capacity, selecting teacher targets by their held-source predictive value beyond a declared posture/quality reference improves motion learning and identifies conditions where additional transfer should be rejected.

The current [paper draft](../manuscript/paper.md) already proposes temporal target selection, exact no-transfer, and matched-pose examples. Renaming those ideas would add little. The expansion should turn them into an explicit algorithm, a mechanism that can be falsified, and an evaluation of whether the selection rule generalizes. GAVD should motivate and stress-test that argument; it should not be the sole basis for a broad representation-learning claim.

**A simple mechanism worth testing.** Consider independent unit-variance posture $S$ and signed velocity $V$. Suppose a standardized teacher emits $q$ redundant posture coordinates and one velocity coordinate:

$$
Y=\left(\underbrace{S,\ldots,S}_{q\text{ coordinates}},V\right).
$$

A linear student with a one-dimensional bottleneck, jointly trained with a linear decoder by full-feature squared error, selects the posture direction when $q>1$. It predicts $q$ coordinates exactly, misses $V$, and achieves mean featurewise $R^2=q/(q+1)$. This approaches one as $q$ grows, while its optimal linear readout of $V$ still has $R^2$ zero. Each teacher coordinate already has variance one, so ordinary coordinate-wise standardization does not remove the example.

If $C=S$, removing the current-state conditional mean leaves $(0,\ldots,0,V)$. The same one-dimensional student can now retain $V$, giving perfect velocity readout in this idealized noiseless model. This is ordinary reduced-rank approximation, not a new theorem or an explanation already established on GAVD. It supplies an exact prediction: correlated, readily predicted state features can dominate a bottleneck even when a small motion component would be more useful.

The rank-one construction was checked directly by SVD for $q=2,9,99$: full-teacher $R^2$ was $2/3$, 0.9, and 0.99, respectively, while velocity-readout $R^2$ remained zero; the residual student's velocity-readout $R^2$ was one. These are constructed analytical values, not GAVD results.

The experiment should replace exact duplicates with controlled correlations, add observation noise, vary capacity and training sources, and independently vary appearance/camera. Include full whitening and dimension-matched compression: if they remove the deficit as well as the proposed method, the simpler explanation wins. This mechanism also exposes why improving global teacher prediction need not improve laterality or forecasting. It provides a possible connection between the two historical studies without pretending they measured the same failure.

**The strongest method candidate: a conditional error-reduction spectrum.** Start with a small, interpretable selector before adding a complicated architecture. In training-only partitions, obtain out-of-source predictions $m_C(C)$ and $m_H(H)$ in common raw teacher units. Define residuals $e_C=Y-m_C$ and $e_H=Y-m_H$, and estimate a source-balanced matrix

$$
\widehat{D}_h=
\widehat{\mathbb{E}}_{\mathrm{source}}
[e_Ce_C^{\top}-e_He_H^{\top}].
$$

For any *fixed* direction $u$, $u^{\top}\widehat{D}_h u$ is the measured squared-error reduction for predicting $u^{\top}Y$. Unlike simply taking the covariance of $m_H-m_C$, this construction charges the history predictor for its errors. Two noisy predictors can disagree strongly without either improving prediction. With exact conditional expectations and $C$ contained in $H$, the population matrix equals $\mathbb{E}[(m_H-m_C)(m_H-m_C)^{\top}]$. That positive-semidefinite identity is standard conditional-expectation algebra. The finite estimate can be indefinite; negative directions should remain visible.

A concrete selector learns leading directions subject to a fixed training-only metric:

$$
\max_{U\in\mathbb{R}^{d\times k}}
\mathrm{tr}\!\left(U^{\top}\widehat{D}_h U\right),
\qquad U^{\top}G U=I_k.
$$

$G$ can be a declared, regularized training covariance of $Y$ or of the $C$ residual. These answer different normalization questions; choose one prospectively and compare ordinary whitening as a baseline. Select $k$ from a short grid including zero, for example $\{0,4,8,16,32\}$, using inner held-source performance. These values are proposed design choices, not a frozen protocol. Use shrinkage because 256 target dimensions are large relative to the independent source count. Full covariance estimation and an unrestricted neural target mapper would be poor first steps.

The resulting auxiliary target is

$$
\begin{aligned}
t_i&=U^{\top}\left\{Y_{h,i}-\widehat{m}_C^{(-g(i))}(C_i)\right\},\\
\mathcal{L}&=\mathcal{L}_{\mathrm{skeleton}}+
\beta\left\lVert g_\theta(H_i)-t_i\right\rVert^2,
\end{aligned}
$$

where $g(i)$ identifies the source/participant group excluded from the baseline fit. Include $\beta=0$ as well as $k=0$. This is a proposed method, not implemented student training. For final training, targets use cross-fitted baseline predictions on the training groups; target-map and hyperparameter selection must also exclude each inner-validation group. Every map, scaler, rank and residual model must be rebuilt within the outer-training boundary. Do not learn $U$ on saved outer predictions and then report those same sources as independent evidence.

The spectrum gives more information than the average increment: a small mean can coexist with a small useful subspace, but only a held-out selected-subspace test can establish that. The current mean is not evidence that such a subspace exists. Projection has also already discarded information: selecting $U$ inside the cached 256 columns cannot recover directions removed by the original random projection.

Predictive subspaces and operator spectra are established ideas. [Predictive-state regression](https://arxiv.org/abs/1505.05310) and [VAMP](https://arxiv.org/abs/1707.04659) are relevant conceptual precedents; residual reduced-rank regression and partial CCA are mandatory simple alternatives. The proposed contribution would be the operational reference-conditioned target choice and its demonstrated ability to predict useful finite-data transfer, not the eigendecomposition itself.

**The selector needs a stronger reference hierarchy.** A positive history-versus-single-pose contrast can merely reflect denoising a poorly observed endpoint. It can also reflect confidence-route changes, as the previous bridge review already demonstrated. The next panel should use a hierarchy:

| Reference | Information retained | What a history improvement would mean |
| --- | --- | --- |
| $Q$ | All declared confidence, validity, transition support and timing channels, included once | More than tracking/support information |
| $Q$ plus current pose | The same $Q$ and endpoint coordinates in prefix-only normalization | More than the observed endpoint |
| $Q$ plus an order-invariant history summary | Same observations without their temporal order | More than additional unordered measurements/state denoising |
| $Q$ plus estimated current pose and velocity | A strong, prefix-only kinematic state estimate | More than the chosen local kinematic extrapolator |

Use the posture reference for a named primary temporal-accessibility question and the stronger references for mechanism checks. Do not redefine the primary comparator after seeing which wins. If velocity explains the improvement, that is useful kinematic supervision; it is not evidence of higher-order dynamics. If a sufficiently rich current state explains everything, zero residual increment is expected in a Markov system and does not mean forecasting is impossible.

A prospective panel must place raw and valid-conditioned confidence/support summaries in the common block exactly once. Lock the selected reference prediction for the primary residual-correction diagnostic, with a separately reported joint-refit sensitivity analysis. Validate the diagnostic on planted signals so that an overly restrictive reference does not manufacture a null. Remove duplicate feature routes; equal nominal penalties alone do not make duplicated representations equivalent.

Controls require care. Reordering raw history changes tracking patterns and transitions; replacing it with a donor changes support as well as motion. Use complete controlled trajectories with independently applied masks to establish the clean mechanism. For real clips, restrict coordinate controls to admissible observations, preserve the declared recipient quality information, and report common-support coverage. Do not call these observational controls pure causal interventions.

**Five creative extensions, ranked by their role in the paper.**

| Extension | New, testable question | Value and principal risk |
| --- | --- | --- |
| Source-dependent target rank | Does the amount of transferable temporal information grow with training sources, even when the global teacher curve gives poor guidance? | Directly exploits the new learning-curve infrastructure; eigenvalue selection can overfit small cohorts |
| Matched-state future branches | Can the representation distinguish histories ending at the same posture but moving toward different futures? | A clean mechanism linking time and laterality; unrealistic reversal fixtures can make success too easy |
| Teacher temporal-context audit | Is the nominal future target expressing nearby motion or depending substantially on later context? | Addresses a concrete limitation of the present cache; changing input support can itself change teacher quality |
| Observation-to-prediction decoder audit | Does a good teacher motion decoder remain useful when fed student-predicted teacher features? | Connects feature error to observable error; requires independent decoder fitting and a direct forecasting baseline |
| Distributional temporal supervision | Can history improve the predicted distribution of futures when it cannot improve their conditional mean? | Opens a distinct long-horizon direction; adds substantial estimation and evaluation demands |

**Source-dependent target rank could become the paper's distinctive empirical result.** Estimate accessibility and optimal transfer rank at several training-source counts and physical-time horizons. The hypothesis is that more sources support more useful temporal directions, not necessarily that they produce a larger average gain over RGB. Report rank-selection stability and held-source utility, not just the selected eigenvalues. Use the term learning curve or frontier, not scaling law, unless a functional law is fitted and tested on genuinely new sizes.

The current curve changes both source and window counts. Add a small feasible factorial comparison: hold the total window budget fixed while varying recordings, then hold recordings fixed while varying windows. Freeze admissible cells after an availability-only audit, not after score inspection. Keep source weighting explicit; having many windows from a recording does not turn it into many independent observations. This would tell the paper whether diverse recordings improve temporal target identification or merely overall estimation.

**Matched-state future branches should be the mechanism test.** Construct pairs with the same final posture and observation support but different incoming velocities and actual subsequent trajectories. Add a stronger challenge with similar current pose and velocity but different acceleration or phase history. Compare current pose, a denoising set encoder, constant velocity, and the proposed history representation. Beating the first comparator alone is insufficient.

In controlled data, independently cross motion identity, direction, left/right amplitude, camera, clothing and missingness. Split subjects and underlying motion sequences before rendering variants; views of the same motion cannot cross evaluation boundaries. Require prediction of subsequently generated or recorded futures, not just the last observed velocity. In real GAVD, find candidate matches from the prefix only and accept imperfect matching as observational evidence. The expanded 1,403-window cache improves the opportunity for within-recording pairs, but their number and quality must be measured; 1,403 windows is not 1,403 matched pairs.

Reflection has a precise supporting role. Retain the reflection-odd/time-even signed laterality observable and add a direction-sensitive target. Compare learned representations with exact post-hoc symmetry projection and matched initialization. An odd/even construction should not be the novelty claim: [SIE](https://proceedings.mlr.press/v202/garrido23b.html) already studies invariant/equivariant separation. Transforming video targets requires actual paired teacher encodings and alignment; changing a joint label is not a counterfactual change in pathology.

**Teacher temporal-context auditing is a useful target-construction experiment.** The present target tokens at frames 38–39 are encoded using frames 0–63. This is an explicitly declared target, not evidence of leakage into student inputs. However, “forecasting eight frames ahead” would overstate its temporal locality.

Hold the prefix and target region fixed and alter only video after the declared target cutoff. Measure target sensitivity, then compare full-context and explicitly bounded-block targets under matched student inputs. A bounded target might summarize $[t+h,t+h+\ell]$, where both the horizon $h$ and block duration $\ell$ are specified in seconds. If the encoder requires fixed input length, design the sampling/padding convention before scoring and test for distribution-shift artifacts; do not silently change attention masks in pretrained weights and assume equivalence. Measure how well each observed teacher target itself predicts motion, so an apparent accessibility improvement is not merely degradation to an easier target.

[V-JEPA 2.1](https://arxiv.org/abs/2603.14482) makes dense target extraction a plausible resource to explore. Candidate targets include bounded pooled person features, aligned body-region temporal differences, and unprojected features before the 256-column compression. Their usefulness remains an experiment. [Causal Forcing](https://arxiv.org/abs/2602.02214) is nearby work on teacher–student temporal conditioning mismatch in diffusion distillation; its ODE argument does not automatically apply to regression on video representations. The distinction and baseline should be discussed rather than claiming this general issue is new.

**The decoder audit is the strongest connection to the original laterality concern.** Train a fixed decoder $d$ from observed future teacher features to a physical motion outcome $O$ on training sources. Compare $d(Y)$, $d(\widehat{Y}_H)$, and $d(\widehat{Y}_C)$ on the same held-out sources, alongside a direct $H\to O$ predictor. The first measures target expressivity; the second and third assess whether accessible prediction preserves that useful content. Residual-code experiments must reconstruct the appropriate common target before applying its decoder, or fit the corresponding decoder prospectively.

For a fixed linear decoder $A$ and target-prediction error covariance $\Sigma_e$, the decoder's teacher-to-prediction discrepancy is $\mathrm{tr}(A\Sigma_e A^{\top})$. A small unweighted $\mathrm{tr}(\Sigma_e)$ need not imply a small discrepancy in the directions $A$ uses. This is a standard matrix identity, not a new theorem. Nor is it automatically equal to downstream prediction risk against $O$, because the observed-teacher decoder has its own error and cross term. Score the actual $O$ outcome. If direction selection uses $O$ or $A$, label the method motion-supervised and give every comparator the same label budget; it cannot retain a task-free claim.

**Distributional supervision is a separate, higher-risk branch.** Suppose $H$ determines the amplitude $a$ of a future displacement but not its sign, with $Y=\pm a$ equally likely. If $C$ omits $a$, $\mathbb{E}[Y\mid H]=\mathbb{E}[Y\mid C]=0$, so conditional-mean improvement is zero even though $H$ predicts the variance and the future distribution. This gives a precise alternative to treating all small MSE increments as inaccessible information.

Test this first on controlled trajectories, then with a heteroscedastic Gaussian or small mixture before a diffusion model. Use proper distributional scores and coverage on actual future motion; sample-based “best of many” error can reward producing more guesses without improving calibrated prediction. This would be a different research question requiring a new protocol. The current run supplies no evidence that distributional uncertainty explains its result, and generic probabilistic forecasting is not a novelty claim. Keep this branch outside the main method unless a mean-versus-distribution comparison provides a clear result.

**Novelty must survive the strongest nearby comparisons.**

| Prior work | Already established | What our work would have to add |
| --- | --- | --- |
| [Modality Focusing Hypothesis](https://zihuixue.github.io/MFH/index.html), ICLR 2023 | Crossmodal transfer depends on useful shared information; stronger teachers need not help | A temporal, reference-conditioned rule that predicts trained-student effects in new regimes |
| [DFA](https://arxiv.org/abs/2008.00506), 2020 | Searches teacher feature aggregation for expressivity and student learnability | Incremental value beyond current state/quality, source-size dependence, and robust rejection |
| [Student Customized KD](https://openaccess.thecvf.com/content/ICCV2021/html/Zhu_Student_Customized_Knowledge_Distillation_Bridging_the_Gap_Between_Student_and_ICCV_2021_paper.html), ICCV 2021 | Adapts transfer using student benefit and gradient similarity | A pretraining target selector tied to declared temporal information and observable forecasting |
| [C²KD](https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html), CVPR 2024 | Customizes crossmodal knowledge, addressing imbalance and soft-label mismatch | Temporal target/rank selection rather than a generic crossmodal customization claim |
| [Overlooked Poses](https://arxiv.org/abs/2208.01302), ECCV 2022, and [SGDD](https://openreview.net/pdf?id=P6F4MxtOKp), ICLR 2026 | Future privileged information for motion/dynamics distillation; SGDD adds spectral guidance | Evidence that choosing which future content to transfer improves over strong future-supervision alternatives |
| [VAMP](https://arxiv.org/abs/1707.04659), predictive-state regression, partial CCA | Low-dimensional predictive dynamical representations | The conditional target criterion and empirical transfer prediction; the spectrum is not itself novel |
| [LogME](https://proceedings.mlr.press/v139/you21b.html), ICML 2021 | Cheap assessment of model transferability without full fine-tuning | Selection of temporal supervision and prediction of its incremental effect; compare with label-budget-matched scores |
| [Learn then Test](https://arxiv.org/abs/2110.01052) | Risk control for selection/calibration with appropriate assumptions | Any claimed no-harm result must instantiate valid independent calibration, not rename baseline fallback |

The SGDD comparison here uses its indexed primary-paper abstract and official code; direct full-paper access encountered an OpenReview browser check. A detailed implementation comparison remains necessary before finalizing the benchmark. Similarity to these methods is a reason to sharpen the experimental claim, not to omit the nearest work.

An exact rank-zero option does not guarantee that the procedure never harms a student: noisy selection can still choose a harmful nonzero candidate. A finite-sample guarantee, if pursued, needs a fixed candidate set, independent source/participant calibration data, a bounded or otherwise justified loss, and appropriate multiplicity control. An accessibility guarantee on teacher targets would still not be a guarantee on downstream motion. This is an optional extension; the main paper can make an empirical abstention claim instead.

**The next experiment should answer the cheapest unresolved question.** Expand a corrected skeleton-only accessibility panel onto the 290-source cache. Reuse the frozen availability/reservation and outer source boundaries, but create a new versioned protocol and output root. This experiment asks a different question from the completed RGB gate; its rules must be declared prospectively. Preserve the original 0.05 decision and all prior artifacts.

The existing [scaling orchestrator](../../../../src/gavd6_sjepa/research_directions/source_scaling/nested_selection.py) provides source-group nesting and size-normalized penalties. The [cached bridge panel](../../../../src/gavd6_sjepa/research_directions/target_accessibility/cached_panel.py) provides the student-input reference construction, but its confidence-route overlap must not be carried forward as a coordinate-only comparison. A new module should reuse the stable contracts and add a corrected schema rather than modify sealed historical implementations.

Run one prespecified corrected linear panel first, with exact fallback and a small model-capacity sensitivity check. Evaluate person and background contextual targets separately—the [expanded cache code](../../../../src/gavd6_sjepa/research_directions/source_scaling/data.py) already stores both. Background accessibility is a diagnostic of what may be explained by context/quality, not a pure negative-control guarantee. This first stage needs original cache artifacts on HAIC or a verified local copy; the inspection notebook alone is insufficient.

The implementation extracts skeletons for the observed 32 frames only. There is no hidden full future-pose dataset waiting inside these cached skeleton arrays. Observable future outcomes require new pose extraction or a dataset with ground-truth motion. New bounded, reflected, dense, or unprojected teacher targets likewise require new encoding. These are concrete work items, not alternate plots of the existing notebook.

| Stage | Required work | Decision it resolves |
| --- | --- | --- |
| 1. Archive and corrected cache panel | Full run evidence; quality-matched skeleton references; source learning curve | Does more data reveal skeleton-only temporal accessibility? |
| 2. Observable targets | Timestamped future pose or motion capture; current-state/velocity baselines; teacher-to-motion readouts | Does the teacher represent motion that the student can actually forecast? |
| 3. Conditional target selector | Low-rank error-reduction directions; shrinkage; source-nested selection; PCA/CCA alternatives | Is there stable, useful target content beyond easier compression? |
| 4. Matched student training | Same encoder, sources, label budget, updates and downstream readout across methods | Does target selection improve an actual student? |
| 5. Transfer-decision generalization | Freeze scores/rules; evaluate new target/teacher/task configurations and independent people | Does the rule predict benefit outside the configurations that motivated it? |

A corrected null at stage 1 would prioritize target construction or model-family diagnosis, not automatically a larger student. A positive panel followed by no student gain falsifies the proposed accessibility-to-transfer hypothesis in that regime. If direct motion supervision matches or beats the selective teacher at equal data/compute, the teacher is not justified for that task; the paper must either show a distinct benefit under sparse labels/shift or narrow to an evaluation contribution. If simple residual CCA matches the full method, prefer that simpler method and make the empirical insight explicit.

**A credible main experiment should measure transfer decisions, not only selected-code loss.** Predeclare a modest family of teacher targets, two physical horizons, and limited source-count regimes. Screen with cheap predictors, then train students for enough favorable and unfavorable cases to test whether the score predicts their actual changes. Do not train only the cases selected as promising: that makes false negatives and abstention quality unmeasurable. If subsampling the candidate set, choose it prospectively and explain the sampling policy.

Compare the selector with global teacher predictive $R^2$, unconditioned skeleton predictability, fixed-rank PCA/whitening, residual reduced-rank regression/partial CCA, and a task-supervised score when equal labels are available. Assess ranking of actual transfer gains, frequency of harmful transfer, realized gain at a fixed selection budget, and regret relative to an oracle choosing among the evaluated candidates. Treat teacher layers and nearby ranks as correlated configurations, not independent datasets. Hold out a teacher family or a dataset when assessing whether the decision rule generalizes.

For trained students, retain these comparisons under matched opportunity:

1. Matched initialization and ordinary skeleton pretraining.
2. Direct kinematic/future-motion supervision and a strong motion-learning baseline.
3. Full teacher-feature distillation.
4. Current-state residual distillation without direction selection.
5. Equally small targets chosen by PCA/whitening and residual CCA, plus a frozen random projection.
6. The proposed conditional selector, with and without its abstention rule.

Count selector training, teacher encoding, auxiliary heads and hyperparameter search in compute comparisons. On compute-intensive candidates, use a shared small pilot policy and freeze finalists before the main evaluation. Include multiple actual neural-training seeds; unlike the ridge subset labels, these assess optimization variability. Source/participant uncertainty and optimization uncertainty should be reported separately. Resample/refit the selection pipeline where feasible or use repeated independent splits; a bootstrap of saved predictions alone does not cover adaptive target learning.

**Data should give GAVD a specific role.** [Human3.6M](https://vision.imar.ro/human3.6m/pami-h36m.pdf) provides synchronized images and motion capture for a clean geometry/forecasting comparison, but its small participant count must remain visible. [AMASS](https://amass.is.tue.mpg.de/) provides a broader motion-capture resource for controlled renderings and varied motions; it is not an off-the-shelf synchronized real-video/pose substitute. Split subjects and underlying source motions before rendering and audit constituent-dataset overlap. GAVD then tests noisy real-world observations and recording shift. This combination is more persuasive than additional windows from the same inspected videos alone.

Use physical units only where calibration supports them. GAVD's image-derived normalized coordinates should not be reported as metric 3D errors. Preserve timestamps and ensure no future frames influence prefix normalization, imputation, tracking state or pair selection. Differences in joint layouts need an explicit common anatomical mapping. Motion targets generated with the same pose estimator as the input also warrant a ground-truth or independently measured check so detector consistency is not mistaken for physical prediction.

The 43 reserved GAVD confirmation recordings remain reserved; the latest availability counts imply only 41 have available media, and eligibility is still unknown. Do not use this set to search targets, rank thresholds, or reference definitions. A fixed final evaluation can use it, but unknown participant overlap prevents a stronger participant-independent claim. Independent motion-capture participants or genuinely new data are needed for that part of the argument.

**How to use the new evidence in the manuscript.** Add the source-learning curve after the repaired RGB gate as a separate development study. Plot RGB and augmented prediction, the matched increment with its own scale, and exact raw MSE once recovered. A small inset or annotation should show the frozen 0.05 threshold; a zoomed increment axis alone makes the effect look larger than its prospective importance. Show mismatch explicitly.

A suitable factual paragraph is:

> In an expanded available-development cohort of 1,403 windows from 290 recordings, increasing outer-training size from 40 to 231–233 recordings improved RGB predictive $R^2$ from 0.4510 to 0.6628. The largest-endpoint skeleton increment was positive but small: real-minus-RGB was 0.0003571, with a conditional source-bootstrap interval [0.0000414, 0.0006697]. This failed the prospectively specified 0.05 effect threshold. Clip mismatch produced a larger point gain of 0.0006029. The comparison strengthens the motivation to study target suitability and the student's information boundary; it does not establish a failure or benefit of skeleton-only distillation.

The new main figure should eventually juxtapose the motivating curve with the *actual student* utility curve and the selector's target-rank decisions. The latter two do not yet exist and should not be filled with simulated values labeled as outcomes. A separate, explicitly theoretical schematic can show the redundant-posture counterexample. The laterality result supplies a secondary observable-preservation test, while source-independent future motion supplies the primary method outcome.

For the existing eight-page draft target, budget roughly one page for the evidence and question, one for related work and the counterexample, two for the selector and controlled mechanism, two for real student comparisons, one for transfer-decision generalization/ablations, and one for limitations and reproducibility. Detailed historical debugging belongs in the appendix. Do not make the manuscript an omnibus collection of negative experiments, symmetry identities and speculative architectures.

**My assessment of the paper paths.**

| Paper path | What would make it convincing | Assessment |
| --- | --- | --- |
| Conditional temporal target selection | Trained-student gains beyond residual CCA and direct motion; correct rejection; independent validation | Strongest recommended method direction |
| Predictability versus motion-retention benchmark | Consistent dissociations across representation families, datasets, observables and readout opportunities | Credible alternative if transfer fails; must generalize beyond current local runs |
| Temporal-context audit of video teachers | Broadly reproduced mismatch between nominal target horizon, actual context dependence and forecast utility | Promising focused paper if target dependence is substantial and consequential |
| Distributional temporal distillation | Mean-based tests miss calibrated, useful distributional information in real motion | Interesting higher-risk project; not the immediate main experiment |
| More data or another masking/reflection loss alone | A mechanism and independent utility evidence beyond existing methods | Insufficiently distinctive as presently specified |

The strongest eventual result would be a double dissociation: improving global teacher prediction fails to improve motion, while a selected temporal target improves motion despite lower full-teacher reconstruction—and a rule learned on development configurations predicts both cases on new ones. That outcome is a research hypothesis, not a reinterpretation of the latest STOP. The immediate concrete step is the corrected expanded skeleton-only panel and acquisition of independent future-motion outcomes. Those measurements determine whether selective distillation earns the center of the paper.
