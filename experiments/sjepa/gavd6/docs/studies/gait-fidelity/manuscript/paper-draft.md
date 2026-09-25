# Evaluating Feature Prediction for 2D Pose Trajectory Restoration with Paired Synthetic Supervision

**Working manuscript, 24 September 2026. Not submission-ready.** The title is the author's registered title. The follow-up numbers below come from the rounded means supplied by the author, not a locally verified follow-up export. Resolve every `TODO` before submission. Use the official ICLR 2027 LaTeX template for the submitted PDF. This draft develops the expanded gait-fidelity study; it does not combine its population or measurements with the initial eight-person pilot described in the registered abstract.

## Abstract — provisional

Pose restoration should correct joint positions while preserving movement. We investigate what predicting reference features adds beyond direct coordinate training under paired synthetic supervision. Estimated poses from rendered movement are paired with projected body-model references, and controlled movement changes are crossed with observation perturbations. We compare direct restoration, coordinate and joint-embedding predictive pretraining, and initialized and shuffled-reference controls on a development cohort of 14 people across three training seeds. A follow-up trains feature-difference, endpoint-feature and coordinate-difference objectives, then fits two readouts to each frozen encoder. Direct coordinate restoration obtains the strongest reported means across response, nuisance, coordinate, waveform and direction metrics. Feature-difference JEPA reduces mean response error by 0.373 degrees relative to endpoint supervision with a change-supervised readout, but increases it by 0.712 degrees with a coordinate-only readout. Across all eight evaluated model families, adding movement-change supervision increases coordinate and waveform error under the fixed protocol. These development results expose a dependence of apparent representation benefit on downstream supervision and demonstrate the need to evaluate scalar movement responses together with their supporting trajectories and prediction reliability. The experiment does not establish independent or clinical generalization.

**TODO: Update the abstract after reading the primary and interaction intervals and failure decomposition. Retain descriptive language if uncertainty is unresolved.**

## 1. Introduction

Learned pose restoration is often judged by how well its outputs match reference coordinates. In movement analysis, those outputs support additional operations: comparing two conditions, estimating a limb's angular excursion, or deciding whether an apparent response reflects movement rather than changed visibility. A training objective can improve one of these measurements while changing other properties of the trajectory. Evaluating the intended response and the geometry that supports it is therefore necessary to understand what a representation contributes.

Predictive representation learning offers a plausible route to useful motion features. A student encoder learns from partially observed trajectories, predicts features of a reference sequence, and supplies a frozen representation to a supervised coordinate readout. Yet the readout can determine which aspects of the representation affect the final measurement. Direct access to observed coordinates through a residual connection further means that useful reconstruction need not demonstrate useful pretraining. These possibilities motivate matched readout objectives, initialized-encoder controls, and a strong direct-restoration baseline.

We ask whether additional feature-difference supervision during predictive pretraining improves recovered movement changes beyond independent endpoint supervision, and whether that comparison is stable across downstream objectives. We use a controlled synthetic gait procedure in which registered movement edits are crossed with observation conditions. References are defined separately for each camera, and evaluation uses fixed reference support. The target response is the change in a signed right-minus-left knee-excursion measurement.

The completed core experiment motivates a follow-up with three pretraining interventions and two readouts for each frozen encoder. The new comparison preserves the original primary endpoint: feature-difference JEPA versus endpoint-supervised JEPA under the paired-change readout. Base-readout controls and the interaction were specified after inspecting the core and before inspecting the follow-up outcomes. Both stages use the same development population; the follow-up is not independent confirmation.

The reported means reveal three observations. First, the paired-change readout increases waveform and coordinate error in all eight evaluated model families. Second, the delta-versus-endpoint response comparison reverses between readouts. Third, direct coordinate restoration remains the strongest practical baseline across the five reported metrics. Our contribution is a controlled evaluation of these relationships under the tested protocol, together with explicit accounting for prediction failures and person/seed uncertainty.

**TODO: Insert verified uncertainty for the central findings and revise the strength of each sentence accordingly.**

## 2. Related work and scope

S-JEPA studies masked predictive skeleton representations for action recognition [1]. Our experiment uses privileged clean projected reference trajectories during training and evaluates coordinate restoration and physical-response measurements. It is an application-specific predictive architecture inspired by this line of work, rather than a claim that action-recognition findings transfer to gait measurement.

Supervising changes has precedents. Sobolev training incorporates derivative information alongside target values [2]; our registered finite differences are not automatically derivatives with respect to a physical variable. The added difference loss and its algebraic expansion are experimental instruments, not standalone novelty claims. Pose-refinement methods such as SmoothNet [3] motivate assessing trajectories as well as coordinates. AMASS supplies the underlying motion/body-model source [4]; it does not provide independent clinical references for this experiment.

**TODO: Verify and complete a focused bibliography against the original papers. Add the closest movement-aware representation, equivariance, and pose-refinement papers only where their actual methods bear on the claim. Explain the distinction from each close precedent; do not use a broad literature catalogue as a substitute for positioning.**

## 3. Measurement and controlled observations

Let an endpoint be a sequence window from one movement state. For each limb, image-plane hip-knee-ankle angles are evaluated on a fixed set of reference-supported timestamps. For limb l, define its excursion as the 95th percentile minus the 5th percentile of those angles, using the saved percentile convention. The signed bilateral measurement and the paired movement response are

$$q_l=P_{95}(\theta_l)-P_5(\theta_l),\qquad A=q_R-q_L,\qquad \Delta A=A_b-A_a.$$

Given restored trajectories, the response error is

$$E_\Delta=| (\widehat A_b-\widehat A_a)-(A_b-A_a) |.$$

Movement contrasts hold observation and camera realization fixed. Nuisance contrasts hold movement fixed and compare the reference-corrected measurement change under altered observation conditions. Each camera has its own projected reference; a change in projection is not assumed to leave a two-dimensional angle unchanged.

We also evaluate coordinate normalized landmark error, full angular-waveform error, and response-direction accuracy. The report's coordinate column is all-valid-synthetic-joint NLE, not visible-joint NLE. Direction accuracy retains reference-resolvable responses using the saved tolerance and counts failed predictions as incorrect. Small true responses remain in the primary absolute-response score; no-change controls are reported separately.

Invalid predictions remain in the reference-eligible denominator. The inherited response scoring penalty is 720 degrees, which is a scoring cost rather than an anatomical measurement. At the same aggregation hierarchy, response error decomposes into the contribution of successful predictions plus 720 times the weighted failure fraction. We report both terms, conditional error on successful predictions, and a zero-response benchmark.

**TODO: Insert exact support thresholds, normalization definition, sign tolerance, failure counts and coverage from the frozen configuration and exports. Include the separate penalties for level, waveform and nuisance metrics.**

### A response loss leaves common errors unconstrained

Let e_i = Ahat_i - A_i. Then response error is |e_b-e_a|. Equal nonzero errors at the two endpoints cancel in this contrast. More generally, two trajectories can share an excursion statistic while differing in temporal waveform. These elementary observations motivate evaluating levels, waveforms and coordinates beside the scalar response. They do not establish that learned common-error cancellation caused an observed gain.

## 4. Models and controlled interventions

The core includes direct coordinate restoration; coordinate and paired predictive pretraining with frozen coordinate readouts; and initialized-encoder and shuffled-reference controls. Each family is evaluated with a base coordinate objective and with the inherited additional paired-change term. The direct model is optimized end to end, whereas pretrained models use frozen encoders. It supplies a practical reference but is not an architecture-matched intervention on pretraining alone.

The follow-up adds feature-difference JEPA, independently supervised endpoint JEPA, and coordinate-difference pretraining. It trains each variant with seeds 17, 29 and 43. Every resulting encoder checkpoint is shared by two readouts, initialized and trained under the registered matched procedure: coordinate-only base and paired-change. The follow-up therefore has nine pretraining phases and eighteen readouts, while the retained core supplies thirty fitted readouts/restorers across the five earlier families and two objectives.

For a common queried token, let e_i be the centered, temperature-scaled student-predictor residual relative to a detached reference-teacher feature at endpoint i, of dimension D. The two JEPA auxiliaries are

$$L_\Delta=\frac{\|e_b-e_a\|^2}{2D},\qquad L_E=\frac{\|e_a\|^2+\|e_b\|^2}{2D}.$$

On the same tensors,

$$L_\Delta=L_E-\frac{e_a^\top e_b}{D}.$$

The comparison matches the support intersection and the training-calibrated auxiliary coefficient. This identity isolates a difference in loss definition; separate training trajectories can still produce different moving-average teachers. A coordinate-difference arm supplies a control outside latent prediction and expresses the two residuals in a common observation-derived scale.

The encoder consumes available observations, masks/confidences and timestamps through the inherited adapter. Reference coordinates and pair labels supply privileged training supervision but are absent from the deployed restorer. The teacher and predictor are discarded for the frozen-encoder readout deployment. The readout also has the inherited residual pathway from observed coordinates, which motivates the initialized control.

**TODO: Add exact architecture, parameter counts, mask policy, optimizer, learning rates, loss coefficients, auxiliary calibration and deployment normalization from the frozen child and bound parent configs. State that a broad loss-weight sweep and a new pretraining re-pairing control were not performed.**

## 5. Experimental protocol and uncertainty

The follow-up reuses the completed core's prepared AMASS-based bundle, split identities, renderings, pose estimates and reference geometry. No new patients or real-video reference population enter the child. The retained core evaluation includes 14 development people, with 12 from BioMotionLab_NTroje and two from KIT. Three seeds represent training randomness; they do not add people. Exact child support must be confirmed against its exported population.

**TODO: Insert training-person count, motion/window counts, source/extractor/camera distributions, actual held-out factors, split construction and exclusions from the manifests. Reconcile sample counts with coverage. Do not copy pilot counts from older studies.**

The source status records 2,000 updates per pretraining phase and per readout, with 4,000 for inherited direct end-to-end fits. Setup binds the parent's actual admitted update schedule. Completed follow-up status records all 27 optimization phases and 18 final models, without failed worker attempts. Source allocation accounting is 0.437 H100-hours for profiling, 2.292 for follow-up training and 0.669 for diagnostics. These totals exclude earlier core training, rendering and pose extraction and should not be presented as total project compute.

Aggregation averages repeated conditions within windows, windows within raw motions, motions within people, and fitted seeds. Paired contrasts compare the same person and seed. The saved crossed bootstrap resamples people and seeds separately while preserving method pairing. All intervals are descriptive development intervals because the development population was reused. Secondary comparisons are unadjusted and retain their declared status.

The primary contrast is endpoint-minus-delta response error under paired-change readout. Base-readout contrast and their difference quantify dependence on the downstream objective. With D_base and D_change defined as endpoint-minus-delta error, the interaction is D_change-D_base.

**TODO: Confirm evaluation completion receipt, verify the saved comparisons and add exact bootstrap settings, primary interval, per-seed effects and interaction interval.**

## 6. Results

### 6.1 The primary contrast is small in the reported means

With paired-change readouts, delta JEPA has response error 9.9880 degrees versus 10.3611 for endpoint JEPA, a descriptive reduction of 0.3731 degrees (3.6%). Nuisance error decreases by 1.9109 degrees, waveform error by 0.2663 degrees, and all-joint NLE by 0.0005. Direction accuracy increases by 0.60 percentage points.

**TODO_PRIMARY_INTERVAL:** report the saved response interval and each seed's effect. If the interval includes zero, state that superiority remains unresolved. Do not substitute a favorable secondary endpoint for this comparison.

### 6.2 The comparison reverses with base readouts

With base readouts, delta response error is 10.9448 degrees and endpoint error is 10.2327, giving D_base=-0.7121 degrees. Together with D_change=+0.3731, the descriptive interaction is +1.0852 degrees. Delta also has higher base-readout nuisance and waveform error. Its apparent response advantage is thus conditional on the readout in the reported means.

**TODO_INTERACTION_INTERVAL:** include the paired interval and participant-level pattern. If uncertain, describe a point-estimate reversal rather than an established interaction.

### 6.3 Movement-change supervision trades geometric fidelity for selected scalar gains

Across the eight reported families, paired-change supervision increases mean coordinate NLE and waveform error. Waveform deterioration ranges from 4.3878 to 7.2075 degrees. Within delta JEPA, paired-change supervision reduces response error by 0.9568 degrees and nuisance error by 2.1547, while increasing waveform error by 4.3878 and NLE by 0.0071. Direction accuracy decreases by 0.87 percentage points. Coordinate-difference pretraining shows a larger adverse response effect: paired-change versus base increases response error by 4.3425 degrees.

This pattern concerns the frozen inherited loss coefficient and update schedule. It does not show that all possible response-aware objectives or coefficient choices degrade trajectories. The eight family means are correlated evidence on one development population, not independent replications.

**TODO_PAIRED_EFFECTS:** add within-person/seed effects and uncertainty; check whether failure scoring accounts for waveform as well as response differences. Show all families in the tradeoff figure.

### 6.4 Strong controls limit the incremental representation claim

Direct/base leads all five reported metrics: response error 7.5442 degrees, nuisance error 10.4929 degrees, all-joint NLE 0.0298, waveform error 12.0733 degrees and direction accuracy 66.09%. Delta JEPA with paired-change readout has response error only 0.0782 degrees below original paired JEPA and 0.1711 below initialized features with the same readout objective. These small descriptive differences require their own uncertainty.

Initialized and shuffled-reference base readouts have response errors 9.4331 and 9.3450 degrees, respectively, below the new pretrained base readouts. Low response error alone consequently does not demonstrate useful movement information from pretraining. The zero-response benchmark and response calibration are necessary to distinguish accurate tracking from attenuation.

### 6.5 Failure, attenuation and feature accessibility

**TODO_FAILURE_ANALYSIS:** decompose the primary contrast into successful-error and failure contributions. A weighted failure-rate difference of about 0.052 percentage points contributes approximately 0.373 degrees under the 720-degree score, illustrating the importance of reporting this decomposition. This arithmetic is not evidence that failures caused the result.

**TODO_CALIBRATION:** compare each key method with zero response; inspect reference-versus-predicted curves, slopes, direction eligibility, no-change controls and held-dose/observation strata. Curves and slopes conditional on success must display coverage. A direction accuracy near 50% is not automatically a demonstrated chance result without an appropriate sign-distribution baseline.

**TODO_FEATURE_DIAGNOSTICS:** inspect the already-computed training-only encoder/predictor/teacher diagnostics. Include a compact result if it explains the restoration findings and its protocol is valid. Otherwise keep it in the appendix or omit the mechanistic claim. Probe failure does not prove that information is absent.

## 7. Discussion and limitations

The strongest supported implication of the reported means is that the apparent value of a predictive representation depends on downstream supervision and on which aspects of movement fidelity are measured. A favorable scalar contrast can coexist with worse trajectories. A matched endpoint control, a second readout of the same encoder, initialized features and a direct-restoration baseline make these distinctions visible.

This is a bounded synthetic development study. Its independent sample is small and source-imbalanced; the follow-up reuses people that informed the core analysis. Projected two-dimensional knee measurements are not validated anatomical range of motion, diagnoses or patient outcomes. Privileged projected references and controlled edits may differ from errors and changes in real video. The fixed readout coefficient and training schedule leave hyperparameter robustness unresolved. Separately evolving teachers and the absence of a new pairing-randomization arm limit unique mechanistic attribution. Failure penalties and conditional calibration require joint reporting. Three seeds provide limited resolution of training variation.

These limitations constrain the claim to the tested procedure. Independent population validation and robustness to training choices would be needed to establish broad representation or clinical benefit.

## 8. Conclusion — provisional

Under the tested protocol, adding movement-change supervision increases mean geometric error across all eight model families, while the delta-versus-endpoint response comparison reverses across readouts. Direct coordinate restoration remains the strongest practical baseline in the reported means. The findings motivate evaluating movement responses jointly with underlying geometry, prediction reliability and matched downstream objectives. **TODO: Refine these sentences against the verified uncertainty and failure analyses.**

## Reproducibility statement — complete after checks

**TODO:** Reference the exact appendix sections documenting data processing, split identities, source restrictions, architecture, training configuration, seeds, saved predictions, failure rules and reconstruction commands. Include an anonymous code/evaluation package if ready. Do not claim publicly released data or successful independent reconstruction until those actions are complete.

## Ethics statement — author review required

This study evaluates projected synthetic references derived from motion-capture data. Its measurements do not establish clinical validity or support patient-level decisions. **TODO:** Document actual data licenses and permissions, source consent/access conditions, redistribution restrictions, representativeness, applicable institutional determinations and intended use. Do not claim an IRB exemption without a basis.

## AI-use statement — factual starter

Generative AI assisted with refinement of hypotheses and experimental design, implementation and validation code, analysis and interpretation of results, literature search, and manuscript planning and drafting. **TODO:** Identify tools and actual scope; document the human review and verification that occurred, including code, calculations, source attribution, figures and claims. Add the corresponding disclosure in the submission form. Do not state that all AI-assisted work has been verified before completing those checks.

## Reference anchors — verify and convert to BibTeX

1. Mohamed Abdelfattah and Alexandre Alahi. *S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition.* ECCV 2024. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf
2. Wojciech Marian Czarnecki et al. *Sobolev Training for Neural Networks.* NeurIPS 2017. https://proceedings.neurips.cc/paper/2017/file/758a06618c69880a6cee5314ee42d52f-Paper.pdf
3. *SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos.* ECCV 2022. https://arxiv.org/abs/2112.13715 — TODO: verify full authors and bibliography against the original.
4. Naureen Mahmood et al. *AMASS: Archive of Motion Capture as Surface Shapes.* ICCV 2019. https://openaccess.thecvf.com/content_ICCV_2019/html/Mahmood_AMASS_Archive_of_Motion_Capture_As_Surface_Shapes_ICCV_2019_paper.html — TODO: verify bibliography against the original.

## Appendix plan

A. Frozen data manifest, split identities, exclusions and reference review.

B. Model architecture, residual pathway, input contract and exact training settings.

C. Mathematical losses, shared auxiliary support, gradients and coefficient calibration.

D. Evaluation definitions, all-attempted penalties, zero-response benchmark and aggregation.

E. Full 16-method table, participant/seed effects and descriptive secondary intervals.

F. Response curves, held-dose strata, failures and feature diagnostics.

G. Reproducibility commands, provenance and total compute separated by stage.
