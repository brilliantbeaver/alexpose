# Proposal 05 tutorial: learning the odd component of gait

> **Purpose:** a robust 8 to 12 week investigation of signed laterality, reflection parity, and bounded cross-modal evidence for clinical construct validity.
>
> **Compute assumption:** 8 H100 GPUs.
>
> **Relationship to the original:** [README.md](./README.md) is preserved unchanged. GAVD becomes an internal pilot. The primary evidence comes from participant-disjoint external cohorts.
>
> **Detailed protocol:** [METHODOLOGY_8_12_WEEK.md](./METHODOLOGY_8_12_WEEK.md).

## How to read this tutorial

This document explains the full research direction from the physical idea to the model, datasets, tests, and claims. No background in symmetry-aware neural networks or gait biomechanics is assumed. The companion methodology contains the executable details.

The argument has five links:

1. A skeleton sequence records where labelled body joints move over time.
2. A learned representation compresses that sequence into numerical features.
3. Some features should stay unchanged when left and right anatomy are reflected. Other features should reverse sign.
4. A symmetry-aware encoder can keep these two kinds of information separate throughout the network.
5. That structure matters scientifically only if it improves prediction of an independent force measurement for held-out people.

The eight-week study tests this chain in one stroke cohort. The twelve-week extension asks whether the result repeats in a separate Parkinson's disease cohort.

## Step 1: State the research question

**Across participant-disjoint clinical and non-clinical cohorts, do skeleton representations preserve signed anatomical laterality as an odd component under reflection, and does explicit parity-aware learning improve its robustness and sample-efficient recovery?**

The specific, measurable, achievable, relevant, and time-bound version is:

> Within 12 weeks, train a standard and a fully $C_2$-equivariant S-JEPA on participant-disjoint non-clinical 3D motion, then compare them under matched data, optimization, parameter, and compute budgets. Test whether the equivariant model lowers participant-held-out error and low-label error area for an independent signed force target in the stroke cohort, preserves its advantage under corruption, and repeats that benefit with participant-level uncertainty in the independent Parkinson's disease cohort.

The minimum complete study ends after week 8 with the standard model, exact odd readout, full $C_2$ model, geometry tests, and stroke evaluation. The stronger cross-clinical claim requires weeks 9 to 12 and successful Parkinson's disease, or PD, replication. The two stopping points are reported separately.

## Step 2: Understand the main idea

Gait has information that should react differently to a mirror.

- **Shared gait**, such as overall walking speed, should stay the same.
- **Signed laterality**, such as right propulsion minus left propulsion, should reverse sign.

A standard neural network has to discover this distinction from examples. A **parity-aware** network is built so that anatomical reflection swaps two internal feature streams. Their average describes what is shared. Their difference describes which side is greater.

This creates an **even channel** for shared gait and an **odd channel** for signed laterality:

$$
e(Mx)=e(x), \qquad o(Mx)=-o(x).
$$

Correct geometry alone is not enough. The odd channel is scientifically useful only if it predicts a separate clinical or biomechanical measurement for people who were absent from pretraining, tuning, and model selection.

### A small numerical example

Suppose one branch produces the value 10 for the original skeleton and the other produces 6 for its anatomical reflection. Their even value is $(10+6)/2=8$. Their odd value is $(10-6)/2=2$. Reflecting the input exchanges 10 and 6. The even value remains 8, while the odd value becomes $-2$. This is the behavior the architecture must preserve at every claimed equivariant layer.

### Essential vocabulary

| Term | Plain meaning |
|---|---|
| Skeleton sequence | Labelled body-joint positions recorded over time |
| Representation | Numerical features that summarize a sequence |
| Encoder | The network that creates the representation |
| Readout or probe | A small model that converts features into the target |
| Anatomical reflection | Exchange left and right joints and reverse the side-to-side axis |
| Even component | Information that remains the same under reflection |
| Odd component | Information that changes sign under reflection |
| Equivariance | A guarantee that internal features transform by a declared rule when the input is transformed |
| Participant-disjoint | No person contributes data to both the training and test sides of a split |

![Parity-aware architecture](./images/8-12-week/parity-architecture.svg)

*Figure 1. Reflection swaps shared-weight branches. Their sum is even and their difference is odd. The model still has to earn predictive and clinical validity on held-out participants.*

## Step 3: Separate the pilot from the main evidence

The original idea asked whether the existing GAVD representation contains a signed coordinate signal. That remains useful, but it cannot carry the main paper because:

- the local GAVD subset is small and grouped by source video rather than verified participant;
- the encoder already saw the evaluation sequences;
- later curriculum stages used a supervised group loss;
- its target is computed from the same coordinates given to the model;
- its sign is not a validated affected-side label.

The revision gives every claim an appropriate source of evidence.

### How this extends earlier work

I-JEPA, V-JEPA, and S-JEPA motivate prediction in latent space, and VICReg motivates explicit anti-collapse regularization [2-5]. MotionBERT and ST-GCN provide strong transformer and graph baselines for learned human-motion features [14, 15]. None of these references alone guarantees that a signed anatomical output is odd under reflection.

Group-equivariant networks provide the mathematical template for layers that commute with a known symmetry, while e3nn shows how parity-aware representations can be engineered and tested [16, 17]. This proposal uses the smaller reflection group $C_2$, separates exact output antisymmetry from encoder-wide equivariance, and tests both against a compute-matched two-view model. Gait symmetry and propulsion studies motivate the signed clinical targets [20, 21]. The distinctive experiment is the combination of participant separation, explicit parity channels, low-label curves, corruption tests, and force measured outside the skeleton input.

![Full evidence ladder](./images/8-12-week/evidence-ladder.svg)

*Figure 2. No dataset is asked to prove retention, geometric behavior, generalization, and clinical validity at once.*

## Step 4: Give each dataset one job

| Dataset | Verified scale and role | What it can support | What it cannot support |
|---|---|---|---|
| **GAVD local subset** | 96 canonical sequences from 18 source videos; historical $d0acc262$ checkpoint | Transductive retention audit for signed coordinate excursion | Participant-level or clinical validity |
| **AMASS** | 344 participants, 11,265 motions, about 40 hours in the referenced release [6] | Participant-disjoint 3D motion pretraining and controlled reflection | Clinical laterality |
| **OUMVLP-Pose** | Pose sequences for more than 10,000 people across 14 views [7, 8] | Optional large 2D gait and view stress test using a separate ankle-ending schema | Clinical laterality, foot-level analysis, or anatomical mirroring from camera angle |
| **MoVi** | 90 actors, synchronized optical motion capture and calibrated stationary video [9] | 3D-to-2D geometry, pose error, view stability, controlled perturbation | Clinical validity |
| **Stroke cohort** | 50 stroke survivors and 138 able-bodied adults with kinematics, kinetics, EMG, and metadata [10] | Primary participant-held-out cross-modal concurrent validity from kinematics to force | Full clinical construct validation or broad cross-pathology claims |
| **PD cohort** | 26 participants measured ON and OFF medication with markers, bilateral forces, and condition-specific categorical most-affected-side metadata [11] | Eligibility-gated clinical replication and secondary side agreement | A stable participant-level affected side or large-sample accuracy |
| **GaitRec** | 2,084 patients, 211 controls, and 75,732 bilateral GRF trials [12] | Large convergent evaluation of force asymmetry against affected-side metadata | Skeleton encoder evaluation or target validation by itself |
| **CP release** | 356 patients and 1,719 gait trials with processed gait variables [13] | Optional pattern-level sensitivity analysis after a compatibility and signed-side audit | Assumed raw-skeleton or signed-side validation |

AMASS pretraining excludes MoVi and every downstream evaluation collection. All constituent datasets and participant manifests are pinned because AMASS combines multiple sources and has expanded over time.

OUMVLP-Pose is a secondary 2D track, not a required input to the primary 3D clinical comparison. Its released joints stop at the ankle, so it uses a separate reduced schema and never enters a foot-level comparison. The original OU-MVLP paper introduced the multi-view dataset; the pose release is cited separately [7, 8].

Every revised result uses right minus left as its positive sign convention. The historical GAVD diagnostic is left minus right, so only its displayed sign is reversed. Its magnitude and all fit metrics remain unchanged.

The GAVD primary pilot table is restricted to the common canonical extraction path. The full local set is a provenance-explicit sensitivity analysis, not evidence for a normal-versus-abnormal effect.

## Step 5: Build reflection into the encoder

### Why the reflection group has only two elements

The relevant group is:

$$
C_2=\{e,m\},
$$

where $e$ means “leave the anatomy unchanged” and $m$ means “reflect the anatomy.” There are only two possibilities, and reflecting twice returns the original. Mathematicians call this two-element set the group $C_2$. The reflection negates the mediolateral, or side-to-side, coordinate and swaps all semantic left-right joints.

The model first creates two synchronized views of the same motion:

$$
H^0(x)=\left[\phi(x),\phi(Mx)\right].
$$

One branch receives the original and the other receives its reflection. When the input itself is reflected, the branches exchange positions:

$$
H^0(Mx)=\left[\phi(Mx),\phi(x)\right].
$$

To **commute with the swap** means that “reflect then process” gives the same result as “process then exchange the branches.” Every layer in the full model must pass this test. Shared weights, attention, masks, positional information, normalization, residual paths, and nonlinearities are all checked. If only the final number is odd, the paper calls it **output antisymmetry**, not an equivariant encoder.

### Why keep the two branches until readout

Ordinary pointwise nonlinearities remain equivariant when the same operation is applied to both group branches. Even and odd components are formed only when read out:

$$
h_{even}=\frac{h_e+h_m}{2}, \qquad
h_{odd}=\frac{h_e-h_m}{2}.
$$

Here $h_e$ and $h_m$ are the two branch features. The average $h_{even}$ is unchanged by a branch swap. The difference $h_{odd}$ changes sign. Keeping the two branches intact allows ordinary shared nonlinear operations without losing the guarantee. The model forms even and odd features only when it reads them out.

## Step 6: Compare models that differ in one important way

![Matched experiment arms](./images/8-12-week/experiment-arms.svg)

*Figure 3. The main comparison changes parity structure while holding participants, updates, tuning, and evaluation fixed. A two-pass standard control separates parity from compute.*

| Arm | Training or architecture | What it establishes |
|---|---|---|
| A | Pure S-JEPA, no clinical labels and no mirror rule | Standard self-supervised comparator |
| A2 | Frozen A encoder with an unconstrained shared-head two-view readout | Compute and extra-view comparator |
| B | Frozen A encoder with sign-aware supervised mirror pairs | Learned output parity without a guarantee |
| C | Frozen A encoder with deliberately invariant supervised mirror pairs | Sign-erasure negative intervention |
| D | Frozen standard encoder with exact odd output projection | Output repair |
| E | Frozen A encoder with a shared paired-joint right-minus-left head | Structured head, no encoder guarantee |
| F | Fully $C_2$-equivariant S-JEPA | Main architectural intervention |
| G | Historical hybrid S-JEPA with supervised group loss | Supervision audit, GAVD only |
| H | MotionBERT and ST-GCN-family encoders | External representation baselines [14, 15] |
| I | Raw phase-aware kinematics and established gait features | Strong non-neural references |
| J | Random encoder, even-only, side-agnostic, nuisance-only, and shuffled controls | Falsification |

The primary architecture contrast is **F versus A2**. Both receive an original and reflected view, so the comparison does not reward F merely for seeing twice as much input. They also receive the same people, training updates, tuning opportunities, and nearly the same computation. **F versus A** is the practical comparison with a usual one-view model. **F versus D** asks the most important scientific follow-up: does organizing the entire encoder help beyond repairing only the final output?

Only A and F are separately pretrained. A2, B, C, D, and E reuse the exact frozen A checkpoint for each seed. This removes pretraining luck as a confound.

The fairness rules are:

- every A and F update receives the same sampled window, its anatomical reflection, and paired masks;
- A treats the two views as ordinary augmented examples, while F treats them as its two group branches;
- both use the same loss coefficients, token normalization, branchwise batch statistics, view-token exposure, update count, optimizer, tuning trials, and five paired seeds;
- their parameter counts and measured training FLOPs per orbit must each be within 5 percent;
- A2's two-pass inference FLOPs must be within 5 percent of F;
- a separate sensitivity analysis gives both models the same frozen total measured-FLOP budget.

If all matching gates cannot be satisfied at once, the paper reports exposure-matched and compute-matched effects separately. It does not claim a fully compute-matched causal architecture comparison.

The key readouts are explicit. With the frozen A encoder and shared scalar head $q$, A2 uses

$$
s_{A2}(x)=a q(E_A(x))+b q(E_A(Mx))+c,
$$

while D uses

$$
s_D(x)=\frac{q(E_A(x))-q(E_A(Mx))}{2}.
$$

All head parameters are optimized directly through the displayed output. For F, the primary signed probe is a zero-bias linear map of its odd channel. This preserves the parity being tested.

## Step 7: Define a target measured outside the skeleton

### Primary force target

From the stroke force plates, compute body-weight-normalized positive AP propulsive impulse for each side and form:

$$
y_{prop}=\log\left(\frac{J_R+\epsilon}{J_L+\epsilon}\right).
$$

Here $J_R$ and $J_L$ are the positive forward impulses produced by the right and left limbs. AP means anterior-posterior, or forward-backward. The logarithmic ratio is zero when the impulses are equal, positive when right is greater, and negative when left is greater. It changes sign under side exchange and is measured independently of the skeleton coordinates. Raw AP force crosses zero, so only the positive propulsion impulse enters the ratio [21].

Stroke kinetics have limited and incompletely audited coverage. The force target remains primary only if at least 30 stroke participants have two clean contacts per side and the two contact-based target estimates pass the frozen consistency gate.

If this gate fails, documented paretic side becomes the replacement primary before model fitting. Every fallback head then uses class-weighted logistic loss and predicts $+1$ exactly when its score is at least zero. The same arms and label budgets remain fixed. The outcome becomes class-balanced error AULC, with a $-0.03$ superiority margin and a $+0.02$ full-label noninferiority margin.

The fallback is confirmatory only if every outer-training fold and retained budget prefix contains at least two people per side and at least two common budgets remain. Signed temporal and spatial ratios remain secondary. The paper removes the independent-force and cross-clinical headlines.

### Secondary and replication targets

- the same signed propulsion definition in the PD cohort;
- the released condition-specific categorical most-affected-side field in PD, after a separate field audit;
- paretic side in stroke;
- signed step-length, stance-time, and swing-time ratios;
- GaitRec affected side against the force target, restricted to unilateral records;
- controlled unilateral attenuation with known sign and dose in AMASS or MoVi.

The PD side field is a broad clinical motor-laterality measure, not a gait-specific ground truth. Missing, undetermined, and ON/OFF-switching labels remain condition-specific and are never collapsed into one stable participant side. The synthetic attenuation check measures mechanism recovery, not simulated disease. A signed coordinate value is not automatically the affected side because compensatory motion can reverse an expected direction.

## Step 8: Keep every evaluation participant unseen

### Participant separation

- No clinical evaluation participant appears in generic pretraining.
- Every trial, stride, visit, medication state, camera view, and mirrored copy stays with its participant.
- All views of an OUMVLP-Pose or MoVi actor stay in one split.
- All ON and OFF sessions from one PD participant stay together.
- All visits from one CP participant stay together.
- GAVD stays source-video-grouped and explicitly transductive.
- The PD replication data do not tune the architecture, target, corruption set, or statistical rule.

### Ask how much labelled data the model needs

Within each stroke outer-training fold, use only eligible stroke survivors and freeze 20 nested, paretic-side-stratified participant rankings. Every arm receives the same ranking at budgets of 4, 8, 16, and all eligible stroke participants.

Each budget is a prefix of the same participant ranking, so a larger budget contains the people in the smaller one. Every model sees exactly the same labelled participants. Predictions are averaged across paired seeds before loss is calculated. The errors across budgets are summarized by normalized error AULC, the area under the learning curve. Lower AULC means less error over the entire range of label budgets. At the full budget, the main prediction summary is mean absolute error, or MAE, on the independent force target.

### Test reflection, viewpoint, and damaged inputs separately

Test separately:

$$
s(Mx)\approx-s(x)
$$

for anatomical reflection, and:

$$
s(Vx)\approx s(x)
$$

for view change.

The confirmatory view operator is fixed sensor-frame yaw on clean outer-test stroke skeletons: rotate each canonical 3D cycle by $-30^\circ$ and $+30^\circ$ about the saved vertical axis, keep anatomical labels fixed, and recompute one prediction per participant and angle. This does not establish real camera-view invariance. Calibrated MoVi cameras test that stronger claim separately.

The confirmatory corruption area averages four equally weighted families: coordinate noise, missingness, time, and joint-schema reduction. Each family uses severities $0,1/3,2/3,1$. The calculation subtracts the clean loss before integrating degradation across severity and does not discard negative changes.

The schema family always exists because it removes universally required bilateral ankle, knee, and hip levels in a fixed order. Every stochastic setting averages 20 fixed, shared realizations. Force error is divided by the training-target MAD, while fallback class-weighted 0-1 error is already dimensionless. Left-right identity errors, viewpoint, gait speed, and estimator changes remain separate falsification or sensitivity analyses.

Exact oddness in Arm D and layerwise commutation in F are manipulation checks. Scientific benefit requires better independent prediction, sample efficiency, or robustness.

## Step 9: Decide in advance what counts as success

### Week-8 single-cohort claim

Using the paired-exposure, equal-update checkpoints, an equivariant-model benefit over the standard two-view control first requires the 5 percent parameter, per-orbit training-FLOP, and downstream inference-FLOP gates. It then requires all of the following:

1. layerwise commutation tests pass for every layer called equivariant;
2. the two-sided participant-bootstrap 95% interval for $MAE_F-MAE_{A2}$ lies below zero in stroke;
3. $AULC_F-AULC_{A2}\le -0.03$, with its Holm-adjusted one-sided paired sign-flip p-value at most 0.05;
4. the upper 95% confidence limit for $e_{view,F}-e_{view,A2}$ is at most $+0.02$;
5. the Holm-adjusted one-sided paired sign-flip p-value for corruption area $cAUC_F-cAUC_{A2}$ is at most 0.05.

If the force fallback is active, item 2 is replaced by an upper 95% confidence limit of at most $+0.02$ for full-label balanced error of F minus A2, and balanced error replaces normalized absolute error in item 3. The result is a narrower side-label study. A stronger claim that encoder equivariance adds value beyond output repair additionally requires a Holm-adjusted one-sided paired sign-flip p-value at most 0.05 for full-label error of F minus D. The methodology fixes the three-test Holm family, 100,000 sign flips, seed, step-down thresholds, and ordinary effect-size intervals.

### Week-12 cross-clinical claim

Before any PD model result is inspected, the released fields and condition mappings must pass audit. At least 15 of the 26 participants must also have valid bilateral force targets under the same contact and consistency rules used for stroke.

ON- and OFF-medication features and targets remain separate records. One frozen five-fold participant partition is used, with no repeated splits and no inner tuning. Each person receives total training weight one, even if that person has two conditions. Condition losses are averaged into one F-versus-A2 effect per person before bootstrapping.

Cross-clinical replication requires all three checks:

1. the 95% interval for the mean participant effect lies below zero;
2. F's out-of-fold calibration slope is between 0.5 and 1.5;
3. F's absolute out-of-fold mean bias is no larger than 0.5 times the participant-condition-weighted PD target standard deviation.

The methodology freezes the regression direction, intercept, standard-deviation divisor, and clustered bootstrap. Strict stroke-to-PD transfer is a harder secondary test.

If the PD force gate fails, the PD analysis is descriptive and there is no cross-clinical claim. A negative point estimate whose interval overlaps zero is reported only as **directional consistency**, not replication. Condition-specific clinical-side agreement is reported only if that separate metadata gate passes.

If F does not meet the superiority rule against D, the data do not support encoder-level benefit beyond output repair. That is not evidence of equivalence. If parity metrics improve without force prediction, the result is geometric success without clinical value. If raw kinematics win, the paper reports that learned representations add no predictive advantage.

## Step 10: State the possible contributions

1. **LateralityBench:** a participant-safe protocol for odd and even skeleton targets, anatomical reflection, view tests, corruption, and low-label curves.
2. **A corrected evidence hierarchy:** GAVD as pilot, non-clinical data for geometry, and independently measured force for bounded concurrent-validity evidence.
3. **A true $C_2$-equivariant skeleton JEPA:** reflection parity preserved through every claimed layer.
4. **A controlled architecture comparison:** exposure-matched primary training plus a separately defined equal-FLOP sensitivity, with matched participants, tuning, and seeds.
5. **Cross-clinical evidence:** stroke primary evaluation and an inferentially qualified, locked PD replication.
6. **A useful null result path:** a direct test of detectable encoder benefit beyond exact output repair, without treating nonsignificance as equivalence.

## Step 11: Follow the staged timeline

![Eight to twelve week timeline](./images/8-12-week/splits-timeline.svg)

*Figure 4. The protocol freezes before stroke testing. PD remains untouched until the architecture and analysis are fixed.*

| Weeks | Work | Milestone |
|---|---|---|
| 1 to 2 | Access, licenses, checksums, common schema, force audit, participant manifests, GAVD reproduction | Targets, folds, and claims frozen |
| 3 to 4 | Standard, sign-aware, exact-output, two-pass, and raw baselines; implement group lifting and tests | Baselines complete; layerwise commutation passes |
| 5 to 6 | Matched AMASS pretraining across five seeds; collapse and parity-channel diagnostics | Frozen encoders |
| 7 to 8 | Stroke nested evaluation, low-label curves, corruption, MoVi geometry | Minimum complete single-clinical study |
| 9 to 10 | PD field and force gates, then locked replication; GaitRec convergent target evaluation; optional OUMVLP-Pose stress test | Qualified cross-clinical result |
| 11 to 12 | Ablations, statistics, adversarial leakage audit, paper, benchmark package | Full study and release |

Estimated compute is 4,000 to 8,000 H100-hours. Re-estimate from measured week-2 throughput and publish actual use. Spare compute is for prespecified seeds and robustness tests, not post hoc model search.

## Step 12: Keep the conclusions within the evidence

- GAVD folder labels are not diagnoses.
- Signed coordinate excursion is not a clinical symmetry biomarker.
- Camera view is not anatomical reflection.
- A signed head is not an equivariant encoder.
- Exact mirror behavior does not prove clinical usefulness.
- Force association does not establish diagnosis, causality, prognosis, or treatment benefit.
- Stroke success alone does not establish cross-pathology generalization.
- CP data are not a signed skeleton replication unless compatibility and affected-side metadata are verified.

## References

1. Ranjan et al., GAVD, *IEEE Access* 2025. [DOI 10.1109/ACCESS.2025.3545787](https://doi.org/10.1109/ACCESS.2025.3545787).
2. Abdelfattah and Alahi, S-JEPA, ECCV 2024. [DOI 10.1007/978-3-031-73411-3_21](https://doi.org/10.1007/978-3-031-73411-3_21).
3. Assran et al., I-JEPA, CVPR 2023. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243).
4. Bardes et al., V-JEPA, 2024. [arXiv:2404.08471](https://arxiv.org/abs/2404.08471).
5. Bardes, Ponce, and LeCun, VICReg, ICLR 2022. [arXiv:2105.04906](https://arxiv.org/abs/2105.04906).
6. Mahmood et al., AMASS, ICCV 2019. [arXiv:1904.03278](https://arxiv.org/abs/1904.03278), [official site](https://amass.is.tue.mpg.de/).
7. Takemura et al., OU-MVLP, 2018. [DOI 10.1186/s41074-018-0039-6](https://doi.org/10.1186/s41074-018-0039-6).
8. An et al., OUMVLP-Pose, *IEEE Transactions on Biometrics, Behavior, and Identity Science*, 2020. [DOI 10.1109/TBIOM.2020.3008862](https://doi.org/10.1109/TBIOM.2020.3008862).
9. Ghorbani et al., MoVi, *PLOS ONE* 2021. [DOI 10.1371/journal.pone.0253157](https://doi.org/10.1371/journal.pone.0253157), [data DOI](https://doi.org/10.5683/SP2/JRHDRN).
10. Van Criekinge et al., stroke and able-bodied full-body gait data, *Scientific Data* 2023. [DOI 10.1038/s41597-023-02767-y](https://doi.org/10.1038/s41597-023-02767-y), [data DOI](https://doi.org/10.6084/m9.figshare.c.6503791.v1).
11. Shida et al., PD ON and OFF medication gait data, *Frontiers in Neuroscience* 2023. [DOI 10.3389/fnins.2023.992585](https://doi.org/10.3389/fnins.2023.992585), [data DOI](https://doi.org/10.6084/m9.figshare.14896881).
12. Horsak et al., GaitRec, *Scientific Data* 2020. [DOI 10.1038/s41597-020-0481-z](https://doi.org/10.1038/s41597-020-0481-z), [data DOI](https://doi.org/10.6084/m9.figshare.c.4788012).
13. Nieuwenhuys et al., CP gait classification data, *PLOS ONE* 2017. [DOI 10.1371/journal.pone.0178378](https://doi.org/10.1371/journal.pone.0178378), [data DOI](https://doi.org/10.6084/m9.figshare.4877432.v1).
14. Zhu et al., MotionBERT, ICCV 2023. [arXiv:2210.06551](https://arxiv.org/abs/2210.06551).
15. Yan, Xiong, and Lin, ST-GCN, AAAI 2018. [arXiv:1801.07455](https://arxiv.org/abs/1801.07455).
16. Cohen and Welling, group-equivariant convolutional networks, ICML 2016. [arXiv:1602.07576](https://arxiv.org/abs/1602.07576).
17. Geiger and Smidt, e3nn, 2022. [arXiv:2207.09453](https://arxiv.org/abs/2207.09453).
18. Kapoor and Narayanan, leakage in ML-based science. [arXiv:2207.07048](https://arxiv.org/abs/2207.07048).
19. Varoquaux, cross-validation with small samples, *NeuroImage* 2018. [DOI 10.1016/j.neuroimage.2017.06.061](https://doi.org/10.1016/j.neuroimage.2017.06.061).
20. Patterson et al., gait symmetry after stroke, *Gait & Posture* 2010. [DOI 10.1016/j.gaitpost.2009.10.014](https://doi.org/10.1016/j.gaitpost.2009.10.014).
21. Bowden et al., AP ground reaction force as paretic-limb propulsion, *Stroke* 2006. [DOI 10.1161/01.STR.0000204063.75779.8d](https://doi.org/10.1161/01.STR.0000204063.75779.8d).
