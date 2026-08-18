# Proposal 05 tutorial: signed laterality in three weeks

> **Purpose:** a rapid, participant-disjoint study of whether skeleton representations retain which side differs, not only how much the two sides differ.
>
> **Schedule:** 21 calendar days with access to 8 H100 GPUs.
>
> **Relationship to the original:** [`README.md`](./README.md) is preserved unchanged. GAVD is now an internal pilot audit. It is not the clinical test set.
>
> **Detailed protocol:** [`METHODOLOGY_3_WEEK.md`](./METHODOLOGY_3_WEEK.md).

## How to read this tutorial

This proposal begins with the problem in everyday language and adds technical detail one layer at a time. You do not need prior knowledge of gait analysis, representation learning, or clinical statistics. The companion methodology turns the same idea into an executable protocol.

The whole study can be summarized as one chain:

1. Record a person's movement as a sequence of body-joint locations.
2. Ask a model to compress that sequence into useful numerical features, called a representation.
3. Check whether those features retain not only that the two legs differ, but also which leg is greater.
4. Reflect the anatomy and verify that a right-minus-left prediction changes sign.
5. Compare a readout that is forced to obey this rule with an equally informed readout that is not.
6. Test both readouts on people who were never used to train or tune them.
7. Compare their predictions with a separate force-plate measurement.

The central distinction is simple: **asymmetry magnitude** says how different the sides are, while **signed laterality** says both how different they are and which side is greater.

## Step 1: State the research question

**Across participant-disjoint clinical and non-clinical cohorts, do skeleton representations preserve signed anatomical laterality as an odd component under reflection, and does explicit parity-aware learning improve its robustness and sample-efficient recovery?**

For this three-week version, the broad question becomes a decision that can be completed in 21 days:

> By day 21, compare matched standard and parity-aware readouts of the same frozen skeleton representation on participant-held-out prediction of an independent, force-derived signed propulsion target from one public stroke cohort, after using GAVD only as an internal pilot. Test whether the exact odd readout reduces normalized low-label error area by at least 0.03, with a participant-bootstrap 95% confidence interval below zero, while meeting fixed clean-data, sensor-frame-yaw, and corruption noninferiority margins.

In simpler terms, the study asks whether enforcing the correct left-right rule helps when labels are scarce and inputs are imperfect. This is an ambitious rapid study. It can support a bounded claim in one stroke cohort. It cannot support diagnosis, treatment, or generalization across diseases.

## Step 2: Build an intuitive picture

Most gait summaries discard direction. They say that two legs differ, but not whether the left or right side drives the difference. That lost sign can matter in unilateral conditions such as stroke.

The study asks three linked questions:

1. **Retention:** Is left-versus-right information present in a learned skeleton representation?
2. **Parity:** When anatomy is reflected, does that information change sign exactly as a right-minus-left quantity should?
3. **Usefulness:** Does teaching the model this rule make the signed information easier to recover with fewer labeled participants and more resistant to missing or noisy joints?

The intervention is simple. A standard readout must discover the reflection rule from examples. A sign-aware readout is shown that a reflected skeleton must receive the negative label. An exact odd readout goes one step further and builds the rule into its output.

Think of a temperature difference written as “room A minus room B.” If the room names are exchanged, the number must change sign. The laterality target behaves in the same way: if right and left anatomy are exchanged, “right minus left” becomes “left minus right.”

The **encoder** is the part of the model that turns a long skeleton sequence into a shorter feature vector. The **readout**, also called a probe or head, converts that vector into the final number. In the three-week study the encoder is frozen. Only the readout changes. The study therefore tests parity-aware **readout learning**, not parity-aware representation learning throughout the encoder.

![Reflection rules for signed laterality](./images/3-week/reflection-rules.svg)

*Figure 1. Anatomical reflection and camera viewpoint are different transformations. A signed right-minus-left quantity should flip under the first and remain stable under the second.*

## Step 3: Learn the reflection rule from first principles

### The four objects in the equations

| Symbol | Plain meaning |
|---|---|
| $x$ | One time sequence of labelled body joints |
| $M$ | Anatomical reflection: exchange left and right body parts and reverse the side-to-side axis |
| $E$ | The frozen encoder that turns a sequence into features |
| $q$ or $s$ | A readout that turns features into a prediction |

Let $x$ be a skeleton sequence. Before modeling, it is pelvis-centered, scaled, and placed in a canonical gait coordinate system with forward, vertical, and mediolateral axes.

Let $M$ be the **anatomical reflection** operation. It negates the mediolateral coordinate and swaps every left and right landmark. Applying it twice returns the original sequence:

$$
M^2x=x.
$$

An **even** quantity is unchanged by this reflection:

$$
e(Mx)=e(x).
$$

Overall speed and total movement amplitude are expected to be mostly even.

An **odd** quantity changes sign:

$$
o(Mx)=-o(x).
$$

A right-minus-left clinical side label and a signed right-minus-left propulsion difference are odd.

For example, suppose right propulsion is 12 units and left propulsion is 9 units. The signed difference is $12-9=+3$. After exchanging the sides, it is $9-12=-3$. The magnitude is still 3, but the sign has reversed.

Any scalar readout $q$ can be separated into exact even and odd parts:

$$
q_{even}(x)=\frac{q(x)+q(Mx)}{2}, \qquad
q_{odd}(x)=\frac{q(x)-q(Mx)}{2}.
$$

The first equation averages a prediction and its reflected prediction, so the sign-changing part cancels. The second subtracts them, so the unchanged part cancels. This projection guarantees that the final odd output changes sign. It does **not** prove that every internal encoder feature follows the reflection rule. That stronger architectural claim belongs to the 8 to 12 week study.

A camera transformation $V$ is not the same as $M$. Changing viewpoint should preserve anatomical identity:

$$
o(Vx)\approx o(x).
$$

Confusing $V$ with $M$ would reward a model that mistakes camera orientation for the affected body side. Anatomical reflection changes body-side identity. A viewpoint change only changes where the observer stands.

### Three ideas that must remain separate

| Idea | Question it answers | What would count as evidence? |
|---|---|---|
| Decodability | Is signed information available in the features? | A held-out probe predicts the signed target |
| Odd output behavior | Does the final number reverse correctly under reflection? | $s(Mx)=-s(x)$ within numerical tolerance |
| Clinical usefulness | Does the number agree with an independent measurement? | It predicts held-out force-plate asymmetry |

A model can pass one row and fail another. For example, a readout may predict reasonably well but use a camera cue that does not reverse correctly.

## Step 4: Place the idea relative to earlier work

JEPA methods learn by predicting missing latent content rather than reconstructing every input detail. I-JEPA and V-JEPA established the broader approach, while S-JEPA adapts predictive latent learning to skeletons [2-4]. VICReg supplies widely used variance and covariance regularization against representation collapse [5]. These methods do not by themselves require a signed output to reverse under anatomical reflection.

Group-equivariant networks show how known symmetries can be built into a model [11]. The rapid study uses the smallest consequence of that idea, an exact odd output projection, and carefully avoids calling the frozen encoder equivariant. Clinical gait work motivates signed temporal and propulsion asymmetry, while also showing why compensation prevents a movement sign from being treated as an automatic diagnosis or affected-side label [8, 12]. The study's gap is therefore specific: test signed parity with participant separation and a force measurement that is independent of skeleton coordinates.

## Step 5: Correct the evidence hierarchy

The revision makes six corrections that change what the evidence can support.

| Previous risk | Correct treatment in this study |
|---|---|
| Calling the final `d0acc262` lineage purely self-supervised | Report that stages 1 to 4 used the supervised `L_group` term. Treat the checkpoint as a historically exposed pilot model. |
| Treating raw coordinates as a null | Call them a coordinate reference. On GAVD, the coordinate-derived target makes them an oracle-like upper reference. |
| Calling GAVD's scalar a clinical biomarker | Name it **signed coordinate excursion**. It is a representation diagnostic, not a validated symmetry ratio and not proof of affected side. |
| Using ordinary mirror augmentation | Use sign-aware pairs `(x, y)` and `(Mx, -y)` for odd targets. Ordinary invariant augmentation would erase the sign. |
| Equating horizontal reflection with viewpoint change | Test anatomical reflection and camera or rigid-view perturbation separately. |
| Evaluating clips as independent samples | Split and resample by participant in external data, and by source video only in the GAVD pilot where participant IDs are unavailable. |

## Step 6: Give each dataset one clear job

![Evidence ladder for the rapid study](./images/3-week/evidence-ladder.svg)

*Figure 2. Each dataset answers a narrower question than the one after it. Dataset size and fields are documented by the GAVD paper, the MoVi paper, and the public stroke dataset descriptor [1, 6, 7].*

| Dataset | Role | What enters the model | Independent target or test | Licensed conclusion |
|---|---|---|---|---|
| **GAVD**, 96 canonical sequences from 18 source videos | Internal pilot and code audit | Existing 2D BlazePose skeletons and frozen `d0acc262` features | Signed coordinate excursion derived from coordinates | Whether the current exposed representation retains its own coordinate diagnostic |
| **MoVi**, 90 actors with synchronized video and motion capture | Day-3 gated non-clinical geometry check | Common skeleton from held-out actors and views | Known view identity, synthetic anatomical reflection, and optional mocap agreement | Whether mirror and view behavior generalizes outside GAVD |
| **Public stroke and healthy gait cohort**, 50 stroke survivors and 138 able-bodied adults | Main clinical study | Full-body Plug-in Gait kinematics | Quality-gated force-derived bilateral asymmetry; paretic-side metadata | Participant-disjoint cross-modal concurrent validity in one stroke cohort |

MoVi is included only if access, licensing, and common-schema conversion work by the end of day 3. If that gate fails, the paper reports two datasets, labels the external geometry result absent, and does not replace it with a convenient weak dataset.

All new plots use one sign convention: positive means **right greater than left**. The historical GAVD function computes left minus right, so the revised audit multiplies that stored scalar by -1 for display and comparison. This reorientation changes no magnitude, fit, or oddness result.

The GAVD pilot's primary table uses the common canonical extraction path. The full local subset is a sensitivity analysis with extraction provenance shown explicitly. This prevents the largely augmented normal rows and canonical abnormal rows from turning acquisition route into an apparent gait effect.

The stroke cohort is the decisive dataset. It provides source C3D files plus postprocessed MAT and spreadsheet files containing left and right stride data, kinematics, force plates, and EMG. Kinetic data in the stroke portion were not the collection priority and require explicit quality checks [7]. Therefore:

- the **gated primary target** is a signed bilateral anterior-posterior propulsive impulse log-ratio from raw, rechecked force traces;
- the **secondary clinical target** is documented paretic side: $y_{side}=+1$ for right and $-1$ for left;
- the **even control target** is stroke versus healthy status, which must not flip under $M$;
- the force primary proceeds only if at least 30 stroke participants have two clean contacts from each side and split-contact target consistency passes its frozen gate;
- otherwise, documented paretic side becomes the replacement primary among eligible stroke survivors;
- in that fallback, $y_{side}=+1$ means right paretic and $-1$ means left paretic, every head uses class-weighted logistic loss and the frozen rule $\hat y=+1$ exactly when its score is at least zero, the primary metric is class-balanced error AULC at the same four label budgets, and the primary contrast and success margins remain C versus D, $-0.03$ for AULC superiority, and $+0.02$ for full-label noninferiority;
- the fallback remains confirmatory only if every outer-training fold and every retained label-budget prefix has at least two participants per side and at least two common budgets remain;
- signed step-length, stance-time, and swing-time ratios remain secondary, and the fallback removes the independent-force concurrent-validity claim from the rapid paper.

This hierarchy prevents a noisy force subset from silently changing the study question.

The force target uses the positive time integral of anterior-posterior ground reaction force, a standard measure of each limb's contribution to post-stroke propulsion [12].

## Step 7: Put every dataset into a common body format

Different datasets name and record joints differently. To compare them fairly, all retained data are mapped to the same small body schema: pelvis center and the left and right hip, knee, ankle, heel, and toe or forefoot. Shoulders are included only when their mapping is unambiguous. Each joint also carries a mask that says whether it was observed or missing.

The mapping is frozen before model comparisons. No landmark is inferred from the test labels. Gait cycles are normalized jointly, not one side at a time, so left-right timing differences are not erased.

## Step 8: Compare matched model readouts

Every learned arm receives the same participants, input schema, optimizer budget, folds, seeds, and tuning budget.

![Matched experiment arms for the rapid study](./images/3-week/experiment-arms.svg)

*Figure 3. The short study changes the readout or parity training rule. It does not claim a fully equivariant encoder.*

### Baselines that show what “good” means

- **Coordinate reference:** regularized models on flattened or biomechanical input features. On GAVD this is an oracle-like reference because the pilot target comes from those coordinates.
- **Random encoder:** same architecture with random frozen weights.
- **Side-agnostic control:** pool corresponding left and right features before the probe. This should remove sign while retaining overall gait information.
- **Nuisance control:** predict source, view, missingness, and speed where available.

### The four learned arms

- **A. Standard S-JEPA:** a compact pure skeleton JEPA, trained without clinical labels or a reflection constraint and then frozen [2-5].
- **B. Sign-aware training:** the same frozen encoder and readout family, trained with both `(x, y)` and `(Mx, -y)` for odd supervised targets. Even targets use `(Mx, y)`.
- **C. Exact odd readout:** $s_C(x)=[q(E(x))-q(E(Mx))]/2$. The shared scalar head $q$ is optimized directly through this two-pass expression. This is an output guarantee, not encoder-wide equivariance.
- **D. Unconstrained two-view readout:** $s_D(x)=a q(E(x))+b q(E(Mx))+c$, with $q,a,b,c$ optimized jointly. It uses the same two encoder passes but has no parity constraint. Its three extra scalar parameters keep the readout within 5 percent of C when the embedding has at least 64 dimensions.

Arm C is selected before evaluation as the primary parity intervention. The key comparison is **C versus D**. Both see the original and reflected inputs, so this comparison asks whether the rule itself helps rather than whether a second model pass helps. **C versus A** measures the practical change from an ordinary one-pass readout. Arm B asks whether correct reflected examples are already sufficient without an exact guarantee.

The existing `d0acc262` checkpoint is not placed in this matched causal comparison. It remains a GAVD pilot because its later stages used condition labels and its evaluation rows were seen during representation training.

## Step 9: Evaluate without information leakage

### Keep each person entirely on one side of a split

- Freeze a participant manifest before training.
- Use five outer participant-disjoint folds, stratified by paretic side where possible.
- Select hyperparameters only with grouped inner folds from the outer training participants.
- Permit self-supervised training only on participants in the current outer training fold.
- Aggregate all windows and trials to one prediction per participant before inference.
- Keep GAVD source-video-grouped and label all GAVD results transductive.

![Leakage-safe split and execution timeline](./images/3-week/splits-timeline.svg)

*Figure 4. Outer test participants never choose a target, landmark mapping, model, or hyperparameter. Eight GPUs parallelize matched runs, not scientific decisions.*

### Measure performance across four label budgets

Fit a small regularized probe with 4, 8, 16, and then all eligible outer-training stroke participants. These are the **label budgets**. They reveal whether a method helps most when labelled clinical data are scarce.

Each larger budget contains the people from the smaller budget, and every arm receives exactly the same people. The experiment repeats the subset construction with 20 fixed participant rankings and five paired model seeds. Predictions are averaged before loss is calculated. The losses across budgets are summarized as the **area under the learning curve**, or AULC. Lower AULC means lower error across the full range of label budgets, not merely at the largest budget. The methodology gives the exact calculation. Healthy participants are not given invented side labels and never enter this supervised clinical head.

The primary estimate is:

$$
\Delta_{AULC}=AULC_C-AULC_D.
$$

The confirmatory view test never depends on optional MoVi access. It applies fixed $-30^\circ$ and $+30^\circ$ vertical-axis rotations to clean outer-test stroke cycles after canonical preprocessing, recomputes one prediction per participant and angle, and averages participant-level prediction drift. This tests sensor-frame yaw stability, not real camera-view invariance. MoVi camera results remain secondary external geometry evidence.

The confirmatory corruption area averages four equally weighted families: coordinate noise, missingness, temporal corruption, and window-start shift. Each uses normalized severities $0,1/3,2/3,1$, clean loss is subtracted before trapezoidal integration, and negative degradation is retained. Every stochastic setting averages 20 fixed, shared realizations. Force loss is divided by the outer-training target MAD with a fixed numerical floor. Fallback class-weighted 0-1 loss is already dimensionless. Left-right identity corruption is a separate destructive control and never enters cAUC.

Exact oddness within numerical tolerance is a wiring check that must pass before inference. It is not an outcome. The result supports the rapid-study claim only when all of the following are true:

1. the point estimate of $\Delta_{AULC}$ is at most $-0.03$;
2. its participant-bootstrap 95% confidence interval is below zero;
3. the upper 95% confidence limit for full-label $nMAE_C-nMAE_D$ is at most $+0.02$;
4. the upper 95% confidence limit for $e_{view,C}-e_{view,D}$ is at most $+0.02$;
5. the upper 95% confidence limit for corruption-area difference $cAUC_C-cAUC_D$ is at most $+0.02$.

If the paretic-side fallback is activated, class-balanced error replaces nMAE in items 1 to 3, the participant bootstrap is stratified by side, and every threshold stays fixed.

If the fallback's class-count or retained-budget gate fails, side recovery is secondary and there is no confirmatory clinical headline.

Failure of any item is informative. It means the result is mixed or inconclusive, not that the protocol should be changed after seeing the test predictions.

### Check why a model succeeds or fails

**Oddness error** measures how far the output is from exact sign reversal:

$$
\epsilon_{odd}=\frac{\mathbb{E}|s(Mx)+s(x)|}{\mathbb{E}|s(Mx)|+\mathbb{E}|s(x)|+\epsilon}.
$$

The other secondary checks are:

- **Even leakage:** fit $q_{even}(x)=[q_e(E(x))+q_e(E(Mx))]/2$ directly to paretic side among stroke survivors. Above-chance recovery flags nuisance or cohort structure in an even output.
- **Parity selectivity:** fit the same exact even form directly to stroke status, using stroke and healthy participants, and compare it with odd-channel side recovery. Each target receives its own nested-training head.
- **View stability:** sign agreement and prediction drift under physically valid rigid rotations or paired camera views.
- **Corruption robustness:** degradation under calibrated coordinate noise, confidence-weighted joint dropout, temporal gaps, and frame-rate change.
- **Cross-modal concurrent validity:** participant-held-out prediction and signed agreement with the contemporaneous force-derived bilateral target in the quality-qualified subset. This is one bounded source of construct evidence, not full clinical construct validation.
- **Affected-side recovery:** balanced accuracy and calibration for left-versus-right paretic side, treated as secondary because compensation can weaken a simple force-sign mapping.
- **Calibration and uncertainty:** balanced accuracy, AUROC, Brier score, and participant-bootstrap confidence intervals.

Exact oddness for arm C is a manipulation check. It is not a discovery and cannot substitute for held-out accuracy.

## Step 10: Execute and freeze decisions on schedule

| Days | Work and freeze point | Compute use |
|---|---|---|
| 1 to 3 | Verify licenses and identifiers; build C3D and GAVD adapters; audit paretic-side and bilateral-force coverage; implement `M`; run involution and sign unit tests | One GPU for smoke tests; CPUs for data audit |
| 4 to 7 | Freeze the common schema, target registry, participant folds, exclusions, primary contrast, and statistical script; complete GAVD pilot | One to two GPUs |
| 8 to 13 | Train arms A to D across folds and five paired seeds; extract frozen features; run label-budget probes | Up to eight GPUs in parallel |
| 14 to 17 | Run corruption and optional MoVi geometry checks; create participant-level prediction table; execute frozen inference | Up to eight GPUs for independent jobs |
| 18 to 21 | Bootstrap and permutation inference; adversarial leakage audit; paper figures; limitations; reproducibility bundle | Mostly CPU; short reruns only for verified implementation defects |

Eight H100s make the matched training grid practical. They do not solve dataset mapping, target validity, or small clinical sample size. The day-3 gate is therefore the most important schedule control.

## Step 11: Interpret the result conservatively

If successful, the paper contributes:

1. a precise odd-versus-even formulation of anatomical laterality for skeleton representations;
2. an honest reclassification of GAVD as a transductive pilot audit;
3. participant-disjoint concurrent biomechanical evidence for recovery of an independent signed propulsion target, with affected side as a secondary endpoint;
4. a matched test of whether a lightweight parity-aware readout improves low-label and corrupted-input recovery;
5. open manifests, transformation unit tests, and participant-level evaluation code.

The bounded conclusion would be:

> In one public stroke cohort, explicit reflection parity improved sample-efficient recovery of independently measured signed propulsion under participant-disjoint evaluation, while GAVD served only as a pilot retention audit.

The study must not claim that the representation diagnoses stroke, that coordinate excursion is a validated clinical biomarker, or that head-level oddness proves an equivariant backbone.

## Relationship to Proposal 09

Proposal 05 supplies the **measurement instrument**: signed targets, oddness tests, leakage-safe splits, robustness curves, and concurrent-validity checks. Proposal 09 supplies the **architectural intervention**: force reflection parity into the model.

In this rapid version, only inexpensive Proposal 09 interventions are tested at the head or output level. The full $C_2$-equivariant encoder and cross-pathology replication are reserved for the 8 to 12 week version.

## References

1. Ranjan et al., “Computer Vision for Clinical Gait Analysis: A Gait Abnormality Video Dataset,” *IEEE Access*, 2025. [DOI 10.1109/ACCESS.2025.3545787](https://doi.org/10.1109/ACCESS.2025.3545787).
2. Abdelfattah and Alahi, “S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition,” ECCV 2024. [DOI 10.1007/978-3-031-73411-3_21](https://doi.org/10.1007/978-3-031-73411-3_21).
3. Assran et al., “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture,” CVPR 2023. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243).
4. Bardes et al., “Revisiting Feature Prediction for Learning Visual Representations from Video,” 2024. [arXiv:2404.08471](https://arxiv.org/abs/2404.08471).
5. Bardes, Ponce, and LeCun, “VICReg,” ICLR 2022. [arXiv:2105.04906](https://arxiv.org/abs/2105.04906).
6. Ghorbani et al., “MoVi: A Large Multi-Purpose Human Motion and Video Dataset,” *PLOS ONE*, 2021. [DOI 10.1371/journal.pone.0253157](https://doi.org/10.1371/journal.pone.0253157), [data DOI](https://doi.org/10.5683/SP2/JRHDRN).
7. Van Criekinge et al., “A full-body motion capture gait dataset of 138 able-bodied adults across the life span and 50 stroke survivors,” *Scientific Data*, 2023. [DOI 10.1038/s41597-023-02767-y](https://doi.org/10.1038/s41597-023-02767-y), [data DOI](https://doi.org/10.6084/m9.figshare.c.6503791.v1).
8. Patterson et al., “Evaluation of gait symmetry after stroke,” *Gait & Posture*, 2010. [PubMed 19932621](https://pubmed.ncbi.nlm.nih.gov/19932621/).
9. Kapoor and Narayanan, “Leakage and the Reproducibility Crisis in ML-based Science,” 2022. [arXiv:2207.07048](https://arxiv.org/abs/2207.07048).
10. Varoquaux, “Cross-validation failure: Small sample sizes lead to large error bars,” *NeuroImage*, 2018. [DOI 10.1016/j.neuroimage.2017.06.061](https://doi.org/10.1016/j.neuroimage.2017.06.061).
11. Cohen and Welling, “Group Equivariant Convolutional Networks,” ICML 2016. [arXiv:1602.07576](https://arxiv.org/abs/1602.07576).
12. Bowden et al., “Anterior-Posterior Ground Reaction Forces as a Measure of Paretic Leg Contribution in Hemiparetic Walking,” *Stroke*, 2006. [DOI 10.1161/01.STR.0000204063.75779.8d](https://doi.org/10.1161/01.STR.0000204063.75779.8d).
