# From pose estimation to trustworthy movement measurements

## A plan for an accessible research writeup on JEPA and gait

[View the rebuilt figures and full captions](FIGURES.md) · [Read the independent-review dispositions](reviews/DISPOSITION.md)

**Prepared 27 September; revised 28 September 2026 after independent review.** This is a writing and research plan, accompanied by rebuilt figures. It is not a new experimental report. The evidence comes from [paper v08](../iclr/versions/v08/paper-v08.pdf), its methods, and its retained numerical exports. Future experiments below are proposals. The supplied *HAI Internship Summer 2025* report supplies a style reference and historical context, not instructions or new study evidence.

**Recommended central question:** When a model repairs noisy observations of walking, does it preserve the movement differences we need to measure?

**Recommended contribution statement:** This study evaluates JEPA-inspired pose restoration through three complementary checks: recovery of a change in knee-motion asymmetry, accuracy of the knee-angle trajectories, and geometric left-right assignment. The findings show why improvement on one check cannot establish fidelity on the others. They motivate a research program in measurement-preserving, physically constrained movement analysis; they do not yet demonstrate a gait world model or clinical fall-risk prediction.

**Recommended title:** *Can a motion model preserve how someone walks? JEPA, gait asymmetry, and the path toward ambient biomechanics.*

The most promising next contribution is **preserving side-specific, atypical movement while improving 3D measurement under occlusion**. Physics constraints, generation, and a tool-using agent should support that testable objective. Simply connecting JEPA, WHAM, OpenSim, and a VLM would be a system integration project with unclear scientific novelty.

## 1. Preserve the earlier report's voice; strengthen its argument

The earlier report has two pages of approachable prose followed by a schematic. Its useful structure is personal and chronological: a motivating problem, a system built to investigate it, an expectation, a disappointing result, and a next step. Preserve that candor and progression. Use first person only for contributions the author can substantiate; use “we” for the research team's work and “the study” when individual attribution is unclear. Do not invent a mentor's involvement or a research chronology beyond the recorded stages.

| Earlier report | What the new writeup should do |
|---|---|
| Starts with older adults and fall risk | Start with a specific measurement failure during everyday mobility monitoring, then explain its relevance to rehabilitation and eventual risk research. |
| Explains pose extraction before VLM prompting | Explain observed joints, learned features, restored joints, and measurements in that order. Define a feature vector as a learned numerical description. |
| Lists many platform features and prompting variants | Organize around three scientific questions. Put model inventories and optimizer details in a companion methods section. |
| Candidly reports that prompting failed to meet expectations | Make the zero-response baseline and uncertain JEPA benefits visible before discussing ambitions. |
| Attributes hallucination to zero-shot classification | Describe observed failures and competing explanations; do not convert an association into a causal diagnosis. |
| Suggests heuristically mapping gait classes to fall risk | Separate movement measurement, gait classification, and prospective fall-risk prediction. Each requires different labels and validation. |
| Ends with a small, dense architecture diagram | Integrate legible, vector figures next to the claims they support, with units, uncertainty, and explicit evidence status. |

Aim for **3,200-3,600 words of main narrative, roughly 9-11 designed pages with six figures**, plus a 1,200-1,600-word methods/research companion and references. The detailed plan here is deliberately longer than that proposed narrative. Do not force every extension into a dense two-page internship summary. A later two-page summary can be derived from the complete writeup.

## 2. Motivation: an important, bounded ambient-intelligence problem

Use a clearly hypothetical opening scene: a fixed camera observes a person walking through a rehabilitation space. One leg is briefly obscured by a chair. The pose estimator jitters or exchanges leg labels. A restoration model makes the skeleton appear smoother. Has it recovered the person's movement, or replaced an informative irregularity with a more typical pattern?

The immediate problem is **reliable measurement of side-specific movement across imperfect observations**. An ambient system should help a researcher or clinician inspect whether knee motion has changed within a person, and whether the recording supports that conclusion. Chair occlusion supplies the observation failure in this example; identifying an actual chair and reasoning about support belong to future work. Ambient intelligence provides the setting in which people and their surroundings must be interpreted together. [Haque, Milstein & Fei-Fei, 2020](https://www.nature.com/articles/s41586-020-2669-y)

The biomechanics motivation should be concrete: different gait measures describe different aspects of movement. A study of nine people after stroke found persistent asymmetric joint mechanics even when step lengths became more symmetric. This supports preserving measurements rather than treating symmetry as an automatic goal. It does not establish that the present projected knee-excursion measure is diagnostic. [Padmanabhan et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7397591/)

Connect to the earlier project in one paragraph: the earlier work asked how pose and scene context could help a VLM interpret mobility. The new study examines an upstream assumption: whether the movement representation preserves the evidence that a downstream model would receive. A sophisticated explanation cannot repair a lost or mislabeled measurement without additional evidence. This is a proposed conceptual connection, not an experiment comparing the two systems.

Keep three levels of impact distinct:

1. **Demonstrated:** a controlled evaluation exposes disagreements among restoration, response, and naming metrics in the tested procedures.
2. **Plausible near-term use:** a quality-assurance layer for video-based movement measurement, after independent 3D and real-video validation.
3. **Longer-term clinical use:** longitudinal mobility assessment or fall-risk research, requiring outcome-specific cohorts, calibration, and prospective validation. No fall-risk thresholds can be inferred from this paper.

### Proposed opening passage

> In my earlier work, I explored how pose estimates and video context could help a vision-language model interpret a person's mobility. This raised a more basic question: can we trust the movement information that reaches the model? A pose sequence can look smoother after processing while losing an important difference between the legs. It can also describe a movement accurately but assign it to the wrong side.
>
> This study examines that problem in a controlled setting. We start with recorded body motions, create paired versions with a synthetic knee edit, and generate noisy two-dimensional pose estimates. We then ask whether a JEPA-inspired model can restore the poses while preserving the change in knee-motion asymmetry. The result is mixed: direct coordinate training improves the noisy observations, but none of the sixteen trained variants beats an always-zero prediction on the pooled asymmetry-change score. The lesson is that reliable movement understanding needs several kinds of evidence, not a single accuracy measure.

The first-person sentence assumes the supplied report accurately describes the intended author's earlier work. The final document should retain a personal voice without attributing all team experiments to one person.

## 3. A section-by-section writing blueprint

| Section and approximate words | The reader's question | Required content and visual |
|---|---|---|
| **Overview: why preserving movement matters** · 350 | Why should I care? | Hypothetical occlusion example, connection to ambient monitoring, central question, honest result preview. |
| **What exactly are we measuring?** · 350 | What does “asymmetry change” mean? | Knee angle, percentile excursion, right-minus-left difference, paired change; one worked example. Introduce rebuilt Figure 1. |
| **Building a controlled movement experiment** · 450 | Where do the inputs and references come from? | AMASS, person split, edits, projection, rendering, pose estimators, corruption, fixed array and support. Figure 1. |
| **What the model learns** · 500 | How does JEPA enter the pipeline? | Student/teacher, masking, endpoint versus delta, coordinate readout, controls and matched-versus-unmatched comparisons. No unqualified “self-supervised” label. |
| **What the experiments actually showed** · 750 | Did it work, and how do we know? | Restoration versus zero response; three primary effects; readout tradeoff; naming. Rebuilt Figures 2, 3, 5, 6. Reliability Figure 4 supports the companion discussion. |
| **What this means for ambient biomechanics** · 250 | What contribution survives the negative results? | Measurement validity as a design requirement; synthetic and development-only limits; no clinical utility claim. |
| **Next steps: from measurement to grounded prediction** · 650 | Which extensions are worth doing first? | Prioritized research program, all seven requested ideas briefly addressed, concrete first study and stopping criteria. Rebuilt Figure 8. Details below become a research companion. |

Each results paragraph should follow **question → comparison → numerical result → interpretation → limitation**. Introduce a graph's question in the preceding sentence; explain its main result immediately afterward. Put the “what this does not show” sentence next to the result, not in a remote disclaimer section.

## 4. Methodology: the explanation the final document must contain

### 4.1 From recorded motion to paired observations

Use a numbered preparation sequence with a single flow diagram:

1. Select walking candidates from AMASS using filenames and projected geometry. These are motion-capture-derived candidates, not a clinically verified disease cohort. Assign people to splits before generating variants; known duplicate motion hashes cannot cross splits.
2. Resample to 25 Hz and select nonoverlapping 128-sample windows. Each spans 5.08 seconds between its first and last sample. Fitting uses **112 people, 692 motions, and 1,645 windows**. Evaluation uses **14 different development people and 155 windows**. Twelve development people come from BioMotionLab_NTroje and two from KIT. Do not confuse these 14 people with the earlier Toronto report's 14 videos.
3. Pair an original body-model motion with an edited version. Nominal right-knee edits are 0°, 5°, 10°, and 15°, gated by an ankle-height proxy and tapered at window boundaries. They are controlled perturbations, not models of disease or treatment. Nonzero response evaluation excludes the zero edit.
4. Apply optional physical mirroring after the edit. Use fixed oblique and side cameras (45° and 90°); camera fitting considers the paired variants. Project body-model joints to obtain clean 2D references, including image-hidden joints. Recompute the target after projection: a 10° 3D edit need not produce a 10° measured response.
5. Render clear or occluded images and obtain RTMPose-M, HRNet-W32, and ViTPose-base estimates. Apply correct, globally swapped, or temporarily swapped input names. Rendering-derived person boxes remove person-detector uncertainty. Real deployment would add that source of error.
6. Preserve **128 frames × 12 joints × 2 coordinates**, with confidence, availability, and timestamps. A missing joint keeps its slot. Naming corruption permutes coordinates, confidence, and availability together; reference anatomy remains fixed. ViTPose and the 15° edit are excluded from optimization and calibration, but their evaluation still reuses the same development people.

Explain mirroring with care: exchanging measured left/right excursions negates their difference. Reflecting and relabeling the 3D motion before projection does not necessarily produce that exact exchange in the image plane.

### 4.2 Define the outcome before using its abbreviation

The knee angle is the image-plane interior angle between hip-to-knee and ankle-to-knee segments. For each leg, summarize motion by its 95th-percentile angle minus its 5th-percentile angle. Then compute:

$$
q_{i,\ell}=P_{95}(\theta_{i,\ell})-P_5(\theta_{i,\ell}),\quad
A_i=q_{i,R}-q_{i,L},\quad \Delta A=A_b-A_a.
$$

An **illustrative**, invented arithmetic example makes the distinction clear: original excursions of 40° left and 50° right give A = 10°; edited excursions of 40° and 55° give A = 15°, so ΔA = 5°. These are teaching numbers, not participant data. The same ΔA could arise through different changes in the two legs. Excursion also discards temporal order. Thus this scalar cannot establish which leg changed, recover the waveform, or measure gait-cycle timing. It is not a time derivative, a clinical flexion measurement, or a fall-risk score.

### 4.3 Explain JEPA as a training idea, then specify this adaptation

A student encoder converts noisy observed joints into features. A predictor estimates teacher features at masked joint/time locations. The teacher receives clean projected references during training and updates slowly from the student. It provides privileged training information; it is not available as an inference input. Published S-JEPA studies masked skeletal representation learning for action recognition. The present adaptation restores coordinates and deploys the student, rather than reproducing that benchmark. [Abdelfattah & Alahi, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4755_ECCV_2024_paper.php)

Four successive frames of one joint form a token: 32 time patches × 12 joints = **384 tokens**. Each patch has 20 inputs: x, y, confidence, availability, and time for four frames. The encoder has width 96, four layers, and four heads; the predictor has two layers. Graph-time masks hide connected joints and time regions, approximately half the available tokens, while retaining context and all slot identities. Full-window attention means this is restoration of an observed window, **not future prediction or a causal streaming model**.

Translation and isotropic scale use only available, artificially unmasked observations. References use the same transform, preventing hidden coordinates or reference geometry from leaking into normalization. The base objective uses centered teacher-student cross-entropy, with teacher/student temperatures 0.06/0.1, plus 0.05 times VICReg to discourage uninformative features. This is a DINO-style objective with paired synthetic supervision, not a claim of unmodified I-JEPA or fully label-free learning.

Explain the two auxiliary objectives without opening with logits:

- **Endpoint:** make each motion's predicted features match its own reference features.
- **Delta:** make the difference between predicted features match the reference-feature difference across the pair. A shared error in both endpoints can cancel, so a correct difference does not guarantee correct endpoints.

The exact centered-logit formulas, support, and reductions belong in the companion methods. Both objectives query common valid patches, average within pairs, and weight pairs equally. Their shared coefficient is 0.0126468, but their initial auxiliary gradient magnitudes are markedly unequal: **10% versus about 0.0013% of the base magnitude** for endpoint and delta. Six fits clip gradients on 11,999/12,000 updates. State this as a limitation on attribution, not proof of why a method performed as it did.

### 4.4 Separate learning the representation from training the readout

The readout converts features into restored joint coordinates: corrections where joints are observed and absolute predictions where they are missing. It is a LayerNorm and 96→96→8 MLP with GELU and a zero-initialized final layer. Except for direct fitting, encoders stay frozen during readout fitting.

At evaluation, each window is restored independently. Paired motions supply training auxiliaries and the response comparison; the model does not receive both development windows together to make a restoration.

Coordinate supervision is Lx. The original change package is **Lx + Ls + Lg**: coordinate error, squared scalar asymmetry-change error, and a penalty for predicted segments shorter than two pixels. Comparing this package with coordinates alone changes two losses together. The repair study keeps coordinate and geometry terms fixed and compares **Lx + 0.1Ls + Lg** against **Lx + λdLd + Lg**, where Ld compares paired angular changes at each supported leg and timestamp. Training-only batches match the dense term's initial gradient magnitude to the low scalar term.

Pretraining and frozen readout fitting each use 2,000 updates; direct training uses 4,000 joint updates. All use batch size 16 and seeds 17, 29, 43. AdamW peaks at 3×10⁻⁴ with 5% warmup, cosine decay, 0.01 weight decay, and clipping at norm one. Teacher momentum increases from 0.99 to 0.999. Equal update totals do not match coordinate-supervised exposure or encoder adaptation.

Explain the controls by purpose: zero response asks whether any response estimate beats returning zero; unchanged poses establish input quality; direct training tests practical restoration; initialized, shuffled-reference, coordinate-pretrained, and coordinate-delta variants constrain what feature learning adds. Retain all eight families under both objectives in Figure 2. They are configurations in one cohort, not sixteen independent studies.

### 4.5 Explain evaluation and uncertainty in ordinary language

| Check | Definition and unit | What it can establish |
|---|---|---|
| Asymmetry-change error | Absolute difference between predicted and reference ΔA; degrees | Fidelity of the selected paired scalar. |
| Knee-angle trajectory error | Mean absolute error over both legs and reference-valid times; degrees | Fidelity of the full projected knee-angle waveform. |
| Normalized location error | Joint distance divided by rendered person-box diagonal | Coordinate restoration quality; not metric 3D error. |
| Anatomical assignment | Post hoc distance comparison under original versus exchanged names | Geometric naming reliability on its eligible joint pairs. |

Eligibility comes from references, not from whether predictions succeeded. Angular checks require finite geometry, reference segments at least two pixels, at least 16 valid frames and 80% coverage. Prediction failures stay in evaluation, costing 180° for waveform and 720° for response. The zero-response baseline predicts ΔA = 0; it produces no pose and has no waveform score.

Average conditions within windows, windows within motions, motions within people, and then seeds. Retain the executed 2:1 nonheld/held response weighting and 4:1 endpoint weighting; do not rebalance the figure packet silently. Extra views do not create new people. Main/core comparisons use 2,000 paired bootstrap draws that independently resample people and seeds. Repair intervals use a t interval across 14 seed-averaged person effects; these condition on the three fitted seeds. Both miss uncertainty introduced by repeated development-driven choices.

Naming eligibility is different: bilateral hip, knee, or ankle references must be separated by at least four pixels. Predictions fail if swapping names improves summed distance by more than two pixels, if the assignment is ambiguous within two pixels, or if predictions are missing/nonfinite. Only the combined failure rate is retained. Do not relabel it “wrong-leg rate” or use 50% as an established chance line.

## 5. The results narrative and its claim boundaries

These are the numerical anchors to preserve. The source paper and its [audited evidence directory](../iclr/versions/v08/evidence/) govern every value.

| Finding | Evidence to report | Interpretation the writeup should use |
|---|---|---|
| Direct coordinate fitting improves inputs | Response 12.69° → 7.54°; waveform 18.57° → 12.07°; NLE 0.0718 → 0.0298. Paired response gain 5.14° [2.15, 8.65]. | Useful restoration in this benchmark. Improvement on a response metric occurs for 10/14 people's seed means, not everyone. |
| Zero response remains stronger pooled | 5.811°, below all 16 trained variants with the original failure scoring. Direct is 1.73° worse [0.53, 3.15]. | Smoother/more accurate poses do not establish useful pooled recovery of ΔA. The baseline is a scalar predictor, not a competing pose reconstruction. |
| Three primary gains remain unresolved | Core JEPA vs direct/change: 0.69° [−0.64, 1.98]; delta vs endpoint/change: 0.37° [−1.11, 1.76]; delta dense vs low scalar: 0.28° [−0.19, 0.75]. | All intervals span zero. The third outcome and interval method differ from the first two; no pooled effect. “Declared” does not mean preregistered. |
| Readout choice changes the point estimates | Delta/endpoint response: 9.99°/10.36° with change; 10.94°/10.23° with coordinates alone. Interaction interval spans zero. | No reliable delta advantage or established readout interaction. |
| A small difference is sensitive to failure accounting | Endpoint/delta failure rates 0.563%/0.525%; penalties contribute 4.055°/3.778°. About 74% of their 0.373° gap comes from that component. | An arithmetic explanation of the score gap, not a mechanism or a 74% failure rate. Successful contributions 6.306°/6.210° already exceed zero response, so penalties alone do not explain their pooled deficits. |
| Training toward the scalar can accompany worse trajectories | Adding the original package increases waveform means in all eight families. Core JEPA response 10.69° → 10.07°, waveform 17.45° → 22.55°. | Demonstrates disagreement between outcomes. Adding scalar and geometry together cannot identify which term caused it. |
| Lower scalar weight matters more than the extra dense term | Delta ViTPose waveform 23.17° → 19.47° → 19.19°. Original-to-low gain 3.70° [2.58, 4.81]; dense's additional benefit uncertain. | Supports sensitivity to readout weighting. Response preservation is not established; no noninferiority margin was specified. |
| Lower scalar error does not certify naming | Global-swap assignment failures: direct 23.29%, endpoint 81.12%, delta 83.40%. Pooled delta exceeds endpoint by 0.77 percentage points [0.17, 1.53]. | An exploratory geometric diagnostic with a different support and combined failure types. |

Follow with the visibility result: direct coordinate fitting beats zero response in the clear-image edit strata but not the occluded strata; endpoint and delta/change lose in all four. Rebuilt Figure 7 retains all 14 people, showing that the pooled result hides an important visibility difference. Do not call the nominal-edit groups bins of true projected change: the retained packet lacks the per-pair values needed for that analysis.

### A useful interpretation paragraph

> The most useful finding was not a winning architecture. It was the disagreement between reasonable measures of success. Direct training brought the reconstructed motion closer to the reference, but estimating an asymmetry change remained difficult. A model could also have a lower change-error estimate while failing more often on left-right assignment. This suggests that an ambient movement system should report several validated measurements and their uncertainty before a language model turns them into an explanation.

Close the completed-work discussion with the limits that change its interpretation: synthetic edits; projected rather than anatomical 3D angles; 14 reused development people and three seeds; unequal auxiliary influence and supervised exposure; no fitted PoseBERT/MotionBERT/SmoothNet external baselines; no locked-person, GAVD, or natural-video confirmation. Retained summaries permit figure regeneration, but the available packet does not include all reconstructed trajectories and checkpoints required to repeat training. The writeup must not invent patient examples or claim a fully reproducible training release.

## 6. Professional figure plan and rebuilt assets

The accompanying figures are redrawn from numerical exports, not enlarged screenshots. Six empirical figures use existing results; two diagrams are explicitly conceptual. Source hashes and plotted values are recorded in `figures/provenance.json` and `figures/plotted-values.csv`. This is figure regeneration, not a new experiment or a new statistical analysis.

| New figure | Original source | Main message and placement |
|---|---|---|
| **1. From movement to measurement** | Paper Figures 1 and 6, Equation 1 | Person split → paired synthetic motions → observations/references → feature/readout training → independent restoration/scoring. Includes explicitly illustrative arithmetic. Main methods. |
| **2. Restoration is not response recovery** | Paper Figure 2 | All eight families × two objectives; response, waveform, and paired waveform effect. Keep the zero-response line on the response panel only. Main results. |
| **3. Three questions, uncertain benefits** | Paper Appendix Figure 7 | Promote all three primary effects into the main narrative. Separate response from waveform and label interval procedures. Main results. |
| **4. Reliability changes the score** | Paper Figure 3 | Failure frequency, successful/failure contributions, all four failure-cost contrasts. Companion results. |
| **5. What readout repair changed** | Paper Figure 4 | Both encoder families, original/low/dense readouts, direct reference, paired effects. Main results. |
| **6. Geometry needs anatomical names** | Paper Figure 5 | All three naming conditions, full 0-100% scale, combined failures, no chance line. Main results. |
| **7. Every development participant** | Paper Appendix Figure 8 | Fixed person order, all seeds, clear versus occluded comparisons; ranges are seed variation, not confidence intervals. Companion results. |
| **8. A testable route to physical grounding** | New proposed architecture/research sequence | Measurements → calibrated 3D/contact evidence → dynamics checks → future prediction and optional generation/tool routing. Label every future component proposed. Main discussion. |

Production specification: 7-inch-wide originals, vector PDF and editable-text SVG, 300-dpi PNG, white background, restrained grid lines, consistent 9-11-point labels. Use blue for direct fitting, purple for endpoint features, vermilion for delta features, and neutral gray for controls. Shapes and labels carry meaning independently of color. Anatomical left/right labels never depend on these method colors. Provide grayscale inspection files. At final layout, keep effective labels at least 8 points and rebuild at the actual width if necessary.

Captions must state population, seeds, aggregation, failure costs, interval method, and exploratory status where applicable. Do not infer a paired effect from marginal interval overlap. Preserve degree versus percentage-point units. Full publication captions and alt text accompany the figure gallery. No generated picture should masquerade as a reconstructed participant or a future experimental result.

## 7. Evaluate the seven research extensions as separate hypotheses

### 7.1 How can a vision model become good at understanding physics?

**Operational definition:** predict how an observed state changes under specified conditions, remain consistent across camera views, and respond appropriately to interventions outside the training distribution. A visually plausible skeleton or low latent prediction loss is insufficient. Physion motivates testing physical outcomes rather than verbal descriptions, while V-JEPA 2 illustrates the additional role of action-conditioned training for planning. Neither supplies evidence that the present gait model already has these abilities. [Physion](https://arxiv.org/abs/2106.08261), [V-JEPA 2](https://arxiv.org/abs/2506.09985)

**Proposed experiment:** first forecast short-horizon 3D gait/contact states from strictly past observations. Compare persistence, constant-velocity or phase-aware prediction, a non-JEPA sequence model, and a matched JEPA predictor. Add controlled support, friction, or obstacle interventions only in simulation or safely observed reference data, with the intervention represented explicitly. Specify whether conditioning is on an observed environment, intended task, or genuine control signal; a knee edit is not an action label.

Train temporal predictive features together with measurable state/contact objectives. Hold out people and physical conditions; randomize appearance separately from physics. Evaluate horizon-dependent trajectory/contact error, outcome accuracy, uncertainty, and response to the intervention. Add oracle-state and oracle-contact variants to distinguish perception failure from predictive failure. No future frames may enter the context encoder.

Apply that restriction to the entire input pipeline: pose extraction, temporal smoothing, normalization, camera/scale fitting, SLAM, and agent tools must use only the observed prefix. Rerun temporal tools on prefixes when needed. A truncation check must confirm that changing withheld future frames cannot change inputs or forecasts. Report observed duration, prediction horizon, and algorithmic latency. Full-clip restoration remains a separate, legitimately noncausal task.

**Decision:** high scientific value after measurement validation. A forecast model must beat matched simple predictors on untouched conditions while preserving atypical motion. If it only reconstructs observed windows or recognizes familiar-looking clips, keep the claim at representation/measurement learning. General “physics understanding” remains broader than the bounded tasks demonstrated.

### 7.2 How can we estimate whether movement is physically grounded?

Use **an evidence profile**, not one universal realism score:

| Level | Proposed checks | Additional evidence needed |
|---|---|---|
| Observation and identity | Reprojection, missingness, anatomical sides, view consistency | Calibrated images and independent landmarks/identity labels |
| Kinematics | Bone-length stability, joint ranges, temporal continuity | Metric 3D, joint conventions, participant-specific variation |
| Contact and environment | Penetration, foot sliding during independently labeled support, hand-chair contact | Floor/object geometry and contact labels with uncertainty |
| Dynamics | Unactuated-base residuals, plausible forces/torques, contact constraints | Body inertial parameters, external loads or validated estimates, appropriate biomechanical model |
| Predictive validity | Accuracy under new support/scene conditions | Held-out interventions and future-state references |

Anatomical identity matters for measurement even when either labeling is mechanically plausible. A real asymmetric movement may be physically valid; smooth, symmetric output may be wrong. Include atypical and assisted movements in the valid reference set. Use controlled hard negatives with one corruption at a time, plus independently reviewed real reconstruction errors; hold out corruption mechanisms so a detector cannot simply learn an editing artifact.

Report false acceptance of invalid motion, false rejection of valid atypical motion, calibration, and risk-versus-coverage when the system abstains. If a probability of validity is reported, define the labeled event and validation population. “Uncertain” should remain available when scale, support, or external force is unobserved. Confidence is not physical truth.

### 7.3 S-JEPA for discriminative and generative purposes

**Discriminative first:** freeze the representation and test readouts for continuous validated movement measures, contact phase, laterality, or properly labeled gait categories. Compare raw coordinates, simple kinematic features, random features, direct supervision, and established motion models under matched splits and head capacity. Then compare controlled fine-tuning. Gait classification and fall-risk prediction remain separate tasks.

**Generative second:** add a separately trained stochastic motion decoder or conditional diffusion model. Condition on observed motion, intended edit or task, and eventually scene/contact state. JEPA's feature predictor is not itself a joint-sequence generator. Motion Diffusion Model is a relevant generative baseline. [MDM](https://arxiv.org/abs/2209.14916)

Define the output precisely: metric 3D joints, or SMPL pose rotations, body shape, global orientation and translation, from which joints are computed. “SMPL keypoints” means joints derived from a body model; sparse keypoints do not uniquely identify all its parameters. Fixed body shape across a sequence and a tested skeleton mapping are necessary. [SMPL](https://smpl.is.tue.mpg.de/)

Compare an identical generator conditioned on JEPA features, coordinate features, and no pretrained features. Evaluate requested-edit fidelity, distribution coverage/diversity, side identity, trajectory and contact accuracy, and physics violations. A distributional similarity score alone can reward visually plausible but clinically misleading normalization. Independent reference measurements must verify that the generator preserves the specified asymmetry. This branch is promising but not required for the first measurement paper.

### 7.4 OpenSim or robot simulation as a training signal

Start with an **offline diagnostic**, then a constrained training experiment. OpenSim inverse dynamics uses kinematics, inertial parameters, and measured or modeled external forces. It does not turn uncertain 2D joints into ground-truth kinetics. Inverse kinematics estimates a motion consistent with a body model; inverse dynamics estimates forces/torques; forward simulation predicts movement given forces/controls. Keep those roles separate. [OpenSim inverse dynamics](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53090063)

Evaluate decoded metric 3D states in a shared coordinate frame. A conceptual residual is M(q)q̈ + h(q,q̇) − Sᵀτ − Jᵀf. Freely choosing torques and contacts can make this residual uninformative. Constrain external forces/contact and inspect unactuated-root residuals, feasible torques, and independent measurements. Low residual under one assumed body/force model is conditional evidence, not a certificate. Differentiate only when the solver and gradients are validated; otherwise use cached teacher outputs, a checked surrogate, or explicit black-box optimization.

Propose an ablation: observation loss alone; plus geometric/contact losses; plus physics-derived supervision, holding model, data, optimization budget, and readout adaptation comparable. Vary force and anthropometric uncertainty. Evaluate against synchronized motion capture/force plates, including valid atypical or assisted movement. Penalizing symmetry or forcing every trace onto a healthy-motion prior would defeat the goal.

Robot/humanoid simulation is useful for controlled perturbations and contact experiments, but robot torque limits and control policies do not validate human muscle forces. Physics-guided motion generation already has precedents such as PhysDiff. [PhysDiff](https://arxiv.org/abs/2212.02500)

**Novelty check:** OpenCap's 2023 validation is a reference for the measurement study design. The **March 2026 OpenCap Monocular preprint** already refines WHAM estimates and uses biomechanical modeling to estimate kinematics/kinetics. Compare against it if compatible artifacts are available; otherwise disclose that an essential external comparison remains unexecuted. The proposed novelty is measurement preservation under occlusion and atypical movement, not “WHAM plus OpenSim.” [OpenCap 2023](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462), [OpenCap Monocular preprint](https://arxiv.org/abs/2603.24733)

### 7.5 Accurate depth on top of the movement embedding

Treat the existing embedding as a candidate motion prior. Normalized 2D joints do not uniquely determine metric scale, depth, camera motion, or object geometry. Add RGB features, camera intrinsics/extrinsics where available, a scene/floor representation, and an independent metric anchor such as calibrated multiview/depth observations or known geometry. A monocular learned metric prediction relies on priors and must be checked in the deployment setting.

Compare a verified depth tool alone, WHAM alone, tool fusion with frozen JEPA features, and controlled encoder adaptation. Depth Anything V2 distinguishes relative-depth models from metric variants trained with metric labels; these outputs are not interchangeable. WHAM fuses 2D motion with image and camera-motion evidence and contact-aware refinement. [Depth Anything V2](https://arxiv.org/abs/2406.09414), [WHAM](https://arxiv.org/abs/2312.07531)

Validate body and scene estimates in the same frame: metric joint/root error without per-example similarity alignment, depth error at relevant surfaces, floor height, drift, minimum foot clearance with its uncertainty, and contact accuracy. Report alignment-normalized metrics only as supplementary diagnostics because they can hide scale and placement errors. Evaluate held-out rooms, views, clothing, and people; audit anatomy mappings between tool conventions.

**Decision:** a high-priority enabling study. Proceed only if motion features improve independent 3D measurements over the tools alone, rather than merely producing smoother depth. Recovering unobserved scale from the old embedding alone is not a defensible promise.

### 7.6 A Qwen agent that selects tools and learns with RL

Formulate this as **sequential acquisition of measurement evidence**, not unrestricted clinical reasoning. Given a clip, current estimates, missingness and uncertainty, the agent can request segmentation, depth, WHAM reconstruction, or a validated biomechanical check, then produce a structured measurement report or abstain. A pinned Qwen-VL checkpoint with structured tool support is a candidate controller; the exact version should be selected by a small reproducible benchmark. Qwen2.5-VL establishes a relevant multimodal/agentic starting point, not demonstrated gait competence. [Qwen2.5-VL](https://qwenlm.github.io/blog/qwen2.5-vl/)

First implement a deterministic pipeline and explicit tool contracts. Every result needs units, coordinate frame, joint convention, timestamp alignment, validity mask, uncertainty, tool version, and provenance. SAM supplies masks/tracks, not metric depth, anatomical labels, or physical contact; WHAM and depth outputs remain estimates. Shared training priors can make tool errors correlated, so agreement is not independent confirmation. [SAM 2](https://arxiv.org/abs/2408.00714)

Learn from successful tool traces before RL. Compare fixed all-tools, a hand-coded uncertainty rule, a supervised router, and RL under the same cost budget and test clips. Use a reward such as **reduction in independently measured error − tool/latency cost − unsupported-measurement penalty**; define abstention utility and target coverage so “always abstain” cannot win. Train rewards use separate reference data; references are unavailable to the deployed controller. Test selected tool failures, inconsistent frames, occlusion, and novel rooms. Do not reward the agent with only its own VLM score or the same simulator residual it can exploit.

A contextual bandit may suffice if one call settles the question; use multi-step RL only if sequential observations demonstrably change the optimal next action. Tool-selection actions are distinct from physical actions in a gait world model. A defensible contribution is lower measurement error at matched cost, or lower cost at prespecified error/coverage, on untouched people and scenes. This is conditional, later work.

### 7.7 Generate human and object motion together

Begin with one bounded interaction: a chair-rise task with known chair geometry, before walking with arbitrary moving objects. This task is a new data domain; it is not contained in the completed gait benchmark. A later extension could study a cane or walker. Predict human body motion, object pose, and contact state in a common metric frame. Objects need geometry/extent and rigid or articulated structure; keypoints alone do not describe penetration or support. Add mass/friction only when making dynamic claims, and propagate uncertainty in those parameters.

Represent a rigid object's tracked keypoints through one rigid pose, rather than allowing each keypoint to drift independently. Include stable body shape, explicit anatomical side identity, foot-floor contacts, and hand/seat contacts. Condition a stochastic generator on the observed prefix, object geometry, and a bounded task or waypoint. CHOIS and InterDiff are existing human-object generation precedents, so appending object tokens is not sufficient novelty. [CHOIS](https://arxiv.org/abs/2312.03913), [InterDiff](https://arxiv.org/abs/2308.16905)

Compare human-only generation, object-conditioned generation without contact supervision, and contact/physics-informed generation. Hold out people and chair shapes, sizes, and support configurations. Score task completion, human/object penetration, rigidity, support and contact timing, rollout diversity, and preservation of instructed asymmetry or assistance. This creates a direct ambient-intelligence contribution: explaining movement in relation to an environment, while exposing which physical assumptions remain unobserved.

The fixed-chair task is a **contact-conditioned body-generation milestone**, not evidence of generating object motion. Follow it with a bounded moving-object task, such as carrying a rigid box along specified waypoints. Generate the human and box trajectories jointly; compare against object persistence and separately generated human/object streams, as well as a coupled model without contact constraints. Hold out object geometry and paths. Evaluate object translation/rotation, inter-object keypoint rigidity, contact timing, and human-object coordination. This second milestone directly tests the requested joint movement generation.

## 8. A focused research sequence with decision criteria

The following is a proposed order, not a calendar promise. Fresh data, assets, and compute availability determine scheduling. Each study should retain a useful negative outcome rather than requiring a JEPA win to justify publication.

Maintain separate data for fitting, model/policy/reward/threshold selection, and final evaluation, split by person and by scene/session where applicable. Select practical margins using a separate pilot or external requirements. Reserve a final cohort until the complete selected pipeline is fixed, or obtain fresh confirmation data whenever an earlier result influences a later stage. Never reuse the final cohort to tune reward weights, abstention thresholds, tool versions, or noninferiority margins. Availability and license compatibility of body models, tools, reference datasets, and external baseline artifacts are concrete feasibility dependencies.

| Stage | Minimum experiment and controls | Evidence required before advancing |
|---|---|---|
| **1. Confirm measurement fidelity** | Restore access to predictions/checkpoints or rerun transparently; record per-pair targets; compare direct, core/endpoint/delta with matched auxiliary influence and separately matched adaptation/exposure. Retain zero and simple temporal controls; add a task-compatible external refiner. | Pre-record primary contrasts, failure costs, exclusions and practical margins; evaluate untouched people once. Estimate participant requirements from desired interval precision, not the number of rendered clips. If locked people were inspected, obtain a fresh cohort. |
| **2. Establish 3D and contact validity** | Paired natural videos plus calibrated 3D/contact references; tools alone versus JEPA fusion. Include typical, atypical, occluded and assisted motion as the sample permits. | Better reference-based measurement at prespecified coverage without erasing side-specific change. Report inability to measure when scale/contact are unresolved. A null JEPA result still defines a strong tool-only baseline. |
| **3. Test physical constraints** | Offline diagnostics, then geometric-only versus dynamics-informed training. Include oracle geometry/force conditions and parameter sensitivity. | Independent accuracy or calibrated rejection improves while atypical-motion error stays within a predeclared acceptable margin. Thresholds come from pilot repeatability and domain requirements, not invented clinical cutoffs. |
| **4. Branch into prediction or generation** | Choose causal gait forecasting first, or a bounded chair interaction generator; use matched simple/generative baselines and held-out conditions. | Demonstrated future-state/contact or interaction fidelity, beyond observed-window reconstruction. No claim about every form of physics. |
| **5. Learn selective tool use** | Fixed, rule-based, supervised, and RL routers with common tool versions and budgets. | A measured error-cost advantage and calibrated abstention on new people/scenes; otherwise keep the simpler router. |
| **Separate clinical program** | Prospectively define target population, outcome horizon, reference assessments, confounders, and fall/mobility outcomes with clinical collaborators. | Calibration and incremental value over relevant clinical baselines. Gait labels and synthetic asymmetry do not substitute for these outcomes. |

**Recommended first new paper:** *Preserving atypical gait in video-based movement reconstruction under occlusion.* Make the primary question whether a representation or constraint improves independently measured 3D knee trajectories while preserving the magnitude and side of a movement change. Measure response error and naming on common support, and report abstention. A prespecified noninferiority margin for preservation must come from measurement repeatability and the intended use, not from the current development scores.

The first executable experiment is the **locked-person 2D confirmation** in stage 1, once required assets are recovered or the pipeline is rerun. The proposed 3D paper is a subsequent study with a new acquisition/reference protocol:

- **Observation contrast:** digitally occlude an otherwise identical target-camera recording while retaining independent synchronized 3D references from unobstructed measurement sources. Clear and corrupted inputs represent the same movement. This isolates observation robustness; it is not an intervention on the person's motion. Physical occlusion and natural scenes remain later external-validity checks.
- **Movement contrast:** acquire prespecified within-person trial conditions or sessions, with each leg's actual change computed from independent 3D references. An instruction to alter a movement does not define the target magnitude. Predefine trial/window pairing; do not interpret nonrandom session differences as causal treatment effects.
- **Measurement contract:** fix an anatomical coordinate convention and reference method for 3D knee flexion, left/right identity, support, time alignment, window duration, and handling of assistance. Use the same trial pairing for original and reconstructed estimates. Report per-leg trajectory and excursion-change errors alongside any right-minus-left scalar.
- **Primary preservation outcome:** for each paired trial, average the absolute error in the reference-defined excursion change of the left and right legs. This avoids equal errors in the two legs canceling inside one signed asymmetry difference. Compare candidate versus the prespecified strongest fitting-stage baseline on the same occluded trials. Predefine whether the decision requires noninferiority on this outcome followed by superiority on trajectory error; use a fixed testing hierarchy and a pilot-derived practical margin. Report side-assignment failure separately and retain all eligible prediction failures. The existing projected ΔA outcome remains a continuity measure, not an interchangeable 3D endpoint.

This protocol distinguishes removing observation errors from preserving an actual movement difference. It also makes the proposed 3D primary an explicit new outcome rather than a silent redefinition of the paper's endpoint.

The plausible broader contribution is a **measurement-preserving evaluation protocol and, if validated, a selective reconstruction system**. This can contribute to biomechanics by preventing misleading normalization and to ambient intelligence by making contextual estimates auditable. Neither contribution requires presenting JEPA as already superior.

## 9. Writing, review, and production workflow

1. **Freeze the evidence.** Record source PDF and CSV hashes. Map every numeric claim to its evidence file and comparison. Distinguish completed primary, exploratory, illustrative, and proposed material.
2. **Draft the six-figure narrative.** Use the proposed section sequence and word budget. Explain ΔA before introducing performance, and teacher/readout supervision before labeling the method JEPA-inspired. Keep the research companion separate from the main story.
3. **Integrate the rebuilt figures.** Place Figures 1, 2, 3, 5, 6, and 8 in the main writeup; Figures 4 and 7 in the companion. Do not add invented pose reconstructions to make the article look more complete.
4. **Run evidence and adversarial review.** Check claims, units, interval definitions, denominator changes, and novelty against current literature. The independent reviewer must be able to reject the narrative premise or proposed experiment. Record each objection and the revision, not just a favorable score.
5. **Edit for accessibility.** Ask whether a reader can explain the question, data pipeline, negative result, and next experiment after reading the opening and figure captions. Remove software inventory that does not affect interpretation.
6. **Produce editable and shareable versions.** Use a single prose source for an editable Word/Google Docs version and a designed PDF. Keep full-resolution SVG/PDF figures and numerical provenance. Render every page and inspect final-size labels, page breaks, grayscale contrast, citations and cross-references. Verify equation glyphs and alt text.

The current deliverable completes the planning, source analysis, figure rebuilding, and independent critique. Drafting the final narrative and conducting any proposed experiments are distinct subsequent activities; neither is represented as completed here.

### Acceptance criteria for the final writeup

- A reader can distinguish 2D restoration from forecasting, physics validation, and fall-risk prediction.
- All three uncertain primary results, the zero-response benchmark, and the most useful positive restoration finding appear in the main narrative.
- Data preparation, split inheritance, model training, readout supervision, support/failures, weighting, and uncertainty are explainable without reading code.
- All seven research ideas have a role, a baseline, an independent evaluation, and an explicit dependency; the first next experiment is unambiguous.
- Every empirical figure is reproducible from its cited numerical inputs; illustrative and future diagrams cannot be mistaken for results.
- The adversarial review's material objections are resolved or retained as explicit limitations.

## 10. Source and evidence map

**Local primary evidence:** [paper](../iclr/versions/v08/paper-v08.pdf), [main source](../iclr/versions/v08/paper-v08.tex), [appendix](../iclr/versions/v08/appendix.tex), [technical supplement](../iclr/versions/v08/supplement/technical-details.tex), [numerical exports](../iclr/versions/v08/evidence/), and [figure provenance](figures/provenance.json). The earlier HAI PDF is a supplied historical/style reference, not confirmation of clinical claims.

**External sources:** inline links identify primary papers and official documentation. The research suggestions and priority order are proposed deductions from the evidence, not results reported by those sources. The OpenCap Monocular item is explicitly a 2026 preprint. Tool names illustrate a reproducible prototype choice rather than a claim that a particular release is the newest or universally best. The final narrative should cite these works where the argument needs them; it need not reproduce a catalog of every tool.

**Review record:** [independent initial critique](reviews/initial-adversarial-review.md), followed by a separate draft review and a disposition record. Those records describe critical assessment, not independent experimental replication.
