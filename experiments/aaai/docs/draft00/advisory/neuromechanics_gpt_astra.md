**Neuromechanics and world-model research advisory. GPT-6 Astra. November 14, 2026.**

I reviewed the canonical README, research roadmap, and executive summary, then checked primary research on gait control, muscle synergies, predictive simulation, and learned motion models. No experiments were run. Everything described as a proposed model, hypothesis, or expected response below is a proposal, not a GAVD result. Literature findings are identified separately.

My recommendation has two parts. This week, test **whether abnormal gait departs from the normal pattern of predictive relationships, including both unusually weak and unusually strong relationships**. In the longer research program, build **a forward model that returns several compatible neuromusculoskeletal explanations and predicts which additional observation would distinguish them**. A video that reveals unusual movement need not reveal its neurological cause.

**The medical motivation changes the research question.**

The earlier proposed direction, that abnormal gait loses help from other joints, is plausible but not a general consequence of neurological impairment. Some impairments produce stereotyped, constrained, or excessively coupled movement. A model might find those movements easier to predict. Neither “more coordinated” nor “more predictable” is equivalent to healthier.

| Condition label | Relevant primary evidence | Modeling implication and boundary |
| --- | --- | --- |
| Stroke | Clark and colleagues found that reduced complexity of EMG patterns in the paretic leg could be described through merging of modules identified in healthy walking, and related this structure to walking performance. [Primary study](https://doi.org/10.1152/jn.00825.2009). | Acquired brain injury can alter selective control, but the observed gait also reflects weakness, compensation, and task demands. Test changes in predictive relationships in either direction. Joint trajectories alone cannot establish merged neural modules or locate a lesion. |
| Parkinson's disease | Morris and colleagues experimentally separated stride-length regulation from cadence regulation; their studies showed that these gait components need not be impaired identically. [Primary stride study](https://doi.org/10.1093/brain/117.5.1169), [primary cadence study](https://pmc.ncbi.nlm.nih.gov/articles/PMC1073238/). | Preserve amplitude and real time separately. A smooth, small-amplitude gait can be abnormal without high temporal surprise. Do not equate a latent gain parameter with dopamine loss, or assume every Parkinsonian clip contains freezing. |
| Cerebral palsy | Steele and colleagues studied EMG from a heterogeneous clinical population and found reduced synergy complexity associated with functional impairment and clinical measures. They also examined walking-speed effects. [Primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4683117/). | CP is a heterogeneous group of developmental motor disorders, with diverse brain injuries and secondary musculoskeletal effects. A single “CP controller” would erase meaningful variation. An observed kinematic pattern is not a measurement of spasticity, contracture, or neural-module count. |
| Myopathic gait | Muscle-driven simulations found that the consequences of weakness depend on which muscles are weakened and how other muscles compensate. [van der Krogt and colleagues](https://pmc.ncbi.nlm.nih.gov/articles/PMC4890623/). | Myopathy primarily concerns muscle disease, not a single central neurological lesion. Model muscle capacity and neural drive as distinct possible contributors. A proximal compensation pattern does not identify a particular myopathy or quantify strength from video. |

The strongest cross-condition insight is that similar visible movements can result from different combinations of control, muscle capacity, body structure, environment, and compensation. Conversely, the same diagnostic category can contain several movement strategies. Disease names should motivate hypotheses and stratified descriptions, not become assumed latent causes.

**What the neuromotor literature actually licenses.**

[Ivanenko, Poppele, and Lacquaniti](https://pmc.ncbi.nlm.nih.gov/articles/PMC1664897/) extracted a small set of temporal components from measured muscle activity under different walking speeds and body-weight support conditions. That is evidence about the organization of EMG in their experiment. It does not establish that five arbitrary latent coordinates learned from 2D joints recover five neural primitives.

There is a more fundamental caution. [Kutch and Valero-Cuevas](https://pubmed.ncbi.nlm.nih.gov/22570602/) used experiments and computational examples to show that biomechanical constraints can produce apparent low-dimensional muscle coordination without assuming neural coupling. [De Groote and colleagues](https://pmc.ncbi.nlm.nih.gov/articles/PMC4167006/) showed that task constraints and muscle-effort minimization can produce a small number of gait synergies in simulation. Thus, even EMG factorization requires care when interpreted neurally. Skeleton factorization is further removed from the underlying commands.

Call factors learned from GAVD **kinematic components**, **motion motifs**, or **latent predictive states**. Reserve “muscle activation,” “neural primitive,” “reflex gain,” and “motor command” for variables grounded in a specified physiological model and independently evaluated measurements. These names are scientific commitments, not interchangeable presentation choices.

**Forward simulation is established; identifying its inverse is the difficult opportunity.**

[Geyer and Herr](https://pubmed.ncbi.nlm.nih.gov/20378480/) demonstrated a reflex-controlled muscle model that produces walking dynamics and responds to environmental changes. [Song and Geyer](https://pmc.ncbi.nlm.nih.gov/articles/PMC4560581/) extended this approach to a controller containing spinal feedback modules and a higher control layer in three-dimensional locomotion. These are explicit hypotheses about how control and body mechanics generate movement, supported by simulation comparisons. Their success does not prove that an observed walker uses that unique controller.

[Falisse and colleagues](https://pmc.ncbi.nlm.nih.gov/articles/PMC6731507/) generated healthy and altered gaits using similar optimization principles with changed musculoskeletal properties. This directly undermines the shortcut “different gait implies different neural controller.” [OpenSim Moco](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008493) already supports motion and control optimization, including weakness examples. [MyoSuite](https://proceedings.mlr.press/v168/caggiano22a.html) provides musculoskeletal control environments. Reusing such infrastructure is sensible, but not itself a methodological contribution.

Several closer precedents must enter the novelty discussion:

| Existing work | Consequence for the proposed research |
| --- | --- |
| [Generative GaitNet, 2022](https://arxiv.org/abs/2201.12044) generates gait conditioned on anatomy and gait settings using a detailed muscle-driven simulation. | Generating weakness- or contracture-dependent walking is already established. |
| [Bidirectional GaitNet, 2023](https://arxiv.org/abs/2306.04161) learns mappings between gait and anatomical conditions. | Inferring anatomical parameters from gait is already a direct research topic. Claims of unique recovery still depend on its model, observations, and validation setting. |
| [Yamada and colleagues, 2025](https://www.nature.com/articles/s41467-025-61292-1) use simulated musculoskeletal gaits to improve analysis across populations and sensor settings. | Synthetic augmentation and simulation-to-video transfer alone are insufficient novelty. |
| [Muscle Synergy Priors, 2026 preprint](https://arxiv.org/abs/2603.10474) constrains reinforcement-learning control with a low-dimensional synergy basis. | A low-dimensional physiological control prior is already studied. |
| [Learned state priors for gait simulation, 2026 preprint](https://arxiv.org/abs/2610.08506) combines learned gait priors with predictive simulation using kinematics and external forces. | “Learned world model plus biomechanics” is too broad an originality claim. |
| [HuMoR, ICCV 2021](https://geometry.stanford.edu/projects/humor/) learns generative motion dynamics and fits motion from incomplete observations. | Stochastic latent transitions and completing noisy body motion are established. |

The targeted search does not prove that the longer-term proposal below is unprecedented. It identifies what cannot credibly serve as the novelty claim.

The immediate anomaly study also inherits close precedents: [Skeleton S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) provides masked skeleton latent prediction; [FSGait](https://openaccess.thecvf.com/content/ACCV2024/html/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.html) provides normal-only predictive and reconstructive gait analysis; [GenGait](https://arxiv.org/abs/2604.01997) provides masked normative gait reconstruction; and [SeeKer](https://arxiv.org/abs/2506.18368) provides conditional keypoint density scoring. [Beyond Marginals](https://arxiv.org/abs/2509.15033) already distinguishes marginal, temporal, and inter-variable behavior. Adding anatomical vocabulary to these ingredients would not create novelty. The contribution must be a demonstrated failure of a specific score interpretation and a controlled, useful correction, or a new validated mechanism-identification protocol.

**A one-week experiment: distinguish normal relationships from merely predictable motion.**

The plain question is: **“Does the rest of the body help predict a joint in the usual way, or in an unusually weak or strong way?”** This is a measured relationship in the model, not an estimate of neural connectivity.

Retain normal-only fitting, all 16 allowed joints, and uniform random masking independent of movement or labels. Fit small predictors on the identical hidden local-coordinate targets with three declared context sets:

1. **Own context** contains visible patches of the target joint at other times.
2. **Contemporaneous context** adds other visible joints from the target's four-frame patch.
3. **Full context** adds other visible joints throughout the window.

The second comparison is essential: anatomical geometry can predict a missing coordinate without learning temporal relationships. Four-frame patches already contain short temporal information, so call this a contemporaneous-patch control rather than a perfectly static model. Restrict information before contextualization and throughout attention. Evaluate identical targets and report their support.

Define two source-level summaries after the fixed window-to-sequence-to-source aggregation:

```text
b_all = mean(loss_own - loss_full)
b_time = mean(loss_contemporaneous - loss_full)
```

The proposed score is a two-sided distance of `(b_all, b_time)` from its normal reference. Start with a diagonal standardized distance whose location and scale are fitted from inner held-out normal sources, with a prespecified variance floor. Do not fit a large covariance matrix or a flexible density estimator from five or six validation sources. Report sensitivity to leaving out each reference source. If reference instability dominates, the two-sided detector is not validated.

Prespecify the existing one-sided `-b_all` as a comparator, not as a direction to choose after testing. This revision is motivated before the new experiment by the literature and by the acknowledged predictable-abnormal failure case. It remains exploratory because original tests have been inspected. Normal-reference deviation is an established scoring principle; novelty would require evidence of the specific failure it repairs, not a claim to invent two-sided detection.

The operational hypothesis is: **A normal-reference predictive-support profile detects some abnormal-labeled, individually predictable motion that raw discrepancy or a one-sided loss-of-benefit score misses, without merely detecting camera or normalization effects.** The falsifier is equally concrete: no adequate normal predictive benefit; no benefit beyond contemporaneous geometry; unstable normal references; or a paired result excluding the prespecified useful discrimination gain.

Keep the raw own-joint prediction error as a separately reported dimension or comparator. A detector based only on support can miss a wholly unfamiliar trajectory if both predictors fail similarly. Do not add arbitrary disease-specific score weights using the abnormal test labels.

Repeat the decisive comparison with contextual JEPA targets only after the local-target result is understood. A full-input teacher can place other-joint information inside the target. Its apparent benefit is therefore a different prediction problem. Freeze teacher weights, center, temperature, masks, and support. Decompose any intervention response using the actual inference loss, separately for own and full branches, then subtract if explaining the contrast. Cross-entropy and KL have different target-change terms because target entropy changes; their within-target predictor contrast cancels that entropy, but a single-branch target-effect analysis does not.

There is also a deterministic preprocessing shortcut: after pelvis centering, the two normalized hips are exact negatives at each frame. A masked hip can be reconstructed from its visible counterpart without any temporal model. Keep the 16-joint allowlist, but report joint-level contributions and the existing contemporaneous-context control. Any claim about dynamics must survive this check. Local coordinates mean no transformer contextualization, conditional on the shared body-reference convention; they are not independent of other-joint preprocessing.

**The decisive matrix should stay small.**

| Test | Necessary comparison | What would count against the proposal |
| --- | --- | --- |
| Learning adequacy | Prespecified update ladder and normal-validation curves; the matched linear version of the three predictors. | The result is a consequence of an undertrained full model or difficult optimization. |
| Timing sensitivity | A relative articulated-limb timing edit versus a synchronous crop shift, identical masks, fixed body reference, and common interior support. | The response comes from boundaries, implausible anatomy, missingness, or contemporaneous geometry. |
| Observation sensitivity | Bounded coordinate corruption and observation deletion on the same eligible windows. | Predictive-support distance mostly measures those measurement edits. |
| Actual detection | Two-sided support distance, one-sided contrast, constituent losses, raw JEPA discrepancy, normal embedding distance, and the planned validated normal-only baseline. | The proposed scoring gain disappears against serious baselines or derives from one recording subgroup. |

Do not promise that a synthetic “overcoupled” skeleton represents stroke or CP. A perfectly correlated pair of arbitrary joint signals is a useful unit-level diagnostic, not a clinically plausible intervention. The neurological motivation makes a two-sided test reasonable; it does not predict the sign of the GAVD result.

Use the existing exploratory source cross-fitting protocol and within-fold ranking metrics. Binary positives are every non-normal folder, including exercise, style, and pregnant under the user's definition. Show each folder's usable source count and missing categories. All abnormal categories are unseen during normal-only fitting. Only the seven additional folders are unseen categories for a comparator trained on the original four abnormal conditions.

There are 32 annotated normal sources and 28 in the reported processed cohort. Their small number limits the range of normal speed, view, morphology, age, and task represented, even if optimization converges. It does not become a large normative cohort through windowing. The 2.13-second windows can be too short for stable cycle-level estimates. Fifteen-hertz poses are not measurements of neural conduction delay. Do not add clinical phase inference from the sparse event labels, or erase cadence and amplitude by treating all time warps and rescalings as nuisance.

For a strong paper, require adequate normal information, a contribution beyond geometry, selective intervention evidence, and a meaningful source-level detection gain. A 0.05 AUROC target can remain the predeclared study target. A positive paired interval and point estimate exceeding 0.05 support a positive estimated gain; a lower bound exceeding 0.05 is required to claim the gain itself exceeds that threshold. Wide intervals mean inconclusive evidence, not a successful negative finding. No 5% clinical false-positive guarantee follows from these normal counts and unknown participant identities.

The week should have three gates: finish the learning and normal-reference stability checks by day two; finish the interpretation controls by day three; lock scores and complete broader evaluation by day four. Days five and six belong to the manuscript and verification. A full musculoskeletal simulation should not enter this deadline's critical path.

**The longer-term world model should represent what is unobserved explicitly.**

A useful conceptual model is:

```text
goal and sensory history -> controller -> muscle excitation
muscle excitation + muscle properties -> muscle force
muscle force + body geometry + ground contact -> 3D movement
3D movement + camera + tracking error -> observed 2D joints
```

GAVD observes the last box imperfectly. It supplies neither controlled actions nor force, EMG, strength, lesion, medication-state, or intervention measurements. A masked-completion JEPA learns conditional statistical predictions. A separately tested suffix model supplies an actionless temporal predictor. Neither becomes an action-conditioned physiological world model simply because a latent variable is named “control.” [V-JEPA 2](https://arxiv.org/abs/2506.09985) itself distinguishes actionless video pretraining from action-conditioned training using interaction trajectories.

In a future model, keep separate latent variables for movement state, body properties, effective control, contact, and camera. Use a simulator or constrained differentiable dynamics for the forward map and an observation model for 2D projection. Allow a distribution or set of compatible latent explanations. External physical parameters are assumptions or separately supervised quantities. They are not recovered diagnoses merely because a network produces values for them.

The elementary ambiguity is visible even in a toy force model: `force = strength × activation`. Within feasible activation bounds, changing strength and compensating activation can preserve force and movement. Real biomechanics adds more ambiguity through redundant muscles, passive forces, contacts, and camera projection. In a learned model `state_next = f(state) + B u`, the transformation `B -> B R` and `u -> R^-1 u` preserves predictions for invertible `R`. Thus, the magnitude of an unconstrained latent command is not an identified physiological effort. A latent penalty can choose one representation without making it biological truth.

A recent primary preprint, [Imitation Learning from Human Motion Alone Does Not Guarantee Biomechanically Plausible Gait Kinetics](https://arxiv.org/abs/2603.12408), reports that similar kinematic tracking can coexist with substantially different kinetic accuracy; adding ground-reaction-force and center-of-pressure information improved its evaluated estimates. That is evidence within its studied setup, not proof that every motion-only inference task is impossible. It reinforces the need to validate internal variables separately from visible motion.

**The more creative contribution is a test of distinguishable explanations.**

Build a benchmark of **different known simulation mechanisms with deliberately similar projected motion**. For example, adjust a strength deficit, a passive joint restriction, and an effective control-timing change so that their projected short gait clips are difficult to distinguish after realistic camera and observation variation. These are parameter interventions in the simulator, not simulated disease diagnoses. Include familiar and changed tasks, such as walking speeds or turns, under explicit simulator conditions.

The model must first fit the original observation while retaining compatible explanations. It then predicts whether the explanations disagree on a withheld observation, such as a second view, a force trace, selected EMG channels, or another task. Evaluate which measurement reduces ambiguity and whether the true generating explanation remains in the returned set. Train the normal-reference detector only on normal-reference material. Any simulator model trained with impairment interventions is a separate supervised synthetic-mechanism experiment and must be reported as such.

The falsifiable research claim is: **A set-valued forward model can preserve anomaly detection while avoiding unjustified mechanism certainty on observationally matched cases, and can predict which added measurement resolves specific ambiguities.** Compare with a single-best inverse model and with a simple ensemble. Require prospective held-out simulation mechanisms, parameter ranges, viewpoints, and tasks. A broad set that contains everything is not success; report coverage together with set size and measurement cost. A narrow incorrect set is false confidence. If additional measurements do not improve identification, report that those mechanisms remain indistinguishable under the tested protocol.

GAVD then tests robustness of the observable anomaly score and produces hypothesis-generating signatures. It cannot validate the inferred neural or muscle cause. Later validation needs participant-linked kinematics, forces, EMG or strength measures as relevant, clinical characterization, and repeated tasks or interventions. Those are new research requirements, not resources already available this week. Even then, the appropriate result may be compatible mechanisms rather than a unique cause.

**Three vector graphics would explain this without overstating the science.**

The first should show several upstream routes, labeled “changed neural control,” “changed muscle capacity,” and “changed body mechanics,” converging on similar visible movement. Add a separate camera-and-tracking branch before the 2D observation. Use no backward arrow claiming that the image uniquely identifies a cause. Its caption should say “Conceptual possibilities; no mechanisms identified in GAVD.”

The second should show the three predictor context sets around a hidden ankle patch, followed by a two-dimensional normal reference region. Place illustrative points on both sides of its center. Label the horizontal axis “benefit of other joints” and the vertical axis “benefit beyond contemporaneous geometry.” Mark every point as schematic. This conveys that abnormal relationships can be unusually weak or strong, and that “outside normal” is a hypothesis requiring actual evaluation.

The third should show two distinct simulated mechanisms producing almost identical initial 2D observations, then diverging under an added force measurement or task. The output is “both remain possible” before the additional evidence and a smaller compatible set afterward. This figure communicates the proposed scientific advance more accurately than a brain diagram with disease names attached to latent dimensions.

The near-term paper should say what information improves abnormal-gait detection. The longer-term paper should say when a forward model can distinguish explanations and when it cannot. Neither should convert an attractive latent animation into evidence of neurological causation.
