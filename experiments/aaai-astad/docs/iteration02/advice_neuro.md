<!-- iteration02-execution-status -->
> This is the preserved design or feasibility record. Consult the [implemented protocol](protocol.md), [verified results](results.md), and [external execution report](external_replication_results.md) for current execution status.

# Neurological reasoning and novelty advice for the next iteration

This is a proposed research direction, reviewed on November 15, 2026. It is not a new experimental result. I read the current manuscript, its adverse controls and the preceding neurological review, then checked the primary sources linked below. The existing result remains unchanged: the learned coordinate profile does not demonstrate an advantage over the random-encoder coordinate control, and the full-context predictor fails the normal-data adequacy check in most folds.

## Recommendation

The most defensible next question is: **Does the relation between body segments add useful information after we account for how far, how fast and in what posture each segment moves?** Keep both parts of that question visible. An abnormal pattern can have unusual amplitude with otherwise regular timing, unusual timing with ordinary amplitude, or a stable compensatory combination of both. A single score that treats every abnormality as a loss of coordination is poorly matched to the clinical evidence.

For example, consider two walking clips in which the left arm barely moves. One person may have reduced arm excursion while preserving the timing of the remaining arm movement. Another may be holding a walking aid. If phase is unreliable when excursion is very small, the model should report that limitation and retain the amplitude observation. It should not manufacture a precise phase estimate or diagnose Parkinson's disease. Conversely, a person can move both legs through substantial excursions but place successive steps at an unusual relative time. The distinction is between observable movement patterns, not between inferred brain lesions.

I recommend a small, mechanism-sensitive **two-channel study**, using an adequate coordinate predictor and straightforward clinical-motion baselines before relying on JEPA embeddings. One channel preserves individual movement size, posture and cadence. The other measures relationships conditional on the permitted movement history. A source-matched control should change the relationship being tested while preserving as much of the individual motion and observation process as possible. The contribution would be demonstrated separation of these explanations under realistic video limitations. It would not be the invention of phase coordination or normal-only reconstruction.

This is a candidate contribution, not established novelty. Its strongest version needs a real-data result beyond elementary phase/asymmetry features and a measured reduction in dependence on pose quality or source differences. If those results do not occur, retain the existing audit framing.

## What the clinical evidence actually supports

| Family | Relevant primary finding | Observable hypothesis worth testing | Interpretation boundary |
| --- | --- | --- | --- |
| Stroke | A 56-person chronic-stroke study found that interlimb coordination and temporal symmetry had different associations with walking function and motor impairment. A separate study found relationships between asymmetric mechanical contributions and coordination during walking and pedaling. | Separate relative timing from left/right movement amplitude and stance proxies. Test whether an added-context contribution remains after the simpler asymmetry descriptors. | The impaired side is not verified in GAVD. Mirror-sensitive results must not be called lesion-side localization. Pose alone does not measure propulsion or descending neural drive. |
| Parkinsons | Morris and colleagues studied stride-length and cadence regulation; Sterling and colleagues found that medication effects on arm amplitude and symmetry differed from effects on bilateral arm coordination. | Preserve arm excursion and step-length proxies while measuring timing separately. Test whether the relation score supplies information beyond movement size. | A small arm swing, slow walk or regular short step is not a diagnosis. Medication, dual tasks, carrying objects and walking aids can alter the same observations. |
| Cerebral palsy | Clinical studies link selective motor control and gait abnormality, and EMG studies report changes in neuromuscular-control complexity. Different musculoskeletal constraints also contribute to crouch. | Retain posture, projected knee flexion and coupling of segment movements. Look for unusually constrained as well as unusually irregular relations. | Reduced apparent dimensionality can arise from mechanics and task constraints. It does not recover muscle synergies, selective motor control or neural injury from video. CP is heterogeneous. |
| Myopathic | Longitudinal observations in Duchenne muscular dystrophy relate weakness and contractures to changing trunk, pelvis and lower-limb patterns. | Retain trunk inclination, pelvic behavior and lower-limb excursion together. Test whether stable trunk–leg adaptation is missed by a pure irregularity score. | Duchenne findings do not characterize every GAVD myopathic clip. Projected pelvis and trunk geometry cannot establish muscle strength or joint moments. Myopathic is a muscle-related category, not one brain disease. |

The stroke statements are supported by [the 2024 coordination/symmetry study](https://pubmed.ncbi.nlm.nih.gov/38709654/) and [the walking/pedaling study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8904653/). The Parkinsonian distinctions come from [Morris and colleagues](https://pubmed.ncbi.nlm.nih.gov/7953597/) and [Sterling and colleagues](https://pubmed.ncbi.nlm.nih.gov/25502948/). The CP findings come from [selective motor control and gait abnormality](https://pubmed.ncbi.nlm.nih.gov/27889619/) and [Steele and colleagues' EMG study](https://pubmed.ncbi.nlm.nih.gov/26084733/). The muscle-disease example comes from [Vandekerckhove and colleagues' longitudinal Duchenne study](https://link.springer.com/article/10.1186/s12984-025-01718-5).

These papers motivate a measurement design. They do not establish that our extracted landmarks reproduce the clinical measurements, or that the folder labels specify each person's actual impairments.

## Compare three directions before committing

| Candidate direction | Potential scientific value | What makes it vulnerable | Recommendation for this deadline |
| --- | --- | --- | --- |
| Separate individual movement from relational information using an adequate coordinate predictor and source-matched controls. | It directly addresses why the current score is difficult to interpret and could identify a useful anomaly signal beyond simple movement descriptors. | Conditional modeling and gait coordination already have substantial prior art. Artificial disruption may be easier to detect than real abnormal gait. | Make this the principal candidate, provided the predictor first passes known-dependency and normal-development checks. |
| Study which common preprocessing choices erase clinically motivated movement signatures. | A paired comparison could show that normalization removes useful amplitude, asymmetry, trunk compensation or timing information while making a representation look cleaner. | Showing a transformation removes the quantity it is designed to remove is true by construction. It becomes a research result only if the change affects held-out movement interpretation or anomaly discrimination beyond nuisance reduction. | Use as a focused supporting experiment with a few declared transformations, not a large search for the best normalization. |
| Infer multiple compatible impairments with a physical simulator and choose the next measurement that separates them. | This could eventually make a meaningful world-model contribution: preserving ambiguity rather than forcing a diagnosis from video. | GAVD lacks measured forces, EMG, strength and lesion information. Building and validating a biomechanical simulator and clinical inverse model is unlikely in days. | Preserve as a longer-term research program. Do not make it a claimed contribution of the next workshop draft. |

The first two directions can reinforce one paper: a deliberately preserved phenotype channel plus a tested relation channel, with preprocessing treated as an explicit scientific choice. They should not become two unrelated algorithms or a new collection of favorable ablations.

## A concrete measurement design

Use only the existing approved BlazePose joints. Begin with a small set of readable signals: projected knee-angle proxies, pelvis-relative ankle displacement, shoulder-relative wrist displacement, and the trunk axis between shoulder and hip midpoints. Use the relevant view labels and state what the two-dimensional projection permits. These are projected kinematic descriptors, not three-dimensional clinical joint angles.

Keep the following information in separate reported groups rather than blending them immediately into a learned score.

| Group | Example measurements | Why retain it |
| --- | --- | --- |
| Movement size and posture | Robust excursion, side-to-side excursion difference, projected knee-angle distribution and trunk inclination. | Small steps, crouch and trunk compensation may be consistent and easy to predict. |
| Raw-clock timing | Observed cycle durations, step-event proxies when validated, time lag and irregularity on sufficiently long sequences. | Time normalization can remove meaningful cadence and temporal asymmetry. |
| Relative relationships | Phase difference where identifiable, cross-prediction residuals, or a limited arm–leg/leg–leg relation profile. | This tests whether the combination adds information beyond the first two groups. |
| Observation support | View, valid-frame fraction, confidence, track continuity and phase-estimation eligibility. | A movement descriptor is not interpretable when the necessary evidence is absent. |

Do not call the third group independent of amplitude merely because a mathematical phase formula is used. Small excursions make phase estimation noisier, and speed can change normal coordination. The [four-limb coordination study by Galor and colleagues](https://pmc.ncbi.nlm.nih.gov/articles/PMC11435969/) directly reports an amplitude–coordination association in healthy adults. It does not validate a video estimator in PD or CP.

The current 32-sample windows at 15 Hz span approximately 2.07 seconds between the first and last sample, calculated as 31 divided by 15. They can support masked completion, but many contain too few cycles for credible stride-to-stride variability. First inventory longer contiguous walking intervals using the fresh annotations and pose timestamps. Do not join unrelated sequences or bridge occlusion to manufacture cycles. A minimum cycle count is an engineering gate, not a clinical validation claim. Freeze it using normal development data and independently checked visible events before opening any new comparison outcomes. Report the source coverage lost at that gate.

Keep an inclusive source analysis for the existing binary endpoint and a clearly identified longer-sequence analysis for phase-dependent measurements. Do not treat the latter as the entire GAVD population. Do not silently replace the all-non-normal endpoint with the four focal condition labels because it sounds more clinical.

## Why phase normalization can erase the answer

A single shared resampling of a complete gait cycle can make waveform comparisons easier, but its original duration must remain available as a separate measurement. Independently warping the left and right legs to identical phases can remove the temporal asymmetry that the study is meant to measure. Aligning every arm to its corresponding leg can similarly remove arm–leg timing differences. Per-joint unit-variance normalization can erase reduced excursion. Rotating the trunk upright at every frame can remove compensatory trunk inclination and can inject a changing coordinate system into limb trajectories.

A useful small comparison is therefore: pelvis translation plus one fixed scale; the same representation with a shared cycle coordinate; and a declared destructive-control transformation such as independent limb phase alignment. The destructive control is not a candidate production normalization. Its role is to demonstrate which retained phenotypes an analysis is sensitive to. Report real-data discrimination and nuisance sensitivity together. An apparent gain after removing a phenotype may reflect view removal, regularization or a smaller selected cohort, not proof that the phenotype was irrelevant.

Phase extraction also creates a leakage hazard in masked prediction. Estimating the conditioning phase from the complete target trajectory reveals information from the hidden patch. Estimating it from the opposite leg gives an own-joint predictor information from another joint. Either restrict phase estimation to that branch's permitted visible observations, or introduce the shared body clock as an explicitly different information condition. Never describe an own-joint-plus-opposite-leg-clock branch as own-joint-only.

Near-zero excursion, discontinuous tracking and insufficient cycles should yield an unavailable phase measurement. Do not map an undefined phase to zero, normality or high anomaly. Retain the amplitude and coverage observations, and report how often each source is measurable.

## Design controls that test relations without claiming virtual disease

The strongest small experiment starts in an observable feature space with known signal meaning. Hold a target joint's observed trajectory fixed while replacing only permitted non-target context by a within-source alternative. For example, shift the other leg's angle time series relative to the unchanged target leg. Evaluate on the common interior timestamps rather than inventing repeated edges. This can test a model's sensitivity to relative timing, but the cropped alternative does not exactly preserve every finite-sample marginal distribution. Quantify that difference rather than claiming perfect preservation.

A circular shift preserves the sample multiset, but creates a boundary discontinuity unless a cycle representation is genuinely periodic. A Fourier surrogate can preserve a spectrum under specified assumptions, but it does not automatically preserve joint limits, multi-joint anatomy or the observation process. Moving each landmark independently can break limb attachments. Moving an articulated limb relative to its anchor preserves some geometry but changes coordinate trajectories and still may produce an implausible whole-body movement. Choose one method, state exactly which quantities it preserves, and verify those quantities numerically before interpreting its output.

Use feature-space surrogates as diagnostic inputs if physical plausibility is not established. Do not render them as a person with a simulated stroke or label them as a neurological mechanism. The target trajectory, target eligibility, scoring timestamps and missingness masks must match between original and altered context. If those cannot be matched, do not call the score difference a context-only effect.

The finalized next-iteration construction uses permitted distal-joint offsets relative to a visible donor hip, attached to the permitted original opposite hip. It fixes the target, own-leg observations and original body normalization anchor. Donor and current hips must be visible and permitted, and each aligned-versus-shifted pair uses the same target keys and its own permitted-context intersection. Recheck prediction adequacy on those exact sparse aligned inputs. The primary timing effect is frozen both-leg squared error under shifted context minus aligned-context error; calibrated score changes are secondary. This is a distal-leg timing challenge rather than a shift of an entire leg trajectory. The primary relational quantity compares the capacity-matched own-leg correction with the both-leg correction. The [canonical instructions](../../../../notes/aaai/05-iterate-02-INSTR.md) give the binding definitions.

A whole-body temporal shift is a crop/phase control, not spatial translation. A rigid image translation is a separate observation-invariance control. Small observation noise and dropout are separate damage controls. Keep their units distinct from a standardized profile score. The previous draft's audit correctly showed why confusing these changes produces a false bug diagnosis.

## Falsification tests and decision rules

1. **Test the estimator before testing the biological interpretation.** On a synthetic process with an independent target, adding context should not produce a stable advantage. On a process with known cross-variable dependence, the selected predictor must recover a declared useful improvement. Match marginal amplitudes and noise across these two processes. This verifies an estimator property, not gait validity.

2. **Preserve the own-context solution.** Use a model family in which an added-context branch can fall back to the own-context predictor, and evaluate its normal-development generalization. Containment of a function class does not guarantee optimization or better held-out risk. Keep the fallback explicit and diagnose failures before claiming biological information.

3. **Test amplitude and timing separately.** Reduce one signal's amplitude without changing its phase, then change relative timing without changing the target. A purported relation score that responds only to amplitude or confidence has failed the intended interpretation. A combined anomaly detector may still use amplitude, but must say so.

4. **Test the correct known invariances.** Left/right relabeling with coordinate reflection should exchange sided explanations rather than change a side-agnostic anomaly score for purely indexing reasons. A side-specific score can transform predictably instead of being forced invariant. A camera rotation in image space is not necessarily a harmless view change and should not be assumed invariant without a declared observation model.

5. **Test common-support observation damage.** The relation effect must be compared with noise, missingness and tracking discontinuity at the same valid target timestamps. Confidence is not a calibrated error probability. A recent [confidence-filtering study](https://arxiv.org/abs/2610.39704) already distinguishes measurement-error reduction from movement-selection shift, so reproducing that general warning is not a sufficient novelty claim.

6. **Compare with elementary clinical-motion descriptors.** A phase-coordination or asymmetry baseline is required, not optional. If it explains the proposed score's discrimination, there is no demonstrated learned-relation advance. Report source-paired differences and the normal sample supporting them.

7. **Require an observed real-data consequence.** Better recognition of artificial context disruption is insufficient. The score must improve a declared comparison on the same real source cohort or provide an independently reviewed, reliable movement explanation. The existing videos cannot become a fresh confirmatory test because a new seed or split is used.

8. **State a stopping rule before the outcome.** If the normal-data adequacy, phase coverage or observation controls fail, stop the mechanism claim and report the specific failure. Do not choose the condition, normalization, phase extractor or score sign after examining which one produces the most appealing anomaly ranking.

For the remaining days, keep synthetic estimator checks, normal-development learning adequacy and the real-data comparison as three distinct gates. A successful first gate cannot compensate for a failed second or third gate.

## Closest prior art and the novelty boundary

| Prior work | What it already establishes | What the new study would still have to add |
| --- | --- | --- |
| [Plotnik, Giladi and Hausdorff, 2007](https://pubmed.ncbi.nlm.nih.gov/17503027/) | A phase coordination index combines the accuracy and variability of left/right stepping timing and was studied in aging and PD. | A validated video observation procedure and a useful consequence beyond an established timing metric. |
| [Galor, Zeilig and Plotnik, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11435969/) | Four-limb phase coordination and its relationship with arm-swing amplitude in healthy adults. | Evidence beyond merely adding arms to a phase score, with explicit behavior when excursion or observations are inadequate. |
| [Fukino and Tachibana, 2025 preprint](https://arxiv.org/abs/2505.10869) | A convolution/LTI model of intersegmental velocity relationships for gait asymmetry, evaluated on five subjects. | A source-separated abnormality study with matched context/marginal controls. Linear filtering or cross-limb reconstruction alone is not new. |
| [FSGait, ACCV 2024](https://openaccess.thecvf.com/content/ACCV2024/html/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.html) | Normal-only reconstruction and prediction for postural and temporal anomalies, with joint/frame/sequence outputs. | A controlled claim about information beyond marginal motion and observation effects, not just separate posture and timing heads. |
| [GenGait, current author version](https://arxiv.org/abs/2604.01997v2) | Normal-only masked transformer reconstruction, joint inconsistency and normative reconstructions; the reported proof-of-concept uses simulated abnormal patterns from held-out normative participants. | A demonstrated distinction from reconstruction/localization, particularly on heterogeneous real-source recordings. A new mask or normative completion is not sufficient. |
| [SeeKer, ICCV 2025](https://arxiv.org/abs/2506.18368) | Conditional keypoint-density modeling and keypoint-detector-confidence-weighted scoring. | Evidence that the matched information contrast or its interpretation adds value beyond conditional surprise. |
| [Beyond Marginals, TMLR 2025](https://arxiv.org/abs/2509.15033) | Separation of marginal, temporal and inter-variable structure in multivariate anomaly detection. | A gait-specific observation and validation contribution beyond renaming those components. |
| [Lochhead and Fisher, author manuscript](https://homepages.inf.ed.ac.uk/rbf/PAPERS/lochheadcbm24.pdf) | Lightweight gait analysis with a typical-cycle representation and small ST-GCN. | Evidence beyond cycle subtraction or claiming that a small model is intrinsically novel. |
| [Confidence Filtering in Video-Based Gait Assessment, 2026 preprint](https://arxiv.org/abs/2610.39704) | The distinction between confidence-selection effects on movement summaries and local pose error. | A new validated method or controlled relationship result, not the general warning that confidence selection can alter the assessed population. |

This is a targeted novelty review, not an exhaustive priority search. No “first” claim is justified. In particular, the highest-value result may be a well-supported dissociation between movement amplitude and relational evidence under realistic observation constraints. It must be measured rather than built into an attractive diagram.

## A notebook and slide sequence that teaches the reasoning

A first new notebook should show one normal and one labeled non-normal clip, the unnormalized projected signals, and what each permitted normalization removes. Use source labels only to select illustrative context, not to tune the normalization. Mark the movement observations as proxies and show low-confidence intervals.

A second should implement the known-dependency checks and the nested coordinate predictor on small synthetic data, then test it on normal development sources. Show the own prediction, added-context correction and unchanged-target controls side by side. Make the estimated information contribution readable before exposing a neural embedding.

A third should apply the frozen method to the same source-separated real cohort, alongside amplitude/posture, phase/asymmetry, matched calibrated error, random features and a validated published comparison where available. Keep unmatched sources and unavailable phase cases in an explicit coverage table. A fourth, if time permits, should show independently reviewed movement descriptions and test whether the model's proposed relation agrees with those observations. Without independent review it remains an explanatory demonstration, not clinical validation.

The revised slides should lead with the measured failure that motivates this redesign, then explain the two observable channels with one walking example. A simple diagram can show three cases: small but well-timed movement, ordinary-sized but mistimed movement, and stable compensated movement. These are conceptual examples unless tied to verified observed clips. Follow with the allowed-input diagram, the surrogate-preservation table, the three experimental gates and the source-level evidence required for a claim.

## What would make this an impactful result

The important contribution would be a verified answer to a narrow, generalizable methodological question: when does a video-based gait score add information about relationships among movements, and when is it explained by movement amplitude, observation quality or an inadequate predictor? A useful answer should survive simple baselines and identify its observation limits.

GAVD is valuable for testing this question under diverse recording conditions. Its 23 retained normal source videos, unknown participant overlap and observational labels do not establish generalization to new patients, institutions or neurological causes. Focal-condition counts remain small. Any strong manuscript should separate a convincing estimator/observation result from a claim about clinical performance, and should identify independent normal participants as the highest-value next data acquisition.
