# Independent scientific options after notebook 06

Read-only evidence review and proposal screening, 14 September 2026. No models were trained, no HAIC jobs were submitted, and no existing result was changed. This memo proposes experiments; it does not forecast acceptance probabilities.

## The decision that the new evidence changes

The earlier portfolio ranked motion preservation first because it expected an otherwise useful repair model to suppress supported motion. Notebook 06 does not establish that conjunction. The adapted MoMask RVQ path damages clean input, and its pointwise truth-informed correction family barely clears the existing repair target. The correct scientific response is to demote this particular candidate, not to reinterpret rejecting it as successful repair.

The useful remaining observation is narrower: changing the video while keeping skeleton input identical moves the flow score in the expected relative direction in 15 of 16 matched pairs. Only five pairs favor the intended path in both videos. This supports testing visual evidence, but does not demonstrate calibrated single-video decisions. The diagnostic also uses reference event regions. A practical method must choose its regions from observable input.

The older future-feature result supplies a different warning: RGB target prediction improved substantially with more sources while the aligned-skeleton increment remained tiny and below a mismatched-skeleton increment. It does not prove that frozen V-JEPA features cannot improve motion measurement. It does rule out using the global target score alone as evidence of useful temporal transfer.

Sources: [pilot diagnostics](../../../docs/studies/motion-preservation/results/pilot-01-diagnostics.md), [future-feature strategy](../../../docs/studies/future-feature-prediction/scaling/research-strategy.md).

## Candidate A: Spend the next observation where it can change the answer

**Question.** Can a small frozen-JEPA head choose an additional existing video frame or crop that reduces a named movement measurement's error more than uncertainty, surprise, or motion magnitude would suggest?

The distinction is between being uncertain and being able to resolve uncertainty. A hidden foot may have an uncertain trajectory, but inspecting another frame in which it remains hidden buys nothing. A modestly uncertain ankle reversal may become clear from one sharply visible frame. The desired model predicts the value of an observation for the requested quantity, not merely the current uncertainty.

**Concrete first experiment.** Use 128 to 256 untouched AMASS motion windows from multiple existing training and development people. Render the same motion under fixed declared cameras, texture conditions, and occluders. Start with a common sparse observation set, then exhaustively evaluate a small set of permitted additional crops or frames offline. A fixed measurement head estimates continuous quantities such as pelvis-relative foot displacement over a specified interval, or the timing difference between left and right limb reversals. Independent AMASS trajectories define their reference values. The oracle uses these references only to measure the best attainable choice. The deployed selector sees only the initial observations, the query, and the candidate action description.

Fit a small value head to the actual reduction in squared measurement error after each additional observation. Use frozen V-JEPA tokens and available skeleton/flow evidence as inputs. This is ordinary supervised value-of-information learning; the novelty must come from the demonstrated distinction and transfer. It does not need a generative motion candidate or a new world-model backbone.

**Primary comparison.** Plot continuous measurement error against additional pixel, frame, or encoder-call budget. Fix one operational cost and report total preprocessing separately. Compare uniform sampling, highest detector uncertainty, highest flow uncertainty, largest motion, most surprising latent change, a raw-flow value head, and equal-capacity non-JEPA features. At one observation budget, require a development gain large enough to matter, provisionally 15 to 20 percent relative error reduction over the strongest baseline. This is a proposed decision threshold, not a prediction.

**Important constraints.** Do not encode all unrevealed high-resolution frames and then call the policy an acquisition-saving method. If low-resolution preview frames are allowed, give them to every policy and include their processing cost. A retrospective video-analysis task may inspect later frames in the already recorded clip, but must not be described as future prediction from a prefix. A prospective version must retain an exact time boundary.

**What makes the contribution substantial.** The same value estimator should select useful observations for a held-out motion query or observation-failure family. It should know when more of the same evidence is unhelpful. The result becomes a general account of what information a latent model should request, supported by a real error-versus-cost improvement. A generic frame selector on gait data is insufficient.

**Nearest work.** [ActiveMoCap](https://arxiv.org/abs/1912.08568) already chooses views to improve human pose estimation. [Video Active Perception](https://arxiv.org/abs/2605.01662) already uses a video generation model to choose existing frames. [AdaptToken](https://arxiv.org/abs/2603.28696) already uses uncertainty for token allocation and stopping. [Certified World Models as Sensing Clocks](https://arxiv.org/abs/2607.01537) already turns a frozen JEPA prediction-validity estimate into a re-observation schedule. The proposed opening is query-specific reducible measurement error under controlled observation ambiguity, not active perception itself.

**Failure condition.** Stop if the observation-choice oracle has little margin over uniform sampling, if a visibility or uncertainty rule captures the gain, or if no transfer survives holding out an observation process. These first checks should take one to two days with the existing renderer and frozen features. GAVD can test the recorded-video interface and visible movement errors only where independent annotations exist; it does not supply hidden 3D ground truth.

**Evidence strength.** Moderate feasibility, low current evidence of the central effect, conditional medium novelty. It is the freshest practical route because it removes the need for a useful pretrained repair model. It is not currently a high-confidence ICLR result.

## Candidate B: Predict whether a proposed local correction helps

**Question.** Can one small visual verifier estimate the sign and useful magnitude of local repair benefit across correction methods it was not trained on?

The output is not whether a clip looks unusual. It is whether changing this specific trajectory segment would move it closer to the independent motion reference. Train on actual local squared-error differences, using raw motion, the proposed displacement, flow, and optional frozen V-JEPA tokens. Include an exact no-change option. The same verifier should work with different repair generators without retuning its threshold on the held-out generator.

**Smallest first experiment.** Read the existing caches and calculate the oracle for the actual proposed limb/time blocks. Add a small practical repair bank, including calibrated partial temporal filters and a verified public temporal refiner. Full-strength Gaussian and median filtering already fail in the current diagnostic; do not assume they provide sufficient repair. MoMask remains a documented adverse candidate until its bounded reconstruction audit yields a useful regime.

**Primary metric and baseline.** Event preservation at the declared repair level, plus observed-joint squared error that includes damage to initially accurate joints. Compare confidence-weighted filtering, direct flow propagation, a flow-only correction verifier, a coordinate-only head, and the strongest candidate-specific rule. Hold out an entire repair method and observation-failure family. The pointwise MoMask oracle's 27.4/30.0 percent repair result is too narrow a margin to justify starting with that family alone.

**Distinctive mixed case.** Put real movement in one body region and tracking error in another. Exchange their roles across two videos while holding the observed skeleton identical. Correct decisions must change locally and in opposite directions. A clip-level real-versus-error classifier cannot solve the required output.

**Nearest work.** [MFTIQ](https://arxiv.org/abs/2411.09551) already estimates correspondence quality separately from the flow estimator. [HTD-Refine](https://arxiv.org/abs/2605.26879) already uses video-derived temporal dynamics to refine human motion while limiting oversmoothing. [WMReward](https://arxiv.org/abs/2601.10553) already uses V-JEPA to score generative candidates. A generic scorer or gate is therefore not new. The opening is calibrated local *benefit*, demonstrated across generators while preserving supported atypical movement at equal repair.

**Failure condition.** Stop before training if the block-matched oracle cannot substantially exceed the practical repair target. Stop the JEPA branch if direct flow or visibility rules achieve the same frontier. A reject-everything model is not successful repair.

**Evidence strength.** Strongest continuity with implemented assets, but conditional feasibility until candidate headroom exists. The 15/16 paired flow contrasts are motivation, not evidence of deployable correction performance. This belongs behind a strict first-day gate rather than at the head of an unconditional training plan.

## Candidate C: Measure the motion that joint positions leave out

**Question.** Can a small budget of surface-motion observations improve prediction of untouched AMASS segment and surface movement beyond a whole-body joint-position history?

Joint centers do not completely determine how every body segment is oriented. Surface motion can carry extra information, but it also contains texture, estimation error, and model-dependent body deformation. The experiment should measure useful predictive value of this extra information, not assume that every residual-flow vector is hidden biomechanics.

**Smallest first experiment.** Select approximately 500 untouched motion prefixes grouped by person and original trial. Predict model-derived segment orientation and pelvis-relative material-point movement at fixed 0.25- and 0.5-second horizons. Compare joint positions alone, positions plus local velocities, positions plus four or eight prefix flow/track tokens, direct raw flow, constant transport, and positions plus true segment rotations as a privileged input ceiling. Use the existing surface renderer's correspondence as a diagnostic, estimated SEA-RAFT as the deployable measurement.

Exact joint-input aliases can establish the representation's limitation in a constructed setting, but they cannot carry the paper. The principal result must occur on unedited held-out motions. AMASS surface trajectories come from its fitted body model, so they are not direct measurements of clothing or skin motion. No claim about force or impairment follows.

**Method and primary comparison.** Keep the video/flow backbone frozen. Fit a compact cross-attention or residual-fusion head and compare direct fusion at the same token and parameter budget. Plot future transport and rotation error against four, eight, and sixteen added tokens. A provisional development target is at least 10 percent relative error reduction over the strongest practical baseline, supported across held people and a held camera/texture setting. Adding full dense flow at uncounted cost is not evidence of a compact state.

**Nearest work.** [MC-JEPA](https://arxiv.org/abs/2307.12698) already joins motion and content learning. [H-MoRe](https://arxiv.org/abs/2504.10676) already learns human-specific optical flow with skeleton-related constraints. [H-Flow](https://arxiv.org/abs/2605.22629) is especially close to surface-motion modeling. The opening is a measured predictive-state omission and a small observation budget that recovers useful information, including a test of whether skeleton consistency suppresses real surface motion.

**Failure condition.** Stop if only renderer truth flow helps, if adding a few ordinary landmarks or direct raw-flow fusion matches the method, or if the effect exists only in constructed twist examples. Do not escalate into photorealistic rendering to rescue absent natural-motion predictive value.

**Evidence strength.** Medium feasibility and potential significance, low current evidence for the natural-motion effect. It is an independent second pilot, not an established successor to motion preservation.

## What should be demoted or revised elsewhere in the seven

**Cross-activity memory needs a much narrower novelty claim.** [Personalized Pose Forecasting](https://arxiv.org/abs/2312.03528) already adapts actual pose forecasts with a small online autoregressive correction, so the distinction between style generation and factual forecasting is insufficient by itself. [PersonaBooth](https://arxiv.org/abs/2503.07390) and [STyMo](https://arxiv.org/abs/2609.04500) further constrain the personalization claim. The remaining question is whether walking support transfers predictive information to a different activity after morphology and immediate-state controls, and beyond autoregressive personalization.

A fresh manifest join found 8,854 eligible files from 189 approved people: 151 train, 19 validation, and 19 test. Filename matches suggest BioMotionLab contains walking, throwing, and lifting for 81 train, 12 validation, and 11 test people. This is a useful availability lead, not validated activity eligibility. The old proposal's minimum of 20 held-out test people exceeds the complete existing test split before activity filtering. Do not silently treat 19 test people as 19 eligible cross-activity participants or lower the old gate retrospectively. A new proposal may declare a different allocation or precision-based requirement before scoring, while preserving reserved motion-preservation evaluation data.

**A generic JEPA robustness or temporal-sensitivity benchmark is now crowded.** [Latent Video Prediction Learns Better World Models](https://arxiv.org/abs/2605.15618) already compares V-JEPA 2.1/2, VideoPrism, and VideoMAEv2 across corruption, contact, occlusion, and temporal direction. [What, Where, and How](https://arxiv.org/abs/2609.01551) already probes camera motion, physics, and anomalies layer by layer. A new gait subset is not enough. The earlier distillation proposals also face [CCH](https://arxiv.org/abs/2510.13182) and [ATLAS](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7346910).

**Latent capture versus physical prediction is an existing contribution.** [The Intervention Gap in Latent World Models](https://arxiv.org/abs/2608.29998) already separates current-query capture, real-effect resolvability, and model propagation, and reports failure of transferable support-aware error scoring. A seventh proposal built on another latent-versus-physical discrepancy should require a new operational improvement, not just rediscover this distinction in AMASS.

**Distributional futures remain a conditional reserve.** Additional prefix evidence can test whether predictive uncertainty shrinks appropriately, but one actual future per prefix does not identify a person's intrinsic behavioral distribution. Model ensembles do not automatically separate observation ambiguity from behavioral uncertainty. Proper scores and matched evidence reveals can support a useful experiment without making that stronger claim.

**Counterexample witnesses are useful only if they change a real decision.** Monocular depth ambiguity is established. Retrieval and geometry can provide valid alternative trajectories, but failed search is not proof of uniqueness, and a pretrained model must beat equally budgeted geometry/retrieval to earn its role. This is a useful supporting instrument for Candidate A rather than automatically a separate high-confidence paper.

## Recommended use of the week

Pilot Candidate A and Candidate C on days one and two. In parallel, perform only the inexpensive block-oracle/candidate check for Candidate B. Select one full study after those results. Do not equate a verified checkpoint with a verified scientific hypothesis, and do not promise seven equally strong paths to an ICLR paper. The portfolio should make it easy to abandon an unsupported premise early.
