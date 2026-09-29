# Independent adversarial review: proposed ambient JEPA writeup

Reviewed independently, before seeing the proposed plan. Evidence: `docs/iclr/versions/v08/paper-v08.tex`, its appendix, and the supplied Summer 2025 report. The objections below concern the proposed narrative and research roadmap; no new experiments were run. Priority P0 blocks a defensible writeup; P1 blocks a convincing research extension.

## P0 — Make the contribution an evaluation lesson, not a success story about JEPA

The paper does not demonstrate better gait representations, physical understanding, future prediction, fall-risk prediction, or clinical benefit. It restores complete observed **2D** motion windows with privileged clean projected training references. The encoder sees the full window; this is not causal forecasting. The adaptation uses DINO-style cross-entropy, VICReg, and coordinate-supervised readouts, and differs from published S-JEPA, including which encoder is deployed.

The correct central question is: **When a model repairs noisy movement observations, does it preserve the movement differences we intend to measure?** That question connects credibly to ambient monitoring: smoother poses could conceal a change or assign it to the wrong leg. A useful contribution is showing why coordinate accuracy, response accuracy, and anatomical assignment require separate checks.

Required numerical constraints (main Results and Appendix C): zero response is 5.811°, lower than all 16 fitted variants under pooled scoring; direct coordinate fitting improves unchanged observations from 12.69° to 7.54° response error and 18.57° to 12.07° waveform error. All three recorded primary benefit intervals cross zero. Delta versus endpoint gives 0.37° [−1.11, 1.76]°; this is uncertainty, not an improvement established by evidence. The report should put these findings before ambitious future applications.

## P0 — Break the clinical inference chain explicitly

The previous report proposes mapping normal/abnormal/pathological labels to fall-risk categories through a heuristic. Do not carry that inference into the new report. Neither a gait class, asymmetry score, nor “more symmetric” motion establishes individual fall risk. The current 112 training and 14 development people come from synthetic preparation of AMASS motions; they are not the previous Toronto study participants. No GAVD, natural-video, prospective-fall, or locked-person confirmation evaluation was completed (Discussion; Appendix A).

Choose one concrete **future** use case: preserving longitudinal, side-specific movement measurements during camera-based mobility monitoring. State that clinical fall-risk utility would later need a separately designed outcome study, patient-relevant reference measurements, and calibration beyond the present benchmark. Do not imply synthetic right-knee edits simulate disease or treatment.

## P0 — Explain what the measurement cannot identify

The response is the change in right-minus-left image-plane knee excursion, where excursion is P95 minus P5. It loses temporal ordering and cannot alone identify which leg changed. Matching excursion can coexist with incorrect trajectories or swapped anatomy. A nominal 3D knee edit is not the projected 2D response; a physical mirror need not negate that response (Methods, Equation 1).

Use an explicitly illustrative teaching diagram showing these ambiguities, not purported reconstructions: complete reconstructed poses are unavailable. Keep 3D anatomical angles, projected knee angles, and clinical measures visually and verbally distinct. Showing a handsome SMPL body beside 2D results without this distinction would invite a false impression of 3D validation.

## P1 — Acknowledge confounding before explaining mechanisms

Endpoint's auxiliary gradient starts at 10% of the base RMS magnitude; delta's starts at approximately 0.0013%, despite a shared coefficient. Six feature fits clip 11,999 of 12,000 updates. This weakens any attribution to “learning differences.” Direct fitting updates its encoder for 4,000 steps; the other procedures pretrain 2,000 steps and fit a frozen-encoder readout for 2,000 steps. Equal total steps do not equalize supervision or encoder adaptation (Methods; Appendix B).

All follow-ups reuse 14 development people and three seeds. Increased waveform errors after adding two penalties cannot identify which penalty caused harm. Lower scalar weighting explains 93% of delta's mean original-to-dense improvement; dense versus low scalar remains uncertain. These are diagnostic findings, not proof of a percentile-gradient mechanism.

The response gap is not merely a large-failure-penalty artifact: endpoint/delta successful-error contributions, 6.306°/6.210°, already exceed zero response. Conversely, approximately 74% of their small mutual score difference comes from failure-cost contributions. Show both facts together. Naming failure combines wrong, ambiguous, and missing outputs; 50% is not a chance benchmark (Results).

## P1 — Turn the seven extensions into separable hypotheses

1. **“Understanding physics.”** Require out-of-distribution intervention prediction, not just natural-looking outputs. Start with controlled changes to support, friction, object position, or contact, holding appearance comparable. Compare an unconstrained predictor, a constraint-only method, and the proposed learned model. Test multi-step kinematic error, contact error, and dynamic residuals jointly; a motion prior may look realistic while predicting the wrong response. This requires future/action-conditioned training absent from the present work.

2. **Physical groundedness.** Define a scorecard, not one unvalidated scalar: anatomical identity and geometry; temporal/contact consistency; environment compatibility; and dynamic consistency conditional on known physical assumptions. Joint limits and smoothness are necessary checks, not proof of dynamically feasible movement. A real pathological motion can be physically valid, while a “normalizing” model erases the signal. Build paired hard negatives with only one violated property and evaluate sensitivity, specificity, calibration, and abstention.

3. **Discriminative and generative S-JEPA.** These are different tasks. First test a frozen representation against coordinate, randomly initialized, and established motion baselines on prespecified measurements. Generation needs a separately trained stochastic decoder or conditional generative model; JEPA feature prediction alone is not a motion generator. “SMPL keypoints” should be defined as joints derived from a body model, distinct from SMPL pose/shape parameters. Evaluate edit fidelity, diversity, contact, identity, and measurement preservation separately.

4. **OpenSim or robot simulation in the loss.** Start with an offline diagnostic or teacher, not an end-to-end promise. OpenSim inverse dynamics requires inertial parameters and measured or modeled external forces; an ill-fitting body model can manufacture residuals. Robot feasibility does not establish human biomechanics. Compare no-physics, geometric-only, and physics-informed variants, plus an oracle-3D condition to isolate perception error. Audit whether “correction” removes true asymmetry. Differentiating through a solver, fitting a surrogate, and using black-box rewards have different stability and bias risks. [OpenSim documentation](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53090063)

5. **Accurate depth on the existing embedding.** The normalized 2D representation cannot uniquely determine metric depth or scale; the input also lacks appearance and objects. Add image/camera/scene evidence and explicit calibration assumptions, then test frozen versus adapted features and tool-only baselines against synchronized 3D references. Evaluate absolute trajectory and contact error, not only alignment-invariant pose error. WHAM itself combines keypoints, image features, SLAM camera motion, and contact-aware refinement; calling it does not certify recovered motion. [WHAM](https://openaccess.thecvf.com/content/CVPR2024/html/Shin_WHAM_Reconstructing_World-grounded_Humans_with_Accurate_3D_Motion_CVPR_2024_paper.html)

6. **Qwen tool agent with RL.** Defer until fixed pipelines work. Define a tool-selection problem with a budget, observable uncertainty, and independently measured reward. Compare fixed all-tools, a hand-coded uncertainty policy, a supervised router, and RL. Charge latency/tool cost; test tool failures, unsupported claims, and abstention. SAM masks are not contact labels; depth and WHAM estimates are not ground truth. An agent rewarded by the same tools it consults can optimize agreement while becoming less accurate.

7. **Joint human/object generation.** Keypoints alone omit object extent, articulation, mass, friction, and contact surfaces. Begin with one bounded interaction, such as sit-to-stand with fixed chair geometry, before arbitrary scenes. Use object pose/geometry, an explicit shared coordinate frame, contact state, and physical parameters when dynamics are claimed. Hold out chairs, people, and camera conditions. Measure penetration, sliding, support, completion, and preservation of atypical movement as well as visual quality.

## Required sequencing and figure safeguards

First repair experimental validity: preserve source predictions/checkpoints, match supervision and auxiliary influence, add established refinement baselines, prerecord comparisons, and evaluate genuinely untouched people. Then validate metric 3D and contacts against independent measurements; only afterward test physics-guided prediction, generation, and adaptive tool use. OpenCap's laboratory comparison illustrates an appropriate validation pattern, not a transferable accuracy guarantee. [OpenCap study](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462)

Novelty also needs updating: the March 2026 **OpenCap Monocular preprint** already combines WHAM refinement, biomechanical constraints, and physics-based kinetics with laboratory validation. Merely joining these tools is insufficient novelty. A more defensible prospective contribution is preserving side-specific atypical motion with calibrated selective tool use under independent person/scene evaluation. [OpenCap Monocular](https://arxiv.org/abs/2603.24733)

Rebuild the paper's workflow, restoration, reliability, readout, and naming figures from source numbers. Show uncertainty and units, distinguish exploratory contrasts, and preserve different denominators. Label future architecture drawings **proposed** and synthetic teaching poses **illustrative**. Retain the previous report's accessible progression and candid account of a failed hypothesis; replace its tool inventory and speculative clinical shortcut with an evidence-led research narrative.
