# Independent adversarial review of PLAN.md

Reviewed 28 September 2026 against paper v08, its appendix and technical supplement, the initial independent review, and selected primary sources. This review covers the **plan and figure specifications**, not yet the rebuilt assets. Section references identify the reviewed 27 September draft. No paper or PLAN.md edits were made.

**Verdict:** No remaining P0 overclaim in the central narrative. The initial objections about JEPA benefit, clinical inference, 2D ambiguity, unequal training influence, and simulator/tool circularity have been incorporated. The main numerical anchors agree with the paper. Before finalizing, resolve the concrete research-design gaps below; asset validation remains open.

## P1.1 — Specify the paired experiment that makes the proposed next paper identifiable

**Where:** Section 8, “Recommended first new paper”; Section 7.5; opening priority statement.

The plan selects preservation of side-specific change under occlusion, but does not say what the two motions being compared will be in the new natural-video/3D study. The existing synthetic paired edit supplies a known contrast. Two unrelated natural clips do not supply the same controlled experiment. The proposed primary also changes from projected excursion to 3D knee trajectories without fixing anatomical angle conventions.

**Revision:** Add a minimal next-study protocol with two distinct contrasts:

- **Observation contrast:** digitally occlude an otherwise identical target-camera recording, with independent synchronized 3D references available for the whole motion. Compare clear versus corrupted estimates of the same underlying movement. This isolates observation robustness without claiming a movement intervention.
- **Movement contrast:** use separately recorded, prespecified within-person motion conditions or sessions, with the actual per-leg 3D change computed from independent references. Instructions to move differently are not the target value. Do not interpret the difference causally without an appropriate design.

Fix an anatomical knee-angle definition, side labels, trial/window pairing, reference support, one primary preservation contrast, and a hierarchy for trajectory improvement versus noninferiority of side-specific change. Show original and restored per-leg errors as well as a scalar difference. The 2D confirmation study and the new 3D measurement study can be separate milestones; make clear which constitutes the first executable experiment. This would turn a compelling theme into a falsifiable protocol.

## P1.2 — Apply causal access restrictions to the entire future-prediction pipeline

**Where:** Section 7.1, especially “No future frames may enter the context encoder”; Sections 7.5–7.6.

Forbidding future frames at the encoder does not prevent future leakage through an offline pose tool, temporal smoothing, camera/scale fitting, SLAM, normalization, or tool-produced context. WHAM is explicitly a temporal reconstruction system combining image, keypoint and camera-motion information; its use must be audited for the intended prefix protocol. [WHAM paper](https://openaccess.thecvf.com/content/CVPR2024/html/Shin_WHAM_Reconstructing_World-grounded_Humans_with_Accurate_3D_Motion_CVPR_2024_paper.html)

**Revision:** Require every input-producing operation to run only on the observed prefix, with tools rerun on prefixes when necessary. Add a mechanical truncation check: changing held-out future frames must not change a model's inputs or forecast. Report observation duration, forecast horizon and algorithmic latency. Keep full-clip restoration results separate from causal forecasting results. This is a design requirement, not evidence that the current study leaked; current restoration legitimately uses its full window.

## P1.3 — Protect new evaluation data from the same adaptive reuse being criticized

**Where:** Section 8 stage gates; Section 7.6 router rewards and benchmarking.

The plan repeatedly proposes untouched people/scenes, but five successive stages may tempt reuse of one newly collected test cohort. Tool selection, reward weights, abstention thresholds and noninferiority margins can all overfit that cohort without using it in a gradient update.

**Revision:** Explicitly separate model fitting, policy/reward and threshold selection, and final evaluation by person and scene/session where applicable. Select practical margins from external requirements or a separate pilot; lock them before testing. Either reserve a final cohort until the complete selected pipeline is fixed or use fresh confirmation data for each stage whose results influence later choices. Report tool versions and licensing/artifact availability as feasibility dependencies, not merely a reproducibility footnote.

## P1.4 — A fixed-chair generator does not test generated object motion

**Where:** Section 7.7 and Section 8 stage 4.

A known stationary chair is a sensible first scene-conditioned body-motion experiment. However, returning its unchanged pose can win the object-motion component trivially. This does not yet test coupled generation of human and moving-object trajectories, which is one requested research extension.

**Revision:** Label the fixed-chair study as the first contact-conditioned milestone. Add a subsequent bounded moving-object case, such as a human moving a rigid object along a known path, with object motion and contact inferred/generated jointly. Include object-persistence and independently generated human/object baselines, plus held-out object geometry. CHOIS generates synchronized object/human motion conditioned on object waypoints; InterDiff predicts future interactions with dynamic objects. The next contribution must exceed static conditioning and contact plausibility alone. [CHOIS](https://arxiv.org/abs/2312.03913), [InterDiff](https://arxiv.org/abs/2308.16905)

## P2.1 — Close two provenance and exposition gaps

**Where:** Sections 4.3–4.5 and 6.

First, the teacher is correctly restricted to training, but an accessible workflow can still imply both paired motions are fed together during deployment. State explicitly that each development window is restored independently; pairing is used in training auxiliaries and in response scoring. Add this to Figure 1's specification.

Second, Section 6 calls the regeneration “not a new statistical analysis.” That is correct only if every interval is copied from an existing numerical export. If plot code computes a fresh interval, aggregation or weighting, classify and document it accordingly. Match the plotted population, point estimates, limits and contrast signs to the exported values. Retain the two different interval procedures and the 2:1 versus 4:1 weighting. Figure 3 must visibly separate its response and waveform outcomes; one shared “JEPA benefit” axis title would be misleading.

## Verification record and remaining acceptance checks

The headline means, all three primary effect intervals, failure percentages/components, readout repair numbers, and global-swap naming rates agree with v08. The supplement also confirms 128 samples span 5.08 seconds, zero edits are excluded from nonzero response scoring, and the readout loss packages are correctly described. V-JEPA 2 supports the narrow action-conditioned-planning example; CHOIS/InterDiff support the stated prior-work boundary. OpenCap Monocular is correctly identified as a March 2026 preprint and a direct novelty constraint, not a proven baseline result for this task. [V-JEPA 2](https://arxiv.org/abs/2506.09985), [OpenCap Monocular](https://arxiv.org/abs/2603.24733)

Before marking figures complete, inspect final-size PDFs/SVGs for clipped text, readable uncertainty and grayscale legibility, and verify source hashes and numerical provenance. The current plan's prepared date can remain 27 September, with “revised 28 September” added after disposition. This is a review of a plan; neither favorable review nor polished figures substitutes for independent experimental confirmation.
