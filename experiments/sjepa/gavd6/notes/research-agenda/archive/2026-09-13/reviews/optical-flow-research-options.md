# Optical flow: evidence, counterexamples and the smallest defensible contribution

Independent audit, 12 September 2026. This is a proposal and identifiability analysis. No SMPL alias pairs, optical-flow inference or forecasting experiments were run in this audit. A numerical check of elementary rotation algebra is not empirical validation of the body-model pipeline.

## Recommendation

Replace the old constraint-conflict Proposal 7 with a conditional **predictive-state sufficiency** study. Its scientific object is the information a joint-position history omits, the conditions under which an additional visible measurement recovers useful information, and the smallest tested representation that retains that benefit. The object is not a gait score, a new optical-flow estimator, or generic skeleton-plus-video fusion.

Use optical flow in Proposal 1 as additional observation evidence too, but keep the questions distinct. Proposal 1 asks whether a restoration prior deletes a real movement that observations support. Proposal 7 asks whether the chosen input representation discarded a relevant movement before prediction even began.

## The run motivates an audit, not this explanation

The latest saved run has real-skeleton incremental target R-squared of 0.000357099, while mismatched skeleton produces 0.000602912. RGB-only prediction improves from 0.451045 to 0.662830 as source data increase. These figures establish that a strong teacher-target score did not establish useful skeleton-specific information in this setup. They do not establish that axial rotation, surface motion or optical flow explains the failure. The existing target also observes future RGB while the skeleton input ends earlier. See [evidence audit](evidence-and-novelty.md), [run analysis](../../../../../notebook_runs/future-innovation/haic-run-02/ANALYSIS.md), and [research strategy](../../../../../docs/studies/future-feature-prediction/scaling/research-strategy.md).

## A valid finite alias construction

Consider a kinematic joint j with exactly one child c. Let b be the rest offset from j to c in j's local frame. Select a rotation Q around b, so Qb = b. Replace the local rotations by:

```text
R_j' = R_j Q
R_c' = Q^-1 R_c
```

If A is the unchanged global rotation of j's parent, the child offset remains A R_j Q b = A R_j b. The child's global rotation remains A R_j Q Q^-1 R_c = A R_j R_c. Therefore every descendant joint position and orientation remains unchanged. Surface vertices influenced by j can nevertheless move. This construction does not extend to an arbitrary joint with several noncollinear child offsets.

This proves a property of the declared kinematic map. It does not prove equality of surface-regressed joints, confidence scores or the actual S-JEPA tensor. Pose-dependent shape changes can shift the chosen landmarks. Accept a pair only after applying the exact deployed extraction, resampling, centering and normalization pipeline and measuring every input channel. If that check fails, there is no exact alias for that representation. A first-order Jacobian null direction does not certify a finite alias.

For a forecasting fixture, choose smooth Q-positive(t) and Q-negative(t) histories with Q-positive(0) = Q-negative(0) = identity at the observation cutoff, but opposite recent angular motion. Both final body configurations can then coincide even though their incoming surface histories differ. Set future trajectories explicitly and evaluate the future material points they produce. Keep every render parameter identical between pair members. Restrict angles and velocities to declared admissible ranges and report their range; kinematic validity is not proof of human biomechanical plausibility.

The elementary balanced-pair squared-loss lower bound is ||y-positive - y-negative|| squared / 4 when both inputs are identical and both outcomes occur equally often. This follows by minimizing the average distance to two targets. It is a useful check, not a novel theorem. Constant-speed examples may be solved perfectly by flow extrapolation and belong only in the validation fixture.

Root's [algebra script](../checks/verify_joint_preserving_rotation.py) and [receipt](../checks/joint-preserving-rotation-check.json) are independently inspectable chain checks. They do not exercise SMPL, actual tensor conversion, rendering or estimation. Exact-input tolerance must be close to numerical error and compared with the outcome separation; setting a loose tolerance after seeing results invites a false impossibility claim.

## What would make the result significant

The pair construction establishes an information loss. A natural future experiment establishes whether that loss matters. Those are different claims.

Use unedited held-out AMASS trials, fixed prefix visibility and preregistered future horizons. Predict segment-relative orientation and body-relative material-point trajectories, with absolute movement error reported separately. A broad global displacement endpoint could be solved by centroid velocity and conceal the claimed mechanism. Score against the known rendered body state, never the same estimated flow used as input. AMASS surface motion is generated from fitted body parameters, not directly measured human skin motion. The paper must say so.

The proposed method freezes a public flow estimator and S-JEPA, then retains local raw and skeleton-unexplained transport in K = 0, 4, 8 or 16 tokens. Keep token dimensions, time span and normalization fixed. Report floats or bytes and downstream operations as well as token count. A sweep across four encodings estimates a useful budget among those choices; it cannot establish an information-theoretic minimum.

A credible positive result needs useful held-out future prediction, a mechanism-specific failure of a joint-guided prior, and efficient recovery that survives at least one new movement family and observation condition. If simple concatenation works equally well, the finding may still be a useful representation audit, but the proposed new method has not earned its claim.

## Novelty collisions to face directly

[Pose from Flow and Flow from Pose](https://openaccess.thecvf.com/content_cvpr_2013/html/Fragkiadaki_Pose_from_Flow_2013_CVPR_paper.html), [flow-based body estimation](https://arxiv.org/abs/1703.00177), [MC-JEPA](https://arxiv.org/abs/2307.12698), and [H-MoRe](https://arxiv.org/html/2504.10676v1) already cover important pieces of pose-flow reasoning, articulated fitting, shared motion/content learning and human-centric flow. [GaitMDF](https://www.sciencedirect.com/science/article/pii/S003132032600110X) additionally makes motion-field distillation for gait an occupied direction. Its primary abstract was inspected; its full method was not.

H-MoRe's published magnitude penalty is nonzero for nonzero surface flow when its reference joint displacement is zero. Whether the original implementation actually suppresses that motion depends on stationary-joint handling and competing losses. The official repository link returned 404 during review. A controlled implementation of the stated penalty must be labeled a reproduction, not an observed failure of released H-MoRe weights.

[H-Flow](https://arxiv.org/html/2605.22629v1) is another direct neighbor. It couples dense surface motion to endpoint-interpolated skeletal motion with a tolerance margin and introduces simulated material correspondence in DynAct4D. Thus surface dynamics beyond sparse joints and their evaluation are established motivations. A verified case outside that prior's tolerance is a testable limitation, not evidence that the complete method fails in ordinary motion. Its optional dataset and checkpoint require separate release verification.

## Strong comparisons and disconfirming controls

- Same-prefix full RGB mesh recovery followed by prediction. The keypoint representation may lose a variable that an existing mesh method already estimates well.
- Dense raw-flow forecasting, simple skeleton-flow concatenation and local affine flow fitting, each with matched training capacity and tuning effort.
- Last state, constant pixel velocity, angular-velocity extrapolation and a Kalman-style tracker. These expose easy manufactured dynamics.
- Additional named or off-axis landmarks under the same observation budget. More informative landmarks may solve the problem more directly than a learned residual representation.
- Full fitted rotations and exact past renderer transport as information ceilings, clearly separate from deployable observation baselines.
- Static final image, duration, root drift, foreground area, cadence and flow-energy summaries. Predict an actual multidimensional future, then ask whether these summaries account for the gain.
- Static body with moving camera, moving texture on a static body, textureless surfaces, repeated textures and occlusion. Camera parallax is not eliminated exactly by one background homography.
- Unseen texture and camera with the same movement distribution, then a genuinely unseen movement family. Repeated renders of one motion are not independent participants.

All estimated input flow must use frames at or before the cutoff. An offline tracker fed the whole clip leaks the future even if only its early outputs are retained. Select query points by prefix information only. Visibility masks for endpoint scoring must be fixed across methods, with coverage reported. Give any abstention threshold its own calibration split; the main paired error comparison should include every eligible query.

## Portfolio judgment

Proposal 1 remains the best primary investment because its real-event-versus-tracker-artifact distinction is practically meaningful and its failure can be demonstrated directly. New Proposal 7 is the best optical-flow-specific pilot and could move into the top two if estimated flow survives the natural-motion and full-RGB baseline gates. Before those gates, place it behind Proposal 1 and approximately alongside Proposal 2, above the crowded transfer-response and motor-style directions. A useful visual example is likely; a significant new forecasting result is still uncertain. Do not convert this ordinal judgment into a numerical ICLR acceptance probability.
