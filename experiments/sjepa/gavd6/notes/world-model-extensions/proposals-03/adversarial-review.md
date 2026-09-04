# Adversarial review and revision log

Three independent reviews covered prior reports, local experiments, and 2024 to 2026 literature. This log records the strongest objection to each surviving idea and the concrete revision made before the final version.

## 1. Cross-Protocol Perturbation Response Prediction

**Objection:** A first draft called this a phase-conditioned balance envelope. That language overstated the data. Georgia Tech varies phase, but Dryad always targets 32.5% of the gait cycle. Generic perturbed-motion prediction also collides with Latent Differentiable Physics, and the repository already proposed a synthetic Intervention Response Fingerprint.

**Revision:** The final study predicts four common, dimensionless recovery outcomes from real interventions. Phase generalization is tested only on Georgia Tech. Dryad tests apparatus transfer. The primary baseline is the phase, direction, magnitude, and protocol conditional mean. The model gets credit only for person-and-state-specific residual recovery.

## 2. Future-Innovation Distillation

**Objection:** “Distill V-JEPA into S-JEPA” is ordinary cross-modal distillation. It also overlaps the repository's Past-Only Predictive Surplus and Cycle-to-Cycle Innovation Map. A high score could come from copying current appearance, background, crop, or source.

**Revision:** The final target is the future V-JEPA residual after current RGB and a strong nuisance model have predicted everything they can. The main result is a conditional skeleton-explainable fraction across horizons. Time-shuffled and person-mismatched skeletons are mandatory. A deployable student is trained only if the measurement is nonzero.

## 3. Predictive Pose-Tracker Auditor

**Objection:** Motion smoothing, pose refinement, physical plausibility, dual-tracker disagreement, and robust skeleton classification already exist. GAVD has no ground-truth pose, so a GAVD-only success could not establish detector accuracy.

**Revision:** The final study detects and localizes error rather than correcting it or classifying gait. It trains the cross-modal bridge on clean rendered AMASS, evaluates six known error families on held identities, and must beat current-state disagreement as well as confidence, jerk, flow, and dual trackers. GAVD is a blinded transfer review only after controlled AUROC passes.

## 4. Pre-Impact Recoverability Horizon

**Objection:** Fall detection can be solved by body height, vertical speed, box shape, or impact frames. SAFER falls are simulated, so “fall-risk prediction” would be incorrect.

**Revision:** The final benchmark cuts every context before annotated onset, matches falls to safe look-alikes on obvious geometry, and defines a pre-obvious subset where trivial kinematics stay below AUROC 0.65. The result is the earliest reliable scripted-event horizon under person-held conditional log loss. Wheelchair and non-lab domains are never pooled into a flattering average.

## 5. Sparse Anchor Budget for Temporary Correspondence

**Objection:** A global anatomical-side anchor is tautological and directly collides with the local laterality work. The existing temporary-swap benchmark is also trivial because continuity reaches the oracle.

**Revision:** The final study uses smooth, continuity-matched temporary swaps, occlusion bridges, and a held-out corruption generator. It measures error across sparse anchor budgets, placement policies, and label-noise rates. A global bit is explicitly excluded. The output is the minimum independent evidence budget, not recovered side from motion alone.

## 6. Nonlocal Predictive Surplus for Foot Placement

**Objection:** Upper-body contribution to foot placement is known biomechanics. A whole-body model may win only because it has more parameters or recovers activity, speed, phase, and heading. This also overlaps the local Conditional Coordination Graph.

**Revision:** The lower-body comparator receives all major state variables and is widened to equal capacity. Upper-body streams are shuffled only within narrow speed, phase, heading, and action strata. The proposal is ranked as a secondary mechanistic audit. It uses “predictive surplus,” never “compensation,” unless a known perturbation changes the surplus.

## 7. Transition Intent Frontier

**Objection:** GaitIntent already benchmarks motion-intent classification from 45-frame sound-limb windows. A new classifier would be incremental. Ground-contact state and post-transition samples could make the task trivial. Only one participant has an amputation.

**Revision:** The final object is a calibrated time-by-body frontier. One prefix-trained model finds the earliest and smallest input that reaches a fixed error and coverage target. Contact state and post-transition samples are removed from the primary input. The amputee participant is one final transfer case, never population evidence.

## Ideas removed rather than renamed

- Automated gait description generation was removed because recent gait-language systems already occupy the claim and GaitMoText is too small for the proposed two-week headline.
- Prospective fall-risk screening on 100 older adults was removed because the outcome is a later self-reported screen, participant-linked public labels still needed verification, and high-dimensional inference would be fragile.
- Cross-topology predictive transport was removed because recent arbitrary-topology and heterogeneous-skeleton models are direct collisions.
- Generic action anticipation was removed because V-JEPA 2, Human-JEPA, FutureHuman3D, and recent skeleton anticipation already occupy it.
- Generic gait counterfactual editing was removed because GaitDynamics and recent gait generation work already provide the core capability.
