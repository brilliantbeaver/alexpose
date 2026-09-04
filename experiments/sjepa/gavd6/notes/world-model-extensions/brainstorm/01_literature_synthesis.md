# Literature synthesis

## Decision boundary

A world model predicts how a system changes. The useful research object here is not another gait classifier. It is a measurement derived from prediction: error, disagreement, a counterfactual change, an abstention signal, or a recovered latent state. A latent state is an internal vector that compresses motion. The literature leaves room for such measurements only when they survive camera, detector, speed, duration, and source-video controls.

## Adaptation pattern

GoalForce and Masked Visual Actions obtain large behavioral changes without training a foundation model. Both keep a Wan video backbone fixed and add a small trainable route for a new control language. GoalForce adds a ControlNet branch to one expert. Masked Visual Actions adds separate low-rank adaptations to two experts. Their shared pattern has four parts:

1. Start from a public checkpoint that already models broad motion.
2. Express the new question as a structured condition or query.
3. Train only an adapter, control branch, or small head.
4. Evaluate the new behavior against a matched frozen-base control.

The gait analogue should preserve a pretrained normal-motion prior while learning how to query it. Good queries include masked joints, desired corrections, contact hypotheses, view alternatives, and assistive-device tracks. A zero-initialized branch is especially useful because the adapted model equals the frozen base before training.

## What is already occupied

Plain generative surprise is occupied. Zero-shot Gait Classification with Diffusion Models already measures error from a public text-to-motion diffusion model and correlates it with Parkinson gait scores. A proposal must change the object being measured, such as a minimum normalizing edit, a response to a controlled intervention, cross-model disagreement, or calibrated uncertainty.

Generic clinical representation transfer is also occupied. GaitForeMer, GaitEncoder, CARE-PD, and Aggregate, Don't Adapt already cover motion forecasting pretraining, compact clinical gait embeddings, multi-site frozen probes, and test-time subject aggregation. DiffuseGaitNet and GAITGen cover impaired-gait synthesis and augmentation. A new result cannot be only a better frozen probe, a severity-conditioned generator, or a label-efficiency curve.

Generic physical plausibility scoring is occupied at the method level. HumanScore uses deterministic anatomy and smoothness tests. PhyMotion maps recovered motion into MuJoCo. PP-Motion learns from simulator correction. The open question is whether physical feasibility and healthy typicality are different axes for an observed gait, and whether their disagreement is informative.

## Scientific openings

### Predictive error geometry

S-JEPA predicts hidden latent vectors rather than coordinates. Its joint-by-time errors can form an error map, but the current repository shows that a representation can fail even when its architecture sounds appropriate. A credible experiment must compare trained, random, and raw-motion controls, vary the prediction horizon, and hold out source videos. The important result is not high error on pathology. It is a stable anatomical or temporal error pattern that cannot be reproduced by pose quality or walking speed.

### Counterfactual repair

GaitDynamics and GaitEncoder provide public priors in metric gait coordinates. They make it possible to ask for the smallest change that moves an observed gait toward a normal-motion distribution. This differs from a scalar anomaly score because the output is an explicit proposed change. It can still fail by repairing camera or lifting artifacts, so the edit must be stable across views and lift hypotheses.

### Physics versus typicality

GaitDynamics, PhyMotion, HumanScore, and PP-Motion expose different notions of valid motion. A gait can be mechanically possible but clinically atypical, or visually typical but mechanically inconsistent after a bad lift. The disagreement between a normal-motion prior and a simulator-grounded judge is therefore a potential measurement and a built-in failure detector.

### Forward and inverse consistency

Inverse dynamics recovers the action or cause that could produce an observed transition. SC3-Eval shows that one adapted world model can combine forward prediction, inverse recovery, and view consistency. Its training scale is not reproducible here. The query pattern is still testable with small skeleton models: predict the future from the present, recover the preceding joint change from that future, and abstain when the two disagree.

### Richer observation bridges

VideoMDM provides the strongest public 2D-supervised route into 3D motion. PoseAnything provides a public pose-conditioned video checkpoint. ProGait exposes a failure mode that body-only models miss: the prosthesis is part of the motion system. Depth, segmentation, pose, and device tracks should be treated as uncertain modalities, not accepted as ground truth.

## AMASS to GAVD bridge contract

Every candidate must cross the same four-stage boundary:

1. Build the normal prior from AMASS metric 3D motion or from a public checkpoint trained on compatible healthy motion.
2. Project AMASS through camera, occlusion, scale, and pose-noise corruptions that resemble GAVD.
3. Lift GAVD into several plausible 3D trajectories or meet AMASS in a view-robust latent space.
4. Evaluate by source-video-held-out groups, because GAVD provides no participant identifier.

The bridge must preserve uncertainty. A single monocular lift can turn depth error into apparent pathology. Results should therefore include lift disagreement, view strata, pose-confidence strata, and a raw 2D baseline.

## Public-checkpoint rule

An idea is admissible only if the first experiment starts from a checkpoint that was directly downloadable on the audit date, or from this repository's existing checkpoint. A paper promise, request-only model, broken link, or unreleased processed corpus does not qualify. Models with usable release surfaces include GaitDynamics, GaitEncoder, MDM, VideoMDM, PoseAnything, PP-Motion, and several general pose, depth, segmentation, and vision-language components recorded in the checkpoint inventory. S-JEPA has no official public checkpoint, so its local weights are the only immediate S-JEPA starting point.

## Selection implications

A strong candidate should answer both a positive and a negative scientific question. It should return more than a class label. It should name the exact checkpoint, adaptation, bridge, null lesson, shortcut control, and 72-hour test. It should not depend on GAVD severity or laterality labels that do not exist. It should treat `gait_pat` as an observational taxonomy and source-video grouping as a leakage surrogate, not as a participant split.

## Open questions

- All 348 GAVD source videos are operator-confirmed as decodable on HAIC and cover all 1,874 sequences. A complete full-video pose manifest still needs to be created.
- The BABEL annotations are downloaded. They still need extraction and path matching before they can provide defensible walking and running durations.
- Can repeated people across different GAVD source videos be detected without creating an unreliable identity claim?
- Which public monocular lifting route is stable enough on prostheses, walkers, canes, and severe occlusion?
- Can a normal-motion edit be distinguished from an edit that merely makes a lift look more like its training set?
