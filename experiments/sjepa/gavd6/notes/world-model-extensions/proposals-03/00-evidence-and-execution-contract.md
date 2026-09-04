# Evidence and execution contract

This document separates what has been executed from what is proposed. It also fixes the evaluation rules before any new result is seen.

## 1. What the local work actually establishes

### Foundation notebooks 00 to 03

The repository contains a working masked-latent S-JEPA implementation. A 64-frame sequence is divided into 16 four-frame blocks. The historical model uses all 33 MediaPipe landmarks as context, but only 12 shoulder and lower-body landmarks can be masked. The target encoder receives the full sequence, the student sees visible tokens, and the predictor estimates target latents. An executed mask audit reached 59.5% of valid eligible tokens, which is 21.6% of the full token grid. No forbidden target was sampled.

This proves that the training mechanics work. It does not prove forward prediction, clinical validity, or useful transfer.

### Foundation notebooks 01 and 02

The historical GAVD cohort contains 96 sequences from 18 source videos: 12 normal, 9 Parkinsonian, 12 stroke, 47 myopathic, and 16 cerebral-palsy presentations. All 12 normal examples come from one video. Parkinsonian and cerebral-palsy examples each come from only two videos. MediaPipe extraction retained original timelines and reached high mean coverage, but one stroke sequence had only about 57% coverage.

This establishes an end-to-end pose pipeline. It also establishes source and missingness confounds that must be controlled.

### Foundation notebooks 04 to 06

Stage 0 trained on 75 normal sequences. Four later curriculum stages introduced the other presentation labels and added label-aware compactness and separation terms. The final model therefore used presentation labels after Stage 0. The same 96 canonical sequences later appeared in latent inspection and classifier probes.

The exposed 96-sequence Random Forest reached macro-F1 0.754. A missingness-only model reached 0.388. All test videos in that split also appeared in training, and the encoder had seen every evaluated sequence. These numbers are descriptive, not evidence of unseen-source clinical generalization.

The latent geometry itself was weak: cosine silhouette 0.028 and minimum class-centroid distance 0.027. A normal-reference distance is not a clinical severity score.

### Latest AMASS outputs

The latest decisive laterality benchmark has 1,146 validation windows from 15 held-out identities. The sealed test remains unopened and only seed 7 exists. Identity-macro normalized mean absolute errors were:

| Representation | Odd orbit | Even channel |
| --- | ---: | ---: |
| Raw, uncorrected | 1.6310 | 1.0820 |
| Raw, continuity correction | 1.5727 | 1.0221 |
| Raw, oracle correction | 1.4851 | 1.0097 |
| Correction-first learned representation | 0.9300 | 0.7472 |
| SG-JEPA | 0.8671 | 0.6696 |
| Fixed 50/50 uncertainty control | **0.8668** | **0.6690** |

Representation learning helps inside this protocol, but the structured gauge mechanism does not beat fixed uncertainty. This is why Proposal 5 studies how much independent correspondence information is required. It does not present the current gauge mechanism as a success.

The synthetic swap probe is even more cautionary. A simple continuity rule reached perfect path recovery and matched the oracle. Any future correspondence experiment must use smooth, continuity-matched temporary swaps that a local jump detector cannot solve.

## 2. What the attached reports add

The internship report supports a broad pose-analysis platform and records qualitative preference for OpenPose on a 14-video Toronto corpus. Its Gemini fall-risk experiment did not match literature ground truth and produced hallucinated reasoning. The proposed Qwen fine-tuning and heuristic risk tiers were plans, not validated results. No proposal here treats them as evidence.

The S-JEPA research writeup documents the laterality benchmark and its key negative result: the learned structured posterior was marginally worse than uniform uncertainty. It also reports strict GAVD and StrokePIG transfer failures. These negative controls directly shaped Proposals 2, 5, and 6.

## 3. Dataset roles

| Dataset | Role | What it can support | What it cannot support |
| --- | --- | --- | --- |
| GAVD | In-the-wild RGB and 2D-pose stress test | Unseen-source future-latent and tracker-audit transfer | Diagnosis, affected side, severity, participant-held generalization |
| AMASS | Controlled full-body motion and rendering | Held-identity dynamics, known corruptions, full-body ablations | Clinical outcome claims |
| Georgia Tech perturbations | Known disturbance with varied direction, magnitude, and gait phase | Held-intervention and held-person recovery prediction | Full-body arm/trunk analysis in the present public release |
| Stanford Dryad perturbations | Independent treadmill perturbation protocol with OpenSim kinematics | Cross-protocol and fixed-phase transfer | Independent phase generalization |
| SAFER-Activities | Long video, aligned pose, frozen RGB features, frame-level activities | Pre-impact fall-versus-look-alike prediction and OOD tests | Real accidental-fall risk |
| GaitIntent | Public 96 Hz full-body IMU and sound-limb transition windows | Early terrain-transition intent and amputee transfer | Broad clinical effectiveness |

## 4. Fixed split rules

1. Split people first whenever participant identity exists.
2. Split GAVD by source video because participant IDs are absent.
3. Hold out intervention combinations in Proposal 1, not only random trials.
4. Compute normalizers, phase templates, corruption thresholds, and calibration maps inside training folds.
5. Never use one person's repeated trials in both train and test.
6. Treat the single GaitIntent amputee participant as a final transfer case, never pooled training data.
7. Keep the existing AMASS test split sealed until a method and threshold are fixed across at least three validation seeds.

## 5. Mandatory baselines

Every study includes the applicable members of this list:

- raw coordinates, velocities, accelerations, and handcrafted gait measurements;
- duration, first and last frame, bounding-box area, foreground area, centroid drift, image size, and frame rate;
- pose confidence, missingness, bone-length error, jerk, cadence, speed, camera motion, and phase confidence;
- a random encoder with the same downstream head;
- an equal-capacity raw-input head;
- time, person, joint, intervention, teacher, or anchor shuffles appropriate to the mechanism;
- the strongest simple dynamical baseline, such as persistence, periodic template, linear state space, or conditional mean;
- source or participant cluster bootstrap intervals;
- parameter and compute matching for every learned comparison.

A representation can be useful only if it improves the conditional model that already contains raw and nuisance features.

## 6. Compute and checkpoint rule

The local anchor is about three H100-hours for 100 S-JEPA epochs on AMASS. Small heads and adapters should finish much faster. Independent folds and seeds may run in parallel on eight H100s, but total GPU-hours are still reported.

Allowed pretrained dependencies are public and downloadable before day 1:

- the official V-JEPA 2.1 ViT-B checkpoint and code;
- the official GaitDynamics diffusion and refinement checkpoints when used only as baselines or frozen critics;
- public MotionBERT or other documented pose baselines;
- local S-JEPA checkpoints with exact hashes and data lineage.

No proposal depends on an unreleased S-JEPA checkpoint, FoundationGait weights marked as coming soon, private GaitForeMer clinical data, or unavailable GAVD videos.

## 7. Mechanism-first reporting

Each proposal has four reporting layers:

1. **Availability:** Were data, labels, and checkpoints actually accessible?
2. **Identifiability:** Is the target predictable from information available before the prediction point?
3. **Mechanism:** Does the proposed pathway beat shuffles, random encoders, and simpler rules?
4. **Transfer:** Does the passed mechanism survive unseen people, sources, views, or protocols?

Failure at an earlier layer stops stronger claims. A higher downstream score cannot rescue a failed identifiability or mechanism test.

## 8. Language contract

- Say “observed GAVD presentation,” not diagnosis.
- Say “simulated fall event,” not real fall risk.
- Say “model-predicted force,” not measured force.
- Say “unseen source,” not unseen person, for GAVD.
- Say “future-latent predictability,” not causal influence.
- Say “upper-body predictive surplus,” not compensation, unless a known perturbation study shows the surplus changes after disturbance.
- Say “correspondence anchor,” not laterality truth, unless the anchor is independently observed.

## 9. Selection rule after 48 hours

Select the flagship with the strongest mechanism margin, not the highest raw metric. Define margin as improvement over the strongest prespecified baseline divided by its source- or person-bootstrap standard error. A proposal advances only if the sign is positive, the trivial-signal shuffles fail, and the required data can be processed inside the remaining schedule.
