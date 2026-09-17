# Optical flow as movement evidence

Research memo, 12 September 2026. This is a bounded addition to the proposal search, not evidence that the experiments have succeeded. No model was trained or loaded for this memo.

## Recommendation

Optical flow strengthens proposal 1 most directly. It provides a concrete way to ask whether a proposed skeleton change is supported by nearby image motion. A separate flow proposal should proceed only if it establishes a new measurement capability or a failure of existing observation models. Adding a flow stream, training a flow teacher, or removing camera motion is already well established.

Flow and pose are different estimates from the same images. They can fail differently, but they are not statistically independent observations. Agreement can reflect the same blur, occlusion or clothing mistake. In particular, a flow model trained to imitate the pose tracker can repeat that tracker's errors. The benchmark must test this circularity rather than reward agreement by definition.

## What flow measures

A flow vector describes the displacement of an image location between two frames. It is measured in pixels. It is not a joint angle, 3D velocity, force, or direct measurement of a hidden anatomical joint. A sleeve can move relative to an elbow. A shadow can move without the leg moving. A textureless region can admit many equally good matches.

For a candidate image-space keypoint path q, a simple diagnostic is

`r[t] = q[t+1] - q[t] - u[t](q[t])`,

where u is the flow field. A large residual says that the proposed keypoint displacement disagrees with the image transport sampled there. It does not prove which estimate is wrong. Use a robust local neighborhood with part masks and visibility checks, because the sampled pixel usually represents skin or clothing, not the physical joint center.

All terms must use the same image coordinate system. Map the endpoints of cropped flow back to the full frame before subtracting them. A different crop at each time can create or remove apparent motion. If camera compensation is used, transform both the keypoint displacement and flow identically. A single background homography is only an approximation under parallax; test moving cameras separately.

## Primary literature and novelty collisions

| Work | What it already establishes | Consequence for our claim |
| --- | --- | --- |
| [Two-Point Gait, ICCV 2013](https://openaccess.thecvf.com/content_iccv_2013/html/Lombardi_Two-Point_Gait_Decoupling_2013_ICCV_paper.html) | Flow statistics can represent limb motion while reducing body-shape dependence. Primary abstract inspected; direct full text failed in this session. | Shape-separated flow for gait is not a new conceptual contribution. |
| [Optical Flow-based 3D Human Motion Estimation, 2017](https://arxiv.org/pdf/1703.00177) | Fits human motion by matching estimated flow to an artificial flow renderer. Full paper available. | Flow agreement as a motion-fitting constraint is old. |
| [Uncertainty-Aware Human Mesh Recovery, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Lee_Uncertainty-Aware_Human_Mesh_Recovery_From_Video_by_Learning_Part-Based_3D_ICCV_2021_paper.html) | Combines uncertain static pose features with optical-flow dynamics and body-part decoding. Primary abstract and indexed method text inspected. | Local flow, uncertainty and pose fusion alone are insufficient. |
| [Using Motion Cues to Supervise Single-Frame Body Pose and Shape Estimation, TMLR 2024](https://infoscience.epfl.ch/bitstreams/c9dbafbc-3cba-4b0b-96fa-6fb4374ef093/download) | Uses correspondence between predicted body motion and image flow as supervision in low-data settings. Primary abstract inspected; OpenReview browser challenge blocked direct reading. | Flow as cheap supervision for pose already exists. |
| [MC-JEPA, 2023](https://arxiv.org/html/2307.12698v1) | Jointly learns optical flow and content with a shared encoder, including warping and cycle constraints. Full methods inspected. | “Add optical flow to JEPA” is already occupied. |
| [H-MoRe, CVPR 2025](https://arxiv.org/html/2504.10676v1) | Learns human-specific flow using skeleton and boundary constraints, then separates overall and body-relative motion. Evaluates gait recognition, action recognition and video generation. Full methods inspected. | A body-relative flow representation is not new. Its pose-supervised flow is a valuable circularity comparator when pose is corrupted. |
| [Skeleton-optical fusion for post-stroke gait, 2025](https://link.springer.com/article/10.1186/s12984-025-01726-5) | Combines optical flow, kinematics and learned features to distinguish gait groups. Full article accessible. | Clinical classification from skeleton-flow fusion is both occupied and outside this user's desired headline. |
| [SEA-RAFT, ECCV 2024](https://arxiv.org/html/2405.14793v1) | Predicts flow using a mixture-of-Laplace objective, exposing uncertainty estimates. Full method inspected. | A learned flow uncertainty head is not our contribution; its probabilities require evaluation under our shift. |
| [MFTIQ, WACV 2025](https://arxiv.org/html/2411.09551v1) | Separates correspondence-quality estimation from the flow method, supports multiple flow estimators and causal long-term tracking. Full methods inspected. | An independent quality network, flow chaining and cross-estimator quality transfer are strong existing baselines. |
| [TETO, 2026](https://arxiv.org/html/2603.23487v1) | Distills a pretrained RGB tracker into event-camera motion estimation with about 25 minutes of recordings and conditions video diffusion using explicit motion. It removes dominant camera motion during data selection. Full methods inspected. | Small-data tracker distillation and flow-conditioned video generation already meet much of the user's adaptation inspiration. We need a different scientific question. |
| [HTD-Refine, CVPR 2026](https://arxiv.org/html/2605.26879v1) | Predicts position, velocity and acceleration from video, then refines motion while resisting excessive smoothing. Full methods inspected. | Preserving high-frequency movement through temporal evidence is a direct novelty threat, not a remote related paper. |

The targeted search used combinations of optical flow, gait, pose refinement, calibration, uncertainty, distillation, JEPA, transport, observability and motion forecasting. These results do not prove absence of closer work. They are sufficient to reject generic two-stream or confidence-gating novelty claims.

## Public artifacts and access status

**Default measurement model: frozen SEA-RAFT.** The [official repository](https://github.com/princeton-vl/SEA-RAFT) links the author model `MemorySlices/Tartan-C-T-TSKH-spring540x960-M`. Its [public file listing](https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M/tree/main) exposes a 78.8 MB `model.safetensors`. Exact binary URL: `https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M/resolve/main/model.safetensors`. The official example returns both flow and uncertainty from two RGB frames. Keep its backbone frozen.

**Simple fallback: RAFT.** The [official download script](https://raw.githubusercontent.com/princeton-vl/RAFT/master/download_models.sh) publishes `https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip`. Use its `raft-sintel.pth` or `raft-small.pth` with a separate calibration rule. Do not describe RAFT's ordinary output as a calibrated probability distribution.

**Longer tracks and quality baselines.** [CoTracker3](https://github.com/facebookresearch/co-tracker) has [public 102 MB online and offline weight files](https://huggingface.co/facebook/cotracker3/tree/main). Exact online binary: `https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth`. [MFTIQ's official script](https://raw.githubusercontent.com/serycjon/MFTIQ/master/download_model.sh) publishes its quality model at `https://cmp.felk.cvut.cz/~serycjon/MFTIQ/materials/UOM_bs4_200k.pth` and additional flow weights. These are baselines, not reasons to train another tracker.

**HTD-Refine access changed after its paper.** The paper and project page say code is forthcoming, but the current [official repository](https://github.com/ant-research/HTD-Refine) provides working commands and a [public checkpoint folder](https://drive.google.com/drive/folders/1Rg6ajhTRwebUZ_coSvfTlxUfPMD0CR9c?usp=sharing) for `pvanet.pt`. The folder opens, but its individual file was not enumerated by this browser tool. The demo requires 30 FPS, SMPL assets, initial HMR motion and compatible camera inputs. Treat the released implementation as a first-day comparator check. If the checkpoint cannot be obtained, use an explicitly labeled adaptation of its position/velocity/acceleration objective and document what cannot be reproduced. Do not call it an exact reproduction.

**H-MoRe.** The primary paper points to [haku-huang/h-more](https://github.com/haku-huang/h-more), but repository access failed in this session and no weight binary was verified. Its method remains a mandatory conceptual comparator. Reuse its stated skeleton-flow agreement idea in a labeled baseline if released assets remain unavailable; do not pretend that this duplicates its reported results.

All local HTTP HEAD checks failed because this sandbox could not resolve hostnames. Public file listings and source scripts were read through the web tool. Successful artifact loading on HAIC remains an execution check, not a permission request or a reason to build a new foundation model.

## Idea A: Let image transport veto a destructive motion repair

**Question.** Can an inexpensive adapter preserve more true local movement at the same amount of tracking noise removed, when the raw skeleton alone is exactly uninformative about whether the event is real?

This is the optical-flow version of proposal 1. A frozen motion prior proposes a repair. Frozen flow, local point tracks and optional video features assess the raw and repaired paths against image transport. The adapter retains only the supported part of their difference. The new claim is the measured preservation-versus-repair improvement across unseen event types and prior families. It is not the residual formula or flow consistency itself.

The main matched pair has identical skeleton arrays, confidence and timing. In one video a real foot excursion occurs; in the other only the pose estimate has the same excursion. Add a second family that changes arm-leg timing, then hold it out completely. Use crossed event/noise conditions, including simultaneous event and corruption. Score the final constrained motion, not the unprojected gate output.

Use two forms of evidence: local displacement agreement and multi-frame support. The first catches a track jumping while the surrounding surface stays still. The second checks that candidate movement remains supported across several observed frames. Neither is trustworthy during total occlusion or shared failure. In those cases return an unresolved probability instead of forcing a choice. Frame pairs after a forecast cutoff are forbidden in any forecasting use; offline restoration may use the full declared clip.

**Necessary comparisons.** Plain flow gating, local Lucas-Kanade or robust patch transport, SEA-RAFT uncertainty, MFTIQ quality, two pose trackers, HTD-style derivative refinement, H-MoRe-style skeleton-flow regularization, the unmodified prior, and a same-capacity coordinate gate. A learned video representation earns its complexity only beyond these controls.

**First 48 hours.** Start with 128 independent motion instances, not a giant rendering set. Require the gate's improvement beyond the best flow-only rule at calibration-locked noise-removal levels. The existing proposal's 15-point retention and 25% corruption-removal thresholds remain planning gates. Stop if ordinary flow propagation solves the task or if all apparent improvement comes from more confident rejection of easy cases. The 280 GPU-hour cap can cover this branch if flow is cached and the two expensive video encoders are not multiplied into a large grid.

**Principal risk.** HTD-Refine already studies loss of real temporal detail. The potential distinction is exactly matched true-event versus tracker-error evidence, event retention at matched repair quality, and explicit cases where all observations remain ambiguous. A result limited to smoother motion or lower velocity error is insufficient.

## Idea B: Measure a local movement only when the visible evidence supports it

**Question.** Can a small adapter produce narrower, empirically calibrated intervals for a local image-space movement descriptor than strong pose and flow baselines, while correctly leaving unobservable cases unresolved?

This is a possible independent measurement proposal, not a recommendation to replace a stronger proposal automatically. The output is a checkable movement interval in image coordinates, such as torso-relative displacement of a visible shoe region over 0.4 seconds or the delay between left and right arm reversals. It does not require lifting every frame to a single 3D skeleton. It must not be renamed clinical toe clearance, joint force or a diagnosis.

A frozen flow model supplies image transport. A small S-JEPA adapter pools part-local tracks and their visibility to estimate a descriptor and its uncertainty. A skeleton estimate supplies an optional anatomical organization, not mandatory truth. A motion prior may guide missing regions, but a narrower interval is rewarded only when measured coverage is retained. Compare independent flow against flow trained to agree with the same pose tracker. This tests whether a common practice makes two estimates agree by deleting information that would have exposed an error.

Construct a two-by-two observation audit: reliable pose and flow; unreliable pose but visible surface transport; reliable pose with corrupted texture or shadows; and both unreliable. Match motion magnitude, clip length, camera, shape and foreground area within contrasts. The headline is local descriptor error and interval width at common coverage across these cells, with a held camera or clothing condition. A gait label is unnecessary.

An optional stronger behavior is to report how much additional existing video is needed before an interval becomes decisive. Use a fixed delay grid, for example 0.1, 0.2, 0.4 and 0.8 seconds. Calibrate all delays jointly on independent sequences, because selecting a delay after inspecting uncertainty invalidates a naive per-time coverage claim. Never claim conditional or distribution-free coverage on arbitrary GAVD shift from synthetic calibration alone.

**First 48 hours.** Compare raw flow pooling, MFTIQ, confidence-weighted pose, simple constant-velocity models and the proposed adapter on several hundred rendered windows grouped by original motion and person. Require at least a provisionally 20% reduction in interval width at the same target coverage, with no concentrated undercoverage in the rare-event or shared-failure cells. Stop if pooling plus ordinary calibration matches the result. Cap a complete follow-up at 200 H100 GPU-hours, an unmeasured allowance, using frozen caches and three small-head seeds.

**Novelty boundary.** This is not novel merely because it produces uncertainty or avoids 3D. MFTIQ already predicts correspondence quality, and H-MoRe already uses body-relative flow. A plausible contribution requires demonstrating that pose-conditioned agreement corrupts measurement under a known failure regime and providing a general remedy with a better accuracy-coverage tradeoff. If it becomes only a calibrated gait-specific tracking head, its ICLR case is weak. It is currently less compelling than Idea A.

## Rendering exact flow from AMASS

AMASS is motion data, not a ready-made RGB/flow dataset. Existing body-model assets are needed for a faithful mesh render. Whole-body meshes include arms, hands as available in the source representation, trunk and feet; Core11 is insufficient to define all surface motion.

Two practical routes exist. [BlenderProc's official renderer](https://dlr-rm.github.io/BlenderProc/examples/advanced/optical_flow/README.html) exposes forward and backward flow as float arrays. For explicit control, [PyTorch3D's mesh rasterizer](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/mesh/rasterizer.py) exposes visible face IDs, depth and barycentric coordinates. Track each visible pixel's surface point through the same mesh triangle at the next time, project with that frame's camera, and subtract its starting pixel coordinate. Check next-frame visibility against depth. This is an implementation inference from the documented rasterizer interface, not a verified existing repository pipeline.

Validate flow direction, pixel origin, perspective interpolation, moving cameras, occlusion and disocclusion on a rigid translating test object before using body motion. Do not use a stick-figure renderer as evidence that tracking on clothing works. A fixed-topology capsule body can debug the interface without new body-model downloads, but it is not a sufficient final realism test.

Randomize texture, background and lighting while holding true motion fixed. Add separate camera, shadow, cloth-like displacement, blur, compression and occlusion perturbations. Geometry flow is material correspondence; shadows and illumination can change apparent motion without changing that ground truth. This distinction is part of the experiment, not a reason to declare the flow model wrong on every photometric change.

GAVD supplies the relevant in-the-wild stress domain. Its normal/abnormal labels do not validate local flow or metric movement. Without new clinical collection, a small blinded annotation of visible material-point displacement can test image-space accuracy; if that annotation is not done, real-video results must remain stress tests and qualitative audits. Do not score a flow method against its own pseudo-labels.
