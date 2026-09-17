# What optical flow changes in this portfolio

> **Scope: September 12–13 portfolio.** Proposal identifiers and priority statements refer to the [earlier decision record](../archive/2026-09-13/README.md); its P7 is motion beyond joints. These references still support the motion-preservation study. See the [research agenda](../README.md) for later proposals and decisions.

Optical flow strengthens the leading proposal and supplies a better replacement for the weakest candidate. The final seven now include **[P7: Keep motion that the skeleton cannot explain](../archive/2026-09-13/proposals/motion-beyond-joints.md)**. The old constraint-conflict idea is preserved as a [rejected reserve](../archive/2026-09-13/proposals/conflicting-motion-requests.md).

## Start with what is actually observed

A skeleton records a few landmark positions. Optical flow estimates how image locations correspond between frames, potentially at every visible pixel. It can therefore describe motion between landmarks. A rotating sleeve pattern may move even when an elbow-to-wrist line does not.

Flow is not an independent sensor. Pose and flow are two estimates derived from the same video, with different errors and different retained information. Neither estimated flow nor agreement between two algorithms establishes the true motion. Camera motion, loose clothing, shadows, occlusion and missing texture can all matter.

The distinction between **optical flow** and **flow matching** is also important. Optical flow concerns image correspondence. Flow matching is a way to train a generative model. A paper with “flow” in its name may concern the second meaning and supply no optical-flow measurements.

## The two strongest uses

**P1 uses flow to check a proposed repair.** Transport visible image points with a frozen flow model, then compare that transport with the raw and repaired pose trajectories. The test asks whether the pixels support an unusual movement that a motion prior would remove. The adapter must beat a direct flow-based check, with the same video and calibration data. Flow is a baseline and a possible useful input, not an automatic novelty claim.

For a truly tracked surface point, a basic check is `next location ≈ current location + estimated flow`. An anatomical joint projected onto clothing is not necessarily that surface point. Use local support and known rendered correspondences to calibrate the check; do not declare every disagreement a pose error. Convert all crops back to a common coordinate system before comparison.

**P7 asks whether skeleton agreement can itself be harmful.** First construct motions whose final model-visible joint histories are identical but whose visible surface transport differs. Then test whether coupling surface flow to joint displacement removes that valid difference. Finally add a small number of surface-motion tokens to S-JEPA and demand a useful forecast gain on natural motion, beyond raw flow, simple fusion and extrapolation.

This is stronger than saying that flow adds information. The research object is a controlled limitation of the chosen skeleton state, a possible failure of forcing two representations to agree, and the amount of extra observation needed to repair the limitation.

The two directions share extraction and rendering infrastructure, but ask different questions. P1 distinguishes true motion from measurement error. P7 preserves visible motion that accurate joint positions still omit. Do not combine their headline claims until each passes its own test.

## Why this is not simply a new two-stream model

[Pose from Flow and Flow from Pose](https://openaccess.thecvf.com/content_cvpr_2013/papers/Fragkiadaki_Pose_from_Flow_2013_CVPR_paper.pdf) already couples pose and flow. [MC-JEPA](https://arxiv.org/abs/2307.12698) already learns flow and content together. [H-MoRe](https://arxiv.org/abs/2504.10676) already builds human-centric motion representations with skeleton and boundary guidance. These are direct prior work, not peripheral citations.

[HTD-Refine](https://arxiv.org/abs/2605.26879) also targets motion recovery and oversmoothing using high-order joint dynamics. Its [current official repository](https://github.com/ant-research/HTD-Refine) now provides code and a checkpoint link, despite the older paper's release promise. It is a strong P1 comparator, although it is not itself an optical-flow method.

The new proposals must beat these ideas' strongest available implementations or clearly labeled reproductions. If generic flow fusion or a simple transport check is sufficient, the corresponding method claim fails.

## A concrete, inexpensive starting stack

Freeze [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT), whose author-linked [78.8 MB checkpoint](https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M/blob/main/model.safetensors) is public. Use official RAFT weights as a fallback and [CoTracker3](https://github.com/facebookresearch/co-tracker) as an optional longer-track comparison. Do not train a flow foundation model this week.

On AMASS renders, store exact surface correspondence and visibility independently of all estimators. On GAVD, use estimated flow as uncertain evidence and report visible failure cases. Camera stabilization should be tested under its assumptions; one global homography cannot generally remove all parallax. Occluded or textureless regions require uncertainty, not a fabricated zero-motion target.

Four, eight or sixteen flow tokens reduce the downstream state size. They do not eliminate the cost of computing dense flow. Report extraction time, memory and forecast time separately. All forecasting features, including offline point tracks, must use only frames before the prediction cutoff.

## What has been checked and what remains an experiment

The [rigid-chain algebra check](../archive/2026-09-13/checks/verify_joint_preserving_rotation.py) verifies the proposed compensated-rotation construction in 100 randomly generated kinematic configurations. Joint positions remain equal to numerical precision while an off-axis material point moves. Its [saved output](../archive/2026-09-13/checks/joint-preserving-rotation-check.json) is explicitly a mathematical sanity check, not an AMASS, SMPL, optical-flow or forecasting result.

The first 24-hour experiment must establish equality of the actual S-JEPA tensors after the full body-model and preprocessing pipeline. The 48-hour gate must show that estimated flow recovers useful visible evidence and that the natural-motion task survives the strongest simple baselines. The detailed [world-model review](../archive/2026-09-13/reviews/optical-flow-and-world-models.md), [measurement review](../archive/2026-09-13/reviews/optical-flow-measurements.md) and [candidate audit](../archive/2026-09-13/reviews/optical-flow-research-options.md) record sources and unresolved conditions.

I would begin with P1 and this P7 measurement assay. They use the existing AMASS and GAVD resources, need no new clinical collection, and can fail cheaply. Their conceptual upside is stronger than another pose-plus-flow classification result.
