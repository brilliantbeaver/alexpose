**Role**: You are an expert AI/ML researcher specializing in world models and JEPA.

**Task**: You are to carefully and systematically set-up and build the following additional set of experiments serving as inexpensive, decisive checks on the current synthetic training results:

1. Fit simple calibration controls using the cached training data.

    Add a per-joint constant offset and a small affine/ridge correction. Fit them only on
    training people and training extractors; evaluate the pooled correction on the held
    extractor without using its labels for calibration.

    Compare their coordinate and motion errors with initialized, coordinate, direct, and
    paired JEPA.

    If a tiny calibration matches most of the neural improvement, you have evidence that the
    main gain does not require temporal representation learning. This is especially plausible
    because your targets are projected SMPL-H joint centers, while the pose estimators follow
    image-keypoint conventions. Joint-mapping errors are a documented source of misleading
    pose evaluation. Hedlin et al.

    That mechanism remains a hypothesis until these controls are run.

2. Establish the timing metric’s attainable support.

    Evaluate the saved reference coordinates against themselves. This reveals how many
    records could support the timing metric even with perfect predictions.

    Then separate reference-ineligible records from predicted peak-count mismatches. Report
    missed and extra peaks alongside timing error, rather than treating an unsupported record
    as a successful result.

    I found a concrete reason this matters: the current camera is initially frontal, while
    timing uses horizontal ankle separation. Forward/backward leg motion can be foreshortened
    in that view. This may weaken the reference signal; the saved trajectories must determine
    whether it actually does. Camera implementation (src/gavd6_sjepa/research_directions/
    synthetic_training/rendering.py:62)

3. Inspect the existing learning curves and motion outputs.

    Plot reference, unchanged, calibrated, initialized, coordinate, and paired-JEPA ankle
    trajectories for the same preselected windows. Examine clean and corrupted conditions
    separately.

    Also inspect displacement error, amplitude ratios, per-joint residuals, and training
    convergence. These analyses can distinguish several explanations: systematic calibration,
    insufficient training, weak timing observability, or actual distortion of motion.

Ultrathink on how to implement these experiments as cleanly and efficiently as possible. Make sure that there are no errors and that your code is readable. Thoughtfully update the Slurm guide for the current study explaining how the user can run the above 3 checks as bash scripts. Use the same structure and style as the rest of the document.

Use independent adversarial review to thoroughly check your work and thoughtfully incorporate any suggestions that it gives.

Use fan out subagents with dynamic workflows.
