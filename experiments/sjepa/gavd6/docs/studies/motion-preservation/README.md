# Motion preservation during tracking repair

**Current study.** Test whether video-supported repair preserves real movement at comparable tracking-error removal, beyond calibrated confidence and optical-flow baselines. This is offline restoration using the complete declared clip. A successful gate alone does not establish a JEPA representation benefit.

| Read in order | Purpose |
| --- | --- |
| [Protocol](protocol/) | Research question, controlled event/error pairs, calibration, baseline obligations and decision rules. |
| [Execution](execution/) | Six-stage notebook workflow, configuration, pilot launch and validation limits. |
| [Results](results/) | Validation record and evidence boundaries; completed software checks must be distinguished from real pretrained-model experiments. |
| [References](references/) | Motion priors, optical-flow evidence and the shared research agenda. |

The [six notebooks](../../../notebooks/motion_preservation/README.md) expose the experiment in order; [Python implementation](../../../src/gavd6_sjepa/research_directions/motion_preservation/) owns the numerical methods, [notebook builders](../../../scripts/research_directions/motion_preservation/) own the tutorial sources, and the [Slurm guide](../../../slurm/motion-preservation/README.md) owns cluster execution. The [AMASS inspection notebook](../../../notebooks/amass/01_visualize_amass_smplh_poses.ipynb) supports raw-motion checks.

Train, calibration, development and final people remain separate. The reserved final event family is opened only after the development decision. GAVD is a visible-motion stress test and requires the earlier source-reservation CSV; reconstructed trajectories are not 3D ground truth. The [13 September 2026 repository assessment](../../repository/history/archive-assessment-2026-09-13.md) found that reservation file and registered output bundles absent locally. Their scientific identities must be recovered, not replaced with convenient new splits.

The next research step is a real-checkpoint pilot: verify event erasure, calibrate strong simple baselines, and compare retention at locked operating points with comparable achieved error removal. Demonstration outputs and successful software tests do not establish that result.
