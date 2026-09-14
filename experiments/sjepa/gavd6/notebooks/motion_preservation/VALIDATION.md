# Implementation validation

Validated on 13 September 2026. These checks establish that the research workflow runs and preserves its evaluation boundaries. They do not establish that the proposed method works on real motion.

| Check | Result |
| --- | --- |
| Focused data, adapter, learning, metric, notebook and workflow tests | 41 passed in 16.6 seconds, including the optional author-conversion test |
| Fresh Jupyter kernels for all six notebooks | Latest executed copies: 34 code cells completed, zero error outputs |
| Explicit CPU demo | Preparation, estimated flow, baselines, gate fitting, calibration and development reporting completed |
| Final-stage lifecycle test | Final cases required an existing fit and calibration; evaluation preserved both files and subsequent fitting was rejected |
| Exact ambiguity fixtures | Identical measured feature tensors with distinct reference trajectories |
| Person-level inference | Repeated trials did not create additional independent people or change person weighting |
| Official HumanML3D conversion | Author example round-trip mean joint error approximately 0.00000023 m |
| Official RVQ interface | Encode, quantization, decoder, padding and inverse conversion exercised with a small random-weight CPU fixture |
| Slurm and notebook sources | Shell syntax, dependency sequencing, explicit final selection and output-free notebook sources checked |
| Body-model dependency | Project and lock select the official GitHub implementation; a stale local PyPI install was identified and setup instructions corrected |

The notebook smoke used generated articulated meshes, a smoothing stand-in and Farneback flow. Its decision is explicitly `demo_only_no_research_decision`. The GAVD notebook returned `not_run_demo`. Simulated person labels do not establish statistical generalization.

The refreshed walkthrough used one generated motion per role, 1.2-second clips at 64-pixel image size, one training epoch, four feature arms, and seed 17. Notebooks 00 through 05 completed 6, 5, 6, 5, 7, and 5 code cells respectively in separate fresh kernels. The reserved final set remained unopened. Calibration and evaluation were repeated after the observed-joint repair metric was introduced during review: evaluation correctly rejected the earlier operating points, then completed after recalibration. The resulting calibration and decision both record `noise_removal_scope: "observed_joints"`. This tiny run checks execution, not statistical power or model quality.

Tests also cover perspective renderer correspondence, missing-diagnostic masks, crop endpoint transforms, motion-prior frame alignment, differentiable projection, unmodified prior/conversion comparisons, and preservation of GAVD confirmation reservations. An intentionally failing notebook is used to check that partial outputs survive; its expected failure message is not a failed test.

The review added regressions for close-surface occlusion, nonfinite dynamic shape, missing-coordinate score inflation, low-frame-rate video playback, fractional frame indices, explicit configuration handling, and demo/real separation. The integrated test also checks that both methods meet the event-plus-noise repair target and that old all-joint calibration is rejected. See the [implementation review](../../docs/studies/motion-preservation/results/implementation-review.md) for findings and their scientific consequences.

The real AMASS SMPL-H/DMPL assets, released MoMask/SEA-RAFT weights, HAIC GPU execution and real GAVD videos were not available to this local validation. No Slurm jobs were submitted. External MDM and exact paper comparisons remain explicit prediction-import paths, not locally executed reproductions.

To repeat the focused tests in an existing compatible environment:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/motion_preservation -p 'test_*.py' -v
```

The author-conversion test is optional unless `MOTION_PRESERVATION_MOMASK_REPO` points to a local official checkout with its bundled example. The real pipeline setup and model download commands are in the [Slurm guide](../../slurm/motion-preservation/README.md).
