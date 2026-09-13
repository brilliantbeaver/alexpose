# Implementation validation

Validated on 13 September 2026. These checks establish that the research workflow runs and preserves its evaluation boundaries. They do not establish that the proposed method works on real motion.

| Check | Result |
| --- | --- |
| Focused data, adapter, learning, metric, notebook and workflow tests | 33 passed |
| Fresh Jupyter kernels for all six notebooks | 34 code cells completed, zero error outputs |
| Explicit CPU demo | Preparation, estimated flow, baselines, gate fitting, calibration and development reporting completed |
| Final-stage lifecycle test | Final cases required an existing fit and calibration; evaluation preserved both files and subsequent fitting was rejected |
| Exact ambiguity fixtures | Identical measured feature tensors with distinct reference trajectories |
| Person-level inference | Repeated trials did not create additional independent people or change person weighting |
| Official HumanML3D conversion | Author example round-trip mean joint error approximately 0.00000023 m |
| Official RVQ interface | Encode, quantization, decoder, padding and inverse conversion exercised with a small random-weight CPU fixture |
| Slurm and notebook sources | Shell syntax, dependency sequencing, explicit final selection and output-free notebook sources checked |
| Dependency lock | Updated for the optional `motion-preservation` extra; offline lock check passed |

The notebook smoke used generated articulated meshes, a smoothing stand-in and Farneback flow. Its decision is explicitly `demo_only_no_research_decision`. The GAVD notebook returned `not_run_demo`. Simulated person labels do not establish statistical generalization.

Tests also cover perspective renderer correspondence, missing-diagnostic masks, crop endpoint transforms, motion-prior frame alignment, differentiable projection, unmodified prior/conversion comparisons, and preservation of GAVD confirmation reservations. An intentionally failing notebook is used to check that partial outputs survive; its expected failure message is not a failed test.

The real AMASS SMPL-H/DMPL assets, released MoMask/SEA-RAFT weights, HAIC GPU execution and real GAVD videos were not available to this local validation. No Slurm jobs were submitted. External MDM and exact paper comparisons remain explicit prediction-import paths, not locally executed reproductions.

To repeat the focused tests in an existing compatible environment:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_motion_preservation_*.py' -v
```

The author-conversion test is optional unless `MOTION_PRESERVATION_MOMASK_REPO` points to a local official checkout with its bundled example. The real pipeline setup and model download commands are in the [Slurm guide](../../slurm/motion-preservation/README.md).
