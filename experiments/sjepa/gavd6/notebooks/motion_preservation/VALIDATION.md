# Implementation validation

## Diagnostic stage, 14 September 2026

Notebook 06 and the `diagnose` Slurm phase were added after reviewing the real pilot. This stage reads its existing prediction cache to separate projection damage, prior and representation errors, useful repair, and available optical-flow evidence. It does not fit models, replace calibration, or open final cases.

| Check | Result |
| --- | --- |
| Complete motion-preservation test suite | 58 tests ran: 55 passed and 3 optional SEA-RAFT checkpoint tests skipped because the local environment lacks their required dependency |
| New diagnostic tests | 12 passed, covering corrupted versus accurate positions, the unprojected oracle, projection of variable reference lengths, exact zero strength, person weighting, flow support, matched pairs, and reserved roles |
| Fresh Jupyter kernel for notebook 06 | All 8 code cells completed with zero error outputs |
| Cached CPU demo | 32 cases across calibration and development, with two generated people per role; tables, two selected traces, and six SVG figures saved |
| Figure review | Repair, strength, bone-length, flow, and trace layouts inspected visually |
| Slurm launch | Shell syntax and launcher tests passed; `diagnose` selects only notebook 06 and requests 8 CPUs, 32 GB, and no GPU |

The fresh-kernel check used generated motion, a smoothing stand-in, and Farneback flow. It validates execution and diagnostic definitions, not a successful repair mechanism. The run's saved configuration may specify CUDA, but the diagnostic itself uses cached arrays on CPU. A focused test verifies that fit and calibration files remain unchanged.

The real pilot's prediction arrays remain on HAIC and were not available for this local diagnostic execution. No HAIC job was submitted. Run `bash slurm/motion-preservation/submit.sh diagnose` against that original run as described in the [Slurm guide](../../slurm/motion-preservation/README.md#5-diagnose-the-existing-pilot-without-another-model-run).

## Original workflow, 13 September 2026

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
