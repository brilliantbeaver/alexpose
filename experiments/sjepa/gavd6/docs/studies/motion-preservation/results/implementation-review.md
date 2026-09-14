# Motion-preservation implementation review

Reviewed on 13 September 2026 against the [study protocol](../protocol/proposal.md), with separate reviews of data and geometry, pretrained-model interfaces, training and evaluation, and notebook/Slurm execution.

**The audit found and fixed several errors that could block experiments or distort their interpretation. The corrected local workflow passes its focused tests and notebook walkthrough. Real AMASS reconstruction and released-model inference on HAIC still need to run before the implementation can support a scientific result.**

The changes are limited to the study code, focused regression tests, notebooks and launch documentation. No experiment management framework or broad reproducibility system was added.

## 1. Errors that could distort the result

| Finding | Why it matters | Correction |
| --- | --- | --- |
| Gap filling could dominate `noise_removal` | A constructed counterexample reported approximately 98% repair while every observed-coordinate error stayed unchanged. The improvement came entirely from completing missing joints. | Compute the repair ratio on observed joints using the same mask for raw and repaired motion. Report completion error, observed error, all-joint error and missing fraction separately. |
| The same-clip comparison did not require the comparator to reach the repair target | A comparator could pass the overall noisy-case target but miss it on event-plus-noise cases, where preservation is being compared. | Both methods must reach the target on the mixed cases, as well as have comparable achieved removal. |
| Near-surface occlusion could enter the flow reference | The previous 1.5 cm depth tolerance accepted a surface newly hidden by another surface only 1 cm closer. These are invalid image correspondences. | Check projected depth at the actual transported subpixel point. Exclude unresolved boundaries and report reference coverage beside flow error. |
| Demo outputs could be relabeled as real | Evaluation accepted a changed run mode even when the saved fit and cases came from a demo. | Require agreement between configured mode, saved fitting/calibration mode and cached case mode. |
| MoMask accepted motion at the wrong frame rate | A 30 Hz trajectory could be interpreted through its 20 Hz representation, changing physical motion timing. | Reject real MoMask extraction unless the configured data rate is 20 Hz. |
| Fractional frame indices were silently truncated | Imported predictions or GAVD overlays could be attached to the wrong original frames. | Require integral frame indices before conversion to integers. |
| GAVD resize and playback could misrepresent movement | Overlay coordinates omitted the pixel-center offset; deduplicated low-rate source frames were played at the requested higher rate. | Use the OpenCV pixel-center transform and the actual sampled playback rate. |

The repair metric now has a clear interpretation:

```text
noise_removal = 1 - observed_joint_MSE(repaired) / observed_joint_MSE(raw)
```

The correct reference includes the event when it is present. MSE here averages squared 3D joint distances, in square meters. An undefined denominator produces no ratio. Missing positions are interpolated for model input, but their guesses are not counted as tracker observations.

`completion_mse_m2` evaluates missing positions against their reference; `mse_m2` still evaluates every position. Event retention remains an unclipped descriptor-fidelity score, including possible overshoot, rather than a literal fraction of every movement detail preserved. The plot labels now state this and use the configured repair target.

New calibration files record `noise_removal_scope: "observed_joints"`. Evaluation rejects old calibration because those strengths were selected using a different error definition.

## 2. Errors that could block or redirect HAIC runs

All six notebooks imported a removed `reporting` module. They now import the existing `plots` module, and the builder generates the corrected cells. A fresh kernel, rather than source compilation alone, exposed this failure.

The launch path also supplied default mode and device flags that overrode JSON configuration. Omitted flags now preserve the selected configuration. Explicit environment variables or command-line flags still override it. An explicitly named missing `MP_CONFIG` fails clearly instead of silently starting a default experiment.

The installed local PyPI `human_body_prior` had an older constructor and did not apply the required DMPL dynamic shape parameters. The project already selects the official GitHub implementation. The [HAIC guide](../../../../slurm/motion-preservation/README.md) now explains the compatible installation for both the project environment and an alternative interpreter. Real body reconstruction checks for the expected SMPL-H model, 16 shape components and 8 active DMPL components. Nonfinite motion or returned geometry cannot become reference truth.

Slurm submission keeps `afterok` dependencies between notebooks 00 through 04. The pilot leaves final evaluation and GAVD unsubmitted. Dependency syntax, extra arguments, account/partition overrides, paths containing spaces and configuration-only launches were checked. BLAS thread settings follow the selected CPU thread count. The launch instructions preserve other packages in the shared environment and clarify external-prior configuration and multiple-seed execution.

## 3. Scientific boundaries that remain sound

- People are separated before generating event, camera and corruption variants. The current AMASS manifest review found 151 training, 9 calibration, 10 development and 19 final people. These are available manifest groups, not the number included in a pilot.
- Reference trajectories, event support and renderer flow are not model input features. Reference targets supervise fitting or score predictions; estimated image flow supplies the gate's motion evidence.
- Strengths and the comparator are selected on calibration people. Retention comparisons average within people, and the paired bootstrap resamples people.
- Final construction requires an existing fit and calibration. Opening final evaluation prevents further fitting or recalibration in that run.
- The ambiguity fixture provides identical measured features with different hidden references. A fitted model cannot infer the hidden explanation from those inputs.
- Transfer to a second imported prior keeps the fitted gate, normalization and calibration fixed. Exact external paper comparisons remain explicitly missing until their predictions are supplied.
- GAVD respects the existing source reservation. Its current stage provides visual stress diagnostics, not a validated transfer of the trained 3D gate or 3D preservation ground truth.

The final condition changes event family, camera and corruption together. Its result supports a combined stress-test claim, not isolated event generalization. The nuisance probe only tests its listed scalar features; passing it cannot establish the absence of every shortcut. Seed consistency currently checks positive retention differences, not that every seed independently passes the entire decision rule.

## 4. Evidence from the checks

The focused suite passed **41 tests**, including the official HumanML3D example conversion. Its coverage includes coordinate transformations, missing-data behavior, model interfaces, person aggregation, calibration and final boundaries, and the concrete errors above.

All six notebooks completed in fresh CPU kernels: **34 code cells, with zero errors in the final executed copies**. The walkthrough used one generated motion per role, 1.2-second clips, 64-pixel images, one epoch and four feature arms. Evaluation correctly rejected calibration made before the metric correction; recalibration and the remaining notebooks then succeeded. The decision remained `demo_only_no_research_decision`; GAVD returned `not_run_demo`; the final set remained unopened.

Shell syntax and mocked scheduler submission passed. No real Slurm jobs were submitted. See the [validation record](../../../../notebooks/motion_preservation/VALIDATION.md) for the test command and interpretation limits.

## 5. Use the corrected implementation

For a new HAIC pilot, use a fresh run directory and follow the [launch guide](../../../../slurm/motion-preservation/README.md). Inspect notebook 00, then submit the small 00-to-04 chain. Notebook 01 must establish real SMPL-H/DMPL reconstruction; notebook 02 must load the released prior and estimated-flow weights and show whether the prior actually erases the selected event.

For an existing fit with final evaluation still unopened, `workflow.calibrate(cfg)` can replace old operating points and evaluation can be rerun. To include the corrected renderer reference, start a fresh run so old scenes are not reused. If a final result has already informed decisions, it becomes development evidence; moving it into a new folder does not restore independence.

The remaining practical uncertainties are the licensed body assets, released MoMask/SEA-RAFT inference, HAIC runtime and memory, and real GAVD video behavior. They were unavailable to this local audit. A passing CPU walkthrough establishes execution and selected scientific boundaries, not event preservation by a pretrained model, clinical validity or paper readiness.
