# Feasibility of a published skeleton anomaly baseline

Checked on November 15, 2026. **Neither official implementation is currently a validated comparator for our 16-joint, partially observed GAVD protocol.** STG-NF is easier to execute locally. SeeKer is the closer scientific comparison because it predicts conditional keypoint distributions. Their main shared obstacle is the treatment of missing coordinates. A quick zero-fill adapter would change the statistical model and would not establish a published-method reproduction.

This review inspected the official repositories, read the primary papers, and ran limited mechanical checks on unmodified model code. It did not train either method, change our model, install dependencies, or produce GAVD detection scores. The current study's matched ridge predictor is a useful control, but it does not substitute for a validated published anomaly detector.

## What was inspected and tested

| Item | Pinned version or basis | Observed status |
| --- | --- | --- |
| [Official SeeKer repository](https://github.com/adelic99/seeker/tree/7e7ab66a703188616b0efd482947ff2bf2bb26ca) | `7e7ab66a703188616b0efd482947ff2bf2bb26ca` | Source inspected. The unmodified model constructor fails on this laptop because its masks are moved directly to CUDA. |
| [Official STG-NF repository](https://github.com/orhir/STG-NF/tree/edb5f3220332e160e4d20ce258787d5e2d7e0200) | `edb5f3220332e160e4d20ce258787d5e2d7e0200` | Source inspected. An unmodified 18-joint model produced finite CPU outputs on synthetic finite input. |
| Local runtime | Apple arm64; Python 3.12.13; PyTorch 2.15.0; NumPy 2.5.4 | CPU checks only. No CUDA, MPS, optimizer, checkpoint or training validation was performed. |
| Our observation protocol | [data.py](../aaai_gait/data.py), [study_data.py](../aaai_gait/study_data.py), and [research roadmap](draft00/research_roadmap.md) | Sixteen selected joints, 32 frames at 15 Hz, four-frame tokens, confidence filtering, retained NaNs and source-held-out evaluation. |
| Historical completeness audit | [full_window_index.csv](../artifacts/full_window_index.csv) | Read-only count of the completed earlier five-class cohort. This is not the final expanded twelve-folder study cohort. |

Temporary repository checkouts were inspected outside the project. They were not vendored into the tutorial package. The historical window index had SHA-256 `53a08a8dffdfe0a691879587076dd6b04e1668811895cfd84cdf329483284142` at review time.

The STG-NF check used `pose_shape=(2,32,18)`, eight flow steps, one level, zero hidden channels, its default permutation and affine coupling, `R=3`, and CPU execution. This constructed 744 trainable parameters. A batch of two finite synthetic windows returned two finite negative log-likelihoods. After initialization, replacing one coordinate in the first window with NaN made that window's score nonfinite while the second score stayed finite. Reconstructing the same model with 16 joints failed because its graph still had 18 nodes. These checks establish a narrow execution fact, not training correctness, convergence or anomaly performance.

SeeKer's parser check confirmed that `droppout` exists and `dropout` does not. Static inspection confirmed that the trainer does not define `joint_lnp_full_window`. Importing its trainer in the current tutorial environment stopped at the missing `tqdm` package. Constructing its model directly, without the trainer, stopped at the hard-coded CUDA operation. No patches were applied to hide these failures.

## Why both methods matter scientifically

SeeKer models keypoint locations with autoregressive conditional Gaussians and combines keypoint log scores with detector-confidence weights. It already occupies the space of conditional skeletal prediction, compositional anomaly scoring and confidence-aware scoring. Therefore, predicting a hidden joint from other joints, reporting per-joint surprise, or using confidence cannot by itself establish our novelty. Our proposed distinction is the controlled comparison of own-joint, patch-local and full context, including a two-sided normal-reference profile and tests of contextual teacher effects. That distinction still needs evidence. [SeeKer paper](https://arxiv.org/html/2506.18368v1)

STG-NF directly learns a distribution over pose sequences using graph-conditioned normalizing flows. It supports training on normal examples and scores sequence likelihood. Normal-only skeleton anomaly detection and learning normal motion distributions are therefore established prior art. Our study must test a specific information hypothesis rather than claim novelty from normal-only fitting. [STG-NF paper](https://arxiv.org/pdf/2211.10946)

The two published methods are different kinds of baselines. SeeKer predicts ordered keypoints. STG-NF gives a whole-window density. Neither is a natural replacement for the exact matched raw-KL control inside the own-versus-full experiment. Their role is an external anomaly-detection comparison on a declared common source cohort.

## SeeKer: close scientific fit, incomplete released execution path

The released model is a masked fully connected autoregressive network. The masks express a fixed ordering, which is different from the uniform random target masks used by our S-JEPA. Adapting the published baseline does not require replacing its autoregressive factorization with our masking task.

The following issues are visible at the pinned version:

| Area | Verified issue | Required action before a result is credible |
| --- | --- | --- |
| Installation | The README references a `requirements.txt` that is absent. Its clone command contains a placeholder repository name. | Record an explicit dependency environment and use the actual official repository. |
| Entry point | The parser defines `droppout`; `seeker.py` accesses `args.dropout`. | Repair the argument mismatch and test the command-line path. |
| Device handling | The model, connectivity masks and confidence scoring contain direct CUDA calls. | Make device handling explicit and register masks as buffers. Check numerical agreement on an available reference device. |
| Test path | `test()` calls an undefined `joint_lnp_full_window`. It also unpacks three values from a scorer that returns one unless its separate return flag is enabled. | Resolve the intended inference path against the paper and a reference case. Do not guess which method name was intended and immediately report results. |
| Skeleton layout | The entry point uses 18 joints. The causal-mask context boundary subtracts `18*2`; validation reshapes scores with 18 joints. | Parameterize the joint count and audit the full flattening, ordering, output alignment and scoring path. |

These findings come from separate source files: [README](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/README.md), [arguments](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/args.py), [entry point](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/seeker.py), [masked model](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/models/made/made_partial.py), [training and inference](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/training.py), and [evaluation](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/validation.py).

Several methodological details also need adaptation:

- Its dataset loader converts COCO-17 detections into an 18-joint layout by deriving a neck point and reordering joints. Our allowlist cannot be represented by applying that conversion to a 16-joint tensor. The shoulders, hips, heels and foot indices must keep their actual identities. [Dataset conversion](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/dataset.py)
- Its normalization uses the mean position and vertical standard deviation across the whole input window. Our normalization uses each frame's pelvis and a prefix torso scale. Use the shared normalization for a matched-input comparison, and report the change. [Normalization](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/utils/data_utils.py)
- Its sequence utility interpolates missing frames. Our protocol preserves missing observations and never bridges a missing-frame gap. Reuse our accepted windows rather than silently importing that interpolation policy. [Sequence processing](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/utils/sequence_utils.py)
- The released trainer selects epochs by validation AUROC. For datasets other than UBnormal, its loader assigns test data to the validation role. Neither choice can be reused for our normal-only model selection. Our outer test sources must never choose a checkpoint. [Dataset roles](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/dataset.py), [checkpoint selection](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/training.py)

For a finite, fully observed input, a feasible layout adapter would use `[batch,time,16,2]` in the fixed numerical joint order. The flattened dimension would be `32*16*2=1024`; the context boundary would depend on `2*16`, not 36. Both XY coordinates of one target joint must be excluded from the conditioning set for that target. Verify this with numerical perturbation or Jacobian tests for every target position. In particular, audit the trainer's one-keypoint output offset before choosing an inference implementation. A changed order is an adaptation and should be fixed before inspecting abnormal performance.

The confidence weights need a separate decision. MediaPipe confidence values have not been established as calibrated probabilities of a correct joint location in this corpus. A weighted score can still be reported as the published scoring heuristic, with an unweighted sensitivity analysis, but it must not be described as calibrated uncertainty or as a solution to missing conditioning variables.

## STG-NF: executable core, incompatible graph and missingness assumptions

The official setup targets Ubuntu, Python 3.8 and a CUDA GPU. Its environment file pins PyTorch 1.10.1, NumPy 1.23.3 and CUDA 10.2, along with Linux-specific packages. That environment should not replace the tutorial's working Python 3.12 environment. The successful CPU core check suggests that a small modern environment is feasible, but it does not validate the legacy data and training scripts. The loader still contains the removed NumPy alias `np.int`. [Official setup](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/README.md), [environment](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/environment.yml), [segmentation code](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/utils/pose_utils.py)

The model constructs the default OpenPose graph inside each flow step. Passing a new input shape does not change its adjacency matrix. The existing headless option is also not our 16-joint skeleton. A real adapter must introduce an explicit layout and pass it through every flow block. For our numerical order `[11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]`, a documented candidate is the induced body graph with these undirected edges in original BlazePose indices:

```text
11-12, 11-13, 13-15, 12-14, 14-16,
11-23, 12-24, 23-24,
23-25, 25-27, 27-29, 29-31, 27-31,
24-26, 26-28, 28-30, 30-32, 28-32.
```

Add the 16 self connections and convert indices through the fixed order above. Preserve the declared adjacency normalization and hop settings. Do not pad to 18 with duplicate joints or invented face points. Check graph symmetry, reachability, normalized adjacency, gradient finiteness and a finite forward/inverse round trip. Using this graph is an explicit BlazePose adaptation, not an exact OpenPose benchmark reproduction. [Graph construction](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/models/STG_NF/graph.py), [flow graph use](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/models/STG_NF/model_pose.py)

The flow likelihood and its data-dependent normalization operate on every coordinate. They expose no observation mask or marginal likelihood over an observed subset. The confidence option in the trainer is not an invalid-coordinate exclusion mechanism. Replacing NaN with zero makes the flow model the filled observation, while dropping selected latent likelihood terms does not generally marginalize missing input coordinates because the flow mixes coordinates. [Likelihood and normalization](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/models/STG_NF/modules_pose.py), [training confidence handling](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/models/training.py)

## Missing observations are the deciding constraint

Our accepted windows need at least 75% valid four-frame tokens overall and 50% in each half. A valid token has finite XY for its joint in all four frames. Confidence below 0.5 creates missing coordinates. The model excludes invalid keys, targets and pooled contributions. A published baseline must state how it uses that same available information.

Multiplying a final likelihood term by zero does not undo the effect of a missing or filled value on other predicted terms. In SeeKer, a missing early keypoint may be part of later keypoints' conditioning sets. In STG-NF, missing inputs enter coupled transformations. Both need more than a loss mask.

The tempting complete-case fallback was checked against the saved historical cohort:

| Historical folder | Accepted windows | Accepted sources | Fully observed 32-frame windows | Sources with a fully observed window |
| --- | ---: | ---: | ---: | ---: |
| Normal | 482 | 28 | 11 | 5 |
| Parkinsons | 184 | 10 | 70 | 5 |
| Stroke | 779 | 15 | 143 | 9 |
| Cerebral palsy | 390 | 10 | 114 | 6 |
| Myopathic | 669 | 28 | 105 | 18 |
| Total | 2,504 | 91 | 443 | 43 |

The count selects `valid_token_fraction == 1` in the saved window index. The five remaining normal sources cannot support our four outer folds with five separate inner normal reference sources and a remaining normal fitting set. The new all-folder extraction must be audited separately, but adding abnormal folders does not by itself add normal sources. A complete-case result would also describe a strongly selected visibility population. It cannot stand in for the primary partially observed cohort.

There are three honest options:

1. **A fully observed secondary cohort.** Preserve the published density model and explicitly restrict all compared methods to the same complete cases. Proceed only if the final corpus supplies sufficient normal sources for a prespecified fitting and evaluation design. The historical count fails that gate. Shortening windows or lowering confidence thresholds would change the experiment and requires a separately declared sensitivity analysis.
2. **An observation-aware adaptation.** For example, a conditional predictor could read finite values plus explicit availability indicators and score observed targets. This is technically plausible, but changes SeeKer's conditioning model and requires missingness noninterference, calibration, coverage and nuisance tests. Label it a SeeKer-inspired adaptation until validated against the published method. It is not an exact marginal likelihood simply because missing targets are excluded.
3. **A later density-method study.** Proper integration over missing coordinates or a validated conditional-flow construction is a substantive extension. It is not a safe one-line repair before this deadline. For this submission, report the missing published comparator and keep claims limited to the matched internal controls.

Using low-confidence raw coordinates that our primary protocol excludes is another possible sensitivity analysis, but it changes the information supplied to the method. It cannot be presented as the same-observation comparison.

## Dependencies and a bounded validation plan

For an isolated SeeKer audit, direct imports require PyTorch, NumPy, SciPy, scikit-learn, `tqdm` and TensorBoard. STG-NF's model core requires PyTorch and NumPy; its surrounding scripts also use `tqdm`, TensorBoard, SciPy, scikit-learn, Pillow and Matplotlib. Most are already available in the tutorial environment, but `tqdm` and TensorBoard were absent during this review. These are observed import requirements, not a tested replacement lockfile. Create a separate `uv` environment and record resolved versions before a reproduction attempt. Avoid upgrading the running experiment environment to repair another repository.

The release licenses also differ: SeeKer is MIT, while STG-NF is distributed under Attribution-NonCommercial 4.0. Preserve source attribution and the relevant license if code is redistributed. This review did not redistribute either implementation. [SeeKer license](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/LICENSE), [STG-NF license](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/LICENSE)

The following is a planning estimate, not a measured completion schedule:

| Priority | Work package | Gate for moving forward |
| --- | --- | --- |
| First half day | Recompute completeness and source support on the final frozen twelve-folder corpus; choose the observation policy. | Sufficient independent source roles without changing the primary comparison after viewing results. Stop the complete-case route if normal support remains inadequate. |
| Next one to two days | Reproduce a small native-layout reference case in an isolated environment. Prefer STG-NF for immediate execution; use SeeKer if the inference alignment and released defects can be resolved against the paper. | Finite forward/backward passes, deterministic evaluation, correct score direction, stable checkpoint reload, and a reference result or independently checked analytical case. |
| Next one to two days | Add only the declared 16-joint layout and shared preprocessing, keeping a written change log. | Target-order or graph tests pass; no artificial joints; no silent missing-coordinate fill; no test-based checkpoint selection. |
| Remaining time | Fit normal-only folds and aggregate on the common eligible source set. | All source roles and exclusions are recorded, normal development diagnostics are adequate, and paired source-level intervals use the same metric as the point estimate. |

Executing and adapting a small model core within a week is plausible. **A validated published baseline on the full partially observed cohort is not currently a defensible promise for the November 20 deadline.** The missingness policy and scarce complete normal sources are the limiting evidence, even if code repairs are fast. The workshop date is verified on the [official ASTAD4 call](https://ismart.ece.mcgill.ca/astad4/); this review does not infer an unstated cutoff timezone.

For the present paper, prioritize the matched own/full/patch-local comparisons, local-coordinate target control, matched raw KL, ridge predictor and source-level uncertainty. If a published comparator fails the gates above, say so plainly. A supported study of predictive information can remain useful without claiming superiority over SeeKer, STG-NF or the wider literature. No current result should be called state of the art, and the absence of a runnable adapter does not diminish the prior-art relevance of either method.
