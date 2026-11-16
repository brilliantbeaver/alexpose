# Independent method and visual review

Reviewer: the literature specialist, separate from the model, data, notebook, runner, and figure authors. This reviewer wrote `literature.md` and `workshop.md`; those two documents are not independently approved by this report. Review date: November 15, 2026, workspace date.

## Scope and actual checks

Read `model.py`, `masking.py`, `training.py`, `pose.py`, `evaluation.py`, `workflow.py`, `experiment.py`, and the figure generator. Visually opened the rendered PNG for each of the eight SVG diagrams. Read the HTML slide source, but did not claim browser-rendered slide inspection. No clinical training run or published benchmark was reproduced by this review.

Two in-memory synthetic training checks used four distinct source IDs, two update steps, four frames, width eight, and deterministic versus latent-plus-position-jitter models. Both produced finite losses, left every teacher parameter without gradients, and rejected posterior conditioning in evaluation mode. These checks used the project `.venv` and wrote no checkpoint. They establish a small execution property only.

An adversarial inference check perturbed every future coordinate by 1,000 and separately removed future observation validity. Following the implementation correction, both changes produced a maximum absolute prediction difference of **0.0**. This checks the model boundary on normalized inputs; it does not establish the absence of leakage in every possible preprocessing pipeline.

## Findings and resolutions

| Finding | Severity | Resolution or acceptance condition |
| --- | --- | --- |
| Stochastic models originally received an additional MSE loss unavailable to deterministic controls. | High for mechanism claims. | Model author made the MSE weight available to both variants. Any factorial experiment must explicitly match this weight, as well as masks and budgets. |
| Forecast decoder originally received future validity as a key-padding mask. | High for future-only claims. | Model author removed that outcome-dependent decoder mask. The adversarial prediction check now passes for changed future validity. |
| Stochastic sampling changed the global RNG stream used for later masks. | Medium for matched ablations. | Training now uses a dedicated seeded mask generator. Shared seeds alone were insufficient. |
| Downstream `encode` originally used the online encoder without matching the cited skeleton method. | Medium for fidelity. | Default now uses the EMA teacher, with explicit encoder selection. |
| Positional extension perturbs temporal positions jointly in the view and teacher, rather than reproducing StoP's tied projection. | Medium for attribution. | Accept as a declared shared-temporal-jitter adaptation. Do not identify it as a faithful StoP implementation or a proven uncertainty estimator. |
| `jepa.png` put the EMA text on its vertical arrow. | Low, visual. | Figure author moved the label off the arrow. Independent reinspection of the new PNG confirms separation. |
| `stochasticity.png` labeled position noise “Coordinate jitter.” | Medium, teaching accuracy. | Figure author changed the wording to temporal position jitter. Independent reinspection confirms the corrected label is readable. |
| Protocol primary endpoint is binary balanced accuracy, while initial report/bootstrap code exposed five-class balanced accuracy and macro-F1. | High for reporting consistency. | The revised classifier report and bootstrap implement binary balanced accuracy. The runner uses this same endpoint for validation selection. |
| Run status was inferred from a directory-name substring. | Medium, provenance. | Run status now derives from preparation mode and frame limits. |
| Forecast error lacked a persistence baseline. | High for predictive-skill claims. | Runner now compares JEPA and persistence on the identical future-joint subset with past support, then aggregates sequences and sources. |
| Chosen supervised estimators were discarded after reporting. | Medium, tutorial completeness. | The runner persists selected classifiers and encoder checkpoints. The separate inference command checks code, versions, feature order, and artifact identities. |
| Full-image first-pose selection ignored the annotated person's box. | High for cohort validity. | Retained as an explicit pilot limitation in README, data protocol, results metadata, and inference output. Person verification remains a paper-evidence gate. |
| Normalization occurs before random masking. | Medium, task definition. | Prefix-based scaling protects future-only inputs, but masked-completion targets may have contributed to pre-normalization. README and notebook01 now explicitly describe prediction in a pre-normalized articulation representation and avoid claiming raw-coordinate masking invariance. |

A further inference test evaluated five windows in batches of two with three prior samples. Sample shape `(3, 5, 4, 16, 2)`, finite errors, and fixed-seed repeatability all passed. The implementation now bounds attention memory by inference batch size. Changing batch size changes the declared Monte Carlo random stream.

Effective sample rates in the window metadata now use actual frame-index increments. This corrects the assumption that every source annotation row is adjacent in time. Nominal-frame-rate limitations remain documented.

## Accepted visual properties

The cohort, source-split, adaptation, comparison, uniform-mask, and forecast figures were legible in their PNG renders. The reviewer found no overlapping text or arrows in these six diagrams. Original BlazePose indices are ordered correctly in the mask diagram. The cohort explicitly distinguishes 666 sequences from 103 source videos and avoids calling the sources unique patients. The forecast diagram distinguishes bidirectional completion from a withheld suffix.

The comparison diagram's sequence-then-source aggregation matches the implementation when `sequence_ids` is supplied. It would not match a caller that omitted sequence identities. The figure describes the intended controlled evaluation; it does not supply results.

## Scientific limits retained

The finite-sample 5% threshold can be infinite with fewer than 19 normal calibration sources. Reporting this as an unavailable finite operating point is correct. It must not be silently replaced by a favorable threshold selected on abnormal test data.

The staged and mixed adaptation arms initially match update count but not per-condition exposure. The runner explicitly lists this limitation. Their comparison can be exploratory; it does not isolate curriculum order.

The review supports tutorial implementation checks. A successful empirical paper still requires audited person selection, final cohort support, complete matched runs, declared metrics, source-level uncertainty, and honest reporting of incomplete experiments.

## Final integrated verification

The final review inspected all eight generated notebooks and their generator. Every current notebook source matches its executed copy cell-for-cell; all 73 code cells have execution counts and no error outputs. The recorded SHA256 for each source in `artifacts/executed/execution.json` also matches the current file. These executions include synthetic model exercises and actual inventory access, with optional media/extraction disabled by default. They do not establish completion of stochastic clinical experiments.

Notebook05 now applies the same latent-MSE weight of 1.0 to deterministic and stochastic illustration arms and explains the hybrid objective. All notebooks expose optional real-video context. Capstone paths refer to the expanded v2 pilot. README links now use the actual notebook filenames. The only unresolved local documentation link at review time was `docs/run_record.json`, which the coordinator is preparing.

The updated JEPA and stochasticity diagrams were reopened visually and accepted. Eight diagrams have independent rendered inspection; the HTML deck still has source inspection only in this review. Revised runner code uses the primary binary endpoint, persists chosen classifiers, derives pilot status from preparation metadata, and compares forecast errors to persistence on matched future targets after sequence/source aggregation.

**Acceptance:** no blocking defects remain for the reviewed tutorial/model scope. This is not acceptance of clinical generalization or of an empirical manuscript. The v2 real-data comparison was still running/preparing at this review; its results need separate final verification. The normal-reference calibration limit, first-person detection limitation, and missing external/repeated-seed validation remain explicit.

These composite SHA256 identities pin the reviewed snapshot. To reproduce each value, sort its listed paths, join `relative/path + " " + SHA256(file_bytes)` lines using a newline with no trailing newline, and hash that UTF-8 text. Model/training includes `model.py`, `masking.py`, `training.py`; runner/evaluation includes `experiment.py`, `workflow.py`, `evaluation.py`. Notebook sources are the eight root `.ipynb` files. Rendered diagrams are the eight `.png` files under `images/`. Any later material edit reopens its corresponding check.

| Reviewed set | Files | Composite SHA256 |
| --- | ---: | --- |
| Model and training | 3 | `9883e4f5c2297a99c012bf3081f254e4b509f9d59a155275dca2091981eae893` |
| Runner and evaluation | 3 | `005b28e4460ac9f484ee2ad5b8618dfd7d0a97e0a8a760d7ad2259e9e9b85784` |
| Notebook sources | 8 | `b3c15c728b83393dea72153936cdf78599afb3d7235c20d2d320283189f31eb1` |
| Rendered diagrams | 8 | `13fdd096f1f87584696942ac32562c7ba7ffdb78761f5b57b1adc6451dfadac3` |
| Notebook generator | 1 | `82eb9356c594728bddf8e6e9f15e87b45576b985a9374af444296735c198f53d` |
