# Independent review of the iteration 02 research package

The reviewer did not author the instruction, literature report, manuscript, slide generator, or notebook implementation. This review assesses the concrete artifacts and their evidence, rather than the assigned specialist roles. It does not guarantee acceptance by the workshop or establish clinical validity.

## Evidence checks completed

The reviewer independently checked all 94 entries in `artifacts/runs/restart-dynamics-v1/artifact_hashes.json`. Every hash matched. Direct source-pair comparisons reproduced excess-error AUROC 0.7931034483 and negative-persistence AUROC 0.8064516129. The latter remains explicitly post hoc. The five within-fold AUROCs and their positive-negative-pair-weighted averages reproduced: 0.7743732591 for excess error and 0.8105849582 for low motion. No cross-fold pairs enter these averages.

Independent arithmetic from `source_scores.csv` reproduced normal-source MSE 0.1212617936 for persistence, 0.0806318753 for bilateral prediction, and 0.0716359330 for current-posture prediction. Their relative reductions are 33.50595% and 40.92456%. A separate pair-comparison implementation reproduced the 5,000-draw paired interval endpoints to numerical precision: [-0.1112347052, 0.0828837597] pooled and [-0.1448467967, 0.0668523677] within fold. The latter preserves fold and binary-label cell counts. These are conditional descriptions of saved overlapping fits, as the manuscript states.

The original extraction and forecasting invariant tests were rerun with this command:

```bash
PYTHONDONTWRITEBYTECODE=1 experiments/aaai-worldmodels/.venv/bin/python -m pytest \
  -c experiments/aaai-worldmodels/pyproject.toml \
  experiments/aaai-worldmodels/tests/test_restart_data.py \
  experiments/aaai-worldmodels/tests/test_restart_experiment.py \
  -q -p no:cacheprovider
```

The result was **14 passed in 1.46 seconds**. This verifies software invariants covered by those tests. It is not a new empirical study or independent validation of diagnoses and participant identity.

Primary-source checks confirmed that [SLiM v4](https://arxiv.org/abs/2603.10648v4) already introduces semantic tube masking and that [ASMa](https://arxiv.org/abs/2602.06251v1) combines complementary movement and graph-degree masking. Indexed text of the [FSGait conference paper](https://openaccess.thecvf.com/content/ACCV2024/papers/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.pdf) confirms normal-only reconstruction, prediction, and a memory bank. The direct CVF fetch returned an access error during this independent check, so that check used the indexed primary-paper text. These precedents rule out first-method claims for those components.

The [GPJATK owner record](https://zenodo.org/records/15130800) confirms 32 coded participants, 166 sequences, four 25 Hz video streams, 100 Hz Vicon measurements, and scientific-use terms. The [PhysioNet dataset](https://physionet.org/content/gaitpdb/1.0.0/) confirms 93 Parkinson's participants, 73 controls, and 16 plantar-force channels at 100 Hz. Neither source justifies treating all external observations as compatible GAVD video samples. The literature report correctly separates their roles.

## Findings and required resolutions

| ID | Severity | Finding | Requested resolution |
| --- | --- | --- | --- |
| R1 | Moderate | The instruction and literature memo initially named different primary mask rates, 40% and 25%, and used different arm identifiers. | Establish an authoritative execution specification and map the advisory alternatives to it. |
| R2 | Moderate | The proposed two-sided formula used the undefined term residual. A signed coordinate residual, squared forecast error, and excess error have different meanings. | Declare the scalar quantity, calibration unit, aggregation order, and primary nuisance comparator before fitting. |
| R3 | Moderate | The initial slide 28 title equated two-sided calibration with detecting rigidity and disorder. | Replace clinical interpretation with observed distribution tails and define the quantity shown. |
| R4 | Moderate | The initial mask illustration used unequal hidden-cell budgets and hid the latest target anchors, contrary to the proposed contract. | Match schematic counts, preserve anchors, and identify the anatomical grouping. |
| R5 | Minor | The source-fold matrix had unlabeled columns and could imply that abnormal sources fit the model. | Label source groups and specify normal-only fitting and inner validation. |
| R6 | Moderate | The first manuscript render left a dangling prose line on an otherwise sparse fourth body page, and appendix floats preceded their section heading. | Rebalance text and figures and keep appendix reading order stable without changing official typography or margins. |
| R7 | Moderate | Draft03 omitted the exact distal-target support gate and the particularly relevant FSGait precedent. Its figure captions lacked full source counts and normalized error units. | Add the reproducibility contract, cite FSGait, and make captions self-contained. |
| R8 | Moderate | The initial motion-balanced prototype always placed tokens without a measured adjacent motion pair last, giving them zero selection probability when measured tokens filled the budget. | Reserve a uniform component of the exact budget so every eligible observation can be sampled. |
| R9 | Minor | The instruction described all paths as repository-relative while using project-relative `docs/` and `worldmodels/` paths. | Define project-relative and repository-relative path conventions explicitly. |
| R10 | Minor | The first evidence-notebook plot placed the within-fold legend over the x-axis title. | Separate the legend, axis title, and footer in the rendered figure. |
| R11 | Moderate | The current root README still recommended the historical publishing scripts. The historical slide script unconditionally replaces `tutorial.html` with the restart deck. | Document the new draft03, iteration02 notebook, and iteration02 slide generators as the current reproduction path. |

All findings were sent to the respective authors. R1 is resolved by an authoritative 40% protocol and a literature-to-protocol arm mapping. R2 is resolved by an explicitly defined source-level log-MSE residual conditioned on prefix motion and missingness, a nested out-of-fit calibration requirement, and the fixed negative-persistence comparator. The proposal states that insufficient independent support can make calibration infeasible. R3 through R5 are resolved in the revised slides: tails refer to prediction difficulty, schematic masks hide the same 24 cells and preserve the latest anchor, and folds identify source groups and normal-only fits. R6 and R7 are resolved in the final six-page manuscript, whose body occupies three pages. R8 is resolved by reserving 20% of the mask budget for uniform sampling. R9 is resolved by defining PROJECT and both path conventions. R10 is resolved in the final notebook figure, where the axis title, legend, and footer occupy separate lines. R11 is resolved in the README source generator and regenerated README, which now reproduce the current artifacts and identify the earlier builders as historical.

The reviewer reran `tests/test_mask_design.py` after R8 was fixed: **7 passed in 0.27 seconds**. A separate structural check applied all four policies to every one of the 2,818 prepared real windows and verified equal hidden counts, no hidden protected anchors or reference joints, and no hidden naturally missing observations. That structural check used nominal times and therefore does not validate the motion-ranking behavior under actual decoder intervals. The module remains a teaching prototype, not a trained or clinically validated method.

## Scientific boundaries retained

The strongest supported contribution is a controlled, reproducible observation-model and anomaly-score benchmark. The favorable excess-error association does not demonstrate incremental anomaly value beyond low movement. The current-posture transition is stronger than bilateral history, so the evidence does not establish the need for a more complicated temporal model. These are limits on the claim, not reasons to hide measured forecasting gains.

GAVD source grouping is useful but cannot prove participant or site independence. Expanding the remaining labels does not add new annotated normal source IDs. The proposal correctly treats masking, calibration, phase estimation, and external validation as future experiments. No review acceptance should be read as completion of those experiments.

## Visual review scope

The reviewer inspected the initial manuscript and 35-slide contact sheets, then full-size renders of the main evidence charts, cohort flow, source-fold matrix, mask patterns, and calibration illustration. The initial issues above were visible in the actual exports, despite the slide layout script reporting zero errors. A clean automated layout report is therefore only one part of acceptance.

After revision, the reviewer inspected all six final manuscript pages at full size, including the mathematical identity, bibliography, complete appendix tables, support contract, and scatter-plot units. The final manuscript has no dangling body line or appendix-heading displacement. The accepted manuscript PDF hash is `5d54db8359941171c3f739703aa68221e9f8125fc9e85be80d124ed3e61087d2`. The revised deck has 36 slides because it adds a separate calibration-flow diagram. Full-size rechecks of slides 15, 23, 28, and 29 verify the requested corrections. Final integrated identities are recorded in `acceptance.json`.

## Notebook and integration acceptance

The final notebook generator, source notebook, executed notebook, input hashes, and figure hashes agree with `notebook_execution.json`. All four code cells executed without an error. The notebook reads actual decoder timestamps and records 364 passing mask-policy checks across one retained window from each of 91 sources, with zero skips. It independently recomputes point estimates while clearly identifying intervals read from the completed post hoc audit. The reviewer inspected its code and both final figures. The visible-input isolation demonstration is appropriately distinguished from eliminating information carried by an adaptive mask pattern.

The main README, iteration02 README, pipeline review, slide README, and draft03 README consistently separate measured forecasting results, post hoc diagnostics, executed mask construction, and proposed fitting. The current reproduction commands point to the new artifacts. The complete experiment test record reports 64 passing tests; the independent reviewer separately ran the 14 original and seven new focused tests described above. All accepted draft01 PDF, Markdown, and LaTeX hashes remain unchanged.

The current research-document revision is accepted with all eleven findings resolved. This acceptance covers the exact hashes in `acceptance.json`. It does not mean that the proposed calibrated detector, masking comparison, external evaluation, or notebooks 20–26 have run. No remaining scientific or visual blocker was found within this scope. Unverified participant identity, clinical diagnoses, external validity, small normal-source support, and conditional uncertainty remain explicit limitations. Human authorship details and any eventual submission are separate decisions.
