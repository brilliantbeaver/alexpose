# Iteration 02: evidence, masking design, and the next study

The next study asks whether gait anomaly detection can add information beyond motion amount, current posture, and recording quality. The [execution instruction](../../../../notes/aaai-worldmodels/05-iterate-02-INSTR.md) defines the research role, hypotheses, data contract, masking arms, calibration score, evaluation, notebook sequence, and review gates. Its M0–M4 design is authoritative; the [literature review](literature_and_masking.md) documents a broader set of alternatives and their precedents.

The [standalone draft02](../draft02/manuscript.pdf) explains a positive forecasting finding and a methodological contribution to anomaly evaluation. Bilateral forecasting reduces normal-source error by 33.5%, while a transition from current posture reduces it by 40.9%. The prespecified excess-error score has AUROC 0.793. A newly documented post hoc low-motion diagnostic has AUROC 0.806. The paired difference is -0.013, with a conditional 95% interval from -0.111 to 0.083. Thus, this evidence supports useful prediction and reference-aware evaluation, while an incremental learned anomaly advantage remains to be established.

## What was completed

| Artifact | What it does |
| --- | --- |
| [Research instruction](../../../../notes/aaai-worldmodels/05-iterate-02-INSTR.md) | Specifies a concrete next study with neurologically motivated questions and strong alternative explanations. |
| [Literature and masking review](literature_and_masking.md) | Verifies current primary work, identifies prior art, and distinguishes external datasets by the question they can answer. |
| [Pipeline and notebook review](pipeline_and_notebook_review.md) | Connects the scientific claims to the actual ingestion code and execution records. |
| [Notebook 16](../../16_iteration02_evidence_and_mask_design.ipynb) | Recomputes evidence and demonstrates exact-budget masks on real cached poses. It trains no model. |
| [Draft03](../draft02/README.md) | Provides readable Markdown and an AAAI-format PDF, vector figures, source tables, and a post hoc score audit. |
| [Current slides](../../slides/iteration02.html) | Explain the complete study and next experiments with editable vectors, tables, and flowcharts. A [vector PDF](../../slides/iteration02.pdf) is also available. |
| [Independent review](adversarial_review.md) | Records adversarial findings, corrections, checks, and the exact accepted artifact identities. |

The mask laboratory implements uniform, time-block, anatomical-tube, and motion-balanced priorities. It hides the same number of eligible tokens, preserves observed reference joints and latest anchors, and keeps missingness distinct from deliberate masking. The motion policy reserves part of its budget for uniform draws so that observations without a measurable motion pair remain selectable. Structured masks can truncate their final block to meet the budget. This is a tested design example, not evidence that masked models improve gait analysis.

The proposed calibrated detector, fitted masking comparison, notebooks 20–26, phase extensions, stochastic models, and external evaluations have not been executed in this revision. Existing extraction, fitting, and notebook records retain their original identities.

## Does GAVD provide enough generalization evidence?

**GAVD supports a heterogeneous public-video benchmark, but GAVD alone does not establish patient-disjoint, site-disjoint, or prospective clinical generalization.** The full local inventory contains 1,874 sequences and 348 source IDs. The selected five labels contain 666 sequences from 103 sources. The evaluated cohort has 2,815 windows from 456 sequences and 91 sources, including 29 normal sources. All 32 annotated normal sources are already within the selected scope.

Source IDs are grouping proxies, not verified participant identities. Public videos can contain repeated people, multiple people, mirrored uploads, and different recording tasks. Folder labels are annotations rather than independently confirmed diagnoses. Unequal retention and pose uncertainty can change the population being studied. More overlapping windows cannot create independent participants or overcome the normal-source ceiling.

Use the remaining GAVD labels for a separately declared transfer or specificity study. Do not classify exercise, pregnancy, or stylized walking as neurological disease by default, and do not acquire data until a favorable significance threshold is reached. The literature review identifies GPJATK for normal video and view validation, GAITEX for a mechanical perturbation study, and PhysioNet Parkinson gait for a separate clinical force-domain replication. These are different forms of evidence and cannot be pooled as an interchangeable clinical cohort.

All GAVD originals remain under `data/videos-gavd`. Reuse existing files in place; new downloads use `sources/<source_id>.*`, with derived previews under `previews/<source_id>/`. The review and design laboratory require no new downloads.

## Reproduce the revision

Use the existing Python 3.12 experiment environment and local frozen input artifacts. These commands run from the repository root:

```bash
experiments/aaai-worldmodels/.venv/bin/python experiments/aaai-worldmodels/scripts/build_draft03.py --compile --render
experiments/aaai-worldmodels/.venv/bin/python experiments/aaai-worldmodels/scripts/build_iteration02_notebook.py --execute
experiments/aaai-worldmodels/.venv/bin/python experiments/aaai-worldmodels/scripts/build_iteration02_slides.py
experiments/aaai-worldmodels/.venv/bin/python experiments/aaai-worldmodels/scripts/build_restart_readme.py
experiments/aaai-worldmodels/.venv/bin/python -m pytest -c experiments/aaai-worldmodels/pyproject.toml experiments/aaai-worldmodels/tests -q
```

The manuscript and slide generators use the existing local PDF/rendering tools; their READMEs and build records document those prerequisites. Editing a generated product requires editing its prose source or generator first. New model fitting needs a new run identity and the complete protocol in the execution instruction. Do not overwrite completed runs to reproduce this document revision.

The [notebook execution record](notebook_execution.json), [mask laboratory checks](mask_lab_validation.json), and [test record](test_execution.json) distinguish actual execution from proposed work. The independent acceptance record applies only to its listed final hashes. Human author information and submission decisions remain outside the completed research-document revision.
