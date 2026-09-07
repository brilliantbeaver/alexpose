# Research-extension verification

Completed September 7, 2026. This record describes software verification and limited read-only analyses. It does not report completion of the proposed real-data training studies or an external evaluation.

## Checks completed

| Check | Result |
|---|---|
| Existing suite plus extension unit tests | 124 passed, including 51 new research-extension tests |
| Tutorial source/notebook consistency | All four generated notebooks match their editable sources |
| Canonical notebook format | Valid notebook files, with no saved execution counts or outputs |
| Fresh-kernel execution | All 29 code cells across notebooks 07–10 completed without errors |
| Inline graphics | Six PNG figures and seven SVG outputs rendered across the four notebooks |
| Visual review | Main information-path, masking, symmetry-comparison, and forecasting figures inspected |
| Original-source preservation | All 24 checked original notebook, core Python, and protocol files retain their initial contents |
| Local data availability | Read-only cohort/split loading succeeds for 625 clips from 93 sources |
| Forecast preparation | Read-only raw-pose preparation succeeds for 611 clips and 1,814 eligible clip/horizon examples from 93 sources |

The new helper package is outside `laterality/`, preserving the implementation fingerprint used by the completed checkpoints. No original notebook, protocol, governance status, trained checkpoint, or registered result was edited. The separate, untracked research note in the workspace was not part of this implementation and was left untouched.

## Scientific checks covered by tests

The tests exercise reflection of coordinates and masks, equal realized masking budgets, preservation of context, teacher gradient exclusion, matched initializations and source draws, training-only scaling/read-out fitting, and source-weighted aggregation. Feature-variation checks expose constant representations that would otherwise pass a consistency test.

Forecasting tests change future coordinates, visibility, and timestamps while retaining the prediction boundary. They verify unchanged context and predictions. A stronger source-isolation test changes held-out future values and verifies that fitted training parameters remain identical. The shuffled-future control now samples an eligible source uniformly before selecting a window, preventing long recordings from dominating its targets.

## Adversarial findings resolved

- Matching hidden-token counts was initially described too broadly as controlling prediction difficulty. The research plan now distinguishes the common budget from intrinsic differences between target regions.
- The shuffled-future control initially drew uniformly from examples, which differed from the source-balanced main training distribution. It now draws sources before windows, with a regression test for unequal source sizes.
- Editable forecasting settings now reject nonfinite values, invalid teacher momentum, negative variance weights, and invalid fitting parameters.
- Notebook figures initially depended on the kernel's default plotting backend. Every tutorial now explicitly enables inline rendering with SVG and PNG support, and the fresh-kernel checker validates the outputs.
- Claims about timestamp changes now specify that the affected observations must remain after the prediction boundary.

## What was and was not executed

Small synthetic training comparisons were executed to validate the new code. The processed-input target reconstruction was also run as a read-only exploratory analysis: 623 clips from 92 sources give direct source-balanced agreement of R² 0.218, correlation 0.652, and mean absolute error 0.041. These are agreement statistics between two measurement paths, not a learned-model score or a performance ceiling.

The new real-data masking grid, symmetry-learning pilot, and forecasting training were not run. No external dataset was downloaded or evaluated. Optional local-data cells remain disabled, and Notebook 06 remains the existing prerequisites check. The proposal for a later external evaluation is documented separately in [the external assessment](external_evaluation_assessment.md).

## Repeating the checks

```bash
.venv/bin/python neurips-laterality/scripts/verify_suite.py
.venv/bin/python neurips-laterality/scripts/verify_research_notebooks.py --execute-smoke --save-executed
```

Executed review copies and exported graphics are saved separately under `executed/research_extensions/`; the checked-in notebooks remain output-free. PyTorch's existing nested-tensor warning concerns an unused optimization, and Jupyter reports its local kernel transport configuration. Neither warning was an execution failure. No error output occurred in the validated notebooks.
