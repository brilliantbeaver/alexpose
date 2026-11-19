"""Build the current study guide from completed v2 evidence, preserving history."""
from pathlib import Path
import json
import shutil

PROJECT=Path(__file__).resolve().parents[1]


def load(name):
    p=PROJECT/'artifacts/runs'/name/'results.json'
    value=json.loads(p.read_text())
    if value['status']!='complete':raise ValueError('Incomplete research run: '+name)
    return value


def build():
    core=load('reference-v2-core-v1');ext=load('reference-v2-transfer-v1')
    sensitivity=load('reference-v2-identity-sensitivity-v1')
    load('reference-v2-robustness-v1')
    normal={r['arm']:r['mse'] for r in core['normal_forecast'] if r['horizon']==8}
    primary=core['primary_anomaly'];transfer=ext['primary_anomaly']
    gain=next(r for r in core['forecast_contrasts'] if r['method']=='ridge_posture' and r['reference']=='persistence')
    previous=PROJECT/'README.before_reference_v2.md'
    if not previous.exists():shutil.copy2(PROJECT/'README.md',previous)
    text=f'''# A reference-controlled benchmark for gait observation models

This study asks whether a model recognizes departures from normal movement organization beyond movement amount, current posture, camera view, and pose quality. The completed method is **`medwm-reference-masking-v2`**. It tests compact normal-only forecasting, seven controlled training-mask policies, a conditional normal-reference anomaly score, and frozen transfer to a separate participant-coded video dataset.

The clearest positive capability is short-horizon forecasting. A current-posture transition reduces normal-source error by **{100*gain['relative_reduction']:.1f}%** relative to repeating the latest pose. The controls also resolve a practical evaluation question: useful forecasting does not establish incremental anomaly detection. The locked conditional score does not improve on the low-motion baseline internally or externally. Structured masks do not improve on uniform masking under the fixed coordinate objective. A separately reviewed post hoc diagnostic also compares two untuned normal-training mean priors, which remain weaker forecasts than current posture. The paper reports these findings together.

Start with the [standalone draft03 PDF](docs/draft03/manuscript.pdf), [plain-language manuscript](docs/draft03/manuscript.md), [manuscript evidence and reproduction record](docs/draft03/README.md), and [new illustrated slides](slides/reference_v2.html), also available as a [vector PDF](slides/reference_v2.pdf). The [protocol and study guide](docs/reference_v2/README.md) index the frozen design, data audits, independent review, and notebook execution record. The [research instruction](../../notes/aaai-worldmodels/05-iterate-02-INSTR.md) remains the design brief, rather than evidence that every proposed extension was executed.

## What actually ran

| Experiment | Independent groups and observation counts | Main measured result |
| --- | --- | --- |
| Internal forecasting and masks | 91 GAVD source videos, including 29 normal; 456 sequences and 2,815 windows | Posture MSE {normal['ridge_posture']:.5f}; persistence {normal['persistence']:.5f}; posture plus velocity {normal['ridge_posture_velocity']:.5f}. |
| Conditional anomaly score | Same five grouped outer folds | Pair-weighted within-fold AUROC {primary['candidate_auroc']:.3f} versus low motion {primary['reference_auroc']:.3f}; paired difference {primary['estimate']:.3f} [{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}]. |
| Frozen external transfer | 90 coded participants: 27 normal, 49 osteoarthritis, 14 Parkinson's; 168 recordings and 773 windows | AUROC {transfer['candidate_auroc']:.3f} versus low motion {transfer['reference_auroc']:.3f}; paired difference {transfer['estimate']:.3f} [{transfer['ci95'][0]:.3f}, {transfer['ci95'][1]:.3f}]. |
| External identity sensitivity | 89 coded participants after a fixed whole-participant exclusion | The same remaining scores are evaluated. Exclusion removes the only retained severe-Parkinson's participant. |
| Robustness and nuisance checks | Common targets under dropout, noise and opposite-leg shifts; view/motion/quality matching retains 40 sources, with 38 in estimable folds | No structured-mask forecasting advantage over uniform is established. Matching changes the population and does not establish causal control. |

The primary error averages all eight future offsets from nominal 0.1 to 0.8 seconds. Coordinates are body-relative projected articulation in prefix-torso units, and MSE has squared units. The sixteen-observation prefix spans 1.5 seconds between endpoints. The windows overlap and do not represent independent patients. Paired 95% group-bootstrap intervals condition on saved fits; optimization seeds and repeated partitions do not create independent cohorts.

All GAVD outcomes remain exploratory because earlier outcomes influenced this design. The post hoc low-motion direction from the earlier audit becomes a declared baseline for v2; its original status is preserved. The excess-error association remains visible alongside both directions of persistence and the other named controls. Supervised historical JEPA probes and condition adaptation use different label access and cohorts, so their headline scores are not a matched leaderboard.

## Dataset coverage and generalizability

The all-label annotation inventory has **1,874 sequences, 458,116 rows, and 348 source IDs**. The pinned `data/gavd/source/` copy is provenance and is not counted again. Five-label scope contains 666 sequences from 103 sources. Media are available for 652 sequences from 98 sources; preserving two source-reference exclusions leaves 641 candidates from 96 sources. Fixed observation and common-target gates yield the internal cohort. All 32 annotated normal source IDs are already inside the five-label scope, and 29 survive. The other labels cannot increase independent normal support.

**GAVD supports a heterogeneous public-video benchmark, but GAVD alone does not establish patient-disjoint, site-disjoint, or prospective clinical generalization.** Source IDs are not patient IDs, labels are not independently confirmed diagnoses, and even the full inventory has limited normal-source support. A source-relation graph records exact duplicates and known links; unknown repeated participants and reencoded copies remain possible. More windows cannot repair that limitation. Exercise, pregnancy, stylized movement, prosthetic use, and intoxication retain their actual meanings rather than becoming neurological diagnoses.

The external [KOA-PD-NM owner dataset](https://data.mendeley.com/datasets/44pfnysy89/1) adds a separate participant-coded transfer test. Its 189 local originals represent 95 coded participants. Two corrupt files leave 187 decodable recordings from 94 people before quality gates, and 90 people remain afterward. IDs include severity group because numbering restarts across groups. `PD_SV_003` is absent locally. Duplicate caches are counted once. The GAVD-derived clips under `data/videos-ms/gavd-sequences` are the same sources and are not independent external data.

External age, native FPS, camera conditions, markers, clothing, person selection, and retention can affect the scores. GAVD source calibration and external participant-averaged observations have different measurement units. The frozen pipeline performs no target-domain fitting or normal calibration. Byte hashes exclude exact file overlap but do not prove clinical participant independence across datasets. The owner page declares CC BY 4.0; the bundled README asks research/noncommercial use. This research does not redistribute the videos. Read the [compatibility contract](docs/reference_v2/external_contract.md) and [inventory audit](docs/reference_v2/data_external_summary.json).

## Ambient ingestion and future isolation

The shared `ambient/` package owns video and pose interfaces. The experiment adapter already uses `ambient.utils.youtube_cache.download_youtube_urls`. All GAVD originals stay under `data/videos-gavd`. Verified existing condition-folder paths are reused in place; new originals belong in `data/videos-gavd/sources/<source_id>.<extension>`, with previews under `previews/<source_id>/`. No new video download was required in this iteration. We do not duplicate originals across conditions or include bulk media in Git.

Sequential decoding calls `SequenceKeypointExtractor.extract_from_image` on RGB images. GAVD frames are one-based, decoded indices zero-based, and conversion occurs exactly once. The sixteen BlazePose indices are `(11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32)`. Distal targets are `(25,27,29,31)` and `(26,28,30,32)`. Confidence, original missingness, dimensions, actual times, source and sequence IDs, video hashes, and weight hashes remain explicit. COCO indices cannot be substituted by position.

Confidence at least 0.5 and 70% observed support are study gates. Per-frame pelvis centering uses both hips; torso scale uses the prefix only. Predictions use fixed future queries, without future coordinates, visibility, or confidence. Future observation support enters loss and offline scoring after the suffix arrives. Tests poison future arrays and masks while requiring unchanged prefix inputs, prefix normalization, anchors, masks, and predictions. Future-relative target normalization and evaluation support may change. A prefix-only early warning score would be a different endpoint.

## Reproduce and inspect

Discover this checkout's repository root and use the experiment's Python 3.12 environment. From the repository root:

```bash
uv sync --project experiments/aaai-worldmodels --python 3.12
uv run --project experiments/aaai-worldmodels jupyter lab experiments/aaai-worldmodels
```

Full replay needs the pinned local media, annotations, pose weights, and immutable prepared runs. Missing artifacts cause an explicit error. From `experiments/aaai-worldmodels`, reproduce reports without refitting:

```bash
.venv/bin/python scripts/build_next_draft03.py --compile --render
.venv/bin/python scripts/build_reference_v2_notebooks.py --execute
.venv/bin/python scripts/build_reference_v2_slides.py
.venv/bin/python scripts/build_reference_v2_readme.py
.venv/bin/python -m pytest -c pyproject.toml tests -q
```

The manuscript uses official AAAI style and pdfTeX. Its builder accepts `--pdftex` and `--render-python` for equivalent installations. The portable paths in the build record describe this execution rather than a required username. The old `scripts/build_draft03.py` remains a compatibility command for preserved draft02 and must not be used to build the new draft. Historical README/slide builders can restore their historical entry points, so use the v2 builders for the current study.

The full experiment commands are `python -m worldmodels.v2_runner`, `python -m worldmodels.v2_external`, `python -m worldmodels.v2_transfer`, and `python -m worldmodels.v2_robustness`. Completed output directories are immutable; a new fit requires a new identity, matching protocol/code freeze, and fresh output path. Exact commands, seeds, package versions, fit memberships, checkpoint hashes, and per-source scores are retained. Arrays and checkpoints stay local under ignored run directories; reviewable source-level tables and figures are exported with the manuscript. The post hoc mean-prior and numerical-floor sensitivity analyses have their own declarations and identities; neither replaces the frozen primary endpoint. A help/import check is not counted as a full experiment.

## Progressive notebooks

| Notebook | What it executes and explains |
| --- | --- |
| [20: Cohort and coverage](20_cohort_identity_and_coverage.ipynb) | Rebuilds all-label inventory, attrition, grouping, and external feasibility. |
| [21: Reference scores](21_reference_scores_and_calibration.ipynb) | Recomputes source scores and explains score direction, disjoint calibration, and tail resolution. |
| [22: Mask laboratory](22_structured_masking_laboratory.ipynb) | Calls Ambient on one real frame, draws masks, checks exact counts and future isolation. |
| [23: Models and fitting](23_compact_models_and_fitting.ipynb) | Fits a labeled teaching ridge model and audits full-run fitting, sampling, and model health. |
| [24: Evaluation and transfer](24_source_and_external_evaluation.ipynb) | Rebuilds paired independent-group endpoints and frozen external transfer evidence. |
| [25: Robustness and interpretation](25_robustness_and_movement_interpretation.ipynb) | Loads actual stress tests, matching, views, source diagnostics, and condition profiles. |
| [26: Manuscript evidence](26_manuscript_evidence_and_claims.ipynb) | Exposes all candidates and maps manuscript claims to their measured evidence. |

Each notebook runs from its own clean kernel. Executed copies and exact input/output hashes are indexed by [the execution record](docs/reference_v2/notebook_execution.json). Notebooks 00–16, draft01, draft02, and their run identities remain preserved. The new study defers latent objectives, learned masks, phase conditioning, and stochastic futures because their declared prerequisites are unmet. Deferral is recorded in [the run ledger](docs/reference_v2/run_record.json), with no invented gains.

The [independent review](docs/reference_v2/independent_review.md) records invariant checks, full score reconstruction, numerical precision findings, publication issues, and their dispositions. Final acceptance applies only to its recorded hashes. Author names, affiliations, disclosures, submission choices, and data-release rights still require author completion before an actual submission. No workshop submission is performed here.
'''
    (PROJECT/'README.md').write_text(text)
    (PROJECT/'docs/reference_v2/README.md').write_text('''# Reference-controlled gait study, version 2

The completed research question separates useful future prediction from anomaly ranking beyond movement amount, posture, view, and observation quality. Read [the current project guide](../../README.md), [draft03](../draft03/README.md), and [the slides](../../slides/reference_v2.html) for the measured findings.

The [immutable protocol](protocol.md), [accepted freeze](protocol_freeze.json), [model contract](model_contract.md), [evaluation contract](evaluation_contract.md), and [external contract](external_contract.md) specify the experiment. The [run record](run_record.json) declares ownership, resources, results, and deferred extension gates. [Primary-source notes](primary_sources.json) distinguish precedent from contributions.

The [data summary](data_summary.md), [all-label audit](data_audit.json), [external inventory](data_external_summary.json), [preparation audit](data_external_preparation_audit.json), and [visual review](data_external_visual_review.json) record acquisition and attrition. The pre-score [identity sensitivity](external_identity_sensitivity_protocol.json) keeps its purpose and reduced severity coverage explicit.

The [notebook execution record](notebook_execution.json) identifies clean-kernel executions. [Independent review](independent_review.md) and the final acceptance record describe reconstruction, visual QA, issues, remedies, and exact accepted hashes. Large media, pose arrays, and checkpoints remain local; the manuscript exports complete source and participant scores for review.

**GAVD supports a heterogeneous public-video benchmark, but GAVD alone does not establish patient-disjoint, site-disjoint, or prospective clinical generalization.** Source IDs are not patient IDs, folder labels are not independently confirmed diagnoses for this analysis, and the normal-source ceiling remains small. External coded-participant transfer adds evidence under a documented acquisition shift without establishing prospective clinical validity.
''')
    print('Built README.md and docs/reference_v2/README.md from completed measured runs.')


if __name__=='__main__':build()
