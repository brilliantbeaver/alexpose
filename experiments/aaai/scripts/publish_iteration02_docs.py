"""Publish active documentation from the shared evidence record."""
from pathlib import Path
import json,shutil
BASE=Path(__file__).resolve().parents[1];PAPER=BASE/'docs/draft02';DOC=BASE/'docs/iteration02'
e=json.loads((PAPER/'evidence.json').read_text());run=e['protocol_hash'];ext=e['external_protocol_hash']
def auc(endpoint,method):return e['main'][endpoint][method]['weighted_auroc']
def interval(v):return f"[{v[0]:.3f}, {v[1]:.3f}]"
gap=e['primary_difference'];egap=e['external']['primary_difference'];c=e['cohort']['retained']
notebooks=[('12_observation_population_and_video_context','Freeze the fresh population, inspect video and pose, and verify source roles.'),
('13_own_leg_and_opposite_leg_prediction','Understand matched masks, permitted inputs, retained prediction and normal-data adequacy.'),
('14_matched_distal_leg_timing_controls','Construct fixed-answer timing edits and see why sparse paired support limits interpretation.'),
('15_local_and_relational_anomaly_scores','Calculate M, R and S, preserve aggregation order, and inspect reference instability.'),
('16_baselines_and_source_level_evaluation','Compare frozen endpoints, controls and the separate five-class classifiers.'),
('17_failure_analysis_and_manuscript_decision','Inspect real failures, the independent replication, scientific gates and justified claims.')]
nav='\n'.join(f'| [{name.split("_")[0]}]({name}.ipynb) | {text} |' for name,text in notebooks)
if not (BASE/'README.before_iteration02.md').exists():shutil.copy2(BASE/'README.md',BASE/'README.before_iteration02.md')
readme=f'''# Useful prediction does not automatically improve gait anomaly detection

This package now contains an executed study of a precise question: **after a leg's own visible movement is accounted for, does the opposite leg add useful prediction information, and does that information improve detection of unusual gait?** The new study preserves a local predictor, adds an own-leg capacity control, compares three motion-independent masking policies, and checks the observation support needed for a timing interpretation.

The answer is qualified. All three policies passed the planned internal normal-data prediction gate. The unchanged uniform task still has uncertain outer-normal gains for U and A. The fixed combined anomaly score performed worse than local movement alone. A separate laboratory study of178 participants reproduced the detection loss despite clear held-out normal prediction improvement. The GAVD timing diagnostic failed its paired-support adequacy gate, so it cannot establish a reliable coordination measurement.

The [standalone manuscript in draft03](docs/draft03/README.md), **Controlled Cross-Leg Prediction for Gait Anomaly Detection**, presents the controlled prediction method, measured completion gains and independent participant evaluation. It explains how prediction adequacy, observation support and score integration determine the interpretation of relational gait evidence.

Start with [the six new notebooks](#tutorials), [verified results](docs/iteration02/results.md), [the draft02 manuscript](docs/draft02/manuscript.pdf), and [submission readiness](docs/draft02/submission_readiness.md). The [slides](slides/tutorial.pdf) and [offline viewer](slides/index.html) explain the complete study in plain language. Each slide has detailed [speaker notes](slides/tutorial.md), and the viewer provides individual light/dark controls with saved preferences. A [score guide](slides/score_guide.md) explains movement M, cross-leg R and their fixed average S. The [draft01 manuscript](docs/draft01/manuscript.pdf), historical runs, and [previous README snapshot](README.before_iteration02.md) remain preserved.

![The implemented measurement workflow separates biological motivation from evidence.](docs/draft02/figures/method.svg)

## What actually ran

| Evaluation | Independent evaluation unit | Local M AUROC | Fixed S AUROC | S minus M, conditional95% interval |
| --- | --- | ---: | ---: | --- |
| GAVD focal primary |90 source recordings,28 normal |{auc('focal','M'):.3f}|{auc('focal','S'):.3f}|{gap['observed']:.3f}, {interval(gap['conditional_95_interval'])}|
| GAVD broad secondary |304 source recordings,28 normal |{auc('broad','M'):.3f}|{auc('broad','S'):.3f}|{e['broad_difference']['observed']:.3f}, {interval(e['broad_difference']['conditional_95_interval'])}|
| Independent laboratory replication |178 participants,128 healthy |{e['external']['metrics']['M']['weighted_auroc']:.3f}|{e['external']['metrics']['S']['weighted_auroc']:.3f}|{egap['observed']:.3f}, {interval(egap['conditional_95_interval'])}|

These are weighted means of within-fold AUROCs, not pooled cross-model ranks. Intervals resample sources or participants while retaining fitted models, folds and the selected protocol. They do not account for training uncertainty or the prior exploratory redesign. The laboratory analysis uses simultaneous six-angle signals and independently fitted models. It is not transfer of video weights, a BlazePose mapping, or validation of all four condition families. The age and sex matched sensitivity also favors M over the fixed fusion.

The primary GAVD endpoint is normal versus Parkinsons, stroke, cerebral palsy and myopathic folders. Only `normal` is normal. Every other folder remains abnormal in the required broad secondary endpoint. Stroke is one category. The focal primary was frozen after draft01 exploration, so it is an explicitly exploratory redesign. It does not replace draft01's all-folder primary.

![The fixed fusion loses ranking performance in both measurement systems.](docs/draft02/figures/detection_difference.svg)

## Tutorials

Each notebook repeats its setup and loads an exact verified real-data run. A fresh kernel can execute it independently. It reports the producer command if an artifact is absent and never substitutes synthetic data for a result. Small software fixtures are explicitly labeled and excluded from efficacy tables.

| Notebook | Learning objective |
| --- | --- |
{nav}

Notebooks00–07 teach the original ingestion, pose processing, Skeleton S-JEPA, VICReg, progressive adaptation, forecasting and Random Forest workflow. Notebooks08–11 document the completed historical context study. Their original scientific outputs retain their historical identities. Navigation is updated; new execution reports distinguish an isolated historical replay from the new experiment.

## Local setup

From the repository root:

```bash
uv sync --project experiments/aaai
uv run --project experiments/aaai jupyter lab experiments/aaai
```

Use Python3.12 and the existing [pyproject.toml](pyproject.toml). Install the optional raw laboratory parser with `uv sync --project experiments/aaai --extra external`. There is no competing requirements file. The repository-root `.env` contains `AAAI_ITERATION02_RUN_DIR=experiments/aaai/artifacts/iteration02/{run}`. Existing API credentials are neither copied nor printed. Gait training and scoring require no language-model API call.

Before running the adapted STG-NF comparison, retrieve its pinned upstream code from the repository root:

```bash
git submodule update --init -- experiments/aaai/vendor/STG-NF
```

The submodule is fixed at `edb5f3220332e160e4d20ce258787d5e2d7e0200`, the revision verified by the baseline loader. Videos, model weights and run artifacts remain local and are excluded from Git. Reports and notebooks identify their producer commands and required run identities; a fresh clone must prepare or restore those local inputs before replaying the experiments.

Every notebook finds the repository root, changes to it, and loads:

```python
from dotenv import load_dotenv as load_env, find_dotenv as find_env
load_env(find_env())
```

`load_env` and `find_env` are aliases for python-dotenv's actual function names. The run records the interpreter and package versions. The default execution uses CPU; the compact ridge study and separately supervised probes fit on the laptop.

## Data, provenance and generalization

The fresh inventory has1,874 sequences,458,116 annotation rows and348 source IDs. The source distribution folder is excluded from counting. The new primary cohort retains9,442 windows,1,381 sequences and304 source recordings, including28 normals. Its focal membership counts are28 normal,10 Parkinsons,15 stroke,10 cerebral palsy and27 myopathic. One source belongs to both `abnormal` and `cerebral palsy`; it counts once in binary endpoints and is excluded from the separate five-class task under the frozen conflict rule. That task therefore has89 sources. Category memberships are not independent people.

The [fresh inventory authority](../../data/gavd/README.md), [preparation ledger](artifacts/iteration02/preparation/cohort_v2/provenance.json), and [source registry](artifacts/iteration02/{run}/source_registry.csv) explain counts and exclusions. A byte-different duplicate was confirmed by decoded frames in the full inventory. Both copies already failed pose-window quality checks, so none of the retained304 sources share that confirmed duplicate footage. Unknown repeated people cannot be resolved from upload IDs.

GAVD supports an internal study of observable movement across heterogeneous web recordings. **It does not by itself establish generalization to independent patients, new clinics or neurological causes.** All28 normal recordings appeared in earlier exploration. The normal sample is small; folder labels have not been independently clinically revalidated; view, activity, assistance, camera motion and pose quality differ. More windows and abnormal videos do not replace independent normal participants. No complete four-fold view-matched analysis resolves all these alternatives, and the small camera review is descriptive.

The ingestion wrappers reuse `ambient.utils.youtube_cache.download_youtube_urls` and `SequenceKeypointExtractor`. They restore full-frame coordinates after per-frame GAVD box cropping. The image detector returns the first pose in its crop and is not a validated identity tracker. The study uses `raw_xy`, confidence0.5, adjacent-valid resampling at15Hz, and the declared shared pelvis/first-half torso reference. It preserves projected posture, excursion and raw timing. Torso-scaled image motion is not physical walking speed or a measured3D joint angle.

All GAVD recordings and generated previews remain under `data/videos-gavd`. New source-organized recordings use `<source_id>/`, with previews in `<source_id>/previews/`. Existing root-level cached videos remain at their recorded paths. Videos are not duplicated into condition folders. The notebooks display local videos and labeled pose overlays. Source video rights have not been cleared for manuscript redistribution; the paper's failure illustrations use measured skeleton traces instead of photographs.

## Models and fair masking comparisons

The new core is a coordinate estimator, not a claim that a JEPA encoder learned a disease mechanism. The retained baseline is a shared77-feature ridge predictor. Own-leg and both-leg residual branches each use154 features and1,240 coefficients including intercepts. Their numerical feature ranks and inductive biases are reported. A zero multiplier returns the saved baseline exactly. Three grouped internal folds select regularization and correction multipliers using fitting normals only. Every outer fit has16 fitting normals, five separate reference normals and seven test normals.

U chooses64 of128 joint-patch tokens uniformly. J first hides one joint for all eight patches, then uniformly fills56 tokens. A hides an eight-token anatomical core, then fills56. Every policy hides half of the tokens and gives each token marginal probability0.5, without consulting motion, labels or attention. Each policy retains the same U-trained baseline. All are evaluated on the identical16-mask U bank with seed17 and the separately defined E_leg challenge with seed117. The [implemented protocol](docs/iteration02/protocol.md) states the exact anatomy, permitted-input rules and aggregation.

These are controlled adaptations of existing masking ideas. Skeleton S-JEPA and MAMP originally use motion-aware masking. VideoMAE and SLiM already motivate tube masks. Stochastic positional embeddings change positional queries; they are not mask selection or physiological uncertainty. New encoder-mask, positional-noise and latent-variable ablations are deferred. The fresh learned/random JEPA probes are a separate supervised five-class comparison with normal-only encoder pretraining, EMA and VICReg. VICReg discourages collapse and redundancy; it does not force disease labels apart.

## Reproduce the scientific runs

```bash
uv run --project experiments/aaai python -m aaai_gait.iteration_runner --cohort experiments/aaai/artifacts/iteration02/preparation/cohort_v2 --mode development
uv run --project experiments/aaai python -m aaai_gait.iteration_runner --cohort experiments/aaai/artifacts/iteration02/preparation/cohort_v2 --mode full --resume
uv run --project experiments/aaai python experiments/aaai/scripts/run_iteration02_auxiliary.py --run experiments/aaai/artifacts/iteration02/{run}
uv run --project experiments/aaai python experiments/aaai/scripts/run_iteration02_timing.py --run experiments/aaai/artifacts/iteration02/{run}
```

Development mode has a separate identity, uses real normal data, and is incomplete for efficacy. Full mode freezes the current cohort, code, source roles, masks and configuration before fitting. A changed scientific configuration creates a new identity. Read [the reproduction guide](docs/iteration02/protocol.md) for preparation, external replication, verification and report commands. Historical latest-result pointers are unchanged.

The raw-prediction verifier reconstructed16,801,488 core mask-target records and the timing verifier reconstructed5,870,520 diagnostic records. These are repeated measurements, not millions of independent samples. [Readiness](docs/draft02/submission_readiness.md) distinguishes an executable package and valid format from the scientific strength of an exploratory negative-result manuscript. The [official workshop](https://ismart.ece.mcgill.ca/astad4/) allows seven content pages plus two reference pages in anonymous AAAI-27 format and lists November20,2026 as the deadline.
'''
(BASE/'README.md').write_text(readme)

protocol=f'''# Implemented iteration02 protocol

The scientific identity is `{run}`. The [machine-readable freeze](../../artifacts/iteration02/{run}/protocol.json) records configuration, fresh cohort, source roles, code hashes, both evaluation banks and the freeze time before fitting. This is an exploratory redesign after the completed draft01 results were seen. The historical study remains unchanged.

## Observation and roles

The current GAVD CSVs produced 1,874 sequences and348 source IDs. Hash-compatible poses, decoded observations and fixed lower-limb support retain304 sources, 28 normal, 1,381 sequences and9,442 windows. Every32-sample window is resampled at15Hz. It spans31/15 seconds and contains eight four-frame patches. Confidence below0.5 is missing; interpolation uses adjacent valid source observations only. A fixed per-frame pelvis reference and first-half torso scale are shared by all branches. Masked hip/shoulder information already used by that reference is an explicitly disclosed conditioning variable Q.

Each of four outer folds reserves seven normal test sources and five normal reference sources. The other16 normal sources fit models. Three grouped internal folds select residual regularization and correction multipliers. Normal references never choose models, masks, stopping, signs or fusion. All overlapping windows from a recording stay in one role. Upload IDs are not verified participant identities. The confirmed duplicate pair in the inventory was already outside the eligible cohort.

The primary endpoint is normal versus Parkinsons, stroke, cerebral palsy and myopathic. The secondary endpoint uses all non-normal folders with unchanged scores. A source with two abnormal memberships counts once. The separately supervised five-class comparison excludes every multi-membership source according to the frozen rule, leaving89 sources. This is conservative label handling, not evidence that the folder label is clinically verified.

## Allowed inputs and targets

Training masks use only BlazePose indices11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32. The new source readout uses25–32. Own-leg learned inputs are23/25/27/29/31 or24/26/28/30/32. All derived prediction features are constructed after restrictions. Coordinate targets have four XY samples; loss is their mean squared error. Completion can use visible observations before and after the answer. It is not causal forecasting.

The U evaluation keeps the historical target rule: a valid hidden target and at least two visible valid patches of that target joint. New correction training and E_leg use a separate paired-leg rule: a valid target plus at least two permitted valid tokens at distinct temporal positions on each leg. Masks are drawn before validity checks, with no redrawing. A source needs four distinct original temporal target patches for each of the eight readouts. Repeated masks do not increase that count.

## Matched mask policies

| Policy | Construction | Hidden tokens |
| --- | --- | ---: |
| U | Uniform sampling without replacement. |64 of128|
| J | Uniformly choose one joint, hide its eight temporal tokens, then fill56 from the other120. |64 of128|
| A | Choose one of16 equiprobable anatomical cores, hide its eight tokens, then fill56 from the other120. |64 of128|

The A partition contains torso reference joints11/12/23/24, left arm13/15, right arm14/16, left distal leg25/27/29/31, and right distal leg26/28/30/32. Four-joint groups use two consecutive patches starting at0,2,4 or6. Two-joint groups use four patches starting at0 or4. Group probability is its width divided by16. Every core has eight tokens and each token belongs to one core, so every token's final hiding probability is `1/16 + (15/16)*(56/120) = 1/2`.

![Implemented mask comparison.](../../images/iteration02/mask_comparison.svg)

Mask structure changes higher-order dependencies and usable context. Matching marginal probabilities does not equalize difficulty. U/J/A correction training shares the exact same U-trained baseline weights in every internal and outer fit. The common U evaluation bank has16 draws, seed17, and the original Torch mask bytes. E_leg enumerates both distal sides and starts0/2/4/6 twice, fills to64 with seed117, and scores only the forced core. It is a distinct task. A versus U on the unchanged U bank is the primary mask contrast; A versus J and J versus U are explanatory contrasts.

## Estimator and operational gates

The baseline f has77 permitted-input features. Each residual has154 features and eight outputs. The own-capacity correction adds own-leg temporal moments; the both-leg correction adds opposite-leg observations. The parameter counts match, but effective rank and inductive bias can differ. Shared ridge also outperforms direct own-joint interpolation on every outer aggregate of normal internal predictions.

```text
base = f(O; Q)
own_capacity = f(O; Q) + alpha_own * h(O; Q)
both_legs = f(O; Q) + alpha_both * g(O, C; Q)
benefit = MSE(own_capacity, Y) - MSE(both_legs, Y)
```

Each fit uses512 source-balanced window/mask presentations and equal source/readout-joint loss weights. Residual ridge penalties are0.1,1 and10; multipliers are0,0.5 and1. The internal selection requires both-leg improvement over f and f+h in at least two of three folds, plus improved mean source risk. It is an operational selection rule, not a significance test. All12 GAVD outer-policy fits pass. Outer-test benefits are reported separately; the same selection data are not mislabeled as an independent confirmation.

The feature-level positive and negative dependency fixtures preserve matching marginal laws and use separate generated training/test samples. They are software checks, not gait simulations. Hidden-input invariance, exact zero correction, source disjointness, fixed target keys and exposure checks also pass.

## Scores and aggregation

M has16 full-observation descriptors. Each leg contributes projected knee-angle median/IQR, ankle X/Y IQR and X/Y velocity MAD. Four additional values capture trunk-inclination median/IQR, left-minus-right knee median and ankle-X excursion. They retain posture, amplitude and raw timing. Descriptor vectors are averaged within sequences and then within sources before fitting normal-only imputation, RobustScaler and Ledoit-Wolf covariance. M is squared Mahalanobis distance divided by16.

Prediction errors first average masks within a target, then targets within windows, windows within sequences, and sequences within sources while retaining the joint axis. Four joint-specific source benefits average equally on each side. Five separate normal references define the bilateral mean and `v_k=0.5*s_k^2+0.5*mean(s^2)+1e-8`, with sample-variance denominator n−1. R is the mean squared standardized departure. Both positive and negative departures can be unusual; benefit is not mutual information, causality, or a health ordering. The fixed combined score is `S=0.5*(M+R)`. These dimension-normalized distances are not equally calibrated probabilities.

The error and own-capacity sham fusions use the same reference identities and distance form. The36-feature motion baseline, fixed bilateral correlation/waveform summaries, quality distance, supervised nuisance classifier, Isolation Forest and adapted STG-NF all use the new source roles. STG-NF uses the authors' pinned code with a declared16-joint graph and fitting-normal median imputation, so it is an adapted comparator, not a faithful benchmark reproduction. Fresh learned/random JEPA probes use512 normal-only encoder updates and a separately supervised five-class linear head. They do not test an encoder-mask contribution to the coordinate method.

## Timing and uncertainty

The independent48-frame parent grid retains the central8:40 interval. Opposite distal context comes from offsets−4 or+4 frames without wrap or padding. A donor distal-minus-hip vector is attached to the original permitted hip with the original scale and Q. Donor joint, donor hip, original joint and attachment hip must be valid and visible in the original192-token mask. Original and shifted predictions share the pair's exact support and hidden answer. Sixteen parent masks use seed217; an eight-parent-per-source budget is fixed by evenly spaced temporal order before scoring.

The all-parent and budget coverage censuses retain27 normals at each offset, with test counts7/7/6/7. Zero offset, unchanged own inputs and source-frame provenance pass. However, every exact-paired-support adequacy gate fails. The mean editable opposite-distal context is only about1.75 tokens. Some shifts change error, but this cannot establish a reliable timing or neurological-coordination measurement. Noise, dropout and whole-body crop controls retain separate meanings and populations.

AUROC is the normal-by-abnormal-pair-count weighted mean of within-fold AUROCs. Average precision is reported per fold and as an equal-fold mean with prevalence. Paired bootstraps resample whole source/duplicate-recording clusters within fold and label, with original fold weights. Laboratory sensitivity resamples matched pairs. Intervals condition on fitted models, split, feature definitions and exploratory protocol. A clinical alarm threshold,95% specificity or calibrated disease risk is not supported.

## Reproduction commands

The [README](../../README.md) gives setup. Run the core producer, auxiliary comparators and timing driver, then `scripts/analyze_iteration02.py`, `scripts/normal_iteration02_diagnostics.py`, `scripts/finalize_iteration02_evidence.py` and `python -m aaai_gait.iteration_verify`, each with the exact `--run` path shown in their help (the verifier uses a positional path). These scripts reconstruct tables from saved predictions and scores. The full model run takes roughly ten minutes on this laptop including conditional bootstraps; timings are recorded, not performance promises.

The external commands are `uv run --project experiments/aaai --extra external python -m aaai_gait.external_iteration02 prepare` and the same command with `run`. The executed version is preserved in that run's `source_snapshot`. A later safety guard requires demographic matching to be frozen before scoring; it does not alter the saved fitting or scores. The [external report](external_replication_results.md) explains byte-range downloads, units, exclusions, age/sex matching and checks.

Use `scripts/build_iteration02_reports.py` and `scripts/publish_iteration02_docs.py` to refresh shared evidence and documentation. The manuscript has its own `docs/draft02/build_manuscript.py`; notebooks and slides retain their source builders. A changed endpoint, observation rule, mask, source role or score requires a new scientific identity. Post-outcome observation audits are labeled as such and never silently change this cohort.
'''
(DOC/'protocol.md').write_text(protocol)

import pandas as pd
RUNDIR=BASE/'artifacts/iteration02'/run
names={'M':'Local movement M, 16 descriptors','R':'Bilateral benefit departure R','S':'Fixed fusion S, U training','motion36':'Richer motion distance, 36 descriptors','S_sham':'M plus own-capacity sham','S_constituent':'M plus constituent-error distance','S_base_error':'M plus baseline-error distance','timing8':'Timing and waveform descriptors','isolation_forest':'Isolation Forest on movement descriptors','stgnf_adapted':'Adapted STG-NF, missing-data imputation','nuisance_distance':'Normal-only availability and viewpoint distance','nuisance_supervised':'Supervised nuisance classifier'}
rows='\n'.join(f'| {title} | {auc("focal",name):.3f} | {auc("broad",name):.3f} |' for name,title in names.items())
normal=pd.read_csv(RUNDIR/'normal_error_table.csv')
def markdown_table(table,precision=3):
    def cell(v):return f'{v:.{precision}f}' if isinstance(v,float) else str(v)
    return '| '+' | '.join(map(str,table.columns))+' |\n| '+' | '.join(['---']*len(table.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in table.itertuples(index=False,name=None))
normal_table=markdown_table(normal,6)
supervised=markdown_table(pd.DataFrame(e['supervised']).groupby('method')[['macro_f1','balanced_accuracy']].mean().reset_index())
mask_matrix=markdown_table(pd.DataFrame([dict(policy=policy,evaluation_bank=bank,focal_S=v['S']['weighted_auroc'],broad_S=e['all_mask_metrics'][f'{policy}/{bank}/broad']['S']['weighted_auroc'],sources=304,reference_normals_each_fold=5) for policy in ['U','J','A'] for bank in ['U','E_leg'] for v in [e['all_mask_metrics'][f'{policy}/{bank}/focal']]]))
results=f'''# Verified iteration02 results

The fixed combination of local movement and bilateral prediction benefit reduced anomaly-ranking performance. This is a result about the specified estimator, normal reference and equal-weight score. It does not show that opposite-leg information is useless. The independent laboratory study shows a clear completion benefit, while its relational score alone has AUROC {e['external']['metrics']['R']['weighted_auroc']:.3f}.

The GAVD run is `{run}`. The separate participant run is `{ext}`. The [shared evidence record](../draft02/evidence.json), [raw source scores](../../artifacts/iteration02/{run}/source_scores.csv), and [protocol](protocol.md) are the authorities for repeated numbers. Every table below comes from executed models and saved records. Synthetic dependency fixtures are excluded.

## Detection on fixed endpoints

The focal primary has 90 source recordings, including 28 normal. The broad secondary has 304 recordings, including the same 28 normals. Scores are identical between endpoints. AUROC is a fixed pair-count-weighted mean of four within-fold AUROCs. Average precision, fold values, prevalences, per-condition results and secondary pooled rankings are saved with the run.

| Method | Focal AUROC | Broad AUROC |
| --- | ---: | ---: |
{rows}

For U training on the fixed U bank, S minus M is {gap['observed']:.3f}, conditional 95% interval {interval(gap['conditional_95_interval'])}, on the focal endpoint. The broad difference is {e['broad_difference']['observed']:.3f}, {interval(e['broad_difference']['conditional_95_interval'])}. The fixed fusion therefore does not establish added detection value. Neither unfavorable nuisance nor pose-baseline scores were reversed after seeing labels.

![Conditional differences for the fixed score.](../draft02/figures/detection_difference.svg)

## Prediction and masking

Every one of the 12 outer-policy fits passed the planned internal normal gate. Selection used the three grouped fitting-normal folds. This is an operational adequacy check, not an independent significance test. Internal candidate selection averages source/joint-weighted prediction requests; final scoring first merges repeated masks for each physical window target. Their weighting differs and is disclosed rather than called identical.

The following are errors averaged over held-out normal sources. The same U-trained baseline is preserved in every training arm. Equal nominal residual feature widths do not prove equal effective capacity: recorded numerical feature ranks and inductive biases differ.

{normal_table}

![Normal prediction errors on the unchanged and focused tasks.](../draft02/figures/normal_prediction.svg)

The outer-normal U-bank mean benefit for U and A remains uncertain. E_leg exposes more completion headroom. A versus U on the unchanged U bank improves the focal S point estimate by 0.055, with conditional interval [−0.018, 0.127]. Its broad interval also includes zero. These comparisons do not establish a superior masking policy. A result on E_leg is a different task, not a substitute for the primary U task. The full U/J/A by U/E_leg matrix, every paired interval and bank-specific reference are retained in [the machine-readable metrics](../../artifacts/iteration02/{run}/rawmetrics.json).

{mask_matrix}

The [per-condition metrics](../../artifacts/iteration02/{run}/per_condition_metrics.json), [common historical-source sensitivity](../../artifacts/iteration02/{run}/historical_common_source_sensitivity.json), and [normal-reference sensitivity](../../artifacts/iteration02/{run}/reference_sensitivity_metrics.csv) retain the full secondary analyses. The common-source comparison uses refitted current models, not old scores paired with a different population.

Direct interpolation is weaker than shared ridge in all four aggregates of GAVD internal held-out normals. This post-outcome inspection used the existing frozen masks, permitted values and targets; it did not change recipe selection. The [saved comparison](../../artifacts/iteration02/{run}/interpolation_normal_adequacy.csv) gives its population and losses.

## Timing support changes the interpretation

The paired timing census retains 27 normal sources, with 7, 7, 6 and 7 outer-test normals across folds. It passes the planned recording-count floor. However, every one of the 24 outer fold/policy/offset combinations fails prediction adequacy on the exact paired original inputs. Only about 1.75 editable distal context tokens remain on average. The internally selected models also fail this paired-support check in all 72 internal combinations.

For U, the minus-four-frame error effect is about 0.000393, with interval [0.000177, 0.000618]. The plus-four-frame effect is about 0.000064, with interval [−0.000232, 0.000368]. A positive response on an inadequate predictor is not reliable evidence of coordination measurement. The code passes zero-offset, own-branch invariance, donor visibility, fixed target and fixed reference checks. These software properties do not repair the sparse observation task. [Timing summary](../../artifacts/iteration02/{run}/timing_summary.json).

![Timing response beside the failed adequacy gate.](../draft02/figures/timing_limits.svg)

## Independent laboratory method replication

All 188 frozen participant trials were retrieved and parsed. Ten healthy trials fail the unchanged observation support, leaving 128 healthy and 50 stroke participants, with 433 windows. Six simultaneous hip, knee and ankle flexion signals retain a shared raw clock. This is a separately fitted method replication, with different units and descriptors, not video-weight transfer or a BlazePose mapping.

All four internal gates pass. On held-out healthy participants, both-leg prediction reduces error relative to the own-capacity control by 4.78% to 5.52% across folds. Nevertheless, M reaches {e['external']['metrics']['M']['weighted_auroc']:.3f} and S reaches {e['external']['metrics']['S']['weighted_auroc']:.3f}; S minus M is {egap['observed']:.3f}, {interval(egap['conditional_95_interval'])}. The prespecified age and sex matched analysis retains 45 pairs and also favors M. The [external report](external_replication_results.md) documents trial selection, metadata, simultaneous units, exclusions and matching. Walking speed was not matched; it may be part of the movement phenotype, and its influence remains unresolved.

## Separate supervised classification

The five-class task uses 89 source recordings after the frozen multi-membership exclusion. Encoders are newly trained on each outer fitting-normal pool; the label probe and Random Forest can use the permitted training abnormal sources. This is supervised classification and is not a normal-only anomaly benchmark. These are equal means over fold-level scores, with every confusion matrix retained.

{supervised}

The compact learned JEPA probe does not establish an advantage over the random encoder. The encoder U/J/A ablation, stochastic positional queries, stochastic latent variables and phase modeling were deferred. Fresh learned/random U probes are not evidence that an unrun encoder-mask ablation succeeded.

## Observation review and uncertainty

Ten prespecified high/low prediction-error cases span all five focal labels. Each has a local video preview, pose overlay and three inspected frames. Visible concerns include camera movement, split-screen composition, assistance, limb overlap and feet leaving the frame. This is a small descriptive review, not clinical relabeling or continuous identity validation.

A separate post-outcome image-boundary audit finds high-confidence distal points outside the decoded image in frozen-window spans from 13 of 28 normal and 83 of 276 abnormal recordings. Their pooled point fractions are about 0.43% and 0.24%, respectively. Confidence is not proof of visible measurement. One requested frame could not be decoded in this audit. No scores, exclusions or fits were changed after inspecting these findings. [Audit ledger](../../artifacts/iteration02/{run}/video_review/boundary_audit.md).

Bootstrap intervals condition on fitted models, roles, selected protocol and known recording groups. They omit training uncertainty, unknown repeated-person dependence and adaptive redesign. Four overlapping fits are not four independent studies. The mixed-view group is not a matched viewpoint, and single-view strata lack adequate four-fold normal support. Neither a global nuisance AUROC nor the small camera review resolves these alternatives. See [limitations](limitations.md) and [submission readiness](../draft02/submission_readiness.md).
'''
(DOC/'results.md').write_text(results)

limitations='''# What this study does and does not establish

The experiments establish an executed, auditable negative result for a fixed local-plus-relational score. They also show that an adequate prediction relationship and useful anomaly ranking are separate questions. They do not establish a new disease mechanism, a clinical diagnostic device or state-of-the-art detection.

1. GAVD source IDs identify uploads, not verified people. All 28 normal recordings were previously explored. The changed focal primary and lower-limb cohort are explicit exploratory decisions made after draft01, even though the new configuration was frozen before new fits.
2. Labels are folder annotations, not independent clinical examinations. Myopathic gait is a neuromuscular category, not one neurological lesion. Cerebral palsy, stroke and Parkinsons have heterogeneous manifestations. Regular compensation may remain predictable.
3. Image coordinates retain projection, crop, camera and detector effects. The shared pelvis/torso reference already uses body measurements. Own-leg input independence is conditional on that reference. Laboratory Plug-in Gait angles also share an upstream body model.
4. Matching residual feature width and coefficient count does not match effective rank or inductive bias. The own-capacity control removes one simple explanation, not every possible capacity difference. Prediction gain is not mutual information.
5. The timing diagnostic passes recording coverage but fails exact paired-support adequacy. Its response cannot establish reliable coordination measurement. Shifts also change instantaneous geometry and finite-interval marginal distributions, so even a successful future test would not isolate neural timing causally.
6. M and R are dimension-normalized distances, not equally calibrated probabilities. R uses shrunk component variances without a bilateral covariance term. Fixed equal fusion can amplify noise or correlated evidence. The observed loss rejects this recipe's advantage, not all relational information or all possible fusion rules. Abnormal labels must not retrospectively tune a new headline recipe.
7. Five GAVD reference normals give uncertain reference estimates. Leave-one-reference-out and variance-floor results are saved. Conditional bootstrap intervals omit fitting uncertainty, unknown person dependence and exploratory protocol selection. No well-resolved clinical specificity, alarm threshold or calibrated risk follows.
8. The adapted STG-NF uses a changed graph and fitting-normal median imputation. It is a bounded pose-method comparator, not a faithful published reproduction. Native complete-case support has only five normal recordings. Small JEPA probes and limited optimization do not establish the limits of representation learning.
9. Independent participant replication improves source independence and supports the fixed-score finding in another measurement system. It covers stroke and healthy walking, uses separately fitted angle models, and does not validate automatic video transfer or the other condition families. Age/sex matching leaves speed and other differences unresolved. Ten selected healthy trials fail observation support.
10. The high/low-case review and later boundary audit reveal measurement limitations without changing the evaluated cohort. Neither all-person tracking nor full camera-motion adjustment is validated. Future extraction changes require a new cache identity and cohort, applied across categories before scoring.

The most useful next experiment is a new participant-held-out replication that first demonstrates adequate prediction on the exact observation support, compares direct interpolation and other simple predictors, and learns any score calibration entirely inside independent normal development data. Its evaluation must retain posture and amplitude alongside relational information. A larger encoder or favorable post hoc mask is not a substitute for that design.
'''
(DOC/'limitations.md').write_text(limitations)

# Repair prose-only spacing without changing code blocks, links or identifiers.
import re
def prose_spacing(text):
    pieces=re.split(r'(```[\s\S]*?```|`[^`]*`|\]\([^)]*\))',text)
    prefixes=['of','has','and','including','All','all','with','at','below','seed','retained','Python','Notebooks','reconstructed','conditional','November','contains','have','uses','use','every','Every','Each','each','are','by','from','fills','fill','choose','for','into','across','confidence','probability','chooses','retain','retains','identical','spans','counts','about','measured','indices','readouts','tokens','other','contains','leaving','or','joints','arm','leg','starts','to','The','declared','independent','central','original','shared']
    for i in range(0,len(pieces),2):
        pieces[i]=re.sub(r'\b('+'|'.join(prefixes)+r')(?=\d)',r'\1 ',pieces[i])
        pieces[i]=re.sub(r'(?<=\d)(?=Hz\b)', ' ',pieces[i])
        pieces[i]=re.sub(r'(?<=\d),(?=2026\b)', ', ',pieces[i])
    return ''.join(pieces)
for path in [BASE/'README.md',DOC/'protocol.md',DOC/'results.md',DOC/'limitations.md']:
    path.write_text(prose_spacing(path.read_text()).replace('roughly ten minutes','roughly eleven minutes'))

(PAPER/'README.md').write_text(f'''# Draft02: controlled cross-leg prediction and anomaly detection

This manuscript reports the executed GAVD protocol `{run}` and the independent participant replication `{ext}`. The defensible finding is that useful completion does not automatically improve anomaly ranking: the frozen equal-weight score loses to local movement on both datasets. The GAVD timing diagnostic has enough recordings but fails prediction adequacy on its paired support.

Read the [manuscript PDF](manuscript.pdf), [editable manuscript](manuscript.md), [verified results](../iteration02/results.md), [evidence record](evidence.json), [readiness assessment](submission_readiness.md), and [responses to independent reviews](response_to_reviews.md). The [active slides](../../slides/tutorial.pdf) and [six new tutorials](../../README.md#tutorials) explain the method and its failures.

This is an exploratory measurement and negative-result paper. It does not claim a clinical mechanism, a new tube-mask family, automatic transfer to patients, or a detection advantage. Formatting readiness and scientific strength are assessed separately. The completed [draft01](../draft01/README.md) remains a historical study with its original results.
''')
print(DOC/'results.md')
