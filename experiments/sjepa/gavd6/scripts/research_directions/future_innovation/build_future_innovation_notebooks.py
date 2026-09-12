"""Editable sources for five connected Future Innovation tutorials.

Use --only 00 01 to regenerate selected lessons, --check for exact source
notebooks, or --check-sources to permit retained execution outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from textwrap import dedent

import nbformat


ROOT = Path(__file__).resolve().parents[3]
DESTINATION = ROOT / "notebooks/experiments/future_innovation"
BUILDER = Path(__file__).relative_to(ROOT).as_posix()
NAMES = {
    "00": "00_question_and_worked_example.ipynb",
    "01": "01_cohort_and_alignment.ipynb",
    "02": "02_teacher_features_and_validity.ipynb",
    "03": "03_matched_predictors_and_controls.ipynb",
    "04": "04_results_and_next_decision.ipynb",
}


def md(source):
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source):
    return nbformat.v4.new_code_cell(dedent(source).strip())


STARTUP = r'''
from pathlib import Path
import os
import sys
from time import perf_counter

started = perf_counter()
override = os.environ.get("GAVD6_ROOT")
if override:
    candidates = [Path(override).expanduser().resolve()]
else:
    candidates = []
    for base in (Path.cwd(), *Path.cwd().parents):
        candidates.extend((base, base / "gavd6", base / "experiments/sjepa/gavd6"))
PROJECT_ROOT = next((p for p in candidates if (p / "src/gavd6_sjepa").is_dir()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Set GAVD6_ROOT to the checkout containing src/gavd6_sjepa.")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from matplotlib_inline.backend_inline import set_matplotlib_formats
get_ipython().run_line_magic("matplotlib", "inline")
set_matplotlib_formats("svg", "png")
plt.rcParams.update({"figure.figsize": (8, 3), "axes.spines.top": False,
                    "axes.spines.right": False, "font.size": 11})

from gavd6_sjepa.research_directions.future_innovation.fi_tutorial_inspection import (
    artifact_inventory, inspect_report, read_optional_table, inspection_audit_path,
)

# Use "execute" for real stages or "inspect" for saved artifacts.
# Relative paths resolve from GAVD6_ROOT. Execution requires an explicit run root.
MODE = os.environ.get("FI_TUTORIAL_MODE", "teach")
if MODE not in {"teach", "inspect", "execute"}:
    raise ValueError("FI_TUTORIAL_MODE must be teach, inspect, or execute.")
if MODE == "execute" and not os.environ.get("FI_RUN_ROOT"):
    raise ValueError("Set FI_RUN_ROOT explicitly before executing real experiment stages.")
RUN_ROOT = Path(os.environ.get("FI_RUN_ROOT", "outputs/future-innovation-direct-v3-dev-20260911")).expanduser()
if not RUN_ROOT.is_absolute():
    RUN_ROOT = PROJECT_ROOT / RUN_ROOT
RUN_ROOT = RUN_ROOT.resolve()
print("Teaching examples only; no empirical gait findings." if MODE == "teach"
      else f"{MODE.upper()} mode: {RUN_ROOT}")
if MODE == "execute":
    from gavd6_sjepa.research_directions.future_innovation.fi_notebook_workflow import (
        initialize_from_environment, run_stage, build_notebook_report, finish_notebook_report,
        attempt_stage, require_stage_success,
    )
    if (RUN_ROOT / "config/run-contract.json").is_file():
        import json
        saved_run = json.loads((RUN_ROOT / "config/run-contract.json").read_text())
        print("Frozen protocol:", saved_run.get("protocol", "legacy-v1"),
              "— gate clips:", saved_run.get("cohort_size"))
        if saved_run.get("protocol", "legacy-v1") == "legacy-v1":
            print("This run retains legacy selectivity gates. Use a new run root for direct-v2.")
'''


def execution_cells(number):
    descriptions = {
        "00": "Initialize the immutable protocol and input provenance, or validate the existing run. Cached direct-v3 uses FI_PARENT_ROOT and a passing FI_CALIBRATION record; other protocols use the original HAIC input setup.",
        "01": "Original runs build candidates and aligned poses. Cached direct-v3 verifies and reuses the frozen parent cohort and pose evidence. This can take hours. The existing stages verify and reuse completed work; inspect the resulting exclusions and overlays below.",
        "02": "Cache the frozen teacher features and check input integrity, teacher repeatability and future isolation. The new direct-v2 protocol performs no person/background replacement or selectivity tests. Original encoding normally uses one H100. Cached direct-v3 verifies inherited arrays and audits entirely on CPU. A verified audit rejection finishes with TRAINING BLOCKED and no fitting; missing or corrupt evidence remains an execution error. Completed stages are verified and reused without loading the teacher again.",
        "03": "Run five outer folds, all frozen arms and the complete frozen inner grid. Direct-v3 has one deterministic fit; historical protocols retain three seeds. With FI_NOTEBOOK_FOLD set, run just that outer fold (all seeds and arms); HAIC supplies five CPU array tasks. Without it, run all folds sequentially. A verified audit rejection skips fitting and records a blocked outcome. No teaching settings enter this branch.",
        "04": "Score the out-of-fold predictions and build the sealed production report. If scoring fails, still attempt a diagnostic STOP report. A verified audit rejection builds an unsealed diagnostic STOP and finishes with TRAINING BLOCKED, measurement_complete=False. Unexpected scoring failures and missing or corrupt evidence still fail. Complete STOP and INCONCLUSIVE results are also successful executions.",
    }
    operations = {
        "00": 'initialize_from_environment(RUN_ROOT)',
        "01": 'stage_error = attempt_stage("build-cohort", RUN_ROOT)\nif stage_error is None:\n    stage_error = attempt_stage("extract-poses", RUN_ROOT)',
        "02": 'device = os.environ.get("FI_NOTEBOOK_DEVICE", "cuda")\nstage_error = attempt_stage("cache-teacher", RUN_ROOT, "--device", device)\nif stage_error is None:\n    stage_error = attempt_stage("audit-teacher", RUN_ROOT, "--device", device)',
        "03": '''fold = os.environ.get("FI_NOTEBOOK_FOLD")
if fold is not None and fold not in {"0", "1", "2", "3", "4"}:
    raise ValueError("FI_NOTEBOOK_FOLD must be 0–4; unset it to run all five folds.")
options = [] if fold is None else ["--outer-fold", fold]
stage_error = attempt_stage("run-gate", RUN_ROOT, "--device", "cpu", *options)''',
        "04": 'scoring_succeeded = build_notebook_report(RUN_ROOT)',
    }
    return [md(f"## Execute this stage\n\n{descriptions[number]}\n\nThis cell runs only in `execute` mode. Each command uses this kernel's Python and the existing production CLI; stage logs are retained alongside the executed notebook."),
            code('if MODE == "execute":\n' + '\n'.join('    ' + line for line in operations[number].splitlines()))]


def opening(number, title, text):
    return [
        md(f"# {number} · {title}\n\n{text}"),
        md("""
        **Run this notebook independently in a fresh kernel.** The default
        `teach` mode uses small generated examples. Set `FI_TUTORIAL_MODE=inspect`
        and `FI_RUN_ROOT` before starting the kernel to read saved artifacts.
        Set `FI_TUTORIAL_MODE=execute` with an explicit `FI_RUN_ROOT` to run the
        production stages below. Execute notebooks **00 → 04** in order for the
        full Experiment 0; each uses a fresh kernel and the same run directory.
        Use the [notebook HAIC launchers](../../../slurm/future-innovation/NOTEBOOKS.md)
        for scheduled execution. Inspection remains read-only. An absent local
        file says nothing about the current state of a remote HAIC job.

        [Study overview](../../../docs/studies/future-innovation/README.md) ·
        [Historical direct-v2 specification](../../../docs/studies/future-innovation/direct-gate-protocol.md) ·
        [Calibrated direct-v3 specification](../../../docs/studies/future-innovation/direct-v3-repair-protocol.md)
        """),
        code(STARTUP),
    ] + execution_cells(number)


def ending(number, text):
    following = int(number) + 1
    link = (f"Continue with [{NAMES[f'{following:02d}']}]({NAMES[f'{following:02d}']})."
            if following < 5 else
            "Return to the [study overview](../../../docs/studies/future-innovation/README.md) to record the next decision.")
    completion = ([code('if MODE == "execute":\n    completed_decision = finish_notebook_report(RUN_ROOT, scoring_succeeded=scoring_succeeded)')]
                  if number == "04" else [])
    if number in {"01", "02", "03"}:
        completion = [code('if MODE == "execute":\n    require_stage_success(stage_error)')]
    return completion + [md(f"## What this step establishes\n\n{text}\n\n{link}"),
            code('print(f"Notebook elapsed time: {perf_counter() - started:.2f} seconds ({MODE} mode).")')]


def lesson00():
    cells = opening('00', 'What can past skeleton motion add?', '''
We predict a teacher's future person-region feature vector from the first 32
frames of a clip. The RGB reference already sees video features, framing,
recording conditions and observation quality. The comparison asks whether an
explicit skeleton representation improves that prediction. Skeletons come from
RGB, so an improvement concerns representation for these predictors.
''')
    cells += [md('''
The inspected `gate-v2` / `direct-v2` run completed with STOP. Its residual head
added an unrestricted RGB map to ridge's own training errors. Unsupported weights
and unstable missingness scaling damaged prediction. The saved result stays
historical evidence; deleting weights afterward was a diagnostic intervention.

`direct-v3` fits `intercept + X W_x + S W_s` jointly. X is safely transformed RGB
and nuisance information; S contains ordered skeleton summaries. Separate ridge
penalties restrain the two coefficient blocks. The squared Frobenius penalty is
the sum of squared matrix entries. Source weights balance videos, and the
intercept is unpenalized. Inner validation can choose the exact RGB baseline.
Legacy-v1 retains its selectivity rules; direct-v2 and direct-v3 omit them.

For an invented target of 0.8, a baseline of 0.5 and full prediction of 0.7,
the improvement is visible as smaller squared error. A baseline-only outcome
returns 0.5 exactly, including after saving and reloading. The difference
`full - baseline` describes a prediction difference; direct-v3 does not train a
second RGB map on ridge residuals.
'''), code('''
if MODE == "teach":
    display(pd.DataFrame({"target": [0.8]*3, "prediction": [0.5, 0.7, 0.5]},
                         index=["RGB reference", "Illustrative joint prediction", "Exact baseline fallback"]))
    from gavd6_sjepa.research_directions.future_innovation.fi_metrics import score_arrays
    target = np.array([[-1.], [0.], [1.]])
    reference = np.zeros_like(target)
    base = target * 0.5
    fallback = score_arrays(target, base, base.copy(), np.ones(3), np.ones(1,dtype=bool))[0]
    assert fallback['delta_r2'] == 0.0
    display(pd.DataFrame([fallback]))
'''), md('''
The main estimate is real-skeleton R² minus no-skeleton R². No-skeleton retains
time-varying joint validity while zeroing coordinates and confidence. Gain over
the shared RGB-only reference is a separate requirement. R² compares prediction
error with the outer-training-mean error; it can be negative. The target mean is
zero in training-standardized units, never the test-set mean.

The inspected cohort is development data. Passing all repaired criteria would
justify independent-source confirmation before scaling. Synthetic examples in
teach mode illustrate software behavior and cannot authorize scientific ADVANCE.
'''), code('''
if MODE != "teach":
    from gavd6_sjepa.research_directions.future_innovation.fi_contracts import read_json
    path = RUN_ROOT / 'config/run-contract.json'
    if path.is_file():
        run = read_json(path)
        display(pd.DataFrame([{'run_id':run['run_id'], 'protocol':run.get('protocol','legacy-v1'),
                               'root':str(RUN_ROOT), 'synthetic':run['synthetic'],
                               'development':run.get('development',True)}]))
    display(artifact_inventory(RUN_ROOT))
    evidence = inspect_report(RUN_ROOT)
    print('Report integrity checked:', evidence['seal_verified'])
    numerical = RUN_ROOT / 'reports/numerical-verification.json'
    print('Saved numerical reconstruction record present:', numerical.is_file())
    print('This inventory checks local presence. The report seal checks report integrity; neither is a fresh numerical reconstruction.')
''')]
    return cells + ending('00', 'The question, RGB reference and matched skeleton increment are distinct. Next establish which clips and source partitions supply the inputs and targets.')


def lesson01():
    cells = opening('01', 'Align clips, joints and source folds', '''
A useful predictor comparison needs a fixed boundary between past inputs and
the teacher target. All clips from a source video stay together in five outer
folds. Three inner source folds choose settings within each outer-training set.
A source split is not proof that participants differ across recordings.
''')
    cells += [md('''
The existing 50 clips and their source labels are reused unchanged in direct-v3.
The original cohort has 43 sources and 39–41 training clips per outer fold.
These sources have been inspected during debugging and are development data.
No clip is removed because a held-out joint becomes missing or a transform
produces an unusually difficult prediction.

| Quantity | Boundary and meaning |
|---|---|
| Prefix RGB, pose and normalization | frames 0–31 only |
| Skeleton history | 32 frames × 33 joints × x, y, confidence, validity |
| Teacher target region | frames 38–39 |
| Teacher target encoding context | full 64-frame clip, including later observations |
| Missing joint | no invented coordinate or velocity observation |
| Velocity summary | adjacent valid endpoints within each ordered 8-frame bin |
'''), code('''
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import FRAME
fig, ax = plt.subplots(figsize=(9,2.2))
ax.broken_barh([(0,32)],(0,0.7),facecolors='#2f6f99',label='Predictor inputs')
ax.broken_barh([(38,2)],(0,0.7),facecolors='#d4803f',label='Target region')
ax.broken_barh([(0,64)],(1,0.5),facecolors='#bdd5c8',label='Full teacher context')
ax.set(xlim=(0,64),xticks=[0,8,16,24,32,38,40,64],yticks=[],xlabel='Clip frame boundary')
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.4),ncol=3); plt.show()
if MODE != "teach":
    cohort = read_optional_table(RUN_ROOT,'manifests/gate-windows.csv')
    if cohort is not None:
        display(cohort.groupby('outer_fold').agg(test_clips=('window_id','size'),test_sources=('video_id','nunique'))
                .assign(training_clips=lambda t: len(cohort)-t.test_clips))
        print(f'{len(cohort)} clips from {cohort.video_id.nunique()} source videos; inspected development cohort.')
        columns=[c for c in ['window_id','video_id','outer_fold','source_first_frame','source_last_frame',
                             'context_pose_coverage','minimum_person_crop_retention','horizon_seconds'] if c in cohort]
        display(cohort[columns])
    else:
        print('No local frozen cohort. Source availability and cohort eligibility are separate stages.')
'''), code('''
if MODE != "teach":
    from gavd6_sjepa.research_directions.future_innovation.fi_contracts import read_json
    run_path=RUN_ROOT/'config/run-contract.json'
    if run_path.is_file() and read_json(run_path).get('protocol')=='direct-v3':
        from gavd6_sjepa.research_directions.future_innovation.fi_cache_reuse import load_reused_cache
        cohort, arrays=load_reused_cache(RUN_ROOT)  # Verified and read-only; no raw media.
        sk=arrays['skeleton']
        display(pd.DataFrame({'window_id':cohort.window_id,'valid_fraction':sk[...,3].mean(axis=(1,2)),
                              'confidence_mean':sk[...,2].mean(axis=(1,2)),
                              'missing_fraction':1-sk[...,3].mean(axis=(1,2))}))
        fig,ax=plt.subplots(figsize=(9,3))
        ax.imshow(sk[0,...,3].T,aspect='auto',vmin=0,vmax=1,cmap='Blues')
        ax.set(xlabel='Prefix frame',ylabel='Joint',title=f'Validity: {cohort.window_id.iloc[0]}'); plt.show()
        print('Inherited cache integrity checked here. Raw alignment evidence is reused from parent receipts.')
''')]
    return cells + ending('01', 'The same cohort and source folds now define every fitting boundary. Next establish how teacher features were encoded and which validity evidence is reused.')


def lesson02():
    cells = opening('02','Teacher features, prefix isolation and lineage','''
A frozen teacher supplies features without learning from this predictor study.
Its input encoding must exclude future information before attention mixes tokens.
Its target encoding deliberately uses the full clip. These are different roles.
''')
    cells += [md('''
At 384 × 384 pixels, 16-pixel patches form a 24 × 24 grid. Two-frame tubelets
cover the temporal axis. Prefix encoding keeps only frames 0–31 before attention;
masking after full-clip attention would already contain future information.
The target pools the person region at frames 38–39 and applies the same frozen
256-dimensional projection in every fold.

RGB inputs concatenate global prefix pooling, the last prefix person region,
prefix background pooling, and nuisance summaries. Unavailable background
measurements have separate support fractions. Recording conditions may help
explain the contextual target, so a gain is not automatically a gait-dynamics claim.
'''), code('''
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import FRAME
from gavd6_sjepa.research_directions.future_innovation.fi_token_regions import context_indices
past_ids=context_indices()
assert past_ids[-1] < FRAME.context_stop_exclusive // FRAME.tubelet_size * FRAME.grid**2
display(pd.DataFrame({'tokens':[len(past_ids),FRAME.frames_per_clip//FRAME.tubelet_size*FRAME.grid**2]},
                     index=['Prefix before attention','Full target context']))
'''), md('''
Direct-v2 retained three preselected windows for repeatability and randomized
future-pixel checks. Direct-v3 reuses those measurements and cached arrays after
checking their original contracts, file hashes, identities, shapes and audit
arithmetic. It needs neither raw-video mounts nor a GPU. It does not rerun
teacher encoding or certify person/background selectivity. Legacy-v1 still
requires its original selectivity measurements.

| Evidence | What it establishes |
|---|---|
| Cache and projection hashes | Identity of reused arrays and projection |
| Repeated inference measurements | Teacher numerical stability on audited windows |
| Randomized future-pixel measurements | Prefix feature isolation on audited windows |
| Training target variance | Usable dimensions, with a training-only mask |
| Original cohort receipts | Inherited alignment and pose provenance |
'''), code('''
if MODE != "teach":
    from gavd6_sjepa.research_directions.future_innovation.fi_contracts import read_json
    lineage=RUN_ROOT/'config/parent-lineage.json'
    if lineage.is_file():
        record=read_json(lineage)
        display({k:record[k] for k in ['parent_root','parent_run_id','parent_protocol','teacher_evidence','raw_alignment_evidence','identities']})
    audit=inspection_audit_path(RUN_ROOT)
    if audit.is_file():
        record=read_json(audit)
        display(pd.DataFrame(list(record.get('checks',{}).items()),columns=['saved check','passed']))
        print('Teacher evidence:',record.get('teacher_evidence','original run audit; consult saved protocol'))
        print('Failed checks:',[k for k,v in record.get('checks',{}).items() if not v])
    for name in ['teacher-stability.csv','causal-leakage.csv','target-sensitivity.csv']:
        table=read_optional_table(RUN_ROOT,'qc/'+name)
        if table is not None:
            print(name); display(table)
    print('Inspect mode reads saved audit evidence; execution verifies the applicable stage.')
''')]
    return cells + ending('02','The cached predictor inputs and contextual targets have explicit lineage. Next fit matched models using only training sources and expose each candidate’s validation evidence.')


def lesson03():
    cells=opening('03','Select jointly regularized predictors and controls','''
Direct-v3 fits one deterministic model per outer fold and arm. It does not repeat
the same solution under three seed labels. Legacy-v1 and direct-v2 keep their
historical stochastic heads and loading paths. All model choices are confined to
inner source validation; outer-test scores never choose a penalty or checkpoint.
''')
    cells += [md('''
Four ordered bins `[0,8)`, `[8,16)`, `[16,24)`, `[24,32)` preserve temporal position.
Each of 33 joints contributes mean valid x/y, valid adjacent-frame velocity x/y,
confidence, frame support and transition support: 924 fixed features. Missing
observations are imputed from training means only. Unsupported or near-constant
training columns become exact zeros in every partition, with their positions
retained. Target scaling is a separate operation and keeps target variation.

| Arm | Transformation before feature construction |
|---|---|
| Real | Original coordinate/confidence/validity history |
| Time shuffle | Permute four-frame blocks, moving all channels together |
| Clip mismatch | Different-source, context-matched donor inside the current partition |
| No skeleton | Zero x/y/confidence; retain the original time-varying validity |

All arms receive 36 pairs from the same positive RGB/skeleton penalty grids plus
the same selected RGB-only baseline. Inner loss pools source-weighted error sums
and weight totals. Ties within the frozen numerical tolerance prefer baseline;
other ties prefer stronger penalties. Failed candidates stay visible and prevent
a complete scientific result. The exact fallback can win every comparison.
'''), code('''
if MODE=='teach':
    from gavd6_sjepa.research_directions.future_innovation.fi_joint_calibration import exact_calibration
    from gavd6_sjepa.research_directions.future_innovation.fi_joint_models import JointRidge
    demo=exact_calibration()
    display(pd.DataFrame([demo]))
    rng=np.random.default_rng(41)
    x=rng.normal(size=(20,4)); s=rng.normal(size=(20,2)); y=x[:,:1]+s[:,:1]
    joint=JointRidge.fit(x,s,y,np.ones(20),10.,10.)
    display(pd.DataFrame({'illustrative_target':y[:,0],'training_prediction':joint.predict(x,s)[:,0]}))
    print('Synthetic training illustration only; real calibration uses complete source-held selection.')
'''), md('''
The historical all-candidates-loss problem is visible in each inner partition,
not just its pooled score. Large losses should stay in the numerical table even
when a logarithmic plot is easier to read. Direct-v3 train and validation losses
come from the same closed-form fit within each inner partition. Historical
neural training loss was recorded before an update and validation afterward;
new timing labels describe that difference. It was not the headline R² cause.

Inspect supported feature counts alongside nominal dimensions. Baseline-only
uses no skeleton coefficients. A joint candidate that wins inner validation can
still lose on unseen sources; fallback eligibility does not guarantee outer gain.
'''), code('''
if MODE!='teach':
    from gavd6_sjepa.research_directions.future_innovation.fi_tutorial_inspection import prediction_fit_tables
    from gavd6_sjepa.research_directions.future_innovation.fi_contracts import read_json
    path=RUN_ROOT/'config/model-contract.json'
    if path.is_file(): display(read_json(path))
    tables=prediction_fit_tables(RUN_ROOT,os.environ.get('FI_NOTEBOOK_FOLD'))
    for name,table in tables.items():
        if not table.empty:
            print(name, f'({len(table)} records; full machine-readable records under models/fold-*/.)')
            with pd.option_context('display.max_rows',None,'display.max_columns',None,'display.precision',10):
                display(table)
    candidates=tables['candidates']
    if not candidates.empty:
        selected=tables['selected']
        print('Baseline fallback frequency:', (selected.selected_type=='baseline_only').sum(), '/',len(selected))
        fig,ax=plt.subplots(figsize=(9,3))
        for arm,rows in candidates.groupby('arm'):
            ax.scatter(np.arange(len(rows)), rows.pooled_loss, s=12, alpha=.55, label=arm)
        ax.set(yscale='log',xlabel='Candidate record within arm',ylabel='Pooled inner MSE',title='All candidate losses; numeric values retained above')
        ax.legend(); plt.show()
    elif not tables['historical_inner_fits'].empty:
        history=tables['historical_inner_fits']
        fig,ax=plt.subplots(figsize=(9,3))
        ax.scatter(np.arange(len(history)),history.validation_squared_error/history.validation_weight,s=8)
        ax.set(yscale='log',xlabel='Historical inner fit record',ylabel='Validation MSE'); plt.show()
    display(artifact_inventory(RUN_ROOT))
''')]
    return cells+ending('03','Every selected model has an inner-validation reason, a fitted preprocessing record and an explicit type. Next reconstruct its held-out predictions and assess the paired effect and uncertainty.')


def lesson04():
    cells=opening('04','Measure the increment and decide what follows','''
A valid negative result is useful. A missing prediction, a corrupted mask and a
complete failed effect criterion are different outcomes. Execution reconstructs
and scores the saved models before sealing a result. Inspect mode reads the
saved evidence without rewriting it.
''')
    cells += [md('''
Pool held-out predictions over the five folds. For each target feature, R² is
one minus source-weighted squared prediction error divided by the error of its
outer-training mean. Average featurewise R² on the intersection of training-valid
dimensions. Stochastic historical runs then average seed scores; deterministic
direct-v3 has one score. Neither fold-score averaging nor test-mean centering
implements this metric.

The 2,000 paired bootstrap draws resample whole source videos with replacement.
Keep clips together and count a source twice when drawn twice. The same draws
serve every arm. These intervals condition on saved models and omit repeated
fitting, selection and this cohort's adaptive redesign. They add no independent
sources. Report the 95% interval separately from the 90%-positive decision rule.

| Required criterion | Threshold |
|---|---|
| Mean real gain over shared RGB ridge | at least +0.05 R² |
| Shuffle | real gain ≥ 2 × max(shuffled gain, 0) |
| Mismatch | gain ≤ +0.01 R² |
| Matched skeleton increment | real minus no-skeleton > 0 |
| Paired bootstrap | real gain and matched increment each positive in ≥90% of draws |
| Measurement evidence | all required input, target, audit, source, model and numerical checks valid |

Direct-v3's prospective deterministic policy makes stochastic seed stability
inapplicable. Historical stochastic runs still require all three gains positive,
at least two ≥0.05, and matched increment positive in every seed. Point failures
produce complete STOP. Passed points with insufficient stability produce
INCONCLUSIVE. A passing direct-v3 result is development ADVANCE and requires
independent-source confirmation before scaling.
'''), code('''
if MODE=='teach':
    from gavd6_sjepa.research_directions.future_innovation.fi_joint_reporting import decide_joint_gate
    metrics={'delta_r2_real':.06,'delta_r2_time_shuffle':.02,'delta_r2_clip_mismatch':.005,'delta_r2_no_skeleton':.005,
             'seed_real_gains':[.06],'seed_skeleton_increments':[.055],
             'bootstrap_positive_fraction':.95,'skeleton_increment_positive_fraction':.95,
             **{k:True for k in ('data_contract_valid','evaluation_contract_valid','controls_complete','input_audit_complete',
                                  'target_variance_valid','teacher_stable','causal_leakage_absent')}}
    cases={'Invented passing development evidence':metrics,
           'Invented uncertainty':{**metrics,'bootstrap_positive_fraction':.7},
           'Exact fallback / zero increment':{**metrics,'delta_r2_real':0.,'delta_r2_no_skeleton':0.,'seed_real_gains':[0.],'seed_skeleton_increments':[0.]}}
    display(pd.DataFrame([{'synthetic illustration':name,**{k:decide_joint_gate(m)[k] for k in ('decision','allow_full_experiment','stochastic_seed_stability')}} for name,m in cases.items()]))
'''), code('''
if MODE!='teach':
    evidence=inspect_report(RUN_ROOT)
    print(evidence['state'],evidence['explanation'])
    print('Report seal integrity checked:',evidence['seal_verified'])
    for path in ['reports/aggregate-metrics.csv','reports/paired-controls.csv']:
        table=read_optional_table(RUN_ROOT,path)
        if table is not None:
            with pd.option_context('display.precision',10,'display.max_columns',None): display(table)
    from gavd6_sjepa.research_directions.future_innovation.fi_contracts import read_json
    for path in ['reports/uncertainty.json','reports/numerical-verification.json']:
        p=RUN_ROOT/path
        if p.is_file():
            print(path,'— saved evidence; numerical reconstruction is not rerun in inspect mode')
            display(read_json(p))
    aggregate=read_optional_table(RUN_ROOT,'reports/aggregate-metrics.csv')
    if aggregate is not None:
        points=aggregate.groupby('arm',sort=False).delta_r2.mean()
        fig,ax=plt.subplots(figsize=(9,3)); ax.barh(points.index,points.values,color='#2f6f99')
        ax.axvline(0,color='black',linewidth=.7); ax.axvline(.05,color='#d4803f',linestyle='--',label='Required real gain')
        ax.set(xlabel='Gain over shared RGB reference (R²)'); ax.legend(); plt.show()
    if evidence['report_text'] is not None: display(Markdown(evidence['report_text']))
'''), md('''
A negative repaired result concerns this predictor, these ordered summaries,
the contextual teacher target and the small data regime. A separately specified
representation, target or data study may be reasonable, but this result does
not justify increasing student capacity by itself. A positive development result
requires independent-source confirmation. Neither outcome trains S-JEPA,
adapters or a skeleton-only distilled student; those remain subsequent experiments.
''')]
    return cells+ending('04','Record the exact run, measured effect, uncertainty and evidence status. A complete negative measurement closes this comparison; an incomplete result identifies required recovery work.')


LESSONS = {"00": lesson00, "01": lesson01, "02": lesson02, "03": lesson03, "04": lesson04}


def render(number):
    notebook = nbformat.v4.new_notebook(cells=LESSONS[number]())
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "editable_source": BUILDER,
        "evidence_boundary": "Teach is non-evidentiary; inspect is read-only; execute runs the existing Experiment 0 CLI.",
    })
    for index, cell in enumerate(notebook.cells):
        cell.id = hashlib.sha256(f"{number}:{index}:{cell.source}".encode()).hexdigest()[:12]
        if cell.cell_type == "code":
            compile(cell.source, f"{NAMES[number]}:cell-{index}", "exec")
    nbformat.validate(notebook)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=tuple(NAMES), nargs="+", default=list(NAMES))
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check", action="store_true")
    checks.add_argument("--check-sources", action="store_true")
    args = parser.parse_args()
    for number in args.only:
        expected = render(number)
        path = DESTINATION / NAMES[number]
        if args.check or args.check_sources:
            actual = nbformat.read(path, as_version=4)
            if args.check_sources:
                comparable = lambda nb: [(cell.cell_type, cell.source) for cell in nb.cells]
            else:
                comparable = lambda nb: json.loads(nbformat.writes(nb))
            if comparable(actual) != comparable(expected):
                raise SystemExit(f"Notebook differs from its builder: {path}")
            print(f"Checked {path.name}")
        else:
            if path.exists() and any(c.get("outputs") for c in nbformat.read(path, as_version=4).cells):
                raise SystemExit(f"Preserve executed {path.name} outside the source tree before regeneration.")
            path.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(expected, path)
            print(f"Built {path.name}")


if __name__ == "__main__":
    main()
