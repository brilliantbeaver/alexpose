"""Generate source tutorials 19–22; executed copies belong in work/artifacts.

Default execution inspects sealed evidence and runs explicitly synthetic examples.
It never starts a new real-data comparison, encodes video, or writes into
historical runs. Notebook 21 optionally refits saved CPU models to verify them.
"""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / 'notebooks/iclr_bridge'


def md(text): return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())
def code(text): return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


SETUP = code('''
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from IPython.display import display, SVG, Markdown
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src/gavd6_sjepa').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
EVIDENCE = ROOT / 'work/artifacts/iclr-bridge-2026-09-11'
PANEL = ROOT / 'outputs/iclr-bridge-cached-20260911'
def read_json(path):
    return json.loads(Path(path).read_text())
pd.set_option('display.precision', 8)
print('Repository:', ROOT)
print('Execution: read-only evidence inspection + labeled synthetic calculations')
''')


def figure(name):
    return code(f"display(SVG(filename=str(ROOT / 'docs/studies/iclr/figures/{name}.svg')))")


def evidence_notebook():
    return [md('''
    # 19 — Evidence and observability: what did the two studies establish?

    This tutorial continues the external laterality notebooks 00–18. It lives in
    the current S-JEPA repository so the historical notebooks remain unchanged.
    The objective is to turn two difficult results into a more precise research
    question. We will distinguish information that a model contains, information
    that a particular readout can recover, and information a student can use at
    inference. These are related quantities, but none implies the others.

    **By the end:** you will be able to explain the laterality readout decline,
    the completed Experiment 0 STOP, and why neither directly answers whether
    selected video supervision improves a skeleton-only student. Run cells in
    order. No training or teacher extraction starts from this notebook.
    '''), SETUP, figure('01_evidence_bridge'), md('''
    ## 1. Establish the evidence level before reading a number

    A notebook can contain a historical result in Markdown without having been
    executed in its current form. A file can exist locally without its identity
    or arithmetic having been verified. The inventory below records those
    distinctions. Laterality's raw predictions and checkpoints are absent from
    this copied checkout. Its seed aggregates can be checked; new inference and
    a new source bootstrap cannot be reconstructed here. The repaired future
    experiment does have cached arrays, selected models and numerical verification.

    The inventories retain every relevant code/Markdown cell, execution count
    and output text. Counts describe notebook files, not independent experiments.
    Historical copies and source tutorials sometimes describe the same fit.
    '''), code('''
    later_inventory = read_json(EVIDENCE / 'laterality/notebook_inventory.json')
    future_inventory = read_json(EVIDENCE / 'future/notebook-inventory.json')
    print('Laterality inventory type:', type(later_inventory).__name__)
    print('Future inventory type:', type(future_inventory).__name__)
    print('Read the complete notebook maps:')
    display(Markdown('[Laterality audit](../../work/artifacts/iclr-bridge-2026-09-11/laterality/evidence-and-critical-analysis.md)'))
    display(Markdown('[Future / S-JEPA audit](../../work/artifacts/iclr-bridge-2026-09-11/future/evidence-audit.md)'))
    ''') , md('''
    ## 2. Read the laterality comparison within its own protocol

    The observable averages normalized differences between left and right
    median coordinate speeds for five landmark pairs. Both sides must be valid
    at both ends of a transition. It measures an image-normalized movement
    contrast, not clinical impairment or loading. Reflection swaps anatomical
    sides, so it reverses the sign. Time reversal leaves speed magnitude unchanged.

    The main masking grid used 625 clips from 93 recording sources, five outer
    folds and five seeds. Initialization scores 0.222544 with the expanded
    readout. All five trained-teacher arms score lower. The following arithmetic
    was recomputed from retained seed records. Its intervals are historical
    saved intervals; the current audit did not regenerate their bootstrap draws.
    '''), code('''
    later = read_json(EVIDENCE / 'laterality/aggregate_recomputation.json')
    rows = [{k: r.get(k) for k in ('label', 'recomputed_mean_r2', 'mean_difference', 'retained_ci95')}
            for r in later['checks']]
    display(pd.DataFrame(rows))
    print('Audit status:', later['status'])
    print('Verification scope:', later['scope'])
    assert later['raw_artifacts_present'] is False
    '''), md('''
    ## 3. Separate a sound negative result from an invalid experiment

    Experiment 0 originally had an unsafe input scaler and an unrestricted second
    RGB mapping. The repaired joint linear model uses separate penalties and
    can select an exact baseline. The repair makes a negative answer possible
    without being evidence of a software failure.

    On 50 clips from 43 sources, the shared RGB reference scores 0.33421517 and
    real skeleton scores 0.33397275. Real minus validity-only no-skeleton is
    −0.00024242 R², with paired 95% source interval [−0.00147402, +0.00091980].
    Fourteen of twenty fold/arm fits select baseline. This is a **complete
    development STOP** for a particular predictor and contextual target: the
    pipeline completed validly, but its scientific advance criteria did not pass.
    It is not proof that skeleton-only distillation is impossible.
    '''), code('''
    direct = ROOT / 'outputs/future-innovation-direct-v3-dev-20260911'
    report_files = sorted(p.name for p in (direct / 'reports').glob('*.json'))
    display(pd.DataFrame({'report_present_locally': report_files}))
    display(pd.DataFrame([
        ['RGB + nuisance', .33421516826816244],
        ['Real skeleton', .33397275078231303],
        ['Time shuffle', .334246668285791],
        ['Clip mismatch', .33545259363857877],
        ['No skeleton (validity retained)', .33421516826816244],
    ], columns=['direct-v3 arm', 'predictive_R2']))
    print('Table values are retained direct-v3 results; file presence alone is not verification.')
    '''), md('''
    ## 4. Why the two R² values cannot be ranked together

    R² compares prediction error with a specified reference error. The laterality
    evaluation uses the evaluated weighted target mean. Future innovation uses
    each outer-training target mean. In standardized target units that training
    mean is zero. Test-mean centering would silently change the experiment.

    The synthetic example makes this visible: targets shift from a training
    mean of zero to 10 and 12. A constant prediction of 11 beats the historical
    training mean strongly, but offers no reduction relative to the test mean.
    Neither denominator is universally correct; preserving the declared one is.
    '''), code('''
    # SYNTHETIC arithmetic example, not a GAVD result.
    y = np.array([10., 12.]); prediction = np.array([11., 11.])
    mse = np.mean((y - prediction)**2)
    display(pd.DataFrame([
        ['training mean 0', 1 - mse / np.mean(y**2)],
        ['test mean 11', 1 - mse / np.mean((y-y.mean())**2)],
    ], columns=['reference', 'R2']))
    '''), md('''
    ## 5. Identify the missed experiment

    RGB complementarity asks what skeletons add after an already informative
    video representation. A skeleton student only receives its pose history.
    Therefore a useful teacher component can be redundant with RGB and still
    be valuable supervision for the student. Conversely, a component useful
    only through an RGB–skeleton interaction may be inaccessible from skeletons
    alone. Notebook 22 constructs both cases explicitly.

    The next cached comparison replaces RGB with a declared student-accessible
    reference: current posture, confidence, validity and frame rate. This tests
    a different question and has a separate frozen protocol. Its primary effect
    is real history minus a matched validity-only history. A negative answer
    remains useful because it discourages further investment in that target
    and representation before evaluating alternatives.

    Continue with **20** for exact symmetry and temporal-order examples, **21**
    for the actual cached comparison, and **22** for a selective-target research
    design. The full critique and step-by-step research plan are in
    [the tutorial](../../docs/studies/iclr/01_critique_and_research_tutorial.md).
    ''')]


def symmetry_notebook():
    return [md('''
    # 20 — Reflection, time direction and observable motion

    This notebook makes the laterality study's transformation laws executable.
    We use **synthetic trajectories throughout**. An observable is a declared
    numerical function of measured movement. Equivariance specifies how that
    value changes under a transformation; it does not guarantee the value is
    present in a learned representation.

    Reflection M swaps anatomical left/right landmarks and negates centered
    horizontal coordinates. It moves confidence and validity with the joints.
    Reversal T reverses the trajectory and its sequence of physical time intervals.
    We will test M² = T² = identity and MT = TM, then show why a speed target
    cannot by itself establish that a model understands temporal order.
    '''), SETUP, figure('03_reflection_and_time'), md('''
    ## 1. Run the fixed calibration

    The fixture's seed, source allocation and tolerances are declared in its
    module before computation. Its algebra checks use a 10⁻¹² tolerance.
    The source-held prediction example uses 48 generated sources, each with
    two opposite-order trajectories; 36 sources train and 12 test. This is a
    controlled example with known structure, not a sample of real people.

    Calling `run_calibration()` without an output path has no filesystem writes.
    Every result below is freshly calculated. The saved JSON in the audit folder
    records the same fixture together with predictions and fit identities.
    '''), code('''
    from gavd6_sjepa.research_directions.iclr_bridge.symmetry_calibration import run_calibration
    calibration = run_calibration()
    assert calibration['status'] == 'passed'
    display(pd.DataFrame([calibration['group_errors']]))
    display(pd.DataFrame(calibration['observable_parities']))
    '''), md('''
    ## 2. Interpret the four transformation classes

    Total speed stays unchanged under reflection and reversal. The bilateral
    speed contrast changes sign under reflection but is unchanged by reversal.
    Signed vertical displacement changes under reversal; averaging the two
    sides makes it reflection-even, while contrasting sides makes it
    reflection-odd. Here “vertical” is the image-coordinate y direction,
    not a calibrated gravity vector.

    Speeds use only valid adjacent observations. With an occluded gap, subtracting
    the last visible coordinate from a later coordinate would be a different
    measurement. Reversal must reverse unequal time intervals too; otherwise
    it changes speed estimates and invalidates the intended parity test.

    Biological dynamics need not be reversible. T is a mathematical diagnostic
    on a bounded observed sequence. It does not justify putting future frames
    into a deployed prefix encoder or treating reverse videos as valid forecasts.
    '''), md('''
    ## 3. Exact geometry can coexist with no information

    For a feature function h in a shared coordinate basis, its parity component
    is P_ab h(x) = [h(x) + a h(Mx) + b h(Tx) + ab h(MTx)] / 4, with signs
    a and b each ±1. This ordinary finite-group projection enforces the desired
    sign laws. It is not a novel learning theorem. A constant zero feature
    satisfies every odd transformation constraint perfectly.

    Thus a symmetry penalty or exact architecture needs two separate checks:
    whether its output transforms correctly, and whether that output predicts
    a nontrivial independent observable. Energy or rank alone cannot replace
    the second check; random features can have both.
    '''), code('''
    display(pd.DataFrame(calibration['projector_checks']))
    display(pd.DataFrame([calibration['zero_feature_control']]))
    '''), md('''
    ## 4. Create an ambiguity that genuinely requires history

    Each generated source supplies a path and its reversal. The paths end at
    the same current posture and have the same distribution of coordinates,
    confidence and support. Their last velocities have opposite signs. We
    define a synthetic one-step continuation using that last velocity. This
    continuation rule is part of the fixture, not an assumption about humans.

    Means, standard deviations and absolute adjacent changes are identical
    within each pair. A model using only those summaries cannot tell the two
    futures apart. Ordered signed velocities can. This explains why the
    laterality expanded-summary gain cannot be called proof of temporal-order
    learning: several features it adds are invariant to reversal.
    '''), code('''
    import matplotlib.pyplot as plt
    temporal = calibration['temporal_observability']
    wave = np.asarray(temporal['illustrative_wave'])
    times = np.asarray(temporal['illustrative_times'])
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(times, wave, '-o', label='Generated history A', color='#087F8C')
    ax.plot(times, wave[::-1], '-s', label='Reversed history B', color='#7953A5')
    ax.set(xlabel='Synthetic time (seconds)', ylabel='Synthetic y coordinate',
           title='Same endpoint; opposite final velocity')
    ax.legend(); ax.grid(alpha=.2); plt.show()
    display(pd.DataFrame([{k:r[k] for k in ('features','r2_training_mean_reference','nominal_features','supported_features')}
                          for r in temporal['scores']]))
    assert set(temporal['train_sources']).isdisjoint(temporal['test_sources'])
    '''), md('''
    ## 5. Transfer the lesson to the real experiment without transferring the result

    The fixture shows that the selected readout can recover a declared signal
    and that order-even summaries miss it by construction. The cached GAVD
    panel uses its existing 32-frame, four-bin representation and its designated
    four-frame block shuffle, not this eleven-frame teaching fixture. Its
    production-path planted-history test separately exercises the actual finite
    search. Neither calibration predicts the sign of the real-data result.

    A useful real extension will measure both the existing reflection-odd,
    time-even contrast and a time-odd future observable. It will match current
    posture and support, include motion-direction controls, and test an actual
    student's held-source readout. For a video teacher, paired reflection
    components require encoding two real videos in a common feature basis.
    Existing pooled vectors cannot be “mirrored” by swapping skeleton columns.

    Continue to **21** to inspect the new real cached comparison and to **22**
    for the target-selection and distillation design.
    ''')]


def panel_notebook():
    return [md('''
    # 21 — Student-accessible prediction of contextual teacher features

    This notebook inspects the separately frozen `student-accessibility-v1`
    experiment. It answers a limited but necessary question: does ordered
    skeleton history improve prediction of the current cached target after
    observation quality and current posture are represented? It does not train
    an S-JEPA encoder or demonstrate distillation.

    By default, Run All performs CPU reconstruction refits of the saved selected
    models without changing any artifact. Set RECONSTRUCT_MODELS=False in section
    2 for strictly no-fit file-integrity inspection. A missing or altered run
    causes an error; there is no synthetic replacement for real results. Starting
    a new comparison is an explicit CLI operation. A new target or changed search
    requires a new run root and prospective protocol.
    '''), SETUP, figure('02_information_boundary'), md('''
    ## 1. Confirm the frozen question and lineage

    Both panels retain the original 50 windows, 43 sources, five outer folds,
    three inner folds, 256-dimensional target and teacher projection. The
    teacher saw all 64 frames; target tokens at 38–39 are contextual features.
    The first 32 skeleton frames are the only history input. Source separation
    controls recordings, not necessarily people appearing in different uploads.

    The support reference contains four bins of confidence/validity, endpoint
    confidence/validity and frame rate. The posture reference adds valid
    frame-31 coordinates. The parent normalization uses a prefix-derived torso
    scale, so this is a declared function of history rather than an independently
    measured single frame. That qualification matters when interpreting “current.”
    '''), code('''
    spec = read_json(PANEL / 'config/specification.json')
    report = read_json(PANEL / 'reports/panel-report.json')
    contract = read_json(PANEL / 'config/run-contract.json')
    display(pd.DataFrame([{'version':spec['version'], 'clips':report['clips'],
                          'sources':report['sources'], 'seed_policy':spec['seed_policy'],
                          'teacher_evidence':contract['teacher_evidence']}]))
    display(pd.DataFrame({'panel':list(spec['baseline_dimensions']),
                          'reference_features':list(spec['baseline_dimensions'].values())}))
    print('Primary:', spec['primary_contrast'])
    print('Created UTC:', contract['created_utc'])
    '''), md('''
    ## 2. Reconstruct before interpreting

    The verifier checks source/cache snapshots, feature schemas, train-only
    preprocessing, controls and typed models. It refits the selected linear
    solutions, reconstructs each held-out prediction in standardized and raw
    units, and recomputes scores and all paired bootstrap draws. Candidate
    records are checked for pooling and selection; all rejected candidates
    are not independently refitted by this command. That is its stated boundary.

    Verification is read-only. It requires the compatible frozen implementation
    and intact parent artifacts. Passing the following cell is stronger than
    seeing a checkpoint count or a correct checksum.
    '''), code('''
    from gavd6_sjepa.research_directions.iclr_bridge.verification_supplement import verify_panel_supplement
    from gavd6_sjepa.research_directions.iclr_bridge.inspection import inspect_cached_panel
    RECONSTRUCT_MODELS = True  # False checks lineage/digests only, with strictly no fitting.
    verification = verify_panel_supplement(PANEL) if RECONSTRUCT_MODELS else inspect_cached_panel(PANEL)
    display(pd.DataFrame([verification]))
    if RECONSTRUCT_MODELS:
        assert verification['status'] == 'passed'
        assert verification['original_verification']['parent_artifacts_unchanged']
    else:
        assert verification['status'] == 'integrity_checked'
        print('No numerical reconstruction performed in this mode.')
    '''), md('''
    ## 3. Understand what was fitted and selected

    Each reference has six ridge penalties. Each arm has 36 pairs of positive
    penalties for the reference and history blocks, plus one exact baseline.
    Training weights give each source equal total influence. The intercept is
    unpenalized. Inner validation chooses a model; the outer sources evaluate
    that choice. A nonzero winner can still perform worse on outer sources.

    No-skeleton retains time-varying validity. Shuffle moves coordinates,
    confidence and validity together in four-frame blocks. Mismatch uses a
    different-source donor from the same partition and never uses targets.
    All arms retain the recipient's reference inputs. Supported feature counts
    can differ after controls, so nominal parameter counts are only one part
    of the comparison.

    The real history block also supplies validity-conditioned confidence means;
    the reference uses raw confidence means. These are different features under
    missingness. Real minus no-skeleton therefore tests a coordinate/confidence
    bundle, not isolated motion. A subsequent confidence/validity-only history
    arm would control that route more tightly. In two posture folds, real also
    selected a different reference-block penalty. The shared-baseline gain
    includes this regularization change.
    '''), code('''
    selections = pd.read_csv(PANEL / 'reports/selections.csv')
    display(pd.crosstab([selections.panel, selections.arm], selections.type))
    candidate_rows, diagnostic_rows = [], []
    for panel in spec['panels']:
        for fold in range(5):
            folder=PANEL / f'models/{panel}/fold-{fold}'
            for arm, record in read_json(folder/'selection.json').items():
                for candidate in record['candidates']:
                    candidate_rows.append({'panel':panel,'fold':fold,'arm':arm,
                        **{k:candidate[k] for k in ('candidate_id','pooled_loss','improvement_over_baseline','valid','selected','selection_reason')}})
            diagnostic_rows.extend({'panel':panel,'fold':fold,**r} for r in read_json(folder/'diagnostics.json'))
    candidates=pd.DataFrame(candidate_rows)
    diagnostics=pd.DataFrame(diagnostic_rows)
    print('Pooled candidates:',len(candidates),'valid:',int(candidates.valid.sum()))
    display(candidates[candidates.selected])
    display(diagnostics[['panel','fold','arm','x_features','x_supported','s_features','s_supported','training_mse','selected_inner_mse']])
    '''), md('''
    ## 4. Read the matched effect and its uncertainty

    All scores use source-balanced featurewise predictive R² relative to the
    outer-training mean. The declared primary contrast is posture-panel real
    minus no-skeleton. Other contrasts help explain the result but cannot
    replace the primary after inspection. Displaying eight decimal places
    preserves the sign and scale of small effects.

    The 2,000 bootstrap draws resample whole sources with replacement and keep
    repeated-source multiplicities. They reuse fitted predictions; they do not
    repeat fitting or model selection. Thus these are conditional development
    intervals, not independent replication. This deterministic predictor has
    one result, even though the artifact identity uses seed 0.
    '''), code('''
    score_rows, contrast_rows = [], []
    for panel,p in report['panels'].items():
        score_rows += [{'panel':panel,'arm':arm,**scores} for arm,scores in p['scores'].items()]
        contrast_rows += [{'panel':panel,'contrast':name,**values} for name,values in p['contrasts'].items()]
    display(pd.DataFrame(score_rows))
    display(pd.DataFrame(contrast_rows))
    print('Measurement complete:',report['measurement_complete'])
    print('Frozen descriptive status:',report['status'])
    print('Scientific advance:',report['scientific_advance'])
    '''), md('''
    ## 5. Decide what follows, without changing the rule

    The frozen descriptive `development_lead` requires the primary 95% interval
    to exclude zero positively and real to beat shuffle and mismatch in point
    estimates. Otherwise a complete run reports `no_supported_temporal_lead`.
    An invalid required comparison reports `incomplete_measurement`. Neither
    descriptive outcome changes the earlier direct-v3 STOP, and neither can
    authorize a paper claiming successful student transfer.

    If there is a lead, freeze a future-only, physical-time target and one
    student-training comparison before collecting confirmation data. If there
    is no lead, first examine target meaning and direct pose prediction, then
    choose a separately specified representation or target study. Repeatedly
    replacing fifty videos until an effect appears would confound model
    development with confirmation.

    ## 6. Explicit reproduction and recovery

    The following commands are shown as text. Running this notebook does not
    execute them. `run` resumes verified completed folds and verifies a sealed
    completed stage. `freeze` requires a new directory. A protocol or code
    amendment belongs in a separate experiment.

    ```bash
    .venv/bin/python scripts/research_directions/iclr_bridge/run_cached_panel.py run --output-root outputs/iclr-bridge-cached-20260911
    .venv/bin/python scripts/research_directions/iclr_bridge/run_cached_panel.py verify --output-root outputs/iclr-bridge-cached-20260911
    ```

    The [frozen protocol](../../docs/studies/iclr/02_cached_panel_protocol.md) specifies
    the construction and the [validation report](../../docs/studies/iclr/03_implementation_and_validation.md)
    records what actually ran. Continue with notebook **22** to design the
    next target and student study.
    ''')]


def distillation_notebook():
    return [md(r'''
    # 22 — Selective future distillation: a falsifiable next study

    This tutorial develops the most promising research extension while keeping
    its implementation status explicit. The cached comparison is real; the
    calculations below are **synthetic teaching examples**. No video teacher is
    loaded and no real skeleton student is trained here. The goal is to define
    a method whose assumptions can be challenged before an expensive run.

    Let H be all permitted student history and C a declared function of H
    containing current state and observation quality. For a teacher target Y,
    define population temporal innovation as
    $u(H)=E[Y\mid H]-E[Y\mid C]$.
    This is information accessible to the student beyond its current reference.
    It is different from the residual of a predictor that can see RGB.
    '''), SETUP, figure('05_matched_pose_futures'), md(r'''
    ## 1. Derive the quantity before constructing an estimator

    Under finite second moments, common squared-error units, and C measurable
    from H, the tower property gives $E[u\mid C]=0$. Orthogonality gives
    $E\|Y-E[Y\mid C]\|^2-E\|Y-E[Y\mid H]\|^2=E\|u\|^2$.
    This is standard conditional-expectation algebra, not a new theorem.
    A finite ridge family estimates neither expectation exactly. Its held-source
    performance difference is a model comparison, not conditional mutual
    information or proof that the entire skeleton contains no extra signal.

    In practice, target units, dimension weights and source weights must remain
    common when comparing predictors. Fold-specific target normalization is
    appropriate for the declared score but must be inverted before combining
    residual arrays from different subfits in one learned target basis.
    '''), md('''
    ## 2. Show why RGB complementarity is neither necessary nor sufficient

    We construct independent signs R and S with all four combinations equally
    likely. A “student” observes S and an “RGB reference” observes R. In the
    shared-signal case RGB also contains S and the target equals S: adding S
    after RGB gives zero improvement, yet the student predicts perfectly.
    In the interaction case Y=R×S: the joint observations determine Y, but S
    alone has conditional mean zero. This second example concerns rich
    predictors that can represent the interaction; the current linear panel
    does not claim to estimate it.
    '''), code('''
    # Exact finite probability space; no fitted GAVD model is involved.
    R=np.array([-1.,-1.,1.,1.]); S=np.array([-1.,1.,-1.,1.])
    def mse(y,p): return float(np.mean((y-p)**2))
    shared=S; interaction=R*S
    examples=pd.DataFrame([
        {'example':'Shared student signal', 'RGB_reference_MSE':mse(shared,S),
         'joint_MSE':0., 'student_only_MSE':mse(shared,S)},
        {'example':'Pure interaction', 'RGB_reference_MSE':mse(interaction,np.zeros(4)),
         'joint_MSE':0., 'student_only_MSE':mse(interaction,np.zeros(4))},
    ])
    examples['gain_after_RGB']=examples.RGB_reference_MSE-examples.joint_MSE
    display(examples)
    assert examples.loc[0,'gain_after_RGB']==0 and examples.loc[0,'student_only_MSE']==0
    assert examples.loc[1,'gain_after_RGB']==1 and examples.loc[1,'student_only_MSE']==1
    '''), md('''
    ## 3. Build a genuine paired teacher target

    Reflection-even and reflection-odd pooled components are (Y+Y_M)/2 and
    (Y−Y_M)/2. We must actually encode both original and mirrored RGB videos
    using the same model, temporal boundaries and projection. If spatial tokens
    are used, their positions need a declared alignment. Mirroring a pose
    array does not produce Y_M. The helper below validates array arithmetic,
    identities and an explicit shared basis; it does not verify external video
    files or invent missing teacher evidence.

    A teacher's odd component might reflect camera layout, clothing or readable
    text rather than anatomical motion. Laterality probes and nuisance controls
    are therefore needed even when the decomposition is exact.
    '''), code('''
    from gavd6_sjepa.research_directions.iclr_bridge.symmetry_calibration import paired_teacher_components
    rng=np.random.default_rng(2201)
    original=rng.normal(size=(8,6)); mirrored=rng.normal(size=(8,6))
    pair=paired_teacher_components(original,mirrored,window_ids=[f'toy_{i}' for i in range(8)],
        original_evidence_id='synthetic_original_2201',mirrored_evidence_id='synthetic_mirror_2201',
        shared_basis_id='synthetic_basis_6',evidence_kind='synthetic_teaching')
    np.testing.assert_allclose(pair['even']+pair['odd'],original)
    np.testing.assert_allclose(pair['even']-pair['odd'],mirrored)
    display(pd.DataFrame([pair['provenance']]))
    '''), md('''
    ## 4. Select directions for temporal accessibility, not merely large variance

    The proposed method learns teacher directions using only current training
    sources. Within that partition, fit current-state and history predictors,
    construct candidate directions from cross-fitted raw-unit predictions or
    a jointly regularized model, and evaluate their extra predictable error
    reduction on source-held inner data. Include rank zero, meaning no transfer.
    Refit the chosen construction using outer-training sources and test it
    once on outer sources. Candidate ranks, penalties and target families must
    be finite and fixed before fitting. Do not rank directions by outer scores.

    Large teacher variance is insufficient: a background-sensitive direction
    can vary strongly while being unpredictable from a skeleton. Conversely,
    a low-variance directional-motion feature may be useful. The following
    exact synthetic table separates those properties. It illustrates the
    selection criterion; it does not implement a validated real target selector.
    '''), code('''
    # Enumerate all independent signs: appearance A, current state C and history V.
    import itertools
    A,C,V=np.asarray(list(itertools.product([-1.,1.],repeat=3))).T
    Y=np.column_stack([10*A, C, .5*V])
    pred_current=np.column_stack([np.zeros(8),C,np.zeros(8)])
    pred_history=np.column_stack([np.zeros(8),C,.5*V])
    gain=np.mean((Y-pred_current)**2,axis=0)-np.mean((Y-pred_history)**2,axis=0)
    display(pd.DataFrame({'teacher_direction':['appearance','current posture','history velocity'],
        'variance':Y.var(axis=0),'extra_predictable_MSE_reduction':gain}))
    selected=np.flatnonzero(gain>0)
    null_selected=np.flatnonzero(np.zeros_like(gain)>0)
    print('Oracle teaching selection:',selected.tolist(),'rank-zero control:',null_selected.tolist())
    assert selected.tolist()==[2] and len(null_selected)==0
    '''), md('''
    ## 5. Make the distillation comparison capable of disproving the idea

    The actual student experiment must compare the same encoder, training
    sources, optimization budget and evaluation decoder under: ordinary S-JEPA,
    full-feature video distillation, selected-target distillation, rank-matched
    random or variance-selected targets, and no transfer. Include matched
    initialization and direct kinematics. If a student simply copies an
    already sufficient handcrafted forecast, it has not demonstrated added
    teacher knowledge. A common independent future-motion endpoint is needed;
    lower loss on a smaller or easier selected target is not a fair success metric.

    Use a future block encoded without the observed prefix when claiming
    future-only supervision. Set horizons in physical time and audit each
    encoder's temporal receptive field. Preserve both an anatomical time-even
    probe and a time-odd probe. Test camera/view changes, missing joints and
    matched current posture. Keep all augmented or paired clips from one source
    on the same side of every split; participant separation requires identities
    or a dataset that supplies them.

    A minimal study first tests one horizon, one frozen teacher layer, one
    candidate family and one independent endpoint. Expand only after this
    bounded comparison reveals an interpretable effect. Current caches lack
    future pose and transformed video encodings, so these real experiments
    remain explicit next steps rather than disabled cells masquerading as results.
    '''), figure('04_research_workflow'), md('''
    ## 6. Position the contribution honestly

    Future privileged supervision and spectral target selection already exist:
    see [Overlooked Poses](https://arxiv.org/abs/2208.01302) and
    [Spectral-guided Physical Dynamics Distillation](https://openreview.net/forum?id=P6F4MxtOKp).
    Split invariant/equivariant representations and skeleton masked-feature
    prediction also have close precedents. The defensible opportunity is a
    tested selection rule for **extra student-accessible temporal information
    beyond current state/support**, an explicit no-transfer outcome, and
    verified improvement on independent motion observables. Combining named
    losses alone would be a weak contribution.

    The paper draft reports current evidence and identifies missing experiments.
    A strong main-track claim still needs useful student transfer, comparison
    with close alternatives, and confirmation on independent sources or people.
    The absence of a positive result is a reason to sharpen the experiment,
    not to convert synthetic calibration into scientific evidence.
    ''')]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notebooks={
        '19_evidence_and_observability':evidence_notebook,
        '20_symmetry_and_temporal_information':symmetry_notebook,
        '21_student_accessible_future_features':panel_notebook,
        '22_selective_future_distillation':distillation_notebook,
    }
    for name,builder in notebooks.items():
        nb=nbf.v4.new_notebook(cells=builder(),metadata={
            'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python'},
            'iclr_bridge':{'version':'tutorials-v1','default':'read-only inspection and explicitly synthetic examples',
                'real_run':'outputs/iclr-bridge-cached-20260911','generator':str(Path(__file__).relative_to(ROOT))}})
        # Deterministic cell IDs keep regeneration diffs reviewable.
        for i,cell in enumerate(nb.cells): cell['id']=f'{name[:2]}-{i:03d}'
        nbf.validate(nb)
        destination = OUTPUT_DIR / f'{name}.ipynb'
        nbf.write(nb, destination)
        print(f'{destination.relative_to(ROOT)}: {len(nb.cells)} cells')


if __name__=='__main__':main()
