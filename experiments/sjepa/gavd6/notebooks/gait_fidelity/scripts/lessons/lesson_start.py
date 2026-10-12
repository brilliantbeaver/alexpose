"""Build-time overview cells; emitted directly into notebook 00."""


def build_start_cells(md, code):
    return [md(r'''
    ## Follow the quantities through the experiment

    A pose estimator produces an imperfect sequence of joint positions. We
    learn to correct those positions, then measure whether the corrections
    preserve differences between the legs and responses to controlled changes
    in movement. The reference trajectory is obtained by projecting body-model
    joints into the rendered camera, so its meaning is a projected engineering
    reference rather than an independently measured clinical landmark.

    | Stage | Quantity and meaning | Worked notebook |
    | --- | --- | --- |
    | Observations | $X\in\mathbb R^{B\times T\times12\times2}$, confidence, availability and time | 01 |
    | Preprocessing | Input-only origin/scale; four-frame joint tokens; stochastic query mask | 01–02 |
    | Representation | Transformer features $Z=f_\theta(X)$ | 03 |
    | Feature prediction | Predict reference features at queried locations; teacher follows the encoder by a moving average | 04 |
    | Restoration | A coordinate output network predicts residual corrections; pretrained encoders are fixed in this stage | 03–04 |
    | Movement supervision | Error in the change of signed knee excursion; later repair compares whole angle trajectories | 04 and [G](experiments/G_readout_repair.ipynb) |
    | Evaluation | Position, displacement, knee waveforms, movement response, nuisance sensitivity and coverage | 05 |
    | Evidence | Average within source families and people; paired people/seed uncertainty; artifact reconstruction | 06 |

    Here $B$ counts examples in a batch, $T$ counts frames, and 12 is the joint
    count. Each notebook introduces its own notation before using it. JEPA in
    this implementation predicts learned features of reference poses at queried
    locations within the window. It does not forecast an unseen future window.

    The code cells expose the scientific calculations and assert agreement with
    the production functions. Standard matrix operations, attention layers and
    optimizers remain PyTorch/NumPy primitives. Preparation and scheduled fitting
    still use the shared runner so every saved result retains its provenance.
    '''), md(r'''
    ## Place each completed experiment in the sequence

    The current results come from three connected experiments. The walking
    core compares direct coordinate training with several frozen-feature
    readouts. JEPA response changes the feature-pretraining loss while retaining
    the same restoration and measurement task. Readout repair then keeps the
    trained response encoders fixed and changes only their coordinate-output
    training objectives. The repeated development participants allow matched
    comparisons; they do not constitute three independent evaluation cohorts.

    | Experiment | What changes? | What carries forward? | Where to inspect it |
    | --- | --- | --- | --- |
    | Walking core | Encoder training and coordinate-only versus paired-change output loss | Shared data, seeds and evaluation | 00–06; core recipes in A, C and D |
    | JEPA response | Paired feature-residual or coordinate-residual loss during pretraining | Parent data and retained core comparators | Experiment F |
    | Readout repair | Scalar-loss weight or dense paired angle-trajectory loss | Frozen response encoders and matched readout setup | [G](experiments/G_readout_repair.ipynb) and 07 |

    Notebook 07 reads the downloaded evidence for all three stages and shows
    which conclusions are supported by their saved comparisons. The local
    `outputs/gait-fidelity/config.json` and report describe the walking-core
    source run; its full arrays and checkpoints remain on HAIC. The compact
    `outputs/iclr` packet additionally contains the later response and repair
    evidence. A CPU fixture uses generated arrays solely to exercise the
    computations and cannot reproduce the source-study scores.

    The full proposed matrix also includes mask-structure and re-pairing
    controls that were not fitted in the latest core run. Their worked
    examples remain useful for understanding the design, but their presence
    in a notebook is not evidence that the corresponding experiment ran.
    **Re-pairing** changes which movement endpoints are matched; **readout
    repair** changes a loss while retaining the original matched endpoints.
    These are separate experiments despite the similar names in the code.
    '''), code('''
    config = study.artifact('config.json')
    print(json.dumps(config, indent=2))
    print('Evidence mode:', 'SOFTWARE FIXTURE' if study.fixture else config['evidence_scope'])
    print('Model:', config['model'])
    print('Training updates:', {key: value for key, value in config['training'].items() if 'updates' in key})
    '''), md('''
    ## Check the environment before reading or fitting data

    The fixture contains small generated tracks and runs on the CPU. Its results
    verify software only. A source study reuses the HAIC assets and its saved
    session. Preflight checks local imports and paths; the allocated GPU worker
    separately checks CUDA and compiled operators. Follow the HAIC guide to
    select a source run before opening the kernel.
    '''), code("study.command('preflight')"), md(r'''
    ## Count the matrix and shared optimization phases

    A new source run defaults to ten core recipes and three seeds, hence 30
    final models. The full matrix has 34 recipes and 102 final models; choose
    it explicitly before initializing a run. The software fixture defaults to
    the full matrix; selecting `--experiment-set core` exercises the latest
    source matrix and its hierarchical sampler with generated data. Feature
    pretraining is shared when encoder objective, mask and seed agree. Different
    output objectives receive fresh output networks, with matching initial
    weights for the same seed; they are subsequently fitted separately.
    This reuse saves computation and must not be counted as independent
    pretraining evidence. We reconstruct the phase count below from the recipe
    table, then compare it with the saved plan.
    '''), code('''
    import pandas as pd
    from IPython.display import display
    study.command('plan', quiet=True)
    plan = study.artifact('plan.json')
    recipes = pd.DataFrame(plan['recipes'])
    pretraining_keys = {(r['encoder'], r['pretraining_mask'], seed)
        for r in plan['recipes'] if r['mode'] == 'frozen_readout' and r['encoder'] != 'initialized'
        for seed in plan['seeds']}
    final_fits = len(plan['recipes']) * len(plan['seeds'])
    assert final_fits == plan['counts']['final_fits']
    assert len(pretraining_keys) == plan['counts']['pretraining_phases']
    assert final_fits + len(pretraining_keys) == plan['counts']['optimization_phases']
    display(recipes.groupby(['group', 'mode']).size().rename('recipes').reset_index())
    print(json.dumps(plan['counts'], indent=2))
    '''), md('''
    ## Keep teaching calculations separate from retained results

    The additional NumPy and PyTorch examples use small in-memory arrays and
    scratch models. Their checks do not overwrite checkpoints, training data,
    predictions or evaluation settings. The normal prepare, launch, evaluate
    and verify cells still execute the registered pipeline. Source training is
    submitted through Slurm, and notebooks 05–06 require those jobs to finish.

    Run **01–06 in order**, then read the experiment notebooks for the matched
    comparisons, the response objective, and the readout repair. Read **07**
    alongside them to connect the calculations to retained results. Each notebook can reopen
    the same saved run without sharing Python variables with another kernel.
    A source run now scans the full audited AMASS inventory and saves a
    motion-window selection. Original AMASS training and validation identities
    become training and development; original test identities remain locked.
    Previous exposure must be reviewed before admitting confirmation data.
    GAVD follows its own recording-grouped plan because the supplied manifests
    do not establish person identity or paired joint-coordinate references.
    More views or training seeds do not create independent participants.
    ''')]
