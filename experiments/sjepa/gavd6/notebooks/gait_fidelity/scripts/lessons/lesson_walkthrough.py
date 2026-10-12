"""Build a read-only visual account of the three completed development experiments.

Cells remain visible in the generated notebook. Evidence is read directly from
outputs/iclr; this module is not imported by the scientific training pipeline.
"""


def build_walkthrough_cells(md, code):
    return [
        md(r'''
        # 07 · Follow the completed gait-fidelity experiments

        **Question:** Does learning to predict reference-pose features help restore
        the movement measured from imperfect pose tracks, beyond learning coordinate
        corrections directly?

        Follow the experiment in order: construct matched movement states, measure
        their difference, learn a representation, train its coordinate readout, and
        evaluate the same people under matched conditions. The three completed runs
        change different parts of this path. Each result below is recomputed from
        the downloaded evidence in `outputs/iclr`.

        Figures marked **SOURCE RESULTS** use saved measurements. Figures marked
        **ILLUSTRATION** explain an operation using constructed inputs; they are
        not trajectories or activations from a fitted model. The compact packet
        contains no raw trajectories, videos or checkpoints. This notebook runs
        locally on CPU, reads no HAIC session, and submits no jobs.

        The completed runs are development experiments on synthetic walking. The
        fourteen evaluated people were excluded from model training but reused
        while developing the follow-ups. Protected confirmation and GAVD evaluation
        are not present in this packet.
        '''),
        code('''
        from pathlib import Path
        import hashlib, json, os, sys, tempfile
        import numpy as np
        import pandas as pd
        from IPython.display import display, Markdown

        ROOT = Path.cwd().resolve()
        while not (ROOT / 'src/gavd6_sjepa').is_dir() and ROOT != ROOT.parent:
            ROOT = ROOT.parent
        assert (ROOT / 'src/gavd6_sjepa').is_dir(), 'Open from the GAVD6 checkout.'
        # An optional alternative compact packet; GF_WORK and GF_ROOT are ignored.
        DATA = Path(os.environ.get('GF_EVIDENCE_ROOT', ROOT / 'outputs/iclr')).resolve()
        os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'gf-walkthrough-mpl'))
        import matplotlib.pyplot as plt
        get_ipython().run_line_magic('matplotlib', 'inline')
        from matplotlib.patches import FancyBboxPatch, Patch
        from matplotlib.colors import ListedColormap
        sys.path.insert(0, str(ROOT / 'src'))

        def read_json(path):
            return json.loads(path.read_text())

        inventories = sorted(DATA.glob('transfer-inventory-*.json'))
        assert inventories, f'No transfer inventory in {DATA}; download the paper evidence first.'
        inventory = read_json(inventories[-1])
        assert not inventory['transfer_blocked'] and not inventory['missing_required']
        selected = [e for e in inventory['files'] if e['status'] == 'selected']
        for entry in selected:
            path = (DATA / entry['destination']).resolve()
            assert path.is_relative_to(DATA), 'Inventory path escapes the evidence directory.'
            assert path.stat().st_size == entry['bytes'], str(path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], str(path)

        RUNS = ['walking-core', 'jepa-response', 'readout-repair']
        paths = {r: DATA / r / ('development/evaluation' if r == 'readout-repair' else 'evaluation')
                 for r in RUNS}
        configs = {r: read_json(DATA / r / 'config.json') for r in RUNS}
        plans = {r: read_json(DATA / r / 'plan.json') for r in RUNS}
        frames = {r: pd.read_csv(paths[r] / 'per-person.csv') for r in RUNS}
        comparisons = {r: read_json(paths[r] / 'comparisons.json') for r in RUNS}
        assert all(c['fixture'] is False for c in configs.values())
        people = set(frames[RUNS[0]].canonical_person_id)
        seeds = sorted(frames[RUNS[0]].seed.unique())
        assert len(people) == 14 and seeds == [17, 29, 43], 'Revisit the scope for a different packet.'
        for frame in frames.values():
            assert set(frame.canonical_person_id) == people
            assert not frame.duplicated(['method', 'seed', 'canonical_person_id']).any()
            assert frame.groupby('method').size().eq(len(people) * len(seeds)).all()
            assert set(frame.seed) == set(seeds)
        for parent, child in zip(RUNS[:-1], RUNS[1:]):
            shared = set(frames[parent].method) & set(frames[child].method)
            index = ['method', 'seed', 'canonical_person_id']
            a = frames[parent][frames[parent].method.isin(shared)].set_index(index).sort_index()
            b = frames[child][frames[child].method.isin(shared)].set_index(index).sort_index()
            numeric = sorted(set(a.select_dtypes('number').columns) & set(b.select_dtypes('number').columns))
            assert a.index.equals(b.index)
            assert np.allclose(a[numeric], b[numeric], equal_nan=True, atol=1e-12, rtol=0), 'Retained results changed.'

        # These CSV rows have already averaged conditions, windows and motions
        # within each person. Equal-person and equal-seed means come next.
        def means(frame):
            metrics = frame.select_dtypes('number').columns.drop('seed', errors='ignore')
            return frame.groupby(['method', 'seed'])[metrics].mean().groupby('method').mean()

        results = {r: means(f) for r, f in frames.items()}
        by_extractor = pd.read_csv(paths['readout-repair'] / 'per-person-by-extractor.csv')
        PRIMARY_EXTRACTOR = comparisons['readout-repair']['primary_extractor']
        assert PRIMARY_EXTRACTOR == 'vitpose_base'
        repair_people = by_extractor[by_extractor.extractor.eq(PRIMARY_EXTRACTOR)].copy()
        repair_means = means(repair_people)
        print(f'Verified {len(selected)} files; {len(people)} shared development people; seeds {seeds}.')
        print('Evidence:', DATA)
        '''),
        code('''
        BLUE, ORANGE, TEAL, GRAY = '#275d82', '#ba613a', '#267970', '#69727d'
        PALE_BLUE, PALE_ORANGE, PALE_TEAL = '#e9f1f7', '#fbefe7', '#e9f3f0'
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                             'axes.titlesize': 12, 'axes.titlelocation': 'left',
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                             'svg.fonttype': 'none', 'pdf.fonttype': 42})
        FIGURES = {}  # Keep figures in memory for the optional export cell at the end.

        def finish(fig, name):
            FIGURES[name] = fig
            display(fig)
            plt.close(fig)

        def box(ax, x, y, w, h, title, detail='', color=PALE_BLUE):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
                                       facecolor=color, edgecolor='#bac4ce', linewidth=1))
            ax.text(x + w/2, y + h*.73, title, ha='center', va='center', weight='bold', fontsize=11)
            ax.text(x + w/2, y + h*.32, detail, ha='center', va='center', fontsize=10, linespacing=1.45)

        def arrow(ax, start, end, color=GRAY, dashed=False):
            ax.annotate('', xy=end, xytext=start,
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.6,
                                        linestyle='--' if dashed else '-'))
        '''),
        md(r'''
        ## 1 · Locate the intervention in each experiment

        The walking core compares representation families and output objectives.
        The response follow-up adds paired residual losses during **pretraining**.
        The repair holds those pretrained encoders fixed and changes **readout
        training**, where features are turned into coordinate corrections.

        A final fit is one output model for one random seed. Pretraining may be
        shared by several readouts, so final-fit counts and optimization-phase
        counts differ. Retained predictions remain the same evidence when reused
        in a later comparison.
        '''),
        code('''
        core_counts, response_counts, repair_counts = [plans[r]['counts'] for r in RUNS]
        fig, ax = plt.subplots(figsize=(13, 4.4))
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
        box(ax, .02, .40, .28, .39, 'Walking core',
            f"{core_counts['pretraining_phases']} pretraining fits; {core_counts['final_fits']} final fits\\n5 families × 2 output losses × 3 seeds")
        box(ax, .36, .40, .28, .39, 'JEPA response',
            f"{response_counts['pretraining_phases']} new pretraining fits; {response_counts['final_fits']} final fits\\n3 variants × 2 output losses × 3 seeds", PALE_TEAL)
        box(ax, .70, .40, .28, .39, 'Readout repair',
            f"{repair_counts['new_pretraining']} new pretraining; {repair_counts['new_readouts']} new readouts\\n2 encoders × 2 new losses × 3 seeds", PALE_ORANGE)
        arrow(ax, (.305, .60), (.345, .60)); arrow(ax, (.645, .60), (.685, .60))
        ax.text(.33, .86, 'Retain core predictions', ha='center', color=GRAY, fontsize=10)
        ax.text(.67, .86, 'Reuse frozen encoders', ha='center', color=GRAY, fontsize=10)
        box(ax, .08, .05, .84, .19, 'One shared development population',
            f'{len(people)} people excluded from training · three fitted seeds · repeated across all stages', '#f3f4f6')
        ax.set_title('SOURCE RESULTS · Saved plans identify what changed', pad=15)
        finish(fig, '01-experiment-lineage')
        display(pd.DataFrame([
            {'Stage': 'Core', 'Primary candidate': 'Paired JEPA / paired-change',
             'Matched comparator': 'Direct / paired-change', 'Outcome': 'Response; all 3 estimators'},
            {'Stage': 'Response', 'Primary candidate': 'Delta JEPA / paired-change',
             'Matched comparator': 'Endpoint JEPA / paired-change', 'Outcome': 'Response; all 3 estimators'},
            {'Stage': 'Repair', 'Primary candidate': 'Delta JEPA / dense change',
             'Matched comparator': 'Delta JEPA / scalar × 0.1', 'Outcome': 'Waveform; ViTPose only'},
        ]))
        '''),
        md(r'''
        ## 2 · Construct the paired observations and reference poses

        A recorded AMASS movement drives a body model. The pipeline renders the
        original and edited movement states, estimates joint positions from each
        video, and projects the body-model joints into the same camera to obtain
        references. Within a pair, the intended movement changes while camera,
        naming condition and observation condition stay matched. The **response**
        measures the difference between these two motion sequences; it is not a
        forecast of a future frame.

        The five saved state records are baseline, its no-change duplicate, and
        the three nonzero edit levels. Derived observations multiply coverage of
        a source movement. They do not multiply the number of independent people.
        Counts below use admitted data rather than the larger proposed cohort.
        '''),
        code('''
        admission = read_json(DATA / 'walking-core/data/admission.json')['counts']
        cfg = configs['walking-core']
        factors = {'Physical states': len(cfg['data']['physical_states']),
                   'Cameras': len(cfg['data']['cameras']),
                   'Pose estimators': by_extractor.extractor.nunique(),
                   'Joint-naming conditions': len(cfg['data']['naming']),
                   'Observation conditions': len(cfg['data']['observations']),
                   'Saved movement-state records': len(cfg['data']['movement_levels_deg']) + 1}
        records_per_window = int(np.prod(list(factors.values())))
        dev_records = admission['splits']['development']
        assert dev_records % records_per_window == 0
        dev_windows = dev_records // records_per_window
        display(pd.DataFrame({'Factor': list(factors), 'Count': list(factors.values())}))
        display(pd.DataFrame([
            {'Split': 'Training', 'People': admission['people'] - len(people),
             'Source windows': admission['windows'] - dev_windows, 'Records': admission['splits']['train']},
            {'Split': 'Development', 'People': len(people), 'Source windows': dev_windows, 'Records': dev_records},
        ]))
        print(f'{dev_windows} windows × {records_per_window} derived records = {dev_records:,} development records.')

        fig, ax = plt.subplots(figsize=(12.8, 4.3))
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
        box(ax, .025, .32, .155, .35, 'AMASS movement', 'Original state a\\nEdited state b')
        box(ax, .25, .59, .24, .30, 'Render → pose estimator', 'Imperfect observed tracks', PALE_BLUE)
        box(ax, .25, .11, .24, .30, 'Project body-model joints', 'Reference tracks', PALE_ORANGE)
        box(ax, .56, .59, .22, .30, 'Coordinate restoration', 'Predicted angles\\nand response', PALE_TEAL)
        box(ax, .56, .11, .22, .30, 'Reference measurement', 'Target angles\\nand response', PALE_ORANGE)
        box(ax, .86, .33, .12, .31, 'Score errors', 'Compare\\noutputs', '#f3f4f6')
        for a, b in [((.19,.58),(.24,.74)), ((.19,.42),(.24,.26)),
                     ((.50,.74),(.55,.74)), ((.50,.26),(.55,.26)),
                     ((.79,.74),(.85,.59)), ((.79,.26),(.85,.38))]: arrow(ax,a,b)
        ax.set_title('ILLUSTRATION · The reference and observed tracks share the same rendered movement', pad=12)
        finish(fig, '02-paired-data-path')
        '''),
        md(r'''
        ## 3 · See what the scalar movement measurement leaves unspecified

        At each frame, a projected knee angle is measured from hip, knee and ankle.
        For each leg, **excursion** is the 95th percentile minus the 5th percentile
        of its angle over the window. Subtracting left excursion from right gives
        an asymmetry measurement, $A$. The paired response is $A_b-A_a$.

        Waveform error compares the angle at corresponding frames. A scalar
        excursion discards their order, so correct excursion alone cannot establish
        faithful movement over time. The constructed example below circularly
        shifts both predicted waveforms. Their scalar response stays unchanged,
        although the framewise error grows. This demonstrates a property of the
        measurement, without asserting that fitted models made this particular
        error. Change `SHIFT_FRAMES` and rerun the cell to explore it.
        '''),
        code(r'''
        SHIFT_FRAMES = 12  # Teaching control only; never changes the scientific evaluation.
        samples, hz = cfg['model']['window_size'], cfg['data']['hz']
        t = np.arange(samples) / hz
        phase = 2 * np.pi * 4 * np.arange(samples) / samples
        # [endpoint, time, leg], constructed projected-angle signals in degrees.
        reference_angles = np.stack([
            np.stack([140 + 20*np.sin(phase), 140 + 24*np.sin(phase + np.pi)], axis=-1),
            np.stack([140 + 20*np.sin(phase), 140 + 32*np.sin(phase + np.pi)], axis=-1),
        ])
        shifted_angles = np.roll(reference_angles, SHIFT_FRAMES, axis=1)

        def asymmetry(angles):
            excursion = np.quantile(angles, .95, axis=1) - np.quantile(angles, .05, axis=1)
            return excursion[:, 1] - excursion[:, 0]

        ref_A, pred_A = asymmetry(reference_angles), asymmetry(shifted_angles)
        ref_response, pred_response = np.diff(ref_A).item(), np.diff(pred_A).item()
        assert np.allclose(ref_A, pred_A)
        toy_waveform_error = np.abs(reference_angles - shifted_angles).mean()
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.6), sharey=True)
        for endpoint, ax in enumerate(axes):
            ax.plot(t, reference_angles[endpoint, :, 1], color=BLUE, label='Reference right knee')
            ax.plot(t, shifted_angles[endpoint, :, 1], color=ORANGE, ls='--', label='Shifted right knee')
            ax.set(title=f'State {"a" if endpoint == 0 else "b"}: A = {ref_A[endpoint]:.2f}°', xlabel='Time (s)')
            ax.grid(alpha=.15)
        axes[0].set_ylabel('Projected knee angle (degrees)'); axes[1].legend(frameon=False, fontsize=9)
        fig.suptitle(f'ILLUSTRATION · Response error = {abs(ref_response-pred_response):.2f}°; '
                     f'waveform error across both legs = {toy_waveform_error:.2f}°', x=.04, ha='left')
        fig.tight_layout(rect=(0,0,1,.93)); finish(fig, '03-scalar-and-waveform')
        display(pd.DataFrame({'Quantity': ['A in state a', 'A in state b', 'Response b − a'],
                              'Reference (degrees)': [*ref_A, ref_response],
                              'Shifted (degrees)': [*pred_A, pred_response]}))
        '''),
        md(r'''
        ## 4 · Follow the features through training and deployment

        Four consecutive frames of one joint form a token. A 128-frame, 12-joint
        window therefore has 384 token positions. Each contains coordinates,
        confidence, availability and time; the encoder produces 96 learned
        features per position. Artificial query masks hide some available inputs
        during pretraining. Image occlusion and missing pose detections are
        separate sources of missing information.

        The heatmap calls the implementation's graph-time sampler on a constructed
        availability pattern. It illustrates its exact hiding budget, including
        the possibility of truncated regions. The completed core used graph-time
        masking; it did not run the full topology-control matrix described in
        tutorial B.
        '''),
        code('''
        from gavd6_sjepa.research_directions.gait_fidelity.masking import sample_mask, patch_support
        MASK_SEED = 17  # An illustrative draw, not a recovered training batch.
        patch = cfg['model']['patch_size']
        observed = np.ones((1, samples, 12), dtype=bool)
        observed[:, 32:48, 10:12] = False  # Constructed absent ankle observations.
        hidden, mask_receipt = sample_mask(observed, 'graph_time', rng=np.random.default_rng(MASK_SEED),
                                          fraction=cfg['training']['mask_fraction'],
                                          patch_size=patch, return_receipt=True)
        available = patch_support(observed, patch)
        display_mask = np.where(~available[0], 0, np.where(hidden[0], 2, 1))
        joints = [f'{side} {joint}' for joint in ['shoulder','elbow','wrist','hip','knee','ankle']
                  for side in ['L','R']]
        fig, ax = plt.subplots(figsize=(12.5, 4.1))
        ax.imshow(display_mask.T, aspect='auto', interpolation='nearest',
                  cmap=ListedColormap(['#d5d9dd', BLUE, '#d69464']), vmin=0, vmax=2)
        ax.set_yticks(range(12), joints)
        ax.set_xticks(np.arange(0, samples//patch, 4), np.arange(0, samples, patch*4))
        ax.set_xlabel(f'First frame in each {patch}-frame token'); ax.set_ylabel('Body-12 joint')
        ax.set_title('ILLUSTRATION · Graph-time masking on a constructed availability pattern', pad=12)
        ax.legend(handles=[Patch(color=c, label=s) for c,s in [('#d5d9dd','Unavailable input'),
                  (BLUE,'Retained context'),('#d69464','Artificial query mask')]],
                  loc='upper center', bbox_to_anchor=(.5,-.22), ncol=3, frameon=False)
        fig.tight_layout(); finish(fig, '04-token-mask')
        print('Available tokens:', mask_receipt['observed_tokens'][0],
              'Hidden:', mask_receipt['hidden_tokens'][0],
              'Retained:', mask_receipt['context_tokens'][0])
        '''),
        md(r'''
        A **JEPA** (joint-embedding predictive architecture) trains one network to
        predict features produced by a second network. Here the teacher processes
        reference poses. Its weights track a moving average of the student encoder;
        gradients do not update the teacher directly. During pretraining, the
        predictor and teacher are compared at selected feature positions.

        At deployment, the learned encoder processes observed poses. A coordinate
        readout predicts a correction added to the input coordinates. Its encoder
        is frozen, and neither clean reference poses nor teacher features are
        available as inputs. The direct-coordinate baseline instead trains its
        encoder and coordinate output jointly. Its 4,000 end-to-end updates differ
        from 2,000 pretraining plus 2,000 frozen-encoder readout updates in which
        parameters and supervision are different.
        '''),
        code('''
        fig, ax = plt.subplots(figsize=(13, 5.6))
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
        ax.text(.01,.98,'PRETRAINING',weight='bold',color=GRAY,va='top')
        box(ax,.02,.64,.20,.20,'Observed poses','Normalize → mask')
        box(ax,.29,.64,.20,.20,'Student encoder','384 × 96 features')
        box(ax,.56,.64,.18,.20,'Predictor','Predicted features')
        box(ax,.80,.64,.18,.20,'Feature loss','Compare to teacher',PALE_TEAL)
        box(ax,.29,.32,.20,.20,'Reference poses','Training targets only',PALE_ORANGE)
        box(ax,.56,.32,.18,.20,'EMA teacher','No target gradient',PALE_ORANGE)
        for a,b in [((.23,.74),(.28,.74)),((.50,.74),(.55,.74)),((.75,.74),(.79,.74)),
                    ((.50,.42),(.55,.42)),((.75,.42),(.88,.62))]: arrow(ax,a,b)
        arrow(ax,(.49,.64),(.59,.53),ORANGE,True)
        ax.text(.49,.55,'Weight averaging',color=ORANGE,fontsize=9,ha='right')
        ax.axhline(.24,color='#cbd1d7',lw=1)
        ax.text(.01,.205,'READOUT TRAINING / DEPLOYMENT',weight='bold',color=GRAY,va='top',fontsize=10)
        ax.text(.5,.06,'Observed poses → frozen encoder → learned coordinate correction + input → restored poses',
                ha='center',va='center',fontsize=11,color=BLUE)
        ax.set_title('ILLUSTRATION · Teacher features are training targets, not deployment inputs',pad=14)
        finish(fig, '05-training-and-deployment')
        '''),
        md(r'''
        ## 5 · Establish the walking-core result before changing JEPA

        Connect the two output objectives within each representation family. The
        base objective trains coordinate correction. The original paired-change
        package adds the scalar response loss and a short-segment geometry penalty.
        Thus, this comparison changes that package; it cannot isolate either
        added term on its own.

        **NLE** is coordinate distance divided by the rendered person's bounding-box
        diagonal, so it has no degree unit. Waveform and response scores use
        projected angles and include declared costs for invalid outputs, examined
        below. Each point averages the same fourteen people and three seeds.
        '''),
        code('''
        families = [('P-direct-none','Direct coordinates'),
                    ('M-coordinate-graph_time','Coordinate pretraining'),
                    ('M-paired_jepa-graph_time','Paired JEPA'),
                    ('I-initialized-none','Untrained features'),
                    ('I-shuffled_jepa-graph_time','Shuffled-reference JEPA')]
        cm = results['walking-core']
        fig, axes = plt.subplots(1,3,figsize=(13.2,4.5),sharey=True)
        for ax,metric,title in zip(axes,['synthetic_all_nle','response_error','waveform_error'],
                                  ['Coordinate error (NLE)','Response score (degrees)','Waveform score (degrees)']):
            for y,(family,label) in enumerate(families):
                a,b = cm.loc[family+'-base',metric],cm.loc[family+'-paired_change',metric]
                ax.plot([a,b],[y,y],color='#b9bdc3',lw=2)
                ax.scatter(a,y,color=BLUE,s=48,label='Base' if y==0 else None,zorder=3)
                ax.scatter(b,y,color=ORANGE,s=48,marker='s',label='Original paired-change' if y==0 else None,zorder=3)
            ax.set_title(title); ax.set_xlim(left=0); ax.grid(axis='x',alpha=.15)
            ax.set_yticks(range(len(families)))
        axes[0].set_yticklabels([label for _,label in families]); axes[0].invert_yaxis()
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
        fig.suptitle('SOURCE RESULTS · What changes when the output loss changes?',x=.03,ha='left')
        fig.tight_layout(rect=(0,.07,1,.94)); finish(fig,'06-core-objectives')
        gains = [cm.loc[f+'-paired_change','waveform_error']-cm.loc[f+'-base','waveform_error'] for f,_ in families]
        print(f'Original paired-change increases mean waveform score in all {len(gains)} families '
              f'(range {min(gains):.2f}–{max(gains):.2f} degrees; lower scores are better).')
        '''),
        md(r'''
        ## 6 · Distinguish the response-pretraining variants

        Let $e_a$ and $e_b$ be student-minus-teacher feature residuals for two
        matched movement states, after temperature scaling and removal of the
        feature-channel mean. The delta term is
        $\|e_b-e_a\|^2/(2D)$; its endpoint control is
        $(\|e_a\|^2+\|e_b\|^2)/(2D)$, where $D$ is feature width. Both retain
        the original feature-prediction objective. Their difference is
        $-e_a^\top e_b/D$, which couples the two residuals.

        Equal nonzero residuals cancel in the delta term. This makes the endpoint
        control necessary: a lower delta loss alone cannot establish accurate
        endpoint poses. The third variant applies a corresponding paired residual
        objective in coordinate space. Tutorial F derives these terms and checks
        their gradients. Here we connect them to the completed plan and result.
        '''),
        code('''
        # A two-dimensional residual illustration, not saved model activations.
        ea = np.array([1., -1.])
        residual_rows = []
        for name,eb in [('Same residual at both endpoints',ea),
                        ('Opposite residuals',-ea), ('Zero residual at both endpoints',np.zeros(2))]:
            a = np.zeros(2) if name.startswith('Zero') else ea
            delta = np.square(eb-a).sum()/(2*len(a))
            endpoint = (np.square(a).sum()+np.square(eb).sum())/(2*len(a))
            assert np.isclose(delta-endpoint,-a.dot(eb)/len(a))
            residual_rows.append({'Illustrative case':name,'Delta term':delta,'Endpoint term':endpoint})
        display(pd.DataFrame(residual_rows))
        response_ids = [f'F-response-{v}-graph_time-{o}'
                        for v in ['jepa_delta_v1','jepa_endpoint_v1','coordinate_delta_v1']
                        for o in ['base','paired_change']]
        display(results['jepa-response'].loc[response_ids,
                ['synthetic_all_nle','response_error','waveform_error','direction_accuracy']].round(4))
        response_primary = comparisons['jepa-response']['response_error']
        print(f"Primary delta advantage: {response_primary['improvement']:.3f} degrees; "
              f"crossed people/seed 95% interval {response_primary['crossed_person_seed_ci95']}.")
        '''),
        md(r'''
        ## 7 · Repair the readout while holding its encoder fixed

        The original scalar loss summarizes each motion window before comparing
        the paired response. The dense change loss compares the two states'
        angle differences **at each matching frame and leg**. Dense therefore
        refers to supervision distributed over time, not to a larger model or
        temporal velocity. All new repair arms retain coordinate supervision and
        the same geometry penalty.

        A gradient measures how a loss changes when a parameter changes. Its
        root-mean-square (RMS) size summarizes the strength of that update signal.
        The second calibration plot measures something different: the fraction of
        eligible predicted angles receiving a nonzero derivative from each loss.

        With $\theta$ denoting a reference angle and $\hat\theta$ a predicted
        angle, the dense term averages
        $[(\hat\theta_{b,t,\ell}-\hat\theta_{a,t,\ell})-
        (\theta_{b,t,\ell}-\theta_{a,t,\ell})]^2/180^2$ over eligible frames
        $t$ and legs $\ell$. The scalar term instead squares the error in
        $A_b-A_a$, also normalized by $180^2$. Both use reference-defined support.

        The low-scalar comparator reduces the original scalar coefficient to 0.1.
        Dense-loss coefficients are calibrated separately for each encoder and
        seed so that the weighted dense term has the same initial gradient RMS
        as the low-scalar term. The coordinate-gradient RMS is measured as a
        diagnostic; it is not the quantity used to set the dense coefficient. This control
        matters because a gain over the original scalar coefficient could arise
        from changing loss strength. Calibration is an initial measurement; it
        does not establish gradient balance throughout training.
        '''),
        code('''
        display(pd.DataFrame([
            {'Readout':'Base (retained)','Coordinate':1,'Geometry':0,'Scalar response':0,'Dense change':'0'},
            {'Readout':'Original scalar (retained)','Coordinate':1,'Geometry':1,'Scalar response':1,'Dense change':'0'},
            {'Readout':'Low scalar (new)','Coordinate':1,'Geometry':1,'Scalar response':.1,'Dense change':'0'},
            {'Readout':'Dense change (new)','Coordinate':1,'Geometry':1,'Scalar response':0,'Dense change':'Calibrated per encoder/seed'},
        ]))
        calibration_rows = []
        ledger = read_json(DATA / 'readout-repair/ledger.json')
        for name, record in ledger['completed'].items():
            if not name.startswith('calibrate-'): continue
            r = record['result']
            row = {'variant':r['representation_variant'], 'seed':r['seed'],
                   'scalar_ratio':r['gradients']['scalar']['rms']/r['gradients']['coordinate']['rms']}
            for term in ['scalar','dense']:
                numerator = sum(b['angle_gradient_support'][term]['nonzero_angle_gradients'] for b in r['batches'])
                denominator = sum(b['angle_gradient_support'][term]['eligible_angle_entries'] for b in r['batches'])
                row[term+'_support'] = numerator / denominator
            calibration_rows.append(row)
        calibration = pd.DataFrame(calibration_rows).sort_values(['variant','seed'])
        assert len(calibration) == plans['readout-repair']['counts']['calibrations'] == 6
        fig, axes = plt.subplots(1,2,figsize=(12.5,4))
        x = np.arange(len(calibration))
        axes[0].scatter(x,calibration.scalar_ratio,color=ORANGE,label='Original scalar coefficient 1',s=45)
        axes[0].scatter(x,.1*calibration.scalar_ratio,color=BLUE,label='Scalar coefficient 0.1',s=45)
        axes[0].axhline(1,color=GRAY,ls='--',lw=1)
        axes[0].set(ylabel='Scalar / coordinate gradient RMS',title='Loss strength at initialization',ylim=(0,None))
        axes[1].scatter(x,100*calibration.scalar_support,color=ORANGE,label='Scalar response',s=45)
        axes[1].scatter(x,100*calibration.dense_support,color=TEAL,label='Dense change',s=45)
        axes[1].set(ylabel='Eligible angle entries with nonzero gradient (%)',
                    title='Where angle supervision reaches',ylim=(0,35))
        labels = [f'{v.removeprefix("jepa_").removesuffix("_v1")}\\n{s}' for v,s in zip(calibration.variant,calibration.seed)]
        for ax in axes:
            ax.set_xticks(x,labels); ax.grid(axis='y',alpha=.15); ax.legend(frameon=False,fontsize=9)
        fig.suptitle('SOURCE RESULTS · Six calibration runs, before readout training',x=.04,ha='left')
        fig.tight_layout(rect=(0,0,1,.94)); finish(fig,'07-readout-calibration')
        '''),
        code('''
        objectives = [('base','Base readout'),('paired_change','Original scalar'),
                      ('scalar_low','Scalar × 0.1'),('dense_change','Dense change')]
        fig, axes = plt.subplots(1,2,figsize=(12.5,4.4),sharey=True)
        for ax,metric,title in zip(axes,['waveform_error','response_error'],
                                  ['Waveform score (degrees)','Response score (degrees)']):
            for variant,color,offset,marker in [('jepa_delta_v1',BLUE,-.09,'o'),
                                              ('jepa_endpoint_v1',TEAL,.09,'s')]:
                for y,(objective,label) in enumerate(objectives):
                    method = (f'F-response-{variant}-graph_time-{objective}' if objective in ['base','paired_change']
                              else f'R-repair-{variant}-{objective}')
                    ax.scatter(repair_means.loc[method,metric],y+offset,color=color,marker=marker,s=55,
                               label=variant.removeprefix('jepa_').removesuffix('_v1').title()+' JEPA' if y==0 else None)
            ax.axvline(repair_means.loc['P-direct-none-base',metric],color=ORANGE,ls='--',label='Direct / base')
            ax.set_title(title); ax.grid(axis='x',alpha=.15); ax.set_yticks(range(4))
        axes[0].set_yticklabels([label for _,label in objectives]); axes[0].invert_yaxis()
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
        fig.suptitle('SOURCE RESULTS · Readout repair, ViTPose only (the primary estimator)',x=.04,ha='left')
        fig.tight_layout(rect=(0,.08,1,.94)); finish(fig,'08-readout-repair')
        vm = repair_means
        original = vm.loc['F-response-jepa_delta_v1-graph_time-paired_change','waveform_error']
        low = vm.loc['R-repair-jepa_delta_v1-scalar_low','waveform_error']
        dense = vm.loc['R-repair-jepa_delta_v1-dense_change','waveform_error']
        print(f'Delta JEPA waveform: original {original:.3f}, low scalar {low:.3f}, dense {dense:.3f}.')
        print(f'Lowering the scalar coefficient recovers {(original-low)/(original-dense):.1%} '
              'of the mean gain from original to dense; this ratio is descriptive, not a causal decomposition.')
        '''),
        md(r'''
        ## 8 · Keep failed outputs in the comparison

        Some outputs cannot produce valid angles on the reference-defined frame
        support. The response score assigns such pairs a fixed 720-degree cost.
        Its mean is the successful-error contribution plus the failure contribution.
        The cost is a declared scoring rule, not a physically observed 720-degree
        knee error. A small failure fraction can consequently affect the ranking.

        The blue segment below retains the full denominator and contributes zero
        for failed cases; it is **not** the mean error among successful cases alone.
        The zero-response benchmark always predicts no change in the scalar
        measurement. It is useful for judging response prediction but does not
        produce restored coordinates.
        '''),
        code('''
        rm = results['jepa-response']
        selected_methods = ['P-direct-none-base','M-paired_jepa-graph_time-base',
                            'F-response-jepa_delta_v1-graph_time-paired_change',
                            'F-response-jepa_endpoint_v1-graph_time-paired_change']
        labels = ['Direct / base','Paired JEPA / base','Delta JEPA / original scalar','Endpoint JEPA / original scalar']
        totals = rm.loc[selected_methods]
        assert np.allclose(totals.response_error,
                           totals.response_success_contribution + totals.response_failure_contribution)
        assert np.allclose(totals.response_failure_contribution,720*totals.response_failure_rate)
        assert np.allclose(rm.zero_response_error,rm.zero_response_error.iloc[0])
        fig, ax = plt.subplots(figsize=(11.8,3.9))
        y = np.arange(len(labels))
        ax.barh(y,totals.response_success_contribution,color=BLUE,label='Successful-error contribution')
        ax.barh(y,totals.response_failure_contribution,left=totals.response_success_contribution,
                color=ORANGE,label='Invalid-pair cost contribution')
        ax.axvline(rm.zero_response_error.iloc[0],color=TEAL,ls='--',lw=2,label='Zero-response benchmark')
        for i,(_,r) in enumerate(totals.iterrows()):
            ax.text(r.response_error+.1,i,f'{r.response_error:.2f}  ({100*r.response_failure_rate:.2f}% failed)',va='center',fontsize=9)
        ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlim(0,totals.response_error.max()+4)
        ax.set_xlabel('Person- and seed-averaged response score (degrees, including declared cost)')
        ax.set_title('SOURCE RESULTS · All three pose estimators, full paired population')
        ax.legend(loc='upper center',bbox_to_anchor=(.4,-.22),ncol=2,frameon=False,fontsize=9)
        fig.tight_layout(); finish(fig,'09-failure-accounting')
        '''),
        md(r'''
        ## 9 · Inspect uncertainty at the person level

        Frames, camera views, corruptions and repeated seeds are related
        observations. Published summaries average conditions within source windows,
        windows within raw motions, and raw motions within people before giving
        each person equal weight. The CSVs used here start at that person level.
        Three fitted seeds describe some training variation; they do not turn
        fourteen people into forty-two independent participants.

        A **crossed bootstrap** resamples people and training seeds separately,
        keeping the candidate and comparator paired. The repair's declared primary
        interval instead applies a Student-t interval to the fourteen differences
        after averaging each person's three seeds. It is conditional on these
        fitted seeds; both intervals are shown. They describe different uncertainty
        assumptions, so neither should be selected because it looks more favorable.
        '''),
        code('''
        from scipy.stats import t as student_t
        entries = [comparisons['walking-core']['response_error'],
                   comparisons['jepa-response']['response_error'],
                   comparisons['readout-repair']['primary']['waveform_error']]
        contrast_frames = [frames['walking-core'],frames['jepa-response'],repair_people]
        contrast_metrics = ['response_error','response_error','waveform_error']
        differences = []
        # Reconstruct the paired means and the saved bootstrap, not independent row resampling.
        for frame,metric,entry in zip(contrast_frames,contrast_metrics,entries):
            pair = frame[frame.method.isin([entry['candidate'],entry['comparator']])].pivot(
                index=['canonical_person_id','seed'],columns='method',values=metric)
            assert not pair.isna().any().any()
            d = (pair[entry['comparator']]-pair[entry['candidate']]).unstack('seed').sort_index()
            x = d.to_numpy(); assert x.shape == (14,3)
            assert np.isclose(x.mean(),entry['improvement'])
            rng = np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
            draws = []
            for _ in range(cfg['evaluation']['bootstrap_draws']):
                i = rng.integers(len(x),size=len(x)); j = rng.integers(x.shape[1],size=x.shape[1])
                draws.append(x[i][:,j].mean())
            saved_ci = entry.get('crossed_person_seed_ci95',entry.get('crossed_bootstrap',{}).get('crossed_person_seed_ci95'))
            assert np.allclose(np.quantile(draws,[.025,.975]),saved_ci)
            differences.append(d)
        repair_person_effect = differences[-1].mean(axis=1)
        half = student_t.ppf(.975,len(repair_person_effect)-1)*repair_person_effect.std(ddof=1)/np.sqrt(len(repair_person_effect))
        assert np.allclose([repair_person_effect.mean()-half,repair_person_effect.mean()+half],entries[-1]['person_averaged_t_ci95'])

        fig, axes = plt.subplots(1,2,figsize=(13.4,5.2),gridspec_kw={'width_ratios':[1.25,1]})
        primary_labels = ['Core: paired JEPA vs direct\\nResponse; all estimators',
                          'Response: delta vs endpoint\\nResponse; all estimators',
                          'Repair: dense vs low scalar\\nWaveform; ViTPose only']
        for y,entry in enumerate(entries):
            mid = entry['improvement']
            lo,hi = entry.get('crossed_person_seed_ci95',entry.get('crossed_bootstrap',{}).get('crossed_person_seed_ci95'))
            axes[0].errorbar(mid,y,xerr=[[mid-lo],[hi-mid]],fmt='o',color=BLUE,capsize=4,
                            label='Crossed people/seed 95% interval' if y==0 else None)
            if y==2:
                lo,hi = entry['person_averaged_t_ci95']
                axes[0].errorbar(mid,y+.18,xerr=[[mid-lo],[hi-mid]],fmt='s',color=ORANGE,capsize=4,
                                label='Repair primary: person t interval')
        axes[0].set_yticks(range(3),primary_labels); axes[0].set_ylim(2.6,-.5)
        axes[0].set_title('Declared primary comparisons')
        axes[0].legend(loc='upper center',bbox_to_anchor=(.35,-.19),frameon=False,fontsize=9)
        for i,seed in enumerate(seeds):
            axes[1].scatter(differences[-1][seed],np.arange(len(people))+(i-1)*.14,
                            s=25,alpha=.65,label=f'Seed {seed}')
        axes[1].scatter(repair_person_effect,np.arange(len(people)),marker='D',s=29,color='#303c49',label='Person mean')
        person_labels = [p.replace('BioMotionLab_NTroje::','BML ').replace('KIT::','KIT ') for p in differences[-1].index]
        axes[1].set_yticks(np.arange(len(people)),person_labels,fontsize=8); axes[1].invert_yaxis()
        axes[1].set_title('Repair: each person and seed')
        axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.19),frameon=False,ncol=2,fontsize=9)
        for ax in axes:
            ax.axvline(0,color=GRAY,ls='--',lw=1); ax.grid(axis='x',alpha=.15)
            ax.set_xlabel('Comparator − candidate score (degrees)\\nPositive favors candidate')
        fig.suptitle('SOURCE RESULTS · Reconstructing uncertainty without treating frames as people',x=.03,ha='left')
        fig.tight_layout(rect=(0,0,1,.94)); finish(fig,'10-person-level-uncertainty')
        '''),
        md(r'''
        ## 10 · Check which branch contains the diagnostic signal

        A **linear probe** fits a simple weighted sum of frozen features to predict
        the reference response. Its error tests what that particular readout can
        recover from that feature branch. Here the diagnostic uses 333 pairs from
        112 training people, with people separated between probe fitting and probe
        evaluation. The representation encoder had already trained on these
        people, so this is a training-population diagnostic, not a new test cohort.

        The teacher sees reference poses; the deployed encoder sees imperfect
        observations. Showing their probe errors separately prevents teacher-side
        information from being attributed to deployed features. Probe error is
        **mean squared error** in degrees squared, unlike the absolute-error
        metrics above. A failed linear probe does not prove all usable information
        is absent or that the representation has completely collapsed.
        '''),
        code('''
        probe_families = {
            'parent-pretrain-shuffled_jepa-graph_time':'Shuffled JEPA',
            'parent-pretrain-paired_jepa-graph_time':'Paired JEPA',
            'child-pretrain-response-jepa_delta_v1-graph_time':'Delta JEPA',
            'child-pretrain-response-jepa_endpoint_v1-graph_time':'Endpoint JEPA',
        }
        probe_rows = []
        for path in sorted((DATA / 'jepa-response/diagnostics').glob('*/diagnostics.json')):
            family = path.parent.name.rsplit('-seed-',1)[0]
            if family not in probe_families: continue
            d = read_json(path)
            assert d['training_only'] is True
            for branch in ['deployment_encoder','deployment_teacher']:
                p = d['probes'][branch]
                probe_rows.append({'family':probe_families[family], 'branch':branch,
                                   'mse':p['mse'],'zero_mse':p['zero_change_mse'],
                                   'pairs':p['rows'],'people':len(p['per_person'])})
        probes = pd.DataFrame(probe_rows)
        assert probes.pairs.eq(333).all() and probes.people.eq(112).all()
        assert np.allclose(probes.zero_mse,probes.zero_mse.iloc[0])
        probe_means = probes.groupby(['family','branch']).mse.mean().unstack().reindex(list(probe_families.values()))
        fig, ax = plt.subplots(figsize=(11.6,3.9))
        x = np.arange(len(probe_means))
        ax.bar(x-.18,probe_means.deployment_encoder,.34,color=BLUE,label='Observed-input encoder')
        ax.bar(x+.18,probe_means.deployment_teacher,.34,color=ORANGE,label='Reference-input teacher')
        ax.axhline(probes.zero_mse.iloc[0],color=TEAL,ls='--',label='Zero-response probe baseline')
        ax.set_xticks(x,probe_means.index); ax.set_ylabel('Probe mean squared error (degrees²)')
        ax.set_title('SOURCE RESULTS · Feature branch and input access must accompany the probe score')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2,frameon=False,fontsize=9)
        fig.tight_layout(); finish(fig,'11-feature-branch-probes')
        '''),
        md(r'''
        ## 11 · Carry the evidence into the paper

        | Observation | What it supports | Boundary |
        | --- | --- | --- |
        | The original paired-change package increases waveform error in all five core families. | A scalar movement objective can accompany worse temporal restoration in this setup. | The comparison adds both scalar supervision and geometry regularization. |
        | Reducing the scalar coefficient recovers most of the repair's mean waveform gain. | Loss scale is an important alternative explanation for improvement over the original readout. | Initial gradient measurements do not identify a unique failure mechanism. |
        | The dense-versus-low-scalar primary interval includes zero. | The current estimate does not establish an additional dense-loss advantage on the declared primary comparison. | This is not evidence of equivalence or proof that dense supervision cannot help. |
        | Delta JEPA's primary advantage over endpoint JEPA remains uncertain. | The study has not established that paired feature-residual coupling improves restoration. | This applies to the implemented model, budget and population, not all JEPAs. |
        | Teacher probes outperform observed-input encoder probes. | Reference-side diagnostic information must be distinguished from information available at deployment. | These probes use training people and one probe family. |

        For a paper figure, combine the experiment-lineage diagram with the core
        objective comparison and the primary-contrast panel. Keep the repair
        calibration and failure decomposition nearby: they explain why a change
        in aggregate error cannot, by itself, settle the representation question.
        Use the schematic waveforms only to explain the measurement. Actual
        before/after motion examples require saved reference and prediction arrays
        with matched identifiers and predeclared selection rules.

        The [full results analysis](../../docs/studies/gait-fidelity/results/iclr-analysis-20260925/README.md)
        records additional condition analyses and the independent adversarial
        review of the scientific claims. The [visualization guide](docs/VISUAL_WALKTHROUGH.md)
        maps these panels to the original tutorials. Source tutorials 00–06 remain
        available for the underlying calculations and run workflow.
        '''),
        code('''
        # Optional export. No files are written unless you turn this on.
        EXPORT_FIGURES = False
        if EXPORT_FIGURES:
            export_dir = ROOT / 'outputs/gait-fidelity/notebook-walkthrough-20260925/figures'
            export_dir.mkdir(parents=True, exist_ok=True)
            for name,fig in FIGURES.items():
                for extension in ['png','svg','pdf']:
                    fig.savefig(export_dir / f'{name}.{extension}', dpi=180, bbox_inches='tight')
            print(f'Exported {len(FIGURES)} figures to {export_dir}')
        print('Completed: verified evidence, reconstructed primary contrasts, and displayed',len(FIGURES),'figures.')
        '''),
    ]
