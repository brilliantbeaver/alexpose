"""Build-time cell text for the five worked experimental comparisons.

These functions are used by build_notebooks.py; their computations are emitted
directly into the notebooks, never imported as runtime teaching abstractions.
"""


def build_experiment_cells(group, md, code):
    return {'M': masking_change, 'T': structure, 'P': practical,
            'I': information, 'L': pairing}[group](md, code)


def masking_change(md, code):
    return [md(r'''
    ## Calculate the comparison within each person and training seed

    Let $E_{p,s,e,m,l}$ be response error for person $p$, seed $s$, encoder $e$,
    mask $m$, and output loss $l$. The gain from change supervision is
    $G_{p,s,e,m}=E_{p,s,e,m,\mathrm{base}}-E_{p,s,e,m,\mathrm{change}}$.
    Positive values favor change supervision. Keeping person, seed, encoder,
    and mask fixed prevents their differences from entering this contrast.

    The interaction $G_{\mathrm{JEPA}}-G_{\mathrm{coordinate}}$ asks whether
    adding change supervision helps JEPA more than coordinate pretraining.
    It does not estimate JEPA's overall advantage. We retain all masks and all
    people below. These are descriptive development comparisons; notebook 06
    explains paired uncertainty for the frozen primary comparison.
    '''), code('''
    import numpy as np
    person_file = study.work / 'evaluation/per-person.csv'
    if person_file.exists():
        person = pd.read_csv(person_file)
        design = pd.DataFrame(group_recipes).rename(columns={'recipe_id': 'method'})
        cells = person.merge(design, on='method', how='inner', validate='many_to_one')
        key = ['canonical_person_id', 'seed', 'encoder', 'pretraining_mask']
        errors = cells.pivot(index=key, columns='readout_or_training_objective', values='response_error')
        assert errors[['base', 'paired_change']].notna().all().all(), 'Report missing support before comparison.'
        gains = (errors['base'] - errors['paired_change']).rename('change_supervision_gain_deg')
        interaction = gains.unstack('encoder')
        interaction['JEPA_minus_coordinate_gain_deg'] = interaction['paired_jepa'] - interaction['coordinate']
        display(gains.reset_index())
        display(interaction.reset_index())
        print('Positive gain = lower response error; these rows reuse the same people.')
    else:
        print('Run notebooks 04–05 to calculate these contrasts from retained predictions.')
    '''), md('''
    The saved response error already gives equal weight to source families
    within each person and excludes exact no-change duplicates from its primary
    average. It includes penalties for failed predictions on eligible references.
    Inspect coverage in the final table below before interpreting a small gain;
    missing reference support can still limit what the population mean covers.
    ''')]


def structure(md, code):
    return [md(r'''
    ## Measure the remaining differences between masking policies

    For token $k$, estimate its hiding probability by
    $\hat p_k=\sum_d H_{d,k}/\sum_d O_{d,k}$, where $O$ indicates a token with
    at least one observed frame. Equal hidden-token counts do not imply equal
    $\hat p_k$. Anatomical regions have different joint memberships, and overlap
    or truncation changes the final lengths of hidden intervals.

    We use a small deterministic training subset to make the calculation visible.
    This tutorial audit is separate from the retained full-run audit. Linear run
    lengths describe the crop seen by the encoder; a cyclic interval crossing
    the crop boundary contributes two runs. The normalized Wasserstein distance
    measures the average displacement between these run-length distributions,
    divided by the number of temporal patches.
    '''), code('''
    import numpy as np
    from scipy.stats import wasserstein_distance
    from gavd6_sjepa.research_directions.gait_fidelity.data import load_dataset
    from gavd6_sjepa.research_directions.gait_fidelity.masking import sample_mask, audit_matching
    bundle = load_dataset(study.bundle_path()).subset('train')
    count = min(6, len(bundle.records))
    selected_rows = np.linspace(0, len(bundle.records) - 1, count, dtype=int)
    observed = bundle.inputs['observed'][selected_rows]
    cfg = study.artifact('config.json')
    patch = cfg['model']['patch_size']
    fraction = cfg['training']['mask_fraction']
    available = observed.reshape(count, -1, patch, 12).any(axis=2)
    draws, seed = 32, 17
    probabilities, runs, budgets = {}, {}, {}
    for policy in ['graph_time', 'shuffled_topology', 'random_joint_intervals']:
        rng = np.random.default_rng(seed)
        bank = np.stack([sample_mask(observed, policy, rng=rng, fraction=fraction,
                                     patch_size=patch) for _ in range(draws)])
        denominator = available.sum(axis=0) * draws
        probabilities[policy] = np.divide(bank.sum(axis=(0, 1)), denominator,
            out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
        budgets[policy] = bank.sum(axis=(-1, -2))
        lengths = []
        for row in bank.reshape(-1, *available.shape[1:]):
            for joint in range(12):
                transitions = np.diff(np.r_[False, row[:, joint], False].astype(int))
                lengths.extend((np.flatnonzero(transitions == -1) - np.flatnonzero(transitions == 1)).tolist())
        runs[policy] = lengths
    reference = audit_matching(observed, draws=draws, seed=seed, fraction=fraction, patch_size=patch)
    rows = []
    for policy in ['shuffled_topology', 'random_joint_intervals']:
        usable_slots = denominator > 0
        maximum = (float(abs(probabilities[policy] - probabilities['graph_time'])[usable_slots].max())
                   if usable_slots.any() else None)
        distance = (wasserstein_distance(runs[policy], runs['graph_time']) / available.shape[1]
                    if runs[policy] and runs['graph_time'] else None)
        for value, name in [(maximum, 'maximum_joint_time_probability_difference'),
                            (distance, 'normalized_run_length_Wasserstein')]:
            expected = reference['comparisons'][policy][name]
            if value is None:
                assert expected is None  # Unsupported is not a successful match.
            else:
                np.testing.assert_allclose(value, expected)
        assert np.array_equal(budgets[policy], budgets['graph_time'])
        rows.append({'control': policy, 'max_probability_difference': maximum,
                     'normalized_run_distance': distance,
                     'tutorial_default_tolerance_passed': reference['comparisons'][policy]['tolerance_passed']})
    display(pd.DataFrame(rows))
    print('These 32-draw examples reproduce the audit calculation; inspect the retained run audit too.')
    '''), md('''
    A failed matching tolerance leaves joint exposure or persistence as an
    alternative explanation for a performance difference. The production audit
    retains that failure and the control's predictions. Do not drop a control
    or adjust its tolerance after seeing which method wins.
    ''')]


def practical(md, code):
    return [md(r'''
    ## Fit the spatial controls using training people only

    In normalized coordinates, the joint offset is the weighted mean training
    residual $b_j=\sum w(y_j-x_j)/\sum w$. The affine control fits
    $W_j=(D_j^T W D_j+\Lambda)^{-1}D_j^T W(y_j-x_j)$, where each design row is
    $D_j=[x_j^x,x_j^y,1]$ and $\Lambda=\operatorname{diag}(0.001,0.001,0)$.
    The intercept is unpenalized; `solve` evaluates the expression without
    explicitly forming an inverse. A training observation needs both an
    observed input joint and a valid reference. Every development extractor
    receives the same fitted correction. Source runs give equal mass to people,
    then raw motions within a person, windows within a motion, and variants
    within a window; available frames within a joint share the window weight.
    The software fixture retains its legacy equal-frame calibration. The
    sufficient statistics below are accumulated in bounded batches, so the
    full training arrays need not be copied into RAM.

    Notebook 01 derives the input-only normalization. Here its transform is
    reused to isolate the fitting calculation. The correction is converted
    back to pixels with the development input's own origin and scale. Missing
    input joints remain missing in these two spatial controls.
    '''), code('''
    import numpy as np
    from gavd6_sjepa.research_directions.gait_fidelity.data import load_dataset
    from gavd6_sjepa.research_directions.gait_fidelity.training import normalize_inputs
    data = load_dataset(study.bundle_path())
    train, development = data.subset('train'), data.subset('development')
    train_people = {r['canonical_person_id'] for r in train.records}
    assert train_people.isdisjoint(r['canonical_person_id'] for r in development.records)
    config = study.artifact('config.json')
    balanced = config['training'].get('sampling') == 'person_motion'
    groups = {}
    for i, row in enumerate(train.records):
        groups.setdefault(row['canonical_person_id'], {}).setdefault(row['motion_hash'], {}).setdefault(row['source_family_id'], []).append(i)
    row_weights = np.ones(len(train.records))
    if balanced:
        for person in groups.values():
            for motion in person.values():
                for rows in motion.values():
                    row_weights[rows] = 1 / (len(groups) * len(person) * len(motion) * len(rows))
    gram, cross = np.zeros((12,3,3)), np.zeros((12,3,2))
    total, residual_sum = np.zeros(12), np.zeros((12,2))
    for start in range(0, len(train.records), 256):
        sl = slice(start, start + 256)
        raw = {k: v[sl] for k,v in train.inputs.items()}
        x, transform = normalize_inputs(raw)
        y = transform.apply(train.targets['xy'][sl])
        for joint in range(12):
            keep = raw['observed'][:,:,joint] & train.targets['valid'][sl,:,joint]
            weight = row_weights[sl] / np.maximum(keep.sum(1),1) if balanced else row_weights[sl]
            weight = np.broadcast_to(weight[:,None], keep.shape)[keep]
            source = x['xy'][:,:,joint][keep]
            residual = y[:,:,joint][keep] - source
            design = np.column_stack([source, np.ones(len(source))])
            gram[joint] += design.T @ (design * weight[:,None])
            cross[joint] += design.T @ (residual * weight[:,None])
            residual_sum[joint] += (residual * weight[:,None]).sum(0)
            total[joint] += weight.sum()
    assert np.all(total > 0)
    offset = residual_sum / total[:,None]
    affine = np.stack([np.linalg.solve(g + np.diag([.001,.001,0.]), c) for g,c in zip(gram,cross)])
    saved = study.artifact('evaluation/calibration.json')
    np.testing.assert_allclose(offset, saved['offset'], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(affine, saved['affine'], rtol=1e-6, atol=1e-8)
    preview = {k: v[:4] for k,v in development.inputs.items()}
    z, development_transform = normalize_inputs(preview)
    design = np.concatenate([z['xy'], np.ones((*z['observed'].shape, 1))], axis=-1)
    normalized = z['xy'] + np.einsum('ntjk,jkc->ntjc', design, affine)
    restored = np.where(z['observed'][..., None], development_transform.invert(normalized), np.nan)
    exported = np.load(study.artifact('evaluation/joint_affine.npy'), mmap_mode='r', allow_pickle=False)
    np.testing.assert_allclose(restored, exported[:4], rtol=1e-6, atol=1e-6, equal_nan=True)
    print('Training-only coefficients and development predictions match the retained calibration.')
    '''), md(r'''
    ## Reproduce interpolation and temporal smoothing on one track

    The fixed filters first linearly interpolate observed positions in physical
    time, extending the first and last observed values to the crop edges.
    They then apply centered triangular weights. Strength 1 uses $[1,2,1]/4$;
    strength 2 uses $[1,2,3,2,1]/9$. Strength 0 performs interpolation alone.
    A wholly missing joint remains missing. These are offline baselines because
    a centered filter uses frames on both sides of each output time.
    '''), code('''
    import matplotlib.pyplot as plt
    from io import BytesIO
    from IPython.display import Image, display
    from gavd6_sjepa.research_directions.synthetic_training_v2.data import filter_tracks
    one = {k: v[:1].copy() for k, v in development.inputs.items()}
    times = one['timestamps'][0]
    all_filtered = {}
    for strength in [0, 1, 2]:
        result = np.full_like(one['xy'], np.nan)
        for joint in range(12):
            keep = one['observed'][0, :, joint]
            if not keep.any():
                continue
            for axis in range(2):
                values = np.interp(times, times[keep], one['xy'][0, keep, joint, axis])
                if strength:
                    weights = np.r_[np.arange(1, strength + 2), np.arange(strength, 0, -1)]
                    weights = weights / weights.sum()
                    values = np.convolve(np.pad(values, (strength, strength), mode='edge'), weights, mode='valid')
                result[0, :, joint, axis] = values
        np.testing.assert_allclose(result, filter_tracks(one, strength), equal_nan=True)
        all_filtered[strength] = result
    fig, ax = plt.subplots(figsize=(9, 3.5), layout='constrained')
    ax.plot(times, one['xy'][0, :, 10, 0], '.', label='Estimated left ankle')
    ax.plot(times, development.targets['xy'][0, :, 10, 0], color='black', label='Projected reference')
    for strength, values in all_filtered.items():
        ax.plot(times, values[0, :, 10, 0], label=f'Filter {strength}')
    label = 'Software fixture' if study.fixture else 'Source development example'
    ax.set(xlabel='Time (s)', ylabel='Horizontal position (pixels)',
           title=label + ': first record selected by metadata order')
    ax.legend(fontsize=8)
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    display(Image(data=buffer.getvalue()))
    plt.close(fig)
    '''), md('''
    An offset preserves pixel displacement within a window; the affine and
    temporal controls can alter it. The static neural benchmark receives
    coordinates from the current frame together with an encoded full-window
    vector of confidence, availability and relative time. Its normalization
    also uses the whole input window. It removes cross-frame coordinate
    trajectories while retaining this auxiliary information. The temporal
    refiner is a study-specific adaptation, not a reproduction with published
    pretrained weights. Read both position and movement errors in notebook 05.
    ''')]


def information(md, code):
    return [md('''
    ## Construct a shuffled-reference control without crossing people or splits

    A donor replaces only the reference window used in feature pretraining.
    Donors keep the training person, physical state, camera, naming condition,
    observation, extractor, movement role and magnitude. Their source family
    must differ. A bijection within each such stratum preserves reference
    frequency. This retains nuisance information while removing the correct
    input-to-reference temporal correspondence.

    Below we reproduce the production donor assignment as short cycles across
    source families. Pairing two distinct families, with one three-cycle when
    needed, avoids a quadratic assignment matrix as the cohort grows. Seeded
    tie-breaking makes the choice reproducible. Development
    rows do not enter this calculation. Initialization-only encoders have no
    donor selection because they have no pretraining phase.
    '''), code('''
    import numpy as np
    from collections import defaultdict
    import heapq
    from gavd6_sjepa.research_directions.gait_fidelity.data import load_dataset
    from gavd6_sjepa.research_directions.gait_fidelity.training import shuffled_reference_indices
    bundle = load_dataset(study.bundle_path())
    records = bundle.records
    fields = ['canonical_person_id', 'physical_state', 'camera_id', 'naming',
              'observation', 'extractor', 'movement_state', 'movement_magnitude']
    strata = defaultdict(list)
    for index, record in enumerate(records):
        if record['split'] == 'train':
            strata[tuple(str(record.get(key)) for key in fields)].append(index)
    seed = plan['seeds'][0]
    rng = np.random.default_rng(seed + 65537)
    donors = np.arange(len(records))
    for indices in strata.values():
        families = np.array([records[i]['source_family_id'] for i in indices])
        groups = defaultdict(list)
        for local_index, family_id in enumerate(families):
            groups[str(family_id)].append(local_index)
        count = len(indices)
        assert count >= 2 and max(map(len, groups.values())) <= count // 2
        for members in groups.values(): rng.shuffle(members)
        permutation = np.full(count, -1, dtype=np.int64)
        heap = [(-len(members), float(rng.random()), key) for key, members in groups.items()]
        heapq.heapify(heap)
        def largest(number):
            return [heapq.heappop(heap)[2] for _ in range(min(number, len(heap)))]
        def replace(keys):
            for key in keys:
                if groups[key]: heapq.heappush(heap, (-len(groups[key]), float(rng.random()), key))
        if count % 2:
            keys = largest(3)
            assert len(keys) == 3
            cycle = [groups[key].pop() for key in keys]
            if rng.random() < .5: cycle.reverse()
            permutation[cycle] = np.roll(cycle, -1)
            replace(keys)
        while heap:
            keys = largest(2)
            assert len(keys) == 2
            a, b = [groups[key].pop() for key in keys]
            permutation[a], permutation[b] = b, a
            replace(keys)
        assert np.all(families != families[permutation])
        assert len(np.unique(permutation)) == count
        donors[indices] = np.asarray(indices)[permutation]
    np.testing.assert_array_equal(donors, shuffled_reference_indices(records, seed))
    example_rows = next(iter(strata.values()))
    display(pd.DataFrame([{'input_person': records[i]['canonical_person_id'],
                          'input_family': records[i]['source_family_id'],
                          'reference_family': records[donors[i]]['source_family_id'],
                          'split': records[donors[i]]['split']} for i in example_rows]))
    assert all(donors[i] == i for i, r in enumerate(records) if r['split'] != 'train')
    '''), md('''
    Notebook 04 shows which weights change during feature training and which
    remain fixed during output training. Compare these controls with graph-time
    paired JEPA under the same output loss and seed. Similar downstream errors
    would limit evidence that aligned feature prediction contributes useful
    information; they would not establish that the learned representations are
    identical. Any favorable difference still needs the direct-training and
    coordinate-pretraining benchmarks.
    ''')]


def pairing(md, code):
    return [md(r'''
    ## Separate endpoint labels from the relationship between endpoints

    Write $a_0,a_1$ for reference right-minus-left knee excursions and
    $\hat a_0,\hat a_1$ for restored values. Per-example supervision minimizes
    $\{(\hat a_0-a_0)^2+(\hat a_1-a_1)^2\}/(2\cdot180^2)$.
    Change supervision minimizes
    $\{(\hat a_1-\hat a_0)-(a_1-a_0)\}^2/180^2$.
    Both retain coordinate supervision and a short-limb penalty, derived in
    notebook 04. Change supervision can cancel a shared endpoint bias, which
    is why endpoint accuracy and change accuracy are evaluated separately.

    Valid re-pairing keeps baseline endpoints fixed and permutes intervention
    endpoints across source families within movement/nuisance strata. Its
    reference differences must be calculated again. The example below
    reproduces the core permutation and shared-minibatch mechanism on six
    **illustrative scalar labels**, not on research predictions.
    '''), code('''
    import numpy as np
    from gavd6_sjepa.research_directions.gait_fidelity.training import paired_batch_cycles, draw_pair_batch
    # Rows 0–5 are baseline endpoints, 6–11 their intervention endpoints.
    pairs = np.column_stack([np.arange(6), np.arange(6, 12)])
    permutation = np.array([1, 0, 3, 2, 5, 4])  # Three closed two-family cycles.
    repaired = np.column_stack([pairs[:, 0], pairs[permutation, 1]])
    original_labels = np.array([0., 2., 1., 4., 3., 6., 5., 10., 9., 7., 13., 8.])
    original_change = original_labels[pairs[:, 1]] - original_labels[pairs[:, 0]]
    repaired_change = original_labels[repaired[:, 1]] - original_labels[repaired[:, 0]]
    assert not np.array_equal(original_change, repaired_change)
    # Traverse the intervention permutation to form complete cycles.
    visited, cycles = set(), []
    for start in range(len(permutation)):
        if start in visited:
            continue
        current, cycle = start, []
        while current not in visited:
            visited.add(current)
            cycle.append(current)
            current = int(permutation[current])
        assert current == start
        cycles.append(np.asarray(cycle))
    for explicit, production in zip(cycles, paired_batch_cycles(pairs, repaired)):
        np.testing.assert_array_equal(explicit, production)
    # Complete cycles make the exact endpoint multiset equal at every update.
    rng = np.random.default_rng(17)
    chosen, size = [], 0
    while size < 3:
        cycle = cycles[int(rng.integers(len(cycles)))]
        chosen.append(cycle)
        size += len(cycle)
    batch = np.concatenate(chosen)
    np.testing.assert_array_equal(batch, draw_pair_batch(cycles, 3, np.random.default_rng(17)))
    np.testing.assert_array_equal(np.sort(pairs[batch].ravel()), np.sort(repaired[batch].ravel()))
    display(pd.DataFrame({'baseline': pairs[:, 0], 'original_endpoint': pairs[:, 1],
                          'new_endpoint': repaired[:, 1], 'original_change': original_change,
                          'recomputed_change': repaired_change}))
    print('Requested pairs: 3; complete-cycle batch pairs:', len(batch))
    '''), md(r'''
    ## Check how the real training control was matched

    The training code proposes two-family cycles, with one three-family cycle
    for an odd stratum. Across a fixed number of seeded trials, it selects the
    proposal minimizing reference-change distribution mismatch. For an equally
    sized stratum this is
    $W=\operatorname{mean}|\operatorname{sort}(\Delta a')-
    \operatorname{sort}(\Delta a)|/180$.
    Selection uses training references only. The worst stratum must satisfy the
    declared tolerance to support attribution specifically to meaningful pairing.
    Pairwise common reference support is recomputed when the new endpoints have
    different validity patterns. The displayed toy labels assume full support.
    '''), code('''
    mismatch = np.mean(abs(np.sort(repaired_change) - np.sort(original_change))) / 180.
    print('Illustrative normalized distribution mismatch:', mismatch)
    # Inspect retained receipts rather than rerunning the matching search.
    ledger = study.artifact('ledger.json')
    audits = []
    for phase in plan['phases']:
        recipe = phase['recipe']
        if phase['phase'] == 'pretrain' or recipe['group'] != 'L':
            continue
        result = ledger.get('completed', {}).get(phase['phase_id'], {}).get('result', {})
        checkpoint = result.get('checkpoint')
        if checkpoint:
            audit_path = Path(checkpoint).parent / 'pairing-audit.json'
            receipt = json.loads(audit_path.read_text())
            audits.append({'phase_id': phase['phase_id'],
                           **{k: receipt[k] for k in ['endpoint_frequency_preserved',
                                'worst_stratum_mismatch', 'tolerance', 'tolerance_passed']}})
    display(pd.DataFrame(audits))
    '''), md('''
    Batch size is nominal because a complete three-cycle may cross its boundary.
    All compared objectives use the same cycle sampler, so this overshoot does
    not give the re-paired arm extra endpoint exposure. A failed distribution
    tolerance stays in the report and limits the pairing claim. It must not be
    repaired by searching for a more favorable tolerance after evaluation.
    ''')]
