"""Independent physical-time metrics, complete response contrasts and uncertainty.

The evaluator never uses restored predictions to select eligible reference
timestamps. It retains failures with declared bounded angular penalties.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .common import atomic_json, read_json, sha256,locked


def angular_waveform(xy, min_segment_px=2.):
    xy=np.asarray(xy,float)
    hip,knee,ankle=xy[...,6:8,:],xy[...,8:10,:],xy[...,10:12,:]
    u,v=hip-knee,ankle-knee
    a,b=np.linalg.norm(u,axis=-1),np.linalg.norm(v,axis=-1)
    valid=np.isfinite(u).all(-1)&np.isfinite(v).all(-1)&(a>=min_segment_px)&(b>=min_segment_px)
    cosine=np.divide((u*v).sum(-1),a*b,out=np.zeros_like(a),where=valid)
    angles=np.degrees(np.arccos(np.clip(cosine,-1.,1.)))
    return np.where(valid,angles,np.nan),valid


def common_reference_support(bundle, cfg):
    ref,geometry=angular_waveform(bundle.targets['xy'],cfg['measurement']['min_segment_px'])
    valid=bundle.targets['valid'][:,:,6:8]&bundle.targets['valid'][:,:,8:10]&bundle.targets['valid'][:,:,10:12]&geometry
    support=valid.all(-1)
    families={}
    for i,r in enumerate(bundle.records):
        families.setdefault(r['source_family_id'],[]).append(i)
    # The full panel shares one physical grid and one reference-supported set.
    # More stringent than pairwise deletion; losses in coverage stay explicit.
    for rows in families.values():
        times=bundle.inputs['timestamps'][rows]
        if not np.allclose(times,times[:1],atol=1e-6,rtol=0):
            raise ValueError('Source-family physical grids disagree')
        support[rows]=np.logical_and.reduce(support[rows],axis=0)
    return ref,support


def measurement_rows(bundle,predictions,cfg,method,seed):
    from ..synthetic_training_v2.evaluation import evaluate_predictions
    ref,support=common_reference_support(bundle,cfg)
    pred,usable=angular_waveform(predictions,cfg['measurement']['min_segment_px'])
    records=[dict(r,seed=seed) for r in bundle.records]
    coordinate=evaluate_predictions(predictions,bundle.targets,bundle.inputs['timestamps'],records,
                                    method=method,evidence_status=bundle.evidence_status)
    rows=[];minimum=cfg['measurement']['min_frames'];fraction=cfg['measurement']['min_coverage']
    for i,r in enumerate(records):
        ok=support[i]; eligible=int(ok.sum())>=minimum and float(ok.mean())>=fraction
        success=eligible and bool(usable[i,ok].all())
        y=np.quantile(ref[i,ok], [.05,.95],axis=0,method='linear') if eligible else None
        yq=y[1]-y[0] if eligible else [np.nan,np.nan]
        pq=np.diff(np.quantile(pred[i,ok],[.05,.95],axis=0,method='linear'),axis=0)[0] if success else [np.nan,np.nan]
        target=float(yq[1]-yq[0]);estimated=float(pq[1]-pq[0])
        row=dict(r,method=method,reference_eligible=eligible,prediction_success=success,
                 support_frames=int(ok.sum()),reference_left_excursion=float(yq[0]),reference_right_excursion=float(yq[1]),
                 reference_A=target,predicted_left_excursion=float(pq[0]),predicted_right_excursion=float(pq[1]),predicted_A=estimated,
                 A_error=abs(estimated-target) if success else (360. if eligible else np.nan),
                 A_bias=estimated-target if success else np.nan,
                 waveform_error=float(np.mean(abs(pred[i,ok]-ref[i,ok]))) if success else (180. if eligible else np.nan),
                 sign_eligible=bool(eligible and abs(target)>cfg['measurement']['sign_tolerance_deg']),
                 sign_correct=bool(success and np.sign(estimated)==np.sign(target)))
        # Independent coordinate evaluator maintains its missing-prediction rule.
        row.update({k:v for k,v in coordinate.iloc[i].to_dict().items() if k not in row})
        # Geometric naming diagnostic; unresolved reference separation stays
        # explicit and is never counted as a correct anatomical assignment.
        truth=bundle.targets['xy'][i,:,6:12].reshape(-1,3,2,2)
        output=np.asarray(predictions[i,:,6:12]).reshape(-1,3,2,2)
        ref_valid=bundle.targets['valid'][i,:,6:12].reshape(-1,3,2).all(-1)
        separation=np.linalg.norm(truth[:,:,0]-truth[:,:,1],axis=-1)
        resolvable=ref_valid & (separation>=cfg['measurement'].get('assignment_separation_px',4.))
        finite=np.isfinite(output).all((-1,-2))
        named=np.linalg.norm(output-truth,axis=-1).sum(-1)
        swapped=np.linalg.norm(output-truth[:,:,::-1],axis=-1).sum(-1)
        margin=cfg['measurement'].get('assignment_margin_px',2.)
        wrong=resolvable & finite & (swapped+margin<named)
        ambiguous=resolvable & finite & (abs(named-swapped)<=margin)
        denom=int(resolvable.sum())
        row.update(assignment_reference_pairs=denom,assignment_wrong_pairs=int(wrong.sum()),
                   assignment_ambiguous_pairs=int(ambiguous.sum()),assignment_missing_pairs=int((resolvable&~finite).sum()),
                   assignment_failure_rate=float((wrong|ambiguous|(resolvable&~finite)).sum()/denom) if denom else np.nan,
                   assignment_status='geometric lower-limb diagnostic; no anatomical anchor or switch-probability claim')
        rows.append(row)
    return pd.DataFrame(rows)


def response_contrasts(table):
    """Compare movement levels within a fixed observation and camera realization."""
    rows=[]
    group=['method','seed','canonical_person_id','source_family_id','physical_state','camera_id','naming','observation','extractor']
    for keys,values in table.groupby(group,dropna=False,sort=True):
        level='movement_level_deg'
        values=values.sort_values(level)
        zero=values[values.movement_state.eq('baseline')]
        if len(zero)!=1:
            raise ValueError('Each response stratum needs exactly one zero-change endpoint')
        base=zero.iloc[0]
        for _,other in values[~values.movement_state.eq('baseline')].iterrows():
            eligible=bool(base.reference_eligible and other.reference_eligible)
            success=eligible and bool(base.prediction_success and other.prediction_success)
            reference=other.reference_A-base.reference_A
            predicted=other.predicted_A-base.predicted_A
            rows.append(dict(zip(group,keys)) | dict(split=other['split'],movement_level_deg=other[level],movement_state=other.movement_state,
                        reference_eligible=eligible,prediction_success=success,reference_change=reference,
                        predicted_change=predicted,response_error=abs(predicted-reference) if success else (720. if eligible else np.nan),
                        response_bias=predicted-reference if success else np.nan))
    return pd.DataFrame(rows)


def nuisance_contrasts(table):
    rows=[]
    group=['method','seed','canonical_person_id','source_family_id','physical_state','camera_id','movement_state','movement_level_deg','extractor']
    for keys,values in table.groupby(group,dropna=False,sort=True):
        base=values[values.naming.eq('correct')&values.observation.eq('clear')]
        if len(base)!=1:
            raise ValueError('Each nuisance stratum needs one correctly named clear reference condition')
        base=base.iloc[0]
        for _,row in values.iterrows():
            if row.naming=='correct' and row.observation=='clear':
                continue
            eligible=bool(row.reference_eligible and base.reference_eligible)
            success=eligible and bool(row.prediction_success and base.prediction_success)
            delta=(row.predicted_A-base.predicted_A)-(row.reference_A-base.reference_A)
            rows.append(dict(zip(group,keys)) | dict(split=row['split'],naming=row.naming,observation=row.observation,
                        reference_eligible=eligible,prediction_success=success,nuisance_error=abs(delta) if success else (720. if eligible else np.nan),
                        nuisance_bias=delta if success else np.nan))
    return pd.DataFrame(rows)


def interaction_contrasts(responses):
    rows=[]
    group=['method','seed','canonical_person_id','source_family_id','physical_state','camera_id','movement_state','movement_level_deg','extractor']
    for keys,values in responses.groupby(group,dropna=False,sort=True):
        baseline=values[values.naming.eq('correct')&values.observation.eq('clear')]
        if len(baseline)!=1:
            raise ValueError('Interaction requires one correctly named clear movement contrast')
        base=baseline.iloc[0]
        for _,row in values.iterrows():
            if row.naming=='correct' and row.observation=='clear':
                continue
            eligible=bool(row.reference_eligible and base.reference_eligible)
            success=eligible and bool(row.prediction_success and base.prediction_success)
            signed=row.response_bias-base.response_bias
            rows.append(dict(zip(group,keys))|dict(naming=row.naming,observation=row.observation,
                        reference_eligible=eligible,prediction_success=success,interaction_bias=signed if success else np.nan,
                        interaction_error=abs(signed) if success else (1440. if eligible else np.nan)))
    return pd.DataFrame(rows)


def crossed_interval(person_table,candidate,comparator,metric,*,draws=2000,seed=731):
    """Resample people and fitted seeds as crossed factors, with paired methods."""
    selected=person_table[person_table.method.isin([candidate,comparator])]
    pivot=selected.pivot(index=['canonical_person_id','seed'],columns='method',values=metric)
    if candidate not in pivot or comparator not in pivot or pivot.isna().any().any():
        return dict(status='insufficient_support',metric=metric)
    delta=(pivot[comparator]-pivot[candidate]).unstack('seed')
    if delta.isna().any().any() or len(delta)<2:
        return dict(status='insufficient_people_or_seed_coverage',metric=metric)
    x=delta.to_numpy();rng=np.random.default_rng(seed)
    person_draws=[];crossed=[]
    for _ in range(draws):
        p=rng.integers(len(x),size=len(x));s=rng.integers(x.shape[1],size=x.shape[1])
        person_draws.append(float(x[p].mean()));crossed.append(float(x[p][:,s].mean()))
    return dict(status='descriptive_development_estimate',metric=metric,candidate=candidate,comparator=comparator,
                people=len(x),seeds=x.shape[1],improvement=float(x.mean()),
                person_conditional_ci95=np.quantile(person_draws,[.025,.975]).tolist(),
                crossed_person_seed_ci95=np.quantile(crossed,[.025,.975]).tolist(),
                per_seed_improvement={str(k):float(v) for k,v in delta.mean().items()},
                interpretation='Positive favors candidate; three seeds limit training-variation precision. No confirmatory significance claim.')


def load_predictions(receipt):
    """Verify either legacy compressed predictions or memory-mapped exports."""
    path = Path(receipt['predictions'])
    if sha256(path) != receipt['predictions_sha256']:
        raise RuntimeError('Retained predictions changed before evaluation')
    if path.suffix == '.npy':
        index_path = Path(receipt['prediction_indices'])
        if sha256(index_path) != receipt['prediction_indices_sha256']:
            raise RuntimeError('Prediction population indices changed')
        return np.load(path, mmap_mode='r', allow_pickle=False), np.load(index_path, allow_pickle=False)
    with np.load(path, allow_pickle=False) as saved:
        return saved['xy'], saved['indices']


def calibration_row_weights(records):
    """Equal people, then raw motions, then windows, then observed variants."""
    groups = {}
    for i, row in enumerate(records):
        groups.setdefault(row['canonical_person_id'], {}).setdefault(row['motion_hash'], {}).setdefault(row['source_family_id'], []).append(i)
    weights = np.empty(len(records), float)
    for person in groups.values():
        for motion in person.values():
            for rows in motion.values():
                weights[rows] = 1. / (len(groups) * len(person) * len(motion) * len(rows))
    return weights


def fit_calibration(train, *, balanced=False, batch_size=256):
    """Accumulate training-only normal equations without loading the full cohort.

    Balanced mode gives each supported window its fixed hierarchical weight;
    frames within that window share its weight. Joint-specific missing support
    remains explicit in the accumulated mass. The ridge is fixed, never selected
    using development or confirmation references.
    """
    from .training import normalize_inputs
    if any(r['split'] != 'train' for r in train.records):
        raise ValueError('Calibration fitting accepts training people only')
    weights = calibration_row_weights(train.records) if balanced else None
    gram = np.zeros((12, 3, 3)); cross = np.zeros((12, 3, 2))
    total = np.zeros(12); residual_sum = np.zeros((12, 2))
    for start in range(0, len(train.records), batch_size):
        sl = slice(start, start + batch_size)
        raw = {k: v[sl] for k, v in train.inputs.items()}
        normalized, transform = normalize_inputs(raw)
        x = normalized['xy']; y = transform.apply(train.targets['xy'][sl])
        for joint in range(12):
            ok = raw['observed'][:, :, joint] & train.targets['valid'][sl, :, joint]
            counts = ok.sum(1)
            row_weight = np.ones(len(ok)) if weights is None else weights[sl] / np.maximum(counts, 1)
            frame_weight = np.broadcast_to(row_weight[:, None], ok.shape)[ok]
            source = x[:, :, joint][ok]; residual = y[:, :, joint][ok] - source
            design = np.column_stack([source, np.ones(len(source))])
            gram[joint] += design.T @ (design * frame_weight[:, None])
            cross[joint] += design.T @ (residual * frame_weight[:, None])
            residual_sum[joint] += (residual * frame_weight[:, None]).sum(0)
            total[joint] += frame_weight.sum()
    if (total <= 0).any():
        raise ValueError('Training calibration has an unsupported joint')
    offset = residual_sum / total[:, None]
    penalty = np.diag([.001, .001, 0.])
    affine = np.stack([np.linalg.solve(g + penalty, c) for g, c in zip(gram, cross)])
    return dict(offset=offset.tolist(), affine=affine.tolist(), ridge=.001,
                joint_support_mass=total.tolist(), fitting_split='train',
                weighting='person_motion_window_variant' if balanced else 'observed_frame',
                repeat_seed_rows='same deterministic predictions')


def apply_calibration(inputs, calibration):
    """Apply fixed training parameters using observation-only normalization."""
    from .training import normalize_inputs
    normalized, transform = normalize_inputs(inputs)
    observed = inputs['observed']; x = normalized['xy']
    offset = np.asarray(calibration['offset']); affine = np.asarray(calibration['affine'])
    design = np.concatenate([x, np.ones((*observed.shape, 1))], -1)
    corrected = x + np.einsum('ntjk,jkc->ntjc', design, affine)
    return dict(joint_offset=np.where(observed[..., None], transform.invert(x + offset), np.nan),
                joint_affine=np.where(observed[..., None], transform.invert(corrected), np.nan))


def baseline_predictions(train, development):
    from ..synthetic_training_v2.data import filter_tracks
    calibration = fit_calibration(train)
    inputs = {k: v[:] for k, v in development.inputs.items()}
    result = dict(unchanged=inputs['xy'].copy(), **apply_calibration(inputs, calibration))
    result.update({f'filter{i}': filter_tracks(inputs, i) for i in range(3)})
    return result, calibration


def write_baseline_predictions(bundle, calibration, output):
    """One small observation batch at a time; output fields stay disk-backed."""
    from ..synthetic_training_v2.data import filter_tracks
    output = Path(output)
    names = ['unchanged', 'joint_offset', 'joint_affine', 'filter0', 'filter1', 'filter2']
    arrays = {name: np.lib.format.open_memmap(output / f'{name}.npy', mode='w+', dtype=np.float32,
                                            shape=bundle.inputs['xy'].shape) for name in names}
    for start in range(0, len(bundle.records), 128):
        sl = slice(start, start + 128)
        inputs = {k: v[sl] for k, v in bundle.inputs.items()}
        values = dict(unchanged=inputs['xy'], **apply_calibration(inputs, calibration))
        values.update({f'filter{i}': filter_tracks(inputs, i) for i in range(3)})
        for name, value in values.items():
            arrays[name][sl] = value
    for value in arrays.values():
        value.flush()
    return arrays


def _person_means(families, keys, columns):
    # Windows within raw motions precede motion/person averaging, so long
    # recordings cannot become extra independent people or dominate the mean.
    return families.groupby(keys + ['motion_hash'], dropna=False)[columns].mean().groupby(keys, dropna=False).mean().reset_index()


def summarize_predictions(bundle, exports, cfg, output):
    """Stream complete source families; retain detailed CSVs and balanced means.

    ``exports`` yields (method, seed, prediction_array). Each family's complete
    reference panel defines its support before any method predictions are read.
    Only small per-family summaries accumulate in memory across methods.
    """
    from .data import select_rows
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    families = {}
    for index, row in enumerate(bundle.records):
        families.setdefault(row['source_family_id'], []).append(index)
    files = ['per-window', 'responses', 'nuisance', 'interaction', 'per-family']
    for name in files:
        (output / f'{name}.csv').unlink(missing_ok=True)
    person_tables, condition_tables, coverage_tables = [], [], []
    metrics = ['A_error', 'waveform_error', 'visible_nle', 'synthetic_all_nle', 'displacement_nle', 'assignment_failure_rate']
    context = [column for column in ('source_dataset', 'motion_label') if all(column in r for r in bundle.records)]
    keys = ['method', 'seed', 'canonical_person_id']
    all_metrics = metrics + ['response_error', 'nuisance_error', 'interaction_error']
    strata = ['method', 'seed', 'extractor', 'camera_id', 'naming', 'observation'] + context
    condition_keys = ['method', 'seed', 'canonical_person_id', 'motion_hash', 'source_family_id', 'extractor', 'camera_id', 'naming', 'observation'] + context
    for method, seed, prediction in exports:
        family_means, condition_means, coverages = [], [], []
        if prediction.shape != bundle.inputs['xy'].shape:
            raise ValueError('Prediction shape differs from declared evaluation population')
        for family, indices in families.items():
            part = select_rows(bundle, indices)
            table = measurement_rows(part, prediction[indices], cfg, method, seed)
            responses = response_contrasts(table)
            nuisance = nuisance_contrasts(table)
            interaction = interaction_contrasts(responses)
            for name, frame in zip(files[:4], [table, responses, nuisance, interaction]):
                path = output / f'{name}.csv'
                frame.to_csv(path, index=False, mode='a', header=not path.exists())
            first = part.records[0]
            common = dict(method=method, seed=seed, canonical_person_id=first['canonical_person_id'],
                          source_family_id=family, motion_hash=first['motion_hash'])
            common.update({column:first[column] for column in context})
            means = {metric: table[metric].mean() for metric in metrics}
            for values, metric in [(responses[responses.movement_state.ne('no_change')], 'response_error'),
                                    (nuisance, 'nuisance_error'),
                                    (interaction[interaction.movement_state.ne('no_change')], 'interaction_error')]:
                means[metric] = values[metric].mean()
            family_means.append(dict(common, **means))
            condition_means.append(table.groupby(condition_keys, dropna=False)[metrics].mean().reset_index())
            coverages.append(dict(common, records=len(table), reference_eligible=int(table.reference_eligible.sum()),
                                  successful=int(table.prediction_success.sum())))
        family_table = pd.DataFrame(family_means)
        path = output / 'per-family.csv'
        family_table.to_csv(path, index=False, mode='a', header=not path.exists())
        person_tables.append(_person_means(family_table, keys, all_metrics))
        conditions = pd.concat(condition_means, ignore_index=True)
        condition_tables.append(_person_means(conditions, strata + ['canonical_person_id'], metrics))
        coverage_tables.append(pd.DataFrame(coverages).groupby(keys + context, dropna=False)[['records', 'reference_eligible', 'successful']].sum().reset_index())
    person = pd.concat(person_tables, ignore_index=True)
    person.to_csv(output / 'per-person.csv', index=False)
    condition_people = pd.concat(condition_tables, ignore_index=True)
    condition_people.to_csv(output / 'by-condition-person.csv', index=False)
    condition_people.groupby(strata, dropna=False)[metrics].mean().reset_index().to_csv(output / 'by-condition.csv', index=False)
    coverage_people = pd.concat(coverage_tables, ignore_index=True)
    coverage_people.to_csv(output / 'coverage-per-person.csv', index=False)
    if context:
        coverage_people.groupby(['method','seed'] + context, dropna=False)[['records','reference_eligible','successful']].sum().reset_index().to_csv(output/'coverage-by-source-motion.csv',index=False)
    coverage = coverage_people.groupby(['method', 'seed'])[['records', 'reference_eligible', 'successful']].sum().reset_index()
    coverage.to_csv(output / 'coverage.csv', index=False)
    primary = cfg['evaluation']
    comparisons = {metric: crossed_interval(person, primary['primary_candidate'], primary['primary_comparator'], metric,
                   draws=primary['bootstrap_draws'], seed=primary['bootstrap_seed'])
                   for metric in ['response_error', 'nuisance_error', 'A_error', 'waveform_error']}
    atomic_json(output / 'comparisons.json', comparisons)
    return person, coverage, comparisons


def evaluate_study(cfg):
    from .data import load_dataset
    from .scheduler import verify_completed, _verify_frozen
    work = Path(cfg['work']); state = read_json(work / 'ledger.json'); plan = read_json(work / 'plan.json')
    _verify_frozen(cfg); verify_completed(state['completed']['prepare'])
    data = load_dataset(state['completed']['prepare']['result']['bundle'])
    dev = data.subset('development'); train = data.subset('train')
    indices = np.flatnonzero([r['split'] == 'development' for r in data.records])
    output = work / 'evaluation'; output.mkdir(exist_ok=True)
    prediction_hashes = {}
    calibration = fit_calibration(train, balanced=cfg.get('training', {}).get('sampling') == 'person_motion')
    atomic_json(output / 'calibration.json', calibration)
    def exports():
        for phase in plan['phases']:
            if phase['phase'] == 'pretrain':
                continue
            result = state['completed'][phase['phase_id']]['result']
            prediction, retained = load_predictions(result)
            if not np.array_equal(retained, indices):
                raise ValueError('Prediction export population or order changed')
            prediction_hashes[result['predictions']] = result['predictions_sha256']
            yield phase['recipe']['recipe_id'], phase['seed'], prediction
        baselines = write_baseline_predictions(dev, calibration, output)
        for name, prediction in baselines.items():
            for seed in cfg['seeds']:
                yield name, seed, prediction
    person, coverage, comparisons = summarize_predictions(dev, exports(), cfg, output)
    status = dict(status='SOFTWARE_FIXTURE_COMPLETE' if cfg['fixture'] else 'DEVELOPMENT_MATRIX_COMPLETE',
                  trained_models=plan['counts']['final_fits'], evidence_status=data.evidence_status,
                  independent_confirmation=False, clinical_validation=False,
                  prediction_hashes=prediction_hashes, scientific_gate='insufficient_evidence',
                  report=str(work / 'report.md'),
                  aggregation='equal variants within windows, windows within raw motions, motions within people',
                  effective_training_budget=state['completed'].get('profile', {}).get('result', {}).get('budget', {}).get('selected', cfg['training']))
    atomic_json(output / 'summary.json', status)
    _write_report(cfg, person, coverage, comparisons, status)
    retain_evaluation_receipt(cfg)
    return status


def retain_evaluation_receipt(cfg):
    """Bind every published baseline, metric table and report to the run ledger."""
    work=Path(cfg['work']);output=work/'evaluation'
    receipt=output/'complete.json'
    artifacts={str(path.resolve()):sha256(path) for path in output.rglob('*')
               if path.is_file() and path!=receipt}
    artifacts[str((work/'report.md').resolve())]=sha256(work/'report.md')
    atomic_json(receipt,dict(phase_id='evaluation',config_sha256=sha256(work/'config.json'),
                            result=dict(summary=str(output/'summary.json'),report=str(work/'report.md')),
                            artifacts=artifacts))
    with locked(work/'locks/ledger.lock'):
        state=read_json(work/'ledger.json')
        state['completed']['evaluation']=dict(receipt=str(receipt),sha256=sha256(receipt),
                                              result=dict(summary=str(output/'summary.json'),report=str(work/'report.md')))
        atomic_json(work/'ledger.json',state)
    return receipt


def _write_report(cfg,person,coverage,comparisons,status):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    work=Path(cfg['work']);summary=person.groupby('method')[['A_error','response_error','nuisance_error','waveform_error']].mean()
    # Separate rows keep every method readable even when its values coincide.
    height=max(5.,.34*len(summary)+1.8)
    fig,axes=plt.subplots(1,2,figsize=(16,height),sharey=True,layout='constrained')
    positions=np.arange(len(summary))
    for ax,metric,title in zip(axes,('A_error','response_error'),
                              ('Excursion-difference error','Movement-response error')):
        values=summary[metric].to_numpy(float)
        ax.scatter(values,positions,s=32,color='#246b8e',zorder=3)
        ax.set(title=title,xlabel='Image-plane degrees · lower is better')
        ax.set_xlim(left=0)
        ax.set_ylim(len(summary)-.4,-.6)
        ax.grid(axis='x',color='#dbe3e9',linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y',length=0,labelsize=9)
        for index,value in enumerate(values):
            if not np.isfinite(value):
                ax.text(.02,index,'unsupported',transform=ax.get_yaxis_transform(),
                        va='center',fontsize=9,color='#666666')
        for spine in ('top','right','left'):
            ax.spines[spine].set_visible(False)
    axes[0].set_yticks(positions,summary.index.tolist())
    axes[1].tick_params(labelleft=False)
    fig.suptitle('Gait Fidelity: descriptive means across people and fitted seeds\n'
                 'One row per method; uncertainty is reported in comparisons.json',fontsize=13)
    fig.savefig(work/'evaluation/measurement-response.png',dpi=150,bbox_inches='tight');plt.close(fig)
    lines=['# Gait Fidelity experiment report','',f"Execution status: **{status['status']}**. Evidence: **{status['evidence_status']}**.",'',
           'The table averages source windows within raw motions, then motions within people, then people and fitted seeds. '
           'Repeated renders, extraction methods and seeds are not additional participants. '
           'Angular measurements use image-plane knee excursion and do not establish clinical range of motion.','',
           '| Method | Excursion-difference error | Movement-response error | Observation-induced error | Angle-waveform error |',
           '| --- | ---: | ---: | ---: | ---: |']
    for name,row in summary.iterrows():
        lines.append('| '+name+' | '+' | '.join('unsupported' if not np.isfinite(v) else f'{v:.4f}' for v in row)+' |')
    lines+=['','Reference eligibility is fixed across the complete source-family panel. '
            'Missing predictions receive bounded worst-case angular penalties: 360 degrees for the signed difference, '
            '720 for a movement or observation contrast, and 180 for the angular waveform. '
            'Reference-ineligible cases remain in coverage.csv; supported means cannot establish a population claim without coverage.','',
            'The primary comparison and person/seed uncertainty are in `evaluation/comparisons.json`. '
            'The primary response averages nonzero intervention levels; the exact duplicated no-change control remains a separate diagnostic in responses.csv. '
            'These are development estimates; confirmation, useful-effect margins and clinical references remain separate requirements.','',
            'Calibration uses training people only. The same deterministic predictions are repeated across seed rows for paired bookkeeping. '
            'The temporal refiner is a source-trained body-12 adaptation, not a reproduction using the original authors’ weights.','',
            'Inspect `evaluation/per-window.csv`, `responses.csv`, `nuisance.csv`, `per-person.csv`, '
            '`by-condition.csv`, and `coverage.csv` alongside the retained videos and predictions.','']
    (work/'report.md').write_text('\n'.join(lines))
