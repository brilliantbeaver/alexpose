"""Independent development evaluation for the frozen response follow-up.

Only saved child predictions and receipt-bound parent predictions are consumed.
The ordinary gait-fidelity evaluator and its historical output remain unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import atomic_json, locked, read_json, sha256
from .evaluation import (crossed_interval, interaction_contrasts, load_predictions,
                         measurement_rows, nuisance_contrasts, response_contrasts,
                         _person_means)


def verified_response_completion(completed, phase=None):
    """Bind the ledger result pointer to its hashed completion and plan cell."""
    from .scheduler import verify_completed
    verify_completed(completed)
    receipt = read_json(completed['receipt'])
    if completed.get('result') != receipt.get('result'):
        raise RuntimeError('Completion ledger result differs from its hashed receipt')
    result = receipt['result']
    if phase is not None and (result.get('recipe_id') != phase['recipe']['recipe_id']
            or result.get('seed') != phase['seed'] or result.get('phase') != phase['phase']):
        raise RuntimeError('Completion recipe, seed or phase differs from its frozen plan')
    return result


def response_metrics(table, cfg):
    """Retain failed predictions in the reference-resolvable direction denominator."""
    rows = response_contrasts(table)
    tolerance = float(cfg['measurement']['sign_tolerance_deg'])
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Use the saved finite nonnegative reference sign tolerance')
    eligible = rows.reference_eligible & (rows.reference_change.abs() > tolerance)
    correct = rows.prediction_success & (np.sign(rows.predicted_change) == np.sign(rows.reference_change))
    rows['response_direction_eligible'] = eligible
    rows['response_direction_correct'] = eligible & correct
    rows['direction_accuracy'] = np.where(eligible, correct.astype(float), np.nan)
    rows['response_magnitude_error'] = np.where(rows.prediction_success,
        (rows.predicted_change.abs() - rows.reference_change.abs()).abs(),
        np.where(rows.reference_eligible, 360., np.nan))
    held = cfg.get('data', {}).get('held_level_deg')
    rows['held_intervention'] = False if held is None else np.isclose(rows.movement_level_deg, float(held), atol=1e-9, rtol=0)
    rows['reference_direction_tolerance_deg'] = tolerance
    return rows


def _limb_metrics(table):
    table = table.copy()
    for side in ('left', 'right'):
        reference, predicted = f'reference_{side}_excursion', f'predicted_{side}_excursion'
        table[f'{side}_excursion_error'] = np.where(table.prediction_success,
            (table[predicted]-table[reference]).abs(), np.where(table.reference_eligible, 180., np.nan))
    return table


def _response_fit(rows):
    """Descriptive calibration; failures are counted and never fitted as coordinates."""
    valid = rows.reference_eligible & rows.prediction_success
    x, y = rows.loc[valid, 'reference_change'].to_numpy(), rows.loc[valid, 'predicted_change'].to_numpy()
    if len(x) < 2 or np.ptp(x) < 1e-12:
        return dict(response_slope=np.nan, response_intercept=np.nan,
                    response_fit_pairs=int(valid.sum()), response_failed_pairs=int((rows.reference_eligible & ~rows.prediction_success).sum()))
    # These descriptive coefficients never enter the primary hypothesis test.
    slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
    return dict(response_slope=float(slope), response_intercept=float(intercept),
                response_fit_pairs=int(valid.sum()), response_failed_pairs=int((rows.reference_eligible & ~rows.prediction_success).sum()))


def summarize_response_predictions(bundle, exports, cfg, output, *, enforce_deadline=False):
    """Stream families; windows -> raw motions -> people precedes seed summaries."""
    from .data import select_rows
    if any(row['split'] != 'development' for row in bundle.records):
        raise PermissionError('Response follow-up evaluation admits development people only')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    streams = ('per-window', 'responses', 'nuisance', 'interaction', 'per-family')
    for name in streams:
        (output/f'{name}.csv').unlink(missing_ok=True)
    families = {}
    for index, record in enumerate(bundle.records):
        families.setdefault(record['source_family_id'], []).append(index)
    base_metrics = ['A_error', 'A_bias', 'waveform_error', 'visible_nle', 'synthetic_all_nle',
                    'displacement_nle', 'assignment_failure_rate', 'left_excursion_error', 'right_excursion_error',
                    'reference_left_excursion', 'reference_right_excursion', 'predicted_left_excursion', 'predicted_right_excursion']
    change_metrics = ['response_error', 'response_bias', 'response_magnitude_error', 'direction_accuracy']
    extended = cfg.get('followup', {}).get('include_base_readouts', False)
    if extended:
        change_metrics += ['zero_response_error', 'response_failure_rate',
                           'response_failure_contribution', 'response_success_contribution',
                           'response_success_conditional_error']
    all_metrics = base_metrics + change_metrics + ['nuisance_error', 'interaction_error']
    keys = ['method', 'seed', 'canonical_person_id']
    condition = ['camera_id', 'naming', 'observation', 'extractor', 'physical_state', 'held_intervention']
    people_tables, conditions, coverage_tables, fits, curves = [], [], [], [], []
    observed_keys = set()
    for method, seed, prediction in exports:
        if (method, seed) in observed_keys:
            raise ValueError('Duplicate method/seed prediction export')
        observed_keys.add((method, seed))
        if prediction.shape != bundle.inputs['xy'].shape:
            raise ValueError('Prediction shape differs from the frozen development population')
        family_means, family_conditions, coverages = [], [], []
        for family, indices in families.items():
            if enforce_deadline and not cfg.get('fixture', False) and cfg.get('followup', {}).get('deadline_utc'):
                from datetime import datetime, timezone
                from .followup import _time
                if datetime.now(timezone.utc) >= _time(cfg['followup']['deadline_utc']):
                    raise RuntimeError('Follow-up evaluation reached its saved deadline; partial tables are not a completed result')
            part = select_rows(bundle, indices)
            table = _limb_metrics(measurement_rows(part, prediction[indices], cfg, method, seed))
            held = cfg.get('data', {}).get('held_level_deg')
            table['held_intervention'] = False if held is None else np.isclose(table.movement_level_deg, float(held), atol=1e-9, rtol=0)
            responses = response_metrics(table, cfg)
            if extended:
                eligible, success = responses.reference_eligible, responses.prediction_success
                responses['zero_response_error'] = np.where(eligible, responses.reference_change.abs(), np.nan)
                responses['response_failure_rate'] = np.where(eligible, (~success).astype(float), np.nan)
                responses['response_failure_contribution'] = np.where(eligible, np.where(success, 0., 720.), np.nan)
                responses['response_success_contribution'] = np.where(eligible, np.where(success, responses.response_error, 0.), np.nan)
                responses['response_success_conditional_error'] = responses.response_error.where(success)
            nuisance = nuisance_contrasts(table)
            interaction = interaction_contrasts(responses)
            for name, frame in zip(streams[:4], (table, responses, nuisance, interaction)):
                path = output/f'{name}.csv'
                frame.to_csv(path, index=False, mode='a', header=not path.exists())
            record = part.records[0]
            common = dict(method=method, seed=seed, canonical_person_id=record['canonical_person_id'],
                          source_family_id=family, motion_hash=record['motion_hash'])
            nonzero = responses[responses.movement_state.ne('no_change')]
            curve_condition = ['camera_id', 'observation', 'physical_state', 'held_intervention',
                               'movement_level_deg', 'movement_state']
            curve_keys = keys + ['motion_hash', 'source_family_id'] + curve_condition
            curve_source = responses.assign(motion_hash=record['motion_hash'])
            curve_values = curve_source.groupby(curve_keys, dropna=False)[
                ['reference_change', 'predicted_change', 'response_error']].mean().reset_index()
            curve_counts = curve_source.groupby(curve_keys, dropna=False)[
                ['reference_eligible', 'prediction_success']].sum().reset_index()
            curves.append(curve_values.merge(curve_counts, on=curve_keys, validate='one_to_one'))
            means = {metric: table[metric].mean() for metric in base_metrics}
            means.update({metric: nonzero[metric].mean() for metric in change_metrics})
            means.update(nuisance_error=nuisance.nuisance_error.mean(),
                         interaction_error=interaction.loc[interaction.movement_state.ne('no_change'), 'interaction_error'].mean())
            family_means.append(dict(common, **means))
            condition_keys = keys + ['motion_hash', 'source_family_id'] + condition
            # response_contrasts intentionally retains no motion hash; recover
            # this reference metadata from the current single family only.
            nonzero = nonzero.assign(motion_hash=record['motion_hash'])
            grouped = nonzero.groupby(condition_keys, dropna=False)
            per_condition = grouped[change_metrics].mean().reset_index()
            counts = grouped[['reference_eligible', 'prediction_success', 'response_direction_eligible',
                              'response_direction_correct']].sum().reset_index()
            per_condition = per_condition.merge(counts, on=condition_keys, validate='one_to_one')
            endpoints = table.groupby(condition_keys, dropna=False)[base_metrics].mean().reset_index()
            per_condition = per_condition.merge(endpoints, on=condition_keys, how='outer', validate='one_to_one')
            for source, metric in ((nuisance, 'nuisance_error'), (interaction, 'interaction_error')):
                source = source.assign(motion_hash=record['motion_hash'], held_intervention=(False if held is None
                    else np.isclose(source.movement_level_deg, float(held), atol=1e-9, rtol=0)))
                values = source.groupby(condition_keys, dropna=False)[metric].mean().reset_index()
                per_condition = per_condition.merge(values, on=condition_keys, how='outer', validate='one_to_one')
            family_conditions.append(per_condition)
            coverages.append(dict(common, records=len(table), reference_eligible=int(table.reference_eligible.sum()),
                successful=int(table.prediction_success.sum()), response_pairs=len(nonzero),
                response_reference_eligible=int(nonzero.reference_eligible.sum()),
                response_successful=int(nonzero.prediction_success.sum()),
                direction_eligible=int(nonzero.response_direction_eligible.sum()),
                direction_correct=int(nonzero.response_direction_correct.sum())))
            fits.append(dict(common, **_response_fit(nonzero)))
        family_frame = pd.DataFrame(family_means)
        path = output/'per-family.csv'
        family_frame.to_csv(path, index=False, mode='a', header=not path.exists())
        people_tables.append(_person_means(family_frame, keys, all_metrics))
        conditions.append(pd.concat(family_conditions, ignore_index=True))
        coverage_tables.append(pd.DataFrame(coverages))
    if not people_tables:
        raise ValueError('No verified predictions were supplied')
    people = pd.concat(people_tables, ignore_index=True)
    people.to_csv(output/'per-person.csv', index=False)
    conditional = pd.concat(conditions, ignore_index=True)
    condition_keys = keys + condition
    condition_people = _person_means(conditional, condition_keys, all_metrics)
    count_columns = ['reference_eligible', 'prediction_success', 'response_direction_eligible', 'response_direction_correct']
    counts = conditional.groupby(condition_keys, dropna=False)[count_columns].sum().reset_index()
    condition_people = condition_people.merge(counts, on=condition_keys, validate='one_to_one')
    condition_people.to_csv(output/'response-by-condition-person.csv', index=False)
    strata = ['method', 'seed'] + condition
    mean = condition_people.groupby(strata, dropna=False)[all_metrics].mean().reset_index()
    count = condition_people.groupby(strata, dropna=False)[count_columns].sum().reset_index()
    mean.merge(count, on=strata, validate='one_to_one').to_csv(output/'response-by-condition.csv', index=False)
    coverage_family = pd.concat(coverage_tables, ignore_index=True)
    numeric = [column for column in coverage_family if column not in keys + ['source_family_id', 'motion_hash']]
    coverage_people = coverage_family.groupby(keys, dropna=False)[numeric].sum().reset_index()
    coverage_people.to_csv(output/'coverage-per-person.csv', index=False)
    coverage = coverage_people.groupby(['method', 'seed'])[numeric].sum().reset_index()
    coverage.to_csv(output/'coverage.csv', index=False)
    pd.DataFrame(fits).to_csv(output/'response-calibration-per-family.csv', index=False)
    curve_family = pd.concat(curves, ignore_index=True)
    curve_keys = keys + curve_condition
    curve_people = _person_means(curve_family, curve_keys, ['reference_change', 'predicted_change', 'response_error'])
    curve_counts = curve_family.groupby(curve_keys, dropna=False)[['reference_eligible', 'prediction_success']].sum().reset_index()
    curve_people = curve_people.merge(curve_counts, on=curve_keys, validate='one_to_one')
    # Seed averaging is explicit; scatter points are people, not correlated seed fits.
    curve_keys = [key for key in curve_keys if key != 'seed']
    averaged = curve_people.groupby(curve_keys, dropna=False)[['reference_change', 'predicted_change', 'response_error']].mean().reset_index()
    counts = curve_people.groupby(curve_keys, dropna=False)[['reference_eligible', 'prediction_success']].sum().reset_index()
    averaged = averaged.merge(counts, on=curve_keys, validate='one_to_one')
    averaged['prediction_coverage'] = averaged.prediction_success / averaged.reference_eligible.replace(0, np.nan)
    averaged.to_csv(output/'response-curves-person.csv', index=False)
    averaged[averaged.movement_state.eq('no_change')].to_csv(output/'response-no-change-person.csv', index=False)
    primary = cfg['evaluation']
    comparisons = {metric: crossed_interval(people, primary['primary_candidate'], primary['primary_comparator'], metric,
        draws=primary['bootstrap_draws'], seed=primary['bootstrap_seed'])
        for metric in ('response_error', 'nuisance_error', 'A_error', 'waveform_error')}
    atomic_json(output/'response-comparisons.json', comparisons)
    atomic_json(output/'comparisons.json', comparisons)
    if extended:
        atomic_json(output/'readout-control-comparisons.json', readout_control_comparisons(people, cfg))
    return people, coverage, comparisons


def readout_control_comparisons(people, cfg):
    """Prespecified secondary contrasts; the paired-change primary is unchanged."""
    from .spec import RESPONSE_VARIANTS
    primary = cfg['evaluation']
    options = dict(draws=primary['bootstrap_draws'], seed=primary['bootstrap_seed'])
    metrics = ['response_error', 'nuisance_error', 'A_error', 'waveform_error',
               'visible_nle', 'synthetic_all_nle', 'response_failure_contribution',
               'response_success_contribution', 'zero_response_error']
    def name(variant, objective):
        return f'F-response-{variant}-graph_time-{objective}'
    pairs = [('delta_vs_endpoint_base', name('jepa_delta_v1','base'), name('jepa_endpoint_v1','base'))]
    for variant in RESPONSE_VARIANTS:
        parent = 'M-coordinate-graph_time' if variant.startswith('coordinate') else 'M-paired_jepa-graph_time'
        pairs.append((f'{variant}_change_vs_base', name(variant,'paired_change'), name(variant,'base')))
        for objective in ('base', 'paired_change'):
            candidate = name(variant, objective)
            pairs += [(f'{variant}_{objective}_vs_parent', candidate, f'{parent}-{objective}'),
                      (f'{variant}_{objective}_vs_direct_base', candidate, 'P-direct-none-base'),
                      (f'{variant}_{objective}_vs_initialized', candidate, f'I-initialized-none-{objective}')]
    result = {label: {metric: crossed_interval(people, candidate, comparator, metric, **options)
                     for metric in metrics} for label, candidate, comparator in pairs}
    effects = []
    for variant in ('jepa_delta_v1', 'jepa_endpoint_v1'):
        base = people[people.method.eq(name(variant,'base'))].set_index(['canonical_person_id','seed'])
        change = people[people.method.eq(name(variant,'paired_change'))].set_index(['canonical_person_id','seed'])
        values = (change[metrics]-base[metrics]).reset_index()
        effects.append(values.assign(method=variant))
    effect_table = pd.concat(effects, ignore_index=True)
    result['coupling_by_readout_interaction'] = {
        metric: crossed_interval(effect_table, 'jepa_delta_v1', 'jepa_endpoint_v1', metric, **options)
        for metric in metrics}
    return dict(status='prespecified_secondary_development_estimates', comparisons=result,
                multiplicity='No adjustment; these contrasts do not replace the primary endpoint or establish confirmation.',
                interaction='Endpoint (change-base) error minus delta (change-base) error; positive means the delta advantage is larger with the change readout.',
                failure_decomposition='Response error equals success contribution plus 720-degree failure contribution at the identical aggregation hierarchy. Conditional error is descriptive.')


def declared_response_recipes(final, cfg):
    """Check every variant/seed/readout cell and preserve the paired-change primary."""
    from .spec import RESPONSE_VARIANTS
    objectives = ('paired_change', 'base') if cfg.get('followup', {}).get('include_base_readouts', False) else ('paired_change',)
    expected = {(variant, seed, objective) for variant in RESPONSE_VARIANTS
                for seed in (17,29,43) for objective in objectives}
    actual = {(p['recipe'].get('representation_variant'), p['seed'],
               p['recipe'].get('readout_or_training_objective','paired_change')) for p in final}
    if len(final) != len(expected) or actual != expected:
        raise ValueError('Follow-up must contain exactly the declared variant/seed/readout matrix')
    recipes = {}
    for variant in RESPONSE_VARIANTS:
        for objective in objectives:
            ids = {p['recipe']['recipe_id'] for p in final
                   if p['recipe']['representation_variant'] == variant
                   and p['recipe'].get('readout_or_training_objective','paired_change') == objective}
            if len(ids) != 1:
                raise ValueError('A representation/readout variant changed recipe identity across seeds')
            recipes[variant, objective] = next(iter(ids))
    if len(set(recipes.values())) != len(recipes):
        raise ValueError('Distinct representation/readout cells must have distinct recipe identities')
    return recipes


def _plot_response(people, output, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    evidence = 'Software fixture: generated data, not research results\n' if cfg.get('fixture') else ''
    mean = people.groupby('method')[['response_error', 'nuisance_error', 'direction_accuracy']].mean().sort_index()
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4., .38*len(mean)+1.5)), sharey=True)
    for axis, metric, title in zip(axes, mean.columns,
            ('Movement-response error (degrees)', 'Nuisance error (degrees)', 'Response direction accuracy')):
        axis.barh(np.arange(len(mean)), mean[metric].to_numpy(), color='#377b91')
        axis.set_xlabel(title)
        axis.grid(axis='x', alpha=.2)
    axes[0].set_yticks(np.arange(len(mean)), mean.index)
    axes[2].set_xlim(0, 1)
    fig.suptitle(evidence + 'Development means across people and seeds; uncertainty is in the paired comparison table')
    fig.tight_layout()
    fig.savefig(Path(output)/'response-diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    conditions = pd.read_csv(Path(output)/'response-by-condition-person.csv')
    factors = ('camera_id', 'observation', 'naming', 'held_intervention')
    fig, axes = plt.subplots(1, 4, figsize=(18, max(4., .38*len(mean)+1.5)), sharey=True)
    for axis, factor in zip(axes, factors):
        balanced = conditions.groupby(['method', 'seed', 'canonical_person_id', factor]).response_error.mean().reset_index()
        grid = balanced.groupby(['method', factor]).response_error.mean().unstack(factor).reindex(mean.index)
        image = axis.imshow(grid.to_numpy(), aspect='auto', cmap='Blues', vmin=0)
        axis.set_xticks(np.arange(len(grid.columns)), [str(value) for value in grid.columns], rotation=35, ha='right')
        axis.set_title(factor.replace('_', ' '))
        fig.colorbar(image, ax=axis, shrink=.6, label='Response error (degrees)')
    axes[0].set_yticks(np.arange(len(mean)), mean.index)
    fig.suptitle(evidence + 'Development response error by condition; equal people and seeds, no confirmatory inference')
    fig.tight_layout()
    fig.savefig(Path(output)/'response-by-condition.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    curves = pd.read_csv(Path(output)/'response-curves-person.csv')
    methods = [cfg['evaluation']['primary_candidate'], cfg['evaluation']['primary_comparator']]
    curves = curves[curves.method.isin(methods) & curves.movement_state.ne('no_change')]
    cameras, observations = sorted(curves.camera_id.unique()), sorted(curves.observation.unique())
    fig, axes = plt.subplots(len(cameras), len(observations), figsize=(6*len(observations), 4.6*len(cameras)), squeeze=False)
    for row, camera in enumerate(cameras):
        for column, observation in enumerate(observations):
            axis = axes[row, column]
            selected = curves[curves.camera_id.eq(camera) & curves.observation.eq(observation)]
            finite = selected[np.isfinite(selected.reference_change) & np.isfinite(selected.predicted_change)]
            reference_values = selected.loc[np.isfinite(selected.reference_change), 'reference_change'].to_numpy()
            bound = max(1., float(np.abs(reference_values).max()) if len(reference_values) else 1.,
                        float(np.abs(finite.predicted_change).max()) if len(finite) else 1.)
            axis.plot([-bound, bound], [-bound, bound], '--', color='.5', linewidth=1)
            for index, method in enumerate(methods):
                for held, marker in ((False, 'o'), (True, '^')):
                    points = finite[finite.method.eq(method) & finite.held_intervention.eq(held)]
                    axis.scatter(points.reference_change, points.predicted_change, marker=marker,
                        color=('#287f8e', '#cb7543')[index], alpha=.55, s=22,
                        label=f"{'Delta' if index == 0 else 'Endpoint'} / {'held dose' if held else 'seen doses'}")
            admitted, succeeded = selected.reference_eligible.sum(), selected.prediction_success.sum()
            axis.set_title(f'{camera} / {observation}\nPrediction coverage: {int(succeeded)}/{int(admitted)} endpoint contrasts')
            axis.set_xlabel('Reference response (degrees)'); axis.set_ylabel('Predicted response (degrees)')
            if finite.empty:
                axis.text(.5, .65, 'No successful predictions', transform=axis.transAxes,
                          ha='center', color='.35', bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
            axis.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(evidence + 'Person-level response means conditional on successful outputs; coverage shown; seeds averaged')
    fig.tight_layout()
    fig.savefig(Path(output)/'response-curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _evaluate_response_followup(cfg):
    """Verify the declared child fits and import only receipt-bound parent exports."""
    from .data import load_dataset
    from .scheduler import _verify_frozen, verify_completed
    from .followup import verify_parent
    if cfg.get('study_kind') != 'jepa_response_followup':
        raise ValueError('Response evaluator requires a separately frozen follow-up')
    _verify_frozen(cfg)
    verify_parent(cfg)
    work = Path(cfg['work'])
    state, plan = read_json(work/'ledger.json'), read_json(work/'plan.json')
    binding = cfg['followup']['parent_binding']
    data = load_dataset(binding['bundle'])
    indices = np.flatnonzero([row['split'] == 'development' for row in data.records])
    dev = data.subset('development')
    final = [phase for phase in plan['phases'] if phase['phase'] != 'pretrain']
    recipe_ids = declared_response_recipes(final, cfg)
    evaluated_cfg = dict(cfg, evaluation=dict(cfg['evaluation'],
        primary_candidate=recipe_ids['jepa_delta_v1','paired_change'],
        primary_comparator=recipe_ids['jepa_endpoint_v1','paired_change']))
    output = work/'evaluation'; output.mkdir(parents=True, exist_ok=True)
    prediction_receipts = []
    def exports():
        for phase in final:
            completed = state['completed'][phase['phase_id']]
            result = verified_response_completion(completed, phase)
            prediction, retained = load_predictions(result)
            if (not np.array_equal(retained, indices) or prediction.shape != dev.inputs['xy'].shape
                    or result['recipe_id'] != phase['recipe']['recipe_id'] or result['seed'] != phase['seed']):
                raise RuntimeError('Child predictions differ from frozen parent development indices')
            prediction_receipts.append(dict(origin='child', phase_id=phase['phase_id'], completed=completed))
            yield phase['recipe']['recipe_id'], phase['seed'], prediction
        for phase_id, phase in binding['baseline_phases'].items():
            completed = binding['parent_completed'][phase_id]
            result = verified_response_completion(completed, phase)
            if 'predictions' not in result:
                raise RuntimeError('A registered parent comparison lacks retained predictions')
            prediction, retained = load_predictions(result)
            if not np.array_equal(retained, indices) or prediction.shape != dev.inputs['xy'].shape:
                raise RuntimeError('Parent predictions differ from the frozen development indices')
            prediction_receipts.append(dict(origin='parent', phase_id=phase_id, completed=completed))
            if result['recipe_id'] != phase['recipe']['recipe_id'] or result['seed'] != phase['seed']:
                raise RuntimeError('Parent prediction receipt differs from its bound recipe/seed')
            yield phase['recipe']['recipe_id'], phase['seed'], prediction
    if 'evaluation' in state['completed']:
        # An evaluation receipt protects derived files; its upstream prediction
        # artifacts and population indices must independently remain unchanged.
        for _ in exports():
            pass
        verified_response_completion(state['completed']['evaluation'])
        return read_json(work/'evaluation/response-summary.json')
    people, coverage, comparisons = summarize_response_predictions(dev, exports(), evaluated_cfg, output, enforce_deadline=True)
    _plot_response(people, output, evaluated_cfg)
    status = dict(status='RESPONSE_FOLLOWUP_DEVELOPMENT_COMPLETE', trained_models=len(final),
        primary_candidate=evaluated_cfg['evaluation']['primary_candidate'],
        primary_comparator=evaluated_cfg['evaluation']['primary_comparator'],
        sign_tolerance_deg=cfg['measurement']['sign_tolerance_deg'],
        prediction_receipts=prediction_receipts, independent_confirmation=False, clinical_validation=False,
        gavd_evaluation=False, evidence_status=data.evidence_status,
        aggregation='variants within source windows, windows within motions, motions within people; seeds crossed with people',
        direction_denominator='all reference-resolvable changes; missing predictions are incorrect',
        response_calibration='descriptive within-family fit on successful outputs; failures separately retained',
        response_bias='conditional signed bias; missing outputs are penalized in primary absolute response error',
        scientific_gate='development evidence only; no confirmatory or clinical claim')
    atomic_json(output/'response-summary.json', status)
    atomic_json(output/'summary.json', status)
    primary = comparisons['response_error']
    if primary.get('status') == 'descriptive_development_estimate':
        low, high = primary['crossed_person_seed_ci95']
        estimate = (f"Control minus candidate response error: **{primary['improvement']:.4f} degrees**, "
            f"with a descriptive 95% crossed person/seed interval of **[{low:.4f}, {high:.4f}] degrees**. "
            f"Positive values favor delta JEPA. This comparison includes {primary['people']} people and {primary['seeds']} fitted seeds.\n\n")
    else:
        estimate = f"Primary comparison status: `{primary.get('status', 'unavailable')}`.\n\n"
    scope = ('**Software fixture: these numbers do not describe human movement.**\n\n' if cfg.get('fixture')
             else '**Synthetic development evidence; no independent confirmation or clinical validation.**\n\n')
    means = people.groupby('method')[['response_error', 'nuisance_error', 'synthetic_all_nle', 'waveform_error', 'direction_accuracy']].mean()
    table_lines = ['| Method | Response error (deg) | Nuisance error (deg) | Coordinate NLE | Waveform error (deg) | Direction accuracy |',
                   '| --- | ---: | ---: | ---: | ---: | ---: |']
    for method, row in means.iterrows():
        table_lines.append('| ' + method + ' | ' + ' | '.join('unsupported' if not np.isfinite(value) else f'{value:.4f}' for value in row) + ' |')
    candidate_coverage = coverage[coverage.method.eq(status['primary_candidate'])]
    coverage_text = (f"Across the candidate's {len(candidate_coverage)} seed exports, "
        f"{int(candidate_coverage.response_successful.sum())}/{int(candidate_coverage.response_reference_eligible.sum())} "
        'reference-eligible movement contrasts had successful predictions. '
        f"The direction denominator contains {int(candidate_coverage.direction_eligible.sum())} reference-resolvable contrasts across seeds; "
        'these repeated contrasts are not independent people.\n\n')
    report = ('# Response preservation follow-up\n\n' + scope +
        'This is a development comparison of three new representations across three seeds. '
        'Parent comparisons use their retained predictions and original population indices.\n\n'
        f"The primary comparison is `{status['primary_candidate']}` versus `{status['primary_comparator']}`. "
        'Its paired person/seed intervals are in `evaluation/response-comparisons.json`.\n\n' + estimate + coverage_text +
        'Descriptive means give equal weight to people and fitted seeds.\n\n' + '\n'.join(table_lines) + '\n\n'
        f"Direction accuracy is restricted only by reference |change| > {status['sign_tolerance_deg']:g} degrees; "
        'missing predictions count as incorrect within that denominator. '
        'Signed bias and slope/intercept remain conditional diagnostics; absolute response error retains the declared failure penalties. '
        'Plots show descriptive means. No confirmation, GAVD, or clinical validation is asserted.\n')
    if cfg['followup'].get('include_base_readouts', False):
        report += ('\nEach new frozen encoder has matched base and paired-change readouts. '
                   'Direct/base and both objectives of the five core controls are imported without retraining. '
                   'Prespecified secondary contrasts and the coupling-by-readout interaction are in '
                   '`evaluation/readout-control-comparisons.json`. The primary comparison above is unchanged. '
                   'Person tables include the zero-response benchmark and an exact decomposition of response '
                   'error into successful-output and failure-penalty contributions.\n')
    (work/'response-report.md').write_text(report)
    (work/'report.md').write_text(report)
    receipt = output/'complete.json'
    artifacts = {str(path.resolve()): sha256(path) for path in output.rglob('*') if path.is_file() and path != receipt}
    artifacts[str((work/'response-report.md').resolve())] = sha256(work/'response-report.md')
    artifacts[str((work/'report.md').resolve())] = sha256(work/'report.md')
    retained = dict(summary=str(output/'response-summary.json'), report=str(work/'report.md'))
    atomic_json(receipt, dict(phase_id='evaluation', config_sha256=sha256(work/'config.json'), result=retained, artifacts=artifacts))
    with locked(work/'locks/ledger.lock'):
        ledger = read_json(work/'ledger.json')
        ledger['completed']['evaluation'] = dict(receipt=str(receipt), sha256=sha256(receipt), result=retained)
        atomic_json(work/'ledger.json', ledger)
    return status


def _verify_response_followup(cfg):
    """Recompute saved statistics from verified predictions in temporary storage."""
    import tempfile
    from .data import load_dataset
    from .followup import verify_parent
    from .scheduler import _verify_frozen, verify_completed
    _verify_frozen(cfg); verify_parent(cfg)
    work = Path(cfg['work']); output = work/'evaluation'
    state, plan = read_json(work/'ledger.json'), read_json(work/'plan.json')
    if 'evaluation' not in state['completed']:
        return dict(status='RESPONSE_EVALUATION_NOT_COMPLETE', recomputed=False)
    verified_response_completion(state['completed']['evaluation'])
    binding = cfg['followup']['parent_binding']
    bundle = load_dataset(binding['bundle'])
    indices = np.flatnonzero([row['split'] == 'development' for row in bundle.records])
    dev = bundle.subset('development')
    final = [phase for phase in plan['phases'] if phase['phase'] != 'pretrain']
    recipes = declared_response_recipes(final, cfg)
    evaluated_cfg = dict(cfg, evaluation=dict(cfg['evaluation'],
        primary_candidate=recipes['jepa_delta_v1','paired_change'], primary_comparator=recipes['jepa_endpoint_v1','paired_change']))
    def exports():
        selected = [(phase, state['completed'][phase['phase_id']]) for phase in final]
        selected += [(phase, binding['parent_completed'][phase_id]) for phase_id, phase in binding['baseline_phases'].items()]
        for phase, completed in selected:
            result = verified_response_completion(completed, phase)
            prediction, retained = load_predictions(result)
            if not np.array_equal(retained, indices):
                raise RuntimeError('Recomputation prediction indices changed')
            yield phase['recipe']['recipe_id'], phase['seed'], prediction
    checked = []
    with tempfile.TemporaryDirectory(prefix='gait-response-verify-') as temporary:
        summarize_response_predictions(dev, exports(), evaluated_cfg, temporary)
        for regenerated in sorted(Path(temporary).glob('*.csv')):
            saved = output/regenerated.name
            if not saved.is_file():
                raise RuntimeError(f'Missing published response table: {saved.name}')
            try:
                pd.testing.assert_frame_equal(pd.read_csv(saved), pd.read_csv(regenerated),
                    check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
            except AssertionError as error:
                raise RuntimeError(f'Response metric recomputation disagrees: {saved.name}') from error
            checked.append(saved.name)
        json_names = ['comparisons.json', 'response-comparisons.json']
        if cfg['followup'].get('include_base_readouts', False):
            json_names.append('readout-control-comparisons.json')
        for name in json_names:
            if read_json(output/name) != read_json(Path(temporary)/name):
                raise RuntimeError(f'Response uncertainty recomputation disagrees: {name}')
            checked.append(name)
    return dict(status='RESPONSE_METRICS_RECOMPUTED', recomputed=True, checked=checked,
                published_outputs_unchanged=True, tolerance=dict(relative=1e-10, absolute=1e-10),
                independent_confirmation=False, clinical_validation=False)


def evaluate_response_followup(cfg):
    with locked(Path(cfg['work'])/'locks/response-evaluation.lock'):
        return _evaluate_response_followup(cfg)


def verify_response_followup(cfg):
    with locked(Path(cfg['work'])/'locks/response-evaluation.lock'):
        return _verify_response_followup(cfg)
