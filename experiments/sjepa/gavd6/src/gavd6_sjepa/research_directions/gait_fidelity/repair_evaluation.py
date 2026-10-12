"""Split-preserving, paired evaluation of the frozen readout-repair experiment.

The primary endpoint is waveform error on the declared extractor. Response
error remains an explicit tradeoff: an interval crossing zero never establishes
noninferiority. Publication is atomic, including all 27 declared model exports.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import nct, t

from .common import atomic_json, digest, locked, read_json, sha256, utc_now
from .data import select_rows
from .evaluation import (_person_means, crossed_interval, interaction_contrasts,
                         measurement_rows, nuisance_contrasts)
from .response_evaluation import response_metrics


SEEDS = (17, 29, 43)
VARIANTS = ('jepa_delta_v1', 'jepa_endpoint_v1')
REPAIR_METHODS = tuple(f'R-repair-{variant}-{objective}' for variant in VARIANTS
                       for objective in ('dense_change', 'scalar_low'))
INHERITED_METHODS = tuple(f'F-response-{variant}-graph_time-{objective}'
                          for variant in VARIANTS for objective in ('base', 'paired_change')) + ('P-direct-none-base',)
METHODS = REPAIR_METHODS + INHERITED_METHODS
PRIMARY_CANDIDATE = 'R-repair-jepa_delta_v1-dense_change'
PRIMARY_COMPARATOR = 'R-repair-jepa_delta_v1-scalar_low'
KEYS = ['method', 'seed', 'canonical_person_id']
BASE_METRICS = ['A_error', 'waveform_error', 'visible_nle', 'synthetic_all_nle',
                'displacement_nle', 'assignment_failure_rate', 'waveform_failure_rate',
                'waveform_failure_contribution', 'waveform_success_contribution',
                'waveform_success_conditional_error']
RESPONSE_METRICS = ['response_error', 'direction_accuracy', 'zero_response_error',
                    'response_failure_rate', 'response_failure_contribution',
                    'response_success_contribution', 'response_success_conditional_error']
METRICS = BASE_METRICS + RESPONSE_METRICS + ['nuisance_error', 'interaction_error']
EXPECTED_FITS = {(method, seed) for method in METHODS for seed in SEEDS}


def _check_deadline(config):
    """Cooperative source cutoff; fixture verification has no wall-clock gate."""
    if config.get('fixture', False):
        return
    value = config.get('repair', {}).get('deadline_utc')
    try:
        deadline = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except (TypeError, ValueError) as error:
        raise ValueError('Source repair evaluation requires an explicit ISO deadline_utc') from error
    if deadline.tzinfo is None:
        raise ValueError('Repair evaluation deadline must have an explicit timezone')
    if datetime.now(timezone.utc) >= deadline:
        raise TimeoutError('Repair evaluation reached its frozen deadline; partial results are not complete')


def _json_safe(value):
    """Undefined descriptive quantities are null, never nonstandard JSON NaN."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _protocol(config):
    options = deepcopy(config.get('repair', {}).get('statistical_protocol', {}))
    if tuple(config.get('seeds', SEEDS)) != SEEDS or tuple(options.get('seeds', SEEDS)) != SEEDS:
        raise ValueError('Repair evaluation requires the three fixed seeds 17, 29, 43')
    for key, expected in (('primary_candidate', PRIMARY_CANDIDATE),
                          ('primary_comparator', PRIMARY_COMPARATOR), ('primary_metric', 'waveform_error')):
        if options.get(key, expected) != expected:
            raise ValueError(f'The fixed repair {key} cannot be changed during evaluation')
        options[key] = expected
    if not str(options.get('primary_extractor', '')).strip():
        raise ValueError('Declare a primary extractor before repair evaluation')
    for key, default in (('bootstrap_draws', 2000), ('bootstrap_seed', 731)):
        options[key] = int(options.get(key, config.get('evaluation', {}).get(key, default)))
    if options['bootstrap_draws'] < 20:
        raise ValueError('At least 20 bootstrap draws are required, including fixtures')
    options['seeds'] = list(SEEDS)
    return options


def _extractor_name(records, declared):
    exact = {str(row['extractor']) for row in records if str(row['extractor']) == declared}
    matches = exact or {str(row['extractor']) for row in records if str(row.get('extractor_family', '')) == declared}
    if len(matches) != 1:
        raise ValueError('Primary extractor must resolve to exactly one retained extractor identity')
    return next(iter(matches))


def response_failure_decomposition(responses):
    """The primary 720-degree penalty decomposes at the same averaging level."""
    rows = responses.copy()
    eligible, success = rows.reference_eligible, rows.prediction_success
    rows['zero_response_error'] = np.where(eligible, rows.reference_change.abs(), np.nan)
    rows['response_failure_rate'] = np.where(eligible, (~success).astype(float), np.nan)
    rows['response_failure_contribution'] = np.where(eligible, np.where(success, 0., 720.), np.nan)
    rows['response_success_contribution'] = np.where(eligible, np.where(success, rows.response_error, 0.), np.nan)
    rows['response_success_conditional_error'] = rows.response_error.where(success)
    valid = rows.loc[eligible]
    if not np.allclose(valid.response_error,
                       valid.response_failure_contribution + valid.response_success_contribution,
                       atol=1e-10, rtol=1e-10):
        raise ValueError('Response failure decomposition does not reconstruct absolute error')
    return rows


def waveform_failure_decomposition(table):
    """Distinguish angular-shape error from the unchanged 180-degree failure score."""
    rows = table.copy()
    eligible, success = rows.reference_eligible, rows.prediction_success
    rows['waveform_failure_rate'] = np.where(eligible, (~success).astype(float), np.nan)
    rows['waveform_failure_contribution'] = np.where(eligible, np.where(success, 0., 180.), np.nan)
    rows['waveform_success_contribution'] = np.where(eligible, np.where(success, rows.waveform_error, 0.), np.nan)
    rows['waveform_success_conditional_error'] = rows.waveform_error.where(success)
    valid = rows.loc[eligible]
    if not np.allclose(valid.waveform_error,
                       valid.waveform_failure_contribution + valid.waveform_success_contribution,
                       atol=1e-10, rtol=1e-10):
        raise ValueError('Waveform failure decomposition does not reconstruct absolute error')
    return rows


def paired_comparison(people, candidate, comparator, metric, *, split, draws=2000, seed=731):
    """People are units; fitted seeds are paired and averaged within each person."""
    metadata = dict(metric=metric, candidate=candidate, comparator=comparator, evaluation_split=split,
                    effect_definition='Comparator error minus candidate error; positive favors candidate.',
                    independent_unit='canonical_person_id', statistical_significance_claim=False,
                    clinical_meaningfulness_claim=False, noninferiority_established=False)
    selected = people[people.method.isin([candidate, comparator])]
    if selected.duplicated(['method', 'canonical_person_id', 'seed']).any():
        raise ValueError('Duplicate person/seed cells would inflate uncertainty sample size')
    grid = selected.pivot(index=['canonical_person_id', 'seed'], columns='method', values=metric)
    if candidate not in grid or comparator not in grid or grid.isna().any().any():
        return metadata | dict(status='insufficient_reference_support', people=int(selected.canonical_person_id.nunique()))
    delta = (grid[comparator] - grid[candidate]).unstack('seed')
    if tuple(sorted(delta.columns)) != SEEDS or delta.isna().any().any():
        return metadata | dict(status='incomplete_person_seed_pairing', people=len(delta))
    if not np.isfinite(delta.to_numpy()).all():
        return metadata | dict(status='nonfinite_paired_measurements', people=len(delta))
    n = len(delta)
    values = delta.mean(axis=1).to_numpy()
    estimate = float(values.mean())
    sd = float(values.std(ddof=1)) if n >= 2 else None
    ci = ([estimate - float(t.ppf(.975, n - 1)) * sd / np.sqrt(n),
           estimate + float(t.ppf(.975, n - 1)) * sd / np.sqrt(n)] if n >= 2 else None)
    crossed = crossed_interval(people, candidate, comparator, metric, draws=draws, seed=seed)
    crossed['status'] = ('descriptive_locked_confirmation_estimate' if split == 'confirmation'
                         else 'descriptive_development_estimate') if n >= 2 else 'insufficient_people'
    crossed['evaluation_split'] = split
    crossed['interpretation'] = ('Paired crossed person/seed bootstrap; only three fitted seeds, so training-variation '
                                 'precision is limited. Secondary intervals are unadjusted.')
    return _json_safe(metadata | dict(status='paired_estimate' if n >= 2 else 'insufficient_people',
        people=n, seeds=len(SEEDS), paired_person_seed_cells=int(delta.size), improvement=estimate,
        paired_person_sd=sd, person_averaged_t_ci95=ci,
        person_interval_interpretation='Student-t interval over person means, conditional on the three fitted seeds.',
        person_effects={str(k): float(v) for k, v in delta.mean(axis=1).items()},
        people_favoring_candidate=int((values > 0).sum()), people_favoring_comparator=int((values < 0).sum()),
        per_seed_improvement={str(k): float(v) for k, v in delta.mean().items()}, crossed_bootstrap=crossed,
        interval_direction=('candidate_favored' if ci is not None and ci[0] > 0 else
                            'comparator_favored' if ci is not None and ci[1] < 0 else 'unresolved'),
        practical_interpretation='No clinical or engineering meaningful-change margin was inferred from this interval.'))


def power_sensitivity(comparison, *, split):
    """Planning scenarios use this actual development contrast's person SD only."""
    if split != 'development':
        return dict(status='not_computed_on_confirmation',
                    reason='Do not use opened confirmation outcomes to redesign or enlarge its sample.')
    sd = comparison.get('paired_person_sd')
    if sd is None or not np.isfinite(sd) or sd <= 0:
        return dict(status='insufficient_or_zero_development_variance')
    cells = []
    for n in (14, 18, 40, 60, 100, 200):
        critical = float(t.ppf(.975, n - 1))
        for effect in (.25, .3731, .5, 1., 2., 5.):
            noncentrality = effect * np.sqrt(n) / sd
            power = float(nct.cdf(-critical, n - 1, noncentrality) + nct.sf(critical, n - 1, noncentrality))
            cells.append(dict(people=n, hypothetical_improvement_deg=effect,
                              power_two_sided_alpha_05=float(np.clip(power, 0, 1)) if np.isfinite(power) else None))
    return dict(status='illustrative_development_sd_scenarios', paired_person_sd=sd,
                variance_people=comparison['people'], candidate=comparison['candidate'], comparator=comparison['comparator'],
                metric=comparison['metric'], scenarios=cells,
                limitations='Normal paired-person model conditional on fitted seeds; development selection, few people, '
                            'new-source variation and changed numbers of windows can make these optimistic. '
                            'This does not establish a clinically meaningful effect or permit outcome-dependent stopping.')


def _confirmation(config, bundle, lock):
    """Verify a new repair declaration; never adapt or relabel legacy test rows."""
    from .repair_cohort import verify_slim_lock
    if lock is None:
        lock = config.get('repair', {}).get('confirmation_lock')
    if not lock or isinstance(lock, dict):
        raise ValueError('Confirmation evaluation requires an immutable lock file path')
    path = Path(lock)
    declared = read_json(path)
    if declared.get('schema') != 'gf-repair-confirmation-lock-v1':
        raise ValueError('Unknown repair confirmation declaration')
    if digest({k: v for k, v in declared.items() if k != 'identity'}) != declared.get('identity'):
        raise ValueError('Repair confirmation lock identity changed')
    verified = verify_slim_lock(config, path)
    if verified.get('identity') != declared['identity']:
        raise ValueError('Verified repair lock differs from requested declaration')
    fits = declared.get('fit_manifest', [])
    fit_keys = [(entry['method'], int(entry['seed'])) for entry in fits]
    if len(fit_keys) != len(EXPECTED_FITS) or set(fit_keys) != EXPECTED_FITS:
        raise ValueError('Confirmation must bind the complete fixed 27-fit evaluation set')
    for fit in fits:
        if declared.get('checkpoint_hashes', {}).get(fit['checkpoint']) != fit['sha256']:
            raise ValueError('Confirmation fit manifest differs from checkpoint hash bindings')
    actual_people = {row['canonical_person_id'] for row in bundle.records}
    actual_families = {row['source_family_id'] for row in bundle.records}
    if actual_people - set(declared['person_ids']) or actual_families - set(declared['source_family_ids']):
        raise ValueError('Confirmation bundle contains identities or windows outside the lock')
    if any(row.get('original_split') != 'test' for row in bundle.records):
        raise PermissionError('Repair confirmation requires unchanged original-test provenance')
    plan = read_json(declared['cohort_plan'])
    planned = {row['source_family_id']: row for row in plan['records']}
    for row in bundle.records:
        expected = planned[row['source_family_id']]
        if any(row.get(key) != expected.get(key) for key in ('canonical_person_id', 'motion_hash', 'start_s', 'end_s')):
            raise ValueError('Confirmation family identity or physical interval differs from its plan')
    provenance = bundle.provenance
    if provenance.get('repair_confirmation_lock_sha256') != sha256(path):
        raise ValueError('Confirmation bundle is not bound to this repair declaration')
    if provenance.get('repair_cohort_identity') != declared['cohort_identity']:
        raise ValueError('Confirmation bundle cohort identity changed')
    if config.get('fixture', False):
        if bundle.evidence_status != 'fixture-tested':
            raise PermissionError('Fixture confirmation cannot claim source evidence')
    elif bundle.evidence_status not in {'technical-source-screen', 'audited-source'}:
        raise PermissionError('Source confirmation requires reviewed source-evidence provenance')
    return declared, path


def _prediction_receipts(config, bundle, split):
    rows = config.get('repair', {}).get('prediction_receipts', [])
    if not rows and config.get('fixture', False):
        return {}
    keys = [(row['method'], int(row['seed'])) for row in rows]
    if len(keys) != len(EXPECTED_FITS) or set(keys) != EXPECTED_FITS:
        raise ValueError('Prediction receipts must bind the complete fixed 27-fit set')
    identity = digest(bundle.records)
    result = {}
    for row, key in zip(rows, keys):
        if row.get('split') != split or row.get('record_identity') != identity:
            raise ValueError('Prediction receipt split or ordered population identity differs')
        if not row.get('checkpoint_sha256'):
            raise ValueError('Prediction receipt lacks its frozen checkpoint identity')
        result[key] = row
    return result


def _verify_prediction(prediction, receipt, shape, declared):
    if tuple(prediction.shape) != tuple(shape):
        raise ValueError('Prediction shape differs from the unchanged evaluation population')
    if receipt is None:
        return
    path = Path(receipt['predictions'])
    if path.suffix != '.npy' or sha256(path) != receipt['predictions_sha256']:
        raise ValueError('Prediction receipt array changed or is not a plain NPY export')
    saved = np.load(path, mmap_mode='r', allow_pickle=False)
    if saved.shape != tuple(shape):
        raise ValueError('Receipt-bound prediction shape differs')
    for start in range(0, len(saved), 64):
        if not np.array_equal(saved[start:start + 64], prediction[start:start + 64], equal_nan=True):
            raise ValueError('Supplied predictions differ from their verified receipt')
    if declared is not None:
        expected = next(row for row in declared['fit_manifest']
                        if row['method'] == receipt['method'] and int(row['seed']) == int(receipt['seed']))
        if receipt['checkpoint_sha256'] != expected['sha256']:
            raise ValueError('Prediction checkpoint differs from frozen confirmation fit')


def _append(output, name, frame):
    path = output / f'{name}.csv'
    frame.to_csv(path, index=False, mode='a', header=not path.exists())


def _means(table, responses, nuisance, interaction):
    active = responses[responses.movement_state.ne('no_change')]
    values = {metric: float(table[metric].mean()) for metric in BASE_METRICS}
    values.update({metric: float(active[metric].mean()) for metric in RESPONSE_METRICS})
    values['nuisance_error'] = float(nuisance.nuisance_error.mean()) if len(nuisance) else np.nan
    values['interaction_error'] = (float(interaction.loc[interaction.movement_state.ne('no_change'),
                                                       'interaction_error'].mean()) if len(interaction) else np.nan)
    return values


def _coverage(table, responses):
    active = responses[responses.movement_state.ne('no_change')]
    return dict(records=len(table), reference_eligible=int(table.reference_eligible.sum()),
                successful=int(table.prediction_success.sum()), response_pairs=len(active),
                response_reference_eligible=int(active.reference_eligible.sum()),
                response_successful=int(active.prediction_success.sum()),
                response_failed=int((active.reference_eligible & ~active.prediction_success).sum()))


def _summarize(bundle, exports, config, output, receipts, declared, split):
    families = {}
    for index, row in enumerate(bundle.records):
        families.setdefault(row['source_family_id'], []).append(index)
    all_people, all_extractor_people, all_coverage = [], [], []
    observed = set()
    for method, seed, prediction in exports:
        _check_deadline(config)
        key = (str(method), int(seed))
        if key not in EXPECTED_FITS or key in observed:
            raise ValueError('Undeclared or duplicate method/seed export')
        observed.add(key)
        _verify_prediction(prediction, receipts.get(key), bundle.inputs['xy'].shape, declared)
        family_rows, extractor_rows, coverage_rows = [], [], []
        for family, indices in families.items():
            _check_deadline(config)
            part = select_rows(bundle, indices)
            table = waveform_failure_decomposition(measurement_rows(part, prediction[indices], config, method, seed))
            responses = response_failure_decomposition(response_metrics(table, config))
            nuisance = nuisance_contrasts(table)
            interaction = interaction_contrasts(responses)
            record = part.records[0]
            common = dict(method=method, seed=seed, canonical_person_id=record['canonical_person_id'],
                          source_family_id=family, motion_hash=record['motion_hash'], split=split)
            for name, frame in (('per-window', table), ('responses', responses),
                                ('nuisance', nuisance), ('interaction', interaction)):
                _append(output, name, frame.assign(evaluation_split=split))
            family_rows.append(common | _means(table, responses, nuisance, interaction))
            for extractor, rows in table.groupby('extractor', sort=True):
                changed = responses[responses.extractor.eq(extractor)]
                nuis = nuisance[nuisance.extractor.eq(extractor)] if len(nuisance) else nuisance
                interact = interaction[interaction.extractor.eq(extractor)] if len(interaction) else interaction
                extractor_rows.append(common | dict(extractor=extractor) | _means(rows, changed, nuis, interact))
                coverage_rows.append(common | dict(extractor=extractor) | _coverage(rows, changed))
        family_frame = pd.DataFrame(family_rows)
        _append(output, 'per-family', family_frame)
        all_people.append(_person_means(family_frame, KEYS, METRICS).assign(split=split))
        all_extractor_people.append(_person_means(pd.DataFrame(extractor_rows), KEYS + ['extractor'], METRICS).assign(split=split))
        all_coverage.append(pd.DataFrame(coverage_rows))
    if observed != EXPECTED_FITS:
        raise ValueError(f'Incomplete repair fit set: expected 27 exports, received {len(observed)}')
    people = pd.concat(all_people, ignore_index=True)
    by_extractor = pd.concat(all_extractor_people, ignore_index=True)
    coverage_family = pd.concat(all_coverage, ignore_index=True)
    counts = ['records', 'reference_eligible', 'successful', 'response_pairs',
              'response_reference_eligible', 'response_successful', 'response_failed']
    coverage_people = coverage_family.groupby(KEYS + ['extractor'], dropna=False)[counts].sum().reset_index().assign(split=split)
    coverage = coverage_people.groupby(['method', 'seed', 'extractor'], dropna=False)[counts].sum().reset_index().assign(split=split)
    for name, frame in (('per-person', people), ('per-person-by-extractor', by_extractor),
                        ('coverage-per-family', coverage_family), ('coverage-per-person', coverage_people), ('coverage', coverage)):
        frame.to_csv(output / f'{name}.csv', index=False)
    return people, by_extractor, coverage


def _comparisons(people, by_extractor, primary_extractor, protocol, split, config):
    options = dict(split=split, draws=protocol['bootstrap_draws'], seed=protocol['bootstrap_seed'])
    primary_people = by_extractor[by_extractor.extractor.eq(primary_extractor)]
    metrics = ['waveform_error', 'response_error', 'synthetic_all_nle', 'response_failure_contribution',
               'response_success_contribution', 'zero_response_error', 'waveform_failure_contribution',
               'waveform_success_contribution']
    def pair(table, candidate, comparator):
        result = {}
        for metric in metrics:
            _check_deadline(config)
            result[metric] = paired_comparison(table, candidate, comparator, metric, **options)
        return result
    primary = pair(primary_people, PRIMARY_CANDIDATE, PRIMARY_COMPARATOR)
    pairs = [('endpoint_dense_vs_scalar_low', REPAIR_METHODS[2], REPAIR_METHODS[3])]
    for variant in VARIANTS:
        dense = f'R-repair-{variant}-dense_change'
        low = f'R-repair-{variant}-scalar_low'
        original = f'F-response-{variant}-graph_time-paired_change'
        base = f'F-response-{variant}-graph_time-base'
        pairs += [(f'{variant}_dense_vs_original_scalar', dense, original),
                  (f'{variant}_scalar_low_vs_original_scalar', low, original),
                  (f'{variant}_dense_vs_base', dense, base),
                  (f'{variant}_original_scalar_vs_base', original, base),
                  (f'{variant}_dense_vs_direct_base', dense, 'P-direct-none-base')]
    return dict(primary_extractor=primary_extractor, primary=primary,
                secondary_primary_extractor={label: pair(primary_people, candidate, comparator)
                                             for label, candidate, comparator in pairs},
                secondary_all_extractors=pair(people, PRIMARY_CANDIDATE, PRIMARY_COMPARATOR),
                secondary_by_extractor={str(extractor): pair(rows, PRIMARY_CANDIDATE, PRIMARY_COMPARATOR)
                                        for extractor, rows in by_extractor.groupby('extractor')},
                multiplicity='One fixed waveform primary. Response is a reported tradeoff; other comparisons and '
                             'intervals are descriptive and unadjusted, with no selection of a best arm.',
                noninferiority='Never inferred from an interval crossing zero; no margin is invented.',
                failure_decomposition='Absolute response error = successful-output contribution + '
                                      '720 degrees times response failure rate; waveform error = waveform successful-output '
                                      'contribution + 180 degrees times waveform failure rate, at identical hierarchical weights.',
                mechanism_boundary='Dense paired angular supervision changes temporal information as well as '
                                   'gradient support. Initial gradient matching does not establish a unique causal '
                                   'effect of quantile-gradient sparsity or match updates throughout training.')


def _report(summary, comparisons):
    primary = comparisons['primary']
    lines = ['# Frozen readout-repair evaluation', '',
             f"Evaluation split: **{summary['evaluation_split']}**. Evidence: **{summary['evidence_status']}**.", '',
             'Software fixtures provide no human evidence.' if summary['fixture'] else
             ('Locked synthetic confirmation; clinical validity is not established.' if summary['independent_confirmation']
              else 'Locked confirmation with incomplete planned support; results describe only the retained population.'
              if summary['independent_confirmation_eligible']
              else 'Synthetic development estimates; these are not independent confirmation.'), '',
             f"Primary extractor: `{summary['primary_extractor']}`. Fixed comparison: `{PRIMARY_CANDIDATE}` "
             f"versus `{PRIMARY_COMPARATOR}`. Waveform error is the only primary endpoint.", '',
             'Positive differences below mean comparator error minus candidate error. Person intervals average '
             'the three fitted seeds within each person; repeated frames, windows and seed fits do not add participants.', '',
             '| Metric | Improvement (deg) | Person 95% t interval | People |',
             '| --- | ---: | --- | ---: |']
    for metric in ('waveform_error', 'response_error'):
        item = primary[metric]
        ci = item.get('person_averaged_t_ci95')
        estimate = item.get('improvement')
        lines.append(f"| {metric} | {'unsupported' if estimate is None else f'{estimate:.4f}'} | "
                     f"{'unsupported' if ci is None else f'[{ci[0]:.4f}, {ci[1]:.4f}]'} | {item.get('people', 0)} |")
    lines += ['', 'Response error remains an explicit tradeoff. An interval crossing zero does not demonstrate '
              'equivalence, preservation or noninferiority. No clinical meaningful-change threshold is assumed.', '',
              'The comparison file also reports each seed, the crossed person/seed bootstrap (limited by three '
              'seeds), original-scalar and low-weight controls, remaining differences from matched base readouts, '
              'and the direct/base practical benchmark. Secondary intervals are unadjusted.', '',
              f"Retained people: {summary['evaluated_people']}; planned people: {summary['planned_people']}. "
              f"Retained source windows: {summary['evaluated_windows']}; planned source windows: {summary['planned_windows']}.", '',
              'Reference-ineligible observations and prediction failures remain in the coverage tables. Failed '
              'eligible response pairs incur 720 degrees; success and failure contributions are reported separately '
              'without dropping failed predictions. Waveform error likewise separates its 180-degree failure '
              'contribution from successful-output error; an improvement may reflect fewer geometric failures, '
              'better successful waveforms, or both.', '',
              'Dense supervision changes temporal information as well as gradient support. A favorable result '
              'supports an effect of this tested supervision change; it does not uniquely establish sparse '
              'quantile gradients as the cause. Matching initial gradient magnitudes does not match optimizer '
              'updates throughout training.', '']
    return '\n'.join(lines)


def evaluate_repair(bundle, exports, config, output, *, split='development', lock=None):
    """Evaluate a fixed split without relabeling; publish only a complete fit set.

    Source exports require ``repair.prediction_receipts`` with method, seed,
    split, predictions, predictions_sha256, record_identity, checkpoint_sha256.
    Each prediction is checked against its receipt-bound NPY array in chunks.
    ``lock`` is a new repair confirmation lock path, never a legacy declaration.
    """
    if split not in {'development', 'confirmation'}:
        raise ValueError('Repair evaluation accepts development or confirmation only')
    _check_deadline(config)
    if not bundle.records or {row['split'] for row in bundle.records} != {split}:
        raise PermissionError('Bundle records must retain exactly the requested evaluation split')
    fixture = bool(config.get('fixture', False))
    if fixture != (bundle.evidence_status == 'fixture-tested'):
        raise PermissionError('Configuration and bundle fixture/source evidence disagree')
    if not fixture and bundle.evidence_status not in {'technical-source-screen', 'automated-source-screen', 'audited-source'}:
        raise PermissionError('Repair evaluation requires recorded source-evidence provenance')
    family_identity = {}
    for row in bundle.records:
        value = (row['canonical_person_id'], row['motion_hash'], row['split'])
        if family_identity.setdefault(row['source_family_id'], value) != value:
            raise ValueError('Source family crosses a person, motion or split')
    protocol = _protocol(config)
    primary_extractor = _extractor_name(bundle.records, protocol['primary_extractor'])
    declared, lock_path = _confirmation(config, bundle, lock) if split == 'confirmation' else (None, None)
    receipts = _prediction_receipts(config, bundle, split)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with locked(output.parent / f'.{output.name}-publication.lock'):
        if output.exists():
            raise FileExistsError('Preserve the first completed repair evaluation; output already exists')
        staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-staging-', dir=output.parent))
        try:
            people, by_extractor, coverage = _summarize(bundle, exports, config, staging, receipts, declared, split)
            comparisons = _comparisons(people, by_extractor, primary_extractor, protocol, split, config)
            actual_people = sorted({row['canonical_person_id'] for row in bundle.records})
            actual_families = sorted({row['source_family_id'] for row in bundle.records})
            planned_people = declared['person_ids'] if declared else actual_people
            planned_families = declared['source_family_ids'] if declared else actual_families
            primary_people = by_extractor[by_extractor.extractor.eq(primary_extractor)
                                         & by_extractor.method.eq(PRIMARY_CANDIDATE)]
            source_people = {}
            for row in bundle.records:
                inferred_source = row['canonical_person_id'].split('::')[0] if '::' in row['canonical_person_id'] else 'unspecified'
                source = str(row.get('source_dataset', inferred_source))
                source_people.setdefault(source, set()).add(row['canonical_person_id'])
            complete_population = (set(planned_people) == set(actual_people) and set(planned_families) == set(actual_families))
            supported_waveform = int(primary_people.loc[primary_people.waveform_error.notna(), 'canonical_person_id'].nunique())
            supported_response = int(primary_people.loc[primary_people.response_error.notna(), 'canonical_person_id'].nunique())
            confirmation_eligible = bool(split == 'confirmation' and not fixture)
            complete_primary_support = supported_waveform == supported_response == len(planned_people)
            summary = dict(schema='gf-repair-evaluation-v1', status='REPAIR_EVALUATION_COMPLETE',
                evaluation_split=split, fixture=bool(config.get('fixture', False)), evidence_status=bundle.evidence_status,
                independent_confirmation=confirmation_eligible and complete_population and complete_primary_support,
                independent_confirmation_eligible=confirmation_eligible,
                confirmation_scope=('complete_planned_population' if confirmation_eligible and complete_population and complete_primary_support
                                    else 'limited_retained_population' if confirmation_eligible else 'not_source_confirmation'),
                clinical_validation=False, primary_candidate=PRIMARY_CANDIDATE, primary_comparator=PRIMARY_COMPARATOR,
                primary_metric='waveform_error', primary_extractor=primary_extractor, methods=len(METHODS), fits=len(EXPECTED_FITS),
                seeds=list(SEEDS), evaluated_people=len(actual_people), planned_people=len(planned_people),
                primary_waveform_supported_people=supported_waveform,
                primary_response_supported_people=supported_response,
                source_people={source: len(ids) for source, ids in source_people.items()},
                evaluated_windows=len(actual_families), planned_windows=len(planned_families),
                missing_people=sorted(set(planned_people) - set(actual_people)),
                missing_source_families=sorted(set(planned_families) - set(actual_families)),
                complete_planned_population=complete_population, complete_primary_person_support=complete_primary_support,
                observation_records=len(bundle.records), independent_unit='canonical_person_id',
                record_identity=digest(bundle.records), lock_sha256=sha256(lock_path) if lock_path else None,
                statistical_protocol=protocol, noninferiority_established=False, best_arm_selection=False,
                scientific_claim='Fixed-comparison synthetic measurement evidence; inspect effect sizes, coverage and intervals.',
                aggregation='Conditions within source windows, windows within raw motions, motions within people; '
                            'paired seeds averaged within people for t intervals.',
                response_failure_penalty_deg=720., waveform_failure_penalty_deg=180.)
            powers = {metric: power_sensitivity(comparisons['primary'][metric], split=split)
                      for metric in ('waveform_error', 'response_error')}
            atomic_json(staging / 'comparisons.json', _json_safe(comparisons))
            atomic_json(staging / 'summary.json', _json_safe(summary))
            atomic_json(staging / 'power-sensitivity.json', _json_safe(powers))
            (staging / 'report.md').write_text(_report(summary, comparisons))
            artifacts = {str(output / path.relative_to(staging)): sha256(path)
                         for path in sorted(staging.rglob('*')) if path.is_file()}
            result = dict(status=summary['status'], summary=str(output / 'summary.json'),
                          report=str(output / 'report.md'), evaluation_split=split,
                          independent_confirmation=summary['independent_confirmation'])
            complete = dict(schema='gf-repair-evaluation-complete-v1', created_utc=utc_now(), result=result,
                            config_identity=digest(config), record_identity=summary['record_identity'],
                            lock_sha256=summary['lock_sha256'], artifacts=artifacts,
                            prediction_inputs=[deepcopy(row) for row in receipts.values()])
            _check_deadline(config)
            atomic_json(staging / 'complete.json', complete)
            os.rename(staging, output)
            return complete | dict(receipt=str(output / 'complete.json'), sha256=sha256(output / 'complete.json'))
        except Exception as error:
            (staging / 'complete.json').unlink(missing_ok=True)
            atomic_json(staging / 'failure.json', dict(status='unpublished', error=str(error), destination=str(output)))
            raise
