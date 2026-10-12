"""Enumerate the preregistered matrix without result-dependent branches."""
from pathlib import Path
from .common import read_json, digest


CORE_RECIPE_IDS = frozenset(
    f'{prefix}-{objective}'
    for prefix in ('M-coordinate-graph_time', 'M-paired_jepa-graph_time',
                   'P-direct-none', 'I-initialized-none', 'I-shuffled_jepa-graph_time')
    for objective in ('base', 'paired_change')
)

RESPONSE_VARIANTS = ('jepa_delta_v1', 'jepa_endpoint_v1', 'coordinate_delta_v1')


def build_response_plan(seeds=(17, 29, 43), *, include_base_readouts=False):
    """A separate, collision-proof extension; historical plan identities do not change."""
    if list(seeds) != [17, 29, 43]:
        raise ValueError('The response follow-up requires the registered seeds 17, 29, 43')
    recipes, phases = [], []
    for variant in RESPONSE_VARIANTS:
        recipe = dict(recipe_id=f'F-response-{variant}-graph_time-paired_change',
                      group='F', encoder='coordinate' if variant.startswith('coordinate') else 'paired_jepa',
                      representation_variant=variant, pretraining_mask='graph_time',
                      mode='frozen_readout', readout_or_training_objective='paired_change', comparator_policy=None)
        recipes.append(recipe)
        for seed in seeds:
            upstream = f'pretrain-response-{variant}-graph_time-seed-{seed}'
            phases.append(dict(phase_id=upstream, phase='pretrain', recipe=recipe,
                               seed=seed, depends_on=['followup-profile']))
            phases.append(dict(phase_id=f"{recipe['recipe_id']}-seed-{seed}", phase='readout',
                               recipe=recipe, seed=seed, depends_on=[upstream]))
    plan = dict(schema='gait-fidelity-response-plan-v1', experiment_set='response_followup',
                recipes=recipes, seeds=list(seeds), phases=phases,
                counts=dict(recipes=3, final_fits=9, pretraining_phases=9, optimization_phases=18))
    if include_base_readouts:
        # One additional readout per frozen encoder; never repeat pretraining.
        for original in list(recipes):
            recipe = dict(original, recipe_id=original['recipe_id'].replace('-paired_change', '-base'),
                          readout_or_training_objective='base')
            recipes.append(recipe)
            for seed in seeds:
                upstream = f"pretrain-response-{recipe['representation_variant']}-graph_time-seed-{seed}"
                phases.append(dict(phase_id=f"{recipe['recipe_id']}-seed-{seed}", phase='readout',
                                   recipe=recipe, seed=seed, depends_on=[upstream]))
        plan.update(schema='gait-fidelity-response-plan-v2', experiment_set='response_followup_readout_control',
                    counts=dict(recipes=6, final_fits=18, pretraining_phases=9, optimization_phases=27))
    plan['identity'] = digest(plan)
    return plan


def build_plan(seeds=(17, 29, 43), experiment_set='full'):
    if experiment_set == 'response_followup':
        return build_response_plan(seeds)
    if experiment_set == 'response_followup_readout_control':
        return build_response_plan(seeds, include_base_readouts=True)
    source = read_json(Path(__file__).with_name('recipes.json'))
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    if experiment_set not in {'core', 'full'}:
        raise ValueError('experiment_set must be core or full')
    recipes = [recipe for recipe in source['recipes']
               if experiment_set == 'full' or recipe['recipe_id'] in CORE_RECIPE_IDS]
    if experiment_set == 'core' and {r['recipe_id'] for r in recipes} != CORE_RECIPE_IDS:
        raise RuntimeError('The registered recipe table is missing a core control')
    phases, upstream = [], {}
    for recipe in recipes:
        if recipe['mode'] == 'frozen_readout' and recipe['encoder'] != 'initialized':
            for seed in seeds:
                key = (recipe['encoder'], recipe['pretraining_mask'], seed)
                if key not in upstream:
                    phase_id = f'pretrain-{key[0]}-{key[1]}-seed-{seed}'
                    upstream[key] = phase_id
                    phases.append(dict(phase_id=phase_id, phase='pretrain', recipe=recipe,
                                       seed=seed, depends_on=['prepare']))
    for recipe in recipes:
        for seed in seeds:
            parent = upstream.get((recipe['encoder'], recipe['pretraining_mask'], seed))
            phases.append(dict(phase_id=f"{recipe['recipe_id']}-seed-{seed}",
                               phase='readout' if recipe['mode'] == 'frozen_readout' else 'end_to_end',
                               recipe=recipe, seed=seed, depends_on=[parent or 'prepare']))
    ids = [p['phase_id'] for p in phases]
    assert len(ids) == len(set(ids))
    plan = dict(schema='gait-fidelity-plan-v1', recipes=recipes, seeds=list(seeds), phases=phases,
                counts=dict(recipes=len(recipes), final_fits=len(recipes)*len(seeds),
                            pretraining_phases=len(upstream), optimization_phases=len(phases)))
    # Preserve the historical full-plan identity; existing frozen runs remain readable.
    if experiment_set != 'full':
        plan['experiment_set'] = experiment_set
    plan['identity'] = digest(plan)
    return plan
