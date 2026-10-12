"""Visible calculations for the separate, registered JEPA response follow-up."""


def build_response_cells(md, code):
    return [md(r'''
    # F · JEPA response coupling

    This tutorial asks whether supervising prediction errors across two movement
    states during pretraining helps a frozen encoder retain useful movement
    information. The existing output network already learns movement changes;
    the new comparison moves an additional constraint into pretraining.

    The cells below calculate the losses on small tensors and compare them with
    production code. These are **software illustrations**, not research results.
    No cell initializes a source run, modifies its artifacts or submits jobs.
    For a source run, start the kernel after sourcing `jepa-response-02/session.env`.
    Without a selected response run, the final cells can read the downloaded
    `outputs/iclr/jepa-response` evidence. This compact packet has summaries and
    diagnostics, not source trajectories or trained checkpoints.

    [Scientific protocol](../../../docs/studies/gait-fidelity/methods/jepa-response.md)
    · [HAIC commands](../../../slurm/gait-fidelity/JEPA_RESPONSE.md)
    '''), code('''
    from pathlib import Path
    import json, os, sys
    ROOT = Path(os.environ.get('GF_ROOT', Path.cwd())).resolve()
    while not (ROOT / 'src/gavd6_sjepa').is_dir() and ROOT != ROOT.parent:
        ROOT = ROOT.parent
    assert (ROOT / 'src/gavd6_sjepa').is_dir(), 'Open from the checkout or release.'
    sys.path.insert(0, str(ROOT / 'src'))
    import numpy as np
    import pandas as pd
    import torch
    from torch.nn import functional as F
    from IPython.display import display, Image
    from gavd6_sjepa.research_directions.synthetic_training_v2.models import ModelConfig
    from gavd6_sjepa.research_directions.gait_fidelity.response_objectives import (
        latent_response_loss, coordinate_response_loss, response_support)
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(17)
    work = Path(os.environ['GF_WORK']).resolve() if os.environ.get('GF_WORK') else None
    cfg = ModelConfig(width=6, heads=2, encoder_layers=1, predictor_layers=1,
                      window_size=8, patch_size=4)
    print('Software illustration: 2 pairs, 8 frames, 12 joints, 6 feature channels.')
    print('Saved run:', work if work else 'none selected')
    '''), md(r'''
    ## 1. Identify the same supported tokens in both arms

    A token contains four frames for one joint. The new auxiliaries require that
    both endpoints query the token and that all four reference frames are valid
    at both endpoints. Reference validity decides loss support; it is never an
    online/student encoder input. The privileged teacher receives reference
    validity during training. Endpoint rows are adjacent: `[a0, b0, a1, b1]`.

    Reduce in two stages: average admitted tokens within each pair, then average
    supported pairs equally. A pair with no auxiliary support still receives its
    original base loss. It is excluded from the auxiliary mean and reported.
    '''), code('''
    endpoints, joints, width = 4, 12, cfg.width
    patches = cfg.window_size // cfg.patch_size
    queries = torch.ones(endpoints, patches * joints, dtype=torch.bool)
    valid = torch.ones(endpoints, cfg.window_size, joints, dtype=torch.bool)
    valid[0, 1, 0] = False  # One invalid frame excludes its whole paired token.
    queries[3, 4] = False  # One unqueried endpoint excludes that paired token.

    complete = valid.reshape(endpoints, patches, cfg.patch_size, joints).all(dim=2)
    manual_support = (queries & complete.flatten(1)).reshape(2, 2, -1).all(dim=1)
    support = response_support(queries, valid, cfg)
    assert torch.equal(support, manual_support)
    assert not support[0, 0] and not support[1, 4]
    print('Admitted tokens per pair:', support.sum(1).tolist())

    def equal_pair_mean(token_values, flags):
        counts = flags.sum(1)
        safe = torch.where(flags, token_values, 0)
        if bool((flags & ~torch.isfinite(token_values)).any()):
            raise FloatingPointError('A supported token has a nonfinite loss.')
        per_pair = safe.sum(1) / counts.clamp_min(1)
        # A graph-connected zero skips only this unsupported auxiliary.
        return per_pair[counts > 0].mean() if bool((counts > 0).any()) else safe.sum()*0
    '''), md(r'''
    ## 2. Derive the coupled and endpoint feature losses

    For predicted features $p$ and detached teacher features $t$, subtract the
    existing teacher center $c$, apply the two temperatures, and remove each
    token's mean across feature channels:

    $$e_i=H[p_i/0.1-\operatorname{stopgrad}((t_i-c)/0.06)].$$

    The paired objective uses $\|e_b-e_a\|^2/(2D)$; the endpoint control uses
    $(\|e_a\|^2+\|e_b\|^2)/(2D)$. With identical support and one shared weight,
    their difference is the negative cross-endpoint inner product $-e_a^Te_b/D$.
    Their reference tensors and model capacity are identical.
    '''), code('''
    predicted = torch.randn(endpoints, patches*joints, width, generator=rng, requires_grad=True)
    teacher = torch.randn(endpoints, patches*joints, width, generator=rng, requires_grad=True)
    center = torch.zeros(width)
    residual = predicted / .1 - (teacher.detach() - center.detach()) / .06
    residual = residual - residual.mean(-1, keepdim=True)
    ea, eb = residual.reshape(2, 2, patches*joints, width).unbind(1)
    delta_tokens = (eb-ea).square().sum(-1) / (2*width)
    endpoint_tokens = (ea.square()+eb.square()).sum(-1) / (2*width)
    coupling_tokens = -(ea*eb).sum(-1) / width

    delta, delta_info = latent_response_loss(predicted, teacher, queries, valid, cfg,
        variant='jepa_delta_v1', center=center)
    endpoint, endpoint_info = latent_response_loss(predicted, teacher, queries, valid, cfg,
        variant='jepa_endpoint_v1', center=center)
    torch.testing.assert_close(delta, equal_pair_mean(delta_tokens, support))
    torch.testing.assert_close(endpoint, equal_pair_mean(endpoint_tokens, support))
    torch.testing.assert_close(delta-endpoint, equal_pair_mean(coupling_tokens, support))
    manual_delta_gradient = torch.autograd.grad(equal_pair_mean(delta_tokens, support), predicted,
                                                retain_graph=True)[0]
    source_delta_gradient = torch.autograd.grad(delta, predicted, retain_graph=True)[0]
    torch.testing.assert_close(manual_delta_gradient, source_delta_gradient)
    delta.backward(retain_graph=True)
    assert teacher.grad is None and torch.isfinite(predicted.grad).all()
    display(pd.DataFrame([{'illustration': 'paired', **delta_info},
                          {'illustration': 'endpoint', **endpoint_info}]))

    # Disjoint query sets produce no paired auxiliary tokens; this does not
    # turn the ordinary per-endpoint pretraining objective into zero loss.
    disjoint_queries = torch.zeros_like(queries)
    disjoint_queries[0::2, ::2] = True
    disjoint_queries[1::2, 1::2] = True
    unsupported_loss, unsupported_info = latent_response_loss(
        predicted, teacher, disjoint_queries, valid, cfg, variant='jepa_delta_v1', center=center)
    explicit_zero = equal_pair_mean(delta_tokens, torch.zeros_like(support))
    torch.testing.assert_close(unsupported_loss, explicit_zero)
    zero_gradient = torch.autograd.grad(unsupported_loss, predicted, retain_graph=True)[0]
    torch.testing.assert_close(zero_gradient, torch.zeros_like(predicted))
    assert unsupported_info['supported_pairs'] == 0 and unsupported_info['loss'] is None
    print('No shared auxiliary queries:', unsupported_info)
    '''), md(r'''
    ## 3. Understand what the paired loss cannot detect

    If both endpoints have the same error, $e_a=e_b\ne0$, the difference loss is
    zero. It can preserve a change while permitting a shared position bias.
    This is why the original anchoring losses remain, and why response error
    must be accompanied by coordinate and nuisance measurements. A small latent
    difference also does not establish a small physical movement difference.
    '''), code('''
    shared_error = torch.tensor([2., -1., -1.])
    shared_delta = (shared_error-shared_error).square().mean()/2
    shared_endpoint = (shared_error.square()+shared_error.square()).mean()/2
    assert shared_delta == 0 and shared_endpoint > 0
    print('Shared nonzero error: delta loss =', shared_delta.item(),
          '; endpoint loss =', shared_endpoint.item())
    '''), md(r'''
    ## 4. Put coordinate residuals in common units

    Each endpoint has its own input-derived normalization scale $s_i$. Subtracting
    normalized residuals directly would mix units. Set $s_{ab}=(s_a+s_b)/2$ and
    $r_i=(s_i/s_{ab})(\hat x_i^{norm}-y_i^{norm})$. Then $r_i$ is pixel error
    divided by the shared pair scale, because the normalization origins cancel.

    The loss is $\|r_b-r_a\|^2/4$ per frame, followed by the four-frame mean,
    the supported-token mean and the supported-pair mean. The divisor four is
    $2\times2$ for the residual difference convention and xy dimensions; it is
    not an additional frame average.
    '''), code('''
    scales = torch.tensor([2., 4., 3., 9.])
    targets = torch.zeros(endpoints, cfg.window_size, joints, 2)
    targets[~valid] = float('nan')  # Invalid placeholders must not poison backward.
    pixel_errors = torch.randn(targets.shape, generator=rng)
    pred_coordinates = (pixel_errors/scales[:, None, None, None]).requires_grad_()
    common_scale = scales.reshape(2, 2).mean(1)
    safe_targets = torch.where(valid[...,None], targets, 0)
    explicit_pixel_residual = (pred_coordinates-safe_targets)*scales[:,None,None,None]
    pixel_a, pixel_b = explicit_pixel_residual.reshape(2, 2, cfg.window_size, joints, 2).unbind(1)
    reference_frame_loss = ((pixel_b-pixel_a)/common_scale[:,None,None,None]).square().sum(-1)/4
    reference_token_loss = reference_frame_loss.reshape(2, patches, cfg.patch_size, joints).mean(2).flatten(1)
    coordinate, coordinate_info = coordinate_response_loss(
        pred_coordinates, targets, queries, valid, scales, cfg)
    explicit_coordinate = equal_pair_mean(reference_token_loss, support)
    torch.testing.assert_close(coordinate, explicit_coordinate)
    expected_gradient = torch.autograd.grad(explicit_coordinate, pred_coordinates, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(coordinate, pred_coordinates, retain_graph=True)[0]
    torch.testing.assert_close(expected_gradient, actual_gradient)
    coordinate.backward()
    assert torch.isfinite(pred_coordinates.grad).all()
    print('Common-scale coordinate loss:', coordinate.item())
    '''), md(r'''
    ## 5. Assemble a complete JEPA pretraining forward pass

    The paired auxiliary augments the same centered feature cross-entropy and
    translated-view regularizer derived in notebook 04:

    $$L_{\rm base}=L_{\rm CE}+0.05L_{\rm VICReg},\qquad
      L_{\rm response}=L_{\rm base}+\lambda L_{\rm auxiliary}.$$

    The auxiliary requires **all** reference frames in a token at both
    endpoints. The base cross-entropy needs **any** valid reference frame at
    each queried endpoint token. Changing the auxiliary support must leave
    the base support unchanged. A pair without auxiliary support can therefore
    still contribute to base pretraining.

    This generated batch is already in normalized coordinates. It uses a fresh
    small model and masks about half of its token slots; notebook 01 explains
    the input-only normalization performed before these operations. We repeat
    the important reductions here so the complete loss can be followed without
    treating `response_forward_losses` as the algorithm.
    '''), code('''
    from gavd6_sjepa.research_directions.synthetic_training_v2.models import RestorationModel
    from gavd6_sjepa.research_directions.gait_fidelity.response_calibration import (
        response_forward_losses, gradient_statistics)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(621)
        response_model = RestorationModel('paired_jepa', cfg).cpu()
    response_model.requires_grad_(False)
    for module in (response_model.encoder, response_model.predictor, response_model.projector):
        module.requires_grad_(True)
    response_model.train()
    demo_target = torch.randn(endpoints, cfg.window_size, joints, 2, generator=rng)
    demo_observed = torch.ones_like(valid)
    demo_observed[1, 0, 11] = False
    demo_inputs = dict(xy=demo_target+.2*torch.randn(demo_target.shape, generator=rng),
        confidence=torch.ones_like(valid, dtype=torch.float32), observed=demo_observed,
        timestamps=torch.arange(cfg.window_size, dtype=torch.float32)[None].expand(endpoints,-1)/25)
    demo_hidden = torch.rand(endpoints, patches, joints, generator=rng) < .5
    demo_queries = demo_hidden | (~demo_observed).reshape(endpoints, patches, cfg.patch_size, joints).any(2)
    base_support = demo_queries.flatten(1) & valid.reshape(endpoints, patches, cfg.patch_size, joints).any(2).flatten(1)
    auxiliary_support = response_support(demo_queries, valid, cfg)

    tokens = response_model.encoder(demo_inputs, demo_hidden)
    predictor_tokens = tokens
    for layer in response_model.predictor[0].layers:
        predictor_tokens = layer(predictor_tokens)
    predictor_norm, predictor_linear = response_model.predictor[1:]
    demo_prediction = F.linear(F.layer_norm(predictor_tokens, predictor_norm.normalized_shape,
        predictor_norm.weight, predictor_norm.bias, predictor_norm.eps),
        predictor_linear.weight, predictor_linear.bias)
    teacher_inputs = dict(xy=torch.where(valid[...,None], demo_target, 0), observed=valid,
        confidence=valid.float(), timestamps=demo_inputs['timestamps'])
    with torch.no_grad():
        demo_teacher = response_model.teacher(teacher_inputs)
    probabilities = ((demo_teacher-response_model.center)/.06).softmax(-1)
    log_predictions = (demo_prediction/.1).log_softmax(-1)
    token_ce = -(probabilities*log_predictions).sum(-1)
    endpoint_counts = base_support.sum(1)
    endpoint_ce = torch.where(base_support, token_ce, 0).sum(1)/endpoint_counts.clamp_min(1)
    supported_endpoints = endpoint_counts > 0
    assert int(supported_endpoints.sum()) == endpoints, 'This teaching batch needs four supported endpoints.'
    explicit_ce = endpoint_ce[supported_endpoints].mean()

    # Replay exactly the same two random translations for the source comparison.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(622)
        views = []
        for _ in range(2):
            translation = (2*torch.rand(endpoints,1,1,2)-1)*.02
            moved = dict(demo_inputs, xy=torch.where(demo_observed[...,None],
                demo_inputs['xy']+translation, demo_inputs['xy']))
            pooled = response_model.encoder(moved, demo_hidden).mean(1)
            projector = response_model.projector
            projected = F.linear(F.gelu(F.linear(pooled, projector[0].weight, projector[0].bias)),
                                 projector[2].weight, projector[2].bias)
            views.append(projected[supported_endpoints])
    u, v = views
    invariance = (u-v).square().mean()
    variance = .5*(torch.relu(1-(u.var(0,unbiased=False)+1e-4).sqrt()).mean()
                   + torch.relu(1-(v.var(0,unbiased=False)+1e-4).sqrt()).mean())
    covariance = u.new_zeros(())
    for view in (u,v):
        centered_view = view-view.mean(0)
        covariance_matrix = centered_view.T@centered_view/(len(view)-1)
        off_diagonal = covariance_matrix-torch.diag(torch.diagonal(covariance_matrix))
        covariance = covariance+off_diagonal.square().sum()/(2*view.shape[1])
    explicit_regularizer = 25*invariance+25*variance+covariance
    explicit_base = explicit_ce+.05*explicit_regularizer
    demo_error = demo_prediction/.1-(demo_teacher.detach()-response_model.center.detach())/.06
    demo_error = demo_error-demo_error.mean(-1,keepdim=True)
    a, b = demo_error.reshape(2,2,patches*joints,width).unbind(1)
    explicit_delta = equal_pair_mean((b-a).square().mean(-1)/2, auxiliary_support)
    explicit_endpoint = equal_pair_mean((a.square()+b.square()).mean(-1)/2, auxiliary_support)

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(622)
        source_forward = response_forward_losses(response_model,
            dict(inputs=demo_inputs, target=demo_target, valid=valid, queries=demo_queries,
                 hidden=demo_hidden, scale=np.ones(endpoints)), {}, 'jepa_delta_v1')
    torch.testing.assert_close(explicit_base, source_forward['base_loss'])
    torch.testing.assert_close(explicit_delta, source_forward['auxiliary_loss'])
    assert torch.equal(base_support, source_forward['query_valid'])
    display(pd.DataFrame([dict(base_cross_entropy=float(explicit_ce.detach()),
        regularizer=float(explicit_regularizer.detach()), base_total=float(explicit_base.detach()),
        paired_auxiliary=float(explicit_delta.detach()), endpoint_auxiliary=float(explicit_endpoint.detach()),
        base_supported_tokens=int(base_support.sum()), auxiliary_supported_tokens=int(auxiliary_support.sum()))]))
    '''), md(r'''
    ## 6. Calibrate a weight without selecting on development results

    The implementation measures gradients on 32 fixed training batches before
    optimization. For JEPA, $\lambda=0.1\sqrt{G_{base}/\max(G_\Delta,G_E)}$;
    both variants use that value across every seed. Coordinate calibration uses
    its own units. The receipt binds this calculation to the source data and
    protocol; final training initializes afresh. Here $G$ is the sum of squared
    gradient norms over the 32 training batches, with unused parameter gradients
    counted as zero. Equivalently, the gradient RMS is
    $\sqrt{G/(32N_{\rm parameters})}$. The parameter count cancels within a
    comparison because its losses use the same trainable parameter set.

    One shared JEPA coefficient makes the **larger** initial auxiliary gradient
    RMS equal to 10% of the base gradient RMS. It does not make both auxiliaries
    individually equal to 10%; their initial gradient magnitudes can differ
    substantially. The following single-batch calculation illustrates the
    reduction and checks it against the implementation. It is not the registered
    32-batch source calibration.

    '''), code('''
    response_parameters = [parameter for parameter in response_model.parameters() if parameter.requires_grad]
    def explicit_gradient_statistics(loss, parameters):
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        energy = sum(float(g.detach().float().square().sum()) for g in gradients if g is not None)
        number = sum(parameter.numel() for parameter in parameters)
        return dict(squared_l2=energy, parameters=number, rms=(energy/number)**.5)
    energies = {}
    for name, term in [('base',explicit_base), ('delta',explicit_delta), ('endpoint',explicit_endpoint)]:
        explicit_stats = explicit_gradient_statistics(term, response_parameters)
        source_stats = gradient_statistics(term, response_parameters)
        np.testing.assert_allclose(explicit_stats['squared_l2'], source_stats['squared_l2'], rtol=2e-6)
        assert explicit_stats['parameters'] == source_stats['parameters']
        energies[name] = explicit_stats
    teaching_weight = .1*(energies['base']['squared_l2']/max(
        energies['delta']['squared_l2'], energies['endpoint']['squared_l2']))**.5
    print('Single generated-batch coefficient:', teaching_weight)
    display(pd.DataFrame(energies).T)
    explicit_total = explicit_base+teaching_weight*explicit_delta
    source_total = source_forward['base_loss']+teaching_weight*source_forward['auxiliary_loss']
    manual_gradients = torch.autograd.grad(explicit_total, response_parameters, retain_graph=True, allow_unused=True)
    source_gradients = torch.autograd.grad(source_total, response_parameters, retain_graph=True, allow_unused=True)
    for parameter, manual, source in zip(response_parameters, manual_gradients, source_gradients):
        manual = torch.zeros_like(parameter) if manual is None else manual
        source = torch.zeros_like(parameter) if source is None else source
        torch.testing.assert_close(manual, source, atol=2e-5, rtol=2e-4)
    print('Complete loss and every trainable parameter gradient match the production forward pass.')
    '''), md(r'''

    The coordinate head starts with zero final weights. Its first backward pass
    therefore gives zero encoder gradients even though the output weights can
    learn. The following small network demonstrates why calibration measures
    **all trainable parameters**, rather than just the encoder.
    '''), code('''
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(623)
        encoder = torch.nn.Linear(4, 6)
        head = torch.nn.Linear(6, 2)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    inputs = torch.randn(5, 4, generator=rng)
    labels = torch.ones(5, 2)
    loss = (head(encoder(inputs))-labels).square().mean()
    loss.backward()
    encoder_norm2 = sum(p.grad.square().sum().item() for p in encoder.parameters())
    total_norm2 = encoder_norm2 + sum(p.grad.square().sum().item() for p in head.parameters())
    assert encoder_norm2 == 0 and total_norm2 > 0
    print('Encoder squared gradient:', encoder_norm2, '; all parameters:', total_norm2)
    print('Calibration defines an initial scale. Later gradients and clipping are logged.')
    '''), md(r'''
    ## 7. Apply the combined gradient before updating the teacher

    The optimizer sees the sum of base and weighted auxiliary gradients.
    Gradient clipping is applied once to that sum, before AdamW updates the
    student. The teacher moving average and center update follow the optimizer
    step, using the pre-update teacher tokens for the center. This order is
    unchanged by the response extension. Clipping can change how an initial
    calibration relates to subsequent updates, so its retained diagnostics
    remain relevant when interpreting the results.
    '''), code('''
    optimizer = torch.optim.AdamW(response_parameters, lr=3e-4/100, weight_decay=.01)
    # First warmup step of the source's 2,000-update phase: 5% = 100 steps.
    old_teacher = [parameter.detach().clone() for parameter in response_model.teacher.parameters()]
    old_center = response_model.center.detach().clone()
    old_student = [parameter.detach().clone() for parameter in response_model.encoder.parameters()]
    optimizer.zero_grad(set_to_none=True)
    explicit_total.backward()
    assert all(parameter.grad is None for parameter in response_model.teacher.parameters())
    assert all(parameter.grad is None for parameter in response_model.readout.parameters())
    norm = torch.nn.utils.clip_grad_norm_(response_parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(response_model.teacher.parameters(), response_model.encoder.parameters()):
            teacher_parameter.mul_(.99).add_(student_parameter, alpha=.01)
        for teacher_buffer, student_buffer in zip(response_model.teacher.buffers(), response_model.encoder.buffers()):
            teacher_buffer.copy_(student_buffer)
        endpoint_teacher = (demo_teacher*base_support[...,None]).sum(1)/endpoint_counts[:,None].clamp_min(1)
        response_model.center.mul_(.9).add_(endpoint_teacher[supported_endpoints].mean(0), alpha=.1)
    for old, current, online in zip(old_teacher, response_model.teacher.parameters(), response_model.encoder.parameters()):
        torch.testing.assert_close(current, .99*old+.01*online)
    torch.testing.assert_close(response_model.center, .9*old_center+.1*endpoint_teacher[supported_endpoints].mean(0))
    assert any(not torch.equal(old,current) for old,current in zip(old_student,response_model.encoder.parameters()))
    print({'generated_batch_update': 1, 'gradient_norm_before_clipping': float(norm),
           'teacher_gradient': None, 'retained_study_checkpoints_modified': False})
    '''), md(r'''
    ## 8. Inspect the saved experiment and outcomes

    The source matrix has three variants and three seeds. The completed response-02
    run shares each pretraining fit across two fresh frozen readouts (base and
    paired-change): nine pretraining fits and eighteen final readouts. Inspect the
    saved plan for an earlier child run's counts. Parent predictions provide the original
    comparisons. The primary contrast is delta versus endpoint JEPA on
    person-balanced response error. Windows from one person are correlated;
    three seeds are not three independent datasets.

    The next cells read a selected child run, or the local downloaded packet
    when no response child was selected. They neither launch fits nor regenerate
    evaluation. An incomplete selected child remains visibly incomplete; it is
    never replaced by another run's completed scores.
    For the downloaded response-02 evidence and the subsequent readout repair,
    use [notebook 07](../07_completed_study_walkthrough.ipynb).
    '''), code('''
    is_child, result_work, evidence_kind = False, None, None
    if work is not None and (work/'config.json').is_file():
        saved_config = json.loads((work/'config.json').read_text())
        is_child = saved_config.get('study_kind') == 'jepa_response_followup'
        if is_child:
            result_work, evidence_kind = work, 'selected source/fixture run'
    compact_work = Path(os.environ.get('GF_EVIDENCE_ROOT', ROOT/'outputs/iclr')).expanduser().resolve()/'jepa-response'
    if not is_child and (compact_work/'config.json').is_file():
        saved_config = json.loads((compact_work/'config.json').read_text())
        if saved_config.get('study_kind') != 'jepa_response_followup' or saved_config.get('fixture', True):
            raise ValueError('The compact packet must identify non-fixture response evidence.')
        result_work, evidence_kind, is_child = compact_work, 'downloaded compact evidence', True
    if is_child:
        print('Evidence:', evidence_kind, '| path:', result_work, '| fixture:', saved_config.get('fixture'))
        if saved_config.get('fixture'):
            print('Software fixture only: these scores are not scientific evidence.')
        plan = json.loads((result_work/'plan.json').read_text())
        display(pd.DataFrame(plan['recipes']))
        print('Saved counts:', plan['counts'])
        print('Saved seeds:', plan['seeds'])
        for name in ('summary.json', 'comparisons.json'):
            path = result_work/'evaluation'/name
            if path.is_file():
                document = json.loads(path.read_text())
                if name == 'summary.json':
                    print('Summary fields:', sorted(document))
                else:
                    # Response comparisons are keyed directly by metric in the
                    # canonical packet; accept a named wrapper only if present.
                    primary = document.get('primary', document)
                    if 'response_error' not in primary:
                        raise ValueError('Response comparisons must contain the declared response_error contrast.')
                    rows = [dict(metric=metric, improvement_deg=value['improvement'],
                        crossed_person_seed_ci95=value['crossed_person_seed_ci95'],
                        people=value['people'], seeds=value['seeds'])
                        for metric,value in primary.items()]
                    display(pd.DataFrame(rows))
                    print('Positive favors delta JEPA; these reused development results are descriptive.')
            else:
                print('Pending:', name)
    else:
        print('No response child selected. Mathematical examples completed; no scientific results implied.')
    '''), code('''
    if is_child:
        for name in ('per-person.csv', 'response-by-condition-person.csv',
                     'response-curves-person.csv', 'coverage.csv'):
            path = result_work/'evaluation'/name
            if path.is_file():
                print(name)
                display(pd.read_csv(path, nrows=8))
            else:
                print('Not available in this run/packet:', name)
        for name in ('response-diagnostics.png', 'response-curves.png'):
            figure = result_work/'evaluation'/name
            if figure.is_file():
                display(Image(filename=str(figure)))
            else:
                print('Optional figure not included:', name)
        calibration_path = result_work/'diagnostics/loss-calibration.json'
        if calibration_path.is_file():
            calibration = json.loads(calibration_path.read_text())
            gradients = calibration['gradients']
            totals = {name: value['sum_squared_l2'] for name,value in gradients.items()}
            inferred_jepa = .1*(totals['jepa_base']/max(totals['jepa_delta_v1'],totals['jepa_endpoint_v1']))**.5
            inferred_coordinate = .1*(totals['coordinate_base']/totals['coordinate_delta_v1'])**.5
            np.testing.assert_allclose(inferred_jepa, calibration['coefficients']['jepa_delta_v1'])
            np.testing.assert_allclose(inferred_jepa, calibration['coefficients']['jepa_endpoint_v1'])
            np.testing.assert_allclose(inferred_coordinate, calibration['coefficients']['coordinate_delta_v1'])
            rows = []
            for variant, base in [('jepa_delta_v1','jepa_base'),('jepa_endpoint_v1','jepa_base'),
                                  ('coordinate_delta_v1','coordinate_base')]:
                coefficient = calibration['coefficients'][variant]
                rows.append(dict(variant=variant, coefficient=coefficient,
                    initial_weighted_auxiliary_over_base_RMS=coefficient*gradients[variant]['rms']/gradients[base]['rms']))
            print('Retained calibration:', len(calibration['batches']), 'training batches; no optimization')
            display(pd.DataFrame(rows))
    '''), md(r'''
    ## 9. Interpret the mechanism with its controls

    A reliable improvement of delta JEPA over the matched endpoint regression
    control would support the value of residual coupling under this observation
    procedure. Similar estimates with broad intervals leave that comparison
    unresolved; they do not demonstrate equivalence. The coordinate-difference
    arm asks whether any gain also occurs with coordinate supervision. A lower
    latent loss without improved restored response does not establish useful
    representation transfer.

    Training-only person-separated ridge probes compare encoder, predictor and
    teacher features. They preserve joint/time positions, fit standardization on
    training folds only and use a fixed 0.01 penalty. These probe folds hold
    people out of the ridge fit; their encoders have already seen all of these
    training people during pretraining. They do not constitute held-out-person
    restoration evaluation. Independently masking identical
    baseline observations changes both masked context and input-derived normalization
    in the student branches. The clean teacher reference stays fixed, so its changes
    isolate normalization effects. Probe failure does not prove information is absent, and probe
    success does not replace the restoration evaluation.

    Read direction accuracy only on reference-resolvable changes, retaining
    missing predictions as failures. Position and nuisance errors expose shared
    bias or other tradeoffs. These development results cannot establish clinical
    validity or a unique benefit of anatomically correct pairing, because this
    follow-up has no latent re-pairing arm or independent clinical references.
    ''')]
