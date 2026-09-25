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
    For saved results, start the kernel after sourcing `jepa-response-01/session.env`.

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
        per_pair = torch.where(flags, token_values, 0).sum(1) / counts.clamp_min(1)
        return per_pair[counts > 0].mean()
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
    residual = predicted / .1 - (teacher.detach() - center) / .06
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
    delta.backward()
    assert teacher.grad is None and torch.isfinite(predicted.grad).all()
    display(pd.DataFrame([{'illustration': 'paired', **delta_info},
                          {'illustration': 'endpoint', **endpoint_info}]))
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
    pixel_errors = torch.randn(targets.shape, generator=rng)
    pred_coordinates = (pixel_errors/scales[:, None, None, None]).requires_grad_()
    common_scale = scales.reshape(2, 2).mean(1)
    pixel_a, pixel_b = pixel_errors.reshape(2, 2, cfg.window_size, joints, 2).unbind(1)
    reference_frame_loss = ((pixel_b-pixel_a)/common_scale[:,None,None,None]).square().sum(-1)/4
    reference_token_loss = reference_frame_loss.reshape(2, patches, cfg.patch_size, joints).mean(2).flatten(1)
    coordinate, coordinate_info = coordinate_response_loss(
        pred_coordinates, targets, queries, valid, scales, cfg)
    torch.testing.assert_close(coordinate, equal_pair_mean(reference_token_loss, support))
    coordinate.backward()
    assert torch.isfinite(pred_coordinates.grad).all()
    print('Common-scale coordinate loss:', coordinate.item())
    '''), md(r'''
    ## 5. Calibrate a weight without selecting on development results

    The implementation measures gradients on 32 fixed training batches before
    optimization. For JEPA, $\lambda=0.1\sqrt{G_{base}/\max(G_\Delta,G_E)}$;
    both variants use that value across every seed. Coordinate calibration uses
    its own units. The receipt binds this calculation to the source data and
    protocol; final training initializes afresh.

    The coordinate head starts with zero final weights. Its first backward pass
    therefore gives zero encoder gradients even though the output weights can
    learn. The following small network demonstrates why calibration measures
    **all trainable parameters**, rather than just the encoder.
    '''), code('''
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
    ## 6. Inspect the saved experiment and outcomes

    The source matrix has three variants and three seeds, each with a pretraining
    phase and a fresh frozen readout. Parent predictions provide the original
    comparisons. The primary contrast is delta versus endpoint JEPA on
    person-balanced response error. Windows from one person are correlated;
    three seeds are not three independent datasets.

    The next cells read a child run if one is selected. They neither launch fits
    nor regenerate evaluation. An incomplete run remains visibly incomplete.
    '''), code('''
    is_child = False
    if work is not None and (work/'config.json').is_file():
        saved_config = json.loads((work/'config.json').read_text())
        is_child = saved_config.get('study_kind') == 'jepa_response_followup'
    if is_child:
        plan = json.loads((work/'plan.json').read_text())
        display(pd.DataFrame(plan['recipes']))
        print('Saved counts:', plan['counts'])
        print('Saved seeds:', plan['seeds'])
        for name in ('response-summary.json', 'response-comparisons.json'):
            path = work/'evaluation'/name
            if path.is_file():
                display(json.loads(path.read_text()))
            else:
                print('Pending:', name)
    else:
        print('No response child selected. Mathematical examples completed; no scientific results implied.')
    '''), code('''
    if is_child:
        for name in ('per-person.csv', 'response-by-condition.csv',
                     'response-no-change-person.csv', 'coverage.csv'):
            path = work/'evaluation'/name
            if path.is_file():
                print(name)
                display(pd.read_csv(path).head(24))
            else:
                print('Pending:', name)
        for name in ('response-diagnostics.png', 'response-curves.png'):
            figure = work/'evaluation'/name
            if figure.is_file():
                display(Image(filename=str(figure)))
            else:
                print('Pending:', name)
    '''), md(r'''
    ## 7. Interpret the mechanism with its controls

    If delta JEPA improves over endpoint regression, residual coupling has helped
    under the registered observation procedure. If the endpoint control matches
    it, additional continuous supervision is sufficient to explain the gain.
    If coordinate-difference pretraining matches it, the remedy need not be
    specific to latent prediction. A lower latent loss without improved restored
    response does not establish useful representation transfer.

    Training-only person-separated ridge probes compare encoder, predictor and
    teacher features. They preserve joint/time positions, fit standardization on
    training folds only and use a fixed 0.01 penalty. Independently masking identical
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
