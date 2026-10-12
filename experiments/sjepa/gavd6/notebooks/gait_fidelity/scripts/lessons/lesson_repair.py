"""Visible scalar/dense readout repair, emitted into standalone tutorial G."""


def build_repair_cells(md, code):
    return [md(r'''
    # G · Derive and check the readout repair

    The completed response experiment produced delta-JEPA and endpoint-JEPA
    encoders. The repair reuses those encoders and fits new coordinate readouts.
    It tests whether supervision distributed over the angle waveform helps beyond
    reducing the strength of the original whole-window scalar loss. This is
    separate from experiment E's **re-pairing** control, which changes which
    motion endpoints are paired.

    Read 01–06 and F first. This notebook exposes the repair's angle geometry,
    percentile gradients, common reference support, three loss terms, calibration,
    and frozen-readout optimizer step. Equations and code are followed by value
    **and gradient** checks against the production implementation. All training
    examples use constructed CPU arrays and a fresh random encoder; no saved
    model is trained or modified. Optional final cells read completed receipts.
    '''), code(r'''
    from pathlib import Path
    from copy import deepcopy
    from types import SimpleNamespace
    import json, math, os, sys
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from IPython.display import display, Image
    from io import BytesIO
    import matplotlib.pyplot as plt

    ROOT = Path(os.environ.get('GF_ROOT', Path.cwd())).resolve()
    while not (ROOT / 'src/gavd6_sjepa').is_dir() and ROOT != ROOT.parent:
        ROOT = ROOT.parent
    assert (ROOT / 'src/gavd6_sjepa').is_dir()
    sys.path.insert(0, str(ROOT / 'src'))
    from gavd6_sjepa.research_directions.synthetic_training_v2.models import ModelConfig, RestorationModel, CoordinateReadout
    from gavd6_sjepa.research_directions.gait_fidelity.training import normalize_batch
    from gavd6_sjepa.research_directions.gait_fidelity.measurements import torch_knee_angles
    from gavd6_sjepa.research_directions.gait_fidelity.repair_objectives import repair_measurement_terms, angle_gradient_support
    from gavd6_sjepa.research_directions.gait_fidelity.repair_training import _forward as source_forward, _gradient_audit
    torch.set_num_threads(1)
    random_state_before = torch.random.get_rng_state().clone()
    settings = dict(min_segment_px=2., min_frames=16, min_fraction=.8)
    Q, T, J = 3, 32, 12
    cfg = ModelConfig(width=16, heads=4, encoder_layers=1, predictor_layers=1, window_size=T, patch_size=4)
    print('ILLUSTRATION:', Q, 'pairs; endpoints [a0,b0,a1,b1,a2,b2];', T, 'frames.')
    print('Source runs use 128 frames, width 96, 4 encoder and 2 predictor layers.')
    '''), md(r'''
    ## 1 · Construct paired coordinates with unequal reference support

    Each endpoint is a full motion window: tensors have shape
    `[pair, endpoint, time, joint, xy]`. The two endpoint rows are adjacent when
    flattened for the model. The example moves the ankles along arcs around the
    knees. It introduces an observation bias, independently missing observations,
    and missing reference coordinates. These are separate masks: observed-input
    availability belongs to the encoder; reference validity belongs to the loss.

    Reference support is deliberately unequal. The third pair will be ineligible
    for angular supervision, although its valid coordinates still contribute to
    the coordinate objective. Predictions cannot remove troublesome frames from
    the angular loss by making the predicted limbs too short.
    '''), code(r'''
    template = np.array([[-20,-70],[20,-70],[-35,-45],[35,-45],[-45,-20],[45,-20],
                         [-15,-35],[15,-35],[-15,0],[15,0],[-15,45],[15,45]], dtype=np.float32)
    truth = np.broadcast_to(template, (Q,2,T,J,2)).copy()
    phase = np.arange(T, dtype=np.float32)*.31
    for pair in range(Q):
        for endpoint in range(2):
            for leg,(knee,ankle) in enumerate([(8,10),(9,11)]):
                bend = (30 + (9+endpoint*(4+pair)*(leg==1))*np.sin(phase + leg*.7 + pair*.19))*np.pi/180
                truth[pair,endpoint,:,ankle] = truth[pair,endpoint,:,knee] + 45*np.stack([np.sin(bend),np.cos(bend)],-1)
    raw_xy = truth.copy()
    raw_xy[...,10,0] += 3*np.sin(phase)[None,None]
    raw_xy[...,11,0] += 5*np.cos(phase*.8)[None,None] + 2
    observed = np.ones((Q,2,T,J),bool)
    observed[0,1,4:8,11] = False
    raw_xy[~observed] = np.nan
    reference_valid = np.ones((Q,2,T,J),bool)
    reference_valid[1,1,:4,6] = False
    reference_valid[2,0,:10,7] = False
    truth[~reference_valid] = np.nan
    raw = dict(xy=raw_xy.reshape(2*Q,T,J,2), observed=observed.reshape(2*Q,T,J),
               confidence=observed.reshape(2*Q,T,J).astype(np.float32),
               timestamps=np.broadcast_to(np.arange(T,dtype=np.float32)/25,(2*Q,T)).copy())
    targets = torch.as_tensor(truth)
    valid = torch.as_tensor(reference_valid)
    # Finite candidate outputs even where the observed coordinate was absent.
    candidate = torch.as_tensor(np.nan_to_num(raw_xy)).clone().requires_grad_(True)
    assert candidate.shape == (Q,2,T,12,2) and valid.dtype == torch.bool
    '''), md(r'''
    ## 2 · Translate coordinates into differentiable knee angles

    For hip–knee vector $u$ and ankle–knee vector $v$, the projected angle is
    $\theta=\operatorname{atan2}(|u_xv_y-u_yv_x|,u^\top v)180/\pi$.
    The absolute cross product restricts the angle to $[0,180]$ degrees.
    `atan2(0,0)` has an undefined derivative; the implementation supplies a small
    positive dot product at that degeneracy and retains a separate length penalty.
    At an exactly straight knee the absolute cross product has a zero subgradient.
    Coordinate supervision remains part of the objective.
    '''), code(r'''
    def explicit_angles(xy):
        xy = xy.float()
        angles, lengths = [], []
        for hip,knee,ankle in [(6,8,10),(7,9,11)]:
            u, v = xy[...,hip,:]-xy[...,knee,:], xy[...,ankle,:]-xy[...,knee,:]
            cross = u[...,0]*v[...,1] - u[...,1]*v[...,0]
            dot = (u*v).sum(-1)
            guarded_dot = torch.where(cross.abs()+dot.abs()>1e-12, dot, torch.full_like(dot,1e-12))
            angles.append(torch.atan2(cross.abs(),guarded_dot)*(180/math.pi))
            lengths.append(torch.stack([(u.square().sum(-1)+1e-12).sqrt(),
                                        (v.square().sum(-1)+1e-12).sqrt()],-1))
        return torch.stack(angles,-1), torch.stack(lengths,-2)

    theta, segment_lengths = explicit_angles(candidate)
    source_theta, source_lengths = torch_knee_angles(candidate)
    torch.testing.assert_close(theta,source_theta)
    torch.testing.assert_close(segment_lengths,source_lengths)
    assert theta.shape == (Q,2,T,2) and segment_lengths.shape == (Q,2,T,2,2)
    '''), md(r'''
    ## 3 · Compute support and the percentile measurement before reducing losses

    A time index is admitted only when both legs have valid reference hip, knee
    and ankle coordinates and sufficiently long reference segments at **both**
    movement endpoints. The pair needs at least 16 admitted frames and 80% of its
    window. This uses the latest source thresholds. The teaching window is shorter.

    A linear percentile sorts the admitted angles. For percentile $q$, let
    $r=(n-1)q$, $i=\lfloor r\rfloor$ and $\alpha=r-i$; the result is
    $(1-\alpha)x_{(i)}+\alpha x_{(\lceil r\rceil)}$. Only the neighboring
    order statistics receive its derivative. Excursion is $P_{95}-P_5$, and
    $A=\text{right excursion}-\text{left excursion}$.
    '''), code(r'''
    def explicit_percentiles(angles):
        # Input [endpoint, admitted time, leg]; output [endpoint, quantile, leg].
        ordered = angles.sort(dim=1).values
        position = angles.new_tensor([.05,.95])*(angles.shape[1]-1)
        lower, upper = position.floor().long(), position.ceil().long()
        alpha = (position-lower)[None,:,None]
        # lerp computes (1-alpha)*lower + alpha*upper with PyTorch's stable arithmetic.
        return torch.lerp(ordered[:,lower,:],ordered[:,upper,:],alpha)

    def explicit_repair(prediction, reference, flags):
        assert torch.isfinite(prediction).all(), 'Nonfinite predictions cannot remove support.'
        assert not (flags & ~torch.isfinite(reference).all(-1)).any()
        safe_reference = torch.where(flags[...,None],reference,0)
        angles, lengths = explicit_angles(prediction)
        reference_angles, reference_lengths = explicit_angles(safe_reference)
        reference_angles = reference_angles.detach()
        fixed = (flags[...,[6,8,10,7,9,11]].all(-1)
                 & (reference_lengths >= settings['min_segment_px']).all(-1).all(-1)).all(1)
        counts = fixed.sum(1)
        eligible = (counts >= settings['min_frames']) & (counts >= settings['min_fraction']*prediction.shape[2])
        scalar_per_pair, dense_per_pair, geometry_per_pair = [], [], []
        for i in range(len(prediction)):
            if not eligible[i]: continue
            p, y = angles[i][:,fixed[i]], reference_angles[i][:,fixed[i]]
            pq, yq = explicit_percentiles(p), explicit_percentiles(y)
            pe, ye = pq[:,1]-pq[:,0], yq[:,1]-yq[:,0]
            pa, ya = pe[:,1]-pe[:,0], ye[:,1]-ye[:,0]
            scalar_per_pair.append((((pa[1]-pa[0])-(ya[1]-ya[0]))/180).square())
            dense_per_pair.append((((p[1]-p[0])-(y[1]-y[0]))/180).square().mean())
            # Average endpoint, admitted frame, leg and segment dimensions within this pair.
            relative_shortfall = torch.relu((settings['min_segment_px']-lengths[i][:,fixed[i]])/settings['min_segment_px'])
            geometry_per_pair.append(relative_shortfall.square().mean())
        connected_zero = prediction.sum()*0
        average = lambda values: torch.stack(values).mean() if values else connected_zero
        return dict(scalar=average(scalar_per_pair),dense=average(dense_per_pair),
                    geometry=average(geometry_per_pair),angles=angles,frame_support=fixed,eligible=eligible)

    manual = explicit_repair(candidate,targets,valid)
    production = repair_measurement_terms(candidate,targets,valid,**settings)
    assert torch.equal(manual['frame_support'],production['frame_support'])
    assert torch.equal(manual['eligible'],torch.tensor([True,True,False]))
    for term in ['scalar','dense','geometry']:
        torch.testing.assert_close(manual[term],production[term],atol=2e-7,rtol=2e-5)
        a = torch.autograd.grad(manual[term],candidate,retain_graph=True)[0]
        b = torch.autograd.grad(production[term],candidate,retain_graph=True)[0]
        torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-5)
        assert torch.isfinite(a).all()
    print('Common reference frames:',manual['frame_support'].sum(1).tolist())
    print('Matched source loss values and coordinate gradients:',{k:float(manual[k].detach()) for k in ['scalar','dense','geometry']})
    '''), md(r'''
    ## 4 · Separate scalar, dense and geometric supervision

    The scalar term averages squared response errors across eligible pairs:
    $L_s=\operatorname{mean}_i\{[(\Delta\widehat A_i-\Delta A_i)/180]^2\}$.
    The dense term first averages
    $[(\Delta\widehat\theta_{i,t,\ell}-\Delta\theta_{i,t,\ell})/180]^2$
    over admitted frames and legs **within each pair**, then averages pairs equally.
    Here $\Delta$ means state $b$ minus state $a$ at a matching time; it is not
    a time derivative. Unequal frame counts do not give one pair more weight.

    Geometry contributes the squared relative shortfall of predicted limb
    segments below 2 pixels on the same support. The original scalar readout and
    both new repair arms include this penalty. The coordinate-only base does not.
    Neither reference eligibility nor the geometry term uses a predicted angle to
    decide which frames may be scored.
    '''), code(r'''
    # A collapsed prediction still receives the original reference support and a penalty.
    collapsed = candidate.detach().clone()
    collapsed[0,0,:,10] = collapsed[0,0,:,8]
    collapsed.requires_grad_()
    collapse_terms = explicit_repair(collapsed,targets,valid)
    collapse_source = repair_measurement_terms(collapsed,targets,valid,**settings)
    assert torch.equal(collapse_terms['frame_support'],manual['frame_support'])
    assert collapse_terms['geometry'] > 0
    torch.testing.assert_close(collapse_terms['geometry'],collapse_source['geometry'])
    assert torch.isfinite(torch.autograd.grad(sum(collapse_terms[k] for k in ['scalar','dense','geometry']),collapsed)[0]).all()
    # An unsupported angular batch has a graph-connected zero, not NaN or a success.
    unsupported = explicit_repair(candidate,torch.full_like(targets,float('nan')),torch.zeros_like(valid))
    unsupported_source = repair_measurement_terms(candidate,torch.full_like(targets,float('nan')),torch.zeros_like(valid),**settings)
    for term in ['scalar','dense','geometry']:
        torch.testing.assert_close(unsupported[term],unsupported_source[term])
        assert not torch.autograd.grad(unsupported[term],candidate,retain_graph=True)[0].any()
    assert not unsupported['eligible'].any()
    print('The full source trainer rejects a batch with no eligible angular pair; this checks its loss primitive only.')

    active = (manual['frame_support'][:,None,:,None] & manual['eligible'][:,None,None,None]).expand_as(manual['angles'])
    fractions, angle_grads = {}, {}
    for term in ['scalar','dense']:
        gradient = torch.autograd.grad(manual[term],manual['angles'],retain_graph=True)[0]
        angle_grads[term] = gradient
        fractions[term] = float(((gradient.abs()>1e-12)&active).sum()/active.sum())
        assert np.isclose(fractions[term],angle_gradient_support(production)[term]['fraction'])
    fig,axes = plt.subplots(2,1,figsize=(9,3.6),sharex=True,constrained_layout=True)
    for ax,term in zip(axes,['scalar','dense']):
        ax.imshow((angle_grads[term][0].abs()>1e-12).reshape(2,T,2).permute(0,2,1).reshape(4,T),
                  aspect='auto',vmin=0,vmax=1,cmap='Blues')
        ax.set_yticks(range(4),['a left','a right','b left','b right']);ax.set_title(term+' angle-gradient support')
    axes[-1].set_xlabel('Frame in one constructed pair')
    fig.suptitle('ILLUSTRATION · Where the two angle losses supply a derivative')
    buffer=BytesIO();fig.savefig(buffer,format='png',dpi=130);display(Image(data=buffer.getvalue()));plt.close(fig)
    print('Constructed-batch nonzero angle-gradient fractions:',fractions)
    '''),
        *build_repair_training_cells(md, code),
    ]


def build_repair_training_cells(md, code):
    return [md(r'''
    ## 5 · Trace a frozen-encoder forward pass and coordinate reduction

    Notebook 01 derives normalization from observed inputs; notebook 03 derives
    the encoder. We reuse those established operations here. The newly expanded
    part is the readout-to-loss path: LayerNorm → linear → GELU → linear → unpack
    the four-frame joint token → add the usable input coordinates → undo the
    input-only normalization. The angle loss then receives pixel coordinates.

    Coordinate training uses squared error in input-normalized coordinates,
    averaged over x/y, then valid joints and frames within each endpoint, then
    supported endpoints. This differs from the evaluator's unsquared distance
    normalized by a reference rendering box. References remain loss-only data.
    '''), code(r'''
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        initial_model = RestorationModel('paired_jepa',cfg).cpu()
        initial_model.requires_grad_(False)
        torch.random.default_generator.manual_seed(17+100003)
        initial_model.readout = CoordinateReadout(cfg)
    initial_model.train(); initial_model.encoder.eval()
    assert all(not p.requires_grad for p in initial_model.encoder.parameters())
    assert all(p.requires_grad for p in initial_model.readout.parameters())
    assert all(not p.requires_grad for p in initial_model.teacher.parameters())
    bundle = SimpleNamespace(inputs=raw,targets=dict(xy=truth.reshape(2*Q,T,J,2),valid=reference_valid.reshape(2*Q,T,J)))
    context = dict(cfg=cfg,device='cpu',precision='float32',measurement=settings)

    def explicit_forward(model,selected):
        inputs = {k:v[selected] for k,v in raw.items()}
        normalized,origin,scale,fallback = normalize_batch(inputs,patch_size=cfg.patch_size)
        tensors = {k:torch.as_tensor(v) for k,v in normalized.items()}
        with torch.no_grad():
            tokens = model.encoder(tensors)  # Explicit encoder derivation is in 03.
        layers = model.readout.network
        correction = layers[3](F.gelu(layers[1](layers[0](tokens))))
        correction = correction.reshape(len(selected),T//cfg.patch_size,J,cfg.patch_size,2).permute(0,1,3,2,4).reshape(len(selected),T,J,2)
        predicted = correction + torch.where(tensors['observed'][...,None],tensors['xy'],0)
        flags = torch.as_tensor(bundle.targets['valid'][selected])
        ref = bundle.targets['xy'][selected]
        safe_ref = np.where(flags.numpy()[...,None],ref,origin[:,None,None])
        target = torch.as_tensor((safe_ref-origin[:,None,None])/scale[:,None,None,None])
        residual = torch.where(flags[...,None],predicted,0)-torch.where(flags[...,None],target,0)
        errors = residual.square().mean(-1)
        counts = flags.sum((1,2)); supported = counts>0
        per_endpoint = errors.sum((1,2))/counts.clamp_min(1)
        coordinate = per_endpoint[supported].mean()
        pixels = predicted*torch.as_tensor(scale[:,None,None,None])+torch.as_tensor(origin[:,None,None])
        terms = explicit_repair(pixels.reshape(-1,2,T,J,2),torch.as_tensor(ref).reshape(-1,2,T,J,2),flags.reshape(-1,2,T,J))
        assert supported.all() and terms['eligible'].any()
        return coordinate,terms,predicted

    selected = np.arange(2*Q)
    coordinate,terms,predicted = explicit_forward(initial_model,selected)
    source_coordinate,source_terms,_ = source_forward(bundle,initial_model,context,selected)
    torch.testing.assert_close(coordinate,source_coordinate)
    for name in ['scalar','dense','geometry']:
        torch.testing.assert_close(terms[name],source_terms[name],atol=2e-7,rtol=2e-5)
    print('Coordinate loss, normalized squared units:',float(coordinate.detach()))
    print('Only these parameter tensors are trainable:',[n for n,p in initial_model.named_parameters() if p.requires_grad])
    '''), md(r'''
    ## 6 · Calibrate the new loss without updating any parameters

    For each encoder and seed, production draws 32 training batches using the
    inherited sampler, at the same initialized readout. It differentiates each
    loss with respect to **all readout parameters**. An unused parameter has a
    zero gradient. If $K$ is parameter count and $M$ is batch count,

    $$G_s=\sqrt{\frac{\sum_{m=1}^{M}\|\nabla_w L_s^{(m)}\|^2}{MK}},\qquad
    G_d=\sqrt{\frac{\sum_{m=1}^{M}\|\nabla_w L_d^{(m)}\|^2}{MK}}.$$

    With inherited change weight $w=1$, $\lambda_s=0.1w$ and
    $\lambda_d=\lambda_s G_s/G_d$. Therefore $\lambda_dG_d=\lambda_sG_s$.
    The target is the **downweighted scalar loss's gradient strength**. Coordinate
    gradients are recorded as a diagnostic, not used as this equality's target.
    Squared gradient energies are summed before the square root; averaging
    separate per-batch gradient norms would produce a different coefficient.

    The small demonstration uses four constructed batches. It preserves the
    exact reduction, includes zero gradients, and checks each gradient vector's
    statistics against production. The zero-initialized final layer initially
    blocks gradients into earlier readout layers; those parameters still count.
    '''), code(r'''
    # These batches stand in for training-only hierarchical draws derived in 04.
    # Both objectives will replay the same ordered endpoints after calibration.
    endpoint_batches = [np.array(x) for x in [[0,1,2,3],[2,3,0,1],[0,1,0,1],[2,3,2,3]]]
    parameters = list(initial_model.readout.parameters())
    K = sum(p.numel() for p in parameters)
    snapshot = {k:v.clone() for k,v in initial_model.state_dict().items()}

    def gradient_vector(loss,parameters):
        gradients = torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten()
                          for p,g in zip(parameters,gradients)])

    energies = {name:0. for name in ['coordinate','scalar','dense','geometry']}
    for selected in endpoint_batches:
        base,terms,_ = explicit_forward(initial_model,selected)
        source_base,source_terms,_ = source_forward(bundle,initial_model,context,selected)
        oracle = _gradient_audit(source_base,source_terms,parameters)
        for name,loss in [('coordinate',base),*[(name,terms[name]) for name in ['scalar','dense','geometry']]]:
            vector = gradient_vector(loss,parameters)
            energy = float(vector.double().square().sum())
            np.testing.assert_allclose(energy,oracle[name]['squared_l2'],rtol=3e-5,atol=1e-10)
            energies[name] += energy
    assert energies['scalar']>0 and energies['dense']>0
    assert all(torch.equal(v,snapshot[k]) for k,v in initial_model.state_dict().items())
    assert all(p.grad is None for p in initial_model.parameters()), 'autograd.grad did not accumulate optimizer gradients.'
    rms = {k:math.sqrt(v/(len(endpoint_batches)*K)) for k,v in energies.items()}
    coefficients = {'scalar_low':.1,'dense_change':.1*math.sqrt(energies['scalar']/energies['dense']),'geometry':1.}
    assert np.isclose(coefficients['dense_change']*rms['dense'],coefficients['scalar_low']*rms['scalar'])
    display(pd.DataFrame({'Term':list(rms),'Initial gradient RMS':list(rms.values())}))
    print('ILLUSTRATIVE coefficients:',coefficients)
    '''), md(r'''
    ## 7 · Perform matched readout updates, leaving the encoder fixed

    The original readout uses $L_x+wL_s+wL_g$. The new low-scalar arm uses
    $L_x+0.1wL_s+wL_g$, and the dense arm uses $L_x+\lambda_dL_d+wL_g$.
    $L_x$ is coordinate loss and $L_g$ is the geometry penalty. Each new arm gets
    the same initial head, encoder, ordered batches, optimizer, and learning-rate
    schedule. The trained head and its optimizer then evolve separately.

    The source schedule warms up linearly for 5% of 2,000 readout updates, then
    follows a cosine decay. Each update clears accumulated gradients, computes
    the total loss, differentiates, clips the combined parameter-gradient norm
    at 1, and takes one AdamW step. There is no teacher averaging during repair.
    We execute two scratch updates per arm and compare complete model states
    and AdamW moment estimates with the same updates using the source loss path.
    '''), code(r'''
    def scheduled_rate(step,total=2000,base_rate=3e-4):
        warmup=max(1,round(.05*total))
        progress=(step-warmup)/max(total-warmup,1)
        return base_rate*((step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*progress)))

    trained,history = {},[]
    for objective,angular_term in [('scalar_low','scalar'),('dense_change','dense')]:
        model,oracle_model = deepcopy(initial_model),deepcopy(initial_model)
        optimizer = torch.optim.AdamW(model.readout.parameters(),lr=3e-4,weight_decay=.01)
        oracle_optimizer = torch.optim.AdamW(oracle_model.readout.parameters(),lr=3e-4,weight_decay=.01)
        for step,selected in enumerate(endpoint_batches[:2]):
            for current,opt,use_source in [(model,optimizer,False),(oracle_model,oracle_optimizer,True)]:
                if use_source:
                    base,terms,_ = source_forward(bundle,current,context,selected)
                else:
                    base,terms,_ = explicit_forward(current,selected)
                loss = base + coefficients[objective]*terms[angular_term] + coefficients['geometry']*terms['geometry']
                for group in opt.param_groups:group['lr']=scheduled_rate(step)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                assert all(p.grad is None for p in current.encoder.parameters())
                assert all(p.grad is None for p in current.teacher.parameters())
                trainable = list(current.readout.parameters())
                before_clip = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in trainable if p.grad is not None]))
                reported_norm = torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
                torch.testing.assert_close(before_clip,reported_norm)
                opt.step()
                if not use_source:history.append(dict(objective=objective,step=step+1,loss=float(loss.detach()),
                                                       learning_rate=scheduled_rate(step),gradient_norm=float(reported_norm)))
            for name,value in model.state_dict().items():
                torch.testing.assert_close(value,oracle_model.state_dict()[name],atol=2e-7,rtol=2e-5)
            # AdamW retains a first and second moment for every fitted parameter.
            for parameter,oracle_parameter in zip(model.readout.parameters(),oracle_model.readout.parameters()):
                for key,value in optimizer.state[parameter].items():
                    torch.testing.assert_close(value,oracle_optimizer.state[oracle_parameter][key],atol=2e-7,rtol=2e-5)
        assert all(torch.equal(v,snapshot[k]) for k,v in model.state_dict().items() if not k.startswith('readout.'))
        assert any(not torch.equal(v,snapshot[k]) for k,v in model.state_dict().items() if k.startswith('readout.'))
        trained[objective]=model
    display(pd.DataFrame(history))
    assert torch.equal(random_state_before,torch.random.get_rng_state())
    print('Source-parity updates passed. Only readout parameters changed; shared CPU Torch RNG is unchanged.')
    '''), md(r'''
    ## 8 · Deploy on observed inputs, then compare with retained source receipts

    Deployment receives only estimated coordinates, scores, availability and
    time. The teacher, reference validity, source identity and intervention label
    are not prediction inputs. Normalization is recomputed from these observed
    inputs without the artificial pretraining mask, and predictions are returned
    to pixels before measurement. The final readout predicts full coordinate
    trajectories; it does not directly output the scalar excursion score.

    The optional receipt replay below reconstructs all six source calibration
    coefficients from their 32 recorded gradient-energy batches. These are
    measurements from the completed experiment, separate from the scratch models
    above. Their presence does not turn a CPU demonstration into HAIC validation.
    '''), code(r'''
    normalized,origin,scale,_ = normalize_batch(raw,patch_size=cfg.patch_size)
    inference_inputs = {k:torch.as_tensor(v) for k,v in normalized.items()}
    assert set(inference_inputs)=={'xy','confidence','observed','timestamps'}
    deployed = trained['dense_change'].eval()
    with torch.no_grad():
        output = deployed(inference_inputs)
        output_px = output.numpy()*scale[:,None,None,None]+origin[:,None,None]
    assert output_px.shape==(2*Q,T,12,2) and np.isfinite(output_px).all()
    print('Deployment output:',output_px.shape,'pixel coordinates; no references passed to the model.')

    DATA=Path(os.environ.get('GF_EVIDENCE_ROOT',ROOT/'outputs/iclr')).resolve()
    receipt_path=DATA/'readout-repair/ledger.json'
    if receipt_path.exists():
        saved_config=json.loads((DATA/'readout-repair/config.json').read_text())
        assert saved_config['fixture'] is False and saved_config['study_kind']=='gait_fidelity_readout_repair'
        source_rows=[]
        for phase,record in json.loads(receipt_path.read_text())['completed'].items():
            if not phase.startswith('calibrate-'):continue
            receipt=record['result'];batches=receipt['batches']
            assert receipt['training_only'] is True and len(batches)==32
            scalar_energy=sum(b['gradients']['scalar']['squared_l2'] for b in batches)
            dense_energy=sum(b['gradients']['dense']['squared_l2'] for b in batches)
            reconstructed=receipt['coefficients']['scalar_low']*math.sqrt(scalar_energy/dense_energy)
            np.testing.assert_allclose(reconstructed,receipt['coefficients']['dense_change'],rtol=1e-12)
            np.testing.assert_allclose(reconstructed*receipt['gradients']['dense']['rms'],
                                      receipt['coefficients']['scalar_low']*receipt['gradients']['scalar']['rms'],rtol=1e-12)
            source_rows.append({'Encoder':receipt['representation_variant'],'Seed':receipt['seed'],
                               'Batches':len(batches),'Dense coefficient':reconstructed,
                               'Scalar coefficient':receipt['coefficients']['scalar_low']})
        assert len(source_rows)==6
        display(pd.DataFrame(source_rows))
    else:
        print('No compact source repair ledger found; source-calibration replay skipped explicitly.')
    '''), md(r'''
    ## 9 · Interpret what the repair can establish

    Initialization matching does not guarantee that the losses retain the same
    gradient scale or direction throughout optimization. Dense supervision also
    changes which temporal information is supplied, so a gain cannot uniquely
    identify sparse scalar gradients as the cause. The primary repair comparison
    is **delta JEPA / dense versus delta JEPA / low scalar on ViTPose waveform
    error**. Notebook 06 reconstructs its person-level uncertainty; notebook 07
    displays the completed results and retained original/base comparators.

    The study shares the same fourteen development people across core, response
    and repair. The protected confirmation run is not part of the downloaded
    result. Successful calculations here establish agreement with the implemented
    procedure, while generalization and scientific claims come from the saved
    evaluation and its stated population.
    ''')]
