"""Executable architecture and optimization lessons, emitted as notebook cells.

Scratch examples operate on local CPU tensors and independently initialized
models. They never load, overwrite, or train a retained study checkpoint.
"""


def build_architecture_cells(md, code):
    return [
        md(r"""
## From observed coordinates to a restored trajectory

The production model consumes a fixed window of 12 body joints. For each joint
and frame it receives `(x, y)`, the estimator's confidence, an observed flag,
and elapsed time in seconds. Reference coordinates enter training losses and
the training-only teacher; they never enter the restoration model's input.

The encoder groups four consecutive frames of **one joint** into a token.
For a window of $T$ frames, patch length $P$, 12 joints, and feature width $D$,
the shapes are

$$[B,T,12,5]\longrightarrow[B,T/P,12,5P]
  \longrightarrow[B,12T/P,D].$$

We will trace a fresh, small CPU model with $T=16$, $P=4$, $D=16$, and two
examples. The saved source study normally uses 128 frames and width 96;
the table below reads your actual configuration. This local numerical example
uses the same operators and parameter layout, with fewer tokens and layers.
Its random weights and generated trajectories provide software checks only.

This is the repository's body-12 restoration architecture. The
multiple-sclerosis tutorials provide the explanatory style, but their
33-landmark model and classification task are different implementations.
"""),
        code(r"""
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from IPython.display import display
from gavd6_sjepa.research_directions.synthetic_training_v2.models import ModelConfig, RestorationModel
from gavd6_sjepa.research_directions.gait_fidelity.training import normalize_batch

saved_config = study.artifact('config.json')
small = ModelConfig(width=16, encoder_layers=1, predictor_layers=1,
                    heads=2, patch_size=4, window_size=16)
display(pd.DataFrame({'saved_study': saved_config['model'], 'teaching_example': vars(small)}))

# Each construction restores the caller's random state on exit.
torch_state_before = torch.random.get_rng_state().clone()
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(601)
    scratch = RestorationModel('paired_jepa', small).cpu().eval()
assert torch.equal(torch_state_before, torch.random.get_rng_state())

B, T, J, P, D = 2, small.window_size, 12, small.patch_size, small.width
S = T // P
template = np.array([[-25,20],[25,20],[-35,55],[35,55],[-40,85],[40,85],
                     [-20,95],[20,95],[-20,145],[20,145],[-20,195],[20,195]], np.float32)
seconds = np.arange(T, dtype=np.float32) / 25
reference_px = np.broadcast_to(template, (B,T,J,2)).copy()
reference_px[:, :, 8, 0] += 12 * np.sin(2*np.pi*seconds)[None]
reference_px[:, :, 9, 0] += 16 * np.sin(2*np.pi*seconds + .3)[None]
observed_px = reference_px + np.array([3., -2.], np.float32)
observed = np.ones((B,T,J), bool)
observed[0, :P, 10] = False
observed_px[~observed] = np.nan
raw = dict(xy=observed_px, confidence=np.full((B,T,J), .8, np.float32),
           observed=observed, timestamps=np.broadcast_to(seconds, (B,T)).copy())
hidden_np = np.zeros((B,S,J), bool)
hidden_np[:, 2, 8:10] = True
normalized, origin, scale, fallbacks = normalize_batch(raw, hidden_np, patch_size=P)
inputs = {key: torch.as_tensor(value) for key, value in normalized.items()}
hidden = torch.as_tensor(hidden_np)
print({'example_scope': 'generated CPU teaching tensors',
       'input_shape': tuple(inputs['xy'].shape), 'context_normalization_fallbacks': fallbacks})
"""),
        md(r"""
## Pack the five channels without leaking hidden coordinates

Artificial masks are repeated across the four frames of their patch. A usable
coordinate must be observed and not artificially hidden. Hidden coordinates
and their confidence become zero; elapsed time remains available.

Natural absence has a separate convention: its coordinates and observed flag
are zeroed, while a finite native confidence is retained unless that slot is
also artificially hidden. Thus confidence and visibility are distinct inputs.
The input-only normalization computed above excludes artificially hidden
coordinates before calculating its origin and scale, as notebook 01 explains.

The following code reproduces the production channel order and token order.
Token index `patch * 12 + joint` identifies one joint in one temporal patch.
"""),
        code(r"""
artificial = hidden.repeat_interleave(P, dim=1)
usable = inputs['observed'] & ~artificial
safe_xy = torch.where(usable[..., None], inputs['xy'], 0)
confidence = torch.where(~artificial, inputs['confidence'], 0)
elapsed = inputs['timestamps'] - inputs['timestamps'][:, :1]
channels = torch.cat([safe_xy, confidence[..., None], usable[..., None].float(),
                      elapsed[:, :, None, None].expand(-1, -1, J, -1)], dim=-1)
patches = channels.reshape(B,S,P,J,5).permute(0,1,3,2,4).reshape(B,S*J,P*5)
assert torch.equal(patches[0, 2*J+8], channels[0, 2*P:3*P, 8].flatten())
assert torch.isfinite(patches).all()
print('Channels:', tuple(channels.shape), 'Patch vectors:', tuple(patches.shape))
display(pd.DataFrame(channels[0, 2*P:3*P, 8].numpy(),
                     columns=['x', 'y', 'confidence', 'usable', 'elapsed_seconds']))
"""),
        md(r"""
## Add position and joint identity

A linear projection maps the $5P$ numbers into $D$ learned features. The model
adds a fixed sinusoidal encoding of the patch index and a learned embedding
of the joint identity. Physical seconds remain in the input channels; patch
indices supply a separate ordered query grid.

$$z_{s,j}=W\,\mathrm{patch}_{s,j}+b+e^{\mathrm{time}}_s
          +e^{\mathrm{joint}}_j
          +\mathbf{1}[\text{no usable frame}]\,e^{\mathrm{missing}}.$$

All patch–joint slots remain in the sequence, including missing slots. The
learned missing-query vector starts at zero. A missing slot can attend to
other slots and eventually receive a prediction; it is not removed by an
attention padding mask.
"""),
        code(r"""
patch_number = torch.arange(S, dtype=torch.float32)[:, None]
frequency = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.) / D))
position = torch.zeros(S, D)
position[:, 0::2] = torch.sin(patch_number * frequency)
position[:, 1::2] = torch.cos(patch_number * frequency[:D//2])
patch_supported = usable.reshape(B,S,P,J).any(dim=2).reshape(B,S*J)
tokens = F.linear(patches, scratch.encoder.projection.weight, scratch.encoder.projection.bias)
tokens = tokens + (position[:, None] + scratch.encoder.joints.weight[None]).reshape(1,S*J,D)
tokens = tokens + (~patch_supported)[..., None] * scratch.encoder.missing_query
print({'tokens': tuple(tokens.shape),
       'entirely_missing_patch_slots': int((~patch_supported).sum()),
       'all_slots_retained': tokens.shape[1] == S*J})
"""),
        md(r"""
## Calculate one self-attention layer explicitly

Each attention head makes query, key, and value vectors from normalized
tokens. Their dot products determine the weights used to combine values:

$$Q=XW_Q^\top+b_Q,\quad K=XW_K^\top+b_K,\quad V=XW_V^\top+b_V,$$
$$\mathrm{Attention}(X)=\mathrm{softmax}(QK^\top/\sqrt{d_h})V.$$

The production Transformer uses normalization **before** each sublayer, a
residual addition around attention, and another around its GELU feed-forward
network. The feed-forward hidden width is $4D$ and dropout is zero. Attention
uses the full window in both temporal directions, so this is offline
restoration rather than causal forecasting.
"""),
        code(r"""
layer = scratch.encoder.blocks.layers[0]
normalized_tokens = layer.norm1(tokens)
q, k, v = F.linear(normalized_tokens, layer.self_attn.in_proj_weight,
                    layer.self_attn.in_proj_bias).chunk(3, dim=-1)
H, Dh, N = small.heads, D // small.heads, S * J
split_heads = lambda a: a.reshape(B,N,H,Dh).transpose(1,2)
q, k, v = map(split_heads, (q,k,v))
attention_weights = (q @ k.transpose(-2,-1) / math.sqrt(Dh)).softmax(dim=-1)
context = (attention_weights @ v).transpose(1,2).reshape(B,N,D)
attention_output = F.linear(context, layer.self_attn.out_proj.weight,
                            layer.self_attn.out_proj.bias)
after_attention = tokens + attention_output
feedforward = layer.linear2(F.gelu(layer.linear1(layer.norm2(after_attention))))
after_layer = after_attention + feedforward
torch.testing.assert_close(after_layer, layer(tokens), atol=2e-6, rtol=2e-5)
torch.testing.assert_close(attention_weights.sum(-1), torch.ones(B,H,N))

encoded = scratch.encoder.norm(after_layer)
torch.testing.assert_close(encoded, scratch.encoder(inputs, hidden), atol=2e-6, rtol=2e-5)
print('Explicit attention and encoder match production:', tuple(encoded.shape))
"""),
        code(r"""
import matplotlib.pyplot as plt
from io import BytesIO
from IPython.display import Image
fig, ax = plt.subplots(figsize=(6,5), constrained_layout=True)
plot = ax.imshow(attention_weights[0].mean(0).detach().numpy(), aspect='equal', cmap='viridis')
ax.set(xlabel='Key token (patch × 12 + joint)', ylabel='Query token (patch × 12 + joint)',
       title='Generated CPU example: random attention')
fig.colorbar(plot, ax=ax, label='Attention weight (head mean)')
buffer = BytesIO(); fig.savefig(buffer, format='png', dpi=120)
display(Image(data=buffer.getvalue())); plt.close(fig)
print('These weights illustrate the operator; they are not learned anatomical importance.')
"""),
        md(r"""
## Decode features into coordinates

The output network applies LayerNorm, a linear layer, GELU, and a final linear
layer that emits $2P$ coordinate corrections per token. Reshaping restores the
frame order. At observed locations the network predicts a residual added to
the input coordinate; at missing locations it predicts an absolute normalized
coordinate, with zero as the addition base.

$$\widehat{x}_{t,j}=r_{t,j}+
  \begin{cases}x_{t,j},&\text{usable input};\\0,&\text{otherwise.}\end{cases}$$

The final linear layer starts at zero. Thus this fresh model initially passes
through usable observations and emits the normalization origin in pixel space
at missing slots. Those missing predictions can be inaccurate; the loss and
evaluation retain their errors.
"""),
        code(r"""
readout_layers = scratch.readout.network
patch_correction = readout_layers[3](F.gelu(readout_layers[1](readout_layers[0](encoded))))
correction = patch_correction.reshape(B,S,J,P,2).permute(0,1,3,2,4).reshape(B,T,J,2)
restored_normalized = correction + torch.where(usable[..., None], inputs['xy'], 0)
torch.testing.assert_close(restored_normalized, scratch(inputs, hidden), atol=2e-6, rtol=2e-5)
assert torch.count_nonzero(correction) == 0
restored_pixels = restored_normalized.detach().numpy() * scale[:,None,None,None] + origin[:,None,None]
np.testing.assert_allclose(restored_pixels[usable.numpy()], observed_px[usable.numpy()], atol=2e-5)
print({'feature_shape': tuple(encoded.shape), 'prediction_shape': tuple(restored_pixels.shape),
       'initial_correction_maximum': float(correction.abs().max().detach())})
"""),
        md(r"""
The numerical derivation used a small model to keep the tensors readable. We
also check the source study's default dimensions on CPU: 128 frames form
384 patch–joint tokens, each with 96 features. This shape check uses another
fresh model and has no bearing on trained accuracy or HAIC GPU compatibility.
"""),
        code(r"""
source_dimensions = ModelConfig(window_size=128)
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(602)
    shape_model = RestorationModel('direct', source_dimensions).cpu().eval()
shape_inputs = dict(xy=torch.as_tensor(template)[None,None].expand(1,128,12,2)/200,
                    confidence=torch.ones(1,128,12), observed=torch.ones(1,128,12,dtype=torch.bool),
                    timestamps=torch.arange(128,dtype=torch.float32)[None]/25)
with torch.no_grad():
    source_features = shape_model.encoder(shape_inputs)
    source_output = shape_model(shape_inputs)
assert source_features.shape == (1,384,96) and source_output.shape == (1,128,12,2)
assert torch.isfinite(source_output).all()
assert torch.equal(torch_state_before, torch.random.get_rng_state())
print('Default source shapes:', tuple(source_features.shape), '→', tuple(source_output.shape))
del shape_model
"""),
        md(r"""
## Connect the architecture to the experiment matrix

Coordinate pretraining fits the encoder and coordinate readout at queried
patches. Paired JEPA fits an encoder and feature predictor against an
exponential-moving-average teacher that sees matching reference trajectories.
Its predictor produces $D$ features per token and is distinct from the
coordinate readout. Notebook 04 derives both losses and executes the updates.

| Phase or model | Updated parameters | Inputs in this phase |
| --- | --- | --- |
| Coordinate pretraining | Encoder and coordinate readout | Masked observations |
| Paired or shuffled-reference JEPA pretraining | Encoder, feature predictor, and regularization projector; teacher by moving average | Masked observations |
| Frozen-feature output training | A freshly initialized coordinate readout | Ordinary observed inputs, without an artificial pretraining mask |
| Direct training | Encoder and coordinate readout together | Ordinary observed inputs |
| Initialized encoder control | A new readout over fixed random encoder features | Ordinary observed inputs |
| Static benchmark | Framewise coordinate network plus full-window confidence/visibility/time context | Current-frame coordinates and shared auxiliary context |
| Temporal refiner | Shared per-joint temporal MLP | Each joint's full-window channels |

The static benchmark removes neighboring-frame **coordinates**, while retaining
full-window auxiliary information. The temporal refiner is a local 2D
SmoothNet-style adaptation; the study does not claim to reproduce author
weights or the original benchmark. These practical models are explained further
in experiment C.

The saved matrix fixes every comparison before ranking results. Three seeds
measure training variability; they do not add independent evaluation people.
"""),
        code(r"""
study.command('plan', quiet=True)
plan = study.artifact('plan.json')
recipes = pd.DataFrame(plan['recipes'])
phases = pd.DataFrame(plan['phases'])
print(json.dumps(plan['counts'], indent=2))
display(recipes)
display(phases[['phase_id', 'phase', 'seed', 'depends_on']].head(20))
assert len(recipes) == plan['counts']['recipes']
assert plan['counts']['final_fits'] == len(recipes) * len(plan['seeds'])
"""),
        md(r"""
Identical pretraining can supply several independently fitted output networks
only when its data, mask, objective and seed agree. The three-seed core plan
contains nine pretraining phases, 24 frozen-feature output phases and six
end-to-end phases, producing 30 final models. The full plan contains 33
pretraining phases and 102 final models. The table above reports the selected
plan, so results from a core run cannot imply that omitted controls were fitted.

The source schedule starts at 2,000 updates per pretraining or output phase and
4,000 for end-to-end training. Hardware profiling can select the declared
common fallback before final fits. The scratch calculations above do not
change that plan, its random states, or its checkpoints. Continue to
**04_run_and_monitor.ipynb** to calculate the losses and follow optimization.
"""),
    ]


def build_training_cells(md, code):
    return [
        md(r"""
## Build a small training example before launching the study

This notebook first performs a few CPU updates on newly created teaching
models. Every tensor, optimizer, and random generator is local to the example;
no retained checkpoint is loaded or changed. The official study launch appears
after the calculations.

Two generated movement pairs supply four endpoints: a baseline and a changed
movement for each pair. We make reference coordinates, add an estimation bias,
and mark one observation as missing. The geometry is deliberately simple and
does not represent AMASS data or a gait disorder. The operations below are
checked against the production functions used by the actual study.
"""),
        code(r"""
import math
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from IPython.display import display
from gavd6_sjepa.research_directions.synthetic_training_v2.models import ModelConfig, RestorationModel, CoordinateReadout
from gavd6_sjepa.research_directions.synthetic_training_v2.training import coordinate_loss, _coordinate_patch_loss
from gavd6_sjepa.research_directions.temporal_gait.objectives import predictive_loss, vicreg_loss
from gavd6_sjepa.research_directions.gait_fidelity.training import normalize_batch
from gavd6_sjepa.research_directions.gait_fidelity.masking import sample_mask
from gavd6_sjepa.research_directions.gait_fidelity.measurements import measurement_loss

saved = study.artifact('config.json')
tc = saved['training']
small = ModelConfig(width=16, encoder_layers=1, predictor_layers=1,
                    heads=2, patch_size=4, window_size=32)
B, T, J, P, D = 4, small.window_size, 12, small.patch_size, small.width
S = T // P
seconds = np.arange(T, dtype=np.float32) / 25
template = np.array([[-25,20],[25,20],[-35,55],[35,55],[-40,85],[40,85],
                     [-20,95],[20,95],[-20,145],[20,145],[-20,195],[20,195]], np.float32)
truth_px = np.broadcast_to(template, (B,T,J,2)).copy()
for row in range(B):
    phase = 2*np.pi*seconds + .25 * (row // 2)
    truth_px[row,:,8,0] += 12*np.sin(phase)
    truth_px[row,:,9,0] += (14 + 9*(row % 2))*np.sin(phase + .2)
xy = truth_px + np.array([3., -2.], np.float32)
observed = np.ones((B,T,J), bool)
observed[0, :P, 10] = False
xy[~observed] = np.nan
raw = dict(xy=xy, confidence=np.full((B,T,J), .8, np.float32), observed=observed,
           timestamps=np.broadcast_to(seconds, (B,T)).copy())
target_valid = np.ones((B,T,J), bool)
target_valid[0, 0, 0] = False  # A partly valid reference patch still has three valid frames.
hidden_np = sample_mask(observed, 'graph_time', rng=np.random.default_rng(607),
                        fraction=.5, patch_size=P)
normalized, origin, scale, _ = normalize_batch(raw, hidden_np, patch_size=P)
inputs = {key: torch.as_tensor(value) for key,value in normalized.items()}
hidden = torch.as_tensor(hidden_np)
valid = torch.as_tensor(target_valid)
target = torch.as_tensor((truth_px-origin[:,None,None])/scale[:,None,None,None])
before_rng = torch.random.get_rng_state().clone()
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(609)
    student = RestorationModel('paired_jepa', small).cpu()
assert torch.equal(before_rng, torch.random.get_rng_state())
student.requires_grad_(False)
for component in (student.encoder, student.predictor, student.projector):
    component.requires_grad_(True)
student.train()
print('Teaching endpoints:', B, '| patches per endpoint:', S*J)
"""),
        md(r"""
## Derive the query set and coordinate objective

A patch is queried if it is artificially hidden **or any of its input frames
are naturally missing**. A queried patch is eligible for supervision if at
least one reference frame is valid. The reference support determines the loss
mask only; every patch–joint slot remains an encoder query regardless of its
reference support.

For coordinate pretraining, we first average squared $x/y$ errors over valid
frames within a patch, then over eligible queried patches within each window,
and finally over supported windows. This gives windows equal weight even if
their numbers of valid reference frames differ:

$$L_{\mathrm{coord,pre}}=\frac1{|\mathcal B_+|}\sum_{b\in\mathcal B_+}
  \frac1{|Q_b|}\sum_{q\in Q_b}\frac1{|V_{bq}|}
  \sum_{t\in V_{bq}}\frac{\|\widehat x_{btj}-y_{btj}\|_2^2}{2}.$$

Here $Q_b$ contains eligible queries and $V_{bq}$ contains valid frames in
that query patch. Output training instead averages valid frame–joint errors
within each window without the pretraining query mask.
"""),
        code(r"""
natural_missing = (~inputs['observed']).reshape(B,S,P,J).any(dim=2)
queries = hidden | natural_missing
reference_counts = valid.reshape(B,S,P,J).sum(dim=2)
query_valid = (queries & (reference_counts > 0)).flatten(1)
assert reference_counts[0,0,0] == 3  # Eligibility is any-reference-frame, not all frames.

def equal_window_mean(errors, support):
    counts = support.flatten(1).sum(1)
    per_window = torch.where(support, errors, 0).flatten(1).sum(1) / counts.clamp_min(1)
    supported = counts > 0
    if not bool(supported.any()):
        raise ValueError('A teaching batch without supervision cannot be counted as zero loss.')
    return per_window[supported].mean(), supported

with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(610)
    coordinate_model = RestorationModel('coordinate', small).cpu()
coordinate_prediction = coordinate_model(inputs, hidden)
safe_target = torch.where(valid[...,None], target, 0)
frame_errors = (coordinate_prediction-safe_target).square().mean(-1)
patch_errors = torch.where(valid.reshape(B,S,P,J), frame_errors.reshape(B,S,P,J), 0).sum(2)
patch_errors = patch_errors / reference_counts.clamp_min(1)
explicit_coordinate_pre, _ = equal_window_mean(patch_errors.flatten(1), query_valid)
production_coordinate_pre, _, _, production_queries = _coordinate_patch_loss(
    coordinate_prediction, target, valid, queries, small)
torch.testing.assert_close(explicit_coordinate_pre, production_coordinate_pre)
assert torch.equal(query_valid, production_queries)
print({'coordinate_pretraining_MSE': float(explicit_coordinate_pre.detach()),
       'eligible_queries_per_window': query_valid.sum(1).tolist()})
"""),
        md(r"""
## Predict reference features with paired JEPA

The student encoder receives masked estimated coordinates. Its feature
predictor emits one $D$-dimensional vector per patch–joint slot. The teacher
encodes matching reference coordinates using its own parameters, which follow
the student by an exponential moving average. The teacher receives reference
validity and a confidence of one at valid reference slots; invalid reference
coordinates are zeroed. It receives no artificial mask.

Both paths use the input/context-derived normalization. The reference does
not choose the student's origin or scale. The shuffled-reference control
instead supplies another training window from the same person and condition
stratum, with its own input-derived transform.

The local JEPA objective is cross-entropy between distributions over **feature
dimensions**. These dimensions do not represent joints, diagnoses or class
labels. For eligible query $q$:

$$p_q=\mathrm{softmax}(z_q^{\mathrm{student}}/\tau_s),\qquad
  r_q=\mathrm{softmax}((\operatorname{stopgrad}(z_q^{\mathrm{teacher}})-c)/\tau_t),$$
$$\ell_q=-\sum_{d=1}^D r_{qd}\log p_{qd}.$$

The default temperatures are $\tau_s=0.10$ and $\tau_t=0.06$. We average this
loss over eligible queries within each window, then over supported windows.
This centered, sharpened objective is the repository's declared JEPA variant;
the term JEPA alone does not prescribe this particular loss.
"""),
        code(r"""
teacher_inputs = dict(xy=torch.where(valid[...,None], target, 0), observed=valid,
                      confidence=valid.float(), timestamps=inputs['timestamps'])
predicted_tokens = student.predictor(student.encoder(inputs, hidden))
with torch.no_grad():
    teacher_tokens = student.teacher(teacher_inputs)
student_temperature = float(tc.get('student_temperature', .1))
teacher_temperature = float(tc.get('teacher_temperature', .06))
teacher_distribution = ((teacher_tokens.detach().float()-student.center) / teacher_temperature).softmax(-1)
student_log_distribution = (predicted_tokens.float()/student_temperature).log_softmax(-1)
token_cross_entropy = -(teacher_distribution * student_log_distribution).sum(-1)
cross_entropy, supported = equal_window_mean(token_cross_entropy, query_valid)
production_ce, _, production_supported = predictive_loss(predicted_tokens, teacher_tokens, query_valid,
    objective='centered_ce_v1', center=student.center,
    student_temperature=student_temperature, teacher_temperature=teacher_temperature)
torch.testing.assert_close(cross_entropy, production_ce)
assert torch.equal(supported, production_supported) and not teacher_tokens.requires_grad
print({'feature_tensor_shape': tuple(predicted_tokens.shape),
       'feature_cross_entropy': float(cross_entropy.detach()),
       'uniform_student_reference': math.log(D)})
"""),
        md(r"""
## Show the representation regularizer

Feature agreement can become uninformative if all inputs receive nearly the
same representation. The production recipe adds a VICReg-style regularizer
to pooled encoder features from two small translated views. Translation is
applied consistently over each normalized window, with each coordinate drawn
from the declared range; the same artificial mask is used in both views.
Pooling averages all patch–joint features, including queries with missing
input information; the supported-window mask is applied after this average.

For projected feature matrices $U,V\in\mathbb R^{B\times D}$, the loss is
$25L_{\mathrm{invariance}}+25L_{\mathrm{variance}}+L_{\mathrm{covariance}}$.
The first term is their mean squared difference. The second penalizes feature
standard deviations below one. The third penalizes squared off-diagonal
covariances, discouraging redundant feature dimensions. The code shows the
precise reductions and stabilization constant.

This regularizer is included only when at least two windows have supported
queries. Its default multiplier in the total JEPA loss is 0.05. It does not
replace checking feature diversity or evaluating restored movement.
"""),
        code(r"""
view_generator = torch.Generator(device='cpu').manual_seed(611)
views = []
for _ in range(2):
    shift = (2*torch.rand(B,1,1,2, generator=view_generator)-1) * float(tc.get('translation_magnitude', .02))
    translated = dict(inputs, xy=torch.where(inputs['observed'][...,None], inputs['xy']+shift, inputs['xy']))
    views.append(student.projector(student.encoder(translated, hidden).mean(1))[supported])
u, v = views
invariance = (u-v).square().mean()
variance = .5 * (torch.relu(1-(u.var(0,unbiased=False)+1e-4).sqrt()).mean()
                 + torch.relu(1-(v.var(0,unbiased=False)+1e-4).sqrt()).mean())
covariance = u.new_zeros(())
for view in (u,v):
    centered_view = view-view.mean(0)
    covariance_matrix = centered_view.T @ centered_view / (len(view)-1)
    off_diagonal = covariance_matrix-torch.diag(torch.diagonal(covariance_matrix))
    covariance = covariance + off_diagonal.square().sum() / (2*view.shape[1])
regularizer = 25*invariance + 25*variance + covariance
torch.testing.assert_close(regularizer, vicreg_loss(u,v))
regularization_weight = float(tc.get('vicreg_weight', .05))
pretraining_loss = cross_entropy + regularization_weight*regularizer
display(pd.DataFrame([{'cross_entropy': float(cross_entropy.detach()),
                      'regularizer': float(regularizer.detach()), 'multiplier': regularization_weight,
                      'total_pretraining_loss': float(pretraining_loss.detach())}]))
"""),
        md(r"""
## Execute one optimizer step, then update teacher and center

AdamW updates the student encoder, predictor and projector using the loss
gradient. The coordinate readout remains inactive during JEPA pretraining.
The production learning rate warms up and then follows a cosine schedule;
gradient clipping bounds the combined gradient norm before the optimizer step.

After the student update, teacher parameters follow
$\theta_{\mathrm{teacher}}\leftarrow m\theta_{\mathrm{teacher}}
+(1-m)\theta_{\mathrm{student}}$. The scheduled coefficient increases from
0.99 toward 0.999. The center follows 90% old center plus 10% the batch mean
of the **pre-update teacher** features at eligible queries, with equal window
weight. Neither the teacher nor the center receives backpropagation gradients.

The printed loss is one generated-batch optimization diagnostic. A successful
gradient update establishes neither convergence nor useful gait features.
"""),
        code(r"""
parameters = [p for p in student.parameters() if p.requires_grad]
learning_rate = float(tc.get('learning_rate', 3e-4))
optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=float(tc.get('weight_decay', .01)))
step, schedule_updates = 0, int(tc['pretraining_updates'])
warmup = max(1, round(float(tc.get('warmup_fraction', .05))*schedule_updates))
progress = (step-warmup)/max(schedule_updates-warmup, 1)
scheduled_lr = learning_rate*((step+1)/warmup if step < warmup else .5*(1+math.cos(math.pi*progress)))
for group in optimizer.param_groups:
    group['lr'] = scheduled_lr
old_teacher = [p.detach().clone() for p in student.teacher.parameters()]
old_center = student.center.detach().clone()
old_encoder = [p.detach().clone() for p in student.encoder.parameters()]
optimizer.zero_grad(set_to_none=True)
pretraining_loss.backward()
assert all(p.grad is None for p in student.teacher.parameters())
assert all(p.grad is None for p in student.readout.parameters())
gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, float(tc.get('gradient_clip', 1.)), error_if_nonfinite=True)
optimizer.step()
assert any(not torch.equal(a,b) for a,b in zip(old_encoder, student.encoder.parameters()))
momentum = .999-(.999-.99)*.5*(1+math.cos(math.pi*step/max(1,schedule_updates-1)))
with torch.no_grad():
    for teacher_parameter, student_parameter in zip(student.teacher.parameters(), student.encoder.parameters()):
        teacher_parameter.mul_(momentum).add_(student_parameter, alpha=1-momentum)
    for teacher_buffer, student_buffer in zip(student.teacher.buffers(), student.encoder.buffers()):
        teacher_buffer.copy_(student_buffer)
    per_window_teacher_mean = (teacher_tokens.float()*query_valid[...,None]).sum(1)/query_valid.sum(1)[:,None].clamp_min(1)
    student.center.mul_(.9).add_(per_window_teacher_mean[supported].mean(0), alpha=.1)
for old, current, online in zip(old_teacher, student.teacher.parameters(), student.encoder.parameters()):
    torch.testing.assert_close(current, momentum*old+(1-momentum)*online)
torch.testing.assert_close(student.center, .9*old_center+.1*per_window_teacher_mean[supported].mean(0))
print({'scratch_update': 1, 'learning_rate': scheduled_lr, 'gradient_norm_before_clipping': float(gradient_norm),
       'teacher_momentum': momentum, 'teacher_has_gradients': False})
"""),
        md(r"""
## Change from pretraining to coordinate output training

The output phase loads its matching encoder, freezes it, and creates a fresh
coordinate readout and optimizer. Matching readout seeds give the comparisons
the same initialization. The feature predictor and teacher are not used for
deployment predictions. Ordinary observed inputs receive no artificial mask
in this phase, so the input-only origin and scale are recomputed from all
available observations.

The scratch model below reuses our single illustrative encoder update, rather
than a study checkpoint. In the actual run, the checkpoint must match the
declared data, model, mask, seed and training configuration.
"""),
        code(r"""
output_model = deepcopy(student)
output_model.requires_grad_(False)
for parameter in output_model.parameters():
    parameter.grad = None
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(17+100003)
    output_model.readout = CoordinateReadout(small)
output_model.readout.requires_grad_(True)
output_model.train(); output_model.encoder.eval()
unmasked, output_origin, output_scale, _ = normalize_batch(raw)
output_inputs = {key: torch.as_tensor(value) for key,value in unmasked.items()}
output_target = torch.as_tensor((truth_px-output_origin[:,None,None])/output_scale[:,None,None,None])
output_prediction = output_model(output_inputs)
errors = (output_prediction-torch.where(valid[...,None], output_target, 0)).square().mean(-1)
coordinate_term, _ = equal_window_mean(errors, valid)
torch.testing.assert_close(coordinate_term, coordinate_loss(output_prediction, output_target, valid)[0])
predicted_px = output_prediction*torch.as_tensor(output_scale[:,None,None,None]) + torch.as_tensor(output_origin[:,None,None])
print('Trainable modules:', [name for name,module in output_model.named_children()
                              if any(p.requires_grad for p in module.parameters())])
"""),
        md(r"""
## Derive the movement measurement and paired-change loss

For each frame and leg, let $u=\mathrm{hip}-\mathrm{knee}$ and
$v=\mathrm{ankle}-\mathrm{knee}$. The projected knee angle in degrees is

$$\theta=\frac{180}{\pi}\operatorname{atan2}(|u_xv_y-u_yv_x|,u\cdot v).$$

For each leg, excursion is the linear-interpolated 95th percentile minus the
5th percentile over fixed reference-supported frames. Our signed measurement
is $A=\mathrm{excursion}_{R}-\mathrm{excursion}_{L}$. It measures image-plane
geometry, so it is not anatomical three-dimensional range of motion.

For baseline and changed endpoints, paired-change supervision minimizes

$$L_{\Delta}=\left[
  \frac{(\widehat A_1-\widehat A_0)-(A_1-A_0)}{180}\right]^2.$$

The per-example control instead uses
$L_A=\tfrac12\sum_{e=0}^1[(\widehat A_e-A_e)/180]^2$.
The factor 180 expresses the training error in a common dimensionless scale;
evaluation reports errors in degrees. Coordinate loss remains present.

The next cell independently implements the training calculation. Common
support requires valid reference hips, knees and ankles for **both legs and
both endpoints**. Short predicted limbs receive an additional penalty, so a
model cannot remove difficult frames by collapsing its joints. Degenerate
angle derivatives are stabilized at the origin; exact percentile gradients
are sparse, and straight knees have a zero subgradient through the absolute
cross product. Notebook 05 explains the stricter evaluation failure rules.
"""),
        code(r"""
def explicit_angles_and_lengths(coordinates):
    angles, lengths = [], []
    for hip, knee, ankle in ((6,8,10), (7,9,11)):
        u = coordinates[...,hip,:]-coordinates[...,knee,:]
        v = coordinates[...,ankle,:]-coordinates[...,knee,:]
        cross = u[...,0]*v[...,1]-u[...,1]*v[...,0]
        dot = (u*v).sum(-1)
        dot = torch.where(cross.abs()+dot.abs() > 1e-12, dot, torch.full_like(dot,1e-12))
        angles.append(torch.atan2(cross.abs(),dot)*(180/np.pi))
        lengths.append(torch.stack([(u.square().sum(-1)+1e-12).sqrt(),
                                    (v.square().sum(-1)+1e-12).sqrt()], -1))
    return torch.stack(angles,-1), torch.stack(lengths,-2)

def explicit_measurement_loss(prediction, reference, flags, objective, min_segment, min_frames, min_fraction):
    pairs, _, frames, joints, dimensions = prediction.shape
    prediction = prediction.reshape(-1,frames,joints,dimensions).float()
    reference = reference.reshape_as(prediction).float()
    flags = flags.reshape(-1,frames,joints)
    safe_reference = torch.where(flags[...,None], reference, 0)
    target_angles, target_lengths = explicit_angles_and_lengths(safe_reference)
    good = flags[:,:,[6,8,10,7,9,11]].all(-1) & (target_lengths >= min_segment).all(-1).all(-1)
    fixed = good.reshape(pairs,2,frames).all(1)[:,None].expand(-1,2,-1).reshape(-1,frames)
    counts = fixed.sum(1)
    eligible = (counts >= min_frames) & (counts >= min_fraction*frames)
    predicted_angles, predicted_lengths = explicit_angles_and_lengths(prediction)
    def measurement(angles):
        values = []
        for row in range(len(angles)):
            if bool(eligible[row]):
                quantiles = torch.quantile(angles[row,fixed[row]], angles.new_tensor([.05,.95]),
                                           dim=0, interpolation='linear')
                excursion = quantiles[1]-quantiles[0]
                values.append(excursion[1]-excursion[0])
            else:
                values.append(angles[row].sum()*0)  # Excluded internal placeholder.
        return torch.stack(values).reshape(pairs,2)
    p, y = measurement(predicted_angles), measurement(target_angles).detach()
    if objective in ('paired_change', 'repaired_change'):
        error = (((p[:,1]-p[:,0])-(y[:,1]-y[:,0]))/180).square()
    else:
        assert objective == 'per_example_measurement'
        error = ((p-y)/180).square().mean(1)
    short = torch.relu((min_segment-predicted_lengths)/min_segment).square().mean((-1,-2))
    short = ((short*fixed).sum(1)/counts.clamp_min(1)).reshape(pairs,2).mean(1)
    supported_pairs = eligible.reshape(pairs,2).all(1)
    assert bool(supported_pairs.any()), 'The teaching reference needs supported pairs.'
    return (error+short)[supported_pairs].mean(), p, y, fixed

measurement_settings = dict(min_segment_px=float(saved['measurement']['min_segment_px']),
                            min_frames=int(saved['measurement']['min_frames']),
                            min_fraction=float(saved['measurement']['min_coverage']))
paired_prediction = predicted_px.reshape(-1,2,T,J,2)
paired_reference = torch.as_tensor(truth_px).reshape(-1,2,T,J,2)
paired_valid = valid.reshape(-1,2,T,J)
loss_terms = {}
for objective in ('paired_change', 'per_example_measurement'):
    value, predicted_A, reference_A, fixed_frames = explicit_measurement_loss(
        paired_prediction, paired_reference, paired_valid, objective,
        measurement_settings['min_segment_px'], measurement_settings['min_frames'], measurement_settings['min_fraction'])
    production_value, _ = measurement_loss(paired_prediction, paired_reference, paired_valid,
                                          objective=objective, **measurement_settings)
    torch.testing.assert_close(value, production_value)
    loss_terms[objective] = value
display(pd.DataFrame({'reference_baseline_A_deg': reference_A[:,0].detach().numpy(),
                      'reference_changed_A_deg': reference_A[:,1].detach().numpy(),
                      'reference_change_deg': (reference_A[:,1]-reference_A[:,0]).detach().numpy(),
                      'predicted_change_deg': (predicted_A[:,1]-predicted_A[:,0]).detach().numpy()}))
print('Training objective values:', {name: float(value.detach()) for name,value in loss_terms.items()})
"""),
        md(r"""
## Understand what pairing changes

The untrained readout above places the missing ankle at the normalization
origin. That can produce a large movement-measurement error even when the
average coordinate error is small; the deliberately missing observation is
retained in the example rather than discarded from the measurement.

If both endpoint measurements have the same five-degree offset, the paired
change can be correct while both endpoint measurements are inaccurate.
Per-example supervision penalizes those offsets directly. These objectives
therefore test different constraints, with coordinate supervision retained
for both.

Valid re-pairing changes which baseline and changed endpoints are compared,
then recomputes the reference difference for those new partners. Every
endpoint must retain its exposure count. In the full study, the pairing
sampler additionally enforces endpoint-role and condition strata, excludes
same-source-family partners, and audits the reference-change distribution.
Closed two- or three-family cycles give every compared objective the same
endpoint multiset **within each minibatch**. Experiment E examines the
saved pairing audit and those controls in detail.
"""),
        code(r"""
# These scalar examples explain the distinction; their values are not study results.
truth_A = torch.tensor([[2.,10.], [7.,18.]])
estimate_A = truth_A+5
paired_error = ((estimate_A[:,1]-estimate_A[:,0])-(truth_A[:,1]-truth_A[:,0])).square()
endpoint_error = (estimate_A-truth_A).square().mean(1)
assert torch.count_nonzero(paired_error) == 0 and torch.all(endpoint_error == 25)
print('Common five-degree offset:', {'change_error_squared': paired_error.tolist(),
                                    'endpoint_error_squared': endpoint_error.tolist()})

original_pairs = np.array([[0,1],[2,3]])
new_pairs = np.array([[0,3],[2,1]])
assert np.array_equal(np.sort(original_pairs.flatten()), np.sort(new_pairs.flatten()))
new_prediction = predicted_px[new_pairs.flatten()].reshape(-1,2,T,J,2)
new_reference = torch.as_tensor(truth_px)[new_pairs.flatten()].reshape(-1,2,T,J,2)
new_valid = valid[new_pairs.flatten()].reshape(-1,2,T,J)
new_loss, _, new_A, _ = explicit_measurement_loss(new_prediction, new_reference, new_valid,
    'repaired_change', measurement_settings['min_segment_px'], measurement_settings['min_frames'], measurement_settings['min_fraction'])
torch.testing.assert_close(new_loss, measurement_loss(new_prediction,new_reference,new_valid,
    objective='repaired_change', **measurement_settings)[0])
print('Recomputed reference changes for the new pairs:', (new_A[:,1]-new_A[:,0]).tolist())
"""),
        md(r"""
## Update the readout and verify that the encoder stays fixed

For the paired-change arm, the output loss is
$L=L_{\mathrm{coordinate}}+\lambda_\Delta L_\Delta$, including the short-limb
penalty inside the movement term. The saved default coefficient is one.
The following step shows the backward pass and checks every encoder tensor
before and after optimization. The same feature extractor will be used at
inference, where only observed coordinates, confidence, visibility and
timestamps are available.
"""),
        code(r"""
frozen_before = {name:value.detach().clone() for name,value in output_model.encoder.state_dict().items()}
readout_before = [parameter.detach().clone() for parameter in output_model.readout.parameters()]
output_parameters = list(output_model.readout.parameters())
output_optimizer = torch.optim.AdamW(output_parameters, lr=learning_rate,
                                     weight_decay=float(tc.get('weight_decay', .01)))
output_updates = int(tc['readout_updates'])
output_warmup = max(1, round(float(tc.get('warmup_fraction', .05))*output_updates))
for group in output_optimizer.param_groups:
    group['lr'] = learning_rate/output_warmup  # First warmup step of this phase.
total_output_loss = coordinate_term + float(tc.get('change_weight',1.))*loss_terms['paired_change']
output_optimizer.zero_grad(set_to_none=True)
total_output_loss.backward()
assert all(parameter.grad is None for parameter in output_model.encoder.parameters())
torch.nn.utils.clip_grad_norm_(output_parameters, float(tc.get('gradient_clip',1.)), error_if_nonfinite=True)
output_optimizer.step()
assert all(torch.equal(value, output_model.encoder.state_dict()[name]) for name,value in frozen_before.items())
assert any(not torch.equal(a,b) for a,b in zip(readout_before, output_model.readout.parameters()))
with torch.no_grad():
    after_update = output_model(output_inputs)
print({'scratch_output_loss_before_update': float(total_output_loss.detach()),
       'encoder_unchanged': True,
       'maximum_normalized_prediction_change': float((after_update-output_prediction.detach()).abs().max())})
"""),
        md(r"""
## Contrast frozen features with end-to-end training

Direct training updates both the encoder and readout. Because the final
readout layer starts at zero, the first backward pass gives the encoder zero
gradient; after that output layer changes, gradients can reach the encoder.
AdamW weight decay can still shrink encoder weights during that first step,
so a zero loss gradient does not mean its parameters remain unchanged.
Two short coordinate-only steps make this initialization behavior visible.
This demonstration has its own newly initialized model and optimizer.
"""),
        code(r"""
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(612)
    direct = RestorationModel('direct', small).cpu()
direct.requires_grad_(False)
direct.encoder.requires_grad_(True); direct.readout.requires_grad_(True)
direct_parameters = [parameter for parameter in direct.parameters() if parameter.requires_grad]
direct_optimizer = torch.optim.AdamW(direct_parameters, lr=learning_rate,
                                     weight_decay=float(tc.get('weight_decay',.01)))
direct_history = []
direct_updates = int(tc['end_to_end_updates'])
direct_warmup = max(1, round(float(tc.get('warmup_fraction',.05))*direct_updates))
for update in range(2):
    progress = (update-direct_warmup)/max(direct_updates-direct_warmup,1)
    lr = learning_rate*((update+1)/direct_warmup if update < direct_warmup else .5*(1+math.cos(math.pi*progress)))
    for group in direct_optimizer.param_groups:
        group['lr'] = lr
    estimate = direct(output_inputs)
    loss = coordinate_loss(estimate,output_target,valid)[0]
    direct_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    encoder_grad = sum(float(p.grad.square().sum()) for p in direct.encoder.parameters() if p.grad is not None)**.5
    torch.nn.utils.clip_grad_norm_(direct_parameters,float(tc.get('gradient_clip',1.)),error_if_nonfinite=True)
    direct_optimizer.step()
    direct_history.append({'update':update+1,'coordinate_MSE':float(loss.detach()),
                           'encoder_gradient_norm':encoder_grad})
assert direct_history[0]['encoder_gradient_norm'] == 0
assert direct_history[1]['encoder_gradient_norm'] > 0
display(pd.DataFrame(direct_history))
assert torch.equal(before_rng, torch.random.get_rng_state())
print('Two generated-batch steps explain gradient flow; they do not measure generalization.')
"""),
        md(r"""
## Launch the saved study

The calculations above have not changed the saved configuration, training
data, plan or checkpoints. Now the canonical launcher executes that plan.
The software fixture runs the full dependency graph with a small CPU budget.
A source study submits one CPU coordinator on HAIC, which dispatches up to
eight independent one-H100 allocations. Each allocation has its own attempt
directory and reservation against the study's compute budget.

For source mode, run this cell on HAIC with Slurm available. It submits jobs
and returns; closing the notebook does not stop the coordinator. The fixture
waits for its local run to finish. Completed phases are retained on resumption.
Keep the source release, frozen configuration and shared bundle unchanged
while jobs are active.
"""),
        code("study.launch()"),
        md(r"""
## Read progress, failures and allocation costs

Repeat the status cell while a source run is active, and inspect its worker
log if a phase fails. A saved checkpoint, scheduler completion and verified
evaluation are different stages of evidence.

GPU-hours equal allocated GPU count multiplied by allocation time, including
idle time and failed attempts. Queue time affects the deadline but does not
consume allocated GPU time. The coordinator counts parent allocations once,
without adding their `.batch` and `.extern` entries again. To stop a source
study, cancel the coordinator and its active worker job IDs shown in status.
"""),
        code(r"""
study.command('status')
for path in sorted((study.work/'logs').glob('*.out'))[-8:]:
    print(path)
print('Full study directory:', study.work)
"""),
        md(r"""
The completed run retains update histories, exact endpoint indices, pairing
audits, normalization fallbacks and model identities. Read those records with
the representation and movement diagnostics rather than treating a decreasing
training loss as sufficient evidence. Continue to
**05_evaluate_and_visualize.ipynb** after all required phases have completed.
"""),
    ]
