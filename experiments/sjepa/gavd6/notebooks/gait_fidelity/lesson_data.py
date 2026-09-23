"""Explicit data and masking lessons inserted after each notebook's setup cell.

The returned code is visible in the generated notebooks. Scientific examples
operate on copies in memory; canonical preparation/validation remain unchanged.
"""


def build_data_cells(md, code):
    return [
        *build_manifest_cells(md, code),
        md(r'''
## Prepare the shared data before inspecting its mathematics

An **estimated/reference pair** contains two trajectories of the same joints at
the same instants: the pose estimator supplies the input, while the projected
body-model joints supply the training target. A **movement pair** contains two
such examples with different movement states. We use the first pairing for pose
restoration and the second to ask whether restoration preserves a change in
movement. The cells below keep these two meanings separate.

Preparation remains the production operation because it includes licensed body
assets, rendering, extraction, and immutable provenance. The fixture runs here
on the CPU. For source data, run the printed Slurm command on HAIC and wait for
preparation to finish before executing the next cell. The remaining examples
read the prepared bundle and work on small copies; they do not change it.
'''),
        code('''
if study.fixture:
    study.command('prepare')
else:
    import shlex
    command = ['bash', str(study.root / 'slurm/gait-fidelity/run.sh'),
               'launch', str(study.work), '--prepare-only']
    print('Run on HAIC, then continue after preparation completes:')
    print(shlex.join(command))
'''),
        code('''
study.command('validate')
import numpy as np
import pandas as pd
from IPython.display import display
from gavd6_sjepa.research_directions.gait_fidelity.data import load_dataset
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import JOINTS

bundle = load_dataset(study.bundle_path())
records = pd.DataFrame(bundle.records)
config = study.artifact('config.json')
display(records.groupby(['split', 'extractor']).size().rename('track records').reset_index())
display(records[['split', 'canonical_person_id']].drop_duplicates()
        .groupby('split').size().rename('independent people'))
print('Evidence:', bundle.evidence_status)
print('All demonstrations below select training rows by metadata, before examining error.')
'''),
        md(r'''
## Follow one aligned example through its arrays

The batch dimension selects examples, the time dimension selects frames, and
the joint dimension follows the fixed body-12 order printed below. Input
coordinates are measured in pixels. A native confidence score comes from the
pose estimator; it is not assumed to be a calibrated probability. The boolean
`observed` array states whether a joint position was supplied. Missing input
positions remain missing rather than being interpolated.

References are stored separately. `valid` records reference availability and
`visible` records the synthetic visibility proxy. A joint may have a valid
reference while being hidden in the image. `eval_scale` is a reference-derived
box diagonal used for coordinate-error evaluation; it is never an input to the
restoration model. Person IDs, movement labels, camera IDs, and reference masks
are likewise absent from inference inputs.
'''),
        code('''
selected = records.index[
    records['split'].eq('train') & records['movement_state'].eq('baseline')
    & records['physical_state'].eq('original') & records['naming'].eq('correct')
    & records['observation'].eq('clear')
]
assert len(selected), 'Preparation must retain a clear, correctly named training baseline.'
row_id = int(selected[0])
raw = {key: values[[row_id]].copy() for key, values in bundle.inputs.items()}
reference = {key: values[[row_id]].copy() for key, values in bundle.targets.items()}
assert set(raw) == {'xy', 'confidence', 'observed', 'timestamps'}
assert set(reference) == {'xy', 'valid', 'visible', 'eval_scale'}
B, T, J, D = raw['xy'].shape
assert (B, J, D) == (1, 12, 2)
assert raw['observed'].shape == reference['valid'].shape == (B, T, J)
assert reference['xy'].shape == raw['xy'].shape
assert np.all(~reference['visible'] | reference['valid'])
assert np.isfinite(raw['xy'][raw['observed']]).all()
assert np.allclose(np.diff(raw['timestamps'], axis=1), 1 / config['data']['hz'])

display(pd.DataFrame({'joint index': range(J), 'joint name': JOINTS}))
display(pd.DataFrame([
    {'role': role, 'field': key, 'shape': str(value.shape), 'dtype': str(value.dtype)}
    for role, values in [('input', raw), ('reference', reference)]
    for key, value in values.items()
]))
print('Selected metadata:', records.loc[row_id, ['canonical_person_id', 'source_family_id',
      'movement_state', 'camera_id', 'observation', 'extractor']].to_dict())
print('Window span:', float(raw['timestamps'][0, -1] - raw['timestamps'][0, 0]), 'seconds')
'''),
        md(r'''
## See how AMASS supplies a reference in the camera image

For source data, AMASS animates a body model in three dimensions. Rendering and
reference projection use the same fixed camera. Write its camera-to-world
rotation as $R$ and position as $c$. For row-vector world points $X$, camera
coordinates are $X_c=(X-c)R$. This renderer looks along the camera's negative
$z$ axis, so positive depth is $d=-X_{c,z}$. With vertical field of view $v$,
image height $H$, and width $W$, the pixel projection is

$$f=\frac{H}{2\tan(v/2)},\qquad
u=f\frac{X_{c,x}}{d}+\frac{W-1}{2},\qquad
w=-f\frac{X_{c,y}}{d}+\frac{H-1}{2}.$$

The minus sign in the second coordinate makes image rows increase downward.
The half-pixel convention matches the renderer's array coordinates. The tiny
three-point example below verifies this projection against production code;
it does not replace an AMASS recording or demonstrate rendering realism.
Source preparation additionally rejects near-plane failures and clipped
physical variants, and freezes each camera across the paired movement states.
'''),
        code('''
from gavd6_sjepa.research_directions.synthetic_training.rendering import project_points

world = np.array([[0., 0., -3.], [.2, .4, -3.], [-.2, -.4, -3.]])
camera_pose = np.eye(4)  # illustrative camera at the origin, looking along -z
width, height, vertical_fov = 640, 480, np.deg2rad(50.)
camera = (world - camera_pose[:3, 3]) @ camera_pose[:3, :3]
depth = -camera[:, 2]
assert np.all(depth > .05)
focal = height / (2 * np.tan(vertical_fov / 2))
projected = np.column_stack([
    focal * camera[:, 0] / depth + (width - 1) / 2,
    -focal * camera[:, 1] / depth + (height - 1) / 2,
]).astype(np.float32)
production_xy, production_depth = project_points(world, camera_pose, width, height, vertical_fov)
np.testing.assert_allclose(projected, production_xy, rtol=0, atol=1e-5)
np.testing.assert_allclose(depth, production_depth)
display(pd.DataFrame(projected, columns=['horizontal pixel u', 'vertical pixel w']))
'''),
        md(r'''
The reference is the projected body model's joint center. Its anatomical
definition is an approximation to the estimator's corresponding landmark,
which can introduce systematic offsets. Alignment in time and camera does not
make this a direct measurement of anatomical truth. Real-video evaluation
requires independent references and its own landmark checks.

## Inspect the second kind of pair: a controlled movement change

A `pair_id` fixes the source window, person, camera, observation condition,
joint-naming condition, and extractor while allowing the movement state to
change. The source preparation composes a local right-knee rotation during
reference-supported swing frames; its requested edit in degrees is an input
to the simulator, **not** the resulting image-plane excursion difference.
The latter must be measured from the projected references in notebook 05.
The analytic software fixture only mimics these relationships for code tests.
'''),
        code('''
pair_rows = records.index[records['pair_id'].eq(records.loc[row_id, 'pair_id'])].to_numpy()
pair = records.loc[pair_rows]
for field in ['canonical_person_id', 'source_family_id', 'split', 'physical_state',
              'camera_id', 'observation', 'naming', 'extractor']:
    assert pair[field].nunique() == 1, f'Uncontrolled change in {field}'
for k in pair_rows:
    np.testing.assert_array_equal(bundle.inputs['timestamps'][k], raw['timestamps'][0])
display(pair[['movement_state', 'movement_magnitude', 'camera_id', 'observation', 'naming']])

# The exact duplicate is retained as a zero-change diagnostic, not another person.
unchanged_rows = pair.index[pair['movement_state'].eq('no_change')]
assert len(unchanged_rows) == 1
np.testing.assert_array_equal(bundle.targets['xy'][unchanged_rows[0]], reference['xy'][0])
print('The no-change reference is exactly equal to its baseline.')
'''),
        md(r'''
## Normalize using only the context that the encoder may see

Let $C$ contain the observed joint positions remaining after artificial query
masking. For each window, production computes a two-coordinate origin $o$ and
one positive scale $s$:

$$o=\operatorname{median}_{x\in C}x,\qquad
s=\lVert Q_{.95}(C)-Q_{.05}(C)\rVert_2,\qquad
\widetilde{x}=(x-o)/s.$$

The quantiles act independently on horizontal and vertical coordinates; their
difference forms a diagonal whose Euclidean length is the common scale.
Using one scale preserves angles and relative geometry. Both quantities are
constant throughout the window. If fewer than two context points remain, or
their span is degenerate, production uses $o=(0,0)$ and $s=1$ and counts the
fallback. Neither hidden values nor references determine this fallback.

The next cell shows the complete computation. We hide the first four-frame
patch of both knees as a deterministic teaching example. Notebook 02 replaces
this illustrative mask with the study's stochastic sampler.
'''),
        code('''
patch_size = int(config['model']['patch_size'])
assert T % patch_size == 0
hidden = np.zeros((B, T // patch_size, J), dtype=bool)
hidden[:, 0, [8, 9]] = True
hidden_frames = np.repeat(hidden, patch_size, axis=1)

def normalize_from_context(inputs, hidden_frames):
    xy = np.asarray(inputs['xy'], dtype=np.float32)
    context = inputs['observed'] & ~hidden_frames
    origins, scales, fallbacks = [], [], 0
    for points, keep in zip(xy, context):
        retained = points[keep]
        if len(retained) >= 2:
            origin = np.median(retained, axis=0)
            span = np.quantile(retained, .95, axis=0) - np.quantile(retained, .05, axis=0)
            scale = float(np.linalg.norm(span))
        else:
            origin, scale = np.zeros(2), 0.
        if not np.isfinite(scale) or scale < 1e-6:
            origin, scale = np.zeros(2), 1.
            fallbacks += 1
        origins.append(origin)
        scales.append(scale)
    origins = np.asarray(origins, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    normalized = {key: values.copy() for key, values in inputs.items()}
    normalized['xy'] = (xy - origins[:, None, None]) / scales[:, None, None, None]
    normalized['confidence'] = np.where(np.isfinite(inputs['confidence']),
                                        inputs['confidence'], 0).astype(np.float32)
    return normalized, origins, scales, fallbacks

normalized, origin, scale, fallback_count = normalize_from_context(raw, hidden_frames)
from gavd6_sjepa.research_directions.gait_fidelity.training import normalize_batch
actual, actual_origin, actual_scale, actual_fallbacks = normalize_batch(raw, hidden, patch_size=patch_size)
for key in normalized:
    np.testing.assert_allclose(normalized[key], actual[key], rtol=0, atol=0, equal_nan=True)
np.testing.assert_array_equal(origin, actual_origin)
np.testing.assert_array_equal(scale, actual_scale)
assert fallback_count == actual_fallbacks
print({'origin_px': origin.tolist(), 'scale_px': scale.tolist(), 'fallbacks': fallback_count})
'''),
        md(r'''
### Check the leakage boundary and restore pixel geometry

Changing a hidden input must leave the normalization and visible context
unchanged. We test that property by moving the hidden coordinates a million
pixels, then rebuilding the context channels. The normalized hidden values
may differ in memory, but token construction replaces them with zero before
the encoder sees them. References are transformed by the **same input-derived**
$o,s$ for the training target; references never provide their own normalization.

The model predicts normalized positions. Applying
$\widehat{x}=s\widehat{\widetilde{x}}+o$ returns them to pixels before gait
measurements and the minimum limb-length check. This cell verifies the inverse
on the original observations, without claiming an improvement from a model.
'''),
        code('''
perturbed = {key: values.copy() for key, values in raw.items()}
perturbed['xy'][hidden_frames & raw['observed']] += 1_000_000
altered, altered_origin, altered_scale, _ = normalize_from_context(perturbed, hidden_frames)
np.testing.assert_array_equal(origin, altered_origin)
np.testing.assert_array_equal(scale, altered_scale)
context = raw['observed'] & ~hidden_frames
safe_context = np.where(context[..., None], normalized['xy'], 0.)
altered_context = np.where(context[..., None], altered['xy'], 0.)
np.testing.assert_array_equal(safe_context, altered_context)

target_normalized = (reference['xy'] - origin[:, None, None]) / scale[:, None, None, None]
restored_input_px = normalized['xy'] * scale[:, None, None, None] + origin[:, None, None]
restored_reference_px = target_normalized * scale[:, None, None, None] + origin[:, None, None]
np.testing.assert_allclose(restored_input_px[raw['observed']], raw['xy'][raw['observed']],
                           rtol=2e-6, atol=1e-4)
np.testing.assert_allclose(restored_reference_px[reference['valid']],
                           reference['xy'][reference['valid']], rtol=2e-6, atol=1e-4)

empty = {key: values.copy() for key, values in raw.items()}
empty['observed'][:] = False
empty['xy'][:] = np.nan
_, empty_origin, empty_scale, empty_count = normalize_from_context(empty, hidden_frames)
assert empty_count == B
np.testing.assert_array_equal(empty_origin, np.zeros((B, 2)))
np.testing.assert_array_equal(empty_scale, np.ones(B))
_, production_empty_origin, production_empty_scale, production_empty_count = normalize_batch(
    empty, hidden, patch_size=patch_size)
np.testing.assert_array_equal(empty_origin, production_empty_origin)
np.testing.assert_array_equal(empty_scale, production_empty_scale)
assert production_empty_count == empty_count
print('Hidden-coordinate perturbation, pixel round trip, and empty-context fallback passed.')
'''),
        md('''
## Inspect positions and trajectories before fitting

The following picture uses the selected training row's middle frame and its
complete ankle trajectory. The choice is made from metadata and frame index,
without ranking prediction errors. Blue points/curves are references; orange
ones are the estimator inputs. The connecting lines describe body-12 geometry.
These two panels are data checks and contain no trained restoration prediction.
The extra previews below may include retained evaluation figures when reopening
an already completed study; their labels identify what they summarize.
'''),
        code('''
import matplotlib.pyplot as plt
from io import BytesIO
from IPython.display import Image
from gavd6_sjepa.research_directions.gait_fidelity.masking import EDGES

frame = T // 2
seconds = raw['timestamps'][0] - raw['timestamps'][0, 0]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
for points, color, label in [(reference['xy'][0, frame], '#2468a0', 'Reference'),
                             (raw['xy'][0, frame], '#cb7022', 'Estimator input')]:
    for a, b in EDGES:
        axes[0].plot(points[[a, b], 0], points[[a, b], 1], color=color, alpha=.65)
    axes[0].scatter(points[:, 0], points[:, 1], s=22, color=color, label=label)
axes[0].set(xlabel='Horizontal position (pixels)', ylabel='Vertical position (pixels)',
            title=f'Middle frame: {seconds[frame]:.2f} seconds')
axes[0].invert_yaxis()
axes[0].set_aspect('equal', adjustable='datalim')
axes[0].legend(loc='best', fontsize=9)
axes[1].plot(seconds, reference['xy'][0, :, 11, 0], color='#2468a0', label='Reference')
axes[1].plot(seconds, raw['xy'][0, :, 11, 0], color='#cb7022', label='Estimator input')
axes[1].set(xlabel='Time since window start (s)', ylabel='Horizontal position (pixels)',
            title='Right ankle through time')
axes[1].legend(loc='best', fontsize=9)
fig.suptitle(('Software fixture' if study.fixture else 'Selected training example') +
             ' — aligned reference and input')
png = BytesIO()
fig.savefig(png, format='png', dpi=120, bbox_inches='tight')
display(Image(data=png.getvalue()))
plt.close(fig)

for path in dict.fromkeys(sorted(study.work.rglob('*viewer*.html')) +
                          sorted(study.work.rglob('*gallery*.html'))):
    print('Interactive prepared-data viewer:', path)
preview_images(study)
'''),
        md('''
The production viewer also exposes camera, movement, visibility, and naming
conditions. Check every declared condition and retained failure before fitting.
All variants of a person remain in that person's original split; render counts
do not increase the number of independent people. Full-manifest selection
retains all relevant audited candidates under the saved duration and cohort
rules; actual admitted counts come from this run's preparation report.
Original test people remain locked while development choices are made.
Continue to **02_masking_and_controls**
to turn these arrays into query masks and model tokens.
'''),
    ]


def build_manifest_cells(md, code):
    return [md(r'''
## Establish the population before rendering

AMASS stores motions and body-model parameters, which supply paired projected
references after rendering. GAVD stores real videos with sequence-level gait
and camera annotations; these labels support a separate observational analysis
and cannot serve as ground-truth joint coordinates.

The AMASS cohort joins eligible motion paths to audited person identities and
the original person splits. A subject folder is not necessarily a distinct
person: activity folders can be aliases. A motion is eligible for a window
only if its duration is at least $(T-1)/h$, where $T$ is the sample count and
$h$ is the requested sampling rate. At 128 frames and 25 Hz this is 5.08 s.
The saved plan records every candidate and exclusion. Filenames identifying
walking are a reproducible candidate screen; they are not a clinical diagnosis
or a substitute for inspecting the actual movement.
'''), code('''
import numpy as np
import pandas as pd
from IPython.display import display
config = study.artifact('config.json')
minimum_span = (config['data']['samples'] - 1) / config['data']['hz']
print('Required window span (seconds):', minimum_span)
if study.fixture:
    print('Generated software fixture: no source inventory or clinical evidence.')
elif config['data'].get('source_selection', 'legacy_roster') == 'full_manifest':
    manifest_dir = Path(config['preparation']['manifest_dir'])
    inventory = pd.read_csv(manifest_dir / 'amass_raw_inventory_eligible.csv')
    registry = pd.read_csv(manifest_dir / 'amass_subject_registry.csv', keep_default_na=False)
    splits = pd.read_csv(manifest_dir / 'amass_subject_splits.csv', keep_default_na=False)
    person_splits = splits[['identity', 'split']].drop_duplicates()
    assert person_splits.groupby('identity')['split'].nunique().max() == 1
    duration = (inventory['num_frames'] - 1) / inventory['mocap_framerate']
    display(inventory.assign(long_enough=duration >= minimum_span)
            .groupby('source_dataset').agg(motions=('relative_path', 'size'),
                                          duration_candidates=('long_enough', 'sum')))
    display(person_splits.groupby('split').size().rename('audited people'))
    cohort_path = Path(config['cohort']['plan_path'])
    cohort = json.loads(cohort_path.read_text())
    print('Frozen cohort selection:', cohort_path)
    print('Read the cohort summary and exclusion files before interpreting candidate counts.')
    print(json.dumps(cohort.get('summary', {}), indent=2))
else:
    print('Explicit legacy roster:', config['source_bundle'])
'''), md(r'''
## Keep real-video groups intact

GAVD sequences from the same recording can overlap or show the same person
from different directions. Partition complete video groups, and merge groups
when a reviewed identity map establishes that they share a person. Unknown
identity is recorded rather than treated as proof that videos show different
people. The fields `dataset_annotation` and `gait_pattern_annotation` describe
different annotation levels; use the latter for the declared gait-pattern task.
Camera labels such as `left side` describe view direction, not affected side.

The optional GAVD plan below inventories the complete manifest and local video
availability. Its planned test groups stay locked. Run `gavd-plan`, then
`launch --gavd-only` from the HAIC guide to extract train/development tracks
through the same GPU queue used for AMASS. Run these stages sequentially; a
second stage request while a coordinator is active does not enqueue that stage.
'''), code('''
gavd_plan_path = study.work / 'gavd/plan.json'
if gavd_plan_path.exists():
    real_plan = json.loads(gavd_plan_path.read_text())
    print('GAVD plan:', gavd_plan_path)
    print(json.dumps(real_plan.get('summary', {}), indent=2))
elif not study.fixture:
    import shlex
    print(shlex.join(['bash', str(study.root / 'slurm/gait-fidelity/run.sh'),
                     'gavd-plan', str(study.work)]))
    print('This CPU inventory does not claim that real-video reference coordinates exist.')
else:
    print('GAVD data are not synthesized by the teaching fixture.')
''')]


def build_mask_cells(md, code):
    return [
        md(r'''
## Define a token before defining a mask

For the body-12 model, one token contains **four adjacent frames of one joint**.
For a window of $T$ frames, patch size $p=4$, and $J=12$ joints, there are
$S=T/p$ patches and $N=SJ$ tokens. Token $(s,j)$ is flattened at index $sJ+j$.
Source windows have 128 frames, hence 384 tokens; the smaller software fixture
uses the dimensions saved in its configuration.

A naturally unobserved joint is different from an artificially hidden query.
We may add an artificial mask only when a patch contains at least one observed
frame. Naturally missing frames also create pretraining queries, as notebook
04 shows. The
model still retains output locations for all patches and joints, including
those with no observations. A reference-valid mask never determines which
input locations are eligible for masking.

For the worked example, we prefer a correctly named, occluded training
baseline with at least two observed tokens, using manifest order to break ties.
If none qualifies, we use the first training row with that input support. We
report sparse candidates passed over for the picture and keep them in the
study; explicit sparse-grid tests below still verify their sampler behavior.
'''),
        code('''
import numpy as np
import pandas as pd
from IPython.display import display
from gavd6_sjepa.research_directions.gait_fidelity.data import load_dataset
from gavd6_sjepa.research_directions.gait_fidelity.masking import (
    sample_mask, patch_support, POLICIES, REGIONS, EDGES, coverage_audit, audit_matching)
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import JOINTS

bundle = load_dataset(study.bundle_path())
records = pd.DataFrame(bundle.records)
config = study.artifact('config.json')
# This metadata rule deliberately includes natural missingness when available.
chosen = records.index[records['split'].eq('train') & records['movement_state'].eq('baseline')
                       & records['naming'].eq('correct') & records['observation'].eq('occluded')]
p = int(config['model']['patch_size'])
candidates = list(dict.fromkeys([*chosen, *records.index[records['split'].eq('train')]]))
row_id, sparse_candidates = None, 0
for candidate in candidates:
    candidate_observed = bundle.inputs['observed'][candidate]
    candidate_tokens = candidate_observed.reshape(-1, p, 12).any(axis=1)
    if candidate_tokens.sum() >= 2:
        row_id = int(candidate)
        break
    sparse_candidates += 1
assert row_id is not None, 'No training row has two observed tokens; inspect extraction before continuing.'
print('Sparse candidates passed over for this demonstration:', sparse_candidates,
      '(all remain in the study)')
raw = {key: value[[row_id]].copy() for key, value in bundle.inputs.items()}
observed = raw['observed']
B, T, J = observed.shape
S = T // p
assert T % p == 0 and J == 12
eligible = observed.reshape(B, S, p, J).any(axis=2)
np.testing.assert_array_equal(eligible, patch_support(observed, p))
fraction = float(config['training']['mask_fraction'])
available_count = eligible.sum(axis=(1, 2))
budgets = np.array([min(max(1, round(int(n) * fraction)), int(n) - 1) if n > 1 else 0
                    for n in available_count])
display(pd.DataFrame({'frames': [T], 'patches': [S], 'joints': [J], 'grid tokens': [S * J],
                      'eligible tokens': available_count, 'hidden budget': budgets}))
print('Natural missing joint-frame values:', int((~observed).sum()))
'''),
        md(r'''
## What the anatomical graph controls

The implementation samples from the connected regions listed below. Each
region is a limb, part of a limb, or a connected torso region in the body-12
graph. It does not grow an unrestricted random subgraph and does not weight
clinically interesting joints more heavily. Overlapping region definitions
can still give some joints a higher selection probability, which we audit.

For each sampled region, draw a duration from 1 through $S/2$ patches and a
uniform start patch. Time wraps cyclically at the window boundary to avoid
automatically under-sampling the edges. The final region can be truncated
when the exact token budget is reached. A wrapped interval appears as two
separate pieces in the model's finite window.
'''),
        code('''
neighbors = {joint: set() for joint in range(J)}
for a, b in EDGES:
    neighbors[a].add(b)
    neighbors[b].add(a)

def connected(region):
    reached, pending = set(), [region[0]]
    while pending:
        joint = pending.pop()
        if joint in reached:
            continue
        reached.add(joint)
        pending.extend(neighbors[joint].intersection(region) - reached)
    return reached == set(region)

assert all(connected(region) for region in REGIONS)
display(pd.DataFrame([{'region': k, 'joints': ', '.join(JOINTS[j] for j in region),
                       'connected': connected(region)} for k, region in enumerate(REGIONS)]))
'''),
        md(r'''
## Implement the masking policies in ordinary NumPy

The sampler below shows the complete decisions applied to a single example.
Uniform-token masking samples eligible positions without replacement. Time
blocks visit consecutive patches, using a shuffled joint order if the final
patch only partly fits the budget. Graph-time masking places anatomical
regions over cyclic intervals. Its two controls retain the region-size rule:
one relabels the graph by a fixed permutation, while the other draws random
joints for every interval.

Every policy hides the same number of eligible tokens and leaves at least one
context token when two or more exist. A bounded retry loop plus a declared
uniform fill handles sparse observations. The code uses its own NumPy
generator, so these examples cannot advance a training job's random state.
'''),
        code('''
def explicit_mask(available, policy, *, seed, fraction=.5, topology_seed=271828):
    rng = np.random.default_rng(seed)
    available = np.asarray(available, dtype=bool)
    blocks, joints = available.shape
    n = int(available.sum())
    budget = min(max(1, round(n * fraction)), n - 1) if n > 1 else 0
    hidden = np.zeros_like(available)
    trace = []
    if budget == 0:
        return hidden, trace
    if policy == 'uniform_tokens':
        hidden.flat[rng.choice(np.flatnonzero(available), budget, replace=False)] = True
    elif policy == 'time_blocks':
        joint_order = rng.permutation(joints)
        start = int(rng.integers(blocks))
        order = [(int((start + k) % blocks), int(j))
                 for k in range(blocks) for j in joint_order if available[(start + k) % blocks, j]]
        for time_patch, joint in order[:budget]:
            hidden[time_patch, joint] = True
    else:
        assert policy in ('graph_time', 'shuffled_topology', 'random_joint_intervals')
        topology = np.random.default_rng(topology_seed).permutation(joints)
        maximum_duration, attempts = max(1, blocks // 2), 0
        while hidden.sum() < budget and attempts < 100 * blocks:
            attempts += 1
            region = np.asarray(REGIONS[int(rng.integers(len(REGIONS)))])
            duration = int(rng.integers(1, maximum_duration + 1))
            start = int(rng.integers(blocks))
            if policy == 'shuffled_topology':
                region = topology[region]
            elif policy == 'random_joint_intervals':
                region = rng.choice(joints, len(region), replace=False)
            before = int(hidden.sum())
            for offset in range(duration):
                for joint in rng.permutation(region):
                    time_patch = (start + offset) % blocks
                    if available[time_patch, joint] and not hidden[time_patch, joint]:
                        hidden[time_patch, joint] = True
                        if hidden.sum() == budget:
                            break
                if hidden.sum() == budget:
                    break
            trace.append({'region': tuple(int(j) for j in region), 'start patch': start,
                          'requested duration': duration, 'new hidden tokens': int(hidden.sum()) - before})
        if hidden.sum() < budget:
            remaining = np.flatnonzero(available & ~hidden)
            hidden.flat[rng.choice(remaining, budget - int(hidden.sum()), replace=False)] = True
    assert int(hidden.sum()) == budget
    assert not (hidden & ~available).any()
    assert n <= 1 or (available & ~hidden).any()
    return hidden, trace

manual, graph_trace = explicit_mask(eligible[0], 'graph_time', seed=17, fraction=fraction)
display(pd.DataFrame(graph_trace).head(12))
print('Sampling trace above shows interval requests; overlapping requests can add fewer tokens.')
'''),
        md('''
### Verify exact agreement, including sparse inputs

Matching only the hidden-token count would miss a joint-order or random-draw
error. The following checks compare every mask bit against the production
sampler for all five policies, several seeds, the selected observations, an
entirely observed grid, an empty grid, and a grid with one eligible token.
The empty and one-token cases intentionally add no artificial mask.
'''),
        code('''
one_token = np.zeros_like(observed)
one_token[:, 0, 0] = True
test_observations = [observed, np.ones_like(observed), np.zeros_like(observed), one_token]
checked = 0
for test_observed in test_observations:
    test_eligible = test_observed.reshape(B, S, p, J).any(axis=2)
    for policy in POLICIES:
        for seed in (0, 17, 29):
            inline, _ = explicit_mask(test_eligible[0], policy, seed=seed, fraction=fraction)
            actual, receipt = sample_mask(test_observed, policy, rng=np.random.default_rng(seed),
                fraction=fraction, patch_size=p, return_receipt=True)
            np.testing.assert_array_equal(inline, actual[0])
            assert int(inline.sum()) == receipt['hidden_tokens'][0]
            checked += 1
print(f'{checked} complete mask arrays match production bit for bit.')
'''),
        md(r'''
## Construct the five channels and pack the tokens

For each frame and joint, the encoder receives
$[\widetilde{x},\widetilde{y},c,a,\tau]$: normalized position, native confidence,
context availability, and seconds since the window starts. Artificially
hidden positions and scores become zero. Naturally missing coordinates also
become zero, and availability distinguishes these placeholders from valid
coordinates at zero. Native finite scores retain production's value even
when a location is naturally unobserved; missing scores are adapted to zero.

Grouping four frames produces a vector of $4\times5=20$ numbers for each
patch/joint. A learned linear projection maps this vector into a feature
embedding in notebook 03. Time and joint positions are then added, allowing
the model to distinguish identical numerical values at different locations.
'''),
        code('''
from gavd6_sjepa.research_directions.gait_fidelity.training import normalize_batch

hidden, _ = sample_mask(observed, 'graph_time', rng=np.random.default_rng(17),
                       fraction=fraction, patch_size=p, return_receipt=True)
normalized, origin, scale, _ = normalize_batch(raw, hidden, patch_size=p)
artificial = np.repeat(hidden, p, axis=1)
usable = observed & ~artificial
safe_xy = np.where(usable[..., None], normalized['xy'], 0.)
safe_confidence = np.where(~artificial, normalized['confidence'], 0.)
seconds = normalized['timestamps'] - normalized['timestamps'][:, :1]
clock = np.broadcast_to(seconds[:, :, None, None], (B, T, J, 1))
channels = np.concatenate([safe_xy, safe_confidence[..., None],
                           usable[..., None].astype(np.float32), clock], axis=-1)
patches = channels.reshape(B, S, p, J, 5).transpose(0, 1, 3, 2, 4).reshape(B, S * J, p * 5)
context_support = usable.reshape(B, S, p, J).any(axis=2).reshape(B, S * J)
assert channels.shape == (B, T, J, 5)
assert patches.shape == (B, S * J, p * 5)
assert np.isfinite(patches).all()
assert not context_support.reshape(B, S, J)[hidden].any()

# Check the indexing with the corresponding unflattened slice.
supported_slots = np.flatnonzero(context_support[0])
assert len(supported_slots), 'This demonstration requires at least one context token.'
token_index = int(supported_slots[0])
time_patch, joint = divmod(token_index, J)
expected = channels[0, time_patch * p:(time_patch + 1) * p, joint].reshape(-1)
np.testing.assert_array_equal(patches[0, token_index], expected)
display(pd.DataFrame(patches[0, token_index].reshape(p, 5),
                     columns=['normalized x', 'normalized y', 'native score', 'context available', 'seconds']))
print({'token_index': token_index, 'patch': time_patch, 'joint': JOINTS[joint],
       'packed_shape': patches.shape, 'context_tokens': int(context_support.sum())})
'''),
        md('''
## Compare the visible masks and their coverage

Orange cells are artificially hidden queries, blue cells retain observed
context, and gray cells have no observed frame in the patch. The same input
and seed are used for all policies, so the exact hiding counts are directly
comparable. The patterns need not have equal duration or per-joint frequency.
'''),
        code('''
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from io import BytesIO
from IPython.display import Image

fig, axes = plt.subplots(len(POLICIES), 1, figsize=(12, 13.5), constrained_layout=True)
colormap = ListedColormap(['#dfdfdf', '#c9e4f6', '#e89045'])
mask_rows = []
for ax, policy in zip(axes, POLICIES):
    policy_mask, receipt = sample_mask(observed, policy, rng=np.random.default_rng(17),
        fraction=fraction, patch_size=p, return_receipt=True)
    state = eligible[0].astype(int)
    state[policy_mask[0]] = 2
    ax.imshow(state.T, aspect='auto', interpolation='nearest', cmap=colormap, vmin=0, vmax=2)
    ax.set_yticks(range(J), JOINTS, fontsize=8)
    ax.set(title=policy.replace('_', ' '), xlabel=f'Temporal patch ({p} frames each)')
    mask_rows.append({'policy': policy, 'hidden tokens': receipt['hidden_tokens'][0],
                      'context tokens': receipt['context_tokens'][0],
                      'sparse-fill tokens': receipt['sparse_uniform_fill_tokens']})
axes[0].legend(handles=[Patch(color=color, label=label) for color, label in
                      zip(colormap.colors, ['Naturally unavailable', 'Visible context', 'Hidden query'])],
               loc='upper center', bbox_to_anchor=(.5, 1.47), ncol=3, fontsize=9)
fig.suptitle(('Software fixture' if study.fixture else 'Selected training example') +
             ' — equal budgets, different query patterns', fontsize=13)
png = BytesIO()
fig.savefig(png, format='png', dpi=120, bbox_inches='tight')
display(Image(data=png.getvalue()))
plt.close(fig)
display(pd.DataFrame(mask_rows))
'''),
        md(r'''
For a bank of $K$ fresh masks, the empirical hiding probability at patch/joint
$(s,j)$ is the number of eligible examples hidden there divided by the number
of eligible examples drawn. Counting natural absence as hiding would corrupt
this denominator. Coverage is a property of repeated draws: a joint can be
hidden throughout one window while still serving as context in other draws.
'''),
        code('''
draws, demo_seed = 128, 17
mask_rng = np.random.default_rng(demo_seed)
bank = np.stack([sample_mask(observed, 'graph_time', rng=mask_rng, fraction=fraction, patch_size=p)
                 for _ in range(draws)])
denominator = eligible.sum(axis=0) * draws
hidden_count = bank.sum(axis=(0, 1))
probability = np.divide(hidden_count, denominator,
    out=np.full_like(hidden_count, np.nan, dtype=float), where=denominator > 0)
audit = coverage_audit(observed, draws=draws, seed=demo_seed, policy='graph_time',
                       fraction=fraction, patch_size=p)
np.testing.assert_allclose(probability, audit['masked_probability_by_patch_joint'], equal_nan=True)
joint_denominator = denominator.sum(axis=0)
joint_rate = np.divide(hidden_count.sum(axis=0), joint_denominator,
                      out=np.full(J, np.nan), where=joint_denominator > 0)
display(pd.DataFrame({'joint': JOINTS, 'eligible-token hiding probability': joint_rate,
                     'draws with context somewhere': audit['context_somewhere_by_joint']}))
print('Always-hidden eligible slots:', audit['always_hidden_eligible_slots'])
print('Never-hidden eligible slots:', audit['never_hidden_eligible_slots'])

matching = audit_matching(observed, draws=128, seed=17, fraction=fraction, patch_size=p)
display(pd.DataFrame(matching['comparisons']).T)
print(matching['claim_limit'])
'''),
        md('''
The displayed audit is a teaching check on one metadata-selected training
example. The full prepared-run audit retains its own population and settings.
Equal token budgets alone do not isolate anatomical connectivity: graph
relabeling can change joint exposure, and overlapping intervals can change
realized run lengths. Failed matching tolerances remain part of the report
and limit the interpretation of a winning mask; they do not justify removing
an inconvenient control.

The stochastic query mask is used during pretraining. Frozen-readout training
and final restoration receive ordinary observed inputs, without an extra
artificial query mask. Continue to **03_experiment_matrix** for the explicit
encoder, feature predictor, coordinate readout, and registered comparisons.
'''),
    ]
