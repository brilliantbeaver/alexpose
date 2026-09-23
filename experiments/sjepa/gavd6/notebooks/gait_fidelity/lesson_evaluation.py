"""Visible metric calculations and evidence checks for tutorials 05 and 06.

Cells are inserted after the shared setup. Demonstrations read retained arrays;
only the existing evaluate/report/verify commands operate on study artifacts.
"""


def build_evaluation_cells(md, code):
    return [
        md(r"""
## 1 · Reconstruct the canonical tables, then explain one actual source family

The evaluation command reads the saved predictions from every completed model.
It never fits a model or chooses a favorable checkpoint. We keep that command
as the authoritative batch operation, then independently calculate its main
measurements below. The code cells use one source family chosen by its metadata,
so their examples remain readable without selecting a favorable result.

A **source family** is one recorded movement interval and all of the image and
movement conditions derived from it. Its many rows do not represent independent
people. Fixture arrays exercise the same calculations but remain software tests.
"""),
        code("if study.fixture:\n    study.command('evaluate')\nelse:\n    assert (study.work / 'evaluation/summary.json').is_file(), 'Wait for the Slurm coordinator to finish evaluation.'\nstudy.command('report')"),
        code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
from io import BytesIO
from gavd6_sjepa.research_directions.gait_fidelity.data import load_dataset, TrackBundle

config = study.artifact('config.json')
plan = study.artifact('plan.json')
ledger = study.artifact('ledger.json')
coverage = pd.read_csv(study.artifact('evaluation/coverage.csv'))
bundle = load_dataset(study.bundle_path())
dev = bundle.subset('development')

# Selection uses only the prespecified method, first seed, and sorted metadata.
method = config['evaluation']['primary_candidate']
seed = config['seeds'][0]
phase = next(p for p in plan['phases']
             if p['phase'] != 'pretrain' and p['seed'] == seed
             and p['recipe']['recipe_id'] == method)
family_id = sorted({r['source_family_id'] for r in dev.records})[0]
family_rows = np.array([i for i, r in enumerate(dev.records)
                        if r['source_family_id'] == family_id])
family = TrackBundle(
    {k: v[family_rows] for k, v in dev.inputs.items()},
    {k: v[family_rows] for k, v in dev.targets.items()},
    [dev.records[i] for i in family_rows], dev.evidence_status, dev.provenance)
from gavd6_sjepa.research_directions.gait_fidelity.evaluation import load_predictions
saved_xy, saved_indices = load_predictions(ledger['completed'][phase['phase_id']]['result'])
expected_indices = np.flatnonzero([r['split'] == 'development' for r in bundle.records])
np.testing.assert_array_equal(saved_indices, expected_indices)
predicted_xy = saved_xy[family_rows].astype(float)
reference_xy = family.targets['xy'].astype(float)
saved_chunks = []
for chunk in pd.read_csv(study.artifact('evaluation/per-window.csv'), chunksize=10000):
    selected = chunk.method.eq(method) & chunk.seed.eq(seed) & chunk.source_family_id.eq(family_id)
    if selected.any(): saved_chunks.append(chunk.loc[selected])
saved_family = pd.concat(saved_chunks, ignore_index=True)
assert len(saved_family) == len(family.records)
identity = ['person_id', 'motion_id', 'window_id', 'variant', 'extractor']
assert saved_family[identity].astype(str).equals(
    pd.DataFrame(family.records)[identity].astype(str))
print('Method:', method, '| seed:', seed, '| family:', family_id)
print('Array shape [conditions, frames, joints, xy]:', predicted_xy.shape)
display(coverage.loc[coverage.method.eq(method)])
"""),
        md(r"""
## 2 · Calculate a projected knee angle from three points

The body-12 order puts left/right hips at indices 6/7, knees at 8/9, and ankles
at 10/11. For each knee, form vectors pointing from the knee to the hip and
ankle, then measure their included angle:

\[
u=h-k,\quad v=a-k,\qquad
\theta=\frac{180}{\pi}\arccos\!\left[
\operatorname{clip}\!\left(\frac{u\cdot v}{\|u\|\,\|v\|},-1,1\right)\right].
\]

This is an angle **in the original image**, with a straight projected leg near
180 degrees. It depends on the camera view and is not a clinical 3D joint angle.
Clipping prevents round-off from moving a valid cosine outside its domain.
Missing joints or segments shorter than the saved pixel threshold cannot supply
a usable angle. The training lesson uses the equivalent `atan2(abs(cross), dot)`
form; the evaluator uses `arccos` as an independent numerical implementation.
"""),
        code(r"""
def projected_knee_angles(xy, minimum_length):
    # Each vector has shape [conditions, frames, left/right, xy].
    hips, knees, ankles = xy[..., 6:8, :], xy[..., 8:10, :], xy[..., 10:12, :]
    u, v = hips - knees, ankles - knees
    u_length = np.linalg.norm(u, axis=-1)
    v_length = np.linalg.norm(v, axis=-1)
    usable = (np.isfinite(u).all(-1) & np.isfinite(v).all(-1)
              & (u_length >= minimum_length) & (v_length >= minimum_length))
    cosine = np.divide((u * v).sum(-1), u_length * v_length,
                       out=np.zeros_like(u_length), where=usable)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return np.where(usable, angle, np.nan), usable

minimum_length = config['measurement']['min_segment_px']
reference_angles, reference_geometry = projected_knee_angles(reference_xy, minimum_length)
predicted_angles, predicted_geometry = projected_knee_angles(predicted_xy, minimum_length)

from gavd6_sjepa.research_directions.gait_fidelity.evaluation import angular_waveform
canonical_angles, canonical_geometry = angular_waveform(reference_xy, minimum_length)
np.testing.assert_allclose(reference_angles, canonical_angles, equal_nan=True)
np.testing.assert_array_equal(reference_geometry, canonical_geometry)
print('Angle tensor [conditions, frames, left/right]:', reference_angles.shape)
"""),
        md(r"""
## 3 · Fix the scoring timestamps using references alone

The comparison must not let a method discard frames where its predictions are
bad. A reference frame is usable only when both legs have valid hip, knee and
ankle coordinates and sufficiently long projected segments. We intersect those
reference frames across **every condition in the source family**, including
the movement levels and cameras. Every method then receives the same fixed set
\(S\). Reference validity is independent of image visibility: modeled synthetic
joints may have usable references even when an occluder hides them in the image.

A family is eligible if \(|S|\) meets both the minimum frame count and minimum
fraction of the interval. Prediction failures cannot change eligibility.
"""),
        code(r"""
valid = family.targets['valid']
reference_ok_per_leg = (valid[..., 6:8] & valid[..., 8:10]
                        & valid[..., 10:12] & reference_geometry)
support_by_condition = reference_ok_per_leg.all(-1)
times = family.inputs['timestamps']
np.testing.assert_allclose(times, np.broadcast_to(times[:1], times.shape), atol=1e-6, rtol=0)
common_support = np.logical_and.reduce(support_by_condition, axis=0)
fixed_support = np.broadcast_to(common_support, support_by_condition.shape)
eligible = (int(common_support.sum()) >= config['measurement']['min_frames']
            and float(common_support.mean()) >= config['measurement']['min_coverage'])

from gavd6_sjepa.research_directions.gait_fidelity.evaluation import common_reference_support
_, canonical_support = common_reference_support(family, config)
np.testing.assert_array_equal(fixed_support, canonical_support)
print(f'Reference support: {common_support.sum()} / {len(common_support)} frames; eligible={eligible}')
print('Required:', config['measurement']['min_frames'], 'frames and',
      config['measurement']['min_coverage'], 'coverage')
"""),
        md(r"""
## 4 · Turn an angle trajectory into an excursion difference

For each leg, **excursion** is its 95th-percentile angle minus its
5th-percentile angle over \(S\). Percentiles reduce the influence of isolated
extreme values, although they do not test timing or waveform shape. The saved
uniform physical clock makes each admitted time sample equally weighted.

\[
E_L=Q_{.95}(\theta_L[S])-Q_{.05}(\theta_L[S]),\quad
E_R=Q_{.95}(\theta_R[S])-Q_{.05}(\theta_R[S]),\quad A=E_R-E_L.
\]

Positive \(A\) means larger right-knee excursion **in this view**. It does not
identify the clinically affected side. Linear interpolation between adjacent
ordered values defines the percentiles in both the model loss and evaluation.

We report \(|\widehat A-A|\) and the mean absolute angle-waveform error. If any
predicted leg is missing or degenerate on \(S\), the endpoint fails and receives
360 degrees for \(A\) error and 180 degrees for waveform error. These are bounded
worst-case penalties because each angle is in [0,180] and \(A\) is in [-180,180].
Reference-ineligible endpoints remain missing, with their count retained in
coverage. They are never assigned zero error.
"""),
        code(r"""
def excursion_difference(angles, support):
    quantiles = np.quantile(angles[support], [0.05, 0.95], axis=0, method='linear')
    left, right = quantiles[1] - quantiles[0]
    return float(left), float(right), float(right - left)

reconstructed = []
for i in range(len(family.records)):
    success = eligible and bool(predicted_geometry[i, common_support].all())
    reference_left, reference_right, reference_A = (
        excursion_difference(reference_angles[i], common_support)
        if eligible else (np.nan, np.nan, np.nan))
    output_left, output_right, output_A = (
        excursion_difference(predicted_angles[i], common_support)
        if success else (np.nan, np.nan, np.nan))
    reconstructed.append(dict(
        reference_A=reference_A, predicted_A=output_A,
        reference_eligible=eligible, prediction_success=success,
        A_error=abs(output_A - reference_A) if success else (360.0 if eligible else np.nan),
        waveform_error=float(np.abs(predicted_angles[i, common_support]
                                    - reference_angles[i, common_support]).mean())
                       if success else (180.0 if eligible else np.nan),
        support_frames=int(common_support.sum())))
reconstructed = pd.DataFrame(reconstructed)
for column in reconstructed:
    np.testing.assert_allclose(reconstructed[column].to_numpy(float),
                               saved_family[column].to_numpy(float),
                               atol=1e-8, rtol=1e-7, equal_nan=True)
display(pd.concat([saved_family[['movement_state', 'movement_level_deg', 'camera_id',
                                'naming', 'observation']], reconstructed], axis=1).head(12))
print('Every angular score in the selected family matches the retained table.')
"""),
        md(r"""
### Read the actual trajectories as well as the scalar

The first metadata-selected row is shown below. Shaded intervals indicate the
fixed reference support. A matching excursion can coexist with incorrect timing
or a distorted waveform, so the curves should be read with both angular errors.
The example selection is fixed before reading the errors.
"""),
        code(r"""
example = 0
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharex=True, sharey=True, layout='constrained')
for side, ax in enumerate(axes):
    ax.plot(times[example], reference_angles[example, :, side], label='Reference', color='#1f618d')
    ax.plot(times[example], predicted_angles[example, :, side], label='Restored', color='#c96b26')
    ax.fill_between(times[example], 0, 180, where=common_support, color='#1f618d', alpha=.07)
    ax.set(title=['Left knee', 'Right knee'][side], xlabel='Physical time (seconds)', ylim=(0, 180))
axes[0].set_ylabel('Projected angle (degrees)')
axes[0].legend(frameon=False)
fig.suptitle(('Software fixture: ' if study.fixture else 'Development example: ')
             + f"{family.records[example]['camera_id']} | {family.records[example]['movement_state']} | "
             f"{family.records[example]['naming']} | {family.records[example]['observation']}")
buffer = BytesIO(); fig.savefig(buffer, format='png', dpi=130); plt.close(fig)
display(Image(data=buffer.getvalue()))
"""),
        md(r"""
## 5 · Reconstruct the position and 0.20-second movement errors

The position score uses the reference bounding-box diagonal \(s_t\), in pixels,
as an evaluation scale. This scale never enters the restoration model. It differs
from the input-only normalization used for training. For a finite prediction,
\(e_{tj}=\|\widehat y_{tj}-y_{tj}\|_2/s_t\). A missing prediction receives 1.0;
finite errors are not clipped at 1.0. Visible-joint errors are averaged within
each frame, then across frames with at least one visible reference joint.

The movement score compares position **changes** separated by exactly 0.20 s:

\[
d_{tj}=\frac{\| (\widehat y_{t+.20,j}-\widehat y_{tj})
                    -(y_{t+.20,j}-y_{tj})\|_2}{s_t}.
\]

Both reference endpoints must be visible. A missing prediction at either end
receives 1.0. The evaluator averages all supported joint/endpoint pairs together;
this is different from the frame-balanced position average. There is no time
warping or interpolation to create a missing 0.20-second match. At 25 Hz the
interval normally corresponds to five frame steps.
"""),
        code(r"""
i = example
truth, output = reference_xy[i], predicted_xy[i]
visible = family.targets['visible'][i]
scale = family.targets['eval_scale'][i].astype(float)
finite = np.isfinite(output).all(-1)
joint_error = np.linalg.norm(output - truth, axis=-1) / scale[:, None]
joint_error = np.where(finite, joint_error, 1.0)
joint_counts = visible.sum(-1)
frame_error = np.divide(np.where(visible, joint_error, 0.0).sum(-1), joint_counts,
                        out=np.full(len(scale), np.nan), where=joint_counts > 0)
visible_nle = float(np.nanmean(frame_error)) if joint_counts.any() else np.nan

displacement_values = []
missing_pairs = 0
for start in range(len(times[i])):
    desired_time = times[i, start] + 0.20
    end = int(np.argmin(np.abs(times[i] - desired_time)))
    if end <= start or abs(times[i, end] - desired_time) > 1e-6:
        continue
    supported = visible[start] & visible[end]
    predicted_both = finite[start] & finite[end]
    movement_error = (output[end] - output[start]) - (truth[end] - truth[start])
    errors = np.linalg.norm(movement_error, axis=-1) / scale[start]
    errors = np.where(predicted_both, errors, 1.0)
    displacement_values.extend(errors[supported])
    missing_pairs += int((supported & ~predicted_both).sum())
displacement_nle = float(np.mean(displacement_values)) if displacement_values else np.nan
np.testing.assert_allclose([visible_nle, displacement_nle],
    saved_family.loc[i, ['visible_nle', 'displacement_nle']].to_numpy(float),
    atol=1e-8, rtol=1e-7, equal_nan=True)
assert len(displacement_values) == saved_family.loc[i, 'displacement_count']
assert missing_pairs == saved_family.loc[i, 'displacement_missing_count']
display(pd.DataFrame([dict(visible_nle=visible_nle, displacement_nle=displacement_nle,
                          visible_joints=int(visible.sum()),
                          displacement_pairs=len(displacement_values), missing_pairs=missing_pairs)]))
"""),
        md(r"""
The separate `synthetic_all_nle` column applies the same position calculation to
all valid synthetic reference joints, including modeled joints hidden in the
image. It is available only for records explicitly marked `synthetic_proxy`.
For real footage, a hidden joint does not become a valid reference because a
different pose estimator produced a coordinate.

### Check geometric left/right assignment separately

For each hip, knee and ankle pair, compare the total distance to named reference
positions with the total distance to swapped positions:

\[
C_{\rm named}=\|\hat y_L-y_L\|+\|\hat y_R-y_R\|,\qquad
C_{\rm swap}=\|\hat y_L-y_R\|+\|\hat y_R-y_L\|.
\]

The references must have both sides valid and separated by the saved minimum
pixel distance. A swapped cost lower by more than the saved margin counts as a
wrong assignment. Costs within the margin are ambiguous; a missing prediction
is also a failure. Unlike the knee-excursion endpoint, this diagnostic scores
each reference-resolvable joint pair independently and uses its own denominator.
It is a geometric comparison to the projected naming convention, not independent
proof of anatomical side or a clinical affected-side prediction.
"""),
        code(r"""
lower_reference = reference_xy[..., 6:12, :].reshape(len(family.records), -1, 3, 2, 2)
lower_prediction = predicted_xy[..., 6:12, :].reshape(lower_reference.shape)
reference_pair_valid = family.targets['valid'][..., 6:12].reshape(
    len(family.records), -1, 3, 2).all(-1)
separation = np.linalg.norm(lower_reference[..., 0, :] - lower_reference[..., 1, :], axis=-1)
resolvable = reference_pair_valid & (separation >= config['measurement']['assignment_separation_px'])
finite_pair = np.isfinite(lower_prediction).all((-1, -2))
named_cost = np.linalg.norm(lower_prediction - lower_reference, axis=-1).sum(-1)
swap_cost = np.linalg.norm(lower_prediction - lower_reference[..., ::-1, :], axis=-1).sum(-1)
margin = config['measurement']['assignment_margin_px']
wrong = resolvable & finite_pair & (swap_cost + margin < named_cost)
ambiguous = resolvable & finite_pair & (np.abs(named_cost - swap_cost) <= margin)
missing = resolvable & ~finite_pair
denominator = resolvable.sum((1, 2))
failure_count = (wrong | ambiguous | missing).sum((1, 2))
assignment = pd.DataFrame(dict(
    assignment_reference_pairs=denominator,
    assignment_wrong_pairs=wrong.sum((1, 2)),
    assignment_ambiguous_pairs=ambiguous.sum((1, 2)),
    assignment_missing_pairs=missing.sum((1, 2)),
    assignment_failure_rate=np.divide(failure_count, denominator,
        out=np.full(len(denominator), np.nan), where=denominator > 0)))
for column in assignment:
    np.testing.assert_allclose(assignment[column], saved_family[column], equal_nan=True, atol=1e-8)
display(pd.concat([saved_family[['naming', 'observation']], assignment], axis=1).head(12))
"""),
        md(r"""

## 6 · Separate movement response, observation sensitivity, and their interaction

Let \(A_{mz}\) denote reference excursion difference for movement level \(m\)
and observation condition \(z\). Subscript 0 means the baseline movement or
the correctly named clear observation. A hat denotes the restored estimate.
We keep the source family, camera, physical mirror state and extractor fixed.

| Quantity | Signed error before taking its absolute value | Question |
| --- | --- | --- |
| Movement response | \(b_{mz}=(\widehat A_{mz}-\widehat A_{0z})-(A_{mz}-A_{0z})\) | Is the measured movement change preserved? |
| Observation sensitivity | \(n_{mz}=(\widehat A_{mz}-\widehat A_{m0})-(A_{mz}-A_{m0})\) | Does obscuring or renaming the same movement change its estimate? |
| Interaction | \(b_{mz}-b_{m0}\) | Does the observation problem alter how accurately the movement change is preserved? |

The **reference change is calculated**, rather than assumed to equal the
declared body-model intervention angle. Projection and percentile excursion can
make those quantities differ. We also retain reference terms in the nuisance
comparison instead of assuming they cancel. The following example selects the
lowest nonzero movement level and compares correctly named clear tracks against
globally swapped clear tracks. This choice uses metadata alone; obscured rows
remain in the full score and coverage tables.
"""),
        code(r"""
def read_selected_family(relative):
    parts = []
    for chunk in pd.read_csv(study.artifact(relative), chunksize=10000):
        keep = chunk.method.eq(method) & chunk.seed.eq(seed) & chunk.source_family_id.eq(family_id)
        if keep.any(): parts.append(chunk.loc[keep])
    return pd.concat(parts, ignore_index=True)
responses = read_selected_family('evaluation/responses.csv')
nuisance = read_selected_family('evaluation/nuisance.csv')
interaction = read_selected_family('evaluation/interaction.csv')

def exactly_one(frame, conditions):
    keep = np.ones(len(frame), dtype=bool)
    for key, value in conditions.items():
        keep &= frame[key].eq(value).to_numpy()
    matched = frame.loc[keep]
    assert len(matched) == 1, f'Expected one endpoint, found {len(matched)}'
    return matched.iloc[0]

anchor = saved_family.loc[saved_family.movement_state.eq('baseline')
                         & saved_family.naming.eq('correct')
                         & saved_family.observation.eq('clear')].iloc[0]
fixed_keys = ['method', 'seed', 'canonical_person_id', 'source_family_id',
              'physical_state', 'camera_id', 'extractor']
stratum = {k: anchor[k] for k in fixed_keys}
same_view = saved_family.loc[
    saved_family.physical_state.eq(anchor.physical_state)
    & saved_family.camera_id.eq(anchor.camera_id)
    & saved_family.extractor.eq(anchor.extractor)]
changed = same_view.loc[same_view.movement_level_deg.gt(0)].sort_values('movement_level_deg').iloc[0]
movement = dict(movement_state=changed.movement_state, movement_level_deg=changed.movement_level_deg)
nuisance_condition = dict(naming='global_swap', observation='clear')
clear = dict(naming='correct', observation='clear')
baseline = dict(movement_state='baseline', movement_level_deg=0.0)
y00 = exactly_one(saved_family, stratum | baseline | clear)
ym0 = exactly_one(saved_family, stratum | movement | clear)
y0z = exactly_one(saved_family, stratum | baseline | nuisance_condition)
ymz = exactly_one(saved_family, stratum | movement | nuisance_condition)

reference_response = ymz.reference_A - y0z.reference_A
predicted_response = ymz.predicted_A - y0z.predicted_A
response_bias = predicted_response - reference_response
clear_response_bias = (ym0.predicted_A - y00.predicted_A) - (ym0.reference_A - y00.reference_A)
nuisance_bias = (ymz.predicted_A - ym0.predicted_A) - (ymz.reference_A - ym0.reference_A)
interaction_bias = response_bias - clear_response_bias

# Apply the same endpoint failure rule, including its bounded penalties.
checks = []
for name, bias, endpoints, penalty, table in [
    ('response', response_bias, [ymz, y0z], 720.0, responses),
    ('nuisance', nuisance_bias, [ymz, ym0], 720.0, nuisance),
    ('interaction', interaction_bias, [ymz, y0z, ym0, y00], 1440.0, interaction),
]:
    admitted = all(bool(r.reference_eligible) for r in endpoints)
    success = admitted and all(bool(r.prediction_success) for r in endpoints)
    error = abs(bias) if success else (penalty if admitted else np.nan)
    saved_row = exactly_one(table, stratum | movement | nuisance_condition)
    np.testing.assert_allclose(error, saved_row[name + '_error'], equal_nan=True, atol=1e-8)
    np.testing.assert_allclose(bias if success else np.nan, saved_row[name + '_bias'],
                               equal_nan=True, atol=1e-8)
    checks.append(dict(contrast=name, signed_error=bias if success else np.nan,
                      absolute_error=error, reference_eligible=admitted, prediction_success=success))
print('Declared intervention:', movement, '| observation:', nuisance_condition)
print('Measured reference change:', reference_response, '| restored change:', predicted_response)
display(pd.DataFrame(checks))
"""),
        md(r"""
The 720-degree response/nuisance penalties and 1440-degree interaction penalty
follow from the bounds on the corresponding differences of \(A\). These are
failure-accounting rules, not successful predictions of a large error.
An exact duplicated `no_change` endpoint is retained as a diagnostic, but is
excluded from the primary response mean so that trivial zero changes cannot
dilute that endpoint.

## 7 · Check coverage and conditions before interpreting a mean

`reference_eligible` counts the rows the references permit us to measure;
`successful` counts predictions that are complete on those fixed frames. They
answer different questions. The condition table contains supported means and
can conceal absent reference coverage unless read with those counts. Camera
views change projected anatomy, so compare errors against each view's own
reference rather than comparing raw knee angles between views.
"""),
        code(r"""
selected = saved_family
condition_summary = selected.groupby(['extractor', 'camera_id', 'naming', 'observation']).agg(
    records=('reference_eligible', 'size'),
    reference_eligible=('reference_eligible', 'sum'),
    successful=('prediction_success', 'sum'),
    angular_error=('A_error', 'mean'),
    waveform_error=('waveform_error', 'mean'),
    visible_position_error=('visible_nle', 'mean'))
display(condition_summary)
print('This displayed condition breakdown covers the metadata-selected source family only.')
print('Full population condition scores:', study.work / 'evaluation/by-condition.csv')
print('Evidence:', dev.evidence_status)
"""),
        md(r"""
These cells explain how position accuracy and movement fidelity can disagree.
Neither a low coordinate error nor a matching excursion proves that the original
timing and all movement features survived restoration. The geometric side
assignment diagnostic addresses a separate kind of error and has a different
reference-support rule, as its explicit calculation shows.

Continue to **06_verify_and_write.ipynb** to calculate equal-person summaries,
reconstruct uncertainty across people and training seeds, and check the evidence
behind a manuscript claim.
"""),
        *build_gavd_cells(md,code),
    ]


def build_gavd_cells(md,code):
    return [md(r'''
## 8 · Evaluate real-video information without inventing reference joints

The GAVD branch reads the original per-frame annotation boxes, scales each box
from its recorded image size to the decoded frame size, and extracts only the
annotated subject. Missing boxes, unreadable frames and unsupported sampling
rates remain exclusions. Consecutive windows never overlap, and timestamps
come from the selected original frames. A source video and every linked
identity remain in one partition. The evaluation script retains sample counts
and failed extractions alongside its scores.

Each AMASS restorer is fixed before real-video evaluation. For every sequence,
we calculate input-normalized joint-position quantiles, velocity root-mean-square
values, joint availability and projected knee excursions. We average windows
within their original sequence. A small **linear probe** is a classifier fitted
to these fixed numerical descriptors; it tests whether gait-pattern label
information remains accessible after restoration. Training the probe uses only
GAVD training groups. This endpoint cannot establish that corrected coordinates
are anatomically accurate, or that a measured asymmetry identifies disease.

For feature matrix $F$, the training set supplies missing-value medians, weighted
means and scales. Standardized features with an intercept form $D$. One-hot
class labels form $Y$, and each recording group receives equal total weight
through diagonal $W$. With fixed ridge penalty $\Lambda$, the coefficients are

$$B=(D^TWD+\Lambda)^{-1}D^TWY.$$

The intercept is unpenalized. The code below reproduces the production solve on
small illustrative arrays; no research checkpoint or GAVD result is changed.
'''),code('''
from gavd6_sjepa.research_directions.gait_fidelity.gavd import fit_probe, probe_predict
training_features = np.array([[.1, .3], [.2, np.nan], [.4, .2],
                              [.8, .6], [.9, .7], [.7, .5]])
labels = np.array(['normal', 'normal', 'normal', 'abnormal', 'abnormal', 'abnormal'])
groups = np.array(['video-a', 'video-a', 'video-b', 'video-c', 'video-c', 'video-d'])
medians = np.nanmedian(training_features, axis=0)
x = np.where(np.isfinite(training_features), training_features, medians)
weights = np.array([1 / np.sum(groups == group) for group in groups])
weights /= weights.sum()
mean = np.sum(x * weights[:, None], axis=0)
scale = np.maximum(np.sqrt(np.sum((x-mean)**2 * weights[:,None], axis=0)), 1e-6)
design = np.column_stack([(x-mean)/scale, np.ones(len(x))])
classes = np.unique(labels)
one_hot = (labels[:,None] == classes).astype(float)
penalty = np.eye(design.shape[1]); penalty[-1,-1] = 0
coefficients = np.linalg.solve(design.T @ (weights[:,None] * design) + penalty,
                               design.T @ (weights[:,None] * one_hot))
production = fit_probe(training_features, labels, groups, ridge=1.)
np.testing.assert_allclose(coefficients, production['coef'])
new_features = np.array([[.3, np.nan], [.8, .7]])
new_x = np.where(np.isfinite(new_features), new_features, medians)
scores = np.column_stack([(new_x-mean)/scale, np.ones(len(new_x))]) @ coefficients
predicted_labels = classes[scores.argmax(axis=1)]
np.testing.assert_array_equal(predicted_labels, probe_predict(production, new_features))
display(pd.DataFrame({'example': [0, 1], 'illustrative prediction': predicted_labels}))
'''),md(r'''
The real comparison includes unchanged tracks, a fixed temporal filter and a
camera/height-only classifier. A strong camera-only result signals that the
acquisition setup can predict the label, which limits interpretation of a pose
classifier's accuracy. Macro recall averages class-specific recall; uncertainty
resamples recording groups and, for the primary neural contrast, training seeds.
Read unsupported classes and extraction coverage before interpreting the mean.

After `gavd-plan` and `launch --gavd-only` have completed, use the CPU allocation
command in the HAIC guide to run `gavd-evaluate`. The default comparison uses
the saved primary AMASS candidate and direct-training comparator across all
seeds. Confirmation remains separate and requires a frozen checkpoint selection
and verified exposure review. The following cell lists existing reports only.
'''),code('''
real_reports = []
for path in sorted((study.work / 'gavd/evaluation').glob('*/summary.json')):
    value = json.loads(path.read_text())
    real_reports.append({'output': value['output'], 'split': value['split'],
                         'status': value['status'], 'clinical_validation': value['clinical_validation']})
if real_reports:
    display(pd.DataFrame(real_reports))
else:
    print('No retained GAVD evaluation yet. The synthetic results above do not imply real-video transfer.')
''')]


def build_verification_cells(md, code):
    return [
        md(r"""
## 1 · Verify the retained evidence before summarizing it

The canonical verifier checks the frozen configuration and source, retained
artifact hashes, and reconstruction of each trained model's saved `A_error`,
`waveform_error`, and `support_frames`. Notebook 05 separately reconstructs
position/displacement and the three contrasts for a metadata-selected example.
Below we expose the remaining calculation from records to people and seeds.

A byte-for-byte check establishes that files have not changed since their
receipt. It cannot establish that the study's assumptions are scientifically
correct, so numerical reconstruction and methodological checks remain separate.
"""),
        code("study.command('verify')\nstudy.command('report')"),
        code(r"""
import hashlib
import numpy as np
import pandas as pd
from IPython.display import Markdown, display, Image
from io import BytesIO
import matplotlib.pyplot as plt

config = study.artifact('config.json')
plan = study.artifact('plan.json')
ledger = study.artifact('ledger.json')
frozen = study.artifact('frozen.json')

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

assert file_sha256(study.work / 'config.json') == frozen['config_sha256']
assert file_sha256(study.work / 'plan.json') == frozen['plan_sha256']
phase = next(p for p in plan['phases'] if p['phase'] != 'pretrain')
completed = ledger['completed'][phase['phase_id']]
assert file_sha256(completed['receipt']) == completed['sha256']
receipt = json.loads(Path(completed['receipt']).read_text())
artifact_checks = []
for path, expected in receipt['artifacts'].items():
    actual = file_sha256(path)
    assert actual == expected, f'Changed artifact: {path}'
    artifact_checks.append(dict(artifact=Path(path).name, sha256=actual[:16], matches=True))
display(pd.DataFrame(artifact_checks))
print('Illustrated phase:', phase['phase_id'], '| canonical verifier checks all completed phases.')
"""),
        md(r"""
## 2 · Average conditions, windows, raw motions and people

An extractor, view, movement variant or repeat seed does not create a new
participant. Let \(e_{psfc}\) be an error for person \(p\), seed \(s\), source
family \(f\), and condition \(c\). The saved summary first computes each
family's mean over its supported conditions, averages windows belonging to the
same raw motion, then gives each supported motion equal weight within a person:

\[
\bar e_{psf}=\operatorname{mean}_{c}(e_{psfc}),\qquad
\bar e_{psm}=\operatorname{mean}_{f\in m}(\bar e_{psf}),\qquad
\bar e_{ps}=\operatorname{mean}_{m}(\bar e_{psm}).
\]

The code follows pandas' missing-value convention: reference-ineligible rows
are skipped by these means, and an entirely unsupported group remains missing.
Prediction failures stay in the mean through their declared penalties. A result
therefore describes the **reference-supported population** and must be reported
with coverage; it is not an unconditional claim about every prepared recording.
"""),
        code(r"""
saved_person = pd.read_csv(study.artifact('evaluation/per-person.csv'))
from gavd6_sjepa.research_directions.gait_fidelity.data import load_dataset
bundle = load_dataset(study.bundle_path())
motion_for_family = {r['source_family_id']: r['motion_hash'] for r in bundle.records}
keys = ['method', 'seed', 'canonical_person_id']
metrics = ['A_error', 'waveform_error', 'visible_nle', 'synthetic_all_nle',
           'displacement_nle', 'assignment_failure_rate']

def reconstruct_people(relative, metrics, exclude_no_change=False):
    # Sums and nonmissing counts permit bounded-memory CSV reads. Averaging
    # chunk means would overweight a small fragment at a chunk boundary.
    sums, counts = [], []
    family_keys = keys + ['source_family_id']
    for frame in pd.read_csv(study.artifact(relative), chunksize=10000):
        if exclude_no_change: frame = frame.loc[frame.movement_state.ne('no_change')]
        grouped = frame.groupby(family_keys)[metrics]
        sums.append(grouped.sum())
        counts.append(grouped.count())
    total = pd.concat(sums).groupby(level=family_keys).sum()
    supported = pd.concat(counts).groupby(level=family_keys).sum()
    families = (total / supported.replace(0, np.nan)).reset_index()
    families['motion_hash'] = families.source_family_id.map(motion_for_family)
    assert families.motion_hash.notna().all()
    motions = families.groupby(keys + ['motion_hash'])[metrics].mean()
    return motions.groupby(keys).mean().reset_index()

person_means = reconstruct_people('evaluation/per-window.csv', metrics)
for relative, metric, exclude in [('responses', 'response_error', True),
                                 ('nuisance', 'nuisance_error', False),
                                 ('interaction', 'interaction_error', True)]:
    values = reconstruct_people('evaluation/' + relative + '.csv', [metric], exclude)
    person_means = person_means.merge(values, on=keys, how='left', validate='one_to_one')

reconstructed = person_means.set_index(keys).sort_index()
retained = saved_person.set_index(keys).sort_index()
assert reconstructed.index.equals(retained.index)
np.testing.assert_allclose(reconstructed.to_numpy(float), retained[reconstructed.columns].to_numpy(float),
                           atol=1e-8, rtol=1e-7, equal_nan=True)
print('All person-level metrics reconstruct:', len(reconstructed), 'person × seed × method rows')
display(person_means.head(12))
"""),
        md(r"""
## 3 · Define the primary paired comparison before reading its outcome

The saved configuration specifies the candidate and comparator. For a
lower-is-better error, define the person-and-seed improvement as

\[
D_{ps}=\bar e_{ps}^{\mathrm{comparator}}-\bar e_{ps}^{\mathrm{candidate}}.
\]

Positive values favor the candidate. Subtraction occurs **within the same
person and seed**, rather than between two unrelated collections of scores.
The point estimate is the mean of the complete person-by-seed matrix. A missing
method, unsupported value, or incomplete seed grid blocks the uncertainty
calculation rather than silently reducing the comparison population.
"""),
        code(r"""
evaluation = config['evaluation']
candidate = evaluation['primary_candidate']
comparator = evaluation['primary_comparator']
metric = 'response_error'
selected = person_means.loc[person_means.method.isin([candidate, comparator])]
wide = selected.pivot(index=['canonical_person_id', 'seed'], columns='method', values=metric)
has_methods = candidate in wide and comparator in wide
complete_pairs = has_methods and not wide.isna().any().any()
delta = (wide[comparator] - wide[candidate]).unstack('seed') if complete_pairs else pd.DataFrame()
supported = complete_pairs and len(delta) >= 2 and not delta.isna().any().any()
print('Candidate:', candidate, '\nComparator:', comparator)
print('Complete person/seed grid:', supported)
display(delta)
"""),
        md(r"""
## 4 · Reproduce the crossed person-and-seed bootstrap

A **bootstrap** approximates sampling variation by repeatedly sampling observed
units with replacement. We draw whole people, keeping their correlated families
and conditions together. We also draw fitted seeds as a second, crossed factor:
the same sampled seed columns apply to every sampled person. Methods stay paired
because we resample their precomputed differences.

For each draw, the person-conditional estimate averages the sampled people over
the original seed columns. The crossed estimate additionally resamples those
columns. The 2.5th and 97.5th percentiles of these means form the two displayed
95% intervals. We reproduce the production random-number draw order exactly;
the demonstration does not consume the training random-number stream.

With only three fitted seeds, the empirical seed distribution remains sparse.
These intervals describe variation in the available development data and fits;
they do not account for earlier method selection, dataset screening decisions,
or reuse of development people. Crossing zero is not the sole criterion for a
scientifically useful or generalizable result.
"""),
        code(r"""
saved_comparison = study.artifact('evaluation/comparisons.json')[metric]
if supported:
    values = delta.to_numpy(float)  # [people, fitted seeds]
    rng = np.random.default_rng(evaluation['bootstrap_seed'])
    person_draws, crossed_draws = [], []
    for _ in range(evaluation['bootstrap_draws']):
        sampled_people = rng.integers(len(values), size=len(values))
        sampled_seeds = rng.integers(values.shape[1], size=values.shape[1])
        person_draws.append(float(values[sampled_people].mean()))
        crossed_draws.append(float(values[sampled_people][:, sampled_seeds].mean()))
    improvement = float(values.mean())
    person_interval = np.quantile(person_draws, [0.025, 0.975])
    crossed_interval = np.quantile(crossed_draws, [0.025, 0.975])
    np.testing.assert_allclose(improvement, saved_comparison['improvement'], atol=1e-8)
    np.testing.assert_allclose(person_interval, saved_comparison['person_conditional_ci95'], atol=1e-8)
    np.testing.assert_allclose(crossed_interval, saved_comparison['crossed_person_seed_ci95'], atol=1e-8)
    display(pd.DataFrame([
        dict(summary='Person-conditional', improvement=improvement, lower=person_interval[0], upper=person_interval[1]),
        dict(summary='Crossed person and seed', improvement=improvement, lower=crossed_interval[0], upper=crossed_interval[1]),
    ]))
    print('Per-seed improvements:', delta.mean(axis=0).to_dict())
else:
    assert saved_comparison['status'].startswith('insufficient')
    print('The fixed comparison is unsupported:', saved_comparison)
"""),
        md(r"""
### Inspect whether the average hides inconsistent people or seeds

Each dot below represents the same person's paired improvement for one fitted
seed, averaged over that person's supported source families. The line at zero
separates improvement from harm. These dots are related observations; their
number is not the evaluation sample size. Deterministic baselines have identical
predictions repeated across seed rows for bookkeeping, not independently fitted
calibrations.
"""),
        code(r"""
if supported:
    fig, ax = plt.subplots(figsize=(9, max(3.5, .36 * len(delta) + 1.5)), layout='constrained')
    vertical = np.arange(len(delta))
    offsets = np.linspace(-.18, .18, delta.shape[1])
    for j, (seed_value, offset) in enumerate(zip(delta.columns, offsets)):
        ax.scatter(delta.iloc[:, j], vertical + offset, s=28, label=f'Seed {seed_value}')
    ax.axvline(0, color='#777777', linewidth=1)
    ax.set_yticks(vertical, delta.index.astype(str))
    ax.set(xlabel='Comparator error − candidate error (degrees; positive favors candidate)',
           ylabel='Development person',
           title=('Software fixture: ' if study.fixture else 'Development study: ')
                 + 'movement-response improvement for each person and seed')
    ax.legend(frameon=False, ncol=min(3, delta.shape[1]))
    buffer = BytesIO(); fig.savefig(buffer, format='png', dpi=130); plt.close(fig)
    display(Image(data=buffer.getvalue()))
"""),
        md(r"""
## 5 · Read the report with a claim-specific evidence checklist

The final report is generated from the retained summaries, not from these
teaching calculations. The equality checks above show that the tutorial follows
the same rules. A reproducible result can still be limited by small or previously
inspected populations and by the difference between synthetic and real references.

| Intended conclusion | Required comparison and evidence |
| --- | --- |
| Restoration improves position accuracy | Same-population coordinate error and coverage against unchanged poses and training-only calibration |
| Change supervision preserves movement | Matched base versus paired-change objectives, with identical endpoint exposure and useful effect sizes |
| Anatomical masking contributes | Graph masks versus topology/duration controls with acceptable coverage and run-length matching |
| JEPA contributes useful features | Matched coordinate pretraining, initialized encoder and shuffled-reference controls, plus direct training |
| Meaningful pairing contributes | Correct pairing versus per-example labels and valid re-pairing, with the distribution audit reported |
| The method generalizes | A frozen protocol evaluated on new people or conditions, with reference coverage and prediction failures retained |
| The method supports clinical gait assessment | Independent clinical measurements and an appropriate real-world population |

This report uses the saved development population; some people may have been
inspected in earlier studies. The original manifest test split remains locked
until the protocol and exposure review are fixed. A complete fixture or a
favorable interval does not establish independent confirmation. The data are image-plane
proxies for gait measurements; a result cannot establish preservation of disease
severity, anatomical 3D range of motion or clinical change without additional
reference evidence. Additional entries in the selected core or full matrix
provide attribution and exploratory comparisons under the same primary claim.
"""),
        code(r"""
summary = study.artifact('evaluation/summary.json')
print(json.dumps({k: summary[k] for k in ('status', 'evidence_status', 'scientific_gate',
                 'independent_confirmation', 'clinical_validation')}, indent=2))
display(Markdown(study.artifact('report.md').read_text()))
"""),
        md(r"""
## 6 · Retain enough evidence for another reader to reconstruct the result

On your Mac, use `slurm/gait-fidelity/retrieve.sh` with the HAIC work directory
and a new local destination. That helper copies predictions, tables and viewer
artifacts while leaving model checkpoints on HAIC. Full artifact-hash verification
also needs the checkpoints; the Slurm guide includes the complete-transfer
command. Preserve the original release and configuration rather than replacing
them with a newer implementation under the same run name.

Write the evaluated participant count, retained source-family count, fitted
seeds, reference coverage, prediction failure rate, effect size and uncertainty
alongside the comparison. State which decisions used development outcomes and
which conditions were genuinely held out. Report unresolved masking or pairing
audit failures as limits on the corresponding interpretation.
"""),
    ]
