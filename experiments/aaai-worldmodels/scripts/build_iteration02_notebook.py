"""Build and optionally execute the nine-cell iteration02 evidence/masking notebook.

This notebook reads frozen results and tests mask designs on real prepared data.
It neither trains new models nor changes immutable experiment outputs.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

PROJECT=Path(__file__).resolve().parents[1]
NAME='16_iteration02_evidence_and_mask_design.ipynb'
OUT=PROJECT/'docs/iteration02'
EXECUTED=PROJECT/'artifacts/executed-iteration02'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

SETUP=r'''from pathlib import Path
import hashlib, json, sys, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import roc_auc_score, average_precision_score
assert sys.version_info[:2] == (3, 12), 'Use the experiment Python 3.12 environment.'
candidates = [Path.cwd(), *Path.cwd().parents]
PROJECT = next((p for p in candidates if (p/'worldmodels/mask_design.py').exists()), None)
if PROJECT is None:
    PROJECT = next(p/'experiments/aaai-worldmodels' for p in candidates
                   if (p/'experiments/aaai-worldmodels/worldmodels/mask_design.py').exists())
REPOSITORY = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from worldmodels.data import build_inventory, JOINT_IDS
from worldmodels.mask_design import design_prefix_mask, POLICIES, REFERENCE_JOINTS
DATA = PROJECT/'artifacts/runs/restart-full10hz-v1'
RUN = PROJECT/'artifacts/runs/restart-dynamics-v1'
OUT = PROJECT/'docs/iteration02'
OUT.mkdir(parents=True, exist_ok=True)
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
required = [RUN/'results.json', RUN/'artifact_hashes.json', DATA/'preparation.json',
            DATA/'windows.npz', DATA/'windows.csv', PROJECT/'docs/draft02/posthoc_score_audit.json']
for path in required:
    assert path.is_file(), f'Required real artifact is missing: {path}'
result = json.loads((RUN/'results.json').read_text())
assert result['status'] == 'complete'
frozen_hashes = json.loads((RUN/'artifact_hashes.json').read_text())
for name, expected in frozen_hashes.items():
    assert digest(RUN/name) == expected, f'Frozen artifact mismatch: {name}'
preparation = json.loads((DATA/'preparation.json').read_text())
assert digest(DATA/'windows.npz') == preparation['data_sha256']
assert digest(DATA/'windows.csv') == preparation['metadata_sha256']
metadata = pd.read_csv(DATA/'windows.csv')
cohort = pd.read_csv(RUN/'cohort.csv')
assert cohort.window_id.is_unique and metadata.window_id.is_unique
assert cohort.groupby('source_id').outer_fold.nunique().max() == 1
inventory, fresh_inventory = build_inventory(REPOSITORY)
frozen_inventory = json.loads((PROJECT/'docs/restart/data_audit.json').read_text())['inventory']
assert fresh_inventory['annotation_identity'] == frozen_inventory['annotation_identity']
display(pd.DataFrame([
    {'Population':'Complete annotation inventory', 'Sequences':fresh_inventory['all_sequences'],
     'Source videos':fresh_inventory['all_sources'], 'Windows':'Not an inventory unit'},
    {'Population':'Five requested annotation labels', 'Sequences':len(inventory),
     'Source videos':inventory.source_id.nunique(), 'Windows':'Not an inventory unit'},
    {'Population':'Retained evaluation cohort', 'Sequences':cohort.sequence_id.nunique(),
     'Source videos':cohort.source_id.nunique(), 'Windows':len(cohort)}]))
display(cohort.groupby('label').agg(windows=('window_id','size'),
                                  sequences=('sequence_id','nunique'), sources=('source_id','nunique')))
print(f'Python {platform.python_version()}; verified {len(frozen_hashes)} frozen result files.')
print('Prepared windows:', len(metadata), '| Common-support windows:', len(cohort))
print('Normal sources in the inventory:', fresh_inventory['by_label']['normal']['sources'],
      '| Retained:', cohort.loc[cohort.label.eq('normal'), 'source_id'].nunique())
'''

EVIDENCE=r'''source = pd.read_csv(RUN/'source_scores.csv')
normal = source.loc[source.label.eq('normal') & source.horizon.eq(8)].pivot(
    index='source_id', columns='arm', values='mse')
forecast_rows = []
for arm in ['persistence', 'own', 'bilateral', 'shifted', 'static', 'bilateral_shrunk']:
    measured = normal[arm].mean()
    stored = next(x for x in result['normal_forecast'] if x['arm']==arm and x['horizon']==8)
    np.testing.assert_allclose(measured, stored['estimate'], rtol=1e-12, atol=1e-14)
    forecast_rows.append({'Forecast':arm, 'Normal source MSE':measured,
                         'Reduction from persistence (%)':100*(normal.persistence.mean()-measured)/normal.persistence.mean()})
display(pd.DataFrame(forecast_rows).round(5))
scores = pd.read_csv(RUN/'source_anomaly_scores.csv')
scores = scores.merge(cohort[['source_id','outer_fold']].drop_duplicates(),
                      on='source_id', validate='one_to_one')
y = scores.label.ne('normal').to_numpy(int)
np.testing.assert_allclose(scores.excess_error, scores.bilateral-scores.persistence,
                           rtol=1e-12, atol=1e-14)
ranking_rows = []
for key, stored in result['anomaly'].items():
    auc = roc_auc_score(y, scores[key]); ap = average_precision_score(y, scores[key])
    np.testing.assert_allclose([auc, ap], [stored['auroc'], stored['average_precision']], atol=1e-12)
    ranking_rows.append({'Score':key, 'AUROC':auc, 'Average precision':ap,
                         'Status':'Prespecified exploratory score'})
scores['low_motion_diagnostic'] = -scores.persistence
ranking_rows.append({'Score':'Negative persistence',
    'AUROC':roc_auc_score(y, scores.low_motion_diagnostic),
    'Average precision':average_precision_score(y, scores.low_motion_diagnostic),
    'Status':'Post hoc diagnostic; direction chosen after outcomes'})
display(pd.DataFrame(ranking_rows).round(4))
posthoc = json.loads((PROJECT/'docs/draft02/posthoc_score_audit.json').read_text())
for path, expected in posthoc['inputs'].items():
    assert digest(PROJECT/path) == expected
within = []
for fold, group in scores.groupby('outer_fold'):
    label = group.label.ne('normal')
    within.append({'fold':int(fold), 'normal':int((~label).sum()), 'abnormal':int(label.sum()),
        'excess_auroc':roc_auc_score(label, group.excess_error),
        'low_motion_auroc':roc_auc_score(label, group.low_motion_diagnostic)})
within = pd.DataFrame(within)
weights = within.normal*within.abnormal
within_point = float(np.average(within.excess_auroc-within.low_motion_auroc, weights=weights))
np.testing.assert_allclose(within_point, posthoc['within_fold_excess_minus_low_motion']['estimate'], atol=1e-12)
display(within.round(4))
interval = posthoc['within_fold_excess_minus_low_motion']['ci95']
print(f'Pair-weighted within-fold AUROC difference: {within_point:.5f}, conditional 95% interval {interval}.')
print('The point estimate is independently recomputed. The interval is read from the hash-checked post hoc audit,')
print('which uses 5,000 paired source resamples within fold/class cells. No model is refitted here.')
plt.rcParams.update({'svg.fonttype':'none', 'pdf.fonttype':42, 'font.size':10,
                     'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for label, marker, color in [('normal','o','#18766c'), ('abnormal','^','#b7642a')]:
    g = scores[scores.label.eq('normal') if label=='normal' else scores.label.ne('normal')]
    axes[0].scatter(g.persistence*1000, g.bilateral*1000, s=28, marker=marker, color=color,
                    alpha=.8, label=f'{label.capitalize()} ({len(g)} sources)')
limit = max(scores.persistence.max(), scores.bilateral.max())*1050
axes[0].plot([0,limit],[0,limit], '--', color='#777777', linewidth=1)
axes[0].set(xlim=(0,limit), ylim=(0,limit), xlabel='Persistence MSE × 1,000',
            ylabel='Bilateral forecast MSE × 1,000', title='Smaller raw error can reflect less movement')
axes[0].legend(frameon=False, fontsize=9)
positions = np.arange(len(within)); width=.35
axes[1].bar(positions-width/2, within.excess_auroc, width, color='#18766c', label='Excess error')
axes[1].bar(positions+width/2, within.low_motion_auroc, width, color='#b7642a', label='Low motion (post hoc)')
axes[1].axhline(.5, linestyle='--', color='#777777', linewidth=1)
axes[1].set(xticks=positions, xticklabels=[f'{int(r.fold)}\n{int(r.normal)}/{int(r.abnormal)}' for r in within.itertuples()],
            ylim=(0,1), xlabel='Outer fold and normal/abnormal source counts', ylabel='Within-fold AUROC',
            title='The simple reference remains competitive')
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc='lower right', bbox_to_anchor=(.99,.055), ncol=2, fontsize=9)
fig.suptitle('Measured frozen outputs: 91 source videos, not 91 verified participants', fontsize=13)
fig.text(.5, .01, 'Squared 2D coordinates use prefix torso normalization. Fold models share training sources. No new fit is shown.',
         ha='center', fontsize=9)
fig.subplots_adjust(left=.07, right=.99, top=.80, bottom=.25, wspace=.24)
for suffix in ['svg','png']:
    fig.savefig(OUT/f'motion_reference_audit.{suffix}', dpi=170, bbox_inches='tight')
plt.show()
'''

MASKS=r'''with np.load(DATA/'windows.npz', allow_pickle=False) as arrays:
    x, observed = arrays['x'], arrays['valid']
assert x.shape == (len(metadata), 24, len(JOINT_IDS), 2)
assert observed.shape == x.shape[:-1] and observed.dtype == np.bool_
index_by_window = pd.Series(np.arange(len(metadata)), index=metadata.window_id)
selected = cohort.sort_values(['source_id','sequence_id','first_frame','window_id']).iloc[0]
example_index = int(index_by_window.loc[selected.window_id])
full_x, full_valid = x[example_index], observed[example_index]
prefix, prefix_valid = full_x[:16], full_valid[:16]
# Read the actual decoder times from the source cache, not an invented evenly spaced grid.
def decoder_times(row):
    frames = np.asarray(json.loads(row.frame_numbers), dtype=np.int64)
    path = DATA/'source_poses'/f'{row.source_id}.npz'
    with np.load(path, allow_pickle=False) as cache:
        lookup = {int(frame):i for i,frame in enumerate(cache['frame_numbers'])}
        positions = np.asarray([lookup[int(frame)] for frame in frames])
        np.testing.assert_array_equal(cache['decoded_indexes'][positions], frames-1)
        times = cache['timestamp_seconds'][positions].copy()
    np.testing.assert_allclose(times, np.asarray(json.loads(row.timestamp_seconds)), rtol=0, atol=1e-10)
    return times, path
all_times, example_cache = decoder_times(selected)
assert np.isfinite(all_times).all() and np.all(np.diff(all_times)>0)
prefix_times = all_times[:16]
DISTAL_JOINTS = (25,26,27,28,29,30,31,32)
def future_queries(horizon=8):
    # Query locations depend only on the declared horizon and joint IDs.
    # Future coordinate values and future validity never select queries.
    query = np.zeros((horizon, len(JOINT_IDS)), dtype=bool)
    query[:, [JOINT_IDS.index(j) for j in DISTAL_JOINTS]] = True
    return query
queries = future_queries()
assert queries.sum() == 64
masks = {}; counts = []
for policy in POLICIES:
    hidden, protected, info = design_prefix_mask(prefix, prefix_valid, prefix_times,
                                                policy, ratio=.4, seed=27)
    masks[policy] = (hidden, protected)
    counts.append(dict(info, protected_tokens=int(protected.sum()),
                       future_query_tokens=int(queries.sum()),
                       observed_future_query_tokens=int((queries & full_valid[16:]).sum())))
display(pd.DataFrame(counts))
print('Example source:', selected.source_id, '| sequence:', selected.sequence_id)
print('Native one-based frames:', json.loads(selected.frame_numbers))
print('Actual prefix gaps (seconds):', np.unique(np.round(np.diff(prefix_times), 7)).tolist())
print('The exact integer budget rounds 40% of eligible tokens. Realized fractions are reported above.')
print('Every policy protects observed shoulders/hips and each joint\'s latest observation.')
print('Motion-balanced selection reserves at least 20% of its hidden budget for uniform draws.')
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
colors = ['#d9dee3','#cbe6df','#ed9b55','#4389bd','#ffffff','#9d87cb']
state_names = ['Naturally missing','Visible prefix','Artificially hidden','Protected reference/anchor',
               'Future non-query','Fixed future query']
short_names = {11:'L shoulder',12:'R shoulder',13:'L elbow',14:'R elbow',15:'L wrist',16:'R wrist',
               23:'L hip',24:'R hip',25:'L knee',26:'R knee',27:'L ankle',28:'R ankle',
               29:'L heel',30:'R heel',31:'L foot index',32:'R foot index'}
fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharex=True, sharey=True)
for ax, policy, info in zip(axes.flat, POLICIES, counts):
    hidden, protected = masks[policy]
    state = np.full((24,len(JOINT_IDS)), 4, dtype=int)
    state[:16] = np.where(prefix_valid, 1, 0)
    state[:16][hidden] = 2; state[:16][protected] = 3
    state[16:][queries] = 5
    ax.imshow(state.T, aspect='auto', interpolation='nearest',
              cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(-.5,6.5,1),6))
    ax.axvline(15.5, color='#333333', linewidth=1.4)
    ax.set(xticks=[0,5,10,15,16,19,23], xticklabels=['0','5','10','15','16','19','23'],
           yticks=np.arange(len(JOINT_IDS)), yticklabels=[f'{short_names[j]} ({j})' for j in JOINT_IDS],
           xlabel='Sample index; future begins at 16',
           title=f"{policy.replace('_',' ').title()}: {info['hidden_tokens']} / {info['eligible_tokens']} hidden")
    ax.tick_params(labelsize=8, labelleft=True, labelbottom=True)
fig.suptitle('Four mask designs on one real window; no model is trained here', fontsize=14)
fig.legend(handles=[Patch(facecolor=c,edgecolor='#777777',label=n) for c,n in zip(colors,state_names)],
           loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(.5,.02))
fig.text(.5,.005,'Purple queries remain fixed even when future joints are unobserved. Only scoring support uses future validity.',
         ha='center',fontsize=9)
fig.tight_layout(rect=(0,.09,1,.95))
for suffix in ['svg','png']:
    fig.savefig(OUT/f'mask_design_examples.{suffix}', dpi=170, bbox_inches='tight')
plt.show()
'''

VALIDATION=r'''# One deterministic retained window per source tests coverage, not statistical performance.
representatives = cohort.sort_values(['source_id','sequence_id','first_frame','window_id']).drop_duplicates('source_id')
assert len(representatives) == 91
validation_rows, skipped, source_hashes = [], [], {}
for row in representatives.itertuples(index=False):
    idx = int(index_by_window.loc[row.window_id])
    original, valid = x[idx], observed[idx]
    times, cache_path = decoder_times(row)
    source_hashes[str(cache_path.relative_to(PROJECT))] = digest(cache_path)
    if not np.isfinite(times[:16]).all() or np.any(np.diff(times[:16]) <= 0):
        skipped.append({'source_id':row.source_id, 'window_id':row.window_id,
                        'reason':'invalid or nonincreasing actual prefix decoder timestamps'})
        continue
    hidden_counts = []
    for policy in POLICIES:
        hidden, protected, info = design_prefix_mask(original[:16], valid[:16], times[:16],
                                                    policy, ratio=.4, seed=27)
        eligible = valid[:16] & ~protected
        expected = max(1,min(int(eligible.sum())-1,int(np.floor(.4*eligible.sum()+.5))))
        assert info['hidden_tokens'] == expected == int(hidden.sum())
        assert not (hidden & ~valid[:16]).any() and not (hidden & protected).any()
        assert np.all(protected[:,[JOINT_IDS.index(j) for j in REFERENCE_JOINTS]] ==
                      valid[:16][:,[JOINT_IDS.index(j) for j in REFERENCE_JOINTS]])
        for j in range(len(JOINT_IDS)):
            available = np.flatnonzero(valid[:16,j])
            if len(available): assert protected[available[-1],j]
        # Natural missingness is separate from artificial masking.
        missing_changed = original[:16].copy(); missing_changed[~valid[:16]] = np.nan
        hidden_missing, protected_missing, _ = design_prefix_mask(
            missing_changed, valid[:16], times[:16], policy, ratio=.4, seed=27)
        np.testing.assert_array_equal(hidden, hidden_missing)
        np.testing.assert_array_equal(protected, protected_missing)
        # A fixed mask removes hidden values from the visible model input.
        visible = valid[:16] & ~hidden
        hidden_changed = original[:16].copy(); hidden_changed[hidden] = 1e6
        np.testing.assert_array_equal(np.where(visible[...,None], original[:16], 0.),
                                      np.where(visible[...,None], hidden_changed, 0.))
        # Neither future coordinates nor future validity enter the mask API.
        changed = original.copy(); changed[16:] = -1e6
        changed_valid = valid.copy(); changed_valid[16:] = ~changed_valid[16:]
        hidden_future, protected_future, _ = design_prefix_mask(
            changed[:16], changed_valid[:16], times[:16], policy, ratio=.4, seed=27)
        np.testing.assert_array_equal(hidden, hidden_future)
        np.testing.assert_array_equal(protected, protected_future)
        np.testing.assert_array_equal(future_queries(), queries)
        assert not (hidden & ~eligible).any()
        validation_rows.append({'source_id':row.source_id,'window_id':row.window_id,
            'policy':policy,'eligible_tokens':int(eligible.sum()),'hidden_tokens':int(hidden.sum()),
            'realized_fraction':float(hidden.sum()/eligible.sum()),
            'partial_final_block':info['partial_final_block'],'status':'passed'})
        hidden_counts.append(int(hidden.sum()))
    assert len(set(hidden_counts)) == 1, 'Policies must have exactly matched budgets.'
checks = ['exact rounded token budget','equal budgets across four policies',
          'reference and latest-observation protection','natural-missing-value invariance',
          'hidden-value isolation after fixing a mask','future-coordinate and future-validity invariance',
          'fixed future query geometry','cache and metadata decoder-time agreement',
          'one-based annotation to zero-based decoded-index agreement']
report = {'identity':'iteration02-mask-design-lab-v1','status':'complete design validation; no fitted mask experiment',
    'sources_selected':len(representatives),'sources_passed':len(representatives)-len(skipped),
    'policy_checks_passed':len(validation_rows),'skipped':skipped,'policies':list(POLICIES),
    'requested_fraction':.4,'seed':27,'motion_policy_uniform_budget_floor':.2,
    'query_tokens':int(queries.sum()),'query_coordinate_count':int(queries.sum()*2),
    'checks':checks,'results':validation_rows,'example_counts':counts,
    'example_source_id':selected.source_id,'example_window_id':selected.window_id,
    'inputs':{str(p.relative_to(PROJECT)):digest(p) for p in [
        DATA/'windows.npz',DATA/'windows.csv',RUN/'cohort.csv',RUN/'results.json',
        PROJECT/'worldmodels/mask_design.py',PROJECT/'docs/draft02/posthoc_score_audit.json']},
    'source_cache_sha256':source_hashes,
    'uncertainty_note':'Point estimates recomputed; conditional intervals read from the verified draft02 audit.',
    'limitations':['This is design validation on one retained window per source, not model training or performance evaluation.',
      'Motion-aware mask choice can encode information about values it later hides; a future study needs mask-pattern-only controls.',
      'Exact budgets may truncate the final block or anatomical tube.',
      'Future query positions are fixed; observed future support is used only when computing a loss or score.']}
(OUT/'mask_lab_validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(f"Passed {len(validation_rows)} policy checks across {report['sources_passed']} sources; explicitly skipped {len(skipped)} sources.")
if skipped: display(pd.DataFrame(skipped))
display(pd.DataFrame(validation_rows).groupby('policy').agg(
    sources=('source_id','nunique'), hidden_min=('hidden_tokens','min'), hidden_max=('hidden_tokens','max'),
    fraction_min=('realized_fraction','min'), fraction_max=('realized_fraction','max'),
    partial_blocks=('partial_final_block','sum')))
print('Saved:', (OUT/'mask_lab_validation.json').relative_to(PROJECT))
'''


def build():
    nb=nbf.v4.new_notebook()
    nb.metadata={'kernelspec':{'display_name':'Med-WM iteration02 (Python 3.12)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}
    cells=[
    ('markdown',"""# Read the evidence and design the next masking comparison

This notebook asks two practical questions. What do the completed gait forecasts establish, and how can the next study compare masking policies fairly? It reads real, immutable GAVD results and tests mask construction on actual prepared windows. It does not train a new model or claim that a masking alternative improves performance.

We first verify artifact identities and rebuild the annotation inventory. The full inventory includes all labels, while the evaluated cohort uses normal, Parkinson's, stroke, cerebral palsy, and myopathic annotations. Source videos are grouping proxies. They are not verified independent patients. Run this notebook with the experiment's Python 3.12 environment and the local artifacts described in the README."""),
    ('code',SETUP),
    ('markdown',"""## Read forecast skill and anomaly ranking separately

Persistence repeats the latest observed pose. Its error therefore reflects displacement away from that pose. A learned forecast can improve on persistence while its raw error remains small for restricted or repeatable abnormal movement. The excess-error score subtracts persistence error from learned forecast error on the same observed targets.

The next cell independently rebuilds normal-source forecast means and all five original anomaly rankings. AUROC is the chance that an abnormal source receives a larger score than a normal source. We preserve the original score directions and show negative persistence only as a post hoc low-movement diagnostic. The within-fold comparison avoids directly comparing scores from different fitted models, although their training sources still overlap. Its saved interval is conditional on those fits and does not quantify patient-population uncertainty."""),
    ('code',EVIDENCE),
    ('markdown',"""## Hide observations while keeping targets and information access clear

A natural missing observation is not an artificial mask. We hide only valid prefix tokens, where one token contains the two coordinates of one joint at one time. Observed shoulders and hips remain available as reference joints, and each joint's latest observation remains available as an anchor. The four policies use the same integer budget, obtained by rounding 40% of eligible tokens. A final time block or anatomical tube may be truncated to meet that budget.

Uniform masking selects individual tokens. Time blocks prioritize groups of four adjacent samples. Anatomical tubes prioritize a limb group over four samples. Motion-balanced masking uses valid adjacent prefix observations and actual decoder time gaps; it reserves at least 20% of the hidden budget for uniform draws. It does not treat unknown motion as zero motion.

The purple future queries always request the same eight distal joints at eight future times. Their positions do not depend on future values or visibility. Future validity determines which requested coordinates can later contribute to an observed loss. The first retained window in source, sequence, and frame order supplies the illustration, without choosing an especially favorable example."""),
    ('code',MASKS),
    ('markdown',"""## Try to break the information boundary

We now select one retained window from each of the 91 source videos and test every policy. Actual decoder timestamps must agree with the metadata and increase through the prefix; invalid times are recorded as explicit skips. We verify exact budgets, protected joints, natural-missing-value invariance, and future isolation.

Changing a value hidden by a fixed mask must not change the visible model input. That is different from rebuilding a motion-aware mask after changing its input. A motion-aware policy can inspect an observed value and then choose to hide it, so the resulting mask pattern can carry information. This demonstration does not eliminate that issue. The next fitted study must include a mask-pattern-only control and a randomized-policy comparison."""),
    ('code',VALIDATION),
    ('markdown',"""## Keep three evidence states separate

**Measured results.** The completed run supports useful normal-source forecasts and a favorable association between excess error and abnormal annotations. Current-posture forecasting is stronger than bilateral history, and the post hoc low-movement diagnostic remains competitive for anomaly ranking. These findings motivate the next comparison without establishing clinical diagnosis or model superiority.

**Executed design demonstration.** This notebook constructs and checks four masks using real prepared observations. It confirms their information boundaries and matched budgets for the recorded sample of sources. It does not fit a masked model, evaluate learned representations, or estimate an anomaly-detection gain from masking.

**Proposed research.** The next protocol compares compact models under fixed masks, matched training budgets, source-grouped selection, and normal-only calibration. The proposed two-sided, motion-conditioned score requires new fitting and independent calibration groups. It is not implemented or validated by this notebook. Read `notes/aaai-worldmodels/05-iterate-02-INSTR.md` for the full experiment ladder and review gates.

GAVD supports a heterogeneous public-video benchmark. GAVD alone does not establish patient-disjoint, site-disjoint, or prospective clinical generalization. Additional independent normal participants and external clinical observations remain more valuable than simply multiplying overlapping windows.""")]
    for i,(kind,source) in enumerate(cells):
        c=nbf.v4.new_code_cell(source) if kind=='code' else nbf.v4.new_markdown_cell(source)
        c['id']=hashlib.sha256((NAME+str(i)+source).encode()).hexdigest()[:12]
        nb.cells.append(c)
    nbf.write(nb,PROJECT/NAME)
    return PROJECT/NAME


def execute(path):
    assert sys.version_info[:2]==(3,12), 'Execute using the project Python 3.12 environment.'
    EXECUTED.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    kernel=EXECUTED/'kernel';kernel.mkdir(exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'MedWM iteration02 verified Python 3.12','language':'python'},indent=2)+'\n')
    manager=KernelSpecManager();manager.install_kernel_spec(str(kernel),kernel_name='medwm-iteration02-verified',user=True,replace=True)
    nb=nbf.read(path,as_version=4);start=time.monotonic()
    report={'status':'running','notebook':path.name,'source_sha256':digest(path),'generator_sha256':digest(__file__),'interpreter':sys.executable,'python_version':sys.version,'cell_count':len(nb.cells),'code_cells':sum(c.cell_type=='code' for c in nb.cells),'inputs':{}}
    for p in [PROJECT/'artifacts/runs/restart-dynamics-v1/results.json',PROJECT/'artifacts/runs/restart-dynamics-v1/artifact_hashes.json',PROJECT/'docs/draft02/posthoc_score_audit.json',PROJECT/'worldmodels/mask_design.py']:
        report['inputs'][str(p.relative_to(PROJECT))]=digest(p)
    error=None
    try:
        NotebookClient(nb,timeout=600,kernel_name='medwm-iteration02-verified',resources={'metadata':{'path':str(PROJECT)}}).execute()
        report['status']='complete'
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc)[-5000:]);error=exc
    target=EXECUTED/path.name;nbf.write(nb,target)
    report.update(seconds=time.monotonic()-start,output_sha256=digest(target),output=str(target.relative_to(PROJECT)))
    report['artifacts']={str(p.relative_to(PROJECT)):digest(p) for p in [OUT/'mask_lab_validation.json',OUT/'mask_design_examples.svg',OUT/'mask_design_examples.png',OUT/'motion_reference_audit.svg',OUT/'motion_reference_audit.png'] if p.exists()}
    (OUT/'notebook_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if error:raise RuntimeError('Notebook failed; see saved output and execution record.') from error

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    path=build()
    if args.execute:execute(path)
    else:print(path)
