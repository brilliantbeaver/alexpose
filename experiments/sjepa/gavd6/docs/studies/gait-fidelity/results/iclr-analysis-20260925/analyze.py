"""Reproduce the paper-facing analysis from the immutable downloaded evidence.

Run: .venv/bin/python docs/studies/gait-fidelity/results/iclr-analysis-20260925/analyze.py
No training, source-file modifications, network access, or participant selection.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'gait-iclr-matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'gait-iclr-cache'))
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
DATA = ROOT/'outputs/iclr'
FIG = OUT/'figures'
FIG.mkdir(exist_ok=True)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def means(frame, extra=()):
    metrics = frame.select_dtypes('number').columns.drop('seed', errors='ignore').tolist()
    return frame.groupby([*extra, 'method', 'seed'])[metrics].mean().groupby([*extra, 'method']).mean()


def contrast(frame, candidate, comparator, metric):
    p = frame[frame.method.isin([candidate, comparator])].pivot(
        index=['canonical_person_id', 'seed'], columns='method', values=metric)
    assert not p.isna().any().any()
    d = (p[comparator]-p[candidate]).unstack('seed').sort_index()
    x = d.to_numpy()
    rng = np.random.default_rng(731)
    crossed, people = [], []
    for _ in range(2000):
        i = rng.integers(len(x), size=len(x))
        j = rng.integers(x.shape[1], size=x.shape[1])
        people.append(float(x[i].mean()))
        crossed.append(float(x[i][:, j].mean()))
    person = x.mean(axis=1)
    half = t.ppf(.975, len(person)-1)*person.std(ddof=1)/np.sqrt(len(person))
    return dict(candidate=candidate, comparator=comparator, metric=metric,
                improvement=float(x.mean()),
                crossed_person_seed_ci95=np.quantile(crossed, [.025, .975]).tolist(),
                person_conditional_ci95=np.quantile(people, [.025, .975]).tolist(),
                person_averaged_t_ci95=[float(person.mean()-half), float(person.mean()+half)],
                people_favoring_candidate=int((person>0).sum()),
                per_seed_improvement={str(k):float(v) for k,v in d.mean().items()},
                person_effects={str(k):float(v) for k,v in d.mean(axis=1).items()},
                interpretation='Descriptive development comparison; secondary analyses unadjusted.')


inventory_path = max(DATA.glob('transfer-inventory-*.json'))
inventory = read_json(inventory_path)
entries = [e for e in inventory['files'] if e['status']=='selected']
assert all((DATA/e['destination']).stat().st_size == e['bytes'] and
           digest(DATA/e['destination']) == e['sha256'] for e in entries)
paths = {r: DATA/r/('development/evaluation' if r=='readout-repair' else 'evaluation')
         for r in ['walking-core', 'jepa-response', 'readout-repair']}
frames = {r:pd.read_csv(p/'per-person.csv') for r,p in paths.items()}
all_means = {r:means(f) for r,f in frames.items()}
repair_extractors = pd.read_csv(paths['readout-repair']/'per-person-by-extractor.csv')
repair_means = means(repair_extractors, ('extractor',))
vit = repair_extractors[repair_extractors.extractor.eq('vitpose_base')]
vm = repair_means.loc['vitpose_base']
people_sets = [set(f.canonical_person_id) for f in frames.values()]
assert people_sets[0] == people_sets[1] == people_sets[2] and len(people_sets[0])==14
for frame in [*frames.values(), vit]:
    assert not frame.duplicated(['method','seed','canonical_person_id']).any()
    assert frame.groupby('method').size().eq(42).all()
    assert set(frame.seed)=={17,29,43}

retained = []
for old,new in [('walking-core','jepa-response'), ('jepa-response','readout-repair')]:
    keys = ['method','seed','canonical_person_id']
    shared = set(frames[old].method)&set(frames[new].method)
    a = frames[old][frames[old].method.isin(shared)].set_index(keys).sort_index()
    b = frames[new][frames[new].method.isin(shared)].set_index(keys).sort_index()
    cols = sorted(set(a.select_dtypes('number').columns)&set(b.select_dtypes('number').columns))
    assert a.index.equals(b.index)
    assert np.allclose(a[cols], b[cols], equal_nan=True, atol=1e-12, rtol=0)
    retained.append(dict(parent=old, child=new, methods=len(shared), person_seed_rows=len(a)))

saved = {r:read_json(p/'comparisons.json') for r,p in paths.items()}
checks = []
for run in ['walking-core','jepa-response']:
    for metric, entry in saved[run].items():
        check = contrast(frames[run], entry['candidate'], entry['comparator'], metric)
        for key in ['improvement','person_conditional_ci95','crossed_person_seed_ci95']:
            assert np.allclose(check[key], entry[key], atol=1e-10, rtol=0)
        checks.append(dict(run=run,metric=metric,matched=True))
for metric, entry in saved['readout-repair']['primary'].items():
    check = contrast(vit, entry['candidate'], entry['comparator'], metric)
    for key in ['improvement','person_averaged_t_ci95']:
        assert np.allclose(check[key], entry[key], atol=1e-10, rtol=0)
    assert np.allclose(check['crossed_person_seed_ci95'],
                       entry['crossed_bootstrap']['crossed_person_seed_ci95'], atol=1e-10, rtol=0)
    checks.append(dict(run='readout-repair',metric=metric,matched=True))

for frame in [frames['jepa-response'], frames['readout-repair'], repair_extractors]:
    assert np.allclose(frame.response_error,
                       frame.response_success_contribution+frame.response_failure_contribution)
    assert np.allclose(frame.response_failure_contribution, 720*frame.response_failure_rate)
for frame in [frames['readout-repair'], repair_extractors]:
    assert np.allclose(frame.waveform_error,
                       frame.waveform_success_contribution+frame.waveform_failure_contribution)
    assert np.allclose(frame.waveform_failure_contribution, 180*frame.waveform_failure_rate)

for name,m in all_means.items():
    m.to_csv(OUT/f'{name}-method-means.csv')
repair_means.to_csv(OUT/'repair-extractor-means.csv')

# The exported held/nonheld strata do not have equal numbers of state levels.
# Preserve 5+10 vs 15 (2:1) for responses, and four nonheld endpoint records
# versus one held record (4:1) for waveform/coordinate metrics. Reference support
# is complete in this packet; assert that these marginals reproduce its totals.
condition_frame = pd.read_csv(paths['jepa-response']/'response-by-condition-person.csv')
condition_metrics = ['response_error','zero_response_error','response_failure_contribution',
                     'response_success_contribution','waveform_error','synthetic_all_nle']
condition_means = {}
for dimension in ['observation','held_intervention']:
    weighted = condition_frame.copy()
    for metric in condition_metrics:
        nonheld_weight = 4 if metric in ['waveform_error','synthetic_all_nle'] else 2
        weight = np.where(weighted.held_intervention,1.,nonheld_weight)
        weighted[metric+'_numerator'] = weighted[metric]*weight
        weighted[metric+'_denominator'] = weight
    cols = [m+s for m in condition_metrics for s in ['_numerator','_denominator']]
    sums = weighted.groupby(['method','seed','canonical_person_id',dimension])[cols].sum()
    for metric in condition_metrics:
        sums[metric] = sums[metric+'_numerator']/sums[metric+'_denominator']
    result = sums[condition_metrics].groupby(['method',dimension]).mean()
    condition_means[dimension] = result
    result.to_csv(OUT/f'response-by-{dimension}.csv')
assert np.allclose(condition_means['observation'].groupby('method').mean(),
                   all_means['jepa-response'][condition_metrics], atol=1e-10, rtol=0)

families = [('P-direct-none','Direct coordinates'), ('M-coordinate-graph_time','Coordinate pretraining'),
            ('M-paired_jepa-graph_time','Paired JEPA'), ('I-initialized-none','Untrained features'),
            ('I-shuffled_jepa-graph_time','Shuffled JEPA')]
secondary = {}
for metric in ['response_error','waveform_error','synthetic_all_nle','displacement_nle']:
    secondary['direct_vs_unchanged_'+metric] = contrast(
        frames['walking-core'],'P-direct-none-base','unchanged',metric)
for method,label in families:
    secondary['core_'+method+'_base_vs_scalar_waveform'] = contrast(
        frames['walking-core'],method+'-base',method+'-paired_change','waveform_error')
(OUT/'exploratory-comparisons.json').write_text(json.dumps(secondary,indent=2)+'\n')

diagnostics = []
for path in sorted((DATA/'jepa-response/diagnostics').glob('*/diagnostics.json')):
    d = read_json(path)
    seed = int(path.parent.name.rsplit('-',1)[1])
    family = path.parent.name.rsplit('-seed-',1)[0]
    for branch, probe in d['probes'].items():
        diagnostics.append(dict(family=family, seed=seed, branch=branch,
                                mse=probe['mse'], zero_change_mse=probe['zero_change_mse'],
                                rows=probe['rows'],people=len(probe['per_person']),
                                fixed_slot_variance=d['statistics'][branch]['fixed_slot_variance'],
                                positive_status=d['probe_result'],source=str(path.relative_to(ROOT))))
diagnostics = pd.DataFrame(diagnostics)
diagnostics.to_csv(OUT/'diagnostic-probes.csv',index=False)

calibrations = []
ledger = read_json(DATA/'readout-repair/ledger.json')
for name,record in ledger['completed'].items():
    if not name.startswith('calibrate-'): continue
    r=record['result']
    support={s:sum(b['angle_gradient_support'][s]['nonzero_angle_gradients'] for b in r['batches'])/
               sum(b['angle_gradient_support'][s]['eligible_angle_entries'] for b in r['batches'])
             for s in ['scalar','dense']}
    calibrations.append(dict(variant=r['representation_variant'],seed=r['seed'],
                             scalar_to_coordinate_gradient=r['gradients']['scalar']['rms']/r['gradients']['coordinate']['rms'],
                             scalar_support=support['scalar'],dense_support=support['dense'],
                             dense_coefficient=r['coefficients']['dense_change']))
pd.DataFrame(calibrations).to_csv(OUT/'readout-calibration.csv',index=False)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                     'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titlelocation':'left','savefig.facecolor':'white','figure.facecolor':'white',
                     'svg.fonttype':'none','pdf.fonttype':42})
BLUE, ORANGE, TEAL, GRAY = '#275d82','#ba613a','#267970','#777777'


def finish(fig, name):
    for ext in ['png','svg','pdf']:
        fig.savefig(FIG/f'{name}.{ext}', dpi=190, bbox_inches='tight')
    plt.close(fig)


# Descriptive matched-objective means. Lines connect the same method family.
fig,axes=plt.subplots(1,3,figsize=(12.4,4.4),sharey=True)
cm=all_means['walking-core']
for ax,metric,title in zip(axes,['synthetic_all_nle','response_error','waveform_error'],
                          ['Coordinate error (NLE)','Response error (degrees)','Waveform error (degrees)']):
    for y,(method,label) in enumerate(families):
        a,b=cm.loc[method+'-base',metric],cm.loc[method+'-paired_change',metric]
        ax.plot([a,b],[y,y],color='#b9bdc3',lw=2,zorder=1)
        ax.scatter(a,y,color=BLUE,s=55,label='Base objective' if y==0 else None,zorder=3)
        ax.scatter(b,y,color=ORANGE,s=55,marker='s',label='Original paired-change objective' if y==0 else None,zorder=3)
    ax.set_title(title);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    ax.set_xlim(left=0);ax.set_yticks(range(len(families)))
axes[0].set_yticklabels([label for _,label in families]);axes[0].invert_yaxis()
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.53,-.015))
fig.suptitle('Walking core: the original change objective increases waveform error in every family',
             x=.03,ha='left',fontsize=13,y=1.01)
fig.tight_layout(rect=(0,.08,1,.95));finish(fig,'01-core-objectives')

# Each row is a different declared scientific comparison, not a meta-analysis.
fig,ax=plt.subplots(figsize=(9.6,3.9))
entries_primary=[saved['walking-core']['response_error'],saved['jepa-response']['response_error'],
                 saved['readout-repair']['primary']['waveform_error']]
labels=['Core: paired JEPA vs direct\nPaired-change readouts; response; 3 estimators',
        'Response: delta vs endpoint JEPA\nPaired-change readouts; response; 3 estimators',
        'Repair: dense vs low scalar, delta JEPA\nViTPose only; waveform error']
for y,e in enumerate(entries_primary):
    lo,hi=e.get('crossed_person_seed_ci95',e.get('crossed_bootstrap',{}).get('crossed_person_seed_ci95'))
    mid=e['improvement'];ax.errorbar(mid,y,xerr=[[mid-lo],[hi-mid]],fmt='o',color=BLUE,capsize=4,lw=2,
                                   label='Crossed person/seed 95% interval' if y==0 else None)
    if y==2:
        lo,hi=e['person_averaged_t_ci95']
        ax.errorbar(mid,y+.17,xerr=[[mid-lo],[hi-mid]],fmt='s',color=ORANGE,capsize=4,lw=2,
                    label='Repair primary: person t interval')
ax.axvline(0,color=GRAY,ls='--',lw=1);ax.set_yticks(range(3),labels);ax.invert_yaxis()
ax.set_xlim(-1.45,2.2);ax.set_ylim(2.65,-.45);ax.set_xlabel('Comparator minus candidate error (degrees); positive favors candidate')
ax.set_title('None of the three declared primary advantages is resolved')
ax.grid(axis='x',alpha=.15);ax.legend(loc='lower center',bbox_to_anchor=(.36,-.46),frameon=False,ncol=1)
fig.tight_layout();finish(fig,'02-primary-contrasts')

# Repair primary scope is ViTPose, and must not be mixed with pooled figures.
fig,axes=plt.subplots(1,2,figsize=(11.8,4.5),sharey=True)
objectives=[('base','Base readout'),('paired_change','Original scalar loss'),
            ('scalar_low','Scalar loss × 0.1'),('dense_change','Dense temporal loss')]
for ax,metric,title in zip(axes,['waveform_error','response_error'],
                          ['Waveform error (degrees)','Response error (degrees)']):
    for variant,color,offset,marker in [('jepa_delta_v1',BLUE,-.09,'o'),('jepa_endpoint_v1',TEAL,.09,'s')]:
        for y,(obj,label) in enumerate(objectives):
            method=(f'F-response-{variant}-graph_time-{obj}' if obj in ['base','paired_change']
                    else f'R-repair-{variant}-{obj}')
            ax.scatter(vm.loc[method,metric],y+offset,c=color,marker=marker,s=52,
                       label=('Delta JEPA' if variant=='jepa_delta_v1' else 'Endpoint JEPA') if y==0 else None)
    ax.axvline(vm.loc['P-direct-none-base',metric],color=ORANGE,lw=1.8,ls='--',label='Direct / base')
    ax.set_title(title);ax.set_xlabel('Lower is better');ax.grid(axis='x',alpha=.15)
    ax.set_xlim(9 if metric=='response_error' else 11,16 if metric=='response_error' else 25)
    ax.set_yticks(range(4))
axes[0].set_yticklabels([label for _,label in objectives]);axes[0].invert_yaxis()
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,frameon=False,loc='lower center')
fig.suptitle('Readout repair on ViTPose: most waveform recovery also occurs with lower scalar weight',
             x=.03,ha='left',fontsize=13,y=1.01)
fig.tight_layout(rect=(0,.09,1,.95));finish(fig,'03-repair-readouts')

fig,axes=plt.subplots(1,2,figsize=(12.8,5),gridspec_kw={'width_ratios':[1.3,1]})
rm=all_means['jepa-response']
models=[('P-direct-none-base','Direct / base'),('I-initialized-none-base','Untrained features / base'),
        ('I-shuffled_jepa-graph_time-base','Shuffled JEPA / base'),
        ('M-paired_jepa-graph_time-base','Paired JEPA / base'),
        ('F-response-jepa_endpoint_v1-graph_time-paired_change','Endpoint JEPA / change'),
        ('F-response-jepa_delta_v1-graph_time-paired_change','Delta JEPA / change')]
for ax,table,labels in [(axes[0],rm.loc[[m for m,_ in models]],[label for _,label in models]),
                       (axes[1],repair_means.xs('P-direct-none-base',level='method').loc[['hrnet_w32','rtmpose_m','vitpose_base']],
                        ['HRNet','RTMPose','ViTPose (held extractor)'])]:
    y=np.arange(len(table))
    ax.barh(y,table.response_success_contribution,color=BLUE,label='Successful-output contribution')
    ax.barh(y,table.response_failure_contribution,left=table.response_success_contribution,color=ORANGE,
            label='Invalid-measurement penalty contribution')
    ax.axvline(float(table.zero_response_error.iloc[0]),color=TEAL,ls='--',lw=2,label='Predict zero response')
    for yy,value in zip(y,table.response_error):ax.text(value+.12,yy,f'{value:.2f}',va='center',fontsize=9)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,12.6);ax.set_xlabel('Response error (degrees)')
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
axes[0].set_title('All extractors; selected controls and JEPA variants')
axes[1].set_title('Direct / base, separated by extractor')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=1,frameon=False,bbox_to_anchor=(.55,-.1))
fig.suptitle('Small invalid-measurement rates make a substantial contribution to response error',
             x=.03,ha='left',fontsize=13,y=1.02)
fig.tight_layout(rect=(0,.1,1,.96));finish(fig,'04-response-decomposition')

fig,ax=plt.subplots(figsize=(9.3,4.6))
diag_families=[('parent-pretrain-paired_jepa-graph_time','Paired JEPA'),
               ('parent-pretrain-shuffled_jepa-graph_time','Shuffled JEPA'),
               ('child-pretrain-response-jepa_endpoint_v1-graph_time','Endpoint JEPA'),
               ('child-pretrain-response-jepa_delta_v1-graph_time','Delta JEPA')]
dm=diagnostics.groupby(['family','branch']).mse.mean()
for branch,color,offset,label in [('deployment_encoder',BLUE,-.15,'Encoder features from estimated poses'),
                                 ('deployment_teacher',TEAL,.15,'Teacher features from reference poses')]:
    ax.barh(np.arange(4)+offset,[dm.loc[(f,branch)] for f,_ in diag_families],height=.27,color=color,label=label)
zero=float(diagnostics.zero_change_mse.iloc[0])
ax.axvline(zero,color=GRAY,ls='--',label=f'Zero-change predictor ({zero:.2f})')
ax.set_yticks(range(4),[label for _,label in diag_families]);ax.invert_yaxis()
ax.set_xlabel('Linear-probe mean squared response error (degrees²); lower is better')
ax.set_title('A positive teacher probe does not establish useful deployed encoder features')
ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
ax.legend(loc='lower center',bbox_to_anchor=(.49,-.47),frameon=False)
fig.tight_layout();finish(fig,'05-feature-probes')

verification=dict(source_inventory=str(inventory_path.relative_to(ROOT)),
                  source_inventory_sha256=digest(inventory_path), verified_files=len(entries),
                  verified_bytes=sum(e['bytes'] for e in entries),
                  source_missing=[e['destination'] for e in inventory['files'] if e['status']=='missing'],
                  people=sorted(people_sets[0]),seeds=[17,29,43],retained_predictions=retained,
                  saved_comparison_recalculations=checks,additive_failure_decompositions_verified=True,
                  condition_reaggregation_matches_published_means=True,
                  statistic_protocol='Saved primary comparisons reproduced; new secondary bootstrap uses 2000 draws, RNG731.',
                  scope='Synthetic development evidence. No raw prediction arrays or independent confirmation available.')
(OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
if (OUT/'README.md').exists():
    import mistune
    markdown = mistune.create_markdown(plugins=['table'])
    article = markdown((OUT/'README.md').read_text())
    html = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>What the completed gait-fidelity experiments establish</title>
<style>
body{margin:0;background:#fff;color:#202731;font-family:Georgia,"Times New Roman",serif}
main{max-width:1050px;margin:56px auto 80px;padding:0 30px}
h1{font:600 2.25rem/1.18 system-ui,sans-serif;max-width:880px;margin:0 0 28px}
p{font-size:1.075rem;line-height:1.7;margin:0 0 22px}
a{color:#235e83;text-decoration-thickness:1px;text-underline-offset:3px}
strong{font-weight:700} em{color:#404955}
table{width:100%;border-collapse:collapse;font:0.94rem/1.45 system-ui,sans-serif;margin:24px 0 30px}
th{background:#f0f4f6;color:#233b4b;font-weight:600;text-align:left}
th,td{padding:11px 12px;border-bottom:1px solid #dce2e5;vertical-align:top}
tr:nth-child(even) td{background:#f9fafb}
img{display:block;width:100%;height:auto;margin:28px 0 8px}
code{font:0.88em ui-monospace,monospace;background:#f2f4f6;padding:1px 4px;overflow-wrap:anywhere}
@media(max-width:720px){main{margin-top:30px;padding:0 18px}h1{font-size:1.8rem}p{font-size:1rem}table{font-size:.79rem}th,td{padding:7px 6px}}
@media print{main{max-width:none;margin:0;padding:0}body{font-size:10pt}p{font-size:10pt;line-height:1.5}h1{font-size:22pt}table{font-size:8pt}img,table{break-inside:avoid}a{color:inherit}}
</style></head><body><main>'''+article+'</main></body></html>\n'
    (OUT/'index.html').write_text(html)
print(json.dumps({'verified_files':len(entries),'people':len(people_sets[0]),
                  'saved_comparison_metrics_reproduced':len(checks),'figures':5,'output':str(OUT)},indent=2))
