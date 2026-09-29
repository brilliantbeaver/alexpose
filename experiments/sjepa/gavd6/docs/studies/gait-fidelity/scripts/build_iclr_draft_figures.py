#!/usr/bin/env python3
"""Build evidence-grounded figures for new ICLR scientific drafts (25 Sep 2026).

All values come from the transferred outputs/iclr development evidence. No raw
images or trajectories are fabricated. The measurement example is an explicitly
aggregated, real participant curve, chosen by a stated, result-independent rule.
Run from any directory: .venv/bin/python <this script>
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'docs/studies/gait-fidelity/images/iclr-draft-20260925'
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {}
AUDIT = {'scope': 'Development evidence; no clinical or independent confirmation.',
         'date': '2026-09-25', 'figures': {}, 'values': {}}
INK = '#26313d'
GRAY = '#64717d'
LIGHT = '#dde3e8'
BLUE = '#0072b2'
ORANGE = '#b96b18'
TEAL = '#168574'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
    'axes.titlesize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 9,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': GRAY,
    'xtick.color': INK, 'ytick.color': INK, 'svg.fonttype': 'none',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'xtick.major.width': .65,
    'ytick.major.width': .65, 'lines.linewidth': 1.5,
    'savefig.bbox': None,
})

def source(relative, kind='csv'):
    p = ROOT / relative
    SOURCES[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p) if kind == 'csv' else json.loads(p.read_text())

def means(frame):
    # Explicitly equal-weight fitted seeds within each person, then people.
    return frame.groupby(['method', 'canonical_person_id'], sort=False).mean(numeric_only=True).groupby('method').mean(numeric_only=True)

def clean(ax, xgrid=False):
    ax.set_axisbelow(True)
    if xgrid:
        ax.grid(axis='x', color=LIGHT, lw=.55)
    ax.tick_params(length=3)

def save(fig, stem, caption, input_files):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    outside = []
    from matplotlib.text import Text
    for obj in fig.findobj(match=Text):
        if obj.get_visible() and obj.get_text().strip():
            box = obj.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
                outside.append(obj.get_text())
    if outside:
        raise AssertionError(f'{stem}: text outside figure: {outside}')
    for suffix in ('svg', 'pdf', 'png'):
        fig.savefig(OUT / f'{stem}.{suffix}', dpi=200, metadata={'Creator':'Matplotlib; build_iclr_draft_figures.py'} if suffix != 'png' else None)
        if suffix == 'svg':
            # Matplotlib bundles DejaVu, but a browser need not have it installed.
            # Preserve editable text while avoiding a serif fallback in HTML/PDF.
            svg = OUT / f'{stem}.svg'
            svg.write_text(svg.read_text().replace(
                "font-family: 'DejaVu Sans'",
                "font-family: 'DejaVu Sans', Arial, sans-serif"))
    AUDIT['figures'][stem] = {'size_inches': fig.get_size_inches().tolist(),
        'caption': caption, 'source_files': input_files, 'text_outside_canvas': outside}
    plt.close(fig)

core_path = 'outputs/iclr/walking-core/evaluation/per-person.csv'
response_path = 'outputs/iclr/jepa-response/evaluation/per-person.csv'
curves_path = 'outputs/iclr/jepa-response/evaluation/response-curves-person.csv'
repair_path = 'outputs/iclr/readout-repair/development/evaluation/per-person-by-extractor.csv'
core = source(core_path); response = source(response_path); curves = source(curves_path)
repair = source(repair_path)
cm, rm = means(core), means(response)
pm = means(repair[repair.extractor.eq('vitpose_base')])
assert response.canonical_person_id.nunique() == 14 and response.seed.nunique() == 3
assert np.allclose(response.groupby('method').zero_response_error.mean(), 5.810829989789)

# Actual data example: fixed first person, conventional observation, no outcome search.
person = sorted(curves.canonical_person_id.unique())[0]
selected = curves[curves.canonical_person_id.eq(person) & curves.camera_id.eq('side') &
                  curves.observation.eq('clear') & curves.physical_state.eq('original')]
methods = [('P-direct-none-base', 'Direct / base', BLUE, 's', '--'),
           ('F-response-jepa_delta_v1-graph_time-paired_change', 'Delta JEPA / change', ORANGE, '^', '-.')]
a = selected[selected.method.eq(methods[0][0])].sort_values('movement_level_deg')
assert len(a) == 4 and np.allclose(a.prediction_coverage, 1)
for method, *_ in methods:
    rows = selected[selected.method.eq(method)].sort_values('movement_level_deg')
    assert np.allclose(rows.reference_change, a.reference_change)
    assert np.allclose(rows.prediction_coverage, 1)
AUDIT['values']['actual_measurement_example'] = {
    'selection_rule': 'Lexicographically first canonical person; side camera, clear observation, original physical state.',
    'person': person, 'camera': 'side', 'observation': 'clear', 'physical_state': 'original',
    'aggregation': 'Published person curves average naming/extractor conditions and windows within raw motions, raw motions within person, then three fitted seeds.',
    'is_raw_sample': False,
    'records': selected[selected.method.isin([x[0] for x in methods])].to_dict('records')}
example_caption = (
    'Actual aggregate measurement example, not a raw pose sequence. The displayed person is '
    f'{person}, selected as the first canonical ID in lexicographic order, in a clear side view '
    'with original physical orientation. A is right-minus-left image-plane knee excursion; '
    'ΔA compares the edited and original motion. Curves are published person means across '
    'source motions, naming conditions, three pose estimators and three fitted seeds. Each dose '
    'contains 135 repeated contrasts per method, all with successful predictions, not 135 people. '
    'The 15° edit was excluded from training. The nominal body-model edit on the horizontal axis '
    'is different from the measured reference change on the vertical axis. This single-person '
    'example illustrates attenuation and does not estimate population accuracy. Raw frames, '
    'reference trajectories and restored trajectories are absent from the local evidence transfer.')
for compact in (False, True):
    fig = plt.figure(figsize=(6.8, 1.72 if compact else 2.22))
    ax = fig.add_axes([.10, .29 if compact else .23, .59, .51 if compact else .51])
    clean(ax)
    ax.axhline(0, color=LIGHT, lw=.7)
    ax.plot(a.movement_level_deg, a.reference_change, color=INK, marker='o', ms=3.7, label='Reference')
    ax.text(16.0, float(a.reference_change.iloc[-1]), 'Reference', va='center', fontsize=9, color=INK, clip_on=False)
    for method, label, color, marker, style in methods:
        rows = selected[selected.method.eq(method)].sort_values('movement_level_deg')
        ax.plot(rows.movement_level_deg, rows.predicted_change, color=color, marker=marker, ms=3.7, ls=style)
        ax.text(16.0, float(rows.predicted_change.iloc[-1]), label, va='center', fontsize=9, color=color, clip_on=False)
    ax.set(xlim=(-.25,15.4), ylim=(-.5,10.5), xticks=[0,5,10,15], yticks=[0,5,10])
    ax.set_xlabel('Nominal knee-flexion edit (°)', labelpad=3)
    ax.set_ylabel('Change in A, ΔA (°)', labelpad=5)
    fig.text(.10,.95, 'One development participant: measured and recovered change',
             fontsize=9.5, weight='bold', va='top')
    if not compact:
        fig.text(.10,.865, f'{person}  ·  Side view, clear, original orientation', fontsize=8.5, color=GRAY, va='top')
    save(fig, 'actual-response-example' + ('-compact' if compact else ''), example_caption, [curves_path])

# Main benchmark means: an explicit no-change diagnostic accompanies restorers.
benchmark_rows = [
    ('unchanged', 'Unchanged observations', 'core'),
    ('P-direct-none-base', 'Direct training / base', 'response'),
    ('I-initialized-none-base', 'Untrained features / base', 'response'),
    ('M-coordinate-graph_time-base', 'Coordinate pretraining / base', 'response'),
    ('F-response-jepa_endpoint_v1-graph_time-paired_change', 'Endpoint JEPA / change', 'response'),
    ('F-response-jepa_delta_v1-graph_time-paired_change', 'Delta JEPA / change', 'response'),
]
zero = float(rm.zero_response_error.iloc[0])
fig = plt.figure(figsize=(6.8,2.75))
axes=[fig.add_axes([.31,.19,.285,.63]),fig.add_axes([.685,.19,.265,.63])]
y=np.arange(len(benchmark_rows))[::-1]
for i,(ax,metric,title,lim,ticks) in enumerate(zip(axes,['response_error','waveform_error'],
        ['A  Movement response','B  Knee-angle trajectory'],[16,26],[[0,5,10,15],[0,10,20]])):
    clean(ax, True)
    ax.spines['left'].set_visible(False)
    ax.set(xlim=(0,lim), ylim=(-.6,len(y)-.4), xticks=ticks, yticks=y)
    ax.tick_params(axis='y',length=0)
    ax.set_yticklabels([r[1] for r in benchmark_rows] if i==0 else ['']*len(y), fontsize=8.6)
    ax.set_xlabel('Error (°); lower is better')
    ax.set_title(title, loc='left', pad=15, fontsize=9.5,weight='bold')
    for yp,(method,_,which) in zip(y,benchmark_rows):
        val=float((cm if which=='core' else rm).loc[method,metric])
        color=ORANGE if method.startswith('F-') else BLUE if method.startswith('P-') else GRAY
        marker='D' if method.startswith('F-') else 'o'
        ax.scatter([val],[yp],color=color,marker=marker,s=24,zorder=4)
        ax.text(val+.4,yp,f'{val:.2f}',fontsize=8.3,va='center')
    if metric=='response_error':
        ax.axvline(zero,color=INK,ls=':',lw=1.2)
        ax.text(zero+.13,5.47,'No-change: 5.81°',fontsize=8,va='bottom',ha='left')
caption = ('Selected restoration benchmarks on the same 14 development people and three fitted seeds. '
    'Dots are descriptive means with equal weight per person and seed; no error bars are implied. '
    'Base denotes coordinate-supervised restoration, and change denotes the original paired-change readout objective. '
    'The direct model trains end to end; the other learned representations are frozen during readout training, '
    'so these are practical benchmarks rather than a pure pretraining ablation. Both angular errors include '
    'declared measurement-failure penalties. The dotted line is the 5.81° response error obtained by always '
    'predicting ΔA = 0; it is a diagnostic of response recovery, not a pose-restoration model and has no '
    'waveform score. The six selected rows do not replace the complete method table.')
AUDIT['values']['benchmark_means']=[{'method':m,'response_error':float((cm if w=='core' else rm).loc[m,'response_error']),
    'waveform_error':float((cm if w=='core' else rm).loc[m,'waveform_error'])} for m,_,w in benchmark_rows]
AUDIT['values']['zero_response_error']=zero
save(fig,'response-benchmarks',caption,[core_path,response_path])

# Exactly the prespecified primary outcome from each completed stage.
cp='outputs/iclr/walking-core/evaluation/comparisons.json'
jp='outputs/iclr/jepa-response/evaluation/comparisons.json'
rp='outputs/iclr/readout-repair/development/evaluation/comparisons.json'
c,j,r=source(cp,'json'),source(jp,'json'),source(rp,'json')
contrasts=[
 ('A  Walking core','Paired JEPA vs direct','Response · all extractors',c['response_error'],'crossed_person_seed_ci95',BLUE,'o'),
 ('B  Response pretraining','Delta vs endpoint JEPA','Response · all extractors',j['response_error'],'crossed_person_seed_ci95',BLUE,'o'),
 ('C  Delta-JEPA repair','Dense vs low scalar','Waveform · ViTPose',r['primary']['waveform_error'],'person_averaged_t_ci95',ORANGE,'D')]
# Labels describe candidate vs comparator; signs are comparator error minus candidate.
for compact in (False,True):
    fig=plt.figure(figsize=(6.8,1.88 if compact else 2.4))
    for idx,(title,contrast,pop,d,key,color,marker) in enumerate(contrasts):
        x=.065+idx*.322
        ax=fig.add_axes([x,.39 if compact else .34,.26,.24 if compact else .32])
        clean(ax)
        ax.spines['left'].set_visible(False)
        ax.set(xlim=(-2,2.5),ylim=(-.6,.6),xticks=[-2,0,2],yticks=[])
        ax.axvline(0,ls=':',color=GRAY,lw=.9)
        lo,hi=d[key];mean=d['improvement']
        ax.errorbar([mean],[0],xerr=[[mean-lo],[hi-mean]],fmt=marker,color=color,capsize=3,ms=5,lw=1.6)
        fig.text(x,.96,title,fontsize=9.4,weight='bold',va='top')
        fig.text(x,.85,contrast,fontsize=8.6,va='top')
        fig.text(x,.75,pop,fontsize=8.1,color=GRAY,va='top')
        fig.text(x+.13,.255 if compact else .205,f'{mean:+.2f}°  [{lo:+.2f}, {hi:+.2f}]',fontsize=8.7,ha='center',va='top')
        if not compact:
            fig.text(x+.13,.105,'Person + seed bootstrap' if idx<2 else 'Person t interval',fontsize=8.0,color=GRAY,ha='center')
    fig.text(.5,.025,'Improvement = comparator error − candidate error (°); positive favors candidate',fontsize=8.5,ha='center')
    caption=('The three declared primary comparisons, each on the same 14 development people and three fitted seeds. '
        'Core and response-pretraining arms both use the original paired-change readout; their bars are 95% crossed '
        'person/seed bootstrap intervals. The repair fixes the delta-JEPA encoder; its bar is a 95% Student-t interval over 14 paired person effects '
        'after averaging the three fitted seeds, conditional on those fits; it uses the held-out ViTPose extractor '
        'and waveform error, while the first two use movement-response error pooled across three extractors. '
        'Positive values favor the candidate named first. All intervals span zero. These are successive development '
        'comparisons on reused participants, not independent replications, and the three effects must not be pooled.')
    save(fig,'primary-contrasts'+('-compact' if compact else ''),caption,[cp,jp,rp])
AUDIT['values']['primary_contrasts']=[{'stage':title,'metric':d['metric'],'improvement':d['improvement'],'interval':d[key],'interval_key':key} for title,_,_,d,key,_,_ in contrasts]

# Repair comparisons: keep the primary extractor population fixed for all arms.
repair_rows=[
 ('F-response-jepa_delta_v1-graph_time-paired_change','Original scalar change'),
 ('R-repair-jepa_delta_v1-scalar_low','Lower-weight scalar'),
 ('R-repair-jepa_delta_v1-dense_change','Dense angular change'),
 ('F-response-jepa_delta_v1-graph_time-base','Base (no change loss)')]
fig=plt.figure(figsize=(6.8,2.45))
axes=[fig.add_axes([.285,.24,.29,.49]),fig.add_axes([.68,.24,.27,.49])]
y=np.arange(4)[::-1]
for idx,(ax,metric,title,limit) in enumerate(zip(axes,['waveform_error','response_error'],
    ['A  Primary: trajectory error','B  Response tradeoff'],[27,19])):
    clean(ax,True);ax.spines['left'].set_visible(False)
    ax.set(xlim=(0,limit),ylim=(-.55,3.55),yticks=y,xticks=[0,10,20] if idx==0 else [0,5,10,15])
    ax.tick_params(axis='y',length=0)
    ax.set_yticklabels([r[1] for r in repair_rows] if idx==0 else ['']*4,fontsize=8.8)
    ax.set_xlabel('Error (°); lower is better')
    ax.set_title(title,loc='left',fontsize=9.3,weight='bold',pad=16)
    for yp,(method,label) in zip(y,repair_rows):
        val=float(pm.loc[method,metric]);color=ORANGE if 'dense_change' in method else BLUE if 'scalar_low' in method else GRAY
        ax.scatter([val],[yp],marker='D' if 'dense_change' in method else 'o',color=color,s=25,zorder=4)
        ax.text(val+.5,yp,f'{val:.2f}',fontsize=8.4,va='center')
fig.text(.285,.96,'Frozen delta-JEPA features · ViTPose inputs · 14 people × 3 fitted seeds',fontsize=8.7,va='top',color=GRAY)
caption=('Readout repair with the same frozen delta-JEPA encoder and held-out ViTPose inputs. '
    'Dots are descriptive means over 14 people and three seeds; lower is better. The original scalar '
    'paired-change loss produces greater trajectory error than the reduced-weight scalar loss or dense '
    'angular-change alternative. The declared dense-versus-low-scalar advantage is only 0.28° '
    '(Figure primary contrasts); this controlled comparison does not establish a benefit from dense '
    'supervision. The base readout uses no change loss. Response error retains the 720° penalty for '
    'failed eligible measurement pairs, and waveform error retains its 180° penalty, so numerical '
    'improvements may combine successful-output accuracy with fewer failures. This plot does not mix '
    'the ViTPose primary population with all-extractor summaries.')
AUDIT['values']['repair_means']=[{'method':m,'waveform_error':float(pm.loc[m,'waveform_error']),'response_error':float(pm.loc[m,'response_error'])} for m,_ in repair_rows]
save(fig,'readout-repair',caption,[repair_path])

# Original downstream loss: all five matched model families, both relevant outcomes.
family_rows=[('P-direct-none','Direct training'),('I-initialized-none','Untrained features'),
 ('I-shuffled_jepa-graph_time','Shuffled-reference JEPA'),('M-coordinate-graph_time','Coordinate pretraining'),
 ('M-paired_jepa-graph_time','Paired JEPA')]
fig=plt.figure(figsize=(6.8,2.6))
axes=[fig.add_axes([.29,.20,.285,.54]),fig.add_axes([.68,.20,.27,.54])]
y=np.arange(5)[::-1]
for idx,(ax,metric,title,limit) in enumerate(zip(axes,['response_error','waveform_error'],
 ['A  Movement response','B  Knee-angle trajectory'],[15,26])):
    clean(ax,True);ax.spines['left'].set_visible(False)
    ax.set(xlim=(0,limit),ylim=(-.5,4.5),yticks=y,xticks=[0,5,10,15] if idx==0 else [0,10,20])
    ax.tick_params(axis='y',length=0)
    ax.set_yticklabels([r[1] for r in family_rows] if idx==0 else ['']*5,fontsize=8.6)
    ax.set_title(title,loc='left',pad=16,fontsize=9.5,weight='bold')
    ax.set_xlabel('Error (°); lower is better')
    for yp,(prefix,label) in zip(y,family_rows):
        base=float(cm.loc[prefix+'-base',metric]);change=float(cm.loc[prefix+'-paired_change',metric])
        ax.plot([base,change],[yp,yp],color=GRAY,lw=.9)
        ax.scatter([base],[yp],color=BLUE,marker='o',s=24,zorder=3)
        ax.scatter([change],[yp],color=ORANGE,marker='D',s=24,zorder=3)
fig.legend([plt.Line2D([],[],color=BLUE,marker='o',ls=''),plt.Line2D([],[],color=ORANGE,marker='D',ls='')],
 ['Base coordinate objective','Original paired-change objective'],loc='upper center',bbox_to_anchor=(.6,1.01),
 frameon=False,ncol=2,fontsize=8.6,columnspacing=1.4,handletextpad=.4)
caption=('Matched downstream objectives in all five walking-core model families, with 14 development people '
    'and three fitted seeds. Each connector joins descriptive means for the same model family; circles use '
    'the base coordinate objective and diamonds add the original scalar paired-change and short-segment '
    'geometry terms. All five families have greater trajectory error after the addition, whereas response '
    'error moves in both directions. Core reference-paired JEPA slightly improves the scalar response while its '
    'trajectory worsens. This identifies a tradeoff in the tested objective package; it does not by itself '
    'isolate the scalar weight, quantile differentiation or geometry penalty as the mechanism.')
save(fig,'core-objective-tradeoff',caption,[core_path])

AUDIT['sources_sha256']=SOURCES
(OUT/'figure-provenance.json').write_text(json.dumps(AUDIT,indent=2)+'\n')
readme=['# ICLR scientific-draft figures (25 September 2026)','',
 'Generated by `docs/studies/gait-fidelity/scripts/build_iclr_draft_figures.py` from the local `outputs/iclr` evidence.',
 'SVG files retain editable text. PDF equivalents and 200-dpi PNG previews are provided. No existing proposal assets were overwritten.',
 '', '## Evidence boundaries', '',
 'The sample figure is an actual **aggregated person-level response curve**, not a raw frame, individual walking window, or reconstructed pose sequence. The transferred package explicitly excludes raw data. The selection rule and plotted values are in `figure-provenance.json`. No generic clinical footage is substituted.',
 'All results are development evidence, reuse the same 14 people, and do not establish clinical validity or an independently confirmed JEPA benefit.', '']
for stem,record in AUDIT['figures'].items():
 readme.extend([f'## {stem}', '', record['caption'], '',f'Print size: {record["size_inches"]}; all text checked to remain within canvas.', ''])
(OUT/'README.md').write_text('\n'.join(readme))
print(json.dumps({'output':str(OUT),'figures':list(AUDIT['figures']),'verified_source_files':len(SOURCES)},indent=2))
