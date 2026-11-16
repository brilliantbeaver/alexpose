"""Build the iteration02 vector tutorial from frozen evidence and declared proposals.

Run from any directory with the experiment Python environment. This generator
never modifies research runs, previous slides, or manuscript artifacts.
"""
from __future__ import annotations
import hashlib
import html
import json
import math
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'slides'
ASSETS = OUT / 'iteration02_images'
RUN = ROOT / 'artifacts/runs/restart-dynamics-v1'
DATA = ROOT / 'artifacts/runs/restart-full10hz-v1'
NAVY, TEAL, ORANGE = '#153448', '#087d78', '#bb6234'
BLUE, GREEN, AMBER, GRAY = '#e5eef5', '#def0eb', '#f8e6d7', '#526a79'
BG = '#fbfcfe'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'font.size': 16})
BOXES = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wrap(text, width):
    return '\n'.join(textwrap.wrap(str(text), width, break_long_words=False,
                                  break_on_hyphens=False))


def txt(ax, x, y, text, size=19, color=NAVY, width=90, **kw):
    return ax.text(x, y, wrap(text, width), fontsize=size, color=color,
                   linespacing=1.32, **kw)


def box(ax, x, y, w, h, text, color=BLUE, size=19, width=None):
    patch = Rectangle((x, y), w, h, facecolor=color, edgecolor='none')
    ax.add_patch(patch)
    label = txt(ax, x+w/2, y+h/2, text, size=size,
                width=width or max(12, int(w*6)), ha='center', va='center')
    BOXES.append((patch, label))
    return patch


def arrow(ax, start, end, color=TEAL):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                mutation_scale=16, color=color, linewidth=2))


def flow(ax, nodes, colors=None):
    n = len(nodes)
    gap = .42
    width = (14.5-gap*(n-1))/n
    for i, node in enumerate(nodes):
        x = .75+i*(width+gap)
        box(ax, x, 3.25, width, 2.25, node,
            color=(colors[i] if colors else (GREEN if i == n-1 else BLUE)),
            size=20 if n < 4 else 18)
        if i < n-1:
            arrow(ax, (x+width+.03, 4.375), (x+width+gap-.03, 4.375))


def table(ax, headers, rows, widths=None, size=16):
    x0, y0, totalw, totalh = .75, 2.0, 14.5, 4.45
    widths = np.array(widths or [1]*len(headers), float)
    widths = widths/widths.sum()*totalw
    h = totalh/(len(rows)+1)
    for rowno, row in enumerate([headers]+rows):
        y = y0+totalh-(rowno+1)*h
        x = x0
        for colno, value in enumerate(row):
            color = NAVY if rowno == 0 else (BLUE if rowno % 2 else '#f0f4f7')
            ax.add_patch(Rectangle((x, y), widths[colno]-.035, h-.025,
                                   facecolor=color, edgecolor='none'))
            t = txt(ax, x+.15, y+h/2, value, size=size if rowno else size-1,
                    color='white' if rowno == 0 else NAVY,
                    width=max(8, int(widths[colno]*6.8)), va='center')
            BOXES.append((Rectangle((x, y), widths[colno]-.035, h-.025,
                                    transform=ax.transData), t))
            x += widths[colno]


def canvas(slide, index, count):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_position([0, 0, 1, 1])
    ax.set(xlim=(0, 16), ylim=(0, 9))
    ax.axis('off')
    state_color = ORANGE if slide['status'].startswith('PROPOSED') else TEAL
    txt(ax, .7, 8.66, 'GAIT OBSERVATIONS FOR MEDICAL WORLD MODELS', size=11, color=GRAY)
    txt(ax, 15.3, 8.66, slide['status'], size=11, color=state_color, ha='right')
    txt(ax, .7, 8.04, slide['title'], size=27, width=83, weight='bold', va='top')
    txt(ax, .7, 7.23, slide['subtitle'], size=17, width=118, color=GRAY, va='top')
    txt(ax, .75, 1.34, slide['caption'], size=15, width=128, color=GRAY, va='top')
    ax.plot([.7, 15.3], [.78, .78], color='#c8d6de', linewidth=1)
    txt(ax, .7, .4, slide['source_short'], size=10, width=145, color=GRAY)
    txt(ax, 15.3, .4, f'{index:02d} / {count}', size=11, color=GRAY, ha='right')
    return fig, ax


def mask_grid(ax, x, y, title, family, w=6.55, h=1.7):
    """Schematic group-time cells: 16 prefix columns and 8 future columns."""
    txt(ax, x, y+h+.35, title, size=18, width=50, weight='bold')
    rng = np.random.default_rng(2701)
    hidden = np.zeros((4, 24), dtype=bool)
    if family == 'uniform':
        candidates = [(j,t) for j in range(4) for t in range(15)]
        for k in rng.choice(len(candidates),24,replace=False):
            j,t=candidates[k];hidden[j,t]=True
    elif family == 'block':
        hidden[:, 5:11] = True
    elif family == 'tube':
        hidden[[0,2], :12] = True
    elif family == 'anatomical':
        hidden[2:, :12] = True
    elif family == 'complementary':
        hidden[0, 2:14] = True
        hidden[2, :12] = True
    elif family == 'phase':
        hidden[2, 4:8] = True
        hidden[3, 12:16] = True
    if family in ('uniform','block','tube','anatomical','complementary'):
        assert int(hidden[:,:16].sum()) == 24
        assert not hidden[:,15].any()
    cw, rh = w/24, h/4
    for row in range(4):
        txt(ax,x-.1,y+(3-row)*rh+rh/2,['Knee','Ankle','Heel','Foot'][row],size=11,ha='right',va='center')
        for col in range(24):
            fc = NAVY if col >= 16 else (AMBER if hidden[row, col] else GREEN)
            ax.add_patch(Rectangle((x+col*cw, y+(3-row)*rh), cw-.025,
                                   rh-.035, facecolor=fc, edgecolor='none'))
    ax.plot([x+16*cw-.015]*2, [y-.05, y+h+.05], color=ORANGE, linewidth=2)
    txt(ax, x, y-.32, '16 prefix samples', size=11, color=GRAY)
    txt(ax,x+15.4*cw,y-.32,'Anchor',size=10,color=TEAL,ha='center')
    txt(ax, x+w, y-.32, '8 future samples', size=11, color=GRAY, ha='right')


def draw(slide, fig, ax):
    kind, p = slide['kind'], slide['payload']
    if kind == 'flow':
        flow(ax, p)
    elif kind == 'table':
        table(ax, p['headers'], p['rows'], p.get('widths'), p.get('size', 16))
    elif kind == 'timeline':
        for i in range(24):
            x = .85+i*.595
            ax.add_patch(Rectangle((x, 4.2), .51, 1.15,
                                   facecolor=BLUE if i < 16 else GREEN, edgecolor='none'))
            if i in (0, 5, 10, 15, 16, 23):
                txt(ax, x+.25, 3.8, str(i), size=13, ha='center')
        txt(ax, 5.1, 2.85, 'The prefix contains 16 observations.', size=22, ha='center')
        txt(ax, 12.6, 2.85, 'The target has 8.', size=22, ha='center', color=TEAL)
        arrow(ax, (10.36, 5.8), (15.07, 5.8))
        txt(ax, 12.7, 6.1, '0.1 to 0.8 seconds ahead', size=17, ha='center')
    elif kind == 'skeleton':
        joints = {'ls': (3.4, 6), 'rs': (4.8, 6), 'lh': (3.65, 4.5), 'rh': (4.55, 4.5),
                  'lk': (3.1, 3.2), 'rk': (5.1, 3.25), 'la': (2.85, 2.15), 'ra': (5.35, 2.15),
                  'lheel': (3.05, 2), 'rheel': (5.15, 2), 'lf': (2.35, 2), 'rf': (5.85, 2)}
        for a, b in [('ls','rs'),('ls','lh'),('rs','rh'),('lh','rh'),('lh','lk'),
                     ('rh','rk'),('lk','la'),('rk','ra'),('la','lheel'),('lheel','lf'),('ra','rheel'),('rheel','rf')]:
            ax.plot([joints[a][0],joints[b][0]], [joints[a][1],joints[b][1]],
                    color=TEAL if a[0]=='l' and b[0]=='l' else NAVY, linewidth=5)
        for name, (x,y) in joints.items():
            ax.scatter(x,y,s=80,color=TEAL if name.startswith('l') else NAVY,zorder=4)
        ax.scatter(4.1,4.5,s=100,color=ORANGE,zorder=5)
        txt(ax, 7, 5.9, 'The pelvis supplies a shared origin.', size=23, width=45)
        txt(ax, 7, 4.55, 'The observed prefix supplies torso scale.', size=23, width=45)
        txt(ax, 7, 3.2, 'Each leg has four distal target joints.', size=23, width=45)
    elif kind == 'shift':
        labels = ['0','2','4','6','9','11','13','15']
        for row, label in enumerate(['Aligned opposite leg', 'Shifted opposite leg']):
            yy = 5.25-row*1.65
            txt(ax, .8, yy+.35, label, size=16, width=20)
            order = list(range(8)) if row == 0 else [4,5,6,0,1,2,3,7]
            for j, idx in enumerate(order):
                box(ax, 4.2+j*1.25, yy-.1, 1.05, .9, labels[idx],
                    color=GREEN if idx==7 else BLUE, size=18)
        txt(ax, 8.2, 2.4, 'A different shift is assigned to each window.', size=22, ha='center')
        txt(ax, 13.5, 6.4, 'The anchor stays fixed.', size=15, width=20, ha='center')
        arrow(ax,(13.5,6.3),(13.5,6.08))
    elif kind == 'folds':
        for col in range(5):
            txt(ax,4.5+col*2.25,6.65,f'Source group {col+1}',size=13,ha='center')
        for row in range(5):
            txt(ax,.8,6.0-row*.82,f'Outer fold {row+1}',size=17)
            for col in range(5):
                box(ax,3.5+col*2.25,5.7-row*.82,2.0,.61,
                    'Held out' if row==col else 'Normal fit / inner',
                    GREEN if row==col else BLUE,size=12,width=20)
    elif kind in ('errors','effects','auc'):
        chart = fig.add_axes([.34,.26,.59,.43])
        chart.set_facecolor(BG)
        values = np.array([v['value'] for v in p])
        bounds = np.array([v['ci'] for v in p])
        for j,v in enumerate(p):
            color = ORANGE if v.get('posthoc') else TEAL
            chart.errorbar(v['value'],j,xerr=[[max(0,v['value']-v['ci'][0])],
                                            [max(0,v['ci'][1]-v['value'])]],
                          fmt='s' if v.get('posthoc') else 'o',color=color,capsize=4,markersize=8)
        chart.set_yticks(np.arange(len(p)), [v['label'] for v in p], fontsize=14)
        chart.invert_yaxis()
        chart.spines[['top','right','left']].set_visible(False)
        chart.tick_params(axis='y',length=0,pad=12)
        chart.grid(axis='x',color='#dce5eb',linewidth=.8)
        if kind == 'errors':
            chart.set_xlim(0,.36)
            chart.set_xticks(np.arange(0,.36,.05))
            chart.set_xlabel('Source-average squared articulation error; lower is better',fontsize=13,labelpad=12)
        elif kind == 'effects':
            chart.axvline(0,color=GRAY,linewidth=1)
            chart.set_xlabel('Reference error minus bilateral error; positive favors bilateral',fontsize=13,labelpad=12)
        else:
            chart.axvline(.5,color=GRAY,linestyle=':',linewidth=1)
            chart.set_xlim(0,1)
            chart.set_xlabel('Source-level AUROC; higher scores rank sources as more abnormal',fontsize=13,labelpad=12)
    elif kind == 'formula':
        box(ax, .85, 4.0, 4.1, 1.5, 'Learned forecast error', BLUE)
        txt(ax, 5.5, 4.75, '−', size=42, ha='center', va='center')
        box(ax, 6.0, 4.0, 4.1, 1.5, 'Persistence error', BLUE)
        txt(ax, 10.65, 4.75, '=', size=35, ha='center', va='center')
        box(ax, 11.15, 4.0, 4.0, 1.5, 'Excess error', GREEN)
        txt(ax, 8, 2.6, 'A higher score means less benefit from the learned forecast.',size=23,ha='center',width=80)
    elif kind == 'masks':
        mask_grid(ax,1.2,4.4,'Uniform prefix masking','uniform',w=6.25,h=1.4)
        mask_grid(ax,8.8,4.4,'Contiguous temporal block','block',w=6.25,h=1.4)
        mask_grid(ax,1.2,2.0,'Random joint tube','tube',w=6.25,h=1.4)
        mask_grid(ax,8.8,2.0,'Anatomical foot tube','anatomical',w=6.25,h=1.4)
    elif kind == 'complementary':
        mask_grid(ax,.9,4.5,'One branch hides selected motion.','complementary',w=13.9,h=1.7)
        flow_y = 2.1
        for i,t in enumerate(['Estimate reliable\nprefix motion','Balance low and\nhigh motion','Match masks and\nfuture support']):
            box(ax,.9+i*4.8,flow_y,4.3,1.2,t,BLUE,size=17)
            if i<2:arrow(ax,(5.25+i*4.8,flow_y+.6),(5.65+i*4.8,flow_y+.6))
    elif kind == 'gate':
        box(ax,.9,4.2,4.0,1.6,'Are at least two cycles reliable?',BLUE,size=21)
        box(ax,6.5,5.2,7.8,1.0,'Yes: add phase-aligned support masks.',GREEN,size=20)
        box(ax,6.5,2.7,7.8,1.0,'No: keep fixed-time masks and report coverage.',AMBER,size=19)
        arrow(ax,(4.95,5.05),(6.45,5.7));arrow(ax,(4.95,4.55),(6.45,3.2))
    elif kind == 'objectives':
        box(ax,.9,3.6,3.4,1.8,'The same visible prefix',BLUE,size=21)
        box(ax,5.5,5.0,4.5,1.2,'Predict coordinates.',BLUE,size=21)
        box(ax,5.5,2.4,4.5,1.2,'Predict target representations.',GREEN,size=20)
        box(ax,11.4,3.6,3.7,1.8,'Compare on identical future coordinates.',BLUE,size=19)
        arrow(ax,(4.35,4.8),(5.45,5.6));arrow(ax,(4.35,4.05),(5.45,3.0))
        arrow(ax,(10.05,5.6),(11.35,4.8));arrow(ax,(10.05,3.0),(11.35,4.05))
    elif kind == 'twotails':
        xx=np.linspace(-3,3,400); yy=np.exp(-xx**2/2)
        x=.95+(xx+3)*2.35; y=2.7+yy*3.1
        ax.plot(x,y,color=TEAL,lw=3)
        ax.fill_between(x,y,2.7,where=xx < -1.6,color=AMBER)
        ax.fill_between(x,y,2.7,where=xx > 1.6,color=AMBER)
        ax.plot([.95,15.05],[2.7,2.7],color=GRAY,lw=1)
        txt(ax,2,2.48,'Unusually easy prediction',size=17,ha='center',va='top',width=24)
        txt(ax,8,2.05,'Normal calibration range',size=19,ha='center')
        txt(ax,14,2.48,'Unusually difficult prediction',size=17,ha='center',va='top',width=24)
        txt(ax,8,6.25,'Conceptual calibration distribution; no measured density is shown.',size=15,color=GRAY,ha='center')
    else:
        raise ValueError(kind)


def make_slides(r, prep, inv):
    slides=[]
    def add(title, subtitle, kind, payload, caption, notes, status='MEASURED / EXPLORATORY', sources=None, source_short=None):
        default_source = 'Source: restart-dynamics-v1; exact inputs and hashes are in iteration02_manifest.json.'
        if status.startswith(('PROPOSED','PLAN')):
            default_source = 'Source: iteration02 research instruction and linked primary literature; the new experiments are proposed.'
        elif status.startswith('REVIEW'):
            default_source = 'Source: .agents/protocol.md and the iteration02 acceptance criteria.'
        elif status.startswith('SYNTHESIS'):
            default_source = 'Sources: completed restart evidence, draft02 and the iteration02 research plan.'
        slides.append(dict(title=title,subtitle=subtitle,kind=kind,payload=payload,
            caption=caption,notes=notes,status=status,sources=sources or [],
            source_short=source_short or default_source))
    def tab(headers,rows,widths=None,size=16):return dict(headers=headers,rows=rows,widths=widths,size=size)
    result_source='../artifacts/runs/restart-dynamics-v1/results.json'
    neuro_sources=['https://doi.org/10.1007/s00221-007-0955-7','https://doi.org/10.1152/jn.00825.2009','https://doi.org/10.1111/dmcn.12826']
    add('Reference-aware scoring reveals a useful gait signal.',
        'A transparent study connects short-term movement prediction with exploratory anomaly detection.',
        'flow',['Observe real gait','Predict future articulation','Compare with a simple reference','Rank source videos'],
        'The contribution is a controlled evaluation of observed movement. Clinical diagnosis and treatment effects remain outside its evidence.',
        'This deck separates measured results from proposed experiments. The favorable excess-error ranking motivates stronger confound controls rather than a claim of clinical diagnostic accuracy.',sources=[result_source])
    add('Medical world models need a reliable observation layer.',
        'Video poses provide a short-term motor observation, while disease trajectories require additional data.',
        'flow',['Video observations','Short-term movement transitions','Longitudinal patient states','Treatment-aware decisions'],
        'This project tests the first transition. Later stages require clinical labels, follow-up and intervention observations.',
        'The official workshop includes dynamics, uncertainty, validation and benchmarks. A short-term observation transition is relevant as a tested component, but it is not an action-conditioned clinical simulator.',
        status='SCOPE / CONCEPTUAL',sources=['https://med-wm.github.io/aaai27-mwm/'],source_short='Source: official Med-WM scope; the final two stages are conceptual future requirements.')
    add('Neurology motivates observable coordination hypotheses.',
        'Clinical physiology suggests useful questions without making a video pose a direct neural measurement.',
        'flow',['Neural and muscular control','Coordinated limb movement','Camera projection and occlusion','Estimated 2D articulation'],
        'The chain has many possible causes. Its arrows describe measurement pathways, not an identified causal model.',
        'Timing studies in Parkinson disease motivate bilateral tests. EMG studies in stroke and cerebral palsy also show why simplified movement can accompany impairment. Camera view, speed and pose error remain alternative explanations.',
        status='RATIONALE / CONCEPTUAL',sources=neuro_sources,source_short='Sources: Plotnik et al. 2007; Clark et al. 2010; Steele et al. 2015.')
    add('Different annotations suggest different checks.',
        'These hypotheses concern visible movement and need independent clinical validation.',
        'table',tab(['Annotation','Observable hypothesis','Required restraint'],[
            ['Parkinsons','Test bilateral timing.','Do not infer a specific neural cause.'],
            ['Stroke','Test asymmetry and low variability.','A simple trajectory may be impaired.'],
            ['Cerebral palsy','Test constrained coordination.','Pose does not measure muscle activation.'],
            ['Myopathic','Test weak or compensated motion.','Do not call every condition central.']], [2.1,3.1,4.3]),
        'No disease-specific direction is assumed for raw prediction error. The same condition can contain several movement patterns.',
        'The dataset folders supply annotations, not independent diagnoses. Neurological reasoning sets falsifiable comparisons; it does not replace patient identity, severity, muscle recordings or validated gait events.',
        status='RATIONALE / PROPOSED',sources=neuro_sources,source_short='Sources: neurological primary studies in the iteration02 literature registry; GAVD labels.')
    add('The full inventory defines the available evidence.',
        'Source videos, sequences and patients are different units.',
        'table',tab(['Scope','Sequences','Source IDs','Interpretation'],[
            ['All GAVD labels',f"{inv['all_sequences']:,}",str(inv['all_sources']),'The complete annotation inventory.'],
            ['Five requested labels',str(inv['selected_sequences']),str(inv['selected_sources']),'The disease-focused scope.'],
            ['Normal annotations',str(inv['by_label']['normal']['sequences']),str(inv['by_label']['normal']['sources']),'Already included in that scope.'],
            ['Evaluated cohort',str(r['sequences']),str(r['sources']),'After media and quality gates.']], [2.9,1.2,1.2,4.8]),
        'The upstream source CSVs are provenance. Counting them again would duplicate the annotation inventory.',
        'The repeated stroke entry in the task is one class. GAVD supplies heterogeneous videos, but source IDs are not verified participant IDs. Extra windows improve coverage without creating independent patients.',
        sources=['../artifacts/runs/restart-full10hz-v1/inventory.json',result_source],source_short='Sources: restart-full10hz-v1/inventory.json and restart-dynamics-v1/results.json.')
    add('Every attrition step remains visible.',
        'The study uses complete eligible spans and applies the same target gate to every model.',
        'flow',[f"{inv['media_eligible_sequences']} media-eligible sequences",f"{prep['selected_sequences']} candidates after exclusions",f"{prep['n_windows']:,} prepared windows",f"{r['windows']:,} evaluated windows"],
        f"The final cohort contains {r['sequences']} sequences and {r['sources']} source videos. It is not a cohort of {r['windows']:,} patients.",
        'Two documented whole-source identity exclusions are preserved. Pose quality and common distal-target support determine eligibility. All rejection reasons remain available in the frozen run; thresholds are not changed to improve a result.',
        sources=['../docs/restart/data_audit.md',result_source])
    add('The condition table reveals unequal source support.',
        'The smaller source groups need especially cautious subgroup interpretation.',
        'table',tab(['Dataset annotation','Windows','Sequences','Sources'],[
            [v['label'],f"{v['windows']:,}",str(v['sequences']),str(v['sources'])] for v in r['cohort_by_label']], [3,1.4,1.5,1.4],size=18),
        'All sources receive equal final weight. A large number of windows cannot compensate for few independent sources.',
        'The table is loaded directly from the completed results artifact. The normal group contains 29 sources, while cerebral palsy has eight. Individual-disease estimates should show their denominators and broad uncertainty.',sources=[result_source])
    add('Shared Ambient interfaces keep ingestion traceable.',
        'New videos belong under data/videos-gavd, with one original per source ID.',
        'flow',['Shared YouTube cache helper','Source video and hash','Ambient pose extraction','Frames, poses and quality ledger'],
        f"The completed extraction made {prep['actual_model_calls_this_invocation']:,} real pose-model calls. No new download was needed for that run.",
        'Reuse ambient.utils.youtube_cache.download_youtube_urls and cache lookups. Store future originals under data/videos-gavd/sources/<source_id>.* and previews under data/videos-gavd/previews/. Existing unique originals remain reusable. The adapter uses SequenceKeypointExtractor.extract_from_image and records model and video identities.',
        sources=['../worldmodels/restart_data.py','../docs/restart/data_contract.md'],source_short='Sources: Ambient downloader and pose APIs; restart_data.py; restart-full10hz-v1/preparation.json.')
    add('Frame numbering changes exactly once.',
        'Annotation frames are one-based, while OpenCV indexes are zero-based.',
        'flow',['GAVD frame n','Subtract one at the decoder boundary','Decode native frame n − 1','Record time and dimensions'],
        'A second subtraction shifts the observation. Cached poses must retain the exact frame identity and conversion contract.',
        'SequenceKeypointExtractor frame arguments are also one-based. The current restart uses extract_from_image after explicit decoding. Do not apply a second frame conversion inside another adapter. Source IDs, sequence IDs, frame tuples, decoder times, confidence and missingness remain explicit.',
        sources=['../docs/restart/data_contract.md'],source_short='Source: restart observation contract and shared Ambient extraction interface.')
    add('A fixed time horizon makes predictions comparable.',
        'Nearest native frames approximate a nominal 10 Hz sampling grid.',
        'timeline',None,
        'The observed prefix spans 1.5 seconds between endpoints. The final future sample is nominally 0.8 seconds later.',
        'The complete target contains eight samples at approximately 0.1 through 0.8 seconds ahead. The saved metadata retain actual decoded times and sampling deviations. No skeleton interpolation converts missing data into observations.',
        sources=['../docs/restart/data_contract.md'])
    add('The model predicts body-relative articulation.',
        'A shared pelvis origin and prefix-only torso scale define the coordinate system.',
        'skeleton',None,
        'The diagram is schematic. Projected 2D coordinates are neither physical speed nor measured three-dimensional anatomy.',
        'The declared representation selects 16 BlazePose joints from 33 raw landmarks. Each target leg uses knee, ankle, heel and foot-index joints. Both hips contribute to the shared reference, so own-leg input is not purely unilateral raw sensory access.',
        sources=['../docs/restart/model_contract.md'])
    add('Missingness is separate from a measured zero.',
        'Confidence, validity and available target support remain explicit arrays.',
        'flow',['Apply confidence and person gates','Fill prefix only from the past','Retain observation indicators','Score observed common targets'],
        'Future coordinates and future validity may define scoring support, but they may not change the prediction inputs.',
        'The frozen run uses confidence at least 0.5, separate prefix and future support gates, and a shared distal-target gate. A future target also needs a valid prefix anchor. New masks must never convert missing keypoints into visible inputs or training targets.',
        sources=['../docs/restart/data_contract.md','../docs/restart/model_contract.md'])
    add('Simple references make the learned claim testable.',
        'Every arm predicts the same future coordinates from the same eligible windows.',
        'table',tab(['Prediction arm','Available information','Question answered'],[
            ['Persistence','Latest observed target pose.','Does learning beat no change?'],
            ['Current-pose transition','Current bilateral articulation.','Is history necessary?'],
            ['Own-leg history','Target-leg distal prefix.','What does local history provide?'],
            ['Bilateral history','Both distal-leg prefixes.','Does extra context help?'],
            ['Shifted opposite history','Same values with changed alignment.','Does their timing matter?']], [2.9,3.5,3.5]),
        'The current-pose transition is a learned forecast. It is different from copying the last pose.',
        'The fitted models are normal-only ridge residual predictors. Damped velocity and shrinkage variants are retained as additional references. Own-leg and bilateral arms share the same pelvis reference.',sources=['../docs/restart/model_contract.md'])
    add('The timing control preserves values and the anchor.',
        'A window-specific cyclic shift changes alignment rather than the selected coordinate set.',
        'shift',None,
        'This schematic shows one shift. The actual offset varies by window and coordinates move with their validity indicators.',
        'The eight selected prefix positions are 0, 2, 4, 6, 9, 11, 13 and 15. Only the first seven opposite-leg positions are shifted. A globally fixed permutation can be absorbed by a linear model, so it is an inadequate timing control. Refit controls and inference-only perturbations answer different questions.',
        sources=['../docs/restart/model_contract.md'])
    add('Source grouping protects related observations.',
        'All windows and sequences from a source remain in the same outer fold.',
        'folds',None,
        'Five outer folds estimate internal source transfer. Three inner folds select settings using normal training sources.',
        'Feature scaling, imputation parameters, model fitting and model selection must remain inside the relevant training data. Prior GAVD exposure means the current evaluation is exploratory. A new split does not create an untouched dataset.',sources=['../docs/restart/model_contract.md'])
    add('The aggregation gives each source equal weight.',
        'The model is evaluated at the source level rather than treating every frame as independent.',
        'flow',['Coordinates within each leg','Equal weight for both legs','Windows then sequences','Equal source averages'],
        'The reported bootstrap intervals resample fixed source predictions. They do not include full retraining or patient-identity uncertainty.',
        'The complete hierarchy matters because sources have unequal numbers of sequences and windows. A frame bootstrap would give a misleading impression of sample size. Future work should repeat training partitions and evaluate a separate external cohort.',sources=['../docs/restart/model_contract.md'])
    norm={v['arm']:v for v in r['normal_forecast'] if v['horizon']==8}
    labels=[('static','Current-pose transition'),('bilateral','Bilateral history'),('own','Own-leg history'),('shifted','Shifted history'),('bilateral_shrunk','Shrunk bilateral history'),('persistence','Persistence'),('damped_velocity','Damped velocity')]
    add('Current-pose prediction is the strongest forecast.',
        'The chart averages all eight future samples on 29 held-out normal sources.',
        'errors',[dict(label=label,value=norm[a]['estimate'],ci=norm[a]['ci95']) for a,label in labels],
        'Bilateral history beats persistence, but current-pose prediction is better. Bars show 95% source-bootstrap intervals conditional on fixed fits.',
        'The bilateral forecast has 0.08063 MSE versus 0.12126 for persistence, a 33.5% relative reduction. The current-pose transition reaches 0.07164. These comparisons do not establish that ordered history is necessary. Intervals condition on the fixed fitted folds.',sources=[result_source])
    pairs=[v for v in r['paired_contrasts'] if v['horizon']==8 and v['method']=='bilateral' and v['reference'] in ['persistence','own','shifted','static']]
    pretty={'persistence':'Against persistence','own':'Against own-leg history','shifted':'Against shifted history','static':'Against current-pose transition'}
    add('Paired controls narrow the interpretation.',
        'A positive effect favors bilateral history over the listed reference.',
        'effects',[dict(label=pretty[v['reference']],value=v['estimate'],ci=v['ci95']) for v in pairs],
        'The own-leg gain is uncertain and current-pose prediction is better. Bars show 95% source-bootstrap intervals conditional on fixed fits.',
        'All paired estimates use the same 29 normal sources. A useful forecast does not establish a neural mechanism. The complete evidence supports forecast utility and targeted evaluation, while leaving timing-specific biological claims unresolved.',sources=[result_source])
    add('Excess error asks whether learning helps this source.',
        'The score compares a learned forecast with the same source’s persistence reference.',
        'formula',None,
        'The score direction was fixed before the completed evaluation. Higher excess error was declared more abnormal.',
        'Subtracting persistence changes the question from absolute movement error to relative forecast benefit. A high excess-error score can still arise from low movement amplitude or simple motion. Therefore, reference adjustment requires a direct low-motion control.',sources=[result_source,'../docs/restart/model_contract.md'])
    ar=r['anomaly'];anames={'bilateral':'Raw bilateral error','persistence':'Persistence error','excess_error':'Excess error','coordination_deficit':'Bilateral minus own error','prefix_missingness':'Prefix missingness'}
    add('Excess error ranks annotations with AUROC 0.793.',
        'These five scores retain their original larger-is-abnormal direction.',
        'auc',[dict(label=anames[a],value=v['auroc'],ci=v['auroc_ci95']) for a,v in ar.items()],
        'The 95% conditional interval for excess error is 0.680 to 0.896. Ranking is exploratory and is not a clinical threshold.',
        'The pooled out-of-fold comparison covers 91 sources, including 62 annotated abnormal sources. Average precision is 0.865 for excess error at prevalence 0.681. Cross-fold score comparability and unknown patient identity remain limitations.',sources=[result_source])
    pr=ar['persistence']; ex=ar['excess_error']
    add('A low-motion control challenges the anomaly story.',
        'A post hoc diagnostic reverses persistence to ask whether limited movement explains the association.',
        'auc',[dict(label='Excess error: declared score',value=ex['auroc'],ci=ex['auroc_ci95']),dict(label='Low-motion score: post hoc',value=1-pr['auroc'],ci=[1-pr['auroc_ci95'][1],1-pr['auroc_ci95'][0]],posthoc=True)],
        f"Negated persistence has AUROC {1-pr['auroc']:.3f}. The excess-error result does not demonstrate superiority to this diagnostic control. Intervals condition on fixed fits.",
        'Negating the persistence error is an outcome-informed diagnostic, not a newly prespecified endpoint. Its AUROC equals one minus the original AUROC; the reflected interval is conditional on the original bootstrap distribution. This finding motivates freezing motion and static-posture controls in the next study. It does not license reversing other scores to report success.',status='POST HOC / DIAGNOSTIC',sources=[result_source],source_short='Source: results.json; AUROC(−score) = 1 − AUROC(score), including the standard half-credit treatment of ties.')
    add('The next study must test what the score adds.',
        'A stronger result would separate abnormal movement patterns from amplitude and acquisition effects.',
        'table',tab(['Alternative explanation','Required comparator','Useful interpretation'],[
            ['Low movement amplitude','Negated persistence and prefix motion.','The new score adds information.'],
            ['Static posture','Current-pose and pose-distance models.','Dynamics provide measurable value.'],
            ['Occlusion or view','Missingness, quality and view strata.','Results survive observation changes.'],
            ['Source composition','Source and external participant tests.','The effect transfers beyond uploads.']], [2.8,3.8,4.4]),
        'The existing result motivates this comparison. It cannot substitute for the comparison.',
        'Freeze the complete baseline menu, score direction and aggregation before opening new evaluation labels. Use a supervised RF only as a separately identified label-assisted reference. Avoid comparing it as if it had the same normal-only information budget.',status='PROPOSED / NOT RUN',sources=['../docs/iteration02/literature_and_masking.md'])
    add('Future targets stay hidden in every mask experiment.',
        'Each pattern hides 24 prefix cells, protects the latest anchor, and predicts the same hidden future.',
        'masks',None,
        'Green cells are visible, pale orange cells are hidden prefix inputs, and navy cells are future targets. The four rows are labeled joints.',
        'The grids illustrate knee, ankle, heel and foot joints over the declared 16-plus-8 time contract. They are conceptual mask patterns, not observed poses. Each hides exactly 24 valid prefix cells and preserves the latest target anchors. The anatomical tube hides a connected foot region; the random tube uses separated joint rows over identical times. Match realized valid hidden-token counts in the actual implementation and retain the same future support. Random and anatomical tubes must be matched for target count and visibility. SLiM already uses connected anatomical semantic tubes, so anatomical masking itself is not claimed as new.',status='PROPOSED / NOT RUN',sources=['https://arxiv.org/abs/2308.07092','https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf','https://arxiv.org/html/2603.10648v4'],source_short='Sources: MAMP (2023), S-JEPA (2024) and SLiM (2026). The forecasting comparison is proposed.')
    add('Motion-aware masking must retain quiet movement.',
        'A complementary design tests high-motion and low-motion regions under the same target budget.',
        'complementary',None,
        'Slow or constrained motion may be informative. Mask selection must not depend on the future or a diagnosis label.',
        'MAMP and S-JEPA motivate motion-aware masking. ASMa motivates complementary high- and low-motion or structural selection. Estimate motion only from reliable prefix pairs, cap sensitivity to pose jumps, and compare with a shuffled-weight control. Treat an implementation inspired by these papers as an adaptation unless their objective is reproduced.',status='PROPOSED / NOT RUN',sources=['https://arxiv.org/abs/2308.07092','https://openreview.net/forum?id=kIFo1q3VMS'],source_short='Sources: MAMP; ASMa, TMLR 2026. Prefix-only reliability gating is a proposed adaptation.')
    add('Phase-based masks require reliable cycle support.',
        'A short prefix may not contain enough observed gait cycles to estimate phase.',
        'gate',None,
        'Coverage is an outcome of the eligibility rule. It must be reported by source and annotation before interpretation.',
        'Use a blinded event-validation sample and a declared minimum of two reliable cycles before testing phase-aligned complementary masks. Longer prefixes need a separately identified cohort and matched fixed-time comparator. Do not infer a Parkinson phase-coordination index from unvalidated 2D peaks.',status='PROPOSED / CONDITIONAL',sources=['https://doi.org/10.1007/s00221-007-0955-7'],source_short='Source: Plotnik et al. 2007 motivates timing; phase eligibility and mask tests are proposed.')
    add('A masking comparison needs a matched contract.',
        'Changing the objective, data and training budget at once prevents a clear interpretation.',
        'table',tab(['Keep fixed','Match or measure','Do not allow'],[
            ['Sources and future targets','Valid hidden-token count.','Different test support by mask.'],
            ['Encoder and parameter budget','Updates, seeds and wall time.','A larger model for one mask.'],
            ['Normal-only training inputs','Reliable prefix-only mask features.','Future or abnormal-label access.'],
            ['Full-prefix primary inference','Separate stress-mask evaluation.','Test masks chosen from errors.']], [3,3.3,4.2]),
        'Learned or adversarial masks are an optional second stage after deterministic comparisons pass.',
        'Prevent a learned mask selector from removing almost all useful observations or optimizing on evaluation labels. Bound its visibility and entropy, train it only within the fitting split, and compare with the same-budget fixed policy. Learned-mask results remain exploratory until a fresh evaluation.',status='PROPOSED / NOT RUN',sources=['../docs/iteration02/literature_and_masking.md'])
    add('Representation prediction needs a coordinate check.',
        'A low embedding error can reflect a weak representation rather than a useful forecast.',
        'objectives',None,
        'Fit any coordinate probe on training sources only. Evaluate the same future coordinates for every objective.',
        'Compare coordinate prediction with a separately declared S-JEPA-inspired latent prediction objective. Freeze architecture and masks for this objective comparison. Inspect latent variance, covariance, effective rank, target similarity and constant-input controls. A stop-gradient teacher alone does not prove that collapse is absent.',status='PROPOSED / NOT RUN',sources=['https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf'],source_short='Source: S-JEPA motivates latent targets; the paired coordinate evaluation is proposed.')
    add('Two-sided calibration tests both tails of normal variation.',
        'Normal-only calibration tests unusually easy or difficult prediction relative to observed prefix conditions.',
        'twotails',None,
        'This is a proposed score. The displayed curve is conceptual and is not a fitted clinical distribution.',
        'Calibrate a predeclared forecast-residual statistic on cross-fitted normal development predictions, conditional on a small set of prefix covariates. Use a tail rule defined in advance and report coarse resolution from the small normal-source sample. Low and high tails do not identify rigidity, disorder or any clinical cause. Combining several tails requires a frozen combination and new calibration. Clinical coverage cannot be promised under source shift.',status='PROPOSED / NOT RUN',sources=['../docs/iteration02/literature_and_masking.md'])
    add('Calibration uses development sources, not test labels.',
        'Fit the reference on cross-fitted normal development predictions, then freeze its transformation.',
        'flow',['Normal development sources','Cross-fitted forecast residuals','Small conditional reference','Frozen held-out tail score'],
        'A small normal-source sample limits calibration resolution. Empirical ranks do not establish clinical coverage under source shift.',
        'Declare a scalar residual statistic, the conditioning variables and the two-sided transformation before outer evaluation. Keep marginal and conditional calibration as distinct arms. Freeze any combination of multiple tails. Report within-fold ranking as well as pooled calibrated ranking. Thresholds require a separately valid calibration mechanism.',status='PROPOSED / NOT RUN',sources=['../../../notes/aaai-worldmodels/05-iterate-02-INSTR.md'])
    add('Evaluate deterministic models before stochastic ones.',
        'More forecast diversity is useful only if it improves a declared calibrated endpoint.',
        'flow',['Persistence and current pose','Small deterministic forecaster','Matched mask and objective tests','Optional stochastic extension'],
        'Use identical data and compute limits. Keep means, uncertainty and anomaly ranking as separate endpoints.',
        'A stochastic model must be evaluated with proper distribution scores, coverage and calibration, not only its best sampled trajectory. Do not use minimum-of-many-samples error as evidence of reliable clinical uncertainty. Model complexity is a staged decision, not a prerequisite for novelty.',status='PROPOSED / NOT RUN',sources=['../docs/iteration02/literature_and_masking.md'])
    add('The notebooks should teach one decision at a time.',
        'Each notebook needs independent setup, an input identity and an explicit execution status.',
        'table',tab(['Notebook','Purpose','Required evidence'],[
            ['16: Evidence and mask design','Audit scores and illustrate masks.','Use frozen results and declared examples.'],
            ['20: Cohort and identities','Audit full GAVD and external eligibility.','Record attrition and source links.'],
            ['21: Calibration and confounds','Freeze motion and posture controls.','Use training-only calibration.'],
            ['22: Mask laboratory','Compare visible-token policies.','Test isolation and matched support.'],
            ['23: Objectives and fitting','Fit controlled model comparisons.','Save seeds, budgets and checkpoints.']], [2.7,4.1,4.5],size=15),
        'Notebook 16 is the bridge audit. Notebooks 20 through 26 describe the proposed next execution sequence.',
        'Historical notebooks 00 through 07 and completed restart notebooks 10 through 15 remain intact. New demonstrations must distinguish synthetic masks, cached result inspection, real fitting and completed evaluation. A successful tutorial execution is not new efficacy evidence.',status='PLAN / EXECUTION STATUS IN NOTEBOOKS',sources=['../../../notes/aaai-worldmodels/05-iterate-02-INSTR.md'])
    add('Evaluation and interpretation complete the sequence.',
        'The final notebooks should make every manuscript claim reproducible.',
        'table',tab(['Notebook','Purpose','Required evidence'],[
            ['24: Source and external evaluation','Estimate paired effects and transfer.','Source tables and frozen score directions.'],
            ['25: Robustness and interpretation','Challenge quality and biological claims.','Condition counts and alternative explanations.'],
            ['26: Manuscript evidence','Export figures and a bounded narrative.','Claim ledger, source hashes and review.']], [3,4.2,4.8],size=17),
        'External evaluation must use compatible observations and verified participant identities, where available.',
        'A model chosen using abnormal labels belongs to a separately identified supervised analysis. Newly acquired external data remain untouched until the protocol, calibration and comparison menu are frozen. Missing external evidence remains an explicit limit.',status='PROPOSED / NOT RUN',sources=['../../../notes/aaai-worldmodels/05-iterate-02-INSTR.md'])
    add('Broader GAVD use needs a separate transfer question.',
        'The seven remaining labels add diverse sources but no new annotated normal sources.',
        'flow',['Audit every available label','Declare transfer label eligibility','Keep related uploads together','Evaluate the frozen model once'],
        'Exercise, pregnancy and style labels are not automatically disease. A larger denominator alone does not improve clinical validity.',
        'The current inventory adds 245 previously unselected source IDs outside the five-label scope. These videos were not cached in the completed audit. Future downloads must use the shared source directory and record unavailable media. Acquisition stops at the declared inventory or resource boundary, not when a desired significance level is reached.',status='PROPOSED / NOT RUN',sources=['../docs/restart/data_audit.md'],source_short='Source: complete GAVD inventory audit; expanded-label transfer is proposed.')
    add('GAVD establishes only bounded generalization.',
        'The strongest immediate claim is heterogeneous-source internal evaluation.',
        'table',tab(['Generalization question','Current evidence','Needed extension'],[
            ['New source video','Evaluated by source folds.','More independently acquired sources.'],
            ['New participant','Identity is not verified.','Participant-linked external data.'],
            ['New clinical site','No site-disjoint cohort.','Independent sites and protocols.'],
            ['Treatment response','No interventions are observed.','Longitudinal action and outcome data.']], [2.8,3.7,4.4]),
        'More windows cannot establish independent patients, verified diagnoses or safe clinical deployment.',
        'Selection by confidence, view and person gates can alter the population. The README documents these limits explicitly. External generalization should evaluate both predictive performance and calibration, with compatibility checks for pose format and capture conditions.',sources=['../docs/restart/data_audit.md','../README.md'])
    add('Adversarial review should challenge every claimed gain.',
        'The next manuscript becomes stronger when its most plausible alternative explanations are tested.',
        'table',tab(['Reviewer challenge','Acceptance evidence'],[
            ['Did the model see the future?','Future perturbations leave inputs and predictions unchanged.'],
            ['Did more capacity cause the gain?','Matched architectures, visible tokens and training budgets.'],
            ['Is the score only low movement?','Frozen amplitude and current-pose controls.'],
            ['Are the samples independent?','Source and duplicate audits, plus explicit patient limits.'],
            ['Can the result be replayed?','Saved predictions, source tables, environment and hashes.']], [3.3,7],size=16),
        'A review must examine the exact delivered files. A material edit reopens the affected checks.',
        'Independent reviewers should reproduce source metrics, inspect figure geometry and language, and challenge novelty against related skeletal anomaly work. Fix the issue or narrow the claim; do not silently replace an unfavorable endpoint.',status='REVIEW / ACCEPTANCE CONTRACT',sources=['../../../.agents/protocol.md'])
    add('A strong contribution links evidence to a precise claim.',
        'The current signal supports a transparent benchmark and a focused next experiment.',
        'flow',['Traceable observation pipeline','Reference-aware anomaly audit','Neurologically motivated mask tests','Independent validation'],
        'The next claim should grow only when the corresponding evidence grows.',
        'Read draft02 for the standalone manuscript, the iteration02 instruction for execution requirements, and the literature registry for primary sources. The active deck preserves the completed restart evidence while marking unrun methods as proposed.',status='SYNTHESIS / EVIDENCE AND PLAN',sources=['../docs/draft02/README.md','../../../notes/aaai-worldmodels/05-iterate-02-INSTR.md','../docs/iteration02/literature_and_masking.md'])
    return slides


def build():
    global BOXES
    r=json.loads((RUN/'results.json').read_text())
    prep=json.loads((DATA/'preparation.json').read_text())
    inv=json.loads((DATA/'inventory.json').read_text())
    assert r['status']=='complete'
    slides=make_slides(r,prep,inv)
    assert all('—' not in json.dumps(s,ensure_ascii=False) for s in slides)
    ASSETS.mkdir(parents=True,exist_ok=True)
    manifest=[]
    checks=[]
    with PdfPages(OUT/'iteration02.pdf', metadata={'Title':'Reference-aware gait anomaly detection: evidence and next experiments','Author':'AlexPose research artifacts'}) as pdf:
        for i,s in enumerate(slides,1):
            BOXES=[]
            fig,ax=canvas(s,i,len(slides));draw(s,fig,ax)
            fig.canvas.draw();renderer=fig.canvas.get_renderer()
            issues=[]
            for patch,label in BOXES:
                pb=patch.get_window_extent(renderer);tb=label.get_window_extent(renderer)
                if tb.x0<pb.x0+4 or tb.x1>pb.x1-4 or tb.y0<pb.y0+4 or tb.y1>pb.y1-4:
                    issues.append('Box overflow: '+label.get_text().replace('\n',' '))
            bounds=fig.bbox
            text_bounds=[]
            for text in fig.findobj(match=matplotlib.text.Text):
                if not text.get_visible() or not text.get_text():continue
                b=text.get_window_extent(renderer)
                if b.width and b.height and (b.x0 < 0 or b.y0 < 0 or b.x1 > bounds.x1 or b.y1 > bounds.y1):
                    issues.append('Canvas overflow: '+text.get_text().replace('\n',' '))
                if b.width and b.height:
                    text_bounds.append((text,b))
            for j,(ta,ba) in enumerate(text_bounds):
                for tb,bb in text_bounds[j+1:]:
                    if min(ba.x1,bb.x1)-max(ba.x0,bb.x0)>2 and min(ba.y1,bb.y1)-max(ba.y0,bb.y0)>2:
                        issues.append('Text overlap: '+ta.get_text().replace('\n',' ')+' / '+tb.get_text().replace('\n',' '))
            for ext in ('svg','png','pdf'):
                fig.savefig(ASSETS/f'slide-{i:02d}.{ext}',dpi=100,facecolor=BG)
            pdf.savefig(fig,facecolor=BG);plt.close(fig)
            checks.append(dict(slide=i,title=s['title'],issues=issues))
            manifest.append(dict(slide=i,title=s['title'],status=s['status'],svg_sha256=sha(ASSETS/f'slide-{i:02d}.svg')))
    cols=4;rows=math.ceil(len(slides)/cols)
    fig,axs=plt.subplots(rows,cols,figsize=(20,rows*3),squeeze=False)
    for a in axs.flat:a.axis('off')
    for i,a in enumerate(axs.flat):
        if i<len(slides):a.imshow(plt.imread(ASSETS/f'slide-{i+1:02d}.png'))
    fig.tight_layout(pad=.4);fig.savefig(OUT/'iteration02_contact_sheet.png',dpi=100);plt.close(fig)
    parts=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reference-aware gait anomaly detection</title><style>body{margin:0;background:#e9eff3;font:18px/1.6 system-ui;color:#153448}header,main{max-width:1280px;margin:auto}header{padding:28px}section{background:white;margin:28px 0;padding-bottom:24px}img{display:block;width:100%;height:auto}details{padding:0 30px}a{color:#087d78}nav{position:sticky;top:0;background:#153448;color:white;padding:10px;text-align:center;z-index:3}nav a,nav button{color:white;margin:0 16px;background:none;border:0;font:inherit;cursor:pointer}li{margin:5px 0}summary{font-weight:600}@media print{nav,header,details{display:none}section{margin:0;padding:0;break-after:page}main{max-width:none}@page{size:landscape;margin:0}}</style></head><body><header><h1>Reference-aware gait anomaly detection</h1><p>Illustrated evidence, neurological reasoning, and the next controlled experiments. Every slide distinguishes measured evidence, post hoc diagnostics, and proposed work.</p><p><a href="iteration02.pdf">Vector PDF</a> · <a href="../docs/draft02/README.md">Standalone manuscript</a> · <a href="../../../notes/aaai-worldmodels/05-iterate-02-INSTR.md">Iteration instructions</a> · <a href="restart.html">Preserved restart deck</a></p></header><nav><button onclick="move(-1)">Previous</button><button onclick="move(1)">Next</button><a href="iteration02.pdf">Download PDF</a></nav><main>']
    for i,s in enumerate(slides,1):
        refs=''.join(f'<li><a href="{html.escape(url)}">{html.escape(url)}</a></li>' for url in s['sources'])
        parts.append(f'<section id="s{i}"><img src="iteration02_images/slide-{i:02d}.svg" alt="{html.escape(s["title"]+" "+s["subtitle"]+" "+s["caption"])}"><details open><summary>Explanation, status and evidence</summary><p><strong>{html.escape(s["status"])}</strong></p><p>{html.escape(s["notes"])}</p><ul>{refs}</ul></details></section>')
    parts.append(f'</main><script>let current=Number(location.hash.slice(2))||1;function move(d){{current=Math.max(1,Math.min({len(slides)},current+d));document.getElementById("s"+current).scrollIntoView({{behavior:"smooth"}});history.replaceState(null,"","#s"+current)}}document.addEventListener("keydown",e=>{{if(e.key==="ArrowRight")move(1);if(e.key==="ArrowLeft")move(-1)}});</script></body></html>')
    (OUT/'iteration02.html').write_text(''.join(parts))
    (OUT/'tutorial.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=iteration02.html"><title>Current gait tutorial</title><p><a href="iteration02.html">Open the current illustrated gait tutorial.</a></p></html>\n')
    (OUT/'iteration02_source.json').write_text(json.dumps(slides,indent=2,ensure_ascii=False)+'\n')
    (OUT/'iteration02_manifest.json').write_text(json.dumps(dict(identity='medwm-iteration02-deck-v1',results_identity=r['identity'],inputs={str(path.relative_to(ROOT)):sha(path) for path in [RUN/'results.json',DATA/'preparation.json',DATA/'inventory.json']},generator_sha256=sha(Path(__file__)),slide_count=len(slides),slides=manifest,pdf_sha256=sha(OUT/'iteration02.pdf')),indent=2)+'\n')
    (OUT/'iteration02_layout_checks.json').write_text(json.dumps(dict(slide_count=len(slides),checks=checks,issue_count=sum(len(x['issues']) for x in checks)),indent=2)+'\n')
    print(json.dumps({'slides':len(slides),'layout_issues':[x for x in checks if x['issues']]},indent=2))


if __name__=='__main__':
    build()
