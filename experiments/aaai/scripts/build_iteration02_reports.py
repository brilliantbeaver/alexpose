"""Build the shared evidence record and publication figures from saved runs."""
from pathlib import Path
import json,sys,shutil,hashlib
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'experiments/aaai';sys.path.insert(0,str(BASE))
from aaai_gait.data import file_sha256,JOINTS
from aaai_gait.media import EDGES
from aaai_gait.iteration_data import load_iteration_data
from aaai_gait.study_runner import dump_json
RUN=BASE/'artifacts/iteration02/d66e1b126c9d9a9adf1b'
EXTERNAL=BASE/'artifacts/iteration02/external_fb9d0d23b56ce29fd846'
PAPER=BASE/'docs/draft02';FIG=PAPER/'figures';FIG.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
m=read(RUN/'rawmetrics.json');intervals=read(RUN/'paired_intervals.json')
data=read(RUN/'run_manifest.json');cohort=read(Path(data['cohort_directory'])/'provenance.json')
timing=read(RUN/'timing_summary.json');external=read(EXTERNAL/'metrics.json');external_interval=read(EXTERNAL/'paired_effects.json')['M'];matched=read(EXTERNAL/'age_sex_sensitivity.json')
scores=pd.read_csv(RUN/'source_scores.csv');adequacy=pd.read_csv(RUN/'adequacy.csv');normal=pd.read_csv(RUN/'normal_error_table.csv')
main={ep:{key:value for key,value in m[f'U/U/{ep}'].items()} for ep in ('focal','broad')}
primary=next(x for x in intervals if x['contrast']=='method' and x['endpoint']=='focal' and x['second']=='M')
broad=next(x for x in intervals if x['contrast']=='method' and x['endpoint']=='broad' and x['second']=='M')
evidence=dict(protocol_hash=RUN.name,external_protocol_hash=EXTERNAL.name,cohort=cohort,
    main=main,primary_difference=primary,broad_difference=broad,all_mask_metrics=m,paired_intervals=intervals,
    normal_adequacy=adequacy.to_dict('records'),normal_test_error=normal.to_dict('records'),
    normal_test_benefit=read(RUN/'normal_test_benefit.json'),timing=timing,
    external=dict(metrics=external,primary_difference=external_interval,age_sex_sensitivity=matched,
                  verification=read(EXTERNAL/'adversarial_verification.json')),
    supervised=read(RUN/'five_class_metrics.json'),view_sensitivity=read(RUN/'view_sensitivity.json'),
    verification=read(RUN/'verification.json'),claims=dict(
      learned_context='All12 GAVD operational internal gates passed; uniform-bank outer-normal benefit remains uncertain for U and A.',
      mask_novelty='Known tube families, controlled budgets and marginal probabilities; no new mask-family claim.',
      timing='Coverage passes but every exact-paired-support adequacy gate fails. No reliable coordination interpretation.',
      detection='Fixed fusion lowers AUROC relative to the local channel in GAVD and the independent laboratory replication.',
      external='Participant-separated six-angle method replication, not transfer of video weights or clinical validation.'),
    numerical_sources={str(p.relative_to(BASE)):file_sha256(p) for p in [RUN/'source_scores.csv',RUN/'rawmetrics.json',RUN/'paired_intervals.json',RUN/'timing_summary.json',EXTERNAL/'scores.csv',EXTERNAL/'metrics.json',EXTERNAL/'paired_effects.json',EXTERNAL/'age_sex_sensitivity.json']})
dump_json(PAPER/'evidence.json',evidence)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
records=[]
def save(fig,name,caption,inputs):
    for suffix in ('svg','pdf','png'):fig.savefig(FIG/f'{name}.{suffix}',bbox_inches='tight',dpi=180)
    records.append(dict(name=name,caption=caption,inputs=[str(p.relative_to(BASE)) for p in inputs],
                        hashes={str(p.relative_to(BASE)):file_sha256(p) for p in inputs},raster_photo_content=False))
    plt.close(fig)

# A readable diagram separates biological motivation from measured quantities.
fig,ax=plt.subplots(figsize=(7.1,2.5));ax.set_xlim(0,10);ax.set_ylim(0,3);ax.axis('off')
nodes=[(.05,1.8,2.25,.95,'Control, muscle,\nand body structure\nshape movement'),(2.65,1.8,2.15,.95,'Camera and pose\nestimation alter\nwhat is observed'),(5.15,1.8,2.1,.95,'Preserve posture,\namplitude and\nraw timing'),(7.6,1.8,2.3,.95,'M: local movement\nDistance from\nfitting normals'),
       (.05,.2,2.25,.95,'Shared answer Y\nand shared body\nreference Q'),(2.65,.2,2.15,.95,'Retain own-leg f\nCompare corrections\nf+h and f+g'),(5.15,.2,2.1,.95,'Check prediction\nand permitted\ntiming support'),(7.6,.2,2.3,.95,'R: benefit departure\nFixed score\nS = (M + R) / 2')]
for x,y,w,h,text in nodes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',facecolor='#e8f4f3',edgecolor='#17858a',linewidth=.9));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8.2)
for y in (2.2,.65):
    for x in (2.32,4.85,7.3):ax.annotate('',xy=(x+.29,y),xytext=(x,y),arrowprops=dict(arrowstyle='->',lw=1,color='#345'))
ax.text(0,1.4,'Biology motivates the measurements. Prediction does not identify a neurological cause.',fontsize=8,color='#704c27')
save(fig,'method','Implemented measurement workflow; neurological mechanisms are motivation, not recovered latent causes.',[RUN/'protocol.json'])

fig,ax=plt.subplots(figsize=(3.45,2.45))
items=[('GAVD focal',primary),('GAVD broad',broad),('Laboratory',external_interval),('Age/sex matched',dict(observed=matched['S_minus_M'],conditional_95_interval=matched['conditional_95_interval']))]
for y,(label,v) in enumerate(items):
    lo,hi=v['conditional_95_interval'];point=v['observed'];ax.errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt='o',color='#ac462f',capsize=3)
ax.axvline(0,color='#6a7078',lw=1);ax.set_yticks(range(4),[a for a,_ in items]);ax.invert_yaxis();ax.set_xlabel('AUROC difference: fixed S minus local M');ax.grid(axis='x',alpha=.18)
save(fig,'detection_difference','All differences and intervals are conditional on fitted models and declared protocols; the laboratory matched interval resamples pairs.',[RUN/'paired_intervals.json',EXTERNAL/'paired_effects.json',EXTERNAL/'age_sex_sensitivity.json'])

fig,axes=plt.subplots(1,2,figsize=(7.0,2.9))
for ax,bank,title in zip(axes,['U','E_leg'],['Unchanged uniform task','Distal-leg challenge task']):
    table=normal[normal.bank.eq(bank)].groupby('policy')[['loss_base','loss_own','loss_both']].mean().reindex(['U','J','A'])
    x=np.arange(3)
    for k,(col,name,color) in enumerate([('loss_base','Retained f','#a8b1ba'),('loss_own','Own-capacity f+h','#4073ad'),('loss_both','Both-leg f+g','#17858a')]):
        ax.bar(x+(k-1)*.23,table[col],.22,label=name,color=color)
    ax.set_xticks(x,['U','J','A']);ax.set_xlabel('Correction-training mask policy');ax.set_ylabel('Mean held-out normal XY MSE');ax.set_title(title);ax.grid(axis='y',alpha=.15)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=7);fig.tight_layout(rect=(0,.10,1,1))
save(fig,'normal_prediction','All28 held-out normal sources contribute. Columns are different observation tasks, so cross-column differences do not isolate training policy.',[RUN/'normal_error_table.csv'])

fig,axes=plt.subplots(1,2,figsize=(7.0,2.5));summ=[v for v in timing['summaries'] if v['scope']=='outer_test_normal' and v['policy']=='U' and v['variant']=='timing']
summ=sorted(summ,key=lambda r:r['offset'])
axes[0].bar(['-4 frames','+4 frames'],[v['paired_gain_over_own'] for v in summ],color='#b95b39');axes[0].axhline(0,color='#555',lw=1);axes[0].set_ylabel('Aligned error: own minus both');axes[0].set_title('Paired-support adequacy fails')
for i,v in enumerate(summ):
    lo,hi=v['conditional_source_bootstrap_95_interval'];point=v['mean_effect'];axes[1].errorbar(i,point,yerr=[[point-lo],[hi-point]],fmt='o',capsize=3,color='#516a86')
axes[1].set_xticks([0,1],['-4 frames','+4 frames']);axes[1].axhline(0,color='#555',lw=1);axes[1].set_ylabel('Shifted error minus aligned error');axes[1].set_title('Sensitivity is not coordination evidence');fig.tight_layout()
save(fig,'timing_limits','27 held-out normal recordings per offset. About1.75 editable distal tokens remain; a positive response does not rescue failed prediction adequacy.',[RUN/'timing_summary.json'])

# A camera-free failure illustration uses measured skeleton traces, not photos.
d=load_iteration_data(data['cohort_directory']);review=pd.read_csv(RUN/'video_review/review_ledger.csv')
cases=[review[(review.label=='normal')&(review.selection=='high')].iloc[0],review[(review.label=='parkinsons')&(review.selection=='low')].iloc[0]]
fig,axes=plt.subplots(1,2,figsize=(3.45,2.7));index={j:k for k,j in enumerate(JOINTS)}
for ax,case in zip(axes,cases):
    matches=d.metadata[d.metadata.sequence_id.eq(case.sequence_id)]
    ix=(matches.start_seconds-case.preview_start_seconds).abs().idxmin();window=d.x[ix]
    for color,t in zip(['#a9bcc8','#588d9a','#185f70'],[0,15,31]):
        for a,b in EDGES:
            pair=window[t,[index[a],index[b]]]
            ax.plot(pair[:,0],pair[:,1],color=color,lw=1)
    ax.set_aspect('equal');ax.invert_yaxis();ax.set_title(f'{case.label} folder\n{case.selection} completion error',fontsize=8);ax.set_xlabel('Torso-scaled image X');ax.grid(alpha=.12)
axes[0].set_ylabel('Torso-scaled image Y');fig.tight_layout()
save(fig,'failure_cases','Prespecified extreme-error examples use actual projected poses. Different viewpoints and regular motion prevent raw error from being a health ordering.',[RUN/'video_review/review_ledger.csv',RUN/'window_metadata.csv'])
dump_json(FIG/'manifest.json',records)
for name in ['template.tex','build_manuscript.py','references.bib']:
    if not (PAPER/name).exists():shutil.copy2(BASE/'docs/draft01'/name,PAPER/name)
if not (PAPER/'author_kit').exists():shutil.copytree(BASE/'docs/draft01/author_kit',PAPER/'author_kit')
print(PAPER/'evidence.json')
