"""Render a source-backed vector teaching deck from the completed restart run."""
from pathlib import Path
import json,html,textwrap,hashlib,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'slides';ASSETS=OUT/'restart_images'
RUN=ROOT/'artifacts/runs/restart-dynamics-v1';DATA=ROOT/'artifacts/runs/restart-full10hz-v1'
NAVY='#17334a';TEAL='#14857a';BLUE='#e0edf6';GREEN='#dcefe9';ORANGE='#f9e7cf';GRAY='#587084'
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,'font.size':17})

def wrap(s,width=36):return '\n'.join(textwrap.wrap(s,width))
def canvas(title,subtitle):
    fig,ax=plt.subplots(figsize=(16,9));fig.patch.set_facecolor('#fbfcfe');ax.set(xlim=(0,16),ylim=(0,9));ax.axis('off')
    ax.text(.65,8.3,title,fontsize=28,weight='bold',color=NAVY)
    ax.text(.65,7.65,wrap(subtitle,113),fontsize=16,color=GRAY,va='top',linespacing=1.4)
    return fig,ax

def box(ax,x,y,w,h,text,color=BLUE,size=18,width=28):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03,rounding_size=.12',fc=color,ec='#b8cbd8',lw=1))
    ax.text(x+w/2,y+h/2,wrap(text,width),ha='center',va='center',fontsize=size,color=NAVY,linespacing=1.4)

def arrow(ax,a,b,label=None):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':2,'color':TEAL})
    if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.2,label,ha='center',color=GRAY,fontsize=12)

def diagram(ax,nodes):
    w=13.9/len(nodes);y=3.4
    for i,text in enumerate(nodes):
        x=.7+i*(w+.1);box(ax,x,y,w-.3,2,text,GREEN if i==len(nodes)-1 else BLUE,size=17,width=max(12,int(w*5)))
        if i<len(nodes)-1:arrow(ax,(x+w-.22,y+1),(x+w+.02,y+1))

def table(ax,cols,rows):
    t=ax.table(cellText=rows,colLabels=cols,loc='center',cellLoc='left',bbox=[.05,.16,.9,.53]);t.auto_set_font_size(False);t.set_fontsize(15)
    for (i,j),cell in t.get_celld().items():
        cell.set_edgecolor('white');cell.set_facecolor(NAVY if i==0 else (BLUE if i%2 else '#eef4f7'))
        if i==0:cell.set_text_props(color='white',weight='bold')
    return t

def build():
    r=json.loads((RUN/'results.json').read_text());prep=json.loads((DATA/'preparation.json').read_text())
    ASSETS.mkdir(parents=True,exist_ok=True);slides=[]
    def add(title,subtitle,kind,payload,notes):slides.append(dict(title=title,subtitle=subtitle,kind=kind,payload=payload,notes=notes))
    add('Gait forecasting can be tested before clinical claims.', 'A transparent observation layer for medical world models.','flow',['Real gait videos','Observed articulation','Future prediction','Evidence and limits'],
        'The restart examines a local movement transition. It does not model disease progression, treatment effects, or clinical decisions. All new results are exploratory because earlier GAVD outcomes were known.')
    add('We turned the earlier study into a clearer question.','Classification accuracy did not establish useful temporal dynamics.','flow',['Review old results','Fix time and coverage','Compare strong references','Test the claimed information'],
        'The prior short-prefix JEPA and RF results remain preserved under their original run identities. The new method is a separate experiment, not a relabeling of those results.')
    add('Neurological reasoning motivates coordination tests.','The evidence supports a hypothesis about observed movement, not a measurement of neural circuits.','table',(['Evidence','Motivation','Claim boundary'],[['PD bilateral timing','Compare both legs','No basal-ganglia inference'],['Stroke EMG modules','Inspect coordination','No EMG measured here'],['CP movement complexity','Test structured history','No neural-module estimate'],['Myopathic annotation','Include neuromuscular gait','Not exclusively central disease']]),
        'Primary references include Plotnik et al. 2007, Clark et al. 2010, and Steele et al. 2015. The literature registry records what each source does and does not support.')
    add('We audited the complete GAVD annotation inventory.','Sequence counts, source counts, and patient counts answer different questions.','flow',['1,874 sequences in all labels','348 source IDs overall','666 selected sequences','103 selected source IDs'],
        'The seven other labels add 245 source IDs, but the selected scope already contains all 32 annotated normal sources. Extra abnormal labels cannot increase the normal-source support for the forecasting endpoint.')
    add('Full-duration processing expands movement coverage.','The extraction covers every eligible annotated span rather than its first short prefix.','flow',['652 media-eligible sequences','641 after source exclusions',f"{prep['n_windows']:,} quality windows",f"{r['windows']:,} model windows"],
        f"The completed model cohort has {r['sequences']} sequences and {r['sources']} source videos. Repeated windows remain dependent. Existing reference mismatches and two whole-source identity exclusions were preserved.")
    add('The final cohort keeps each condition visible.','These are dataset annotations, not independently verified diagnoses.','table',(['Annotation','Windows','Sequences','Sources'],[[v['label'],v['windows'],v['sequences'],v['sources']] for v in r['cohort_by_label']]),
        'The cohort table is generated directly from the completed model run. The final common distal-target gate is applied to every comparison before folds are assigned.')
    add('Every observation has a traceable path from video.','Source bytes, annotation rows, model weights and adapter identities are recorded.','flow',['One source file in videos-gavd','Ambient image pose API','Box and confidence checks','Native frames and timestamps'],
        f"Fresh extraction executed {prep['actual_model_calls_this_invocation']:,} actual pose model calls. New downloads use data/videos-gavd/sources, and previews use a separate previews directory. Existing nonduplicated originals are reused in place.")
    add('The observation contract has an explicit time scale.','A nominal 10 Hz grid selects native frames without interpolating skeletons.','timeline',None,
        'The prefix contains 16 samples spanning 1.5 seconds between endpoints. The final target is nominally 0.8 seconds after the last prefix frame. Actual native frames and decoder timestamps remain in the metadata; sampling is not presentation-time resampling.')
    add('The model sees articulation in a shared body reference.','Prefix-only torso scaling prevents future observations from changing the input scale.','flow',['33 raw pose joints','16 declared model joints','Pelvis-centered articulation','4 distal targets per leg'],
        'Both hips already enter the shared pelvis reference. Own-leg means own distal history conditional on that reference. Coordinates are image-plane articulation, not physical velocity or three-dimensional anatomy.')
    add('Missingness remains visible throughout the pipeline.','A missing coordinate never becomes an observed zero target.','flow',['Confidence at least 0.5','Prefix and future support','Same distal targets for all arms','Masked fitting and scoring'],
        'The study invalidates unresolved annotation geometry. Each leg needs at least four observed joint-time targets by horizon four and eight by horizon eight. Future target joints also need a valid prefix observation.')
    add('Bilateral information is a precise comparison.','The target and own-leg prefix stay fixed when opposite-leg observations are added.','bilateral',None,
        'Each learned model predicts residual articulation from the latest observed target pose. Ridge regularization controls capacity. The extra context has to improve held-out normal-source prediction to demonstrate useful conditional information.')
    add('A timing control must preserve the available values.','Each window receives a different hashed cyclic shift of the same selected opposite-leg pairs.','flow',['Select identical 8 times','Keep latest anchor fixed','Shift 7 historical pairs','Refit the matched control'],
        'A single fixed permutation would be absorbable by linear regression. Window-specific shifts preserve coordinates, validity and feature count while disrupting their alignment. This computational perturbation is not a biological intervention.')
    add('Persistence is part of the reliability strategy.','The prospective shrinkage rule did not improve the unshrunk bilateral forecast in this run.','flow',['Anchor pose','Learned future residual','Inner-selected beta','Anchor + beta × residual'],
        'Beta is selected from 0, 0.25, 0.5 and 1 using normal training-source inner validation. In the completed run, shrinkage did not improve unshrunk bilateral prediction. Every learned arm has the same grid. Beta zero means persistence. Original unshrunk arms remain primary.')
    add('Every source is held out once.','All windows and related annotations from one source remain together.','folds',None,
        'Five outer folds estimate source-transfer behavior. Three inner folds select regularization and shrinkage using normal sources only. Outer abnormal labels do not select forecasting settings. Earlier test exposure means this is exploratory cross-validation.')
    add('The scoring hierarchy gives sources equal weight.','Repeated clips provide temporal coverage rather than independent sample size.','flow',['Observed coordinates per leg','Equal left and right legs','Windows then sequences','Equal source averages'],
        'Masked training weights follow the same source, sequence and window hierarchy. Bootstrap intervals resample fixed out-of-fold source predictions. They condition on the trained folds and do not represent verified patient-level uncertainty.')
    normal=pd.DataFrame(r['normal_forecast']).query('horizon == 8').set_index('arm')
    shown=['persistence','damped_velocity','static','own','bilateral','shifted','bilateral_shrunk']
    add('The current-pose learned transition forecasts best.',f"Current pose: {normal.loc['static','estimate']:.5f} MSE. Bilateral history: {normal.loc['bilateral','estimate']:.5f}. Persistence: {normal.loc['persistence','estimate']:.5f}. Lower is better.",'errors',[(a,float(normal.loc[a,'estimate']),normal.loc[a,'ci95']) for a in shown],
        'The static-context arm is a learned, current-pose-conditioned future transition, not a constant output. It forecasts better than bilateral history. Errors average all eight nominal future samples from 0.1 to 0.8 seconds. Error bars are conditional source bootstrap intervals. The complete tables include every unshrunk and shrunk arm, both horizons, and inference perturbations. No unfavorable comparison is omitted from the run.')
    comparisons=[v for v in r['paired_contrasts'] if v['horizon']==8 and (v['reference'],v['method']) in [('persistence','bilateral'),('own','bilateral'),('shifted','bilateral'),('static','bilateral'),('persistence','bilateral_shrunk')]]
    add('Paired effects separate utility from timing evidence.','Positive differences favor the listed forecast method over its reference.','effects',comparisons,
        'A positive forecast gain does not by itself establish a neural mechanism. Read the matched shifted-context result separately. Intervals condition on overlapping cross-validation fits, and no confirmatory clinical significance claim is made.')
    add('Abnormality ranking is a separate exploratory endpoint.','Score directions were fixed before evaluation; a poor ranking is not reversed afterward.','table',(['Score','AUROC','95% conditional interval'],[[a,f"{v['auroc']:.3f}",f"{v['auroc_ci95'][0]:.3f} to {v['auroc_ci95'][1]:.3f}"] for a,v in r['anomaly'].items()]),
        'Higher error and higher deficit scores were declared to mean more abnormal. Pooled out-of-fold AUROC assumes score comparability across fitted folds. Persistence and prefix missingness are controls. These associations do not provide diagnostic sensitivity or clinical thresholds.')
    add('The notebooks execute the entire reasoning chain.','Each new notebook initializes its own paths and uses actual GAVD observations.','table',(['Notebook','Executed activity'],[['10 Inventory','Fresh labels, media and source audit'],['11 Video and time','Decode a real frame and call pose model'],['12 Representation','Check past-only inputs and target support'],['13 Modeling','Fit a real ridge model and inspect nested folds'],['14 Results','Read source effects and uncertainty'],['15 Claims','Connect evidence to manuscript boundaries']]),
        'The six restart notebooks supplement the preserved eight historical JEPA tutorials. Their execution record pins sources, outputs, environment and actual input identities.')
    add('A useful positive result still needs a bounded claim.','The manuscript distinguishes prediction, coordination evidence, and clinical meaning.','flow',['Forecast utility','Timing-specific information','Annotation association','External clinical validation'],
        'Success at one level does not establish the next. The draft reports its strongest supported contribution while keeping failed hypotheses, measurement limits, and the need for independent patients visible.')
    add('GAVD cannot establish every form of generalization.','Source grouping is necessary but does not prove patient independence or clinical transportability.','table',(['Question','Evidence status'],[['Different source videos','Evaluated internally with source folds'],['Different participants','Identity not verified across uploads'],['Different clinical sites','No prospective site-disjoint study'],['Treatment response','No action or intervention data'],['Broader labels','Full inventory audited; separate study needed']]),
        'Further progress needs clinician-reviewed identity and labels, external normal participants, stable capture conditions, and longitudinal or intervention observations. Additional windows alone cannot satisfy these requirements.')
    add('All deliverables point back to the same run.','Editable diagrams and published numbers are regenerated from saved evidence.','flow',['Frozen protocol and input hashes','Saved fits and predictions','Executed notebooks','Reviewed manuscript and slides'],
        'Read docs/restart/run_record.json, docs/restart/implementation_review.md, and docs/draft01/README.md for current completion and submission readiness. The workshop page lists November 20, 2026 Anywhere on Earth; the independent OpenReview deadline check remains unresolved.')
    manifest=[]
    with PdfPages(OUT/'restart.pdf') as pdf:
        for i,s in enumerate(slides,1):
            fig,ax=canvas(s['title'],s['subtitle']);kind=s['kind'];p=s['payload']
            if kind=='flow':diagram(ax,p)
            elif kind=='table':table(ax,*p)
            elif kind=='timeline':
                for t in range(24):
                    xx=.9+t*.59;ax.add_patch(plt.Rectangle((xx,4),.48,1.2,fc=BLUE if t<16 else GREEN,ec='white'))
                    if t%4==0 or t==23:ax.text(xx+.2,3.6,str(t),ha='center',fontsize=14)
                ax.axvline(10.1,ymin=.4,ymax=.64,color=TEAL,lw=2)
                ax.text(4.9,2.6,'16 prefix observations',ha='center',fontsize=22,color=NAVY)
                ax.text(12.8,2.6,'8 future observations',ha='center',fontsize=22,color=TEAL)
            elif kind=='bilateral':
                box(ax,.8,4.4,4.2,1.4,'Own distal-leg history',BLUE,width=20)
                box(ax,.8,2,4.2,1.4,'Opposite distal-leg history',GREEN,width=20)
                box(ax,6.3,3.2,3.1,1.6,'Regularized predictor',ORANGE,width=16)
                box(ax,11,3.2,4.1,1.6,'Future target-leg articulation',GREEN,width=20)
                arrow(ax,(5.1,5.1),(6.2,4.2));arrow(ax,(5.1,2.7),(6.2,3.8));arrow(ax,(9.5,4),(10.9,4))
            elif kind=='folds':
                for row in range(5):
                    ax.text(.8,5.8-row*.85,f'Outer fold {row+1}',fontsize=17)
                    for col in range(5):
                        box(ax,3.5+col*2.1,5.5-row*.85,1.8,.6,'Held out' if row==col else 'Fit / inner',GREEN if row==col else BLUE,size=13,width=15)
            elif kind in ('errors','effects'):
                aa=fig.add_axes([.29,.28,.64,.43]);aa.set_facecolor('#fbfcfe')
                if kind=='errors':
                    names=[a for a,_,_ in p];values=np.array([v for _,v,_ in p]);bounds=np.array([b for _,_,b in p]);aa.set_xlabel('Source-average squared articulation error',fontsize=14)
                else:
                    names=[v['method']+' vs '+v['reference'] for v in p];values=np.array([v['estimate'] for v in p]);bounds=np.array([v['ci95'] for v in p]);aa.axvline(0,color=GRAY,lw=1);aa.set_xlabel('Reference error minus method error',fontsize=14)
                yy=np.arange(len(names));aa.errorbar(values,yy,xerr=np.maximum(0,np.array([values-bounds[:,0],bounds[:,1]-values])),fmt='o',color=TEAL,capsize=4)
                aa.set_yticks(yy,names,fontsize=13);aa.invert_yaxis();aa.spines[['top','right']].set_visible(False);aa.grid(axis='x',alpha=.15)
            ax.text(.7,.55,'Med-WM | Exploratory fixed-time GAVD study',fontsize=12,color=GRAY)
            ax.text(15.2,.55,f'{i:02d} / {len(slides)}',ha='right',fontsize=12,color=GRAY)
            for ext in ('svg','png','pdf'):fig.savefig(ASSETS/f'slide-{i:02d}.{ext}',dpi=100)
            pdf.savefig(fig);plt.close(fig)
            manifest.append(dict(slide=i,title=s['title'],svg_sha256=hashlib.sha256((ASSETS/f'slide-{i:02d}.svg').read_bytes()).hexdigest()))
    html_parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gait dynamics research restart</title><style>body{margin:0;background:#eaf0f5;font:18px/1.6 system-ui;color:#17334a}header,main{max-width:1280px;margin:auto}header{padding:24px}section{background:white;margin:24px 0;padding:12px 12px 24px;border-radius:12px}img{display:block;width:100%;height:auto}details{padding:0 30px}nav{position:sticky;top:0;background:#17334a;padding:10px;text-align:center;z-index:2}nav a,nav button{color:white;margin:0 14px;background:transparent;border:0;font:inherit;cursor:pointer}a{color:#086e65}@media print{nav,header,details{display:none}section{margin:0;padding:0;break-after:page}body{background:white}main{max-width:none}@page{size:landscape;margin:0}}</style><header><h1>Gait dynamics research restart</h1><p>Twenty-two illustrated slides connect neurological motivation, full-duration ingestion, controlled forecasting, actual results, and manuscript limits.</p><a href="restart.pdf">Download the vector PDF</a> · <a href="../docs/draft01/README.md">Read submission readiness</a></header><nav><button onclick="move(-1)">Previous</button><button onclick="move(1)">Next</button><a href="restart.pdf">PDF</a></nav><main>']
    for i,s in enumerate(slides,1):html_parts.append(f'<section id="s{i}"><img src="restart_images/slide-{i:02d}.svg" alt="{html.escape(s["title"])}"><details open><summary>Explanation and evidence</summary><p>{html.escape(s["notes"])}</p></details></section>')
    html_parts.append('</main><script>let current=1;function move(d){current=Math.max(1,Math.min(22,current+d));document.getElementById("s"+current).scrollIntoView({behavior:"smooth"});history.replaceState(null,"","#s"+current)}document.addEventListener("keydown",e=>{if(e.key==="ArrowRight")move(1);if(e.key==="ArrowLeft")move(-1)});</script></html>')
    (OUT/'restart.html').write_text(''.join(html_parts))
    if (OUT/'tutorial.html').exists() and not (OUT/'tutorial_before_restart.html').exists():shutil.copyfile(OUT/'tutorial.html',OUT/'tutorial_before_restart.html')
    shutil.copyfile(OUT/'restart.html',OUT/'tutorial.html')
    (OUT/'restart_source.json').write_text(json.dumps(slides,indent=2)+'\n')
    (OUT/'restart_manifest.json').write_text(json.dumps(dict(results_sha256=hashlib.sha256((RUN/'results.json').read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),slides=manifest),indent=2)+'\n')
    fig,axs=plt.subplots(6,4,figsize=(20,17))
    for a in axs.flat:a.axis('off')
    for i,a in enumerate(axs.flat):
        if i<len(slides):a.imshow(plt.imread(ASSETS/f'slide-{i+1:02d}.png'))
    fig.tight_layout();fig.savefig(OUT/'restart_contact_sheet.png',dpi=100);plt.close(fig)
    print('Rendered',len(slides),'vector slides with current evidence.')
if __name__=='__main__':build()
