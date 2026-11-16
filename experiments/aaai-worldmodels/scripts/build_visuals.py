"""Regenerate editable SVG teaching diagrams and a local HTML slide deck."""
from pathlib import Path
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'images'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':12})
NAVY, BLUE, TEAL, ORANGE = '#18324b','#dceafa','#d9f0e8','#fbe7cc'

def canvas(title, subtitle):
    fig, ax = plt.subplots(figsize=(12,5.2))
    fig.patch.set_facecolor('#fafcfe'); ax.set(xlim=(0,12),ylim=(0,5.2)); ax.axis('off')
    ax.text(.35,4.8,title,fontsize=21,weight='bold',color=NAVY)
    ax.text(.35,4.32,subtitle,fontsize=11,color='#526578')
    return fig,ax

def box(ax,x,y,w,h,text,color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=.1',fc=color,ec='#aebdca',lw=.9))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12,color=NAVY,linespacing=1.5)

def arrow(ax,a,b,label=''):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=NAVY,lw=1.5))
    if label: ax.text((a[0]+b[0])/2+(.38 if a[0]==b[0] else 0),(a[1]+b[1])/2+.14,label,ha='center',fontsize=10,color=NAVY)

def save(fig,name):
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',pad_inches=.15)
    fig.savefig(OUT/f'{name}.png',dpi=100,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)

fig,ax=canvas('One source stays in one partition','Split original video IDs before pose windows, model fitting, or classifier selection.')
for y,label,c in [(3.0,'Train sources',BLUE),(1.8,'Validation sources',TEAL),(.6,'Test sources',ORANGE)]:
    box(ax,.4,y,2.5,.8,label,c); box(ax,4.1,y,2.5,.8,'Sequences → windows',c); box(ax,8,y,3.5,.8,{'Train sources':'Fit encoders and classifiers','Validation sources':'Select settings and threshold','Test sources':'Evaluate frozen choices'}[label],c)
    arrow(ax,(2.95,y+.4),(4.05,y+.4)); arrow(ax,(6.65,y+.4),(7.95,y+.4))
save(fig,'source_split')

fig,ax=canvas('Predict a hidden representation','The teacher observes the target; gradients update the student and predictor only.')
box(ax,.35,2.7,2.1,.9,'Visible joint tokens'); box(ax,3.2,2.7,2.1,.9,'Student encoder'); box(ax,6.1,2.7,2.1,.9,'Predictor'); box(ax,9,2.7,2.5,.9,'Masked-token loss',ORANGE)
for a,b in [(2.5,3.15),(5.35,6.05),(8.25,8.95)]:arrow(ax,(a,3.15),(b,3.15))
box(ax,.35,.8,2.1,.9,'Full pose sequence',TEAL); box(ax,3.2,.8,2.1,.9,'EMA target encoder',TEAL); box(ax,6.1,.8,2.1,.9,'Stop-gradient target',TEAL)
arrow(ax,(2.5,1.25),(3.15,1.25)); arrow(ax,(5.35,1.25),(6.05,1.25)); arrow(ax,(8.25,1.25),(10.25,2.65)); arrow(ax,(4.25,2.65),(4.25,1.75),'EMA')
ax.text(.4,.2,'VICReg regularizes student variance and covariance. It does not assign diagnosis clusters.',fontsize=11,color=NAVY)
save(fig,'jepa')

fig,ax=canvas('Uniform masking treats allowed tokens equally','Only the 16 joints in the declared BlazePose table can enter the masking vocabulary.')
import numpy as np
rng=np.random.default_rng(27)
mask=rng.random((5,16))<.5
for t in range(5):
    for j in range(16):
        ax.add_patch(plt.Rectangle((.9+j*.51,3.3-t*.48),.43,.4,fc=ORANGE if mask[t,j] else BLUE,ec='white'))
    ax.text(.65,3.48-t*.48,str(t+1),ha='right',va='center')
for j,idx in enumerate([11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]):ax.text(1.1+j*.51,.95,str(idx),ha='center',fontsize=10)
ax.text(.4,.4,'Columns: original BlazePose indices. Rows: frames. Orange: prediction targets.',fontsize=11)
box(ax,9.5,2.6,1.9,.7,'Visible',BLUE);box(ax,9.5,1.5,1.9,.7,'Masked',ORANGE)
save(fig,'uniform_mask')

fig,ax=canvas('Completion and forecasting answer different questions','A future prediction must be computed from past context alone.')
for row,y in [('Completion',2.8),('Forecast',1.4)]:
    ax.text(.4,y+.3,row,fontsize=14,va='center',weight='bold',color=NAVY)
    for t in range(8):
        hidden=(t in [1,3,6]) if row=='Completion' else t>=4
        box(ax,2.7+t*1.05,y,.85,.65,str(t+1),ORANGE if hidden else BLUE)
ax.text(2.7,.65,'Blue is observed. Orange is withheld. Normalization uses observed context only.',fontsize=11)
save(fig,'forecast')

fig,ax=canvas('Keep a stable definition of normal motion','Copy the normal checkpoint before adding condition-labeled training examples.')
box(ax,.4,2,2.5,1,'Normal train sources');box(ax,4.3,2,2.5,1,'Normal-only checkpoint',TEAL);arrow(ax,(2.95,2.5),(4.25,2.5))
box(ax,8.5,2.9,3,.9,'Frozen anomaly reference',TEAL);box(ax,8.5,1.1,3,.9,'Adapted encoder + probe',ORANGE)
arrow(ax,(6.85,2.5),(8.45,3.35));arrow(ax,(6.85,2.5),(8.45,1.55))
ax.text(.4,.4,'Adaptation stages: parkinsons → stroke → cerebral palsy → myopathic, with normal replay.',fontsize=11)
save(fig,'adaptation')

fig,ax=canvas('Two kinds of stochasticity need separate controls','A factorial experiment separates location robustness from conditional outcome diversity.')
for x,y,txt in [(1.9,2.8,'Neither\nDeterministic reference'),(6.8,2.8,'Positions only\nTemporal position jitter'),(1.9,1.1,'Latents only\nConditional samples'),(6.8,1.1,'Both\nInteraction test')]:box(ax,x,y,3.8,1.1,txt,BLUE if y>2 else TEAL)
ax.text(.4,.35,'Match source folds, masks, seeds, data exposure, and update budget in every cell.',fontsize=11)
save(fig,'stochasticity')

fig,ax=canvas('Compare models on identical observations','A matched cohort separates model differences from differences in available data.')
box(ax,.4,2.1,2.2,1,'Same pose windows');box(ax,3.6,3,2.4,.8,'Gait features');box(ax,3.6,1.2,2.4,.8,'JEPA embeddings',TEAL)
box(ax,7,3,1.8,.8,'Random Forest');box(ax,7,1.2,1.8,.8,'Linear probe',TEAL);box(ax,9.7,2.1,1.9,1,'Paired source\nevaluation',ORANGE)
for a,b in [((2.65,2.6),(3.55,3.4)),((2.65,2.6),(3.55,1.6)),((6.05,3.4),(6.95,3.4)),((6.05,1.6),(6.95,1.6)),((8.85,3.4),(9.65,2.7)),((8.85,1.6),(9.65,2.5))]:arrow(ax,a,b)
ax.text(.4,.4,'Aggregate windows → sequences → sources. Bootstrap source IDs, never individual frames.',fontsize=11)
save(fig,'comparison')

fig,ax=canvas('The source count limits the evidence','666 selected sequences come from 103 source videos; clinical independence is unknown.')
labels=['Normal','Parkinsons','Stroke','Cerebral palsy','Myopathic'];counts=[32,11,19,11,30]
for i,(label,n) in enumerate(zip(labels,counts)):
    y=3.5-i*.62;ax.text(.4,y,label,va='center',fontsize=12);ax.barh(y,n/4,height=.36,left=2.5,color='#4c87b9');ax.text(2.7+n/4,y,str(n),va='center',fontsize=12)
ax.text(.4,.2,'Source IDs are grouping proxies. They are not verified unique patients.',fontsize=11)
save(fig,'cohort')

slides=[
('Learn normal motion, then test unfamiliar gait','A reproducible tutorial study for the Medical World Models workshop. All diagrams describe the protocol, not demonstrated clinical performance.','jepa'),
('Start with the current GAVD inventory','Read all transformed annotations fresh. The requested labels define a focused five-class benchmark; other abnormal gaits remain outside the primary claim.','cohort'),
('Protect the evaluation before making windows','Original source videos stay in one partition. Fit models on training sources, choose settings on validation sources, then report the held-out test.','source_split'),
('Map videos to explicit skeletal observations','Use Ambient extraction and one-based GAVD frames. Keep image dimensions, timestamps, confidence and missingness. Watch the source excerpt before trusting an extracted pose.','uniform_mask'),
('Mask only the declared sixteen joints','Uniform random masks never use velocity or motion magnitude. This is an intentional adaptation of the motion-aware published methods.','uniform_mask'),
('The student learns to predict a teacher target','The target encoder follows an exponential moving average. The predictor reconstructs masked target representations. Gradients never flow through teacher targets.','jepa'),
('Prevent collapse without inventing clinical clusters','Variance and covariance regularization maintain useful variation. Label separation comes from a supervised probe or an explicitly supervised objective.','jepa'),
('A future forecast needs a withheld suffix','Train and evaluate a suffix mask explicitly. A bidirectional masked completion can use later observations and cannot establish forecasting ability.','forecast'),
('Keep the normal reference frozen','Train on normal sources first. Save the reference, copy it, and add each condition in stages with normal replay. The reference remains the anomaly detector.','adaptation'),
('Test stochasticity through controlled experiments','Positional jitter perturbs token locations. A conditional latent variable can represent multiple possible futures. Neither guarantees calibrated uncertainty.','stochasticity'),
('Use the same cohort for RF and JEPA','Classify five labels and the binary endpoint. Use training-only imputation and scaling, equal source weighting, and validation-based choices.','comparison'),
('Report evidence at the source level','Include support, balanced accuracy, per-class recall, AUROC, threshold sensitivity and specificity, paired uncertainty and representative failures.','source_split'),
('Treat GAVD as exploratory medical evidence','Unknown participant overlap, camera variation and web selection limit generalization. A source split cannot establish prospective clinical performance.','cohort'),
('Build the paper from completed evidence','Inspect artifacts and the run ledger before writing results. If empirical evidence is incomplete, the workshop allows a position-paper route. Verify current submission requirements.','comparison'),
]
# Result slides are generated only from completed, released experiments.
if (ROOT/'docs/research_results.json').exists():
    import json
    report=json.loads((ROOT/'docs/research_results.json').read_text())
    primary=report['binary']['paired_against_rf']['jepa_mixed']
    slides.extend([
        ('The primary classifier comparison remains uncertain',f"Mixed JEPA balanced accuracy is 0.818 versus RF 0.788 on 17 sources. The paired difference is {primary['estimate']:.3f}, with a 95% interval from {primary['ci95'][0]:.3f} to {primary['ci95'][1]:.3f}. Staged JEPA is secondary.",'binary_revision'),
        ('Persistence remains a demanding forecast reference','Three training seeds, six normal test sources. The residual mean gain is 0.000513 with an interval crossing zero. Replacing ordered context with a static prefix improves residual error.','forecast_revision'),
        ('Anomaly ranking and forecast skill separate','All nine prespecified scores are shown on 19 sources. The direct model has poor forecast skill but a strong excess-error ranking. This supports a controlled case study, with no claim that the best score proves useful dynamics.','anomaly_revision'),
        ('Distribution quality and uncertainty ranking separate','Latent samples improve the energy score across all three seeds. Their error-deferral ranking is worse than prefix confidence on the six normal sources. Neither result establishes clinical calibration.','uncertainty_tradeoff'),
    ])
    slides.extend([('Read the endpoints separately','Binary annotation screening and five-condition recognition have different targets and decision rules. These are point estimates; paired intervals and source support are in the report.','__result_table__'),('Keep the two study cohorts distinct','Every source stays in its original split. Short-window person gating and longer forecast support lead to different retained test sets. Windows are not independent patients.','__cohort_table__')])

style='''body{margin:0;background:#112538;font-family:system-ui,sans-serif;color:#18324b}section{box-sizing:border-box;min-height:100vh;max-width:1400px;margin:auto;padding:4vh 5vw;background:#fafcfe;page-break-after:always}h1{font-size:clamp(26px,3vw,44px);line-height:1.12;margin:.3em 0}p{font-size:clamp(17px,1.6vw,23px);line-height:1.5;max-width:1120px}img{width:100%;max-height:65vh;object-fit:contain}footer{font-size:14px;color:#526578}a{color:#246a98}table{border-collapse:collapse;width:100%;font-size:clamp(16px,1.6vw,24px);margin:5vh 0}td,th{text-align:left;padding:.8em .5em;border-bottom:1px solid #b9c8d6}th{background:#dceafa}@media print{section{height:190mm;min-height:0}img{max-height:130mm}}'''
body=[]
for i,(title,desc,fig) in enumerate(slides,1):
    visual=f'<img src="../images/{fig}.svg" alt="{html.escape(title)}">'
    if fig=='__result_table__':
        visual='<table><tr><th>Model</th><th>Binary BA</th><th>Binary AUROC</th><th>Five-class BA</th></tr>'
        for key,label in [('rf','Random Forest'),('untrained_probe','Untrained probe'),('jepa_normal','Normal JEPA'),('jepa_adapted','Staged JEPA (secondary)'),('jepa_mixed','Mixed JEPA (primary)')]:
            b=report['binary']['test_reports'][key];f=report['five_class_baseline']['classifiers'][key]
            visual+=f"<tr><td>{label}</td><td>{b['binary_balanced_accuracy']:.3f}</td><td>{b['abnormal_auroc']:.3f}</td><td>{f['balanced_accuracy']:.3f}</td></tr>"
        visual+='</table>'
    elif fig=='__cohort_table__':
        visual='<table><tr><th>Observation contract</th><th>Sequences</th><th>Train sources</th><th>Validation sources</th><th>Test sources</th></tr><tr><td>16-frame classifier</td><td>405</td><td>53</td><td>19</td><td>17</td></tr><tr><td>32-frame forecast</td><td>407</td><td>53</td><td>19</td><td>19</td></tr></table>'
    body.append(f'<section id="slide-{i}"><footer>MED-WM GAIT TUTORIALS · {i:02d} / {len(slides)}</footer><h1>{html.escape(title)}</h1><p>{html.escape(desc)}</p>{visual}<footer>Evidence and references: <a href="../docs/literature.md">literature review</a> · <a href="../docs/protocol.md">study protocol</a> · <a href="../docs/results_tables.md">completed result tables</a></footer></section>')
(ROOT/'slides/tutorial.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Medical world models for gait</title><style>'+style+'</style>'+''.join(body)+'</html>')
(ROOT/'slides/README.md').write_text(f"# Teaching and results slides\n\nOpen `tutorial.html` in a browser. Its {len(slides)} pages pair teaching steps and completed results with editable SVG diagrams. Print to PDF with landscape pages if desired. Source: `../scripts/build_visuals.py`. Run `scripts/analyze_research.py` and `scripts/build_results_supplement.py` after the joint test release to regenerate the result figures, then rebuild this deck. The first fourteen pages teach the method; the final six summarize released evidence and endpoint tables. See `../docs/results_tables.md` for all secondary scores and uncertainty.\n")
print(f'Generated eight teaching SVGs and {len(slides)} HTML slides.')
