"""Rebuild the foundation SVG teaching diagrams; revised slides have their own builder."""
from pathlib import Path
from html import escape
import json

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'images'
OUT.mkdir(exist_ok=True)
NAVY, TEAL, BLUE, ORANGE, PALE = '#172e47', '#007f82', '#315cb5', '#b45413', '#f3f6fa'

def text(x,y,content,size=23,color=NAVY,weight='normal',anchor='start'):
    return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" font-family="Arial, sans-serif">{escape(str(content))}</text>'
def lines(x,y,items,size=22,gap=34,**kwargs):
    return ''.join(text(x,y+i*gap,s,size,**kwargs) for i,s in enumerate(items))
def box(x,y,w,h,title,body,color=TEAL):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="white" stroke="{color}" stroke-width="2"/>'
            +text(x+22,y+42,title,25,color,'bold')+lines(x+22,y+85,body,21))
def arrow(x1,y1,x2,y2,color=TEAL):
    return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="3" marker-end="url(#arrow)"/>'
def save(name,title,body,desc):
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="600" viewBox="0 0 1200 600" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L8,3 z" fill="{TEAL}"/></marker></defs><rect width="1200" height="600" fill="{PALE}"/>{text(42,58,title,32,weight='bold')}{body}</svg>'''
    (OUT/f'{name}.svg').write_text(svg)

def flow(name,title,stages,footer):
    n=len(stages); gap=32; width=(1116-gap*(n-1))/n
    body=''
    for i,(heading,items) in enumerate(stages):
        x=42+i*(width+gap)
        body+=box(x,145,width,280,heading,items)
        if i<n-1 and name not in ('vicreg','stochasticity','generalization','evaluation'): body+=arrow(x+width+3,285,x+width+gap-5,285)
    body+=lines(42,492,footer,22)
    save(name,title,body, '; '.join(h+': '+', '.join(items) for h,items in stages))

flow('research_map','A gait video supports two research tasks',[
 ('Observed motion',['Video frames','Annotated person','16 selected joints']),
 ('Learned features',['Normal-only JEPA','Uniform hidden tokens','EMA target features']),
 ('Measured outcomes',['Anomaly ranking','Condition prediction','Source-held-out tests'])],
 ['Anomaly detection uses a frozen normal reference. Condition classification uses labeled training data.',
  'Dataset labels describe this collection. They do not establish a clinical diagnosis.'])
flow('data_pipeline','Fresh annotations become traceable skeleton windows',[
 ('1. Read CSVs',['Five named folders','Hash every CSV','Count unique sources']),
 ('2. Recover motion',['Download by source','Use annotated box','Audit missing poses']),
 ('3. Make windows',['Split sources first','Resample to 15 Hz','Keep cache provenance'])],
 ['Source CSVs remain the authority. Old feature tables and old manifests are never inputs.',
  'A failed download or pose-quality exclusion stays visible in the attrition report.'])
flow('source_split','All windows from one source stay together',[
 ('Training sources',['Fit preprocessing','Train normal JEPA','Fit labeled classifiers']),
 ('Validation sources',['Choose checkpoints','Set anomaly threshold','Select hyperparameters']),
 ('Test sources',['Lock all choices first','Predict once','Bootstrap by source'])],
 ['Split before clipping or windowing. Source identity can otherwise leak into the test set.',
  'Different source IDs can still contain the same person or reposted footage.'])
# Tokenization and masking grid, 8 temporal patches x 16 joints.
body=lines(42,108,['32 frames / 4 frames per patch = 8 time patches. Each patch retains joint identity.'],23)
for t in range(8):
    body+=text(245+t*90,159,str(t),20,anchor='middle')
    for j in range(16):
        hidden=((t*17+j*7)%13)<6
        body+=f'<rect x="{215+t*90}" y="{178+j*19}" width="60" height="14" rx="3" fill="{ORANGE if hidden else TEAL}"/>'
body+=text(42,215,'16 permitted',23)+text(42,249,'joint indices',23)+text(42,284,'only',23)
body+=text(985,239,'Visible',22,TEAL)+text(985,280,'Hidden',22,ORANGE)
body+=lines(42,534,['Uniform samples choose hidden tokens without consulting displacement, speed, or label.'],23)
save('uniform_mask','Mask time and joint tokens uniformly',body,'Grid illustrates random hidden and visible tokens. Orange is hidden. Motion does not determine selection.')
flow('tokenization','A token describes one joint over four frames',[
 ('Input window',['32 frames','16 joints','2 image coordinates']),
 ('Patch embedding',['4 frames × 2 values','One linear projection','Time + joint position']),
 ('Token sequence',['8 time patches','16 joint positions','128 tokens per window'])],
 ['Coordinates are pelvis-centered and body-scaled. Camera geometry remains a possible confound.',
  'Confidence supports quality control. It is not an encoder input.'])
# JEPA two routes; distinguish gradient/EMA with legend.
body=box(42,130,285,158,'Context branch',['Visible tokens only','Online encoder θ'])
body+=box(435,130,295,158,'Predictor',['Context + target queries','Predict hidden features'])+arrow(330,210,428,210)
body+=box(42,350,285,158,'Target branch',['All observed tokens','EMA encoder ξ'])
body+=box(860,240,298,158,'Feature loss',['Predictions vs targets','No target gradients'])
body+=arrow(734,210,810,210)+f'<path d="M810,210 L810,275 L852,275" fill="none" stroke="{TEAL}" stroke-width="3" marker-end="url(#arrow)"/>'
body+=f'<path d="M330,430 L808,430 L808,355 L852,355" fill="none" stroke="{TEAL}" stroke-width="3" marker-end="url(#arrow)"/>'
body+=lines(395,483,['After the optimizer step:', 'ξ ← m ξ + (1 − m) θ'],24)
body+=text(42,563,'The teacher supplies targets during training. It never supplies hidden coordinates to the predictor.',21)
save('jepa_branches','The encoder learns by predicting hidden features',body,'Online visible-only encoder feeds predictor. Frozen teacher receives full input. EMA updates teacher only after optimizer.')
flow('vicreg','VICReg discourages collapse and redundancy',[
 ('Invariance',['Two context views','Same walking window','Encourage agreement']),
 ('Variance',['Batch standard deviation','Discourage constant codes','Watch small batches']),
 ('Covariance',['Off-diagonal penalty','Reduce repeated signals','Track effective rank'])],
 ['L = latent prediction + λ VICReg. VICReg is label-free and does not repel health-condition classes.',
  'A supervised probe learns class boundaries after the encoder has learned features.'])
flow('curriculum','Retain a normal reference during adaptation',[
 ('A. Normal only',['Training normal sources','Save immutable anchor','Measure normal surprise']),
 ('B. Add conditions',['PD, stroke, CP, myopathy','Retain normal replay','Copy before adapting']),
 ('C. Compare probes',['Same source partitions','Same condition labels','Compare normal / final'])],
 ['An adapted encoder may become familiar with abnormal gait. Keep the normal anchor for anomaly scoring.',
  'Progressive order is a hypothesis. Compare with pooled training and an untrained encoder.'])
flow('stochasticity','Three different uses of randomness',[
 ('Uniform masks',['Choose hidden tokens','Changes observed context','Required in main model']),
 ('Position noise',['Perturb token position','StoP-inspired ablation','Evaluate without noise']),
 ('Latent variables',['Model possible futures','Posterior and prior','Requires separate study'])],
 ['Random masking, positional noise, and a stochastic latent variable solve different problems.',
  'The 2024 skeleton S-JEPA and the 2026 stochastic-latent S-JEPA share a name, not an implementation.'])
flow('forecast','Forecasting hides every future token',[
 ('Past context',['Observe first half','No later coordinates','Encode past tokens']),
 ('Future queries',['Future time positions','Joint identities','Predict latent targets']),
 ('Held-out error',['Compare future features','Use one suffix mask','Visualize time and joint'])],
 ['Random in-window completion can use later context. It is not a forecasting result.',
  'Latent residuals explain model mismatch. They do not identify a medical cause.'])
flow('anomaly','Calibrate the normal reference without test labels',[
 ('Fit reference',['Normal training only','Keep anchor weights','Fixed scoring procedure']),
 ('Set threshold',['Normal validation scores','Finite-sample quantile','Record calibration size']),
 ('Evaluate once',['Normal vs abnormal','AUROC and AUPRC','Sensitivity and false alarms'])],
 ['Too few normal calibration sources give an infinite threshold at the requested false-alarm level.',
  'Prediction error can reflect pose failure, camera view, or unfamiliar motion. Review the video.'])
flow('evaluation','Compare methods on the same held-out sequences',[
 ('Classical baseline',['Joint-angle summaries','Motion descriptors','Train-only RF pipeline']),
 ('JEPA features',['Frozen normal encoder','Progressive encoder','Linear supervised probe']),
 ('Fair comparison',['Same windows and labels','Sequence aggregation','Source-cluster intervals'])],
 ['Report macro-F1, balanced accuracy, per-class recall, and binary anomaly metrics.',
  'Train-only feature selection and grouped validation replace optimistic random-window splits.'])
flow('generalization','More frames do not mean more independent people',[
 ('Observed coverage',['666 sequences','140,641 annotation rows','103 source videos']),
 ('Unknown variation',['Participant overlap','Severity and treatment','Camera and source bias']),
 ('Needed evidence',['Independent cohort','Participant IDs','Site and view stress tests'])],
 ['The selected GAVD subset supports exploratory source-held-out evaluation.',
  'Clinical generalization remains unproven without independent participant and site evidence.'])
# Exact joint list split into sides, concise clean anatomy.
body=''; coords={11:(470,165),12:(730,165),13:(420,250),14:(780,250),15:(390,330),16:(810,330),23:(520,330),24:(680,330),25:(520,405),26:(680,405),27:(520,480),28:(680,480),29:(500,505),30:(700,505),31:(550,515),32:(650,515)}
for a,b in [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),(31,27),(28,30),(30,32),(32,28)]:
    x,y=coords[a];u,v=coords[b];body+=f'<line x1="{x}" y1="{y}" x2="{u}" y2="{v}" stroke="{TEAL}" stroke-width="4"/>'
for i,(x,y) in coords.items():body+=f'<circle cx="{x}" cy="{y}" r="7" fill="{BLUE}"/>'
body+=lines(45,150,['11, 12: shoulders','13, 14: elbows','15, 16: wrists','23, 24: hips','25, 26: knees','27, 28: ankles','29, 30: heels','31, 32: foot indices'],23,43)
body+=lines(880,220,['16 unique joints','Sorted numerically','Face excluded','Hand tips excluded'],23,42)
body+=text(42,565,'Feature definitions select the whitelist before masking. Every mask is restricted to these joints.',21)
save('joint_whitelist','Only these 16 BlazePose joints enter the model',body,'Schematic skeleton with selected shoulder, elbow, wrist, hip, knee, ankle, heel and foot index pairs.')

print('Wrote',len(list(OUT.glob('*.svg'))),'foundation SVGs. Use scripts/build_research_slides.py for slides.')
