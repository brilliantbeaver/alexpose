"""Build the proposed second-iteration deck and shared, editable vector diagrams.

All empirical values are read from completed-run or read-only availability artifacts.
This script does not fit models, alter scientific results, or change data selection.
"""
from pathlib import Path
import hashlib, json, math, subprocess, re
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlsplit
from html import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw

BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parents[1];OUT=BASE/'slides';IMG=OUT/'images';SHARED=BASE/'images/iteration02'
W,H=1280,720
INK='#172e47';TEAL='#087f82';BLUE='#315cb5';ORANGE='#ad521b';MUTED='#506478';PALE='#f2f6fa';LINE='#d9e3ec';WHITE='#ffffff'
plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.family':'DejaVu Sans','savefig.facecolor':'white'})
FONT=FontProperties(family='DejaVu Sans');BOLD=FontProperties(family='DejaVu Sans',weight='bold')
SOURCES={
 'instructions':('Canonical next-iteration instructions','../../../notes/aaai/05-iterate-02-INSTR.md'),
 'brief':('Working design and scientific scope','../docs/iteration02/design_brief.md'),
 'method':('Methodological advice','../docs/iteration02/advice_method.md'),
 'neuro':('Neurological evidence and novelty advice','../docs/iteration02/advice_neuro.md'),
 'data':('Read-only data and ingestion review','../docs/iteration02/advice_data.md'),
 'coverage':('Read-only availability calculations','../docs/iteration02/data_feasibility_probe.json'),
 'results':('Completed draft01 results','../docs/context_study_results.md'),
 'sensitivity':('Completed sensitivity results','../docs/context_study_sensitivity.md'),
 'paper':('Completed draft01 manuscript','../docs/draft01/manuscript.pdf'),
 'audit':('Completed execution validation','../docs/current_execution_validation.md'),
 'baseline':('Published-baseline feasibility audit','../docs/published_baseline_feasibility.md'),
 'workshop':('ASTAD4 official requirements','https://ismart.ece.mcgill.ca/astad4/'),
 'pci':('Original bilateral phase coordination study','https://pubmed.ncbi.nlm.nih.gov/17503027/'),
 'qpci':('Four-limb coordination in healthy adults','https://pmc.ncbi.nlm.nih.gov/articles/PMC11435969/'),
 'stroke':('Stroke coordination and temporal symmetry','https://pubmed.ncbi.nlm.nih.gov/38709654/'),
 'pd':('PD arm amplitude, symmetry and coordination','https://pubmed.ncbi.nlm.nih.gov/25502948/'),
 'cp':('Selective motor control and gait in CP','https://pubmed.ncbi.nlm.nih.gov/27889619/'),
 'dmd':('Longitudinal Duchenne muscle and gait observations','https://link.springer.com/article/10.1186/s12984-025-01718-5'),
 'gengait':('GenGait author manuscript','https://arxiv.org/abs/2604.01997v2'),
 'fsgait':('FSGait','https://openaccess.thecvf.com/content/ACCV2024/html/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.html'),
 'seeker':('SeeKer','https://arxiv.org/abs/2506.18368'),
 'marginals':('Beyond Marginals','https://arxiv.org/abs/2509.15033'),
 'lti':('Intersegmental LTI asymmetry metric','https://arxiv.org/abs/2505.10869'),
 'selection':('Confidence filtering and selection shift','https://arxiv.org/abs/2610.39704'),
 'lightweight':('Typical-cycle representation and lightweight ST-GCN','https://homepages.inf.ed.ac.uk/rbf/PAPERS/lochheadcbm24.pdf'),
 'external':('Bounded independent replication review','../docs/iteration02/external_replication_feasibility.md'),
 'mask_method':('Controlled masking methods review','../docs/iteration02/masking_method_review.md'),
 'mask_neuro':('Masking and neurological interpretation','../docs/iteration02/masking_neuro_review.md'),
 'mamp':('MAMP primary paper','https://arxiv.org/abs/2308.07092'),
 'sjepa':('Skeleton S-JEPA primary paper','https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf'),
 'slim':('SLiM current author manuscript','https://arxiv.org/html/2603.10648v4'),
 'asma':('ASMa author publication record','https://23wm13.github.io/ASMa/'),
 'external_paper':('Laboratory control and stroke dataset','https://pmc.ncbi.nlm.nih.gov/articles/PMC10692332/'),
}

def width(text,size=24,bold=False):
 if not text:return 0
 return TextPath((0,0),str(text),size=size*.72,prop=BOLD if bold else FONT).get_extents().width/.72

def wrap(text,max_width,size=24,bold=False):
 out=[];line=''
 for word in str(text).split():
  trial=(line+' '+word).strip()
  if line and width(trial,size,bold)>max_width:out.append(line);line=word
  else:line=trial
 if line:out.append(line)
 return out

class Canvas:
 def __init__(self,w=W,h=H):
  self.w,self.h=w,h;self.fig=plt.figure(figsize=(w/100,h/100));self.ax=self.fig.add_axes([0,0,1,1]);self.ax.set(xlim=(0,w),ylim=(0,h));self.ax.axis('off');self.texts=[]
 def text(self,x,y,t,size=24,color=INK,bold=False,align='left'):
  obj=self.ax.text(x,self.h-y,str(t),fontsize=size*.72,color=color,fontweight='bold' if bold else 'normal',ha=align,va='baseline');self.texts.append(obj);return obj
 def para(self,x,y,t,w,size=24,color=INK,bold=False,leading=1.32):
  lines=wrap(t,w,size,bold)
  for i,line in enumerate(lines):self.text(x,y+i*size*leading,line,size,color,bold)
  return y+max(0,len(lines)-1)*size*leading
 def box(self,x,y,w,h,fill=WHITE,stroke=LINE,radius=12):
  self.ax.add_patch(FancyBboxPatch((x,self.h-y-h),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',facecolor=fill,edgecolor=stroke,linewidth=1.1))
 def line(self,x1,y1,x2,y2,color=TEAL,lw=2,dash=None):
  self.ax.plot([x1,x2],[self.h-y1,self.h-y2],color=color,lw=lw,ls=dash or '-',solid_capstyle='round')
 def arrow(self,x1,y1,x2,y2,color=TEAL):
  self.ax.add_patch(FancyArrowPatch((x1,self.h-y1),(x2,self.h-y2),arrowstyle='-|>',mutation_scale=13,linewidth=1.5,color=color))
 def card(self,x,y,w,h,title,body,color=TEAL,size=24):
  self.box(x,y,w,h);bottom=self.para(x+22,y+35,title,w-44,24,color,True,1.22);end=self.para(x+22,bottom+39,body,w-44,size)
  if end>y+h-17:raise ValueError(f'Card text overflows: {title}: {end} > {y+h-17}')
 def banner(self,y,t,color=TEAL,size=25):
  self.box(60,y,1160,66,'#e7f1f1','#b6d9da',9);lines=wrap(t,1110,size,True)
  if len(lines)>1:raise ValueError('Banner too long '+t)
  self.text(640,y+42,t,size,color,True,'center')
 def table(self,headers,rows,widths,y=214,row_h=58,size=23):
  self.box(60,y,1160,row_h,INK,INK,6);x=60
  for a,w in zip(headers,widths):self.para(x+16,y+34,a,w-30,size,WHITE,True,1.15);x+=w
  for i,row in enumerate(rows):
   yy=y+(i+1)*row_h;self.box(60,yy,1160,row_h,WHITE if i%2==0 else PALE,WHITE,0);x=60
   for a,w in zip(row,widths):
    lines=wrap(a,w-30,size)
    if len(lines)>1 and 25+(len(lines)-1)*size*1.13>row_h-5:raise ValueError(f'Table cell overflows: {a}')
    self.para(x+16,yy+(34 if len(lines)==1 else 25),a,w-30,size,leading=1.13);x+=w
 def save(self,path,png=True):
  self.fig.canvas.draw();renderer=self.fig.canvas.get_renderer();bad=[]
  for t in self.texts:
   bb=t.get_window_extent(renderer)
   if bb.x0<-.5 or bb.y0<-.5 or bb.x1>self.w+.5 or bb.y1>self.h+.5:bad.append(t.get_text())
  if bad:raise ValueError('Text outside canvas: '+str(bad))
  for ext in ['svg','pdf']+(['png'] if png else []):self.fig.savefig(path.with_suffix('.'+ext),dpi=130)
  plt.close(self.fig)

def frame(title,lede,tag,number,total=40,takeaway=''):
 c=Canvas();c.box(0,0,18,H,TEAL,TEAL,0);c.text(60,38,tag,15,TEAL,True)
 title_size=min(34,34*1155/max(1155,width(title,34,True)))
 if title_size<29:raise ValueError('Shorten title: '+title)
 c.text(60,87,title,title_size,INK,True);end=c.para(60,142,lede,1150,24,MUTED)
 if end>181:raise ValueError('Shorten introduction: '+lede)
 if takeaway:c.box(45,638,1190,47,PALE,PALE,8);c.text(60,668,takeaway,min(19,19*1150/max(1150,width(takeaway,19))),INK)
 c.text(60,704,'Gait prediction | Proposed second iteration | November 2026',15,MUTED);c.text(1220,704,f'{number:02d} / {total}',16,MUTED,align='right');return c

def rationale(c,top=215):
 cols=[('Movement size and posture','A small arm swing or a flexed knee can be regular and predictable.',TEAL),('Relative timing','Ordinary excursions can occur at an unusual time relative to the other leg.',BLUE),('Stable compensation','Trunk and leg changes can help a person keep walking in a repeated pattern.',ORANGE)]
 for i,(t,b,col) in enumerate(cols):c.card(60+i*400,top,360,220,t,b,col,25)
 c.banner(top+266,'Preserve local movement, then test what cross-leg context adds.',size=25)

def residual(c,top=210):
 c.card(60,top+5,250,108,'O: own leg','Visible points.',TEAL,22)
 c.card(60,top+195,250,108,'C: other leg','Visible points.',BLUE,22)
 c.card(465,top+5,300,108,'Frozen f0(O)','Keep its output.',TEAL,22)
 c.card(465,top+180,300,148,'Learn g(O, C)','Start at zero. Fit on normal sources.',BLUE,22)
 c.arrow(315,top+59,459,top+59);c.line(315,top+92,395,top+92);c.line(395,top+92,395,top+235);c.arrow(395,top+235,459,top+235)
 c.arrow(315,top+251,459,top+269,BLUE)
 c.arrow(770,top+59,929,top+157);c.arrow(770,top+245,929,top+177,BLUE)
 c.ax.add_patch(Circle((951,c.h-(top+165)),23,facecolor=WHITE,edgecolor=TEAL,linewidth=1.6));c.text(951,top+173,'+',27,TEAL,True,'center')
 c.card(1020,top+110,200,112,'Prediction','f0 + g',TEAL,20);c.arrow(978,top+165,1014,top+165)
 c.banner(top+350,'Zero correction preserves the baseline. It does not guarantee test improvement.',size=23)

def timing(c,top=210):
 c.text(60,top+5,'Shift distal-joint offsets; keep the target and own-leg observations fixed.',23,INK,True)
 x0=430;cell=55;row_y=top+54
 for i,(label,start,color) in enumerate([('Earlier context: -4 frames',1,BLUE),('Aligned context: 0 frames',2,TEAL),('Later context: +4 frames',3,ORANGE)]):
  y=row_y+i*84;c.text(60,y+30,label,23,color,True)
  for j in range(12):
   active=start<=j<start+8;c.box(x0+j*cell,y,cell-5,46,color if active else PALE,color if active else LINE,4)
   if j==start:c.text(x0+j*cell+7,y+29,'start',16,WHITE,True)
  c.text(1120,y+30,'32 frames',20,MUTED)
 c.text(430,row_y+254,'One 48-frame parent segment; each tile spans four frames.',22,MUTED)
 c.banner(top+350,'Fixed visible hip anchors, paired support and measured donor drift.',size=23)

def gates(c,top=210):
 items=[('1. Observe','Count support, inspect actual video and preserve missingness.'),('2. Learn','Recover known dependencies and improve normal development.'),('3. Isolate','Keep targets and masks fixed; compare timing and noise controls.'),('4. Evaluate','Use frozen source roles and strong matched comparators.')]
 for i,(t,b) in enumerate(items):
  x=60+i*300;c.card(x,top+15,260,242,t,b,TEAL if i<2 else BLUE,23)
  if i<3:c.arrow(x+265,top+136,x+294,top+136)
 c.banner(top+310,'A failed gate narrows the claim; it does not trigger a search for a better score.',size=22)

DIAGRAMS={'rationale':('Three observable patterns require different explanations.',rationale),'residual_predictor':('Keep a competent local prediction when adding context.',residual),'timing_controls':('Change the timing of the other-leg distal context.',timing),'execution_gates':('Require evidence before strengthening the claim.',gates)}


def mask_families(c,top=210):
 import random
 rows=[11,12,23,24,13,15,14,16,25,27,29,31,26,28,30,32]
 rng=random.Random(20261115)
 for k,(name,desc) in enumerate([('U: uniform','Draw 64 tokens.'),('J: joint tube','One joint; all eight patches.'),('A: anatomical tube','Example: 4 joints, 2 patches.')]):
  x=60+400*k;c.box(x,top,360,334);c.text(x+22,top+34,name,25,TEAL if k else BLUE,True);c.text(x+22,top+67,desc,20,MUTED)
  core=set() if k==0 else ({(8,t) for t in range(8)} if k==1 else {(j,t) for j in range(8,12) for t in [2,3]})
  pool=[(j,t) for j in range(16) for t in range(8) if (j,t) not in core];fill=set(rng.sample(pool,64-len(core)));assert len(core|fill)==64
  gx=x+100;gy=top+100
  for t in range(8):c.text(gx+t*28+12,gy-9,str(t),13,MUTED,align='center')
  for j,idx in enumerate(rows):
   c.text(gx-12,gy+j*14+10,str(idx),12,MUTED,align='right')
   for t in range(8):
    col=TEAL if (j,t) in core else BLUE if (j,t) in fill else PALE;c.box(gx+t*28,gy+j*14,25,11,col,col,1)
 c.text(60,top+369,'Structured core',20,TEAL,True);c.text(260,top+369,'Uniform fill',20,BLUE,True);c.text(420,top+369,'Pale cells stay visible before validity checks.',20,MUTED)
 c.text(60,top+402,'Illustrative draws. Rows group joint indices by anatomy; columns are four-frame patches.',18,MUTED)

def mask_comparison(c,top=210):
 c.table(['Training recipe','Frozen E_U: primary','Frozen E_leg: diagnostic'],[
  ('U','Same uniform target bank','Same distal-core target bank'),('J','Same uniform target bank','Same distal-core target bank'),('A','Same uniform target bank','Same distal-core target bank')],[280,440,440],y=top,row_h=56,size=22)
 c.card(60,top+256,560,134,'E_U retains the reference task.','Use 16 uniform draws with seed 17.',TEAL,23)
 c.card(652,top+256,568,134,'E_leg tests local geometry.','Balance two sides and four starts, twice.',BLUE,23)

DIAGRAMS.update({'mask_families':('Change mask structure while retaining the token budget.',mask_families),'mask_comparison':('Compare training recipes within one frozen evaluation task.',mask_comparison)})

def add_mask_slides():
 d=Deck()
 d.add('Structured masking is established. Test its added value.','The literature supports alternatives, but no cited action-recognition result establishes the best mask for pathological gait.',
  lambda c:cards(c,[('Published defaults','MAMP and skeleton S-JEPA use motion-aware masking. Our uniform variant is a declared adaptation.'),('Close structural precedent','SLiM masks anatomical groups across time to reduce easy local interpolation.'),('Our proposed comparison','Match total mask count and token probabilities. Test useful added-leg prediction on a fixed task.')],size=23),
  tag='MASKING PRIOR ART AND PROPOSED TEST',notes='The original S-JEPA uses a 0.9 motion-aware mask on resized 120-frame action sequences. Our 32-frame, 16-joint projected gait input does not justify copying that ratio. The current SLiM author record identifies NeurIPS 2026 and its semantic tube strategy is close prior art. ASMa uses motion and topology and is reported as TMLR 2026 by its authors. U, J and A are controlled motion-independent adaptations, not reproductions of those complete systems. Do not claim a new anatomical tube or state-of-the-art gait accuracy.',refs=('mask_method','mask_neuro','mamp','sjepa','slim','asma'),takeaway='A useful contribution needs a matched experiment and a supported result, not a new mask name.')
 d.add('Every recipe hides exactly 64 of the same 128 tokens.','U samples freely. J and A first hide an eight-token core, then uniformly sample 56 of the remaining 120 tokens.',
  lambda c:mask_families(c,top=210),tag='ILLUSTRATIVE MASKS, NOT MODEL RESULTS',notes='A 32-sample window contains eight four-frame temporal patches and the same 16 allowed joints. U samples 64 distinct token positions uniformly. J selects one joint uniformly and masks all eight temporal positions. A selects an anatomical group with probability equal to its width divided by 16, then an aligned nonwrapping temporal block of length eight divided by its width. Both structured families fill 56 further positions uniformly without replacement. The pictured A draw is one distal-leg example; A also samples the torso and arm groups. Mask generation cannot read motion, confidence, labels or loss. The graphic groups row labels by anatomy for readability and does not redefine the index mapping.',refs=('mask_method','instructions'),takeaway='The pictures show declared sampling rules, not measured model behavior.')
 def anatomical(c):
  c.table(['Anatomical group','BlazePose indices','Width','Time patches','Probability'],[
   ('Torso anchors','11, 12, 23, 24','4','2','4/16'),('Left elbow and wrist','13, 15','2','4','2/16'),('Right elbow and wrist','14, 16','2','4','2/16'),('Left distal leg','25, 27, 29, 31','4','2','4/16'),('Right distal leg','26, 28, 30, 32','4','2','4/16')],[305,310,115,230,200],y=211,row_h=63,size=22)
 d.add('Match how often each token is hidden.','For two-patch cores, starts are 0, 2, 4 or 6. For four-patch cores, starts are 0 or 4. No interval wraps.',anatomical,
  notes='The five groups are disjoint and cover exactly the 16 allowed joints. For a group of width w, choose it with probability w/16 and choose one of the aligned blocks of length h=8/w uniformly. Every token therefore enters the core with probability (w/16)*(h/8)=1/16. After uniform fill, its hidden probability is 1/16+(15/16)*(56/120)=1/2. J has the same core and final marginal probabilities. This matches first-order token probabilities and exact budgets, not higher-order dependence, valid support or task difficulty. Torso or filler masks can hide hip tokens. All branches remain conditional on the explicitly shared body reference Q; no additional hidden hip may enter a derived feature.',refs=('mask_method','instructions'),takeaway='For every token: P(core) = 1/16; P(hidden) = 1/16 + (15/16)(56/120) = 1/2.')
 d.add('Evaluate every training arm on the same frozen masks.','Compare U, J and A within an evaluation column. Native-mask versus native-mask scores answer different questions.',
  lambda c:mask_comparison(c,top=215),tag='ILLUSTRATION OF THE PROPOSED COMPARISON',notes='The primary bank E_U is the unchanged 16-draw uniform bank with seed 17. A versus U is the primary masking-policy contrast on this fixed bank. A versus J tests anatomical co-masking beyond a joint tube; J versus U is secondary. These contrasts are distinct from the primary S-versus-M detection comparison. E_leg contains two sides times four aligned starts times two repetitions, giving 16 masks. Each E_leg draw forces the four distal joints of that side over two temporal patches, then uniformly fills 56 additional tokens using dedicated seed 117. Save the exact banks and their hashes. Score valid core targets only in E_leg. Match target keys, loss weights and support across training arms within each bank. E_leg emphasizes lower-limb cores by design and is a separate diagnostic, not a marginally matched replacement for E_U. Fit separate reserved-normal relational references for each bank using the same five reserved source identities. If a required reference lacks support, report the diagnostic unavailable; never substitute a test source. Disclose diagnostic coverage and do not form one intersection across all mask banks and timing controls.',refs=('mask_method','instructions'),takeaway='Keep the evaluation question fixed when attributing a difference to the training mask.')
 d.add('Separate training factors and their random streams.','A masking comparison must not silently change initialization, sample order, regularization or the second training view.',
  lambda c:cards(c,[('Coordinate corrections first','Freeze one U-trained local baseline. Vary only the correction-training masks.'),('JEPA as a separate factor','Vary encoder pretraining masks. Keep downstream coordinate-head masks fixed to U.'),('Reproducible paired views','Separate mask RNG from all other randomness. Use the assigned recipe for both VICReg masks.')],size=23),
  notes='Compare correction-training masks with one common retained U-trained baseline f. Do not refit f under every mask recipe and then call it the same isolated comparison. If JEPA is evaluated, vary encoder pretraining masks separately while keeping head training on U, with the random-encoder control retained. Use dedicated random generators for masking, initialization, source sampling and augmentations. Derive the second VICReg mask from the assigned U, J or A recipe using its own declared stream; do not leave an implicit uniform second mask in structured arms. Match updates and nominal target counts, and report effective valid-target exposures. Any operational selection uses grouped fitting-normal validation, not calibration normals or disease AUROC.',refs=('mask_method','instructions'),takeaway='Independent bookkeeping is necessary before a performance difference can be attributed to masks.')
 def mask_gates(c):
  c.table(['Gate','Required check'],[
   ('Mask construction','Exactly 64 tokens; correct groups, starts and one-half marginals.'),('Information access','Mask before derived features and attention; keep the shared body reference fixed.'),('Normal prediction','Compare added-leg, local and extra-capacity predictions on identical support.'),('Missing measurements','Apply validity after drawing; report unsupported targets without redraws.')],[355,805],y=216,row_h=83,size=23)
 d.add('Check support and learning before interpreting a mask.','Report support and normal prediction competence before interpreting a condition-associated score.',mask_gates,
  notes='Run known-dependence, conditional-independence, zero-gate and hidden-input invariance fixtures for each arm. Invariance is conditional on the declared shared reference Q, not proof of independence from every body measurement used to construct Q. A J tube removes all history of its selected joint; an old eligibility rule that requires visible history of that very joint would exclude precisely those targets. Use a declared own-leg context rule for training and diagnostics while preserving the frozen primary U population. Audit available own-leg and opposite-leg tokens, repeated target keys and source weighting. Keep failed arms in the report. A failed normal relational gate sets R to zero within the fold; it does not authorize changing the endpoint or masking recipe after evaluation.',refs=('mask_method','mask_neuro','instructions'),takeaway='Higher reconstruction error does not establish better representations or a neurological mechanism.')
 return d.records

def build_diagrams():
 SHARED.mkdir(parents=True,exist_ok=True)
 for name,(title,func) in DIAGRAMS.items():
  c=Canvas(1280,600);c.text(60,34,'PROPOSED STUDY | ILLUSTRATION, NOT A MEASURED RESULT',16,TEAL,True);c.text(60,83,title,31,INK,True);func(c,top=130)
  c.text(60,582,'The figures describe planned comparisons. They do not identify neurological causes.',17,MUTED);c.save(SHARED/name)

class Deck:
 def __init__(self):self.records=[]
 def add(self,title,lede,draw,tag='PROPOSED NEXT ITERATION',notes='',refs=('instructions','brief'),takeaway=''):
  self.records.append(dict(title=title,lede=lede,draw=draw,tag=tag,notes=notes,refs=list(refs),takeaway=takeaway))

def cards(c,items,y=220,h=280,size=24):
 n=len(items);gap=32;w=(1160-(n-1)*gap)/n
 for i,(title,body) in enumerate(items):c.card(60+i*(w+gap),y,w,h,title,body,TEAL if i%2==0 else BLUE,size)

def load():
 p=BASE/'docs/draft01/evidence.json';q=BASE/'docs/iteration02/data_feasibility_probe.json';e=json.loads(p.read_text());a=json.loads(q.read_text())
 assert e['primary_run']=='3044566091261819';assert a['status'].startswith('Read-only')
 for run in [e['primary_run'],e['sensitivity_run']]:assert json.loads((BASE/'artifacts/context_study/gavd'/run/'verification.json').read_text())['passed']
 return e,a,{'files':[{'path':str(x.relative_to(BASE)),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in [p,q]],'status':'Completed results and read-only availability only; new method not executed'}

def build_deck(e,a):
 d=Deck();metric={r['Method']:r for r in e['budget_comparison']};cov=a['lower_limb_candidate'];labels={r['label']:r['sources'] for r in cov['by_label']};ci=e['primary_difference']['interval_95'];old=e['cohort'];focal=sum(labels[k] for k in ['normal','parkinsons','stroke','cerebral palsy','myopathic'])
 def m(name,col='Primary512'):return metric[name][col]
 d.add('Can the other leg improve a reliable local prediction?',
  'The next study asks whether projected cross-leg relationships add useful information beyond movement size, posture and local geometry.',
  lambda c:cards(c,[('Learn the local movement','Predict a hidden joint patch from observations of its own leg.'),('Add the other leg','Retain the local prediction and learn only a correction from extra context.'),('Challenge the explanation','Test timing, missingness, capacity and source identity before making a claim.')]),
  notes='This is a proposed second iteration. It has not been implemented or evaluated. Neurological reasoning motivates observable comparisons; projected video does not identify a neural lesion or establish a clinical diagnosis.',takeaway='The scientific opportunity is an auditable test of added relational information.')
 def observed(c):
  rows=[('Latent benefit profile',f"{m('Latent benefit profile'):.3f}"),('Raw full latent error',f"{m('Raw full latent error'):.3f}"),('Calibrated full-error profile',f"{m('Full-error reference distance'):.3f}")]
  c.table(['Completed method','Source AUROC'],rows,[900,260],y=220,row_h=66,size=25)
  c.banner(530,f"Profile minus raw error: {e['primary_difference']['point_difference']:+.3f}; 95% interval [{ci[0]:+.3f}, {ci[1]:+.3f}].",size=24)
 d.add('The completed result does not establish an improvement.','The primary completed run includes 251 sources, of which only 23 have a normal label.',observed,'MEASURED DRAFT01 RESULT',
  notes='These are completed source-level results from run 3044566091261819, not outcomes of the proposed method. The paired source bootstrap interval conditions on the trained models, reference sets and fixed folds. It crosses zero. Calibrated full-error distance also exceeds the latent benefit profile point estimate. The AUROC is computed over all non-normal folders in the historical endpoint.',refs=('results','paper'),takeaway='A larger point estimate alone does not establish added value.')
 def random_control(c):
  c.table(['Common coordinate target','512 head updates','4,096 head updates'],[
   ('Learned encoder',f"{m('Coordinate benefit profile'):.3f}",f"{m('Coordinate benefit profile','Sensitivity4096'):.3f}"),
   ('Random encoder',f"{m('Random encoder, coordinate heads'):.3f}",f"{m('Random encoder, coordinate heads','Sensitivity4096'):.3f}")],[580,290,290],y=215,row_h=68,size=24)
  bad={u:sum(r['full_coordinate']>r['own_coordinate'] for r in e['normal_validation'] if r['head_updates']==u) for u in ['512','4096']}
  c.card(60,460,560,140,'Primary development failure',f"Full coordinate error is worse than own-only in {bad['512']}/4 folds.",ORANGE,24)
  c.card(652,460,568,140,'Longer training does not resolve it',f"The same failure remains in {bad['4096']}/4 folds.",ORANGE,24)
 d.add('A random encoder matches or exceeds the learned one.','A common coordinate target makes the encoder comparison more interpretable than comparing separate latent target spaces.',random_control,'MEASURED DRAFT01 RESULT',notes='The longer-budget sensitivity changes predictor training, not the source-level sample size. The learned contextual predictor has not yet shown reliable predictive competence on normal development data. A poor context model can create an apparent abnormality score without measuring a meaningful biological relation.',refs=('results','sensitivity','method'),takeaway='First establish that added context can help a competent predictor.')
 d.add('Keep three evidence statuses separate.','Completed efficacy, read-only feasibility and a research proposal answer different questions.',
  lambda c:cards(c,[('Completed measurements','The draft01 models and notebooks ran. Their results remain the historical record.'),('Read-only availability','New support counts inspect existing poses and annotations. No model was trained.'),('Proposed next experiments','Notebooks 12 through 17, residual predictors and timing controls are still planned.')]),
  tag='EVIDENCE AND PROVENANCE',notes='The new deck is a planning document. It must not be cited as a new accuracy result or a completed validation. The exact prior 40-page deck is preserved under archive/draft01_completed.',refs=('audit','coverage','instructions'),takeaway='Availability is not performance, and a proposal is not a result.')
 def neuro(c):
  c.table(['Family','Plausible origin','Observable candidate'],[
   ('Stroke','Injury to motor pathways','Unequal stance or repeated compensation'),
   ('Parkinsons','Basal-ganglia dysfunction','Reduced excursion and altered coordination'),
   ('Cerebral palsy','Early motor-system injury','Selective-control limits and flexed posture'),
   ('Myopathic','Muscle disease or weakness','Proximal weakness and compensatory motion')],[235,405,520],y=211,row_h=77,size=23)
 d.add('Different origins can produce overlapping visible adaptations.','A neurological or muscle mechanism can motivate a measurement without making that measurement a diagnosis.',neuro,
  tag='NEUROLOGICAL RATIONALE',notes='Stroke can alter selective activation, strength and balance. Parkinson disease can affect automatic movement and movement amplitude. Cerebral palsy involves heterogeneous early motor-system injury and secondary musculoskeletal changes. Myopathic is a broad muscle-disease label, not one neurological lesion. The Duchenne study is a mechanistic example, not evidence that every GAVD myopathic sequence has Duchenne muscular dystrophy. Camera view, speed, aids and individual compensation can create overlapping projected signatures.',refs=('neuro','stroke','pd','cp','dmd'),takeaway='Treat these as label-associated movement hypotheses with explicit counterexamples.')
 d.add('A regular movement can still differ from typical gait.','Small movement, unusual timing and stable compensation require different tests.',lambda c:rationale(c,top=214),
  tag='ILLUSTRATION OF THE PROPOSED STUDY',notes='Example one: a person with small but regular arm swing may have a local amplitude difference while cross-leg prediction remains accurate. Example two: ordinary leg excursion may occur at an unusual relative time. Example three: repeated trunk compensation can make movement more predictable despite a marked impairment. None of these signs uniquely identifies a condition.',refs=('neuro','pd','stroke'),takeaway='A relational score complements local movement; it cannot replace it.')
 d.add('Normal-only masked gait modeling is already established.','Using a world-model label or adding another mask is not enough to distinguish the paper.',
  lambda c:cards(c,[('FSGait','Prior work separates spatial and temporal gait abnormalities using self-supervised modeling.'),('GenGait','Prior work uses normal-gait generative modeling and masked reconstruction for abnormality detection.'),('SeeKer','Prior work estimates keypoint confidence through masked conditional prediction.')],size=23),
  tag='NOVELTY BOUNDARY',notes='Read and cite the primary papers. The proposed study must outperform and explain stronger matched baselines before attributing value to the encoder. GenGait has an available author manuscript and publisher record; publication timing does not establish its applicability or our performance against it.',refs=('fsgait','gengait','seeker','baseline'),takeaway='The contribution must come from the tested comparison and its safeguards.')
 d.add('Coordination and dependence scores also have close precedents.','The plausible new contribution is a careful separation of local motion, extra information and predictor failure.',
  lambda c:cards(c,[('Clinical coordination','PCI studies bilateral timing. Q-PCI studies four-limb coordination in healthy adults.'),('Statistical dependence','Beyond Marginals studies individual, temporal and inter-variable dependence.'),('Gait and selection controls','Intersegmental dynamics, typical-cycle subtraction and confidence selection are existing ideas.')],size=23),
  tag='NOVELTY BOUNDARY',notes='Q-PCI was evaluated in healthy adults and must not be presented as established PD or CP clinical evidence. The LTI asymmetry study is a small preprint. The recent confidence-filtering study directly motivates measuring the selection shift caused by support gates. No claim of being the first phase or dependence-based gait method is supported.',refs=('pci','qpci','marginals','lti','lightweight','selection'),takeaway='Novelty remains conditional on empirical evidence that the controls isolate useful structure.')
 def pipeline(c):
  steps=[('Fresh labels','Read data/gavd and rebuild the source manifest.'),('Shared videos','Keep one source copy under data/videos-gavd.'),('Ambient poses','Crop the person, restore image coordinates and retain confidence.'),('Honest windows','Resample valid neighbors and preserve gaps.')]
  for i,(t,b) in enumerate(steps):
   x=60+i*300;c.card(x,221,260,280,t,b,TEAL,22)
   if i<3:c.arrow(x+264,360,x+294,360)
  c.banner(540,'Inspect first, middle and last frames together with the exact walking interval.',size=23)
 d.add('Follow the existing ingestion pipeline and inspect its weak points.','A single cached source video can appear in multiple labeled sequences. Source identity must follow every derived window.',pipeline,
  tag='VERIFIED INGESTION DESIGN',notes='The study manifest reads the refreshed condition folders. ambient.utils.youtube_cache.download_youtube_urls downloads videos. Ambient SequenceKeypointExtractor uses MediaPipe IMAGE mode and the first pose in a crop, so it is not a validated temporal person tracker. The ingestion code handles one-based annotation frames, scaled bounding boxes and a small crop margin. Use raw image coordinates for new prefix-based normalization. Preserve confidence and interpolation provenance. Never store duplicate condition-specific source videos.',refs=('data','instructions'),takeaway='Bad crops, person switches and missing joints can imitate an unusual gait.')
 def coverage(c):
  c.table(['Support definition','Sources','Normal sources','Windows'],[
   ('Completed 16-joint profile',str(old['sources']),str(old['normal_sources']),f"{old['windows']:,}"),
   ('Proposed 8-joint readout',str(cov['sources']),str(cov['normal_sources']),f"{cov['windows']:,}"),
   ('Focal five-label subset',str(focal),str(labels['normal']),'Not yet scored')],[620,170,200,170],y=218,row_h=76,size=23)
  c.banner(545,'The five added normal sources were already explored during development.',size=24)
 d.add('More supported sources do not create independent validation.','The lower-leg readout improves coverage without requiring reliable arm observations for every score.',coverage,
  tag='READ-ONLY AVAILABILITY, NOT NEW MODEL RESULTS',notes='The candidate readout includes BlazePose indices 25 through 32 and requires four distinct usable patches per joint. Inputs and uniform training masks still cover the original 16-joint allowlist. The read-only candidate cohort has 304 sources, 28 normal sources, 1,381 sequences and 9,442 windows. The five recovered normal sources were excluded previously by arm support. All 28 normal sources have appeared in prior exploration. Source identity is only a proxy for a person; person independence is not established.',refs=('coverage','data'),takeaway='Describe the next evaluation as exploratory reuse of the available cohort.')
 def duration(c):
  rows=[(str(r['frames']),f"{r['sample_span_seconds']:.2f} s",str(r['sources']),str(r['normal_sources']),f"{r['windows']:,}") for r in a['rewindowing']]
  c.table(['Samples','Observed span','Sources','Normal','Windows'],rows,[180,260,230,230,260],y=214,row_h=60,size=23)
 d.add('Longer windows sharply reduce the normal evidence.','These upper-bound counts precede the final joint-profile and phase-reliability gates.',duration,
  tag='READ-ONLY AVAILABILITY, NOT NEW MODEL RESULTS',notes='The sample span is (samples minus one) divided by 15 Hz. It is not samples divided by 15. At 96 samples, only 12 normal sources survive the initial rewindowing probe. The 32-sample upper-bound cohort has 311 sources before the eight-joint readout gate, compared with 304 after that gate. The analysis must not confuse these two denominators. Keep 32 samples as the main proposed setting. Treat phase analyses as optional, separately reported subsets.',refs=('coverage','data'),takeaway='A longer cycle estimate can reduce generalizability through selective exclusion.')
 def contacts(c):
  ev=[r for r in a['event_types'] if 'initial contact' in r['gait_event']];n=sum(r['rows'] for r in ev)
  cards(c,[('Existing contact annotations',f'{n} left or right initial-contact labels occur across 80 sequences.'),('Limited source coverage','They cover 11 non-normal sources and no normal sources.'),('A narrow validation role','Audit a fixed event rule. Separate any threshold-tuning sources from later evaluation.')],size=24)
 d.add('The contact labels cannot validate normal phase estimation.','They are useful auxiliary annotations, but their source and condition coverage is limited.',contacts,
  tag='READ-ONLY ANNOTATION AUDIT',notes='The event inventory also contains seven other nonempty rows. The 751 initial contacts are not 751 independent subjects. If abnormal contact labels influence a phase estimator or reliability threshold, this is supervised development on abnormal sources. Reserve these sources from the relevant evaluation and label the analysis accordingly. Do not silently describe such a pipeline as fully normal-only.',refs=('coverage','data','method'),takeaway='Annotation availability is a constraint on the claim, not a reason to infer missing labels.')
 def joints(c):
  rows=[('11, 12','Left and right shoulder','Optional arm and torso context'),('13, 14','Left and right elbow','Optional arm context'),('15, 16','Left and right wrist','Optional arm context'),('23, 24','Left and right hip','Pelvis and leg context'),('25, 26','Left and right knee','Primary lower-leg readout'),('27, 28','Left and right ankle','Primary lower-leg readout'),('29, 30','Left and right heel','Primary lower-leg readout'),('31, 32','Left and right foot index','Primary lower-leg readout')]
  c.table(['BlazePose indices','Keypoint pair','Features involved'],rows,[270,400,490],y=204,row_h=46,size=21)
 d.add('Keep the mask allowlist and readout set explicit.','The table groups left and right pairs for space. The notebook must also print all 16 individual indices in numerical order.',joints,
  notes='The U reference masks uniformly over indices 11 through 16 and 23 through 32. J and A use declared structured masks on that identical allowlist, without motion-intensity or velocity weighting. No other joints may become masked prediction targets. The candidate score readout is the subset 25 through 32. Removing arm observations from a required score is separate from the declared training-mask comparison.',refs=('instructions','data'),takeaway='The allowlist stays fixed. Uniform masking is the reference for the structured alternatives.')
 d.add('Separate local geometry from the opposite-leg increment.','A knee often helps predict its own ankle simply because the leg is connected. That is a useful control, but a different question.',
  lambda c:cards(c,[('Own joint','Visible observations of the target joint establish a basic temporal prediction.'),('Own leg','Visible observations along the target leg add local geometric information.'),('Both legs','The other leg adds a relational increment beyond the retained own-leg prediction.')]),
  notes='Use the same hidden target patch, visible-support rules and loss weights in each comparison. Shared torso or pelvis information must be assigned explicitly and consistently. The primary added-context comparison is own-leg versus both-leg prediction. Own-joint versus own-leg is the geometry control. Do not rename a model that receives the opposite leg or target-derived phase as own-only.',refs=('method','instructions'),takeaway='The primary question concerns the added information from the other leg.')
 d.add('Preserve the local predictor when adding context.','Fit a competent local baseline, freeze it, then learn a correction that starts at zero.',lambda c:residual(c,top=204),
  tag='ILLUSTRATION OF THE PROPOSED MODEL',notes='Let O denote visible own-leg information, C visible opposite-leg information and Y the same hidden coordinate patch. First fit f0(O) using normal development sources. Then predict f0(O) plus g(O,C), with the correction initialized to zero. The original f0 remains an available function when the correction is zero. This function-class containment does not guarantee better held-out performance, so normal development gates remain necessary.',refs=('method','instructions'),takeaway='A nested prediction family reduces one avoidable failure; held-out evidence is still required.')
 d.add('Give the own-leg control the same extra capacity.','A more flexible model may improve even when the additional leg contains no useful information.',
  lambda c:cards(c,[('Retained baseline','f0(O) is the frozen local coordinate predictor.'),('Capacity control','f0(O) + h(O) gives the own-leg input the same correction architecture.'),('Extra-context model','f0(O) + g(O, C) adds opposite-leg observations with matched optimization.')]),
  notes='Match correction width, depth, update budget, initialization and source sampling as closely as possible. Record effective input dimension and parameter count. Use a shared coordinate target rather than comparing arbitrary latent errors from separate teachers. Start with shared ridge and a small temporal model before adding a JEPA encoder.',refs=('method','instructions'),takeaway='The other-leg contribution must exceed the benefit of extra capacity alone.')
 d.add('Make the estimator pass processes with known answers.','Controlled synthetic data validate the measurement logic. They are not clinical simulations or efficacy results.',
  lambda c:cards(c,[('Independent processes','Match excursion and noise while making the other leg uninformative.'),('Coupled processes','Introduce a known extra predictive relation and require the estimator to recover it.'),('Low headroom','Reduce local prediction error until an apparent null becomes unsurprising.')]),
  notes='Generate independent and coupled processes with comparable marginals, missingness and noise. These fixtures answer whether the estimator can distinguish known statistical structures. They do not reproduce stroke, Parkinson disease, cerebral palsy or myopathy. Predeclare tolerance and reproducibility rules before viewing GAVD outcome scores. A contiguous-gap sensitivity can test a low-headroom concern, but it is separate from the registered U, J and A training comparison.',refs=('method','instructions'),takeaway='A method that cannot recover a known dependence cannot explain a GAVD dependence.')
 d.add('Phase estimation can accidentally reveal the hidden answer.','A phase variable computed from the full target sequence can carry information about the masked movement.',
  lambda c:cards(c,[('Permitted predictor input','Estimate any predictor phase from visible, permitted observations only.'),('Separate descriptive analysis','Full-sequence phase may describe a clip if it never enters masked prediction.'),('Preserve the phenotype','Keep raw time, excursion and asymmetry. Do not independently warp each leg.')],size=24),
  notes='Independent leg warping can erase timing asymmetry. Joint-wise variance normalization can erase small excursion. Removing trunk tilt can erase compensation. Very small or irregular excursions can make phase undefined, so report an unavailable phase rather than a zero anomaly. The main proposed predictor keeps the raw clock and does not depend on successful cycle estimation.',refs=('neuro','method','pci','qpci'),takeaway='A cleaner-looking cycle can be a less faithful representation of the movement.')
 d.add('Shift distal-leg timing without changing the target.','Use a longer source segment and visible hip-relative offsets to construct aligned and shifted contexts without circular wrap.',lambda c:timing(c,top=205),
  tag='ILLUSTRATION OF A PROPOSED TIMING CONTROL',notes='For a 48-frame parent segment, fix Y and O to frames 8:40 in zero-based half-open indexing. Use C from 4:36, 8:40 and 12:44. Apply uniform masks on the original 12-by-16 parent token grid before deriving features. Move donor values with their original visibility flags, so a hidden value cannot reappear after shifting. For each visible distal joint, subtract the visible donor hip and attach that offset to the permitted original opposite hip. Require valid, permitted donor and anchor hips. Never derive a feature from a hidden hip. Keep the original body normalization anchor and use the intersection of permitted context within each original-versus-shifted pair. Restrict the aligned input to that exact paired support. Evaluate offsets separately; the symmetric average on sources eligible for both is secondary. Do not force all timing, noise and dropout controls onto one intersection. This is distal-leg timing, not a shift of the complete leg trajectory. This is a predictor-input challenge, not a physically realizable counterfactual body or a synthetic disease.',refs=('method','instructions'),takeaway='A same-target probe is interpretable only when its support and normalization also stay fixed.')
 d.add('No-wrap changes still need geometry and drift controls.','Hip-relative donor offsets can change distal geometry as well as timing. Measure both effects before interpreting the error change.',
  lambda c:cards(c,[('Exact paired support','Require common target and hip support. Recheck normal prediction on these sparse inputs.'),('Measure donor drift','Report projected bone, angle, amplitude, speed and missingness changes.'),('Raw timing effect','Frozen both-leg squared error: shifted minus aligned context, on identical targets.')],size=24),
  notes='This is a compound distal timing-and-geometry input challenge, not an isolated physiological phase intervention. Report projected bone-length and joint-angle changes as well as motion and support drift. Do not claim exact marginal preservation. No-wrap shifts remove a wrap discontinuity but use different donor frames. A coherent global temporal shift changes both the target and the context; it is not the same-target negative control. Timing changes are input interventions on a predictor and do not establish neural causation. The primary timing effect is source-aggregated squared error of the frozen both-leg predictor with shifted context minus aligned-context error, on identical targets and paired support. A positive effect means aligned context helped this predictor. Calibrated R or S changes are secondary. Recheck normal-development adequacy, available context-token counts and remaining headroom on the exact paired-support aligned inputs. A null response with almost no permitted context is uninformative; do not relax its masking rules.',refs=('method','neuro'),takeaway='An error change caused by visibility or motion drift is not evidence of unusual coordination.')
 d.add('Report local movement and relational evidence separately.','Both signals may be useful, and each has a different failure mode.',
  lambda c:cards(c,[('Local movement channel','Describe excursion, posture, asymmetry and velocity. Fit M on fitting normals.'),('Relational channel','Compare matched own-leg and both-leg losses. Calibrate R on reserved normal sources.'),('Fixed score and fallback','Use S = (M + R)/2. A failed relational gate sets R to zero for every source in its fold.')]),
  notes='M contains only the declared visible-movement descriptors, not local prediction error. Fit its imputation, scaling and covariance on fitting-normal sources. Fit the relational reference R on separate reserved normal sources. The primary relation is loss of f0(O) + h(O) minus loss of f0(O) + g(O,C), measured on the same target and support. A two-sided deviation from a normal relational pattern is more defensible than assuming all abnormal clips must have less context benefit. Stable compensation may increase predictability. Use a fixed aggregation rule and source-balanced weighting. The frozen combined score is S = 0.5(M + R). If a fold fails the normal relational gate, mark R unavailable and set it to zero for every source in that fold. S = M/2 then preserves the local-motion ranking within the fold. Keep the fold and its sources, report which folds support relational learning, and do not claim a successful mechanism from fallback performance. The primary method comparison is S versus M. Relational references remain fitted on the separate reserved normal sources.',refs=('neuro','method','instructions'),takeaway='With R = 0, the combined score preserves the local-motion ranking within that fold.')
 def roles(c):
  c.table(['Per outer fold, if all 28 normals qualify','Sources','Role'],[
   ('Normal model development','16','Inner fit and validation only'),('Normal reference calibration','5','Freeze the score scale'),('Normal outer evaluation','7','Evaluate fixed scores')],[640,160,360],y=220,row_h=72,size=24)
  c.banner(542,'All 28 normal sources were previously explored. The redesign remains exploratory.',size=22)
 d.add('Separate fitting, calibration and evaluation by source.','The proposed four-fold split needs a manifest audit before these planned counts become binding.',roles,
  notes='The 16 development sources must be split again by source for fitting and model selection. Five reference sources must never select a predictor, hyperparameter or phase threshold. Seven normal sources per outer fold are planned for evaluation. Group overlapping clips and known duplicates, and document unresolved identity links. Historical source reuse prevents describing the redesign as a fresh confirmatory study, even when the new pipeline obeys source separation internally.',refs=('method','data','instructions'),takeaway='Many windows from one source do not replace independent normal sources.')
 d.add('Compare every method on the same frozen source cohort.','Refit every comparator on the same source roles. Compare rankings within a fold before aggregating across folds.',
  lambda c:cards(c,[('Within-fold AUROC','Weight fold AUROCs by fixed normal-by-abnormal source-pair counts.'),('Paired uncertainty','Resample sources within each fold and binary label. Keep the original fold weights.'),('Average precision','Report each fold and its prevalence, plus the equally weighted fold mean.')],size=24),
  notes='Do not compare an old 251-source AUROC with a new 304-source AUROC as if the difference were caused by the method. Refit old-style baselines under the new source manifest. Report focal primary and all-folder secondary endpoints with the same frozen scores. The primary AUROC summary is the fixed pair-count-weighted mean of within-fold AUROCs, not a pooled out-of-fold ranking. Show every fold and an equally weighted fold mean as a sensitivity. Report average precision per fold, its prevalence and its equally weighted fold mean. The paired bootstrap resamples whole sources or confirmed duplicate-recording groups within each fold and binary-label stratum, retaining all method scores and the original fold weights. Pooled rankings are secondary because five reference normals do not establish common score calibration across folds. Source bootstrap intervals do not measure training-seed uncertainty or person-generalization. Disease-family slices are small and descriptive.',refs=('method','instructions'),takeaway='Comparability is a property of the evaluation design, not a shared metric name.')
 def baselines(c):
  c.table(['Comparator','Question it answers'],[
   ('Own leg and capacity-matched correction','Does extra flexibility explain the change?'),('Shared ridge and small temporal model','Is a complex encoder necessary?'),('Motion, amplitude and valid phase features','Do simple gait descriptions explain the score?'),('Random and learned encoder, same target','Does learned representation add value?'),('Exp5-style Random Forest; verified prior method','How does it compare with practical alternatives?')],[600,560],y=210,row_h=70,size=22)
 d.add('A strong result must survive inexpensive alternatives.','Choose one primary model family before inspecting the focal endpoint and keep the comparator set fixed.',baselines,
  notes='The Random Forest follows the existing exp5 feature pipeline with matched source splits and the same focal labels. Train supervised family classifiers only on training sources and report them separately from normal-only anomaly detection. Include normal-only local-motion and nuisance controls. A published baseline is a reproduced baseline only after its adaptation and data compatibility have been verified; otherwise label it an explicitly adapted method and avoid an unsupported state-of-the-art comparison.',refs=('baseline','method','instructions'),takeaway='A useful small-model result is stronger than an unsupported architecture claim.')
 def endpoints(c):
  c.card(60,218,560,326,'New exploratory primary',f"Normal versus the four requested families uses {focal} candidate sources: normal {labels['normal']}, Parkinsons {labels['parkinsons']}, stroke {labels['stroke']}, cerebral palsy {labels['cerebral palsy']} and myopathic {labels['myopathic']}.",TEAL,25)
  c.card(652,218,568,326,'Required secondary',f"All 12 folders contribute {cov['sources']} candidate sources. Apply the identical frozen scores and report the broader non-normal endpoint, including activity and style labels.",BLUE,25)
 d.add('Freeze the new endpoint choice before scoring the redesign.','The focal question follows the requested neurological and muscle families. The broader task remains mandatory.',endpoints,
  notes='This is a post-historical-results redesign. It does not replace the original all-folder primary result or create a new confirmatory claim. Stroke appears twice in the request but represents one class. Folder labels are not verified clinical diagnoses. The pooled primary endpoint is normal versus the four requested abnormal families. Five-class classification and individual family analyses are secondary and descriptive, with source-balanced evaluation and broad uncertainty.',refs=('instructions','coverage','data'),takeaway='Do not select the endpoint, labels or model because their observed result looks better.')
 d.add('Notebooks 12 through 14 establish whether the study is possible.','Each notebook needs its own setup, explicit data status and inspectable outputs.',
  lambda c:cards(c,[('12. Data and coverage','Rebuild fresh labels, verify source roles, display walking intervals and quantify support.'),('13. Nested prediction','Run known-answer fixtures, fit local predictors and test capacity-matched corrections.'),('14. Timing probes','Construct no-wrap context shifts and verify fixed targets, masks and normalization.')],size=24),
  notes='These notebooks are new planned deliverables. They must not be numbered as completed experiments in the evidence ledger. Notebook 12 writes an immutable cohort and exclusion manifest. Notebook 13 writes synthetic checks and normal-development reports before model choice is frozen. Notebook 14 validates pairing and drift before any condition-associated interpretation.',refs=('instructions','method'),takeaway='Stop downstream interpretation when support, estimator or control checks fail.')
 d.add('Notebooks 15 through 17 test value and document the limits.','The final evidence package must preserve unsuccessful results as well as favorable ones.',
  lambda c:cards(c,[('15. Reference scores','Fit movement descriptors on fitting normals. Calibrate relational scores on reserved normals.'),('16. Matched evaluation','Refit comparators, evaluate the focal primary and broad secondary endpoints, and show uncertainty.'),('17. Evidence and decision','Assemble provenance, failures, manuscript tables and a claim-by-claim readiness review.')],size=24),
  notes='The score definition and aggregation rule must be frozen before outcome evaluation. Notebook 16 pairs methods at the source level and displays subgroup sample sizes. Notebook 17 must generate reported numbers from saved artifacts, link every figure to its inputs and classify each manuscript claim as supported, descriptive or untested.',refs=('instructions','method'),takeaway='A reproducible negative result can be scientifically useful if its scope is clear.')
 d.add('Make every notebook understandable on the laptop.','A reader should be able to open one notebook, run its setup and understand the exact movement being scored.',
  lambda c:cards(c,[('Independent setup','Use uv and the existing pyproject. Load the project root .env through dotenv aliases.'),('Visible movement','Show the source video interval, overlay joints and explain any unreadable or missing clip.'),('Explicit artifacts','Save configuration, source hashes, support counts, fitted roles and output status.')],size=24),
  notes='Use from dotenv import load_dotenv as load_env and find_dotenv as find_env, then load_env(find_env()) after resolving the project root. Never print secrets. Use a local uv kernel, explain CPU or MPS execution, and make cached pose reuse dependent on configuration and source provenance. For video display, use existing media.show_window_context and annotated-frame helpers with explicit source timestamps. Store all downloads under data/videos-gavd.',refs=('instructions','data'),takeaway='A figure without its walking interval and support record is difficult to interpret.')
 d.add('Require evidence before strengthening the claim.','Each gate has a declared failure response. The study does not continue by searching for favorable subgroups.',lambda c:gates(c,top=211),
  tag='ILLUSTRATION OF PROPOSED EXECUTION GATES',notes='The observation gate audits support and person visibility. The learning gate requires known-dependence recovery and competent normal prediction. The isolation gate requires same-target pairing, capacity controls, documented donor drift and adequate prediction on the exact sparse paired-support inputs. The evaluation gate requires matched source roles, uncertainty and comparison against strong inexpensive baselines. Gate thresholds and fallback claims must be frozen before focal endpoint scores are inspected. A failed relational gate sets R = 0 for all sources in that fold, preserving the M ranking within that fold while retaining all eligible sources.',refs=('instructions','method'),takeaway='Passing a gate permits a narrower inference; it never proves neurological causation.')
 def week(c):
  c.table(['Date','Required output','Decision'],[
   ('Nov 15','Cohort and support audit; optional external pilot','Freeze the eligible study'),('Nov 16','Nested models and normal development','Stop if the predictor is inadequate'),('Nov 17','Timing checks and matched baselines','Freeze the scoring recipe'),('Nov 18','Final paired evaluations and uncertainty','Freeze evidence and claims'),('Nov 19–20','Manuscript, figures and submission checks','Submit only supported conclusions')],[200,580,380],y=211,row_h=71,size=22)
 d.add('Use the remaining week for one falsifiable study.','The workshop page lists November 20, 2026. It does not state a submission timezone.',week,
  notes='Verify the live workshop instructions before submission, because deadlines and format can change. Freeze research evidence by November 18 to leave time for writing and final checks. A failed gate should trigger the predefined narrower paper, not a larger architecture search. The proposed schedule is a plan and is not evidence that experiments have run.',refs=('instructions','workshop'),takeaway='A complete, modest claim is more credible than an unfinished broad claim.')
 d.add('Keep the compute budget small and the choice process visible.','Additional capacity is useful only when it resolves an identified uncertainty.',
  lambda c:cards(c,[('Start simple','Run shared ridge, the retained local model and one small correction family.'),('Freeze a small grid','Choose optimization settings on normal development sources, not abnormal labels.'),('Add JEPA conditionally','Evaluate learned and random representations only after the coordinate estimator passes.')],size=24),
  notes='Stochastic positional embeddings, stochastic latent variables and large architecture searches are optional follow-up ablations. They are not the critical path to this workshop study. VICReg can guard against representational collapse, but it does not force clinical classes apart or guarantee a useful anomaly score. Retain uniform masking as the reference and keep all proposed alternatives motion-independent. Record all attempted settings and failed runs.',refs=('instructions','method'),takeaway='A model ingredient is not a contribution until its added value survives controlled comparison.')
 d.add('The strongest plausible claim is conditional and specific.','The proposed method is valuable only if a measured relational signal survives the obvious alternative explanations.',
  lambda c:cards(c,[('Added information','Opposite-leg observations improve a competent local predictor beyond extra capacity.'),('Interpretable sensitivity','The declared timing challenge changes the relation after accounting for support and drift.'),('Practical value','The frozen relational score improves or complements inexpensive movement descriptions.')],size=24),
  notes='All three statements require new evidence. A successful result would support an observable relational signature in projected video from this source dataset. It would not prove a neural world model, a diagnostic mechanism, a causal effect of disease or generalization to an unseen clinical population. Compare against the closest dependence and normal-gait methods in the manuscript.',refs=('instructions','neuro','method'),takeaway='The manuscript should state exactly which alternative explanations were tested.')
 d.add('Precommit to three honest manuscript outcomes.','The result determines the paper scope. The paper title should not determine the result.',
  lambda c:cards(c,[('Added value survives','Write a constrained relational-anomaly study with paired uncertainty and transparent limits.'),('Structure, little detection gain','Write a measurement study explaining when local and relational views differ.'),('A key gate fails','Write a clearly scoped failure audit if the evidence is substantial enough.')],size=24),
  notes='A null result is not automatically a strong paper. A negative or audit paper still needs a precise research question, systematic controls, reproducible findings and a generalizable methodological lesson. Do not claim workshop acceptance probability from the current exploratory scores. The decision should include independent adversarial review of the final evidence and novelty.',refs=('instructions','method'),takeaway='Do not rebrand a failed adequacy test as a discovered clinical abnormality.')
 d.add('A useful world model must first predict what can be observed.','The near-term goal is a trustworthy comparison. Mechanistic interpretation comes after the comparison works.',
  lambda c:cards(c,[('This week','Test retained local prediction, extra-leg information and falsification controls.'),('Optional independent pilot','Inspect simultaneous laboratory signals for 138 controls and 50 stroke participants.'),('Longer-term research','Study phase, compensation and uncertainty with richer measurements and prospective validation.')],size=24),
  notes='Schedule the optional two-hour external parsing pilot early, before committing the remaining week to GAVD-only execution. The independent laboratory dataset has 138 controls and 50 stroke participants. Its two C3D archives total 779,608,713 bytes. No scientific data were downloaded in this review. Reserve two hours to establish participant identity, common time and signal meaning; stop if a complete measurement replication exceeds one working day. Refit within the external dataset and describe this as method replication, not transfer of GAVD weights. Future studies could test mechanism-sensitive hypotheses using synchronized multi-view motion, force plates or EMG. The current projected GAVD sequences do not identify muscle force, central motor commands or disease etiology. The immediate deliverable is a defensible evidence trail, not an ungrounded neurological explanation.',refs=('instructions','neuro','external','external_paper','workshop'),takeaway='The best next step is the smallest study that can disprove its own explanation.')
 d.records[13:13]=add_mask_slides()
 assert len(d.records)==40,len(d.records)
 return d

def export_deck(d,e,a,provenance):
 OUT.mkdir(exist_ok=True);IMG.mkdir(exist_ok=True);review=OUT/'review';review.mkdir(exist_ok=True)
 for p in IMG.glob('slide-*'):
  if p.suffix in {'.svg','.png','.pdf'}:p.unlink()
 for p in review.glob('*'):
  if p.is_file():p.unlink()
 records=[]
 for i,s in enumerate(d.records,1):
  c=frame(s['title'],s['lede'],s['tag'],i,len(d.records),s['takeaway']);s['draw'](c);stem=IMG/f'slide-{i:02d}';c.save(stem)
  records.append({k:v for k,v in s.items() if k!='draw'})
 pdfs=[str(IMG/f'slide-{i:02d}.pdf') for i in range(1,len(records)+1)]
 subprocess.run(['pdfunite',*pdfs,str(OUT/'tutorial.pdf')],check=True)
 html=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gait prediction: proposed second iteration</title><style>body{margin:0;background:#eef3f7;color:#172e47;font:17px/1.55 system-ui,sans-serif}header{position:sticky;top:0;background:#fff;padding:12px 4vw;box-shadow:0 1px 4px #0002;display:flex;gap:20px;align-items:center;z-index:2}a{color:#2357a5}main{max-width:1280px;margin:auto;padding:18px}section{margin:0 0 36px;background:white;border-radius:12px;overflow:hidden;box-shadow:0 3px 18px #102b4710;scroll-margin-top:80px}img{width:100%;height:auto;display:block}details{padding:16px 28px;border-top:1px solid #d9e3ec}summary{cursor:pointer;font-weight:650}p{max-width:1080px}button{padding:8px 12px;font:inherit;border:1px solid #cbd8e3;border-radius:7px;background:white;cursor:pointer}#count{margin-left:auto}footer{padding:24px 4vw}@media print{header,details,footer{display:none}main{padding:0}section{break-after:page;box-shadow:none;margin:0}@page{size:landscape;margin:0}}</style></head><body><header><b>Proposed second iteration</b><a href="tutorial.pdf">PDF</a><a href="tutorial.md">Notes</a><a href="archive/draft01_completed/tutorial.pdf">Completed draft01 deck</a><button id="prev" aria-label="Previous slide">←</button><button id="next" aria-label="Next slide">→</button><span id="count">40 slides</span></header><main>']
 md=['# Gait prediction: proposed second iteration','', 'This deck distinguishes completed draft01 measurements, read-only availability checks and new proposals. Notebooks 12 through 17 and the proposed residual models have not been implemented or run. The exact historical 40-page deck is preserved in [the completed draft01 archive](archive/draft01_completed/tutorial.pdf).','', '[Open the current PDF](tutorial.pdf) or [the offline slide viewer](index.html).','']
 for i,s in enumerate(records,1):
  links=' · '.join(f'<a href="{escape(SOURCES[k][1],quote=True)}">{escape(SOURCES[k][0])}</a>' for k in s['refs'])
  html.append(f'<section id="slide-{i}"><img src="images/slide-{i:02d}.svg" alt="{escape(s["title"],quote=True)}"><details><summary>Slide {i}: explanation and sources</summary><p><b>{escape(s["tag"])}</b></p><p>{escape(s["lede"])}</p><p>{escape(s["notes"])}</p><p><b>{escape(s["takeaway"])}</b></p><p>{links}</p></details></section>')
  md.extend([f'## {i:02d}. {s["title"]}','',f'**Status: {s["tag"]}.**','',s['lede'],'',f'![Slide {i:02d}](images/slide-{i:02d}.svg)','',s['notes'],'',f'**{s["takeaway"]}**','','Sources: '+ ' · '.join(f'[{SOURCES[k][0]}]({SOURCES[k][1]})' for k in s['refs']), ''])
 html.append('''</main><footer>All proposed diagrams are illustrations. Numerical results are loaded from completed-run evidence or read-only availability files. Source-level support is not participant-level validation.</footer><script>const slides=[...document.querySelectorAll('section')];let current=0;function go(n){current=Math.max(0,Math.min(slides.length-1,n));slides[current].scrollIntoView({behavior:'smooth',block:'start'});document.getElementById('count').textContent=(current+1)+' / '+slides.length;}document.getElementById('prev').onclick=()=>go(current-1);document.getElementById('next').onclick=()=>go(current+1);document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown'].includes(e.key)){e.preventDefault();go(current+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(current-1)}if(e.key==='Home')go(0);if(e.key==='End')go(slides.length-1)});const observer=new IntersectionObserver(entries=>{for(const e of entries){if(e.isIntersecting){current=slides.indexOf(e.target);document.getElementById('count').textContent=(current+1)+' / '+slides.length}}},{threshold:.55});slides.forEach(s=>observer.observe(s));</script></body></html>''')
 (OUT/'index.html').write_text('\n'.join(html));(OUT/'tutorial.md').write_text('\n'.join(md));(OUT/'deck_spec.json').write_text(json.dumps(records,indent=2)+'\n')
 (OUT/'result_input.json').write_text(json.dumps({'completed_run':e['primary_run'],'historical_cohort':e['cohort'],'metrics':e['budget_comparison'],'primary_difference':e['primary_difference'],'read_only_candidate':{k:v for k,v in a['lower_limb_candidate'].items() if k!='prior_use'},'rewindowing_upper_bounds':a['rewindowing'],'provenance':provenance},indent=2)+'\n')
 subprocess.run(['pdftoppm','-r','80','-png',str(OUT/'tutorial.pdf'),str(review/'render')],check=True)
 pages=sorted(review.glob('render-*.png'));assert len(pages)==len(records)
 for start in range(0,len(pages),10):
  subset=pages[start:start+10];thumbw=640;thumbh=360;sheet=Image.new('RGB',(thumbw*2,385*math.ceil(len(subset)/2)), '#e4ebf1');draw=ImageDraw.Draw(sheet)
  for j,p in enumerate(subset):
   im=Image.open(p).convert('RGB');im.thumbnail((thumbw,thumbh));x=(j%2)*thumbw;y=(j//2)*385;sheet.paste(im,(x,y));draw.text((x+15,y+362),f'Slide {start+j+1:02d}',fill=INK)
  sheet.save(review/f'contact-sheet-{start//10+1:02d}.png')
 (OUT/'build_manifest.json').write_text(json.dumps({'status':'Proposal deck; no new model execution','slides':len(records),'builder':'scripts/build_iteration02_slides.py','empirical_inputs':provenance['files'],'pdf_sha256':hashlib.sha256((OUT/'tutorial.pdf').read_bytes()).hexdigest(),'archived_pdf_sha256':hashlib.sha256((OUT/'archive/draft01_completed/tutorial.pdf').read_bytes()).hexdigest(),'vector_figures':list(DIAGRAMS),'rendered_from_final_pdf_pages':len(pages)},indent=2)+'\n')

def validate_mask_design():
 from fractions import Fraction
 groups=[[11,12,23,24],[13,15],[14,16],[25,27,29,31],[26,28,30,32]]
 allow=[11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]
 assert sorted(j for g in groups for j in g)==allow
 probabilities={(t,j):Fraction(0) for t in range(8) for j in allow};cores=0
 for group in groups:
  w=len(group);h=8//w;starts=list(range(0,8,h))
  for start in starts:
   core={(t,j) for t in range(start,start+h) for j in group};assert len(core)==8;cores+=1
   for key in core:probabilities[key]+=Fraction(w,16)*Fraction(1,len(starts))
 assert all(p==Fraction(1,16) for p in probabilities.values())
 assert all(p+(1-p)*Fraction(56,120)==Fraction(1,2) for p in probabilities.values())
 result={'status':'Analytical proposal check, not model execution','tokens':128,'hidden_per_draw':64,'structured_core':8,'uniform_fill':56,'anatomical_cores_enumerated':cores,'token_core_probability':'1/16','token_hidden_probability':'1/2','E_U_draws':16,'E_U_seed':17,'E_leg_draws':2*4*2}
 (OUT/'review/mask_design_validation.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

def validate_outputs():
 report=OUT/'review/text-bounds.html'
 subprocess.run(['pdftotext','-bbox',str(OUT/'tutorial.pdf'),str(report)],check=True)
 root=ET.parse(report).getroot();bad=[];pages=0;words=0
 for p in root.iter():
  if p.tag.endswith('page'):
   pages+=1;w,h=float(p.attrib['width']),float(p.attrib['height'])
   for t in p.iter():
    if not t.tag.endswith('word'):continue
    words+=1;z=t.attrib
    if float(z['xMin'])<0 or float(z['yMin'])<0 or float(z['xMax'])>w+.1 or float(z['yMax'])>h+.1:bad.append({'page':pages,'text':t.text})
 broken=[];links_checked=0
 paths=[OUT/'README.md',OUT/'tutorial.md',OUT/'index.html']+[OUT/'archive/draft01_completed'/n for n in ['index.html','tutorial.md','README.md']]
 for p in paths:
  text=p.read_text();links=re.findall(r'(?:href|src)="([^"]+)"',text) if p.suffix=='.html' else re.findall(r'\]\(([^\)]+)\)',text)
  for link in links:
   if link.startswith(('https:','http:','mailto:','#')):continue
   links_checked+=1;target=(p.parent/unquote(urlsplit(link).path)).resolve()
   if not target.exists():broken.append({'file':str(p.relative_to(BASE)),'link':link})
 fonts=subprocess.check_output(['pdffonts',str(OUT/'tutorial.pdf')],text=True)
 rasters=subprocess.check_output(['pdfimages','-list',str(OUT/'tutorial.pdf')],text=True)
 mask_design=validate_mask_design()
 result={'mask_design':mask_design,'pages':pages,'words':words,'out_of_bounds_words':bad,'local_links_checked':links_checked,'broken_links':broken,'type3_fonts':'Type 3' in fonts,'raster_objects':max(0,len(rasters.strip().splitlines())-2),'archive_pdf_unchanged':hashlib.sha256((OUT/'archive/draft01_completed/tutorial.pdf').read_bytes()).hexdigest()=='bb0502c207305ac9a5f2746c3af5dfaac7eda8c5d7805aa6007726f86fc317d8'}
 (OUT/'review/validation.json').write_text(json.dumps(result,indent=2)+'\n')
 assert pages==40 and not bad and not broken and not result['type3_fonts'] and result['raster_objects']==0 and result['archive_pdf_unchanged'],result
 return result

if __name__=='__main__':
 e,a,provenance=load();build_diagrams();d=build_deck(e,a);export_deck(d,e,a,provenance);validate_outputs();print(f'Built and checked {len(d.records)} proposed slides and six shared diagrams.')
