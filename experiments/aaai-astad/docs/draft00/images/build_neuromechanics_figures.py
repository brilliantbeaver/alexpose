"""Build original explanatory figures. All plotted curves and values are illustrative.

Run from the repository root:
MPLCONFIGDIR=/tmp/aaai-figures experiments/aaai/.venv/bin/python \
  experiments/aaai/docs/draft00/images/build_neuromechanics_figures.py
"""
from pathlib import Path
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
INK, MUTED = '#203547', '#526575'
BLUE, TEAL, GOLD, RED = '#28659B', '#178276', '#AA721D', '#A84F58'
PALE, GREEN, CREAM, ROSE = '#EAF2FA', '#E8F5F1', '#FBF1DF', '#F8EBED'
FIGURES = []

def canvas(title, subtitle='', size=(14, 7)):
    fig = plt.figure(figsize=size, facecolor='white')
    ax = fig.add_axes([.035, .07, .93, .77])
    ax.set(xlim=(0, 14), ylim=(0, 7)); ax.axis('off')
    fig.text(.04, .94, title, fontsize=23, color=INK, weight='bold', va='top')
    if subtitle:
        fig.text(.04, .875, subtitle, fontsize=12, color=MUTED, va='top')
    return fig, ax

def box(ax, x, y, w, h, title, body='', color=PALE, edge=BLUE, fs=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12',
                              facecolor=color, edgecolor=edge, linewidth=1.2))
    ax.text(x+w/2, y+h-.28, title, ha='center', va='top', fontsize=fs,
            color=INK, weight='bold', linespacing=1.3)
    if body:
        ax.text(x+w/2, y+h/2-.15, body, ha='center', va='center', fontsize=11.5,
                color=INK, linespacing=1.5)

def arrow(ax, start, end, color=MUTED, style='-', rad=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15,
                              color=color, lw=1.5, linestyle=style,
                              connectionstyle=f'arc3,rad={rad}'))

def note(fig, text):
    fig.text(.04, .035, text, fontsize=10.5, color=MUTED, va='bottom')

def save(fig, name, caption):
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=130, facecolor='white')
    plt.close(fig)
    FIGURES.append((name, caption))

fig, ax = canvas('From a biological change to a walking video',
                 'The forward chain has several hidden steps. GAVD observes the far end of the chain.')
nodes = [(.1, 'Origin', 'Brain injury,\ndegeneration,\nor muscle disorder'),
         (3.0, 'Movement limits', 'Selective control,\nforce capacity,\nor joint range'),
         (5.9, 'Motion pattern', 'Direct effects and\npossible adaptation'),
         (8.8, 'Physical motion', 'Body movement\nand ground contact')]
for x, title, body in nodes:
    box(ax,x,4.5,2.45,2.0,title,body)
for a,b in zip(nodes,nodes[1:]):
    arrow(ax,(a[0]+2.5,5.5),(b[0]-.07,5.5))
box(ax,8.8,1.2,2.45,1.9,'Observed pose','Camera projection\nand tracking errors',GREEN,TEAL)
arrow(ax,(10.02,4.45),(10.02,3.18))
box(ax,.1,1.2,6.8,1.9,'A reverse inference can have many answers',
    'The same visible movement can arise from different causes.\nA folder label does not identify the hidden steps.',CREAM,GOLD)
ax.text(12.0,5.4,'Mostly\nunmeasured\nin GAVD',ha='center',color=MUTED,fontsize=12)
ax.text(12.0,2.1,'Available\n2D tracks',ha='center',color=TEAL,fontsize=12)
note(fig,'Conceptual causal chain, not a fitted causal model. Age, task, aids, terrain, and experience can affect several steps.')
save(fig,'neurological_chain','Biological origins influence movement through control, body capacity, and adaptation. Video alone does not reveal each step.')

fig, ax = canvas('Four origins, overlapping movement effects',
                 'Examples guide hypotheses. They are not a diagnostic lookup table.',size=(14,8))
ax.text(.2,6.8,'CONDITION',color=MUTED,weight='bold')
ax.text(3.05,6.8,'ORIGIN AND LIMITATION',color=MUTED,weight='bold')
ax.text(8.2,6.8,'POSSIBLE VISIBLE EFFECT',color=MUTED,weight='bold')
rows=[('Stroke','Acquired brain injury;\nweakness or less selective control','Unequal leg motion;\nchanges in foot-clearing strategy'),
      ("Parkinson's",'Degeneration in movement networks;\nchanges in scaling and automaticity','Small steps or arm excursions;\ntask-dependent changes'),
      ('Cerebral palsy','Developing-brain disturbance;\ncontrol and body growth interact','Varied knee, foot, and trunk patterns;\nunilateral or bilateral effects'),
      ('Myopathic gait','Muscle disorders;\nreduced capacity or contracture','Changes in trunk or leg strategy;\npatterns depend on affected muscles')]
for i,(label,origin,effect) in enumerate(rows):
    y=5.2-i*1.55
    box(ax,.15,y,2.45,1.25,label,'',GREEN,TEAL,fs=13)
    box(ax,3.0,y,4.65,1.25,'',origin,PALE,BLUE)
    box(ax,8.15,y,5.6,1.25,'',effect,CREAM,GOLD)
    arrow(ax,(2.65,y+.62),(2.95,y+.62)); arrow(ax,(7.7,y+.62),(8.1,y+.62))
note(fig,'Myopathic does not name one brain disorder. None of these patterns is present in every person or unique to one condition.')
save(fig,'condition_routes','Clinical mechanisms suggest observable questions while leaving substantial overlap between conditions.')

fig, axs = plt.subplots(1,3,figsize=(14,4.7),sharey=True)
fig.subplots_adjust(left=.06,right=.98,bottom=.24,top=.73,wspace=.22)
t=np.linspace(0,4*np.pi,250)
ys=[-np.sin(t),np.sin(t+.6),.9*np.sin(t)]
titles=['Alternating signals','Changed timing','Tightly linked signals']
for i,(ax,y,title) in enumerate(zip(axs,ys,titles)):
    ax.plot(t,np.sin(t),color=BLUE,lw=2.4,label='Signal A')
    ax.plot(t,y,color=RED,lw=2.4,ls='--',label='Signal B')
    ax.set(title=title,xlabel='Time',ylim=(-1.25,1.25),xticks=[])
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=10)
axs[0].set_ylabel('Illustrative movement signal')
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.09),ncol=2,fontsize=10,frameon=False)
fig.text(.04,.94,'A movement can repeat reliably and still have an unusual relationship',fontsize=20,weight='bold',color=INK,va='top')
fig.text(.04,.83,'All three examples are smooth. Predictability alone does not tell us whether a pattern is normal.',color=MUTED,fontsize=12)
note(fig,'Invented signals, not measured gait or simulated disease. No panel is assigned a neurological diagnosis.')
save(fig,'predictable_patterns','Toy signals explain why altered relationships need not make movement noisy or unpredictable.')

fig, ax = plt.subplots(figsize=(13,5.5))
fig.subplots_adjust(left=.19,right=.97,bottom=.29,top=.73)
groups=['Left leg targets','Right leg targets','Arm targets']
y=np.arange(3)[::-1]
ax.axvspan(-.7,.7,color=GREEN,label='Illustrative normal reference')
ax.axvline(0,color=TEAL,lw=1.2)
ax.scatter([-1.8,.1,1.6],y,color=RED,s=115,marker='D',label='One invented abnormal profile',zorder=3)
for yi,xi in zip(y,[-1.8,.1,1.6]):
    ax.plot([0,xi],[yi,yi],color=RED,lw=2)
ax.set(yticks=y,yticklabels=groups,xlim=(-2.8,2.8),ylim=(-.6,2.65),
       xlabel='Deviation in help from other joints (illustrative standardized units)')
ax.set_xticks([-2,-1,0,1,2]); ax.spines[['top','right']].set_visible(False)
ax.text(-2.7,2.42,'Less help',color=BLUE); ax.text(1.75,2.42,'More help',color=BLUE)
fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.55,.10),ncol=2,fontsize=10,frameon=False)
fig.text(.04,.94,'Look for changed support across body regions',fontsize=23,weight='bold',color=INK,va='top')
fig.text(.04,.845,'A two-sided score can flag either less help or more help than the normal reference.',fontsize=12,color=MUTED)
note(fig,'All values and the shaded band are invented. Grouping target joints does not identify which input joint or muscle caused a change.')
save(fig,'support_profile','The proposed profile detects departures in either direction. This figure contains no experimental result.')

fig, ax = canvas('The experiment that fits the coming week',
                 'Preserve normal-only learning, matched targets, and source-separated evaluation.')
box(ax,.15,4.7,3.3,1.6,'Normal training videos','Same 16 allowed joints',GREEN,TEAL)
box(ax,5.0,4.7,3.6,1.6,'Uniform random masks','Same hidden patches')
box(ax,10.05,4.7,3.7,1.6,'Matched predictors','Own joint / all joints')
arrow(ax,(3.5,5.5),(4.92,5.5)); arrow(ax,(8.65,5.5),(9.97,5.5))
box(ax,10.05,1.8,3.7,1.6,'Compare errors','Local and JEPA targets\nreported separately')
box(ax,5.0,1.8,3.6,1.6,'Normal reference','Three-region support profile\nand direct-motion controls',CREAM,GOLD)
box(ax,.15,1.8,3.3,1.6,'Score held-out sources','Normal vs all other folders\nOne score per source',GREEN,TEAL,fs=12.5)
arrow(ax,(11.9,4.64),(11.9,3.48)); arrow(ax,(10,2.6),(8.68,2.6)); arrow(ax,(4.94,2.6),(3.53,2.6))
ax.text(7,.65,'Required checks: patch-local context  |  relative timing  |  view and observation quality',ha='center',fontsize=12,color=MUTED)
note(fig,'Proposed workflow. Existing training results do not validate it. Source groups stay separate across fitting, validation, and testing.')
save(fig,'neuromechanics_experiment','The immediate experiment reuses the existing pose pipeline and tests a specific motion prediction claim.')

fig, ax = canvas('Separate movement changes from observation changes',
                 'Each comparison uses the same masks, body reference, and common valid target support.',size=(14,8))
ax.text(.2,6.75,'TEST',weight='bold',color=MUTED)
ax.text(4.1,6.75,'WHAT CHANGES',weight='bold',color=MUTED)
ax.text(9.0,6.75,'WHAT IT CHECKS',weight='bold',color=MUTED)
rows=[('Relative limb timing','One articulated limb is shifted\non common interior frames.','Sensitivity to a changed\nrelationship between limbs.'),
      ('Whole-body timing','All limbs shift together\nunder matched crop rules.','A crop or global timing\neffect rather than relative timing.'),
      ('Patch-local context','Keep own-joint history; other joints\nonly in the target patch.','Whether local geometry and motion\nexplain the full-context benefit.'),
      ('Observation damage','Specified coordinate noise\nor missing observations.','Sensitivity to the chosen\nmeasurement corruption.')]
for i,(title,change,meaning) in enumerate(rows):
    y=5.15-i*1.5
    box(ax,.15,y,3.3,1.22,title,'',GREEN,TEAL,fs=12.5)
    box(ax,3.95,y,4.5,1.22,'',change)
    box(ax,8.95,y,4.75,1.22,'',meaning,CREAM,GOLD)
note(fig,'Coordinate edits are controlled sensitivity tests. They are not clinical interventions or validated simulations of stroke, PD, CP, or myopathy.')
save(fig,'intervention_controls','Controls test the interpretation of model behavior, rather than assigning synthetic diagnoses.')

fig, ax = canvas('A forward world model needs a body and an observation process',
                 'Longer-term design: predict movement under specified control and physical conditions.',size=(14,8))
box(ax,.15,4.5,3.3,1.8,'Controller','Goal, sensory state,\nand control parameters')
box(ax,5.0,4.5,3.5,1.8,'Muscles and body','Activation, force,\nanatomy, and joint limits')
box(ax,10.05,4.5,3.6,1.8,'Next physical state','3D motion, contact,\nand environment')
arrow(ax,(3.5,5.4),(4.94,5.4)); arrow(ax,(8.56,5.4),(9.99,5.4))
ax.plot([11.85,11.85,1.8,1.8],[4.43,3.75,3.75,4.25],color=MUTED,lw=1.4)
arrow(ax,(1.8,4.18),(1.8,4.45))
ax.text(6.9,3.82,'Sensory feedback',ha='center',va='bottom',color=MUTED,fontsize=11)
box(ax,7.4,.6,3.2,1.65,'Camera and tracker','Projection, occlusion,\nand noise',GREEN,TEAL,fs=12)
ax.plot([12.9,12.9,9.0],[4.43,2.85,2.85],color=TEAL,lw=1.4)
arrow(ax,(9.0,2.85),(9.0,2.3),color=TEAL)
box(ax,11.25,.6,2.4,1.65,'2D pose','GAVD\nobservations',GREEN,TEAL)
arrow(ax,(10.66,1.4),(11.19,1.4),color=TEAL)
box(ax,.15,.6,6.15,1.65,'External measurements validate the components','3D motion, contacts, EMG, strength,\nand known participant information',CREAM,GOLD,fs=12)
note(fig,'A JEPA completion model is one possible representation component. It does not by itself identify a controller, muscle forces, or neural pathology.')
save(fig,'physical_world_model','A future model separates physical dynamics, latent control, and camera observations, with additional measurements for validation.')

fig, ax = canvas('When two causes look similar, keep both explanations',
                 'Proposed simulation benchmark: test what a video can and cannot distinguish.')
box(ax,.15,4.9,3.75,1.4,'Simulated limitation A','For example, weaker push-off')
box(ax,.15,2.8,3.75,1.4,'Simulated limitation B','For example, restricted joint range')
box(ax,5.1,3.75,3.85,1.7,'Matched 2D observations','Similar projected trajectories\nunder explicit matching rules',GREEN,TEAL)
arrow(ax,(3.96,5.6),(5.04,4.9)); arrow(ax,(3.96,3.5),(5.04,4.15))
box(ax,10.15,3.75,3.7,1.7,'Compatible explanations','Report a set and uncertainty,\nnot a unique cause',CREAM,GOLD)
arrow(ax,(9.01,4.6),(10.09,4.6))
box(ax,5.1,.7,8.75,1.5,'Which extra measurement would separate them?',
    'Compare another view, contact timing, strength, or force data in a validated study.',CREAM,GOLD)
arrow(ax,(12,3.69),(12,2.27))
note(fig,'Illustrative candidate pair, not an established equivalence. First demonstrate matched observations and distinct mechanisms inside a validated simulator.')
save(fig,'mechanism_ambiguity','The ambitious contribution is reliable ambiguity and measurement selection, beyond choosing one simulated cause.')

cards=[]
for name,caption in FIGURES:
    cards.append(f'<section><img src="{name}.svg" alt="{html.escape(caption)}"><p>{html.escape(caption)}</p><a href="{name}.pdf">Vector PDF</a></section>')
(OUT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gait research visual guide</title><style>body{font:17px/1.6 system-ui;margin:35px auto;max-width:1150px;padding:0 20px;color:#203547}section{margin:30px 0 60px}img{width:100%;height:auto;border:1px solid #d8e1e8}p{max-width:950px}a{color:#28659b}@media print{section{break-inside:avoid}}</style><h1>Gait research visual guide</h1><p>These are conceptual illustrations of proposed research. Curves and profiles contain no experimental results. The linked research documents provide the evidence and limitations.</p>'+''.join(cards)+'</html>')

# A raster overview makes layout review quick; the individual files remain vector.
fig,axs=plt.subplots(4,2,figsize=(18,19))
for ax,(name,_) in zip(axs.flat,FIGURES):
    ax.imshow(plt.imread(OUT/f'{name}.png')); ax.axis('off')
fig.subplots_adjust(hspace=.04,wspace=.02,left=.01,right=.99,top=.99,bottom=.01)
fig.savefig(OUT/'neuromechanics_contact_sheet.png',dpi=110)
plt.close(fig)
print(f'Built {len(FIGURES)} SVG/PDF figures and an HTML visual guide.')
