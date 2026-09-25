"""Rebuild 18 conceptual SVGs, PNG previews and measured layout reports.

DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib .venv/bin/python \
    docs/studies/gait-fidelity/scripts/build_figures.py

Requires Pillow and CairoSVG. Schematic drawings are not results. This builder
writes only SVGs, previews and layout JSON; build_paper.py owns the HTML gallery.
"""
from pathlib import Path
from html import escape
import hashlib
import json
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'images'
W = 1200
INK, MUTED, GRID = '#183247', '#566978', '#d9e2e5'
TEAL, BLUE, AMBER, RED = '#007f79', '#4c60af', '#986325', '#b44c58'
GREEN, PALE, SAND, ROSE = '#e7f4ef', '#edf0fa', '#fbf1de', '#faecee'
SCOPES = {'FOLLOW-UP':(TEAL,GREEN,158), 'INHERITED CORE':(BLUE,PALE,216),
          'INHERITED DESIGN':(BLUE,PALE,238),
          'DIAGNOSTIC':(TEAL,GREEN,168), 'DEFERRED':(AMBER,SAND,146)}


class Figure:
    def __init__(self,name,title,subtitle,scope='FOLLOW-UP',height=630,description=None,
                 status='CONCEPTUAL · RESULTS PENDING'):
        self.name,self.title,self.subtitle,self.scope,self.height=name,title,subtitle,scope,height
        self.description=description or subtitle
        self.labels,self.connectors=[],[]
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(scope+". "+self.description)}</desc>',
            '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#566978"/></marker></defs>',
            f'<rect width="{W}" height="{height}" fill="#fcfcf9"/>']
        color,fill,width=SCOPES[scope]
        self.rect(36,18,width,30,fill,stroke='none',radius=5)
        self.text(48,40,scope,18,color=color,weight='bold')
        self.text(1164,40,status,18,color=MUTED,anchor='end')
        self.text(36,91,title,32,weight='bold')
        self.text(36,124,subtitle,20,color=MUTED)

    def text(self,x,y,text,size=20,color=INK,anchor='start',weight='normal',container=None):
        self.labels.append(dict(x=x,y=y,text=text,size=size,color=color,anchor=anchor,weight=weight,container=container))
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(text)}</text>')

    def rect(self,x,y,w,h,fill=PALE,stroke=GRID,radius=12):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def circle(self,x,y,r,fill=TEAL,stroke='white',width=2):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def line(self,points,color=MUTED,arrow=True,dashed=False,width=2.5,check=True):
        self.parts.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"'+(' marker-end="url(#arrow)"' if arrow else '')+(' stroke-dasharray="7 6"' if dashed else '')+'/>')
        if check: self.connectors.extend(zip(points,points[1:]))

    def panel(self,x,y,w,h,label,fill='white'):
        self.rect(x,y,w,h,fill)
        self.text(x+20,y+34,label,23,weight='bold',container=[x+12,y+10,x+w-12,y+h-12])

    def footer(self,note):
        self.line([(36,self.height-57),(1164,self.height-57)],color=GRID,arrow=False,check=False,width=1)
        self.text(36,self.height-26,note,19,color=MUTED)

    def save(self):
        svg='\n'.join(self.parts+['</svg>']); ET.fromstring(svg)
        (OUT/f'{self.name}.svg').write_text(svg)
        return dict(file=f'{self.name}.svg',title=self.title,subtitle=self.subtitle,
                    description=self.scope+'. '+self.description,scope=self.scope,
                    width=W,height=self.height,labels=self.labels,connectors=self.connectors)


POINTS={'LS':(-37,0),'RS':(37,0),'LE':(-65,50),'RE':(65,50),'LW':(-88,95),'RW':(88,95),
        'LH':(-25,88),'RH':(25,88),'LK':(-31,148),'RK':(31,148),'LA':(-38,208),'RA':(38,208)}
EDGES=[('LS','RS'),('LS','LE'),('LE','LW'),('RS','RE'),('RE','RW'),('LS','LH'),('RS','RH'),('LH','RH'),('LH','LK'),('LK','LA'),('RH','RK'),('RK','RA')]


def skeleton(f,cx,y,scale=1.,hidden=(),motion=False,color=TEAL,noise=False):
    points=dict(POINTS)
    if motion: points.update(RK=(53,140),RA=(12,195))
    if noise: points.update(LK=(-55,131),RA=(60,195))
    for a,b in EDGES:
        f.line([(cx+scale*points[a][0],y+scale*points[a][1]),(cx+scale*points[b][0],y+scale*points[b][1])],color='#a5b8bf',arrow=False,width=3,check=False)
    for joint,(x,yy) in points.items():
        f.circle(cx+x*scale,y+yy*scale,7*scale,RED if joint in hidden else color)
        if joint in hidden: f.circle(cx+x*scale,y+yy*scale,2.3*scale,'white',stroke='none')


def wave(f,x,y,w,h,color=TEAL,amplitude=1.,phase=0.,noise=False):
    pts=[]
    for i in range(81):
        v=math.sin(i/80*math.pi*3.5+phase)*amplitude
        if noise: v+=.16*math.sin(i*1.8)+.10*math.cos(i*.83)
        pts.append((x+i*w/80,y-v*h/2))
    f.line(pts,color=color,arrow=False,width=3,check=False)


def legend(f,x,y,label,color=TEAL,ring=False):
    f.circle(x,y-6,6,color)
    if ring: f.circle(x,y-6,2,'white',stroke='none')
    f.text(x+15,y,label,18,color=MUTED)


def tokens(f,x,y,values,size=22,gap=7,color=BLUE):
    for i,v in enumerate(values): f.rect(x+i*(size+gap),y,size,size,color if v else 'white',stroke=color,radius=4)


def research_question():
    f=Figure('01-research-question','Can learning movement changes improve pose restoration?',
      'Compare errors across two movement states, then test the same type of output network.',height=630,
      description='Two reference-verified movement states feed a residual-coupling pretraining test. A frozen encoder and matched fresh coordinate readout are evaluated for restored movement response and coordinate error. This is a planned comparison, not a result.')
    f.panel(36,166,326,365,'1  Two movement states')
    skeleton(f,116,235,.65); skeleton(f,279,235,.65,motion=True)
    f.text(116,409,'State a',21,anchor='middle',weight='bold'); f.text(279,409,'State b',21,anchor='middle',weight='bold')
    f.line([(151,345),(240,345)]); f.text(199,473,'Same source + view',20,anchor='middle',color=MUTED)
    f.panel(408,166,340,365,'2  Train on error changes',GREEN)
    f.text(486,270,'Error a',27,color=TEAL,anchor='middle'); f.text(671,270,'Error b',27,color=BLUE,anchor='middle')
    f.line([(539,258),(556,258)],arrow=False,width=3); f.line([(600,258),(619,258)],arrow=False,width=3); f.circle(578,258,19,'white',TEAL)
    f.text(578,265,'−',23,anchor='middle',weight='bold')
    f.text(578,336,'Difference of errors',24,anchor='middle',weight='bold')
    f.text(578,378,'vs separate errors',20,anchor='middle',color=MUTED)
    f.text(578,473,'Same model + hidden inputs',20,anchor='middle',color=MUTED)
    f.panel(794,166,370,365,'3  Test restored response')
    f.text(979,263,'Encoder weights fixed',24,anchor='middle',weight='bold'); f.line([(979,281),(979,316)])
    f.text(979,352,'New output network',23,anchor='middle',weight='bold')
    f.text(979,415,'Movement response',22,anchor='middle',color=TEAL)
    f.text(979,449,'+ coordinate accuracy',20,anchor='middle',color=MUTED)
    f.line([(362,348),(408,348)]); f.line([(748,348),(794,348)])
    f.footer('Movement states are paired intervals, not adjacent frames. No paired input is added at deployment.')
    return f.save()


def crossed_design():
    f=Figure('02-crossed-design','Separate movement from how it is observed',
      "Reuse the parent's crossed families, references and physical-time grid.",'INHERITED DESIGN',680,
      description='The inherited experiment crosses paired movement states with observation conditions. '
                  'Drawn skeletons and tracks are illustrative, not measured results.',status='SCHEMATIC')
    f.text(445,178,'Cleaner observation',22,anchor='middle',weight='bold'); f.text(925,178,'Degraded observation',22,anchor='middle',weight='bold')
    for y,state,motion in [(202,'State a',False),(405,'State b',True)]:
        f.text(42,y+81,state,25,weight='bold')
        for x,noisy in [(258,False),(738,True)]:
            f.rect(x,y,374,172,PALE if noisy else GREEN)
            skeleton(f,x+87,y+25,.53,motion=motion,noise=noisy,color=BLUE if noisy else TEAL)
            wave(f,x+168,y+81,165,53,BLUE if noisy else TEAL,.67 if motion else 1.,noise=noisy)
            f.text(x+252,y+146,'Observed track',18,anchor='middle',color=MUTED)
    for y in (288,491): f.line([(632,y),(738,y)])
    for x in (445,925): f.line([(x,374),(x,405)])
    f.footer('Movement contrasts stay within an observation condition. Each camera uses its own projected 2D reference.')
    return f.save()


def graph_masks():
    f=Figure('03-changing-graph-masks','Hide joint regions using the existing body graph',
      'Masking hides part of the observed input during training. All variants use the same policy.','INHERITED CORE',635)
    for cx,title,hidden in [(216,'Leg region',{'LH','LK','LA'}),(600,'Arm region',{'RS','RE','RW'}),(984,'Trunk region',{'LS','RS','LH','RH'})]:
        f.rect(cx-176,164,352,350,'white'); f.text(cx,204,title,23,anchor='middle',weight='bold'); skeleton(f,cx,240,1.05,hidden)
    legend(f,105,552,'Visible context',TEAL); legend(f,315,552,'Hidden for training',RED,True)
    f.text(715,552,'Same masking policy in all three variants',20,color=MUTED)
    f.footer('The follow-up tests auxiliary losses. It introduces no graph, mask-rate or target-exposure ablation.')
    return f.save()


def mask_contract():
    f=Figure('04-mask-contract','Keep three kinds of availability separate',
      'Training queries include deliberately hidden blocks and blocks with missing detections.','INHERITED CORE',610)
    for i in range(6): f.text(476+i*119,190,f'Block {i+1}',19,anchor='middle',weight='bold')
    for title,y,values,color in [('Detector found joint',246,[1,1,0,1,0,1],BLUE),('Training query',328,[0,1,1,1,1,0],RED),('Reference available',410,[1,1,1,0,1,1],TEAL)]:
        f.text(56,y+8,title,24,weight='bold')
        for i,v in enumerate(values):
            f.circle(476+i*119,y,18,color if v else 'white',color,2)
            if v: f.text(476+i*119,y+7,'+',23,color='white',anchor='middle',weight='bold')
            else: f.line([(469+i*119,y),(483+i*119,y)],arrow=False,color=color,check=False)
    f.rect(36,470,1128,66,GREEN)
    f.text(600,511,'Added loss: queried in both states + reference available in all four frames',23,anchor='middle',weight='bold')
    f.footer('One block = one joint over four frames. Pairs excluded from the added loss still receive the original loss.')
    return f.save()


def coverage_audit():
    f=Figure('05-coverage-audit','Check what hidden input and scaling change',
      'Diagnostic examples show available context and a control with no real movement change.','DIAGNOSTIC',675,
      description='The left schematic tracks how often each joint and time block is available. The right control repeats identical raw observations and references with independent hidden-input patterns and input-derived normalization. Teacher feature differences then isolate normalization, while student differences also include hidden context. No measured results are shown.')
    f.panel(36,166,584,354,'How often is each input available?')
    for r,label in enumerate(['Shoulder','Elbow','Hip','Knee','Ankle']):
        f.text(63,255+r*43,label,20)
        for c in range(8):
            a=.22+((r*7+c*3)%9)/12
            f.rect(200+c*47,229+r*43,38,32,f'rgb({int(225-150*a)},{int(244-70*a)},{int(236-65*a)})',stroke='white',radius=3)
    f.text(328,466,'Time-block position →',20,anchor='middle',color=MUTED)
    f.text(328,500,'Darker = more retained context (schematic)',19,anchor='middle',color=MUTED)
    f.panel(664,166,500,354,'Control: no movement change')
    f.text(914,257,'Same raw reference + validity',22,anchor='middle',weight='bold')
    f.line([(914,274),(914,295),(797,295),(797,326)])
    f.line([(914,295),(1030,295),(1030,326)])
    for x,label in [(711,'Shift / scale a'),(944,'Shift / scale b')]:
        f.rect(x,333,172,47,PALE)
        f.text(x+86,363,label,20,anchor='middle',weight='bold')
        f.line([(x+86,387),(x+86,413)])
    tokens(f,735,431,[1,1,0,1],23,8,TEAL)
    tokens(f,968,431,[1,0,1,1],23,8,BLUE)
    f.text(914,488,'Teacher features can differ after scaling',20,anchor='middle',color=MUTED)
    f.text(36,568,'Features = learned numbers. Measure their variation at fixed joint / time positions.',22)
    f.footer('Schematic only. Hidden inputs also affect the student; the unchanged teacher reference isolates shift / scale effects.')
    return f.save()


def matched_experiment():
    f=Figure('06-matched-mask-experiment','Three variants separate three kinds of training',
      'Features are learned numbers. First learn them; fix encoder weights; then train a new output network.',height=700)
    for x,s in [(480,17),(716,29),(952,43)]: f.text(x,185,f'Seed {s}',21,anchor='middle',weight='bold')
    for y,name,sub,color,fill in [(219,'JEPA error change','difference of feature errors',TEAL,GREEN),(330,'JEPA separate errors','independent feature errors',BLUE,PALE),(441,'Coordinate error change','difference of position errors',AMBER,SAND)]:
        f.text(36,y+35,name,25,weight='bold',color=color); f.text(36,y+67,sub,18,color=MUTED)
        for cx in (480,716,952):
            f.rect(cx-104,y,208,86,fill)
            f.text(cx,y+29,'Learn features',21,anchor='middle',weight='bold',color=color)
            f.line([(cx,y+37),(cx,y+47)],width=1.5)
            f.text(cx,y+72,'Train output',20,anchor='middle',color=MUTED)
    f.text(36,594,'3 variants × 3 seeds = 9 models · 18 fitting phases',27,weight='bold')
    f.footer('Reuse parent baselines: plain JEPA, coordinate, direct, initialized and shuffled-reference controls.')
    return f.save()


def video_evidence():
    f=Figure('07-video-evidence','Real-video transfer needs its own reference study',
      'These extensions remain outside the focused synthetic follow-up.','DEFERRED',630)
    for x,title in [(36,'Original footage'),(426,'Independent annotation'),(816,'Supported comparison')]: f.panel(x,168,348,358,title,SAND if x==816 else 'white')
    f.rect(78,238,264,164,'#eff2f1'); f.circle(210,270,15,'#97aaa9'); skeleton(f,210,305,.39,color=BLUE)
    f.text(210,464,'Local videos / GAVD',21,anchor='middle')
    skeleton(f,531,261,.59); f.text(637,290,'Visible',20,color=TEAL); f.text(637,327,'joints',20,color=TEAL)
    f.text(600,464,'Side + time + provenance',21,anchor='middle')
    f.circle(987,304,43,'white',AMBER,2); f.text(987,317,'?',43,color=AMBER,anchor='middle')
    f.text(990,395,'Observation recovery',23,anchor='middle',weight='bold')
    f.text(990,462,'Clinical claims need more',20,anchor='middle',color=MUTED)
    f.line([(384,342),(426,342)]); f.line([(774,342),(816,342)])
    f.footer('No new dataset, GAVD evaluation or clinical reference enters this run. Folder labels are not motion truth.')
    return f.save()


def review_view():
    f=Figure('08-data-review-view','A later viewer would align video and measurements',
      'Proposed inspection display for a separately admitted real-video extension.','DEFERRED',665)
    f.panel(36,167,450,338,'Video + reference overlays')
    f.rect(67,233,388,227,'#edf1f1'); skeleton(f,254,266,.76)
    f.line([(350,246),(350,447)],arrow=False,color=AMBER,dashed=True,check=False)
    f.panel(526,167,638,338,'Left / right traces on one clock')
    f.text(553,267,'L',23,color=TEAL,weight='bold'); wave(f,596,262,527,61,TEAL)
    f.text(553,381,'R',23,color=BLUE,weight='bold'); wave(f,596,379,527,76,BLUE,phase=.28)
    f.line([(849,228),(849,454)],arrow=False,color=AMBER,dashed=True,check=False)
    f.line([(66,556),(1130,556)],arrow=False,color=GRID,width=5,check=False); f.circle(742,556,10,AMBER)
    f.text(600,594,'One physical-time cursor · fixed units · visible missingness',22,anchor='middle',weight='bold')
    f.footer('Schematic only: synchronized overlays and annotated references are not produced by the current follow-up.')
    return f.save()


def source_splits():
    f=Figure('09-source-splits','Inherit the parent population and its split roles',
      'Bind the completed walking-core-01 bundle and receipts; do not rebuild or resplit it.','INHERITED CORE',635)
    f.panel(36,172,320,338,'Completed parent',PALE)
    for y in (252,294,336): f.rect(79,y,228,26,'white',radius=4)
    f.text(196,420,'Manifest + admissions',22,anchor='middle',weight='bold'); f.text(196,461,'Actual counts + updates',20,anchor='middle',color=MUTED)
    f.panel(408,172,338,338,'Bound follow-up',GREEN)
    f.text(577,277,'Training',25,anchor='middle',weight='bold'); f.text(577,311,'Calibrate / fit / diagnose',20,anchor='middle',color=MUTED)
    f.line([(444,347),(710,347)],arrow=False,color=GRID,check=False)
    f.text(577,396,'Development',25,anchor='middle',weight='bold'); f.text(577,433,'Frozen comparison',20,anchor='middle',color=MUTED)
    f.panel(798,172,366,338,'Confirmation stays closed',SAND)
    f.rect(945,279,71,75,'none',AMBER,8)
    f.parts.append(f'<path d="M958 279v-22a23 23 0 0 1 46 0v22" fill="none" stroke="{AMBER}" stroke-width="4"/>')
    f.circle(981,310,5,AMBER); f.text(981,428,'No new people',22,anchor='middle',weight='bold')
    f.text(981,461,'No confirmation inference',20,anchor='middle',color=MUTED); f.line([(356,342),(408,342)])
    f.footer('All derived windows retain their source roles. Historical pilot sizes do not describe this bound population.')
    return f.save()


def uncertainty():
    f=Figure('10-assignment-uncertainty','Uncertain anatomical side can reverse a signed measure',
      'A bilateral interpretation extension requires independent side evidence.','DEFERRED',620)
    f.panel(36,167,456,353,'Two plausible assignments')
    f.text(73,265,'Assignment 1',22,weight='bold'); f.text(73,393,'Assignment 2',22,weight='bold')
    for y,label,col in [(260,'+a',TEAL),(388,'−a',BLUE)]:
        f.circle(383,y,45,GREEN if col==TEAL else PALE,stroke=col); f.text(383,y+11,label,31,color=col,anchor='middle',weight='bold')
    f.text(264,475,'Illustrative equal probabilities',20,anchor='middle',color=MUTED)
    f.panel(573,167,591,353,'A zero average can hide ambiguity',SAND)
    f.line([(681,326),(1052,326)],arrow=False,color=GRID,width=3,check=False)
    f.circle(731,326,15,BLUE); f.circle(1004,326,15,TEAL)
    for x,label,col in [(731,'−a',BLUE),(868,'0',MUTED),(1004,'+a',TEAL)]: f.text(x,374,label,25,anchor='middle',color=col)
    f.line([(868,309),(868,341)],arrow=False,color=MUTED,check=False)
    f.text(869,444,'Magnitude a in both hypotheses',24,anchor='middle',weight='bold'); f.line([(492,342),(573,342)])
    f.footer("The synthetic follow-up uses the parent's anatomical references; it adds no side-assignment model.")
    return f.save()


def transforms():
    f=Figure('11-transformation-contracts','Separate movement, camera view and joint names',
      'The parent transformation rules and projected references stay fixed.','INHERITED CORE',635)
    for x,title,fill in [(36,'Movement change',GREEN),(420,'Camera change',PALE),(804,'Joint-slot exchange',SAND)]: f.panel(x,171,360,344,title,fill)
    skeleton(f,123,253,.64); skeleton(f,303,253,.64,motion=True); f.line([(174,326),(250,326)])
    f.text(216,452,'Retain actual response',21,anchor='middle',weight='bold')
    f.rect(475,261,98,116,'white',BLUE,4); f.parts.append(f'<path d="M620 277L731 253L731 383L620 362Z" fill="white" stroke="{BLUE}" stroke-width="2"/>')
    f.line([(584,317),(610,317)]); f.text(600,452,'Reference from same view',21,anchor='middle',weight='bold')
    for x,y,label,col in [(872,279,'L',BLUE),(1083,279,'R',TEAL),(872,383,'R',TEAL),(1083,383,'L',BLUE)]: f.text(x,y,label,35,color=col,weight='bold')
    f.line([(917,270),(1059,367)],arrow=False,color=BLUE,check=False); f.line([(1066,270),(916,367)],arrow=False,color=TEAL,check=False)
    f.text(984,452,'Anatomical truth stays fixed',21,anchor='middle',weight='bold')
    f.footer('Projected quantities may change with camera geometry. Image coordinates are not clinical 3D angles.')
    return f.save()


def evidence_workflow():
    f=Figure('12-evidence-workflow','Set loss weights on training data, then run the comparison',
      'Calibration sets one fixed weight for each added loss; model weights do not update during this step.',height=685)
    for i,(title,sub) in enumerate([('Reuse','Completed parent'),('Set loss weights','32 training batches'),('Train','18 phases'),('Evaluate','Same people + seeds')]):
        x=36+i*290
        f.circle(x+125,219,26,GREEN,TEAL); f.text(x+125,227,str(i+1),23,color=TEAL,anchor='middle',weight='bold')
        f.text(x+125,293,title,26,anchor='middle',weight='bold'); f.text(x+125,332,sub,20,anchor='middle',color=MUTED)
        if i<3: f.line([(x+166,219),(x+375,219)])
    f.panel(36,386,553,204,'One shared weight for the JEPA losses',GREEN)
    f.text(60,466,'Original gradient',21,weight='bold')
    f.rect(279,445,274,23,TEAL,stroke='none',radius=3)
    f.text(60,518,'Added gradient (max)',21,weight='bold')
    f.rect(279,497,27.4,23,BLUE,stroke='none',radius=3)
    f.text(335,518,'10% initial gradient scale',19,color=MUTED)
    f.text(60,565,'Gradient = how a loss pushes model weights',20,color=MUTED)
    f.panel(617,386,547,204,'Check what the model learns')
    f.text(642,468,'Training-only checks; people kept separate',22,weight='bold')
    f.text(642,509,'Hide input / full input / unchanged motion',21,color=MUTED)
    f.text(642,550,'Feature variation + error fit + training traces',20,color=MUTED)
    f.footer('Use all trainable parameters for gradient energy. Coordinate units get separate calibration; final fits start afresh.')
    return f.save()


def data_claims():
    f=Figure('13-data-to-claims','Existing synthetic data support a limited claim',
      'No new data acquisition enters the focused response follow-up.',height=700)
    for y,title,mid,end,color,fill in [(173,'Completed parent','Projected 2D reference','Response preservation',TEAL,GREEN),(332,'Real video','Independent annotation','Observation recovery',AMBER,SAND),(491,'Clinical extension','Events + side + synchronization','Clinical measurement',AMBER,SAND)]:
        f.rect(36,y,1128,128,fill)
        f.text(59,y+36,'FOLLOW-UP' if y==173 else 'DEFERRED',18,color=color,weight='bold')
        f.text(59,y+86,title,24,weight='bold'); f.text(574,y+61,mid,22,anchor='middle',weight='bold')
        f.text(990,y+61,end,22,anchor='middle',weight='bold')
        f.line([(312,y+54),(357,y+54)],dashed=y!=173); f.line([(801,y+54),(835,y+54)],dashed=y!=173)
        f.text(990,y+96,'Evaluate; do not assume' if y==173 else 'Separate admission needed',18,anchor='middle',color=MUTED)
    f.footer('Distance between learned features is not a physical gait measure. Real-video and clinical claims need new evidence.')
    return f.save()


def paired_jepa():
    f=Figure('14-paired-jepa-method','Compare prediction errors while the encoder is learning',
      'Features are learned numbers. The student predicts them from observed poses; the teacher uses references.',height=755)
    for y,sub in [(197,'State a'),(339,'State b')]:
        f.rect(36,y,190,104,'white'); f.text(131,y+33,sub,24,anchor='middle',weight='bold'); tokens(f,63,y+56,[1,0,1,1,0],21,8)
        f.rect(285,y,263,104,PALE); f.text(416,y+43,'Encoder + predictor',22,anchor='middle',weight='bold')
        f.text(416,y+78,'Some input is hidden',18,anchor='middle',color=MUTED)
        f.rect(607,y,190,104,GREEN); f.text(702,y+42,'Error a' if sub=='State a' else 'Error b',32,anchor='middle',color=TEAL,weight='bold')
        f.text(702,y+79,'Prediction − reference',18,anchor='middle',color=MUTED)
        f.line([(226,y+51),(285,y+51)]); f.line([(548,y+51),(607,y+51)])
    f.text(416,325,'Shared weights for both states',19,anchor='middle',color=BLUE)
    f.panel(866,197,298,246,'Reference teacher',SAND)
    f.text(1015,284,'Reference for each state',21,anchor='middle',weight='bold'); f.text(1015,326,'Learned numbers',23,anchor='middle',color=AMBER)
    f.text(1015,392,'Targets fixed per step',21,anchor='middle',color=MUTED)
    f.line([(866,247),(797,247)]); f.line([(866,391),(797,391)])
    f.line([(416,443),(416,487),(1015,487),(1015,443)],dashed=True); f.text(700,478,'Moving average of encoder weights',18,anchor='middle',color=MUTED)
    f.rect(36,525,1128,131,GREEN); f.text(58,568,'PRETRAIN',18,color=TEAL,weight='bold')
    f.text(58,614,'Original loss + added error term',25,weight='bold')
    f.text(886,568,'THEN FIX ENCODER WEIGHTS',18,color=TEAL,anchor='middle',weight='bold')
    f.text(886,614,'New coordinate output network',22,anchor='middle',weight='bold'); f.line([(636,599),(694,599)])
    f.footer('Each reference uses its input-derived shift and scale. Deployment keeps only the encoder and output network.')
    return f.save()


def graph_time_mask():
    f=Figure('15-graph-time-mask','Compare errors where both movement states have references',
      'A target block is one joint over four frames; each reference frame must be available.','FOLLOW-UP',685)
    for x,state,vals in [(36,'State a',[1,1,0,1,1,0]),(648,'State b',[1,0,0,1,1,1])]:
        f.panel(x,169,516,293,state); f.text(x+23,246,'Target blocks',20,weight='bold'); tokens(f,x+194,224,vals,32,10,RED)
        f.text(x+23,322,'Reference frames',20,weight='bold')
        for c in range(6):
            for r in range(4):
                good=not(c==4 and r==2 and state=='State b')
                f.rect(x+194+c*42,280+r*23,32,17,TEAL if good else 'white',TEAL,2)
        f.text(x+258,421,'Same joint / time block order',20,anchor='middle',color=MUTED)
    f.text(600,307,'∩',36,anchor='middle',weight='bold'); f.text(39,535,'Eligible blocks',24,weight='bold')
    tokens(f,438,506,[1,0,0,1,0,0],40,18,TEAL)
    f.text(609,590,'Average blocks within pairs, then eligible pairs equally',22,anchor='middle',weight='bold')
    f.footer('Illustrative intersection. Other pairs keep the original loss and remain in reports of available targets.')
    return f.save()


def response_estimand():
    f=Figure('16-response-estimand','Measure restored change against the projected reference',
      'Illustrative arithmetic: state b − state a at the same observation condition.',height=730)
    f.panel(36,168,361,466,'Measure knee excursion'); f.text(217,242,'Knee angle over time',21,anchor='middle',weight='bold')
    wave(f,77,312,275,89,TEAL)
    for y in (267,357): f.line([(76,y),(352,y)],color=GRID,arrow=False,dashed=True,check=False)
    f.text(217,419,'q = 95th − 5th percentile',24,anchor='middle',weight='bold'); f.text(217,482,'A = q right − q left',26,anchor='middle',weight='bold')
    f.text(217,548,'Image-plane degrees',21,anchor='middle',color=MUTED)
    f.panel(438,168,726,466,'Illustration: attenuated movement response')
    f.text(468,264,'Reference',22,color=TEAL,weight='bold'); f.line([(734,256),(1059,256)],arrow=False,color=TEAL,width=4,check=False)
    for x,label in [(734,'2°'),(1059,'6°')]: f.circle(x,256,9,TEAL); f.text(x,303,label,24,anchor='middle',color=TEAL,weight='bold')
    f.text(468,368,'Restored',22,color=BLUE,weight='bold'); f.line([(774,360),(937,360)],arrow=False,color=BLUE,width=4,check=False)
    for x,label in [(774,'2.5°'),(937,'4.5°')]: f.circle(x,360,9,BLUE); f.text(x,407,label,24,anchor='middle',color=BLUE,weight='bold')
    f.text(805,476,'Δrestored − Δreference = 2° − 4° = −2°',25,anchor='middle',weight='bold')
    f.text(805,526,'Signed error = −2°; absolute response error = 2°',20,anchor='middle',color=MUTED)
    f.text(805,574,'Also report each leg, joint positions and motion traces',20,anchor='middle',color=MUTED)
    f.footer('Main contrast: delta vs endpoint JEPA, balanced by person and seed. Missing predictions remain failures.')
    return f.save()


def execution():
    f=Figure('17-parallel-execution','A separate follow-up within 48 additional H100-hours',
      "Admission preserves all controls and the parent's actual selected update budget.",height=740)
    f.panel(36,168,510,286,'3 variants × 3 seeds → 9 models',GREEN)
    for r in range(3):
        for c in range(3):
            x,y=72+c*158,235+r*60; f.rect(x,y,126,40,'white',TEAL,5); f.text(x+63,y+27,'P → R',22,color=TEAL,anchor='middle',weight='bold')
    f.text(291,427,'18 phases · P: learn features · R: train output',19,anchor='middle',weight='bold')
    f.panel(586,168,578,286,'Reserved GPU time')
    segments=[(36,'Fits',TEAL),(2,'Calibrate / profile',BLUE),(4,'Diagnostics',AMBER),(6,'Recovery',RED)]
    x=615
    for hrs,label,color in segments:
        width=519*hrs/48; f.rect(x,246,width,50,color,stroke='none',radius=0); x+=width
    for i,(hrs,label,color) in enumerate(segments): legend(f,621+(i%2)*270,343+(i//2)*50,f'{hrs} h  {label}',color)
    f.rect(36,493,1128,151,PALE); f.text(61,535,'RESULTS CUTOFF',18,color=BLUE,weight='bold')
    f.text(61,584,'24 September 2026 · 18:00 Pacific',31,weight='bold')
    f.text(898,544,'2 hours reserved',23,anchor='middle',weight='bold'); f.text(898,583,'for queue uncertainty',22,anchor='middle',color=MUTED)
    f.footer('Measured profiling decides admission. Allowances are not measured throughput or completed fits.')
    return f.save()


def response_loss():
    f=Figure('jepa-response','Train on the difference between two prediction errors',
      'Features are learned numbers. Compare their scaled errors across two movement states.',height=1010,
      description='An error, also called a residual, is predicted minus reference features after fixed scaling and removal of the channel mean. Delta JEPA couples the errors; endpoint JEPA scores them separately. The exact difference is a residual cross term. A coordinate control expresses position errors in a shared pair scale. Calibration uses 32 training batches, followed by fresh pretraining. All variants then fix encoder weights and train a new coordinate output network.')
    f.rect(36,163,1128,151,'white')
    f.text(600,207,'e(i) = H[p(i) / 0.1 − stopgrad((t(i) − c) / 0.06)]',29,anchor='middle',weight='bold')
    f.text(600,248,'e = error (residual) · p = prediction · t = teacher reference · c = teacher average',21,anchor='middle',color=MUTED)
    f.text(600,287,'H subtracts the mean across features; stopgrad holds the reference target fixed for this update.',20,anchor='middle',color=MUTED)
    f.panel(36,345,550,266,'Delta JEPA: compare the errors',GREEN)
    f.panel(614,345,550,266,'Endpoint JEPA: score each error',PALE)
    for cx,col,label in [(180,TEAL,'e(a)'),(440,BLUE,'e(b)'),(738,TEAL,'e(a)'),(1038,BLUE,'e(b)')]:
        f.circle(cx,445,30,'white',col,2); f.text(cx,456,label,28,color=col,anchor='middle',weight='bold')
    f.line([(211,445),(286,445)],arrow=False,color=TEAL,width=3)
    f.line([(334,445),(409,445)],arrow=False,color=TEAL,width=3)
    f.circle(310,445,21,GREEN,TEAL); f.text(310,453,'−',27,color=TEAL,anchor='middle',weight='bold')
    f.text(310,546,'LΔ = ‖e(b) − e(a)‖² / (2D)',28,anchor='middle',weight='bold')
    f.text(889,546,'LE = (‖e(a)‖² + ‖e(b)‖²) / (2D)',27,anchor='middle',weight='bold')
    f.text(600,652,'LΔ = LE − e(a)ᵀe(b) / D',32,anchor='middle',weight='bold')
    f.text(600,693,'D = number of learned features · a and b = two movement states',21,anchor='middle',color=MUTED)
    f.text(600,731,'Equal errors can cancel; check restored joint positions and motion traces too.',22,anchor='middle',color=MUTED)
    f.rect(36,769,1128,109,SAND)
    f.text(57,808,'COORDINATE CONTROL',18,color=AMBER,weight='bold')
    f.text(57,851,'Compare changes in position errors',25,weight='bold')
    f.text(916,815,'Use one common pair scale',22,anchor='middle',weight='bold')
    f.text(916,852,'s pair = (s a + s b) / 2',23,anchor='middle',color=MUTED)
    f.text(600,902,'Both JEPA variants share the added-loss weight and keep the original losses.',21,anchor='middle',color=MUTED)
    f.line([(36,916),(1164,916)],color=GRID,arrow=False,check=False,width=1)
    f.text(600,951,'32 training batches set loss weights → Pretrain the encoder',22,anchor='middle',color=MUTED)
    f.text(600,986,'→ Fix encoder weights → Train a new coordinate output network',22,anchor='middle',color=MUTED)
    return f.save()


BUILDERS=[research_question,crossed_design,graph_masks,mask_contract,coverage_audit,matched_experiment,
          video_evidence,review_view,source_splits,uncertainty,transforms,evidence_workflow,data_claims,
          paired_jepa,graph_time_mask,response_estimand,execution,response_loss]


def build():
    OUT.mkdir(exist_ok=True)
    return [builder() for builder in BUILDERS]


def segment_hits_rect(a,b,box):
    """Liang–Barsky segment/rectangle intersection, including diagonals."""
    x0,y0,x1,y1=box; dx,dy=b[0]-a[0],b[1]-a[1]; lo,hi=0.,1.
    for p,q in [(-dx,a[0]-x0),(dx,x1-a[0]),(-dy,a[1]-y0),(dy,y1-a[1])]:
        if p==0:
            if q<=0: return False
        elif p<0: lo=max(lo,q/p)
        else: hi=min(hi,q/p)
    return lo<hi


def font_path(bold=False):
    candidates=[Path('/System/Library/Fonts/Supplemental')/('Arial Bold.ttf' if bold else 'Arial.ttf'),
        Path('/usr/share/fonts/truetype/liberation2')/('LiberationSans-Bold.ttf' if bold else 'LiberationSans-Regular.ttf'),
        Path('/usr/share/fonts/truetype/dejavu')/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
    return next((p for p in candidates if p.is_file()),candidates[0])


def render_and_validate(records):
    import cairosvg
    from PIL import ImageFont
    previews=OUT/'previews'; previews.mkdir(exist_ok=True)
    findings,assets=[],[]
    for fig in records:
        boxes=[]
        for label in fig['labels']:
            font=ImageFont.truetype(str(font_path(label['weight']=='bold')),label['size'])
            left,top,right,bottom=font.getbbox(label['text'],anchor='ls'); length=font.getlength(label['text'])
            offset=length/2 if label['anchor']=='middle' else length if label['anchor']=='end' else 0
            box=(label['x']+left-offset,label['y']+top,label['x']+right-offset,label['y']+bottom); label['measured_bounds']=box
            if label['size']<18: findings.append([fig['file'],'font below 18px',label['text']])
            c=label['container']
            if c and any((box[0]<c[0],box[1]<c[1],box[2]>c[2],box[3]>c[3])): findings.append([fig['file'],'text outside panel',label['text']])
            if box[0]<16 or box[1]<8 or box[2]>W-16 or box[3]>fig['height']-8: findings.append([fig['file'],'text outside safe canvas',label['text']])
            for prev,txt in boxes:
                if max(prev[0],box[0])<min(prev[2],box[2]) and max(prev[1],box[1])<min(prev[3],box[3]): findings.append([fig['file'],'text overlap',txt,label['text']])
            for a,b in fig['connectors']:
                if segment_hits_rect(a,b,box): findings.append([fig['file'],'connector crosses text',label['text']])
            boxes.append((box,label['text']))
        for width,suffix in [(W,''),(900,'-900')]:
            path=previews/(Path(fig['file']).stem+suffix+'.png')
            cairosvg.svg2png(url=str(OUT/fig['file']),output_width=width,write_to=str(path))
            assets.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        path=OUT/fig['file']; assets.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    subsets=[('figure-layout.json','layout-check.json',records),
             ('proposal-figure-layout.json','proposal-layout-check.json',[f for f in records if f['file'][:2] in {'13','14','15','16'}]),
             ('execution-figure-layout.json','execution-layout-check.json',[f for f in records if f['file'].startswith('17-')])]
    for layout,report,subset in subsets:
        names={f['file'] for f in subset}
        (OUT/layout).write_text(json.dumps(subset,indent=2)+'\n')
        result=dict(figure_count=len(subset),preview_count=2*len(subset),findings=[f for f in findings if f[0] in names],
            assets=[a for a in assets if any(Path(a['path']).stem in {Path(n).stem,Path(n).stem+'-900'} for n in names)],
            checks=['SVG XML parsing','18px minimum font','Pillow text bounds','panel containment','canvas safe area','text intersections','connector/text intersections'],
            fonts={'normal':str(font_path()),'bold':str(font_path(True))},
            limitation='Schematic artwork is excluded from connector checks. Geometry checks supplement rendered visual review and scientific review.')
        (OUT/report).write_text(json.dumps(result,indent=2)+'\n')
    if findings: raise ValueError(json.dumps(findings,indent=2))
    print(f'{len(records)} SVGs and {len(records)*2} previews generated; measured layout checks passed.')


def main():
    render_and_validate(build())


if __name__=='__main__': main()
