"""Draw editable vector figures from anonymous retained aggregate evidence.

Requires reportlab. No generated poses represent participants or forecasts.
"""
from pathlib import Path
import argparse,json,math
from reportlab.graphics.shapes import Drawing,Rect,Line,String,Circle,Polygon,PolyLine
from reportlab.graphics import renderPDF,renderSVG
from reportlab.lib.colors import HexColor, Color

INK='#172B3A'; MUTED='#4E6270'; TEAL='#087F8C'; PURPLE='#7052A1'
ORANGE='#BD5D18'; GRAY='#DDE3E7'; PALE='#F1F6F8'; WHITE='#FFFFFF'

class Canvas:
    def __init__(self,w,h): self.w=w;self.h=h;self.d=Drawing(w,h)
    def rect(self,x,y,w,h,fill=WHITE,stroke=GRAY,r=8):
        self.d.add(Rect(x,self.h-y-h,w,h,rx=r,ry=r,fillColor=HexColor(fill),strokeColor=HexColor(stroke),strokeWidth=1.6))
    def text(self,x,y,s,size=20,color=INK,bold=False,anchor='start'):
        self.d.add(String(x,self.h-y,s,fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,fillColor=HexColor(color),textAnchor=anchor))
    def line(self,x1,y1,x2,y2,color=MUTED,width=2,dash=None):
        self.d.add(Line(x1,self.h-y1,x2,self.h-y2,strokeColor=HexColor(color),strokeWidth=width,strokeDashArray=dash))
    def arrow(self,points,color=MUTED,dashed=False):
        # An explicit final segment of >= 14 px keeps arrowheads aligned.
        for a,b in zip(points,points[1:]): self.line(*a,*b,color,2.2,[6,4] if dashed else None)
        a,b=points[-2:];dx=b[0]-a[0];dy=b[1]-a[1];n=math.hypot(dx,dy)
        assert n>=14
        ux,uy=dx/n,dy/n
        pts=[b[0],self.h-b[1],b[0]-9*ux+4*uy,self.h-(b[1]-9*uy-4*ux),b[0]-9*ux-4*uy,self.h-(b[1]-9*uy+4*ux)]
        self.d.add(Polygon(pts,fillColor=HexColor(color),strokeColor=HexColor(color)))
    def circle(self,x,y,r,color=TEAL,open=False):
        self.d.add(Circle(x,self.h-y,r,fillColor=HexColor(WHITE if open else color),strokeColor=HexColor(color),strokeWidth=1.8))
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        renderPDF.drawToFile(self.d,str(path.with_suffix('.pdf')))
        renderSVG.drawToFile(self.d,str(path.with_suffix('.svg')))

def skeleton(c,x,y,phase,missing=False):
    # Anatomical left and right are color-coded; this is a drawn schematic.
    pts={'head':(x,y),'neck':(x,y+15),'lh':(x+12,y+38),'rh':(x-12,y+38),
         'ls':(x+16,y+18),'rs':(x-16,y+18),'lk':(x+12+phase,y+60),
         'rk':(x-12-phase,y+60),'la':(x+14+2*phase,y+83),'ra':(x-14-2*phase,y+83)}
    for a,b in [('head','neck'),('neck','ls'),('neck','rs'),('ls','lh'),('rs','rh'),('lh','rh'),('lh','lk'),('lk','la'),('rh','rk'),('rk','ra')]:
        col=TEAL if b.startswith('l') else PURPLE if b.startswith('r') else MUTED
        c.line(*pts[a],*pts[b],col,2.4,[3,3] if missing and b=='la' else None)
    for k,(a,b) in pts.items():c.circle(a,b,3.8,TEAL if k.startswith('l') else PURPLE if k.startswith('r') else MUTED,missing and k=='la')

def pipeline(path,version):
    if version >= 5:
        return pipeline_print(path)
    c=Canvas(1000,512)
    c.rect(2,2,996,508,WHITE,WHITE)
    c.text(15,24,'SOURCE-VIDEO SPLIT  |  every clip and derived view keeps its source assignment',20,bold=True)
    c.rect(12,39,244,126,PALE)
    for x,ph,missing in [(56,-3,False),(133,2,True),(210,-2,False)]:skeleton(c,x,51,ph,missing)
    c.arrow([(34,146),(232,146)],INK)
    for x,t in [(52,'t0'),(127,'t1'),(205,'t2')]:c.text(x,162,t,16)
    c.text(274,65,'Timestamped observations',22,bold=True)
    c.text(274,90,'T × 33 × (x, y, z, visibility)',20)
    c.circle(281,114,4,TEAL);c.text(293,120,'observed',18)
    c.circle(402,114,4,TEAL,True);c.text(414,120,'missing',18)
    c.text(274,150,'Schematic poses; no forecast',18,MUTED)
    c.rect(600,42,365,105,'#FFF5EC','#E4CBB8')
    c.text(617,68,'Movement-measure path',22,ORANGE,True)
    c.text(617,94,'Observed transitions + original time',19)
    c.text(617,120,'Five paired median speeds → signed y',19)
    c.arrow([(540,85),(600,85)],ORANGE)
    c.arrow([(965,97),(986,97),(986,467),(945,467)],ORANGE)
    c.arrow([(134,166),(134,196)],TEAL)
    c.rect(12,196,244,146,'#EDF7F7','#B7D8D9')
    c.text(27,223,'Encoder-input path',22,TEAL,True)
    c.text(27,249,'Fill ≤4 samples; normalize',19)
    c.text(27,275,'Resize by index: 64 × 33 × 3',19)
    c.text(27,301,'Original time intervals omitted',18)
    c.text(27,326,'Validity kept separately',19)
    c.arrow([(256,235),(285,235)],TEAL)
    c.rect(285,196,196,146,PALE)
    c.text(299,222,'Four-position patches',19,bold=True)
    c.text(299,246,'16 blocks × 33 joints',18)
    for r in range(3):
        for t in range(8):
            col=ORANGE if (r,t) in [(1,2),(1,3),(2,5)] else GRAY if (r,t)==(0,6) else TEAL
            c.rect(301+t*20,260+r*13,16,9,col,col,1)
    c.arrow([(302,313),(458,313)],INK)
    c.text(300,336,'Context can flank targets',17,MUTED)
    c.arrow([(481,235),(515,235)],TEAL)
    c.rect(515,200,132,71,'#EDF7F7','#B7D8D9')
    c.text(529,226,'Online',22,TEAL,True);c.text(529,253,'encoder',22,TEAL,True)
    c.arrow([(647,235),(682,235)],TEAL)
    c.rect(682,200,136,71,PALE)
    c.text(693,231,'Predictor',22,bold=True);c.text(693,257,'mask tokens',18)
    c.arrow([(818,235),(851,235)],TEAL)
    c.rect(851,199,114,129,'#F5F0FA','#D8C9E7')
    c.text(865,225,'Hidden',21,PURPLE,True);c.text(865,251,'feature',21,PURPLE,True)
    c.text(865,277,'loss',21,PURPLE,True);c.text(865,310,'CE',20,PURPLE)
    c.rect(682,290,136,55,'#F5F0FA','#D8C9E7')
    c.text(695,314,'Teacher',21,PURPLE,True);c.text(695,337,'no gradient',17)
    c.arrow([(818,315),(851,315)],PURPLE)
    c.arrow([(581,271),(581,305),(682,305)],PURPLE,True)
    c.text(594,293,'EMA',17,PURPLE)
    # Unmasked teacher input routes below the patch and online branches.
    c.arrow([(256,328),(269,328),(269,365),(751,365),(751,345)],PURPLE)
    c.text(376,361,'Full prepared clip to teacher',18,PURPLE)
    c.rect(285,382,680,44,PALE)
    c.text(301,410,'Two full geometric views → same online encoder → 12-joint pool → VICReg',18)
    c.arrow([(134,342),(134,404),(285,404)],TEAL)
    c.line(12,440,965,440,GRAY,1.8)
    c.text(15,466,'FROZEN EVALUATION',20,bold=True)
    c.text(15,493,'Held-out source videos',18,MUTED)
    c.rect(259,448,263,52,'#EDF7F7','#B7D8D9')
    c.text(272,470,'Initial / trained encoder',20,TEAL,True)
    c.text(272,492,'paired feature summary',18)
    c.arrow([(522,474),(559,474)],TEAL)
    c.rect(559,448,180,52,PALE);c.text(572,470,'Ridge readout',20,bold=True);c.text(572,492,'fit on train sources',17)
    c.arrow([(739,474),(774,474)],TEAL)
    c.rect(774,448,171,52,'#FFF5EC','#E4CBB8');c.text(788,470,'Estimate y',20,ORANGE,True);c.text(788,491,'score against y',17)
    c.save(path)

def pipeline_print(path):
    c=Canvas(1000,512)
    c.text(15,24,'GAIT OBSERVATIONS  |  source videos define all train/test assignments',21,bold=True)
    c.rect(12,40,244,126,PALE)
    for x,ph,missing in [(56,-3,False),(133,2,True),(210,-2,False)]:skeleton(c,x,49,ph,missing)
    c.arrow([(35,143),(232,143)],INK)
    for x,t in [(49,'t0'),(124,'t1'),(201,'t2')]:c.text(x,163,t,20.5)
    c.text(15,191,'Schematic; original time',20.5,MUTED)
    c.text(290,191,'Tokens carry anatomical joint and position identities',20.5,MUTED)
    c.arrow([(256,101),(290,101)],ORANGE)
    c.rect(290,46,396,120,'#FFF5EC','#E4CBB8')
    c.text(307,73,'Movement-measure path',23,ORANGE,True)
    c.text(307,101,'Observed transitions + timestamps',21)
    c.text(307,127,'Pelvis / body-width normalization',21)
    c.text(307,153,'Five left/right median-speed pairs',21)
    c.arrow([(686,103),(729,103)],ORANGE)
    c.rect(729,67,236,78,'#FFF5EC','#E4CBB8')
    c.text(748,94,'Signed movement y',22,ORANGE,True)
    c.text(748,124,'Left–right contrast',20.5)
    c.arrow([(965,105),(988,105),(988,471),(954,471)],ORANGE)
    c.arrow([(134,195),(134,216)],TEAL)
    c.rect(12,216,244,137,'#EDF7F7','#B7D8D9')
    c.text(28,243,'Encoder input',23,TEAL,True)
    c.text(28,270,'Fill gaps; normalize',20.5)
    c.text(28,297,'64 × 33 × 3 + validity',21)
    c.text(28,322,'Resize by index',20.5)
    c.text(28,346,'Time intervals omitted',20.5)
    c.arrow([(256,253),(290,253)],TEAL)
    c.rect(290,216,196,137,PALE)
    c.text(305,240,'4-position patch',20.5,bold=True)
    c.text(305,266,'16 × 33 tokens',20.5)
    for row in range(3):
        for t in range(8):
            col=ORANGE if (row,t) in [(1,2),(1,3),(2,5)] else GRAY if (row,t)==(0,6) else TEAL
            c.rect(306+t*20,280+row*13,16,9,col,col,1)
    c.arrow([(306,328),(465,328)],INK)
    c.text(304,349,'Two-sided context',20.5)
    c.arrow([(486,253),(518,253)],TEAL)
    c.rect(518,216,127,74,'#EDF7F7','#B7D8D9')
    c.text(533,245,'Online',23,TEAL,True);c.text(533,275,'encoder',23,TEAL,True)
    c.arrow([(645,253),(682,253)],TEAL)
    c.rect(682,216,136,74,PALE)
    c.text(695,245,'Predictor',23,bold=True);c.text(695,275,'mask tokens',20.5)
    c.arrow([(818,253),(851,253)],TEAL)
    c.rect(851,216,114,141,'#F5F0FA','#D8C9E7')
    c.text(866,246,'Hidden',22,PURPLE,True);c.text(866,276,'feature',22,PURPLE,True)
    c.text(866,306,'loss',22,PURPLE,True);c.text(866,341,'CE',21,PURPLE)
    c.rect(682,312,136,55,'#F5F0FA','#D8C9E7')
    c.text(695,336,'Teacher',22,PURPLE,True);c.text(695,360,'no gradient',20.5)
    c.arrow([(818,338),(851,338)],PURPLE)
    c.arrow([(581,290),(581,333),(682,333)],PURPLE,True)
    c.text(598,321,'EMA',20.5,PURPLE)
    c.arrow([(256,341),(273,341),(273,388),(751,388),(751,367)],PURPLE)
    c.text(326,381,'Full prepared clip to teacher',20.5,PURPLE)
    c.rect(290,405,675,40,PALE)
    c.text(304,432,'Full views → online → 12-joint pool → projector → VICReg',20.5)
    c.arrow([(134,353),(134,425),(290,425)],TEAL)
    c.line(12,455,965,455,GRAY,1.8)
    c.text(15,479,'FROZEN READOUT',21,bold=True);c.text(15,505,'Held-out source videos',20.5,MUTED)
    c.rect(253,462,261,47,'#EDF7F7','#B7D8D9')
    c.text(266,483,'Initial/trained encoder',20.5,TEAL,True);c.text(266,505,'Paired summary',20.5)
    c.arrow([(514,485),(550,485)],TEAL)
    c.rect(550,462,191,47,PALE);c.text(563,483,'Ridge regression',21,bold=True);c.text(563,505,'Train-source fit',20.5)
    c.arrow([(741,485),(777,485)],TEAL)
    c.rect(777,462,177,47,'#FFF5EC','#E4CBB8');c.text(792,483,'Estimate y',21,ORANGE,True);c.text(792,505,'Score against y',20.5)
    c.save(path)

def results(path,evidence,version):
    c=Canvas(1000,345)
    if version >= 5:
        original_text=c.text
        def print_text(x,y,s,size=20,color=INK,bold=False,anchor='start'):
            replacements={
                'Expanded summary; higher source-balanced R² is better':'Expanded summary; source-balanced R²',
                'Dots: five seeds. Tick: mean. Same 93 videos.':'Dots: seeds; tick: mean. Same 93 videos.',
                'Counts summarize repeated model checks;':'Repeated checks on fitted models;',
                'they are not independent experiments.':'same videos, not independent trials.',
                'Neither panel evaluates future movement.':'Future movement is untested here.'}
            original_text(x,y,replacements.get(s,s),max(size,20.5),color,bold,anchor)
        c.text=print_text
    rows=evidence['readout_rows']; ini=rows[1]
    c.text(15,25,'A  Movement estimate from frozen features',23,bold=True)
    c.text(15,49,'Expanded summary; higher source-balanced R² is better',18,MUTED)
    labels=['Initial encoder','Random: motion','MAMP-style','Motion mixture','Random: region','Connected region']
    chosen=[ini]+rows[2:7]
    def xx(v):return 204+v/0.30*356
    for tick in [0,.1,.2,.3]:
        c.line(xx(tick),69,xx(tick),270,GRAY,1)
        c.text(xx(tick),296,f'{tick:.1f}',18,MUTED,anchor='middle')
    for i,(r,label) in enumerate(zip(chosen,labels)):
        y=82+i*35
        c.text(15,y+5,label,19,bold=i==0)
        for j,val in enumerate(r['per_seed_r2']):c.circle(xx(val),y+(j-2)*2.2,3.6,TEAL if i==0 else PURPLE)
        c.line(xx(r['mean_r2']),y-9,xx(r['mean_r2']),y+9,INK,2.2)
        c.text(580,y+6,f"{r['mean_r2']:.3f}",18,INK,anchor='end')
    c.text(202,325,'Dots: five seeds. Tick: mean. Same 93 videos.',18,MUTED)
    c.line(608,15,608,331,GRAY,1.5)
    c.text(632,25,'B  Hidden-feature matching',23,bold=True)
    c.text(632,52,'Correct clip beats another source',19,MUTED)
    for y,label,num,den,col in [(90,'Initial',33,75,TEAL),(174,'Trained',375,375,PURPLE)]:
        c.text(632,y,label,21,bold=True)
        c.rect(632,y+12,331,24,PALE,PALE,2)
        c.rect(632,y+12,331*num/den,24,col,col,2)
        c.text(963,y+59,f'{num}/{den} checks',21,col,True,anchor='end')
    c.text(632,272,'Counts summarize repeated model checks;',18,MUTED)
    c.text(632,296,'they are not independent experiments.',18,MUTED)
    c.text(632,325,'Neither panel evaluates future movement.',18,INK)
    c.save(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,required=True);parser.add_argument('--assets',type=Path,required=True)
    args=parser.parse_args();evidence=json.loads((args.assets/'numerical_evidence.json').read_text())
    pipeline(args.assets/'figures/training_pipeline_compact',args.version)
    results(args.assets/'figures/learning_results',evidence,args.version)
    (args.assets/'figures/FIGURE_NOTES.md').write_text('Vector geometry drawn with ReportLab; SVG and PDF use the same drawing objects. Poses and mask cells are schematic, not participant data or model forecasts. The mask schematic illustrates hidden versus missing tokens, not the realized fraction in any training arm. Solid arrows carry data; dashed EMA arrow updates teacher weights. Orange is the measured target or hidden position; teal is context; gray is missing. Target y supervises the separate readout only. The full-view regularizer uses twelve gait joints, including hips; the signed target and bilateral readout use five pairs, excluding hips. The training grid includes neither reflection augmentation nor explicit reflection loss.\n')

if __name__=='__main__':main()
