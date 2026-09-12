"""Build editable vector diagrams for the ICLR research tutorial.

Concept diagrams are schematics, never images of measured poses. Results are
loaded from saved records by a separate function after the cached panel runs.
"""
from pathlib import Path
from html import escape
import argparse
import json

from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, Circle
from reportlab.graphics import renderSVG, renderPDF
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'docs/studies/iclr/figures'
C = dict(ink='#182B3A', muted='#536675', teal='#087F8C', purple='#7953A5',
         orange='#C06C30', bg='#F5F8FB', line='#CDDAE3', white='#FFFFFF',
         pale='#E8F4F4', violet='#F0EBF7', amber='#FFF0E3', red='#AF4848')


class Figure:
    def __init__(self, title, subtitle, width=1000, height=560):
        self.w, self.h = width, height
        self.d = Drawing(width, height)
        self.rect(0, 0, width, height, 'bg', stroke=None)
        self.text(35, 38, title, 25, 'ink', bold=True)
        self.text(35, 68, subtitle, 15, 'muted')

    def rect(self, x, y, w, h, fill='white', stroke='line', r=10):
        self.d.add(Rect(x, self.h-y-h, w, h, rx=r, ry=r,
                        fillColor=HexColor(C.get(fill, fill)),
                        strokeColor=HexColor(C.get(stroke, stroke)) if stroke else None,
                        strokeWidth=1.2))

    def text(self, x, y, text, size=16, color='ink', bold=False):
        self.d.add(String(x, self.h-y, text, fontName='Helvetica-Bold' if bold else 'Helvetica',
                          fontSize=size, fillColor=HexColor(C.get(color, color))))

    def lines(self, x, y, texts, size=16, color='ink', step=25):
        for i, t in enumerate(texts): self.text(x, y+i*step, t, size, color)

    def arrow(self, x1, y1, x2, y2, color='muted'):
        import math
        self.d.add(Line(x1, self.h-y1, x2, self.h-y2, strokeColor=HexColor(C[color]), strokeWidth=2))
        a=math.atan2(y2-y1,x2-x1)
        pts=[x2,self.h-y2,x2-9*math.cos(a-.4),self.h-(y2-9*math.sin(a-.4)),
             x2-9*math.cos(a+.4),self.h-(y2-9*math.sin(a+.4))]
        self.d.add(Polygon(pts,fillColor=HexColor(C[color]),strokeColor=None))

    def tag(self, x, y, label, color='teal', w=160):
        self.rect(x,y,w,28,color,stroke=None,r=5);self.text(x+10,y+19,label,12,'white',True)

    def save(self, name, description):
        OUT.mkdir(parents=True,exist_ok=True)
        svg=OUT/f'{name}.svg'
        renderSVG.drawToFile(self.d,str(svg))
        s=svg.read_text();i=s.index('>',s.index('<svg'))
        s=s[:i]+ ' role="img" aria-labelledby="title desc"'+s[i:i+1]+f'<title id="title">{escape(name.replace("_"," "))}</title><desc id="desc">{escape(description)}</desc>'+s[i+1:]
        svg.write_text(s)
        renderPDF.drawToFile(self.d,str(OUT/f'{name}.pdf'))


def evidence():
    f=Figure('Two studies expose different breaks in the evidence chain',
             'Each result has its own cohort, target, metric and verification boundary.')
    for x,title,color in [(35,'LATERALITY','purple'),(520,'FUTURE INNOVATION','teal')]:
        f.rect(x,100,445,285);f.tag(x+20,120,title,color,225)
    f.lines(55,181,['625 clips / 93 source videos','Hidden-feature matching becomes more consistent.','Signed movement readout becomes worse.'],17,step=35)
    f.text(55,298,'Initial R2 0.223  >  trained 0.101-0.114',18,'purple',True)
    f.lines(55,334,['Retained aggregate evidence; local raw predictions','and fitted checkpoints unavailable.'],14,'muted',21)
    f.lines(540,181,['50 clips / 43 source videos','Safe scaling and exact baseline fallback pass checks.','Skeletons add no supported value beyond RGB.'],17,step=35)
    f.text(540,298,'Matched increment: -0.00024242 R2',18,'teal',True)
    f.lines(540,334,['Saved models and paired source-bootstrap scores','independently reconstructed.'],14,'muted',21)
    f.arrow(255,387,410,431,'purple');f.arrow(745,387,590,431,'teal')
    f.rect(160,434,680,83,'white');f.text(187,465,'Next measurement: access to a declared motion observable',20,'ink',True)
    f.text(187,494,'Separate symmetry, temporal order, prediction and student benefit.',16,'muted')
    f.text(35,546,'Evidence synthesis. The two R2 definitions and target spaces must not be compared numerically.',13,'muted')
    f.save('01_evidence_bridge','The laterality and future-innovation results motivate separate tests of observable motion; their R squared values are not comparable.')


def information():
    f=Figure('Measure teacher information accessible to the student',
             'RGB complementarity and skeleton-only temporal learning are separate questions.')
    f.rect(35,110,435,128,'white');f.tag(55,126,'EXISTING QUESTION','teal',190)
    f.lines(55,180,['RGB prefix + support  ->  contextual future Y','Does adding skeleton history improve prediction?'],17,step=31)
    f.rect(530,110,435,128,'white');f.tag(550,126,'NEW DEVELOPMENT QUESTION','purple',270)
    f.lines(550,180,['Current posture + support  ->  contextual future Y','What does ordered skeleton history add?'],17,step=31)
    f.rect(35,281,255,126,'pale');f.text(55,310,'H: student input',19,'teal',True)
    f.lines(55,344,['Coordinates, quality and timing','Observed frames 0-31 only'],15)
    f.rect(365,281,275,126,'white');f.text(385,310,'C: current-state reference',19,'ink',True)
    f.lines(385,344,['Frame-31 posture + support','C is a declared function of H'],15)
    f.arrow(290,345,365,345)
    f.rect(715,281,250,126,'violet');f.text(735,310,'Y: teacher target',19,'purple',True)
    f.lines(735,344,['Frozen contextual features','Training supervision only'],15)
    f.rect(160,449,680,68,'white');f.text(187,479,'History contribution: E[Y | H] - E[Y | C]',22,'ink',True)
    f.text(187,504,'A population definition; finite fitted models still require held-source tests.',14,'muted')
    f.text(35,546,'Current cache Y uses full-clip context. A future-only target requires new encoding and audits.',13,'muted')
    f.save('02_information_boundary','Two prediction questions and a student-accessible population definition of temporal innovation. Existing cached target is contextual.')


def parity():
    f=Figure('Left-right sensitivity does not establish temporal-order sensitivity',
             'Schematic observables under anatomical reflection M and time reversal T.')
    f.text(306,119,'Reflection-even',19,'teal',True);f.text(640,119,'Reflection-odd',19,'purple',True)
    f.text(35,214,'Time-even',18,'ink',True);f.text(35,374,'Time-odd',18,'ink',True)
    specs=[(225,145,'pale','Overall median speed',['Keeps its sign under M and T.','Magnitude can ignore ordering.']),
           (590,145,'violet','Left-right speed contrast',['Changes sign under M.','Unchanged under reversal T.']),
           (225,305,'pale','Mean vertical displacement',['Unchanged by anatomical M.','Changes sign under reversal T.']),
           (590,305,'violet','Left-right displacement contrast',['Changes sign under M and T.','A direction-sensitive observable.'])]
    for x,y,bg,title,lines in specs:
        f.rect(x,y,340,132,bg);f.text(x+18,y+31,title,17,'ink',True);f.lines(x+18,y+66,lines,15,step=27)
    f.rect(35,469,930,52,'white');f.text(55,501,'Exact symmetry can hold for a zero representation. Accuracy must be tested separately.',18,'ink',True)
    f.text(35,548,'T is an analysis operation on a bounded sequence, not a label-preserving forecast augmentation.',13,'muted')
    f.save('03_reflection_and_time','Four parity classes distinguish anatomical reflection from time reversal. Perfect transformation behavior alone does not prove information retention.')


def workflow():
    f=Figure('A staged workflow keeps development and confirmation distinct',
             'Use the available cache first; acquire new teacher evidence only for the question that needs it.',height=610)
    rows=[('1','Audit and calibrate','Trace notebooks; verify arithmetic; test known symmetry and order signals.','teal'),
          ('2','Freeze the cached panel','Fix support/current-posture references, grids, controls and source partitions.','teal'),
          ('3','Fit and reconstruct','Train-only transforms; inner selection; outer predictions; paired uncertainty.','teal'),
          ('4','Test future targets','New physical-time windows and mirrored teacher encoding; causal input audits.','purple'),
          ('5','Compare student training','Ordinary S-JEPA, full targets, selected targets and exact no-transfer option.','purple'),
          ('6','Confirm on new sources','Freeze one method and endpoint; include subject-separated evidence.','purple')]
    for i,(n,title,body,color) in enumerate(rows):
        y=102+i*74;f.rect(35,y,930,61,'white');f.tag(48,y+14,n,color,35)
        f.text(103,y+25,title,18,'ink',True);f.text(103,y+47,body,14,'muted')
        if i<5:f.arrow(65,y+62,65,y+72,color)
    f.text(35,579,'Stages 1-3 can use current cached arrays. Later stages need additional empirical evidence.',15,'muted')
    f.save('04_research_workflow','Six-stage workflow separating current cached implementation from future teacher extraction, student training and confirmation.')


def matched():
    import math
    f=Figure('Similar current pose can conceal different futures',
             'Teaching schematic: opposite velocities distinguish two histories that meet at the current state.')
    f.rect(35,105,930,332,'white');f.text(57,137,'Joint displacement',16,'muted')
    f.arrow(90,364,920,364);f.text(880,392,'Time',15,'muted')
    f.d.add(Line(525,f.h-162,525,f.h-384,strokeColor=HexColor(C['line']),strokeWidth=2,strokeDashArray=[5,5]))
    f.text(473,414,'CURRENT POSE',13,'ink',True)
    for sign,col in [(1,'teal'),(-1,'purple')]:
        points=[]
        for i in range(81):
            x=125+i*9.5;y=260-sign*83*math.sin((x-525)/185)
            points.append((x,y))
        for (x1,y1),(x2,y2) in zip(points[:-1],points[1:]):
            f.d.add(Line(x1,f.h-y1,x2,f.h-y2,strokeWidth=3,strokeColor=HexColor(C[col]),
                            strokeDashArray=[7,4] if x1>525 else None))
    f.d.add(Circle(525,f.h-260,7,fillColor=HexColor(C['ink']),strokeColor=None))
    f.text(139,354,'History A',18,'teal',True);f.text(139,169,'History B',18,'purple',True)
    f.text(674,178,'Future A',18,'teal',True);f.text(674,340,'Future B',18,'purple',True)
    f.rect(35,460,930,61,'white');f.text(55,486,'Match posture, view and support using the prefix; then evaluate the observed futures.',17,'ink',True)
    f.text(55,509,'Within-recording contrasts may reduce appearance differences; cancellation must be tested.',14,'muted')
    f.text(35,548,'All curves are synthetic illustrations. Real matched clips do not establish a causal intervention.',13,'muted')
    f.save('05_matched_pose_futures','Synthetic opposite-direction trajectories meet at the same current pose and diverge in the future. History can resolve a state ambiguity.')


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    evidence();information();parity();workflow();matched()
    report_path=ROOT/'outputs/iclr-bridge-cached-20260911/reports/panel-report.json'
    if report_path.exists(): results(report_path)
    (OUT/'README.md').write_text('# Figure provenance\n\nFigures 01-05 are original editable vector diagrams. Coordinates and trajectories are schematic, not extracted poses. Figure 01 uses reported laterality and verified direct-v3 aggregates, keeping their metrics separate. Figure 06 reads `outputs/iclr-bridge-cached-20260911/reports/panel-report.json` directly and shows the paired conditional source-bootstrap intervals, distinguishing the primary posture contrast from secondary comparisons. SVG files are editable vectors; PDF companions support manuscript typesetting. Regenerate with `.venv/bin/python scripts/research_directions/iclr_bridge/build_figures.py`.\n')
    print(f'Built SVG/PDF figures in {OUT}')


def results(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    report=json.loads(path.read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none',
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9.0,4.4),sharey=True)
    contrasts=['real_minus_no-skeleton','real_minus_baseline','real_minus_time-shuffle','real_minus_clip-mismatch']
    labels=['Real − no-skeleton','Real − reference','Real − shuffle','Real − mismatch']
    for ax,panel,title in zip(axes,['posture','support'],['Current posture + support','Support only (secondary)']):
        rows=[report['panels'][panel]['contrasts'][k] for k in contrasts]
        for i,row in enumerate(rows):
            color=C['teal'] if panel=='posture' and i==0 else C['muted']
            ax.errorbar(row['estimate'],i,xerr=[[row['estimate']-row['ci025']],
                        [row['ci975']-row['estimate']]],fmt='o',color=color,capsize=4,linewidth=1.8)
        ax.axvline(0,color=C['line'],linewidth=1.5)
        ax.set_title(title,fontweight='bold',color=C['ink'],pad=13)
        ax.set_yticks(np.arange(4),labels)
        ax.grid(axis='x',alpha=.15);ax.set_xlabel('Difference in predictive R²')
        ax.tick_params(axis='y',length=0)
        ax.set_xlim(-.01,.022);ax.set_xticks([-.01,0,.01,.02])
    axes[0].invert_yaxis()
    fig.suptitle('Cached development comparison: the primary interval includes zero',
                 x=.03,ha='left',fontweight='bold',fontsize=12.5,color=C['ink'])
    fig.text(.03,.02,'50 clips / 43 sources · 2,000 paired source draws · fixed fitted models\n'
             'Teal identifies the prospectively named primary contrast. Other comparisons are descriptive.',
             fontsize=10,color=C['muted'])
    fig.tight_layout(rect=(0,.11,1,.92))
    for extension in ['svg','pdf']: fig.savefig(OUT/f'06_cached_panel_results.{extension}',bbox_inches='tight')
    svg=OUT/'06_cached_panel_results.svg';s=svg.read_text();i=s.index('>',s.index('<svg'))
    s=s[:i]+' role="img" aria-labelledby="title desc"'+s[i:i+1]+'<title id="title">Cached panel results</title><desc id="desc">Point estimates and paired 95 percent source-bootstrap intervals for both panels. The primary posture real-minus-no-skeleton interval includes zero.</desc>'+s[i+1:]
    svg.write_text(s)
    plt.close(fig)


if __name__=='__main__':main()
