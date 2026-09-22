"""Rebuild editable conceptual SVGs, two PNG review sizes, and the figure gallery.

Run from the repository: DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
    .venv/bin/python docs/studies/gait-fidelity/scripts/build_figures.py
Dependencies: Pillow, CairoSVG. No video or experiment data are changed.
"""
from pathlib import Path
from html import escape
import json
import hashlib
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'images'
INK, MUTED = '#183247', '#536777'
BLUE, TEAL, RED, AMBER = '#326bb2', '#087e78', '#b64b49', '#a56819'
PALE, GREEN, ROSE, SAND = '#edf3fb', '#eaf6f3', '#fbeeed', '#fff5e6'


class Figure:
    def __init__(self, name, title, subtitle, height=460):
        self.name, self.title, self.subtitle, self.height = name, title, subtitle, height
        self.labels, self.lines = [], []
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>',
                      '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6.5" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#536777"/></marker></defs>',
                      f'<rect width="1000" height="{height}" fill="white"/>']
        self.text(32, 40, title, 27, weight='bold')
        self.text(32, 72, subtitle, 18, color=MUTED)

    def text(self, x, y, s, size=20, color=INK, anchor='start', weight='normal'):
        self.labels.append(dict(x=x, y=y, text=s, size=size, anchor=anchor, weight=weight))
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(s)}</text>')

    def rect(self, x, y, w, h, fill=PALE, stroke='#d5e0e7', radius=12):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def box(self, x, y, w, title, lines=(), h=116, fill=PALE):
        self.rect(x, y, w, h, fill)
        first = len(self.labels)
        self.text(x+w/2, y+32, title, 21, anchor='middle', weight='bold')
        for i, s in enumerate(lines):
            self.text(x+w/2, y+64+27*i, s, 18, color=MUTED, anchor='middle')
        for label in self.labels[first:]:
            label['container'] = [x+8, y+8, x+w-8, y+h-8]

    def line(self, pts, color=MUTED, arrow=True, dashed=False, check=True):
        self.parts.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in pts)}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"' + (' marker-end="url(#arrow)"' if arrow else '') + (' stroke-dasharray="6 5"' if dashed else '') + '/>')
        if check:
            self.lines.extend(zip(pts, pts[1:]))

    def circle(self, x, y, r, fill, stroke='white'):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

    def save(self):
        svg = '\n'.join(self.parts + ['</svg>'])
        ET.fromstring(svg)
        (OUT / f'{self.name}.svg').write_text(svg)
        return dict(file=f'{self.name}.svg', title=self.title, subtitle=self.subtitle,
                    height=self.height, labels=self.labels, lines=self.lines)


def build():
    OUT.mkdir(exist_ok=True)
    records = []
    f = Figure('01-research-question', 'Preserve the movement being measured', 'Three different causes of an unusual pose trace require different responses.', 390)
    for x,title,lines,fill in [
        (32,'Real movement',['Unequal left / right steps','Retain the difference'],GREEN),
        (352,'Observation error',['Blur, gaps or jitter','Reduce estimation error'],PALE),
        (672,'Uncertain side',['Left / right assignment','Resolve or report uncertainty'],SAND)]:
        f.box(x,126,296,title,lines,h=146,fill=fill)
    f.text(500,332,'Judge restoration against an independent reference for the named outcome.',19,anchor='middle')
    records.append(f.save())

    f = Figure('02-crossed-design', 'Separate movement change from observation change', 'Each four-cell family comes from one source identity. This is a proposed design.', 500)
    f.text(417,124,'Clean observation',20,anchor='middle',weight='bold')
    f.text(795,124,'Degraded observation',20,anchor='middle',weight='bold')
    f.text(32,221,'Motion A',21,weight='bold')
    f.text(32,374,'Motion B',21,weight='bold')
    for x,y,title,lines,fill in [
        (265,148,'A · clean',['Reference Y00','Restored estimate'],GREEN),
        (643,148,'A · degraded',['Reference Y01','Restored estimate'],PALE),
        (265,300,'B · clean',['Reference Y10','Restored estimate'],GREEN),
        (643,300,'B · degraded',['Reference Y11','Restored estimate'],PALE)]:
        f.box(x,y,305,title,lines,fill=fill)
    f.line([(570,205),(643,205)])
    f.line([(570,357),(643,357)])
    f.line([(417,264),(417,300)])
    f.line([(795,264),(795,300)])
    f.text(500,465,'Score changes in error; projected references can change with the camera.',19,anchor='middle')
    records.append(f.save())

    f = Figure('03-changing-graph-masks', 'Give each joint opportunities to supply context', 'Illustrative body12 masks at one time block; each draw hides a different connected region.', 480)
    points={'LS':(-37,0),'RS':(37,0),'LE':(-65,55),'RE':(65,55),'LW':(-88,103),'RW':(88,103),
            'LH':(-25,94),'RH':(25,94),'LK':(-30,159),'RK':(30,159),'LA':(-37,216),'RA':(37,216)}
    edges=[('LS','RS'),('LS','LE'),('LE','LW'),('RS','RE'),('RE','RW'),('LS','LH'),('RS','RH'),('LH','RH'),('LH','LK'),('LK','LA'),('RH','RK'),('RK','RA')]
    for cx,title,hidden in [(180,'Draw 1: left leg',{'LH','LK','LA'}),(500,'Draw 2: right arm',{'RS','RE','RW'}),(820,'Draw 3: trunk',{'LS','RS','LH','RH'})]:
        f.text(cx,124,title,20,anchor='middle',weight='bold')
        for a,b in edges:
            f.line([(cx+points[a][0],165+points[a][1]),(cx+points[b][0],165+points[b][1])],arrow=False,color='#b4c2cc',check=False)
        for joint,(x,y) in points.items():
            f.circle(cx+x,165+y,9,RED if joint in hidden else TEAL)
            if joint in hidden:
                f.circle(cx+x,165+y,3,'white')
        f.text(cx-37,411,'L',18,anchor='middle')
        f.text(cx+37,411,'R',18,anchor='middle')
    f.circle(220,451,7,TEAL); f.text(236,457,'Context',18)
    f.circle(390,451,7,RED); f.circle(390,451,2,'white'); f.text(406,457,'Training mask',18)
    f.text(685,457,'L / R are anatomical labels',18,color=MUTED)
    records.append(f.save())

    f = Figure('04-mask-contract', 'Keep artificial masks separate from missing detections', 'The following contract is proposed for the new masking experiment.', 430)
    for x,title,lines in [(32,'Retain raw input',['Coordinates + times','Detection validity']),
                          (280,'Draw a mask',['Graph region + span','Independent RNG']),
                          (528,'Encode context',['No hidden-coordinate','features or shortcuts']),
                          (776,'Score targets',['Reference validity','Normalize loss count'])]:
        f.box(x,130,192,title,lines,h=140)
    for a,b in [(224,280),(472,528),(720,776)]:
        f.line([(a,200),(b,200)])
    f.text(500,326,'Training target mask ≠ detector missingness ≠ image visibility.',21,anchor='middle',weight='bold')
    f.text(500,370,'Deployment receives observed tracks and validity; references are evaluator-only.',18,anchor='middle')
    records.append(f.save())

    f = Figure('05-coverage-audit', 'Audit every joint at every time position', 'A changing mask can still leave the same central tokens rarely visible.', 445)
    f.box(32,130,280,'Window-level count',['“Hip seen at least once”','Can hide central starvation'],h=136,fill=SAND)
    f.box(360,130,280,'Joint × time coverage',['Count context and targets','Across an advancing RNG'],h=136,fill=GREEN)
    f.box(688,130,280,'Accept the sampler',['Both roles if eligible','Check paired-side balance'],h=136)
    f.line([(312,198),(360,198)]); f.line([(640,198),(688,198)])
    f.text(500,320,'Retain achieved mask rate, span lengths, rejection rate and natural missingness.',18,anchor='middle')
    f.rect(95,354,810,57,GREEN)
    f.text(500,390,'No permanent hip or ankle mask. Verify this empirically before training.',20,anchor='middle',weight='bold')
    records.append(f.save())

    f = Figure('06-matched-mask-experiment', 'Test masking and prediction objectives separately', 'Six pretraining arms share the same new data, mask budget and coordinate readout.', 440)
    f.text(82,163,'Objective',18,weight='bold')
    for x,title in [(340,'Current blocks'),(586,'Random tokens'),(832,'Graph + time')]:
        f.text(x,123,title,20,anchor='middle',weight='bold')
    for y,label in [(166,'Coordinates'),(277,'Paired JEPA')]:
        f.text(32,y+47,label,20,weight='bold')
        for x in (229,475,721):
            f.box(x,y,222,'Matched arm',['Same target exposure'],h=90,fill=GREEN if x==721 else PALE)
    f.text(500,410,'Also retain direct end-to-end, no-pretraining, calibration and smoothing controls.',18,anchor='middle')
    records.append(f.save())

    f = Figure('07-video-evidence', 'Use the local videos for the claims they can support', 'Footage and folder labels are available; independent gait references remain to be established.', 470)
    f.box(32,126,292,'Available now',['Natural imaging challenges','Qualitative failure review'],h=139,fill=GREEN)
    f.box(354,126,292,'After annotation',['Visible joint / side checks','Reviewer disagreement'],h=139,fill=PALE)
    f.box(676,126,292,'Needs added reference',['Clinical measurement error','Affected side or real change'],h=139,fill=SAND)
    f.text(500,326,'An MS or PD folder label does not identify an affected limb.',21,anchor='middle',weight='bold')
    f.text(500,366,'Another pose estimator supplies a comparison, not independent ground truth.',18,anchor='middle')
    f.text(500,414,'Keep this collection as development until source overlap and rights are reviewed.',18,anchor='middle')
    records.append(f.save())

    f = Figure('08-data-review-view', 'Review footage together with the quantities being measured', 'Viewer design: the local gallery currently plays raw video and retains review metadata.', 520)
    f.rect(32,113,936,328,'#f7fafc')
    f.box(52,138,280,'Original video',['Native timestamps','View, cut and visibility'],h=134,fill=PALE)
    f.box(360,138,280,'Overlay comparison',['Input / restored skeleton','Reference if available'],h=134,fill=GREEN)
    f.box(668,138,280,'Left / right traces',['Shared physical time','Fixed axes and units'],h=134,fill=PALE)
    f.line([(62,316),(938,316)],arrow=False)
    f.circle(397,316,8,BLUE)
    f.text(500,355,'One linked time cursor; no independent warping of the two legs.',18,anchor='middle')
    f.text(500,398,'Annotate: visible · uncertain side · cut · missing output · reference unavailable',18,anchor='middle')
    f.text(500,485,'Synchronized overlays and quantitative traces are planned work, not generated results.',18,anchor='middle')
    records.append(f.save())

    f = Figure('09-source-splits', 'Split source families before extracting windows', 'The current 41 filename-derived sources are not verified independent people.', 470)
    f.box(32,126,268,'Audit original sources',['Across GAVD and MS / PD','Link duplicate people if known'],h=146,fill=SAND)
    f.box(365,126,270,'Assign one partition',['All clips from that group','All derived versions'],h=146,fill=GREEN)
    f.box(700,126,268,'Then create windows',['Overlaps stay together','No split by frame or clip'],h=146)
    f.line([(300,199),(365,199)]); f.line([(635,199),(700,199)])
    f.box(198,330,604,'Report uncertainty by source family',['Person identity and cross-source dependence remain unverified'],h=97,fill=GREEN)
    records.append(f.save())

    f = Figure('10-assignment-uncertainty', 'Retain plausible side assignments before measuring gait', 'Illustrative case: two equally plausible assignments imply opposite signed differences.', 450)
    f.box(32,135,258,'Assignment 1',['Signed difference: +a','Probability: 0.5'],h=136,fill=GREEN)
    f.box(32,294,258,'Assignment 2',['Signed difference: −a','Probability: 0.5'],h=136,fill=PALE)
    f.box(386,138,582,'Signed mean is zero',['This alone cannot establish a symmetric movement.'],h=104,fill=SAND)
    f.box(386,298,582,'Magnitude is a in both assignments',['Measure each hypothesis, then summarize uncertainty.'],h=104,fill=GREEN)
    f.line([(290,203),(337,203),(337,282)],arrow=False)
    f.line([(290,362),(337,362),(337,282)],arrow=False)
    f.circle(337,282,5,MUTED)
    f.line([(337,282),(360,282),(360,190),(386,190)])
    f.line([(337,282),(360,282),(360,350),(386,350)])
    records.append(f.save())

    f = Figure('11-transformation-contracts', 'An image transform and a joint-label error are different', 'Keep coordinates, anatomical names and validity aligned under every operation.', 406)
    for x,title,lines,fill in [
        (32,'Camera / image',['Update the projection','Use same-view references'],PALE),
        (352,'Joint-slot exchange',['Permute input metadata','Keep anatomical truth fixed'],SAND),
        (672,'Physical movement',['Retain its actual change','Check coupled joint effects'],GREEN)]:
        f.box(x,124,296,title,lines,h=151,fill=fill)
    f.text(500,338,'Do not force equal left–right motion or identical pixels across cameras.',20,anchor='middle',weight='bold')
    records.append(f.save())

    f = Figure('12-evidence-workflow', 'Advance from verified inputs to a frozen comparison', 'New masks and real-video references must pass their checks before any new efficacy claim.', 440)
    for x,title,lines in [(32,'Audit inputs',['Anatomy + time','Groups + rights']),
                          (280,'Check masks',['Coverage audit','Gradient check']),
                          (528,'Compare on dev',['Position + response','Matched controls']),
                          (776,'Confirm once',['Freeze protocol','Fresh source groups'])]:
        f.box(x,135,192,title,lines,h=151,fill=GREEN if x==776 else PALE)
    for a,b in [(224,280),(472,528),(720,776)]: f.line([(a,211),(b,211)])
    f.text(500,346,'Choose useful margins from reference precision; repeated renders are not new people.',18,anchor='middle')
    f.text(500,394,'If references cannot resolve the endpoint, retain a bounded engineering conclusion.',18,anchor='middle')
    records.append(f.save())
    return records


def segment_intersects_rect(a,b,box):
    # The design uses orthogonal connectors. Diagonal skeleton edges are excluded.
    x0,y0,x1,y1=box
    if a[0] == b[0]:
        return x0 < a[0] < x1 and max(min(a[1],b[1]),y0) < min(max(a[1],b[1]),y1)
    if a[1] == b[1]:
        return y0 < a[1] < y1 and max(min(a[0],b[0]),x0) < min(max(a[0],b[0]),x1)
    return False


def render_and_validate(records):
    import cairosvg
    from PIL import ImageFont
    previews=OUT/'previews'; previews.mkdir(exist_ok=True)
    findings=[]
    for fig in records:
        boxes=[]
        for label in fig['labels']:
            suffix=' Bold' if label['weight']=='bold' else ''
            font=ImageFont.truetype(f'/System/Library/Fonts/Supplemental/Arial{suffix}.ttf',label['size'])
            left,top,right,bottom=font.getbbox(label['text'],anchor='ls')
            offset=font.getlength(label['text'])/2 if label['anchor']=='middle' else 0
            box=(label['x']+left-offset,label['y']+top,label['x']+right-offset,label['y']+bottom)
            if 'container' in label:
                cx0,cy0,cx1,cy1=label['container']
                if box[0]<cx0 or box[1]<cy0 or box[2]>cx1 or box[3]>cy1:
                    findings.append((fig['file'],'text outside assigned box padding',label['text']))
            if box[0]<16 or box[1]<8 or box[2]>984 or box[3]>fig['height']-8:
                findings.append((fig['file'],'text outside safe bounds',label['text']))
            for prev,txt in boxes:
                if max(prev[0],box[0])<min(prev[2],box[2]) and max(prev[1],box[1])<min(prev[3],box[3]):
                    findings.append((fig['file'],'overlapping text',txt,label['text']))
            for a,b in fig['lines']:
                if segment_intersects_rect(a,b,box): findings.append((fig['file'],'connector crosses text',label['text']))
            boxes.append((box,label['text']))
        for width,suffix in [(1000,''),(900,'-900')]:
            cairosvg.svg2png(url=str(OUT/fig['file']),output_width=width,
                            write_to=str(previews/(Path(fig['file']).stem+suffix+'.png')))
    (OUT/'figure-layout.json').write_text(json.dumps(records,indent=2)+'\n')
    report={'figure_count':len(records),'preview_count':2*len(records),'findings':findings,
            'checks':['SVG XML parses','Arial text bounds','assigned-box text containment','text intersections','orthogonal connector/text intersections'],
            'limit':'Geometry checks supplement independent visual review; they do not establish scientific correctness.'}
    (OUT/'layout-check.json').write_text(json.dumps(report,indent=2)+'\n')
    if findings: raise ValueError(findings)
    print(f'{len(records)} SVGs and {len(records)*2} previews generated; layout checks passed.')


def gallery(records):
    cards=[]
    for fig in records:
        name=fig['file']
        cards.append(f'<section><h2>{escape(fig["title"])}</h2><img src="{name}" alt="{escape(fig["subtitle"])}"><p>{escape(fig["subtitle"])}</p><p><a href="{name}">Editable SVG</a> · <a href="previews/{Path(name).stem}-900.png">900-pixel review</a></p></section>')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Gait Fidelity — figure gallery</title><style>body{margin:30px auto;max-width:1040px;padding:0 20px;background:#f2f6f9;color:#183247;font:17px/1.6 Arial,sans-serif}section{background:white;padding:20px;margin:28px 0;border:1px solid #d5e0e7;border-radius:12px}img{width:100%;height:auto}h2{font-size:22px}a{color:#326bb2}</style><h1>Gait Fidelity: visual guide</h1><p>Conceptual figures for the proposed experiments. These diagrams contain no new efficacy results. <a href="../README.md">Research proposal</a> · <a href="../methods/masking.md">Masking study</a> · <a href="../data/local-videos.md">Local video audit</a></p>'''+''.join(cards)+'</html>\n'
    (OUT/'gallery.html').write_text(html)


if __name__=='__main__':
    records=build()
    render_and_validate(records)
    gallery(records)
