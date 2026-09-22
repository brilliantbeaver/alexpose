"""Build four editable proposal diagrams and geometry-checked PNG previews.

Run from the repository root:
    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib .venv/bin/python \
        docs/studies/gait-fidelity/scripts/build_proposal_figures.py

These are explanatory diagrams. They contain no experimental efficacy results.
Dependencies: Pillow and CairoSVG. No data, model, or historical receipt is changed.
"""
from pathlib import Path
from html import escape
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "images"
W = 1200
INK = "#173247"
MUTED = "#536879"
BLUE = "#2866a8"
TEAL = "#057b75"
ORANGE = "#a7571c"
RED = "#ad4749"
PALE = "#edf3fb"
GREEN = "#e9f5f1"
SAND = "#fff4e5"
GREY = "#dde3e8"


class Figure:
    def __init__(self, name, title, subtitle, height):
        self.name, self.height = name, height
        self.labels, self.connectors = [], []
        self.title, self.subtitle = title, subtitle
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>',
            '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6.5" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#536879"/></marker></defs>',
            f'<rect width="{W}" height="{height}" fill="white"/>',
        ]
        self.text(36, 43, title, 30, weight="bold")
        self.text(36, 78, subtitle, 19, color=MUTED)

    def text(self, x, y, text, size=20, color=INK, anchor="start", weight="normal", container=None):
        self.labels.append(dict(x=x, y=y, text=text, size=size, color=color,
                                anchor=anchor, weight=weight, container=container))
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(text)}</text>')

    def rect(self, x, y, w, h, fill=PALE, stroke="#cfdce5", radius=12):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def card(self, x, y, w, h, title, lines=(), fill=PALE, title_size=22):
        self.rect(x, y, w, h, fill)
        container = [x+13, y+12, x+w-13, y+h-12]
        self.text(x+18, y+34, title, title_size, weight="bold", container=container)
        for i, line in enumerate(lines):
            self.text(x+18, y+68+29*i, line, 19, color=MUTED, container=container)

    def line(self, points, color=MUTED, arrow=True, dashed=False, width=2.5, check=True):
        self.parts.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"' + (' marker-end="url(#arrow)"' if arrow else '') + (' stroke-dasharray="7 5"' if dashed else '') + '/>')
        if check:
            self.connectors.extend(zip(points, points[1:]))

    def circle(self, x, y, r, fill, stroke="white", width=2):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def save(self):
        svg = "\n".join(self.parts+["</svg>"])
        ET.fromstring(svg)
        (OUT/f"{self.name}.svg").write_text(svg)
        return dict(file=f"{self.name}.svg", title=self.title, subtitle=self.subtitle,
                    width=W, height=self.height, labels=self.labels, connectors=self.connectors)


POINTS = {"LS":(-37,0),"RS":(37,0),"LE":(-65,55),"RE":(65,55),"LW":(-88,103),"RW":(88,103),
          "LH":(-25,94),"RH":(25,94),"LK":(-30,159),"RK":(30,159),"LA":(-37,216),"RA":(37,216)}
EDGES = [("LS","RS"),("LS","LE"),("LE","LW"),("RS","RE"),("RE","RW"),("LS","LH"),
         ("RS","RH"),("LH","RH"),("LH","LK"),("LK","LA"),("RH","RK"),("RK","RA")]


def skeleton(f, cx, y, scale=1., hidden=()):
    for a, b in EDGES:
        f.line([(cx+scale*POINTS[a][0], y+scale*POINTS[a][1]),
                (cx+scale*POINTS[b][0], y+scale*POINTS[b][1])], arrow=False, color="#aebfc9", width=3, check=False)
    for joint,(dx,dy) in POINTS.items():
        x, yy = cx+scale*dx, y+scale*dy
        f.circle(x, yy, 8*scale, RED if joint in hidden else TEAL)
        if joint in hidden:
            f.circle(x, yy, 2.5*scale, "white", stroke="none")


def data_to_claims():
    f = Figure("13-data-to-claims", "Connect each data source to the evidence it can supply",
               "Dataset role follows reference quality; every comparison remains within its declared evidence boundary.", 820)
    for x,t in [(36,"Data and current access"),(444,"Reference evidence"),(848,"Evaluation enabled")]:
        f.text(x,131,t,23,weight="bold")
    rows = [
        (158, "AMASS synthetic pairs", ["Recorded motion → rendered video", "Estimated tracks + source identity"],
         "Projected body-model joints", ["Known image geometry and time", "Visibility and validity kept apart"],
         "Controlled restoration", ["Position and movement error", "Known motion / image changes"], GREEN),
        (344, "Local real videos", ["91 clips; 41 filename source IDs", "Development collection; GAVD overlap"],
         "Visible-joint annotations", ["To be created by independent review", "Hidden real joints stay unknown"],
         "Real-image stress tests", ["Reviewed positions and side", "Blur / occlusion; time preserved"], PALE),
        (530, "Independent clinical data", ["Candidate access is conditional", "Fresh participants not yet allocated"],
         "Synchronized references", ["Verify contacts / motion and alignment", "Clinical side needs a separate record"],
         "Clinical change agreement", ["Only after endpoint validation", "Needs a fresh participant cohort"], SAND),
    ]
    for y,t,ls,rt,rl,ct,cl,fill in rows:
        f.card(36,y,366,152,t,ls,fill)
        f.card(444,y,362,152,rt,rl,fill,21)
        f.card(848,y,316,152,ct,cl,fill,21)
        f.line([(402,y+78),(444,y+78)],dashed=y>158)
        f.line([(806,y+78),(848,y+78)],dashed=y>158)
    f.text(36,730,"Solid: existing pairing pipeline; new intervention pairs are pending. Dashed: annotation or access work required.",19,color=MUTED)
    f.rect(36,756,1128,42,GREEN)
    f.text(600,784,"Training references fit models; held-out references score outputs. Deployment uses observations only.",20,anchor="middle",weight="bold")
    return f.save()


def paired_jepa():
    f = Figure("14-paired-jepa-method", "Predict reference features, then learn coordinate corrections",
               "Paired JEPA is a proposed mechanism to test against matched coordinate training and simpler controls.", 992)
    f.rect(24,106,1152,499,"#f7fafc")
    f.text(42,142,"1. Pretraining: learn from paired observations and synthetic references",24,weight="bold")
    f.card(42,184,240,140,"Observed pose track",["Hide sampled tokens", "Separate missingness"],PALE,21)
    f.card(340,184,214,140,"Student encoder",["Turns context into", "numerical features"],PALE,21)
    f.card(614,184,212,140,"Predictor",["Estimates reference", "features at queries"],PALE,21)
    f.card(884,184,272,140,"Predicted features",["No reference coordinates", "enter the student"],PALE,21)
    for a,b in [(282,340),(554,614),(826,884)]:
        f.line([(a,255),(b,255)])

    f.card(42,397,240,142,"Projected reference",["Valid joints; paired time", "Training input only"],GREEN,21)
    f.card(340,397,214,142,"Teacher encoder",["Slow copy of student", "No loss gradient"],GREEN,21)
    f.card(614,397,212,142,"Target features",["Summaries of valid", "reference sequences"],GREEN,21)
    f.card(884,397,272,142,"Feature prediction loss",["Compare eligible queries", "Scale by valid target count"],SAND,21)
    for a,b in [(282,340),(554,614),(826,884)]:
        f.line([(a,468),(b,468)])
    f.line([(1018,324),(1018,397)])
    f.line([(447,324),(447,397)],dashed=True)
    f.text(470,355,"EMA: gradual",18,color=MUTED)
    f.text(470,380,"weight update",18,color=MUTED)
    f.text(42,560,"EMA means exponential moving average: gradual teacher weight updates with no loss gradient.",18,color=MUTED)
    f.text(42,588,"Queries are locations to predict. A token is one joint over a short time block; invalid targets are excluded.",18,color=MUTED)

    f.rect(24,633,1152,335,"#f7fafc")
    f.text(42,672,"2. Coordinate fitting and deployment",24,weight="bold")
    f.text(42,709,"Fix the learned encoder; reference coordinates supervise the output network during training only.",20,color=MUTED)
    f.card(42,750,240,134,"Observed pose track",["Coordinates + times", "Availability indicators"],PALE,21)
    f.card(340,750,214,134,"Frozen encoder",["Weights remain fixed", "Produces features"],PALE,21)
    f.card(614,750,240,134,"Coordinate readout",["Learns joint corrections", "from those features"],GREEN,21)
    f.card(914,750,242,134,"Restored trajectory",["Return to input units", "Measure gait afterward"],GREEN,21)
    for a,b in [(282,340),(554,614),(854,914)]:
        f.line([(a,815),(b,815)])
    f.text(600,929,"Deployment follows this lower row without reference inputs or teacher features.",21,anchor="middle",weight="bold")
    return f.save()


def mask_grid(f, x, y, hidden, title):
    names = ["L hip","R hip","L knee","R knee","L ankle","R ankle"]
    f.text(x+137,y-47,title,22,anchor="middle",weight="bold")
    f.text(x+137,y-15,"Time blocks →",19,anchor="middle",color=MUTED)
    for c in range(8):
        f.text(x+c*34+15,y+18,str(c+1),18,anchor="middle")
    for r,name in enumerate(names):
        f.text(x-12,y+57+r*42,name,19,anchor="end")
        for c in range(8):
            natural = r == 5 and c == 1
            fill = GREY if natural else RED if (r,c) in hidden else TEAL
            f.rect(x+c*34,y+31+r*42,30,30,fill,stroke="white",radius=3)
            if not natural and (r,c) in hidden:
                f.circle(x+c*34+15,y+46+r*42,4,"white",stroke="none")
            if natural:
                f.line([(x+c*34+8,y+39+r*42),(x+c*34+22,y+53+r*42)],arrow=False,color=MUTED,width=2,check=False)
                f.line([(x+c*34+22,y+39+r*42),(x+c*34+8,y+53+r*42)],arrow=False,color=MUTED,width=2,check=False)


def graph_time_mask():
    f = Figure("15-graph-time-mask", "Change both the anatomical region and the time interval",
               "Two illustrative presentations of the same input; cells are joint × time-block tokens, not measured results.", 780)
    f.text(175,145,"Schematic body graph",22,anchor="middle",weight="bold")
    skeleton(f,175,203,1.05,{"LH","LK","LA"})
    f.text(175,475,"Left hip → knee → ankle",19,anchor="middle")
    f.text(175,504,"L / R name anatomical sides.",18,anchor="middle",color=MUTED)
    f.text(175,530,"No camera view is implied.",18,anchor="middle",color=MUTED)

    hidden1 = {(r,c) for r in [0,2,4] for c in [2,3,4]}
    hidden2 = {(r,c) for r in [1,3,5] for c in [4,5,6]}
    mask_grid(f,440,210,hidden1,"Draw 1: left leg hidden")
    mask_grid(f,880,210,hidden2,"Draw 2: right leg hidden")
    f.text(585,514,"Leg rows shown; arm rows omitted.",18,anchor="middle",color=MUTED)
    f.text(1010,514,"Left leg supplies context again.",18,anchor="middle",color=MUTED)

    f.rect(36,589,1128,76,GREEN)
    f.text(600,620,"Across advancing random draws, audit every eligible joint at every time position.",21,anchor="middle",weight="bold")
    f.text(600,649,"A changing mask alone does not guarantee that hips or ankles receive enough context.",19,anchor="middle")
    for x,fill,label in [(52,TEAL,"Available context"),(355,RED,"Artificially hidden"),(673,GREY,"Naturally missing input")]:
        f.rect(x,700,22,22,fill,stroke="none",radius=3)
        if fill == RED:
            f.circle(x+11,711,3,"white",stroke="none")
        elif fill == GREY:
            f.line([(x+5,705),(x+17,717)],arrow=False,color=MUTED,width=2,check=False)
            f.line([(x+17,705),(x+5,717)],arrow=False,color=MUTED,width=2,check=False)
        f.text(x+34,719,label,19)
    f.text(36,759,"Artificial masks select available inputs. Reference-valid training targets are tracked in a separate array.",19,color=MUTED)
    return f.save()


def dumbbell(f, y, name, a, b, color, label_a, label_b, delta):
    x0, scale = 175, 72
    f.text(56,y+6,name,20,weight="bold",color=color)
    xa, xb = x0+a*scale, x0+b*scale
    f.line([(xa,y),(xb,y)],arrow=True,color=color,width=3,check=False)
    f.circle(xa,y,8,color)
    f.circle(xb,y,8,color)
    f.text(xa,y-24,f"{a:g}°",20,anchor="middle",color=color,weight="bold")
    f.text(xb,y-24,f"{b:g}°",20,anchor="middle",color=color,weight="bold")
    f.text(xa,y+33,label_a,18,anchor="middle")
    f.text(xb,y+33,label_b,18,anchor="middle")
    f.text(737,y+7,delta,20,anchor="end",color=color,weight="bold")


def response_estimand():
    f = Figure("16-response-estimand", "Separate distortion of movement change from observation sensitivity",
               "Illustrative arithmetic only: signed right-minus-left knee excursion in the image plane, measured in degrees.", 950)
    f.rect(30,117,1140,359,"#f7fafc")
    f.text(52,157,"A. Change the motion; keep the observation condition fixed",24,weight="bold")
    dumbbell(f,242,"Reference",2,6,TEAL,"Motion A","Motion B","Δ = +4°")
    dumbbell(f,372,"Estimate",2.5,4.5,BLUE,"Motion A","Motion B","Δ = +2°")
    f.card(805,198,342,191,"Response distortion",["Estimated change − true change", "+2° − (+4°) = −2°", "The change is underestimated."],SAND,22)
    f.text(52,449,"Baseline measurement error alone cannot describe this loss of movement response.",19,color=MUTED)

    f.rect(30,507,1140,323,"#f7fafc")
    f.text(52,547,"B. Keep motion B fixed; add image occlusion at the same camera",24,weight="bold")
    f.text(56,615,"Reference",20,color=TEAL,weight="bold")
    f.text(276,615,"6°",22,color=TEAL,weight="bold",anchor="middle")
    f.line([(305,608),(516,608)],arrow=True,color=TEAL,width=3,check=False)
    f.text(546,615,"6°",22,color=TEAL,weight="bold",anchor="middle")
    f.text(737,615,"Δ = 0°",20,color=TEAL,weight="bold",anchor="end")
    f.text(56,709,"Estimate",20,color=BLUE,weight="bold")
    f.text(276,709,"4.5°",22,color=BLUE,weight="bold",anchor="middle")
    f.line([(312,702),(510,702)],arrow=True,color=BLUE,width=3,check=False)
    f.text(546,709,"5.5°",22,color=BLUE,weight="bold",anchor="middle")
    f.text(737,709,"Δ = +1°",20,color=BLUE,weight="bold",anchor="end")
    f.text(276,750,"Clean image",19,anchor="middle")
    f.text(546,750,"Added occlusion",19,anchor="middle")
    f.card(805,581,342,191,"Observation-only error",["Estimated change − true change", "+1° − 0° = +1°", "The observation changes error."],SAND,22)
    f.text(52,801,"If the camera changes, use its own projected reference; a 2D quantity may legitimately change.",19,color=MUTED)

    f.text(36,874,"Knee excursion = 95th − 5th percentile of the 2D hip–knee–ankle angle over a fixed reviewed interval.",19)
    f.text(36,910,"These image-plane quantities do not establish clinical 3D range of motion or a meaningful treatment effect.",19,color=MUTED)
    return f.save()


def segment_hits_rect(a, b, box):
    # Orthogonal flow connectors only; illustrative skeleton and chart geometry is excluded.
    x0,y0,x1,y1 = box
    if a[0] == b[0]:
        return x0 < a[0] < x1 and max(min(a[1],b[1]),y0) < min(max(a[1],b[1]),y1)
    if a[1] == b[1]:
        return y0 < a[1] < y1 and max(min(a[0],b[0]),x0) < min(max(a[0],b[0]),x1)
    return False


def render_and_validate(records):
    import cairosvg
    from PIL import ImageFont
    previews = OUT/"previews"
    previews.mkdir(exist_ok=True)
    findings, assets = [], []
    for fig in records:
        boxes=[]
        for label in fig["labels"]:
            if label["size"] < 18:
                findings.append([fig["file"],"font below 18px",label["text"]])
            suffix = " Bold" if label["weight"] == "bold" else ""
            font = ImageFont.truetype(f"/System/Library/Fonts/Supplemental/Arial{suffix}.ttf",label["size"])
            left,top,right,bottom = font.getbbox(label["text"],anchor="ls")
            length=font.getlength(label["text"])
            offset=length/2 if label["anchor"]=="middle" else length if label["anchor"]=="end" else 0
            box=(label["x"]+left-offset,label["y"]+top,label["x"]+right-offset,label["y"]+bottom)
            label["measured_bounds"] = box
            c=label["container"]
            if c and (box[0]<c[0] or box[1]<c[1] or box[2]>c[2] or box[3]>c[3]):
                findings.append([fig["file"],"text outside assigned card",label["text"]])
            if box[0]<16 or box[1]<8 or box[2]>W-16 or box[3]>fig["height"]-8:
                findings.append([fig["file"],"text outside canvas safe area",label["text"]])
            for previous,txt in boxes:
                if max(previous[0],box[0])<min(previous[2],box[2]) and max(previous[1],box[1])<min(previous[3],box[3]):
                    findings.append([fig["file"],"text overlap",txt,label["text"]])
            for a,b in fig["connectors"]:
                if segment_hits_rect(a,b,box):
                    findings.append([fig["file"],"connector crosses text",label["text"]])
            boxes.append((box,label["text"]))
        for width,suffix in [(W,""),(900,"-900")]:
            path=previews/(Path(fig["file"]).stem+suffix+".png")
            cairosvg.svg2png(url=str(OUT/fig["file"]),output_width=width,write_to=str(path))
            assets.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        path=OUT/fig["file"]
        assets.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT/"proposal-figure-layout.json").write_text(json.dumps(records,indent=2)+"\n")
    report=dict(figure_count=len(records),preview_count=2*len(records),findings=findings,assets=assets,
                checks=["SVG XML parsing","18px minimum font","measured text bounds","card containment",
                        "text-to-text intersections","orthogonal connector/text intersections"],
                limitation="Geometry checks supplement actual image review; they do not establish scientific correctness.")
    (OUT/"proposal-layout-check.json").write_text(json.dumps(report,indent=2)+"\n")
    if findings:
        raise ValueError(findings)
    print(f"{len(records)} SVGs and {len(records)*2} previews generated; geometric checks passed.")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    records=[data_to_claims(),paired_jepa(),graph_time_mask(),response_estimand()]
    render_and_validate(records)
