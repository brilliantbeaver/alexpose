"""Build the proposal's editable SVG diagrams and PNG review previews.

Run with the repository interpreter; CairoSVG needs the local Cairo library.
The diagrams are conceptual illustrations, not measured study results.
"""
from pathlib import Path
from html import escape
import json
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "images"
INK, MUTED, BLUE, TEAL, ORANGE = "#172b40", "#53677b", "#2862ba", "#087f79", "#b96520"
PALE, GREEN, SAND = "#edf3fc", "#eaf7f3", "#fff4e7"


class Figure:
    def __init__(self, name, title, subtitle, height):
        self.name, self.height = name, height
        self.labels = []
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>',
                      '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#53677b"/></marker></defs>',
                      f'<rect width="1200" height="{height}" fill="white"/>']
        self.text(36, 43, title, 28, weight="bold")
        self.text(36, 76, subtitle, 18, color=MUTED)

    def text(self, x, y, value, size=20, color=INK, anchor="start", weight="normal"):
        self.labels.append(dict(x=x, y=y, text=value, size=size, anchor=anchor, weight=weight))
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{escape(value)}</text>')

    def box(self, x, y, w, h, title, lines=(), fill=PALE):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="#d5dfe7"/>')
        self.text(x+w/2, y+34, title, 22, anchor="middle", weight="bold")
        for i, line in enumerate(lines):
            self.text(x+w/2, y+65+i*27, line, 18, anchor="middle", color=MUTED)

    def line(self, points, color=MUTED, arrow=True, dashed=False):
        self.parts.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"' + (' stroke-dasharray="7 5"' if dashed else '') + (' marker-end="url(#arrow)"' if arrow else '') + '/>')

    def wave(self, x, y, w, amp, variant="reference", color=TEAL, dashed=False):
        points = []
        for i in range(161):
            t = i/160
            v = math.sin(t*4*math.pi) + .22*math.sin(t*2*math.pi+.5)
            if variant == "noisy":
                v += .28*math.sin(t*39*math.pi) + .16*math.cos(t*57*math.pi)
            if variant == "flat":
                v *= .30
            points.append((x+t*w, y-amp*v))
        self.line(points, color, False, dashed)

    def save(self):
        OUT.mkdir(parents=True, exist_ok=True)
        svg = "\n".join(self.parts + ["</svg>"])
        ET.fromstring(svg)
        path = OUT / f"{self.name}.svg"
        path.write_text(svg)
        return dict(file=path.name, height=self.height, labels=self.labels)


def build():
    records = []
    f = Figure("01-preserve-motion", "Correct the observation. Preserve the movement.", "Conceptual trajectories: smoothness alone cannot establish faithful restoration.", 410)
    for x, title, note in [(36,"Imperfect observation","Estimation errors obscure motion"), (426,"Useful restoration","Reference amplitude and timing retained"), (816,"Over-smoothed output","Real movement variation is reduced")]:
        f.box(x, 110, 348, 245, title, fill=GREEN if x==426 else PALE)
        f.wave(x+24, 230, 300, 47, color=TEAL, dashed=True)
        f.wave(x+24, 230, 300, 47, "noisy" if x==36 else "flat" if x==816 else "reference", ORANGE if x!=426 else BLUE)
        # Two short lines keep the panel caption legible at embedded size.
        words = {36:["Errors obscure the trajectory"],426:["Amplitude and timing retained"],816:["Variation is reduced"]}[x]
        for j, line in enumerate(words):
            f.text(x+174, 327+j*24, line, 18, anchor="middle", color=MUTED)
    f.line([(36,384),(81,384)],TEAL,False,True)
    f.text(93,390,"Dashed: reference motion",17,color=MUTED)
    f.line([(430,384),(475,384)],BLUE,False)
    f.text(487,390,"Solid: illustrative output / observation",17,color=MUTED)
    records.append(f.save())

    f = Figure("02-paired-data", "One motion, controlled observation changes", "Keep person, motion, timestamps and camera fixed within each intervention pair.", 510)
    f.box(36,195,225,115,"Audited walking",["AMASS motion","Same physical times"])
    f.box(325,110,245,95,"Clean render",["Reference appearance"],GREEN)
    f.box(325,250,245,95,"Degraded render",["Blur or obstruction"],SAND)
    f.line([(261,232),(293,232),(293,157),(325,157)])
    f.line([(293,232),(293,297),(325,297)])
    f.box(640,175,225,115,"Pose estimator",["Coordinates + scores","Observed / missing"])
    f.line([(570,157),(605,157),(605,211),(640,211)])
    f.line([(570,297),(605,297),(605,258),(640,258)])
    f.box(930,175,234,115,"Estimated tracks",["Restoration inputs","No reference labels"])
    f.line([(865,232),(930,232)])
    f.box(325,390,540,90,"Same-view projected joints",["Privileged targets for training and scoring"],GREEN)
    f.line([(148,310),(148,435),(325,435)])
    f.text(1047,409,"Held combination:",18,anchor="middle",weight="bold")
    f.text(1047,438,"blur + obstruction",18,anchor="middle",color=MUTED)
    f.text(1047,467,"on development people",17,anchor="middle",color=MUTED)
    records.append(f.save())

    f = Figure("03-paired-jepa", "Paired JEPA: learn features, then restore coordinates", "The clean teacher is available during training only; the deployed model receives observed tracks.", 590)
    f.text(36,119,"A  REPRESENTATION TRAINING",17,color=BLUE,weight="bold")
    f.box(36,145,230,96,"Imperfect tracks",["Coordinates + support"])
    f.box(340,145,240,96,"Online encoder",["Learned temporal tokens"])
    f.box(650,145,230,96,"Predictor",["Masked target features"])
    f.box(950,222,214,105,"Feature loss",["Compare valid targets","Update online branch"],SAND)
    f.line([(266,193),(340,193)])
    f.line([(580,193),(650,193)])
    f.line([(880,193),(915,193),(915,250),(950,250)])
    f.box(36,310,230,96,"Clean projections",["Same view and times"],GREEN)
    f.box(340,310,240,96,"EMA teacher",["No target gradients"],GREEN)
    f.box(650,310,230,96,"Target features",["Clean motion encoding"],GREEN)
    f.line([(266,358),(340,358)])
    f.line([(580,358),(650,358)])
    f.line([(880,358),(915,358),(915,300),(950,300)])
    f.line([(460,241),(460,310)],dashed=True)
    f.text(483,280,"EMA weights",16,color=MUTED)
    f.text(36,454,"B  FROZEN ENCODER + TRAINED COORDINATE READOUT",17,color=BLUE,weight="bold")
    for x, title, subtitle in [(36,"Observed tracks","Deployment input"),(340,"Frozen encoder","Time and joint tokens"),(650,"Temporal readout","Trained on synthetic pairs"),(950,"Restored tracks","2D coordinates")]:
        f.box(x,478,240 if x==340 else 230 if x!=950 else 214,90,title,[subtitle],GREEN if x==950 else PALE)
    for a,b in [(266,340),(580,650),(880,950)]: f.line([(a,523),(b,523)])
    records.append(f.save())

    f = Figure("04-fair-comparison", "Separate the value of clean targets from the value of JEPA", "Compare restored coordinates, not losses measured in different learned feature spaces.", 505)
    f.box(36,125,270,245,"Matched training data",["Same training people","Same observed clips","Clean labels: training only"])
    f.box(390,110,360,92,"Coordinate pretraining",["Freeze; fit matched readout design"])
    f.box(390,230,360,92,"Paired JEPA",["Freeze; fit matched readout design"],GREEN)
    f.box(390,350,360,92,"Practical denoisers",["Direct model + SmoothNet-style MLP"])
    f.line([(306,248),(348,248),(348,156),(390,156)])
    f.line([(348,248),(348,276),(390,276)])
    f.line([(348,276),(348,396),(390,396)])
    f.box(885,198,279,150,"Common evaluation",["Coordinate accuracy","Motion preservation","Clean-input retention"],GREEN)
    f.line([(750,156),(813,156),(813,233),(885,233)])
    f.line([(750,276),(885,276)])
    f.line([(750,396),(813,396),(813,311),(885,311)])
    f.box(885,375,279,85,"Scoring references",["Evaluation only"],GREEN)
    f.line([(1024,375),(1024,348)])
    f.text(36,484,"Objective test: match data and steps. Practical test: match total compute, including readout training.",19,color=MUTED)
    records.append(f.save())

    f = Figure("05-evidence-workflow", "Advance on evidence; retain useful negative results", "Planned stages: a successful software run does not establish a scientific benefit.", 590)
    f.box(36,130,230,116,"Validate pairs",["Anatomy, time, splits","Input / target isolation"])
    f.box(335,130,250,116,"Compare methods",["Strong direct baselines","Matched JEPA test"])
    f.box(655,130,240,116,"Repeat finalists",["Three training seeds","Held people / extractor"])
    f.box(965,130,199,116,"Real-data check",["Development labels","Check preservation"])
    for a,b in [(266,335),(585,655),(895,965)]: f.line([(a,188),(b,188)])
    f.box(36,332,360,158,"JEPA does not add value?",["Stop JEPA-specific expansion.","A qualifying direct denoiser","can still advance."],SAND)
    f.line([(460,246),(460,290),(216,290),(216,332)])
    f.box(480,366,280,100,"Freeze the protocol",["Method, metrics and margins"],PALE)
    f.box(850,366,314,100,"Independent confirmation",["Previously unexamined recordings"],GREEN)
    f.line([(1064,246),(1064,303),(620,303),(620,366)])
    f.line([(760,416),(850,416)])
    f.text(822,540,"Evidence missing? Pause that claim and report what remains unresolved.",18,anchor="middle",color=MUTED)
    records.append(f.save())
    (OUT / "figure-layout.json").write_text(json.dumps(records,indent=2)+"\n")
    return records


def validate_and_render(records):
    import cairosvg
    from PIL import ImageFont
    previews = OUT / "previews"
    previews.mkdir(exist_ok=True)
    findings = []
    for fig in records:
        boxes = []
        for label in fig["labels"]:
            suffix = " Bold" if label["weight"] == "bold" else ""
            font = ImageFont.truetype(f"/System/Library/Fonts/Supplemental/Arial{suffix}.ttf",label["size"])
            left,top,right,bottom = font.getbbox(label["text"],anchor="ls")
            width = font.getlength(label["text"])
            offset = width/2 if label["anchor"]=="middle" else 0
            box = (label["x"]+left-offset,label["y"]+top,label["x"]+right-offset,label["y"]+bottom)
            if box[0]<0 or box[1]<0 or box[2]>1200 or box[3]>fig["height"]:
                findings.append((fig["file"],"text outside canvas",label["text"]))
            for prev, text in boxes:
                if max(prev[0],box[0]) < min(prev[2],box[2]) and max(prev[1],box[1]) < min(prev[3],box[3]):
                    findings.append((fig["file"],"overlapping text",text,label["text"]))
            boxes.append((box,label["text"]))
        cairosvg.svg2png(url=str(OUT/fig["file"]),write_to=str(previews/fig["file"].replace(".svg",".png")))
    (OUT / "layout-check.json").write_text(json.dumps({"text_bounds_findings":findings,"note":"Text bounds only; arrows, semantics and overall clarity also require visual review."},indent=2)+"\n")
    if findings:
        raise ValueError(findings)
    print(f"Generated and rendered {len(records)} SVGs; no text collisions or canvas overflows.")


def build_html():
    import mistune
    content = mistune.html((ROOT / "README.md").read_text())
    css = """
    :root { color-scheme: light; }
    body { margin: 0; background: #f1f5f8; color: #172b40;
           font: 17px/1.65 Arial, sans-serif; }
    main { max-width: 1000px; margin: 32px auto; padding: 44px 52px;
           background: white; border: 1px solid #d5dfe7; border-radius: 16px; }
    h1 { font-size: 36px; line-height: 1.2; letter-spacing: -.7px; }
    h2 { font-size: 25px; margin-top: 36px; }
    a { color: #2862ba; text-underline-offset: 3px; }
    img { display: block; width: 100%; height: auto; margin: 28px 0 12px; }
    p:has(img) + p { font-size: 14px; color: #53677b; margin-top: 0; }
    @media (max-width: 650px) { main { margin: 0; padding: 24px 18px; border: 0; }
                             h1 { font-size: 29px; } }
    @media print { body { background: white; font-size: 11pt; }
                   main { margin: 0; padding: 0; border: 0; }
                   img { break-inside: avoid; } h2 { break-after: avoid; } }
    """
    (ROOT / "proposal.html").write_text('<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Correct pose errors without erasing gait</title><style>'+css+'</style></head><body><main>'+content+'</main></body></html>\n')


if __name__ == "__main__":
    validate_and_render(build())
    build_html()
