"""Render original SVG proposal diagrams and check text geometry.

Run with the repository Python. Set DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
on macOS when Cairo is installed through Homebrew. All figures are schematics.
"""
from pathlib import Path
from html import escape
import json
import math
import textwrap
import xml.etree.ElementTree as ET

import cairosvg
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
OUT = ROOT
PREVIEW = OUT / "previews"
PREVIEW.mkdir(parents=True, exist_ok=True)
FONT_PATH = font_manager.findfont("DejaVu Sans")
NAVY, BLUE, TEAL, RED, GRAY = "#182b43", "#2767b2", "#087f78", "#ba4256", "#64748b"
LIGHT, GOLD = "#eef4fa", "#9c6800"
W, H = 1440, 800
CHECKS = []
FIGURE_NAMES = {1: 'motion-preservation', 2: 'motion-ambiguity', 3: 'distillation-value', 4: 'probabilistic-forecasting', 5: 'response-distillation', 6: 'cross-activity-prediction', 7: 'motion-beyond-joints'}


class Diagram:
    def __init__(self, title, subtitle):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title>',
                      f'<desc id="desc">{escape(subtitle)} Design schematic. No experimental outcomes are shown.</desc>',
                      '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9" fill="#64748b"/></marker></defs>',
                      f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
        self.text_boxes = []
        self.text(48, 63, title, 34, NAVY, bold=True)
        self.text(48, 104, subtitle, 20, GRAY)

    def rect(self, x, y, w, h, fill=LIGHT, stroke=None, rx=20):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"'+(f' stroke="{stroke}" stroke-width="2"' if stroke else '')+'/>' )

    def line(self, x1, y1, x2, y2, color=GRAY, width=3, arrow=False, dash=False):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-linecap="round"'+(' marker-end="url(#arrow)"' if arrow else '')+(' stroke-dasharray="7 7"' if dash else '')+'/>' )

    def path(self, data, color=BLUE, width=3, fill="none"):
        self.parts.append(f'<path d="{data}" stroke="{color}" stroke-width="{width}" fill="{fill}" stroke-linecap="round" stroke-linejoin="round"/>')

    def circle(self, x, y, r, fill, stroke=None):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"'+(f' stroke="{stroke}" stroke-width="2"' if stroke else '')+'/>' )

    def text(self, x, y, value, size=22, color=NAVY, anchor="start", bold=False):
        # Baseline-aware measured box; the SVG explicitly uses the same font family.
        font = ImageFont.truetype(FONT_PATH, size)
        box = font.getbbox(value, anchor="ls")
        width = font.getlength(value)
        left = x - (width / 2 if anchor == "middle" else width if anchor == "end" else 0)
        self.text_boxes.append((left, y+box[1], left+width, y+box[3], value))
        self.parts.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')

    def lines(self, x, y, lines, size=21, color=GRAY, anchor="start", step=32):
        for i, value in enumerate(lines):
            self.text(x, y+i*step, value, size, color, anchor)

    def wrap(self, value, max_width, size=21):
        font = ImageFont.truetype(FONT_PATH, size)
        result, line = [], ""
        for word in value.split():
            candidate = f"{line} {word}".strip()
            if line and font.getlength(candidate) > max_width:
                result.append(line)
                line = word
            else:
                line = candidate
        if line:
            result.append(line)
        return result

    def skeleton(self, cx, cy, scale=1, color=BLUE, variant=0, arms=True):
        points = {"head": (0,-62), "neck": (0,-40), "hip": (0,5),
                  "ls":(-23,-31),"rs":(23,-31),"le":(-39,-2-variant),"re":(39,-2+variant),
                  "lw":(-45,23-variant),"rw":(45,23+variant),"lk":(-18,42),"rk":(20+variant*.4,40),
                  "lf":(-29,76),"rf":(31+variant*.7,74-variant*.4)}
        edges = [("neck","hip"),("hip","lk"),("hip","rk"),("lk","lf"),("rk","rf")]
        edges += [("neck","ls"),("neck","rs"),("ls","le"),("le","lw"),("rs","re"),("re","rw")]
        for a,b in edges:
            p,q=points[a],points[b]
            c=color if arms or a not in {"neck","ls","rs","le","re"} else "#c5d0dc"
            self.line(cx+p[0]*scale,cy+p[1]*scale,cx+q[0]*scale,cy+q[1]*scale,c,4)
        self.circle(cx,cy-62*scale,13*scale,"white",color)
        for p in ("hip","lk","rk","lf","rf"):
            x,y=points[p]
            self.circle(cx+x*scale,cy+y*scale,4*scale,color)

    def wave(self,x,y,w=210,h=30,color=BLUE,phase=0,bump=False):
        pts=[]
        for i in range(81):
            z=i/80
            yy=math.sin(2*math.pi*(z+phase))*h
            if bump:
                yy+=math.exp(-((z-.65)/.07)**2)*h
            pts.append((x+z*w,y-yy))
        self.path("M"+" L".join(f"{a:.1f},{b:.1f}" for a,b in pts),color,4)

    def footer(self, message):
        self.rect(48, 660, 1344, 78, "#edf7f5")
        self.lines(72, 692, self.wrap(message,1290,21),21,TEAL,step=28)
        self.text(48,774,"Design schematic. No experimental outcomes are shown.",17,GRAY)

    def save(self,name):
        collisions=[]
        for i,a in enumerate(self.text_boxes):
            if a[0]<0 or a[1]<0 or a[2]>W or a[3]>H:
                collisions.append(["outside",a[4]])
            for b in self.text_boxes[i+1:]:
                if min(a[2],b[2])-max(a[0],b[0])>1 and min(a[3],b[3])-max(a[1],b[1])>1:
                    collisions.append([a[4],b[4]])
        CHECKS.append({"file":name,"text_elements":len(self.text_boxes),"overlaps":collisions})
        svg="\n".join(self.parts+["</svg>"])
        ET.fromstring(svg)
        (OUT/name).write_text(svg)
        cairosvg.svg2png(bytestring=svg.encode(),write_to=str(PREVIEW/name.replace(".svg",".png")),output_width=1440,output_height=800)


SPECS = {
1: dict(title="Keep the movement. Repair the measurement.", sub="The same tracked excursion can be a real event or a detector error.",
        steps=[("Observe",["Raw pose and video","Both may be uncertain"]),("Propose a repair",["Frozen motion prior","No backbone training"]),("Check pixel motion",["Flow and video evidence","Small calibrated gate"]),("Retain or repair",["Protect real events","Abstain if unresolved"])],
        foot="Success means more real motion retained at the same tracking-error reduction, including a held-out event type.",
        exp=[("Establish erasure",["Known reference motion","True event plus tracking error","Matched observation tensors"]),("Beat simple checks",["Flow and two trackers","Same video evidence","Calibration-locked repair"]),("Test transfer",["Unseen event family","New prior and camera","Natural motion stress"])],
        gate="48-hour development gate: +15 points in retention at matched error; remove at least 25% of injected error.",
        stop="Stop if a simple evidence gate matches the result."),
2: dict(title="Show why one camera may not settle the answer.", sub="A valid alternative is evidence of ambiguity. Failed search is not evidence of uniqueness.",
        steps=[("Observe",["Partial 2D joint tracks","Declared camera bounds"]),("Search both answers",["Two whole-body motions","Opposite descriptor values"]),("Check constraints",["Reprojection and anatomy","Independent verification"]),("Show the alternative",["A checkable motion pair","Or: no witness found"])],
        foot="Reveal held-out evidence. Reject inconsistent candidates, then search again; a third explanation may remain.",
        exp=[("Construct known pairs",["Identical visible tracks","Different 3D conclusions","Independently bounded controls"]),("Match search cost",["Articulated geometry","Prior and random starts","Every candidate checked"]),("Reveal evidence",["Previously hidden view","Check this witness again","No claim of uniqueness"])],
        gate="48-hour gate: +15 points in valid-witness recovery over geometry at equal search cost.",
        stop="Stop if geometry already finds the relevant alternatives."),
3: dict(title="Judge the lesson by what the student learns.", sub="Teacher-feature prediction and actual student improvement are different measurements.",
        steps=[("Teacher targets",["State, motion and context","Frozen public encoders"]),("Student reference",["Use available inputs only","Separate state from history"]),("Choose a lesson",["Temporal target and rank","Include no distillation"]),("Test the student",["Actual future joint error","Hold out a teacher family"])],
        foot="The contribution must improve training decisions over existing transferability criteria and a short student pilot.",
        exp=[("Correct the panel",["Quality features once","Prefix-only student","Bounded target intervals"]),("Train paired students",["Same data and capacity","Whitening and CCA","Three paired seeds"]),("Judge new choices",["Held-out teacher family","Motion forecast utility","Harm and decision regret"])],
        gate="48-hour gate: a useful student gain beyond whitening and a short pilot, not just positive target R-squared.",
        stop="Stop the method claim if existing criteria choose equally well."),
4: dict(title="Teach possible futures, not only their average.", sub="A richer teacher can see evidence that a partial-observation student cannot see.",
        steps=[("Student's past",["Some body evidence hidden","No access to future frames"]),("Richer teacher",["Additional past body cues","Frozen motion prior"]),("Student distribution",["Keep unresolved branches","Calibrate the spread"]),("Score real futures",["Probability and coverage","Compare strong mixtures"])],
        foot="First show real information about future spread beyond the mean. Ordinary probabilistic distillation is a strong baseline.",
        exp=[("Find real signal",["Held-out mocap futures","Mean-matched references","Quality and order controls"]),("Test the teacher",["Calibrated future samples","Direct probabilistic head","Same outcome labels"]),("Remove extra cues",["Core11 versus whole body","Unseen observation masks","Score width and coverage"])],
        gate="48-hour gate: proper-score gain with mean-error equivalence supported by an interval on real motion.",
        stop="Stop if only a constructed branching toy shows the effect."),
5: dict(title="Transfer how a teacher responds to a change.", sub="Hold appearance fixed and compare paired changes in motion and representation.",
        steps=[("Paired motions",["One controlled change","Same render conditions"]),("Teacher response",["Difference in frozen features","No absolute feature target"]),("Student response",["Match paired differences","Keep a small adapter"]),("Test new changes",["Held-out edit families","Actual future-motion gain"])],
        foot="Response matching must beat Jacobian and relational distillation, plus direct motion supervision.",
        exp=[("Audit teacher response",["Change motion only","Change appearance only","Independent rendering checks"]),("Compare transfer",["Feature and Jacobian KD","Relational matching","Equal data and parameters"]),("Demand utility",["Unseen change family","Independent forecast task","New appearance profiles"])],
        gate="48-hour gate: motion-specific teacher response that supports actual student gains beyond direct supervision.",
        stop="Stop if responses provide no equal-budget predictive benefit beyond direct motion training."),
6: dict(title="Can a short walk help predict another activity?", sub="Test a transferable personal movement pattern, beyond body shape and the current pose.",
        steps=[("Walking memory",["From twenty seconds of walking","Person held out of adaptation"]),("Different activity",["Same query prefix for all","Only past frames visible"]),("Frozen forecaster",["Receives walking memory","And the same query prefix"]),("Measured future",["Compare actual movement","Swap a matched donor"])],
        foot="The key comparison is the same query with same-person support versus matched other-person support.",
        exp=[("Count usable people",["Walk plus other activities","Separate support trials","Known participant IDs"]),("Test raw support",["Match body shape and speed","Same query in every arm","Remove static information"]),("Test transfer",["People held out of adaptation","Held-out query activity","Actual future error"])],
        gate="48-hour gate: support contains a repeatable cross-activity signal beyond morphology and matched donors.",
        stop="Stop if support only reveals body size, posture or activity."),
7: dict(title="Keep visible motion that joint positions can miss.", sub="Test a limitation of the skeleton state, then add only the surface motion needed for a useful forecast.",
        steps=[("Same joint history",["Validate actual input equality","Match the final image too"]),("Different surface motion",["A visible pattern can rotate","Joint endpoints stay fixed"]),("Add a few probes",["4, 8 or 16 flow tokens","Keep uncertain evidence"]),("Forecast new motion",["Unedited AMASS trajectories","Model-derived surface targets"])],
        foot="Flow is additional evidence from the same pixels. It is neither independent ground truth nor a unique 3D explanation.",
        exp=[("Verify the input gap",["Whole-body joint equality","Known rendered transport","Textureless ambiguity tests"]),("Check flow guidance",["Estimated and reference flow","Unmodified frozen model","Does joint guidance erase it?"]),("Test a small state",["K = 0, 4, 8, 16 tokens","Untouched motion forecasts","Count full extraction cost"])],
        gate="48-hour gate: estimated flow exposes the gap and improves natural-motion forecasts beyond cheap baselines.",
        stop="Stop if only synthetic twists work, or ordinary fusion already gives the same useful result."),
}


def icon(d, i, k, cx, cy):
    if i==1:
        if k==0:
            d.skeleton(cx,cy,.8,BLUE,16)
            d.circle(cx+33,cy+53,12,"none",RED)
        elif k==1:
            d.text(cx-100,cy-66,"Observed",17,BLUE)
            d.wave(cx-100,cy-18,200,27,BLUE,bump=True)
            d.text(cx-100,cy+15,"Prior",17,GRAY)
            d.wave(cx-100,cy+40,200,18,GRAY)
        elif k==2:
            d.rect(cx-90,cy-60,180,135,"white",BLUE,10)
            d.skeleton(cx,cy+1,.65,TEAL,16)
            for dx,dy in [(-63,-20),(54,-20),(-57,43),(48,43)]:
                d.line(cx+dx-8,cy+dy,cx+dx+8,cy+dy-9,BLUE,2,True)
        else:
            d.text(cx-100,cy-66,"Retained",17,TEAL)
            d.wave(cx-100,cy-18,200,25,TEAL,bump=True)
            d.text(cx-100,cy+15,"Prior",17,GRAY)
            d.wave(cx-100,cy+40,200,18,GRAY)
    elif i==2:
        if k==0:
            d.rect(cx-85,cy-68,170,157,"white",GRAY,10)
            d.skeleton(cx,cy,.85,BLUE)
        elif k==1:
            d.skeleton(cx-61,cy,.72,TEAL,25)
            d.skeleton(cx+61,cy,.72,RED,-25)
        elif k==2:
            for j in range(3):
                d.circle(cx-62,cy-36+j*44,9,TEAL)
                d.line(cx-39,cy-36+j*44,cx+70,cy-36+j*44,GRAY,3)
        else:
            d.rect(cx-107,cy-52,96,110,"#e5f4f1",TEAL,12)
            d.rect(cx+11,cy-52,96,110,"#faeef0",RED,12)
            d.text(cx-59,cy+12,"A",34,TEAL,"middle",True)
            d.text(cx+59,cy+12,"B",34,RED,"middle",True)
            d.text(cx-59,cy+40,"Low bend",14,TEAL,"middle")
            d.text(cx+59,cy+40,"High bend",14,RED,"middle")
    elif i==3:
        if k==0:
            for j,(label,c,h) in enumerate([("State",GRAY,70),("Motion",TEAL,105),("Context",BLUE,85)]):
                x=cx-103+j*78
                d.rect(x,cy+45-h,48,h,c,rx=8)
                d.text(x+24,cy+76,label,15,c,"middle")
        elif k==1:
            d.skeleton(cx,cy,.8,BLUE)
            d.circle(cx,cy+4,54,"none",TEAL)
        elif k==2:
            for j,(label,c) in enumerate([("0",GRAY),("8",TEAL),("32",GRAY)]):
                x=cx-80+j*80
                d.circle(x,cy+5,32,"white",c)
                d.text(x,cy+14,label,26,c,"middle")
            d.text(cx,cy+69,"Target rank",18,GRAY,"middle")
        else:
            d.line(cx-93,cy+52,cx+97,cy+52)
            d.path(f"M{cx-90},{cy+27} Q{cx-10},{cy-49} {cx+91},{cy-26}",BLUE,5)
            d.path(f"M{cx-90},{cy+27} Q{cx-10},{cy-29} {cx+91},{cy-49}",TEAL,5)
    elif i==4:
        if k<2:
            d.skeleton(cx,cy,.9,BLUE,20,arms=k==1)
            if k==0:
                d.rect(cx-102,cy-42,44,62,"#e2e8f0",rx=8)
        elif k==2:
            for dx,c in [(-49,BLUE),(49,TEAL)]:
                pts=[]
                for n in range(61):
                    x=-110+n*220/60
                    y=70*math.exp(-((x-dx)/30)**2)
                    pts.append((cx+x,cy+46-y))
                d.path("M"+" L".join(f"{a:.1f},{b:.1f}" for a,b in pts),c,4)
            d.line(cx-110,cy+46,cx+110,cy+46)
        else:
            for j in range(3):
                d.line(cx-70,cy-33+j*44,cx+70,cy-33+j*44,BLUE,4)
                d.circle(cx-20+j*22,cy-33+j*44,8,TEAL)
    elif i==5:
        if k==0:
            d.wave(cx-103,cy-25,206,25,BLUE)
            d.wave(cx-103,cy+44,206,25,TEAL,phase=.12)
        elif k in (1,2):
            for j,c in enumerate([BLUE,TEAL]):
                d.rect(cx-98,cy-45+j*65,68,37,c,rx=6)
                d.line(cx-12,cy-26+j*65,cx+60,cy+7,GRAY,3,True)
            d.text(cx+90,cy+17,"Δ",36,TEAL,"middle")
        else:
            d.skeleton(cx,cy,.85,TEAL,20)
    elif i==6:
        if k==0:
            d.skeleton(cx,cy,.83,BLUE,16)
            d.text(cx+90,cy+15,"20 s",19,BLUE,"middle")
        elif k==1:
            d.skeleton(cx,cy,.8,TEAL,38)
            d.line(cx-91,cy+70,cx+90,cy+70,GRAY,3,False,True)
        elif k==2:
            for j in range(3):
                d.rect(cx-75+j*17,cy-60+j*23,132,70,"white",BLUE,10)
            d.circle(cx+62,cy+42,17,TEAL)
        else:
            d.wave(cx-106,cy,212,32,BLUE)
            d.wave(cx-106,cy+25,212,20,TEAL,phase=.07)
    elif i==7:
        if k==0:
            d.skeleton(cx-62,cy,.70,BLUE,0)
            d.skeleton(cx+62,cy,.70,BLUE,0)
        elif k==1:
            for dx,color,sign in [(-60,TEAL,1),(60,RED,-1)]:
                x=cx+dx
                d.rect(x-27,cy-64,54,138,"white",color,26)
                d.line(x,cy-45,x,cy+55,BLUE,5)
                d.circle(x,cy-45,6,BLUE)
                d.circle(x,cy+55,6,BLUE)
                d.circle(x+sign*17,cy-15,6,color)
                d.circle(x-sign*15,cy+33,6,color)
                d.path(f"M{x-37},{cy-5} Q{x},{cy+23} {x+37},{cy-5}",color,3)
        elif k==2:
            d.skeleton(cx,cy,.88,BLUE,0)
            for dx,dy in [(-31,-21),(31,-21),(-39,4),(39,4),(-13,24),(13,24),(-24,57),(25,57)]:
                d.circle(cx+dx,cy+dy,6,TEAL)
                d.line(cx+dx,cy+dy,cx+dx+14,cy+dy-10,TEAL,2)
        else:
            d.wave(cx-106,cy-20,212,25,BLUE)
            d.wave(cx-106,cy+35,212,22,TEAL,phase=.1)
            for dx in [-69,0,70]:
                d.circle(cx+dx,cy+35-22*math.sin(2*math.pi*((dx+106)/212+.1)),5,TEAL)


for i,s in SPECS.items():
    d=Diagram(s["title"],s["sub"])
    xs=[48,391,734,1077]
    for k,(heading,body) in enumerate(s["steps"]):
        x=xs[k]
        d.rect(x,166,315,424,LIGHT)
        d.circle(x+29,196,15,BLUE)
        d.text(x+29,202,str(k+1),16,"white","middle")
        d.text(x+157.5,244,heading,23,NAVY,"middle",True)
        icon(d,i,k,x+157.5,367)
        d.lines(x+157.5,508,body,19,GRAY,"middle",32)
        if k<3 and not (i==6 and k==0):
            d.line(x+319,371,x+339,371,GRAY,2,True)
    if i==6:
        # Walking support creates memory independently of the query activity.
        d.line(205.5,166,205.5,140,GRAY,2)
        d.line(205.5,140,891.5,140,GRAY,2)
        d.line(891.5,140,891.5,162,GRAY,2,True)
        d.text(548.5,131,"Support only",16,GRAY,"middle")
    d.footer(s["foot"])
    d.save(f"{FIGURE_NAMES[i]}-mechanism.svg")
    d=Diagram(f"Proposal {i}: the experiment that can change the decision", "A short mechanism test comes before the full one-week study.")
    for k,(heading,body) in enumerate(s["exp"]):
        x=48+k*459
        d.rect(x,163,426,309,LIGHT)
        d.circle(x+35,200,19,BLUE)
        d.text(x+35,207,str(k+1),20,"white","middle")
        d.text(x+72,208,heading,25,NAVY,bold=True)
        for j,line in enumerate(body):
            d.circle(x+32,271+j*55,4,TEAL)
            d.text(x+49,279+j*55,line,21,GRAY)
        if k<2:
            d.line(x+430,317,x+452,317,GRAY,2,True)
    d.rect(48,505,1344,103,"#f0f5fc",BLUE)
    d.lines(72,546,d.wrap(s["gate"],1290,22),22,NAVY,step=31)
    d.footer(s["stop"])
    d.save(f"{FIGURE_NAMES[i]}-experiment.svg")

d=Diagram("Seven alternatives. Select one flagship.", "Ranked by the current evidence, novelty boundary and one-week execution risk.")
ranked=[(1,"Preserve real movement","Use pixel motion to preserve events during repair",TEAL),
        (7,"Keep motion beyond keypoints","Add the surface motion a skeleton state is missing",TEAL),
        (2,"Show competing explanations","Find checked motions that reverse a 3D conclusion",BLUE),
        (6,"Cross-activity motor memory","Test whether a short walk helps forecast another action",BLUE),
        (5,"Transfer responses","Teach how features change, then measure student benefit",BLUE),
        (3,"Predict distillation value","Choose a target, a rank, or no teacher",GRAY),
        (4,"Teach uncertain futures","Test useful temporal information beyond the mean",GRAY)]
for rank,(i,name,description,color) in enumerate(ranked,1):
    y=140+(rank-1)*73
    d.rect(48,y,1344,62,"#f1f6fa",rx=12)
    d.circle(78,y+31,19,color)
    d.text(78,y+38,str(rank),19,"white","middle")
    d.text(113,y+39,name,24,NAVY,bold=True)
    d.text(604,y+38,description,21,GRAY)
    d.text(1366,y+38,f"P{i}",18,color,"end")
d.rect(48,680,1344,56,"#e8f5f1")
d.text(72,715,"Start with P1 and P7's small flow assays. Keep P6's data audit as a fallback.",23,TEAL)
d.text(48,774,"Research ranking, not an acceptance probability. No proposed result has been measured.",17,GRAY)
d.save("proposal-comparison.svg")

(OUT/"layout-checks.json").write_text(json.dumps(CHECKS,indent=2))
if any(x["overlaps"] for x in CHECKS):
    raise RuntimeError("Text overlap or viewbox overflow. Inspect layout-checks.json.")

# Contact sheets support independent human visual review; originals remain vector.
for kind in ("mechanism","experiment"):
    sheet=Image.new("RGB",(1440,4*430),"#dce5ef")
    for i in range(1,8):
        im=Image.open(PREVIEW/f"{FIGURE_NAMES[i]}-{kind}.png").convert("RGB")
        im.thumbnail((704,391))
        x=8+((i-1)%2)*720
        y=8+((i-1)//2)*430
        sheet.paste(im,(x,y))
    sheet.save(PREVIEW/f"contact-{kind}.png")
print(f"Rendered {len(CHECKS)} SVGs with zero measured text collisions; PNG previews and two contact sheets ready.")
