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
FIGURE_NAMES = {1: 'observation-selection', 2: 'motion-beyond-joints', 3: 'repair-verification', 4: 'cross-activity-prediction', 5: 'uncertainty-and-evidence', 6: 'motion-ambiguity', 7: 'distillation-target-selection'}


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
    1: dict(title="Ask for evidence that will improve the answer.",
        sub="A fixed budget buys one more measurement from the already observed video.",
        steps=[("Start coarse", ["Past video and tracks", "The future stays hidden"]),
               ("Choose a query", ["Predict error reduction", "Do not inspect all crops"]),
               ("Reveal and predict", ["Process the selected crop", "Forecast actual motion"]),
               ("Reuse the selector", ["A different forecaster", "A new observation failure"])],
        foot="The contribution requires transferable evidence value beyond active acquisition and uncertainty rules.",
        exp=[("Measure opportunity", ["Natural captured motion", "Reveal all queries offline", "Label the best-query oracle"]),
             ("Match the budget", ["Confidence, flow and EDDI", "Same pixels and context", "Count all extraction cost"]),
             ("Freeze and transfer", ["New forecaster architecture", "New observation failure", "Actual future-motion error"])],
        gate="48-hour gate: useful oracle headroom, then at least 5% lower error than the strongest practical rule.",
        stop="Stop if extra evidence does not help, or the selector only learns one decoder's weaknesses."),
    2: dict(title="Joint positions do not describe every movement.",
        sub="Test natural motion before using a constructed twist as an explanation.",
        steps=[("Keep all joints", ["22 whole-body landmarks", "Use a strong history model"]),
               ("Observe the surface", ["Past flow and video tokens", "Check measurement quality"]),
               ("Keep a small state", ["Four, eight or sixteen tokens", "Compare simple compression"]),
               ("Predict what moves", ["Future surface transport", "Natural held-person motion"])],
        foot="A few useful tokens must beat ordinary flow fusion; missing information alone does not establish the method.",
        exp=[("Find natural headroom", ["Joints versus past flow", "Exact flow is a ceiling", "Estimated flow must help"]),
             ("Compress fairly", ["Same token and byte budget", "PCA, pooling, raw flow", "Count dense upstream cost"]),
             ("Test real utility", ["Unedited AMASS futures", "New camera and movement", "GAVD visible 2D follow-up"])],
        gate="48-hour gate: estimated transport reduces natural-motion forecast MSE by at least 10% beyond joints.",
        stop="Stop if longer joint history or simple flow extrapolation explains the entire gain."),
    3: dict(title="Judge a correction before applying it.",
        sub="A useful repair candidate is a prerequisite. MoMask is currently a comparison.",
        steps=[("Observe motion", ["Raw trajectory and video", "Real movement plus errors"]),
               ("Propose a change", ["A useful local correction", "Exact identity is allowed"]),
               ("Predict its benefit", ["Signed change in error", "Flow with optional JEPA"]),
               ("Preserve and repair", ["Different choices by region", "Transfer to a new refiner"])],
        foot="A gate that rejects every damaging reconstruction has preserved the input but has not repaired tracking.",
        exp=[("Check useful actions", ["Actual blocks and strengths", "Block oracle; event retention", "Clean-motion damage"]),
             ("Learn local benefit", ["Observed-input regions", "Flow-only strong baseline", "Opposite local decisions"]),
             ("Require transfer", ["Held-out repair generator", "Controlled retention test", "Natural-motion confirmation"])],
        gate="Keep the controlled target: at least 25% repair and +15 retention points at comparable achieved repair.",
        stop="Stop before training if the real correction family has little preservation-and-repair headroom."),
    4: dict(title="Can one walk help predict another activity?",
        sub="Personal memory must explain recorded futures beyond shape and current motion.",
        steps=[("Observe a walk", ["Separate support recording", "At most twenty seconds"]),
               ("Make a memory", ["Frozen JEPA and small pool", "Compare raw full-body data"]),
               ("Observe a new task", ["Same person's query prefix", "No access to query future"]),
               ("Predict that future", ["Two nonwalking activities", "Compare a matched donor"])],
        foot="Personalized generation and prediction exist. The test is actual cross-activity benefit after strong controls.",
        exp=[("Count eligible people", ["Verify support and activities", "About eleven possible test IDs", "Do not promise statistical power"]),
             ("Challenge the memory", ["Donor, shape and raw support", "Autoregressive correction", "Full-rotation information control"]),
             ("Test held people", ["Actual 0.5-second futures", "Both activity families", "Every person's effect shown"])],
        gate="48-hour gate: correct support improves both activities beyond donor and static controls.",
        stop="Stop the JEPA claim if raw support or a cheap personal autoregressive model works equally well."),
    5: dict(title="Check what added evidence really changes.",
        sub="A repeated observation is not an independent measurement. More evidence can also reveal uncertainty.",
        steps=[("Forecast a range", ["A probability distribution", "Frozen features, small head"]),
               ("Add fixed evidence", ["Duplicate, overlap or reveal", "Same extra input for all"]),
               ("Check what is new", ["Raw and residual features", "Compare simple calibration"]),
               ("Score the response", ["Proper score and coverage", "Width alone is insufficient"])],
        foot="The hypothesis is a transferable correction of evidence responses, not a new Gaussian mixture or guarantee.",
        exp=[("Establish the failure", ["Two standard fusion heads", "Unsupported confidence gain", "Untouched captured motion"]),
             ("Try simpler remedies", ["Reject exact duplicates", "Calibration and covariance", "Same observation metadata"]),
             ("Test shifted evidence", ["Unseen overlap and visibility", "Final proper forecast score", "Retain useful new information"])],
        gate="Continue only after a real fusion failure; seek 5% better final CRPS beyond the strongest remedy.",
        stop="Stop the standalone paper if ordinary fusion or calibration solves the observed problem."),
    6: dict(title="Show the alternative that changes the conclusion.",
        sub="The witness concerns declared tracks and geometric bounds, not every RGB pixel.",
        steps=[("Declare observations", ["2D tracks and camera bounds", "A precise movement question"]),
               ("Search both answers", ["Real-motion templates", "JEPA ranks starting points"]),
               ("Check independently", ["Reprojection and joint limits", "Each failure costs budget"]),
               ("Return a witness", ["Two admissible explanations", "Or no witness found"])],
        foot="Not finding another motion does not prove uniqueness. A checked pair can expose a specific ambiguity.",
        exp=[("Build known controls", ["Held real-motion templates", "Verified track ambiguity", "Nontrivial finding difference"]),
             ("Challenge the search", ["Geometry and retrieval", "Same RGB for image methods", "Same total search cost"]),
             ("Reveal more evidence", ["Recheck this particular pair", "Either explanation may fail", "No universal uniqueness claim"])],
        gate="48-hour gate: at least +15 points in verified-witness discovery beyond the strongest geometric search.",
        stop="Stop if geometry already finds the useful alternatives or the checker accepts implausible cases."),
    7: dict(title="Judge a lesson by what the student learns.",
        sub="Teacher-feature prediction is a screening measurement. Physical forecasting is the outcome.",
        steps=[("Offer teacher targets", ["Frozen video representations", "Training labels only"]),
               ("Select a lesson", ["History-accessible directions", "Include no distillation"]),
               ("Train real students", ["Same skeleton inputs", "Matched capacity and budget"]),
               ("Measure the decision", ["Physical forecast improvement", "Regret on held configurations"])],
        foot="Whitening, CCA and short student pilots are strong alternatives. Target selection itself is established.",
        exp=[("Run the small panel", ["No teacher and full targets", "Whitening and selected targets", "Actual natural-motion forecasts"]),
             ("Check the choice", ["Physical error, not latent R²", "Harmful choices stay visible", "Include decision cost"]),
             ("Hold out a family", ["New teacher or student setup", "Three paired training seeds", "Compare choice regret"])],
        gate="48-hour gate: selected targets improve physical error by at least 5% beyond no teacher and simple transforms.",
        stop="Stop the method claim if whitening or a short student pilot makes equally useful decisions."),
}


def icon(d, proposal, stage, x, y):
    if stage == 0 and proposal in (1, 2, 3, 4, 6):
        d.rect(x-103, y-102, 206, 188, "white", "#d3dfeb", 12)
        d.skeleton(x, y-2, 1.0, BLUE, variant=10 if proposal == 4 else 0)
        if proposal == 1:
            for v in (-60, 0, 60):
                d.line(x-96, y+v, x+96, y+v, "#dae5ef", 1)
        if proposal == 3:
            d.circle(x-29, y+74, 13, "none", RED)
        return
    if proposal == 1 and stage == 1:
        for j in range(3):
            xx=x-102+j*72
            d.rect(xx, y-62, 60, 120, "white", TEAL if j == 1 else "#c9d5e3", 8)
            d.line(xx+30,y-40,xx+30,y+28,TEAL if j == 1 else GRAY,4)
            d.circle(xx+30,y+28,6,TEAL if j == 1 else GRAY)
        d.text(x,y+89,"Choose before reveal",16,TEAL,"middle")
    elif proposal == 1 and stage == 3:
        d.rect(x-35,y-80,70,49,"#e6f5f0",TEAL,8)
        d.text(x,y-49,"Rule",18,TEAL,"middle",True)
        for sign,label in [(-1,"Model A"),(1,"Model B")]:
            xx=x+sign*64
            d.line(x,y-28,xx,y+14,GRAY,2,True)
            d.rect(xx-48,y+24,96,49,"white",BLUE,8)
            d.text(xx,y+55,label,17,BLUE,"middle")
    elif proposal == 2 and stage == 1:
        d.line(x-83,y,x+83,y,BLUE,7)
        d.circle(x-83,y,9,BLUE); d.circle(x+83,y,9,BLUE)
        d.circle(x,y-42,8,TEAL); d.circle(x,y+42,8,RED)
        d.path(f"M{x},{y-42} C{x+60},{y-42} {x+60},{y+42} {x},{y+42}",TEAL,3)
        d.text(x,y+91,"Endpoints can stay still",16,GRAY,"middle")
    elif (proposal == 2 and stage == 2) or (proposal == 4 and stage == 1):
        for j in range(8):
            d.rect(x-94+(j%4)*50,y-58+(j//4)*62,38,42,"#e4f4f1",TEAL,8)
        d.text(x,y+93,"Small learned memory" if proposal==4 else "Fixed token budget",16,TEAL,"middle")
    elif proposal == 3 and stage == 1:
        d.wave(x-104,y-36,208,28,BLUE,bump=True)
        d.wave(x-104,y+42,208,22,TEAL)
        d.line(x,y-3,x,y+10,GRAY,2,True)
    elif proposal == 3 and stage == 2:
        for row in range(3):
            for col in range(4):
                xx=x-82+col*51; yy=y-54+row*47
                d.line(xx,yy,xx+25,yy-9,TEAL,2,True)
        d.text(x,y+99,"Does the video agree?",16,GRAY,"middle")
    elif proposal == 3 and stage == 3:
        for dx,color in [(-58,TEAL),(58,BLUE)]:
            d.circle(x+dx,y-25,29,"white",color)
        d.text(x-58,y-17,"Keep",17,TEAL,"middle")
        d.text(x+58,y-17,"Fix",17,BLUE,"middle")
        d.line(x-58,y+9,x-58,y+61,TEAL,3,True)
        d.line(x+58,y+9,x+85,y+61,BLUE,3,True)
    elif proposal == 4 and stage == 2:
        d.skeleton(x,y-5,1.15,BLUE,variant=26)
    elif proposal == 5 and stage == 0:
        for shift,col in [(-40,BLUE),(38,TEAL)]:
            pts=[(x-110+k*2.75,y+48-96*math.exp(-((k*2.75-110-shift)/37)**2)) for k in range(81)]
            d.path("M"+" L".join(f"{a:.1f},{b:.1f}" for a,b in pts),col,3)
        d.line(x-115,y+52,x+115,y+52,GRAY,2)
        d.text(x,y+91,"Possible future values",16,GRAY,"middle")
    elif proposal == 5 and stage == 1:
        for dx,dy in [(-65,-22),(5,12)]:
            d.rect(x+dx,y+dy-66,74,109,"white",BLUE,8)
            d.line(x+dx+37,y+dy-48,x+dx+37,y+dy+23,BLUE,4)
        d.text(x,y+89,"Repeated or informative?",16,GRAY,"middle")
    elif proposal == 5 and stage == 2:
        for j,val in enumerate([76,36,58,91]):
            d.rect(x-90+j*48,y+48-val,28,val,"#cbdff0",None,4)
            d.rect(x-90+j*48,y+48-val,28,12 if j<3 else 44,TEAL,None,4)
        d.text(x,y+91,"Keep raw evidence too",16,GRAY,"middle")
    elif proposal == 5 and stage == 3:
        for j,(label,width,offset) in enumerate([("Before",126,-19),("After",146,16)]):
            yy=y-42+j*69
            d.text(x-104,yy-15,label,16,GRAY)
            d.line(x-width/2+offset,yy,x+width/2+offset,yy,BLUE if j==0 else TEAL,4)
            d.circle(x+offset,yy,7,BLUE if j==0 else TEAL)
            d.line(x+39,yy-14,x+39,yy+14,RED,2)
        d.text(x,y+96,"Check against the outcome",15,GRAY,"middle")
    elif proposal == 6 and stage == 1:
        d.skeleton(x-62,y-5,.75,BLUE,variant=0)
        d.skeleton(x+62,y-5,.75,TEAL,variant=26)
    elif proposal == 6 and stage == 2:
        for j,label in enumerate(["Tracks","Anatomy","Time"]):
            yy=y-67+j*55
            d.rect(x-94,yy,188,39,"white","#ccdce8",8)
            d.text(x-72,yy+26,label,18,NAVY)
            d.path(f"M{x+51},{yy+19} l7,7 l14,-15",TEAL,3)
    elif proposal == 6 and stage == 3:
        d.line(x,y-79,x,y+66,GRAY,2,dash=True)
        d.circle(x-67,y-5,30,"#e6eef9",BLUE)
        d.circle(x+67,y-5,30,"#e4f4f1",TEAL)
        d.text(x-67,y+3,"A",25,BLUE,"middle",True)
        d.text(x+67,y+3,"B",25,TEAL,"middle",True)
        d.text(x,y+97,"Different findings",16,GRAY,"middle")
    elif proposal == 7 and stage == 0:
        for j in range(5):
            d.rect(x-102,y-77+j*34,204,23,"#e5edf7",None,4)
            d.rect(x-102,y-77+j*34,45+j*24,23,BLUE,None,4)
    elif proposal == 7 and stage == 1:
        for j,label in enumerate(["0","1","4","16"]):
            xx=x-90+(j%2)*108; yy=y-72+(j//2)*88
            d.rect(xx,yy,73,59,"#e4f4f1" if j==0 else "white",TEAL if j==0 else BLUE,10)
            d.text(xx+36.5,yy+39,label,26,TEAL if j==0 else BLUE,"middle",True)
    elif proposal == 7 and stage == 2:
        for k in range(3):
            for j in range(3):
                d.circle(x-72+k*72,y-63+j*63,12,BLUE if k!=1 else TEAL)
                if k<2:
                    d.line(x-57+k*72,y-63+j*63,x-15+k*72,y-63+j*63,GRAY,2,True)
    else:
        d.line(x-111,y+63,x+111,y+63,"#a5b6c7",2)
        d.line(x-111,y+63,x-111,y-70,"#a5b6c7",2)
        d.path(f"M{x-103},{y+28} C{x-50},{y+27} {x-34},{y-25} {x},{y-9}",BLUE,4)
        d.path(f"M{x},{y-9} C{x+39},{y+6} {x+50},{y-51} {x+106},{y-49}",TEAL,4)
        d.line(x,y-71,x,y+62,GRAY,1,dash=True)
        d.text(x-62,y+98,"Past",16,BLUE,"middle")
        d.text(x+63,y+98,"Forecast",16,TEAL,"middle")


for i,s in SPECS.items():
    d=Diagram(s["title"],s["sub"])
    for k,(heading,body) in enumerate(s["steps"]):
        x=48+k*343
        d.rect(x,166,315,424,LIGHT)
        d.circle(x+29,196,15,BLUE)
        d.text(x+29,202,str(k+1),16,"white","middle")
        d.text(x+157.5,244,heading,23,NAVY,"middle",True)
        icon(d,i,k,x+157.5,367)
        d.lines(x+157.5,508,body,18,GRAY,"middle",32)
        if k<3 and not (i==4 and k==1):
            d.line(x+320,371,x+337,371,GRAY,2,True)
    if i==4:
        d.line(548.5,166,548.5,145,GRAY,2)
        d.line(548.5,145,1234.5,145,GRAY,2)
        d.line(1234.5,145,1234.5,162,GRAY,2,True)
        d.text(891.5,135,"Support memory",15,GRAY,"middle")
    d.footer(s["foot"])
    d.save(f"{FIGURE_NAMES[i]}-mechanism.svg")
    d=Diagram(f"Proposal {i}: the experiment that can change the decision", "Prospective tests. No positive experimental result is implied.")
    for k,(heading,body) in enumerate(s["exp"]):
        x=48+k*459
        d.rect(x,163,426,309,LIGHT)
        d.circle(x+35,200,19,BLUE)
        d.text(x+35,207,str(k+1),20,"white","middle")
        d.text(x+72,208,heading,24,NAVY,bold=True)
        for j,line in enumerate(body):
            d.circle(x+29,271+j*55,4,TEAL)
            d.text(x+46,279+j*55,line,19,GRAY)
        if k<2:
            d.line(x+431,317,x+451,317,GRAY,2,True)
    d.rect(48,505,1344,103,"#f0f5fc",BLUE)
    d.lines(72,546,d.wrap(s["gate"],1290,22),22,NAVY,step=31)
    d.footer(s["stop"])
    d.save(f"{FIGURE_NAMES[i]}-experiment.svg")

d=Diagram("Seven hypotheses. Choose one after the first evidence.","The ranking changes after notebook 06. None is an established main-track result.")
d.text(78,132,"Rank",13,GRAY,"middle")
d.text(1366,132,"Proposal ID",13,GRAY,"end")
rows=[(1,"Value of an observation","Transfer a useful selection rule to another forecaster",TEAL),
      (2,"Motion beyond joint positions","Keep predictive surface information in a small state",TEAL),
      (3,"Transferable repair judgment","Continue only after useful corrections exist",BLUE),
      (4,"Cross-activity motor memory","Test actual personal prediction beyond simple controls",BLUE),
      (6,"Checked ambiguity witnesses","Beat geometry at finding a finding-changing alternative",GRAY),
      (5,"Evidence-responsive forecasts","Require more than ordinary uncertainty calibration",GRAY),
      (7,"Useful JEPA teaching","Predict actual student benefit beyond existing criteria",GRAY)]
for rank,(i,name,description,color) in enumerate(rows,1):
    y=140+(rank-1)*73
    d.rect(48,y,1344,62,"#f1f6fa",rx=12)
    d.circle(78,y+31,19,color)
    d.text(78,y+38,str(rank),19,"white","middle")
    d.text(113,y+39,name,23,NAVY,bold=True)
    d.text(586,y+38,description,18,GRAY)
    d.text(1366,y+38,f"P{i}",18,color,"end")
d.rect(48,680,1344,56,"#e8f5f1")
d.text(72,715,"Pilot P1 and P2. Check P3's oracle cheaply. Commit the remaining week to one surviving claim.",22,TEAL)
d.text(48,774,"Research ranking, not an acceptance probability. All graphics are conceptual.",17,GRAY)
d.save("proposal-comparison.svg")

(OUT/"layout-checks.json").write_text(json.dumps(CHECKS,indent=2))
if any(x["overlaps"] for x in CHECKS):
    raise RuntimeError("Text collision or canvas overflow. Inspect layout-checks.json.")
for kind in ("mechanism","experiment"):
    sheet=Image.new("RGB",(1440,4*430),"#dce5ef")
    for i in range(1,8):
        im=Image.open(PREVIEW/f"{FIGURE_NAMES[i]}-{kind}.png").convert("RGB")
        im.thumbnail((704,391))
        sheet.paste(im,(8+((i-1)%2)*720,8+((i-1)//2)*430))
    sheet.save(PREVIEW/f"contact-{kind}.png")
print(f"Rendered {len(CHECKS)} SVG schematics and two visual-review contact sheets.")
