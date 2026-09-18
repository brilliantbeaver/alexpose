"""Build the September 13 tutorial's SVG diagrams from retained evidence.

Run from the experiment directory with:
    python scripts/scripts_make_progress_diagrams.py

The generator uses only Python's standard library. Main comparison and control
scores are read from their JSON artifacts. Notebook demonstration numbers are
transcribed from retained cells, documented in the notebook evidence review.
Teaching sketches are explicitly marked schematic.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "images" / "progress-0913"
OUT.mkdir(parents=True, exist_ok=True)
R1 = json.loads((ROOT / "artifacts/runs/r1_g1_1k_s42/results.json").read_text())
E0 = json.loads((ROOT / "artifacts/eval/g1/E0_results.json").read_text())
NAVY, TEAL, ORANGE, MUTED, LINE = "#142b45", "#087e8b", "#c66c24", "#536779", "#cfdae3"
BLUE, PALE, CREAM, WHITE = "#4264a8", "#e6f2f3", "#fff1e4", "#ffffff"


class SVG:
    def __init__(self, title: str, desc: str, height: int = 400):
        self.height = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(desc)}</desc>',
                      '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#536779"/></marker></defs>',
                      '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#142b45} .small{font-size:19px}</style>']

    def rect(self, x, y, w, h, fill=WHITE, stroke=LINE, rx=12):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

    def text(self, x, y, value, size=25, color=NAVY, weight="normal", anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(value))}</text>')

    def lines(self, x, y, values, size=24, step=34, **kw):
        for i, value in enumerate(values):
            self.text(x, y + i * step, value, size=size, **kw)

    def line(self, x1, y1, x2, y2, color=LINE, width=3, arrow=False, dash=False):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"' + (' marker-end="url(#arrow)"' if arrow else '') + (' stroke-dasharray="7 6"' if dash else '') + '/>')

    def circle(self, x, y, r, fill=TEAL, stroke=WHITE):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

    def path(self, points, color=TEAL, width=4, dash=False):
        pts = " ".join(f'{x:.2f},{y:.2f}' for x, y in points)
        self.parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"' + (' stroke-dasharray="8 6"' if dash else '') + '/>')

    def card(self, x, y, w, title, body, fill=PALE, h=180):
        self.rect(x, y, w, h, fill)
        title_size = min(27, (w - 40) / max(1, len(title) * .54))
        body_size = min(23, (w - 40) / max(1, max(map(len, body)) * .52))
        self.text(x + 20, y + 38, title, title_size, weight="bold")
        self.lines(x + 20, y + 80, body, size=body_size, step=32)

    def save(self, name):
        (OUT / name).write_text("\n".join(self.parts + ["</svg>"]))


def skeleton(s, x, y, scale=1, mirrored=False, color=TEAL, angle=0):
    points = [(0, -100), (0, -65), (-33, -60), (33, -60), (-46, -10), (60, -20), (-18, 5), (18, 5), (-46, 70), (52, 64), (-30, 130), (75, 123)]
    rad = math.radians(angle)
    pts = []
    for px, py in points:
        px = -px if mirrored else px
        pts.append((x + scale * (px * math.cos(rad) - py * math.sin(rad)), y + scale * (px * math.sin(rad) + py * math.cos(rad))))
    for a, b in [(0,1),(1,2),(1,3),(2,4),(3,5),(2,6),(3,7),(6,7),(6,8),(7,9),(8,10),(9,11)]:
        s.line(*pts[a], *pts[b], color=color, width=5)
    for xx, yy in pts[1:]:
        s.circle(xx, yy, 4.2, color)
    s.circle(*pts[0], 16 * scale, WHITE, color)


def symmetry():
    s = SVG("Symmetry is a comparison across a gait cycle", "Synthetic angle trajectories illustrate the same unequal-amplitude left and right movements before and after half-cycle phase alignment. The right amplitude stays 41 in both panels and the left stays 74. These are not participant measurements.")
    for x, title in [(45, "Compare corresponding phases"), (595, "Then inspect the difference")]:
        s.text(x, 35, title, 27, weight="bold")
        s.line(x, 305, x + 455, 305, color=MUTED)
        s.line(x, 85, x, 305, color=MUTED)
        s.text(x + 145, 343, "Gait cycle (%) →", 22)
        s.text(x, 330, "0", 19, MUTED)
        s.text(x + 455, 330, "100", 19, MUTED, anchor="end")
    pts1 = [(45 + t * 455 / 120, 195 - 74 * math.sin(t / 120 * 2 * math.pi)) for t in range(121)]
    pts2 = [(45 + t * 455 / 120, 195 - 41 * math.sin(t / 120 * 2 * math.pi + math.pi)) for t in range(121)]
    s.path(pts1, TEAL)
    s.path(pts2, ORANGE, dash=True)
    s.text(60, 76, "Left", 22, TEAL)
    s.text(145, 76, "Right: shifted by half a cycle", 22, ORANGE)
    s.path([(595 + t * 455 / 120, 195 - 74 * math.sin(t / 120 * 2 * math.pi)) for t in range(121)], TEAL)
    s.path([(595 + t * 455 / 120, 195 - 41 * math.sin(t / 120 * 2 * math.pi)) for t in range(121)], ORANGE, dash=True)
    s.text(610, 76, "Aligned left", 22, TEAL)
    s.text(790, 76, "Aligned right", 22, ORANGE)
    s.text(45, 365, "Angle (schematic units). Same unequal amplitudes in both panels; only the right-side phase is aligned.", 18, MUTED)
    s.text(45, 391, "Proposed analysis; these are synthetic curves, not measured participant results.", 18, MUTED)
    s.save("01_symmetry_cycle.svg")


def hypotheses():
    s = SVG("Question, comparison, observation, revision", "Four connected stages describe the study's reasoning: define incremental value, compare on source-held-out clips, inspect failed and successful results, then revise with controls.")
    cards = [(20, "1. Ask", ["Does learned motion", "add useful information", "beyond simple features?"]), (300, "2. Compare", ["Same source folds", "Training-only fitting", "Held-out clip scores"]), (580, "3. Inspect", ["Class-specific errors", "Pose-only controls", "Representation health"]), (860, "4. Revise", ["Test timing", "Measure symmetry", "Use new sources"])]
    for x, title, body in cards:
        s.card(x, 90, 220, title, body, h=205)
        if x < 850:
            s.line(x + 226, 190, x + 269, 190, MUTED, arrow=True)
    s.text(30, 42, "Working null: no added value from the learned representation in this comparison.", 25, weight="bold")
    s.text(30, 365, "This organizing hypothesis was not a preregistered significance test; all retained estimates are developmental.", 20, MUTED)
    s.save("02_hypothesis_loop.svg")


def notebooks():
    s = SVG("Seven notebooks and one retained reference run", "Notebook 00 inventories video, 01 extracts pose, 02 makes tokens and tests masks, 03 trains, 04 continues from weights, 05 has an unsuccessful retained visualization run, 06 evaluates and imports frozen results.")
    nodes = [(20,25,"00","Inventory",["Clips and sources"]),(295,25,"01","Pose",["Coordinates + visibility"]),(570,25,"02","Tokens",["Local joint-time units"]),(845,25,"03","Train",["800-update demo"]),(20,220,"04","Continue",["400 more updates"]),(295,220,"05","Visualize",["Retained run failed"]),(570,220,"06","Evaluate",["Demo + frozen results"]),(845,220,"R1","Reference",["5 folds × 1,000 updates"])]
    for x,y,n,title,body in nodes:
        s.rect(x,y,240,155, CREAM if n=="05" else PALE)
        s.text(x+17,y+35,n,24,ORANGE if n=="05" else TEAL,weight="bold")
        s.text(x+17,y+76,title,29,weight="bold")
        s.text(x+17,y+115,body[0],19)
    s.save("03_notebook_map.svg")


def dataset():
    s = SVG("47 usable clips from 35 source groups", "Normal-labelled clips: 19 from 16 sources. MS-labelled clips: 11 from 11 sources. PD-labelled clips: 17 from 8 sources. Source videos do not establish unique participant identity.")
    s.text(310,35,"Clips",25,TEAL,weight="bold")
    s.text(755,35,"Source groups",25,BLUE,weight="bold")
    for i,(name,clips,sources) in enumerate([("normal label",19,16),("MS label",11,11),("PD label",17,8)]):
        y=85+i*84
        s.text(30,y+30,name,27)
        s.rect(305,y,clips*17,45,TEAL,TEAL,rx=5)
        s.text(318+clips*17,y+31,clips,27,weight="bold")
        s.rect(750,y,sources*14,45,BLUE,BLUE,rx=5)
        s.text(763+sources*14,y+31,sources,27,weight="bold")
    s.text(30,375,"A source is an original video identifier. Different sources can still include the same person.",23,MUTED)
    s.save("04_dataset_counts.svg")


def pose_pipeline():
    s=SVG("From walking video to numerical pose", "Sampling creates frames; MediaPipe estimates 33 two-dimensional landmarks and visibility. Cleaning interpolates missing data before per-frame normalization. The cached coordinates are estimates rather than ground truth.")
    s.card(20,95,220,"Video",["Sample frames", "Nominal 15 fps", "Whole-frame pose"],h=190)
    s.card(300,95,220,"33 landmarks",["x: horizontal", "y: vertical", "v: visibility"],h=190)
    s.card(580,95,220,"Clean",["Trim empty ends", "Interpolate gaps", "Reject poor clips"],h=190,fill=CREAM)
    s.card(860,95,220,"Normalize",["Center pelvis", "Divide by torso", "Keep visibility"],h=190)
    for x in [246,526,806]:s.line(x,190,x+44,190,MUTED,arrow=True)
    s.text(25,40,"Every retained frame has 33 × 3 values; the third channel is visibility, not depth.",26,weight="bold")
    s.text(25,363,"Cleaning can invent intermediate coordinates. The cache lacks explicit validity and timestamp channels.",21,MUTED)
    s.save("05_pose_pipeline.svg")


def normalization():
    s=SVG("What normalization preserves", "A synthetic skeleton is centered on the pelvis and divided by a single torso length. The operation preserves within-frame 2D angles and ratios of distances; it removes image position and per-frame scale.")
    skeleton(s,165,194,.65)
    skeleton(s,545,182,.65)
    skeleton(s,925,173,.9)
    for xx, yy in [(545,185.25),(925,177.5)]:
        s.line(xx-110,yy,xx+110,yy,MUTED,width=1,dash=True)
        s.line(xx,61,xx,298,MUTED,width=1,dash=True)
    s.text(30,35,"Estimated image pose",26,weight="bold")
    s.text(407,35,"Subtract pelvis p",26,weight="bold")
    s.text(802,35,"Divide by torso s",26,weight="bold")
    s.line(275,160,390,160,MUTED,arrow=True)
    s.line(658,160,767,160,MUTED,arrow=True)
    s.text(85,329,"Coordinates q",25)
    s.text(450,329,"q − p",28,weight="bold")
    s.text(836,329,"q′ = (q − p) / s",26,weight="bold")
    s.text(25,384,"Schematic. Preserved within a frame: angles and distance ratios. Removed: root trajectory and absolute scale.",20,MUTED)
    s.save("06_normalization.svg")


def shape_contract():
    s=SVG("Preservation has a precise scope", "A table distinguishes coordinate-preserving reshaping from transformations that remove information or alter coordinates. None guarantees exact preservation of the original video's physical geometry.")
    s.text(25,32,"Operation",25,weight="bold")
    s.text(360,32,"What remains",25,weight="bold")
    s.text(740,32,"What changes",25,weight="bold")
    rows=[("Reshape into tokens","Same ordered input values","Array layout; then learned projection"),("Center + torso scale","2D angles; distance ratios","Root motion; per-frame scale"),("Rotate / scale / reflect","2D shape up to similarity","Coordinates; signed laterality"),("Interpolate / pad","Fixed input dimensions","Unobserved time points are filled"),("Encode + pool","A learned 96-value vector","Detailed coordinates and order")]
    for i,(a,b,c) in enumerate(rows):
        y=60+i*63
        s.rect(15,y,1070,56,WHITE if i%2 else "#f0f5f8",stroke="none",rx=4)
        s.text(27,y+35,a,22)
        s.text(360,y+35,b,22)
        s.text(740,y+35,c,21)
    s.save("07_shape_contract.svg")


def split():
    s=SVG("Split sources before making training windows", "For fold 0, 37 clips from 29 sources train the model. Ten clips from six separate sources form the test set. Five of these test clips come from a single PD source and remain grouped together.")
    s.card(20,55,260,"Original source",["A long recording", "may produce many", "clips and windows"],h=250)
    for i in range(5):s.rect(40+i*43,241,32,38,TEAL,TEAL,rx=4)
    s.line(294,180,374,180,MUTED,arrow=True)
    s.rect(394,25,680,145,PALE)
    s.text(415,61,"TRAIN • fold 0",26,TEAL,weight="bold")
    s.lines(415,99,["37 clips · 29 sources → training windows", "Fit encoder, feature selection, scaler and classifier"],23,33)
    s.rect(394,220,680,145,CREAM)
    s.text(415,256,"TEST • fold 0",26,ORANGE,weight="bold")
    s.lines(415,294,["10 clips · 6 sources → held-out predictions", "All five clips from one PD source stay here together"],23,33)
    s.line(406,194,1060,194,MUTED,width=2,dash=True)
    s.text(20,385,"Grouping prevents this source overlap; participant identity and recording confounding remain unresolved.",20,MUTED)
    s.save("08_source_split.svg")


def tensor():
    s=SVG("A small joint-time patch becomes one token", "A 32-frame window with 33 joints and three channels is grouped into eight blocks of four frames. Each joint-block has 12 values projected to 96 dimensions, yielding 264 tokens with joint and time position embeddings.")
    s.card(20,82,235,"One window",["32 frames", "33 joints", "3 channels: x, y, v"],h=205)
    s.card(310,82,245,"One local patch",["4 adjacent frames", "of one joint", "4 × 3 = 12 values"],h=205)
    s.card(610,82,235,"Token vector",["12 → 96 values", "Learned projection", "+ joint/time position"],h=205)
    s.card(900,82,180,"Full window",["8 × 33", "264 tokens", "264 × 96"],h=205)
    for a,b in [(266,299),(565,598),(855,890)]:s.line(a,185,b,185,MUTED,arrow=True)
    s.text(25,38,"Local order is explicit before compression into a learned representation.",27,weight="bold")
    s.text(25,368,"The 32-frame window is roughly 2 seconds at nominal 15 fps; actual cached timing is not fully verified.",21,MUTED)
    s.save("09_tensor_flow.svg")


def masking():
    s=SVG("Repair fixed hiding with varying anatomical-time masks", "Schematic mask grids compare a fixed target subset reused throughout training with masks changing between examples. The implemented repaired sampler selects connected body regions over time, near 60 percent target tokens.")
    for x,title in [(25,"Historical fixed mask"),(590,"Current varying masks")]:
        s.text(x,35,title,28,weight="bold")
        s.text(x,73,"Joint groups →",22,MUTED)
        for r in range(6):
            for c in range(9):
                target=(c>=5) if x==25 else ((c*3+r*2)%11<6)
                s.rect(x+c*45,95+r*29,38,23,ORANGE if target else TEAL,"none",rx=3)
    s.text(25,309,"Some joint identities never enter context.",23)
    s.text(590,309,"Each joint can serve as context and target.",23)
    s.rect(26,340,20,20,TEAL,TEAL,rx=2);s.text(56,358,"Visible context",21)
    s.rect(242,340,20,20,ORANGE,ORANGE,rx=2);s.text(272,358,"Hidden prediction target",21)
    s.text(26,393,"Schematic grids; actual layout is 8 time blocks × 33 joints. Repaired target fraction is about 60%.",19,MUTED)
    s.save("10_mask_repair.svg")


def positions():
    s=SVG("Give each hidden target a joint and time identity", "The historical predictor reused indistinguishable hidden mask tokens. The repaired predictor adds a learned position vector so each hidden slot specifies the joint and time block being predicted.")
    s.text(20,35,"A target needs a location in the skeleton and in time.",29,weight="bold")
    for x,title,fill,labels in [(30,"Historical hidden slots",CREAM,["mask","mask","mask"]),(580,"Repaired hidden slots",PALE,["mask + L knee, t₁","mask + R ankle, t₂","mask + hip, t₃"])]:
        s.text(x,93,title,26,weight="bold")
        for i,label in enumerate(labels):
            s.rect(x,118+i*70,445,55,fill)
            s.text(x+20,155+i*70,label,25)
    s.text(28,374,"Position vectors distinguish the prediction requests; this structural repair does not establish a score gain.",21,MUTED)
    s.save("11_target_positions.svg")


def teacher():
    s=SVG("Predict teacher features at hidden joint-time positions", "The student encodes visible tokens from an augmented window. A predictor with target position vectors predicts hidden features. A full-window teacher supplies targets and is updated by an exponential moving average of student encoder weights, without backpropagating through teacher targets.")
    s.card(15,32,210,"Student input",["Augmented window", "Visible tokens only"],h=120)
    s.card(290,32,205,"Student encoder",["Learned features"],h=120)
    s.card(565,32,225,"Predictor",["Hidden positions", "+ joint/time identity"],h=120)
    s.card(15,237,210,"Teacher input",["Full original window"],h=120)
    s.card(290,237,205,"Teacher encoder",["Slow moving weights"],h=120)
    s.card(565,237,225,"Target features",["Hidden slots only", "No target gradient"],h=120)
    for y in [90,295]:
        s.line(235,y,278,y,MUTED,arrow=True)
        s.line(505,y,553,y,MUTED,arrow=True)
    s.rect(867,113,213,164,CREAM)
    s.text(891,153,"Match features",24,weight="bold")
    s.lines(890,194,["Update student", "and predictor"],23,32)
    s.line(800,90,857,150,MUTED,arrow=True)
    s.line(800,295,857,245,MUTED,arrow=True)
    s.line(392,163,392,225,TEAL,arrow=True)
    s.text(410,193,"EMA",22,TEAL,weight="bold")
    s.text(18,395,"EMA = exponential moving average. Centering and sharpening shape the loss; collapse is checked separately.",19,MUTED)
    s.save("12_teacher_student.svg")


def sampling():
    s=SVG("Equalize expected training exposure by source", "Source-uniform sampling gives a short and a long source equal total probability even when they produce different numbers of windows. This changes training weights without generating independent observations.")
    s.card(25,35,420,"Long source",["Many clips and windows"],h=190)
    s.card(655,35,420,"Short source",["Few clips and windows"],h=190)
    for i in range(9):s.rect(49+i*39,148,30,44,TEAL,TEAL,rx=3)
    for i in range(2):s.rect(785+i*48,148,37,44,BLUE,BLUE,rx=3)
    s.text(460,126,"Equal total",24,weight="bold")
    s.text(459,161,"probability",24,weight="bold")
    s.line(232,236,462,299,MUTED,arrow=True)
    s.line(862,236,640,299,MUTED,arrow=True)
    s.rect(377,282,346,66,PALE)
    s.text(405,324,"Training minibatch",29,weight="bold")
    s.text(25,389,"The current SSL loop does not read condition labels. Labels enter later when fitting the classifier.",21,MUTED)
    s.save("13_source_sampling.svg")


def training_results():
    # Notebook 03, zero-based cell 14: retained first/last-five loss and final rank.
    s=SVG("Notebook 03's retained training diagnostics", "For the fold-zero 800-update demonstration, first-five-update mean loss is 4.990 and last-five-update mean loss is 1.112. Final effective rank is 9.5, a within-batch representation-spread diagnostic rather than a test accuracy.")
    s.text(25,35,"Notebook 03 • one training fold • 800 updates",28,weight="bold")
    for x, val, label in [(100,4.990,"First 5 updates"),(390,1.112,"Last 5 updates")]:
        h=val*43
        s.rect(x,306-h,180,h,TEAL,TEAL,rx=4)
        s.text(x+90,290-h,f"{val:.3f}",30,weight="bold",anchor="middle")
        s.text(x+90,340,label,23,anchor="middle")
    s.line(80,306,598,306,MUTED,width=2)
    s.text(190,382,"Mean training objective",22,MUTED)
    s.card(680,94,390,"Effective rank: 9.5",["Feature spread within a batch", "Batch size 32 → ceiling 31", "A training diagnostic;", "no clinical meaning assigned."],h=230)
    s.save("14_training_diagnostics.svg")


def continuation():
    # Notebook 04, zero-based cells 9 and 13: final rank and two retained probes.
    s=SVG("More training did not improve this held-out demonstration", "Notebook 04 initializes from notebook 03's 800-update weights and performs 400 updates with restarted optimizer, center and schedule. Held-out fold-zero probe macro F1 remains 0.600 before and after; final batch effective rank is 10.4.")
    s.card(30,70,400,"Saved 800-update weights",["Held-out macro F1", "0.600"],h=215)
    s.card(670,70,400,"400 further updates",["Held-out macro F1", "0.600"],h=215)
    s.line(445,165,655,165,MUTED,arrow=True)
    s.text(468,206,"Same sources",23)
    s.text(468,242,"New optimizer",23)
    s.text(30,339,"The retained continuation changed the feature spread (final rank 10.4), with no probe-score gain.",23)
    s.text(30,384,"This is a single-fold demonstration with restarted training state, not a five-fold or continuous 1,200-step run.",20,MUTED)
    s.save("15_continuation_result.svg")


def protocols():
    # Notebook 06, zero-based cell 12: separate fresh 500-update fold-zero demo.
    s=SVG("Keep three training protocols separate", "The 03 to 04 teaching sequence runs 800 then 400 updates from saved weights on fold zero. Notebook 06 separately runs a fresh 500-update model on fold zero. The retained R1 reference trains 1000 updates independently in each of five folds.")
    s.card(20,60,325,"03 → 04 demonstration",["Fold 0 only", "800 + 400 from weights", "Probe F1: 0.600 → 0.600"],h=235)
    s.card(389,60,325,"06 demonstration",["Fold 0 only; fresh model", "500 updates", "S-JEPA 0.644 · RF 0.915"],h=235)
    s.card(757,60,325,"Frozen older reference",["All five source folds", "1,000 updates per fold", "S-JEPA 0.438 · RF 0.667", "Before final centering fix"],h=235,fill=CREAM)
    s.text(25,350,"All displayed scores are macro F1, but their training budgets and evaluation coverage differ.",24,weight="bold")
    s.text(25,390,"The larger fold-0 scores are not an updated five-fold estimate or evidence that a shorter training run is better.",20,MUTED)
    s.save("16_protocol_separation.svg")


def readout():
    s=SVG("Freeze the learned representation before fitting a classifier", "All tokens from an unaugmented window are encoded by the teacher. A fixed seeded graph-time subset is mean pooled, then window vectors are averaged to one clip vector. A scaler and logistic regression fit on training clip vectors and labels; test clips only undergo transformation and prediction.")
    s.card(20,40,225,"Full test window",["No augmentation", "Teacher encoder"],h=150)
    s.card(295,40,225,"Token mean",["Fixed seeded subset", "of joint-time tokens"],h=150)
    s.card(570,40,225,"Window mean",["One 96-value", "vector per clip"],h=150)
    s.card(845,40,235,"Predict label",["Frozen train scaler", "Fitted linear probe"],h=150)
    for x in [254,529,804]:s.line(x,115,x+31,115,MUTED,arrow=True)
    s.rect(290,240,788,103,PALE)
    s.text(315,277,"Fitted using training clips only",26,TEAL,weight="bold")
    s.text(315,313,"Encoder → scaler → class-balanced logistic regression",25)
    s.text(20,383,"Averaging reduces the representation. Retention of phase timing and left–right differences has not been tested.",20,MUTED)
    s.save("17_frozen_readout.svg")


def results():
    s=SVG("Paired five-fold retained results", "Pooled out-of-fold macro F1 is 0.438 for repaired S-JEPA versus 0.667 for Random Forest; accuracy is 0.447 versus 0.660. Each of 47 clips has one held-out prediction across source-grouped folds. No confidence intervals are estimated.")
    for pi,(key,title) in enumerate([("macro_f1","Macro F1"),("accuracy","Accuracy")]):
        x=40+pi*555
        s.text(x,32,title,28,weight="bold")
        for val in [0,.25,.5,.75,1]:
            yy=310-val*240
            s.line(x+60,yy,x+465,yy,"#e0e7ed",width=1)
            s.text(x+45,yy+7,f"{val:.2f}",18,MUTED,anchor="end")
        for j,(branch,label,color) in enumerate([("sjepa_pooled","S-JEPA",TEAL),("rf_pooled","Random Forest",BLUE)]):
            v=R1[branch][key];xx=x+100+j*200
            s.rect(xx,310-v*240,135,v*240,color,color,rx=4)
            s.text(xx+67.5,295-v*240,f"{v:.3f}",28,weight="bold",anchor="middle")
            s.text(xx+67.5,343,label,22,anchor="middle")
    s.text(35,389,"Frozen reference before final centering fix · 47 clips · 35 sources · one seed · no confidence intervals",20,MUTED)
    s.save("18_paired_results.svg")


def confusions():
    s=SVG("The retained models make different class errors", "Confusion matrices have true class in rows and predicted class in columns, ordered normal, MS, PD. S-JEPA: rows 11 3 5; 0 6 5; 5 8 4. Random Forest: rows 13 1 5; 1 8 2; 5 2 10. Counts are clips, not independent people.")
    names=["normal","MS","PD"]
    for pi,(branch,title,color) in enumerate([("sjepa_pooled","S-JEPA",TEAL),("rf_pooled","Random Forest",BLUE)]):
        x=15+pi*555
        s.text(x+200,32,title,29,weight="bold",anchor="middle")
        s.text(x+255,68,"Predicted label →",21,MUTED,anchor="middle")
        cm=R1[branch]["confusion"]
        for j,label in enumerate(names):s.text(x+145+j*100,104,label,23,anchor="middle")
        for i,label in enumerate(names):
            s.text(x+80,151+i*73,label,23,anchor="end")
            for j,val in enumerate(cm[i]):
                fill=PALE if i==j else (CREAM if val>=5 else "#f1f5f8")
                s.rect(x+98+j*100,117+i*73,94,67,fill,"none",rx=5)
                s.text(x+145+j*100,160+i*73,val,30,weight="bold",anchor="middle")
        s.text(x+200,363,f'PD recall: {cm[2][2]}/17 = {R1[branch]["pd_recall"]:.3f}',25,weight="bold",anchor="middle")
    s.text(20,395,"Rows are true labels. S-JEPA sends 8 PD clips to MS; the Random Forest sends 5 PD clips to normal.",20,MUTED)
    s.save("19_confusion_matrices.svg")


def controls():
    s=SVG("Simple controls also predict the dataset labels", "Pooled macro F1 from five source-grouped folds for both logistic regression and Random Forest control models. Controls use mean-and-spread pose, mean pose, visibility, retained-frame count, and median-pose pixel distances. Both classifiers are retained without selecting a winner. Mean/std pose logistic regression reaches 0.703, and visibility-only Random Forest reaches 0.636. These descriptive controls do not establish a causal explanation.")
    labels=[("mean_std_pose","Pose mean + spread"),("mean_pose","Pose mean"),("visibility_only","Visibility only"),("duration_acq","Retained-frame count"),("body_proportion","Median-pose pixel distances")]
    x0,w=400,600
    for v in [0,.2,.4,.6,.8]:
        x=x0+w*v/.8;s.line(x,50,x,329,LINE,width=1);s.text(x,356,f"{v:.1f}",20,MUTED,anchor="middle")
    for i,(key,label) in enumerate(labels):
        y=73+i*54
        s.text(20,y+9,label,22)
        a=E0["shortcut_controls"][key]["logreg"]["pooled_macro_f1"]
        b=E0["shortcut_controls"][key]["rf"]["pooled_macro_f1"]
        xa=x0+w*a/.8;xb=x0+w*b/.8
        s.line(xa,y,xb,y,MUTED,width=2)
        s.circle(xa,y,8,TEAL)
        s.rect(xb-7,y-7,14,14,BLUE,BLUE,rx=1)
    s.circle(32,388,7,TEAL);s.text(47,395,"Logistic regression",20)
    s.rect(251,381,14,14,BLUE,BLUE,rx=1);s.text(278,395,"Random Forest",20)
    s.text(600,394,"Pooled macro F1 →",22,MUTED)
    s.save("20_shortcut_controls.svg")


def evaluation_units():
    s=SVG("Sampling and inference use different units", "One source can contribute many windows and several clips. Training samples sources uniformly in expectation, but reported scores count clips. More windows do not create new participants, and five overlapping training folds are not five independent experiments.")
    s.card(25,40,285,"Training exposure",["Source-uniform sampling", "Limits domination by", "long source recordings"],h=230)
    s.card(408,40,285,"Reported scores",["One prediction per clip", "47 clips in total", "Sources have unequal", "numbers of clips"],h=230,fill=CREAM)
    s.card(791,40,285,"Future inference",["Verify participants", "Bootstrap source groups", "or participant groups", "Use a fresh test cohort"],h=230)
    s.text(25,319,"Five-fold grouping protects the source boundary.",24,weight="bold")
    s.text(25,351,"A confidence interval requires an explicit uncertainty calculation.",24,weight="bold")
    s.text(25,385,"Current estimates reuse an inspected development collection, so they cannot support confirmatory clinical claims.",20,MUTED)
    s.save("21_evaluation_units.svg")


def symmetry_tests():
    s=SVG("The next tests should measure geometry and time directly", "Three proposed experiments test mirror consistency, time order, and aligned left-right angle differences. Each needs training-only model selection, unchanged source or participant test boundaries, and explicit nuisance controls. None has a retained result in the current notebooks.")
    s.card(20,30,325,"Reflection test",["Mirror x and relabel joints", "Compare model features", "under the defined mapping"],h=240)
    skeleton(s,91,221,.21,color=TEAL)
    skeleton(s,263,221,.21,mirrored=True,color=ORANGE)
    s.line(126,228,226,228,MUTED,arrow=True)
    s.card(389,30,325,"Time-order control",["Keep each pose intact", "Shuffle frame order", "Measure the score change"],h=240)
    for i,n in enumerate(["3","1","4","2"]):
        s.rect(414+i*70,208,48,42,WHITE,LINE,rx=3);s.text(438+i*70,237,n,24,anchor="middle")
    s.card(757,30,325,"Symmetry measure",["Align gait-cycle phase", "Compare left/right angles", "Control camera viewpoint"],h=240)
    s.path([(779+i*2.4,225-18*math.sin(i/100*math.pi*3)) for i in range(101)],TEAL)
    s.path([(779+i*2.4,225-10*math.sin(i/100*math.pi*3)) for i in range(101)],ORANGE,dash=True)
    s.text(25,314,"Proposed experiments: no measured equivariance, temporal ablation",23,weight="bold")
    s.text(25,345,"or phase-aligned asymmetry result is available yet.",23,weight="bold")
    s.text(25,385,"Schematic. Predefine the comparison and fit only on training sources before measuring the held-out result.",20,MUTED)
    s.save("22_proposed_symmetry_tests.svg")


def next_steps():
    s=SVG("What follows from the observed limitations", "The next sequence first verifies pose quality and feature geometry, then tests motion and symmetry against controls, then evaluates on verified participants and fresh sources with group-aware uncertainty.")
    s.card(20,75,325,"1. Verify the input",["Review pose overlays", "Keep time / validity masks", "Check ankle-angle features", "Complete notebook 05"],h=235)
    s.card(389,75,325,"2. Test the mechanism",["Temporal-shuffle control", "Reflection comparison", "Phase-aligned asymmetry", "Matched acquisition cues"],h=235)
    s.card(757,75,325,"3. Strengthen inference",["Verify participant IDs", "Lock model decisions", "Evaluate fresh sources", "Report grouped uncertainty"],h=235)
    s.text(25,37,"Each next experiment resolves a specific ambiguity in the current evidence.",27,weight="bold")
    s.text(25,385,"The present study supports an inspectable development workflow; additional data and tests are still needed.",20,MUTED)
    s.save("23_next_experiments.svg")


def angle_features():
    s=SVG("What the classical baseline receives", "The saved Random Forest comparison uses 15 populated core angle feature slots: six means, three absolute left-right mean differences, and six ranges. An upstream code inspection finds the left ankle range assigned from the right ankle statistics, limiting bilateral interpretation.")
    s.card(20,48,250,"Angles over a clip",["Left/right hip", "Left/right knee", "Left/right ankle"],h=225)
    s.card(350,48,355,"15 populated feature slots",["6 mean angles", "3 absolute left–right differences", "6 angle ranges"],h=225)
    s.card(785,48,290,"Random Forest",["100 decision trees", "Depth at most 5", "Training-only column filter"],h=225)
    s.line(282,155,337,155,MUTED,arrow=True)
    s.line(717,155,772,155,MUTED,arrow=True)
    s.text(25,328,"No explicit cycle-phase alignment is used in these summary features.",25,weight="bold")
    s.text(25,382,"Implementation limitation: the left ankle range slot copies the right ankle range; the saved scores are unchanged.",20,MUTED)
    s.save("24_classical_features.svg")


def main():
    for fn in [symmetry,hypotheses,notebooks,dataset,pose_pipeline,normalization,shape_contract,split,tensor,masking,positions,teacher,sampling,training_results,continuation,protocols,readout,results,confusions,controls,evaluation_units,symmetry_tests,next_steps,angle_features]:
        fn()
    print(f"Wrote {len(list(OUT.glob('*.svg')))} SVG diagrams to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
