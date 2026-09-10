"""Build editable SVG figures from the retained study and its implementation.

Run with the repository Python; no model is loaded or trained.  Numerical
panels are read from the completed Notebook 18 grid, never copied from prose.
The SVGs use only standard vector elements and remain editable as plain text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from html import escape
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
SUITE = HERE.parents[1]
GRID = SUITE / "artifacts/motion_structured/grids/292443b0fab5339f5da7ca566a85d6172ffc5b64abe5febf2546681a0152ff57"
INK = "#203846"
MUTED = "#546675"
LINE = "#D7E1E7"
TEAL = "#137E8A"
PURPLE = "#7857A6"
AMBER = "#A46B1C"
PALE = "#F4F7F9"


def format_r2(value, *, difference=False):
    """Round display labels only; preserve one digit for small signed effects."""
    value = float(value)
    if difference and 0 < abs(value) < 0.001:
        return f"{value:+.1g}"
    if difference and value != 0:
        return f"{value:+.3f}"
    return f"{value:.3f}"


class SVG:
    def __init__(self, width, height, title, description):
        self.parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker></defs>
<style>text{{font-family:Arial,Helvetica,sans-serif;fill:{INK}}}.muted{{fill:{MUTED}}}</style>
<rect width="{width}" height="{height}" fill="white"/>''']

    def text(self, x, y, lines, size=20, fill=INK, weight=400, anchor="start", leading=1.36):
        if isinstance(lines, str):
            lines = [lines]
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" style="fill:{fill}">')
        for i, text in enumerate(lines):
            self.parts.append(f'<tspan x="{x}" dy="{0 if not i else size*leading}">{escape(str(text))}</tspan>')
        self.parts.append("</text>")

    def rect(self, x, y, w, h, fill="white", stroke=LINE, r=16, dash=None):
        dash = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>')

    def line(self, x1, y1, x2, y2, color=LINE, width=1.5, dash=None):
        dash = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dash}/>')

    def path(self, d, color=MUTED, width=2, arrow=True, dash=None):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        dash = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{marker}{dash}/>')

    def circle(self, x, y, r, fill=TEAL, stroke="white", width=1.5):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def panel(self, x, y, w, h, tag, title, fill=PALE, stroke=LINE):
        self.rect(x, y, w, h, fill, stroke)
        self.text(x+22, y+32, tag, 15, TEAL, 700)
        self.text(x+22, y+63, title, 23, INK, 700)

    def save(self, name):
        (HERE / name).write_text("\n".join(self.parts)+"\n</svg>\n", encoding="utf-8")


def pipeline():
    s = SVG(1560, 1120, "Where anatomical laterality enters skeleton JEPA training",
        "Clip-local pose preparation retains 33 landmarks and a separate validity mask. Five source-video folds precede all model training and derived training views. A student predicts hidden teacher features with a stop-gradient EMA teacher. Completed reflection augmentation and anatomical or motion target selection are separate experiments. Explicit token reflection loss is implemented in Notebook 09, with synthetic evidence only. Frozen evaluation uses held-out sources and a signed coordinate-derived endpoint.")
    s.text(40, 45, "Where anatomical laterality enters skeleton JEPA training", 31, weight=700)
    s.text(40, 78, "Completed experiments and the explicit-reflection extension have different evidence status.", 20, MUTED)

    s.panel(40, 110, 300, 195, "01  INPUT", "Pose clips + source IDs")
    s.text(62, 203, ["625 accepted clips / 93 videos", "33 anatomical landmarks", "Coordinates + validity + time", "Five dataset condition annotations"], 18)
    s.path("M 340 208 H 367")
    s.panel(380, 110, 370, 195, "02  CLIP-LOCAL PREPARATION", "Keep anatomy and validity")
    s.text(402, 203, ["Model: short gaps, pelvis centering,", "body scaling, 64-frame resizing", "Input X: [B, 64, 33, 3]", "Validity V: [B, 64, 33]"], 18)
    s.path("M 750 208 H 777")
    s.panel(790, 110, 325, 195, "03  SOURCE SPLIT", "Five outer folds")
    s.text(812, 203, ["Keep every video’s clips together", "Four folds supply training sources", "One fold supplies held-out sources", "All derived views inherit the fold"], 17.5)
    s.panel(1155, 110, 365, 195, "SEPARATE COORDINATE LANE", "Signed movement contrast", "#FFF8ED", "#E5D2AD")
    s.text(1177, 203, ["Original-time, paired-valid motions", "No gap filling or temporal resizing", "Five bilateral pairs define y", "y enters the readout after training"], 18)
    s.text(40, 334, "Preparation occurs before fold assignment in the saved workflow; it fits no cross-clip statistics. Source separation precedes training sampling and augmentation.", 15.5, MUTED)

    s.rect(40, 359, 1480, 407, "#FBFCFD")
    s.text(62, 392, "04  TRAINING ON OUTER-TRAIN SOURCES", 16, TEAL, 700)
    s.text(62, 425, "Anatomy enters through named joints, bilateral transformations, and the choice of hidden targets.", 21)

    s.rect(65, 451, 350, 132, "#EAF5F5", "#A9CED0")
    s.text(85, 482, "Choose the experimental intervention", 19, weight=700)
    s.text(85, 511, ["00–05: reflect whole clips with p = 0.5", "12: gait-joint vs all-joint targets", "15–18: motion / connected-region targets"], 17)
    s.text(85, 607, ["Reflection: x-coordinate sign + joint swap", "Validity and selected targets stay aligned", "15–18 use no reflection augmentation"], 16.5, MUTED)
    s.path("M 415 515 H 442")

    s.rect(455, 451, 330, 132)
    s.text(476, 482, "Four frames form one joint patch", 19, weight=700)
    s.text(476, 512, ["Patches: [B, 16, 33, 12]", "Tokens: [B, 528, 96]", "Valid and hidden masks: [B, 16, 33]"], 17)
    s.text(476, 607, ["All 33 landmarks remain in the input.", "Hidden content is zeroed; grid positions", "remain. A patch needs four valid frames."], 16.5, MUTED)
    s.path("M 785 495 H 830")
    s.path("M 808 495 V 643 H 830")

    s.rect(845, 451, 270, 112, "#EAF5F5", "#A9CED0")
    s.text(866, 483, "Student encoder", 23, TEAL, 700)
    s.text(866, 514, ["Masked, geometric input view", "Gradients update its weights"], 17)
    s.rect(845, 609, 270, 112, "#F1EDF7", "#CBBEDD")
    s.text(866, 641, "Teacher encoder", 23, PURPLE, 700)
    s.text(866, 672, ["Complete permitted clip", "Stop gradient; EMA update"], 17)
    s.path("M 980 563 V 600", PURPLE, dash="5 4")
    s.text(1000, 591, "slow weight average", 14, PURPLE)

    s.path("M 1115 507 H 1142")
    s.rect(1155, 451, 335, 112)
    s.text(1176, 483, "Predictor", 23, weight=700)
    s.text(1176, 514, ["Predict features at hidden positions", "Preserve each clip’s target mask"], 17)
    s.path("M 1320 563 V 600")
    s.path("M 1115 666 H 1142")
    s.rect(1155, 609, 335, 112, "#FFF8ED", "#E5D2AD")
    s.text(1176, 641, "Masked feature objective", 21, AMBER, 700)
    s.text(1176, 672, ["Centered target cross-entropy", "+ feature-variation regularization"], 17)
    s.text(62, 704, ["Anatomical and motion masks change the self-supervised task.", "Condition labels and the signed target y never supervise the encoder."], 17, MUTED)

    s.panel(40, 794, 730, 230, "05  EXPLICIT RELATION · NOTEBOOK 09", "Extra encoder loss on original + reflected clips", "#F7F3FB", "#CBBEDD")
    s.text(62, 887, ["Z(MX) ≈ P Z(X): reflection should exchange anatomical token positions.", "The penalty compares aligned valid tokens; feature channels stay fixed.", "Its energy denominator is detached from gradients; λ = 1 in the demo.", "This adds two encoder passes with gradients and uses no signed labels."], 18)
    s.rect(62, 983, 678, 26, "#EDE5F5", "#EDE5F5", 6)
    s.text(75, 1002, "SYNTHETIC DEMONSTRATION ONLY · 16 frames / 16 channels · no real-data benefit established", 14, PURPLE, 700)

    s.panel(795, 794, 725, 230, "06  FROZEN EVALUATION · HELD-OUT VIDEOS", "Test whether learned features recover laterality")
    s.text(817, 887, ["Freeze the declared encoder; compare with its paired initial weights.", "Fit feature scaling and ridge readout using outer-train sources only.", "Use source groups for readout penalty selection; test on the held-out fold.", "Pool held-out predictions per seed; give every source equal total weight."], 18)
    s.text(817, 1000, "Five seeds repeat the same cohort. Video-bootstrap intervals condition on the fitted models.", 15, MUTED)
    s.path("M 770 815 H 782 V 779 H 837 V 541 H 845", PURPLE, 2, dash="6 5")

    s.text(40, 1061, "Shape contract: transformations preserve the model tensor axes and carry validity alongside coordinates. Interpolation and resizing change the raw trajectory.", 17, MUTED)
    s.text(40, 1088, "Initial control = untrained S-JEPA weights, with the same learned pose extractor and anatomical schema. Source folds do not establish participant separation.", 17, MUTED)
    s.save("training_pipeline.svg")


def compact_pipeline():
    # Keep the canonical command aligned with the current print-ready design.
    from make_physworld_print_figures import compact
    compact()




def skeleton(s, cx, cy, reflected=False):
    # A deliberately reduced body schematic, not a sample of the study data.
    p = {"head":(0,-96), "neck":(0,-59), "ls":(-46,-51), "rs":(43,-45),
         "le":(-62,-4), "re":(66,-3), "lw":(-85,37), "rw":(83,24),
         "lh":(-25,17), "rh":(25,17), "lk":(-38,87), "rk":(44,70),
         "la":(-69,149), "ra":(65,125)}
    if reflected:
        p = {k:(-x,y) for k,(x,y) in p.items()}
    for a,b,colour in [("head","neck",MUTED),("neck","ls",TEAL),("neck","rs",PURPLE),
       ("ls","rs",MUTED),("ls","le",TEAL),("le","lw",TEAL),("rs","re",PURPLE),("re","rw",PURPLE),
       ("ls","lh",TEAL),("rs","rh",PURPLE),("lh","rh",MUTED),("lh","lk",TEAL),("lk","la",TEAL),
       ("rh","rk",PURPLE),("rk","ra",PURPLE)]:
        if reflected:
            colour = PURPLE if colour == TEAL else TEAL if colour == PURPLE else colour
        s.line(cx+p[a][0],cy+p[a][1],cx+p[b][0],cy+p[b][1],colour,5)
    for name,(x,y) in p.items():
        colour = TEAL if name.startswith("l") else PURPLE if name.startswith("r") else MUTED
        if reflected and name not in ("head","neck"):
            colour = PURPLE if colour == TEAL else TEAL
        s.circle(cx+x,cy+y,14 if name=="head" else 6, "white",colour,3)


def reflection():
    s=SVG(1440, 795,"Anatomical reflection reverses a signed movement target",
        "A schematic body and its reflection illustrate a sign reversal in the horizontal coordinate, exchange of anatomical left/right landmark identities, and the corresponding sign reversal in a signed motion target. Equivariance of tokens is distinguished from invariance of pooled features and from an algebraically odd readout. All displayed motion values are illustrative.")
    s.text(40,45,"A known geometric transformation gives a testable prediction",30,weight=700)
    s.text(40,78,"The body may walk asymmetrically; reflecting its measurement should still exchange left and right.",20,MUTED)
    s.panel(40,112,800,478,"A  ANATOMICAL REFLECTION","Coordinate sign, joint identities, and masks transform together")
    skeleton(s,220,329)
    skeleton(s,655,329,True)
    s.text(143,305,"L",17,TEAL,700)
    s.text(285,305,"R",17,PURPLE,700)
    s.text(578,305,"L",17,TEAL,700)
    s.text(724,305,"R",17,PURPLE,700)
    s.text(220,211,"Original X",23,weight=700,anchor="middle")
    s.text(655,211,"Reflected MX",23,weight=700,anchor="middle")
    s.path("M 335 331 H 537")
    s.text(436,291,["x → −x","left ↔ right"],19,anchor="middle")
    s.text(436,375,["time order stays fixed","validity + hidden masks swap"],15.5,MUTED,anchor="middle")
    s.text(220,511,"y(X) = +0.20",24,TEAL,700,anchor="middle")
    s.text(655,511,"y(MX) = −0.20",24,PURPLE,700,anchor="middle")
    s.text(440,558,"Illustrative body and values; the implemented schema contains 33 landmarks.",16,MUTED,anchor="middle")

    s.panel(868,112,532,478,"B  WHAT THE TARGET MEASURES","Paired motion from observed coordinates")
    s.text(890,210,["For each left/right landmark pair:","uL and uR are median movement speeds","over the same valid time transitions."],19)
    s.rect(890,296,488,70,"white")
    s.text(1134,340,"pair contrast = (uL − uR) / (uL + uR + ε)",22,anchor="middle")
    s.text(890,404,["Average the five accepted pair contrasts.","Use shoulder, knee, ankle, heel, and toe pairs.","Each pair needs at least eight valid transitions.","All five pairs must be usable."],18)
    s.text(890,543,["Positive y: positive mean left-minus-right contrast.","Opposing differences between pairs can cancel."],18,MUTED)

    s.rect(40,616,435,143,"#EAF5F5","#A9CED0")
    s.text(62,650,"Token equivariance",21,TEAL,700)
    s.text(62,681,["Z(MX) ≈ P Z(X) exchanges joint positions.","Feature channels are assumed to stay fixed.", "The explicit loss has synthetic evidence only."],17)
    s.rect(502,616,435,143,"#F1EDF7","#CBBEDD")
    s.text(524,650,"Exact oddness at the readout",21,PURPLE,700)
    s.text(524,681,["g(X) = [h(X) − h(MX)] / 2", "Then g(MX) = −g(X), even before training.","Predictive benefit requires a separate test."],17)
    s.rect(964,616,436,143,"#FFF8ED","#E5D2AD")
    s.text(986,650,"Constant-feature control",21,AMBER,700)
    s.text(986,681,["Constant nonzero tokens can match perfectly.","They cannot distinguish clips, so we also", "measure feature variation and held-out R²."],17)
    s.save("reflection_and_target.svg")


def load_csv(name):
    with (GRID/name).open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def results():
    summary, per_seed = load_csv("summary.csv"), load_csv("per_seed.csv")
    intervals, diag = load_csv("paired_intervals.csv"), load_csv("predictor_diagnostics.csv")
    def row(experiment, condition, representation):
        rows=[r for r in summary if r["experiment"]==experiment and r["condition"]==condition and r["representation"]==representation and r["observation"]=="unaltered"]
        assert len(rows)==1
        r=rows[0]
        assert int(r["evaluated_seeds"])==5 and int(r["evaluated_sources_min"])==93 and int(r["evaluated_clips_min"])==625
        return r
    plot_rows=[
        ("Initial · mean", "motion","uniform","initial_online__mean", MUTED),
        ("Initial · motion summary", "motion","uniform","initial_online__mean_motion", TEAL),
        ("Teacher · motion random", "motion","uniform","pretrained_teacher__mean_motion", PURPLE),
        ("Teacher · MAMP motion", "motion","mamp_motion","pretrained_teacher__mean_motion", PURPLE),
        ("Teacher · robust motion", "motion","robust_motion","pretrained_teacher__mean_motion", PURPLE),
        ("Teacher · region random", "regions","uniform","pretrained_teacher__mean_motion", PURPLE),
        ("Teacher · connected region", "regions","connected_region","pretrained_teacher__mean_motion", PURPLE),
        ("Direct pose summary", "motion","uniform","direct_pose", MUTED)]
    plot_data=[]
    for label, experiment, condition, representation, color in plot_rows:
        r=row(experiment,condition,representation)
        seeds=sorted([r for r in per_seed if r["experiment"]==experiment and r["condition"]==condition and r["representation"]==representation and r["observation"]=="unaltered"],key=lambda r:int(r["seed"]))
        assert len(seeds)==5
        numbers=[float(r["r2"]) for r in seeds]
        assert abs(mean(numbers)-float(r["mean_r2"]))<1e-12
        plot_data.append({"label":label,"experiment":experiment,"condition":condition,"representation":representation,"mean_r2":float(r["mean_r2"]),"mean_mae":float(r["mean_mae"]),"per_seed_r2":numbers,"color":color})
    trained=[r for r in diag if r["condition"]!="initial"]
    initial=[r for r in diag if r["condition"]=="initial" and r["experiment"]=="motion"]
    better=lambda r:float(r["mismatched_target_mse"])>float(r["matched_target_mse_on_control_clips"])
    assert len(trained)==375 and sum(map(better,trained))==375
    assert len(initial)==75 and sum(map(better,initial))==33
    assert len(intervals)==3
    s=SVG(1540,1035,"Learned correspondence and laterality readout diverge",
        "Saved results from 625 clips in 93 source videos. Trained predictors consistently favor correct-clip hidden teacher targets, but every trained encoder arm has lower motion-sensitive laterality R squared than the paired initial encoder. All three alternative-mask contrasts have source-bootstrap confidence intervals crossing zero. Individual seed scores and uncertainty interpretation are shown.")
    s.text(40,45,"Learning clip correspondence does not establish useful laterality features",29,weight=700)
    s.text(40,79,"625 clips · 93 source videos · five outer folds · five seeds",20,MUTED)
    s.panel(40,111,925,548,"A  FROZEN LATERALITY READOUT","The initial encoder retains the strongest tested readout")
    x0,x1=365,850
    scale=lambda v:x0+(x1-x0)*v/.30
    for tick in [0,.05,.10,.15,.20,.25,.30]:
        x=scale(tick)
        s.line(x,205,x,579,LINE,1)
        s.text(x,608,f"{tick:.2f}",15,MUTED,anchor="middle")
    s.text((x0+x1)/2,638,"Source-balanced R² · higher is better",17,MUTED,anchor="middle")
    for i,r in enumerate(plot_data):
        y=223+i*48
        if i==1:
            s.rect(57,y-22,890,43,"#EAF5F5","#EAF5F5",5)
        s.text(63,y+6,r["label"],17.5,r["color"],700 if i==1 else 400)
        for j,v in enumerate(r["per_seed_r2"]):
            s.circle(scale(v),y+(j-2)*3.5,3.3,"white",r["color"],1.1)
        s.circle(scale(r["mean_r2"]),y,6.6,r["color"])
        s.text(930,y+6,format_r2(r["mean_r2"]),18,r["color"],700,anchor="end")
    s.panel(990,111,510,548,"B  HIDDEN-FEATURE DIAGNOSTIC","Correct-clip targets are easier after training")
    s.text(1012,209,["Count of diagnostic rows with lower error for", "the correct clip than a different source video:"],17)
    s.text(1012,284,"Initial predictor",19,MUTED)
    s.rect(1012,299,455,34,"#E2E9ED","#E2E9ED",4)
    s.rect(1012,299,455*33/75,34,MUTED,MUTED,4)
    s.text(1458,322,"33 / 75",18,INK,700,anchor="end")
    s.text(1012,380,"Trained predictors",19,TEAL)
    s.rect(1012,395,455,34,TEAL,TEAL,4)
    s.text(1458,418,"375 / 375",18,"white",700,anchor="end")
    s.text(1012,478,["Each row combines one fold, seed, arm,", "and evaluation mask. These are repeated", "checks on the same videos; the counts", "do not represent independent participants."],17,MUTED)
    s.text(1012,600,["Only the sign of the comparison is summarized.", "Teacher-feature error scales differ across arms."],16,MUTED)
    s.panel(40,684,1460,270,"C  MASK-AGAINST-RANDOM CONTRASTS","Alternative target policies do not establish a readout improvement")
    lo,hi=-.04,.025
    sx=lambda v:660+(v-lo)/(hi-lo)*760
    for tick in [-.04,-.02,0,.02]:
        s.line(sx(tick),784,sx(tick),899,MUTED if tick==0 else LINE,2 if tick==0 else 1,dash="4 4" if tick==0 else None)
        s.text(sx(tick),923,f"{tick:+.2f}" if tick else "0",16,MUTED,anchor="middle")
    labels={"mamp_motion minus uniform":"MAMP motion − motion random", "robust_motion minus uniform":"Robust motion − motion random", "connected_region minus uniform":"Connected region − region random"}
    for i,r in enumerate(intervals):
        y=800+i*42
        d,l,u=map(float,(r["difference"],r["lower_95"],r["upper_95"]))
        assert int(r["sources"])==93 and l<0<u
        s.text(62,y+6,labels[r["subtraction"]],19)
        s.text(600,y+6,format_r2(d, difference=True),18,PURPLE,700,anchor="end")
        s.line(sx(l),y,sx(u),y,PURPLE,3)
        s.line(sx(l),y-6,sx(l),y+6,PURPLE,2)
        s.line(sx(u),y-6,sx(u),y+6,PURPLE,2)
        s.circle(sx(d),y,6,PURPLE)
    s.text(40,987,"Panel A: large dots are mean seed scores; small open dots are the five seed scores, not confidence intervals. Motion summaries add variation, feature changes, and support.",16,MUTED)
    s.text(40,1014,"Panel C: 95% intervals use 2,000 paired source-video resamples, conditional on fitted models. Each mask has its own count-matched random reference; intervals do not establish equivalence.",15.5,MUTED)
    s.save("learning_results.svg")
    return {"readout_rows":plot_data,"mask_intervals":intervals,"correspondence":{"trained_positive_rows":375,"trained_rows":375,"initial_positive_rows":33,"initial_rows":75,"initial_duplicate_across_experiments_removed":True,"unit":"fold × seed × arm × evaluation mask; repeated checks on the same source videos"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-only", action="store_true",
                        help="Refresh only the results SVG; preserve the full-precision ledger and other figures.")
    args = parser.parse_args()
    if args.results_only:
        ledger = json.loads((HERE / "physworld_figure_provenance.json").read_text(encoding="utf-8"))
        numbers = results()
        if numbers != ledger["numbers"]:
            raise ValueError("Results differ from the retained full-precision ledger; inspect before publishing.")
        print("Updated learning_results.svg labels; full-precision numbers and other figures unchanged.")
        return
    pipeline()
    compact_pipeline()
    reflection()
    numbers=results()
    files=[GRID/name for name in ("summary.csv","per_seed.csv","paired_intervals.csv","predictor_diagnostics.csv")]
    code_paths=["config/protocol.json","laterality/geometry.py","laterality/model.py","laterality/data.py","laterality/splitting.py","laterality_extensions/symmetry_learning.py","laterality_extensions/motion_structured_training.py","laterality_extensions/motion_gavd.py","tutorials/research_09.py"]
    files += [SUITE/name for name in code_paths]
    provenance={"created_for":"Physical World AI paper revisions, 2026-09-09", "grid":str(GRID.relative_to(SUITE)),"artifact_sha256":{str(p.relative_to(SUITE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},"numbers":numbers,"scope_notes":["No new model training.","The pipeline documents actual clip preparation before source split; no fitted cross-source preparation statistics.","The full-scale model tensor is B×64×33×3; patch length 4; embedding width 96.","The explicit reflection loss is demonstrated on synthetic data only (16 frames, width 16).", "Targets and dataset conditions never enter the encoder objective.","Reflection schematic values are illustrative; they are not observed clip results.","Existing study files and original figures are preserved."]}
    (HERE/"physworld_figure_provenance.json").write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("Created compact and detailed training SVGs, reflection_and_target.svg, learning_results.svg, and physworld_figure_provenance.json")


if __name__=="__main__":
    main()
