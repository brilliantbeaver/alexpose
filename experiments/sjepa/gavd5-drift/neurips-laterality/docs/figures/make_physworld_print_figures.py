"""Create additional print layouts; retain every existing illustration.

All four figures are 990 SVG pixels wide with labels at least 20 pixels high.
At a 396-point manuscript inclusion width, the minimum label is 8 PDF points.
The numerical panel uses the original verified, full-precision figure ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from make_physworld_figures import SVG, skeleton, format_r2, INK, MUTED, LINE, TEAL, PURPLE, AMBER, PALE

HERE = Path(__file__).resolve().parent


def box(s, x, y, w, h, title, lines, fill="white", color=INK, stroke=LINE):
    s.rect(x, y, w, h, fill, stroke)
    s.text(x+18, y+32, title, 22, color, 700)
    s.text(x+18, y+64, lines, 20, INK, leading=1.3)


def compact():
    s=SVG(990,710,"Where laterality enters a skeleton JEPA",
        "Print layout of the source-separated training pipeline. Anatomical identity and masking enter the fixed grid. A separate explicit reflection penalty has synthetic evidence only. The teacher is an exponential average without gradients; the signed target enters only frozen readout and evaluation.")
    s.text(20,37,"Where laterality enters a skeleton JEPA",28,weight=700)
    s.text(20,72,"Encoder updates use training sources; side identity enters through joints and masks.",20,MUTED)

    def small(x,w,title,lines,fill="white",color=INK,stroke=LINE):
        s.rect(x,95,w,125,fill,stroke)
        s.text(x+14,126,title,20,color,700)
        s.text(x+14,158,lines,20,leading=1.3)
    small(20,150,"Pose + folds",["64 × 33 × 3", "Validity V", "Source folds"])
    small(210,155,"Joint tokens",["16 × 33 × 96", "L/R names", "All positions"],"#EAF5F5",TEAL,"#A9CED0")
    s.rect(405,95,155,125,"#EAF5F5","#A9CED0")
    s.text(419,126,["Student", "encoder"],20,TEAL,700,leading=1.3)
    s.text(419,193,"Masked tokens",20)
    small(600,155,"Predictor",["Hidden-token", "features"])
    small(795,175,"Joint loss",["Masked CE", "+ VICReg"],"#FFF8ED",AMBER,"#E5D2AD")
    for a,b in [(170,200),(365,395),(560,590),(755,785)]:
        s.path(f"M {a} 157 H {b}")

    # Every auxiliary path has a dedicated lane. There are no crossings, and
    # the final straight segment before every arrowhead is at least 30 pixels.
    s.rect(270,320,285,158,"#F7F3FB","#CBBEDD",16,"6 5")
    s.text(288,351,"Explicit reflection loss",22,PURPLE,700)
    s.text(288,382,["M(X) = D Xπ", "D = diag(−1, 1, 1)", "Z(MX) ≈ P Z(X)"],20,leading=1.3)
    s.text(288,460,"SYNTHETIC ONLY",20,PURPLE,700)
    s.path("M 450 320 V 230",PURPLE,dash="6 5")

    s.rect(600,320,155,142,"#F1EDF7","#CBBEDD")
    s.text(614,351,["Teacher", "encoder"],20,PURPLE,700,leading=1.3)
    s.text(614,409,["Full input", "No gradient"],20,leading=1.3)
    s.path("M 540 220 V 265 H 677 V 310",PURPLE,dash="6 5")
    s.text(576,249,"EMA",20,PURPLE,700)
    s.path("M 740 320 V 265 H 825 V 230")
    s.text(749,249,"targets",20,MUTED)

    s.rect(795,320,175,142,"#FFF8ED","#E5D2AD")
    s.text(809,351,"VICReg",20,AMBER,700)
    s.text(809,383,["Two unmasked", "student views", "12 gait joints"],20,leading=1.3)
    s.path("M 930 320 V 230",AMBER)

    # The teacher embeds the complete pose through its own patch projection.
    s.path("M 145 220 V 520 H 642 V 472")
    s.text(250,507,"complete input",20,MUTED)
    box(s,600,570,370,115,"Frozen readout + evaluation",[
        "Train-source ridge; held-out test", "Signed target y enters here"])
    s.path("M 710 462 V 560")
    s.text(734,543,"freeze",20,MUTED)
    s.path("M 95 220 V 630 H 590")
    s.text(115,610,"held-out sources: excluded from encoder updates",20,MUTED)
    s.save("training_pipeline_compact_print.svg")
    # Explicitly authorized by the latest user request: replace only the
    # canonical compact diagram with this clean, print-ready layout.
    (HERE/"training_pipeline_compact.svg").write_text(
        (HERE/"training_pipeline_compact_print.svg").read_text(encoding="utf-8"),encoding="utf-8")


def detailed():
    s=SVG(990,1320,"Detailed anatomy-aware JEPA pipeline",
        "Print supplement showing independent model and target preparation, source groups, tensor dimensions, completed target policies, student and teacher training, the synthetic explicit reflection extension, and source-separated frozen readout.")
    s.text(20,38,"Detailed anatomy-aware JEPA pipeline",28,weight=700)
    s.rect(20,67,950,228,PALE)
    s.text(40,101,"A · Prepare each clip independently",23,TEAL,700)
    s.text(40,140,"Model input",22,weight=700)
    s.text(40,176,["Short-gap filling; pelvis centering;", "body scaling; 64-frame resizing.", "X: B × 64 × 33 × 3", "Validity: B × 64 × 33"],20)
    s.line(486,125,486,277)
    s.text(514,140,"Signed movement contrast",22,AMBER,700)
    s.text(514,176,["Original timestamps and valid pairs.", "No filling or temporal resizing.", "Five bilateral pairs define target y.", "Image-derived coordinates."],20)
    box(s,20,315,950,98,"B · Fix five source-video folds",[
        "625 clips / 93 sources; every clip and derived view inherits its source fold."])
    s.rect(20,434,950,136,"#EAF5F5","#A9CED0")
    s.text(40,468,"C · Keep the full anatomical grid; vary hidden targets",23,TEAL,700)
    s.text(40,502,["Patches: B × 16 × 33 × 12 → tokens: B × 528 × 96.",
        "12: gait vs all joints. 15–18: motion or connected-region masks.",
        "All 33 joints stay in the input. Valid and hidden masks: B × 16 × 33."],20)

    s.text(20,606,"D · Train only on outer-training sources",23,TEAL,700)
    box(s,40,630,265,98,"Student encoder",["Zero hidden content"] ,"#EAF5F5",TEAL,"#A9CED0")
    box(s,360,630,250,98,"Predictor",["Predict hidden features"])
    s.path("M 305 679 H 352")
    s.rect(665,630,305,246,"#FFF8ED","#E5D2AD")
    s.text(683,663,"Training loss",22,AMBER,700)
    s.text(683,701,["Masked cross-entropy", "+ gait-pooled VICReg", "on two complete", "geometric student views."],20)
    s.text(683,842,"No signed supervision",20,AMBER,700)
    s.path("M 610 679 H 657")
    box(s,40,784,265,92,"Teacher encoder",["Complete permitted input"],"#F1EDF7",PURPLE,"#CBBEDD")
    s.path("M 173 728 V 776",PURPLE,dash="6 5")
    s.text(196,761,"EMA",20,PURPLE,700)
    s.path("M 305 830 H 657")
    s.text(335,806,"stop-gradient targets",20,MUTED)

    s.rect(20,916,950,144,"#F7F3FB","#CBBEDD",16,"6 5")
    s.text(40,950,"E · Notebook 09 adds an explicit encoder relation",23,PURPLE,700)
    s.text(40,984,["M(X)[t, j] = (−x, y, z)[t, π(j)]; compare Z(MX) with P Z(X).",
        "Aligned valid tokens; identity channel action; detached energy denominator."],20)
    s.text(40,1040,"SYNTHETIC ONLY · 16 frames / 16 channels · two extra encoder passes",20,PURPLE,700)

    box(s,20,1081,950,165,"F · Freeze the encoder, then fit and evaluate the readout",[
        "Fit scaling and ridge only on training-source groups; predict held-out videos.",
        "Compare with initial S-JEPA weights using the same pose extractor and schema.",
        "Pool held-out folds per seed; weight every source video equally.",
        "Video-bootstrap intervals condition on fitted models; seeds reuse the cohort."])
    s.text(20,1282,["Preparation precedes fold-file creation; training sampling follows source grouping.",
        "Tensor axes and joint identities persist; interpolation and resizing change coordinates."],20,MUTED)
    s.save("training_pipeline_print.svg")


def reflection():
    s=SVG(990,790,"Anatomical reflection reverses a signed contrast",
        "Print schematic with original and mirrored anatomy, dimensionless mean paired-speed target, token equivariance, algebraic output oddness, and a constant-feature control. Drawings and target values are illustrative.")
    s.text(20,37,"Anatomical reflection reverses a signed contrast",28,weight=700)
    s.text(20,62,"Back-view schematic; illustrative coordinates and target values.",20,MUTED)
    s.rect(20,68,950,370,PALE)
    s.text(230,107,"Original X",23,weight=700,anchor="middle")
    s.text(745,107,"Reflected MX",23,weight=700,anchor="middle")
    skeleton(s,230,228)
    skeleton(s,745,228,True)
    s.text(150,208,"L",20,TEAL,700)
    s.text(295,208,"R",20,PURPLE,700)
    s.text(665,208,"L",20,TEAL,700)
    s.text(810,208,"R",20,PURPLE,700)
    s.path("M 365 229 H 603")
    s.text(490,175,["x → −x", "left ↔ right"],22,anchor="middle")
    s.text(490,273,["Time order stays fixed.", "Validity and masks swap."],20,MUTED,anchor="middle")
    s.text(230,414,"y(X) = +0.20",24,TEAL,700,anchor="middle")
    s.text(745,414,"y(MX) = −0.20",24,PURPLE,700,anchor="middle")
    s.rect(20,456,950,138,"white")
    s.text(40,488,"Target: average five paired speed contrasts",23,weight=700)
    s.text(40,524,"Each pair: c = (mL − mR) / (mL + mR + ε)",23)
    s.text(40,555,"Use median speeds over at least eight common valid transitions per pair.",20)
    s.text(40,582,"Positive y means a positive average left-minus-right speed contrast.",20,MUTED)
    box(s,20,615,298,154,"Token equivariance",[
        "Z(MX) ≈ P Z(X)","Exchange joint positions;", "keep channel axes."],"#EAF5F5",TEAL,"#A9CED0")
    box(s,346,615,298,154,"Oddness by algebra",[
        "g = [h(X) − h(MX)] / 2", "Then g(MX) = −g(X).", "Test learning separately."],"#F1EDF7",PURPLE,"#CBBEDD")
    box(s,672,615,298,154,"Constant-feature test",[
        "Perfect agreement can", "lose between-clip signal.", "Check useful prediction."],"#FFF8ED",AMBER,"#E5D2AD")
    s.save("reflection_and_target_print.svg")


def results(numbers):
    rows=numbers["readout_rows"]
    intervals=numbers["mask_intervals"]
    s=SVG(990,960,"Learned correspondence and laterality readout diverge",
        "Print results: per-seed frozen laterality readout, positive correct-clip correspondence checks, and the three original source-bootstrap mask contrasts. Direct pose is shown once because it is independent of encoder training seed.")
    s.text(20,36,"Learned correspondence and laterality readout diverge",27,weight=700)
    s.rect(20,65,950,421,PALE)
    s.text(40,100,"A · Frozen laterality readout",23,TEAL,700)
    scale=lambda v:360+450*v/.30
    for t in [0,.05,.10,.15,.20,.25,.30]:
        s.line(scale(t),127,scale(t),420,LINE,1)
        s.text(scale(t),448,f"{t:.2f}",20,MUTED,anchor="middle")
    for i,r in enumerate(rows):
        y=147+38*i
        if i==1:
            s.rect(31,y-23,928,40,"#EAF5F5","#EAF5F5",5)
        s.text(40,y+6,r["label"],20,r["color"],700 if i==1 else 400)
        if r["representation"]!="direct_pose":
            for j,v in enumerate(r["per_seed_r2"]):
                s.circle(scale(v),y+(j-2)*3,3.5,"white",r["color"],1.25)
        s.circle(scale(r["mean_r2"]),y,6,r["color"])
        s.text(940,y+6,format_r2(r["mean_r2"]),20,r["color"],700,anchor="end")
    s.text(40,478,"Large dots: mean R². Open dots: five seed fits. Direct pose is seed-independent.",20,MUTED)

    s.rect(20,503,950,164,PALE)
    s.text(40,538,"B · Correct-clip targets favored over other-source targets",23,TEAL,700)
    for y,label,success,n,colour in [(579,"Initial predictor",33,75,MUTED),(622,"Trained predictors",375,375,TEAL)]:
        s.text(40,y+5,label,21,colour)
        s.rect(360,y-18,450,28,"#E2E9ED","#E2E9ED",4)
        s.rect(360,y-18,450*success/n,28,colour,colour,4)
        s.text(940,y+5,f"{success} / {n}",21,colour,700,anchor="end")
    s.text(40,653,"These are repeated diagnostic rows on the same 93 source videos.",20,MUTED)

    s.rect(20,685,950,255,PALE)
    s.text(40,720,"C · Each alternative minus its matched random mask",23,TEAL,700)
    sx=lambda v:560+(v+.04)/.065*380
    for t in [-.04,-.02,0,.02]:
        s.line(sx(t),745,sx(t),863,MUTED if t==0 else LINE,2 if t==0 else 1,dash="5 4" if t==0 else None)
        s.text(sx(t),890,f"{t:+.2f}" if t else "0",20,MUTED,anchor="middle")
    labels={"mamp_motion minus uniform":"MAMP motion", "robust_motion minus uniform":"Robust motion", "connected_region minus uniform":"Connected region"}
    for i,r in enumerate(intervals):
        y=759+46*i
        d,l,u=map(float,(r["difference"],r["lower_95"],r["upper_95"]))
        s.text(40,y+6,labels[r["subtraction"]],21)
        s.text(515,y+6,format_r2(d, difference=True),21,PURPLE,700,anchor="end")
        s.line(sx(l),y,sx(u),y,PURPLE,3)
        s.line(sx(l),y-6,sx(l),y+6,PURPLE,2)
        s.line(sx(u),y-6,sx(u),y+6,PURPLE,2)
        s.circle(sx(d),y,6,PURPLE)
    s.text(40,926,"ΔR² and 95% source intervals; 2,000 resamples conditional on fitted models.",20,MUTED)
    s.save("learning_results_print.svg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-only", action="store_true",
                        help="Refresh only the print results SVG without rebuilding the pipeline or reflection figures.")
    args = parser.parse_args()
    if args.results_only:
        ledger = json.loads((HERE / "physworld_figure_provenance.json").read_text(encoding="utf-8"))
        results(ledger["numbers"])
        print("Updated learning_results_print.svg labels; other figures unchanged.")
        return
    original_names=("training_pipeline_compact","training_pipeline","reflection_and_target","learning_results")
    originals={f"{name}{suffix}":hashlib.sha256((HERE/f"{name}{suffix}").read_bytes()).hexdigest()
               for name in original_names if name!="training_pipeline_compact" for suffix in (".svg",".png",".pdf")}
    ledger_path=HERE/"physworld_figure_provenance.json"
    ledger=json.loads(ledger_path.read_text(encoding="utf-8"))
    compact()
    detailed()
    reflection()
    results(ledger["numbers"])
    checks={}
    for name in original_names:
        path=HERE/f"{name}_print.svg"
        root=ET.parse(path).getroot()
        sizes=[float(node.attrib["font-size"]) for node in root.iter("{http://www.w3.org/2000/svg}text")]
        assert min(sizes)>=20 and root.attrib["width"]=="990"
        checks[path.name]={"svg_viewbox":root.attrib["viewBox"],"minimum_font_px":min(sizes),"minimum_font_at_396pt_width":min(sizes)*396/990}
    for name,digest in originals.items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    (HERE/"physworld_print_figures.json").write_text(json.dumps({
        "originals_preserved":originals,"source_ledger_sha256":hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "print_checks":checks,"changes":["All labels at least 8pt when included at 396pt width.",
        "Direct-pose control plotted once because its result is seed-independent.",
        "Positive-target wording refers to the mean pair contrast, allowing pairwise cancellation.",
        "Detailed qualifications remain in manuscript captions.",
        "The canonical compact SVG is updated with explicit user authorization; other originals are untouched."]},indent=2)+"\n",encoding="utf-8")
    print("Created four *_print.svg layouts and updated the canonical compact SVG; other original assets are unchanged.")


if __name__=="__main__":
    main()
