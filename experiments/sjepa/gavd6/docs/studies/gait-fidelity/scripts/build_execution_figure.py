"""Generate the eight-H100 planning diagram without rebuilding accepted assets.

Run from the repository root:
    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib .venv/bin/python \
        docs/studies/gait-fidelity/scripts/build_execution_figure.py

The capacity is a planning assumption, not measured throughput. Importing the
shared Figure helper does not run its main block or regenerate earlier figures.
"""
from pathlib import Path
import hashlib
import json

from build_proposal_figures import Figure, ROOT, OUT, W, PALE, GREEN, SAND, MUTED, segment_hits_rect


def build():
    f = Figure("17-parallel-execution", "Share prepared data across independent one-GPU experiments",
               "Eight available H100s provide up to eight concurrent jobs; each small model uses one GPU.", 1036)
    f.text(36,125,"1. Prepare and review shared data once",24,weight="bold")
    f.card(36,151,305,123,"Reviewed source data",
           ["People, references and splits", "Shared by matched methods"],GREEN,22)
    f.card(449,151,304,123,"Render and extract once",
           ["Per frozen dataset version", "Retain timestamps and validity"],GREEN,22)
    f.card(856,151,308,123,"Immutable shared cache",
           ["Hash inputs; record settings", "Read-only inputs for every job"],GREEN,22)
    f.line([(341,213),(449,213)])
    f.line([(753,213),(856,213)])
    f.line([(1010,274),(1010,299),(600,299),(600,328)])

    f.rect(36,328,1128,290,"#f7fafc")
    f.text(56,368,"2. Run a stage in waves of up to eight independent jobs",24,weight="bold")
    f.text(56,399,"Stages cover matched masking, paired-change loss comparisons and required controls.",20,color=MUTED)
    for i in range(8):
        x, y, width, height = 54+i*140, 428, 120, 108
        f.rect(x,y,width,height,PALE)
        bounds=[x+9,y+9,x+width-9,y+height-9]
        f.text(x+width/2,y+37,f"GPU {i+1}",22,anchor="middle",weight="bold",container=bounds)
        f.text(x+width/2,y+78,"One job",20,anchor="middle",container=bounds)
    f.text(600,573,"One H100 per job; unique run / attempt paths and resumable checkpoints.",20,anchor="middle",weight="bold")
    f.text(600,600,"Start each wave only after its required data and comparison dependencies are ready.",19,anchor="middle",color=MUTED)
    f.line([(600,618),(600,664)])

    f.rect(36,664,1128,152,GREEN)
    f.text(56,705,"3. Evaluate the frozen configuration and assemble the paper",24,weight="bold")
    f.text(63,758,"Retained predictions",21,weight="bold")
    f.text(63,792,"Run receipts + checkpoint hashes",18,color=MUTED)
    f.text(415,758,"Frozen reference scoring",21,weight="bold")
    f.text(415,792,"Same people and eligibility rules",18,color=MUTED)
    f.text(836,758,"Tables, figures and writing",21,weight="bold")
    f.text(836,792,"Uncertainty and bounded claims",18,color=MUTED)
    f.line([(286,751),(390,751)])
    f.line([(713,751),(812,751)])

    f.text(36,861,"Resource budget: planning assumptions, not measured throughput",23,weight="bold")
    f.rect(36,883,1128,131,SAND)
    bounds=[49,895,1151,1005]
    f.text(56,914,"GPU window: Tue 22 Sep 00:00 → Thu 24 Sep 12:00 PDT (60 h).",20,weight="bold",container=bounds)
    f.text(56,944,"8 H100s × 60 h = 480 H100-hours: 360 planned + 120 reserve.",20,weight="bold",container=bounds)
    f.text(56,972,"Setup precedes this window: Mon 21 Sep 23:00 → Tue 22 Sep 00:00 PDT (≤1 h assumed).",19,container=bounds)
    f.text(56,1001,"Capacity is a planning ceiling; preparation, queueing and measured runtime determine completed work.",18,color=MUTED,container=bounds)
    return f.save()


def render_and_validate(fig):
    import cairosvg
    from PIL import ImageFont

    findings, boxes, assets = [], [], []
    for label in fig["labels"]:
        if label["size"] < 18:
            findings.append(["font below 18px",label["text"]])
        suffix=" Bold" if label["weight"]=="bold" else ""
        font=ImageFont.truetype(f"/System/Library/Fonts/Supplemental/Arial{suffix}.ttf",label["size"])
        left,top,right,bottom=font.getbbox(label["text"],anchor="ls")
        length=font.getlength(label["text"])
        offset=length/2 if label["anchor"]=="middle" else length if label["anchor"]=="end" else 0
        box=(label["x"]+left-offset,label["y"]+top,label["x"]+right-offset,label["y"]+bottom)
        label["measured_bounds"]=box
        c=label["container"]
        if c and (box[0]<c[0] or box[1]<c[1] or box[2]>c[2] or box[3]>c[3]):
            findings.append(["text outside assigned card",label["text"]])
        if box[0]<16 or box[1]<8 or box[2]>W-16 or box[3]>fig["height"]-8:
            findings.append(["text outside canvas safe area",label["text"]])
        for previous,txt in boxes:
            if max(previous[0],box[0])<min(previous[2],box[2]) and max(previous[1],box[1])<min(previous[3],box[3]):
                findings.append(["text overlap",txt,label["text"]])
        for a,b in fig["connectors"]:
            if segment_hits_rect(a,b,box):
                findings.append(["connector crosses text",label["text"]])
        boxes.append((box,label["text"]))
    previews=OUT/"previews"
    previews.mkdir(exist_ok=True)
    for width,suffix in [(W,""),(900,"-900")]:
        path=previews/(Path(fig["file"]).stem+suffix+".png")
        cairosvg.svg2png(url=str(OUT/fig["file"]),output_width=width,write_to=str(path))
        assets.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    path=OUT/fig["file"]
    assets.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT/"execution-figure-layout.json").write_text(json.dumps([fig],indent=2)+"\n")
    report=dict(figure_count=1,preview_count=2,findings=findings,assets=assets,
                checks=["SVG XML parsing","18px minimum font","measured text bounds","card containment",
                        "text-to-text intersections","orthogonal connector/text intersections"],
                limitation="Geometry checks supplement actual image review; capacity is assumed, not measured throughput.")
    (OUT/"execution-layout-check.json").write_text(json.dumps(report,indent=2)+"\n")
    if findings:
        raise ValueError(findings)
    print("1 SVG and 2 previews generated; geometric checks passed.")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    render_and_validate(build())
