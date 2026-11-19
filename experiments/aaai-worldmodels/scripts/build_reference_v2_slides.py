"""Build the v2 research tutorial from completed, immutable study artifacts.

This is a scientific figure generator, not a training command. It preserves all
earlier slide decks. Run with the experiment's Python 3.12 environment. Every
displayed empirical value is read from a completed run or an audited inventory.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import sys
import textwrap
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
OUT = PROJECT / "slides"
ASSETS = OUT / "reference_v2_images"
DOCS = PROJECT / "docs/reference_v2"
RUNS = PROJECT / "artifacts/runs"
sys.path.insert(0, str(PROJECT))

NAVY = "#17354b"
TEAL = "#087d78"
ORANGE = "#bc653b"
GRAY = "#526879"
PALE = "#e8eff6"
GREEN = "#dcefe9"
AMBER = "#fae7d9"
BG = "#fafbfd"
LIGHT = "#f0f4f8"
MISSING = "#c8cfd8"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "font.size": 16,
                     "axes.titleweight": "bold"})

TEXT_CHECKS = []
PLOT_CHECKS = []
INPUTS = {}
SLIDES = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    INPUTS[str(path.relative_to(PROJECT))] = sha(path)
    return json.loads(path.read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def completed(name):
    directory = RUNS / name
    report = read(directory / "results.json")
    if report.get("status") != "complete" or report.get("test_only", False):
        raise ValueError(f"A completed real-data run is required: {name}")
    hashes = read(directory / "artifact_hashes.json")
    for rel, expected in hashes.items():
        candidate = directory / rel
        if not candidate.is_file() or sha(candidate) != expected:
            raise ValueError(f"Artifact identity changed: {candidate}")
    return report


def add(title, subtitle, kind, payload, caption, notes, sources,
        status="COMPLETED STUDY", section="Method"):
    if any("—" in str(x) for x in (title, subtitle, caption, notes, payload)):
        raise ValueError("The tutorial uses complete sentences without em dashes.")
    SLIDES.append(dict(title=title, subtitle=subtitle, kind=kind, payload=payload,
                       caption=caption, notes=notes, sources=sources,
                       status=status, section=section))


def fit_text(ax, x, y, w, h, message, *, size=18, color=NAVY,
             weight="normal", ha="left", va="top", pad=.12, min_size=12):
    """Place wrapped text inside a declared rectangle and record its extent."""
    if "—" in str(message):
        raise ValueError("Unexpected em dash")
    artist = ax.text(x + (w/2 if ha == "center" else pad),
                     y + (h/2 if va == "center" else h-pad), "",
                     fontsize=size, color=color, weight=weight, ha=ha, va=va,
                     linespacing=1.28)
    original = str(message)
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    available_width = (w-2*pad)*fig.get_figwidth()*fig.dpi/16
    for current in np.arange(size, min_size-.1, -.5):
        artist.set_fontsize(current)
        prop=artist.get_fontproperties()
        lines=[]
        for paragraph in original.split("\n"):
            line=""
            for word in paragraph.split():
                candidate=f"{line} {word}".strip()
                measured=renderer.get_text_width_height_descent(candidate,prop,False)[0]
                if measured>available_width and line:
                    lines.append(line);line=word
                else:line=candidate
            lines.append(line)
        artist.set_text("\n".join(lines))
        extent = artist.get_window_extent(renderer)
        lower = ax.transData.transform((x+pad/2, y+pad/2))
        upper = ax.transData.transform((x+w-pad/2, y+h-pad/2))
        if extent.x0 >= lower[0]-.5 and extent.x1 <= upper[0]+.5 and extent.y0 >= lower[1]-.5 and extent.y1 <= upper[1]+.5:
            break
    TEXT_CHECKS.append((artist, (x, y, w, h), ax, original))
    return artist


def text(ax, x, y, message, size=18, color=NAVY, **kw):
    return ax.text(x, y, message, fontsize=size, color=color, **kw)


def rect(ax, x, y, w, h, color=PALE, edge="none", linewidth=1):
    p = Rectangle((x, y), w, h, facecolor=color, edgecolor=edge, linewidth=linewidth)
    ax.add_patch(p)
    return p


def card(ax, x, y, w, h, heading, body, color=PALE, size=18):
    rect(ax, x, y, w, h, color)
    hh=min(1.2,h*.44)
    heading_artist=fit_text(ax,x+.12,y+h-hh-.08,w-.24,hh,heading,size=size+1,
                            weight="bold",min_size=size,pad=.06)
    bounds=heading_artist.get_window_extent(ax.figure.canvas.get_renderer())
    heading_bottom=ax.transData.inverted().transform((bounds.x0,bounds.y0))[1]
    bh=heading_bottom-y-.26
    fit_text(ax,x+.12,y+.10,w-.24,bh,body,size=size,min_size=size-1,pad=.06)


def arrow(ax, start, end, color=TEAL):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=17, linewidth=2, color=color,
                                shrinkA=0, shrinkB=0))


def flow(ax, items, y=3.65, h=2.2, colors=None, left=.75, total=14.5):
    gap=.42
    w=(total-gap*(len(items)-1))/len(items)
    for i, item in enumerate(items):
        x=left+i*(w+gap)
        rect(ax,x,y,w,h,(colors or [PALE]*(len(items)-1)+[GREEN])[i])
        fit_text(ax,x+.08,y+.1,w-.16,h-.2,item,size=19,ha="center",va="center",min_size=17)
        if i<len(items)-1:
            arrow(ax,(x+w+.055,y+h/2),(x+w+gap-.055,y+h/2))


def table(ax, headers, rows, *, widths=None, size=17, y=1.90, h=4.65):
    widths=np.asarray(widths or [1]*len(headers),float)
    widths=14.5*widths/widths.sum()
    rh=h/(len(rows)+1)
    for r, values in enumerate([headers]+rows):
        yy=y+h-(r+1)*rh
        xx=.75
        for c,value in enumerate(values):
            rect(ax,xx,yy,widths[c]-.035,rh-.03,NAVY if r==0 else (PALE if r%2 else LIGHT))
            fit_text(ax,xx+.03,yy+.035,widths[c]-.095,rh-.10,value,
                     size=size if r else size-.5,
                     color="white" if r==0 else NAVY,
                     weight="bold" if r==0 else "normal",va="center",min_size=size-1)
            xx+=widths[c]


def base(slide, number, total):
    fig=plt.figure(figsize=(16,9),dpi=100,facecolor=BG)
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,16),ylim=(0,9)); ax.axis("off")
    text(ax,.75,8.63,"GAIT OBSERVATIONS FOR MEDICAL WORLD MODELS",size=11,color=GRAY)
    text(ax,15.25,8.63,slide["status"],size=11,color=ORANGE if any(k in slide["status"] for k in ("LIMIT","GATE","POST HOC")) else TEAL,ha="right")
    fit_text(ax,.60,7.61,14.8,.82,slide["title"],size=29,weight="bold",min_size=27)
    fit_text(ax,.60,6.78,14.8,.78,slide["subtitle"],size=18,color=GRAY,min_size=17)
    fit_text(ax,.60,.91,14.8,.78,slide["caption"],size=15.5,color=GRAY,min_size=14.5)
    ax.plot([.75,15.25],[.78,.78],color="#cbd6df",linewidth=1)
    text(ax,.75,.41,f"REFERENCE-AWARE GAIT STUDY V2  |  {slide['section'].upper()}",size=10,color=GRAY)
    text(ax,15.25,.41,f"{number:02d} / {total}",size=11,color=GRAY,ha="right")
    return fig,ax


def chart(fig, records, xlabel, *, xlim=None, zero=False, chance=False,
          pos=(.35,.25,.58,.46), size=16, digits=4, log=False):
    # Leave room for the x-axis label above the full-sentence figure caption.
    pos=(pos[0],pos[1]+.06,pos[2],pos[3]-.06)
    ax=fig.add_axes(pos,facecolor=BG)
    for i,item in enumerate(records):
        value=item["value"]
        color=item.get("color",TEAL)
        lo,hi=item.get("ci",[value,value])
        ax.errorbar(value,i,xerr=[[max(0,value-lo)],[max(0,hi-value)]],fmt=item.get("marker","o"),
                    color=color,capsize=4,markersize=8,linewidth=2)
        label=f"{value:.{digits}f}"
        ax.annotate(label,(value,i),xytext=(8,8),textcoords="offset points",fontsize=12,color=color)
        PLOT_CHECKS.append(dict(value=float(value),ci95=[float(lo),float(hi)],label=item["label"]))
    ax.set_yticks(range(len(records)),[item["label"] for item in records],fontsize=size)
    ax.invert_yaxis()
    ax.tick_params(axis="y",length=0,pad=14)
    ax.spines[["left","right","top"]].set_visible(False)
    ax.grid(axis="x",color="#dce5eb",linewidth=.8)
    ax.set_xlabel(xlabel,fontsize=14,labelpad=14)
    if xlim: ax.set_xlim(*xlim)
    if log:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([.06,.10,.20,.40,.80]))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    if zero: ax.axvline(0,color=GRAY,linewidth=1)
    if chance: ax.axvline(.5,color=GRAY,linewidth=1,linestyle=":")
    ax.margins(y=.17)
    return ax


def draw_mask(ax, x, y, w, h, name, example, policy):
    values=example[policy]
    states=np.asarray(values["states"])
    rh=h/16; cw=w/24
    colors={0:MISSING,1:GREEN,2:AMBER,3:NAVY,4:"#84b9bf",5:"#edf1f5"}
    for row in range(16):
        for col in range(24):
            rect(ax,x+col*cw,y+(15-row)*rh,cw-.015,rh-.012,colors[int(states[row,col])])
    text(ax,x,y+h+.18,name,size=17,weight="bold")
    text(ax,x,y-.28,"16 prefix samples",size=11,color=GRAY)
    text(ax,x+w,y-.28,"8 future times",size=11,color=GRAY,ha="right")
    ax.plot([x+16*cw-.008]*2,[y-.04,y+h+.04],color=ORANGE,linewidth=1.5)
    for idx in (0,6,8,15):
        text(ax,x-.06,y+(15-idx+.5)*rh,str(example["joint_ids"][idx]),size=9,color=GRAY,ha="right",va="center")


def draw(slide, fig, ax):
    kind,p=slide["kind"],slide["payload"]
    if kind=="cards":
        n=len(p); gap=.4; w=(14.5-gap*(n-1))/n
        for i,v in enumerate(p): card(ax,.75+i*(w+gap),2.1,w,4.2,v[0],v[1],GREEN if i==n-1 else PALE,size=19)
    elif kind=="table":
        table(ax,p["headers"],p["rows"],widths=p.get("widths"),size=p.get("size",17))
    elif kind=="flow":
        flow(ax,p)
    elif kind=="funnel":
        for i,(label,count) in enumerate(p):
            xx=.9+i*.35; yy=5.92-i*.78; ww=8.4-i*.4
            rect(ax,xx,yy,ww,.62,GREEN if i==len(p)-1 else PALE)
            fit_text(ax,xx+.08,yy+.01,ww-.16,.60,label,size=17,va="center",min_size=16)
            fit_text(ax,10.1,yy,4.9,.62,count,size=18,va="center",min_size=17)
    elif kind=="timeline":
        for i in range(24):
            xx=.9+i*.595
            rect(ax,xx,4.1,.51,1.1,PALE if i<16 else GREEN)
            if i in (0,5,10,15,16,23): text(ax,xx+.25,3.73,str(i),size=14,ha="center",color=GRAY)
        text(ax,5.3,5.66,"Visible prefix",size=24,weight="bold",ha="center")
        text(ax,12.8,5.66,"Predicted future",size=24,weight="bold",ha="center",color=TEAL)
        text(ax,5.3,2.97,"16 observations spanning 1.5 seconds",size=20,ha="center")
        text(ax,12.8,2.97,"8 offsets: 0.1 to 0.8 s",size=20,ha="center")
        text(ax,8,2.08,"The primary error averages all eight future offsets.",size=21,ha="center",color=TEAL)
    elif kind=="skeleton":
        points={11:(2.8,5.9),12:(4.2,5.9),13:(2.15,4.85),14:(4.85,4.85),15:(1.8,3.9),16:(5.2,3.9),
                23:(3.0,4.4),24:(4.0,4.4),25:(2.65,3.3),26:(4.4,3.3),27:(2.5,2.25),28:(4.65,2.25),
                29:(2.75,2.06),30:(4.4,2.06),31:(1.97,1.96),32:(5.15,1.96)}
        for a,b in [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
                    (23,25),(25,27),(27,29),(29,31),(24,26),(26,28),(28,30),(30,32)]:
            ax.plot([points[a][0],points[b][0]],[points[a][1],points[b][1]],color=TEAL if a%2 else NAVY,linewidth=4)
        offsets={27:(-.42,.12),28:(.20,.12),29:(.09,-.24),30:(-.30,-.24)}
        for j,(xx,yy) in points.items():
            ax.scatter(xx,yy,color=ORANGE if j in (11,12,23,24) else (TEAL if j%2 else NAVY),s=60,zorder=3)
            dx,dy=offsets.get(j,(-.27 if j%2 else .18,.07))
            text(ax,xx+dx,yy+dy,str(j),size=12)
        card(ax,6.5,4.92,8.55,1.32,"A shared body reference defines the input.","Each frame uses its pelvis center. Scale uses the prefix only.",size=17)
        card(ax,6.5,3.35,8.55,1.42,"Four target joints describe each leg.","Knee, ankle, heel, and foot index are predicted on each leg.",size=17)
        card(ax,6.5,1.88,8.55,1.32,"Joint numbers follow BlazePose-33.","The 16-joint order is explicit. COCO indexing differs.",color=GREEN,size=17)
    elif kind=="folds":
        for c in range(5): text(ax,4.2+c*2.3,6.25,f"Group {c}",size=15,ha="center")
        for r in range(5):
            text(ax,.85,5.63-r*.78,f"Outer fold {r}",size=18)
            for c in range(5):
                xx=3.2+c*2.3; yy=5.37-r*.78
                rect(ax,xx,yy,2.03,.60,GREEN if r==c else PALE)
                fit_text(ax,xx,yy,2.03,.60,"Held out" if r==c else "Normal fitting",size=13,ha="center",va="center",min_size=13)
    elif kind=="masks":
        for i,policy in enumerate(p["policies"]):
            draw_mask(ax,1.0+(i%2)*7.7,4.55-(i//2)*2.42,6.65,1.57,p["labels"][i],p["example"],policy)
    elif kind=="chart":
        chart(fig,p["records"],p["xlabel"],xlim=p.get("xlim"),zero=p.get("zero",False),
              chance=p.get("chance",False),size=p.get("size",16),digits=p.get("digits",4),pos=p.get("pos",(.35,.25,.58,.46)),log=p.get("log",False))
    elif kind=="scatter":
        data=pd.DataFrame(p["data"])
        ch=fig.add_axes([.12,.30,.52,.39],facecolor=BG)
        for label,col,mark in [("normal",TEAL,"o"),("abnormal",ORANGE,"s")]:
            q=data[data.label.eq("normal") if label=="normal" else data.label.ne("normal")]
            ch.scatter(q.persistence,q.ridge_posture_velocity,label=label,s=38,color=col,marker=mark,alpha=.8)
        maximum=max(data.persistence.max(),data.ridge_posture_velocity.max())*1.06
        ch.plot([0,maximum],[0,maximum],color=GRAY,linestyle=":")
        ch.set(xlabel="Persistence MSE",ylabel="Posture + velocity MSE",xlim=(0,maximum),ylim=(0,maximum))
        ch.spines[["top","right"]].set_visible(False);ch.legend(fontsize=12)
        card(ax,10.75,3.45,4.35,2.65,"Motion remains a competing explanation.","An almost stationary source can have a small persistence error. Subtracting that error does not remove every dependence on movement amount.",color=GREEN,size=17)
    elif kind=="calibration":
        card(ax,.8,4.0,4.25,2.2,"R: normal forecast fitting", "15 or 16 sources fit a forecast. Inner fits on 10 or 11 sources create out-of-fit errors for the nuisance reference.",size=17)
        card(ax,5.85,4.0,4.25,2.2,"C: separate normal reference", "The 7 or 8 C sources enter no contributing fit. Their ranks define unusual prediction.",size=17,color=GREEN)
        card(ax,10.9,4.0,4.25,2.2,"Outer sources: evaluation", "Normal and abnormal sources receive the same frozen forecast, nuisance reference, and tail transform.",size=17)
        arrow(ax,(5.12,5.1),(5.76,5.1));arrow(ax,(10.17,5.1),(10.81,5.1))
        fit_text(ax,1.05,2.05,13.9,1.15,"Rotate C three times inside each outer fold, then average the three tail scores. Repeat the complete schedule for the current-posture control.",size=22,ha="center",va="center",min_size=21)
    elif kind=="cdf":
        vals=np.asarray(p["residuals"])
        ch=fig.add_axes([.10,.27,.48,.43],facecolor=BG)
        # Construct every empirical interval exactly. Uniform sampling can miss
        # a narrow central interval even when it is scientifically important.
        edges=np.r_[vals.min()-.8,np.unique(np.sort(vals)),vals.max()+.8]
        midpoints=edges[:-1]+np.diff(edges)/2
        mid_cdf=np.array([(.5+(vals<a).sum()+.5*(vals==a).sum())/(len(vals)+1) for a in midpoints])
        tails=abs(2*mid_cdf-1)
        ch.stairs(tails,edges,baseline=None,color=TEAL,linewidth=2.5)
        reference_tails=np.array([abs(2*(.5+(vals<a).sum()+.5*(vals==a).sum())/(len(vals)+1)-1) for a in np.unique(vals)])
        ch.scatter(np.unique(vals),reference_tails,facecolors=BG,edgecolors=TEAL,s=17,zorder=4)
        zero=np.flatnonzero(np.isclose(tails,0,rtol=0,atol=1e-15))
        if len(zero):
            k=int(zero[0]);center=(edges[k]+edges[k+1])/2
            assert edges[k]<center<edges[k+1] and tails[k]==0
            ch.annotate("Narrow zero-tail interval",xy=(center,0),xytext=(.05,.14),
                        textcoords="axes fraction",fontsize=11,color=GRAY,
                        arrowprops=dict(arrowstyle="->",color=GRAY,linewidth=1.1))
        ch.scatter(vals,np.zeros(len(vals)),marker="|",s=130,color=ORANGE,zorder=4,label="Separate normal references")
        ch.set(xlabel="Log-error residual after the prefix reference",ylabel="Two-sided empirical tail score",ylim=(-.08,1.05))
        ch.spines[["top","right"]].set_visible(False)
        fit_text(ax,9.65,4.25,5.3,1.75,"A high score can mean unusually easy or unusually difficult prediction.",size=22,weight="bold",min_size=21)
        fit_text(ax,9.65,2.18,5.3,1.75,"The small normal reference makes ranks coarse. The transform has no automatic clinical or conformal coverage guarantee.",size=19,min_size=18)
    elif kind=="external_flow":
        flow(ax,["Observed coordinates → each leg", "Windows → each recording", "Recordings → coded participant", "Stored GAVD reference → tail"],y=4.0,h=2.0)
        fit_text(ax,1.0,2.05,14.0,1.2,"Only after participant aggregation, apply each of the 15 frozen transforms. Then average the 15 tail scores. No external example fits the model or reference.",size=22,ha="center",va="center",min_size=21)
    else:
        raise ValueError(f"Unknown visual kind: {kind}")


def label(arm):
    return {"persistence":"Persistence", "damped_velocity":"Damped velocity",
            "ridge_posture":"Current posture", "ridge_posture_velocity":"Posture + velocity state",
            "own":"Own-leg history", "bilateral":"Bilateral history", "shifted":"Shifted history",
            "static":"Static-posture ridge", "untrained_probe":"Untrained encoder + probe",
            "mlp_none":"M0: full prefix", "mlp_uniform":"M1: uniform",
            "mlp_time_blocks":"M2: time blocks", "mlp_anatomical_tubes":"M3: anatomical tubes",
            "mlp_motion_balanced":"M4: motion balanced", "mlp_random_groups":"M3 random groups",
            "mlp_motion_focused":"M4 motion focused", "mlp_pattern_only":"Mask pattern + anchor"}.get(arm,arm)


def fmtci(value):
    return f"{value['estimate']:+.3f} [{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}]"


def mask_example():
    from worldmodels.v2_data import load_cohort
    from worldmodels.v2_masks import MaskConfig, make_mask, POLICIES
    from worldmodels.data import JOINT_IDS
    from worldmodels.v2_models import TARGET_IDS
    x,v,meta,t=load_cohort()
    # A fixed identity-based rule avoids choosing an example for visual effect.
    chosen=meta.sort_values(["source_id","sequence_id","window_id"]).index[0]
    record=dict(window_id=str(meta.loc[chosen,"window_id"]), source_id=str(meta.loc[chosen,"source_id"]),
                selection="First source, sequence, and window in lexical identity order; no performance criterion.",
                joint_ids=list(JOINT_IDS),future_query_joint_ids=list(TARGET_IDS))
    target_columns=[JOINT_IDS.index(joint) for joint in TARGET_IDS]
    for policy in POLICIES:
        hidden,protected,audit=make_mask(x[chosen,:16],v[chosen,:16],t[chosen,:16],MaskConfig(policy=policy),seed=27)
        states=np.full((24,16),5,int)
        states[16:,target_columns]=3
        states[:16]=np.where(v[chosen,:16],1,0)
        states[:16][protected]=4
        states[:16][hidden]=2
        assert int((states[16:]==3).sum())==8*len(TARGET_IDS)==64
        assert np.all(states[16:,target_columns]==3)
        assert int((states[16:]==5).sum())==8*(len(JOINT_IDS)-len(TARGET_IDS))
        record[policy]=dict(states=states.T.tolist(),audit=audit)
    return record


def independent_replay(core, transfer, priors):
    internal=pd.read_csv(RUNS/"reference-v2-core-v1/source_scores.csv")
    external=pd.read_csv(RUNS/"reference-v2-transfer-v1/participant_scores.csv")
    def paired(frame,candidate,reference):
        values=[]
        for fold,q in frame.groupby("outer_fold"):
            y=q.label.ne("normal").to_numpy()
            if y.all() or not y.any(): continue
            weight=int(y.sum()*(~y).sum())
            values.append((weight,float(roc_auc_score(y,q[candidate])),float(roc_auc_score(y,q[reference]))))
        weights=np.array([v[0] for v in values],float)
        candidate=np.average([v[1] for v in values],weights=weights)
        reference=np.average([v[2] for v in values],weights=weights)
        return dict(candidate_auroc=candidate,reference_auroc=reference,estimate=candidate-reference)
    checks=[]
    for name,frame,report,candidate,reference in [
            ("internal_primary",internal,core["primary_anomaly"],"calibrated_two_sided","low_motion"),
            ("internal_matched_posture",internal,core["posture_calibration_contrast"],"calibrated_two_sided","posture_two_sided"),
            ("external_primary",external,transfer["primary_anomaly"],"two_sided","low_motion"),
            ("external_matched_posture",external,transfer["posture_calibration_contrast"],"two_sided","posture_two_sided")]:
        replay=paired(frame,candidate,reference)
        for key,val in replay.items():
            if not np.isclose(val,report[key],atol=1e-12,rtol=0): raise AssertionError((name,key,val,report[key]))
        checks.append(dict(endpoint=name,replayed=replay,maximum_absolute_error=max(abs(replay[k]-report[k]) for k in replay)))
    for entry in core["normal_forecast"]:
        if entry["horizon"]!=8:continue
        replay=internal.loc[internal.label.eq("normal"),entry["arm"]].mean()
        if not np.isclose(replay,entry["mse"],atol=1e-10,rtol=0):raise AssertionError((entry["arm"],replay,entry["mse"]))
    prior_rows=pd.read_csv(RUNS/"reference-v2-mean-prior-posthoc-v1/source_scores.csv")
    for name,expected in priors["normal_mse"].items():
        replay=prior_rows.loc[prior_rows.label.eq("normal"),name].mean()
        if not np.isclose(replay,expected,atol=1e-10,rtol=0):raise AssertionError((name,replay,expected))
    return dict(status="passed",point_estimate_checks=checks,
                normal_forecast_means_replayed=sum(v["horizon"]==8 for v in core["normal_forecast"]),
                posthoc_prior_means_replayed=len(priors["normal_mse"]),
                method="Independent scikit-learn ranking from saved source/participant rows; no model fitting.",
                limitation="Confidence intervals are displayed from the hash-verified run, not recomputed by this figure generator.")


def make_slides(core,transfer,robust,sensitivity,priors,audit,external,example):
    c="../artifacts/runs/reference-v2-core-v1/results.json"
    t="../artifacts/runs/reference-v2-transfer-v1/results.json"
    r="../artifacts/runs/reference-v2-robustness-v1/results.json"
    s="../artifacts/runs/reference-v2-identity-sensitivity-v1/results.json"
    prior_source="../artifacts/runs/reference-v2-mean-prior-posthoc-v1/results.json"
    d="../docs/reference_v2/"
    protocol=d+"protocol.md";model=d+"model_contract.md";evaluation=d+"evaluation_contract.md";ext=d+"external_contract.md"
    norm={v["arm"]:v for v in core["normal_forecast"] if v["horizon"]==8}
    ranks={v["score"]:v for v in core["rankings"]}
    contrast={(v["method"],v["reference"]):v for v in core["forecast_contrasts"]}
    primary=core["primary_anomaly"]; ep=transfer["primary_anomaly"]
    gain=contrast["ridge_posture","persistence"]
    selection=read(RUNS/"reference-v2-core-v1/selection.json")
    selected_steps=[v["selected_steps"] for v in selection.values() if v.get("mode")=="mlp_M0"]
    add("Useful movement prediction needs reference-aware evaluation.",
        f"Current posture reduces normal-source forecast error by {100*gain['relative_reduction']:.1f}% relative to persistence.","cards",[
            ["The scientific question is specific.","Can the model recognize departures from normal organization beyond movement amount, current posture, camera view, and pose quality?"],
            ["The study uses strong references.","Compact forecasts, matched masks, and a separate normal reference are evaluated alongside low-motion and current-posture controls."],
            ["Forecasting and ranking differ.",f"The GAVD study uses {core['sources']} source videos. Transfer uses {transfer['participants']} coded participants. The locked calibrated score adds no demonstrated ranking advantage over low motion."]],
        "Source videos and coded participants are different evaluation units and are never pooled into one sample count.",
        "This tutorial reports the completed v2 study. The internal sources were known during earlier research, so GAVD results remain exploratory. All empirical charts are generated from completed, hash-verified artifacts.",[c,t,protocol],section="Question")
    add("A movement-observation model has a clear medical role.",
        "The workshop welcomes dynamics, uncertainty, validation, and benchmark contributions.","flow",[
            "Video observations", "Body-relative movement", "Short-term prediction", "Tested observation component"],
        "The measured endpoint is a short forecast of visible movement. It does not establish disease progression or treatment response.",
        "A larger medical world model would need patient-state and intervention evidence. This study develops and tests the observation component that such a system would first need to trust.",["https://med-wm.github.io/aaai27-mwm/",protocol],section="Question")
    add("Movement physiology motivates tests, not neural diagnoses.",
        "Impaired walking can become more variable, more stereotyped, or systematically compensated.","table",dict(
            headers=["Motivation","Observable question","Interpretation limit"],rows=[
                ["Parkinson gait can alter bilateral timing.","Does opposite-leg timing add useful prediction?","Projected joints do not identify basal-ganglia circuitry."],
                ["Stroke can change coordination and compensation.","Are left/right forecast patterns unusual?","Muscle-activation modules and affected side are not measured."],
                ["Cerebral palsy can alter selective control.","Do multijoint departures repeat across sources?","Age, devices, and view can explain differences."],
                ["Myopathic gait can reflect weakness.","Do hip-near and foot-near patterns differ?","The video does not measure muscle strength."]],widths=[3.4,3.6,4.0],size=17),
        "The physiological literature motivates both unusually easy and unusually difficult prediction as possibilities. The direction is tested empirically.",
        "The observation-level hypotheses do not identify lesions, neural control modules, or a universal disease signature. Myopathy is not exclusively a central neurological disorder.",[d+"primary_sources.json","../docs/iteration02/literature_and_masking.md"],section="Question")
    add("Independent coverage sets the scale of the evidence.",
        "Thousands of overlapping windows provide many observations of a much smaller number of groups.","table",dict(
            headers=["Evaluation setting","Windows / sequences or recordings","Independent-group proxy"],rows=[
                ["GAVD, internal grouped evaluation",f"{core['windows']:,} windows; {core['sequences']} sequences",f"{core['sources']} source videos; {core['normal_sources']} normal"],
                ["KOA-PD-NM, frozen transfer",f"{transfer['windows']} windows; {transfer['recordings']} recordings",f"{transfer['participants']} coded participants; 27 normal"],
                ["What the count does not prove","Windows from the same recording are related.","GAVD source IDs are not verified patient IDs."]],widths=[3.0,4.0,4.0],size=19),
        "GAVD alone does not establish patient-disjoint, site-disjoint, or prospective clinical generalization.",
        "The external coded participant keys keep repeated directions together. Unknown identities or reencoded duplicates can still limit independence across datasets.",[c,t,d+"data_summary.md"],section="Data")
    inventory=audit["fresh_inventory"]
    add("The full annotation inventory is audited before fitting.",
        "Upstream provenance CSVs are excluded from the transformed per-condition counts.","funnel",[
            ["All transformed labels",f"{inventory['all_sequences']:,} sequences / {inventory['all_sources']} sources"],
            ["Five requested annotations",f"{inventory['selected_sequences']} sequences / {inventory['selected_sources']} sources"],
            ["Media-eligible material",f"{inventory['media_eligible_sequences']} sequences / {inventory['media_eligible_sources']} sources"],
            ["After two preserved identity exclusions","641 candidates / 96 sources"],
            ["Prepared, then common target support",f"2,818 → {core['windows']:,} windows / {core['sources']} sources"]],
        "All 32 annotated normal source IDs are already in the five-label scope; 29 survive the fixed study gates.",
        "Processing other labels can test specificity or transfer, but cannot add normal-source support. Unusual movement categories retain their actual annotation meanings.",[d+"data_audit.json",d+"data_attrition_by_label.csv"],section="Data")
    add("The ingestion boundary reuses Ambient interfaces.",
        "The existing native-frame pose caches are verified and reconstructed rather than extracted again.","flow",[
            "GAVD annotation frame (one-based)", "Native decoder index (zero-based)",
            "RGB image → Ambient pose", "Timestamped joints + confidence"],
        "The conversion occurs exactly once. All 96 source caches and all 2,818 prepared windows were checked against pinned inputs.",
        "The experiment adapter uses Ambient's cached downloader and SequenceKeypointExtractor.extract_from_image. Originals remain under data/videos-gavd. A media download alone is not proof of annotation alignment.",[d+"data_audit.json","../worldmodels/pose.py","../../../ambient/pose/keypoint_extractor.py"],section="Observation contract")
    add("Every model sees the same past and predicts the same future.",
        "A window has 24 samples, 16 selected joints, and two image coordinates per joint.","timeline",None,
        "Nearest-native-frame sampling is nominally 10 Hz. Actual decoded times are retained for motion calculations.",
        "The prefix spans 1.5 seconds between its first and last endpoints. Eight future observations cover nominal offsets 0.1 through 0.8 seconds. The primary MSE is an average across that suffix, not only the last point.",[protocol,d+"data_audit.json"],section="Observation contract")
    add("Body-relative articulation has explicit units and joints.",
        "A pose estimates body-joint positions in an image. The diagram labels the joint IDs and reference.","skeleton",None,
        "Coordinates are projected 2D articulation in prefix-torso units. They are not measured 3D angles or physical walking speed.",
        "Both hips contribute to the pelvis reference. Therefore, an own-leg history model already conditions on a shared body reference. Protect reference observations when applying prefix masks.",[model,"../docs/restart/data_contract.md"],section="Observation contract")
    add("Missing observations, training masks, and queries differ.",
        "A confidence gate describes what was observed; a training mask deliberately removes an observed input.","table",dict(
            headers=["Operation","Meaning","Where it may act"],rows=[
                ["Natural missingness","The pose estimate lacks sufficient support.","The prefix keeps validity indicators; missing targets are not scored."],
                ["Artificial prefix masking","A policy hides an otherwise observed xy token.","Coordinates are erased before causal filling and feature calculation."],
                ["Future query","The predictor is asked for a fixed future joint and time.","Queries do not depend on future confidence or target values."],
                ["Future-isolation test","Change every future coordinate and validity bit.","Predictions, masks, anchors, and prefix normalization must stay unchanged."]],widths=[2.6,4.2,5.0],size=17),
        "Confidence ≥ 0.5 and separate 70% prefix/future support gates are study settings. Future-relative target normalization and evaluation support may change.",
        "Offline future support defines which predictions can be evaluated. Prefix quality, alone, must determine any deployment-time abstention. Future-relative target centering and scoring support may change when the future is altered; predictor inputs and outputs must not.",[model,protocol,"../tests/test_v2_models.py","../tests/test_v2_evaluation.py"],section="Observation contract")
    add("Equal-source evaluation prevents long videos from dominating.",
        "Mean squared error averages squared coordinate discrepancies on common observed targets.","flow",[
            "Coordinates within each leg", "Equal weight for both legs", "Windows, then sequences", "Equal source components"],
        "The outcome has squared prefix-torso units. Repeated windows improve measurement, but do not create independent people.",
        "Training samples source, sequence, and window hierarchically. Evaluation averages in the same declared hierarchy. External evaluation replaces sequence/source with recording/participant before calibration.",[evaluation,ext],section="Evaluation")
    add("Simple forecasts make a complex claim earn its place.",
        "Every reference predicts identical observed targets under the same normal-only grouping.","table",dict(
            headers=["Reference","Available information","What it tests"],rows=[
                ["Persistence / damped velocity","Latest pose / last measured slope.","Does learning add useful prediction?"],
                ["Current-posture transition","Latest distal articulation and support.","Is ordered history necessary?"],
                ["Posture + velocity state","Posture, robust velocity, dispersion, support.","Does a small dynamic state help?"],
                ["Own / bilateral / shifted history","Local history and matched timing controls.","Does aligned opposite-leg context add value?"],
                ["Random encoder + fitted probe","An untrained representation with a normal probe.","Can generic features explain the improvement?"]],widths=[3.0,4.0,4.0],size=17),
        "The primary anomaly family was fixed as posture plus velocity before new outer outcomes were viewed.",
        "Current posture is a learned transition rather than persistence. A global feature permutation is not a convincing timing control because a linear fit can absorb it; the refitted shift varies by window.",[protocol,model],section="Models")
    add("A compact movement state is interpretable and inexpensive.",
        "The 64 features describe joint positions and recent changes measured from observed prefix pairs.","cards",[
            ["Current state", "Sixteen xy coordinates describe eight distal joints. Eight indicators record whether their latest anchors are known."],
            ["Recent change", "Sixteen coordinatewise median slopes use the last five prefix intervals, divided by actual measured time gaps."],
            ["Dispersion and support", "Sixteen median absolute deviations describe slope variation. Eight measured-pair counts record observation support."]],
        "PV denotes the combined posture, velocity, dispersion, and support state. Any improvement must be interpreted for that combined input.",
        "The ridge model predicts coordinate residuals from the latest anchors. We abbreviate this combined posture-plus-velocity state as PV in the result tables. Its regularization is chosen from 0.1, 1, 10, and 100 inside permitted normal development sources.",[model,protocol],section="Models")
    add("Five grouped outer folds separate fitting from evaluation.",
        "Only normal sources outside the held-out fold fit the forecasting models.","folds",None,
        "Three inner normal-source folds select settings. All earlier GAVD exposure still makes this internal study exploratory.",
        "Normal outer-training counts are 23 in the first four folds and 24 in the last. The exact source lists, masks, seeds, transforms, and checkpoints are retained in immutable manifests.",[evaluation,d+"data_fold_counts.csv",c],section="Evaluation")
    add("Seven mask policies test what the model can recover.",
        "All policies predict the same hidden future. Only additional training-prefix removal changes.","table",dict(
            headers=["Arm","Prefix information deliberately hidden","Reason for the control"],rows=[
                ["M0","None.","Uncorrupted anchor with more visible input."],
                ["M1 / M2","Uniform tokens / four-sample time blocks.","Random removal versus short tracking gaps."],
                ["M3 / M3-random","Anatomical groups / shuffled groups of the same sizes.","Anatomy versus matched mask geometry."],
                ["M4 / M4-focused","Balanced low/high motion / bounded motion preference.","Compare different motion-informed priorities."]],widths=[2.4,4.9,4.7],size=18),
        "M1 and later arms hide 40% of eligible non-protected tokens. The endpoint uses a full naturally observed prefix.",
        "MAMP, Skeleton S-JEPA, SLiM, and ASMa establish precedents. This experiment adapts mask ideas to a small normal-only forecasting cohort; it is not an official reproduction of their action-recognition systems.",[model,d+"primary_sources.json"],section="Masking")
    add("Real observation support determines each realized mask.",
        "These examples use the first window in lexical source, sequence, and window order.","masks",dict(
            policies=["none","uniform","time_blocks","anatomical_tubes"],labels=["M0: naturally observed prefix","M1: uniform prefix removal","M2: exact-budget time priorities","M3: exact-budget anatomical priorities"],example=example),
        "Green, blue-green, peach, and gray mark visible, protected, hidden, and missing prefix inputs. Navy marks future queries; pale future cells are not queried.",
        f"Example identity: {example['window_id']}. All masks use seed 27. Row labels are BlazePose IDs. The prefix has 16 joint rows, but the eight-step future queries include only joints {', '.join(map(str,example['future_query_joint_ids']))}. This gives 64 future joint-time queries. Natural missingness is shown only in the observed prefix; future query choices do not consult future visibility.",[model,d+"data_audit.json"],section="Masking")
    a=example["anatomical_tubes"]["audit"]
    add("Mask matching concerns both budget and geometry.",
        "The anatomical and random-group policies share group sizes 4, 4, 2, 2 and four-sample time blocks.","masks",dict(
            policies=["random_groups","motion_balanced","motion_focused","uniform"],labels=["M3-random: geometry control","M4: low/high-motion priorities","M4-focused: bounded motion preference","M1: shared random reference"],example=example),
        f"The example hides {a['hidden_tokens']} of {a['eligible_tokens']} eligible tokens ({100*a['realized_fraction']:.1f}%). Reference joints and latest anchors remain visible.",
        "The final structured block can be truncated to achieve the exact budget. These are budget-matched priorities, not guaranteed intact tubes. Motion policies reserve at least 20% of hidden tokens for uniform draws. Navy future cells are the eight distal target rows; pale future cells are not queried. The separate pattern-only diagnostic tests information carried by the mask itself.",[model],section="Masking")
    add("The neural comparison changes masks without changing the task.",
        "A small multilayer perceptron predicts coordinate residuals from 832 prefix features.","flow",[
            "Causal prefix features + visibility", "32-unit nonlinear layer", "32-unit nonlinear layer", "128 residual coordinates + anchors"],
        f"The network has 31,936 parameters. Normal inner validation selects {', '.join(map(str,selected_steps))} updates across the five folds.",
        "The output head starts at zero, so the initial forecast is persistence. AdamW uses learning rate 0.001, weight decay 0.001, gradient clipping at one, and at most 16 distinct normal sources per batch. All mask arms share exposure and four finite corruption banks; M0 still sees more tokens.",[model,c],section="Models")
    refs=["persistence","damped_velocity","ridge_posture","ridge_posture_velocity","own","bilateral","shifted","static","untrained_probe"]
    add("Normal-source forecasts are compared against strong references.",
        f"The primary forecast error averages eight future offsets on {core['normal_sources']} held-out normal sources.","chart",dict(
            records=[dict(label=label(k),value=norm[k]["mse"],ci=norm[k]["ci95"],color=TEAL if k=="ridge_posture_velocity" else GRAY) for k in refs],
            xlabel="Source-average future MSE on a log scale; lower is better",size=14,pos=(.37,.24,.55,.46),log=True),
        "Bars are 95% source-bootstrap intervals conditional on the saved fits. Current posture and random-feature controls remain visible.",
        "This compares useful forecast capability under identical target support. It does not establish a clinical mechanism or imply that the best forecaster must be the best anomaly score.",[c],section="Forecast results")
    pairs=[("ridge_posture","persistence"),("ridge_posture_velocity","ridge_posture"),("bilateral","own"),("bilateral","shifted")]
    pair_records=[dict(label=f"{label(a)} vs\n{label(b)}",value=contrast[a,b]["estimate"],ci=contrast[a,b]["ci95"]) for a,b in pairs]
    for name,title in [("mean_pose","Normal mean future pose"),("mean_residual","Anchor + mean displacement")]:
        v=next(v for v in priors["comparisons"] if v["method"]==name and v["reference"]=="ridge_posture")
        pair_records.append(dict(label=f"Current posture vs\n{title}",value=-v["estimate"],ci=[-v["ci95"][1],-v["ci95"][0]],color=ORANGE,marker="D"))
    add("Paired controls separate prediction from a normal pose prior.",
        "Positive effects favor the first method. Orange diamonds are added post hoc mean-prior controls.","chart",dict(
            records=pair_records,
            xlabel="Reference MSE minus method MSE; positive favors the first method",zero=True,size=14,digits=4),
        "All 29 normal sources are paired. The mean-prior comparisons were added after outcomes were known, under a separate analysis identity.",
        "The constant future-pose prior averages normal training targets without conditioning on a prefix. The mean-displacement prior adds the normal training residual mean to each latest anchor. Both preserve the original normal outer-training groups and common support, with no hyperparameter selection. Their gains and intervals are new outcome-exposed diagnostics, not original primary endpoints. The richer PV state remains weaker than posture alone.",[c,prior_source],status="CORE + POST HOC CONTROLS",section="Forecast results")
    masks=["mlp_none","mlp_uniform","mlp_time_blocks","mlp_anatomical_tubes","mlp_motion_balanced","mlp_random_groups","mlp_motion_focused","mlp_pattern_only"]
    add("The complete masking result keeps every policy visible.",
        "The main endpoint averages three predictions before computing MSE; the pattern diagnostic uses seed 27.","chart",dict(
            records=[dict(label=label(k),value=norm[k]["mse"],ci=norm[k]["ci95"],color=ORANGE if k=="mlp_pattern_only" else TEAL) for k in masks],
            xlabel="Normal-source future MSE; lower is better",size=14,pos=(.36,.24,.56,.46)),
        "Identical future support and full-prefix inference make this a controlled comparison of training corruption.",
        "M0 is a higher-information training anchor. The mask-pattern model still adds persistence anchors at its output and is a diagnostic, not a selected deployment method. No claim about a latent JEPA objective follows from this coordinate-only comparison.",[c,model],section="Masking results")
    mpairs=[("mlp_anatomical_tubes","mlp_uniform"),("mlp_anatomical_tubes","mlp_random_groups"),("mlp_motion_balanced","mlp_uniform"),("mlp_motion_balanced","mlp_motion_focused")]
    add("Structured masks must beat their specific simple controls.",
        "Matched geometry tests anatomy; matched motion preference tests how motion is allocated.","chart",dict(
            records=[dict(label=f"{label(a)} vs\n{label(b)}",value=contrast[a,b]["estimate"],ci=contrast[a,b]["ci95"]) for a,b in mpairs],
            xlabel="Reference MSE minus structured-mask MSE; positive favors structured mask",zero=True,size=15,digits=4),
        "These exploratory intervals condition on saved fits. The complete comparison is retained even when an interval spans zero.",
        "An action-recognition precedent does not guarantee an anomaly-detection benefit. A mask can regularize forecasting without proving that it captures physiological coordination.",[c,model],section="Masking results")
    part_rows=[]
    for result in core["retraining_sensitivity"]:
        e=next(v for v in result["contrasts"] if v["reference"]=="ridge_posture")
        part_rows.append([f"Partition seed {result['partition_seed']}",f"{e['estimate']:+.5f}","PV gain versus current posture."])
    add("Optimization seeds and repeated partitions are sensitivity checks.",
        "Three optimization seeds describe fitting variation. Two alternate source partitions repeat the compact forecast comparison.","table",dict(
            headers=["Sensitivity","Measured value","Meaning"],rows=part_rows+[
                ["Neural seeds","27, 41, 73","All per-seed source scores remain in the registry."],
                ["Primary mask endpoint","Mean of three predictions","The ensemble is scored once per source."],
                ["Independent sample size","29 normal GAVD source videos","Seeds and split repeats do not create participants."]],widths=[3.0,3.2,5.0],size=18),
        "Partition effects use complete refits under the declared normal-only selection. They remain correlated evaluations of the same sources.",
        "The run saves source-level errors for every seed, policy, and repeated partition. These checks quantify limited fitting variation; the main bootstrap intervals still condition on fixed fits.",[c,"../artifacts/runs/reference-v2-core-v1/retraining_summary.json"],section="Sensitivity")
    add("A small error can reflect quiet motion as well as skill.",
        "Each point is one held-out GAVD source under the compact full-training forecast.","scatter",dict(
            data=pd.read_csv(RUNS/"reference-v2-core-v1/source_scores.csv")[["label","persistence","ridge_posture_velocity"]].to_dict("records")),
        "The dotted line is equal forecast error. Colors indicate annotations only after the predictions are frozen.",
        "Persistence repeats the latest observed pose. Its error measures observed displacement from that anchor. Historical excess error subtracts persistence MSE from model MSE, but this does not automatically remove motion amount. The new score therefore uses explicit prefix covariates and retains negative persistence as its locked comparator.",[c,evaluation],section="Anomaly scoring")
    add("The normal reference is separate from every contributing fit.",
        "The same forecast and nuisance reference score C and the held-out outer sources.","calibration",None,
        "The nuisance model sees log prefix motion and prefix missingness. C sources are excluded from all contributing fits and selections.",
        "Out-of-fit R errors fit a small ridge reference for log MSE. That reference is trained from smaller predictors than the final R predictor, which creates a declared training-size approximation. All fit memberships and checkpoint hashes are saved.",[evaluation,protocol],section="Anomaly scoring")
    calibrators=read(RUNS/"reference-v2-core-v1/calibration_models.json")
    if isinstance(calibrators,dict):calibrators=calibrators.get("components",calibrators.get("models",calibrators))
    first=calibrators[0]
    cal=read(RUNS/"reference-v2-core-v1"/first.get("calibrator",first.get("calibrator_path")))
    residuals=cal.get("calibration_residuals")
    if residuals is None:
        residuals=[row["residual"] for row in cal["calibration_rows"]]
    add("A two-sided tail asks what is unusual relative to normal.",
        "The exact step curve uses the first frozen calibration component, selected by its saved order.","cdf",dict(residuals=residuals),
        f"The residual is log forecast MSE minus its normal prediction from the prefix. Orange ticks mark {len(residuals)} normal references; open circles show midrank tails.",
        "The empirical mid-CDF is F=(0.5+number below+0.5×number tied)/(nC+1). The anomaly score is |2F−1|. A high score means unusual ease or difficulty after the declared nuisance reference, not a specific disease. The exact same schedule is fit for current posture.",[evaluation,"../artifacts/runs/reference-v2-core-v1/calibration_models.json"],section="Anomaly scoring")
    add("The main ranking endpoint compares paired within-fold scores.",
        "AUROC is the probability that an abnormal source ranks above a normal source, with half credit for ties.","cards",[
            ["Primary candidate","The locked two-sided posture-plus-velocity score is compared with negative persistence error, the low-motion baseline."],
            ["Within-fold aggregation","Only abnormal-normal pairs inside the same outer fold contribute. Folds are weighted by their available number of pairs."],
            ["Group-level uncertainty","The bootstrap resamples source components within fold and label, using the same draws for both methods."]],
        "An AUROC of 0.5 is chance ranking. AUROC does not define a diagnostic threshold or clinical operating point.",
        "The 5,000 paired bootstrap draws condition on saved overlapping fits and unknown participant relationships. Pooled rankings, average precision with prevalence, and equal-fold results are secondary summaries. Average precision summarizes precision as the threshold changes and depends on abnormal-source prevalence.",[evaluation,c],section="Evaluation")
    def rankval(score):return ranks[score]["within_fold"]["pair_weighted_auroc"]
    controls=[("calibrated_two_sided","Locked two-sided PV"),("posture_two_sided","Matched two-sided posture"),("low_motion","Low motion: −persistence"),("ridge_posture_velocity","Raw PV forecast MSE"),("excess_ridge_posture_velocity","PV minus persistence MSE"),("posture_distance","Normal posture distance"),("negative_prefix_motion","Low prefix motion"),("high_prefix_missingness","High prefix missingness"),("marginal_two_sided","Marginal two-sided PV")]
    add("The locked detector is judged beside all nuisance controls.",
        f"Primary paired AUROC difference versus low motion: {fmtci(primary)}.","chart",dict(
            records=[dict(label=name,value=rankval(key),color=TEAL if key=="calibrated_two_sided" else (ORANGE if key=="low_motion" else GRAY)) for key,name in controls],
            xlabel="Pair-weighted within-fold source AUROC; higher is better",xlim=(0,1),chance=True,size=14,digits=3,pos=(.37,.24,.55,.46)),
        "The plotted controls show point estimates. The subtitle reports the primary 95% paired conditional interval.",
        f"The matched-posture paired difference is {fmtci(core['posture_calibration_contrast'])}. It controls the calibration schedule as well as the observed current posture. Low motion and missingness cannot be omitted because they challenge the central claim.",[c,evaluation],section="Anomaly results")
    folds=primary["folds"]
    foldrows=[]
    for row in folds:
        foldrows.append([str(row.get("outer_fold",row.get("fold"))),str(row.get("normal",row.get("n_normal"))),str(row.get("abnormal",row.get("n_abnormal"))),f"{row.get('candidate_auroc',row.get('candidate')):.3f}",f"{row.get('reference_auroc',row.get('reference')):.3f}"])
    add("Every fold remains visible beside the aggregate.",
        "Different source populations can produce different rankings even under a fixed method.","table",dict(
            headers=["Outer fold","Normal","Abnormal","Candidate AUROC","Low-motion AUROC"],rows=foldrows,widths=[1.5,1.5,1.6,3,3],size=19),
        "Counts are source components. The primary estimate excludes cross-fold comparisons and uses positive-negative pair weights.",
        "Fold heterogeneity, coarse calibration ranks, and small reference sets constrain interpretation. A fresh partition cannot turn already studied GAVD sources into an untouched cohort.",[c,evaluation],section="Anomaly results")
    effects=robust["normal_forecast_contrasts"]
    erows=[]
    names={"observed":"Observed prefix","uniform_dropout":"40% prefix removal","coordinate_noise":"Coordinate noise","opposite_time_shift":"Opposite-leg time shift"}
    for condition in robust["conditions"]:
        row=[names[condition]]
        for method,reference in [("ridge_posture_velocity","ridge_posture"),("mlp_anatomical_tubes","mlp_uniform"),("mlp_motion_balanced","mlp_uniform")]:
            q=next(q for q in effects if q["condition"]==condition and q["method"]==method and q["reference"]==reference)
            row.append(f"{q['estimate']:+.5f}")
        erows.append(row)
    add("Shared corruptions test sensitivity to the observations.",
        "Future targets stay fixed while prefix gaps, coordinate noise, or opposite-leg timing are perturbed.","table",dict(
            headers=["Prefix condition","Posture MSE − PV MSE","Uniform MSE − tube MSE","Uniform MSE − balanced MSE"],rows=erows,widths=[3.2,3.1,3.1,3.4],size=17),
        "Values are paired MSE differences on 29 normal sources; positive favors the second method. Conditional 95% intervals remain in the linked report.",
        "The tested corruptions do not reverse the point-estimate ordering. Uniform hiding removes 40% of eligible inputs, Gaussian noise has SD 0.02 torso units, and timing stress rotates opposite-leg observations before protected anchors. This is observation sensitivity, not a neurological intervention.",[r,protocol],section="Robustness")
    nuisance=robust["nuisance_sensitivity"]
    nrows=[]
    for row in nuisance.get("view_strata",[]):
        if row.get("status")!="estimable":continue
        nrows.append([f"{row['view'].capitalize()} view, pooled",f"{row['normal']} normal / {row['abnormal']} abnormal",f"{row['calibrated_two_sided']:.3f} / {row['low_motion']:.3f}"])
    pair=nuisance["paired_within_fold"]
    nrows.append(["Matched motion + missingness",f"{nuisance['evaluable_matched_sources']} sources / {len(nuisance['evaluable_matched_folds'])} folds",f"{pair['candidate_auroc']:.3f} / {pair['reference_auroc']:.3f}"])
    if not nrows:nrows=[["View and quality sensitivity","Saved per-source outputs","Report the retained population and all pair distances."]]
    add("View and observation quality remain alternative explanations.",
        "Stratification and matching describe supported subsets rather than replacing the full-cohort endpoint.","table",dict(
            headers=["Analysis","Source support","Candidate / low-motion AUROC"],rows=nrows,widths=[3.5,3.0,4.8],size=18),
        f"View rows are pooled. Matching retains {nuisance['retained_sources']} sources, with {nuisance['evaluable_matched_sources']} evaluable across four folds. Its paired within-fold difference is {fmtci(pair)}.",
        "The matcher standardizes log prefix motion and missingness using outer-training normals. Every retained pair and distance is published. The front-view slice has only four normal sources. Neither matching nor reweighting removes unknown confounding.",[r,"../artifacts/runs/reference-v2-robustness-v1/nuisance_matched_pairs.csv"],section="Robustness")
    probe=robust["source_identity_probe"]
    prows=[]
    for row in probe.get("rows",[]):
        prows.append([row["representation"],f"{row['balanced_accuracy']:.3f}",str(row["source_classes"])])
    prows.append(["Uniform class reference",f"{probe['rows'][0]['uniform_class_baseline']:.3f}",str(probe['rows'][0]['source_classes'])])
    add("Feature health and source information need separate checks.",
        "A training-source linear probe asks whether features retain video-source information across sequences.","table",dict(
            headers=["Encoder or condition","Balanced source-label accuracy","Source classes"],rows=prows,widths=[4.1,3.7,3.2],size=18),
        "This is a feature-decoding diagnostic inside normal training sources. It is not participant recognition or an independent generalization test.",
        "One deterministic sequence per source fits the linear probe and other sequences evaluate it. The forecaster has already seen training observations, and annotations may overlap. Random features have higher balanced accuracy than the learned features; this small diagnostic does not establish a source-invariance benefit.",[r,model],section="Diagnostics")
    by=external["by_label"]
    add("Frozen transfer uses an independently sourced video resource.",
        "KOA-PD-NM contains normal, osteoarthritis, and Parkinson video with dataset-coded participant grouping.","table",dict(
            headers=["Annotation","Local recordings","Retained recordings / windows","Retained coded participants"],rows=[
                [q["label"],str(q["catalog_recordings"]),f"{q['retained_recordings']} / {q['retained_windows']}",str(q["retained_participants"])] for q in by],widths=[2.5,2.2,4.1,3.2],size=19),
        "The local inventory has 189 recordings; 187 decode. Identical files in the second cache are counted once.",
        "The original descriptor and local inventory differ. Invalid and absent files remain explicit. Ambient made 17,138 image-pose calls; the final common support contains 773 windows from 168 recordings and 90 coded participants.",[ext,d+"data_external_preparation_audit.json","https://data.mendeley.com/datasets/44pfnysy89/1"],section="External transfer")
    add("Participant aggregation precedes every nonlinear tail transform.",
        "The GAVD forecast, nuisance model, and normal residual distribution are frozen.","external_flow",None,
        "Averaging per-recording tail ranks first would produce a different statistic and is not the declared method.",
        "There is no external model fit, reference fit, threshold choice, or score reversal. The 15 frozen GAVD components each score the same participant-aggregated MSE and covariates. The external averaging unit differs from the GAVD source-video unit.",[ext,t],section="External transfer")
    excontrols=[("two_sided","Locked two-sided PV"),("posture_two_sided","Matched two-sided posture"),("low_motion","Low motion: −persistence"),("mse","Raw PV forecast MSE"),("excess_error","PV minus persistence MSE"),("low_prefix_motion","Low prefix motion"),("prefix_missingness","High prefix missingness"),("marginal_two_sided","Marginal two-sided PV")]
    add("Transfer keeps the low-motion and posture challenges intact.",
        f"Primary paired participant AUROC difference: {fmtci(ep)}.","chart",dict(
            records=[dict(label=name,value=transfer["anomaly_scores"][key]["auroc"],color=TEAL if key=="two_sided" else (ORANGE if key=="low_motion" else GRAY)) for key,name in excontrols],
            xlabel="External coded-participant AUROC; higher is better",xlim=(0,1),chance=True,size=14,digits=3,pos=(.37,.24,.55,.46)),
        "The comparison uses 27 normal and 63 annotated abnormal participants. The interval conditions on frozen GAVD models.",
        f"The matched-posture difference is {fmtci(transfer['posture_calibration_contrast'])}. The combined abnormal endpoint includes osteoarthritis and Parkinson annotations, so it is not neurological specificity.",[t,ext],section="External results")
    condrows=[]
    for q in transfer["condition_results"]:
        condrows.append([q["label"],f"{q['normal']} / {q['abnormal']}",f"{q['scores']['two_sided']['auroc']:.3f}",f"{q['scores']['low_motion']['auroc']:.3f}",fmtci(q["paired_comparison"])])
    add("Condition slices share the same normal comparison group.",
        "Parkinson and osteoarthritis are reported separately without choosing a favorable subset.","table",dict(
            headers=["Annotation","Normal / abnormal","PV AUROC","Low-motion AUROC","Paired difference [95% interval]"],rows=condrows,widths=[2.3,2.4,2.0,2.5,4.0],size=18),
        "The same 27 normal participants appear in both slices. These are descriptive, correlated comparisons.",
        "The dataset labels were not independently clinically reassessed. A disease-associated rank could reflect camera, age, task, or observation support as well as movement organization.",[t,ext],section="External results")
    shifts=[row for row in transfer["covariate_shift"] if row["component"].startswith("primary_")]
    outside=[row["outside_any_training_covariate_range"] for row in shifts]
    tails=[row["below_reference_residual"]+row["above_reference_residual"] for row in shifts]
    add("Transfer exposes acquisition and reference-range shifts.",
        "Nominal 10 Hz sampling does not make recording conditions equivalent.","cards",[
            ["The cameras differ by group.","Among decodable recordings, 51 of 59 normal videos use 30 FPS; 98 of 100 osteoarthritis videos use 50 FPS. Blur and framing can remain different."],
            ["The usable observations differ.","Only 84 of 218 candidate normal windows survive the fixed observation gates. Age, clothing, reflective markers, and hospital backgrounds also differ."],
            ["The reference can saturate.",f"Across the 15 PV components, {min(outside)} to {max(outside)} of 90 participants lie outside a fitted covariate range. Between {min(tails)} and {max(tails)} fall beyond a normal residual range."]],
        "Confidence summaries use different joint sets in some descriptive caches; the declared 16-joint support and prefix covariates remain consistent.",
        "All per-component range exceedances and empirical-rank endpoint counts are available in covariate_shift.json. No external recalibration is performed after seeing them. GAVD source-video references and external participant averages also differ in variance.",[t,d+"data_external_preparation_audit.json","../artifacts/runs/reference-v2-transfer-v1/covariate_shift.json"],section="Transfer limits")
    add("A visual identity concern has a declared secondary analysis.",
        "The exclusion of PD_SV_001 was recorded before external efficacy scores were viewed.","table",dict(
            headers=["Population","Coded participants","PV minus low-motion AUROC [95% interval]"],rows=[
                ["Primary transfer, retained unchanged",str(transfer["participants"]),fmtci(transfer["primary_anomaly"])],
                ["Exclude the flagged participant",str(sensitivity["participants"]),fmtci(sensitivity["primary_anomaly"])],
                ["Reason for the sensitivity","A nearby helper overlaps the walking person.","Still images cannot establish continuous identity."]],widths=[3.6,3.4,5.0],size=18),
        "The exclusion removes 32 windows and two recordings from one severe-Parkinson coded participant. Severity coverage therefore changes.",
        "The primary 90-person population remains preserved. This separate analysis identity reports the predefined observation-identity concern, not an outcome-driven improvement. It does not resolve every tracking ambiguity.",[s,d+"external_identity_sensitivity_protocol.json",d+"data_external_visual_review.json"],section="Identity sensitivity")
    add("The notebook chain teaches one scientific decision at a time.",
        "Each notebook starts from its own kernel and shows its actual artifacts and computation.","table",dict(
            headers=["Notebook","Question and evidence"],rows=[
                ["20: Cohort and identity","Rebuild the inventory, source groups, attrition, and external compatibility."],
                ["21: Reference scores","Explain movement amount, nuisance controls, and separate normal calibration."],
                ["22: Mask laboratory","Decode one Ambient frame, show real masks, and test causal isolation."],
                ["23: Models and fitting","Distinguish a real demonstration fit from loading the full-run checkpoints."],
                ["24–25: Evaluation and interpretation","Replay source and external endpoints, uncertainty, and robustness."],
                ["26: Manuscript evidence","Expose every candidate, claim mapping, figure input, and deferred gate."]],widths=[3.2,8.8],size=17),
        "A teaching fit and synthetic invariant tests are software or tutorial evidence. They do not replace held-out research results.",
        "The notebooks preserve all earlier notebook identities. The full-run commands use immutable input hashes and source memberships; missing required outputs cause explicit failures rather than synthetic performance.",[d+"notebook_execution.json","../20_cohort_identity_and_coverage.ipynb","../26_manuscript_evidence_and_claims.ipynb"],section="Reproduction")
    add("Additional complexity has explicit scientific prerequisites.",
        "The completed core is deterministic and coordinate-based. Extensions remain separately identified decisions.","table",dict(
            headers=["Extension","Required prerequisite","Evidence needed before a claim"],rows=[
                ["Predict a learned summary","Keep the model, masks, and training budget matched.","Measure future coordinates and verify that summaries retain variation."],
                ["Learn which inputs to hide","A remaining question after fixed masks; constrain coverage.","Measure extra compute and add a randomized-policy control."],
                ["Use walking-cycle timing","Validate events blindly and require two credible prefix cycles.","Report timing error, failure rate, and the eligible population."],
                ["Predict a distribution of futures","Show why a deterministic forecast is inadequate.","Use proper distribution scores, coverage, and sharpness."]],widths=[2.9,4.3,4.8],size=17),
        "These are proposed extensions with gates, not completed results or promised performance improvements.",
        "A phase proxy is not a measured heel strike, and a diverse set of samples is not calibrated uncertainty. Each completed extension would need its own protocol, identity, and independent review.",[protocol,d+"run_record.json"],status="EXTENSION GATES",section="Next decisions")
    add("Every number leads back to a declared evidence identity.",
        "The deck checks input hashes and independently replays principal AUROC points and normal-source means.","table",dict(
            headers=["Evidence","Immutable identity or location","Role"],rows=[
                ["Cohort and extraction","reference-v2-cohort-v1 / restart-full10hz-v1","Annotation, source, weights, native times, support."],
                ["Core fits and calibration","reference-v2-core-v1","Checkpoints, fit ledger, masks, source predictions."],
                ["External preparation / transfer","reference-v2-external-v1 / transfer-v1","Participant mapping, poses, frozen scoring."],
                ["Robustness / identity analysis","robustness-v1 / identity-sensitivity-v1","Separate perturbation and exclusion identities."],
                ["Manuscript and visuals","draft03 ledger / reference_v2_manifest.json","Exact claims, figure values, and reviewed hashes."]],widths=[2.8,5.0,4.2],size=17),
        f"The core invocation recorded {core['fits']:,} fits and {core['seconds_this_invocation']:.1f} seconds. Those fits are computation, not independent samples.",
        "The source generator never trains a model or changes a research run. Its numeric replay tests saved source rows with an independent ranking implementation. Conditional intervals are taken from the verified run manifests.",[c,t,r,s,d+"protocol_freeze.json"],section="Reproduction")


def validate_layout(fig, start, slide_number):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();issues=[];sizes=[]
    for artist,(x,y,w,h),ax,original in TEXT_CHECKS[start:]:
        bb=artist.get_window_extent(renderer)
        lo=ax.transData.transform((x,y));hi=ax.transData.transform((x+w,y+h))
        sizes.append(float(artist.get_fontsize()))
        if bb.x0<lo[0]-.7 or bb.x1>hi[0]+.7 or bb.y0<lo[1]-.7 or bb.y1>hi[1]+.7:
            issues.append(dict(slide=slide_number,text=original,extent=bb.extents.tolist(),container=[*lo,*hi]))
    for chart_axis in fig.axes[1:]:
        bb=chart_axis.get_tightbbox(renderer)
        if bb.y0<170 or bb.x0<60 or bb.x1>1540 or bb.y1>675:
            issues.append(dict(slide=slide_number,text="Plot, tick labels, or axis label escapes its reserved panel.",extent=bb.extents.tolist(),container=[60,170,1540,675]))
    return dict(slide=slide_number,minimum_fitted_font_points=min(sizes),issues=issues)


def html_deck():
    encoded=json.dumps(SLIDES).replace("</","<\\/")
    document='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reference-aware gait prediction: executed v2 study</title>
<style>
:root{color-scheme:light;--navy:#17354b;--muted:#526879;--teal:#087d78}*{box-sizing:border-box}
body{margin:0;background:#eef3f8;color:var(--navy);font:17px/1.6 system-ui,sans-serif}header{max-width:1320px;margin:auto;padding:20px 24px 12px}h1{font-size:25px;line-height:1.3;margin:0 0 8px}header p{margin:4px 0;color:var(--muted);font-size:15px}a{color:var(--teal)}
nav{display:flex;flex-wrap:wrap;gap:10px;align-items:center;max-width:1320px;margin:auto;padding:4px 24px 15px}button,select{font:inherit;border:1px solid #b6c7d4;border-radius:7px;background:#fff;color:var(--navy);padding:7px 12px}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}select{flex:1;min-width:220px;max-width:820px}main{max-width:1320px;margin:auto;padding:0 24px 40px}#art{display:block;width:100%;aspect-ratio:16/9;background:#fafbfd;box-shadow:0 7px 25px #17354b15;border:1px solid #d5dfe8}article{background:white;margin-top:18px;padding:20px 26px;border-radius:8px}article h2{font-size:21px;margin:0 0 12px}article p{max-width:1050px}#sources{display:flex;gap:9px 18px;flex-wrap:wrap;font-size:14px}#position{font-variant-numeric:tabular-nums;font-size:14px}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #bc653b;outline-offset:3px}footer{max-width:1320px;margin:auto;padding:0 24px 30px;font-size:14px;color:var(--muted)}@media(max-width:700px){header,main,nav{padding-left:12px;padding-right:12px}h1{font-size:21px}article{padding:14px 18px}select{order:3;max-width:100%;width:100%}}
@media print{header,nav,article,footer{display:none}main{padding:0;max-width:none}#art{border:0;box-shadow:none}}
</style></head><body><header><h1>Reference-aware gait prediction and anomaly detection</h1><p>The executed v2 study explains the observation pipeline, controlled masks, strong simple alternatives, and frozen external transfer.</p><p><a href="reference_v2.pdf">Download the vector PDF</a> · <a href="reference_v2_manifest.json">Evidence manifest</a> · <a href="README.md">Reproduction guide</a> · <a href="iteration02.html">Earlier iteration02 deck</a></p></header>
<nav aria-label="Slide navigation"><button id="prev" aria-label="Previous slide">← Previous</button><span id="position" aria-live="polite"></span><button id="next" aria-label="Next slide">Next →</button><select id="select" aria-label="Choose slide"></select></nav>
<main><img id="art" alt=""><article><h2 id="title"></h2><p id="notes"></p><p id="caption"></p><div id="sources" aria-label="Evidence links"></div></article></main>
<footer>Use Left/Right, Page Up/Page Down, Home/End, or the slide menu. Numerical values are loaded from completed, hash-verified artifacts. Windows, source videos, and coded participants are distinct units.</footer>
<script>
const slides=__DATA__;let current=0;const $=id=>document.getElementById(id);
slides.forEach((s,i)=>{const option=document.createElement('option');option.value=i;option.textContent=String(i+1).padStart(2,'0')+' · '+s.title;$('select').append(option)});
function show(index){current=Math.max(0,Math.min(slides.length-1,index));const s=slides[current];$('art').src='reference_v2_images/slide-'+String(current+1).padStart(2,'0')+'.svg';$('art').alt=s.title+' '+s.subtitle+' '+s.caption;$('title').textContent=s.title;$('notes').textContent=s.notes;$('caption').textContent=s.caption;$('position').textContent=(current+1)+' / '+slides.length;$('select').value=current;$('prev').disabled=current===0;$('next').disabled=current===slides.length-1;$('sources').replaceChildren();s.sources.forEach((src,i)=>{const a=document.createElement('a');a.href=src;a.textContent=src.startsWith('http')?new URL(src).hostname+' · source '+(i+1):src.split('/').pop();$('sources').append(a)});history.replaceState(null,'','#slide-'+(current+1));}
$('prev').onclick=()=>show(current-1);$('next').onclick=()=>show(current+1);$('select').onchange=e=>show(Number(e.target.value));document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;if(['ArrowRight','PageDown','ArrowLeft','PageUp','Home','End'].includes(e.key)){e.preventDefault();show(e.key==='Home'?0:e.key==='End'?slides.length-1:current+(['ArrowRight','PageDown'].includes(e.key)?1:-1))}});show(Number(location.hash.replace('#slide-',''))-1||0);
</script></body></html>'''
    (OUT/"reference_v2.html").write_text(document.replace("__DATA__",encoded))
    (OUT/"tutorial.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=reference_v2.html"><title>Current gait tutorial</title><p><a href="reference_v2.html">Open the executed reference-aware gait tutorial.</a></p></html>\n')


def contact_sheet(paths):
    thumb_w=480;thumb_h=270;pad=14;label_h=30;cols=4;rows=int(np.ceil(len(paths)/cols))
    contact=Image.new("RGB",(cols*(thumb_w+pad)+pad,rows*(thumb_h+label_h+pad)+pad),"#edf2f7")
    pen=ImageDraw.Draw(contact)
    for i,path in enumerate(paths):
        with Image.open(path) as pic:
            pic=pic.convert("RGB");pic.thumbnail((thumb_w,thumb_h))
            x=pad+(i%cols)*(thumb_w+pad);y=pad+(i//cols)*(thumb_h+label_h+pad)
            contact.paste(pic,(x,y));pen.text((x+3,y+thumb_h+5),f"{i+1:02d}  {SLIDES[i]['title']}",fill=NAVY)
    contact.save(OUT/"reference_v2_contact_sheet.png")


def main():
    started=time.monotonic()
    core=completed("reference-v2-core-v1")
    transfer=completed("reference-v2-transfer-v1")
    robust=completed("reference-v2-robustness-v1")
    sensitivity=completed("reference-v2-identity-sensitivity-v1")
    priors=completed("reference-v2-mean-prior-posthoc-v1")
    audit=read(DOCS/"data_audit.json")
    external=read(DOCS/"data_external_preparation_audit.json")
    for name in ("protocol.md","model_contract.md","evaluation_contract.md","external_contract.md",
                 "protocol_freeze.json","analysis_dependency_freeze.json","primary_sources.json",
                 "external_identity_sensitivity_protocol.json","data_external_visual_review.json"):
        INPUTS[str((DOCS/name).relative_to(PROJECT))]=sha(DOCS/name)
    INPUTS["worldmodels/v2_models.py"]=sha(PROJECT/"worldmodels/v2_models.py")
    example=mask_example()
    replay=independent_replay(core,transfer,priors)
    make_slides(core,transfer,robust,sensitivity,priors,audit,external,example)
    if not 32<=len(SLIDES)<=40:raise AssertionError(len(SLIDES))
    OUT.mkdir(exist_ok=True);ASSETS.mkdir(exist_ok=True)
    write(OUT/"reference_v2_source.json",dict(slides=SLIDES,mask_example=example))
    layouts=[];pngs=[]
    with PdfPages(OUT/"reference_v2.pdf",metadata={"Title":"Reference-aware gait prediction: executed v2 study","Author":"AlexPose research study","Subject":"Verified movement-observation study, controls, and frozen external transfer"}) as pdf:
        for number,slide in enumerate(SLIDES,1):
            start=len(TEXT_CHECKS)
            fig,ax=base(slide,number,len(SLIDES));draw(slide,fig,ax)
            layouts.append(validate_layout(fig,start,number))
            stem=ASSETS/f"slide-{number:02d}"
            fig.savefig(stem.with_suffix(".svg"),facecolor=BG)
            fig.savefig(stem.with_suffix(".png"),facecolor=BG,dpi=125)
            fig.savefig(stem.with_suffix(".pdf"),facecolor=BG)
            pdf.savefig(fig,facecolor=BG)
            pngs.append(stem.with_suffix(".png"));plt.close(fig)
    issues=[issue for item in layouts for issue in item["issues"]]
    write(OUT/"reference_v2_layout_checks.json",dict(status="passed" if not issues else "needs_changes",slides=layouts,issues=issues,
        check="Every declared text container is checked against its actual rendered extent; manual page inspection is separate."))
    if issues:raise AssertionError(f"{len(issues)} text containers need repair")
    html_deck();contact_sheet(pngs)
    write(OUT/"reference_v2_numeric_checks.json",replay)
    outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(ASSETS.glob("*")) if p.is_file()}
    for name in ("reference_v2.html","reference_v2.pdf","reference_v2_source.json","reference_v2_layout_checks.json","reference_v2_numeric_checks.json","reference_v2_contact_sheet.png","tutorial.html"):
        outputs[name]=sha(OUT/name)
    write(OUT/"reference_v2_manifest.json",dict(identity="reference-v2-slides-v1",status="ready_for_visual_and_independent_review",slides=len(SLIDES),
        method="medwm-reference-masking-v2",generator=str(Path(__file__).relative_to(REPO)),generator_sha256=sha(__file__),
        inputs=INPUTS,outputs=outputs,seconds=time.monotonic()-started,
        environment=dict(python=sys.version,matplotlib=matplotlib.__version__,numpy=np.__version__,pandas=pd.__version__),
        numerical_replay=replay["status"],format="Editable SVG text and shapes; vector PDF; responsive HTML navigator.",
        uncertainty="Paired group-bootstrap intervals condition on saved models. No clinical or conformal guarantee.",
        visual_review="reference_v2_visual_review.md"))
    print(json.dumps(dict(slides=len(SLIDES),layout_issues=len(issues),numeric_replay=replay["status"],seconds=time.monotonic()-started)))


if __name__=="__main__":main()
