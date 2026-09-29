"""Rebuild the writeup's eight figures from the v08 audit packet.

No training, resampling, invented empirical poses, or new statistical inference.
Run with Python containing numpy, pandas, matplotlib and Pillow.
"""
from pathlib import Path
import csv
import hashlib
import json
import os

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
SOURCE = REPO / "docs/iclr/versions/v08"
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "qa/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

INK, MUTED, GRID = "#243343", "#65717C", "#DDE3E8"
DIRECT, ENDPOINT, DELTA = "#0072B2", "#7B5AA6", "#D55E00"
GREEN = "#007D67"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "mathtext.fontset": "dejavusans", "axes.labelsize": 9,
    "axes.titlesize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 9,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": .65,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "ytick.major.size": 0,
    "xtick.major.size": 3, "xtick.major.width": .65,
    "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "figure.facecolor": "white",
    "savefig.facecolor": "white", "hatch.linewidth": .6,
})
INPUTS, VALUES, VALIDATION, FIGURES = {}, [], {}, []


def read(name):
    p = SOURCE / "evidence" / name
    INPUTS[str(p.relative_to(REPO))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text()) if p.suffix == ".json" else pd.read_csv(p)


def one(df, **selection):
    keep = pd.Series(True, index=df.index)
    for key, value in selection.items():
        keep &= df[key].eq(value)
    out = df.loc[keep]
    assert len(out) == 1, (selection, len(out))
    return out.iloc[0]


def record(figure, item, metric, estimate, low=None, high=None, unit="degrees", interval="none"):
    VALUES.append(dict(figure=figure, item=item, metric=metric,
                       estimate=float(estimate), ci_low=low, ci_high=high,
                       unit=unit, interval=interval))


def color(key):
    return DIRECT if key.startswith("P-direct") else ENDPOINT if "jepa_endpoint" in key else DELTA if "jepa_delta" in key else MUTED


def style(ax):
    ax.grid(axis="x", color=GRID, lw=.6, zorder=0)
    ax.set_axisbelow(True)


def panel(ax, title):
    ax.set_title(title, loc="left", pad=12, fontweight="bold")


def title(fig, heading, subtitle):
    fig.text(.025, .972, heading, va="top", fontsize=12.5, fontweight="bold")
    fig.text(.025, .913, subtitle, va="top", fontsize=8.6, color=MUTED)


def foot(fig, text):
    fig.text(.025, .025, text, va="bottom", fontsize=8, color=MUTED, linespacing=1.35)


def interval(ax, r, y, col, marker="o", keys=("ci_low", "ci_high"), filled=True):
    m, lo, hi = float(r.estimate), float(r[keys[0]]), float(r[keys[1]])
    assert lo <= m <= hi
    ax.errorbar(m, y, xerr=[[m-lo], [hi-m]], fmt=marker, color=col,
                mfc=col if filled else "white", ms=5, capsize=2.5, lw=1.2,
                zorder=3)
    return m, lo, hi


def save(fig, name, sources, kind="empirical"):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    outside = []
    for item in fig.findobj(matplotlib.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        b = item.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > bounds.x1+1 or b.y1 > bounds.y1+1:
            outside.append(item.get_text())
    assert not outside, (name, outside)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    Image.open(OUT / f"{name}.png").convert("L").save(OUT / f"{name}-gray.png")
    FIGURES.append(dict(name=name, kind=kind, sources=sources,
                        width_inches=fig.get_figwidth(), height_inches=fig.get_figheight(),
                        text_outside_canvas=outside))
    plt.close(fig)


def workflow():
    fig = plt.figure(figsize=(7, 5.7))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0,7), ylim=(0,5.7)); ax.axis("off")
    ax.text(.18,5.52,"From recorded movement to a measured change",fontsize=12.5,fontweight="bold",va="top")
    ax.text(.18,5.17,"Completed study | Conceptual diagram; no participant poses shown",fontsize=8.6,color=MUTED)

    def box(x,y,w,h,heading,body,fill="#F4F7F9",edge=GRID):
        ax.add_patch(Rectangle((x,y),w,h,fc=fill,ec=edge,lw=.8))
        ax.text(x+.11,y+h-.13,heading,va="top",fontsize=9.5,fontweight="bold")
        ax.text(x+.11,y+h-.41,body,va="top",fontsize=8.8,linespacing=1.38)
    def arrow(a,b,col=MUTED,dashed=False):
        ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",color=col,lw=1,linestyle="--" if dashed else "-"))
    box(.18,3.91,1.98,1.0,"1  Prepare motions","AMASS walking candidates\nSplit by person first\n128 samples at 25 Hz")
    box(2.51,3.91,1.98,1.0,"2  Form paired states","Original + synthetic knee edit\nThen optional 3D mirror\nFixed camera for each pair")
    box(4.83,3.91,1.99,1.0,"3  Create observations","Render + estimate 2D joints\nOcclusion / naming corruption\nKeep clean projected targets")
    arrow((2.17,4.4),(2.5,4.4));arrow((4.5,4.4),(4.82,4.4))

    ax.text(.18,3.62,"TRAINING",fontsize=8.5,fontweight="bold",color=DIRECT)
    ax.text(1.08,3.62,"112 people · 1,645 windows",fontsize=9.2)
    box(.18,2.37,2.54,1.04,"Observed poses → student","Masked joint/time features\nClean references → teacher targets\nEndpoint OR delta auxiliary")
    box(3.12,2.37,3.70,1.04,"Features → coordinate readout","Freeze encoder; fit against reference joints\nCoordinate, original change, or repair objective\nDirect fitting instead updates encoder + readout")
    arrow((2.73,2.9),(3.11,2.9))
    ax.text(.18,2.10,"DEVELOPMENT",fontsize=8.5,fontweight="bold",color=GREEN)
    ax.text(1.47,2.10,"14 different people · 155 windows · 3 fitted seeds",fontsize=9.2)
    box(.18,.96,2.54,.91,"Restore windows separately","Observed a → predicted a\nObserved b → predicted b")
    box(3.12,.96,3.70,.91,"Compare with references for scoring","Paired asymmetry change · knee trajectories\nLeft-right assignment (post hoc)")
    arrow((2.73,1.42),(3.11,1.42))
    ax.text(.18,.69,"MEASUREMENT",fontsize=8.5,fontweight="bold")
    ax.text(.18,.42,"Knee excursion: P95 − P5     Asymmetry: right − left     Change: edited − original",fontsize=9.2)
    ax.text(.18,.16,"Illustrative arithmetic only: (L, R) = (40°, 50°) → (40°, 55°); asymmetry 10° → 15°; change = 5°.",fontsize=8.2,color=MUTED)
    save(fig,"01-workflow",["paper-v08.tex, sections 3.1–3.3", "appendix.tex, A–B"],"conceptual: completed workflow; invented teaching arithmetic")


FAMILIES = [
    ("P-direct-none", "Direct (joint)"),
    ("I-initialized-none", "Initialized"),
    ("I-shuffled_jepa-graph_time", "Shuffled JEPA"),
    ("M-coordinate-graph_time", "Coordinate"),
    ("M-paired_jepa-graph_time", "Core JEPA"),
    ("F-response-coordinate_delta_v1-graph_time", "Coordinate delta"),
    ("F-response-jepa_endpoint_v1-graph_time", "Endpoint JEPA"),
    ("F-response-jepa_delta_v1-graph_time", "Delta JEPA"),
]


def restoration():
    means, effects = read("restoration_means.csv"), read("restoration_effects.csv")
    summary = read("method_summary.csv")
    zero = float(summary.zero_response_error.iloc[0])
    assert np.allclose(summary.zero_response_error, zero)
    assert (summary.response_error > zero).all()
    assert len(summary) == 16
    VALIDATION["all_16_trained_variants_worse_than_zero_pooled"] = True
    fig = plt.figure(figsize=(7,4.65))
    title(fig,"Restoration is not the same as response recovery","All eight fitted families and both readout objectives; smaller errors are better")
    gs=fig.add_gridspec(1,3,left=.22,right=.978,bottom=.22,top=.73,wspace=.34,width_ratios=[1,1,1.03])
    axs=[fig.add_subplot(gs[0,i]) for i in range(3)]
    ys=[8.5,6.9,5.9,4.9,3.9,2.9,1.9,.9]
    for y,(key,label) in zip(ys,FAMILIES):
        col=color(key)
        for ax,metric in zip(axs[:2],["response_error","waveform_error"]):
            a=one(means,method=key+"-base",metric=metric)
            b=one(means,method=key+"-paired_change",metric=metric)
            ax.plot([a.estimate,b.estimate],[y,y],color=GRID,lw=1.5)
            ax.plot(a.estimate,y,"o",color=col,ms=4.6)
            ax.plot(b.estimate,y,"^",mec=col,mfc="white",ms=5.4,mew=1.1)
            for row in [a,b]:record("02-restoration",row.method,metric,row.estimate)
        r=one(effects,family=key,metric="waveform_error")
        vals=interval(axs[2],r,y,col,marker="s")
        record("02-restoration",key,"change_minus_coordinate_waveform",*vals,interval="crossed person/seed bootstrap 95%")
    for i,ax in enumerate(axs):
        ax.set(ylim=(.2,9.25),yticks=ys,yticklabels=[n for _,n in FAMILIES] if i==0 else [])
        ax.axhline(7.65,color=GRID,lw=.8);style(ax)
    axs[0].text(-.045,7.23,"Frozen encoders",transform=axs[0].get_yaxis_transform(),ha="right",fontsize=8,color=MUTED)
    axs[0].axvline(zero,color=INK,ls=":",lw=1.1)
    for ax,lim,ticks,lab,head in zip(axs,[(0,25),(0,30),(-1,13)],[[0,10,20],[0,10,20,30],[0,5,10]],
            ["Response error (°)","Trajectory error (°)","Change − coordinate (°)"],["A  Asymmetry change","B  Knee trajectory","C  Trajectory effect"]):
        ax.set(xlim=lim,xticks=ticks,xlabel=lab);panel(ax,head)
    axs[2].axvline(0,color=MUTED,lw=.8)
    fig.legend(handles=[Line2D([],[],marker="o",ls="",color=INK,label="Coordinates"),
                        Line2D([],[],marker="^",mfc="white",ls="",color=INK,label="Original change package"),
                        Line2D([],[],ls=":",color=INK,label="Zero response: 5.81°")],
               ncol=3,loc="upper center",bbox_to_anchor=(.52,.858),frameon=False,fontsize=8.4,columnspacing=1.4,handlelength=1.4)
    foot(fig,"14 development people × 3 seeds; hierarchical means. A–B: point estimates. C: exploratory paired 95% intervals.\nFailures retained at 720° (response) / 180° (trajectory). Positive C means the original change package worsens error.")
    record("02-restoration","zero_response","response_error",zero)
    save(fig,"02-restoration",["restoration_means.csv","restoration_effects.csv","method_summary.csv"])


def primary():
    data=read("summary-figure-provenance.json")["selected_comparisons"]
    assert len(data)==3 and all(r["ci95_low_deg"] < 0 < r["ci95_high_deg"] for r in data)
    VALIDATION["three_primary_intervals_span_zero"]=True
    fig=plt.figure(figsize=(7,4.55))
    title(fig,"Three scientific questions; three uncertain benefits","Recorded primary comparisons reuse the same 14 development people and 3 seeds")
    settings=[("1  Does feature prediction help?","Core JEPA/change versus direct/change",MUTED),
              ("2  Does a feature difference help?","Delta/change versus endpoint/change",DELTA),
              ("3  Does dense supervision help?","Delta/dense versus delta/low scalar",DELTA)]
    for i,(r,(heading,comparison,col)) in enumerate(zip(data,settings)):
        y=.755-i*.228
        fig.text(.035,y,heading,fontsize=10.4,fontweight="bold")
        fig.text(.035,y-.057,comparison,fontsize=9.2)
        desc="Response · crossed person/seed 95% interval" if i<2 else "ViTPose trajectory · person-t 95% interval"
        fig.text(.035,y-.106,desc,fontsize=8.3,color=MUTED)
        ax=fig.add_axes([.67,y-.095,.295,.113])
        m,lo,hi=r["gain_deg"],r["ci95_low_deg"],r["ci95_high_deg"]
        ax.errorbar(m,0,xerr=[[m-lo],[hi-m]],fmt="s" if i==2 else "o",ms=5,color=col,capsize=3)
        ax.axvline(0,color=MUTED,lw=.8)
        ax.set(xlim=(-1.4,2.2),ylim=(-1,1),yticks=[],xticks=[-1,0,1,2]);style(ax)
        if i<2:ax.set_xticklabels([])
        else:ax.set_xlabel("Paired gain (°)",fontsize=8.5,labelpad=2)
        fig.text(.817,y+.035,f"{m:.2f} [{lo:.2f}, {hi:.2f}]°",ha="center",fontsize=8.5,color=col)
        record("03-primary-questions",r["id"],r["metric"],m,lo,hi,interval=r["interval_key"])
    foot(fig,"Positive gains favor the candidate. Outcomes and interval procedures differ; there is no pooled effect.\nEvery interval crosses zero. The first comparison uses direct/change, not the stronger direct-coordinate model.")
    save(fig,"03-primary-questions",["summary-figure-provenance.json"])


KEYS=["P-direct-none-base","F-response-jepa_endpoint_v1-graph_time-paired_change","F-response-jepa_delta_v1-graph_time-paired_change"]
NAMES=["Direct / coordinate","Endpoint / change","Delta / change"]


def reliability():
    data=read("method_summary.csv")
    effects=read("penalty_effects.csv")
    effects=effects[effects.contrast.eq("delta versus endpoint")].sort_values("failure_penalty_deg")
    assert len(effects)==4
    fig=plt.figure(figsize=(7,4.5))
    title(fig,"Rare failures contribute materially to the response score","The score retains failed measurements; conditional successful error is a different quantity")
    gs=fig.add_gridspec(1,3,left=.245,right=.98,bottom=.29,top=.68,wspace=.42,width_ratios=[.75,1.35,1.1])
    axs=[fig.add_subplot(gs[0,i]) for i in range(3)]
    for y,k in zip([2,1,0],KEYS):
        r=one(data,method=k)
        assert np.isclose(r.response_success_contribution+r.response_failure_contribution,r.response_error)
        axs[0].plot(r.failure_percent,y,"o",color=color(k),ms=5)
        axs[0].annotate(f"{r.failure_percent:.3f}",(r.failure_percent,y),xytext=(0,9),textcoords="offset points",ha="center",fontsize=8.5)
        axs[1].barh(y,r.response_success_contribution,height=.4,color="#B8C1C9",edgecolor=INK,lw=.6)
        axs[1].barh(y,r.response_failure_contribution,left=r.response_success_contribution,height=.4,color="white",edgecolor=INK,lw=.6,hatch="////")
        for field in ["failure_percent","response_success_contribution","response_failure_contribution"]:
            record("04-reliability",k,field,r[field],unit="percent" if field=="failure_percent" else "degrees")
    for i,ax in enumerate(axs[:2]):
        ax.set(ylim=(-.55,2.65),yticks=[2,1,0],yticklabels=NAMES if i==0 else []);style(ax)
    axs[0].set(xlim=(0,.8),xticks=[0,.4,.8],xlabel="Failure rate (%)")
    axs[1].set(xlim=(0,12),xticks=[0,6,12],xlabel="Score contribution (°)")
    for y,(_,r) in zip([3,2,1,0],effects.iterrows()):
        vals=interval(axs[2],r,y,INK,filled=r.failure_penalty_deg==720)
        record("04-reliability",f"cost={r.failure_penalty_deg:g}","endpoint_minus_delta",*vals,interval="crossed person/seed 95%")
    axs[2].set(ylim=(-.55,3.55),xlim=(-1.5,2),xticks=[-1,0,1,2],yticks=[3,2,1,0],yticklabels=["0°","180°","360°","720°"],xlabel="Endpoint − delta (°)")
    axs[2].axvline(0,color=MUTED,lw=.8);style(axs[2])
    for ax,h in zip(axs,["A  Failure rate","B  Score components","C  Cost sensitivity"]):panel(ax,h)
    fig.legend(handles=[Patch(fc="#B8C1C9",ec=INK,label="Successful contribution (all cases)"),Patch(fc="white",ec=INK,hatch="////",label="Failure rate × 720°")],loc="upper center",bbox_to_anchor=(.53,.81),ncol=2,frameon=False,fontsize=8.5)
    foot(fig,"14 people × 3 seeds. A–B: descriptive means. C: paired 95% crossed-bootstrap intervals; all cross zero.\n720° is the original cost; other costs are exploratory. At zero cost, failures remain with zero error.\nEndpoint/delta successful contributions (6.31°/6.21°) already exceed zero response (5.81°).")
    VALIDATION["response_score_components_add"]=True
    save(fig,"04-reliability",["method_summary.csv","penalty_effects.csv"])


def repair():
    means=read("repair_means.csv");means=means[means.metric.eq("waveform_error")]
    effects=read("repair_effects.csv");effects=effects[effects.metric.eq("waveform_error")]
    fig=plt.figure(figsize=(7,5.9))
    title(fig,"Readout weighting changes knee-trajectory fidelity","ViTPose inputs; original, low-scalar and dense readouts share their family's frozen encoder")
    gs=fig.add_gridspec(2,2,left=.225,right=.98,bottom=.2,top=.78,hspace=.70,wspace=.25,height_ratios=[1.2,1])
    for j,(var,col,lab) in enumerate([("jepa_delta_v1",DELTA,"Delta"),("jepa_endpoint_v1",ENDPOINT,"Endpoint")]):
        ax=fig.add_subplot(gs[0,j])
        keys=["P-direct-none-base",f"F-response-{var}-graph_time-paired_change",f"R-repair-{var}-scalar_low",f"R-repair-{var}-dense_change"]
        for y,k,mark in zip([3.8,2.4,1.4,.4],keys,["o","^","D","s"]):
            r=one(means,method=k)
            vals=interval(ax,r,y,DIRECT if k.startswith("P-") else col,marker=mark,keys=("person_t_low","person_t_high"),filled=mark!="^")
            ax.annotate(f"{r.estimate:.2f}",(r.estimate,y),xytext=(0,9),textcoords="offset points",ha="center",fontsize=8.5)
            record("05-readout-repair",k,"waveform_error",*vals,interval="person-t 95% after seed averaging")
        ax.axhline(3.08,color=GRID,lw=.8)
        ax.set(xlim=(9,28.5),ylim=(-.25,4.55),xticks=[10,15,20,25],yticks=[3.8,2.4,1.4,.4],yticklabels=["Direct (joint)","Original change","Low scalar","Dense"] if j==0 else [],xlabel="Trajectory error (°)")
        style(ax);panel(ax,f"{'AB'[j]}  {lab} encoder")
        ax=fig.add_subplot(gs[1,j])
        for y,contrast in zip([2,1,0],["low scalar versus original","dense versus original","dense versus low scalar"]):
            r=one(effects,variant=var,contrast=contrast)
            vals=interval(ax,r,y,col,marker="s" if y==0 else "o",keys=("person_t_low","person_t_high"),filled=y==0)
            record("05-readout-repair",var+" / "+contrast,"paired_waveform_gain",*vals,interval="person-t 95% after seed averaging")
        ax.set(xlim=(-.7,6.1),ylim=(-.6,2.6),xticks=[0,2,4,6],yticks=[2,1,0],yticklabels=["Low vs original","Dense vs original","Dense vs low"] if j==0 else [],xlabel="Paired trajectory gain (°)")
        ax.axvline(0,color=MUTED,lw=.8);style(ax)
        panel(ax,f"{'CD'[j]}  {'Primary' if j==0 else 'Secondary'} family gains")
    foot(fig,"14 people; 95% person-t intervals after averaging 3 fitted seeds. Failures cost 180°.\nDeclared primary: delta dense versus low scalar. Other displayed effects are secondary or exploratory.\nDirect is jointly trained. Positive gains favor the first named readout. Response preservation is not established.")
    save(fig,"05-readout-repair",["repair_means.csv","repair_effects.csv"])


def naming():
    df=read("condition_summary.csv")
    df=df[df.stratum_type.eq("naming") & df.metric.eq("assignment_failure_rate")]
    fig=plt.figure(figsize=(7,3.75))
    title(fig,"A lower response error does not certify anatomical names","Post hoc geometric assignment failures: wrong, ambiguous or missing predictions combined")
    gs=fig.add_gridspec(1,3,left=.245,right=.98,bottom=.29,top=.7,wspace=.22)
    for j,(cond,lab) in enumerate([("correct","Correct names"),("global_swap","Global swap"),("temporary_swap","Temporary swap")]):
        ax=fig.add_subplot(gs[0,j])
        for y,k in zip([2,1,0],KEYS):
            r=one(df,stratum=cond,method=k).copy()
            for field in ["estimate","ci_low","ci_high"]:r[field]*=100
            vals=interval(ax,r,y,color(k),marker="o" if k.startswith("P-") else "^",filled=k.startswith("P-"))
            ax.annotate(f"{r.estimate:.1f}",(r.estimate,y),xytext=(0,9),textcoords="offset points",ha="center",fontsize=8.2)
            record("06-anatomical-naming",k+" / "+cond,"assignment_failure_rate",*vals,unit="percent",interval="exploratory crossed person/seed 95%")
        ax.set(xlim=(0,100),xticks=[0,50,100],ylim=(-.5,2.7),yticks=[2,1,0],yticklabels=NAMES if j==0 else [],xlabel="Assignment failure (%)")
        style(ax);panel(ax,f"{'ABC'[j]}  {lab}")
    foot(fig,"14 people × 3 seeds; exploratory 95% crossed-bootstrap intervals for each method mean.\nEligibility differs from angular scoring. Endpoint strata retain 4:1 nonheld/held weighting.\nNo 50% chance baseline is established. Methods also differ in encoder adaptation and readout supervision.")
    save(fig,"06-anatomical-naming",["condition_summary.csv"])


def participants():
    means=read("case-figure-person.csv");seeds=read("case-figure-seeds.csv")
    people=sorted(means.person_code.unique());assert len(people)==14
    fig=plt.figure(figsize=(7,5.5))
    title(fig,"Visibility changes the participant-level picture","Direct coordinate restoration; every development person is shown in the same order")
    gs=fig.add_gridspec(1,3,left=.105,right=.98,bottom=.23,top=.77,wspace=.3)
    settings=[("pooled_noisy","A  Pooled observations","Unchanged − direct",(-5,25)),
              ("clear_zero","B  Clear images","Zero − direct",(-.2,3.1)),
              ("occluded_zero","C  Occluded images","Zero − direct",(-28,4))]
    counts=[]
    for j,(p,heading,sub,lim) in enumerate(settings):
        ax=fig.add_subplot(gs[0,j]);v=means[means.panel.eq(p)].set_index("person_code").loc[people]
        count=int((v.gain_deg>0).sum());counts.append(count)
        assert v.seed_min_gain_deg.min()>lim[0] and v.seed_max_gain_deg.max()<lim[1]
        for y,code in enumerate(people):
            r=v.loc[code];s=seeds[seeds.panel.eq(p)&seeds.person_code.eq(code)]
            assert len(s)==3
            assert np.isclose(s.gain_deg.mean(),r.gain_deg)
            ax.plot([r.seed_min_gain_deg,r.seed_max_gain_deg],[y,y],color="#ADB8C2",lw=1.1)
            ax.scatter(s.gain_deg,np.full(3,y),s=9,color="#ADB8C2",zorder=2)
            ax.plot(r.gain_deg,y,"D",color=DIRECT,ms=3.8,zorder=3)
            record("07-participants",p+" / "+code,"response_gain",r.gain_deg,r.seed_min_gain_deg,r.seed_max_gain_deg,interval="observed seed range; NOT confidence interval")
            for _,row in s.iterrows():record("07-participants",p+" / "+code+f" / seed {row.seed}","response_gain",row.gain_deg)
        ax.set(ylim=(13.7,-.7),yticks=np.arange(14),yticklabels=people if j==0 else [],xlim=lim,xlabel="Response gain (°)")
        ax.set_xticks([[0,10,20],[0,1,2,3],[-20,-10,0]][j])
        ax.axvline(0,color=MUTED,lw=.8);style(ax);panel(ax,heading)
        ax.set_title(heading,loc="left",pad=20,fontweight="bold")
        ax.text(.5,1.025,sub,transform=ax.transAxes,fontsize=8.5,ha="center",color=MUTED)
        ax.text(.5,-.2,f"{count}/14 mean gains favor direct",transform=ax.transAxes,ha="center",fontsize=8.2)
    assert counts==[10,14,1],counts
    VALIDATION["participant_favorable_counts_pooled_clear_occluded"]=counts
    foot(fig,"Diamonds: means of seeds 17, 29, 43. Gray points/ranges: fitted seeds, not confidence intervals.\nPositive favors direct. Axes have different ranges. Original 720° failure cost and 2:1 response weighting retained.\nExploratory profiles summarize errors; they are not reconstructed motions or clinical cases.")
    save(fig,"07-participants",["case-figure-person.csv","case-figure-seeds.csv"])


def future():
    fig=plt.figure(figsize=(7,5.5));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,7),ylim=(0,5.5));ax.axis("off")
    ax.text(.18,5.33,"A testable route toward physical grounding",fontsize=12.5,fontweight="bold",va="top")
    ax.text(.18,4.96,"Proposed research | This architecture and its benefits have not been demonstrated",fontsize=8.6,color=MUTED)
    def box(x,y,w,h,head,body,fc="#F4F7F9",edge=GRID):
        ax.add_patch(Rectangle((x,y),w,h,fc=fc,ec=edge,lw=.8))
        ax.text(x+.12,y+h-.14,head,va="top",fontsize=9.5,fontweight="bold")
        ax.text(x+.12,y+h-.44,body,va="top",fontsize=8.6,linespacing=1.38)
    def ar(a,b,col=MUTED,dash=False):
        ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=1,color=col,linestyle="--" if dash else "-"))
    box(.18,3.52,2.00,1.14,"1  Scene evidence","Video + scale anchors\nDepth + body estimates\nObject shape + contacts")
    box(2.52,3.52,2.01,1.14,"2  Metric motion","3D joints + anatomical sides\nJEPA features as a prior\nUncertainty + provenance")
    box(4.86,3.52,1.96,1.14,"3  Measurement checks","Trajectory / change / identity\nContact / dynamics checks\nReport or abstain")
    ar((2.19,4.08),(2.51,4.08));ar((4.54,4.08),(4.85,4.08))
    box(.18,2.15,6.64,.94,"Validate against measurements outside the prediction pipeline","Synchronized motion capture / calibrated multiview; contacts and forces where needed.\nHold out people and scenes. Do not certify a tool using only its own outputs.",fc="#EDF5F2",edge="#BED8CD")
    ar((5.80,3.50),(5.80,3.10),GREEN)
    ax.text(.18,1.88,"AFTER MEASUREMENT VALIDATION",fontsize=8.5,fontweight="bold",color=GREEN)
    box(.18,.55,2.00,1.11,"Causal prediction","Past → future states\nNew support conditions\nCompare simple predictors")
    box(2.52,.55,2.01,1.11,"Conditional generation","Body + moving objects\nGeometry / contact / diversity\nRetain side-specific changes")
    box(4.86,.55,1.96,1.11,"Selective tools","Qwen selects tool calls\nRules → imitation → RL\nIndependent reward + cost")
    ax.text(.18,.23,"OpenSim/simulation: conditional training or diagnostic signal, not ground truth. No clinical risk claim follows.",fontsize=8.2,color=MUTED)
    save(fig,"08-research-roadmap",["PLAN.md, sections 7–8"],"proposed conceptual research architecture; no results")


def main():
    workflow();restoration();primary();reliability();repair();naming();participants();future()
    assert len(FIGURES)==8
    for name in ["paper-v08.pdf","paper-v08.tex","appendix.tex"]:
        p=SOURCE/name;INPUTS[str(p.relative_to(REPO))]=hashlib.sha256(p.read_bytes()).hexdigest()
    INPUTS[str(Path(__file__).relative_to(REPO))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with (OUT/"plotted-values.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(VALUES[0]));writer.writeheader();writer.writerows(VALUES)
    provenance=dict(source="paper-v08, unchanged audited estimates",no_new_training=True,no_new_inference=True,
                    sha256=INPUTS,figures=FIGURES,validation=VALIDATION,
                    libraries={"matplotlib":matplotlib.__version__,"numpy":np.__version__,"pandas":pd.__version__},
                    note="Teaching arithmetic in Figure 1 is invented and labeled. Figure 8 is proposed. All statistical marks use retained audit values.")
    (OUT/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    for mode in ["color","gray"]:
        canvas=Image.new("RGB",(1600,2300),"#E7EBEE");draw=ImageDraw.Draw(canvas)
        for i,entry in enumerate(FIGURES):
            suffix="-gray" if mode=="gray" else ""
            im=Image.open(OUT/f"{entry['name']}{suffix}.png").convert("RGB");im.thumbnail((780,540))
            x=10+(i%2)*800;y=25+(i//2)*575
            draw.text((x,y-18),entry["name"],fill=INK);canvas.paste(im,(x,y))
        qa=HERE/"qa";qa.mkdir(exist_ok=True);canvas.save(qa/f"figures-{mode}-contact.png")
    print(json.dumps({"figures":len(FIGURES),"plotted_values":len(VALUES),"checks":VALIDATION},indent=2))


if __name__=="__main__":main()
