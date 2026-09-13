"""Small plots for inspecting inputs, optimization and the actual tradeoff."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .body_geometry import PARENTS, descriptor


def preview_pair(cfg, case_id=None, split="train"):
    from .workflow import _npz, _camera
    index=pd.read_csv(cfg.root/"cases.csv")
    choices=index.loc[index.role.eq(split)]
    if case_id is not None:
        choices=choices.loc[choices.case_id.eq(case_id)]
    else:
        choices=choices.loc[choices.event_present.astype(bool) & choices.noise_present.astype(bool)]
    if not len(choices):
        raise ValueError(f"No case available for {split}")
    row=choices.iloc[0]
    case=_npz(row.path)
    scene=_npz(cfg.root/"scenes"/f"{row.scene_id}.npz")
    cached=cfg.root/"predictions"/cfg.prior_id/f"{row.case_id}.npz"
    frame_ids=np.arange(len(case["raw"]))
    if cached.exists():
        case=_npz(cached)
        frame_ids=case["frame_indices"]
    camera=_camera(scene)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    frame=len(case["raw"])//2
    first=("Reference",case["truth"],"#087f78") if "prior" in case else ("Clean reference",case["clean"],"#087f78")
    methods=[first,("Tracked",case["raw"],"#bd4650"),
             ("Frozen prior" if "prior" in case else "Edited reference",case.get("prior",case["event_reference"]),"#2767b2")]
    for ax,(label,joints,color) in zip(axes,methods):
        ax.imshow(scene["rgb"][frame_ids[frame]])
        xy,_=camera.project(joints[frame])
        for j in range(1,22):
            ax.plot(xy[[j,PARENTS[j]],0],xy[[j,PARENTS[j]],1],color=color,lw=1.5)
        ax.scatter(xy[:,0],xy[:,1],s=8,color=color)
        ax.set_title(label)
        ax.set_axis_off()
    suffix="SIMULATED MECHANICS ONLY" if cfg.mode=="demo" else "Controlled AMASS observation"
    fig.suptitle(f"{row.event_family}: event + tracking error | {suffix}")
    return fig


def plot_history(history):
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),constrained_layout=True)
    for (mode,seed),rows in history.groupby(["feature_mode","seed"]):
        for ax,metric in zip(axes,["restoration_loss","evidence_loss"]):
            ax.plot(rows.epoch,rows[metric],label=f"{mode}, {seed}")
            ax.set(xlabel="Training epoch",ylabel=metric.replace("_"," "))
    axes[-1].legend(fontsize=8,bbox_to_anchor=(1.02,1),loc="upper left")
    return fig


def plot_tradeoff(report):
    """Plot locked operating points, without inventing an interpolated frontier."""
    summary=report["summary"]
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    primary=report["decision"].get("primary")
    comparator=report["decision"].get("calibration_selected_comparator")
    shown={"raw","prior_unprojected","conversion_only_unprojected","calibrated_flow_gate","flow_gate","gaussian",primary,comparator}
    palette=plt.get_cmap("tab10")
    count=0
    for row in summary.itertuples():
        if row.method not in shown:
            continue
        if not np.isfinite(row.noise_removal) or not np.isfinite(row.retention):
            continue
        is_gate=row.method.startswith("gate_full")
        color="#087f78" if is_gate else palette(count%10)
        ax.scatter(row.noise_removal,row.retention,s=80 if is_gate else 45,color=color,
                   marker="*" if is_gate else "o",label=row.method,alpha=.8)
        count+=1
    ax.axvline(.25,color="#a9b7c8",ls="--",lw=1)
    ax.set(xlabel="Tracking error removed (fraction)",ylabel="True event retained (unclipped)",
           title="Selected comparisons at saved strengths; all methods remain in the table")
    if (summary.noise_removal.abs()>2).any():
        ax.set_xscale("symlog",linthresh=.5)
        ax.set_xlabel("Tracking error removed (fraction; symmetric log beyond ±0.5)")
    ax.legend(fontsize=8,bbox_to_anchor=(1.01,1),loc="upper left")
    if report["decision"].get("mode")=="demo":
        fig.suptitle("SIMULATED MECHANICS ONLY: no pretrained-model result",color="#9c6800")
    ax.grid(alpha=.15)
    return fig


def plot_event_trace(cfg, case_id, method="prior"):
    """Foot-height trace for an explicit case; angles and timing are in score CSVs."""
    from .workflow import _npz
    case=_npz(cfg.root/"predictions"/cfg.prior_id/f"{case_id}.npz")
    fig,ax=plt.subplots(figsize=(9,3.5),constrained_layout=True)
    t=case.get("timestamps",np.arange(len(case["raw"]))/cfg.fps)
    for name in ("truth","raw",method):
        ax.plot(t,case[name][:,10,1],label=name)
    ax.set(xlabel="Time (s)",ylabel="Left foot height (m)")
    ax.legend()
    return fig
