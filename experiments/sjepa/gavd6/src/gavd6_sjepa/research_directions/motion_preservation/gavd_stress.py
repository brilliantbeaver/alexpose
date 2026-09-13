"""Observational GAVD flow/pose stress gallery, without invented 3D truth."""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..future_prediction.video_pose import decode_exact_window
from .pretrained_models import OpticalFlowEstimator, joint_transport_diagnostics
from .motion_data import load_gavd_manifest
from .body_geometry import PARENTS


def discover_reservations(cfg) -> tuple[list[Path], pd.DataFrame | None]:
    """Use existing reservations; a new random split cannot recover old exposure."""
    explicit = getattr(cfg,"gavd_reservation_csv","") or os.environ.get("MP_GAVD_RESERVATION","")
    if explicit:
        paths = [Path(explicit).expanduser()]
        if not paths[0].is_file():
            raise FileNotFoundError(f"Configured GAVD reservation does not exist: {paths[0]}")
    else:
        roots = {Path("outputs"),cfg.root.parent}
        paths = sorted({p.resolve() for root in roots if root.is_dir()
                        for p in root.glob("future-innovation*/config/source-reservation.csv")})
    if not paths:
        return [],None
    rosters=[]
    for path in paths:
        roster=pd.read_csv(path,dtype={"video_id":str})
        if not {"video_id","role"}<=set(roster) or not set(roster.role)<={"development","confirmation"}:
            raise ValueError(f"Expected an existing FI video_id/role reservation: {path}")
        rosters.append(roster[["video_id","role"]])
    merged=pd.concat(rosters,ignore_index=True)
    # Conservatively respect every discovered reservation if there is more than one.
    merged["reserved"]=merged.role.eq("confirmation")
    reserved=merged.groupby("video_id",as_index=False).reserved.max()
    return paths,reserved


def decode_gavd_window(row, duration_s=3.2, target_fps=20, max_side=256):
    """Manifest frames are one-based inclusive; decoder indices are zero-based.

    Nearest original frames are sampled. The returned actual timestamps document
    the small nonuniformity when source FPS is not a multiple of target FPS.
    """
    first,last=int(row["first_frame"])-1,int(row["last_frame"])-1
    capture=cv2.VideoCapture(str(row["video_path"]))
    try:
        fps=float(capture.get(cv2.CAP_PROP_FPS))
        if not capture.isOpened() or not np.isfinite(fps) or fps<=0:
            raise ValueError(f"Cannot decode GAVD recording: {row['video_path']}")
    finally:
        capture.release()
    if first<0 or last<first or target_fps<=0:
        raise ValueError("Invalid GAVD frame interval or target FPS")
    count=min(int(round(duration_s*target_fps)),int(np.floor((last-first)*target_fps/fps))+1)
    offsets=np.rint(np.arange(count)*fps/target_fps).astype(int)
    offsets=np.unique(offsets[offsets<=last-first])
    if len(offsets)<2:
        raise ValueError("GAVD interval supplies fewer than two distinct source frames")
    decoded,actual_fps=decode_exact_window(row["video_path"],first,int(offsets[-1])+1)
    selected=decoded[offsets]
    original_h,original_w=selected.shape[1:3]
    scale=min(1.0,max_side/max(original_h,original_w))
    height,width=max(1,round(original_h*scale)),max(1,round(original_w*scale))
    selected=np.stack([cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA) for frame in selected])
    source_frames=first+offsets
    return selected,source_frames,source_frames/actual_fps,dict(
        source_fps=float(actual_fps),source_height=original_h,source_width=original_w,
        scale_xy=[width/original_w,height/original_h],manifest_frame_base=1,cache_frame_base=0)


def load_pose_overlay(path, source_frames, scale_xy):
    """Read optional aligned full-frame 22-joint exports, without guessing a camera.

    NPZ fields: source_frames[N] (zero-based original video), joints2d[N,22,2]
    (original full-frame pixels), coordinate_system='full_frame_pixels'. Optional
    repaired_joints2d has the same shape. These are model estimates. A joints3d
    array alone is insufficient to place a 2D overlay and is not silently lifted
    or treated as reference truth.
    """
    with np.load(path,allow_pickle=False) as cache:
        required={"source_frames","joints2d","coordinate_system"}
        if not required<=set(cache.files):
            raise ValueError(f"Pose overlay needs {sorted(required)}: {path}")
        if str(cache["coordinate_system"].item())!="full_frame_pixels":
            raise ValueError("Pose overlays must declare original full-frame pixel coordinates")
        frames=np.asarray(cache["source_frames"],int)
        if frames.ndim!=1 or len(np.unique(frames))!=len(frames):
            raise ValueError("Pose source_frames must contain unique original frame indices")
        lookup={int(frame):i for i,frame in enumerate(frames)}
        result={}
        for field in ("joints2d","repaired_joints2d"):
            if field not in cache:
                continue
            data=np.asarray(cache[field],np.float32)
            if data.shape!=(len(frames),22,2):
                raise ValueError(f"{field} must have shape [N,22,2] with standard SMPL joint order")
            aligned=np.full((len(source_frames),22,2),np.nan,np.float32)
            for t,frame in enumerate(source_frames):
                if int(frame) in lookup:
                    aligned[t]=data[lookup[int(frame)]]*np.asarray(scale_xy)
            result[field]=aligned
    return result


def _draw_pose(rgb, joints, color):
    image=rgb.copy()
    if joints is None:
        return image
    valid=np.isfinite(joints).all(axis=-1)
    points=np.rint(np.nan_to_num(joints)).astype(int)
    for j in range(1,22):
        p=PARENTS[j]
        if valid[j] and valid[p]:
            cv2.line(image,tuple(points[j]),tuple(points[p]),color,1,cv2.LINE_AA)
    return image


def _flow_rgb(flow, limit):
    magnitude=np.linalg.norm(flow,axis=-1)
    intensity=np.clip(magnitude/max(limit,1e-6)*255,0,255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(intensity,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)


def _gallery(rgb, forward, source_frames, overlays, folder, fps):
    magnitude=np.linalg.norm(forward,axis=-1)
    limit=float(np.percentile(magnitude,95))
    height,width=rgb.shape[1:3]
    panels=[]
    for t in range(len(forward)):
        frame=_draw_pose(rgb[t],overlays.get("joints2d",[None]*len(rgb))[t],(255,215,35))
        frame=_draw_pose(frame,overlays.get("repaired_joints2d",[None]*len(rgb))[t],(30,255,180))
        joined=np.concatenate([frame,_flow_rgb(forward[t],limit)],axis=1)
        panel=np.zeros((height+28,width*2,3),np.uint8)
        panel[28:]=joined
        cv2.putText(panel,f"frame {source_frames[t]} | motion scale {limit:.1f}px",(5,18),
                    cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)
        panels.append(panel)
    frames=np.unique(np.linspace(0,len(panels)-1,min(5,len(panels))).astype(int))
    sheet=np.concatenate([panels[t] for t in frames],axis=0)
    sheet_path=folder/"flow-contact-sheet.png"
    cv2.imwrite(str(sheet_path),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
    video_path=folder/"flow-gallery.mp4"
    video_height,video_width=panels[0].shape[:2]
    # Common video codecs require even dimensions; pad only the display artifact.
    display_h,display_w=video_height+video_height%2,video_width+video_width%2
    writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(display_w,display_h))
    if not writer.isOpened():
        return str(sheet_path),""
    try:
        for panel in panels:
            padded=cv2.copyMakeBorder(panel,0,display_h-video_height,0,display_w-video_width,cv2.BORDER_CONSTANT)
            writer.write(cv2.cvtColor(padded,cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return str(sheet_path),str(video_path)


def gavd_stress(cfg) -> pd.DataFrame:
    """Write an external observation gallery and flow-consistency diagnostics.

    It does not fit on GAVD, infer diagnoses or score event-retention against a
    generated trajectory. Missing reservation metadata stops only this stage.
    """
    folder=cfg.root/"gavd";folder.mkdir(parents=True,exist_ok=True)
    if cfg.mode=="demo":
        report=pd.DataFrame([dict(status="not_run_demo",reason="No real GAVD videos are fabricated in demo mode")])
        report.to_csv(folder/"stress-summary.csv",index=False)
        return report
    table=load_gavd_manifest(cfg.gavd_manifest_dir,cfg.gavd_video_root)
    paths,reservation=discover_reservations(cfg)
    if reservation is None:
        report=pd.DataFrame([dict(status="not_run_missing_source_reservation",
            reason="Set MP_GAVD_RESERVATION to the existing FI source-reservation.csv before decoding GAVD",
            annotated_sequences=len(table),available_sequences=int(table.available.sum()))])
        report.to_csv(folder/"stress-summary.csv",index=False)
        return report
    table=table.merge(reservation,on="video_id",how="left",validate="many_to_one")
    # Unknown sources have no established reservation role and stay unopened.
    table["exclude_reason"]=np.where(table.reserved.isna(),"not_in_existing_reservation",
                                     np.where(table.reserved.fillna(True),"reserved_confirmation",""))
    table.to_csv(folder/"source-selection.csv",index=False)
    selected=table.loc[table.exclude_reason.eq("") & table.available].sample(frac=1,random_state=cfg.seed)
    selected=selected.drop_duplicates("video_id").head(cfg.max_gavd_sequences)
    notes=dict(reservation_paths=[str(p) for p in paths],reserved_sources=int(reservation.reserved.sum()),
               group_unit="recording_identity_not_known",clinical_ground_truth=False,
               interpretation="Image-motion consistency diagnostics and manual review only",
               model_projection="Optional 2D poses are upstream estimates, never a 3D reference",
               optical_flow="Same RGB as tracking, not an independent sensor")
    (folder/"interpretation.json").write_text(json.dumps(notes,indent=2)+"\n")
    if selected.empty:
        report=pd.DataFrame([dict(status="not_run_no_available_development_videos")])
        report.to_csv(folder/"stress-summary.csv",index=False)
        return report
    estimator=OpticalFlowEstimator(cfg.flow_backend,cfg.flow_checkpoint,repo_dir=cfg.flow_repo or None,
                                  config_path=cfg.flow_config,device=cfg.device)
    rows=[]
    for row in selected.to_dict("records"):
        item=folder/str(row["sequence_id"]);item.mkdir(exist_ok=True)
        record=dict(sequence_id=row["sequence_id"],video_id=row["video_id"],group_unit="recording",
                    status="observational_only",clinical_reference="unavailable")
        try:
            rgb,frames,timestamps,decode_info=decode_gavd_window(row,cfg.duration_s,cfg.fps,max(256,cfg.image_size))
            flow=estimator.estimate(rgb,bidirectional=True)
            arrays=dict(forward=flow.forward,source_frames=frames,timestamps=timestamps)
            if flow.backward is not None:
                arrays["backward"]=flow.backward
            if flow.uncertainty is not None:
                arrays["uncertainty"]=flow.uncertainty
            overlays={}
            pose_path=Path(cfg.gavd_pose_root)/f"{row['sequence_id']}.npz" if cfg.gavd_pose_root else None
            if pose_path and pose_path.is_file():
                overlays=load_pose_overlay(pose_path,frames,decode_info["scale_xy"])
                arrays.update(overlays)
                for name,points in overlays.items():
                    diagnostics=joint_transport_diagnostics(flow,points,rgb=rgb)
                    arrays.update({f"{name}_{key}":value for key,value in diagnostics.items()})
                    valid=diagnostics["evidence_valid"]
                    record[f"{name}_transport_median_px"]=float(np.median(diagnostics["transport_error_px"][valid])) if valid.any() else np.nan
                    record[f"{name}_evidence_coverage"]=float(valid.mean())
            record["pose_overlay"]="provided_estimate" if overlays else "unavailable"
            np.savez_compressed(item/"flow-observations.npz",**arrays)
            sheet,video=_gallery(rgb,flow.forward,frames,overlays,item,cfg.fps)
            magnitude=np.linalg.norm(flow.forward,axis=-1)
            record.update(frames=len(rgb),source_fps=decode_info["source_fps"],
                          median_image_motion_px=float(np.median(magnitude)),
                          median_image_motion_px_per_s=float(np.median(magnitude/np.diff(timestamps)[:,None,None])),
                          contact_sheet=sheet,video=video)
            (item/"observation.json").write_text(json.dumps({**decode_info,**flow.metadata},indent=2)+"\n")
        except (ValueError,OSError) as error:
            record.update(status="unavailable",reason=str(error))
        rows.append(record)
    report=pd.DataFrame(rows)
    report.to_csv(folder/"stress-summary.csv",index=False)
    return report
