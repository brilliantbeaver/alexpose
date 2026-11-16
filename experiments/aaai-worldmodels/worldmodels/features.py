"""Exp5-style observed angle features on the exact JEPA windows.

Ambient supplies lower-limb geometry. Missing angles stay NaN instead of being
silently converted to zero by the legacy GaitFeatureVector convenience method.
"""
from __future__ import annotations
import numpy as np
from .data import JOINT_IDS


def _summary(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return [float(finite.mean()), float(finite.std()), float(np.ptp(finite))] if len(finite) else [np.nan]*3


def _angle(a, b, c):
    u, v = a-b, c-b
    norm = np.linalg.norm(u, axis=-1)*np.linalg.norm(v, axis=-1)
    cosine = np.divide((u*v).sum(-1), norm, out=np.full_like(norm,np.nan), where=norm>1e-8)
    return np.degrees(np.arccos(np.clip(cosine,-1,1)))


def build_features(x, valid=None, fps=15.0):
    """Return [N,37] features and names; x is [N,T,16,2] torso-normalized xy.

    `fps` can be one sampled rate per window. Image coordinates and projected
    angles do not establish calibrated physical speed or measured 3D anatomy.
    """
    from ambient.pose.joint_angles import get_joint_angles
    x = np.asarray(x, float)
    if x.ndim != 4 or x.shape[-2:] != (len(JOINT_IDS),2) or x.shape[1]<2:
        raise ValueError('Expected [N,T>=2,16,2] in JOINT_IDS order')
    valid = np.isfinite(x).all(-1) if valid is None else np.asarray(valid, bool) & np.isfinite(x).all(-1)
    if valid.shape != x.shape[:-1]:
        raise ValueError('Validity shape mismatch')
    rates = np.broadcast_to(np.asarray(fps,float), (len(x),))
    if np.any(~np.isfinite(rates) | (rates<=0)):
        raise ValueError('Sample rates must be positive')
    joints = ['left_hip','left_knee','left_ankle','right_hip','right_knee','right_ankle']
    names = [j+'_mean_deg' for j in joints] + [j+'_asymmetry_deg' for j in ('hip','knee','ankle')]
    names += [j+'_range_deg' for j in joints] + [j+'_std_deg' for j in joints]
    names += [side+'_elbow_'+stat+'_deg' for side in ('left','right') for stat in ('mean','std','range')]
    names += [side+'_foot_direction_'+stat+'_deg' for side in ('left','right') for stat in ('mean','std')]
    names += [side+'_wrist_speed_'+stat+'_torso_per_s' for side in ('left','right') for stat in ('mean','std')]
    names += ['ankle_separation_mean_torso','ankle_separation_std_torso']
    output, j = [], {joint:i for i,joint in enumerate(JOINT_IDS)}
    for window, observed, rate in zip(x, valid, rates):
        raw = np.zeros((len(window),33,3), float)
        raw[:,JOINT_IDS,:2] = np.nan_to_num(window)
        raw[:,JOINT_IDS,2] = observed
        frames = [[dict(x=float(k[0]), y=float(k[1]), confidence=float(k[2])) for k in frame] for frame in raw]
        sequence = get_joint_angles(frames, keypoint_format='BLAZEPOSE_33', confidence_threshold=0.5, fps=float(rate))
        statistics = np.asarray([_summary(sequence.get_joint_angle_series(name)) for name in joints])
        mean, std, extent = statistics.T
        row = list(mean) + list(np.abs(mean[:3]-mean[3:])) + list(extent) + list(std)
        coords = np.where(observed[...,None], window, np.nan)
        for a,b,c in ((11,13,15),(12,14,16)):
            row.extend(_summary(_angle(coords[:,j[a]],coords[:,j[b]],coords[:,j[c]])))
        for heel,toe in ((29,31),(30,32)):
            direction = coords[:,j[toe]]-coords[:,j[heel]]
            # Absolute projected foot-axis inclination, invariant to walking left/right.
            pitch = np.degrees(np.arctan2(np.abs(direction[:,1]),np.abs(direction[:,0])))
            pitch[np.linalg.norm(direction,axis=-1)<1e-8] = np.nan
            row.extend(_summary(pitch)[:2])
        for wrist in (15,16):
            velocity = np.linalg.norm(np.diff(coords[:,j[wrist]],axis=0),axis=-1)*rate
            row.extend(_summary(velocity)[:2])
        separation = np.linalg.norm(coords[:,j[27]]-coords[:,j[28]],axis=-1)
        row.extend(_summary(separation)[:2])
        output.append(row)
    return np.asarray(output,dtype=np.float32).reshape(len(x),len(names)), names
