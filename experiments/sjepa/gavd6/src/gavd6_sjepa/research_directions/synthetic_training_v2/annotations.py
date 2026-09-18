"""Blind human-reference manifests; no model output is exported to annotators."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .contracts import JOINTS,atomic_json,sha256_file,digest


def annotation_template(frames,output,*,split='development'):
    """Frames require recording/alias identity, true seconds and image paths.

    Blank coordinate fields are intentionally not model-generated references.
    Time logs and duplicate annotations are separate annotation work products.
    """
    if split!='development':raise PermissionError('Protected annotation export needs separately locked confirmation workflow')
    required={'recording_id','canonical_recording_id','frame_id','timestamp_s','image_path','split','exposure'}
    if required-set(frames):raise ValueError('Frame inventory lacks recording/time/exposure information')
    if not frames.split.eq(split).all() or frames.frame_id.duplicated().any():raise ValueError('Mixed splits or duplicate frames')
    if frames.timestamp_s.isna().any() or not np.isfinite(frames.timestamp_s).all():raise ValueError('Physical timestamps required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for record in frames.to_dict('records'):
        for joint in JOINTS:
            rows.append({**{k:record[k] for k in required},'joint':joint,'x_px':'','y_px':'','visible':'',
                         'annotator_id':'','annotation_seconds':'','independent_reference':True})
    pd.DataFrame(rows).to_csv(out/'landmarks-blank.csv',index=False)
    frames[list(sorted(required))].to_csv(out/'frame-manifest.csv',index=False)
    atomic_json(out/'instructions.json',dict(candidate_outputs_shown=False,ground_truth_kind='independent_human_visible_2d_reference_pending',
        first_development_clips=4,double_annotation_fraction=.20,
        rules=['Retain anatomical left/right; do not label hidden joints as visible.',
               'Use only source images, never candidate predictions.',
               'Record annotation time; use at least20% independently repeated clips.',
               'No cadence/preservation claim from isolated sparse frames.'],manifest_sha256=sha256_file(out/'frame-manifest.csv')))
    return out


def audit_annotations(path):
    table=pd.read_csv(path)
    required={'recording_id','frame_id','joint','timestamp_s','x_px','y_px','visible','annotator_id','annotation_seconds'}
    if required-set(table):raise ValueError('Missing annotation fields')
    if table[['frame_id','joint','annotator_id']].duplicated().any():raise ValueError('Duplicate annotation key')
    completed=table.annotator_id.notna() & table.visible.notna()
    visible=table.visible.astype(str).str.lower().isin(['true','1'])
    if (visible & (~np.isfinite(pd.to_numeric(table.x_px,errors='coerce'))|~np.isfinite(pd.to_numeric(table.y_px,errors='coerce')))).any():
        raise ValueError('Visible annotations require finite coordinates')
    counts=table.loc[completed].groupby('recording_id').frame_id.nunique()
    return dict(status='insufficient_evidence' if not completed.all() else 'reference_imported_pending_repeatability',
                rows=len(table),completed_rows=int(completed.sum()),recordings=int(table.recording_id.nunique()),
                minimum_annotated_frames=int(counts.min()) if len(counts) else 0,
                annotation_seconds=float(pd.to_numeric(table.annotation_seconds,errors='coerce').fillna(0).sum()),
                temporal_confirmation='pending independent dense reference support, event definitions and repeatability')


def freeze_confirmation(destination,*,method,input_manifest,configuration,decision,protocol_path,prediction_manifest):
    """Lock unlabelled manifests only; this operation NEVER reads reference labels."""
    if decision.get('status')!='pass' or decision.get('evidence_status')!='real-development':
        raise PermissionError('Confirmation freeze requires qualifying independent real-development decision')
    if not decision.get('independent_references_verified') or not decision.get('preservation_margins_calibrated'):
        raise PermissionError('Real references and preservation margins must precede confirmation')
    manifest=json.loads(Path(input_manifest).read_text())
    records=manifest['records']
    if not records or any(r['split']!='confirmation' or r['exposure']!='unexposed_verified' for r in records):
        raise PermissionError('Confirmation groups must have verified protected exposure')
    value=dict(method=method,configuration=configuration,decision=decision,
               input_manifest=str(Path(input_manifest).resolve()),input_hash=sha256_file(input_manifest),
               protocol=str(Path(protocol_path).resolve()),protocol_hash=sha256_file(protocol_path),
               prediction_manifest=str(Path(prediction_manifest).resolve()),prediction_hash=sha256_file(prediction_manifest),
               status='locked_unopened',labels_opened=False)
    value['lock_hash']=digest(value)
    p=Path(destination)
    if p.exists():raise FileExistsError('Confirmation lock is immutable')
    atomic_json(p,value);return value


def confirmation_access(lock_path,*,execute=False):
    """Verify lock before returning permission to a separately reviewed scorer.

    This first milestone does not implement automatic confirmation scoring.
    No reference loader is called here, even when execution is requested.
    """
    if not execute:raise PermissionError('Explicit confirmation execution is separate from freezing')
    lock=json.loads(Path(lock_path).read_text());saved=lock.pop('lock_hash')
    if digest(lock)!=saved:raise ValueError('Confirmation lock changed')
    for path_key,hash_key in [('input_manifest','input_hash'),('protocol','protocol_hash'),('prediction_manifest','prediction_hash')]:
        if sha256_file(lock[path_key])!=lock[hash_key]:raise ValueError('Confirmation input/protocol/predictions changed')
    raise NotImplementedError('Independent annotation import and confirmation scorer review remain pending; no labels opened')
