"""Recording-disjoint GAVD panels and independent human 2D references.

Clinical category labels are never converted into landmark coordinates. Model
predictions are never used to populate the annotation templates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import BOX_COLUMNS, KEYPOINT_NAMES, _read_labels, load_pose_manifest


VIEWS = ("side", "oblique", "frontal_rear")
REFERENCE_BOX_COLUMNS = ["box_x1", "box_y1", "box_x2", "box_y2"]


def _boolean(values, name):
    text = pd.Series(values).astype(str).str.strip().str.lower()
    if not text.isin(["true", "false", "1", "0"]).all():
        raise ValueError(f"{name} must be explicitly true/false or 1/0, with no blanks")
    return text.isin(["true", "1"]).to_numpy()


def eligible_gavd_sequences(cfg):
    """Exclude protected and unknown recording IDs using the established roster."""
    from ..motion_preservation.motion_data import load_gavd_manifest

    reservation_path = Path(cfg.gavd_reservation_csv).expanduser()
    if not cfg.gavd_reservation_csv or not reservation_path.is_file():
        raise FileNotFoundError("Set ST_GAVD_RESERVATION to the existing FI source-reservation.csv")
    roster = pd.read_csv(reservation_path, dtype={"video_id": str})
    if not {"video_id", "role"} <= set(roster) or not set(roster.role) <= {"development", "confirmation"}:
        raise ValueError("Reservation CSV must have video_id and development/confirmation role")
    # If a recording occurs twice, any protected assignment takes precedence.
    protected = set(roster.loc[roster.role.eq("confirmation"), "video_id"])
    permitted = set(roster.loc[roster.role.eq("development"), "video_id"]) - protected
    table = load_gavd_manifest(cfg.gavd_manifest_dir, cfg.gavd_video_root)
    table["excluded_reason"] = np.where(table.video_id.isin(protected), "protected_recording",
                                        np.where(~table.video_id.isin(permitted), "unknown_reservation", ""))
    folder = cfg.root / "data"
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "gavd-eligibility.csv", index=False)
    eligible = table.loc[table.excluded_reason.eq("")].copy()
    eligible.attrs["protected_video_ids"] = sorted(protected)
    return eligible


def create_gavd_view_template(cfg, table=None):
    """Export metadata for manual view, person-size, and fixed crop assignment.

    ``source_height`` is video height, not person resolution. The latter must be
    measured from the selected person's image box before inspecting model errors.
    """
    table = eligible_gavd_sequences(cfg) if table is None else table.copy()
    template = table[["sequence_id", "video_id", "video_path", "available", "first_frame", "last_frame", "cam_view"]].copy()
    template["coarse_view"] = ""
    template["view_checked"] = False
    template["person_height_px"] = np.nan
    for name in BOX_COLUMNS:
        template[name] = np.nan
    template["related_recording_id"] = template.video_id
    output = cfg.root / "data" / "gavd-view-template.csv"
    if not output.exists():
        template.to_csv(output, index=False)
    return output


def select_gavd_panel(table, assignments, counts=(3, 4, 6), seed=17):
    """Select six view-by-person-resolution collections from checked metadata.

    The median height boundary is fixed from all eligible checked assignments.
    A recording, including a manually identified related source, appears once.
    Inadequate cells raise an error rather than borrowing repeated recordings.
    """
    required = {"sequence_id", "coarse_view", "view_checked", "person_height_px", *BOX_COLUMNS}
    if not required <= set(assignments):
        raise ValueError(f"View assignments lack {sorted(required - set(assignments))}")
    if assignments.sequence_id.duplicated().any():
        raise ValueError("View assignments must contain each sequence_id at most once")
    checked = assignments.loc[_boolean(assignments.view_checked, "view_checked")].copy()
    if not set(checked.coarse_view) <= set(VIEWS):
        raise ValueError(f"Checked coarse_view must be one of {VIEWS}")
    # Retain the supplied, manually checked view and box values when merging.
    manual = assignments.loc[_boolean(assignments.view_checked, "view_checked")].copy()
    keep = ["sequence_id", "coarse_view", "person_height_px", *BOX_COLUMNS]
    if "related_recording_id" in manual:
        keep.append("related_recording_id")
    joined = table.merge(manual[keep], on="sequence_id", how="inner", validate="one_to_one")
    joined = joined.loc[joined.available].copy()
    heights = pd.to_numeric(joined.person_height_px, errors="coerce")
    boxes = joined[BOX_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if (joined.empty or not np.isfinite(heights).all() or (heights <= 0).any()
            or not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2])
            or np.any(boxes[:, :2] < 0)):
        raise ValueError("Checked, available clips need positive person heights and valid fixed image boxes")
    joined["person_height_px"] = heights
    # Randomized metadata order chooses one clip per related-recording group.
    # Keep all alias edges until grouping, including other clips of one video.
    joined = joined.sample(frac=1, random_state=seed)
    if "related_recording_id" not in joined:
        joined["related_recording_id"] = joined.video_id
    joined["related_recording_id"] = joined.related_recording_id.fillna("").astype(str)
    joined.loc[joined.related_recording_id.eq(""), "related_recording_id"] = joined.video_id
    # Resolve transitive aliases (A related to B, B related to C) before splitting.
    # A protected original source also protects an otherwise permitted re-upload.
    parents = {}
    def root(value):
        parents.setdefault(value, value)
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value
    for video_id, related_id in zip(joined.video_id.astype(str), joined.related_recording_id):
        first, second = root(video_id), root(related_id)
        if first != second:
            parents[max(first, second)] = min(first, second)
    protected = {root(str(value)) for value in table.attrs.get("protected_video_ids", [])}
    joined["related_recording_id"] = joined.video_id.astype(str).map(root)
    joined = joined.loc[~joined.related_recording_id.isin(protected)]
    joined = joined.drop_duplicates("related_recording_id")
    boundary = float(np.median(joined.person_height_px))
    joined["resolution_group"] = np.where(joined.person_height_px < boundary, "low", "high")
    joined["domain_id"] = joined.coarse_view + "_" + joined.resolution_group
    roles = ("context", "early", "confirmation")
    if len(counts) != 3 or min(counts) < 1:
        raise ValueError("Specify positive context, early and confirmation recording counts")
    selected = []
    for view in VIEWS:
        for resolution in ("low", "high"):
            domain = f"{view}_{resolution}"
            group = joined.loc[joined.domain_id.eq(domain)].copy()
            if len(group) < sum(counts):
                raise ValueError(f"{domain} has {len(group)} eligible recordings; needs {sum(counts)}. "
                                 "Complete more checked assignments or declare a smaller design before outcomes.")
            offset = 0
            for role, count in zip(roles, counts):
                rows = group.iloc[offset:offset + count].copy()
                rows["role"] = role
                selected.append(rows)
                offset += count
    result = pd.concat(selected, ignore_index=True)
    result["resolution_boundary_px"] = boundary
    result["group_unit"] = "recording_not_verified_person"
    return result


def _decode_frames(video_path, source_frames):
    """Read declared zero-based source indices without resizing image pixels."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open GAVD video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid source frame rate: {video_path}")
        first, last = int(source_frames[0]), int(source_frames[-1])
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        wanted = set(map(int, source_frames))
        for frame_index in range(first, last + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Video ended before requested frame {frame_index}: {video_path}")
            if frame_index in wanted:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames, fps


def write_annotation_page(index, output, split):
    """Write a small offline click-annotation tool with resumable human work.

    Image paths are relative to the page, so users can copy the whole data
    directory locally. Nothing runs a pose model or guesses hidden coordinates.
    Save progress as JSON, load it for an independent second pass, then export
    the exact long CSV consumed by :func:`import_gavd_annotations`.
    """
    import os

    output = Path(output)
    frames = [dict(frame_id=str(row.frame_id), image_path=os.path.relpath(row.image_path, output.parent),
                   identity_guide=[float(getattr(row, name)) for name in BOX_COLUMNS]
                   if all(hasattr(row, name) for name in BOX_COLUMNS) else None)
              for row in index.itertuples()]
    payload = json.dumps(dict(frames=frames, names=list(KEYPOINT_NAMES), split=split)).replace("</", "<\\/")
    page = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Independent GAVD reference annotation</title>
<style>
body{font:16px system-ui,sans-serif;max-width:1150px;margin:24px auto;padding:0 18px;color:#173047;background:#f5f8fb}
h1{font-size:25px;margin-bottom:10px}p{line-height:1.5}button,input,select{font:inherit;padding:7px 10px;margin:3px;border:1px solid #afbeca;border-radius:5px;background:white}
button{cursor:pointer}button:hover{background:#e0edf5}.row{display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin:8px 0}
#stage{background:#182633;text-align:center;padding:5px}canvas{max-width:100%;max-height:65vh;width:auto;height:auto;cursor:crosshair;vertical-align:middle}
#prompt{font-weight:650;color:#0b577c}#status{min-height:24px;color:#7d3e16}.small{font-size:14px;color:#42596c}label{white-space:nowrap}a{color:#0b577c}
</style>
<h1>Independent GAVD reference annotation</h1>
<p>Annotate only the selected person. The <b>blue dashed rectangle is an identity guide only</b>. Draw a separate <b>yellow reference box</b> around that person's full visible body with two corner clicks, then place the twelve named landmarks. Choose <b>Hidden / ambiguous</b> when a landmark cannot be located reliably. Reference coordinates and the yellow box come only from your clicks.</p>
<p class="small">Save a progress JSON regularly. A second annotator can load it, inspect each frame, correct labels if needed, and mark that frame reviewed. Copy this HTML file together with its <code>gavd/</code> image directory to use it locally. No images or labels leave your browser.</p>
<div class="row"><button id="previous">Previous frame</button><select id="frame"></select><button id="next">Next frame</button><span id="count"></span></div>
<div class="row"><label>Annotator <input id="annotator" placeholder="Your name or identifier"></label><label>Reviewer <input id="reviewer" placeholder="Independent reviewer"></label><button id="review">Mark frame reviewed</button></div>
<div class="row"><button id="box">Draw reference box</button><select id="joint"></select><button id="skip">Hidden / ambiguous (S)</button><button id="clear">Clear current landmark</button></div>
<p id="prompt"></p><div id="stage"><canvas id="canvas"></canvas></div><p id="status" role="status"></p>
<div class="row"><button id="save">Save progress JSON</button><label>Load progress <input id="load" type="file" accept=".json"></label><button id="export">Export reference CSV</button></div>
<p class="small">Landmark order follows the original COCO body labels: shoulders, elbows, wrists, hips, knees and ankles, left then right. “Left” refers to the person's left. Hidden landmarks are not scored. The confirmation CSV needs an independent reviewer for every frame.</p>
<script id="data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id), canvas=$('canvas'), ctx=canvas.getContext('2d');
let records=data.frames.map(f=>({frame_id:f.frame_id,box:null,points:Array(12).fill(null),annotator:'',reviewer:''}));
let position=0,joint=0,boxMode=true,firstCorner=null,picture=new Image(),lastAnnotator='';
data.frames.forEach((f,i)=>{const o=new Option((i+1)+' / '+data.frames.length+' : '+f.frame_id,i);$('frame').add(o);});
data.names.forEach((name,i)=>$('joint').add(new Option((i+1)+'. '+name.replaceAll('_',' '),i)));
function message(text){$('status').textContent=text;}
function draw(){
 ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(picture,0,0);
 const r=records[position],unit=Math.max(canvas.width/500,1);ctx.lineWidth=2*unit;ctx.font=(12*unit)+'px system-ui';
 const guide=data.frames[position].identity_guide;
 if(guide){ctx.strokeStyle='#67b8ff';ctx.setLineDash([7*unit,5*unit]);ctx.strokeRect(guide[0],guide[1],guide[2]-guide[0],guide[3]-guide[1]);ctx.setLineDash([]);ctx.fillStyle='#67b8ff';ctx.fillText('Identity guide only',guide[0]+3*unit,Math.max(14*unit,guide[1]-5*unit));}
 if(r.box){ctx.strokeStyle='#ffe66d';ctx.strokeRect(r.box[0],r.box[1],r.box[2]-r.box[0],r.box[3]-r.box[1]);}
 r.points.forEach((p,i)=>{if(p&&p.visible){ctx.fillStyle=i===joint?'#ffbc55':'#35ebcb';ctx.beginPath();ctx.arc(p.x,p.y,3.5*unit,0,2*Math.PI);ctx.fill();ctx.strokeStyle='#112530';ctx.strokeText(String(i+1),p.x+5*unit,p.y-5*unit);ctx.fillText(String(i+1),p.x+5*unit,p.y-5*unit);}});
 if(firstCorner){ctx.fillStyle='#ffe66d';ctx.fillRect(firstCorner[0]-3*unit,firstCorner[1]-3*unit,6*unit,6*unit);}
 $('joint').value=String(Math.min(joint,11));
 $('prompt').textContent=boxMode?(firstCorner?'Click the opposite reference-box corner.':'Click the first reference-box corner.'):(joint<12?'Place '+data.names[joint].replaceAll('_',' ')+', or mark it hidden / ambiguous.':'Twelve visibility decisions complete. Check the box and annotation, then go to the next frame.');
 $('count').textContent=records.filter(r=>r.box&&r.points.every(Boolean)&&r.annotator).length+' / '+records.length+' frames annotated';
}
function loadFrame(){
 const r=records[position];$('frame').value=String(position);$('annotator').value=r.annotator||lastAnnotator;$('reviewer').value=r.reviewer;
 joint=r.points.findIndex(p=>p===null);if(joint<0)joint=12;boxMode=!r.box;firstCorner=null;
 picture=new Image();picture.onload=()=>{canvas.width=picture.naturalWidth;canvas.height=picture.naturalHeight;draw();};
 picture.onerror=()=>message('Image not found. Copy the gavd/ folder alongside this page, preserving relative paths.');
 picture.src=data.frames[position].image_path;message('');
}
function author(){const name=$('annotator').value.trim();if(!name){message('Enter the human annotator name before adding labels.');return false;}lastAnnotator=name;records[position].annotator=name;records[position].reviewer='';$('reviewer').value='';return true;}
canvas.addEventListener('click',event=>{
 if(!picture.complete||!picture.naturalWidth||!author())return;
 const rect=canvas.getBoundingClientRect(),x=Math.min(canvas.width-1,Math.max(0,(event.clientX-rect.left)*canvas.width/rect.width)),y=Math.min(canvas.height-1,Math.max(0,(event.clientY-rect.top)*canvas.height/rect.height));
 if(boxMode){if(!firstCorner)firstCorner=[x,y];else{const b=[Math.min(firstCorner[0],x),Math.min(firstCorner[1],y),Math.max(firstCorner[0],x),Math.max(firstCorner[1],y)];if(b[2]-b[0]<1||b[3]-b[1]<1){message('Draw a nonzero reference box.');return;}records[position].box=b;boxMode=false;firstCorner=null;}}
 else if(joint<12){records[position].points[joint]={x,y,visible:true};joint++;}draw();
});
$('box').onclick=()=>{boxMode=true;firstCorner=null;draw();};
$('joint').onchange=()=>{joint=Number($('joint').value);boxMode=false;firstCorner=null;draw();};
$('skip').onclick=()=>{if(joint<12&&author()){records[position].points[joint]={x:null,y:null,visible:false};joint++;boxMode=false;draw();}};
$('clear').onclick=()=>{if(joint<12&&author()){records[position].points[joint]=null;draw();}};
function navigate(delta){position=Math.max(0,Math.min(records.length-1,position+delta));loadFrame();}
$('previous').onclick=()=>navigate(-1);$('next').onclick=()=>navigate(1);$('frame').onchange=()=>{position=Number($('frame').value);loadFrame();};
$('review').onclick=()=>{const r=records[position],name=$('reviewer').value.trim();if(!name||name===r.annotator){message('Use a different, identified human reviewer.');return;}if(!r.box||!r.points.every(Boolean)||!r.annotator){message('Complete this frame before marking it reviewed.');return;}r.reviewer=name;message('This frame marked reviewed by '+name+'.');};
function download(name,text,mime){const link=document.createElement('a'),url=URL.createObjectURL(new Blob([text],{type:mime}));link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('save').onclick=()=>download('gavd-'+data.split+'-progress.json',JSON.stringify({split:data.split,records},null,2),'application/json');
$('load').onchange=async()=>{try{const file=$('load').files[0];if(!file)return;const saved=JSON.parse(await file.text());if(saved.split!==data.split||!Array.isArray(saved.records)||saved.records.length!==records.length||saved.records.some((r,i)=>r.frame_id!==data.frames[i].frame_id||!Array.isArray(r.points)||r.points.length!==12))throw Error('Progress belongs to another panel or has an invalid structure.');records=saved.records;loadFrame();message('Progress loaded.');}catch(error){message(error.message);}};
$('export').onclick=()=>{
 if(records.some(r=>!r.box||!r.points.every(Boolean)||!r.annotator||(data.split==='confirmation'&&!r.reviewer))){message('Finish every frame, including boxes and required confirmation review, before exporting. Save JSON to pause.');return;}
 const columns=['frame_id','landmark','x','y','visible','box_x1','box_y1','box_x2','box_y2','annotator','reviewer'];
 const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"',rows=[columns.map(quote).join(',')];
 records.forEach(r=>r.points.forEach((p,j)=>rows.push([r.frame_id,data.names[j],p.x,p.y,p.visible,...r.box,r.annotator,r.reviewer].map(quote).join(','))));
 download('gavd-'+data.split+'-annotations.csv',rows.join('\n')+'\n','text/csv');message('CSV exported. Keep confirmation labels separate from early evaluation.');
};
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;if(e.key.toLowerCase()==='s'){$('skip').click();e.preventDefault();}else if(e.key==='ArrowRight')navigate(1);else if(e.key==='ArrowLeft')navigate(-1);});
loadFrame();
</script></html>'''
    output.write_text(page.replace("__DATA__", payload))
    return output


def prepare_gavd_panel(cfg):
    """Export unlabeled context, evaluation images, and blank human templates.

    First call without ST_GAVD_VIEWS writes the view template and returns None.
    Once checked assignments are supplied, all selected frames are fixed using
    metadata alone. The early and confirmation label templates are separate.
    """
    from PIL import Image

    table = eligible_gavd_sequences(cfg)
    create_gavd_view_template(cfg, table)
    if not cfg.gavd_view_csv:
        return None
    assignments = pd.read_csv(Path(cfg.gavd_view_csv).expanduser(), keep_default_na=False)
    panel = select_gavd_panel(table, assignments,
                             (cfg.gavd_context_recordings, cfg.gavd_early_recordings,
                              cfg.gavd_confirmation_recordings), cfg.seed)
    folder = cfg.root / "data"
    panel.to_csv(folder / "gavd-panel.csv", index=False)
    records, annotations = [], {"early": [], "confirmation": []}
    for row in panel.to_dict("records"):
        first, last = int(row["first_frame"]) - 1, int(row["last_frame"]) - 1
        if first < 0 or last < first:
            raise ValueError("GAVD manifest frame indices must be one-based inclusive")
        if row["role"] == "context":
            # Preserve a short contiguous motion segment, not a montage over a long recording.
            import cv2
            capture = cv2.VideoCapture(str(row["video_path"]))
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
            finally:
                capture.release()
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"Cannot read source FPS: {row['video_path']}")
            offsets = np.rint(np.arange(cfg.clip_frames) * fps / cfg.clip_fps).astype(int)
            if len(np.unique(offsets)) != len(offsets):
                raise ValueError("clip_fps exceeds source FPS; choose a lower context sampling rate")
            start = first + max(0, (last - first - int(offsets[-1])) // 2)
            frames = start + offsets
            if frames[-1] > last:
                raise ValueError(f"Selected GAVD interval too short for context: {row['sequence_id']}")
        else:
            # Annotate four frames within one short clip, not across unrelated
            # moments of a long sequence interval. Clip location uses metadata.
            import cv2
            capture = cv2.VideoCapture(str(row["video_path"]))
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
            finally:
                capture.release()
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"Cannot read source FPS: {row['video_path']}")
            span = min(last - first, int(round((cfg.clip_frames - 1) * fps / cfg.clip_fps)))
            start = first + (last - first - span) // 2
            frames = np.rint(np.linspace(start, start + span, cfg.gavd_annotation_frames)).astype(int)
            if len(np.unique(frames)) != cfg.gavd_annotation_frames:
                raise ValueError(f"Insufficient distinct annotation frames: {row['sequence_id']}")
        images, fps = _decode_frames(row["video_path"], frames)
        clip_id = f"gavd_{row['sequence_id']}"
        item = folder / "gavd" / row["role"] / clip_id
        item.mkdir(parents=True, exist_ok=True)
        for frame, image in zip(frames, images):
            h, w = image.shape[:2]
            bbox = np.asarray([row[name] for name in BOX_COLUMNS], float)
            if bbox[2] > w or bbox[3] > h:
                raise ValueError(f"Fixed person crop exceeds original image size: {row['sequence_id']}")
            frame_id = f"{clip_id}_f{frame:07d}"
            path = item / f"{frame:07d}.png"
            Image.fromarray(image).save(path)
            records.append(dict(
                frame_id=frame_id, clip_id=clip_id, frame_index=int(frame), image_path=str(path),
                label_path="", label_index=0, **dict(zip(BOX_COLUMNS, bbox)), width=w, height=h,
                person_id="", motion_id=str(row["sequence_id"]), recording_id=str(row["video_id"]),
                domain_id=row["domain_id"], role=row["role"], lesson_id="", source_path=row["video_path"],
                source_fps=fps, timestamp_s=float(frame / fps), related_recording_id=row["related_recording_id"],
            ))
            if row["role"] != "context":
                for landmark in KEYPOINT_NAMES:
                    annotations[row["role"]].append(dict(
                        frame_id=frame_id, image_path=str(path), landmark=landmark,
                        x="", y="", visible="", box_x1="", box_y1="", box_x2="", box_y2="",
                        annotator="", reviewer="",
                    ))
    index = pd.DataFrame(records)
    context_path, evaluation_path = folder / "gavd_context.csv", folder / "gavd_evaluation.csv"
    index.loc[index.role.eq("context")].to_csv(context_path, index=False)
    index.loc[~index.role.eq("context")].to_csv(evaluation_path, index=False)
    for split, rows in annotations.items():
        path = folder / f"gavd-{split}-annotation-template.csv"
        if not path.exists():
            pd.DataFrame(rows).to_csv(path, index=False)
        write_annotation_page(index.loc[index.role.eq(split)], folder / f"annotate-{split}.html", split)
    (folder / "gavd-panel-notes.json").write_text(json.dumps({
        "resolution_boundary_px": float(panel.resolution_boundary_px.iloc[0]),
        "source_frames": "zero-based; original manifest is one-based inclusive",
        "group_unit": "recording, not verified person identity",
        "reference_status": "blank templates require independent human annotation",
        "input_crops": "fixed checked clip boxes, distinct from per-frame reference scale boxes",
    }, indent=2) + "\n")
    return context_path, evaluation_path


def annotation_path_for_split(cfg, split):
    """Resolve exactly one label file, without opening the other reference split."""
    if split not in {"early", "confirmation"}:
        raise ValueError("Reference split must be early or confirmation")
    supplied = str(cfg.gavd_annotations_csv)
    if not supplied:
        raise ValueError("Set ST_GAVD_ANNOTATIONS to a {split} filename pattern or annotation directory")
    path = Path(supplied.format(split=split)).expanduser()
    if path.is_dir():
        path = path / f"gavd-{split}-annotations.csv"
    elif "{split}" not in supplied and split not in path.stem:
        raise ValueError("A single reference file must identify its split in its name; "
                         "prefer a {split} pattern to keep early and confirmation labels separate")
    if not path.is_file():
        raise FileNotFoundError(f"Independent {split} reference annotations are missing: {path}")
    return path


def import_gavd_annotations(cfg, split="early") -> pd.DataFrame:
    """Import only the requested independently annotated reference partition.

    Required long-table fields: frame_id, landmark, x, y, visible, box_x1,
    box_y1, box_x2, box_y2, annotator, reviewer. A frame has twelve landmark rows;
    repeat its independently annotated box on each row. Hidden/ambiguous joints
    have visible=false and may leave coordinates blank. Confirmation requires
    an identified second reviewer. Input estimator crops remain unchanged.
    """
    index = load_pose_manifest(cfg.root / "data" / "gavd_evaluation.csv", roles=[split])
    annotations = pd.read_csv(annotation_path_for_split(cfg, split), keep_default_na=False)
    required = {"frame_id", "landmark", "x", "y", "visible", *REFERENCE_BOX_COLUMNS, "annotator", "reviewer"}
    if not required <= set(annotations):
        raise ValueError(f"Reference annotations lack {sorted(required - set(annotations))}")
    if set(annotations.frame_id) != set(index.frame_id):
        raise ValueError(f"The {split} annotation file must contain exactly its fixed evaluation frames")
    folder = cfg.root / "data" / "gavd_labels" / split
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in index.to_dict("records"):
        labels = annotations.loc[annotations.frame_id.eq(frame["frame_id"])].copy()
        if len(labels) != 12 or set(labels.landmark) != set(KEYPOINT_NAMES):
            raise ValueError(f"Need each of the twelve landmark names exactly once: {frame['frame_id']}")
        labels = labels.set_index("landmark").loc[list(KEYPOINT_NAMES)]
        if labels.annotator.astype(str).str.strip().eq("").any():
            raise ValueError(f"Record the independent human annotator: {frame['frame_id']}")
        if split == "confirmation" and labels.reviewer.astype(str).str.strip().eq("").any():
            raise ValueError(f"Confirmation annotations need a second reviewer: {frame['frame_id']}")
        reviewer = labels.reviewer.astype(str).str.strip().str.casefold()
        annotator = labels.annotator.astype(str).str.strip().str.casefold()
        if (reviewer.ne("") & reviewer.eq(annotator)).any():
            raise ValueError(f"The second reviewer must differ from the original annotator: {frame['frame_id']}")
        visible = _boolean(labels.visible, "visible")
        xy = labels[["x", "y"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        if (not visible.any() or not np.isfinite(xy[visible]).all()
                or np.any(xy[visible] < 0)
                or np.any(xy[visible, 0] >= frame["width"])
                or np.any(xy[visible, 1] >= frame["height"])):
            raise ValueError(f"Visible landmarks must lie inside the original image: {frame['frame_id']}")
        boxes = labels[REFERENCE_BOX_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        box = boxes[0]
        if (not np.isfinite(boxes).all() or not np.allclose(boxes, box)
                or np.any(box[2:] <= box[:2]) or np.any(box[:2] < 0)
                or box[2] > frame["width"] or box[3] > frame["height"]):
            raise ValueError(f"Repeat one valid, independent reference person box on all landmark rows: {frame['frame_id']}")
        label_path = folder / f"{frame['frame_id']}.npz"
        np.savez_compressed(label_path, keypoints=xy, visible=visible)
        frame.update(label_path=str(label_path), label_index=0,
                     reference_box_x1=box[0], reference_box_y1=box[1],
                     reference_box_x2=box[2], reference_box_y2=box[3],
                     reference_scale=float(np.linalg.norm(box[2:] - box[:2])),
                     annotation_source="independent_human_body12")
        rows.append(frame)
    result = pd.DataFrame(rows)
    result.to_csv(cfg.root / "data" / f"gavd_{split}_labeled.csv", index=False)
    _read_labels.cache_clear()
    return result
