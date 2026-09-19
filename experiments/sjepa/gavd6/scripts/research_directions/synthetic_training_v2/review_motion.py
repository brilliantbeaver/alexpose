#!/usr/bin/env python3
"""Create bounded, offline SMPL-H playback for a human locomotion review.

This tool reads source assets and writes a new review directory. It never edits
the audit, reservations, preparation configuration, or a source motion. Playback
is review evidence, not a locomotion or eligibility decision.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys


ROOT = Path(__file__).resolve().parents[3]
FPS = 25
FRAMES = 64
MAX_WINDOWS = 32


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True,
                        help="Saved config/preparation.json; review paths need not be complete")
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--relative-path", help="One exact AMASS manifest relative_path")
    source.add_argument("--audit-csv", type=Path,
                        help="Candidate CSV with relative_path,start_s; human fields may be blank")
    result.add_argument("--start-s", type=float, action="append",
                        help="Repeat for each window with --relative-path; not used with --audit-csv")
    result.add_argument("--output-dir", type=Path, required=True,
                        help="New directory; an existing path is never overwritten")
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return result


def _requests(args):
    if args.audit_csv is not None:
        if args.start_s is not None:
            raise ValueError("--start-s is only valid with --relative-path")
        with args.audit_csv.expanduser().open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not {"relative_path", "start_s"} <= set(reader.fieldnames or []):
                raise ValueError("Candidate CSV requires relative_path,start_s columns")
            values = [(row["relative_path"], row["start_s"]) for row in reader]
    else:
        if not args.start_s:
            raise ValueError("--relative-path requires at least one --start-s")
        values = [(args.relative_path, start) for start in args.start_s]
    if not 1 <= len(values) <= MAX_WINDOWS:
        raise ValueError(f"Request between 1 and {MAX_WINDOWS} windows; got {len(values)}")
    requests, seen = [], set()
    for relative, raw_start in values:
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("Every requested window needs a nonempty relative_path")
        relative = relative.strip()
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative or path.as_posix() != relative:
            raise ValueError(f"Use an exact, normalized AMASS relative_path: {relative!r}")
        try:
            start = float(raw_start)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid start_s for {relative}: {raw_start!r}") from exc
        if not math.isfinite(start) or start < 0:
            raise ValueError(f"start_s must be finite and nonnegative: {raw_start!r}")
        if (relative, start) in seen:
            raise ValueError(f"Duplicate requested window: {relative} at {start:g}s")
        seen.add((relative, start))
        requests.append((relative, start))
    return requests


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_views(joints):
    """One orientation for the whole window, so turns and root travel remain visible."""
    import numpy as np

    lateral = joints[0, 2] - joints[0, 1]
    lateral[1] = 0
    length = float(np.linalg.norm(lateral))
    if length < 1e-6:
        raise ValueError("Initial hip direction is degenerate; cannot define review views")
    lateral = lateral / length
    up = np.array([0.0, 1.0, 0.0])
    forward = np.cross(lateral, up)
    origin = joints[0, 0].copy()
    origin[1] = 0
    centered = joints - origin
    front = np.stack([centered @ lateral, joints[:, :, 1]], axis=-1)
    side = np.stack([centered @ forward, joints[:, :, 1]], axis=-1)
    return front, side, dict(lateral=lateral.tolist(), forward=forward.tolist(),
                            origin=origin.tolist(), up=up.tolist())


def _write_html(path, payload):
    # Escape '<' in JSON so even an unusual local filename cannot close a script.
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    document = _HTML.replace("__PAYLOAD__", data)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(document)


def create_review(args):
    requests = _requests(args)
    config_path = args.config.expanduser().resolve(strict=True)
    config = json.loads(config_path.read_text())
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Review output already exists; choose a new directory: {output}")
    for name in ("amass_root", "body_model_root", "manifest_dir", "dmpl_root"):
        if not config.get(name):
            if name == "dmpl_root":
                continue
            raise ValueError(f"Preparation configuration needs {name}")
        if output.is_relative_to(Path(config[name]).expanduser().resolve()):
            raise ValueError(f"Review output must be outside the source asset directory {name}")
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from gavd6_sjepa.research_directions.motion_preservation.body_geometry import (
        JOINT_NAMES, PARENTS, SMPLHBody,
    )
    from gavd6_sjepa.research_directions.motion_preservation.motion_data import (
        load_amass_manifest, load_motion,
    )

    table = load_amass_manifest(config["manifest_dir"], config["amass_root"])
    root = Path(config["amass_root"]).expanduser().resolve(strict=True)
    selected = []
    # Resolve all requests before loading a model or writing any output.
    for relative, start in requests:
        rows = table.loc[table.relative_path.eq(relative)]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one approved manifest row for {relative}; got {len(rows)}")
        row = rows.iloc[0].to_dict()
        if row["original_split"] not in {"train", "validation"}:
            raise ValueError(f"Review helper excludes original test people: {relative}")
        raw_path = Path(row["raw_path"]).resolve(strict=True)
        if not raw_path.is_relative_to(root) or not raw_path.is_file():
            raise ValueError(f"Motion must be a file within configured AMASS root: {raw_path}")
        end = start + (FRAMES - 1) / FPS
        if not math.isfinite(float(row["duration_s"])) or end > float(row["duration_s"]) + 1e-9:
            raise ValueError(f"Window {start:g}–{end:g}s exceeds manifest duration for {relative}")
        selected.append((row, start, raw_path))

    body = SMPLHBody(config["body_model_root"], config.get("dmpl_root"),
                     device=args.device, batch_size=16)
    if output.is_relative_to(body.dmpl_root.resolve()):
        raise ValueError("Review output must be outside the inferred DMPL source directory")
    windows, positions, timestamps, hashes = [], [], [], {}
    for index, (row, start, raw_path) in enumerate(selected, start=1):
        print(f"Review window {index}/{len(selected)}: {row['relative_path']} @ {start:g}s", flush=True)
        source_hash = hashes.setdefault(str(raw_path), _sha256(raw_path))
        motion = load_motion(row, start_s=start, duration_s=FRAMES / FPS, fps=FPS)
        expected_times = start + np.arange(FRAMES) / FPS
        if motion.poses.shape != (FRAMES, 156) or not np.allclose(
                motion.timestamps, expected_times, rtol=0, atol=1e-9):
            raise ValueError("Motion loader did not supply the required 64 samples at 25 Hz")
        sequence = body.forward(motion)
        joints = np.asarray(sequence.joints, dtype=np.float32)
        if joints.shape != (FRAMES, 22, 3) or not np.isfinite(joints).all():
            raise ValueError("SMPL-H playback requires finite [64,22,3] joint positions")
        if sequence.coordinate_system != "y_up":
            raise ValueError("SMPL-H playback requires the documented y_up coordinates")
        front, side, view = _fixed_views(joints)
        windows.append(dict(
            window=index, relative_path=str(row["relative_path"]), raw_path=str(raw_path),
            person_id=str(row["person_id"]), original_split=str(row["original_split"]),
            gender=motion.gender, start_s=start, end_s=float(expected_times[-1]),
            frames=FRAMES, fps=FPS, source_fps=motion.metadata["source_fps"],
            source_sha256=source_hash, timestamps=expected_times.tolist(),
            evidence_reference=f"{output / 'review.html'}#window={index}",
            view_coordinates=view, front=front.tolist(), side=side.tolist(),
        ))
        positions.append(joints)
        timestamps.append(expected_times)
    for raw_path, digest in hashes.items():
        if _sha256(raw_path) != digest:
            raise RuntimeError(f"Source changed during review generation: {raw_path}")

    output.mkdir(parents=True, exist_ok=False)
    trace = output / "joints.npz"
    with trace.open("xb") as stream:
        np.savez_compressed(stream, joints=np.stack(positions), timestamps=np.stack(timestamps),
                            parents=PARENTS, joint_names=np.asarray(JOINT_NAMES))
    payload = dict(windows=windows, parents=PARENTS.tolist(), joint_names=list(JOINT_NAMES))
    _write_html(output / "review.html", payload)
    metadata = dict(
        schema="stv2-motion-review-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        status="playback_created_human_review_required", device=args.device,
        geometry="SMPL-H+DMPL", coordinates="AMASS converted once to Y up; meters",
        views="Fixed at initial hip direction; root travel retained; equal scales in both views",
        config=str(config_path), config_sha256=_sha256(config_path),
        helper_sha256=_sha256(Path(__file__)), joints_sha256=_sha256(trace),
        audit_csv=str(args.audit_csv.expanduser().resolve()) if args.audit_csv is not None else None,
        body_model_root=str(Path(config["body_model_root"]).expanduser().resolve()),
        dmpl_root=str(body.dmpl_root),
        decisions="No locomotion, reservation, canonical identity or exposure decisions are made",
        windows=[{key: value for key, value in window.items() if key not in {"front", "side"}}
                 for window in windows],
    )
    with (output / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Playback: {output / 'review.html'}", flush=True)
    print("Human review is still required; no audit or reservation file was changed.", flush=True)
    return metadata


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        create_review(args)
    except Exception as exc:
        print(f"STV2 motion review: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


_HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AMASS motion review</title>
<style>
*{box-sizing:border-box}body{font:16px/1.5 system-ui,sans-serif;background:#f4f6fa;color:#142032;margin:0}
main{max-width:1140px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 8px}p{margin:8px 0}
.notice{background:#fff4cf;padding:12px 16px;border-left:4px solid #b87c00;border-radius:4px}
.box{background:white;border:1px solid #d5dce7;border-radius:8px;padding:16px;margin-top:16px}
label{display:inline-block;margin:4px 16px 4px 0}select,button,input{font:inherit}select{max-width:100%;padding:6px}
button{padding:7px 18px;cursor:pointer}#window{width:100%}#scrub{width:100%;margin:12px 0}
.mono{font:13px/1.5 ui-monospace,monospace;overflow-wrap:anywhere}#canvas{display:block;width:100%;height:auto}
.small{font-size:14px;color:#46566b}#clock{font-variant-numeric:tabular-nums;font-weight:650}
.legend{display:flex;gap:22px;flex-wrap:wrap}.left{color:#1769c4}.right{color:#bd471d}
</style><main>
<h1>AMASS motion review</h1>
<p class="notice" id="notice">Playback only. Confirm locomotion and record your observations separately. This page does not approve people, reservations, exposure or audit rows.</p>
<section class="box"><label for="window">Window</label><select id="window"></select>
<p id="source" class="mono"></p><p id="person" class="small"></p>
<canvas id="canvas" width="1080" height="560" aria-label="Synchronized front and side skeleton views"></canvas>
<div class="legend small"><span class="left">Blue: anatomical left</span><span class="right">Orange: anatomical right</span><span>Gray: trunk / full pelvis trajectory</span></div>
<input id="scrub" type="range" min="0" max="63" step="1" value="0" aria-label="Frame">
<button id="play" type="button">Play</button>
<label for="speed">Speed <select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option></select></label>
<span id="clock" aria-live="off"></span>
<p class="small">64 samples at 25 Hz. Views keep the initial hip orientation; turns and root travel remain visible. Both views use the same fixed meter scale. The gray line marks world height 0, not an inferred floor.</p>
</section><section class="box"><p id="instruction"><strong>Record after inspection:</strong> reviewer, review date, whether the full interval contains suitable locomotion, and observed transitions or corruption.</p>
<p>Evidence reference:</p><p id="evidence" class="mono"></p><p id="hash" class="mono small"></p>
<p class="small">The companion metadata.json records the exact source and sample times; joints.npz preserves the derived 3D coordinates. No external network or browser plugin is needed.</p></section>
</main><script type="application/json" id="data">__PAYLOAD__</script><script>
"use strict";
const data=JSON.parse(document.getElementById("data").textContent);
if(data.review_mode==="automated_development"){
  document.getElementById("notice").textContent="Automated development screen. Playback is retained for optional inspection. No human locomotion review or external reservation verification is claimed.";
  document.getElementById("instruction").textContent="screen.json records the heuristic thresholds and measured values. Passing this screen is not a validated locomotion label or confirmatory evidence.";
}
const choose=document.getElementById("window"), scrub=document.getElementById("scrub"),
      play=document.getElementById("play"), speed=document.getElementById("speed"),
      canvas=document.getElementById("canvas"), ctx=canvas.getContext("2d");
let selected=0, frame=0, playing=false, last=null, carry=0, bounds;
const left=new Set([1,4,7,10,13,16,18,20]), right=new Set([2,5,8,11,14,17,19,21]);
data.windows.forEach((w,i)=>{const o=document.createElement("option");o.value=String(i);
o.textContent=`${w.window}. ${w.person_id} · ${w.original_split} · ${w.start_s.toFixed(2)}–${w.end_s.toFixed(2)} s`;choose.appendChild(o);});
function fit(w){const views=[w.front,w.side], flat=views.map(v=>v.flat()),
  xmin=flat.map(v=>Math.min(...v.map(p=>p[0]))), xmax=flat.map(v=>Math.max(...v.map(p=>p[0]))),
  ymin=Math.min(0,...flat.flat().map(p=>p[1])), ymax=Math.max(...flat.flat().map(p=>p[1])),
  span=Math.max(1.8,ymax-ymin,...xmax.map((x,i)=>x-xmin[i]))*1.18;
return {span,xc:xmax.map((x,i)=>(x+xmin[i])/2),yc:(ymin+ymax)/2};}
function draw(){const w=data.windows[selected];ctx.fillStyle="#fff";ctx.fillRect(0,0,1080,560);
  const side=440, top=74, scale=side/bounds.span;
  [w.front,w.side].forEach((view,v)=>{const ox=50+540*v,
    xy=p=>[ox+side/2+(p[0]-bounds.xc[v])*scale,top+side/2-(p[1]-bounds.yc)*scale];
    ctx.fillStyle="#142032";ctx.font="600 18px system-ui";ctx.textAlign="center";
    ctx.fillText(v===0?"Front view (initial orientation)":"Side view (initial orientation)",ox+side/2,30);
    ctx.font="13px system-ui";ctx.fillStyle="#52627a";ctx.fillText(v===0?"Lateral position (m)":"Forward position (m)",ox+side/2,548);
    ctx.save();ctx.beginPath();ctx.rect(ox,top,side,side);ctx.clip();
    ctx.strokeStyle="#e5eaf1";ctx.lineWidth=1;ctx.font="11px system-ui";ctx.textAlign="left";
    const step=bounds.span>6?1:.5;
    for(let y=Math.ceil((bounds.yc-bounds.span/2)/step)*step;y<=bounds.yc+bounds.span/2;y+=step){
      const py=xy([0,y])[1];ctx.beginPath();ctx.moveTo(ox,py);ctx.lineTo(ox+side,py);ctx.stroke();
      ctx.fillStyle="#52627a";ctx.fillText(`${y.toFixed(1)} m`,ox+4,py-3);}
    for(let x=Math.ceil((bounds.xc[v]-bounds.span/2)/step)*step;x<=bounds.xc[v]+bounds.span/2;x+=step){
      const px=xy([x,0])[0];ctx.beginPath();ctx.moveTo(px,top);ctx.lineTo(px,top+side);ctx.stroke();}
    ctx.strokeStyle="#9aa8b8";ctx.beginPath();ctx.moveTo(ox,xy([0,0])[1]);ctx.lineTo(ox+side,xy([0,0])[1]);ctx.stroke();
    ctx.strokeStyle="#c4ccd8";ctx.lineWidth=2;ctx.beginPath();view.forEach((pose,t)=>{const p=xy(pose[0]);t?ctx.lineTo(...p):ctx.moveTo(...p);});ctx.stroke();
    for(let j=1;j<data.parents.length;j++){const a=xy(view[frame][data.parents[j]]),b=xy(view[frame][j]);
      ctx.strokeStyle=left.has(j)?"#1769c4":right.has(j)?"#bd471d":"#445368";ctx.lineWidth=4;
      ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
    view[frame].forEach((p,j)=>{const q=xy(p);ctx.fillStyle=left.has(j)?"#1769c4":right.has(j)?"#bd471d":"#445368";
      ctx.beginPath();ctx.arc(q[0],q[1],3.5,0,Math.PI*2);ctx.fill();});ctx.restore();
    ctx.strokeStyle="#cdd6e2";ctx.lineWidth=1;ctx.strokeRect(ox,top,side,side);
  });
  scrub.value=String(frame);document.getElementById("clock").textContent=
    `Source ${w.timestamps[frame].toFixed(2)} s · frame ${frame+1}/64`;
}
function stop(){playing=false;last=null;carry=0;play.textContent="Play";}
function select(index){stop();selected=index;frame=0;choose.value=String(index);const w=data.windows[index];bounds=fit(w);
  document.getElementById("source").textContent=w.relative_path;
  document.getElementById("person").textContent=`${w.person_id} · original split: ${w.original_split} · ${w.gender} SMPL-H + DMPL`;
  document.getElementById("evidence").textContent=w.evidence_reference;
  document.getElementById("hash").textContent=`Source SHA256: ${w.source_sha256}`;draw();}
function fromHash(){const m=location.hash.match(/^#window=(\d+)$/),index=m?Number(m[1])-1:0;
  select(index>=0&&index<data.windows.length?index:0);}
choose.addEventListener("change",()=>{location.hash=`window=${Number(choose.value)+1}`;select(Number(choose.value));});
window.addEventListener("hashchange",fromHash);
scrub.addEventListener("input",()=>{stop();frame=Number(scrub.value);draw();});
play.addEventListener("click",()=>{playing=!playing;last=null;carry=0;play.textContent=playing?"Pause":"Play";});
speed.addEventListener("change",()=>{last=null;carry=0;});
function tick(now){if(playing){if(last!==null){carry+=Math.min(now-last,250)*25*Number(speed.value)/1000;
  const advance=Math.floor(carry);if(advance){frame=(frame+advance)%64;carry-=advance;draw();}}last=now;}requestAnimationFrame(tick);}
fromHash();requestAnimationFrame(tick);
</script></html>'''


if __name__ == "__main__":
    raise SystemExit(main())
