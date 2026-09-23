"""Portable review pages showing videos, estimates, references and missingness."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

EDGES = ((0, 2), (2, 4), (1, 3), (3, 5), (0, 1), (0, 6), (1, 7),
         (6, 7), (6, 8), (8, 10), (7, 9), (9, 11))


def save_video(images, path, *, fps=25):
    """Encode actual rendered RGB without changing frame count or elapsed time."""
    frames = np.asarray(images)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Video encoding expects RGB uint8[N,H,W,3]")
    executable = shutil.which("ffmpeg")
    if executable is None: raise RuntimeError("ffmpeg is required to retain reviewable source videos; load the HAIC FFmpeg module")
    if not np.isfinite(fps) or fps <= 0: raise ValueError("Positive video rate required")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(f"Video already exists: {path}")
    h, w = frames.shape[1:3]
    # CRF18 is a review artifact; estimation always uses the original RGB arrays.
    command = [executable, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "18",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-n", str(path)]
    completed = subprocess.run(command, input=frames.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode:
        raise RuntimeError(f"Review-video encoding failed: {completed.stderr.decode(errors='replace')}")
    return path


def save_contact_sheet(render, path):
    from PIL import Image, ImageDraw
    frames = render["images"]
    positions = np.linspace(0, len(frames) - 1, 4).round().astype(int)
    canvas = Image.new("RGB", (2 * frames.shape[2], 2 * frames.shape[1]))
    for k, frame in enumerate(positions):
        image = Image.fromarray(frames[frame]); draw = ImageDraw.Draw(image)
        xy = render["keypoints"][frame]
        for a, b in EDGES:
            if np.isfinite(xy[[a, b]]).all(): draw.line([tuple(xy[a]), tuple(xy[b])], fill=(255, 230, 70), width=3)
        for j, point in enumerate(xy):
            if not np.isfinite(point).all(): continue
            x, y = map(float, point)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(40, 160, 255) if j % 2 == 0 else (255, 90, 50))
        draw.text((12, 12), f"Frame {frame} | reference L=blue, R=orange", fill="white", stroke_width=2, stroke_fill="black")
        canvas.paste(image, (k % 2 * image.width, k // 2 * image.height))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path)
    return path


def plot_trajectories(bundle, index, predictions=None, output=None):
    """Plot raw pixel trajectories with gaps, physical seconds and anatomical side."""
    import matplotlib.pyplot as plt
    index = int(index)
    t = bundle.inputs["timestamps"][index]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    for row, (joint, side) in enumerate(((10, "Left ankle"), (11, "Right ankle"))):
        for dim, label in enumerate(("x", "y")):
            ax = axes[row, dim]
            reference = np.where(bundle.targets["valid"][index, :, joint], bundle.targets["xy"][index, :, joint, dim], np.nan)
            ax.plot(t, reference, label="Projected reference", color="black", linewidth=1.6)
            ax.plot(t, bundle.inputs["xy"][index, :, joint, dim], label="Estimator input", alpha=.65)
            for name, xy in (predictions or {}).items(): ax.plot(t, np.asarray(xy)[index, :, joint, dim], label=name, alpha=.8)
            ax.set_ylabel(f"{side} {label} (pixels)"); ax.grid(alpha=.2)
    for ax in axes[-1]: ax.set_xlabel("Original source time (seconds)")
    axes[0, 0].legend(fontsize=8)
    r = bundle.records[index]
    fig.suptitle(f"{r['source_family_id']} · {r['movement_state']} {r['movement_magnitude']:g}° · {r['camera_id']} · {r['naming']} · {r['extractor']}")
    fig.tight_layout()
    if output:
        path = Path(output); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=140)
    return fig


def _finite_json(value):
    if isinstance(value, np.ndarray): return _finite_json(value.tolist())
    if isinstance(value, dict): return {str(k): _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    if isinstance(value, np.generic): return value.item()
    return value


def build_viewer(bundle, output, predictions=None, *, max_tracks=12, indices=None):
    """Self-contained HTML controls; videos use local relative links, no CDN.

    The default selection uses metadata order, never measured model performance.
    Provide explicit ``indices`` when reviewing a frozen selection manifest.
    """
    from .data import validate_bundle
    validate_bundle(bundle)
    output = Path(output)
    if output.suffix.lower() != ".html": output = output / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    if indices is None:
        # Spread a metadata-only sample over the retained file, preserving exact
        # indices in the viewer artifact for repeatable follow-up review.
        count = min(int(max_tracks), len(bundle.records))
        indices = np.unique(np.linspace(0, len(bundle.records) - 1, count).round().astype(int)).tolist()
    rows = []
    for i in indices:
        r = bundle.records[int(i)]
        video = str(r.get("video_path", ""))
        video_url = os.path.relpath(video, output.parent) if video else ""
        rows.append(dict(index=int(i), record=r, video=video_url,
            times=bundle.inputs["timestamps"][i], input=bundle.inputs["xy"][i],
            observed=bundle.inputs["observed"][i], confidence=bundle.inputs["confidence"][i],
            reference=bundle.targets["xy"][i], valid=bundle.targets["valid"][i], visible=bundle.targets["visible"][i],
            predictions={name: np.asarray(xy)[i] for name, xy in (predictions or {}).items()}))
    payload = json.dumps(_finite_json(dict(rows=rows, edges=EDGES, evidence_status=bundle.evidence_status)), allow_nan=False).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gait Fidelity · synchronized data review</title><style>
body{font:16px system-ui,sans-serif;margin:24px auto;max-width:1200px;padding:0 20px;color:#20303c;background:#f7f9fc}h1{font-size:26px}p{line-height:1.5}button,select,input{font:inherit;padding:7px;margin:5px}canvas{max-width:100%;background:#fff;border:1px solid #cad4dc;border-radius:6px}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}.grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(240px,1fr);gap:20px}@media(max-width:800px){.grid{grid-template-columns:1fr}}#time{font-variant-numeric:tabular-nums}video{display:none}.legend span{margin-right:16px}label{display:inline-block}</style>
<h1>Gait Fidelity: synchronized data review</h1>
<p id="evidence"></p><p>Solid reference bones are shown in black; estimated left joints are blue and right joints are orange. Missing estimated joints remain absent. A translucent ring marks a valid reference hidden from view. Reference visibility is a synthetic depth approximation. The horizontal ankle traces below retain the original timestamps.</p>
<select id="track"></select><button id="play">Play</button><input id="frame" type="range" min="0" value="0"><span id="time"></span>
<div class="grid"><div><video id="video" muted playsinline preload="metadata"></video><canvas id="pose" width="640" height="480"></canvas><canvas id="trace" width="640" height="220"></canvas></div><div><label><input id="reference" type="checkbox" checked>Reference</label><label><input id="estimated" type="checkbox" checked>Estimator</label><pre id="meta"></pre></div></div>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.querySelector('#data').textContent), sel=document.querySelector('#track'), slider=document.querySelector('#frame'), video=document.querySelector('#video'), pose=document.querySelector('#pose'), trace=document.querySelector('#trace');
let row,playing=false,last=0;const good=p=>p&&p.length===2&&p.every(Number.isFinite);
data.rows.forEach((x,i)=>{const o=document.createElement('option');o.value=i;const r=x.record;o.textContent=`${x.index}: ${r.canonical_person_id} · ${r.movement_state} ${r.movement_magnitude}° · ${r.physical_state} · ${r.camera_id} · ${r.naming} · ${r.observation} · ${r.extractor}`;sel.append(o)});
document.querySelector('#evidence').textContent=`Evidence: ${data.evidence_status}. Examples were selected by metadata, without model scores.`;
function select(){row=data.rows[Number(sel.value)];slider.max=row.times.length-1;slider.value=0;video.src=row.video;document.querySelector('#meta').textContent=JSON.stringify(row.record,null,2);draw()}
function skeleton(ctx,xy,color,width){ctx.lineWidth=width;ctx.strokeStyle=color;for(const [a,b] of data.edges){if(!good(xy[a])||!good(xy[b]))continue;ctx.beginPath();ctx.moveTo(...xy[a]);ctx.lineTo(...xy[b]);ctx.stroke()}}
function draw(){if(!row)return;const f=Number(slider.value),ctx=pose.getContext('2d');ctx.clearRect(0,0,640,480);ctx.fillStyle='#edf1f5';ctx.fillRect(0,0,640,480);if(row.video&&video.readyState>=2)ctx.drawImage(video,0,0,640,480);const target=row.reference[f],input=row.input[f];
if(document.querySelector('#reference').checked){skeleton(ctx,target,'#111827',2);target.forEach((p,j)=>{if(!good(p)||!row.valid[f][j])return;ctx.beginPath();ctx.arc(p[0],p[1],row.visible[f][j]?2.5:6,0,Math.PI*2);ctx.strokeStyle=row.visible[f][j]?'#111827':'#64748baa';ctx.lineWidth=2;ctx.stroke()})}
if(document.querySelector('#estimated').checked){skeleton(ctx,input,'#64748b',1.2);input.forEach((p,j)=>{if(!good(p)||!row.observed[f][j])return;ctx.beginPath();ctx.arc(p[0],p[1],4,0,Math.PI*2);ctx.fillStyle=j%2?'#e97825':'#1178bd';ctx.fill()})}
Object.entries(row.predictions).forEach(([name,xy],i)=>skeleton(ctx,xy[f],['#a855f7','#16a34a','#db2777'][i%3],2));document.querySelector('#time').textContent=`Frame ${f} · source ${row.times[f].toFixed(3)} s`;drawTrace(f)}
function drawTrace(f){const c=trace.getContext('2d');c.clearRect(0,0,640,220);const a=45,b=620,top=20,bottom=185;const vals=[];[row.reference,row.input].forEach(arr=>arr.forEach(x=>[10,11].forEach(j=>{if(good(x[j]))vals.push(x[j][0])})));const lo=Math.min(...vals)-5,hi=Math.max(...vals)+5;const px=i=>a+i*(b-a)/(row.times.length-1),py=v=>bottom-(v-lo)*(bottom-top)/(hi-lo);c.strokeStyle='#ddd';c.strokeRect(a,top,b-a,bottom-top);for(const j of [10,11]){for(const [arr,dash]of [[row.reference,[]],[row.input,[3,3]]]){c.beginPath();c.setLineDash(dash);c.strokeStyle=j===10?'#1178bd':'#e97825';let started=false;arr.forEach((x,i)=>{if(!good(x[j])){started=false;return}if(started)c.lineTo(px(i),py(x[j][0]));else c.moveTo(px(i),py(x[j][0]));started=true});c.stroke()}}c.setLineDash([]);c.strokeStyle='#111';c.beginPath();c.moveTo(px(f),top);c.lineTo(px(f),bottom);c.stroke();c.fillStyle='#20303c';c.font='12px sans-serif';c.fillText('Horizontal ankle position: solid reference, dashed input',45,210)}
function seek(){if(row.video){video.currentTime=Math.max(0,row.times[Number(slider.value)]-row.times[0])}else draw()}
sel.onchange=select;slider.oninput=seek;video.onseeked=draw;video.onloadeddata=draw;document.querySelector('#reference').onchange=draw;document.querySelector('#estimated').onchange=draw;
document.querySelector('#play').onclick=()=>{playing=!playing;document.querySelector('#play').textContent=playing?'Pause':'Play'};
function tick(now){if(playing&&row&&now-last>1000*(row.times[1]-row.times[0])){slider.value=(Number(slider.value)+1)%row.times.length;seek();last=now}requestAnimationFrame(tick)}select();requestAnimationFrame(tick);
</script></html>'''.replace("__DATA__", payload)
    output.write_text(page)
    (output.parent / (output.stem + "-selection.json")).write_text(json.dumps(dict(indices=indices, selection="metadata_spread_no_model_scores", evidence_status=bundle.evidence_status), indent=2))
    return output
