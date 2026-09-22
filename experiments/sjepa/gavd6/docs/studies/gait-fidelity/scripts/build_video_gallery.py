"""Build a local-only gallery from the audited video manifest, without copying videos.

Run: .venv/bin/python docs/studies/gait-fidelity/scripts/build_video_gallery.py
Optional --thumbnails generates ignored local JPEG previews with ffmpeg.
"""
from pathlib import Path
from html import escape
from urllib.parse import quote
import argparse
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "data"
MS = ROOT.parents[4] / 'multiple-sclerosis'


def build(thumbnails=False):
    rows=json.loads((ROOT/'evidence/video-inventory.json').read_text())
    posters=ROOT/'images/local-video-review'
    posters.mkdir(exist_ok=True)
    (posters/'.gitignore').write_text('*.jpg\n')
    cards=[]
    for i,r in enumerate(rows):
        p=MS/r['path']
        if not p.is_file(): raise FileNotFoundError(p)
        duration=float(r.get('duration') or r['container_duration'])
        poster=posters/(p.stem+'.jpg')
        if thumbnails and not poster.exists():
            subprocess.run(['ffmpeg','-v','error','-ss',str(duration*.5),'-i',str(p),'-frames:v','1',
                            '-vf','scale=480:270:force_original_aspect_ratio=decrease',
                            '-q:v','3','-y',str(poster)],check=True,capture_output=True)
        url=quote(os.path.relpath(p,PAGE),safe='/')
        poster_url=quote(os.path.relpath(poster,PAGE),safe='/')
        fps=r['avg_frame_rate']
        display=f'{duration:.2f} s · {r["width"]} × {r["height"]} · fps {fps}'
        card=f'''<article data-condition="{escape(r['condition_label'])}" data-search="{escape((p.stem+' '+r['source_id']).lower())}">
<h2>{escape(p.name)}</h2><p><span class="condition">Folder: {escape(r['condition_label'])} · </span>Source: {escape(r['source_id'])}<br>{escape(display)}</p>
<video controls preload="none" playsinline poster="{poster_url}" src="{url}"></video>
<p><a href="{url}">Open original video</a> · <button type="button" class="stamp" data-i="{i}">Insert current time</button></p>
<label for="notes-{i}">Observations and usable interval</label><textarea id="notes-{i}" data-i="{i}" rows="3" placeholder="E.g. 0.5–3.2 s: feet visible; side uncertain; cut at 3.3 s"></textarea>
<details><summary>Source checksum</summary><code>{r['sha256']}</code></details></article>'''
        cards.append(card)
    data=[{k:r[k] for k in ['path','source_id','condition_label','sha256']} for r in rows]
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Gait Fidelity — local video review</title>
<style>body{max-width:1240px;margin:28px auto;padding:0 20px;color:#183247;background:#f2f6f9;font:16px/1.55 Arial,sans-serif}h1{font-size:30px}h2{font-size:18px;overflow-wrap:anywhere}a{color:#326bb2}.toolbar{position:sticky;top:0;background:#eaf1f7;padding:16px;border:1px solid #ccd8e3;z-index:2;display:flex;flex-wrap:wrap;gap:14px;align-items:center}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px;margin-top:22px}article{padding:18px;background:white;border:1px solid #d5e0e7;border-radius:10px}video{width:100%;height:230px;object-fit:contain;background:#16232d}input,select,button,textarea{font:inherit}textarea{box-sizing:border-box;width:100%;margin-top:6px}button{cursor:pointer}details{font-size:12px;margin-top:12px}code{overflow-wrap:anywhere}.blind .condition{display:none}[hidden]{display:none!important}.notice{background:#fff5e6;padding:12px 16px;border-left:4px solid #a56819}</style></head><body class="blind">
<h1>Gait Fidelity: inspect the local footage</h1><p><a href="local-videos.md">Audit and interpretation</a> · <a href="../methods/masking.md">Proposed masking experiment</a> · <a href="../images/gallery.html">Conceptual figures</a></p>
<p>91 clips from 41 filename-derived sources, totaling 9.21 minutes. Play the original files before deciding which intervals are usable. A thumbnail is one midpoint frame. Folder names are dataset labels; source IDs are not verified patient identities.</p>
<p class="notice">This page reads local files and sends no data to a server. Video and optional thumbnail links require the sibling multiple-sclerosis checkout. Public redistribution rights and clinical references remain unverified. The gallery has no reference or restored-pose overlay. Display-label hiding is a convenience, not a blinded annotation protocol.</p>
<div class="toolbar"><label>Folder <select id="condition"><option value="">All</option><option>MS</option><option>PD</option><option>Normal</option></select></label><label>Find source <input id="search" type="search" size="16"></label><label><input id="hide-labels" type="checkbox" checked> Hide folder labels</label><label>Reviewer <input id="reviewer" size="12"></label><button id="export" type="button">Export notes (JSON)</button><span id="count"></span></div>
<p id="save-status">Notes are stored in this browser when local storage is available. Export them to retain a portable review record.</p><div class="grid">'''+''.join(cards)+'''</div><script>
const inventory=INVENTORY;
const key='gait-fidelity-video-review-20260921-v1';
let saved={};
try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){document.getElementById('save-status').textContent='Browser storage unavailable: export your notes before closing this page.'}
document.getElementById('reviewer').value=saved.reviewer||'';
for(const node of document.querySelectorAll('textarea')) node.value=(saved.notes||{})[inventory[+node.dataset.i].path]||'';
function snapshot(){const notes={};for(const node of document.querySelectorAll('textarea')){if(node.value.trim()) notes[inventory[+node.dataset.i].path]=node.value}return {reviewer:document.getElementById('reviewer').value,notes};}
function retain(){try{localStorage.setItem(key,JSON.stringify(snapshot()))}catch(e){document.getElementById('save-status').textContent='Notes could not be saved in this browser. Export before closing.'}}
document.addEventListener('input',e=>{if(e.target.matches('textarea,#reviewer'))retain()});
function filter(){const c=document.getElementById('condition').value;const q=document.getElementById('search').value.trim().toLowerCase();let n=0;for(const card of document.querySelectorAll('article')){card.hidden=Boolean((c&&card.dataset.condition!==c)||(q&&!card.dataset.search.includes(q)));if(!card.hidden)n++;else card.querySelector('video').pause()}document.getElementById('count').textContent=n+' / '+inventory.length+' clips';}
document.getElementById('condition').addEventListener('change',filter);document.getElementById('search').addEventListener('input',filter);filter();
document.getElementById('hide-labels').addEventListener('change',e=>document.body.classList.toggle('blind',e.target.checked));
for(const button of document.querySelectorAll('.stamp'))button.addEventListener('click',()=>{const card=button.closest('article');const note=card.querySelector('textarea');note.value+=(note.value?'\n':'')+card.querySelector('video').currentTime.toFixed(3)+' s: ';note.focus();retain()});
document.getElementById('export').addEventListener('click',()=>{const state=snapshot();const payload={created_utc:new Date().toISOString(),scope:'Raw-video feasibility notes, not clinical ground truth',reviewer:state.reviewer,clips:inventory.filter(r=>state.notes[r.path]).map(r=>({...r,note:state.notes[r.path]}))};const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));link.download='gait-fidelity-video-review.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)});
</script></body></html>'''
    # Escape a JS newline represented in the Python triple-quoted template.
    page=page.replace("?'\n':''", "?'\\n':''")
    page=page.replace('INVENTORY',json.dumps(data).replace('</','<\\/'))
    (PAGE/'video-gallery.html').write_text(page)
    print(f'Gallery: {len(rows)} original videos; {sum((posters/(Path(r["path"]).stem+".jpg")).exists() for r in rows)} local thumbnails.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thumbnails',action='store_true')
    build(parser.parse_args().thumbnails)
