"""Audit sibling video files into a separate output directory without changing them."""
import json,csv,subprocess,hashlib,collections,statistics,re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
root=Path(__file__).resolve().parents[6] / 'multiple-sclerosis'
out=Path(__file__).resolve().parent
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
out=parser.parse_args().output
out.mkdir(parents=True,exist_ok=True)

def audit(p):
 b=p.read_bytes(); pointer=b.startswith(b'version https://git-lfs.github.com/spec/v1')
 r={'path':str(p.relative_to(root)),'condition_label':p.parent.name,'source_id':p.stem[:11],'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'lfs_pointer':pointer}
 if not pointer:
  x=subprocess.run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(p)],capture_output=True,text=True)
  if x.returncode: r['probe_error']=x.stderr
  else:
   d=json.loads(x.stdout); v=next(s for s in d['streams'] if s['codec_type']=='video'); r.update({k:v.get(k) for k in ['codec_name','width','height','r_frame_rate','avg_frame_rate','nb_frames','pix_fmt','duration','time_base','start_time']});r['container_duration']=d['format'].get('duration');r['audio_streams']=sum(s['codec_type']=='audio' for s in d['streams']);r['rotation']=v.get('tags',{}).get('rotate',v.get('side_data_list',[]))
 return r
rows=list(ThreadPoolExecutor(max_workers=8).map(audit,sorted((root/'video-data-full').rglob('*.mp4'))))
(out/'video-inventory.json').write_text(json.dumps(rows,indent=2)+'\n')
keys=sorted(set(k for r in rows for k in r));f=(out/'video-inventory.csv').open('w');w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows);f.close()
groups={}
for c in ['MS','PD','Normal','ALL']:
 a=rows if c=='ALL' else [r for r in rows if r['condition_label']==c];ds=[float(r.get('duration') or r['container_duration']) for r in a if not r.get('probe_error')]; hashes=collections.Counter(r['sha256'] for r in a)
 groups[c]={'clips':len(a),'sources':len(set(r['source_id'] for r in a)),'bytes':sum(r['bytes'] for r in a),'duration_seconds_sum':sum(ds),'duration_seconds_min':min(ds),'duration_seconds_median':statistics.median(ds),'duration_seconds_max':max(ds),'avg_frame_rates':dict(collections.Counter(r.get('avg_frame_rate') for r in a)),'resolutions':dict(collections.Counter(f"{r.get('width')}x{r.get('height')}" for r in a)),'lfs_pointers':sum(r['lfs_pointer'] for r in a),'probe_errors':[r for r in a if r.get('probe_error')],'exact_duplicate_groups':[[r['path'] for r in a if r['sha256']==h] for h,n in hashes.items() if n>1],'clips_by_source':dict(collections.Counter(r['source_id'] for r in a))}
(out/'video-summary.json').write_text(json.dumps(groups,indent=2)+'\n');print(json.dumps(groups,indent=2))
