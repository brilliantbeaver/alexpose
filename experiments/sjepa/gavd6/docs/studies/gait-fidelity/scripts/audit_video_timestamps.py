"""Audit decoded presentation times into a separate output directory."""
import json,subprocess,collections,statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path(__file__).resolve().parents[6] / 'multiple-sclerosis';out=Path(__file__).resolve().parent
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
out=parser.parse_args().output
out.mkdir(parents=True,exist_ok=True)
rows=json.loads((out/'video-inventory.json').read_text())
def timing(r):
 cmd=['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','csv=p=0',str(root/r['path'])]
 a=subprocess.run(cmd,capture_output=True,text=True)
 t=[]
 for line in a.stdout.splitlines():
  try:t.append(float(line.split(',')[0]))
  except ValueError:pass
 ds=[t[i+1]-t[i] for i in range(len(t)-1)]; rate=float(__import__('fractions').Fraction(r['avg_frame_rate'])); stride=max(1,round(rate/15))
 return {'path':r['path'],'source_id':r['source_id'],'condition_label':r['condition_label'],'decoded_timestamp_count':len(t),'source_average_fps':rate,'code_stride':stride,'approx_retained_fps_if_constant':rate/stride,'positive_deltas_ms_min':1000*min(ds) if ds else None,'positive_deltas_ms_max':1000*max(ds) if ds else None,'median_delta_ms':1000*statistics.median(ds) if ds else None,'delta_values_ms_rounded':dict(collections.Counter(round(1000*d,3) for d in ds)),'nonincreasing_count':sum(d<=0 for d in ds),'decode_stderr':a.stderr.strip(),'returncode':a.returncode}
a=list(ThreadPoolExecutor(max_workers=6).map(timing,rows)); (out/'timestamps-audit.json').write_text(json.dumps(a,indent=2)+'\n')
print('decoded frames',sum(r['decoded_timestamp_count'] for r in a),'files',len(a),'errors',[r['path'] for r in a if r['returncode'] or r['decode_stderr']]);print('retained fps range',min(r['approx_retained_fps_if_constant'] for r in a),max(r['approx_retained_fps_if_constant'] for r in a));print('below13',[(r['path'],round(r['source_average_fps'],4),round(r['approx_retained_fps_if_constant'],4)) for r in a if r['approx_retained_fps_if_constant']<13]);print('PTS range variation >1ms',[(r['path'],r['positive_deltas_ms_min'],r['positive_deltas_ms_max']) for r in a if r['positive_deltas_ms_max']-r['positive_deltas_ms_min']>1])
