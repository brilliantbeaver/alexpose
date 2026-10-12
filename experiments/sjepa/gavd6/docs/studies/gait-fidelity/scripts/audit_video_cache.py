"""Audit sibling pose caches into a separate output directory without changing them."""
import numpy as np,json,hashlib,collections,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[6] / 'multiple-sclerosis';out=Path(__file__).resolve().parent
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
out=parser.parse_args().output
out.mkdir(parents=True,exist_ok=True)

rows=[]
for p in sorted((root/'artifacts/keypoints-full').glob('*.npz')):
 a=np.load(p,allow_pickle=False);k=a['keypoints'];n=a['keypoints_norm'];rows.append({'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'keys':a.files,'shape':list(k.shape),'frames':len(k),'fps':int(a['fps']),'source_id':str(a['source_id']),'label':str(a['label']),'clip_name':str(a['clip_name']),'all_coordinates_finite':bool(np.isfinite(k[:,:,:2]).all()),'has_timestamps':any('timestamp' in s for s in a.files),'has_imputation_mask':any('mask' in s or 'interpol' in s for s in a.files),'visibility_mean':float(k[:,:,2].mean()),'visibility_lt_05_fraction':float((k[:,:,2]<.5).mean())})
(out/'cache-inventory.json').write_text(json.dumps(rows,indent=2)+'\n')
print({'count':len(rows),'sources':len(set(r['source_id'] for r in rows)),'fps':dict(collections.Counter(r['fps'] for r in rows)),'total_frames':sum(r['frames'] for r in rows),'min_frames':min(r['frames'] for r in rows),'median_frames':statistics.median(r['frames'] for r in rows),'max_frames':max(r['frames'] for r in rows),'all_coordinates_finite':all(r['all_coordinates_finite'] for r in rows),'timestamp_files':sum(r['has_timestamps'] for r in rows),'imputation_mask_files':sum(r['has_imputation_mask'] for r in rows)})
print('low-visibility examples',[(r['clip_name'],round(r['visibility_lt_05_fraction'],3)) for r in sorted(rows,key=lambda r:r['visibility_lt_05_fraction'],reverse=True)[:7]])
