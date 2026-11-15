"""Normal-only interpolation and held-out benefit diagnostics, no refitting."""
from pathlib import Path
import argparse,sys,json
import numpy as np,pandas as pd,joblib
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments/aaai'))
from aaai_gait.iteration_data import load_iteration_data
from aaai_gait.retained_prediction import iter_feature_batches
from aaai_gait.study_runner import dump_json
from aaai_gait.data import file_sha256
parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();p=args.run
manifest=json.loads((p/'run_manifest.json').read_text());data=load_iteration_data(manifest['cohort_directory']);rows=[]
for fold in range(4):
    for internal in range(3):
        bundle=joblib.load(p/f'fold_{fold}/inner_{internal}_models.joblib')
        q=data.metadata.source_id.isin(bundle['validation_sources']).to_numpy();x=data.x[q];meta=data.metadata.loc[q].reset_index(drop=True);parts=[]
        for a,f in iter_feature_batches(x,batch_examples=8192):
            a['source_id']=meta.source_id.to_numpy()[a.window_index]
            a['interpolation']=np.square(f.own[:,:8]-f.target).mean(1)
            a['ridge']=np.square(bundle['base'].predict(f.own)-f.target).mean(1);parts.append(a)
        z=pd.concat(parts).groupby(['source_id','joint'])[['interpolation','ridge']].mean().groupby('source_id').mean().reset_index()
        z['outer_fold']=fold;z['inner_fold']=internal;rows.append(z)
pd.concat(rows,ignore_index=True).to_csv(p/'interpolation_normal_adequacy.csv',index=False)
scores=pd.read_csv(p/'source_scores.csv');rows=[]
for (policy,bank),table in scores[scores.binary_label.eq(0)].groupby(['policy','bank']):
    rng=np.random.default_rng(42);boot=np.zeros(5000);means={}
    for fold,group in table.groupby('outer_fold'):
        values=group[['left_benefit','right_benefit']].mean(axis=1).to_numpy();means[int(fold)]=float(values.mean())
        boot+=len(group)/len(table)*values[rng.integers(len(values),size=(5000,len(values)))].mean(1)
    rows.append(dict(policy=policy,bank=bank,n_normal=len(table),mean_benefit=float(table[['left_benefit','right_benefit']].mean(axis=1).mean()),
        conditional_95_interval=np.quantile(boot,[.025,.975]).tolist(),fold_mean_benefit=means,
        bootstrap='5000 paired source draws within each outer fold, original source-count weights',
        interpretation='conditional fixed-model held-out normal prediction diagnostic'))
dump_json(p/'normal_test_benefit.json',rows)
for name in ['interpolation_normal_adequacy.csv','normal_test_benefit.json']:
    manifest['artifacts'][name]={'path':name,'sha256':file_sha256(p/name)}
dump_json(p/'run_manifest.json',manifest)
