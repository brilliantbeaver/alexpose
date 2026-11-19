"""Publish derived tables and hashes after all scientific runs."""
from pathlib import Path
import argparse,sys,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments/aaai'))
from aaai_gait.iteration_scoring import ranking_metrics,endpoint_frame,collapse_recording_duplicates
from aaai_gait.study_runner import dump_json
from aaai_gait.data import file_sha256
parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();p=args.run
s=pd.read_csv(p/'source_scores.csv');metrics=pd.read_csv(p/'metrics.csv');detail=json.loads((p/'rawmetrics.json').read_text());extra=[]
columns=['R','R_raw','base_error_distance','own_error_distance','both_error_distance','stgnf_adapted']
for (policy,bank),group in s.groupby(['policy','bank']):
    for endpoint in ['focal','broad']:
        t=collapse_recording_duplicates(endpoint_frame(group,endpoint),columns)
        for method in columns:
            value=ranking_metrics(t,method);detail[f'{policy}/{bank}/{endpoint}'][method]=value
            for metric,field in [('auroc','weighted_auroc'),('average_precision','equal_fold_average_precision')]:
                if field in value:extra.append(dict(policy=policy,bank=bank,endpoint=endpoint,method=method,metric=metric,value=value[field],n_sources=value['n_sources'],status=value['status']))
metrics=pd.concat([metrics[~metrics.method.isin(columns)],pd.DataFrame(extra)],ignore_index=True)
metrics.to_csv(p/'metrics.csv',index=False);dump_json(p/'rawmetrics.json',detail)
normal=s[s.binary_label.eq(0)];table=[]
for (policy,bank,fold),g in normal.groupby(['policy','bank','outer_fold']):
    row=dict(policy=policy,bank=bank,outer_fold=int(fold),n_normal=len(g))
    for loss in ['loss_base','loss_own','loss_both','benefit']:
        row[loss]=float(g[[f'left_{loss}',f'right_{loss}']].mean(axis=1).mean())
    table.append(row)
pd.DataFrame(table).to_csv(p/'normal_error_table.csv',index=False)
references=[]
for fold in range(4):
    base=s[s.outer_fold.eq(fold)&s.policy.eq('U')&s.bank.eq('U')]
    perturb=pd.read_csv(p/f'fold_{fold}/U_U/reference_sensitivity.csv',keep_default_na=False)
    for (removed,floor),q in perturb.groupby(['removed_reference','variance_floor']):
        z=base.drop(columns=['R','S']).merge(q[['source_id','R']],on='source_id',validate='one_to_one');z['S']=.5*(z.M+z.R)
        for endpoint in ['focal','broad']:
            value=ranking_metrics(endpoint_frame(z,endpoint),'S')
            references.append(dict(outer_fold=fold,removed_reference=removed,variance_floor=floor,endpoint=endpoint,
                                   auroc=value.get('weighted_auroc'),status=value['status']))
pd.DataFrame(references).to_csv(p/'reference_sensitivity_metrics.csv',index=False)
baseline=json.loads((p/'baseline_status.json').read_text());baseline['local_interpolation']='Completed12 internal normal diagnostics; shared ridge beats interpolation in each outer aggregate. See interpolation_normal_adequacy.csv.'
dump_json(p/'baseline_status.json',baseline)
manifest=json.loads((p/'run_manifest.json').read_text());manifest['status']='complete'
names=[x.name for x in p.iterdir() if x.is_file() and x.suffix in ['.csv','.json','.npz'] and x.name not in ['run_manifest.json','verification.json','artifact_hashes.json']]
manifest['artifacts']={n:{'path':n,'sha256':file_sha256(p/n)} for n in names}
manifest['artifacts']['manifest.csv']={'path':manifest['cohort_directory']+'/manifest.csv','sha256':file_sha256(Path(manifest['cohort_directory'])/'manifest.csv')}
dump_json(p/'run_manifest.json',manifest)
dump_json(p/'artifact_hashes.json',{str(f.relative_to(p)):file_sha256(f) for f in sorted(p.rglob('*')) if f.is_file() and f.name not in ['artifact_hashes.json','run_manifest.json','verification.json'] and 'figures' not in f.parts and 'notebook_records' not in f.parts})
print('Final evidence manifest',p)
