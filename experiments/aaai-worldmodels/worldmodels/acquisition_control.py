"""Source-grouped acquisition-only binary controls, with a sealed test release."""
from pathlib import Path
import argparse,json,joblib,importlib.metadata
import numpy as np
import pandas as pd
from .data import sha256_file
from .settings import PROJECT
from .workflow import load_prepared,json_write
from .evaluation import fit_probe,align_probabilities,aggregate_sources,classification_report

CLASSES=('normal','abnormal')
FEATURES={'prefix_quality':['sample_fps','prefix_valid_fraction','prefix_mean_confidence'],
          'retrospective_quality':['sample_fps','prefix_valid_fraction','prefix_mean_confidence','future_valid_fraction','future_mean_confidence']}

def develop(prepared,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    _,_,m=load_prepared(prepared);m=m[m.split.isin(['train','validation'])].copy()
    y=np.where(m.label=='normal','normal','abnormal');train=(m.split=='train').to_numpy();val=(m.split=='validation').to_numpy()
    output.mkdir(parents=True);chosen={};records=[]
    for name,columns in FEATURES.items():
        x=m[columns].to_numpy(float)
        if not np.isfinite(x).all():raise ValueError('Quality features must be finite')
        trials=[]
        for c in (.1,1.):
            model=fit_probe(x[train],y[train],m.source_id[train],seed=27,C=c)
            ids,truth,p=aggregate_sources(align_probabilities(model,x[val],classes=CLASSES),y[val],m.source_id[val],m.sequence_id[val])
            report=classification_report(truth,p,classes=CLASSES)
            records.append({'model':name,'C':c,'validation':report})
            trials.append((report['binary_balanced_accuracy'],model,c))
        _,model,c=max(trials,key=lambda q:q[0]);joblib.dump(model,output/f'{name}.joblib')
        chosen[name]={'C':c,'columns':columns,'sha256':sha256_file(output/f'{name}.joblib')}
    spec={'identity':'acquisition-only-control-v1','prepared':str(Path(prepared).resolve()),'choices':chosen,'validation':records,
        'data_sha256':sha256_file(Path(prepared)/'windows.npz'),'metadata_sha256':sha256_file(Path(prepared)/'windows.csv'),
        'code_sha256':sha256_file(__file__),'evaluation_sha256':sha256_file(Path(__file__).with_name('evaluation.py')),
        'protocol_sha256':sha256_file(PROJECT/'docs/revision_protocol.md'),
        'versions':{p:importlib.metadata.version(p) for p in ('numpy','scikit-learn','joblib')},
        'interpretation':'Retrospective future quality is a confound audit, never a past-only uncertainty input.'}
    json_write(output/'selection.json',spec);json_write(output/'seal.json',{'selection_sha256':sha256_file(output/'selection.json')})
    print(json.dumps(records,indent=2))

def evaluate(development,output):
    development=Path(development);output=Path(output)
    if output.exists():raise FileExistsError(output)
    seal=json.loads((development/'seal.json').read_text())
    if seal['selection_sha256']!=sha256_file(development/'selection.json'):raise ValueError('Changed selection')
    spec=json.loads((development/'selection.json').read_text())
    for path,key in [(Path(spec['prepared'])/'windows.npz','data_sha256'),(Path(spec['prepared'])/'windows.csv','metadata_sha256'),
                     (Path(__file__),'code_sha256'),(Path(__file__).with_name('evaluation.py'),'evaluation_sha256'),(PROJECT/'docs/revision_protocol.md','protocol_sha256')]:
        if sha256_file(path)!=spec[key]:raise ValueError(f'Changed {key}')
    for name,version in spec['versions'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Changed environment')
    _,_,m=load_prepared(spec['prepared']);m=m[m.split=='test'].copy();y=np.where(m.label=='normal','normal','abnormal')
    output.mkdir(parents=True);reports={}
    for name,choice in spec['choices'].items():
        path=development/f'{name}.joblib'
        if sha256_file(path)!=choice['sha256']:raise ValueError('Changed classifier')
        model=joblib.load(path)
        ids,truth,p=aggregate_sources(align_probabilities(model,m[choice['columns']].to_numpy(float),classes=CLASSES),y,m.source_id,m.sequence_id)
        reports[name]=classification_report(truth,p,classes=CLASSES)
        pd.DataFrame({'source_id':ids,'label':truth,'normal':p[:,0],'abnormal':p[:,1]}).to_csv(output/f'{name}_predictions.csv',index=False)
    json_write(output/'results.json',{'identity':spec['identity'],'selection_sha256':seal['selection_sha256'],'reports':reports,'interpretation':spec['interpretation']})
    print('Quality-control final evaluation completed:',output)

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    d=s.add_parser('develop');d.add_argument('--prepared',required=True);d.add_argument('--output',required=True)
    e=s.add_parser('evaluate');e.add_argument('--development',required=True);e.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='develop':develop(a.prepared,a.output)
    else:evaluate(a.development,a.output)
