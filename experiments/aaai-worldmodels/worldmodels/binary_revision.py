"""Correct the binary endpoint without selecting anything on test sources."""
from pathlib import Path
import argparse,json,joblib,importlib.metadata
from .settings import PROJECT
import numpy as np
import pandas as pd
from .workflow import load_prepared,json_write
from .data import sha256_file
from .features import build_features
from .evaluation import fit_rf,fit_probe,align_probabilities,aggregate_sources,classification_report,paired_source_bootstrap
from .training import load_checkpoint,encode

CLASSES=('normal','abnormal')
ENCODERS={'untrained_probe':'untrained_encoder.pt','jepa_normal':'normal_completion.pt','jepa_adapted':'adaptation_4.pt','jepa_mixed':'mixed_adaptation.pt'}

def matrix(name,x,valid,meta,baseline):
    if name=='rf':return build_features(x,valid,meta.sample_fps.to_numpy())[0]
    return encode(load_checkpoint(Path(baseline)/ENCODERS[name]),x,valid)

def develop(prepared,baseline,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    x,v,m=load_prepared(prepared)
    selected=m.split.isin(['train','validation']).to_numpy()
    x,v,m=x[selected],v[selected],m[selected].reset_index(drop=True)
    y=np.where(m.label=='normal','normal','abnormal')
    train=(m.split=='train').to_numpy();val=(m.split=='validation').to_numpy()
    output.mkdir(parents=True)
    records=[];choices={};prior_diagnostics={}
    for name in ['rf',*ENCODERS]:
        z=matrix(name,x,v,m,baseline)
        old=joblib.load(Path(baseline)/f'{name}_classifier.joblib')
        old_p=old.predict_proba(z[val]);normal_column=list(old.classes_).index('normal')
        sid,truth,p=aggregate_sources(np.column_stack([old_p[:,normal_column],1-old_p[:,normal_column]]),y[val],m.source_id[val],m.sequence_id[val])
        prior_diagnostics[name]=dict(classification_report(truth,p,classes=CLASSES),predicted_abnormal_sources=int((p[:,1]>=.5).sum()))
        candidates=[{'max_depth':d} for d in (4,8)] if name=='rf' else [{'C':c} for c in (.01,.1,1.,10.)]
        runs=[]
        for config in candidates:
            estimator=(fit_rf if name=='rf' else fit_probe)(z[train],y[train],m.source_id[train],seed=27,**config)
            sid,truth,p=aggregate_sources(align_probabilities(estimator,z[val],classes=CLASSES),y[val],m.source_id[val],m.sequence_id[val])
            report=classification_report(truth,p,classes=CLASSES)
            records.append(dict(model=name,config=config,validation=report))
            runs.append((report['binary_balanced_accuracy'],report['macro_f1'],estimator,config))
        best=max(runs,key=lambda item:(item[0],item[1]))
        joblib.dump(best[2],output/f'{name}.joblib')
        choices[name]=dict(config=best[3],validation_binary_ba=best[0],classifier_sha256=sha256_file(output/f'{name}.joblib'))
    selection=dict(identity='direct-binary-v1',development_only=True,prepared=str(Path(prepared).resolve()),baseline=str(Path(baseline).resolve()),
        data_sha256=sha256_file(Path(prepared)/'windows.npz'),primary_jepa=max(ENCODERS,key=lambda name:choices[name]['validation_binary_ba']),choices=choices,validation_candidates=records,old_five_class_threshold_diagnostics=prior_diagnostics,
        criterion='validation binary balanced accuracy, then macro F1; first simplest grid entry on ties',
        training_target='normal versus pooled selected abnormal; equal source and binary-class weights',
        test_metrics_inspected=False,code_sha256=sha256_file(__file__),
        metadata_sha256=sha256_file(Path(prepared)/'windows.csv'),
        baseline_artifact_sha256={p.name:sha256_file(p) for p in Path(baseline).iterdir() if p.suffix in ('.pt','.joblib')},
        dependency_sha256={p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py') if p.name in ('data.py','features.py','evaluation.py','training.py','model.py','masking.py','workflow.py')},
        versions={p:importlib.metadata.version(p) for p in ('torch','numpy','scikit-learn','joblib')},
        protocol_sha256=sha256_file(PROJECT/'docs/revision_protocol.md'))
    json_write(output/'selection.json',selection)
    json_write(output/'seal.json',{'selection_sha256':sha256_file(output/'selection.json'),'protocol_sha256':selection['protocol_sha256']})
    print(json.dumps({k:v['validation_binary_ba'] for k,v in choices.items()},indent=2))

def evaluate(development,output):
    development=Path(development);output=Path(output)
    if output.exists():raise FileExistsError(output)
    seal=json.loads((development/'seal.json').read_text())
    if sha256_file(development/'selection.json')!=seal['selection_sha256']:raise ValueError('Changed selection')
    spec=json.loads((development/'selection.json').read_text())
    if sha256_file(PROJECT/'docs/revision_protocol.md')!=spec['protocol_sha256']:raise ValueError('Changed protocol')
    if sha256_file(Path(spec['prepared'])/'windows.csv')!=spec['metadata_sha256']:raise ValueError('Changed metadata')
    for name,digest in spec['baseline_artifact_sha256'].items():
        if sha256_file(Path(spec['baseline'])/name)!=digest:raise ValueError('Changed baseline artifact')
    for name,digest in spec['dependency_sha256'].items():
        if sha256_file(Path(__file__).with_name(name))!=digest:raise ValueError('Changed dependency')
    for name,version in spec['versions'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Changed environment')
    if sha256_file(__file__)!=spec['code_sha256']:raise ValueError('Changed binary experiment code')
    x,v,m=load_prepared(spec['prepared'])
    if sha256_file(Path(spec['prepared'])/'windows.npz')!=spec['data_sha256']:raise ValueError('Changed input')
    test=(m.split=='test').to_numpy();x,v,m=x[test],v[test],m[test].reset_index(drop=True)
    y=np.where(m.label=='normal','normal','abnormal');reports={};probabilities={};old_reports={};endpoint_differences={}
    output.mkdir(parents=True)
    for name,choice in spec['choices'].items():
        path=development/f'{name}.joblib'
        if sha256_file(path)!=choice['classifier_sha256']:raise ValueError('Changed classifier')
        z=matrix(name,x,v,m,spec['baseline']);estimator=joblib.load(path)
        ids,truth,p=aggregate_sources(align_probabilities(estimator,z,classes=CLASSES),y,m.source_id,m.sequence_id)
        reports[name]=classification_report(truth,p,classes=CLASSES);probabilities[name]=p
        old_estimator=joblib.load(Path(spec['baseline'])/f'{name}_classifier.joblib')
        raw=old_estimator.predict_proba(z);normal_column=list(old_estimator.classes_).index('normal')
        old_ids,old_truth,old_p=aggregate_sources(np.column_stack([raw[:,normal_column],1-raw[:,normal_column]]),y,m.source_id,m.sequence_id)
        assert np.array_equal(ids,old_ids) and np.array_equal(truth,old_truth)
        old_reports[name]=classification_report(truth,old_p,classes=CLASSES)
        endpoint_differences[name]=paired_source_bootstrap(truth,p,old_p,classes=CLASSES,seed=27,repetitions=2000)
        pd.DataFrame({'source_id':ids,'label':truth,'normal':p[:,0],'abnormal':p[:,1]}).to_csv(output/f'{name}_source_predictions.csv',index=False)
    comparisons={name:paired_source_bootstrap(truth,p,probabilities['rf'],classes=CLASSES,seed=27,repetitions=2000) for name,p in probabilities.items() if name!='rf'}
    json_write(output/'results.json',dict(identity='direct-binary-v1',selection_sha256=seal['selection_sha256'],primary_jepa=spec['primary_jepa'],test_reports=reports,paired_against_rf=comparisons,old_five_class_derived_reports=old_reports,direct_minus_old_endpoint=endpoint_differences,
        limitation='Single encoder seed and same fixed held-out source cohort; no external clinical validation.'))
    print('Frozen binary evaluation completed:',output)

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    d=s.add_parser('develop');d.add_argument('--prepared',required=True);d.add_argument('--baseline',required=True);d.add_argument('--output',required=True)
    e=s.add_parser('evaluate');e.add_argument('--development',required=True);e.add_argument('--output',required=True)
    a=p.parse_args()
    if a.cmd=='develop':develop(a.prepared,a.baseline,a.output)
    else:evaluate(a.development,a.output)
