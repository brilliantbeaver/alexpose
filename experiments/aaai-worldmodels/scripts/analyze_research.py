"""Analyze the jointly released, frozen experiments without choosing new models."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from worldmodels.revision import paired_gain_summary
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'artifacts/runs'
release=json.loads((ROOT/'docs/test_release.json').read_text())
if release['status']!='frozen_before_joint_test_release':raise ValueError('No joint release')
for name,digest in release['frozen_files'].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed frozen input '+name)

def load(path):return json.loads(path.read_text())
execution=load(ROOT/'docs/test_execution.json')
if execution['status']!='all_fixed_evaluations_complete':raise ValueError('Final evaluations incomplete')
for name,digest in execution['results'].items():
    if hashlib.sha256((RUNS/name/'results.json').read_bytes()).hexdigest()!=digest:raise ValueError('Changed final results '+name)
for name,digest in execution['source_csv_sha256'].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed final source CSV '+name)


def bootstrap(y,scores,threshold=None,seed=2027,n=2000,reference=None):
    y=np.asarray(y,dtype=bool);scores=np.asarray(scores,float)
    groups=[np.flatnonzero(y==label) for label in (False,True)]
    if any(len(g)==0 for g in groups):return {'status':'both classes required'}
    rng=np.random.default_rng(seed)
    def metric(idx):
        row=[roc_auc_score(y[idx],scores[idx]),average_precision_score(y[idx],scores[idx])]
        if threshold is not None:row.append(balanced_accuracy_score(y[idx],scores[idx]>=threshold))
        if reference is not None:
            row.extend([row[0]-roc_auc_score(y[idx],np.asarray(reference)[idx]),row[1]-average_precision_score(y[idx],np.asarray(reference)[idx])])
        return row
    values=np.asarray([metric(np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])) for _ in range(n)])
    names=['auroc','average_precision']+(['binary_balanced_accuracy'] if threshold is not None else [])
    if reference is not None:names+=['auroc_gain_over_persistence','ap_gain_over_persistence']
    estimates=metric(np.arange(len(y)))
    return dict(n_sources=len(y),abnormal_prevalence=float(y.mean()),unit='source',repetitions=n,
        **{name:{'estimate':float(estimates[i]),'ci95':np.quantile(values[:,i],[.025,.975]).tolist()} for i,name in enumerate(names)})

binary=load(RUNS/'binary-test-clean-v1/results.json');forecast=load(RUNS/'revision-test-v1/results.json')
quality=load(RUNS/'quality-test-clean-v1/results.json');development=load(RUNS/'revision-development-v1/development.json')
binary_ci={}
for path in sorted((RUNS/'binary-test-clean-v1').glob('*_source_predictions.csv')):
    frame=pd.read_csv(path);binary_ci[path.name.replace('_source_predictions.csv','')]=bootstrap(frame.label!='normal',frame.abnormal,threshold=.5)
quality_ci={}
for path in sorted((RUNS/'quality-test-clean-v1').glob('*_predictions.csv')):
    frame=pd.read_csv(path);quality_ci[path.name.replace('_predictions.csv','')]=bootstrap(frame.label!='normal',frame.abnormal,threshold=.5)
persistence=pd.read_csv(RUNS/'revision-test-v1/persistence_sources.csv').set_index('source_id').sort_index()
records=[];effects=[];frames={}
for seed in forecast['protocol']['seeds']:
    for arm in forecast['protocol']['arms']:
        directory=RUNS/f'revision-test-v1/seed_{seed}'
        frame=pd.read_csv(directory/f'{arm}_actual_sources.csv').set_index('source_id').sort_index()
        if not frame.index.equals(persistence.index) or not frame.label.equals(persistence.label):raise ValueError('Source support/labels differ')
        frame['excess_mse']=frame.mse_h16-persistence.mse_h16
        frames[(seed,arm)]=frame
        normal=frame.label=='normal'
        for score,column in [('raw_mse','mse_h16'),('energy','energy_h16'),('excess_mse','excess_mse')]:
            records.append({'seed':seed,'arm':arm,'score':score,**bootstrap(frame.label!='normal',frame[column],reference=persistence.mse_h16.to_numpy())})
        for control in ('static','shuffled'):
            alternative=pd.read_csv(directory/f'{arm}_{control}_sources.csv').set_index('source_id').loc[frame.index]
            for source in frame.index[normal]:
                effects.append({'seed':seed,'arm':arm,'control':control,'source_id':source,'penalty':float(alternative.loc[source,'mse_h16']-frame.loc[source,'mse_h16'])})
ensemble=[]
for arm in forecast['protocol']['arms']:
    source=frames[(forecast['protocol']['seeds'][0],arm)]
    for score,column in [('raw_mse','mse_h16'),('energy','energy_h16'),('excess_mse','excess_mse')]:
        values=np.mean([frames[(seed,arm)][column].to_numpy() for seed in forecast['protocol']['seeds']],axis=0)
        ensemble.append({'arm':arm,'score':score,**bootstrap(source.label!='normal',values,reference=persistence.mse_h16.to_numpy())})
effect_frame=pd.DataFrame(effects);temporal={}
for (arm,control),group in effect_frame.groupby(['arm','control']):
    temporal[f'{arm}_{control}']=paired_gain_summary(group.groupby('source_id').penalty.mean().to_numpy())
energy_differences=[]
for seed in forecast['protocol']['seeds']:
    det,slv=frames[(seed,'residual_det')],frames[(seed,'residual_slv')]
    normal=det.label=='normal'
    energy_differences.append((det.loc[normal,'energy_h16']-slv.loc[normal,'energy_h16']).rename(seed))
energy_effect=paired_gain_summary(pd.concat(energy_differences,axis=1).mean(axis=1).to_numpy())
objective=load(RUNS/'objective-audit-test-v1/results.json')
report=dict(five_class_baseline=load(RUNS/'pilot-comparison-clean-v1/results.json'),objective_weight_audit=objective,person_review=load(ROOT/'docs/forecast_reuse_certificate.json'),identity='released-medwm-study-analysis-v1',primary_binary=binary['primary_jepa'],binary=binary,binary_intervals=binary_ci,
    forecast=forecast,forecast_validation_gate=development['validation_gate'],quality=quality,quality_intervals=quality_ci,
    temporal_controls=temporal,stochastic_energy_gain=energy_effect,anomaly_by_seed=records,anomaly_seed_mean_scores=ensemble,
    persistence_anomaly=bootstrap(persistence.label!='normal',persistence.mse_h16),
    interpretation='Exploratory source-grouped GAVD study. No new model, score direction or threshold selected from these results.',
    analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(ROOT/'docs/research_results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(ROOT/'artifacts/research_analysis').mkdir(exist_ok=True)
effect_frame.to_csv(ROOT/'artifacts/research_analysis/temporal_effects.csv',index=False)
pd.DataFrame([{'seed':r['seed'],'arm':r['arm'],'score':r['score'],'auroc':r['auroc']['estimate'],'ap':r['average_precision']['estimate']} for r in records]).to_csv(ROOT/'artifacts/research_analysis/anomaly_by_seed.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
# Every displayed comparison has a declared primary or descriptive role.
fig,ax=plt.subplots(figsize=(10,4.8));names=['rf','untrained_probe','jepa_normal','jepa_adapted','jepa_mixed']
labels=['Random Forest','Untrained probe','Normal JEPA probe','Staged JEPA probe','Mixed JEPA probe']
for i,name in enumerate(names):
    r=binary_ci[name]['binary_balanced_accuracy'];lo,hi=r['ci95'];v=r['estimate']
    ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color='#21685f' if name=='jepa_mixed' else '#477eac',capsize=4)
ax.set(yticks=range(5),yticklabels=labels,xlim=(0,1),xlabel='Source-level balanced accuracy, 95% source bootstrap interval',title=f"Direct binary evaluation on {binary_ci['rf']['n_sources']} held-out sources");ax.invert_yaxis();ax.axvline(.5,color='gray',ls=':')
fig.tight_layout();fig.savefig(ROOT/'images/binary_revision.svg');fig.savefig(ROOT/'images/binary_revision.png',dpi=120);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4.8))
colors={'direct_det':'#b47755','residual_det':'#21685f','residual_slv':'#477eac'}
for arm in forecast['protocol']['arms']:
    rows=[r for r in forecast['records'] if r['arm']==arm]
    offset={'direct_det':-.2,'residual_det':0.,'residual_slv':.2}[arm]
    ax.scatter(np.arange(len(rows))+offset,[r['normal_persistence_gain'] for r in rows],label=arm,color=colors[arm])
ax.axhline(0,color='gray',ls=':');ax.set(xticks=range(len(forecast['protocol']['seeds'])),xticklabels=forecast['protocol']['seeds'],xlabel='Training seed',ylabel='Persistence MSE minus model MSE',title='Normal-source forecast skill; positive values favor the model');ax.legend();fig.tight_layout();fig.savefig(ROOT/'images/forecast_revision.svg');fig.savefig(ROOT/'images/forecast_revision.png',dpi=120);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4.8))
for i,arm in enumerate(forecast['protocol']['arms']):
    rows=[r for r in ensemble if r['arm']==arm]
    for j,r in enumerate(rows):
        v=r['auroc']['estimate'];lo,hi=r['auroc']['ci95'];ax.errorbar(v,i*4+j,xerr=[[v-lo],[hi-v]],fmt='o',color=colors[arm],capsize=3)
ax.set(yticks=[i*4+j for i in range(3) for j in range(3)],yticklabels=[f'{a}: {s}' for a in forecast['protocol']['arms'] for s in ('MSE','energy','excess MSE')],xlim=(0,1),xlabel='AUROC, 95% source bootstrap interval',title='All predeclared anomaly scores, averaged across training seeds');ax.invert_yaxis();ax.axvline(.5,color='gray',ls=':');fig.tight_layout();fig.savefig(ROOT/'images/anomaly_revision.svg');fig.savefig(ROOT/'images/anomaly_revision.png',dpi=120);plt.close(fig)
print('Wrote docs/research_results.json and three evidence-linked SVG figures.')
