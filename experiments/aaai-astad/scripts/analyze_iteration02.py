"""Prespecified comparator integration and explicitly descriptive sensitivity."""
from pathlib import Path
import argparse,sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments/aaai'))
from aaai_gait.iteration_scoring import ranking_metrics,endpoint_frame,paired_bootstrap,FOCAL
from aaai_gait.study_runner import dump_json

parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
args=parser.parse_args();out=args.run
scores=pd.read_csv(out/'source_scores.csv');registry=pd.read_csv(out/'source_registry.csv')
aux=pd.concat([pd.read_csv(out/f'auxiliary/fold_{f}/stgnf_adapted/source_scores.csv') for f in range(4)])
merged=scores.drop(columns=['stgnf_adapted','stgnf_missing_fraction'],errors='ignore').merge(
    aux.rename(columns={'missing_fraction':'stgnf_missing_fraction'}),on=['source_id','outer_fold'],validate='many_to_one')
merged.to_csv(out/'source_scores.csv',index=False)
base=merged[merged.policy.eq('U')&merged.bank.eq('U')].copy()
extra={}
for endpoint in ('focal','broad'):
    t=endpoint_frame(base,endpoint)
    extra[endpoint]=dict(stgnf_adapted=ranking_metrics(t,'stgnf_adapted'),
        S_vs_stgnf=paired_bootstrap(t,'S','stgnf_adapted',repeats=1000),
        S_vs_richer=paired_bootstrap(t,'S','motion36',repeats=1000))
dump_json(out/'auxiliary_metrics.json',extra)
classification=[];details=[]
for fold in range(4):
    rf=pd.read_csv(out/f'fold_{fold}/five_class_predictions.csv');rf['method']='random_forest'
    probes=pd.read_csv(out/f'auxiliary/fold_{fold}/jepa_probes/five_class_predictions.csv')
    for name,t in [('random_forest',rf)]+[(m,p) for m,p in probes.groupby('method')]:
        if set(t.source_id)!=set(rf.source_id):raise ValueError('Five-class comparators have different sources')
        classification.append(t)
        if name=='random_forest':metric=json.loads((out/f'fold_{fold}/tabular_audit.json').read_text())['five_class_rf']
        else:metric=json.loads((out/f'auxiliary/fold_{fold}/jepa_probes/status.json').read_text())['metrics'][name]
        details.append(dict(outer_fold=fold,method=name,macro_f1=metric['macro_f1'],
                            balanced_accuracy=metric['balanced_accuracy'],n_test_sources=len(t),
                            confusion_matrix=metric['confusion_matrix'],labels=metric['labels']))
pd.concat(classification,ignore_index=True).to_csv(out/'five_class_predictions.csv',index=False)
dump_json(out/'five_class_metrics.json',details)
# Historical cohort sensitivity uses newly fitted iteration02 scores only.
old=pd.read_csv(ROOT/'experiments/aaai/artifacts/context_study/gavd/3044566091261819/source_folds.csv')
common=base[base.source_id.isin(old.source_id)]
dump_json(out/'historical_common_source_sensitivity.json',dict(
    interpretation='Newly fitted methods, evaluated on sources also eligible in draft01; old model scores are not reused',
    n_common_sources=len(common),metrics={m:ranking_metrics(common,m) for m in ['M','S','motion36','timing8','stgnf_adapted']},
    paired=paired_bootstrap(common,'S','M',repeats=1000)))
per_condition={}
for label in FOCAL[1:]:
    t=base[base.categories.map(lambda c:c=='normal' or label in c.split('|'))]
    per_condition[label]={m:ranking_metrics(t,m) for m in ['M','S','motion36','stgnf_adapted']}
dump_json(out/'per_condition_metrics.json',per_condition)
# Frame-aligned labels, restricted to a single common view across a source.
meta=pd.read_csv(out/'window_metadata.csv',keep_default_na=False)
views=meta.groupby('source_id').camera_views.agg(lambda v:next(iter(set(v))) if len(set(v))==1 else 'mixed_or_changing')
base['view_stratum']=base.source_id.map(views)
view_results=[]
for view,t in base.groupby('view_stratum'):
    supported=[f for f,g in t.groupby('outer_fold') if g.binary_label.eq(0).sum()>=2 and g.binary_label.eq(1).sum()>=2]
    subset=t[t.outer_fold.isin(supported)]
    record=dict(view=view,n_sources=len(t),n_normal=int(t.binary_label.eq(0).sum()),supported_folds=supported,
                complete_four_fold_comparison=len(supported)==4,excluded_sources=len(t)-len(subset),
                camera_motion='unresolved; this is view conditioning only',analysis='descriptive after core scoring')
    if len(subset):record['metrics']={m:ranking_metrics(subset,m) for m in ['M','S','motion36']}
    view_results.append(record)
dump_json(out/'view_sensitivity.json',view_results)
dump_json(out/'baseline_status.json',dict(
    local_interpolation='Implemented as permitted-input feature and explicit baseline API; separate error analysis saved below',
    shared_ridge='completed four source folds',own_capacity='completed',both_leg='completed U/J/A on U and E_leg',
    motion16='completed',motion36='completed',timing8='completed; waveform proxy, not validated PCI',
    sham='completed',calibrated_errors='completed',observation_quality='completed',supervised_nuisance='completed',
    isolation_forest='completed',STG_NF='completed adapted16-joint graph with fitting-normal median imputation; not faithful reproduction',
    five_class_RF='completed',five_class_JEPA_and_random_probe='completed with new normal-only512-update backbones',
    encoder_mask_ablation='deferred; independent participant replication prioritized, so no encoder-policy contribution claim',
    PCI='unavailable: no validated normal contact/phase estimator on these short clips',
    stochastic_position_or_latent_variables='deferred; no result claimed'))
print(out/'auxiliary_metrics.json')
