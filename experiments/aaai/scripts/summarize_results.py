"""Write a qualified report from completed real-data runs only."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1]
def f(v):return 'unavailable' if v is None else f'{v:.3f}'
def ci(v):return 'unavailable' if v is None else f'[{v[0]:.3f}, {v[1]:.3f}]'
def main():
    runs=json.loads((BASE/'artifacts/real_run_index.json').read_text())
    if not runs or any(r['synthetic'] for r in runs):raise ValueError('Completed real runs required')
    text=['# Exploratory GAVD results\n','These are source-held-out technical baselines, not clinical validation. The walkthrough schedule does not establish convergence. Participant overlap, label semantics, temporal alignment and recording bias remain unresolved.\n'];records=[];comparisons=[]
    for run in runs:
        path=Path(run['run_dir']);p=json.loads((path/'provenance.json').read_text());m=json.loads((path/'capstone_metrics.json').read_text());meta=pd.read_csv(path/'window_index.csv')
        if p['synthetic'] or m['synthetic']:raise ValueError('Synthetic data cannot enter this report')
        comparisons.append((m['methods']['random_forest']['source_weighted']['macro_f1'],m['methods'][m['selected_jepa']]['source_weighted']['macro_f1'],m['methods']['untrained_encoder']['source_weighted']['macro_f1']))
        text += [f"## Seed {run['seed']}\n",f"Run `{path.name}` retained {p['n_windows']} windows, {p['n_sequences']} sequences and {p['n_sources']} sources. Each stage used {p['configuration']['epochs']} epochs. Validation selected `{m['selected_jepa']}`.\n",'### Retained cohort\n','|Split|Label|Sources|Sequences|Windows|','|---|---|---:|---:|---:|']
        counts=meta.groupby(['split','label']).agg(sources=('source_id','nunique'),sequences=('sequence_id','nunique'),windows=('sequence_id','size'))
        for (split,label),row in counts.iterrows():text.append(f'|{split}|{label}|{row.sources}|{row.sequences}|{row.windows}|')
        text += ['\n### Supervised condition recognition\n','The same sequences support every method. Normal-only describes encoder pretraining; its supervised probe still receives all five training labels.\n','|Method|Source-weighted macro-F1|Balanced accuracy|Abnormal AUROC|Abnormal AP|','|---|---:|---:|---:|---:|']
        for name,result in m['methods'].items():
            v=result['source_weighted'];records.append(dict(seed=run['seed'],method=name,**{k:v[k] for k in ['macro_f1','balanced_accuracy','abnormal_auroc','abnormal_average_precision']}))
            text.append(f"|{name}|{f(v['macro_f1'])}|{f(v['balanced_accuracy'])}|{f(v['abnormal_auroc'])}|{f(v['abnormal_average_precision'])}|")
        test_sources=meta[meta.split.eq('test')].groupby('label').source_id.nunique()
        fragile=test_sources[test_sources<2]
        if len(fragile):text.append('\nSevere class-support limitation: '+', '.join(f'{k} has {v} test source' for k,v in fragile.items())+'. Bootstrap intervals cannot recover missing participant diversity.\n')
        text += ['\n### Per-class recall\n','|Label|Test sources|RF recall|Selected JEPA recall|','|---|---:|---:|---:|']
        for label,n in test_sources.items():
            rf=m['methods']['random_forest']['source_weighted']['per_class'][label]['recall']
            je=m['methods'][m['selected_jepa']]['source_weighted']['per_class'][label]['recall']
            text.append(f'|{label}|{n}|{f(rf)}|{f(je)}|')
        pair=m['paired_selected_minus_rf_sequence_weighted']
        text += [f"\nPaired selected-JEPA minus RF **sequence-weighted** macro-F1 interval: {ci(pair['intervals_95']['macro_f1'])}. {pair['valid_repeats']}/{pair['requested_repeats']} source-cluster resamples retained every class. This weighting differs from the source-weighted points above.\n",'### Normal-only anomaly ranking\n','|Model|AUROC|95% source-cluster interval|Abnormal AP|Normal calibration sources|Finite 5% threshold|','|---|---:|---|---:|---:|---|']
        for filename,label in [('anomaly_metrics.json','Masked completion'),('forecast_anomaly_metrics.json','Suffix forecasting'),('normal_distance_metrics.json','Normal embedding distance')]:
            v=json.loads((path/filename).read_text());b=v.get('source_bootstrap',{})
            text.append(f"|{label}|{f(v['auroc'])}|{ci(b.get('auroc_ci95'))}|{f(v['average_precision'])}|{v['n_normal_calibration_sources']}|{v['finite_threshold']}|")
        if (path/'spe_ablation.json').exists():
            spe=json.loads((path/'spe_ablation.json').read_text())
            text.append('\nMatched seed-42 StoP-inspired position-noise ablation: source-weighted macro-F1 '+f(spe['metrics']['source_weighted']['macro_f1'])+'. Paired sequence-weighted SPE minus deterministic-position interval '+ci(spe['paired_spe_minus_fixed_sequence_weighted']['intervals_95']['macro_f1'])+'. This is a single-seed representation ablation, not a confirmed improvement.\n')
        text.append('\nInsufficient normal calibration support yields an infinite threshold, with zero sensitivity by construction. It cannot support operational detection claims. Ranking scores remain descriptive. Forecasting is retrospective on evaluable futures.\n')
    text += ['## Interpretation\n','Compare the selected JEPA with both RF and the untrained encoder. Read paired intervals alongside point estimates and do not equate a ranking metric with classification accuracy. Supervised abnormal ranking uses a different supervision setting from the normal-only detector. Score direction must remain as prespecified rather than being flipped after test inspection. Repeated source splits are dependent, not independent clinical observations. Extra progressive training changes compute and data exposure. A matched pooled schedule is required to isolate curriculum effects.\n','The [data audit](data_audit.md) documents attrition and semantic limits. Independent participant-identified data are needed for stronger generalization. A single short SPE run cannot establish the general value of positional noise. Longer training must be selected using validation and evaluated on genuinely untouched data. The learned stochastic-latent prior/posterior extension remains unimplemented. Per-sequence predictions, source assignments, checkpoints and software versions live in each recorded run directory.\n']
    rf,je,untrained=map(list,zip(*comparisons))
    summary=(f'Across {len(runs)} prespecified short runs, RF source-weighted macro-F1 ranged from {min(rf):.3f} to {max(rf):.3f}, compared with {min(je):.3f} to {max(je):.3f} for the validation-selected JEPA. RF had higher points in {sum(a>b for a,b in zip(rf,je))}/{len(runs)} runs. The untrained encoder had higher points than the selected trained JEPA in {sum(a>b for a,b in zip(untrained,je))}/{len(runs)} runs. Point estimates alone do not establish a representation-learning advantage.\n')
    text.insert(1,summary)
    text.insert(3,'![Prespecified first-split comparison](../images/real_comparison.svg)\n')
    (BASE/'docs/results.md').write_text('\n'.join(text));df=pd.DataFrame(records);df.to_csv(BASE/'docs/results_summary.csv',index=False)
    first=df[df.seed.eq(runs[0]['seed'])].set_index('method');fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,key,title in zip(axes,['macro_f1','balanced_accuracy'],['Macro-F1','Balanced accuracy']):
        ax.barh(first.index,first[key],color='#007f82');ax.set_xlim(0,1);ax.set_xlabel(title);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f"Exploratory GAVD test comparison, seed {runs[0]['seed']}\nSource-weighted points; no clinical generalization claim")
    fig.tight_layout();fig.savefig(BASE/'images/real_comparison.svg',bbox_inches='tight');plt.close(fig)
    print('Wrote real-data report and vector result chart')
if __name__=='__main__':main()
