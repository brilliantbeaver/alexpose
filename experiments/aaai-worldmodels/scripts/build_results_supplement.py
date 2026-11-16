"""Render evidence-linked supplementary tables and the uncertainty trade-off."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];r=json.loads((ROOT/'docs/research_results.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
rows=r['forecast']['hypotheses']['by_seed'];seeds=[x['seed'] for x in rows]
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
for ax,key,title,label in [(axes[0],'energy_gain','Distribution score','Deterministic energy minus latent energy'),(axes[1],'aurc_gain_over_prefix_confidence','Selective prediction','Prefix-confidence AURC minus latent AURC')]:
 ax.scatter(range(3),[x[key] for x in rows],s=65,color='#21685f');ax.axhline(0,color='gray',ls=':');ax.set(xticks=range(3),xticklabels=seeds,xlabel='Training seed',ylabel=label,title=title);ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
fig.suptitle('Positive values favor latent sampling; six normal test sources',fontsize=13);fig.tight_layout();fig.savefig(ROOT/'images/uncertainty_tradeoff.svg');fig.savefig(ROOT/'images/uncertainty_tradeoff.png',dpi=140);plt.close(fig)
lines=['# Complete result tables','', 'Generated from `research_results.json` and the frozen source CSVs. Values are exploratory source-level estimates. Classification uses 17 test sources; prediction/anomaly scores use 19, including six normal sources. No score direction or primary model was selected from test results.','', '## Direct binary and five-class classification','', '| Model | Binary balanced accuracy | Binary AUROC | Five-class balanced accuracy | Five-class macro-F1 |','|---|---:|---:|---:|---:|']
for name,b in r['binary']['test_reports'].items():
 f=r['five_class_baseline']['classifiers'][name];lines.append(f"| {name} | {b['binary_balanced_accuracy']:.4f} | {b['abnormal_auroc']:.4f} | {f['balanced_accuracy']:.4f} | {f['macro_f1']:.4f} |")
lines+=['','## Forecast accuracy by seed','', '| Seed | Arm | Normal MSE | Normal energy | Persistence minus model MSE |','|---|---|---:|---:|---:|']
for a in r['forecast']['records']:lines.append(f"| {a['seed']} | {a['arm']} | {a['normal_mse']:.6f} | {a['normal_energy']:.6f} | {a['normal_persistence_gain']:+.6f} |")
lines+=['','## Every prespecified anomaly score','', 'Mean scores across the three training seeds are formed per source before evaluation. These are nine secondary comparisons. The source bootstrap has 2,000 paired replicates. Higher score always means abnormal; the direction is not reversed when AUROC is below 0.5.','', '| Arm | Score | AUROC [95% interval] | AP | AUROC gain over persistence [95% interval] |','|---|---|---|---:|---|']
for a in r['anomaly_seed_mean_scores']:
 q=a['auroc'];d=a['auroc_gain_over_persistence'];lines.append(f"| {a['arm']} | {a['score']} | {q['estimate']:.3f} [{q['ci95'][0]:.3f}, {q['ci95'][1]:.3f}] | {a['average_precision']['estimate']:.3f} | {d['estimate']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |")
lines+=['','## Additional fixed forecast baselines','', '| Baseline | Normal-source MSE | All-source MSE |','|---|---:|---:|']
for name in ['persistence','velocity','population_static','untrained_direct','untrained_residual']:
 f=pd.read_csv(ROOT/f'artifacts/runs/revision-test-v1/{name}_sources.csv');lines.append(f"| {name} | {f.loc[f.label=='normal','mse_h16'].mean():.6f} | {f.mse_h16.mean():.6f} |")
lines+=['','## Temporal controls','', '| Arm/control | Controlled minus original normal MSE [95% interval] |','|---|---|']
for name,d in r['temporal_controls'].items():lines.append(f"| {name} | {d['mean_gain']:+.6f} [{d['interval95'][0]:+.6f}, {d['interval95'][1]:+.6f}] |")
lines+=['','## Uncertainty and optimization sensitivity','', '![Distribution quality and selective prediction](../images/uncertainty_tradeoff.svg)','', 'The JSON report retains all source-level uncertainty/risk summaries, including unavailable partial-rank correlations when residual degrees of freedom are insufficient. AURC here is the declared mean of risk at the retained coverage grid, not a continuous integral.','', f"Weight-ten minus weight-one benefit in normal MSE (positive favors weight ten): {r['objective_weight_audit']['gain_over_primary']['mean_gain']:+.8f}; 95% paired interval {r['objective_weight_audit']['gain_over_primary']['interval95']}."]
(ROOT/'docs/results_tables.md').write_text('\n'.join(lines)+'\n')
(ROOT/'docs/results_figure_provenance.json').write_text(json.dumps({'report_sha256':hashlib.sha256((ROOT/'docs/research_results.json').read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'analysis_generator_sha256':hashlib.sha256((ROOT/'scripts/analyze_research.py').read_bytes()).hexdigest(),'source_csv_manifest':'test_execution.json','figure_sha256':{name:hashlib.sha256((ROOT/'images'/name).read_bytes()).hexdigest() for name in ['binary_revision.svg','forecast_revision.svg','anomaly_revision.svg','uncertainty_tradeoff.svg']},'figures':['binary_revision.svg','forecast_revision.svg','anomaly_revision.svg','uncertainty_tradeoff.svg']},indent=2)+'\n')
print('Rendered supplementary result tables and uncertainty figure.')
