"""Render paired uncertainty from the frozen primary real-data artifacts."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
B=Path(__file__).resolve().parent; EXP=B.parents[2]
e=json.loads((B.parent/'evidence.json').read_text())
rows=[('Raw full error',e['primary_difference']['point_difference'],*e['primary_difference']['interval_95'])]
for label,name in [('Full-error distance','Full-error profile distance'),('Own-error distance','Own-error profile distance'),('Random encoder, latent heads','Untrained latent profile')]:
 aliases={'Untrained latent profile':{'Untrained latent profile','Random encoder + fitted latent heads','Random encoder + fitted heads','Random encoder with fitted heads'}}
 r=next(x for x in e['matched_primary'] if x['Comparator'] in aliases.get(name,{name}));rows.append((label,r['Difference'],r['CI_low'],r['CI_high']))
r=e['source_control_intervals']['motion_normal_distance'];rows.append(('Motion-feature distance',r['point_difference'],*r['interval_95']))
plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(6.7,2.75),layout='constrained')
y=np.arange(len(rows));x=np.array([r[1] for r in rows]);low=np.array([r[2] for r in rows]);high=np.array([r[3] for r in rows])
ax.errorbar(x,y,xerr=[x-low,high-x],fmt='o',color='#087f88',capsize=4,linewidth=1.6,markersize=4)
ax.axvline(0,color='#626c7a',ls='--',lw=1);ax.set_yticks(y,[r[0] for r in rows]);ax.invert_yaxis();ax.set_xlim(-.19,.28)
ax.set_xlabel('Latent-profile AUROC minus comparator AUROC')
ax.text(.01,1.07,'Negative: comparator higher',transform=ax.transAxes,ha='left',fontsize=9)
ax.text(.99,1.07,'Positive: profile higher',transform=ax.transAxes,ha='right',fontsize=9)
ax.grid(axis='x',color='#e1e7eb',linewidth=.6);ax.set_axisbelow(True)
for ext in ('svg','pdf','png'):fig.savefig(B/f'paired_differences.{ext}',dpi=180)
plt.close(fig)
(B/'paired_differences_data.json').write_text(json.dumps({'primary_run':e['primary_run'],'rows':rows,'interval':'Paired 1000-replicate source bootstrap, conditional on fitted models, references, folds and observed source identities','multiplicity':'Primary raw-error comparison; remaining contrasts are supporting and unadjusted'},indent=2))
# Refresh the earlier overview with accurate control naming, without resummarizing training artifacts.
metrics=json.loads((EXP/'artifacts/context_study/gavd'/e['primary_run']/'ranking_metrics.json').read_text())['methods']
items=[('JEPA latent profile','trained_latent_all_profile_score'),('JEPA full-context error','trained_latent_all_raw_full'),('JEPA coordinate profile','trained_coordinate_all_profile_score'),('Ridge coordinate profile','ridge_coordinate_all_profile_score'),('Random encoder + fitted latent heads','untrained_latent_all_profile_score')]
fig,ax=plt.subplots(figsize=(9,4),layout='constrained');values=[metrics[k]['primary_auroc'] for _,k in items]
bars=ax.barh([n for n,_ in items],values,color=['#1c7882','#526378','#b67b30','#6e5a9d','#7d8c93']);ax.invert_yaxis();ax.axvline(.5,color='#999999',ls=':');ax.set_xlim(0,1)
ax.set(title='Real GAVD source-held-out comparison',xlabel='Normal-count-weighted within-fold AUROC')
for bar,v in zip(bars,values):ax.text(v+.015,bar.get_y()+bar.get_height()/2,f'{v:.3f}',va='center')
for ext in ('svg','pdf','png'):fig.savefig(EXP/'images'/f'context_real_comparison.{ext}',dpi=160)
