"""Rebuild vector figures from the frozen external run, never from typed scores."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from aaai_gait.external_iteration02 import extract_angles
BASE=Path(__file__).resolve().parents[1]
out=BASE/'images/iteration02/external_replication';out.mkdir(parents=True,exist_ok=True)
run=BASE/'artifacts/iteration02/external_fb9d0d23b56ce29fd846'
m=json.loads((run/'metrics.json').read_text());e=json.loads((run/'paired_effects.json').read_text());demo=json.loads((run/'age_sex_sensitivity.json').read_text());adequacy=json.loads((run/'adequacy.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(14,4.4),layout='constrained',gridspec_kw={'width_ratios':[1,1.1,1.2]})
heldout=pd.read_csv(run/'heldout_prediction_error.csv')
gains=(100*heldout.loc[heldout.binary_label.eq(0)].sort_values('outer_fold').both_vs_own_relative_gain).tolist()
axs[0].bar(range(1,5),gains,color='#307c83',width=.6);axs[0].set(xticks=range(1,5),xlabel='Outer fold',ylabel='Held-out normal MSE reduction (%)',ylim=(0,7),title='Opposite-leg context helps completion')
for i,y in enumerate(gains,1):axs[0].text(i,y+.16,f'{y:.2f}%',ha='center',fontsize=10)
labels=['Motion M','Relational R','Fixed fusion S'];vals=[m[k]['weighted_auroc'] for k in ['M','R','S']]
axs[1].barh(labels,vals,color=['#375e97','#ba7a40','#875898']);axs[1].set(xlim=(.5,1.02),xlabel='Weighted within-fold AUROC',title='It does not improve stroke detection');axs[1].invert_yaxis()
for i,v in enumerate(vals):axs[1].text(v-.012,i,f'{v:.3f}',ha='right',va='center',color='white',fontweight='bold')
for i,(delta,ci) in enumerate([(e['M']['observed'],e['M']['conditional_95_interval']),(demo['S_minus_M'],demo['conditional_95_interval'])]):axs[2].errorbar(delta,i,xerr=[[delta-ci[0]],[ci[1]-delta]],fmt='o',color='#a74340',capsize=5,markersize=7)
axs[2].axvline(0,color='#58646d',linestyle='--',linewidth=1);axs[2].set(yticks=[0,1],yticklabels=['All 178 participants','90 age/sex matched'],ylim=(1.6,-.6),xlim=(-.18,.02),xticks=[-.15,-.10,-.05,0.],xlabel='AUROC difference: S minus M',title='Both conditional intervals are negative');axs[2].grid(axis='x',alpha=.2)
fig.savefig(out/'external_results.svg');fig.savefig(out/'external_results.png',dpi=140);plt.close(fig)
ledger=pd.read_csv(BASE/'artifacts/iteration02/preparation/external_replication/trial_download_ledger.csv');fig,axs=plt.subplots(3,2,figsize=(10,7),layout='constrained',sharex='col')
for col,group in enumerate(['healthy','stroke']):
 row=ledger[ledger.group.eq(group)].sort_values('participant_id').iloc[0];x,t,_=extract_angles(row.selected_file)
 for kind in range(3):
  ax=axs[kind,col];ax.plot(t,x[:,2*kind],color='#2c6ca2',label='Left');ax.plot(t,x[:,2*kind+1],color='#ba6035',label='Right');ax.set_ylabel(['Hip flexion','Knee flexion','Ankle dorsiflexion'][kind]+' (deg)');ax.grid(alpha=.15)
  if kind==0:ax.set_title(f'{group.capitalize()} example: {row.participant_id.split(":")[1]}')
  if kind==2:ax.set_xlabel('Seconds from selected trial start')
fig.suptitle('Both legs share one raw clock\nBlue: left leg. Orange: right leg. Examples selected before scoring.',fontsize=13)
fig.savefig(out/'external_common_clock_examples.svg');fig.savefig(out/'external_common_clock_examples.png',dpi=140);plt.close(fig)
