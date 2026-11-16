"""Plot only completed source-level pilot evidence from its saved JSON report."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('report',type=Path)
a=parser.parse_args();r=json.loads(a.report.read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
keys=['rf','untrained_probe','jepa_normal','jepa_adapted','jepa_mixed']
labels=['Random Forest','Untrained probe','Normal JEPA probe','Staged JEPA probe','Mixed JEPA probe']
fig,(left,right)=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.5,1]})
fig.subplots_adjust(left=.2,right=.97,top=.77,bottom=.23,wspace=.65)
fig.suptitle('Bounded real-data pilot: one training seed',fontsize=19,fontweight='bold',y=.97)
fig.text(.2,.86,f"{r['classifiers']['rf']['n_sources']} test sources; sixteen sampled frames per retained sequence",fontsize=11)
values=[r['classifiers'][k]['binary_balanced_accuracy'] for k in keys]
left.barh(labels,values,color=['#477eac','#a6b0bc','#79b6ad','#458b80','#21685f'])
left.invert_yaxis();left.set_xlim(0,1.08);left.set_xlabel('Source-level binary balanced accuracy')
for i,v in enumerate(values):left.text(v+.015,i,f'{v:.3f}',va='center',fontsize=10)
right.barh(['JEPA forecast','Persistence'],[r['forecast_mse_mean'],r['persistence_mse_mean']],color=['#458b80','#d0a069'])
right.invert_yaxis();right.set_xlabel('Matched future coordinate MSE\nLower is better')
for ax in (left,right):ax.spines[['top','right']].set_visible(False)
ci=r['paired_jepa_minus_rf']['ci95']
fig.text(.2,.10,f"Paired staged-JEPA minus RF 95% source bootstrap interval: [{ci[0]:.3f}, {ci[1]:.3f}]",fontsize=10)
fig.text(.2,.045,'Short, fixed-frame pilot with unknown patient overlap and no external clinical validation.',fontsize=10)
for suffix in ('svg','png'):fig.savefig(ROOT/f'images/pilot_comparison.{suffix}',dpi=120)
print('Wrote images/pilot_comparison.svg and PNG from',a.report)
