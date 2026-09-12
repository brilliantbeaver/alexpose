"""Plot a completed direct-v3 development comparison without modifying its run."""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.run_root.resolve()):
        parser.error('Choose an output outside the sealed run')
    scores=pd.read_csv(args.run_root/'reports/aggregate-metrics.csv').set_index('arm')
    uncertainty=json.loads((args.run_root/'reports/uncertainty.json').read_text())
    arms=['real-skeleton','time-shuffle','clip-mismatch','no-skeleton']
    points=scores.loc[arms,'delta_r2'].to_numpy()
    bounds=np.array([[uncertainty['intervals'][a]['delta_r2'][k] for a in arms] for k in ['ci025','ci975']])
    # Percentile bootstrap intervals need not contain the point estimate.
    fig,axes=plt.subplots(1,3,figsize=(14,4),gridspec_kw={'width_ratios':[1.4,1.1,1]})
    ax=axes[0]
    ax.hlines(np.arange(4),bounds[0],bounds[1],color='#4d788e',linewidth=2)
    ax.scatter(points,np.arange(4),c=['#ba594b','#4d788e','#4d788e','#69836d'],s=45,zorder=3)
    ax.axvline(0,c='gray',linewidth=.8)
    ax.set(yticks=range(4),yticklabels=arms,xlabel='Gain over shared RGB ridge (R²)',title='Held-source gains and 95% intervals')
    ax.invert_yaxis(); ax.ticklabel_format(axis='x',style='sci',scilimits=(-2,2))
    matrix=np.zeros((4,5),dtype=int)
    for fold in range(5):
        for row in json.loads((args.run_root/f'models/fold-{fold}/fit-diagnostics.json').read_text()):
            matrix[arms.index(row['arm']),fold]=row['selected_type']=='joint_ridge'
    from matplotlib.colors import ListedColormap
    axes[1].imshow(matrix,vmin=0,vmax=1,cmap=ListedColormap(['#e8edeb','#4d788e']),aspect='auto')
    for a in range(4):
        for f in range(5): axes[1].text(f,a,'joint' if matrix[a,f] else 'RGB',ha='center',va='center',fontsize=9,color='white' if matrix[a,f] else '#26332b')
    axes[1].set(xticks=range(5),yticks=[],xlabel='Outer fold',title=f'{int((matrix == 0).sum())} / {matrix.size} selections use RGB only')
    new=[]
    for fold,scope in [(1,'inner-1'),(3,'inner-0')]:
        pre=json.loads((args.run_root/f'models/fold-{fold}/preprocessing.json').read_text())
        new.append(pre[scope]['heldout_x']['transformed_max_abs'])
    x=np.arange(2)
    axes[2].bar(x-.16,[1e8]*2,width=.32,label='Historical',color='#ba594b')
    axes[2].bar(x+.16,new,width=.32,label='Repaired',color='#4d788e')
    axes[2].set(yscale='log',xticks=x,xticklabels=['outer 1 / inner 1','outer 3 / inner 0'],ylabel='Maximum |transformed RGB/nuisance input|',title='Ordinary missingness stays finite')
    axes[2].tick_params(axis='x',labelsize=8); axes[2].legend(fontsize=8)
    cohort=pd.read_csv(args.run_root/'manifests/gate-windows.csv')
    decision=json.loads((args.run_root/'reports/gate-decision.json').read_text())['decision']
    fig.suptitle(f'direct-v3 development {decision} · {len(cohort)} clips / {cohort.video_id.nunique()} sources · deterministic prediction',fontsize=13)
    fig.text(.02,.015,'Intervals condition on saved fitted models. Required real gain is +0.05 R² (outside the displayed gain range).',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.93))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180,bbox_inches='tight')
    fig.savefig(args.output.with_suffix('.svg'),bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__': main()
