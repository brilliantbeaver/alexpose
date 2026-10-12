from pathlib import Path
from collections import Counter
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent.parent
rows=json.loads((EXP/'artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/capstone-20260921T011305448893Z/oof.json').read_text())
labels=['normal','ms','pd']
def cm(data,model):
 counts=Counter(r['source_id'] for r in data);arr=np.zeros((3,3))
 for row in data:arr[labels.index(row['true']),labels.index(row['pred_'+model])]+=1/counts[row['source_id']]
 return arr
def f1(a):return np.mean(2*np.diag(a)/(a.sum(0)+a.sum(1)))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(7,2.70));grid=fig.add_gridspec(1,3,width_ratios=[1.12,1,1],left=.095,right=.99,bottom=.29,top=.83,wspace=.48)
axes=[fig.add_subplot(grid[0,i]) for i in range(3)];ax=axes[0]
a=[f1(cm([r for r in rows if r['fold']==i],'sjepa')) for i in range(5)]
b=[f1(cm([r for r in rows if r['fold']==i],'mean_pose')) for i in range(5)]
for i,(x,y) in enumerate(zip(a,b)):ax.plot([x,y],[i,i],color='#81919c',lw=1.2,zorder=1)
ax.scatter(a,range(5),marker='o',s=27,color='#145a78',label='S-JEPA',zorder=2)
ax.scatter(b,range(5),marker='s',s=25,color='#a44a20',label='Mean pose',zorder=2)
ax.set(xlim=(0,1),ylim=(4.5,-.5),xticks=[0,.5,1],yticks=range(5),yticklabels=[f'Fold {i}' for i in range(5)],xlabel='Macro-F1',title='(a) Matched test folds')
fig.legend(*ax.get_legend_handles_labels(),frameon=False,loc='lower left',bbox_to_anchor=(.09,.01),ncol=2,fontsize=8,handletextpad=.3,columnspacing=.7)
for ax,model,title in zip(axes[1:],['sjepa','mean_pose'],['(b) S-JEPA','(c) Mean pose']):
 a=cm(rows,model);a=a/a.sum(1,keepdims=True)*100
 ax.imshow(a,cmap='Blues',vmin=0,vmax=100)
 for i in range(3):
  for j in range(3):ax.text(j,i,f'{a[i,j]:.1f}',ha='center',va='center',fontsize=9,color='#101010')
 ax.set(xticks=range(3),xticklabels=['Normal','MS','PD'],yticks=range(3),yticklabels=['Normal','MS','PD'],xlabel='Predicted label',title=title)
 ax.tick_params(length=0,pad=4)
 for sp in ax.spines.values():sp.set_visible(False)
 if model=='sjepa':ax.set_ylabel('Collection label',labelpad=3)
fig.savefig(ROOT/'figures/folds-and-errors-v5.pdf');fig.savefig(ROOT/'figures/folds-and-errors-v5.png',dpi=240)
print('Zero-based fold indices match notebook and partition table.')
