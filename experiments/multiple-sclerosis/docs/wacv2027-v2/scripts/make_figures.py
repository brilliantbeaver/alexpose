"""Publication figures drawn from the notebook-associated saved predictions."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT.parent.parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
rows=json.loads((EXP/'artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/capstone-20260921T011305448893Z/oof.json').read_text())
if isinstance(rows,dict):
    rows=rows.get('rows',rows.get('predictions'))
labels=['normal','ms','pd']
def cm(data, model):
    count=Counter(r['source_id'] for r in data); a=np.zeros((3,3))
    for r in data: a[labels.index(r['true']),labels.index(r['pred_'+model])]+=1/count[r['source_id']]
    return a

def f1(a):
    return np.mean(np.divide(2*np.diag(a),a.sum(0)+a.sum(1),out=np.zeros(3),where=(a.sum(0)+a.sum(1))>0))

fig,axes=plt.subplots(1,3,figsize=(7,2.32),gridspec_kw={'width_ratios':[1.13,1,1]},layout='constrained')
ax=axes[0]
colors={'sjepa':'#176b89','mean_pose':'#bd5827'}
for model,label,marker in [('sjepa','S-JEPA','o'),('mean_pose','Mean pose','s')]:
 vals=[f1(cm([r for r in rows if r['fold']==i],model)) for i in range(5)]
 ax.plot(range(1,6),vals,marker=marker,color=colors[model],label=label,lw=1.0,ms=4)
ax.set(xlabel='Outer fold',ylabel='Source-weighted macro-F1',ylim=(0,1),xticks=range(1,6),title='(a) Variation across folds')
ax.legend(frameon=False,loc='upper left',fontsize=7)
for ax,model,title in zip(axes[1:],['sjepa','mean_pose'],['(b) S-JEPA','(c) Mean pose']):
 a=cm(rows,model); a=a/a.sum(1,keepdims=True)*100
 ax.imshow(a,vmin=0,vmax=100,cmap='Blues')
 for i in range(3):
  for j in range(3): ax.text(j,i,f'{a[i,j]:.1f}',ha='center',va='center',color='white' if a[i,j]>55 else 'black',fontsize=8)
 ax.set(xticks=range(3),xticklabels=['Normal','MS','PD'],yticks=range(3),yticklabels=['Normal','MS','PD'],xlabel='Predicted label',title=title)
 if model=='sjepa':ax.set_ylabel('Collection label')
 for spine in ax.spines.values():spine.set_visible(False)
 ax.tick_params(length=0)
fig.savefig(ROOT/'figures/folds-and-errors.pdf');fig.savefig(ROOT/'figures/folds-and-errors.png',dpi=200);plt.close(fig)

fig,ax=plt.subplots(figsize=(7,1.78));ax.set_axis_off();ax.set(xlim=(0,10),ylim=(0,2.6))
boxes=[(.05,1.35,2.2,.85,'91 video clips\n88 retained / 41 sources'),(2.65,1.35,2.2,.85,'Pixel landmarks\nBody-relative coordinates'),(5.25,1.35,2.15,.85,'Partition by source\nTrain / validation / test'),(7.95,1.35,2.0,.85,'32-frame windows\n33 landmarks')]
for x,y,w,h,txt in boxes:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025',fc='#eff5f8',ec='#5d7684',lw=.8));ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=8)
for x in [2.28,4.9,7.47]:ax.annotate('',xy=(x+.28,1.775),xytext=(x,1.775),arrowprops={'arrowstyle':'->','lw':.8})
ax.text(0.1,.67,'Training sources: label-free encoder + supervised probe',ha='left',fontsize=8)
ax.text(0.1,.18,'Validation: choose 800 or 800+400 updates',ha='left',fontsize=8)
ax.text(5.15,.67,'Test sources: one prediction per clip',ha='left',fontsize=8)
ax.text(5.15,.18,'Pool scores with equal source or equal clip weights',ha='left',fontsize=8)
fig.savefig(ROOT/'figures/pipeline.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures/pipeline.png',dpi=200,bbox_inches='tight');plt.close(fig)
print('Figures rendered from notebook-associated OOF rows.')
