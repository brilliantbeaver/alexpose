"""Data-derived, version-local figures. No training or invented reconstructions.

Run: .venv/bin/python docs/iclr/versions/v08/scripts/build_figures.py --prototype
or omit --prototype for the complete figure set after analyze_evidence.py.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.stats import t
from paper_style import *
from matplotlib.lines import Line2D

ROOT = next((p for p in HERE.parents if (p/'outputs/iclr').is_dir()), HERE/'inputs')
INPUTS = {}
DERIVED = {}
def read(rel):
    local_prefix = 'docs/iclr/versions/v08/'
    p = HERE / rel[len(local_prefix):] if rel.startswith(local_prefix) else ROOT / rel
    INPUTS[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p) if p.suffix == '.csv' else json.loads(p.read_text())

def person_mean(d, metric):
    return d.groupby('canonical_person_id')[metric].mean()

def mean_ci(d, metric):
    v = person_mean(d, metric).to_numpy()
    assert len(v) == 14
    m = float(v.mean()); h = float(t.ppf(.975,13)*v.std(ddof=1)/np.sqrt(14))
    return m, m-h, m+h

def paired_crossed(d, a, b, metric):
    # Positive is b - a. Same person/seed draws for both methods.
    p = d[d.method.isin([a,b])].pivot(index=['canonical_person_id','seed'], columns='method', values=metric)
    z = (p[b]-p[a]).unstack('seed').sort_index().to_numpy()
    assert z.shape == (14,3) and np.isfinite(z).all()
    rng = np.random.default_rng(731)
    draws = [z[rng.integers(0,14,14)][:,rng.integers(0,3,3)].mean() for _ in range(2000)]
    return float(z.mean()), *np.quantile(draws,[.025,.975]).tolist()

def method_color(key):
    if key.startswith('P-direct'): return DIRECT
    if 'jepa_endpoint' in key: return ENDPOINT
    if 'jepa_delta' in key: return DELTA
    return GRAY

def schematic():
    """Conceptual keypoints and experimental workflow; no empirical pose data."""
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(5.5, 3.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 5.5), ylim=(0, 3.15)); ax.axis('off')
    neutral='#F1F3F5'; edit='#008568'
    def txt(x,y,s,size=8.5,**kw):
        ax.text(x,y,s,fontsize=size,va='center',**kw)
    def box(x,y,w,h,label='',fill='white',edge=GRAY,size=8.5,weight='normal'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,
            boxstyle='round,pad=0.018,rounding_size=0.035',
            facecolor=fill,edgecolor=edge,linewidth=.75))
        if label:txt(x+w/2,y+h/2,label,size,ha='center',fontweight=weight,linespacing=1.08)
    def arr(a,b,color=GRAY):
        ax.annotate('',xy=b,xytext=a,arrowprops={
            'arrowstyle':'->','lw':.8,'color':color,'shrinkA':2,'shrinkB':2})
    def pose(cx,cy,edited=False):
        # Twelve illustrative 2D keypoints: shoulders, elbows, wrists, hips,
        # knees, ankles. These hand-designed icons are not data or predictions.
        pts=np.array([[-.19,.89],[.13,.85],[-.30,.65],[.30,.62],[-.23,.42],
                      [.36,.41],[-.12,.47],[.13,.45],[-.29,.25],[.30,.27],
                      [-.37,.015],[.18,.025]],float)
        if edited:
            theta=np.deg2rad(28.)  # Drawing choice, not a reported intervention.
            rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            pts[11]=pts[9]+rot@(pts[11]-pts[9])
        points=pts*.84+np.array([cx,cy])
        edges=[(0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7),(6,8),(8,10),(7,9),(9,11)]
        for a,b in edges:
            col=edit if edited and (a,b) in [(7,9),(9,11)] else GRAY
            ax.plot(points[[a,b],0],points[[a,b],1],color=col,lw=1.1,zorder=3)
        ax.scatter(points[:,0],points[:,1],s=8,facecolor='white',edgecolor=INK,lw=.7,zorder=4)
        if edited:ax.scatter(points[[7,9,11],0],points[[7,9,11],1],s=9,facecolor='white',edgecolor=edit,lw=.9,zorder=5)
    # A readable motion icon links the abstract paired objective to human gait.
    txt(.05,3.00,'1  Build a controlled gait pair',9.5,fontweight='bold')
    txt(1.11,2.79,'Illustrative keypoints',8.0,ha='center',color=GRAY)
    for x in [.11,1.43]:
        for shift in [.06,.03,0]:
            box(x+shift,1.77+shift,.73,.87,fill='white',edge=LIGHT)
    pose(.53,1.815,False);pose(1.85,1.815,True)
    arr((.91,2.20),(1.38,2.20),edit)
    txt(1.145,2.43,'Synthetic\nknee edit',8.0,ha='center',color=edit,linespacing=1.08)
    txt(.51,1.59,r'Original window $a$',8.5,ha='center')
    txt(1.83,1.59,r'Edited window $b$',8.5,ha='center')
    txt(1.15,1.36,'Each window: 128 frames × 12 joints',8.0,ha='center')
    txt(1.15,1.16,'Noisy 2D poses + projected references',8.0,ha='center')

    ax.plot([2.38,2.38],[1.08,3.08],color=LIGHT,lw=.65)
    txt(2.52,3.00,'2  Fit on 112 training people',9.5,fontweight='bold')
    txt(2.56,2.73,'Noisy poses',8.3)
    arr((3.36,2.73),(3.66,2.73))
    txt(3.75,2.73,'Student + predictor',8.5)
    txt(2.56,2.48,'Clean references',8.3)
    arr((3.57,2.48),(3.85,2.48))
    txt(3.94,2.48,'Teacher targets',8.5)
    txt(3.97,2.23,'Same base loss; separate runs',8.2,ha='center')
    txt(3.14,2.04,'Endpoint',8.8,ha='center',fontweight='bold',color=ENDPOINT)
    txt(4.79,2.04,'Delta',8.8,ha='center',fontweight='bold',color=DELTA)
    box(2.55,1.62,1.18,.31,'Match each\nmotion’s features',fill='#F1EDF6',edge=ENDPOINT,size=8.4)
    txt(3.97,1.775,'OR',7.9,ha='center',color=GRAY)
    box(4.20,1.62,1.18,.31,'Match their\nfeature difference',fill='#FCF0E8',edge=DELTA,size=8.4)
    txt(3.97,1.39,'Freeze encoder → fit a readout',8.7,ha='center',fontweight='bold')
    txt(3.97,1.16,'Readout targets: reference coordinates',8.0,ha='center')

    ax.plot([.05,5.45],[1.00,1.00],color=LIGHT,lw=.65)
    txt(.05,.85,'3  Evaluate on 14 development people',9.5,fontweight='bold')
    txt(5.43,.85,'3 fitted seeds',8.1,ha='right',color=GRAY)
    box(.08,.32,1.20,.33,'Observed window\n'+r'$a$ or $b$',size=8.5)
    box(1.62,.32,1.28,.33,'Frozen encoder\n+ trained readout',fill=neutral,size=8.4)
    arr((1.30,.485),(1.60,.485))
    arr((2.92,.485),(3.42,.485))
    txt(3.18,.65,'Restored\njoints',7.6,ha='center',linespacing=1.05)
    box(3.44,.27,1.97,.45,'Paired asymmetry-change error\nKnee-angle trajectory error\nLeft–right naming (post hoc)',size=8.25)
    arr((4.43,.075),(4.43,.25))
    txt(4.26,.09,r'References $a,b$',8.0,ha='right')
    txt(.08,.12,'Restore separately; score the change from both windows.',7.7)
    save(fig,'method')

FAMILIES = [
 ('P-direct-none','Direct (joint)'),
 ('I-initialized-none','Initialized'),
 ('I-shuffled_jepa-graph_time','Shuffled JEPA'),
 ('M-coordinate-graph_time','Coordinate'),
 ('M-paired_jepa-graph_time','Core JEPA'),
 ('F-response-coordinate_delta_v1-graph_time','Coordinate delta'),
 ('F-response-jepa_endpoint_v1-graph_time','Endpoint JEPA'),
 ('F-response-jepa_delta_v1-graph_time','Delta JEPA')]

def restoration(d):
    fig = plt.figure(figsize=(5.5,3.25))
    gs = fig.add_gridspec(1,3,left=.215,right=.99,bottom=.19,top=.84,wspace=.26,width_ratios=[1,1,1.05])
    axes=[fig.add_subplot(gs[0,i]) for i in range(3)]
    ys=[8.5,6.8,5.8,4.8,3.8,2.8,1.8,.8]
    labels=[n for _,n in FAMILIES]
    rows=[]
    for a in axes:
        a.set_ylim(-.05,9.2);a.set_yticks(ys);axis_style(a)
    axes[0].set_yticklabels(labels);axes[1].set_yticklabels([]);axes[2].set_yticklabels([])
    for y,(key,name) in zip(ys,FAMILIES):
        col=method_color(key)
        for a,metric in zip(axes[:2],['response_error','waveform_error']):
            u=d[d.method.eq(key+'-base')][metric].mean()
            v=d[d.method.eq(key+'-paired_change')][metric].mean()
            a.plot([u,v],[y,y],color=LIGHT,lw=1.1,zorder=2)
            a.plot(u,y,'o',color=col,ms=3.7,zorder=3)
            a.plot(v,y,'^',mec=col,mfc='white',ms=4.5,mew=1,zorder=4)
        eff,lo,hi=paired_crossed(d,key+'-base',key+'-paired_change','waveform_error')
        axes[2].errorbar(eff,y,xerr=[[eff-lo],[hi-eff]],fmt='s',color=col,ms=3.3,lw=1,capsize=2)
        rows.append({'family':key,'change_minus_coordinate_waveform':eff,'ci_low':lo,'ci_high':hi})
    for a in axes:
        a.axhline(7.6,color=LIGHT,lw=.6)
    axes[0].text(-.05,7.25,'Frozen encoder',transform=axes[0].get_yaxis_transform(),ha='right',va='center',fontsize=7.5,color=GRAY)
    z=d.zero_response_error.mean()
    axes[0].axvline(z,color=INK,ls=':',lw=1)
    for a,lim in zip(axes,[(0,25),(0,30),(-1,14)]):a.set_xlim(*lim)
    axes[0].set_xticks([0,10,20]);axes[1].set_xticks([0,10,20,30]);axes[2].set_xticks([0,5,10])
    axes[2].axvline(0,color=GRAY,lw=.6)
    for a,title,xlab in zip(axes,['A  Response','B  Waveform','C  Waveform effect'],['Error (°)','Error (°)','Change − coordinate (°)']):
        panel(a,title);a.set_xlabel(xlab,labelpad=5)
    handles=[Line2D([],[],marker='o',linestyle='',color=INK,label='Coordinate objective'),Line2D([],[],marker='^',mfc='white',linestyle='',color=INK,label='Original change package'),Line2D([],[],linestyle=':',color=INK,label='Zero (5.81°)')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,1.015),ncol=3,frameon=False,columnspacing=.85,handletextpad=.3,handlelength=1.2)
    DERIVED['restoration_waveform_effects']=rows
    save(fig,'restoration')

def reliability(d):
    e=read('docs/iclr/versions/v08/evidence/penalty_effects.csv')
    e=e[e.contrast.eq('delta versus endpoint')].sort_values('failure_penalty_deg')
    assert len(e)==4
    fig=plt.figure(figsize=(5.5,2.55))
    gs=fig.add_gridspec(1,3,left=.19,right=.985,bottom=.25,top=.76,width_ratios=[.9,1.3,1.2],wspace=.46)
    axes=[fig.add_subplot(gs[0,i]) for i in range(3)]
    keys=['P-direct-none-base','F-response-jepa_endpoint_v1-graph_time-paired_change','F-response-jepa_delta_v1-graph_time-paired_change']
    names=['Direct / coordinate','Endpoint / change','Delta / change']
    for yi,k in zip([2,1,0],keys):
        row=d[d.method.eq(k)].mean(numeric_only=True)
        axes[0].plot(100*row.response_failure_rate,yi,'o',color=method_color(k),ms=4.4)
        axes[0].annotate(f'{100*row.response_failure_rate:.3f}',(100*row.response_failure_rate,yi),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7.7)
        axes[1].barh(yi,row.response_success_contribution,height=.43,facecolor='#b4b4b4',edgecolor=INK,lw=.45)
        axes[1].barh(yi,row.response_failure_contribution,left=row.response_success_contribution,height=.43,facecolor='white',edgecolor=INK,lw=.5,hatch='////')
    for a in axes[:2]:
        a.set(ylim=(-.65,2.75),yticks=[2,1,0]);axis_style(a)
    axes[0].set_yticklabels(names);axes[1].set_yticklabels([])
    axes[0].set(xlim=(0,.8),xticks=[0,.4,.8],xlabel='Failure rate (%)')
    axes[1].set(xlim=(0,12),xticks=[0,6,12],xlabel='Response score (°)')
    for yi,(_,r) in zip([3,2,1,0],e.iterrows()):
        axes[2].errorbar(r.estimate,yi,xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],fmt='o' if r.failure_penalty_deg==720 else 'o',mfc=INK if r.failure_penalty_deg==720 else 'white',color=INK,ms=4,capsize=2)
    axes[2].set(ylim=(-.65,3.65),yticks=[3,2,1,0],yticklabels=['0°','180°','360°','720°'],xlim=(-1.45,1.95),xticks=[-1,0,1],xlabel='Endpoint − delta (°)')
    axes[2].axvline(0,color=GRAY,lw=.7);axis_style(axes[2])
    for a,title in zip(axes,['A  Failure frequency','B  Score components','C  Effect by cost']):panel(a,title)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc='#b4b4b4',ec=INK,lw=.4,label='Successful contribution'),Patch(fc='white',ec=INK,lw=.4,hatch='////',label='Failure cost × rate')],loc='upper center',bbox_to_anchor=(.59,1.015),ncol=2,frameon=False,handlelength=1.3,columnspacing=1)
    save(fig,'reliability')

def repair():
    means=read('docs/iclr/versions/v08/evidence/repair_means.csv')
    eff=read('docs/iclr/versions/v08/evidence/repair_effects.csv')
    means=means[means.metric.eq('waveform_error')].set_index('method')
    eff=eff[eff.metric.eq('waveform_error')]
    fig=plt.figure(figsize=(5.5,3.75))
    gs=fig.add_gridspec(2,2,left=.22,right=.985,bottom=.13,top=.90,hspace=.78,wspace=.24,height_ratios=[1.25,1])
    axs=[[fig.add_subplot(gs[i,j]) for j in range(2)] for i in range(2)]
    for j,(var,color,name) in enumerate([('jepa_delta_v1',DELTA,'Delta'),('jepa_endpoint_v1',ENDPOINT,'Endpoint')]):
        ax=axs[0][j]
        keys=['P-direct-none-base',f'F-response-{var}-graph_time-paired_change',f'R-repair-{var}-scalar_low',f'R-repair-{var}-dense_change']
        for y,k,mark in zip([3.8,2.4,1.4,.4],keys,['o','^','D','s']):
            r=means.loc[k];co=DIRECT if k.startswith('P-') else color
            ax.errorbar(r.estimate,y,xerr=[[r.estimate-r.person_t_low],[r.person_t_high-r.estimate]],fmt=mark,color=co,mfc='white' if mark=='^' else co,ms=4,capsize=2)
            ax.text(r.estimate+.1,y+.29,f'{r.estimate:.2f}',fontsize=7.5,ha='center')
        ax.axhline(3.05,color=LIGHT,lw=.6)
        ax.set(ylim=(-.2,4.4),xlim=(9,28),xticks=[10,15,20,25],yticks=[3.8,2.4,1.4,.4],yticklabels=['Direct (joint)','Original change','Low scalar','Dense'] if j==0 else [],xlabel='Waveform error (°)')
        axis_style(ax);panel(ax,f'{"AB"[j]}  {name}'+(' (primary family)' if j==0 else ' (secondary family)'))
        ax=axs[1][j]
        for y,contrast in zip([2,1,0],['low scalar versus original','dense versus original','dense versus low scalar']):
            r=eff[eff.variant.eq(var)&eff.contrast.eq(contrast)].iloc[0]
            ax.errorbar(r.estimate,y,xerr=[[r.estimate-r.person_t_low],[r.person_t_high-r.estimate]],fmt='s' if y==0 else 'o',color=color,mfc=color if y==0 else 'white',ms=4,capsize=2)
        ax.set(ylim=(-.55,2.55),xlim=(-.65,6.2),xticks=[0,2,4,6],yticks=[2,1,0],yticklabels=['Low vs original','Dense vs original','Dense vs low'] if j==0 else [],xlabel='Waveform improvement (°)')
        ax.axvline(0,color=GRAY,lw=.65);axis_style(ax);panel(ax,f'{"CD"[j]}  Paired gains: {name.lower()}')
    save(fig,'repair')

def laterality():
    d=read('docs/iclr/versions/v08/evidence/condition_summary.csv')
    d=d[d.stratum_type.eq('naming')&d.metric.eq('assignment_failure_rate')]
    keys=['P-direct-none-base','F-response-jepa_endpoint_v1-graph_time-paired_change','F-response-jepa_delta_v1-graph_time-paired_change']
    fig=plt.figure(figsize=(5.5,1.95))
    gs=fig.add_gridspec(1,3,left=.21,right=.98,bottom=.27,top=.77,wspace=.20)
    for j,(cond,name) in enumerate([('correct','Correct names'),('global_swap','Global swap'),('temporary_swap','Temporary swap')]):
        ax=fig.add_subplot(gs[0,j])
        for y,k in zip([2,1,0],keys):
            r=d[d.stratum.eq(cond)&d.method.eq(k)].iloc[0]
            ax.errorbar(100*r.estimate,y,xerr=[[100*(r.estimate-r.ci_low)],[100*(r.ci_high-r.estimate)]],fmt='o' if k.startswith('P-') else '^',color=method_color(k),ms=4.1,mfc=method_color(k) if k.startswith('P-') else 'white',capsize=2)
        ax.set(ylim=(-.5,2.5),xlim=(0,100),xticks=[0,50,100],yticks=[2,1,0],yticklabels=['Direct / coordinate','Endpoint / change','Delta / change'] if j==0 else [],xlabel='Assignment failure (%)')
        axis_style(ax);panel(ax,f'{"ABC"[j]}  {name}')
    save(fig,'laterality')

def main():
    args=argparse.ArgumentParser();args.add_argument('--prototype',action='store_true');opt=args.parse_args()
    d=read('outputs/iclr/jepa-response/evaluation/per-person.csv')
    assert d.groupby('method').size().eq(42).all()
    schematic();restoration(d)
    if not opt.prototype:
        reliability(d)
        repair()
        laterality()
    (HERE/'figures'/'provenance.json').write_text(json.dumps({'inputs_sha256':INPUTS,'derived':DERIVED,'population':'14 development people, three fitted seeds; paired hierarchical exported cells','intervals':'Crossed bootstrap: 2000 draws seed731 unless caption explicitly says person-t conditional on three fitted seeds.'},indent=2)+'\n')

if __name__=='__main__': main()
