"""Compact primary-contrast figure; preserves the v08 estimates and intervals."""
from pathlib import Path
import hashlib
import json
import os

BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE.parent/'iclr/versions/v08/evidence/summary-figure-provenance.json'
os.environ.setdefault('MPLCONFIGDIR',str(BASE/'qa/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK,MUTED,GRID='#233343','#556573','#d8e0e5'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,
    'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6})
data=json.loads(SOURCE.read_text())['selected_comparisons']
assert [r['id'] for r in data]==['core','difference','repair']
fig=plt.figure(figsize=(7,2.35))
fig.text(.015,.97,'Three primary comparisons leave the added benefit uncertain',
         fontsize=10.5,weight='bold',va='top')
ax=fig.add_axes([.66,.23,.32,.59])
ax.set(xlim=(-1.55,2.3),ylim=(-.45,2.45),xticks=[-1,0,1,2],yticks=[],xlabel='Gain (degrees)')
ax.tick_params(axis='x',labelsize=8,length=3,pad=3)
ax.set_axisbelow(True);ax.grid(axis='x',color=GRID,lw=.6)
ax.axvline(0,color=MUTED,lw=.8)
for side in ['left','right','top']:ax.spines[side].set_visible(False)
ax.spines['bottom'].set_color(MUTED)
labels=[('Core JEPA vs direct, both with change objective','Asymmetry-response error'),
        ('Delta vs endpoint feature training','Asymmetry-response error'),
        ('Dense vs low-scalar readout, delta encoder','Knee-trajectory error on ViTPose')]
for row,(r,(label,measure)) in enumerate(zip(data,labels)):
    y=2-row
    col='#0072B2' if row==0 else '#D55E00'
    m,lo,hi=[r[k] for k in ['gain_deg','ci95_low_deg','ci95_high_deg']]
    assert lo<0<hi and r['people']==14 and r['fitted_seeds']==3
    ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt='s' if row==2 else 'o',
                color=col,ms=4.5,lw=1.1,capsize=2.5)
    ax.text(2.24,y+.22,f'{m:.2f} [{lo:.2f}, {hi:.2f}]',ha='right',fontsize=8,color=col)
    fy=.23+.59*(y+.45)/2.9
    fig.text(.018,fy+.018,label,fontsize=8.5,weight='bold',va='center')
    fig.text(.018,fy-.051,measure,fontsize=8,color=MUTED,va='center')
fig.text(.018,.02,'Positive gains favor the candidate. Bars show 95% intervals; outcomes are not pooled.',
         fontsize=8,color=MUTED)
fig.canvas.draw(); renderer=fig.canvas.get_renderer()
for txt in fig.findobj(matplotlib.text.Text):
    if txt.get_visible() and txt.get_text():
        b=txt.get_window_extent(renderer)
        assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.x1+1 and b.y1<=fig.bbox.y1+1,txt.get_text()
for ext in ['svg','pdf','png']:fig.savefig(BASE/f'figures/09-overview-results.{ext}',dpi=300)
record={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'width_inches':7,'height_inches':2.35,'new_inference':False,'plotted_values':data}
(BASE/'figures/09-overview-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print('Built compact figure from three unchanged primary contrasts.')
