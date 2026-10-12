from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(7.0,2.25));fig.subplots_adjust(0,0,1,1)
ax.set(xlim=(0,7),ylim=(0,2.25));ax.set_axis_off()
checks=[]
def box(x,y,w,h,title,body):
 rect=Rectangle((x,y),w,h,facecolor='#f2f6f8',edgecolor='#334e5c',linewidth=.9);ax.add_patch(rect)
 a=ax.text(x+w/2,y+h-.20,title,ha='center',va='center',fontweight='bold',fontsize=8.5,color='#10252f')
 b=ax.text(x+w/2,y+.27,body,ha='center',va='center',fontsize=8.2,color='#10252f',linespacing=1.5)
 checks.append((rect,[a,b]))
width=1.50;gap=.28;start=.08;y=1.31;h=.83
items=[('Video collection','91 clips\n41 source videos'),('Pose extraction','88 usable clips\n33 landmarks'),('Body coordinates','Center on hips\nScale by torso'),('Source partitions','5 outer folds\nThen form windows')]
for i,(title,body) in enumerate(items):
 x=start+i*(width+gap);box(x,y,width,h,title,body)
 if i<3:ax.annotate('',xy=(x+width+gap-.04,y+h/2),xytext=(x+width+.035,y+h/2),arrowprops={'arrowstyle':'->','lw':1.05,'color':'#334e5c'})
ax.plot([.08,6.92],[1.11,1.11],color='#9aaeb9',lw=.7)
for x,title,body in [(.08,'Training','Fit label-free encoder\nFit label classifiers'),(2.45,'Validation','Select 800 or 800+400\nupdates before testing'),(4.82,'Test','One prediction per clip\nSource / clip weighting')]:box(x,.08,2.10,.83,title,body)
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for rect,texts in checks:
 rb=rect.get_window_extent(renderer)
 for t in texts:
  tb=t.get_window_extent(renderer)
  assert rb.x0+2<=tb.x0 and tb.x1<=rb.x1-2 and rb.y0+2<=tb.y0 and tb.y1<=rb.y1-2,(t.get_text(),rb,tb)
fig.savefig(ROOT/'figures/pipeline-v5.pdf');fig.savefig(ROOT/'figures/pipeline-v5.png',dpi=240)
(ROOT/'qa/pipeline-v5-checks.json').write_text(json.dumps({'all_text_inside_panels':True,'font_points':[8.2,8.5],'geometry':'four equally sized process boxes; three equally sized partition boxes; straight connectors','panel_count':7},indent=2))
