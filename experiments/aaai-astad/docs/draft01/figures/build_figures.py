"""Render the method diagram with embedded TrueType text and editable SVG."""
from pathlib import Path
out=Path(__file__).resolve().parent
# Use embedded TrueType text for the manuscript PDF. On this macOS setup,
# librsvg's Arial export introduces Type 3 fallback glyphs, which AAAI disallows.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.family':'DejaVu Sans'})
fig=plt.figure(figsize=(9.6,3.7));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,960),ylim=(0,370));ax.axis('off')
def mtext(x,y,t,n=20):ax.text(x,370-y,t,fontsize=n*.72,color='#183344',va='baseline')
def mbox(x,y,w,title,lines):
 ax.add_patch(FancyBboxPatch((x,370-y-105),w,105,boxstyle='round,pad=0,rounding_size=8',facecolor='#eaf4f4',edgecolor='#c2d2d9',linewidth=.7))
 mtext(x+12,y+28,title,20)
 for i,t in enumerate(lines):mtext(x+12,y+57+25*i,t,18)
def marrow(x,y,x2):ax.add_patch(FancyArrowPatch((x,370-y),(x2,370-y),arrowstyle='-|>',mutation_scale=12,linewidth=1.1,color='#526378'))
mtext(20,29,'Normal sources fit the encoder and six separate heads',23)
for x,w,t,ls in [(20,213,'Uniform masking',['16 joints × 8 patches','Same hidden targets']),(270,200,'Frozen encoder',['Own / patch / full','Restricted inputs']),(515,195,'Two target types',['Local coordinates','Contextual teacher']),(755,185,'Matched losses',['Own minus full','12 readout joints'])]:mbox(x,50,w,t,ls)
for x,x2 in [(235,265),(470,510),(710,750)]:marrow(x,100,x2)
mtext(20,197,'Each fold separates fitting, normal reference, and outer test sources',22)
for x,t,ls in [(20,'Three-region profile',['Left leg • right leg • arms','Equal joint/source weights']),(345,'Five normal references',['Freeze mean and variation','Check more or less help']),(670,'Outer test sources',['Rank within each fold','Compare identical targets'])]:mbox(x,218,270,t,ls)
marrow(290,270,340);marrow(615,270,665)
mtext(20,358,'The score describes model prediction. It does not identify neural causes.',18)
for ext in ('svg','pdf','png'):fig.savefig(out/f'method.{ext}',dpi=180)
plt.close(fig)
