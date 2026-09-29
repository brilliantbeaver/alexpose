"""Draw the proposed crossed experiment. Contains no empirical data."""
from pathlib import Path
import hashlib
import json
import os

BASE = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(BASE/'qa/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
INK, MUTED = '#233343', '#556573'
fig = plt.figure(figsize=(7, 1.85))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 7), ylim=(0, 1.85)); ax.axis('off')
ax.text(.02, 1.77, 'Cross actual movement with observation quality',
        size=10.5, weight='bold', color=INK, va='top')
ax.text(6.98, 1.77, 'PROPOSED', size=8, color=MUTED, ha='right', va='top')
for y, trial in [(1.00, 'A'), (.35, 'B')]:
    ax.text(.05, y+.20, f'Trial {trial}', weight='bold', color=INK, va='center')
    for x, condition, fill in [(1.02, 'Clear input', '#e9f3f8'),
                                (4.13, 'Occluded copy', '#f0f2f4')]:
        ax.add_patch(FancyBboxPatch((x, y), 2.8, .4,
                     boxstyle='round,pad=0,rounding_size=0.045',
                     facecolor=fill, edgecolor='#a3b5c0', linewidth=.65))
        ax.text(x+1.4, y+.2, condition, color=INK, ha='center', va='center')
    ax.annotate('', xy=(4.10, y+.2), xytext=(3.85, y+.2),
                arrowprops={'arrowstyle': '<->', 'color': MUTED, 'lw': .8})
ax.text(3.97, 1.48, 'Same recorded movement within each row',
        ha='center', color=MUTED, size=8)
for x in [2.42, 5.53]:
    ax.annotate('', xy=(x, .77), xytext=(x, .98),
                arrowprops={'arrowstyle': '->', 'color': '#0072B2', 'lw': .9})
ax.text(3.97, .86, 'A → B', ha='center', va='center', size=8.5, color='#0072B2')
ax.text(1.02, .10, 'Between rows: preserve each leg’s reference-measured change.',
        size=8.5, color=INK, va='center')
fig.canvas.draw(); renderer=fig.canvas.get_renderer()
for txt in fig.findobj(matplotlib.text.Text):
    if txt.get_visible() and txt.get_text():
        b = txt.get_window_extent(renderer)
        assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.x1+1 and b.y1 <= fig.bbox.y1+1, txt.get_text()
for ext in ['svg', 'pdf', 'png']:
    fig.savefig(BASE/f'figures/10-research-design.{ext}', dpi=300)
(BASE/'figures/10-research-provenance.json').write_text(json.dumps({
    'kind': 'conceptual experiment schematic', 'empirical_values': [],
    'width_inches': 7, 'height_inches': 1.85,
    'source': 'writeup/RESEARCH_DIRECTION.md',
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
}, indent=2)+'\n')
print('Built proposed-study diagram; no empirical marks.')
