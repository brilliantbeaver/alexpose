"""Shared, physical-size figure style for the v08 manuscript."""
from pathlib import Path
import os
HERE = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(HERE / 'qa' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

INK = '#252525'
GRAY = '#787878'
LIGHT = '#dddddd'
DIRECT = '#0072B2'
ENDPOINT = '#7B5AA6'
DELTA = '#D55E00'
COLORS = {'direct': DIRECT, 'endpoint': ENDPOINT, 'delta': DELTA}
plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
    'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8.5, 'legend.fontsize': 8,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': GRAY,
    'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': .65,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': .65, 'ytick.major.width': .65,
    'xtick.major.size': 2.8, 'ytick.major.size': 0,
    'lines.linewidth': 1.1, 'lines.markersize': 4,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'hatch.linewidth': .55,
})

def save(fig, name):
    """Keep physical dimensions fixed; never crop to a changing bounding box."""
    out = HERE / 'figures'
    out.mkdir(exist_ok=True)
    fig.canvas.draw()
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out / f'{name}.{ext}', dpi=200)
    Image.open(out / f'{name}.png').convert('L').save(out / f'{name}-gray.png')
    plt.close(fig)

def axis_style(ax, xgrid=True):
    if xgrid:
        ax.grid(axis='x', color=LIGHT, lw=.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)

def panel(ax, text):
    ax.set_title(text, loc='left', pad=8, fontweight='bold')
