"""Publication figures for the V9 workshop manuscript.

Render at the final 5.5-inch text width; requires Matplotlib and NumPy.
PDF uses embedded TrueType fonts. SVG retains editable text. All numerical
marks come from numerical_evidence.json; no synthetic participant images.
"""
from pathlib import Path
import argparse
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/fmts-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

INK = '#222222'
MUTED = '#565656'
BLUE = '#355F7C'
RULE = '#C8C8C8'
WIDTH = 396.0  # 5.5 inches, the official manuscript text width

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral'],
    'mathtext.fontset': 'stix', 'font.size': 9.5,
    'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'axes.edgecolor': INK, 'axes.linewidth': 0.55,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white', 'svg.hashsalt': 'fmts-v9-publication',
})


class Diagram:
    """Coordinates and font sizes are in physical points, not display pixels."""
    def __init__(self, height):
        self.height = height
        self.fig = plt.figure(figsize=(WIDTH / 72, height / 72))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, WIDTH), ylim=(height, 0))
        self.ax.axis('off')
        self.texts = []

    def text(self, x, y, label, size=9.5, color=INK, bold=False, ha='left', **kw):
        text = self.ax.text(x, y, label, fontsize=size, color=color,
                            fontweight='bold' if bold else 'normal',
                            ha=ha, va='center', linespacing=1.12, **kw)
        self.texts.append(text)
        return text

    def rule(self, y):
        self.ax.plot([0, WIDTH], [y, y], color=RULE, lw=0.55)

    def arrow(self, points, dashed=False, color=INK):
        if len(points) > 2:
            xs, ys = zip(*points[:-1])
            self.ax.plot(xs, ys, color=color, lw=0.65,
                         ls=(0, (2.5, 2)) if dashed else '-')
        self.ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle='-|>', mutation_scale=6,
            linewidth=0.65, color=color, shrinkA=0, shrinkB=0,
            linestyle=(0, (2.5, 2)) if dashed else '-'))

    def box(self, x, y, w, h, title, subtitle=None):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor='white',
                                   edgecolor=INK, linewidth=0.65))
        if subtitle:
            self.text(x+w/2, y+h/2-5, title, ha='center')
            self.text(x+w/2, y+h/2+5, subtitle, size=8.5, color=MUTED, ha='center')
        else:
            self.text(x+w/2, y+h/2, title, ha='center')

    def save(self, stem):
        return save(self.fig, stem, self.texts)


def save(fig, stem, texts=()):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    overflow = []
    for text in texts:
        box = text.get_window_extent(renderer)
        if (box.x0 < bounds.x0-1 or box.y0 < bounds.y0-1 or
                box.x1 > bounds.x1+1 or box.y1 > bounds.y1+1):
            overflow.append(text.get_text())
    if overflow:
        raise ValueError(f'Text outside figure: {overflow}')
    metadata = {'Creator': 'Matplotlib', 'CreationDate': None, 'ModDate': None}
    fig.savefig(stem.with_suffix('.pdf'), metadata=metadata)
    fig.savefig(stem.with_suffix('.svg'), metadata={'Date': None})
    # Review-only previews accompany the editable and typeset figure formats.
    fig.savefig(stem.with_suffix('.png'), dpi=240)
    plt.close(fig)
    return {'width_pt': WIDTH, 'height_pt': float(fig.get_figheight()*72),
            'minimum_font_pt': min(t.get_fontsize() for t in texts) if texts else None,
            'text_within_figure': True}


def pipeline(path, version):
    c = Diagram(202)
    c.text(0, 6, '(a)  Separate measurement and model input', size=10, bold=True)
    c.text(0, 27, 'Gait observations', bold=True)
    c.text(0, 39, 'Pose, visibility, time', size=8.5, color=MUTED)
    c.arrow([(87, 33), (101, 33), (101, 23), (114, 23)])
    c.arrow([(101, 33), (101, 49), (114, 49)])
    c.text(120, 20, 'Observed transitions')
    c.text(120, 32, 'Normalize; retain original time', size=8.5, color=MUTED)
    c.text(120, 46, 'Prepare encoder input')
    c.text(120, 58, 'Fill gaps; normalize; resize', size=8.5, color=MUTED)
    c.arrow([(238, 23), (281, 23)])
    c.arrow([(238, 49), (281, 49)])
    c.text(289, 20, r'Signed speed score $y$', color=BLUE)
    c.text(289, 32, 'Five left-right pair contrasts', size=8.5, color=MUTED)
    c.text(289, 46, r'Clip $X$: $64\times33\times3$')
    c.text(289, 58, 'Validity kept; time omitted', size=8.5, color=MUTED)

    c.rule(68)
    c.text(0, 78, '(b)  Label-free S-JEPA pretraining', size=10, bold=True)
    c.text(396, 78, 'Training sources only', size=8.5, color=MUTED, ha='right')
    c.text(0, 101, 'Visible patches')
    c.text(0, 133, r'Full prepared clip $X$')
    c.arrow([(87, 101), (107, 101)])
    c.arrow([(87, 133), (107, 133)])
    c.box(107, 90, 74, 22, 'Online encoder')
    c.box(213, 90, 68, 22, 'Predictor')
    c.box(107, 122, 74, 22, 'EMA teacher', 'No gradient')
    c.arrow([(181, 101), (213, 101)])
    c.arrow([(144, 112), (144, 122)], dashed=True)
    c.text(159, 117, 'EMA', size=8.5, color=MUTED)
    c.arrow([(281, 101), (326, 101)])
    c.arrow([(181, 133), (306, 133), (306, 113), (326, 113)])
    c.text(359, 100, 'Hidden-feature', ha='center', color=BLUE)
    c.text(359, 113, 'loss (CE)', ha='center', color=BLUE)
    c.text(231, 125, 'Teacher targets', size=8.5, color=MUTED, ha='center')
    c.text(0, 156, 'Two full views', size=9)
    c.arrow([(66, 156), (93, 156)])
    c.text(99, 156, 'Same encoder', size=9)
    c.arrow([(159, 156), (186, 156)])
    c.text(192, 156, 'Pool + projector', size=9)
    c.arrow([(264, 156), (326, 156)])
    c.text(359, 156, 'VICReg', ha='center', color=BLUE)

    c.rule(168)
    c.text(0, 179, '(c)  Frozen evaluation', size=10, bold=True)
    c.text(396, 179, 'Train-source fit; held-out source scores',
           size=8.5, color=MUTED, ha='right')
    c.text(0, 195, 'Initial / trained encoder', size=9)
    c.arrow([(92, 195), (108, 195)])
    c.text(114, 195, 'Paired summary', size=9)
    c.arrow([(181, 195), (213, 195)])
    c.text(219, 195, 'Ridge readout', size=9)
    c.arrow([(276, 195), (326, 195)])
    c.text(359, 195, r'Compare $\hat y$ with $y$', size=9, color=BLUE, ha='center')
    return c.save(path)


def seed_offsets(xs):
    # Exhaustive placement is tiny (five marks) and never changes a score's x.
    candidates = [0, -1.7, 1.7, -3.4, 3.4, -5.1, 5.1]
    def place(ys):
        i = len(ys)
        if i == len(xs):
            return ys
        for y in candidates:
            if all((xs[i]-xs[j])**2+(y-py)**2 >= 3.4**2-1e-9
                   for j, py in enumerate(ys)):
                result = place(ys+[y])
                if result is not None:
                    return result
        return None
    result = place([])
    if result is None:
        raise ValueError('Seed markers need more vertical space')
    return np.asarray(result)


def results(path, evidence, version):
    c = Diagram(136)
    c.text(0, 8, '(a)  Movement readout', size=10, bold=True)
    c.text(0, 21, 'Frozen features; five seeds per condition', size=8.5, color=MUTED)
    c.text(260, 8, '(b)  Latent-feature matching', size=10, bold=True)
    c.text(260, 21, 'Checks favoring the same clip', size=8.5, color=MUTED)

    rows = [evidence['readout_rows'][1]] + evidence['readout_rows'][2:7]
    labels = ['Initialization', 'Random (motion)', 'MAMP-style', 'Motion mixture',
              'Random (region)', 'Connected region']
    x0, x1 = 85, 216
    def xx(value):
        return x0+(x1-x0)*value/0.3
    for tick in [0, 0.1, 0.2, 0.3]:
        c.ax.plot([xx(tick), xx(tick)], [31, 108], color='#E2E2E2', lw=0.45, zorder=0)
        c.text(xx(tick), 120, f'{tick:.1f}', size=8.5, ha='center')
    c.ax.plot([x0, x1], [111, 111], color=INK, lw=0.55)
    c.text(242, 25, 'Mean', size=8.5, ha='right', color=MUTED)
    for i, (row, label) in enumerate(zip(rows, labels)):
        y = 37+i*13.4
        color = INK if i == 0 else BLUE
        c.text(0, y, label, size=9, bold=i == 0)
        values = np.asarray(row['per_seed_r2'])
        assert len(values) == 5
        assert np.isclose(values.mean(), row['mean_r2'], rtol=0, atol=1e-12)
        c.ax.scatter([xx(v) for v in values], y+seed_offsets([xx(v) for v in values]),
                     s=8, facecolors='white' if i == 0 else color,
                     edgecolors=color, linewidths=0.55, zorder=3)
        c.ax.plot([xx(row['mean_r2'])]*2, [y-4.4, y+4.4], color=INK, lw=0.8, zorder=4)
        c.text(242, y, f"{row['mean_r2']:.3f}", size=9, ha='right')
    c.text((x0+x1)/2, 130, r'Source-balanced $R^2$', size=9, ha='center')

    correspondence = evidence['correspondence']
    states = [('Initial', correspondence['initial_positive_rows'], correspondence['initial_rows'], INK),
              ('Trained', correspondence['trained_positive_rows'], correspondence['trained_rows'], BLUE)]
    bx0, bx1 = 266, 388
    for y, (label, num, den, color) in zip([54, 86], states):
        percentage = 100*num/den
        end = bx0+(bx1-bx0)*percentage/100
        c.text(260, y-12, label, size=9)
        c.text(394, y-12, f'{num}/{den} ({percentage:.0f}%)', size=9, ha='right', color=color)
        c.ax.plot([bx0, bx1], [y, y], color='#DEDEDE', lw=0.5, zorder=0)
        c.ax.plot([bx0, end], [y, y], color=color, lw=1.1, zorder=1)
        c.ax.scatter([end], [y], s=18, facecolors='white' if label=='Initial' else color,
                     edgecolors=color, linewidths=0.7, zorder=2)
    c.ax.plot([bx0, bx1], [111, 111], color=INK, lw=0.55)
    for tick in [0, 50, 100]:
        x = bx0+(bx1-bx0)*tick/100
        c.ax.plot([x, x], [111, 113.5], color=INK, lw=0.55)
        c.text(x, 120, str(tick), size=8.5, ha='center')
    c.text((bx0+bx1)/2, 130, 'Diagnostic comparisons (%)', size=9, ha='center')
    return c.save(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version',type=int,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    args=parser.parse_args()
    evidence=json.loads((args.assets/'numerical_evidence.json').read_text())
    metrics = {
        'training_pipeline_compact': pipeline(args.assets/'figures/training_pipeline_compact',args.version),
        'learning_results': results(args.assets/'figures/learning_results',evidence,args.version),
        'numerical_source': 'numerical_evidence.json',
        'seed_marks': 30,
        'same_clip_counts': evidence['correspondence'],
    }
    (args.assets/'figures/layout_checks.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (args.assets/'figures/FIGURE_NOTES.md').write_text('''# V9 figure production notes

Both figures are drawn at the final 5.5-inch manuscript width using Matplotlib.
PDF embeds TrueType fonts; SVG retains editable text; PNG is a review preview.
STIX serif typography, thin rules, square module outlines, white space and a
single muted blue accent replace the former colored cards and pose icons.
The smallest labels are 8.5 pt; panel titles are 10 pt. Output dimensions and
text-boundary checks are recorded in layout_checks.json.

Figure 1 has three ordered panels: (a) separate measurement and model-input
preparation, (b) label-free pretraining, and (c) frozen source-held-out evaluation.
Solid arrows carry data; the dashed EMA arrow updates teacher weights. The
teacher receives the complete prepared clip, and its targets enter the hidden-
feature loss. The full-view regularizer shares the online encoder. The target y
is used only for the separate ridge readout. Joint/position identities and patch
dimensions are described in the methods. The full-view pool uses
twelve gait joints; the signed target and bilateral readout use five pairs.
There are no schematic participant images or invented motion traces.

Figure 2 reads the 30 per-seed R-squared values and all six means directly from
numerical_evidence.json. Open black circles identify initialization; filled blue
circles identify trained teachers. A short black tick marks each mean; vertical
jitter only separates the five seeds. The latent-feature matching panel displays the recorded
33/75 and 375/375 same-clip diagnostic counts as percentages (44% and 100%).
These checks reuse source videos and models; no binomial interval or independent-
trial interpretation is introduced. Source uncertainty remains in Table 1.
No model fitting, inference, or bootstrap was rerun.
''')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
