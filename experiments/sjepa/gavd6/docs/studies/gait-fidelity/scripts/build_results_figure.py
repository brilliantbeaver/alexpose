"""Draw the proposal's completed-core results with Matplotlib.

Full: two matched-scale mean panels and a separate paired-contrast interval panel.
Compact: the same means, sized for the two-page overview. SVGs retain vector text;
PNGs are rendered directly by Matplotlib. All values come from the retained CSVs.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'gait-results-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / 'images'
DATA = ROOT / 'results/core-analysis-20260924'
INK, MUTED = '#223747', '#62717B'
COORD, CHANGE = '#276494', '#B86523'
FAMILIES = [
    ('Direct', 'P-direct-none-base', 'P-direct-none-paired_change'),
    ('JEPA', 'M-paired_jepa-graph_time-base', 'M-paired_jepa-graph_time-paired_change'),
]
METRICS = [('response_error', 'Movement-response error'),
           ('waveform_error', 'Knee-angle trajectory error')]


def read_data():
    with (DATA / 'method-means.csv').open(newline='') as stream:
        means = {row['method']: row for row in csv.DictReader(stream)}
    with (DATA / 'paired-contrasts.csv').open(newline='') as stream:
        contrasts = {row['metric']: row for row in csv.DictReader(stream)
                     if row['comparison'] == 'registered'}
    return means, contrasts


def check_layout(fig):
    """Check actual rendered text extents, not approximate character counts."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    collisions, outside, line_collisions = [], [], []
    for i, (text, box) in enumerate(texts):
        if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1):
            outside.append(text.get_text())
        for other, obox in texts[i + 1:]:
            intersection = Bbox.intersection(box, obox)
            if intersection is not None and intersection.width > 0.7 and intersection.height > 0.7:
                collisions.append([text.get_text(), other.get_text()])
    assert not outside, f'Text outside figure: {outside}'
    assert not collisions, f'Overlapping text: {collisions}'
    for ax in fig.axes:
        for label in getattr(ax, '_result_labels', []):
            box = label.get_window_extent(renderer)
            for line in getattr(ax, '_result_lines', []):
                path = line.get_transform().transform_path(line.get_path())
                if path.intersects_bbox(box, filled=False):
                    line_collisions.append(label.get_text())
    assert not line_collisions, f'Connectors/references intersect value labels: {line_collisions}'
    return {'text_elements': len(texts), 'text_overlaps': collisions,
            'text_outside_figure': outside,
            'value_labels_intersected_by_lines': line_collisions,
            'smallest_font_pt': min(t.get_fontsize() for t, _ in texts)}


def means_panels(fig, means, compact):
    bottom, height = ((.235, .48) if compact else (.555, .285))
    axes = [fig.add_axes([left, bottom, .36, height]) for left in (.125, .595)]
    size = 10.5 if compact else 11.5
    values = []
    for index, (ax, (metric, title)) in enumerate(zip(axes, METRICS)):
        ax.set(xlim=(0, 25), ylim=(-.43, 1.43), xticks=[0, 5, 10, 15, 20, 25])
        ax.set_title(title, loc='center', pad=25 if compact else 29,
                     fontsize=11.6 if compact else 13, fontweight='bold', color=INK)
        for side in ('left', 'right', 'top'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#C7D1D7')
        ax.spines['bottom'].set_linewidth(.7)
        ax.tick_params(axis='x', labelsize=10 if compact else 10.8,
                       colors=MUTED, length=0, pad=6)
        ax.tick_params(axis='y', length=0, pad=12, labelsize=size, colors=INK)
        ax.set_yticks([1, 0], ['Direct', 'JEPA'] if index == 0 else [])
        ax.set_axisbelow(True)
        ax.grid(False)  # Direct values and sparse ticks make extra grid lines unnecessary.
        unchanged = float(means['unchanged'][metric])
        ax._result_lines = [ax.axvline(unchanged, color='#89969F', lw=1, ls=(0, (3, 3)), zorder=1)]
        ax._result_labels = []
        ax.text(unchanged, 1.51, f'Unchanged: {unchanged:.1f}°', ha='center', va='bottom',
                fontsize=9.5 if compact else 10.5, color=MUTED, clip_on=False)
        for family_index, (family, base, change) in enumerate(FAMILIES):
            y = 1 - family_index
            base_value, change_value = float(means[base][metric]), float(means[change][metric])
            # Vertical offsets encode the objective; x always remains the measured mean.
            ys = [y + .18, y - .18]
            ax._result_lines.extend(ax.plot([base_value, change_value], ys, color='#AFBAC2', lw=1.1, zorder=2))
            for key, value, point_y, color, marker, fill in (
                (base, base_value, ys[0], COORD, 'o', COORD),
                (change, change_value, ys[1], CHANGE, 'D', 'white'),
            ):
                ax.plot(value, point_y, marker=marker, markersize=5.6,
                        markeredgewidth=1.25, markeredgecolor=color, markerfacecolor=fill,
                        linestyle='none', zorder=4)
                # Label away from the dashed baseline, without masking it or the connector.
                on_left = value < unchanged
                label = ax.annotate(f'{value:.1f}', (value, point_y),
                                    xytext=(-7 if on_left else 7, -2 if on_left and key == change else 0),
                                    textcoords='offset points', ha='right' if on_left else 'left',
                                    va='center', color=color, fontsize=size, zorder=5)
                ax._result_labels.append(label)
                values.append({'family': family, 'method': key, 'metric': metric,
                               'value': value, 'display': f'{value:.1f}', 'y': point_y})
    fig.text(.54, .063 if compact else .471, 'Mean error (°) · lower is better',
             ha='center', fontsize=10.3 if compact else 11.1, color=MUTED)
    handles = [Line2D([], [], color=COORD, marker='o', markerfacecolor=COORD,
                      markersize=5.6, linestyle='none', label='Coordinate loss'),
               Line2D([], [], color=CHANGE, marker='D', markerfacecolor='white',
                      markeredgewidth=1.25, markersize=5.6, linestyle='none',
                      label='Coordinate + change loss')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, 1.005),
               ncol=2, frameon=False, fontsize=10.5 if compact else 11.5, labelcolor=INK,
               columnspacing=2, handletextpad=.5, handlelength=1, borderpad=0)
    return values


def contrast_panel(fig, contrasts):
    fig.text(.125, .389, 'Declared comparison: JEPA − direct, both with change loss',
             fontsize=12, fontweight='bold', color=INK)
    ax = fig.add_axes([.275, .13, .61, .20])
    ax.set(xlim=(-3, 5), ylim=(-.45, 1.45), xticks=[-2, 0, 2, 4])
    ax.set_yticks([1, 0], ['Response', 'Trajectory'])
    for side in ('left', 'right', 'top'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#C7D1D7')
    ax.spines['bottom'].set_linewidth(.7)
    ax.tick_params(length=0, labelsize=10.8, labelcolor=MUTED, pad=7)
    ax.axvline(0, color='#89969F', lw=1, ls=(0, (3, 3)))
    records = []
    for y, (metric, _) in zip([1, 0], METRICS):
        row = contrasts[metric]
        # Source reports comparator-minus-candidate improvement. Reverse it so
        # negative values on the figure consistently mean lower JEPA error.
        value = -float(row['improvement'])
        lo, hi = -float(row['crossed_high']), -float(row['crossed_low'])
        ax.errorbar(value, y, xerr=[[value - lo], [hi - value]], fmt='o', color=INK,
                    markersize=5.3, lw=1.4, capsize=3, capthick=1.1, zorder=3)
        records.append({'metric': metric, 'jepa_minus_direct': value,
                        'crossed_95_low': lo, 'crossed_95_high': hi})
    fig.text(.58, .028, 'Difference in error (°) · 95% intervals · 0 = equal error',
             ha='center', fontsize=10.5, color=MUTED)
    return records


def save(fig, filename, title, description):
    path = IMAGES / filename
    fig.savefig(path, format='svg', facecolor='white', metadata={'Date': None})
    # Matplotlib's RDF metadata does not supply an accessible SVG title/desc.
    text = path.read_text()
    start = text.index('<svg ')
    end = text.index('>', start)
    from html import escape
    text = (text[:end] + ' role="img" aria-labelledby="title desc"' + text[end:end + 1]
            + '\n<title id="title">' + escape(title) + '</title>'
            + '\n<desc id="desc">' + escape(description) + '</desc>' + text[end + 1:])
    ET.fromstring(text)
    path.write_text(text)
    png = IMAGES / 'previews' / (path.stem + '.png')
    png.parent.mkdir(exist_ok=True)
    fig.savefig(png, dpi=220, facecolor='white')
    return path


def build_results():
    means, contrasts = read_data()
    receipts, paths = {}, []
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'svg.fonttype': 'none', 'svg.hashsalt': 'gait-core-matplotlib-v1',
                         'pdf.fonttype': 42, 'axes.unicode_minus': True}):
        for compact in (False, True):
            name = 'research-core-results' + ('-compact' if compact else '')
            fig = plt.figure(figsize=(7.5, 2.5) if compact else (8.6, 5.0), dpi=150)
            values = means_panels(fig, means, compact)
            effect = [] if compact else contrast_panel(fig, contrasts)
            layout = check_layout(fig)
            description = (
                'Completed-core development results: fourteen people, three training seeds. '
                'Both mean-error panels use a zero-to-25-degree scale. Blue circles indicate '
                'coordinate loss and open orange diamonds indicate coordinate plus change loss. '
                'Direct and JEPA each have two vertically offset points; thin gray connectors '
                'link their two fitted means, not individual observations or confidence intervals. '
                'Dashed vertical lines show unchanged-pose errors. All displayed means are '
                'rounded to one decimal. Lower error is better; means alone do not establish '
                'a JEPA improvement. The proposed follow-up has no results.'
            )
            if not compact:
                description += (' The separate lower panel reports the declared JEPA-minus-direct '
                                'contrasts, both with change loss, with descriptive 95% crossed '
                                'person/seed intervals. The response interval crosses zero; '
                                'the trajectory interval indicates higher JEPA error.')
            path = save(fig, name + '.svg', 'Gait fidelity: completed-core errors' +
                        ('' if compact else ' and declared paired comparison'), description)
            receipts[name] = {'size_inches': list(fig.get_size_inches()), 'mean_axis_limits': [0, 25],
                              'layout': layout, 'points': values, 'paired_contrasts': effect,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            paths.append(path)
            plt.close(fig)
    record = {'renderer': 'Matplotlib', 'version': matplotlib.__version__,
              'source_sha256': {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
                                for name in ['method-means.csv', 'paired-contrasts.csv']},
              'figures': receipts}
    (ROOT / 'records/results-figure-build.json').write_text(json.dumps(record, indent=2) + '\n')
    return paths


if __name__ == '__main__':
    for result in build_results():
        print(result)
