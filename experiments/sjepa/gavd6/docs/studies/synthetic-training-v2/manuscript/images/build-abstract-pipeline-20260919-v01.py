"""Create an append-only conceptual pipeline figure and review artifacts.

Run with the repository .venv interpreter. Requires CairoSVG and Pillow.
On this Mac, prefix the command with DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib.
Layout measurement uses macOS Arial fonts; SVG itself has a sans-serif fallback.
Use --output-dir to reproduce in a fresh directory; existing files are refused.
No experiment data are read and no scientific results are generated.
"""
from argparse import ArgumentParser
from html import escape
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import cairosvg
from PIL import ImageFont

WIDTH, HEIGHT = 1400, 1250
INK, MUTED = "#182c42", "#53677a"
BLUE, TEAL, ORANGE = "#eaf1fb", "#e8f5f0", "#fff1e3"
STEM = "abstract-training-pipeline-20260919-v01"


def build():
    labels, boxes, edges = [], [], []
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title description">',
        '<title id="title">Paired synthetic supervision for 2D pose restoration</title>',
        '<desc id="description">Conceptual training pipeline. Rendered observations and projected joint references share motion and camera. A student predicts the slowly updated teacher features during pretraining. A separately trained coordinate readout uses the frozen encoder. Evaluation separates coordinates, movement and timing support. Calibration and real-video validation are pending.</desc>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#53677a"/></marker></defs>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
    ]

    def text(x, y, value, size=19, bold=False, anchor="start", color=INK, container=None):
        labels.append(dict(x=x, y=y, text=value, size=size, bold=bold, anchor=anchor, container=container))
        parts.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}" fill="{color}">{escape(value)}</text>')

    def box(x, y, w, h, title, lines=(), fill=BLUE):
        key = len(boxes)
        boxes.append([x, y, x+w, y+h])
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#d1dde6"/>')
        text(x+w/2, y+32, title, 21, True, "middle", container=key)
        for i, line in enumerate(lines):
            text(x+w/2, y+59+i*23, line, 17, anchor="middle", color=MUTED, container=key)

    def line(points, dashed=False):
        edges.append(points)
        pts = ' '.join(f'{x},{y}' for x, y in points)
        dash = ' stroke-dasharray="7 5"' if dashed else ''
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{MUTED}" stroke-width="2.5" stroke-linejoin="round" marker-end="url(#arrow)"{dash}/>')

    def stage(y, number, title):
        text(44, y, f'{number}   {title}', 19, True, color="#285aa4")

    text(44, 51, 'Paired synthetic supervision for 2D pose restoration', 29, True)
    text(44, 83, 'Conceptual design · projected references guide training, never inference', 19, color=MUTED)
    stage(129, '1', 'Prepare matched observations and reference targets')
    box(44, 182, 250, 110, 'AMASS motion', ['Body-model animation', 'Person-separated splits'])
    box(357, 157, 270, 108, 'Matched renders', ['Fixed motion and camera', 'Clean / blur / obstruction'])
    box(692, 157, 268, 108, 'Frozen pose estimators', ['HRNet · RTMPose · ViTPose', 'Extract 12 body joints'])
    box(1023, 157, 333, 108, 'Observed tracks', ['Coordinates, scores, missingness', '64 samples at 25 Hz'])
    line([(294, 211), (357, 211)])
    line([(627, 211), (692, 211)])
    line([(960, 211), (1023, 211)])
    box(357, 306, 603, 88, 'Same-view projected joints', ['Synthetic anatomical proxies; training and scoring only'], TEAL)
    line([(294, 265), (322, 265), (322, 350), (357, 350)])
    text(1023, 329, 'Fit on HRNet + RTMPose;', 18, color=MUTED)
    text(1023, 355, 'evaluate ViTPose without fitting', 18, color=MUTED)
    text(1023, 381, 'its development labels.', 18, color=MUTED)

    stage(445, '2', 'Pretrain by predicting reference features')
    box(44, 479, 240, 92, 'Masked observations', ['Estimated tracks only'])
    box(344, 479, 255, 92, 'Online encoder', ['Learn temporal features'])
    box(659, 479, 246, 92, 'Predictor', ['Predict masked features'])
    box(997, 553, 359, 112, 'Feature-matching loss', ['Gradients update the online branch', 'Reference branch is detached'], ORANGE)
    line([(284, 525), (344, 525)])
    line([(599, 525), (659, 525)])
    line([(905, 525), (950, 525), (950, 582), (997, 582)])
    box(44, 653, 240, 92, 'Projected references', ['Same motion and times'], TEAL)
    box(344, 653, 255, 92, 'Teacher encoder', ['Slow average of online weights'], TEAL)
    box(659, 653, 246, 92, 'Target features', ['Training supervision'], TEAL)
    line([(284, 699), (344, 699)])
    line([(599, 699), (659, 699)])
    line([(905, 699), (950, 699), (950, 632), (997, 632)])
    line([(471, 571), (471, 653)], dashed=True)
    text(489, 617, 'Weight average', 16, color=MUTED)

    stage(799, '3', 'Fit a coordinate readout; restore tracks using observed inputs')
    box(44, 831, 240, 91, 'Observed tracks', ['Full window at inference'])
    box(344, 831, 255, 91, 'Frozen encoder', ['Retain pretrained features'])
    box(659, 831, 246, 91, 'Coordinate readout', ['Fit on training pairs only'])
    box(997, 831, 359, 91, 'Restored 2D trajectories', ['Teacher and targets absent at inference'], TEAL)
    for a, b in [(284, 344), (599, 659), (905, 997)]:
        line([(a, 877), (b, 877)])
    text(44, 957, 'Controls: coordinate pretraining, direct restoration, initialized readout, shuffled pairing.', 18, color=MUTED)

    stage(1008, '4', 'Evaluate different people and the held extractor family')
    box(44, 1038, 410, 94, 'Coordinate agreement', ['Error relative to proxy references'], '#f0f3f6')
    box(488, 1038, 430, 94, 'Movement fidelity', ['Displacement, amplitude, L/R separation'], '#f0f3f6')
    box(953, 1038, 403, 94, 'Timing and support', ['Peak error plus eligible-record counts'], '#f0f3f6')
    parts.append('<rect x="44" y="1164" width="1312" height="59" rx="10" fill="#fffaf4" stroke="#c2935d" stroke-dasharray="7 5"/>')
    text(65, 1201, 'Pending: calibration controls, reference timing self-check and independent real-video validation.', 19, color="#85541e")
    svg = '\n'.join(parts + ['</svg>'])
    ET.fromstring(svg)
    return svg, labels, boxes, edges


def check_layout(labels, boxes, edges):
    findings, measured = [], []
    font_dir = Path('/System/Library/Fonts/Supplemental')
    for label in labels:
        name = 'Arial Bold.ttf' if label['bold'] else 'Arial.ttf'
        font = ImageFont.truetype(str(font_dir / name), label['size'])
        left, top, right, bottom = font.getbbox(label['text'], anchor='ls')
        shift = font.getlength(label['text'])/2 if label['anchor'] == 'middle' else 0
        bbox = (label['x']+left-shift, label['y']+top, label['x']+right-shift, label['y']+bottom)
        outer = boxes[label['container']] if label['container'] is not None else [0, 0, WIDTH, HEIGHT]
        if not (outer[0] <= bbox[0] <= bbox[2] <= outer[2] and outer[1] <= bbox[1] <= bbox[3] <= outer[3]):
            findings.append(['outside container', label['text']])
        for previous, value in measured:
            if max(previous[0], bbox[0]) < min(previous[2], bbox[2]) and max(previous[1], bbox[1]) < min(previous[3], bbox[3]):
                findings.append(['text overlap', value, label['text']])
        for edge in edges:
            for (x1, y1), (x2, y2) in zip(edge, edge[1:]):
                if x1 == x2 and bbox[0]-3 < x1 < bbox[2]+3 and max(min(y1,y2),bbox[1]-3) < min(max(y1,y2),bbox[3]+3):
                    findings.append(['vertical arrow intersects text', label['text']])
                if y1 == y2 and bbox[1]-3 < y1 < bbox[3]+3 and max(min(x1,x2),bbox[0]-3) < min(max(x1,x2),bbox[2]+3):
                    findings.append(['horizontal arrow intersects text', label['text']])
        measured.append((bbox, label['text']))
    if findings:
        raise ValueError(findings)
    return dict(findings=findings, checked_text_labels=len(labels), checked_edges=len(edges),
                note='Geometry checks supplement visual review; schematic, not empirical results.')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    paths = {ext: args.output_dir / f'{STEM}.{ext}' for ext in ('svg', 'png', 'pdf', 'layout.json')}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Use a fresh --output-dir; existing review artifacts are preserved.')
    svg, labels, boxes, edges = build()
    layout = check_layout(labels, boxes, edges)
    payloads = dict(svg=svg.encode(), png=cairosvg.svg2png(bytestring=svg.encode()),
                    pdf=cairosvg.svg2pdf(bytestring=svg.encode()),
                    **{'layout.json': (json.dumps(layout, indent=2)+'\n').encode()})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, path in paths.items():
        with path.open('xb') as stream:
            stream.write(payloads[key])
    print(json.dumps(layout, indent=2))


if __name__ == '__main__':
    main()
