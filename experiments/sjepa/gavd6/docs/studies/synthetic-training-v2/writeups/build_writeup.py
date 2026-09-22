"""Build the two-page PDF and its reusable vector illustrations from README.md."""
from pathlib import Path
import html
import re

import pandas as pd
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, Circle
from reportlab.graphics import renderSVG
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether
from pypdf import PdfReader

OUT = Path(__file__).resolve().parent
IMAGES = OUT / 'images'
FONT = Path('/System/Library/Fonts/Supplemental')
for name, filename in [('Body', 'Times New Roman.ttf'), ('BodyBold', 'Times New Roman Bold.ttf'),
                       ('BodyItalic', 'Times New Roman Italic.ttf'), ('Arial', 'Arial.ttf'), ('ArialBold', 'Arial Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT / filename)))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='BodyItalic', boldItalic='BodyBold')
INK = colors.HexColor('#1d2c35')
MUTED = colors.HexColor('#52616a')
TEAL = colors.HexColor('#147d87')
BLUE = colors.HexColor('#32629c')
ORANGE = colors.HexColor('#bc6b2a')
LIGHT = colors.HexColor('#e7edf0')
WIDTH = 518.4

def light_canvas(height, width=WIDTH):
    """Keep exported figures readable when embedded in dark-themed viewers."""
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, fillColor=colors.white, strokeColor=None))
    return drawing

def label(d, x, y, value, size=8, font='Arial', color=INK, anchor='start'):
    d.add(String(x, y, value, fontName=font, fontSize=size, fillColor=color, textAnchor=anchor))

def arrow(d, x1, y1, x2, y2, color=MUTED, dashed=False):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=.9, strokeDashArray=[3,2] if dashed else None))
    if x1 == x2:
        sign = 1 if y2 > y1 else -1
        points = [x2, y2, x2-2.5, y2-5*sign, x2+2.5, y2-5*sign]
    else:
        sign = 1 if x2 > x1 else -1
        points = [x2, y2, x2-5*sign, y2-2.5, x2-5*sign, y2+2.5]
    d.add(Polygon(points, fillColor=color, strokeColor=None))

def box(d, x, y, w, h, lines, fill='#f4f7f8', accent=False):
    d.add(Rect(x, y, w, h, rx=3, ry=3, strokeColor=TEAL if accent else colors.HexColor('#aebdc4'),
               strokeWidth=.8, fillColor=colors.HexColor(fill)))
    for i, text in enumerate(lines):
        label(d, x+w/2, y+h/2+(len(lines)-1)*5-i*10-2, text, 8.1,
              'ArialBold' if i == 0 else 'Arial', anchor='middle')

def design():
    d = light_canvas(107)
    box(d, 0, 25, 93, 49, ['AMASS motion', '24 training people', '8 development people'])
    box(d, 114, 64, 111, 34, ['Projected joints', 'Synthetic references'])
    box(d, 114, 5, 111, 40, ['Rendered images', 'Frozen pose estimators', 'Imperfect 2D tracks'])
    arrow(d, 93, 60, 114, 81)
    arrow(d, 93, 36, 114, 25)
    box(d, 264, 5, 119, 40, ['Restoration methods', 'Paired JEPA', 'and all controls'], '#eaf4f3', True)
    arrow(d, 225, 25, 264, 25, TEAL)
    d.add(Line(225, 80, 324, 80, strokeColor=MUTED, strokeWidth=.9, strokeDashArray=[3,2]))
    arrow(d, 324, 80, 324, 45, dashed=True)
    label(d, 238, 87, 'Training only', 7.4)
    box(d, 413, 5, 105, 40, ['8-person evaluation', 'Position error', 'Movement fidelity'])
    arrow(d, 383, 25, 413, 25, TEAL)
    d.add(Line(324, 80, 465, 80, strokeColor=MUTED, strokeWidth=.9, strokeDashArray=[3,2]))
    arrow(d, 465, 80, 465, 45, dashed=True)
    label(d, 398, 87, 'Scoring reference', 7.4)
    return d

def comparison(detailed=False):
    """Separate extractors into labeled columns; retain seed ranges in a companion."""
    d = light_canvas(494 if detailed else 218, WIDTH * (2 if detailed else 1))
    data = pd.read_csv(OUT / 'analysis/per-seed-primary.csv')
    methods = [('unchanged','Unchanged / filter0'), ('filter1','3-frame filter'), ('filter2','5-frame filter'),
        ('joint_offset','Joint offset'), ('joint_affine','Joint affine'), ('initialized','Untrained encoder'),
        ('coordinate','Coordinate pretraining'), ('ordinary_jepa','Ordinary JEPA'), ('paired_jepa','Paired JEPA'),
        ('shuffled_jepa','Shuffled JEPA'), ('smoothnet','SmoothNet-style'), ('static','Static network'), ('direct','Direct supervision')]
    extractors = [('hrnet_w32','HRNet'), ('rtmpose_m','RTMPose'), ('vitpose_base','ViTPose')]
    factor = 2 if detailed else 1
    columns = [160, 217, 274, 346, 403, 460]
    cell_width = 52 * factor
    row_top, row_step = (384, 25) if detailed else (164, 11.3)
    cell_height = 23 if detailed else 10.4
    cell_bottom_offset = 13 if detailed else 5.2
    group_y, extractor_y = (430, 408) if detailed else (189, 176)
    heading = 'Error reduction relative to unchanged tracks (%)'
    label(d, 6, 474 if detailed else 206, heading, 15 if detailed else 9.2, 'ArialBold')
    label(d, d.width-6, 474 if detailed else 206, 'Higher is better', 11 if detailed else 8.2,
          color=MUTED, anchor='end')
    label(d, 6, extractor_y, 'Method', 11 if detailed else 8.5, 'ArialBold', MUTED)
    if detailed:
        label(d, 6, 454, 'Mean of seeds 17, 29 and 43; brackets show their minimum–maximum, not a confidence interval.', 10, color=MUTED)
    for panel, (metric, title) in enumerate([
            ('visible_nle', 'Position accuracy'),
            ('displacement_nle', 'Movement over 0.20 s')]):
        group_left = columns[panel*3] * factor
        group_right = (columns[panel*3+2]+52) * factor
        label(d, (group_left+group_right)/2, group_y, title, 13 if detailed else 9.2,
              'ArialBold', anchor='middle')
        d.add(Line(group_left, extractor_y-5, group_right, extractor_y-5,
                   strokeColor=colors.HexColor('#b8c8cf'), strokeWidth=.6))
        for j, (extractor, name) in enumerate(extractors):
            x = columns[panel*3+j] * factor
            center = x + cell_width/2
            label(d, center, extractor_y, name, 11 if detailed else 8.4, 'ArialBold', anchor='middle')
            selected = data[data.metric.eq(metric) & data.extractor.eq(extractor)]
            baseline = selected[selected.method.eq('unchanged')].set_index('seed').value.sort_index()
            assert baseline.index.tolist() == [17, 29, 43]
            values = {}
            for method, _ in methods:
                measurements = selected[selected.method.eq(method)].set_index('seed').value.sort_index()
                assert measurements.index.equals(baseline.index) and measurements.notna().all()
                reduction = 100*(1-measurements/baseline)
                values[method] = (reduction.mean(), reduction.min(), reduction.max())
            best = max(values, key=lambda method: values[method][0])
            for i, (method, _) in enumerate(methods):
                y = row_top-i*row_step
                mean, low, high = values[method]
                shown = 0. if abs(mean) < (.005 if detailed else .05) else mean
                fill = improvement_color(shown)
                winner = method == best
                d.add(Rect(x, y-cell_bottom_offset, cell_width, cell_height,
                           fillColor=fill, strokeColor=TEAL if winner else None,
                           strokeWidth=.9 if winner else 0))
                # Avoid reporting floating-point noise as a negative zero.
                number = f'{shown:.2f}' if detailed else f'{shown:.1f}'
                number = number.replace('-', '−')
                label(d, center, y+1 if detailed else y-3, number, 12.2 if detailed else 8.8,
                      'ArialBold' if winner else 'Arial', anchor='middle')
                if detailed:
                    low = 0. if abs(low) < .005 else low
                    high = 0. if abs(high) < .005 else high
                    span = f'[{low:.2f}, {high:.2f}]'.replace('-', '−')
                    label(d, center, y-9, span, 8.4, color=MUTED, anchor='middle')
    # Shared labels eliminate the need to track an extractor-color legend.
    for i, (method, name) in enumerate(methods):
        y = row_top-i*row_step
        label(d, 6, y-3, name, 12 if detailed else 8.6,
              'ArialBold' if method in ['paired_jepa','direct'] else 'Arial')
    # Separate filtering, spatial calibration, frozen-encoder and end-to-end arms.
    for index in (2, 4, 9):
        y = row_top-(index+.5)*row_step
        d.add(Line(6, y, 150*factor, y, strokeColor=LIGHT, strokeWidth=.5))
    legend_y = 45 if detailed else 7
    legend_font = 9.5 if detailed else 7.4
    label(d, 6, legend_y, 'Shading: 0', legend_font, color=MUTED)
    swatch_left = 72 if detailed else 45
    for step in range(20):
        d.add(Rect(swatch_left+step*2*factor, legend_y-1, 2*factor, 7*factor,
                   fillColor=improvement_color(50*step/19), strokeColor=None))
    label(d, swatch_left+43*factor, legend_y, '50% improvement', legend_font, color=MUTED)
    negative_x = 420 if detailed else 209
    d.add(Rect(negative_x, legend_y-1, 9*factor, 7*factor,
               fillColor=improvement_color(-1), strokeColor=None))
    label(d, negative_x+12*factor, legend_y, 'Negative = worse', legend_font, color=MUTED)
    best_x = 750 if detailed else 377
    d.add(Rect(best_x, legend_y-1, 9*factor, 7*factor, fillColor=None, strokeColor=TEAL, strokeWidth=.9))
    label(d, best_x+12*factor, legend_y, 'Best mean in column', legend_font, color=MUTED)
    if detailed:
        label(d, 6, 20, 'Eight development people, reused across seeds. Calibration is shared; unchanged and filter0 coincide.', 10, color=MUTED)
    return d

def improvement_color(value):
    """One positive scale for both endpoints; a distinct light fill for worse values."""
    if value < -.005:
        return colors.HexColor('#f8e2d5')
    fraction = min(1., max(0., value/50))
    target = colors.HexColor('#a8d0df')
    return colors.Color(*(1+(component-1)*fraction for component in (target.red, target.green, target.blue)))

def inline(text):
    text = html.escape(text)
    text = re.sub(r'\[([^]]+)\]\(([^)]+)\)', lambda m: '<link href="'+m.group(2)+'" color="#147d87">'+m.group(1)+'</link>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    return text

def main():
    IMAGES.mkdir(exist_ok=True)
    figures = {'01-study-design.svg': design(), '02-method-comparison.svg': comparison(),
               '03-seed-ranges.svg': comparison(detailed=True)}
    descriptions = {
        '01-study-design.svg': ('Paired synthetic training and evaluation', 'Estimated pose tracks feed restoration methods; projected reference joints guide training and evaluation.'),
        '02-method-comparison.svg': ('Position and movement accuracy by method and extractor', 'A labeled matrix reports three-seed mean error reductions. Direct supervision has the best position means; the five-frame filter has the best HRNet displacement mean.'),
        '03-seed-ranges.svg': ('Full method comparison with training-seed ranges', 'Each cell gives the mean percentage error reduction and the minimum–maximum across seeds 17, 29 and 43. Ranges are not confidence intervals.'),
    }
    for name, drawing in figures.items():
        path = IMAGES / name
        renderSVG.drawToFile(drawing, str(path))
        svg = path.read_text()
        # SVG viewers do not know ReportLab's internal ArialBold font alias.
        svg = svg.replace('font-family: ArialBold;', 'font-family: Arial; font-weight: 700;')
        svg = re.sub(r'viewBox="[^"]+"', f'viewBox="0 0 {drawing.width} {drawing.height}"', svg, count=1)
        title, description = descriptions[name]
        svg = svg.replace('<title>...</title>', '<title>'+html.escape(title)+'</title>')
        svg = svg.replace('<desc>...</desc>', '<desc>'+html.escape(description)+'</desc>')
        path.write_text(svg)
    styles = {
        'body': ParagraphStyle('body', fontName='Body', fontSize=10.5, leading=12.45, textColor=INK, spaceAfter=5),
        'title': ParagraphStyle('title', fontName='ArialBold', fontSize=17, leading=19.3, textColor=INK, spaceAfter=6),
        'heading': ParagraphStyle('heading', fontName='ArialBold', fontSize=10.1, leading=12, textColor=INK, spaceBefore=6, spaceAfter=4, keepWithNext=True),
        'caption': ParagraphStyle('caption', fontName='Body', fontSize=8.5, leading=10.2, textColor=MUTED, spaceAfter=5),
        'meta': ParagraphStyle('meta', fontName='Arial', fontSize=8, leading=10, textColor=MUTED, spaceAfter=8),
        'source': ParagraphStyle('source', fontName='Body', fontSize=7.6, leading=9.1, textColor=MUTED, spaceBefore=1),
    }
    blocks = (OUT / 'README.md').read_text().split('\n\n')
    story = []
    i = 0
    while i < len(blocks):
        b = blocks[i].strip()
        if not b:
            i += 1
            continue
        if b.startswith('<!-- PAGEBREAK'):
            story.append(PageBreak())
        elif b.startswith('# '):
            story.append(Paragraph(inline(b[2:]), styles['title']))
        elif b.startswith('## '):
            story.append(Paragraph(inline(b[3:]), styles['heading']))
        elif b.startswith('!['):
            name = re.search(r'images/([^)]+)', b).group(1)
            caption = blocks[i+1].strip()
            story.append(KeepTogether([Spacer(1,2), figures[name], Paragraph(inline(caption), styles['caption'])]))
            i += 1
        elif b.startswith('*Research results'):
            story.append(Paragraph(inline(b.strip('*')), styles['meta']))
        elif b.startswith('*Sources'):
            story.append(Paragraph(inline(b.strip('*')), styles['source']))
        else:
            story.append(Paragraph(inline(b), styles['body']))
        i += 1
    def footer(c, doc):
        c.setStrokeColor(LIGHT)
        c.line(46.8, 33, 565.2, 33)
        c.setFillColor(MUTED)
        c.setFont('Arial', 7.4)
        c.drawString(46.8, 22, 'Synthetic training v2  |  20 September 2026')
        c.drawRightString(565.2, 22, str(doc.page))
    path = OUT / 'synthetic-training-v2-results.pdf'
    doc = SimpleDocTemplate(str(path), pagesize=(612,792), rightMargin=46.8, leftMargin=46.8,
                            topMargin=36, bottomMargin=43, title='Synthetic pose restoration: better positions, unresolved motion fidelity',
                            author='Synthetic training v2 research notes')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    pdf = PdfReader(path)
    assert len(pdf.pages) == 2, f'Expected two pages; got {len(pdf.pages)}'
    assert all(not page.images for page in pdf.pages), 'Figures must remain vector artwork'
    print(f'{path}: {len(pdf.pages)} pages; vector artwork; {sum(len(p.extract_text().split()) for p in pdf.pages)} words')

if __name__ == '__main__':
    main()
