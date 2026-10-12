"""Build the focused overview as responsive HTML and a two-page vector PDF.

Both outputs use proposal-brief.md. ReportLab lays out the text and photographs;
CairoSVG and pypdf place the original vector diagrams without rasterizing them.
"""
from pathlib import Path
from html import escape
import io
import json
import re
import tempfile

from bs4 import BeautifulSoup
import cairosvg
import mistune
from PIL import Image as PILImage
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, Image, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ['Introduction', 'Methodology', 'Experiments', 'Results', 'Discussion']
INK = '#183247'
MUTED = '#526675'
WIDTH = 540


def source():
    soup = BeautifulSoup(mistune.html((ROOT / 'proposal-brief.md').read_text()), 'html.parser')
    sections = {}
    header = []
    target = header
    for tag in soup.find_all(recursive=False):
        if tag.name == 'h2':
            target = sections.setdefault(tag.get_text(), [])
        else:
            target.append(tag)
    assert list(sections) == SECTIONS, list(sections)
    return header, sections


def photo_band(items, image_side='right'):
    image_at = next(i for i, p in enumerate(items) if p.find('img'))
    image, caption = items[image_at:image_at + 2]
    text = items[:image_at] + items[image_at + 2:]
    return ('<div class="context-row ' + image_side + '"><div class="context-copy">'
            + ''.join(map(str, text)) + '</div><figure>' + str(image)
            + '<figcaption>' + caption.decode_contents() + '</figcaption></figure></div>')


def build_html(header, sections):
    out = []
    for title, items in sections.items():
        if title == 'Introduction':
            content = photo_band(items)
        elif title == 'Discussion':
            # The argument leads; the photograph sits beside the transfer limits.
            content = str(items[0]) + photo_band(items[1:-1], 'left') + str(items[-1])
        else:
            content = ''.join(map(str, items))
            content = re.sub(r'<p>(<img[^>]+>)</p>\s*<p><em>(.*?)</em></p>',
                             r'<figure class="research-figure">\1<figcaption>\2</figcaption></figure>',
                             content, flags=re.S)
        out.append(f'<section id="{title.lower()}"><h2>{title}</h2>{content}</section>')
    css = '''
:root{color-scheme:light;--ink:#183247;--muted:#526675;--line:#d8e3e8}
*{box-sizing:border-box}body{margin:0;background:#edf3f6;color:var(--ink);font:18px/1.6 Georgia,serif}
a{color:#265f9e;text-underline-offset:3px}a:focus-visible{outline:3px solid #ae7120;outline-offset:4px}
.toolbar{max-width:1000px;margin:24px auto;padding:0 32px;font:14px/1.6 Arial,sans-serif;display:flex;gap:24px;flex-wrap:wrap}
main{max-width:1000px;margin:0 auto 40px;background:white;padding:42px 52px;border:1px solid var(--line);box-shadow:0 5px 24px #18324708}
h1,h2{font-family:Arial,sans-serif;line-height:1.25}h1{font-size:40px;margin:0 0 5px;letter-spacing:-.7px}
header p{margin:6px 0}header p:first-of-type{font-size:22px;color:var(--muted)}header p:last-of-type{font:13px/1.5 Arial,sans-serif;color:var(--muted);margin:14px 0 24px}
h2{font-size:25px;margin:28px 0 12px;border-top:1px solid var(--line);padding-top:18px}p{margin:12px 0}
figure{margin:14px 0}figure img{display:block;max-width:100%;height:auto}figcaption{font:13px/1.5 Arial,sans-serif;color:var(--muted);margin-top:8px}figcaption em{font-style:normal}
.context-row{display:grid;grid-template-columns:minmax(0,1fr) 180px;gap:26px;align-items:start}.context-row figure{margin:4px 0}.context-copy p:first-child{margin-top:0}
.context-row.left{grid-template-columns:240px minmax(0,1fr)}.context-row.left figure{grid-column:1;grid-row:1}.context-row.left .context-copy{grid-column:2;grid-row:1}
.research-figure{border:1px solid var(--line);border-radius:6px;padding:10px;background:#fff}.research-figure img{width:100%}
#discussion>p:last-child{font:12px/1.6 Arial,sans-serif;border-top:1px solid var(--line);padding-top:12px;color:var(--muted)}
@media(max-width:650px){body{font-size:17px}main{padding:26px 20px;border:0}.toolbar{padding:0 20px;gap:16px}h1{font-size:34px}.context-row,.context-row.left{display:flex;flex-direction:column;gap:10px}.context-row figure{max-width:210px;align-self:center}.context-row.left figure{max-width:310px}h2{font-size:24px}.research-figure{padding:4px}}
@media print{body{background:white;font-size:10.5pt;line-height:1.35}.toolbar{display:none}main{max-width:none;margin:0;padding:0;border:0;box-shadow:none}h1{font-size:26pt}h2{font-size:13pt;margin:12px 0 7px;padding-top:8px}header p:first-of-type{font-size:13pt}header p:last-of-type{font-size:8pt}p{margin:7px 0}figcaption{font-size:8pt}.research-figure{padding:0;border:0;break-inside:avoid}.context-row{grid-template-columns:1fr 110px;gap:14px}.context-row.left{grid-template-columns:145px 1fr}#results{break-before:page}#discussion>p:last-child{font-size:8pt}@page{size:letter;margin:.5in}}
'''
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Gait Fidelity — two-page research overview</title><style>' + css
            + '</style></head><body><nav class="toolbar" aria-label="Document versions">'
            '<a href="proposal.html">Full proposal</a><a href="proposal-brief.pdf">Two-page PDF</a>'
            '<a href="proposal-brief.md">Overview source</a></nav><main><header>'
            + ''.join(map(str, header)) + '</header>' + ''.join(out) + '</main></body></html>\n')
    (ROOT / 'proposal-brief.html').write_text(page)


def register_fonts():
    mac = Path('/System/Library/Fonts/Supplemental')
    linux = Path('/usr/share/fonts/truetype/dejavu')
    families = {
        'Body': [('Georgia.ttf', 'DejaVuSerif.ttf'), ('Georgia Bold.ttf', 'DejaVuSerif-Bold.ttf'),
                 ('Georgia Italic.ttf', 'DejaVuSerif-Italic.ttf'), ('Georgia Bold Italic.ttf', 'DejaVuSerif-BoldItalic.ttf')],
        'Label': [('Arial.ttf', 'DejaVuSans.ttf'), ('Arial Bold.ttf', 'DejaVuSans-Bold.ttf'),
                  ('Arial Italic.ttf', 'DejaVuSans-Oblique.ttf'), ('Arial Bold Italic.ttf', 'DejaVuSans-BoldOblique.ttf')],
    }
    for family, options in families.items():
        names = [family, family + '-Bold', family + '-Italic', family + '-BoldItalic']
        for name, (mf, lf) in zip(names, options):
            font = mac / mf if (mac / mf).exists() else linux / lf
            pdfmetrics.registerFont(TTFont(name, str(font)))
        pdfmetrics.registerFontFamily(family, normal=names[0], bold=names[1], italic=names[2], boldItalic=names[3])


def paragraph(tag, style):
    html = tag.decode_contents() if hasattr(tag, 'decode_contents') else tag
    html = html.replace('<strong>', '<b>').replace('</strong>', '</b>')
    html = html.replace('<em>', '<i>').replace('</em>', '</i>')
    html = re.sub(r'<a href="([^"]+)">', r'<a href="\1" color="#265f9e">', html)
    return Paragraph(html, style)


class VectorFigure(Flowable):
    """Reserve layout space; merge a vector page at this location after typesetting."""
    placements = []

    def __init__(self, path, width):
        super().__init__()
        self.path = path
        view = re.search(r'viewBox="([^"]+)"', path.read_text()).group(1)
        _, _, w, h = map(float, view.split())
        self.width, self.height = width, width * h / w

    def draw(self):
        x, y = self.canv.absolutePosition(0, 0)
        self.placements.append((self.canv.getPageNumber() - 1, self.path, x, y, self.width, self.height))


def build_pdf(header, sections):
    register_fonts()
    body = ParagraphStyle('Body', fontName='Body', fontSize=10.5, leading=13.4,
                          textColor=colors.HexColor(INK), spaceAfter=5, splitLongWords=False)
    caption = ParagraphStyle('Caption', fontName='Label', fontSize=8, leading=10,
                             textColor=colors.HexColor(MUTED), spaceBefore=3, spaceAfter=7)
    heading = ParagraphStyle('Heading', fontName='Label-Bold', fontSize=13.5, leading=16,
                             textColor=colors.HexColor(INK), spaceBefore=6, spaceAfter=6, keepWithNext=True)
    sources = ParagraphStyle('Sources', parent=caption, fontSize=7.5, leading=9.3, spaceBefore=5)
    title = ParagraphStyle('Title', fontName='Label-Bold', fontSize=25, leading=29,
                           textColor=colors.HexColor(INK), spaceAfter=2)
    subtitle = ParagraphStyle('Subtitle', fontName='Label', fontSize=12.2, leading=16,
                              textColor=colors.HexColor(MUTED), spaceAfter=5)
    status = ParagraphStyle('Status', parent=caption, fontSize=8.3, leading=11, spaceAfter=8)

    def blocks(items, width=WIDTH):
        result = []
        for tag in items:
            img = tag.find('img')
            if img:
                path = ROOT / img['src']
                if path.suffix == '.svg':
                    result += [Spacer(1, 3), VectorFigure(path, width)]
                else:
                    w, h = PILImage.open(path).size
                    result.append(Image(str(path), width=width, height=width * h / w))
            else:
                is_caption = len(tag.contents) == 1 and getattr(tag.contents[0], 'name', '') == 'em'
                is_sources = tag.get_text().startswith('Sources:')
                result.append(paragraph(tag, sources if is_sources else caption if is_caption else body))
        return result

    def photo_row(items, photo_width=120, left=False):
        i = next(i for i, p in enumerate(items) if p.find('img'))
        photo = blocks(items[i:i + 2], photo_width)
        copy = blocks(items[:i] + items[i + 2:])
        gap = 17
        text_width = WIDTH - photo_width - gap
        widths = [photo_width, gap, text_width] if left else [text_width, gap, photo_width]
        cells = [photo, '', copy] if left else [copy, '', photo]
        table = Table([cells], colWidths=widths, hAlign='LEFT')
        table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                   ('LEFTPADDING', (0, 0), (-1, -1), 0),
                                   ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                                   ('TOPPADDING', (0, 0), (-1, -1), 0),
                                   ('BOTTOMPADDING', (0, 0), (-1, -1), 0)]))
        return table

    story = [paragraph(header[0], title), paragraph(header[1], subtitle), paragraph(header[2], status)]
    for name in SECTIONS:
        if name == 'Results':
            story.append(PageBreak())
        story.append(Paragraph(name, heading))
        items = sections[name]
        if name == 'Introduction':
            story.append(photo_row(items))
        elif name == 'Discussion':
            story.extend(blocks(items[:1]))
            story.append(photo_row(items[1:-1], photo_width=150, left=True))
            story.extend(blocks(items[-1:]))
        else:
            story.extend(blocks(items))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#d8e3e8'))
        canvas.line(36, 27, 576, 27)
        canvas.setFont('Label', 7.7)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(36, 16, 'Gait Fidelity  ·  Research overview  ·  24 September 2026')
        canvas.drawRightString(576, 16, f'{doc.page} / 2')
        canvas.restoreState()

    VectorFigure.placements = []
    with tempfile.TemporaryDirectory(prefix='gait-brief-') as temp:
        raw = Path(temp) / 'typeset.pdf'
        doc = BaseDocTemplate(str(raw), pagesize=(612, 792), title='Gait Fidelity: research overview',
                              author='Gait Fidelity study', allowSplitting=1)
        frame = Frame(36, 36, WIDTH, 726, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        doc.addPageTemplates(PageTemplate(id='Overview', frames=[frame], onPage=footer))
        doc.build(story)
        reader = PdfReader(raw)
        if len(reader.pages) != 2:
            raise RuntimeError(f'Overview must be two readable pages, got {len(reader.pages)}; revise layout or prose.')
        writer = PdfWriter()
        writer.clone_document_from_reader(reader)
        for page, path, x, y, w, h in VectorFigure.placements:
            vector = PdfReader(io.BytesIO(cairosvg.svg2pdf(url=str(path)))).pages[0]
            transform = Transformation().scale(w / float(vector.mediabox.width), h / float(vector.mediabox.height)).translate(x, y)
            writer.pages[page].merge_transformed_page(vector, transform)
        with (ROOT / 'proposal-brief.pdf').open('wb') as output:
            writer.write(output)
    pdf = PdfReader(ROOT / 'proposal-brief.pdf')
    receipt = {'source': 'proposal-brief.md', 'outputs': ['proposal-brief.html', 'proposal-brief.pdf'],
               'major_sections': SECTIONS, 'pdf_pages': len(pdf.pages), 'body_font_pt': 10.5,
               'body_leading_pt': 13.4, 'page_size': 'US Letter', 'figures_remain_vector': True,
               'source_words': len((ROOT / 'proposal-brief.md').read_text().split()),
               'page_text_characters': [len(p.extract_text()) for p in pdf.pages],
               'figure_placements': [{'page': p + 1, 'source': str(f.relative_to(ROOT)),
                                      'x_pt': x, 'y_pt': y, 'width_pt': w, 'height_pt': h}
                                     for p, f, x, y, w, h in VectorFigure.placements]}
    (ROOT / 'records/brief-build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    header, sections = source()
    build_html(header, sections)
    build_pdf(header, sections)
