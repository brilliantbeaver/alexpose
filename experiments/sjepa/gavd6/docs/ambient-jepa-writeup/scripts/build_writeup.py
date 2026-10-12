"""Build the full research narrative from one Markdown source.

Dependencies: reportlab, pdfrw, pypdf, Pandoc; no network or model execution.
Vector figure PDFs are embedded without rasterization. The Markdown is canonical.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

from pdfrw import PdfReader as VectorReader
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer,
)

HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / "writeup/WRITEUP.md"
OUTPUT = HERE / "output/pdf/jepa-gait-research-writeup.pdf"
QA = HERE / "qa/writeup"
QA.mkdir(parents=True, exist_ok=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

INK = colors.HexColor("#233343")
BLUE = colors.HexColor("#006a91")
MUTED = colors.HexColor("#556573")
RULE = colors.HexColor("#cdd8df")
WIDTH, HEIGHT, MARGIN = 612, 792, 54
CONTENT = WIDTH - 2 * MARGIN

font_dir = Path("/System/Library/Fonts/Supplemental")
for family in ("Georgia", "Arial"):
    for suffix, name in [("", family), (" Bold", family + "-Bold"),
                         (" Italic", family + "-Italic"),
                         (" Bold Italic", family + "-BoldItalic")]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / f"{family}{suffix}.ttf")))
    pdfmetrics.registerFontFamily(family, normal=family, bold=family + "-Bold",
                                 italic=family + "-Italic", boldItalic=family + "-BoldItalic")

styles = {
    "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=27, leading=30,
                            textColor=INK, spaceAfter=10, keepWithNext=True),
    "subtitle": ParagraphStyle("subtitle", fontName="Arial", fontSize=13, leading=17,
                               textColor=BLUE, spaceAfter=10, keepWithNext=True),
    "byline": ParagraphStyle("byline", fontName="Arial", fontSize=9.3, leading=13,
                             textColor=MUTED, spaceAfter=20, keepWithNext=True),
    "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=15, leading=19,
                         textColor=INK, spaceBefore=15, spaceAfter=8, keepWithNext=True),
    "h3": ParagraphStyle("h3", fontName="Arial-Bold", fontSize=11.2, leading=15,
                         textColor=INK, spaceBefore=11, spaceAfter=6, keepWithNext=True),
    "body": ParagraphStyle("body", fontName="Georgia", fontSize=10.4, leading=14.7,
                           textColor=INK, spaceAfter=8, allowWidows=0, allowOrphans=0),
    "caption": ParagraphStyle("caption", fontName="Arial", fontSize=8.7, leading=11.5,
                              textColor=MUTED, spaceBefore=7, spaceAfter=13),
    "equation": ParagraphStyle("equation", fontName="Georgia", fontSize=10.4, leading=17,
                               textColor=INK, alignment=TA_CENTER, spaceBefore=5, spaceAfter=12),
    "reference": ParagraphStyle("reference", fontName="Arial", fontSize=9.1, leading=12.5,
                                textColor=INK, leftIndent=22, firstLineIndent=-22, spaceAfter=8),
    "bullet": ParagraphStyle("bullet", fontName="Georgia", fontSize=10.4, leading=14.7,
                             textColor=INK, leftIndent=14, firstLineIndent=-12, spaceAfter=4),
}

def plain(inlines):
    result = []
    for item in inlines:
        t, c = item["t"], item.get("c")
        if t == "Str": result.append(c)
        elif t in ("Space", "SoftBreak", "LineBreak"): result.append(" ")
        elif t in ("Strong", "Emph", "Superscript", "Subscript"): result.append(plain(c))
        elif t in ("Link", "Image"): result.append(plain(c[1]))
        elif t == "Math": result.append(c[1])
        else: raise ValueError(f"Unhandled plain inline: {t}")
    return "".join(result)

def inline(inlines):
    result = []
    for item in inlines:
        t, c = item["t"], item.get("c")
        if t == "Str": result.append(html.escape(c))
        elif t in ("Space", "SoftBreak"): result.append(" ")
        elif t == "LineBreak": result.append("<br/>")
        elif t in ("Strong", "Emph", "Superscript", "Subscript"):
            tag = {"Strong":"b", "Emph":"i", "Superscript":"super", "Subscript":"sub"}[t]
            result.append(f"<{tag}>{inline(c)}</{tag}>")
        elif t == "Link":
            target = c[2][0]
            # Resolve the source-paper link relative to the PDF's own directory.
            if not target.startswith(("https://", "http://", "#")):
                import os
                target = os.path.relpath((SOURCE.parent / target).resolve(), OUTPUT.parent)
            result.append(f'<link href="{html.escape(target, quote=True)}" color="#006a91">{inline(c[1])}</link>')
        else: raise ValueError(f"Unhandled formatted inline: {t}")
    return "".join(result)

def equation(tex):
    if tex.startswith("q_"):
        return ('q<sub>i,ℓ</sub> = P<sub>95</sub>(θ<sub>i,ℓ</sub>) - P<sub>5</sub>(θ<sub>i,ℓ</sub>)'
                '<br/>A<sub>i</sub> = q<sub>i,R</sub> - q<sub>i,L</sub> '
                ';&nbsp;&nbsp;&nbsp; ΔA = A<sub>b</sub> - A<sub>a</sub>')
    if tex.startswith("e_i"):
        return 'e<sub>i</sub> = H[p<sub>i</sub>/0.1 - sg((t<sub>i</sub> - c)/0.06)]'
    if tex.startswith("L_{E,k}"):
        return ('L<sub>E,k</sub> = (‖e<sub>a</sub>‖<super>2</super> + ‖e<sub>b</sub>‖<super>2</super>)/(2D)'
                '<br/>L<sub>Δ,k</sub> = ‖e<sub>b</sub> - e<sub>a</sub>‖<super>2</super>/(2D)')
    if tex.startswith("S(C)"):
        return 'S(C) = U + C f.'
    raise ValueError(f"Unhandled equation: {tex}")

class VectorFigure(Flowable):
    def __init__(self, path, alt, width=CONTENT):
        super().__init__()
        self.path, self.alt = path, alt
        self.form = pagexobj(VectorReader(str(path)).pages[0])
        box = [float(v) for v in self.form.BBox]
        self.x0, self.y0 = box[:2]
        self.natural_w, self.natural_h = box[2]-box[0], box[3]-box[1]
        self.width, self.height = width, width * self.natural_h / self.natural_w
        self.hAlign = "CENTER"
    def draw(self):
        c = self.canv
        c.saveState()
        c.scale(self.width/self.natural_w, self.height/self.natural_h)
        c.translate(-self.x0, -self.y0)
        c.doForm(makerl(c, self.form))
        c.restoreState()

class Report(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename), pagesize=(WIDTH,HEIGHT), leftMargin=MARGIN,
                         rightMargin=MARGIN, topMargin=48, bottomMargin=48,
                         initialFontName="Arial",
                         title="Can a motion model preserve how someone walks?",
                         author="Theodore Mui", subject="JEPA, gait asymmetry, and ambient biomechanics")
        frame=Frame(MARGIN, 48, CONTENT, HEIGHT-96, id="body", leftPadding=0,
                    rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="report", frames=frame, onPage=self.decor))
        self.positions=[]
    def decor(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setFont("Arial",8)
            canvas.setFillColor(MUTED)
            canvas.drawString(MARGIN,HEIGHT-29,"JEPA, gait asymmetry, and ambient biomechanics")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(.5)
        canvas.line(MARGIN,34,WIDTH-MARGIN,34)
        canvas.setFont("Arial",8)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN,21,"Theodore Mui  |  Research writeup  |  September 2026")
        canvas.drawRightString(WIDTH-MARGIN,21,str(doc.page))
        canvas.restoreState()
    def afterFlowable(self, flowable):
        if hasattr(flowable,"bookmark"):
            title, anchor, level = flowable.bookmark
            self.canv.bookmarkPage(anchor)
            self.canv.addOutlineEntry(title,anchor,level=level,closed=False)
            self.positions.append({"type":"heading","title":title,"page":self.page})
        if isinstance(flowable,VectorFigure):
            self.positions.append({"type":"figure","asset":flowable.path.name,"page":self.page,
                                   "width_pt":flowable.width,"height_pt":flowable.height,"alt":flowable.alt})

ast=json.loads(subprocess.check_output(["pandoc",str(SOURCE),"-f","markdown-implicit_figures","-t","json"]))
blocks=ast["blocks"]
story=[]
figures=[]
i=0
companion=False
references=False
while i < len(blocks):
    b=blocks[i]; t,c=b["t"],b.get("c")
    if t=="Header":
        level, attr, content=c
        title=plain(content)
        if title in ("Technical companion", "References and source notes"):
            story.append(PageBreak())
            companion=title=="Technical companion"
            references=title=="References and source notes"
        p=Paragraph(inline(content),styles["title" if level==1 else "h2" if level==2 else "h3"])
        # Outline begins with the report title; subordinate sections follow.
        p.bookmark=(title,attr[0],level-1)
        story.append(p)
    elif t in ("Para","Plain"):
        if len(c)==1 and c[0]["t"]=="Image":
            _, alt, target=c[0]["c"]
            path=(SOURCE.parent/target[0]).resolve().with_suffix(".pdf")
            fig=VectorFigure(path,plain(alt))
            assert i+1<len(blocks) and blocks[i+1]["t"]=="Para"
            cap=blocks[i+1]["c"]
            assert plain(cap).startswith("Figure ")
            group=[Spacer(1,8),fig,Paragraph(inline(cap),styles["caption"])]
            story.append(KeepTogether(group))
            figures.append({"number":plain(cap).split(".")[0],"asset":path.name,
                            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
            i+=1
        elif len(c)==1 and c[0]["t"]=="Math":
            math_text=equation(c[0]["c"][1].strip()).replace("‖",'<font name="Arial">‖</font>')
            story.append(Paragraph(math_text,styles["equation"]))
        else:
            name="subtitle" if i==1 else "byline" if i==2 else "body"
            p=Paragraph(inline(c),styles[name])
            # Keep this short interpretive paragraph intact after the large graph.
            story.append(KeepTogether([p]) if plain(c).startswith("However, the zero-response baseline") else p)
    elif t=="BulletList":
        for item in c:
            assert len(item)==1 and item[0]["t"] in ("Para","Plain")
            story.append(Paragraph("• " + inline(item[0]["c"]),styles["bullet"]))
        story.append(Spacer(1,5))
    elif t=="OrderedList":
        attr,items=c
        for n,item in enumerate(items,start=attr[0]):
            assert len(item)==1 and item[0]["t"] in ("Para","Plain")
            story.append(Paragraph(f"{n}. " + inline(item[0]["c"]),styles["reference"]))
    else:
        raise ValueError(f"Unhandled block: {t}")
    i+=1

doc=Report(OUTPUT)
doc.build(story)

# HTML uses the same source, native MathML, and the editable SVG figure assets.
template=HERE/"writeup/template.html"
subprocess.run(["pandoc",str(SOURCE),"-f","markdown-implicit_figures","-t","html5",
                "--standalone","--mathml","--toc","--toc-depth=2",
                f"--template={template}","--metadata","pagetitle=Can a motion model preserve how someone walks?",
                "-o",str(HERE/"writeup/writeup.html")],check=True)

reader=PdfReader(OUTPUT)
pages=[p.extract_text() for p in reader.pages]
assert len(figures)==8
assert len(re.findall(r'^\*\*Figure ',SOURCE.read_text(),re.M))==8
assert all(p.strip() for p in pages)
assert "References and source notes" in "\n".join(pages)
manifest={
    "source":str(SOURCE.relative_to(HERE)),
    "source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "pdf_sha256":hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    "pages":len(reader.pages),"figures":figures,"locations":doc.positions,
    "page_words":[len(p.split()) for p in pages],
    "figure_width_inches":CONTENT/72,
    "empirical_figures_recomputed":False,
    "new_model_fitting":False,
}
(QA/"build-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
(QA/"extracted-text.txt").write_text("\n\n".join(f"PAGE {n+1}\n{p}" for n,p in enumerate(pages)))
print(json.dumps({"pdf":str(OUTPUT),"pages":len(reader.pages),"figures":len(figures),
                  "page_words":manifest["page_words"],"locations":doc.positions},indent=2))
