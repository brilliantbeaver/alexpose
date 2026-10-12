"""Build the unchanged three-page overview followed by a research direction."""
from pathlib import Path
import hashlib
import html
import json
import os
import subprocess
from pdfrw import PdfReader as VectorReader
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate,Flowable,Frame,PageTemplate,Paragraph,PageBreak,KeepTogether,Spacer

BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/'writeup/OVERVIEW.md'
CONTINUATION=BASE/'writeup/RESEARCH_DIRECTION.md'
OUTPUT=BASE/'output/pdf/jepa-gait-overview.pdf'
QA=BASE/'qa/overview';QA.mkdir(parents=True,exist_ok=True)
INK,MUTED=colors.HexColor('#233343'),colors.HexColor('#556573')
for family in ['Georgia','Arial']:
    for suffix,name in [('',family),(' Bold',family+'-Bold'),(' Italic',family+'-Italic'),(' Bold Italic',family+'-BoldItalic')]:
        pdfmetrics.registerFont(TTFont(name,f'/System/Library/Fonts/Supplemental/{family}{suffix}.ttf'))
    pdfmetrics.registerFontFamily(family,normal=family,bold=family+'-Bold',italic=family+'-Italic',boldItalic=family+'-BoldItalic')
styles={
 'title':ParagraphStyle('title',fontName='Arial-Bold',fontSize=23,leading=26,textColor=INK,spaceAfter=10,keepWithNext=True),
 'byline':ParagraphStyle('byline',fontName='Arial',fontSize=9,leading=12,textColor=MUTED,spaceAfter=16,keepWithNext=True),
 'body':ParagraphStyle('body',fontName='Georgia',fontSize=10.6,leading=14.4,textColor=INK,spaceAfter=8,allowOrphans=0,allowWidows=0),
 'h2':ParagraphStyle('h2',fontName='Arial-Bold',fontSize=12.5,leading=16,textColor=INK,spaceBefore=9,spaceAfter=7,keepWithNext=True),
 'h3':ParagraphStyle('h3',fontName='Arial-Bold',fontSize=10,leading=13,textColor=INK,spaceBefore=9,spaceAfter=5,keepWithNext=True),
 'caption':ParagraphStyle('caption',fontName='Arial',fontSize=8.5,leading=11.1,textColor=MUTED,spaceBefore=6,spaceAfter=0),
 'reference':ParagraphStyle('reference',fontName='Arial',fontSize=8.4,leading=10.6,textColor=MUTED,spaceAfter=4,leftIndent=13,firstLineIndent=-13),
}
def inline(items):
    out=[]
    for x in items:
        t,c=x['t'],x.get('c')
        if t=='Str':out.append(html.escape(c))
        elif t in ['Space','SoftBreak']:out.append(' ')
        elif t in ['Emph','Strong']:
            tag='i' if t=='Emph' else 'b';out.append(f'<{tag}>{inline(c)}</{tag}>')
        elif t=='Link':
            url=c[2][0]
            if not url.startswith(('https://','http://')):url=os.path.relpath((SOURCE.parent/url).resolve(),OUTPUT.parent)
            out.append(f'<link href="{html.escape(url,quote=True)}" color="#006a91">{inline(c[1])}</link>')
        else:raise ValueError(t)
    return ''.join(out)
class Vector(Flowable):
    def __init__(self,path):
        Flowable.__init__(self);self.form=pagexobj(VectorReader(str(path)).pages[0]);b=list(map(float,self.form.BBox))
        self.x,self.y=b[:2];self.w,self.h=b[2]-b[0],b[3]-b[1];self.width=504;self.height=self.h*504/self.w
    def draw(self):
        c=self.canv;c.saveState();c.scale(504/self.w,504/self.w);c.translate(-self.x,-self.y);c.doForm(makerl(c,self.form));c.restoreState()
class Report(BaseDocTemplate):
    def __init__(self):
        BaseDocTemplate.__init__(self,str(OUTPUT),pagesize=(612,792),initialFontName='Arial',
          title='Preserving gait measurements in ambient mobility assessment',author='Theodore Mui')
        self.addPageTemplates(PageTemplate('report',[Frame(54,48,504,696,id='body',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=self.decor))
    def decor(self,c,doc):
        c.saveState();c.setFont('Arial',8);c.setFillColor(MUTED)
        if doc.page>1:c.drawString(54,765,'JEPA and gait measurement | '+('Research overview' if doc.page<=3 else 'Proposed continuation'))
        c.setStrokeColor(colors.HexColor('#cdd8df'));c.setLineWidth(.5);c.line(54,34,558,34)
        c.drawString(54,21,'Theodore Mui | September 2026');c.drawRightString(558,21,str(doc.page));c.restoreState()

combined=SOURCE.read_text().rstrip()+'\n\n<!-- PAGEBREAK -->\n\n'+CONTINUATION.read_text()
ast=json.loads(subprocess.check_output(['pandoc','-f','markdown-implicit_figures','-t','json'],input=combined,text=True))
blocks=ast['blocks'];story=[];i=0
while i<len(blocks):
    b=blocks[i];t,c=b['t'],b.get('c')
    if t=='Header':story.append(Paragraph(inline(c[2]),styles['title' if c[0]==1 else 'h2' if c[0]==2 else 'h3']))
    elif t in ['Para','Plain']:
        if len(c)==1 and c[0]['t']=='Image':
            path=(SOURCE.parent/c[0]['c'][2][0]).with_suffix('.pdf');caption=blocks[i+1]['c']
            story.append(KeepTogether([Spacer(1,3),Vector(path),Paragraph(inline(caption),styles['caption'])]));i+=1
            if path.name=='10-research-design.pdf':story.append(Spacer(1,8))
        else:story.append(Paragraph(inline(c),styles['byline' if i>0 and blocks[i-1]['t']=='Header' and blocks[i-1]['c'][0]==1 else 'body']))
    elif t=='RawBlock':
        assert c[1].strip()=='<!-- PAGEBREAK -->';story.append(PageBreak())
    elif t=='OrderedList':
        for n,item in enumerate(c[1],start=c[0][0]):story.append(Paragraph(f'{n}. '+inline(item[0]['c']),styles['reference']))
    else:raise ValueError(t)
    i+=1
Report().build(story)
reader=PdfReader(OUTPUT);pages=[p.extract_text() for p in reader.pages]
assert 'Recommended continuation' in pages[3], 'The original overview must remain exactly three pages'
assert len(pages)==7,f'Unexpected combined length: {len(pages)} pages'
fragment=subprocess.check_output(['pandoc','-f','markdown-implicit_figures','-t','html5'],input=combined,text=True)
document='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Preserving gait measurements in ambient mobility assessment</title><link rel="stylesheet" href="overview.css"></head><body><nav aria-label="Document formats"><a href="../output/pdf/jepa-gait-overview.pdf">Overview and continuation PDF</a><a href="OVERVIEW.md">Overview source</a><a href="RESEARCH_DIRECTION.md">Continuation source</a><a href="writeup.html">Detailed report</a></nav><main>'''+fragment+'</main></body></html>'
(BASE/'writeup/overview.html').write_text(document)
manifest={'pages':len(pages),'words_by_pdf_page':[len(p.split()) for p in pages],
 'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'continuation_sha256':hashlib.sha256(CONTINUATION.read_bytes()).hexdigest(),
 'overview_pages':3,'continuation_pages':len(pages)-3,
 'pdf_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
 'body_font_points':10.6,'body_leading_points':14.4,
 'figure_sha256':hashlib.sha256((BASE/'figures/09-overview-results.pdf').read_bytes()).hexdigest(),
 'research_figure_sha256':hashlib.sha256((BASE/'figures/10-research-design.pdf').read_bytes()).hexdigest(),
 'overview_preserved_from_before_extension':hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='8e9a022d9880a3f2af8f577fca758c8e83bd8227c0645e5f116a53485dec78af'}
(QA/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(QA/'extracted-text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{p}' for i,p in enumerate(pages)))
print(json.dumps(manifest,indent=2))
