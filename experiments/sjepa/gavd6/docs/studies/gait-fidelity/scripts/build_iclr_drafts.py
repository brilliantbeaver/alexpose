"""Build new scientific/overview HTML drafts; never overwrite the proposals.

Run with .venv/bin/python docs/studies/gait-fidelity/scripts/build_iclr_drafts.py.
Empirical plots have a separate source: build_iclr_draft_figures.py.
PDFs are browser prints of these HTML files, not separately typeset documents.
"""
from pathlib import Path
from html import escape
import io
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import mistune

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'images/iclr-draft-20260925'
FULL = 'full-writeup'
BRIEF = 'research-overview'
EQUATIONS = {
    'measurement': r'q_\ell=P_{95}(\theta_\ell)-P_5(\theta_\ell),\quad A=q_R-q_L,\quad\Delta A=A_b-A_a.',
    'response': r'E_\Delta=\left|(\widehat A_b-\widehat A_a)-(A_b-A_a)\right|.',
    'feature': r'L_\Delta=\frac{\|e_b-e_a\|^2}{2D},\qquad L_E=\frac{\|e_a\|^2+\|e_b\|^2}{2D}.',
    'dense': r'L_{\mathrm{dense}}=\frac{1}{|S|}\sum_{(t,\ell)\in S}\left[\frac{(\widehat\theta_{b,t,\ell}-\widehat\theta_{a,t,\ell})-(\theta_{b,t,\ell}-\theta_{a,t,\ell})}{180}\right]^2.',
}
ALT = {
    'measurement': 'Leg excursion q is percentile 95 minus percentile 5 of the knee angle; A is right minus left excursion; delta A is A at state b minus A at state a.',
    'response': 'Response error is the absolute difference between restored and reference changes in A.',
    'feature': 'Difference loss is the squared difference of endpoint feature errors divided by two D. Endpoint loss is the sum of their squared magnitudes divided by two D.',
    'dense': 'Dense loss averages squared angular-response errors, divided by 180 degrees before squaring, over supported times and legs.',
}

CSS = '''
:root{color-scheme:light;--ink:#1c2c37;--muted:#53636d;--blue:#245c83;--line:#d5dfe5}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}
body{margin:0;background:#edf1f4;color:var(--ink);font:18px/1.62 Georgia,'Times New Roman',serif}
a{color:var(--blue);text-decoration-thickness:1px;text-underline-offset:3px;overflow-wrap:anywhere}
a:focus-visible{outline:3px solid #a96b20;outline-offset:3px}
.toolbar{max-width:1000px;margin:20px auto;display:flex;gap:24px;flex-wrap:wrap;padding:0 28px;font:14px/1.5 Arial,sans-serif}
main{max-width:1000px;margin:0 auto 48px;padding:48px 64px;background:#fff;border:1px solid var(--line)}
h1,h2,h3{font-family:Arial,sans-serif;line-height:1.25;font-weight:650;color:#192e3f}
h1{font-size:39px;letter-spacing:-.6px;margin:0 0 14px}
.subtitle{font:21px/1.4 Arial,sans-serif;color:var(--muted);margin:0 0 12px}
.dateline{font:12px/1.5 Arial,sans-serif;letter-spacing:.25px;color:var(--muted);margin:12px 0 30px}
h2{font-size:26px;margin:38px 0 16px;padding-top:18px;border-top:1px solid var(--line)}
h3{font-size:20px;margin:27px 0 12px}p{margin:13px 0}strong{font-weight:700}
.contents{font:14px/1.7 Arial,sans-serif;border-left:3px solid #d5e4ed;padding:9px 0 9px 17px;margin:20px 0 26px}
.contents a{display:block;text-decoration:none}.contents a:hover{text-decoration:underline}
figure{margin:24px 0;break-inside:avoid}figure img{width:100%;height:auto;display:block}
figcaption{font:13px/1.5 Arial,sans-serif;color:var(--muted);margin-top:10px}
table{width:100%;border-collapse:collapse;margin:22px 0 8px;font:14px/1.45 Arial,sans-serif;font-variant-numeric:tabular-nums}
th{border-top:1.4px solid #556a77;border-bottom:1px solid #aebdc6;text-align:left;font-weight:700;color:#263e4d}
td{border-bottom:1px solid #dce3e8}th,td{padding:9px 10px;vertical-align:top}th:first-child,td:first-child{padding-left:0}
td.numeric,th.numeric{text-align:right}td.numeric{white-space:nowrap}td:last-child,th:last-child{padding-right:0}
.table-caption{font:13px/1.5 Arial,sans-serif;color:var(--muted);margin:8px 0 24px}
.table-caption em{font-style:normal}.equation{text-align:center;margin:22px 0;overflow-x:auto;break-inside:avoid}
.table-block{break-inside:avoid;max-width:100%;overflow-x:auto}
.equation img{display:block;margin:auto;max-width:100%;height:auto}
.references{font:14px/1.55 Arial,sans-serif}.references li{margin:8px 0}
code{font:0.86em Menlo,monospace;overflow-wrap:anywhere}.brief-references{font:12px/1.45 Arial,sans-serif;border-top:1px solid var(--line);padding-top:10px;margin-top:15px;color:var(--muted)}
.brief main{max-width:1000px}.brief h2{font-size:23px;margin:24px 0 10px;padding:0;border:0}
.brief figure{margin:16px 0}.brief .dateline{margin-bottom:20px}
@media(max-width:700px){body{font-size:17px}main{padding:26px 22px;border:0}h1{font-size:32px}.subtitle{font-size:18px}.toolbar{gap:12px;margin:15px auto;padding:0 22px}h2{font-size:23px}th,td{font-size:11px;padding:7px 5px}.references{font-size:13px}.equation img{min-width:480px}figcaption{font-size:12px}}
@media print{
 @page{size:letter;margin:.65in .65in .65in .65in}
 body{font-family:'Times New Roman',Times,serif;font-size:11pt;line-height:1.38;background:#fff;color:#152631}
 main,.brief main{max-width:none;margin:0;padding:0;border:0}
 .toolbar,.contents{display:none}h1{font-size:25pt;margin:0 0 10pt;letter-spacing:-.3pt}
 .subtitle{font-size:12pt;margin-bottom:8pt}.dateline{font-size:8.5pt;margin:7pt 0 18pt}
 h2{font-size:15pt;margin:20pt 0 10pt;padding-top:10pt;break-after:avoid}
 h3{font-size:12pt;margin:14pt 0 7pt;break-after:avoid}p{margin:8pt 0;orphans:3;widows:3}
 figure{margin:15pt 0;break-inside:avoid}figcaption,.table-caption{font-size:8.8pt;line-height:1.35;margin-top:7pt}
 table{font-size:9pt;line-height:1.33;margin-top:14pt}th,td{padding:7pt 6pt}thead{display:table-header-group}tr{break-inside:avoid}.table-block{overflow:visible}
 a{color:inherit;text-decoration:none}.equation{margin:14pt 0}.equation img{max-width:100%;min-width:0}
 .references{font-size:9pt;line-height:1.4}.references li{margin:5pt 0}
 .brief{font-size:11pt;line-height:1.32}.brief h1{font-size:22pt;margin-bottom:6pt}
 .brief .subtitle{font-size:10.8pt;line-height:1.3;margin-bottom:5pt}.brief .dateline{font-size:8pt;margin:5pt 0 12pt}
 .brief h2{font-size:12.2pt;margin:11pt 0 5pt;padding:0;border:0}.brief p{margin:5pt 0;orphans:2;widows:2}
 .brief figure{margin:9pt 0}.brief figcaption{font-size:8.3pt;line-height:1.28;margin-top:5pt}
 .brief #s5-experiments{margin-top:11pt}
 .brief-references{font-size:8pt;line-height:1.3;margin-top:9pt;padding-top:6pt}
}
'''


def equations():
    plt.rcParams.update({'svg.fonttype':'path', 'mathtext.fontset':'dejavuserif', 'font.family':'DejaVu Serif'})
    for key, formula in EQUATIONS.items():
        fig = plt.figure(figsize=(8, .6))
        fig.text(.01,.5, '$'+formula+'$', fontsize=17, va='center', color='#1c2c37')
        fig.savefig(ASSETS/f'equation-{key}.svg', bbox_inches='tight', pad_inches=.05, metadata={'Date':None})
        plt.close(fig)


def method_diagram():
    """Hand-laid SVG keeps training and deployment flows in separate bands."""
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="432" viewBox="0 0 1020 432"><title>JEPA-inspired pose restoration: training and deployment</title><desc>Observed poses train a student encoder and predictor against a reference-pose teacher. The teacher follows student parameters. Deployment uses a frozen encoder and readout, with an observed-coordinate residual.</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#526571"/></marker></defs><rect width="1020" height="432" fill="white"/>']
    def text(x,y,t,size=18,weight='normal',fill='#1c2c37',anchor='start'):
        svg.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(t)}</text>')
    def box(x,y,w,h,title,sub,fill='#f0f5f8'):
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="#becbd4"/>')
        text(x+w/2,y+27,title,19,'bold',anchor='middle')
        text(x+w/2,y+52,sub,16,anchor='middle')
    def arrow(points,dashed=False):
        dash=' stroke-dasharray="6 5"' if dashed else ''
        svg.append(f'<polyline points="{points}" fill="none" stroke="#526571" stroke-width="1.8"{dash} marker-end="url(#arrow)"/>')
    text(14,23,'A  PREDICTIVE PRETRAINING',18,'bold')
    box(14,46,215,70,'Estimated poses','artificially masked')
    box(280,46,170,70,'Student encoder','trainable features')
    box(502,46,170,70,'Predictor','trainable')
    arrow('229,81 280,81');arrow('450,81 502,81')
    box(14,175,215,70,'Reference poses','projected body joints','#edf6f2')
    box(280,175,170,70,'Teacher encoder','targets; no gradient','#edf6f2')
    arrow('229,210 280,210')
    arrow('365,116 365,175',True);text(380,149,'moving average',14)
    svg.append('<rect x="745" y="46" width="260" height="199" rx="4" fill="#faf7ef" stroke="#cfc5ad"/>')
    text(875,76,'Supervision',20,'bold',anchor='middle')
    for y,t in [(108,'Match reference features'),(147,'Plus matched auxiliary:'),(177,'each state independently'),(205,'or the difference of states')]:text(875,y,t,17,anchor='middle')
    arrow('672,81 745,81');arrow('450,210 745,210')
    text(14,284,'B  RESTORATION AFTER READOUT FITTING',18,'bold')
    box(14,304,215,70,'Estimated poses','no artificial masking')
    box(280,304,170,70,'Frozen encoder','learned features')
    box(502,304,170,70,'Fitted readout','coordinate output')
    box(745,304,260,70,'Restored trajectory','12 joints × 128 frames','#edf6f2')
    arrow('229,339 280,339');arrow('450,339 502,339');arrow('672,339 745,339')
    arrow('120,374 120,408 875,408 875,374')
    svg.append('<rect x="287" y="395" width="425" height="24" fill="white"/>')
    text(500,414,'Add observed coordinates where available',16,anchor='middle')
    svg.append('</svg>')
    (ASSETS/'method-flow.svg').write_text('\n'.join(svg))


def build(stem, brief=False):
    source=(ROOT/f'{stem}.md').read_text()
    soup=BeautifulSoup(mistune.create_markdown(escape=False,plugins=['table'])(source),'html.parser')
    headings=[]
    for h in soup.find_all('h2'):
        slug=re.sub('[^a-z0-9]+','-',h.get_text().lower()).strip('-')
        if slug[0].isdigit():slug='s'+slug
        h['id']=slug;headings.append((slug,h.get_text()))
    for div in soup.select('[data-equation]'):
        k=div['data-equation'];img=soup.new_tag('img',src=f'images/iclr-draft-20260925/equation-{k}.svg',alt=ALT[k])
        dims=re.search(r'width="([\d.]+)pt" height="([\d.]+)pt"',(ASSETS/f'equation-{k}.svg').read_text())
        if dims:img['style']=f'width:{float(dims[1])*1.16:.1f}px'
        div.append(img)
    for p in soup.find_all('p'):
        if re.match('Table [0-9]',p.get_text()):p['class']='table-caption'
    for img in soup.select('figure img'):
        link=soup.new_tag('a',href=img['src'],attrs={'aria-label':'Open full-size figure: '+img.get('alt','Figure')})
        img.wrap(link)
    for td in soup.find_all('td'):
        if re.fullmatch(r'[+−\-\d.,°%\[\]– ]+',td.get_text().strip()):td['class']='numeric'
    for table in soup.find_all('table'):
        table['aria-label']=table.find('tr').get_text(' ',strip=True)
        rows=table.select('tbody tr')
        for col, th in enumerate(table.select('thead th')):
            cells=[r.find_all('td')[col] for r in rows]
            if cells and sum('numeric' in c.get('class',[]) for c in cells)>=len(cells)/2:
                th['class']='numeric'
        caption=table.find_next_sibling()
        wrapper=soup.new_tag('div',attrs={'class':'table-block'})
        table.wrap(wrapper)
        if caption and caption.name=='p' and 'table-caption' in caption.get('class',[]):
            wrapper.append(caption.extract())
    toc='' if brief else '<nav class="contents" aria-label="Contents">'+''.join(f'<a href="#{slug}">{escape(title)}</a>' for slug,title in headings)+'</nav>'
    h=soup.find('h2')
    if toc:h.insert_before(BeautifulSoup(toc,'html.parser'))
    other=FULL if brief else BRIEF
    otherlabel='Full scientific draft' if brief else 'Two-page overview'
    cls='brief' if brief else 'full'
    html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(soup.h1.get_text())+(' — two-page overview' if brief else ' — scientific draft')+'</title><meta name="description" content="Evidence-grounded gait fidelity study: completed synthetic experiments on predictive representations, loss weighting and measurement reliability."><style>'+CSS+'</style></head><body class="'+cls+'"><nav class="toolbar" aria-label="Document versions"><a href="'+other+'.html">'+otherlabel+'</a><a href="'+stem+'.pdf">Print PDF</a><a href="'+stem+'.md">Editable source</a><a href="proposal.html">Original proposal</a></nav><main>'+str(soup)+'</main></body></html>\n'
    (ROOT/f'{stem}.html').write_text(html)
    return {'file':stem+'.html','words':len(soup.get_text(' ',strip=True).split()),'headings':headings,'figures':len(soup.find_all('figure'))}


if __name__=='__main__':
    ASSETS.mkdir(parents=True,exist_ok=True)
    equations();method_diagram()
    record=[build(FULL),build(BRIEF,True)]
    (ROOT/'reviews/iclr-drafts-20260925/build.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
