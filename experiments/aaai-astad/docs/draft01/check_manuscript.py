"""Check the built anonymous PDF, citations, fonts, bounds and official style."""
from pathlib import Path
import hashlib,json,re,subprocess,xml.etree.ElementTree as ET
B=Path(__file__).resolve().parent;R=B/'rendered';R.mkdir(exist_ok=True)
for args,out in [(['pdftotext','-bbox-layout',str(B/'manuscript.pdf'),'-'],'text_bounds.xhtml'),(['pdftotext','-layout',str(B/'manuscript.pdf'),'-'],'manuscript.txt'),(['pdffonts',str(B/'manuscript.pdf')],'fonts.txt'),(['pdfimages','-list',str(B/'manuscript.pdf')],'images.txt')]:
 r=subprocess.run(args,capture_output=True,text=True,check=True);(R/out).write_text(r.stdout)
root=ET.parse(R/'text_bounds.xhtml').getroot();ns={'x':'http://www.w3.org/1999/xhtml'};pages=root.findall('.//x:page',ns);outside=[];refs=[]
for i,p in enumerate(pages,1):
 for w in p.findall('.//x:word',ns):
  if float(w.attrib['xMin'])<0 or float(w.attrib['yMin'])<0 or float(w.attrib['xMax'])>float(p.attrib['width']) or float(w.attrib['yMax'])>float(p.attrib['height']):outside.append((i,w.text))
  if w.text=='References':refs.append(i)
fontlines=(R/'fonts.txt').read_text().splitlines()[2:]
fonts=[re.split(r'\s{2,}',line.strip()) for line in fontlines if line.strip()]
source=(B/'manuscript.md').read_text();text=(R/'manuscript.txt').read_text();aux=(B/'manuscript.aux').read_text();bbl=(B/'manuscript.bbl').read_text();log=(B/'build_logs/04_pdflatex.log').read_text()
cited={key for group in re.findall(r'\\citation\{([^}]+)\}',aux) for key in group.split(',')};bibkeys=set(re.findall(r'@\w+\{([^,]+)',(B/'references.bib').read_text()))
checks={
 'letter_pages':all(abs(float(p.attrib['width'])-612)<.1 and abs(float(p.attrib['height'])-792)<.1 for p in pages),
 'content_and_reference_limits':bool(refs) and min(refs)<=7 and len(pages)-min(refs)+1<=2,
 'no_words_outside_page':not outside,
 'all_fonts_embedded':bool(fonts) and all(re.search(r'\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$',line).group(1)=='yes' for line in fontlines if line.strip()),
 'no_type3_fonts':all('Type 3' not in line for line in fontlines),
 'vector_figures':len((R/'images.txt').read_text().splitlines())==2,
 'official_style_unchanged':all((B/n).read_bytes()==(B/'author_kit'/n).read_bytes() for n in ['aaai2027.sty','aaai2027.bst']),
 'no_latex_warnings':not any(x in log for x in ['Warning','Overfull','undefined']),
 'citations_defined':bool(cited) and cited<=bibkeys,
 'anonymous_text':all(x not in text for x in ['/Users/','pmui','alexpose']) and 'Anonymous submission' in text,
 'no_em_dashes':chr(0x2014) not in source,
 'figures_and_tables_referenced':all(x in source for x in ['ref{fig:method}','ref{fig:differences}','ref{tab:ranking}','ref{tab:attrition}']),
}
record={'passed':all(checks.values()),'checks':checks,'pages':len(pages),'references_start_pages':refs,'outside_page_words':outside,'cited_reference_count':len(cited),'pdf_sha256':hashlib.sha256((B/'manuscript.pdf').read_bytes()).hexdigest()}
(R/'checks.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if not record['passed']:raise SystemExit(1)
