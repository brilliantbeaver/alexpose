"""Final artifact audit; writes qa/final-validation.json. No evidence mutation."""
from pathlib import Path
import json,re,hashlib,zipfile
import pymupdf
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'docs/iclr'
report={'versions':[],'checks':{}}
for v in range(1,8):
 stem=f'paper-v{v:02d}-final';version_dir=OUT/f'versions/v{v:02d}';src=version_dir/f'{stem}.tex';pdf=version_dir/f'{stem}.pdf'
 if not src.exists() or not pdf.exists():raise SystemExit(f'Missing {stem}')
 text=src.read_text();doc=pymupdf.open(pdf)
 pages=[x.get_text() for x in doc]
 starts=[i+1 for i,t in enumerate(pages) if 'ai use disclosure' in t.lower()]
 assert starts==[10],(stem,starts)
 assert r'\title{Evaluating Feature Prediction for 2D Pose Trajectory Restoration with Paired Synthetic Supervision}' in text
 assert any('NimbusRom' in ft[3] or 'Times' in ft[3] for ft in doc[0].get_fonts()),stem
 assert 'undefined' not in (version_dir/'logs'/f'{stem}.blg').read_text().lower(),stem
 keys=[]
 for match in re.findall(r'\\cite\w*\{([^}]*)\}',text):keys+=match.split(',')
 bib=(OUT/'references.bib').read_text();missing=[k for k in keys if not re.search(r'@\w+\{'+re.escape(k)+',',bib)]
 assert not missing,(stem,missing)
 paper_issues=[]
 for i,page in enumerate(doc):
  for b in page.get_text('dict')['blocks']:
   for line in b.get('lines',[]):
    for sp in line['spans']:
     x0,y0,x1,y1=sp['bbox']
     if x0<0 or y0<0 or x1>page.rect.width+.5 or y1>page.rect.height+.5:paper_issues.append([i+1,sp['text']])
 report['versions'].append({'version':v,'main_pages':9,'total_pages':len(doc),'reference_and_disclosure_start':10,'citation_keys':sorted(set(keys)),'out_of_page_text':paper_issues,'tex_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()})
 assert not paper_issues,(stem,paper_issues)
# Bibliography source exists, and final publication text contains no draft placeholders.
final=(OUT/'versions/v07/paper-v07-final.tex').read_text()
assert not re.search(r'TODO|FIXME|TODO_PRIMARY|TODO_CALIBRATION',final)
report['checks']={'requested_title':True,'all_main_texts_nine_pages':True,'all_cited_keys_present_in_bibliography':True,'all_page_text_within_media_box':True,'final_has_no_draft_placeholders':True,'times_family_embedded':True,'limits':'Text geometry does not replace visual review; prior versions retain recorded scientific/editorial issues. Initial scientific reviews refer to preserved preflight renders; final typeset companions correct font encoding and complete bibliography without changing results.'}
with zipfile.ZipFile(OUT/'template/iclr-2027-style-files.zip') as archive:
 template_checks={name:hashlib.sha256(archive.read(name)).hexdigest()==hashlib.sha256((OUT/'template'/name).read_bytes()).hexdigest() for name in archive.namelist() if not name.endswith('/')}
assert all(template_checks.values())
report['checks']['official_template_unmodified']=template_checks
final_doc=pymupdf.open(OUT/'versions/v07/paper-v07-final.pdf')
final_pdf_text='\n'.join(page.get_text() for page in final_doc)
assert not any(marker in final_pdf_text for marker in ['theodoremui','tedmui','/Users/','/hai/scratch/'])
report['checks']['no_private_identity_paths_in_final_pdf']=True
assert all('undefined' not in (OUT/f'versions/v{v:02d}/logs/paper-v{v:02d}-final.log').read_text().lower() for v in range(1,8))
report['checks']['no_undefined_font_or_citation_warnings_in_delivered_builds']=True
(OUT/'qa/final-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'versions':len(report['versions']),'main_pages':[v['main_pages'] for v in report['versions']],'status':'passed'},indent=2))
