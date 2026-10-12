"""Check paper numbers, citation keys, anonymous layout, and source preservation."""
from pathlib import Path
import json, hashlib, re, subprocess
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent.parent

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
audit=json.loads((ROOT/'evidence/results-audit.json').read_text())
notebooks=json.loads((ROOT/'evidence/notebook-audit-checks.json').read_text())['notebooks']
report={'source_notebooks_unchanged':{},'revisions':{}}
for filename,record in notebooks.items():
 report['source_notebooks_unchanged'][filename]=digest(EXP/filename)==record['sha256']
assert all(report['source_notebooks_unchanged'].values())
report['original_pdf_unchanged']=digest(ROOT/'wacv2007-v1.pdf')==audit['original_draft_audit']['draft_sha256']
assert report['original_pdf_unchanged']
for version in range(2,10):
 p=ROOT/f'wacv2007-v{version}.tex'
 if not p.exists():continue
 source=p.read_text();pdf=p.with_suffix('.pdf')
 text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
 pages=text.split('\f')
 if not pages[-1].strip():pages.pop()
 reference_pages=[i+1 for i,page in enumerate(pages) if re.search(r'\bReferences\b',page)]
 assert reference_pages and reference_pages[0]<=8, (version,reference_pages)
 assert 'pmui' not in text.lower() and '/Users/' not in text
 bibname=re.search(r'\\bibliography\{([^}]+)\}',source)[1]
 bib=(ROOT/(bibname+'.bib')).read_text()
 available=set(re.findall(r'@\w+\{([^,]+),',bib))
 used=set()
 for bundle in re.findall(r'\\cite\w*\{([^}]+)\}',source):used.update(bundle.split(','))
 assert used<=available,used-available
 assert not re.search(r'Citation.+undefined|Reference.+undefined|Overfull \\hbox|Overfull \\vbox', (ROOT/f'qa/wacv2007-v{version}/wacv2007-v{version}.log').read_text()), version
 # Exact values independently derived from notebook-associated OOF rows.
 expected=['0.457&0.397&$0.438\\pm0.196$','0.452&0.410&$0.427\\pm0.132$','0.411&0.397&$0.395\\pm0.098$','0.318&0.318&$0.285\\pm0.104$','0.259&0.202&$0.159\\pm0.022$']
 assert all(value in source for value in expected),(version,'main table mismatch')
 if version>=3:
  assert all(value in source for value in ['[-0.160,0.174]','[-0.133,0.220]','[-0.046,0.304]'])
  protocol=source + ((ROOT/'evidence/v9-methods.md').read_text() if version>=9 else '')
  assert '20261004' in protocol and '10,000' in protocol
 report['revisions'][str(version)]={'pdf_sha256':digest(pdf),'tex_sha256':digest(p),'pages_total':len(pages),'references_begin_page':reference_pages[0],'main_content_within_eight_pages':True,'references_resolved':len(used),'main_result_table_verified':True,'anonymity_scan_pass':True}
report['notes']=['References may share their first page with final main-text paragraphs.','DRAFT is an intentional submission-ID placeholder, not an assigned OpenReview ID.','Numeric checks supplement independent computation and manual page inspection.']
(ROOT/'qa/delivery-verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
