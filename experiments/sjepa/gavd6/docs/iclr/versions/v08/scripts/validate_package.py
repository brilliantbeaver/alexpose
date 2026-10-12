"""Validate the delivered paper, dependencies, evidence, and preserved history."""
from pathlib import Path
import hashlib,json,re,subprocess
from pypdf import PdfReader
HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pdf=HERE/'paper-v08.pdf';r=PdfReader(pdf)
text='\n'.join(p.extract_text() for p in r.pages)
aux=(HERE/'paper-v08.aux').read_text()
main=int(re.search(r'\\newlabel\{maintextend\}\{\{[^}]*\}\{(\d+)\}',aux).group(1))
assert main==9 and len(r.pages)==16,(main,len(r.pages))
assert 'AI USE DISCLOSURE' in r.pages[9].extract_text()
assert 'DATAINTEGRITYANDEVALUATION' in re.sub(r'\s+','',r.pages[11].extract_text())
title='Evaluating JEPA-Inspired Motion Representations through Geometry and Gait Asymmetry'
assert r.metadata.title==title and not r.metadata.get('/Author')
assert all(tuple(map(float,p.mediabox[2:]))==(612.,792.) for p in r.pages)
for token in ['/Users/','/home/','theodoremui','tedmui','TODO','FIXME','TBD']:
    assert token not in text,token
portable_text_tokens=['notebook','/Users/','/home/','outputs/','scripts/','.ipynb','.csv','.json','checkout','local director','remote validation','disk exhaustion','tutorial','fixture','study artifacts','compact exports']
for token in portable_text_tokens:
    assert token.lower() not in text.lower(),('reader-facing local setup',token)
log=(HERE/'paper-v08.log').read_text()
for token in ['Overfull','undefined','Missing character','Font shape']:
    assert token not in log,token
source=(HERE/'paper-v08.tex').read_text()
for part in [source.split('\\begin{abstract}',1)[1].split('\\end{abstract}',1)[0], source.split('\\section{Introduction}',1)[1].split('\\section{Related',1)[0], source.split('\\section{Discussion and limitations}',1)[1].split('\\label{maintextend}',1)[0]]:
    for term in ['self-supervised learning','jepa','world models']:assert term in part.lower(),term
tex=source+(HERE/'appendix.tex').read_text()
keys=set(k.strip() for part in re.findall(r'\\cite\w*\{([^}]+)\}',tex) for k in part.split(','))
bib=set(re.findall(r'@\w+\{([^,]+),',(HERE/'references.bib').read_text()))
assert keys<=bib,keys-bib
font_checks={}
for p in [pdf]+sorted((HERE/'figures').glob('*.pdf')):
    out=subprocess.check_output(['pdffonts',str(p)],text=True)
    rows=out.splitlines()[2:]
    assert rows
    for row in rows:
        m=re.search(r'\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$',row)
        assert m and m.group(1)=='yes',(p,row)
    font_checks[str(p.relative_to(HERE))]={'all_embedded':True,'font_count':len(rows),'times_family':'NimbusRom' in out}
assert font_checks['paper-v08.pdf']['times_family']
styles=[]
for p in (HERE/'template').iterdir():
    q=ROOT/'docs/iclr/template/iclr2027'/p.name
    assert q.is_file() and sha(p)==sha(q),p
    styles.append(p.name)
history=json.loads((ROOT/'docs/iclr/organization-map.json').read_text())['moves']
for item in history.values():
    assert sha(ROOT/'docs/iclr'/item['path'])==item['sha256'],item['path']
prov=json.loads((HERE/'evidence/provenance.json').read_text())
for path,h in prov['input_sha256'].items():assert sha(ROOT/path)==h,path
assert len(prov['people'])==14 and len(prov['seeds'])==3
figprov=json.loads((HERE/'figures/provenance.json').read_text())
for path,h in figprov['inputs_sha256'].items():assert sha(ROOT/path)==h,path
record={'status':'passed','main_pages':main,'total_pages':len(r.pages),'disclosure_page':10,'appendix_start_page':12,'cited_primary_sources':len(keys),'historical_files_preserved':len(history),'official_template_files_unchanged':styles,'fonts':font_checks,'evidence_input_hashes_verified':len(prov['input_sha256']),'saved_comparisons_reproduced':prov['saved_comparisons_reproduced'],'paper_sha256':sha(pdf),'source_sha256':sha(HERE/'paper-v08.tex'),'appendix_sha256':sha(HERE/'appendix.tex'),'reader_facing_local_setup_scan':'passed','visual_review':'qa/portable-manuscript-review.json; reviews/portable-manuscript-methods-review.md and reviews/portable-manuscript-evidence-review.md. Earlier figure and comparison reviews remain historical; visual inspection is separate from these structural assertions.'}
(HERE/'qa/validation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['status','main_pages','total_pages','historical_files_preserved','evidence_input_hashes_verified']},indent=2))
