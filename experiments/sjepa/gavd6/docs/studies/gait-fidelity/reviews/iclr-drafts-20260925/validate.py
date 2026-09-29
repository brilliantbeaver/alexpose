"""Read-only consistency checks, followed by a small validation receipt."""
from pathlib import Path
from urllib.parse import urlsplit, unquote
import csv
import hashlib
import json
import statistics

from bs4 import BeautifulSoup
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[1]
REPO = DOCS.parents[2]
OUT = REPO / 'outputs/iclr'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
transfer = json.loads((OUT / 'transfer-inventory-20260925T172906510508Z.json').read_text())
selected = [r for r in transfer['files'] if r['status']=='selected']
assert len(selected)==69
for row in selected:
    assert sha(OUT/row['destination'])==row['sha256'], row['destination']
originals = json.loads((HERE/'original-hashes.json').read_text())
for p, digest in originals.items():
    assert sha(REPO/p)==digest, f'Original changed: {p}'

def means(relative):
    rows=list(csv.DictReader((OUT/relative).open()))
    result={}
    for method in {r['method'] for r in rows}:
        group=[r for r in rows if r['method']==method]
        result[method]={k:statistics.mean(float(r[k]) for r in group)
                        for k in ['response_error','waveform_error']}
    return rows,result

core_rows,core=means('walking-core/evaluation/per-person.csv')
response_rows,response=means('jepa-response/evaluation/per-person.csv')
direct=core['P-direct-none-base']
assert round(direct['response_error'],2)==7.54
assert round(direct['waveform_error'],2)==12.07
delta=response['F-response-jepa_delta_v1-graph_time-paired_change']['response_error']
endpoint=response['F-response-jepa_endpoint_v1-graph_time-paired_change']['response_error']
assert round(endpoint-delta,4)==.3731
zero=statistics.mean(float(r['zero_response_error']) for r in response_rows)
assert round(zero,2)==5.81 and all(m['response_error']>zero for m in response.values())
assert len({r['canonical_person_id'] for r in core_rows})==14
assert len({r['seed'] for r in core_rows})==3

expected=['Introduction','Data collection','Methodology','AI models and techniques',
          'Experiments','Results','Discussion']
documents=[]
for kind in ['scientific','overview']:
    stem='full-writeup' if kind=='scientific' else 'research-overview'
    html=DOCS/(stem+'.html'); soup=BeautifulSoup(html.read_text(),'html.parser')
    assert '\\operatorname' not in html.read_text()
    broken=[]
    for tag in soup.select('[href],[src]'):
        value=tag.get('href',tag.get('src')); u=urlsplit(value)
        if u.scheme: continue
        target=(DOCS/unquote(u.path)).resolve() if u.path else html
        if not target.exists(): broken.append(value)
        elif u.fragment and target.suffix=='.html':
            dest=BeautifulSoup(target.read_text(),'html.parser')
            if not dest.find(id=u.fragment):broken.append(value)
    assert not broken,broken
    assert all(img.get('alt') for img in soup.find_all('img'))
    pdf=DOCS/(stem+'.pdf'); reader=PdfReader(pdf)
    assert pdf.stat().st_size>50000
    if kind=='overview':
        assert len(reader.pages)==2
        assert [h.get_text().split('. ',1)[1] for h in soup.find_all('h2')]==expected
        assert '5. Experiments' in reader.pages[1].extract_text()[:100]
        assert 'AI models and techniques' in reader.pages[0].extract_text()
    pages=[p.extract_text() for p in reader.pages]
    assert all(len(p)>200 for p in pages)
    documents.append({'kind':kind,'html_sha256':sha(html),'pdf_sha256':sha(pdf),
                      'pages':len(pages),'figures':len(soup.find_all('figure')),
                      'all_local_assets_and_links_resolve':True})
browser=json.loads((HERE/'browser-checks.json').read_text())
assert all(not r['errors'] and not r['overflow'] and not r['mobileOverflow'] for r in browser)
assert all(i['loaded'] for r in browser for i in r['images'])
record={'evidence_files_verified':len(selected),'originals_unchanged':True,
        'people':14,'seeds':3,'direct':direct,'delta_vs_endpoint_mean_improvement':endpoint-delta,
        'zero_response_error':zero,'documents':documents,
        'scope':'Compact evidence and rendered drafts; no raw-data or checkpoint replay.'}
(HERE/'final-validation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
