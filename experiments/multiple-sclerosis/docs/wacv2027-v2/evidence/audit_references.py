"""Retrieve publisher-deposited Crossref metadata for original reference audit.

Candidate matching is evidence collection, not automatic verification.
"""
from pathlib import Path
import concurrent.futures, difflib, json, re, time, urllib.parse, urllib.request

ROOT = Path(__file__).parent
rows = json.loads((ROOT / 'reference-originals.json').read_text())

def normalized(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

def retrieve(row):
    q = urllib.parse.urlencode({'query.title': row['title'], 'rows': 3})
    url = 'https://api.crossref.org/works?' + q
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ResearchBibliographyAudit/1.0'})
            data = json.load(urllib.request.urlopen(req, timeout=45))['message']['items']
            candidates = []
            for item in data:
                title = (item.get('title') or [''])[0]
                score = difflib.SequenceMatcher(None, normalized(row['title']), normalized(title)).ratio()
                candidates.append({'similarity': score, 'metadata': item})
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            return {**row, 'query_url': url, 'candidates': candidates}
        except Exception as e:
            if attempt == 2:
                return {**row, 'error': str(e)}
            time.sleep(attempt + 1)

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    result = list(pool.map(retrieve, rows))
(ROOT / 'reference-crossref-candidates.json').write_text(json.dumps(result, indent=2))
for row in result:
    c = (row.get('candidates') or [{}])[0]
    m = c.get('metadata', {})
    authors = '; '.join((a.get('given','')+' '+a.get('family','')).strip() for a in m.get('author',[]))
    print(row['number'], round(c.get('similarity',0),3), m.get('DOI'), m.get('title'), m.get('published', m.get('issued')), authors)
