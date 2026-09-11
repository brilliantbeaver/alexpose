"""Verify frozen assets, original-file preservation, local links and V7/V8 numbers."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    r=ROOT
    audit=json.loads((r/'review/evidence_check.json').read_text())
    for name,digest in audit['sha256'].items():
        assert sha(r.parent.parent/name)==digest,name
    state=json.loads((r/'review/readme_preservation.json').read_text())
    readme=(r.parent/'README.md').read_bytes()
    assert hashlib.sha256(readme[:state['original_bytes']]).hexdigest()==state['original_sha256']
    links=[(r/'README.md',(r/'README.md').read_text()),
           (r.parent/'README.md',readme.decode().split('<a id="fmts-2026-review"></a>',1)[1])]
    for v in range(1,9):
        md=r/f'paper_v{v}.md';links.append((md,md.read_text()))
        a=r/f'assets/v{v}';manifest=json.loads((a/'manifest.json').read_text())
        for key,suffix in [('markdown_sha256','.md'),('tex_sha256','.tex'),('pdf_sha256','.pdf'),('overleaf_sha256','_overleaf.zip'),('supplement_sha256','_supplement.zip')]:
            assert sha(r/f'paper_v{v}{suffix}')==manifest[key],(v,key)
        for name,digest in manifest['files'].items():
            assert sha(a/name)==digest,(v,name)
        with zipfile.ZipFile(r/f'paper_v{v}_supplement.zip') as z:
            assert z.testzip() is None
            for name in z.namelist():
                assert not re.search(r'/Users/|theodoremui|alexpose|gavd5-drift|physworld_revisions',z.read(name).decode()),(v,name)
            for name in ['draw_figures.py','verify_summary.py','numerical_evidence.json']:
                assert z.read(name)==(a/name).read_bytes(),(v,name)
        texts=[p.extract_text() for p in PdfReader(r/f'paper_v{v}.pdf').pages]
        assert next(i for i,t in enumerate(texts) if 'References' in t)==4
    for path,text in links:
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('http:','https:','#','mailto:')):
                continue
            local=target.split('#',1)[0]
            if local:
                assert (path.parent/local).exists(),(path,target)
    for v in [7,8]:
        text=(r/f'paper_v{v}.md').read_text()
        assert not re.search(r'An earlier reflection|earlier reflection experiment|fully powered null|only route',text,re.I)
        assert re.search(r'375(?: of |/)375',text)
        assert re.search(r'33(?: of |/)75',text)
        evidence=json.loads((r/f'assets/v{v}/numerical_evidence.json').read_text())
        rows=[line for line in text.splitlines() if re.match(r'\|[^|]+\|\s*0\.\d+',line)]
        assert len(rows)==5
        for line,row in zip(rows,evidence['readout_rows'][2:7]):
            cells=[c.strip() for c in line.strip('|').split('|')]
            effect=evidence['trained_minus_initial'][row['experiment']+'/'+row['condition']]
            assert cells[1]==f"{row['mean_r2']:.3f}",(v,cells)
            assert cells[2]==f"{effect['difference_r2']:.3f}",(v,cells)
            assert cells[3]==f"[{effect['r2_ci95'][0]:.3f}, {effect['r2_ci95'][1]:.3f}]",(v,cells)
        for value in ['0.223','49/125','96/125','70.4%','0.218','0.0415','0.0436','0.0440']:
            assert value in text,(v,value)
    result={'original_evidence_files_unchanged':len(audit['sha256']),
            'prior_readme_bytes_preserved':state['original_bytes'],'versions_checked':8,
            'all_main_text_pages':4,'final_total_pages':6,'manifests':'pass',
            'supplement_anonymity_and_private_asset_consistency':'pass','local_links':'pass',
            'v7_v8_paired_tables_repeated_values_and_chronology':'pass',
            'raw_inference_or_bootstrap_rerun':False}
    (r/'review/final_integrity_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    files={str(p.relative_to(r)):sha(p) for p in r.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    (r/'artifact_manifest.json').write_text(json.dumps({'date':'2026-09-10',
        'scope':'Local manuscript revision snapshots; no external release','files':files},indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
