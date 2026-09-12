"""Read-only delivery audit; writes its own report only when requested."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import nbformat

ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/'work/artifacts/iclr-bridge-2026-09-11'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    before=json.loads((WORK/'source-preservation-before.json').read_text())
    after={name:{'sha256':digest(Path(name)), 'size':Path(name).stat().st_size,
                 'mtime_ns':Path(name).stat().st_mtime_ns} for name in before}
    if before != after: raise ValueError('Historical paper or notebook modified')
    notebooks=[]
    for source in sorted(ROOT.glob('2[0-2]_*.ipynb')) + sorted(ROOT.glob('19_*.ipynb')):
        executed=WORK/'executed_notebooks'/source.name
        a,b=nbformat.read(source,as_version=4),nbformat.read(executed,as_version=4)
        nbformat.validate(a);nbformat.validate(b)
        assert [c.source for c in a.cells] == [c.source for c in b.cells]
        errors=[o for c in b.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
        code_cells=[c for c in b.cells if c.cell_type=='code']
        if errors or any(c.execution_count is None for c in code_cells):
            raise ValueError('Notebook did not finish Run All')
        notebooks.append({'source':str(source.relative_to(ROOT)), 'source_sha256':digest(source),
            'executed':str(executed.relative_to(ROOT)), 'executed_sha256':digest(executed),
            'code_cells':len(code_cells),'errors':0})
    links=[]
    # Only local Markdown targets, excluding URL anchors and mathematical syntax.
    for doc in sorted((ROOT/'docs/studies/iclr').glob('*.md')):
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
            if '://' in target or target.startswith('#'):continue
            path=(doc.parent/target.split('#')[0]).resolve()
            links.append({'document':doc.name,'target':target,'exists':path.exists()})
    return {'status':'passed' if all(x['exists'] for x in links) else 'missing_local_links',
        'historical_sources_unchanged':len(before),'source_snapshot_sha256':digest(WORK/'source-preservation-before.json'),
        'notebooks':notebooks,'local_links':links}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args()
    result=check()
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']!='passed':raise SystemExit(1)
