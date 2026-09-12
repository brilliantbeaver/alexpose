"""Read-only delivery audit; writes its own report only when requested."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import nbformat

ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/'work/artifacts/iclr-bridge-2026-09-11'
SOURCE_DIR=ROOT/'notebooks/iclr_bridge'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def notebook_sources(directory=SOURCE_DIR):
    sources=[]
    for number in range(19,23):
        matches=list(directory.glob(f'{number}_*.ipynb'))
        if len(matches)!=1:
            raise ValueError(f'Expected exactly one notebook {number} in {directory}; found {len(matches)}')
        sources.append(matches[0])
    return sources


def check(executed_dir=WORK/'executed_notebooks'):
    before=json.loads((WORK/'source-preservation-before.json').read_text())
    after={name:{'sha256':digest(Path(name)), 'size':Path(name).stat().st_size,
                 'mtime_ns':Path(name).stat().st_mtime_ns} for name in before}
    if before != after: raise ValueError('Historical paper or notebook modified')
    notebooks=[]
    for source in notebook_sources():
        executed=executed_dir/source.name
        a,b=nbformat.read(source,as_version=4),nbformat.read(executed,as_version=4)
        nbformat.validate(a);nbformat.validate(b)
        if [(c.cell_type,c.source) for c in a.cells] != [(c.cell_type,c.source) for c in b.cells]:
            raise ValueError(f'Source differs from {executed}; execute current notebooks into a new directory and pass --executed-dir')
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--executed-dir',type=Path,default=WORK/'executed_notebooks',
                   help='Executed copies matching the current source; historical copies are never overwritten')
    a=p.parse_args()
    result=check(a.executed_dir.resolve())
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']!='passed':raise SystemExit(1)
