"""Check notebook structure, generated figure XML and local Markdown links."""
from pathlib import Path
import ast
import re
import xml.etree.ElementTree as ET
import nbformat
ROOT=Path(__file__).resolve().parents[1]
errors=[]
for path in ROOT.glob('*.ipynb'):
    notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
    for i,cell in enumerate(notebook.cells):
        if cell.cell_type=='code':
            try: ast.parse(cell.source)
            except SyntaxError as exc: errors.append(f'{path.name} cell{i}: {exc}')
for path in (ROOT/'images').glob('*.svg'): ET.parse(path)
for path in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]:
    for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
        if '://' in target or target.startswith('#'):continue
        destination=target.split('#')[0]
        if not (path.parent/destination).exists():errors.append(f'{path.name}: missing link {target}')
if errors:raise SystemExit('\n'.join(errors))
print('Validated eight notebooks, SVG XML, and local README/docs links.')
