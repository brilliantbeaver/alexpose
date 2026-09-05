"""Compare live notebook source with the read-only initial text snapshot."""
from pathlib import Path
import difflib
import hashlib
import json
import re

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parents[1]
inventory = json.loads((DOCS / "reproducibility/notebook_inventory.json").read_text())
records = []
for row in inventory:
    path = ROOT / row["notebook"]
    current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if current_hash == row["sha256"]:
        continue
    old = (DOCS / "reproducibility/notebook_text" / (path.stem + ".txt")).read_text(encoding="utf-8")
    old_cells = re.split(r'\n=== CELL \d+ \[\w+\] execution=[^\n]* ===\n', old)[1:]
    old_sources = [cell.split('\nOUTPUT:\n', 1)[0].split('\nHTML TABLE:\n', 1)[0].strip() for cell in old_cells]
    current = json.loads(path.read_bytes())
    new_sources = [''.join(c.get('source', [])).strip() for c in current['cells']]
    diffs = []
    if len(old_sources) != len(new_sources):
        diffs.append(f'Cell count changed: {len(old_sources)} -> {len(new_sources)}')
    for index, (old_source, new_source) in enumerate(zip(old_sources, new_sources)):
        if old_source != new_source:
            diff = '\n'.join(difflib.unified_diff(old_source.splitlines(), new_source.splitlines(), fromfile='initial', tofile='current', lineterm=''))
            diffs.append(f'CELL {index}\n{diff}')
    records.append({'notebook':row['notebook'], 'initial_sha256':row['sha256'], 'current_sha256':current_hash,
                    'source_diffs':diffs, 'note':'Source-code comparison excludes saved outputs and metadata.'})
manifest = json.loads((DOCS / 'evidence/verification_manifest.json').read_text())
changed_inputs = [r['path'] for r in manifest['inputs'] if hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()!=r['sha256']]
result = {'notebook_updates':records, 'changed_evidence_inputs':changed_inputs}
(DOCS / 'review/concurrent_source_updates.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
