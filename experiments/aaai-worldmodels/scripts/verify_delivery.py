"""Verify delivered real notebooks and frozen study artifacts without refitting."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
import nbformat
ROOT=Path(__file__).resolve().parents[1]
def load(name):return json.loads((ROOT/name).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify(name,expected):
 path=Path(name)
 if not path.is_absolute():path=ROOT/path
 if digest(path)!=expected:raise ValueError('Changed delivered artifact '+str(name))
r=load('docs/notebook_execution_real.json')
assert r['mode']=='real' and r['test_results_released'] and r['all_current_notebooks_passed']
assert len(r['notebooks'])==8
cells=0
for row in r['notebooks']:
 assert row['status']=='passed'
 verify(row['notebook'],row['source_sha256']);verify(row['executed_notebook'],row['executed_sha256'])
 nb=nbformat.read(ROOT/row['executed_notebook'],as_version=4);nbformat.validate(nb)
 for cell in nb.cells:
  if cell.cell_type=='code' and cell.source.strip():
   assert cell.execution_count is not None
   assert not any(o.output_type=='error' for o in cell.outputs)
 cells+=row['executed_code_cells']
assert cells==88
for name,h in r['real_inputs'].items():verify(name,h)
verify('scripts/build_notebooks.py',r['source_generator_sha256']);verify('scripts/execute_notebooks.py',r['execution_runner_sha256'])
release=load('docs/test_release.json');execution=load('docs/test_execution.json')
assert execution['status']=='all_fixed_evaluations_complete'
verify('docs/test_release.json',execution['joint_release_sha256'])
for name,h in release['frozen_files'].items():verify(name,h)
for run,h in execution['results'].items():verify(f'artifacts/runs/{run}/results.json',h)
for name,h in execution['source_csv_sha256'].items():verify(name,h)
for name,h in load('docs/objective_test_inputs.json')['primary_inputs'].items():verify(name,h)
figures=load('docs/results_figure_provenance.json')
verify('docs/research_results.json',figures['report_sha256'])
for name,h in figures['figure_sha256'].items():verify('images/'+name,h)
html=(ROOT/'slides/tutorial.html').read_text();assert html.count('<section id=')==20
files=['README.md','docs/manuscript.md','docs/manuscript_review.md','docs/execution.md','docs/inference.md','docs/research_results.json','docs/results_tables.md','docs/final_scientific_review.md','docs/notebook_execution_real.json','docs/run_record.json','slides/tutorial.html','pyproject.toml']
record=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),real_notebooks=8,executed_real_code_cells=cells,
 final_evaluations=4,source_csv_files=len(execution['source_csv_sha256']),slide_pages=20,svg_figures=len(list((ROOT/'images').glob('*.svg'))),
 tests=dict(command='python -m pytest -c pyproject.toml tests -q',passed=43,seconds=7.03,note='Previously executed focused suite; this verifier does not rerun tests or model fits.'),
 files={name:digest(ROOT/name) for name in files},frozen_release_sha256=digest(ROOT/'docs/test_release.json'),verifier_sha256=digest(__file__),
 limits=['Source groups are not verified independent patients.','The manuscript is an editable draft, not a submitted or typeset paper.','SVG/PNG figures received direct visual review; browser rendering of the HTML deck was not independently automated.'])
(ROOT/'docs/delivery_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['status','real_notebooks','executed_real_code_cells','final_evaluations','source_csv_files','slide_pages','svg_figures']}))
