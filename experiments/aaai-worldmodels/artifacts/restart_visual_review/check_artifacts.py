"""Independent checks of executed notebooks and numerical slide inputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import nbformat

P=Path(__file__).resolve().parents[2]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
r=json.loads((P/'docs/restart/notebook_execution.json').read_text())
assert r['status']=='complete'
checks=[]
for entry in r['notebooks']:
    source=P/entry['name'];out=P/'artifacts/executed-restart'/entry['name']
    nb=nbformat.read(out,as_version=4);original=nbformat.read(source,as_version=4)
    code=[c for c in nb.cells if c.cell_type=='code']
    errors=[o for c in code for o in c.outputs if o.output_type=='error']
    item={'name':entry['name'],'source_hash_matches':sha(source)==entry['source_sha256'],
          'output_hash_matches':sha(out)==entry['output_sha256'],'code_cells':len(code),
          'execution_counts':[c.execution_count for c in code],'error_outputs':len(errors),
          'outputs':[len(c.outputs) for c in code]}
    assert item['source_hash_matches'] and item['output_hash_matches'] and not errors
    assert all(c.execution_count is not None for c in code)
    assert [c.source for c in nb.cells]==[c.source for c in original.cells]
    checks.append(item)
assert len(checks)==6 and sum(c['code_cells'] for c in checks)==30
inputs={name:sha(P/name)==value for name,value in r['inputs'].items()}
assert all(inputs.values())
slides=json.loads((P/'slides/restart_source.json').read_text())
manifest=json.loads((P/'slides/restart_manifest.json').read_text())
assert manifest['results_sha256']==sha(P/'artifacts/runs/restart-dynamics-v1/results.json')
assert manifest['generator_sha256']==sha(P/'scripts/build_restart_slides.py')
assert len(manifest['slides'])==len(slides)==22
assert all(sha(P/'slides/restart_images'/f'slide-{s["slide"]:02d}.svg')==s['svg_sha256'] for s in manifest['slides'])
results=json.loads((P/'artifacts/runs/restart-dynamics-v1/results.json').read_text())
normal={x['arm']:x for x in results['normal_forecast'] if x['horizon']==8}
for arm,estimate,ci in slides[15]['payload']:
    assert estimate==normal[arm]['estimate'] and ci==normal[arm]['ci95']
for contrast in slides[16]['payload']:
    assert contrast in results['paired_contrasts']
for arm,auroc,ci in slides[17]['payload'][1]:
    x=results['anomaly'][arm]
    assert auroc==f"{x['auroc']:.3f}" and ci==f"{x['auroc_ci95'][0]:.3f} to {x['auroc_ci95'][1]:.3f}"
assert slides[5]['payload'][1]==[[v['label'],v['windows'],v['sequences'],v['sources']] for v in results['cohort_by_label']]
report={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'notebooks':checks,
        'inputs_match':inputs,'slide_count':len(slides),'all_plotted_numbers_match_run':True,
        'results_sha256':sha(P/'artifacts/runs/restart-dynamics-v1/results.json'),
        'slides_manifest_sha256':sha(P/'slides/restart_manifest.json'),
        'slides_pdf_sha256':sha(P/'slides/restart.pdf'),
        'notebook_record_sha256':sha(P/'docs/restart/notebook_execution.json'),
        'checker_sha256':sha(Path(__file__))}
(P/'artifacts/restart_visual_review/notebook_and_numeric_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
