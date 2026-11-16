"""Verify current restart artifacts without refitting or editing frozen outputs."""
from pathlib import Path
import hashlib,json,re
from urllib.parse import unquote
import nbformat
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    checks={};run=ROOT/'artifacts/runs/restart-dynamics-v1'
    hashes=json.loads((run/'artifact_hashes.json').read_text())
    assert all(sha(run/p)==h for p,h in hashes.items());checks['model_artifact_hashes']=len(hashes)
    identity=json.loads((run/'identity.json').read_text())
    assert identity['code_sha256']==sha(ROOT/'worldmodels/restart_experiment.py')
    assert identity['protocol_sha256']==sha(ROOT/'docs/restart/protocol.md')
    checks['frozen_code_and_protocol']=True
    rec=json.loads((ROOT/'docs/restart/notebook_execution.json').read_text());assert rec['status']=='complete'
    for p,h in rec['inputs'].items():assert sha(ROOT/p)==h
    cells=0
    for n in rec['notebooks']:
        assert sha(ROOT/n['name'])==n['source_sha256']
        out=ROOT/'artifacts/executed-restart'/n['name'];assert sha(out)==n['output_sha256']
        nb=nbformat.read(out,as_version=4)
        for c in nb.cells:
            if c.cell_type=='code':
                assert c.execution_count is not None
                assert all(o.output_type!='error' for o in c.outputs);cells+=1
    assert len(rec['notebooks'])==6 and cells==30
    checks['executed_notebooks']=6;checks['executed_cells']=cells
    slides=json.loads((ROOT/'slides/restart_manifest.json').read_text());assert slides['results_sha256']==sha(run/'results.json')
    assert slides['generator_sha256']==sha(ROOT/'scripts/build_restart_slides.py')
    for s in slides['slides']:assert sha(ROOT/'slides/restart_images'/f"slide-{s['slide']:02d}.svg")==s['svg_sha256']
    assert len(slides['slides'])==22;checks['vector_slides']=22
    for p in ['docs/draft01/manuscript.pdf','docs/draft01/manuscript.md','docs/draft01/README.md','docs/draft01/reviews/independent_review.md','docs/restart/final_adversarial_review.md','docs/restart/tutorial_visual_review.md','slides/restart.pdf']:
        assert (ROOT/p).is_file(),f'Missing delivery {p}'
    broken=[]
    for path in [ROOT/'README.md',ROOT/'slides/README.md']:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in target or target.startswith('#'):continue
            dest=(path.parent/unquote(target.split('#')[0])).resolve()
            if not dest.exists():broken.append((str(path.relative_to(ROOT)),target))
    assert not broken,broken;checks['current_navigation_links']=True
    reports=['README.md','docs/restart/protocol.md','docs/restart/interpretation_notes.md','docs/restart/data_audit.md','docs/restart/data_contract.md','docs/restart/model_contract.md','docs/restart/implementation_review.md','docs/restart/final_adversarial_review.md','docs/restart/tutorial_visual_review.md','docs/restart/notebook_execution.json','docs/draft01/manuscript.pdf','docs/draft01/manuscript.md','docs/draft01/manuscript.tex','docs/draft01/README.md','docs/draft01/reviews/independent_review.md','slides/restart.pdf','slides/restart.html','slides/restart_manifest.json']
    result=dict(status='complete',checks=checks,artifacts={p:sha(ROOT/p) for p in reports},limitations=['Draft author information remains pending user confirmation.','Exploratory source-level evidence does not establish patient-level clinical generalization.'])
    (ROOT/'docs/restart/delivery_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':main()
