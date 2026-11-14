"""Freeze a manuscript evidence packet and run a read-only Codex review.

No commits are made to the user's repository. Suite freeze helpers were not
installed, so this local packet uses clean Git commits and content checks.
"""
from pathlib import Path
import subprocess, shutil, tempfile, json, argparse, hashlib
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent.parent

def run(command, cwd):
    return subprocess.run(command, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()

def review(version):
    report = ROOT / 'reviews' / f'codex-v{version}'
    report.mkdir(exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix=f'wacv-v{version}-review-'))
    run(['git','init','-q'], target)
    (target/'original-draft.txt').write_bytes((ROOT/'evidence/original-draft.txt').read_bytes())
    run(['git','add','.'],target)
    cfg=['-c','user.name=Research review snapshot','-c','user.email=snapshot@localhost','-c','commit.gpgsign=false']
    run(['git',*cfg,'commit','-qm','Original supplied manuscript'],target)
    base=run(['git','rev-parse','HEAD'],target)
    for directory in ['sjepa']:
        shutil.copytree(EXP/directory,target/directory,ignore=shutil.ignore_patterns('__pycache__','tests'))
    for file in EXP.glob('*.ipynb'):
        notebook=json.loads(file.read_text())
        def safe_outputs(cell):
            return [{key:value for key,value in out.items() if key in ('output_type','text','ename','evalue','traceback') } | {'text/plain':out.get('data',{}).get('text/plain',[])} for out in cell.get('outputs',[])]
        text='\n\n'.join(f'CELL {i} [{cell["cell_type"]}]\n'+''.join(cell['source'])+'\nOUTPUTS (text only; embedded images omitted)\n'+json.dumps(safe_outputs(cell)) for i,cell in enumerate(notebook['cells']))
        (target/(file.stem+'.txt')).write_text(text)
    for file in (ROOT/'evidence').glob('*'):
        if file.is_file() and file.suffix in {'.md','.json','.py'}:
            shutil.copy2(file,target/file.name)
    ambient=EXP.parent.parent/'ambient'
    for relative in ['pose/keypoint_data.py','pose/keypoint_extractor.py','classification/features.py']:
        dst=target/'ambient'/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ambient/relative,dst)
    prior=8 if version>=9 else (5 if version>5 else (4 if version>4 else 2))
    shutil.copy2(ROOT/f'wacv2007-v{prior}.tex',target/'manuscript.tex')
    shutil.copy2(ROOT/'references.bib',target/'references.bib')
    for v in {prior,version}:
        bibliography=ROOT/f'references-v{v}.bib'
        if bibliography.exists():shutil.copy2(bibliography,target/bibliography.name)
    rendered=ROOT/f'qa/wacv2007-v{version}/text.txt'
    if rendered.exists():shutil.copy2(rendered,target/'rendered-paper-text.txt')
    for stamp in ['20260921T011305448893Z','20261009T060955236580Z']:
        path=EXP/'artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5'/f'capstone-{stamp}'
        for name in ['results.json','oof.json','provenance.json']:
            shutil.copy2(path/name,target/f'{stamp}-{name}')
    run(['git','add','.'],target)
    run(['git',*cfg,'commit','-qm','Prior draft and complete frozen evidence'],target)
    base=run(['git','rev-parse','HEAD'],target)
    shutil.copy2(ROOT/f'wacv2007-v{version}.tex',target/'manuscript.tex')
    run(['git','add','manuscript.tex'],target)
    run(['git',*cfg,'commit','-qm',f'Revision {version}'],target)
    head=run(['git','rev-parse','HEAD'],target)
    lens='''First-party scientific manuscript challenge review, not a code merge. The user requests a WACV 2027 Vision4Neuro paper derived from the seven top-level notebooks, which are the source of truth. The original supplied PDF was converted to original-draft.txt. The base manuscript is a previously reviewed revision, and current manuscript is the revision under review. Previous Codex findings about omitted uncertainty, exact source splitting, RF input definition, and an interpolation-reporting overclaim have been addressed; verify that they remain fixed. All57 reference identities have now been checked in reference-audit.md/JSON with primary overrides for failed initial search candidates. Notebook outputs contain text only; embedded images are intentionally omitted. The intended audience is educated researchers who may not know ML or statistical jargon: flag consequential unexplained terminology while preserving reproducibility. Review manuscript.tex and bibliography against notebook text, implementation, OOF artifacts, and evidence audits. Do not treat later RF rerun as the main notebook study. No edits, recursive Codex, MCP, or external tools. Only read files, using nl -ba for exact citations. Ask whether the inference is justified, not merely whether text repeats code. Scope is a bounded empirical pilot, not clinical deployment. For each question give HOLDS, BROKEN or NOT REACHED with quotes: (1) every numerical claim, denominator, weighting, bootstrap and comparison is faithful; (2) Section 4.1 masking, 12 landmarks/all33, coordinate chain and teacher readout are accurate; (3) same-seed runs, source vs participant grouping, train/validation boundaries and notebook vs additional reanalysis are distinguished; (4) citations support scientific claims and do not hallucinate; (5) negative-result wording neither establishes equivalence nor causal temporal benefit and identifies consequential limitations. Name claimed properties for which the retained experiments provide no evidence. Prioritize concrete scientific or numerical defects and missing reproduction details, not generic wish lists for new experiments. Acknowledge known limitations already stated, do not report them as undisclosed. Report line count and last line of manuscript.tex as reading anchor. Each defect requires exact file:line, consequence, fix and severity. End exactly Verdict: approve or Verdict: needs-attention.'''
    if version>=9:
        lens+=''' The user specifically requests constructive v9 framing, foregrounding the observed S-JEPA RF advantage and reducing statistical jargon. This is authorized editorial reframing, not authorization to suppress material contrary evidence. Check the complete abstract and conclusion against Results. Positive descriptions of trained features are allowed if limited to predictive information in these collection labels; do not demand a negative framing or significance-test-centered narrative. Challenge any unsupported causal claim that training improved features versus initialization, clinical meaning, temporal benefit, or general RF superiority. Verify that equal-clip comparison, competitive mean pose, RF defect, and numerical intervals remain accurately available. v9-methods.md is an actual included companion for exact resampling/formulas; read it before alleging omitted details. Confirm all three scores 0.457/0.452/0.411 and training45--58clips/24sources. Named properties must be tied to evidence. No new training or clinical validation was run. Review v8 as base and v9 as current.'''
    framing=Path('/Users/pmui/.codex/skills/codex-adversarial-review/references/adversarial-framing.md').read_text()
    prompt=framing.replace('{{BASE}}',base).replace('{{HEAD}}',head).replace('{{LENS}}',lens)
    (report/'prompt.md').write_text(prompt)
    assert not run(['git','status','--porcelain'],target)
    manifest={'target':str(target),'base':base,'head':head,'model':'gpt-6-astra','effort':'high','helper_status':'suite freeze helpers unavailable; isolated committed packet and clean checks used','same_model_family_review':True}
    (report/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with (report/'prompt.md').open() as src, (report/'review.log').open('w') as log:
        result=subprocess.run(['codex','exec','--sandbox','read-only','--cd',str(target),'-m','gpt-6-astra','-c','model_reasoning_effort=high','-o',str(report/'review.md'),'--','-'],stdin=src,stdout=log,stderr=subprocess.STDOUT)
    manifest['returncode']=result.returncode
    manifest['head_unchanged']=run(['git','rev-parse','HEAD'],target)==head
    manifest['clean_after']=not run(['git','status','--porcelain'],target)
    manifest['report_written']=(report/'review.md').exists() and (report/'review.md').stat().st_size>0
    (report/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
    if result.returncode or not all(manifest[x] for x in ['head_unchanged','clean_after','report_written']):
        raise RuntimeError('Review incomplete or immutable target changed')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('version',type=int)
    review(parser.parse_args().version)
