"""Package v9's sources and compile an independently extracted Overleaf project."""
from pathlib import Path
import hashlib, json, re, shutil, subprocess, tempfile, zipfile
ROOT=Path(__file__).resolve().parents[1]
NAME='wacv2007-v9-overleaf'
QA=ROOT/'qa'/NAME
QA.mkdir(parents=True,exist_ok=True)
workspace=Path(tempfile.mkdtemp(prefix='wacv-overleaf-'))
stage=workspace/'project'
stage.mkdir()
files={
 'main.tex':'wacv2007-v9.tex',
 'references-v9.bib':'references-v9.bib',
 'wacv.sty':'wacv.sty',
 'ieeenat_fullname.bst':'ieeenat_fullname.bst',
 'figures/pipeline-v5.pdf':'figures/pipeline-v5.pdf',
 'figures/folds-and-errors-v5.pdf':'figures/folds-and-errors-v5.pdf',
}
# Keep the methodological records cited by the manuscript readily editable.
# These are documentation; rebuilding the research itself still needs project data.
for name in ['v9-methods.md','method-details.md','reproducibility.md','notebook-audit.md',
             'claim-ledger.md','results-numerical-consistency.md','reference-audit.md',
             'venue-and-literature.md','results_audit.py','results-audit.json',
             'notebook-audit-checks.json']:
 files['evidence/'+name]='evidence/'+name
for target,source in files.items():
 destination=stage/target
 destination.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(ROOT/source,destination)
readme='''# Overleaf project: WACV 2027 manuscript v9

Upload this ZIP as a new Overleaf project. Select `main.tex` as the main document and **pdfLaTeX** as the compiler. The included official WACV 2027 style and bibliography style are used directly; BibTeX generates the references from `references-v9.bib`. Recompile from scratch if an imported project shows stale citation warnings.

The project contains the final v9 manuscript, its two vector figures, bibliography and the companion method/evidence records referred to by the paper. The manuscript source is byte-identical to `wacv2007-v9.tex` in the delivered folder, renamed `main.tex` for convenient import. No external image download or shell-escape command is needed to compile the paper.

The review copy retains an anonymous author block and the intentional `DRAFT` paper-ID placeholder. Replace that placeholder with the assigned submission number when available. Keep anonymity for workshop review. No submission or upload is performed by this package.

The evidence directory is authoring documentation, not an automatically appended paper supplement. Its reconstruction scripts and references to the broader project require the original experiment directory and retained data; videos, pose caches, model checkpoints and a complete training environment are not included in this manuscript-source ZIP. For links to the full original-draft audit and revision history, use the accompanying `docs/wacv2007` folder. Do not assume the ZIP's documentation alone constitutes a cleared data release.

Local verification extracts this ZIP into a fresh directory and compiles `main.tex` with Tectonic (XeTeX). The expected Overleaf compiler is pdfLaTeX using the official template; an actual Overleaf-hosted build has not been performed. The package check verifies the complete required asset set, page count, rendered citations and absence of overfull boxes. It does not validate the scientific data's clinical labels or rights.
'''
(stage/'README.md').write_text(readme)
zip_path=ROOT/(NAME+'.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in sorted(stage.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(stage).as_posix())
extracted=workspace/'extracted'
with zipfile.ZipFile(zip_path) as z:
 assert z.testzip() is None
 z.extractall(extracted)
assert (extracted/'main.tex').read_bytes()==(ROOT/'wacv2007-v9.tex').read_bytes()
source=(extracted/'main.tex').read_text()
for path in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',source):
 assert (extracted/path).is_file(),path
for path in re.findall(r'\\bibliography\{([^}]+)\}',source):
 assert (extracted/(path+'.bib')).is_file(),path
result=subprocess.run(['tectonic','--keep-logs','--keep-intermediates','main.tex'],cwd=extracted,
 text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(QA/'build-output.txt').write_text(result.stdout)
if result.returncode:raise RuntimeError(result.stdout)
shutil.copyfile(extracted/'main.pdf',QA/'main.pdf')
shutil.copyfile(extracted/'main.log',QA/'main.log')
log=(extracted/'main.log').read_text()
assert not re.search(r'Overfull|Missing character|(?:Citation|Reference).*undefined',log)
text=subprocess.check_output(['pdftotext','-layout',str(extracted/'main.pdf'),'-'],text=True)
canonical=subprocess.check_output(['pdftotext','-layout',str(ROOT/'wacv2007-v9.pdf'),'-'],text=True)
assert text==canonical,'Packaged PDF differs from final canonical PDF text'
(QA/'text.txt').write_text(text)
pages=[p for p in text.split('\f') if p.strip()]
refpages=[i+1 for i,p in enumerate(pages) if re.search(r'\bReferences\b',p)]
assert refpages and refpages[0]<=8
assets=[{'path':p.relative_to(stage).as_posix(),'bytes':p.stat().st_size,
 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(stage.rglob('*')) if p.is_file()]
manifest={'zip':zip_path.name,'zip_sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest(),
 'source_matches_final_v9':True,'archive_integrity_pass':True,'extracted_build_pass':True,
 'packaged_and_canonical_pdf_text_identical':True,'pages':len(pages),'references_start_page':refpages[0],
 'compiler_tested':'Tectonic (XeTeX), local extracted ZIP','overleaf_hosted_build_performed':False,
 'overleaf_main':'main.tex','overleaf_recommended_compiler':'pdfLaTeX','build_workspace':str(workspace),
 'assets':assets}
(QA/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='assets'},indent=2))
